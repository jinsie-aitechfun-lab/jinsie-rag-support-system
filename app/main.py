from pathlib import Path
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import jinsie_agent_platform as platform
from jinsie_agent_platform.runner import workflow_runner

from app.rag.retriever import format_context, keyword_retrieve, vector_retrieve
from app.rag.llm_chat import chat_answer, has_chat_env

from app.rag.pipeline import run_rag_pipeline

from app.workflow.runner import WorkflowRunner
from app.workflow.nodes.template_node import TemplateNode
from app.workflow.nodes.retriever_node import RetrieverNode
from app.workflow.nodes.llm_node import LLMNode
from app.workflow.nodes.python_code_node import PythonCodeNode

# ✅ Day30 实战：统一响应结构 + 统一错误语义（最小侵入式，不改主链路）
from uuid import uuid4

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI(title="Jinsie RAG Support System")

# LLMOps Backend Structure
# - API Layer: FastAPI (app/main.py)
# - Orchestration Layer: workflow/ (base_node.py, runner.py, nodes/)
# - Retrieval Layer: rag/retriever.py (keyword/vector), rag/vector_index.py, rag/embeddings.py
# - Inference Layer: rag/llm_chat.py (OpenAI-compatible chat completions)
# - Graph Runner: rag/graph_runner.py (pipeline execution wrapper)
# - Observability: workflow runner step stats + explicit env validation errors


class RagRunRequest(BaseModel):
    query: str
    include_context: bool = False
    retrieval_mode: str = "keyword"  # "keyword" | "vector"
    top_k: int = 3


class WorkflowRunRequest(BaseModel):
    query: str
    retrieval_mode: str = "keyword"  # "keyword" | "vector"
    top_k: int = 3
    template: str = (
        "你将获得一段检索到的上下文，请优先基于上下文完成任务。\n\n"
        "【上下文】\n{context}\n\n"
        "【用户问题】\n{query}"
    )
    python_function: str = "to_text"


@app.get("/health")
def health():
    return {"status": "ok"}


def _ok(data: dict, *, request_id: str) -> dict:
    return {"success": True, "request_id": request_id, "data": data}


def _err(code: str, message: str, *, request_id: str) -> dict:
    return {"success": False, "request_id": request_id, "error": {"code": code, "message": message}}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    FastAPI/Pydantic body/query/path validation errors (HTTP 422).
    Unify into {success, request_id, error{code,message}}.
    """
    request_id = request.headers.get("x-request-id") or str(uuid4())
    # Keep message compact; details remain available in server logs if needed.
    message = "validation failed"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_err("VALIDATION_ERROR", message, request_id=request_id),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # If client provides x-request-id, keep it; otherwise generate one.
    request_id = request.headers.get("x-request-id") or str(uuid4())

    detail = exc.detail
    if isinstance(detail, dict):
        request_id = detail.get("request_id") or request_id
        code = detail.get("code", "HTTP_ERROR")
        message = detail.get("message", "request failed")
    else:
        code = "HTTP_ERROR"
        message = str(detail)

    return JSONResponse(status_code=exc.status_code, content=_err(code, message, request_id=request_id))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    # Dev 期可以把 str(exc) 暴露出去；若你希望更“企业级”，这里可改为固定文案 + 服务器端日志
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_err("INTERNAL_ERROR", str(exc), request_id=request_id),
    )


def _platform_prompt_path() -> str:
    """
    Resolve the platform prompt path regardless of current working directory.
    platform.__file__ -> .../jinsie-ai-agent-platform/jinsie_agent_platform/__init__.py
    """
    repo_root = Path(platform.__file__).resolve().parents[1]
    prompt = repo_root / "app" / "prompts" / "system" / "agent_system.md"
    return str(prompt)


def _augment_query_with_context(query: str, *, mode: str, top_k: int) -> tuple[str, list, str]:
    mode_norm = (mode or "keyword").strip().lower()

    if mode_norm == "vector":
        docs = vector_retrieve(query, top_k=top_k)
        used_mode = "vector"
    else:
        docs = keyword_retrieve(query, top_k=top_k)
        used_mode = "keyword"

    context = format_context(docs)

    if not context:
        return query, docs, ""

    augmented = (
        "你将获得一段检索到的上下文，请优先基于上下文完成任务。\n\n"
        f"【上下文】\n{context}\n\n"
        f"【用户问题】\n{query}"
    )
    # Keep return signature; rag_run will set mode in response
    return augmented, docs, context


@app.post("/v1/rag/run")
def rag_run(req: RagRunRequest):
    request_id = str(uuid4())

    out = run_rag_pipeline(
        req.query,
        retrieval_mode=req.retrieval_mode,
        top_k=req.top_k,
        debug=True,
    )

    result = out["result"]
    docs = out["docs"]
    context = out["context"]
    resp_mode = out["resp_mode"]

    # 可审计：默认不返回上下文；需要时再打开
    data = {
        "answer": result,
        "retrieval": {
            "mode": resp_mode,
            "hit_count": len(docs),
            "doc_ids": [d.doc_id for d in docs],
        },
    }

    if req.include_context:
        data["retrieval"]["context_preview"] = context[:800]

    data["metrics"] = out.get("metrics", {})

    return _ok(data, request_id=request_id)


@app.post("/v1/workflow/run")
def workflow_run(req: WorkflowRunRequest):
    request_id = str(uuid4())

    # 最小 Node + Runner：Retriever -> Template -> LLM -> PythonCode
    runner = WorkflowRunner(
        nodes=[
            RetrieverNode(step_id="step_1_retriever", mode=req.retrieval_mode, top_k=req.top_k),
            TemplateNode(step_id="step_2_template", template=req.template),
            LLMNode(step_id="step_3_llm"),
            PythonCodeNode(step_id="step_4_python", function_name=req.python_function),
        ]
    )

    t0 = time.perf_counter()
    out = runner.run(
        {
            "query": req.query,
            # retriever 会产出 context/docs/mode
            "context": "",
        }
    )
    t1 = time.perf_counter()

    out["metrics"] = {"total_ms": round((t1 - t0) * 1000.0, 1)}

    return _ok(out, request_id=request_id)