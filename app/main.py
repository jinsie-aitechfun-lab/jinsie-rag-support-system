from pathlib import Path
import logging
import time
from datetime import datetime, timezone
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import jinsie_agent_platform as platform
from jinsie_agent_platform.runner import workflow_runner

from app.rag.retriever import format_context, keyword_retrieve, vector_retrieve
from app.rag.llm_chat import chat_answer, has_chat_env

from app.rag.pipeline import run_rag_pipeline

from app.workflow.runner import WorkflowRunner
from app.workflow.graph_runner_langgraph import run_langgraph_workflow
from app.workflow.nodes.template_node import TemplateNode
from app.workflow.nodes.retriever_node import RetrieverNode
from app.workflow.nodes.llm_node import LLMNode
from app.workflow.nodes.python_code_node import PythonCodeNode

from app.observability.emit import emit_metrics

# ✅ Day30 实战：统一响应结构 + 统一错误语义（最小侵入式，不改主链路）
from uuid import uuid4

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI(title="Jinsie RAG Support System")
logger = logging.getLogger(__name__)

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


def _utc_timestamp() -> str:
    # ISO 8601 in UTC with 'Z' suffix, e.g. 2026-02-28T15:32:11Z
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ok(data: dict, *, request_id: str) -> dict:
    return {
        "success": True,
        "request_id": request_id,
        "timestamp": _utc_timestamp(),
        "error_code": None,
        "data": data,
    }


def _err(code: str, message: str, *, request_id: str) -> dict:
    return {
        "success": False,
        "request_id": request_id,
        "timestamp": _utc_timestamp(),
        "error_code": code,
        "error": {"code": code, "message": message},
    }


def _normalize_metrics(metrics: dict | None) -> dict:
    """
    Unify metrics shape for all endpoints:
      { "total_ms": <float>, "retrieval_ms": <float>, "llm_ms": <float> }
    Missing/None/invalid values are filled with 0.0.
    """
    metrics = metrics or {}

    def _as_float(v) -> float:
        if v is None:
            return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0

    return {
        "total_ms": _as_float(metrics.get("total_ms")),
        "retrieval_ms": _as_float(metrics.get("retrieval_ms")),
        "llm_ms": _as_float(metrics.get("llm_ms")),
    }


def _normalize_usage(usage: dict | None) -> dict:
    usage = usage or {}

    def _as_int(v) -> int:
        if v is None:
            return 0
        try:
            return int(v)
        except Exception:
            return 0

    prompt_tokens = _as_int(usage.get("prompt_tokens"))
    completion_tokens = _as_int(usage.get("completion_tokens"))
    total_tokens = _as_int(usage.get("total_tokens"))

    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _estimate_cost(model: str, usage: dict) -> float:
    """
    Minimal cost estimator.
    Default is 0.0 unless env prices are configured.

    Env (optional):
    - OPENAI_PRICE_INPUT_PER_1K
    - OPENAI_PRICE_OUTPUT_PER_1K
    """
    try:
        input_price_per_1k = float(os.getenv("OPENAI_PRICE_INPUT_PER_1K", "0").strip() or 0.0)
        output_price_per_1k = float(os.getenv("OPENAI_PRICE_OUTPUT_PER_1K", "0").strip() or 0.0)
    except Exception:
        return 0.0

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)

    cost = (prompt_tokens / 1000.0) * input_price_per_1k + (completion_tokens / 1000.0) * output_price_per_1k
    return round(cost, 6)


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
    request_id = request.headers.get("x-request-id") or str(uuid4())
    message = str(exc.detail) if exc.detail else "http error"
    return JSONResponse(
        status_code=exc.status_code,
        content=_err("HTTP_ERROR", message, request_id=request_id),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_err("INTERNAL_ERROR", "internal server error", request_id=request_id),
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _augment_query_with_context(query: str, *, mode: str = "keyword", top_k: int = 3):
    mode_norm = (mode or "keyword").strip().lower()

    if mode_norm == "vector":
        docs = vector_retrieve(query, top_k=top_k)
    else:
        docs = keyword_retrieve(query, top_k=top_k)

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

    data["metrics"] = _normalize_metrics(out.get("metrics", {}))

    # -----------------------------
    # Observability (best-effort)
    # -----------------------------
    try:
        engine = (os.getenv("RAG_ENGINE") or "jinsie-rag-support-system").strip()
        m = data["metrics"] or {}
        answer_obj = data.get("answer") or {}
        model = str(answer_obj.get("model") or engine)
        usage = _normalize_usage(answer_obj.get("usage") or {})
        cost = _estimate_cost(model, usage)

        emit_metrics(
            [
                {
                    "request_id": request_id,
                    "total_ms": float(m.get("total_ms") or 0.0),
                    "llm_ms": float(m.get("llm_ms") or 0.0),
                    "retrieval_ms": float(m.get("retrieval_ms") or 0.0),
                    "engine": engine,
                    "status": "COMPLETED",
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "cost": cost,
                    "timestamp": _utc_timestamp(),
                }
            ]
        )
    except Exception:
        # Never affect main response
        pass

    return _ok(data, request_id=request_id)


@app.post("/v1/workflow/run")
def workflow_run(req: WorkflowRunRequest, request: Request):
    request_id = str(uuid4())

    # Header 灰度开关：不改 schema（默认 legacy）
    engine = (request.headers.get("x-workflow-engine") or "legacy").strip().lower()

    # 最小 Node + Runner：Retriever -> Template -> LLM -> PythonCode
    nodes = [
        RetrieverNode(step_id="step_1_retriever", mode=req.retrieval_mode, top_k=req.top_k),
        TemplateNode(step_id="step_2_template", template=req.template),
        LLMNode(step_id="step_3_llm"),
        PythonCodeNode(step_id="step_4_python", function_name=req.python_function),
    ]

    t0 = time.perf_counter()

    if engine == "langgraph":
        out = run_langgraph_workflow(
            nodes,
            {
                "query": req.query,
                "context": "",
            },
        )
    else:
        runner = WorkflowRunner(nodes=nodes)
        out = runner.run(
            {
                "query": req.query,
                # retriever 会产出 context/docs/mode
                "context": "",
            }
        )

    t1 = time.perf_counter()

    # ✅ metrics 真值化（复用 steps[].elapsed_ms）
    steps = out.get("steps", []) or []

    def _step_ms(step_id: str) -> float:
        for s in steps:
            if s.get("step_id") == step_id:
                try:
                    return float(s.get("elapsed_ms") or 0.0)
                except Exception:
                    return 0.0
        return 0.0

    retrieval_ms = _step_ms("step_1_retriever")
    llm_ms = _step_ms("step_3_llm")

    # 可观测：engine 灰度命中（不改 success/metrics 结构；只在 data 里追加）
    meta = out.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        out["meta"] = meta
    meta["engine"] = engine

    out["metrics"] = _normalize_metrics(
        {
            "total_ms": round((t1 - t0) * 1000.0, 1),
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
        }
    )

    # best-effort observability 上报：不能影响主链路
    try:
        workflow_usage_raw = (
            out.get("llm_usage")
            or meta.get("llm_usage")
            or out.get("usage")
            or ((out.get("answer") or {}).get("usage"))
            or {}
        )
        workflow_usage = _normalize_usage(workflow_usage_raw)

        llm_model = (
            out.get("llm_model")
            or meta.get("llm_model")
            or ((out.get("answer") or {}).get("model"))
            or os.getenv("OPENAI_MODEL")
            or "unknown"
        )

        emit_metrics(
            [
                {
                    "request_id": request_id,
                    "engine": f"workflow:{engine}",
                    "status": str(out.get("status") or "COMPLETED"),
                    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "total_ms": float(out["metrics"].get("total_ms") or 0.0),
                    "llm_ms": float(out["metrics"].get("llm_ms") or 0.0),
                    "retrieval_ms": float(out["metrics"].get("retrieval_ms") or 0.0),
                    "prompt_tokens": int(workflow_usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(workflow_usage.get("completion_tokens") or 0),
                    "total_tokens": int(workflow_usage.get("total_tokens") or 0),
                    "cost": _estimate_cost(
                        llm_model,
                        workflow_usage,
                    ),
                }
            ]
        )
    except Exception as e:
        logger.warning("workflow_run emit metrics failed: %s", e)

    return _ok(out, request_id=request_id)