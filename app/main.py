# LLMOps Backend Structure
# - API Layer: FastAPI (app/main.py)
# - Orchestration Layer: workflow/ (base_node.py, runner.py, nodes/)
# - Retrieval Layer: rag/retriever.py (keyword/vector), rag/vector_index.py, rag/embeddings.py
# - Inference Layer: rag/llm_chat.py (OpenAI-compatible chat completions)
# - Graph Runner: rag/graph_runner.py (pipeline execution wrapper)
# - Observability: workflow runner step stats + explicit env validation errors
from pathlib import Path

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

app = FastAPI(title="Jinsie RAG Support System")


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
    try:
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
        resp = {
            "answer": result,
            "retrieval": {
                "mode": resp_mode,
                "hit_count": len(docs),
                "doc_ids": [d.doc_id for d in docs],
            },
        }

        if req.include_context:
            resp["retrieval"]["context_preview"] = context[:800]

        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/workflow/run")
def workflow_run(req: WorkflowRunRequest):
    try:
        # 最小 Node + Runner：Retriever -> Template -> LLM -> PythonCode
        runner = WorkflowRunner(
            nodes=[
                RetrieverNode(step_id="step_1_retriever", mode=req.retrieval_mode, top_k=req.top_k),
                TemplateNode(step_id="step_2_template", template=req.template),
                LLMNode(step_id="step_3_llm"),
                PythonCodeNode(step_id="step_4_python", function_name=req.python_function),
            ]
        )

        out = runner.run(
            {
                "query": req.query,
                # retriever 会产出 context/docs/mode
                "context": "",
            }
        )

        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
