from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import jinsie_agent_platform as platform
from jinsie_agent_platform.runner import workflow_runner

from app.rag.retriever import format_context, keyword_retrieve, vector_retrieve

app = FastAPI(title="Jinsie RAG Support System")


class RagRunRequest(BaseModel):
    query: str
    include_context: bool = False
    retrieval_mode: str = "keyword"  # "keyword" | "vector"
    top_k: int = 3


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
        augmented, docs, context = _augment_query_with_context(
            req.query, mode=req.retrieval_mode, top_k=req.top_k
        )

        result = workflow_runner(
            augmented,
            debug=True,
            prompt_path=_platform_prompt_path(),
        )

        mode_norm = (req.retrieval_mode or "keyword").strip().lower()
        resp_mode = "vector" if mode_norm == "vector" else "keyword"

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
