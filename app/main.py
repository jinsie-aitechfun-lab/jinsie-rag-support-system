from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import jinsie_agent_platform as platform
from jinsie_agent_platform.runner import workflow_runner

from app.rag.retriever import format_context, keyword_retrieve

app = FastAPI(title="Jinsie RAG Support System")


class RagRunRequest(BaseModel):
    query: str
    include_context: bool = False


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


def _augment_query_with_context(query: str) -> tuple[str, list, str]:
    docs = keyword_retrieve(query, top_k=3)
    context = format_context(docs)

    if not context:
        return query, docs, ""

    augmented = (
        "你将获得一段检索到的上下文，请优先基于上下文完成任务。\n\n"
        f"【上下文】\n{context}\n\n"
        f"【用户问题】\n{query}"
    )
    return augmented, docs, context


@app.post("/v1/rag/run")
def rag_run(req: RagRunRequest):
    try:
        augmented, docs, context = _augment_query_with_context(req.query)

        result = workflow_runner(
            augmented,
            debug=False,
            prompt_path=_platform_prompt_path(),
        )

        # 可审计：默认不返回上下文；需要时再打开
        resp = {
            "answer": result,
            "retrieval": {
                "mode": "keyword",
                "hit_count": len(docs),
                "doc_ids": [d.doc_id for d in docs],
            },
        }

        if req.include_context:
            resp["retrieval"]["context_preview"] = context[:800]

        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
