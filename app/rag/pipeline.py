from __future__ import annotations

from pathlib import Path

import jinsie_agent_platform as platform
from jinsie_agent_platform.runner import workflow_runner

from app.rag.llm_chat import chat_answer, has_chat_env
from app.rag.retriever import format_context, keyword_retrieve, vector_retrieve


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


def run_rag_pipeline(
    query: str,
    *,
    retrieval_mode: str = "keyword",
    top_k: int = 3,
    debug: bool = True,
) -> dict:
    """
    Orchestration layer for RAG:
      query -> retrieve -> context -> (llm_chat | workflow_runner) -> result dict

    Return shape:
      {
        "result": <same as old `result`>,
        "docs": <retrieved docs list>,
        "context": <formatted context string>,
        "resp_mode": "keyword" | "vector"
      }
    """
    augmented, docs, context = _augment_query_with_context(
        query, mode=retrieval_mode, top_k=top_k
    )

    # If chat env is ready, use real LLM; otherwise fallback to existing runner behavior.
    if has_chat_env():
        llm = chat_answer(augmented)
        result = {
            "task_status": "COMPLETED",
            "stats": {"total_steps": 1, "ok": 1, "skipped": 0, "failed": 0, "degraded_count": 0},
            "last_step_id": "step_1",
            "tool": "llm_chat",
            "output": {"answer": llm.get("text", "")},
            "model": llm.get("model", ""),
            "usage": llm.get("usage", {}),
        }
    else:
        result = workflow_runner(
            augmented,
            debug=debug,
            prompt_path=_platform_prompt_path(),
        )

    mode_norm = (retrieval_mode or "keyword").strip().lower()
    resp_mode = "vector" if mode_norm == "vector" else "keyword"

    return {"result": result, "docs": docs, "context": context, "resp_mode": resp_mode}
