from __future__ import annotations

from pathlib import Path

import jinsie_agent_platform as platform
from jinsie_agent_platform.runner import workflow_runner

from app.rag.graph_runner import AnswerNode, GraphRunner, RetrieverNode
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


def _retrieve(query: str, *, mode: str, top_k: int) -> tuple[list, str]:
    """
    Stage 1: retrieve documents.
    Returns: (docs, used_mode)
    """
    mode_norm = (mode or "keyword").strip().lower()

    if mode_norm == "vector":
        docs = vector_retrieve(query, top_k=top_k)
        used_mode = "vector"
    else:
        docs = keyword_retrieve(query, top_k=top_k)
        used_mode = "keyword"

    return docs, used_mode


def _build_prompt(query: str, *, context: str) -> str:
    """
    Stage 2: build augmented prompt.
    If no context, return original query.
    """
    if not context:
        return query

    augmented = (
        "你将获得一段检索到的上下文，请优先基于上下文完成任务。\n\n"
        f"【上下文】\n{context}\n\n"
        f"【用户问题】\n{query}"
    )
    return augmented


def _answer(augmented: str, *, debug: bool) -> dict:
    """
    Stage 3: produce answer result dict.
    Prefer real LLM when env is ready; otherwise fallback to workflow runner.
    """
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
        return result

    result = workflow_runner(
        augmented,
        debug=debug,
        prompt_path=_platform_prompt_path(),
    )
    return result


def _augment_query_with_context(query: str, *, mode: str, top_k: int) -> tuple[str, list, str]:
    # Stage 1: retrieve
    docs, used_mode = _retrieve(query, mode=mode, top_k=top_k)

    # Keep existing behavior: format context
    context = format_context(docs)

    # Stage 2: build prompt
    augmented = _build_prompt(query, context=context)

    if not context:
        # Keep return signature; rag_run will set mode in response
        return query, docs, ""

    # Keep return signature; rag_run will set mode in response
    return augmented, docs, context


def _retriever_node_fn(state: dict) -> dict:
    query = state.get("query", "")
    retrieval_mode = state.get("retrieval_mode", "keyword")
    top_k = int(state.get("top_k", 3))

    augmented, docs, context = _augment_query_with_context(query, mode=retrieval_mode, top_k=top_k)

    mode_norm = (retrieval_mode or "keyword").strip().lower()
    resp_mode = "vector" if mode_norm == "vector" else "keyword"

    return {
        "augmented": augmented,
        "docs": docs,
        "context": context,
        "resp_mode": resp_mode,
    }


def _answer_node_fn(state: dict) -> dict:
    augmented = state.get("augmented", state.get("query", ""))
    debug = bool(state.get("debug", True))
    result = _answer(augmented, debug=debug)
    return {"result": result}


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
    graph = GraphRunner(
        nodes=[
            RetrieverNode(retrieve_fn=_retriever_node_fn),
            AnswerNode(answer_fn=_answer_node_fn),
        ]
    )

    final_state = graph.run(
        {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "debug": debug,
        }
    )

    return {
        "result": final_state.get("result"),
        "docs": final_state.get("docs", []),
        "context": final_state.get("context", ""),
        "resp_mode": final_state.get("resp_mode", "keyword"),
    }
