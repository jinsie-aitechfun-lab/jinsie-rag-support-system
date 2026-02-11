from __future__ import annotations

from typing import Any, Dict

from app.rag.retriever import format_context, keyword_retrieve, vector_retrieve
from app.workflow.base_node import BaseNode, NodeResult


class RetrieverNode(BaseNode):
    node_type = "retriever"

    def __init__(self, *, step_id: str, mode: str = "keyword", top_k: int = 3):
        super().__init__(step_id=step_id)
        self.mode = (mode or "keyword").strip().lower()
        self.top_k = int(top_k or 3)

    def run(self, state: Dict[str, Any]) -> NodeResult:
        try:
            query = (state.get("query") or "").strip()
            if not query:
                return NodeResult(ok=False, error="missing required field: query")

            if self.mode == "vector":
                docs = vector_retrieve(query, top_k=self.top_k)
                used_mode = "vector"
            else:
                docs = keyword_retrieve(query, top_k=self.top_k)
                used_mode = "keyword"

            context = format_context(docs)

            state["retrieval_mode"] = used_mode
            state["docs"] = docs
            state["context"] = context

            return NodeResult(
                ok=True,
                output={
                    "mode": used_mode,
                    "hit_count": len(docs),
                    "doc_ids": [d.doc_id for d in docs],
                    "context_preview": (context[:800] if context else ""),
                },
            )
        except Exception as e:
            return NodeResult(ok=False, error=str(e))
