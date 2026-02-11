from __future__ import annotations

from typing import Any, Dict

from app.rag.llm_chat import chat_answer, has_chat_env
from app.workflow.base_node import BaseNode, NodeResult


class LLMNode(BaseNode):
    node_type = "llm"

    def __init__(self, *, step_id: str):
        super().__init__(step_id=step_id)

    def run(self, state: Dict[str, Any]) -> NodeResult:
        try:
            if not has_chat_env():
                return NodeResult(
                    ok=False,
                    error="LLM chat env not configured (missing OPENAI_BASE_URL/OPENAI_MODEL/OPENAI_API_KEY etc.)",
                )

            prompt = (state.get("prompt") or "").strip()
            if not prompt:
                return NodeResult(ok=False, error="missing required field: prompt")

            answer = chat_answer(prompt)
            state["answer"] = answer

            return NodeResult(ok=True, output={"answer": answer})
        except Exception as e:
            return NodeResult(ok=False, error=str(e))
