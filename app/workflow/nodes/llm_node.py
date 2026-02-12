from __future__ import annotations

from typing import Any, Dict

from app.rag.llm_chat import chat_answer, has_chat_env
from app.workflow.base_node import BaseNode, NodeResult


class LLMNode(BaseNode):
    node_type = "llm"

    def __init__(self, *, step_id: str):
        super().__init__(step_id=step_id)

    def run(self, state: Dict[str, Any]) -> NodeResult:
        """
        LLMOps 分层定位：Inference Layer（推理层）

        输入契约（从 state 读取）：
        - prompt: str（必填，由 TemplateNode 产出）
        - 环境变量：OPENAI_*（由 has_chat_env 校验）

        输出契约（写回 state）：
        - answer: dict | str（chat_answer 的返回结构，供 runner / 后处理节点使用）
        - llm_model: str（可选，推理指标）
        - llm_usage: dict（可选，推理指标）

        说明：
        - 这是 RAG 的“推理阶段”，只负责把 prompt 发给 LLM 并写回 answer
        - 不负责召回 docs/context（RetrieverNode 负责）
        - 不负责 prompt 组装（TemplateNode 负责）
        """
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

            # 推理层可观测性：把模型与 token usage 写入 state，便于 runner/上层聚合指标
            if isinstance(answer, dict):
                state["llm_model"] = answer.get("model", "")
                state["llm_usage"] = answer.get("usage", {}) or {}

            return NodeResult(ok=True, output={"answer": answer})
        except Exception as e:
            return NodeResult(ok=False, error=str(e))
