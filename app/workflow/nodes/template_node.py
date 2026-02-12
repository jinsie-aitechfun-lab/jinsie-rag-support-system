from __future__ import annotations

from typing import Any, Dict

from app.workflow.base_node import BaseNode, NodeResult


class TemplateNode(BaseNode):
    node_type = "template"

    def __init__(self, *, step_id: str, template: str):
        super().__init__(step_id=step_id)
        self.template = template

    def run(self, state: Dict[str, Any]) -> NodeResult:
        """
        LLMOps 分层定位：Orchestration Layer（编排层）

        输入契约（从 state 读取）：
        - template.format(**state) 所需的变量（例如：query/context/...）

        输出契约（写回 state）：
        - prompt: str（组装后的最终提示词，供 LLMNode 推理使用）

        说明：
        - 这是 RAG 的“提示词编排阶段”，把 query + context 等组合成可喂给 LLM 的 prompt
        - 不负责召回 docs/context（RetrieverNode 负责）
        - 不负责调用 LLM（LLMNode 负责）
        """
        try:
            # 约定：template 可用的变量来自 state
            prompt = self.template.format(**state)
            state["prompt"] = prompt
            return NodeResult(ok=True, output={"prompt": prompt})
        except KeyError as e:
            return NodeResult(ok=False, error=f"missing template variable: {e}")
        except Exception as e:
            return NodeResult(ok=False, error=str(e))
