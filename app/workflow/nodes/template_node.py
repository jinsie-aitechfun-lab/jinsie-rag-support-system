from __future__ import annotations

from typing import Any, Dict

from app.workflow.base_node import BaseNode, NodeResult


class TemplateNode(BaseNode):
    node_type = "template"

    def __init__(self, *, step_id: str, template: str):
        super().__init__(step_id=step_id)
        self.template = template

    def run(self, state: Dict[str, Any]) -> NodeResult:
        try:
            # 约定：template 可用的变量来自 state
            prompt = self.template.format(**state)
            state["prompt"] = prompt
            return NodeResult(ok=True, output={"prompt": prompt})
        except KeyError as e:
            return NodeResult(ok=False, error=f"missing template variable: {e}")
        except Exception as e:
            return NodeResult(ok=False, error=str(e))
