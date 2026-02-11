from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from app.workflow.base_node import BaseNode, NodeResult


class PythonCodeNode(BaseNode):
    node_type = "python_code"

    def __init__(self, *, step_id: str, function_name: str = "echo"):
        super().__init__(step_id=step_id)
        self.function_name = (function_name or "echo").strip().lower()

    def run(self, state: Dict[str, Any]) -> NodeResult:
        try:
            # 最小安全版本：白名单函数映射（不执行任意代码）
            fn = _get_whitelisted_function(self.function_name)
            if fn is None:
                return NodeResult(ok=False, error=f"python function not allowed: {self.function_name}")

            # 约定：输入来自 state["answer"]（LLM 的产物）
            answer = state.get("answer")
            out = fn(answer)

            state["python_output"] = out
            return NodeResult(ok=True, output={"python_output": out})
        except Exception as e:
            return NodeResult(ok=False, error=str(e))


def _get_whitelisted_function(name: str) -> Optional[Callable[[Any], Any]]:
    name_norm = (name or "").strip().lower()

    def echo(x: Any) -> Any:
        return x

    def to_text(x: Any) -> str:
        # 兼容你当前 LLM 返回 dict 的结构（{"text": "..."}）
        if isinstance(x, dict) and "text" in x:
            return str(x["text"])
        return str(x)

    whitelist: Dict[str, Callable[[Any], Any]] = {
        "echo": echo,
        "to_text": to_text,
    }

    return whitelist.get(name_norm)
