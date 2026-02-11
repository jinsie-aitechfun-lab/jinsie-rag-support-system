from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from app.workflow.base_node import BaseNode, NodeResult


class PluginRegistry:
    def __init__(self):
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register(self, name: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        key = (name or "").strip().lower()
        if not key:
            raise ValueError("plugin name required")
        self._handlers[key] = handler

    def get(self, name: str) -> Optional[Callable[[Dict[str, Any]], Any]]:
        return self._handlers.get((name or "").strip().lower())


registry = PluginRegistry()


class PluginNode(BaseNode):
    node_type = "plugin"

    def __init__(self, *, step_id: str, plugin_name: str):
        super().__init__(step_id=step_id)
        self.plugin_name = (plugin_name or "").strip()

    def run(self, state: Dict[str, Any]) -> NodeResult:
        try:
            handler = registry.get(self.plugin_name)
            if handler is None:
                # 占位：明确失败，不 silent
                return NodeResult(ok=False, error=f"plugin not registered: {self.plugin_name}")

            out = handler(state)
            state["plugin_output"] = out
            return NodeResult(ok=True, output={"plugin_output": out})
        except Exception as e:
            return NodeResult(ok=False, error=str(e))
