from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class NodeResult:
    ok: bool
    output: Any = None
    error: Optional[str] = None


class BaseNode:
    """
    Workflow Node base class.

    - Node 只负责单步能力（可测试）
    - Runner 负责编排与可观测（可审计）
    """

    node_type: str = "base"

    def __init__(self, *, step_id: str):
        self.step_id = step_id

    def run(self, state: Dict[str, Any]) -> NodeResult:
        raise NotImplementedError
