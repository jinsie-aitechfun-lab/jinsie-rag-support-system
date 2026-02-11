from __future__ import annotations

import time
from typing import Any, Dict, List

from app.workflow.base_node import BaseNode


def _safe_summary(v: Any, limit: int = 200) -> str:
    try:
        s = str(v)
        return s[:limit] + ("…" if len(s) > limit else "")
    except Exception:
        return "<unprintable>"


class WorkflowRunner:
    """
    最小可运行 runner：固定顺序串起来（非 DAG）
    目标：可运行 + 可审计 + 失败可定位（step 级别）
    """

    def __init__(self, nodes: List[BaseNode]):
        self.nodes = nodes

    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(initial_state or {})
        steps: List[Dict[str, Any]] = []

        for n in self.nodes:
            # 可审计：记录输入摘要（避免把大段内容全打出来）
            input_summary = {
                "keys": sorted(list(state.keys())),
                "query": _safe_summary(state.get("query", "")),
                "prompt": _safe_summary(state.get("prompt", "")),
                "context": _safe_summary(state.get("context", "")),
            }

            t0 = time.time()
            res = n.run(state)
            elapsed_ms = int((time.time() - t0) * 1000)

            step_record = {
                "step_id": n.step_id,
                "node_type": getattr(n, "node_type", "unknown"),
                "ok": bool(res.ok),
                "elapsed_ms": elapsed_ms,
                "input_summary": input_summary,
                "output_summary": _safe_summary(res.output),
            }

            if not res.ok:
                step_record["error"] = res.error or "unknown error"
                steps.append(step_record)
                return {
                    "status": "FAILED",
                    "steps": steps,
                    "state_keys": sorted(list(state.keys())),
                }

            steps.append(step_record)

        # 最终输出优先级：python_output(文本后处理) > answer(可能是dict/str)
        final_text = state.get("python_output")
        if final_text is None:
            ans = state.get("answer", "")
            if isinstance(ans, dict) and "text" in ans:
                final_text = ans.get("text", "")
            else:
                final_text = ans

        return {
            "status": "COMPLETED",
            "steps": steps,
            "answer": state.get("answer", ""),
            "final_text": final_text,
            "retrieval": {
                "mode": state.get("retrieval_mode", ""),
                "hit_count": (len(state.get("docs", []) or [])),
                "doc_ids": [d.doc_id for d in (state.get("docs", []) or [])],
            },
        }
