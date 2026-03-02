from __future__ import annotations

import time
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from app.workflow.base_node import BaseNode


def _safe_summary(v: Any, limit: int = 200) -> str:
    try:
        s = str(v)
        return s[:limit] + ("…" if len(s) > limit else "")
    except Exception:
        return "<unprintable>"


def run_langgraph_workflow(nodes: List[BaseNode], initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph 版本：最小串行编排（与 legacy WorkflowRunner 行为对齐）
    目标：
    - 不改现有 schema
    - 仅替换“执行引擎”（编排壳）
    - steps 结构与 legacy 对齐，供 metrics 真值化复用
    """
    state: Dict[str, Any] = dict(initial_state or {})
    steps: List[Dict[str, Any]] = []

    # 用闭包把 node 执行封进 graph node
    def _make_node_fn(n: BaseNode):
        def _fn(s: Dict[str, Any]) -> Dict[str, Any]:
            # 可审计：记录输入摘要（避免把大段内容全打出来）
            input_summary = {
                "keys": sorted(list(s.keys())),
                "query": _safe_summary(s.get("query", "")),
                "prompt": _safe_summary(s.get("prompt", "")),
                "context": _safe_summary(s.get("context", "")),
            }

            # ✅ 真值化：使用 perf_counter + 小数 ms，避免 <1ms 被 int 截断为 0
            t0 = time.perf_counter()
            res = n.run(s)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

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
                # 将失败标记写回 state，供后续统一收口
                s["__workflow_failed__"] = True
                s["__workflow_error__"] = step_record.get("error")
                steps.append(step_record)
                return s

            steps.append(step_record)
            return s

        return _fn

    # 构建最小串行 graph：n1 -> n2 -> ... -> END
    g = StateGraph(dict)

    prev_name = None
    for idx, n in enumerate(nodes, start=1):
        name = f"n{idx}_{n.step_id}"
        g.add_node(name, _make_node_fn(n))
        if prev_name is None:
            g.set_entry_point(name)
        else:
            g.add_edge(prev_name, name)
        prev_name = name

    if prev_name is not None:
        g.add_edge(prev_name, END)

    app = g.compile()
    final_state = app.invoke(state)

    # 若某一步失败，返回 FAILED（与 legacy 对齐）
    if final_state.get("__workflow_failed__"):
        return {
            "status": "FAILED",
            "steps": steps,
            "state_keys": sorted(list(final_state.keys())),
        }

    # Output Shaping（与 legacy 对齐）
    final_text = final_state.get("python_output")
    if final_text is None:
        ans = final_state.get("answer", "")
        if isinstance(ans, dict) and "text" in ans:
            final_text = ans.get("text", "")
        else:
            final_text = ans

    return {
        "status": "COMPLETED",
        "steps": steps,
        "answer": final_state.get("answer", ""),
        "final_text": final_text,
        "retrieval": {
            "mode": final_state.get("retrieval_mode", ""),
            "hit_count": (len(final_state.get("docs", []) or [])),
            "doc_ids": [d.doc_id for d in (final_state.get("docs", []) or [])],
        },
    }