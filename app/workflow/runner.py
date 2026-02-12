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

    LLMOps 分层定位：
    - Orchestration Layer（编排层）：负责把各个 node 串成可运行链路，并维护 state/steps 审计
    - Data/Knowledge Layer（数据层）：由具体 node 负责把 docs/context 填入 state（runner 只承载状态流转）
    - Inference Layer（推理层）：由具体 node 负责调用 LLM 产出 answer（runner 只负责调度与记录）
    """

    def __init__(self, nodes: List[BaseNode]):
        self.nodes = nodes

    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        # Orchestration Layer：state 是编排层的“共享状态容器”，贯穿整条链路流转
        state: Dict[str, Any] = dict(initial_state or {})
        # Orchestration Layer：steps 是 LLMOps/LLMOps 的可观测与审计载体（step 级别）
        steps: List[Dict[str, Any]] = []

        # Orchestration Layer：按固定顺序调度 node（最小可运行串行编排，非 DAG）
        for n in self.nodes:
            # 可审计：记录输入摘要（避免把大段内容全打出来）
            # 这些字段分别对应常见 LLMOps 状态面：
            # - query: 用户输入（API 层进入）
            # - prompt: 编排层组装后的提示词（Orchestration）
            # - context: 数据层召回拼装后的上下文（Data/Knowledge）
            input_summary = {
                "keys": sorted(list(state.keys())),
                "query": _safe_summary(state.get("query", "")),
                "prompt": _safe_summary(state.get("prompt", "")),
                "context": _safe_summary(state.get("context", "")),
            }

            t0 = time.time()
            # Orchestration Layer：执行一个 node，让它在 state 上读写并返回 NodeResult
            # - Data/Knowledge node：通常写入 docs/context/retrieval_mode
            # - Inference node：通常写入 answer（或更细粒度中间产物）
            res = n.run(state)
            elapsed_ms = int((time.time() - t0) * 1000)

            # Ops/Observability：记录每一步的输入/输出摘要与耗时，便于故障定位与回放
            step_record = {
                "step_id": n.step_id,
                "node_type": getattr(n, "node_type", "unknown"),
                "ok": bool(res.ok),
                "elapsed_ms": elapsed_ms,
                "input_summary": input_summary,
                "output_summary": _safe_summary(res.output),
            }

            if not res.ok:
                # 失败即停：最小 runner 的故障策略（快速失败 + 可定位）
                step_record["error"] = res.error or "unknown error"
                steps.append(step_record)
                return {
                    "status": "FAILED",
                    "steps": steps,
                    "state_keys": sorted(list(state.keys())),
                }

            steps.append(step_record)

        # Output Shaping（输出整形/后处理）：
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
                # Data/Knowledge Layer 的关键指标：召回模式/命中数/doc_ids
                "mode": state.get("retrieval_mode", ""),
                "hit_count": (len(state.get("docs", []) or [])),
                "doc_ids": [d.doc_id for d in (state.get("docs", []) or [])],
            },
        }
