from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    title: str
    content: str


# ✅ 先用“内置小知识库”跑通链路（后面再换成真实 KB / 向量库）
_BUILTIN_DOCS: List[RetrievedDoc] = [
    RetrievedDoc(
        doc_id="doc_1",
        title="Echo Tool",
        content="echo_tool: 用于把输入原样输出，常用于调试与链路打通。",
    ),
    RetrievedDoc(
        doc_id="doc_2",
        title="Time Tool",
        content="get_time: 返回当前 UTC 时间字符串，常用于演示工具调用。",
    ),
    RetrievedDoc(
        doc_id="doc_3",
        title="RAG 说明",
        content="RAG = 先检索再生成：把检索到的上下文拼进提示词，让模型基于材料回答。",
    ),
]


def keyword_retrieve(query: str, *, top_k: int = 3) -> List[RetrievedDoc]:
    """
    Extremely simple keyword retriever (MVP).

    Strategy:
    - split query into rough tokens
    - score by token hits in doc content/title
    """
    q = (query or "").strip().lower()
    if not q:
        return []

    tokens = [t for t in q.replace("：", " ").replace("，", " ").split() if t]
    if not tokens:
        tokens = [q]

    scored = []
    for d in _BUILTIN_DOCS:
        hay = f"{d.title}\n{d.content}".lower()
        score = sum(1 for t in tokens if t in hay)
        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def format_context(docs: List[RetrievedDoc]) -> str:
    """
    Convert retrieved docs into a compact context block.
    """
    if not docs:
        return ""

    parts = []
    for d in docs:
        parts.append(f"[{d.doc_id}] {d.title}\n{d.content}")
    return "\n\n".join(parts)
