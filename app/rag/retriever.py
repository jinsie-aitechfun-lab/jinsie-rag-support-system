from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    title: str
    content: str
    source_path: str  # e.g. "docs/knowledge/intro.md"


def _repo_root() -> Path:
    # .../app/rag/retriever.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def _knowledge_dir() -> Path:
    return _repo_root() / "docs" / "knowledge"


def _load_knowledge_md() -> List[RetrievedDoc]:
    """
    Load markdown files from docs/knowledge/*.md as a tiny local KB.
    """
    kdir = _knowledge_dir()
    if not kdir.exists():
        return []

    docs: List[RetrievedDoc] = []
    for i, p in enumerate(sorted(kdir.glob("*.md"))):
        try:
            text = p.read_text(encoding="utf-8").strip()
        except Exception:
            # If a file can't be read, skip it (MVP robustness)
            continue

        rel = p.relative_to(_repo_root()).as_posix()
        title = p.stem  # filename without suffix
        docs.append(
            RetrievedDoc(
                doc_id=f"md_{i+1}",
                title=title,
                content=text,
                source_path=rel,
            )
        )
    return docs


def keyword_retrieve(query: str, *, top_k: int = 3) -> List[RetrievedDoc]:
    """
    Extremely simple keyword retriever (MVP).

    Strategy:
    - split query into rough tokens
    - score by token hits in doc fields (title/content/source_path)
    - if no hit at all, fallback to first top_k docs (so user always gets context)
    """
    q = (query or "").strip().lower()
    if not q:
        return []

    docs = _load_knowledge_md()
    if not docs:
        return []

    tokens = [t for t in q.replace("：", " ").replace("，", " ").replace("。", " ").split() if t]
    if not tokens:
        tokens = [q]

    scored = []
    for d in docs:
        hay = f"{d.doc_id}\n{d.title}\n{d.source_path}\n{d.content}".lower()
        score = sum(1 for t in tokens if t in hay)
        scored.append((score, d))

    # Sort by (score desc, title asc) to be stable
    scored.sort(key=lambda x: (-x[0], x[1].title))

    # If nothing matched, fallback: still return some docs (MVP usability)
    if scored and scored[0][0] == 0:
        return [d for _, d in scored[:top_k]]

    return [d for score, d in scored if score > 0][:top_k]


def format_context(docs: List[RetrievedDoc]) -> str:
    """
    Convert retrieved docs into a compact context block.
    """
    if not docs:
        return ""

    parts = []
    for d in docs:
        parts.append(f"[{d.doc_id}] {d.title} ({d.source_path})\n{d.content}")
    return "\n\n".join(parts)
