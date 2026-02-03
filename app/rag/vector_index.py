from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from app.rag.embeddings import embed_texts

try:
    import chromadb  # type: ignore
except Exception as e:  # pragma: no cover
    chromadb = None  # type: ignore
    _CHROMA_IMPORT_ERROR = e
else:
    _CHROMA_IMPORT_ERROR = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    text: str


def _repo_root() -> Path:
    # .../app/rag/vector_index.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def _persist_dir() -> Path:
    return _repo_root() / "var" / "chroma"


def _chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """
    Very small chunker (MVP):
    - normalize whitespace
    - cut by fixed size with overlap
    """
    t = (text or "").strip()
    if not t:
        return []

    # Soft normalization
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")

    chunks: List[str] = []
    i = 0
    n = len(t)
    while i < n:
        j = min(i + chunk_size, n)
        piece = t[i:j].strip()
        if piece:
            chunks.append(piece)
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks


def build_chunks(docs: List[Dict]) -> List[Chunk]:
    """
    docs: list of dict with keys: doc_id/title/content/source_path
    """
    out: List[Chunk] = []
    for d in docs:
        doc_id = str(d["doc_id"])
        title = str(d.get("title", ""))
        source_path = str(d.get("source_path", ""))
        content = str(d.get("content", ""))

        parts = _chunk_text(content)
        for idx, piece in enumerate(parts, start=1):
            chunk_id = f"{doc_id}__c{idx}"
            out.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    title=title,
                    source_path=source_path,
                    text=piece,
                )
            )
    return out


class ChromaVectorIndex:
    """
    Minimal persistent Chroma index (MVP):
    - persistent dir: var/chroma
    - collection: knowledge_md_v1
    - lazy build on first query
    """

    def __init__(self) -> None:
        if chromadb is None:
            raise RuntimeError(f"chroma not installed/importable: {_CHROMA_IMPORT_ERROR}")

        self._persist = _persist_dir()
        self._persist.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._persist))
        self._col = self._client.get_or_create_collection(
            name="knowledge_md_v1",
            metadata={"hnsw:space": "cosine"},
        )

    def _is_built(self) -> bool:
        try:
            cnt = self._col.count()
        except Exception:
            return False
        return cnt > 0

    def ensure_built(self, docs: List[Dict]) -> None:
        if self._is_built():
            return

        chunks = build_chunks(docs)
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vecs = embed_texts(texts)

        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {"doc_id": c.doc_id, "title": c.title, "source_path": c.source_path}
            for c in chunks
        ]

        # Upsert embeddings
        self._col.upsert(ids=ids, embeddings=vecs, documents=texts, metadatas=metadatas)

    def query(self, query: str, *, top_k: int = 3) -> List[Tuple[str, str, Dict]]:
        """
        Return list of (chunk_id, chunk_text, metadata)
        """
        q = (query or "").strip()
        if not q:
            return []

        q_vec = embed_texts([q])[0]
        res = self._col.query(
            query_embeddings=[q_vec],
            n_results=max(1, int(top_k)),
            include=["documents", "metadatas", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        out = []
        for i in range(min(len(ids), len(docs), len(metas))):
            out.append((ids[i], docs[i], metas[i] or {}))
        return out
