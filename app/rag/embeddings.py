from __future__ import annotations

import os
from typing import List

import requests


class EmbeddingError(RuntimeError):
    pass


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def _join_embeddings_url(base_url: str) -> str:
    """
    SiliconFlow / OpenAI-compatible base_url styles:
    - https://api.siliconflow.cn/v1  -> embeddings path should be /embeddings
    - https://api.siliconflow.cn     -> embeddings path should be /v1/embeddings

    We normalize both to a correct final URL.
    """
    b = (base_url or "").rstrip("/")
    if b.endswith("/v1"):
        return f"{b}/embeddings"
    return f"{b}/v1/embeddings"


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Minimal OpenAI-compatible embeddings call.

    Env:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (default: https://api.openai.com)
    - OPENAI_EMBEDDING_MODEL (default: text-embedding-3-small)
    """
    if not texts:
        return []

    api_key = _env("OPENAI_API_KEY")
    base_url = _env("OPENAI_BASE_URL", "https://api.openai.com")
    model = _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if not api_key:
        raise EmbeddingError("missing env: OPENAI_API_KEY")

    url = _join_embeddings_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": texts,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        raise EmbeddingError(f"embedding request failed: {e}") from e

    if r.status_code >= 400:
        raise EmbeddingError(f"embedding http {r.status_code}: {r.text[:300]}")

    try:
        data = r.json()
    except Exception as e:
        raise EmbeddingError(f"embedding response is not json: {e}") from e

    items = data.get("data") or []
    if not items:
        raise EmbeddingError("embedding response missing data")

    # Preserve input order by index
    items_sorted = sorted(items, key=lambda x: int(x.get("index", 0)))
    vecs: List[List[float]] = []
    for it in items_sorted:
        emb = it.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise EmbeddingError("invalid embedding vector in response")
        vecs.append(emb)

    return vecs
