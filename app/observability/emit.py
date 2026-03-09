from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


def _as_float(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _as_int(v: Any) -> int:
    if v is None:
        return 0
    try:
        return int(v)
    except Exception:
        return 0


def _as_bool_env(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _base_url() -> str:
    # Default to local dev Observability service
    return (os.getenv("OBSERVABILITY_BASE_URL") or "http://127.0.0.1:8003").rstrip("/")


def _timeout_seconds() -> float:
    raw = os.getenv("OBSERVABILITY_TIMEOUT_SECONDS", "0.6")
    try:
        value = float(raw)
    except Exception:
        return 0.6
    if value <= 0:
        return 0.6
    return value


def _enabled() -> bool:
    return _as_bool_env("OBSERVABILITY_ENABLED", True)


def build_invocation_metric(
    *,
    request_id: str,
    engine: str,
    status: str,
    timestamp: str,
    total_ms: Any = 0.0,
    llm_ms: Any = 0.0,
    retrieval_ms: Any = 0.0,
    prompt_tokens: Any = 0,
    completion_tokens: Any = 0,
    total_tokens: Any = 0,
    cost: Any = 0.0,
) -> Dict[str, Any]:
    total_tokens_norm = _as_int(total_tokens)
    prompt_tokens_norm = _as_int(prompt_tokens)
    completion_tokens_norm = _as_int(completion_tokens)

    if total_tokens_norm <= 0:
        total_tokens_norm = prompt_tokens_norm + completion_tokens_norm

    return {
        "request_id": str(request_id),
        "engine": str(engine),
        "status": str(status),
        "timestamp": str(timestamp),
        "total_ms": _as_float(total_ms),
        "llm_ms": _as_float(llm_ms),
        "retrieval_ms": _as_float(retrieval_ms),
        "prompt_tokens": prompt_tokens_norm,
        "completion_tokens": completion_tokens_norm,
        "total_tokens": total_tokens_norm,
        "cost": _as_float(cost),
    }


def _post_metrics_sync(metrics: List[Dict[str, Any]]) -> None:
    """
    Best-effort POST metrics to Observability service.
    Must NEVER break the main request path.
    """
    if not metrics:
        return

    if not _enabled():
        return

    url = f"{_base_url()}/metrics/ingest"
    body = json.dumps(metrics).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # Keep timeout small to avoid blocking (even though we run in a thread)
        with urllib.request.urlopen(req, timeout=_timeout_seconds()) as resp:
            # read to finish request; ignore content
            resp.read()
    except Exception as e:
        logger.warning("observability emit failed: %s", e)


def emit_metrics(metrics: List[Dict[str, Any]]) -> None:
    """
    Fire-and-forget emitter.
    Any exception is swallowed.
    """
    if not metrics:
        return

    if not _enabled():
        return

    try:
        t = threading.Thread(target=_post_metrics_sync, args=(metrics,), daemon=True)
        t.start()
    except Exception as e:
        # Never affect main path
        logger.warning("observability emit thread start failed: %s", e)
        return