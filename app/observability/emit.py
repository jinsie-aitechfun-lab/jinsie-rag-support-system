from __future__ import annotations

import json
import os
import threading
import urllib.request
from typing import Any, Dict, List


def _base_url() -> str:
    # Default to local dev Observability service
    return (os.getenv("OBSERVABILITY_BASE_URL") or "http://127.0.0.1:8003").rstrip("/")


def _post_metrics_sync(metrics: List[Dict[str, Any]]) -> None:
    """
    Best-effort POST metrics to Observability service.
    Must NEVER break the main request path.
    """
    if not metrics:
        return

    url = f"{_base_url()}/metrics/ingest"
    body = json.dumps(metrics).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Keep timeout small to avoid blocking (even though we run in a thread)
    with urllib.request.urlopen(req, timeout=0.6) as resp:
        # read to finish request; ignore content
        resp.read()


def emit_metrics(metrics: List[Dict[str, Any]]) -> None:
    """
    Fire-and-forget emitter.
    Any exception is swallowed.
    """
    try:
        t = threading.Thread(target=_post_metrics_sync, args=(metrics,), daemon=True)
        t.start()
    except Exception:
        # Never affect main path
        return