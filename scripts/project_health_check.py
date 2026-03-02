from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# sentinel for "missing field" (different from JSON null -> Python None)
_MISSING = object()


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _safe_json_loads(b: bytes) -> Any:
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        # fallback: show raw text for debugging
        try:
            return {"_raw": b.decode("utf-8", errors="replace")}
        except Exception:
            return {"_raw": "<un-decodable>"}


def _http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: int = 20,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, Any], float]:
    body = None
    req_headers = {
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)

    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = Request(url=url, data=body, headers=req_headers, method=method)

    t0 = _now_ms()
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            status = int(getattr(resp, "status", 200))
            data = resp.read()
            t1 = _now_ms()
            return status, _safe_json_loads(data), (t1 - t0)
    except HTTPError as e:
        try:
            data = e.read()
        except Exception:
            data = b""
        t1 = _now_ms()
        return int(e.code), _safe_json_loads(data), (t1 - t0)
    except URLError as e:
        t1 = _now_ms()
        return 0, {"_error": f"URLError: {e}"}, (t1 - t0)
    except Exception as e:
        t1 = _now_ms()
        return 0, {"_error": f"Exception: {e}"}, (t1 - t0)


def _get(d: Dict[str, Any], path: str, default: Any = _MISSING) -> Any:
    """
    Safe getter for nested dicts using dot path, e.g. 'data.meta.engine'
    """
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict):
            return default
        if p not in cur:
            return default
        cur = cur[p]
    return cur


def _print_kv(k: str, v: Any, indent: int = 2) -> None:
    sp = " " * indent
    if v is _MISSING:
        print(f"{sp}{k}: <missing>")
        return
    # JSON null -> Python None: this is "present but null"
    if v is None:
        print(f"{sp}{k}: null")
        return
    if isinstance(v, (dict, list)):
        try:
            s = json.dumps(v, ensure_ascii=False)
        except Exception:
            s = str(v)
        print(f"{sp}{k}: {s}")
        return
    print(f"{sp}{k}: {v}")


def _print_section(title: str) -> None:
    print()
    print(f"== {title} ==")


def main() -> int:
    ap = argparse.ArgumentParser(description="Project health check (no assertions, print-only).")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001", help="API base url (default: http://127.0.0.1:8001)")
    ap.add_argument("--timeout-s", type=int, default=25, help="HTTP timeout seconds (default: 25)")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    timeout_s = int(args.timeout_s)

    # 1) /health
    _print_section("/health")
    url = f"{base_url}/health"
    code, data, wall_ms = _http_json("GET", url, payload=None, timeout_s=timeout_s)
    _print_kv("http_status", code)
    _print_kv("wall_ms", round(wall_ms, 1))
    # keep raw response visible for debugging
    _print_kv("response", data)

    # 2) /v1/rag/run keyword
    _print_section("/v1/rag/run [keyword]")
    url = f"{base_url}/v1/rag/run"
    payload = {
        "query": "health-check: keyword",
        "mode": "keyword",
        "include_context": False,
    }
    code, data, wall_ms = _http_json("POST", url, payload=payload, timeout_s=timeout_s)
    _print_kv("http_status", code)
    _print_kv("wall_ms", round(wall_ms, 1))

    # Contract-ish fields (print only)
    _print_kv("success", _get(data, "success"))
    _print_kv("request_id", _get(data, "request_id"))
    _print_kv("timestamp", _get(data, "timestamp"))
    _print_kv("error_code", _get(data, "error_code"))

    # data payload (if success)
    resp_data = _get(data, "data", {})
    if isinstance(resp_data, dict):
        _print_kv("retrieval.mode", _get(resp_data, "retrieval.mode"))
        _print_kv("retrieval.hit_count", _get(resp_data, "retrieval.hit_count"))
        _print_kv("metrics", _get(resp_data, "metrics"))
    else:
        _print_kv("data", resp_data)

    # 3) /v1/workflow/run langgraph
    _print_section("/v1/workflow/run [langgraph]")
    url = f"{base_url}/v1/workflow/run"
    payload = {
        "query": "health-check: langgraph workflow",
        # keep schema minimal; server should default reasonable behavior
    }
    headers = {
        "x-workflow-engine": "langgraph",
    }
    code, data, wall_ms = _http_json("POST", url, payload=payload, timeout_s=timeout_s, headers=headers)
    _print_kv("http_status", code)
    _print_kv("wall_ms", round(wall_ms, 1))
    _print_kv("success", _get(data, "success"))
    _print_kv("request_id", _get(data, "request_id"))
    _print_kv("timestamp", _get(data, "timestamp"))
    _print_kv("error_code", _get(data, "error_code"))

    resp_data = _get(data, "data", {})
    if isinstance(resp_data, dict):
        _print_kv("meta.engine", _get(resp_data, "meta.engine"))
        _print_kv("status", _get(resp_data, "status"))
        _print_kv("metrics", _get(resp_data, "metrics"))
        steps = _get(resp_data, "steps", [])
        if isinstance(steps, list):
            _print_kv("steps.count", len(steps))
            # print a short steps summary
            for s in steps[:8]:
                if not isinstance(s, dict):
                    continue
                sid = s.get("step_id")
                ok = s.get("ok")
                elapsed = s.get("elapsed_ms")
                print(f"  step: {sid}  ok={ok}  elapsed_ms={elapsed}")
        else:
            _print_kv("steps", steps)
    else:
        _print_kv("data", resp_data)

    print()
    print("[OK] health-check finished (print-only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())