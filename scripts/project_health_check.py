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


def _is_nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _is_number(v: Any) -> bool:
    # avoid treating bool as int
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, float))


def main() -> int:
    ap = argparse.ArgumentParser(description="Project health check (default: print-only; optional strict assertions).")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001", help="API base url (default: http://127.0.0.1:8001)")
    ap.add_argument("--timeout-s", type=int, default=25, help="HTTP timeout seconds (default: 25)")
    ap.add_argument("--strict", action="store_true", help="Enable strict assertions and exit non-zero on failures.")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    timeout_s = int(args.timeout_s)
    strict = bool(args.strict)

    failures = 0

    def _fail(msg: str) -> None:
        nonlocal failures
        failures += 1
        print(f"[FAIL] {msg}")

    def _assert(cond: bool, msg: str) -> None:
        if not cond:
            _fail(msg)

    # 1) /health
    _print_section("/health")
    url = f"{base_url}/health"
    code, data, wall_ms = _http_json("GET", url, payload=None, timeout_s=timeout_s)
    _print_kv("http_status", code)
    _print_kv("wall_ms", round(wall_ms, 1))
    # keep raw response visible for debugging
    _print_kv("response", data)

    if strict:
        _assert(code == 200, f"/health http_status expected 200, got {code}")

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
    success = _get(data, "success")
    request_id = _get(data, "request_id")
    timestamp = _get(data, "timestamp")
    error_code = _get(data, "error_code")

    _print_kv("success", success)
    _print_kv("request_id", request_id)
    _print_kv("timestamp", timestamp)
    _print_kv("error_code", error_code)

    # data payload (if success)
    resp_data = _get(data, "data", {})
    if isinstance(resp_data, dict):
        _print_kv("retrieval.mode", _get(resp_data, "retrieval.mode"))
        _print_kv("retrieval.hit_count", _get(resp_data, "retrieval.hit_count"))
        _print_kv("metrics", _get(resp_data, "metrics"))
    else:
        _print_kv("data", resp_data)

    if strict:
        _assert(code == 200, f"/v1/rag/run http_status expected 200, got {code}")
        _assert(success is True, f"/v1/rag/run success expected True, got {success}")
        _assert(_is_nonempty_str(request_id), f"/v1/rag/run request_id expected non-empty string, got {request_id}")
        _assert(_is_nonempty_str(timestamp), f"/v1/rag/run timestamp expected non-empty string, got {timestamp}")
        _assert(error_code is None, f"/v1/rag/run error_code expected null (None), got {error_code}")

        metrics = _get(resp_data, "metrics", _MISSING) if isinstance(resp_data, dict) else _MISSING
        total_ms = _get(metrics, "total_ms") if isinstance(metrics, dict) else _MISSING
        retrieval_ms = _get(metrics, "retrieval_ms") if isinstance(metrics, dict) else _MISSING
        llm_ms = _get(metrics, "llm_ms") if isinstance(metrics, dict) else _MISSING

        _assert(_is_number(total_ms) and float(total_ms) >= 0, f"/v1/rag/run metrics.total_ms expected number>=0, got {total_ms}")
        _assert(_is_number(retrieval_ms) and float(retrieval_ms) >= 0, f"/v1/rag/run metrics.retrieval_ms expected number>=0, got {retrieval_ms}")
        _assert(_is_number(llm_ms) and float(llm_ms) >= 0, f"/v1/rag/run metrics.llm_ms expected number>=0, got {llm_ms}")

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

    success = _get(data, "success")
    request_id = _get(data, "request_id")
    timestamp = _get(data, "timestamp")
    error_code = _get(data, "error_code")

    _print_kv("success", success)
    _print_kv("request_id", request_id)
    _print_kv("timestamp", timestamp)
    _print_kv("error_code", error_code)

    resp_data = _get(data, "data", {})
    if isinstance(resp_data, dict):
        engine = _get(resp_data, "meta.engine")
        _print_kv("meta.engine", engine)
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
        engine = _MISSING
        _print_kv("data", resp_data)

    if strict:
        _assert(code == 200, f"/v1/workflow/run http_status expected 200, got {code}")
        _assert(success is True, f"/v1/workflow/run success expected True, got {success}")
        _assert(_is_nonempty_str(request_id), f"/v1/workflow/run request_id expected non-empty string, got {request_id}")
        _assert(_is_nonempty_str(timestamp), f"/v1/workflow/run timestamp expected non-empty string, got {timestamp}")
        _assert(error_code is None, f"/v1/workflow/run error_code expected null (None), got {error_code}")
        _assert(engine == "langgraph", f"/v1/workflow/run data.meta.engine expected 'langgraph', got {engine}")

        metrics = _get(resp_data, "metrics", _MISSING) if isinstance(resp_data, dict) else _MISSING
        total_ms = _get(metrics, "total_ms") if isinstance(metrics, dict) else _MISSING
        retrieval_ms = _get(metrics, "retrieval_ms") if isinstance(metrics, dict) else _MISSING
        llm_ms = _get(metrics, "llm_ms") if isinstance(metrics, dict) else _MISSING

        _assert(_is_number(total_ms) and float(total_ms) >= 0, f"/v1/workflow/run metrics.total_ms expected number>=0, got {total_ms}")
        _assert(_is_number(retrieval_ms) and float(retrieval_ms) >= 0, f"/v1/workflow/run metrics.retrieval_ms expected number>=0, got {retrieval_ms}")
        _assert(_is_number(llm_ms) and float(llm_ms) >= 0, f"/v1/workflow/run metrics.llm_ms expected number>=0, got {llm_ms}")

    print()
    if strict:
        if failures == 0:
            print("[OK] health-check finished (strict).")
            return 0
        print(f"[FAIL] health-check finished (strict), failures={failures}.")
        return 1

    print("[OK] health-check finished (print-only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())