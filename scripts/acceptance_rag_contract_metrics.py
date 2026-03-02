#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Acceptance: Contract + Metrics guardrail (Project 2)

Constraints:
- No API response changes
- No runner changes
- No new dependencies
- Add only ONE script file under scripts/

Usage:
  python scripts/acceptance_rag_contract_metrics.py --base-url http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


# ---------------------------
# tiny http helpers (stdlib)
# ---------------------------

def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: int = 30) -> Tuple[int, Dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        data = {}
        try:
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {"_raw": raw}
        return e.code, data
    except Exception as e:
        raise RuntimeError(f"HTTP POST failed: {url} -> {e}") from e


# ---------------------------
# assertions & parsing
# ---------------------------

class _AcceptanceFail(RuntimeError):
    def __init__(self, tag: str, msg: str) -> None:
        super().__init__(msg)
        self.tag = tag
        self.msg = msg


def _die(tag: str, msg: str) -> None:
    # Do NOT sys.exit here; let main() print SUMMARY + FAILURE DETAILS and then exit with same code.
    raise _AcceptanceFail(tag=tag, msg=msg)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _get_first_number(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    """
    Try a few possible key names for robustness without changing contract.
    Returns float if found and parseable, else None.
    """
    for k in keys:
        if k in d:
            v = d.get(k)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v)
                except Exception:
                    return None
    return None


def _find_metrics(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Locate a dict named 'metrics' anywhere one level down, without guessing deep structures.
    """
    if isinstance(obj, dict):
        if isinstance(obj.get("metrics"), dict):
            return obj["metrics"]
        # one-level scan
        for _, v in obj.items():
            if isinstance(v, dict) and isinstance(v.get("metrics"), dict):
                return v["metrics"]
    return None


def _find_timestamp(obj: Any) -> Optional[str]:
    """
    Locate 'timestamp' field (string) either at top-level or one-level down.
    """
    if isinstance(obj, dict):
        ts = obj.get("timestamp")
        if isinstance(ts, str):
            return ts
        for _, v in obj.items():
            if isinstance(v, dict):
                ts2 = v.get("timestamp")
                if isinstance(ts2, str):
                    return ts2
    return None


def _assert_utc_iso8601(ts: str) -> None:
    """
    Accepts:
      - ...Z
      - ...+00:00
    Rejects naive timestamps.
    """
    s = ts.strip()
    # Normalize Z to +00:00 for fromisoformat
    if s.endswith("Z"):
        s_norm = s[:-1] + "+00:00"
    else:
        s_norm = s

    try:
        dt = datetime.fromisoformat(s_norm)
    except Exception:
        _die("timestamp", f"timestamp is not ISO8601 parseable: {ts!r}")

    if dt.tzinfo is None:
        _die("timestamp", f"timestamp must be timezone-aware (UTC), got naive: {ts!r}")

    # must be UTC (offset 0)
    offset = dt.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        _die("timestamp", f"timestamp must be UTC (Z/+00:00). got: {ts!r}")

    # additionally ensure "looks like" UTC form
    if not (s.endswith("Z") or s.endswith("+00:00")):
        _die("timestamp", f"timestamp must end with 'Z' or '+00:00'. got: {ts!r}")


def _print_metrics(metrics: Dict[str, Any]) -> None:
    total_ms = _get_first_number(metrics, ("total_ms", "total", "total_time_ms"))
    retrieval_ms = _get_first_number(metrics, ("retrieval_ms", "retrieve_ms", "retrieval_time_ms"))
    llm_ms = _get_first_number(metrics, ("llm_ms", "model_ms", "generation_ms", "llm_time_ms"))

    # Always print, even if some are missing, to make drift visible.
    print("  metrics:")
    print(f"    total_ms:     {total_ms}")
    print(f"    retrieval_ms: {retrieval_ms}")
    print(f"    llm_ms:       {llm_ms}")


def _assert_positive_metric(metrics: Dict[str, Any], key_candidates: Tuple[str, ...], label: str, tag: str) -> None:
    v = _get_first_number(metrics, key_candidates)
    if v is None:
        _die(tag, f"{label} missing in metrics (checked keys={list(key_candidates)})")
    if v <= 0:
        _die(tag, f"{label} must be > 0, got {v}")


# ---------------------------
# cases
# ---------------------------

@dataclass
class CaseResult:
    name: str
    ok: bool
    fail_tag: Optional[str] = None


def case_rag(base_url: str, mode: str, timeout_s: int) -> CaseResult:
    """
    mode: keyword | vector
    """
    print(f"\n== CASE: /v1/rag/run [{mode}] ==")
    url = f"{base_url.rstrip('/')}/v1/rag/run"

    # Minimal payload. If your API requires different fields, we do NOT guess here;
    # we fail fast with actionable output.
    payload = {
        "query": "acceptance: contract+metrics guardrail",
        "mode": mode,
    }

    code, data = _http_post_json(url, payload, timeout_s=timeout_s)
    print(f"  http_status: {code}")

    if code != 200:
        print("  response:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("[FAIL] http_status != 200")
        _die("http_status", f"/v1/rag/run [{mode}] expected 200, got {code}")

    # Contract checks
    ts = _find_timestamp(data)
    if not ts:
        print("  response:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("[FAIL] timestamp missing")
        _die("timestamp", "timestamp not found in response (top-level or one-level down)")

    _assert_utc_iso8601(ts)
    _ok(f"timestamp is UTC ISO8601: {ts}")

    metrics = _find_metrics(data)
    if not metrics:
        print("  response:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("[FAIL] metrics missing")
        _die("metrics", "metrics not found in response (expected field: metrics)")

    _print_metrics(metrics)

    # Metrics assertions (per your Day35 spec)
    _assert_positive_metric(metrics, ("retrieval_ms", "retrieve_ms", "retrieval_time_ms"), "retrieval_ms", "retrieval_ms")
    _ok("retrieval_ms > 0")

    return CaseResult(name=f"rag_{mode}", ok=True)


def case_workflow(base_url: str, timeout_s: int) -> CaseResult:
    print("\n== CASE: /v1/workflow/run ==")
    url = f"{base_url.rstrip('/')}/v1/workflow/run"

    # Minimal payload. If your API requires different fields, we do NOT guess.
    payload = {
        "query": "acceptance: workflow contract+metrics guardrail",
    }

    code, data = _http_post_json(url, payload, timeout_s=timeout_s)
    print(f"  http_status: {code}")

    if code != 200:
        print("  response:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("[FAIL] http_status != 200")
        _die("http_status", f"/v1/workflow/run expected 200, got {code}")

    metrics = _find_metrics(data)
    if not metrics:
        print("  response:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("[FAIL] metrics missing")
        _die("metrics", "metrics not found in workflow response (expected field: metrics)")

    _print_metrics(metrics)

    # Minimal requirement: total_ms must be > 0 and not a placeholder.
    _assert_positive_metric(metrics, ("total_ms", "total", "total_time_ms"), "total_ms", "total_ms")
    _ok("total_ms > 0")

    return CaseResult(name="workflow_run", ok=True)


# ---------------------------
# main
# ---------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8001")
    parser.add_argument("--timeout", type=int, default=30, help="request timeout seconds")
    parser.add_argument(
        "--baseline-wall-time-s",
        type=float,
        default=None,
        help="optional baseline wall_time_s for manual regression visibility (no fail), e.g. 15.08",
    )
    parser.add_argument(
        "--drift-threshold-pct",
        type=float,
        default=20.0,
        help="warn when wall_time drift exceeds this percent (default: 20)",
    )
    args = parser.parse_args()

    start = time.time()

    results = []
    failure_pairs = []  # (case_name, fail_tag)

    def _run_case(fn, case_name_hint: str) -> None:
        try:
            r = fn()
            results.append(r)
        except _AcceptanceFail as e:
            results.append(CaseResult(name=case_name_hint, ok=False, fail_tag=e.tag))
            failure_pairs.append((case_name_hint, e.tag))
        except Exception as e:
            # Keep failure readable even when it's a transport/runtime error.
            print(f"[FAIL] exception: {e}")
            results.append(CaseResult(name=case_name_hint, ok=False, fail_tag="exception"))
            failure_pairs.append((case_name_hint, "exception"))

    _run_case(lambda: case_rag(args.base_url, mode="keyword", timeout_s=args.timeout), "rag_keyword")
    _run_case(lambda: case_rag(args.base_url, mode="vector", timeout_s=args.timeout), "rag_vector")
    _run_case(lambda: case_workflow(args.base_url, timeout_s=args.timeout), "workflow_run")

    ok_count = sum(1 for r in results if r.ok)
    total = len(results)
    elapsed = time.time() - start

    print("\n== SUMMARY ==")
    for r in results:
        if r.ok:
            print(f"  - {r.name}: PASS")
        else:
            print(f"  - {r.name}: FAIL ({r.fail_tag})")
    print(f"  passed: {ok_count}/{total}")
    print(f"  wall_time_s: {elapsed:.2f}")

    # Baseline regression visibility (manual, non-failing)
    if args.baseline_wall_time_s is not None and args.baseline_wall_time_s > 0:
        baseline = float(args.baseline_wall_time_s)
        drift_pct = ((elapsed - baseline) / baseline) * 100.0
        print(f"  baseline_wall_time_s: {baseline:.2f}")
        print(f"  wall_time_drift_pct: {drift_pct:+.1f}")
        if abs(drift_pct) >= float(args.drift_threshold_pct):
            print(f"[WARN] wall_time drift >= {float(args.drift_threshold_pct):.0f}% (manual visibility only)")

    if failure_pairs:
        print("\n== FAILURE DETAILS ==")
        for case_name, tag in failure_pairs:
            print(f"  - {case_name} \u2192 {tag}")
        return 2

    _ok("Acceptance guardrail complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())