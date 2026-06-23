#!/usr/bin/env python3
"""Run one guarded EasyICU WebServer external-provider point-fire."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


SAFE_PROVIDER_KEYS = {
    "provider",
    "external",
    "ai_enabled",
    "per_run_opt_in",
    "canonical_opt_in_source",
    "canonical_opt_in_passed",
    "provider_gate",
    "provider_gate_order",
    "credentials_attempted",
    "credentials_loaded",
    "credential_source",
    "credential_fingerprint",
    "base_url_source",
    "base_url_endpoint",
    "base_url_configured",
    "model_source",
    "client",
    "client_constructed",
    "mock_calls",
    "external_calls",
    "max_external_calls_per_run",
    "max_output_tokens",
    "usage",
}


def request_json(
    method: str,
    base_url: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    *,
    timeout: int = 30,
) -> Dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method.upper(),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - localhost tool
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {_safe_error_text(text)}") from exc
    return json.loads(payload or "{}")


def wait_for_job(base_url: str, job_id: str, *, timeout_sec: int = 120) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        snapshot = request_json("GET", base_url, f"/api/jobs/{job_id}", timeout=10)
        if snapshot.get("status") != "running":
            return snapshot
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_sec}s")


def safe_summary(snapshot: Dict[str, Any], review: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = snapshot.get("result") if isinstance(snapshot.get("result"), dict) else {}
    gate = result.get("gate") if isinstance(result.get("gate"), dict) else {}
    provider = result.get("provider") if isinstance(result.get("provider"), dict) else {}
    strict = result.get("strict_evidence_audit") if isinstance(result.get("strict_evidence_audit"), dict) else {}
    checks = gate.get("checks") if isinstance(gate.get("checks"), list) else []
    privacy_check = next(
        (
            row for row in checks
            if isinstance(row, dict) and row.get("id") == "no_patient_rows_persisted"
        ),
        {},
    )
    readiness = review.get("readiness") if isinstance(review, dict) and isinstance(review.get("readiness"), dict) else {}
    return {
        "ok": snapshot.get("status") == "done",
        "job_id": snapshot.get("id"),
        "job_status": snapshot.get("status"),
        "run_id": result.get("run_id"),
        "run_type": result.get("run_type"),
        "project_dir": result.get("project_dir"),
        "provider": sanitize_provider(provider),
        "gate": {
            "status": gate.get("status"),
            "reason": gate.get("reason"),
            "reportable": bool(gate.get("reportable")),
            "draft_unlocked": bool(gate.get("draft_unlocked")),
            "checks_total": len(checks),
            "checks_passed": sum(1 for row in checks if isinstance(row, dict) and row.get("passed")),
        },
        "strict_evidence": {
            "claims_passed": strict.get("claims_passed"),
            "sentences_passed": strict.get("sentences_passed"),
            "missing_evidence": strict.get("missing_evidence", []),
            "unbound_claims": strict.get("unbound_claims", []),
            "unbound_sentences": strict.get("unbound_sentences", []),
        },
        "privacy": {
            "passed": privacy_check.get("passed"),
            "scanned_artifacts": privacy_check.get("scanned_artifacts"),
            "row_level_markers": privacy_check.get("row_level_markers", []),
        },
        "readiness": {
            "status": readiness.get("status"),
            "signable": readiness.get("signable"),
            "reportable": readiness.get("reportable"),
            "draft_unlocked": readiness.get("draft_unlocked"),
        },
        "uploads": result.get("uploads"),
        "tokens": result.get("tokens"),
        "secrets_returned": False,
    }


def sanitize_provider(provider: Dict[str, Any]) -> Dict[str, Any]:
    return {key: provider.get(key) for key in sorted(SAFE_PROVIDER_KEYS) if key in provider}


def safety_failures(summary: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    provider = summary.get("provider") if isinstance(summary.get("provider"), dict) else {}
    gate = summary.get("gate") if isinstance(summary.get("gate"), dict) else {}
    privacy = summary.get("privacy") if isinstance(summary.get("privacy"), dict) else {}
    if provider.get("external_calls") != 1:
        failures.append("expected exactly one external provider call")
    if provider.get("client_constructed") is not True:
        failures.append("provider client was not constructed")
    if gate.get("reportable") is not False:
        failures.append("gate.reportable must remain false")
    if gate.get("draft_unlocked") is not False:
        failures.append("gate.draft_unlocked must remain false")
    if privacy.get("passed") is not True:
        failures.append("privacy scan did not pass")
    if summary.get("uploads") not in {0, None}:
        failures.append("uploads must remain zero")
    if summary.get("tokens") not in {0, None}:
        failures.append("top-level tokens must remain zero")
    return failures


def _safe_error_text(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {"error": text[:400]}
    return json.dumps(_strip_unsafe_values(payload), ensure_ascii=False)


def _strip_unsafe_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _strip_unsafe_values(v)
            for k, v in value.items()
            if str(k) not in {"api_key", "Authorization", "base_url"}
        }
    if isinstance(value, list):
        return [_strip_unsafe_values(v) for v in value]
    if isinstance(value, str) and value.startswith("sk-"):
        return "[redacted]"
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--study-id", default="sepsis")
    parser.add_argument("--mode", default="analysis")
    parser.add_argument("--question", default="Run one bounded provider scaffold from the active export.")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--enable-ai-setting", action="store_true")
    parser.add_argument("--confirm-external-call", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.confirm_external_call:
        print(json.dumps({
            "ok": False,
            "error": "confirm_external_call_required",
            "hint": "pass --confirm-external-call after provider readiness is true",
        }, indent=2), file=sys.stderr)
        return 2
    if args.enable_ai_setting:
        request_json("POST", args.server, "/api/settings", {"ai_enabled": True})
    status = request_json("GET", args.server, f"/api/agent-runs/provider-status?provider={args.provider}")
    provider_status = status.get("provider_status") if isinstance(status.get("provider_status"), dict) else {}
    if not provider_status.get("ready"):
        print(json.dumps({
            "ok": False,
            "error": "provider_not_ready",
            "provider_status": _strip_unsafe_values(provider_status),
        }, indent=2, ensure_ascii=False))
        return 2
    body = {
        "study_id": args.study_id,
        "mode": args.mode,
        "run_type": "full",
        "llm_provider": args.provider,
        "external_llm_opt_in": True,
        "question": args.question,
    }
    if args.project_root:
        body["project_root"] = args.project_root
    started = request_json("POST", args.server, "/api/jobs/agent-run", body)
    snapshot = wait_for_job(args.server, str(started.get("job_id")), timeout_sec=args.timeout_sec)
    review = None
    result = snapshot.get("result") if isinstance(snapshot.get("result"), dict) else {}
    if result.get("project_dir"):
        review = request_json("POST", args.server, "/api/agent-runs/review", {"project_dir": result["project_dir"]})
    summary = safe_summary(snapshot, review=review)
    failures = safety_failures(summary)
    summary["safety_failures"] = failures
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 3 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
