"""Local downloadable diagnostic projection, with no raw logs or data files."""

from __future__ import annotations

import io
import json
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping
import zipfile

from easyicu import __version__
from easyicu.model_request_metadata import STATES, closed
from easyicu.webserver.model_request_records import run_egress

REASON_CODES = frozenset({"provider_request_failed", "request_metadata_available",
                         "request_metadata_not_recorded", "request_metadata_invalid",
                         "run_metadata_invalid", "human_plan_review_required",
                         "research_agent_pipeline_failed_closed",
                         "research_agent_pipeline_complete_human_interpretation_required"})


def scrub_diagnostic_logs(logs: Iterable[Any]) -> list[dict[str, str]]:
    """Keep only closed diagnostic codes; arbitrary text is withheld wholesale.

    Regex substitution cannot prove a log contains no patient text. Even fields
    named 'code' or 'state' must match a closed vocabulary. No free-form string,
    unknown key, row, prompt, path or secret is passed through.
    """
    clean = []
    from easyicu.webserver.stop_next_actions import STOP_POLICIES
    codes = REASON_CODES | frozenset(STOP_POLICIES)
    for index, row in enumerate(logs):
        if index >= 1000:
            break
        row = row if isinstance(row, Mapping) else {}
        clean.append({
            "level": closed(row.get("level"), frozenset({"INFO", "WARNING", "ERROR"}), "INFO"),
            "code": closed(row.get("code"), codes, "diagnostic_detail_withheld"),
            "state": closed(row.get("state"), STATES),
        })
    return clean


def _run_diagnostics(wrapper_dir: Path, fallback: str) -> tuple[str, str, list]:
    from easyicu.webserver.agent_runs import _read_safe_artifact_bytes

    _, raw, error = _read_safe_artifact_bytes(wrapper_dir, "source_run_manifest.json")
    if error == "artifact_not_found":
        return fallback, "provider_task_ledger", []
    try:
        manifest = json.loads(raw) if raw is not None else None
        if not isinstance(manifest, dict):
            raise ValueError("invalid_manifest")
        state = closed(manifest.get("status"), STATES | frozenset({
            "blocked", "analysis_only", "human_review_pending", "locked_pending_human_review",
            "passed", "not_passed", "diagnostic_only",
        }))
        reasons = [manifest.get(key) for key in ("failure_code", "reason_code") if manifest.get(key)]
        return state, "run_manifest", [{"code": code, "state": state} for code in reasons]
    except (ValueError, TypeError):
        return "unknown", "manifest_unavailable", [{"code": "run_metadata_invalid"}]


def build_problem_bundle(wrapper_dir: Path) -> bytes:
    """Build in memory from verified ledger metadata; never scan a run directory."""
    egress = run_egress(wrapper_dir)
    state, state_basis, run_events = _run_diagnostics(wrapper_dir, egress["run_state"])
    events = scrub_diagnostic_logs([
        {"level": "ERROR" if row["failed"] else "INFO",
         "code": "provider_request_failed" if row["failed"] else "request_metadata_available",
         "state": row["state"]} for row in egress["requests"]
    ] + run_events or [{"code": egress["reason_code"]}])
    manifest = {
        "schema_version": "easyicu.problem-bundle/1",
        "versions": {"easyicu": __version__, "python": platform.python_version()},
        "run_state": state, "run_state_basis": state_basis,
        "reason_codes": sorted({event["code"] for event in events}),
        "log_policy": "closed_lifecycle_metadata_only", "upload_performed": False,
        "payload_status": "withheld",
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in (("manifest", manifest), ("events", events), ("egress", egress)):
            archive.writestr(name + ".json", json.dumps(payload, ensure_ascii=False, indent=2))
    return output.getvalue()
