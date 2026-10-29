"""Atomic reports and receipts containing only the closed aggregate schema."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from .contracts import SCHEMA_VERSION, ItemResult, Status
from .privacy import artifact_name, is_generated_job_id, redact as redact, token


def atomic_json(path: Path, payload: dict):
    temporary = path.with_suffix(".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("symlink_output_refused")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ReceiptStore:
    def __init__(self, out: Path, scope: str, *, resume: bool):
        self.out = out
        self.scope = hashlib.sha256(scope.encode()).hexdigest()
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / "walk_state.json"
        if self.path.is_symlink():
            raise ValueError("symlink_output_refused")
        if self.path.exists():
            if not resume:
                raise ValueError("existing_walk_requires_resume")
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state.get("scope") != self.scope or state.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("resume_scope_mismatch")
            self.state = state
        else:
            self.state = {"schema_version": SCHEMA_VERSION, "scope": self.scope, "items": {}}
        self.receipts = out / "receipts"
        if self.receipts.is_symlink():
            raise ValueError("symlink_output_refused")
        self.receipts.mkdir(exist_ok=True)
        self.flush()

    def flush(self):
        self._validate_state()
        atomic_json(self.path, self.state)

    def _validate_state(self):
        state = self.state
        if (set(state) - {"schema_version", "scope", "items", "source_id", "database"}
                or not isinstance(state.get("items"), dict)):
            raise ValueError("resume_state_invalid")
        for key in ("source_id", "database"):
            if key in state and token(state[key]) != state[key]:
                raise ValueError("resume_state_invalid")
        booleans = {"deleted", "source_bound", "question_sent", "provider_retry_used"}
        jobs = {"message_job_id", "plan_job_id", "execution_job_id"}
        identities = {"project_id", "session_id", "study_context_id", "message_job_id", "plan_job_id",
                      "execution_job_id", "run_id"}
        for key, item in state["items"].items():
            if (token(key) != key or not isinstance(item, dict)
                    or set(item) - booleans - identities - {"project_title"}):
                raise ValueError("resume_state_invalid")
            for name, value in item.items():
                if name in booleans and type(value) is not bool:
                    raise ValueError("resume_state_invalid")
                if name in identities and value != "" and not (name in jobs and is_generated_job_id(value)) and (
                        not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,159}", value)
                        or token(value) != value):
                    raise ValueError("resume_state_invalid")
                if name == "project_title" and (not isinstance(value, str) or not re.fullmatch(r"EasyICU walk [a-f0-9]{12}", value)):
                    raise ValueError("resume_state_invalid")

    def _path(self, phase, item):
        identity = hashlib.sha256(f"{phase}/{item}".encode()).hexdigest()[:24]
        return self.receipts / f"{phase}-{identity}.json"

    def save(self, result: ItemResult, *, policy: str):
        atomic_json(self._path(result.phase, result.item_id), {
            "schema_version": SCHEMA_VERSION, "scope": self.scope, "policy": policy,
            "result": result.to_dict(),
        })

    def finished(self, phase: str, item: str, *, policy: str) -> ItemResult | None:
        path = self._path(phase, item)
        if path.is_symlink():
            raise ValueError("symlink_output_refused")
        if not path.exists():
            return None
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if (receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("scope") != self.scope
                    or receipt.get("policy") != policy):
                return None
            result = ItemResult(**receipt["result"])
            if result.phase != phase or result.item_id != item:
                return None
            return result if result.status not in {Status.FAILED, Status.TIMED_OUT} else None
        except (KeyError, TypeError, ValueError):
            return None

    def write_report(self, items: list[ItemResult]) -> dict:
        report = {"schema_version": SCHEMA_VERSION, "aggregate_only": True,
                  "scope_sha256": self.scope, "items": [item.to_dict() for item in items],
                  "status_counts": dict(Counter(item.status.value for item in items))}
        atomic_json(self.out / "walk_report.json", report)
        lines = ["# EasyICU walk", "", "Aggregate metadata only. A pass describes the named check, not scientific validity.", "",
                 "| Phase | Item | Result | Reason | Seconds |", "| --- | --- | --- | --- | ---: |"]
        lines.extend(f"| {r.phase} | {r.item_id} | {r.status.value} | {r.reason_code} | {r.elapsed_seconds:.2f} |" for r in items)
        path = self.out / "walk_report.md"
        if path.is_symlink():
            raise ValueError("symlink_output_refused")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report


def run_metrics(projection: dict) -> dict:
    run = projection.get("latest_run") or {}
    history = projection.get("runs") or []
    # A failed latest attempt may coexist with an older authoritative review.
    # The history is newest-first; do not attribute that older run's success to it.
    if history and (run.get("present") is False or not run or (
            (projection.get("workflow") or {}).get("next_action_code") == "failed_pipeline_requires_fresh_plan"
            and "gate_reason_code" in history[0])):
        run = history[0]
    refs = run.get("artifact_refs") or []
    names = [row.get("artifact", "") for row in refs if isinstance(row, dict)]
    names += list(run.get("artifact_names") or [])
    names = list(dict.fromkeys(name for name in names if isinstance(name, str)))
    safe_names = [artifact_name(name) for name in names]
    counts = Counter(token(name.rsplit(".", 1)[-1], "other") for name in names)
    provenance = sum("provenance" in name for name in safe_names)
    evidence = sum("evidence" in name or name == "manifest.json" for name in safe_names)
    return {
        "artifact_count": max(len(names), run.get("artifact_count", 0)),
        "artifact_names": safe_names, "artifact_counts_by_type": dict(counts),
        "report_present": any("report" in name or name.startswith("manuscript") for name in safe_names),
        "provenance_present": provenance > 0, "provenance_count": provenance,
        "evidence_present": evidence > 0, "evidence_count": evidence,
        "execution_complete": run.get("execution_complete") is True,
        "analysis_validated": run.get("analysis_validated") is True,
        "evidence_complete": run.get("evidence_complete") is True,
        "gate_status": token(run.get("gate_status")),
        "gate_reason_code": token(run.get("gate_reason_code")),
        "gate_detail_code": token(run.get("gate_detail_code")),
        "gate_detail_cause_code": token(run.get("gate_detail_cause_code")),
        "gate_detail_step_id": token(run.get("gate_detail_step_id")),
        "gate_detail_runner_kind": token(run.get("gate_detail_runner_kind")),
    }
