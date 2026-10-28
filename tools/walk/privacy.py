"""A closed projection vocabulary: no response bodies or free text on disk."""

from __future__ import annotations

import re
from typing import Any


_IDENTIFIER = re.compile(
    r"\b(?:patient|subject|stay|icustay|hadm|person|encounter|hospitaladmission|admission|case|pid|mrn)"
    r"(?:[_ -]?(?:id|identifier|number))?\s*[=: _-]+[A-Za-z0-9_.-]+", re.I,
)
_ROW_KEYS = re.compile(
    r"^(?:rows?|records?|samples?|values?|data|transcript|content|text|error|"
    r"artifact_payloads|.*(?:patient|subject|stay|icustay|hadm|mrn)(?:_?id)?)$", re.I,
)
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,159}$")


def redact(value: Any) -> Any:
    """Secondary defense for identifier-like strings; raw payloads are never saved."""
    if isinstance(value, str):
        return _IDENTIFIER.sub("[redacted]", value)
    if isinstance(value, dict):
        return {str(k): redact(v) for k, v in value.items() if not _ROW_KEYS.fullmatch(str(k))}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value if value is None or type(value) in (bool, int, float) else None


def token(value: Any, default: str = "not_reported") -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        return default
    if redact(value) != value or "://" in value or ".." in value or "/" in value:
        return default
    return value


def artifact_name(value: Any) -> str:
    name = token(value, "redacted")
    return "redacted" if re.search(r"\d{6,}", name) or name.split(".", 1)[0].isdigit() else name


_COUNTS = frozenset({
    "artifact_count", "candidate_count", "retrieved", "screened", "included",
    "source_count", "shortfall_count", "mapped", "missing", "changed_meaning",
    "concept_count", "module_count", "planned_count", "item_count", "pending_design_choices",
    "task_count", "attached_skill_count", "mention_count", "provenance_count", "evidence_count",
    "unreported_owner_count", "run_count", "workflow_count", "method_count", "completed_modules",
})
_BOOLEANS = frozenset({
    "plan_valid", "plan_execution_ready", "report_present", "provenance_present", "evidence_present",
    "connection_verified", "shell_ready", "ready", "enabled", "execution_complete", "analysis_validated",
    "evidence_complete", "smoke_executed", "kernel_binding", "removed", "renamed", "attached", "resolved",
})
_TOKENS = frozenset({
    "version", "commit", "runner_image_tag", "database", "source", "basis", "gate_status",
    "gate_reason_code", "run_outcome", "plan_review_status", "current_stage", "provider_status",
    "id", "layer", "implementation", "owner_status", "status", "role", "scope",
})
_LISTS = frozenset({"artifact_names", "refusal_codes", "planned_items", "unavailable", "shortfalls"})
_HISTOGRAMS = frozenset({
    "artifact_counts_by_type", "layer_counts", "role_coverage", "sources", "evidence_verdicts",
    "feasibility_verdicts", "status_counts",
})
_RECORDS = frozenset({"families", "catalogue_items", "modules"})


def metrics_projection(raw: dict) -> dict:
    """Retain only known metadata fields, integer aggregates and typed records.

    There is intentionally no generic JSON fallback. Adding an observation to
    the report requires declaring its representation here first.
    """
    result = {}
    for key, value in raw.items():
        if key in _COUNTS and (value is None or type(value) is int and value >= 0):
            result[key] = value
        elif key in _BOOLEANS and (value is None or type(value) is bool):
            result[key] = value
        elif key in _TOKENS:
            result[key] = token(value)
        elif key in _LISTS and isinstance(value, list):
            result[key] = [artifact_name(v) if key == "artifact_names" else token(v, "redacted") for v in value]
        elif key in _HISTOGRAMS and isinstance(value, dict):
            result[key] = {token(k, "redacted"): v for k, v in value.items()
                           if type(v) is int and v >= 0}
        elif key in _RECORDS and isinstance(value, list):
            result[key] = [metrics_projection(v) for v in value if isinstance(v, dict)]
    return result
