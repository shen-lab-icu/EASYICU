"""PHI-safe projections for model-visible Pi Copilot tools."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .contracts import PiCopilotError

MAX_PROJECTION_BYTES = 32_768
MAX_TEXT_CHARS = 2_000
MAX_LIST_ITEMS = 80

_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"\b(?:subject|stay|hadm|patient|entity)[ _-]?ids?\b", re.I),
    re.compile(
        r"\b(?:medical record number|mrn)\s*[:=#]?\s*[A-Za-z0-9-]+", re.I
    ),
    re.compile(r"\b(?:dob|date of birth)\s*[:=]", re.I),
    re.compile(r"\b(?:note_text|free_text|clinical_note)\b", re.I),
)
_FORBIDDEN_KEYS = {
    "rows",
    "records",
    "observations",
    "series",
    "values",
    "patient",
    "patients",
    "patient_id",
    "patient_ids",
    "subject_id",
    "subject_ids",
    "stay_id",
    "stay_ids",
    "hadm_id",
    "hadm_ids",
    "entity_id",
    "entity_ids",
    "timestamps",
    "notes",
    "credentials",
    "api_key",
    "token",
    "source_path",
    "project_dir",
    "path",
}


def path_digest(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def reject_sensitive_message(text: str) -> None:
    """Reject obvious row/identifier material before it reaches the shell model."""

    for pattern in _SENSITIVE_TEXT_PATTERNS:
        if pattern.search(text):
            raise PiCopilotError(
                "pi_message_phi_risk",
                (
                    "The Pi Copilot message looks like row-level patient data or "
                    "an identifier. Use aggregate study metadata instead."
                ),
                status_code=400,
            )


def _bounded_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def project_study_context(
    context: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not context:
        return {"present": False}
    source = context.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    cohort = context.get("cohort")
    cohort = cohort if isinstance(cohort, Mapping) else {}
    safe_cohort_keys = (
        "preset",
        "label",
        "review_scope",
        "comparison_mode",
        "age_min",
        "age_max",
        "min_icu_los_hours",
        "observation_window_hours",
        "max_patients",
        "entity_count",
        "full_entity_count",
        "cohort_size",
        "source_count",
        "module_count",
        "exclude_readmissions",
    )
    projected_cohort = {
        key: cohort[key]
        for key in safe_cohort_keys
        if key in cohort and not isinstance(cohort[key], (dict, list))
    }
    question = _bounded_text(context.get("question"), 1200)
    if question:
        reject_sensitive_message(question)
    return ensure_safe_projection(
        {
            "present": True,
            "id": context.get("id"),
            "revision": context.get("revision"),
            "title": _bounded_text(context.get("title"), 160),
            "question": question,
            "purpose": _bounded_text(context.get("purpose"), 800),
            "data_source": {
                "source_type": source.get("source_type") or source.get("type"),
                "database": source.get("database") or source.get("source_id"),
                "path_digest": path_digest(source.get("path")),
                "status": source.get("status"),
            },
            "cohort": projected_cohort,
            "modules": [
                str(item)[:120] for item in (context.get("modules") or [])
            ][:MAX_LIST_ITEMS],
            "outcome": _bounded_text(context.get("outcome"), 500),
            "time_window": dict(context.get("time_window") or {}),
            "comparator": _bounded_text(context.get("comparator"), 500),
            "export_format": _bounded_text(context.get("export_format"), 40),
            "analysis_goal": _bounded_text(context.get("analysis_goal"), 1200),
            "current_stage": context.get("current_stage"),
            "last_route": context.get("last_route"),
            "active_job_id": context.get("active_job_id"),
            "confirmations": dict(context.get("confirmations") or {}),
        }
    )


def project_job(snapshot: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not snapshot:
        return {"present": False}
    events = snapshot.get("events")
    events = events if isinstance(events, list) else []
    progress = []
    for event in events[-20:]:
        if not isinstance(event, Mapping):
            continue
        progress.append(
            {
                key: event.get(key)
                for key in (
                    "seq",
                    "type",
                    "status",
                    "current",
                    "total",
                    "step",
                    "label",
                    "reason",
                )
                if event.get(key) is not None
            }
        )
    return ensure_safe_projection(
        {
            "present": True,
            "job_id": snapshot.get("id"),
            "kind": snapshot.get("kind"),
            "status": snapshot.get("status"),
            "cancel_requested": bool(snapshot.get("cancel_requested")),
            "cancel_reason": snapshot.get("cancel_reason"),
            "error_code": _safe_error_code(snapshot.get("error")),
            "progress": progress,
        }
    )


def project_capabilities(payload: Mapping[str, Any]) -> Dict[str, Any]:
    settings = payload.get("settings")
    settings = settings if isinstance(settings, Mapping) else {}
    capabilities = payload.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, Mapping) else {}
    result: Dict[str, Any] = {"settings": dict(settings), "capabilities": {}}
    for name, raw in capabilities.items():
        row = raw if isinstance(raw, Mapping) else {}
        result["capabilities"][str(name)] = {
            key: row.get(key)
            for key in (
                "enabled",
                "available",
                "status",
                "reason",
                "behavior",
                "scope",
            )
            if row.get(key) is not None
        }
    return ensure_safe_projection(result)


def project_run_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return ensure_safe_projection(
        {
            key: row.get(key)
            for key in (
                "run_id",
                "study_id",
                "mode",
                "run_type",
                "gate_status",
                "readiness_status",
                "signed",
                "signoff_stale",
                "integrity_status",
                "reportable",
                "draft_unlocked",
                "artifact_count",
                "artifact_names",
                "updated_at",
            )
            if row.get(key) is not None
        }
    )


def project_artifacts(rows: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    projected = []
    for row in list(rows)[:MAX_LIST_ITEMS]:
        projected.append(
            {
                key: row.get(key)
                for key in ("name", "size", "sha256", "media_type")
                if row.get(key) is not None
            }
        )
    return ensure_safe_projection(projected)


def bounded_json_projection(
    value: Any, *, max_bytes: int = MAX_PROJECTION_BYTES
) -> Any:
    """Keep approved structured artefacts bounded without inventing a summary."""

    ensure_safe_projection(value, max_bytes=max_bytes)
    return value


def ensure_safe_projection(
    value: Any, *, max_bytes: int = MAX_PROJECTION_BYTES
) -> Any:
    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                normalized = str(key).strip().lower().replace("-", "_")
                if normalized in _FORBIDDEN_KEYS:
                    raise PiCopilotError(
                        "pi_projection_blocked",
                        (
                            "The host withheld a model-visible field owned by "
                            f"the patient-data boundary: {normalized}"
                        ),
                        status_code=500,
                        details={"field": normalized},
                    )
                visit(child)
        elif isinstance(node, (list, tuple)):
            if len(node) > MAX_LIST_ITEMS:
                raise PiCopilotError(
                    "pi_projection_too_large",
                    "The host projection contains too many list items.",
                    status_code=500,
                )
            for child in node:
                visit(child)
        elif isinstance(node, Path):
            raise PiCopilotError(
                "pi_projection_blocked",
                "Filesystem paths are not model-visible Pi tool results.",
                status_code=500,
            )

    visit(value)
    encoded = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) > max_bytes:
        raise PiCopilotError(
            "pi_projection_too_large",
            "The host projection exceeded the bounded Pi tool-result contract.",
            status_code=500,
            details={"bytes": len(encoded), "max_bytes": max_bytes},
        )
    return value


def _safe_error_code(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    first = text.split(":", 1)[0]
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", first).strip("_")
    return normalized[:160] or "job_failed"


__all__ = [
    "bounded_json_projection",
    "ensure_safe_projection",
    "path_digest",
    "project_artifacts",
    "project_capabilities",
    "project_job",
    "project_run_row",
    "project_study_context",
    "reject_sensitive_message",
]
