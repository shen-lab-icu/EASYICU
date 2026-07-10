"""Bounded, metadata-only study context persistence for the native WebApp."""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_CONFIG_PATH = Path.home() / ".easyicu" / "webserver_study_contexts.json"
_LOCK = threading.RLock()

_MAX_CONTEXTS = 80
_MAX_COLLECTION_ITEMS = 64
_MAX_NESTED_TEXT = 500
_MAX_PATH_LENGTH = 4096
_MAX_CONTEXT_BYTES = 32_768
_MAX_CONTEXT_NODES = 512
_MAX_STORE_BYTES = 4 * 1024 * 1024
_ROW_LEVEL_KEYS = {
    "tablerows",
    "rowdata",
    "rows",
    "records",
    "values",
    "observations",
    "series",
    "patient",
    "patients",
    "patientid",
    "patientids",
    "stayid",
    "stayids",
    "subjectid",
    "subjectids",
    "hadmid",
    "hadmids",
    "entityid",
    "entityids",
}
_CONTEXT_FIELDS = {
    "id",
    "title",
    "question",
    "purpose",
    "data_source",
    "cohort",
    "modules",
    "outcome",
    "time_window",
    "comparator",
    "export_format",
    "analysis_goal",
    "current_stage",
    "last_route",
    "active_job_id",
    "confirmations",
}
_TEXT_LIMITS = {
    "title": 160,
    "question": 1200,
    "purpose": 800,
    "outcome": 500,
    "comparator": 500,
    "export_format": 40,
    "analysis_goal": 1200,
}
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_COHORT_TEXT_FIELDS = {
    "preset",
    "label",
    "review",
    "review_scope",
    "comparison",
    "source_type",
    "comparison_mode",
    "icd_include",
    "icd_exclude",
}
_COHORT_NUMBER_FIELDS = {
    "age_min",
    "age_max",
    "min_icu_los_hours",
    "observation_window_hours",
    "max_patients",
    "entity_count",
    "full_entity_count",
    "review_entities",
    "review_entity_cap",
    "module_count",
    "cohort_size",
    "source_count",
}
_COHORT_BOOL_FIELDS = {"exclude_readmissions", "icd_enabled"}
_COHORT_LIST_FIELDS = {"include_diagnoses", "exclude_diagnoses"}
_SEPSIS_SCHEMA = {
    "record_scope": "text",
    "runtime_profile": "text",
    "implementation_profile": "text",
    "score_family": "text",
    "definition_locked": "bool",
    "suspected_infection": {
        "mode": "text",
        "abx_win_hours": "number",
        "samp_win_hours": "number",
        "abx_count_win_hours": "number",
        "abx_min_count": "number",
        "positive_cultures_required": "bool",
    },
    "sofa_increase": {
        "si_window": "text",
        "window_before_si_hours": "number",
        "window_after_si_hours": "number",
        "delta_function": "text",
        "threshold": "number",
        "keep_components": "bool",
    },
    "review_options": {"si_window": "text_list"},
    "locked_core": {
        "suspected_infection_windows": "text",
        "sofa_window": "text",
        "delta_rule": "text",
        "sofa_threshold": "text",
    },
}
_TIME_WINDOW_SCHEMA = {
    "hours": "number",
    "observation_hours": "number",
    "anchor": "text",
    "preset": "text",
    "label": "text",
}


class StudyContextError(ValueError):
    """Raised when context metadata violates the local persistence contract."""

    def __init__(self, detail: Dict[str, Any]) -> None:
        super().__init__(str(detail.get("error") or "study_context_error"))
        self.detail = detail


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_raw() -> Dict[str, Any]:
    try:
        if _CONFIG_PATH.stat().st_size > _MAX_STORE_BYTES:
            raise StudyContextError(
                {
                    "error": "study_context_store_too_large",
                    "max_bytes": _MAX_STORE_BYTES,
                }
            )
        payload = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise StudyContextError({"error": "study_context_store_invalid"}) from exc
    if not isinstance(payload, dict):
        raise StudyContextError({"error": "study_context_store_invalid"})
    return payload


def _write_raw(payload: Dict[str, Any]) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        _CONFIG_PATH.parent.chmod(0o700)
    except OSError:
        pass
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{_CONFIG_PATH.name}.",
        suffix=".tmp",
        dir=_CONFIG_PATH.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(_CONFIG_PATH)
        try:
            _CONFIG_PATH.chmod(0o600)
        except OSError:
            # The replacement already inherits the secure mkstemp mode.
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _row_level_markers(value: Any, path: str = "context") -> List[str]:
    markers: List[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            normalized_key = re.sub(r"[^a-z0-9]+", "", key.lower())
            if normalized_key in _ROW_LEVEL_KEYS:
                markers.append(child_path)
            markers.extend(_row_level_markers(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            markers.extend(_row_level_markers(child, f"{path}[{index}]"))
    return markers


def _reject_row_level_metadata(value: Any) -> None:
    markers = _row_level_markers(value)
    if markers:
        raise StudyContextError(
            {
                "error": "row_level_metadata_forbidden",
                "markers": markers[:20],
            }
        )


def _metadata_node_count(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(_metadata_node_count(child) for child in value.values())
    if isinstance(value, list):
        return 1 + sum(_metadata_node_count(child) for child in value)
    return 1


def _enforce_context_budget(value: Any) -> None:
    try:
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise StudyContextError(
            {"error": "study_context_not_json_serializable"}
        ) from exc
    size = len(serialized.encode("utf-8"))
    if size > _MAX_CONTEXT_BYTES:
        raise StudyContextError(
            {
                "error": "study_context_too_large",
                "max_bytes": _MAX_CONTEXT_BYTES,
            }
        )
    nodes = _metadata_node_count(value)
    if nodes > _MAX_CONTEXT_NODES:
        raise StudyContextError(
            {
                "error": "study_context_too_complex",
                "max_nodes": _MAX_CONTEXT_NODES,
            }
        )


def _text(value: Any, *, field: str, max_length: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise StudyContextError(
            {
                "error": "invalid_study_context_field_type",
                "field": field,
                "expected": "string",
            }
        )
    text = " ".join(value.split())
    if len(text) > max_length:
        raise StudyContextError(
            {
                "error": "study_context_field_too_long",
                "field": field,
                "max_length": max_length,
            }
        )
    return text


def _identifier(value: Any, *, field: str, default: str = "") -> str:
    candidate = value if value not in (None, "") else default
    if not isinstance(candidate, str):
        raise StudyContextError(
            {
                "error": "invalid_study_context_field_type",
                "field": field,
                "expected": "string",
            }
        )
    text = candidate.strip()
    if not text or len(text) > 80 or not _IDENTIFIER_PATTERN.fullmatch(text):
        raise StudyContextError(
            {
                "error": "invalid_study_context_identifier",
                "field": field,
                "max_length": 80,
            }
        )
    return text


def normalize_path(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, Path)):
        raise StudyContextError(
            {
                "error": "invalid_study_context_field_type",
                "field": "data_source.path",
                "expected": "string",
            }
        )
    text = str(value).strip()
    if not text:
        return ""
    if len(text) > _MAX_PATH_LENGTH:
        raise StudyContextError(
            {
                "error": "study_context_field_too_long",
                "field": "data_source.path",
                "max_length": _MAX_PATH_LENGTH,
            }
        )
    path = Path(text).expanduser()
    try:
        path = path.resolve(strict=False)
    except OSError:
        pass
    return str(path)


def _data_source(value: Any) -> Optional[Dict[str, str]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise StudyContextError(
            {"error": "invalid_study_context_field", "field": "data_source"}
        )
    unknown = sorted(set(value) - {"path", "label", "database"})
    if unknown:
        raise StudyContextError(
            {
                "error": "unknown_study_context_fields",
                "field": "data_source",
                "fields": unknown,
            }
        )
    source = {
        "path": normalize_path(value.get("path")),
        "label": _text(value.get("label"), field="data_source.label", max_length=160),
        "database": _text(
            value.get("database"), field="data_source.database", max_length=64
        ),
    }
    return source if any(source.values()) else None


def _modules(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > _MAX_COLLECTION_ITEMS:
        raise StudyContextError(
            {
                "error": "invalid_study_context_field",
                "field": "modules",
                "max_items": _MAX_COLLECTION_ITEMS,
            }
        )
    modules: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise StudyContextError(
                {
                    "error": "invalid_study_context_field_type",
                    "field": "modules",
                    "expected": "list[string]",
                }
            )
        module = _text(item, field="modules", max_length=80)
        if module and module not in modules:
            modules.append(module)
    return modules


def _finite_number(value: Any, *, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudyContextError(
            {
                "error": "invalid_study_context_field_type",
                "field": field,
                "expected": "number",
            }
        )
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        raise StudyContextError(
            {"error": "invalid_study_context_number", "field": field}
        )
    return value


def _schema_object(value: Any, *, field: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise StudyContextError(
            {
                "error": "invalid_study_context_field_type",
                "field": field,
                "expected": "object",
            }
        )
    unknown = sorted(set(value) - set(schema))
    if unknown:
        raise StudyContextError(
            {
                "error": "unknown_study_context_fields",
                "field": field,
                "fields": unknown,
            }
        )
    result: Dict[str, Any] = {}
    for key, raw in value.items():
        child_field = f"{field}.{key}"
        kind = schema[key]
        if isinstance(kind, dict):
            result[key] = _schema_object(raw, field=child_field, schema=kind)
        elif kind == "text":
            result[key] = _text(raw, field=child_field, max_length=_MAX_NESTED_TEXT)
        elif kind == "number":
            result[key] = _finite_number(raw, field=child_field)
        elif kind == "bool":
            if not isinstance(raw, bool):
                raise StudyContextError(
                    {
                        "error": "invalid_study_context_field_type",
                        "field": child_field,
                        "expected": "boolean",
                    }
                )
            result[key] = raw
        elif kind == "text_list":
            if not isinstance(raw, list) or len(raw) > _MAX_COLLECTION_ITEMS:
                raise StudyContextError(
                    {
                        "error": "invalid_study_context_field_type",
                        "field": child_field,
                        "expected": "list[string]",
                    }
                )
            result[key] = [
                _text(item, field=child_field, max_length=160) for item in raw
            ]
        else:  # pragma: no cover - schema definitions are module constants
            raise AssertionError(f"unknown StudyContext schema kind: {kind}")
    return result


def _cohort(value: Any) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        **{field: "text" for field in _COHORT_TEXT_FIELDS},
        **{field: "number" for field in _COHORT_NUMBER_FIELDS},
        **{field: "bool" for field in _COHORT_BOOL_FIELDS},
        **{field: "text_list" for field in _COHORT_LIST_FIELDS},
        "sepsis_definition": _SEPSIS_SCHEMA,
    }
    return _schema_object(value, field="cohort", schema=schema)


def _confirmations(value: Any) -> Dict[str, bool]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise StudyContextError(
            {
                "error": "invalid_study_context_field_type",
                "field": "confirmations",
                "expected": "object[boolean]",
            }
        )
    if len(value) > _MAX_COLLECTION_ITEMS:
        raise StudyContextError(
            {
                "error": "invalid_study_context_field",
                "field": "confirmations",
                "max_items": _MAX_COLLECTION_ITEMS,
            }
        )
    result: Dict[str, bool] = {}
    for raw_key, raw_value in value.items():
        key = _identifier(raw_key, field="confirmations.key")
        if not isinstance(raw_value, bool):
            raise StudyContextError(
                {
                    "error": "invalid_study_context_field_type",
                    "field": f"confirmations.{key}",
                    "expected": "boolean",
                }
            )
        result[key] = raw_value
    return result


def _default_context(context_id: str, timestamp: str) -> Dict[str, Any]:
    return {
        "id": context_id,
        "revision": 0,
        "title": "",
        "question": "",
        "purpose": "",
        "data_source": None,
        "cohort": {},
        "modules": [],
        "outcome": "",
        "time_window": {},
        "comparator": "",
        "export_format": "",
        "analysis_goal": "",
        "current_stage": "plan",
        "last_route": "entry",
        "active_job_id": None,
        "confirmations": {},
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def _sanitize_patch(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise StudyContextError({"error": "study_context_body_required"})
    _enforce_context_budget(raw)
    _reject_row_level_metadata(raw)
    unknown = sorted(set(raw) - _CONTEXT_FIELDS)
    if unknown:
        raise StudyContextError(
            {"error": "unknown_study_context_fields", "fields": unknown}
        )

    patch: Dict[str, Any] = {}
    if "id" in raw and raw.get("id"):
        patch["id"] = _identifier(raw.get("id"), field="id")
    for field, max_length in _TEXT_LIMITS.items():
        if field in raw:
            patch[field] = _text(raw.get(field), field=field, max_length=max_length)
    if "data_source" in raw:
        patch["data_source"] = _data_source(raw.get("data_source"))
    if "modules" in raw:
        patch["modules"] = _modules(raw.get("modules"))
    if "cohort" in raw:
        patch["cohort"] = _cohort(raw.get("cohort"))
    if "time_window" in raw:
        patch["time_window"] = _schema_object(
            raw.get("time_window"),
            field="time_window",
            schema=_TIME_WINDOW_SCHEMA,
        )
    if "confirmations" in raw:
        patch["confirmations"] = _confirmations(raw.get("confirmations"))
    for field, default in (("current_stage", "plan"), ("last_route", "entry")):
        if field in raw:
            patch[field] = _identifier(raw.get(field), field=field, default=default)
    if "active_job_id" in raw:
        value = raw.get("active_job_id")
        patch["active_job_id"] = (
            _identifier(value, field="active_job_id") if value else None
        )
    return patch


def _contexts_from_raw(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = raw.get("contexts") if isinstance(raw.get("contexts"), list) else []
    contexts: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        _enforce_context_budget(row)
        _reject_row_level_metadata(row)
        patch = _sanitize_patch(
            {field: row[field] for field in _CONTEXT_FIELDS if field in row}
        )
        context_id = patch.pop("id")
        created_at = (
            _text(row.get("created_at"), field="created_at", max_length=64) or _now()
        )
        context = _default_context(context_id, created_at)
        context.update(patch)
        revision = row.get("revision", 0)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise StudyContextError({"error": "study_context_store_invalid"})
        context["revision"] = revision
        context["updated_at"] = (
            _text(row.get("updated_at"), field="updated_at", max_length=64)
            or created_at
        )
        contexts.append(context)
    return contexts


def list_contexts() -> Dict[str, Any]:
    with _LOCK:
        raw = _read_raw()
        contexts = _contexts_from_raw(raw)
        return {
            "contexts": contexts,
            "active_id": raw.get("active_id"),
        }


def get_context(context_id: str) -> Optional[Dict[str, Any]]:
    clean_id = _identifier(context_id, field="id")
    with _LOCK:
        for context in _contexts_from_raw(_read_raw()):
            if context.get("id") == clean_id:
                return context
    return None


def get_active_context() -> Optional[Dict[str, Any]]:
    with _LOCK:
        raw = _read_raw()
        active_id = raw.get("active_id")
        if not active_id:
            return None
        return next(
            (
                context
                for context in _contexts_from_raw(raw)
                if context.get("id") == active_id
            ),
            None,
        )


def upsert_context(
    raw_context: Dict[str, Any],
    *,
    active: bool = True,
    expected_revision: Optional[int] = None,
    require_revision: bool = False,
    lifecycle_write: bool = True,
) -> Dict[str, Any]:
    patch = _sanitize_patch(raw_context)
    if expected_revision is not None and (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision < 0
    ):
        raise StudyContextError(
            {
                "error": "invalid_study_context_revision",
                "expected": "non_negative_integer",
            }
        )
    with _LOCK:
        raw = _read_raw()
        contexts = _contexts_from_raw(raw)
        context_id = patch.pop("id", None) or f"study_{secrets.token_hex(8)}"
        current = next((row for row in contexts if row.get("id") == context_id), None)
        current_revision = int((current or {}).get("revision") or 0)
        if current is not None and require_revision and expected_revision is None:
            raise StudyContextError(
                {
                    "error": "study_context_revision_required",
                    "study_context_id": context_id,
                    "current_revision": current_revision,
                }
            )
        if current is not None and expected_revision is not None:
            if expected_revision != current_revision:
                raise StudyContextError(
                    {
                        "error": "study_context_revision_conflict",
                        "study_context_id": context_id,
                        "expected_revision": expected_revision,
                        "current_revision": current_revision,
                    }
                )
        elif current is None and expected_revision not in (None, 0):
            raise StudyContextError(
                {
                    "error": "study_context_revision_conflict",
                    "study_context_id": context_id,
                    "expected_revision": expected_revision,
                    "current_revision": None,
                }
            )
        if current is not None and not lifecycle_write:
            # Job lifecycle is server-owned. A stale browser metadata cache
            # must never clear a job pointer or roll back an authoritative
            # terminal stage; those fields move only through handoff/job APIs.
            for field in ("current_stage", "last_route", "active_job_id"):
                patch.pop(field, None)
        elif current is None and not lifecycle_write and patch.get("active_job_id"):
            raise StudyContextError(
                {
                    "error": "study_context_lifecycle_field_forbidden",
                    "field": "active_job_id",
                }
            )
        timestamp = _now()
        context = (
            dict(current)
            if current is not None
            else _default_context(context_id, timestamp)
        )
        if patch or current is None:
            context.update(patch)
            context["revision"] = current_revision + 1
            context["updated_at"] = timestamp
        contexts = [row for row in contexts if row.get("id") != context_id]
        contexts.insert(0, context)
        active_id = context_id if active else raw.get("active_id")
        trimmed = contexts[:_MAX_CONTEXTS]
        if active_id and not any(row.get("id") == active_id for row in trimmed):
            active_row = next(
                (row for row in contexts if row.get("id") == active_id), None
            )
            if active_row is not None and trimmed:
                trimmed[-1] = active_row
            else:
                active_id = context_id
        if active_id and not any(row.get("id") == active_id for row in trimmed):
            active_id = context_id
        _write_raw(
            {
                "schema_version": 1,
                "updated_at": timestamp,
                "active_id": active_id,
                "contexts": trimmed,
            }
        )
        return context


def handoff_context(
    context_id: str,
    *,
    current_stage: Any = None,
    last_route: Any = None,
    active_job_id: Any = ...,
    expected_revision: Optional[int] = None,
) -> Dict[str, Any]:
    clean_id = _identifier(context_id, field="id")
    with _LOCK:
        raw = _read_raw()
        contexts = _contexts_from_raw(raw)
        current = next((row for row in contexts if row.get("id") == clean_id), None)
        if current is None:
            raise StudyContextError(
                {"error": "study_context_not_found", "study_context_id": clean_id}
            )
        patch: Dict[str, Any] = {"id": clean_id}
        if current_stage is not None:
            patch["current_stage"] = current_stage
        if last_route is not None:
            patch["last_route"] = last_route
        if active_job_id is not ...:
            patch["active_job_id"] = active_job_id
        # Keep the read/merge/write inside one re-entrant lock scope so two
        # concurrent handoffs cannot both merge from the same stale snapshot.
        return upsert_context(
            patch,
            active=True,
            expected_revision=expected_revision,
            require_revision=expected_revision is not None,
            lifecycle_write=True,
        )


def clear_active_job_if(
    context_id: str,
    job_id: str,
    *,
    current_stage: Any,
    last_route: Any = "agent",
) -> Dict[str, Any]:
    """Clear one job pointer only if it is still the context's active job.

    Multiple runs may overlap for one study. A terminal callback from an older
    run must not clear the pointer or stage written by a newer run.
    """
    clean_id = _identifier(context_id, field="id")
    clean_job_id = _identifier(job_id, field="active_job_id")
    with _LOCK:
        current = next(
            (
                row
                for row in _contexts_from_raw(_read_raw())
                if row.get("id") == clean_id
            ),
            None,
        )
        if current is None:
            raise StudyContextError(
                {"error": "study_context_not_found", "study_context_id": clean_id}
            )
        if current.get("active_job_id") != clean_job_id:
            return {"context": current, "cleared": False}
        context = upsert_context(
            {
                "id": clean_id,
                "current_stage": current_stage,
                "last_route": last_route,
                "active_job_id": None,
            },
            active=False,
            expected_revision=int(current.get("revision") or 0),
            require_revision=True,
            lifecycle_write=True,
        )
        return {"context": context, "cleared": True}


def build_agent_context_binding(
    context: Dict[str, Any],
    *,
    export_path: str,
    request_question: Any = None,
) -> Dict[str, Any]:
    """Validate an Agent source binding and return artifact-safe metadata."""
    context_id = _identifier(context.get("id"), field="id")
    cohort = context.get("cohort") if isinstance(context.get("cohort"), dict) else {}
    confirmations = (
        context.get("confirmations")
        if isinstance(context.get("confirmations"), dict)
        else {}
    )
    if (
        context.get("current_stage") == "crossdb_plan_only"
        or cohort.get("review") == "crossdb"
        or int(cohort.get("source_count") or 0) > 1
        or confirmations.get("crossdb_plan_only") is True
    ):
        raise StudyContextError(
            {
                "error": "study_context_execution_not_supported",
                "study_context_id": context_id,
                "reason": "crossdb_aggregate_not_bound_to_agent_runner",
            }
        )
    source = (
        context.get("data_source")
        if isinstance(context.get("data_source"), dict)
        else {}
    )
    expected_path = normalize_path(source.get("path"))
    active_path = normalize_path(export_path)
    if not expected_path:
        raise StudyContextError(
            {
                "error": "study_context_source_required",
                "study_context_id": context_id,
            }
        )
    if expected_path != active_path:
        raise StudyContextError(
            {
                "error": "study_context_source_mismatch",
                "study_context_id": context_id,
                "expected_path": expected_path,
                "active_path": active_path,
            }
        )

    requested = _text(request_question, field="question", max_length=1200)
    context_question = _text(context.get("question"), field="question", max_length=1200)
    resolved_question = requested or context_question or None
    return {
        "status": "bound",
        "study_context_id": context_id,
        "context_revision": int(context.get("revision") or 0),
        "context_updated_at": context.get("updated_at"),
        "applied": {
            "data_source.path": active_path,
            "question": resolved_question,
        },
        "applied_from": {
            "data_source.path": "study_context",
            "question": "request" if requested else "study_context",
        },
        "informational": {
            key: context.get(key)
            for key in (
                "title",
                "purpose",
                "data_source",
                "cohort",
                "modules",
                "outcome",
                "time_window",
                "comparator",
                "export_format",
                "analysis_goal",
                "current_stage",
                "last_route",
                "active_job_id",
                "confirmations",
            )
        },
    }


__all__ = [
    "StudyContextError",
    "build_agent_context_binding",
    "clear_active_job_if",
    "get_active_context",
    "get_context",
    "handoff_context",
    "list_contexts",
    "normalize_path",
    "upsert_context",
]
