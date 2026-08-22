"""Bounded, metadata-only study context persistence for the native WebApp."""

from __future__ import annotations

import json
import hashlib
import os
import re
import secrets
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from easyicu.research_agent.planning.sensitivity_authority import (
    normalize_prespecified_sensitivities,
)
from easyicu.webserver import state_paths

_CONFIG_PATH = state_paths.state_root() / "webserver_study_contexts.json"
_LOCK = threading.RLock()

_MAX_LISTED_CONTEXTS = 80
_MAX_STORED_CONTEXTS = 1000
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
    "crossdb_selection",
    "cohort",
    "modules",
    "outcome",
    "primary_exposure",
    "covariates",
    "covariate_selection",
    "covariate_rationales",
    "covariate_temporal_roles",
    "execution_concepts",
    "analysis_design",
    "sensitivity_specs",
    "time_window",
    "comparator",
    "export_format",
    "analysis_goal",
    "current_stage",
    "last_route",
    "active_job_id",
    "confirmations",
    "idea_handoff",
    "literature_authority",
}
_TEXT_LIMITS = {
    "title": 160,
    "question": 1200,
    "purpose": 800,
    "outcome": 500,
    "primary_exposure": 160,
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
_EXECUTION_CONCEPT_FIELDS = frozenset(
    {"outcome", "primary_exposure", "covariates"}
)
_ANALYSIS_DESIGN_FIELDS = frozenset(
    {"analysis_family", "analysis_unit", "variance_estimator", "cluster_unit"}
)
_COVARIATE_SELECTIONS = frozenset({"planner_selectable", "exact"})
_COVARIATE_TEMPORAL_ROLES = frozenset(
    {"baseline_static", "at_or_before_time_zero"}
)
_ANALYSIS_UNITS = frozenset(
    {"row", "icu_stay", "hospital_admission", "patient", "site"}
)
_VARIANCE_ESTIMATORS = frozenset(
    {
        "model_based",
        "heteroskedasticity_robust",
        "cluster_robust",
        "none_counts_only",
    }
)
_CLUSTER_UNITS = frozenset({"hospital_admission", "patient", "site", "custom"})
_IDEA_HANDOFF_SCHEMA = {
    "schema_version": "text",
    "run_id": "text",
    "idea_id": "text",
    "canonical_handoff_sha256": "text",
    "status": "text",
    "accepted_at": "text",
    "go_no_go": "text",
    "go_no_go_reason": "text",
    "prior_art_binding_schema_version": "text",
    "prior_art_sha256": "text",
    "prior_art_status": "text",
    "prior_art_result_count": "number",
    "prior_art_searched_at": "text",
}
_LITERATURE_AUTHORITY_SCHEMA = {
    "schema_version": "text",
    "receipt_id": "text",
    "receipt_sha256": "text",
    "study_context_id": "text",
    "study_context_revision": "number",
    "status": "text",
    "result_count": "number",
    "searched_at": "text",
    "study_configuration_sha256": "text",
}
_LITERATURE_AUTHORITY_V2 = "easyicu.web-literature-authority/2"
_LITERATURE_AUTHORITY_V3 = "easyicu.web-literature-authority/3"
_LITERATURE_SCOPE_FIELDS_V2 = (
    "question",
    "purpose",
    "data_source",
    "crossdb_selection",
    "cohort",
    "modules",
    "outcome",
    "primary_exposure",
    "covariates",
    "covariate_selection",
    "covariate_rationales",
    "covariate_temporal_roles",
    "execution_concepts",
    "analysis_design",
    "sensitivity_specs",
    "time_window",
    "comparator",
    "export_format",
    "analysis_goal",
    "confirmations",
    "idea_handoff",
)
_SCIENTIFIC_CONFIGURATION_FIELDS = (
    *_LITERATURE_SCOPE_FIELDS_V2,
    "literature_authority",
)


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
    if len(serialized.encode("utf-8")) > _MAX_STORE_BYTES:
        raise StudyContextError(
            {
                "error": "study_context_store_capacity_reached",
                "max_bytes": _MAX_STORE_BYTES,
            }
        )
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


def _crossdb_selection(value: Any) -> Dict[str, Any]:
    if value in (None, {}):
        return {}
    if not isinstance(value, dict):
        raise StudyContextError(
            {"error": "invalid_study_context_field", "field": "crossdb_selection"}
        )
    allowed = {"schema_version", "source_count", "sources", "selection_digest"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise StudyContextError(
            {
                "error": "unknown_study_context_fields",
                "field": "crossdb_selection",
                "fields": unknown,
            }
        )
    schema_version = _text(
        value.get("schema_version"),
        field="crossdb_selection.schema_version",
        max_length=40,
    )
    if schema_version != "crossdb-selection-v1":
        raise StudyContextError(
            {
                "error": "crossdb_selection_schema_unsupported",
                "field": "crossdb_selection.schema_version",
            }
        )
    source_count = value.get("source_count")
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count < 2
        or source_count > _MAX_COLLECTION_ITEMS
    ):
        raise StudyContextError(
            {
                "error": "invalid_study_context_field",
                "field": "crossdb_selection.source_count",
            }
        )
    raw_sources = value.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != source_count:
        raise StudyContextError(
            {
                "error": "crossdb_selection_source_count_mismatch",
                "field": "crossdb_selection.sources",
            }
        )
    sources: List[Dict[str, str]] = []
    for index, raw_source in enumerate(raw_sources):
        field = f"crossdb_selection.sources[{index}]"
        if not isinstance(raw_source, dict):
            raise StudyContextError(
                {"error": "invalid_study_context_field", "field": field}
            )
        source_unknown = sorted(
            set(raw_source) - {"source_id", "label", "database", "path_hash"}
        )
        if source_unknown:
            raise StudyContextError(
                {
                    "error": "unknown_study_context_fields",
                    "field": field,
                    "fields": source_unknown,
                }
            )
        path_hash = _text(
            raw_source.get("path_hash"), field=f"{field}.path_hash", max_length=64
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{12,64}", path_hash):
            raise StudyContextError(
                {
                    "error": "invalid_study_context_identifier",
                    "field": f"{field}.path_hash",
                }
            )
        sources.append(
            {
                "source_id": _identifier(
                    raw_source.get("source_id"), field=f"{field}.source_id"
                ),
                "label": _text(
                    raw_source.get("label"), field=f"{field}.label", max_length=160
                ),
                "database": _text(
                    raw_source.get("database"),
                    field=f"{field}.database",
                    max_length=64,
                ),
                "path_hash": path_hash,
            }
        )
    selection_digest = _text(
        value.get("selection_digest"),
        field="crossdb_selection.selection_digest",
        max_length=64,
    ).lower()
    expected_digest = hashlib.sha256(
        json.dumps(
            sources,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if selection_digest != expected_digest:
        raise StudyContextError(
            {
                "error": "crossdb_selection_digest_mismatch",
                "field": "crossdb_selection.selection_digest",
            }
        )
    return {
        "schema_version": schema_version,
        "source_count": source_count,
        "sources": sources,
        "selection_digest": selection_digest,
    }


def _modules(value: Any) -> List[str]:
    return _text_list(value, field="modules", max_length=80)


def _text_list(value: Any, *, field: str, max_length: int) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > _MAX_COLLECTION_ITEMS:
        raise StudyContextError(
            {
                "error": "invalid_study_context_field",
                "field": field,
                "max_items": _MAX_COLLECTION_ITEMS,
            }
        )
    values: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise StudyContextError(
                {
                    "error": "invalid_study_context_field_type",
                    "field": field,
                    "expected": "list[string]",
                }
            )
        text = _text(item, field=field, max_length=max_length)
        if text and text not in values:
            values.append(text)
    return values


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


def normalize_covariate_rationales(value: Any) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > _MAX_COLLECTION_ITEMS:
        raise StudyContextError(
            {
                "error": "invalid_study_context_field_type",
                "field": "covariate_rationales",
                "expected": "object[string]",
            }
        )
    result: Dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _identifier(raw_key, field="covariate_rationales.key")
        text = _text(
            raw_value,
            field=f"covariate_rationales.{key}",
            max_length=_MAX_NESTED_TEXT,
        )
        if len(text) < 8:
            raise StudyContextError(
                {
                    "error": "study_covariate_rationale_too_short",
                    "field": f"covariate_rationales.{key}",
                    "min_length": 8,
                }
            )
        result[key] = text
    return result


def normalize_covariate_temporal_roles(value: Any) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > _MAX_COLLECTION_ITEMS:
        raise StudyContextError(
            {
                "error": "invalid_study_context_field_type",
                "field": "covariate_temporal_roles",
                "expected": "object[string]",
            }
        )
    result: Dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _identifier(raw_key, field="covariate_temporal_roles.key")
        role = _identifier(
            raw_value,
            field=f"covariate_temporal_roles.{key}",
        )
        if role not in _COVARIATE_TEMPORAL_ROLES:
            raise StudyContextError(
                {
                    "error": "study_covariate_temporal_role_invalid",
                    "field": f"covariate_temporal_roles.{key}",
                    "allowed": sorted(_COVARIATE_TEMPORAL_ROLES),
                }
            )
        result[key] = role
    return result


def normalize_execution_concepts(value: Any) -> Dict[str, Any]:
    """Normalize exact source-concept identifiers separately from UI labels."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise StudyContextError(
            {
                "error": "invalid_study_context_field_type",
                "field": "execution_concepts",
                "expected": "object",
            }
        )
    unknown = sorted(set(value) - _EXECUTION_CONCEPT_FIELDS)
    if unknown:
        raise StudyContextError(
            {
                "error": "unknown_study_context_fields",
                "field": "execution_concepts",
                "fields": unknown,
            }
        )
    result: Dict[str, Any] = {}
    for field in ("outcome", "primary_exposure"):
        raw = value.get(field)
        if raw not in (None, ""):
            result[field] = _identifier(
                raw,
                field=f"execution_concepts.{field}",
            )
    if "covariates" in value:
        raw_covariates = value.get("covariates")
        if not isinstance(raw_covariates, list) or len(raw_covariates) > _MAX_COLLECTION_ITEMS:
            raise StudyContextError(
                {
                    "error": "invalid_study_context_field",
                    "field": "execution_concepts.covariates",
                    "max_items": _MAX_COLLECTION_ITEMS,
                }
            )
        covariates: List[str] = []
        for raw in raw_covariates:
            concept_id = _identifier(
                raw,
                field="execution_concepts.covariates",
            )
            if concept_id not in covariates:
                covariates.append(concept_id)
        result["covariates"] = covariates
    return result


def normalize_analysis_design(value: Any) -> Dict[str, str]:
    """Normalize user-approved sampling and variance commitments.

    StudyContext owns this scientific intent.  A data-source adapter may later
    bind ``cluster_unit`` to a private physical coordinate, but neither the
    browser nor Pi is allowed to guess that coordinate from prose.
    """

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise StudyContextError(
            {
                "error": "invalid_study_context_field_type",
                "field": "analysis_design",
                "expected": "object",
            }
        )
    unknown = sorted(set(value) - _ANALYSIS_DESIGN_FIELDS)
    if unknown:
        raise StudyContextError(
            {
                "error": "unknown_study_context_fields",
                "field": "analysis_design",
                "fields": unknown,
            }
        )
    if not value:
        return {}
    missing = sorted(
        field
        for field in ("analysis_unit", "variance_estimator")
        if not value.get(field)
    )
    if missing:
        raise StudyContextError(
            {
                "error": "study_analysis_design_fields_required",
                "field": "analysis_design",
                "fields": missing,
            }
        )
    analysis_family = ""
    if value.get("analysis_family") not in (None, ""):
        analysis_family = _identifier(
            value.get("analysis_family"), field="analysis_design.analysis_family"
        )
        from easyicu.research_agent.planning.analysis_types import (
            canonical_analysis_family,
            list_analysis_types,
        )

        if canonical_analysis_family(analysis_family) != analysis_family:
            raise StudyContextError(
                {
                    "error": "study_analysis_family_unsupported",
                    "field": "analysis_design.analysis_family",
                    "allowed": sorted(spec.key for spec in list_analysis_types()),
                }
            )
    analysis_unit = _identifier(
        value.get("analysis_unit"), field="analysis_design.analysis_unit"
    )
    variance_estimator = _identifier(
        value.get("variance_estimator"),
        field="analysis_design.variance_estimator",
    )
    if analysis_unit not in _ANALYSIS_UNITS:
        raise StudyContextError(
            {
                "error": "study_analysis_unit_unsupported",
                "field": "analysis_design.analysis_unit",
                "allowed": sorted(_ANALYSIS_UNITS),
            }
        )
    if variance_estimator not in _VARIANCE_ESTIMATORS:
        raise StudyContextError(
            {
                "error": "study_variance_estimator_unsupported",
                "field": "analysis_design.variance_estimator",
                "allowed": sorted(_VARIANCE_ESTIMATORS),
            }
        )
    cluster_unit = ""
    if value.get("cluster_unit") not in (None, ""):
        cluster_unit = _identifier(
            value.get("cluster_unit"), field="analysis_design.cluster_unit"
        )
        if cluster_unit not in _CLUSTER_UNITS:
            raise StudyContextError(
                {
                    "error": "study_cluster_unit_unsupported",
                    "field": "analysis_design.cluster_unit",
                    "allowed": sorted(_CLUSTER_UNITS),
                }
            )
    if variance_estimator == "cluster_robust" and not cluster_unit:
        raise StudyContextError(
            {
                "error": "study_cluster_unit_required",
                "field": "analysis_design.cluster_unit",
            }
        )
    if variance_estimator != "cluster_robust" and cluster_unit:
        raise StudyContextError(
            {
                "error": "study_cluster_unit_not_applicable",
                "field": "analysis_design.cluster_unit",
            }
        )
    return {
        **({"analysis_family": analysis_family} if analysis_family else {}),
        "analysis_unit": analysis_unit,
        "variance_estimator": variance_estimator,
        **({"cluster_unit": cluster_unit} if cluster_unit else {}),
    }


def normalize_sensitivity_specs(value: Any) -> List[Dict[str, Any]]:
    """Normalize typed user-reviewed sensitivities at the StudyContext owner."""

    try:
        specs = normalize_prespecified_sensitivities(value)
    except (TypeError, ValueError) as exc:
        raise StudyContextError(
            {
                "error": "study_sensitivity_specs_invalid",
                "field": "sensitivity_specs",
                "reason": str(exc)[:_MAX_NESTED_TEXT],
            }
        ) from exc
    return [spec.model_dump(mode="json") for spec in specs]


def validate_context_update(
    raw_context: Dict[str, Any],
    *,
    current_context: Optional[Mapping[str, Any]] = None,
    lifecycle_write: bool = True,
) -> Dict[str, Any]:
    """Validate and normalize one proposed update without mutating the store.

    Conversational callers need to distinguish a rejected proposal from an
    authorized mutation.  In particular, a one-use host grant must not be
    consumed merely because a typed sensitivity row is malformed.  This owner
    preview applies the same normalizers and cross-field scientific contracts
    as :func:`upsert_context`; the subsequent write still performs revision
    checking and validates again under the store lock.
    """

    patch = _sanitize_patch(raw_context)
    current = dict(current_context or {})
    if current and not lifecycle_write:
        for field in ("current_stage", "last_route", "active_job_id"):
            patch.pop(field, None)
    elif not current and not lifecycle_write and patch.get("active_job_id"):
        raise StudyContextError(
            {
                "error": "study_context_lifecycle_field_forbidden",
                "field": "active_job_id",
            }
        )

    candidate = dict(current)
    candidate.update(patch)
    _validate_covariate_decision_contract(candidate)
    _validate_analysis_dependence_contract(candidate)
    if "time_window" in patch:
        _validate_materialization_window_contract(candidate)
    return patch


def _default_context(context_id: str, timestamp: str) -> Dict[str, Any]:
    return {
        "id": context_id,
        "revision": 0,
        "title": "",
        "question": "",
        "purpose": "",
        "data_source": None,
        "crossdb_selection": {},
        "cohort": {},
        "modules": [],
        "outcome": "",
        "primary_exposure": "",
        "covariates": [],
        "covariate_selection": "planner_selectable",
        "covariate_rationales": {},
        "covariate_temporal_roles": {},
        "execution_concepts": {},
        "analysis_design": {},
        "sensitivity_specs": [],
        "time_window": {},
        "comparator": "",
        "export_format": "",
        "analysis_goal": "",
        "current_stage": "plan",
        "last_route": "entry",
        "active_job_id": None,
        "confirmations": {},
        "idea_handoff": {},
        "literature_authority": {},
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def _sanitize_patch(
    raw: Any,
    *,
    allow_literature_authority: bool = False,
) -> Dict[str, Any]:
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
    if "crossdb_selection" in raw:
        patch["crossdb_selection"] = _crossdb_selection(
            raw.get("crossdb_selection")
        )
    if "modules" in raw:
        patch["modules"] = _modules(raw.get("modules"))
    if "covariates" in raw:
        patch["covariates"] = _text_list(
            raw.get("covariates"), field="covariates", max_length=160
        )
    if "covariate_selection" in raw:
        selection = _identifier(
            raw.get("covariate_selection"),
            field="covariate_selection",
            default="planner_selectable",
        )
        if selection not in _COVARIATE_SELECTIONS:
            raise StudyContextError(
                {
                    "error": "study_covariate_selection_invalid",
                    "field": "covariate_selection",
                    "allowed": sorted(_COVARIATE_SELECTIONS),
                }
            )
        patch["covariate_selection"] = selection
    if "covariate_rationales" in raw:
        patch["covariate_rationales"] = normalize_covariate_rationales(
            raw.get("covariate_rationales")
        )
    if "covariate_temporal_roles" in raw:
        patch["covariate_temporal_roles"] = normalize_covariate_temporal_roles(
            raw.get("covariate_temporal_roles")
        )
    if "execution_concepts" in raw:
        patch["execution_concepts"] = normalize_execution_concepts(
            raw.get("execution_concepts")
        )
    if "analysis_design" in raw:
        patch["analysis_design"] = normalize_analysis_design(
            raw.get("analysis_design")
        )
    if "sensitivity_specs" in raw:
        patch["sensitivity_specs"] = normalize_sensitivity_specs(
            raw.get("sensitivity_specs")
        )
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
    if "idea_handoff" in raw:
        handoff = _schema_object(
            raw.get("idea_handoff"),
            field="idea_handoff",
            schema=_IDEA_HANDOFF_SCHEMA,
        )
        digest = str(handoff.get("canonical_handoff_sha256") or "")
        if digest and not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise StudyContextError(
                {
                    "error": "invalid_idea_handoff_digest",
                    "field": "idea_handoff.canonical_handoff_sha256",
                }
            )
        prior_art_digest = str(handoff.get("prior_art_sha256") or "")
        if prior_art_digest and not re.fullmatch(r"[a-f0-9]{64}", prior_art_digest):
            raise StudyContextError(
                {
                    "error": "invalid_prior_art_handoff_digest",
                    "field": "idea_handoff.prior_art_sha256",
                }
            )
        prior_art_count = handoff.get("prior_art_result_count")
        if prior_art_count is not None and (
            isinstance(prior_art_count, bool)
            or not isinstance(prior_art_count, int)
            or prior_art_count < 0
        ):
            raise StudyContextError(
                {
                    "error": "invalid_prior_art_handoff_count",
                    "field": "idea_handoff.prior_art_result_count",
                }
            )
        patch["idea_handoff"] = handoff
    if "literature_authority" in raw:
        if not allow_literature_authority:
            raise StudyContextError(
                {
                    "error": "study_literature_authority_server_owned",
                    "field": "literature_authority",
                }
            )
        authority = _schema_object(
            raw.get("literature_authority"),
            field="literature_authority",
            schema=_LITERATURE_AUTHORITY_SCHEMA,
        )
        for field in ("receipt_sha256", "study_configuration_sha256"):
            digest = str(authority.get(field) or "")
            if digest and not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise StudyContextError(
                    {
                        "error": "invalid_literature_authority_digest",
                        "field": f"literature_authority.{field}",
                    }
                )
        for field in ("result_count", "study_context_revision"):
            number = authority.get(field)
            if number is not None and (
                isinstance(number, bool) or not isinstance(number, int) or number < 0
            ):
                raise StudyContextError(
                    {
                        "error": (
                            "invalid_literature_authority_count"
                            if field == "result_count"
                            else "invalid_literature_authority_revision"
                        ),
                        "field": f"literature_authority.{field}",
                    }
                )
        patch["literature_authority"] = authority
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
            {field: row[field] for field in _CONTEXT_FIELDS if field in row},
            allow_literature_authority=True,
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


def _validate_covariate_decision_contract(context: Dict[str, Any]) -> None:
    roster = set(context.get("covariates") or [])
    rationale_keys = set((context.get("covariate_rationales") or {}).keys())
    temporal_keys = set((context.get("covariate_temporal_roles") or {}).keys())
    unbound = sorted((rationale_keys | temporal_keys) - roster)
    if unbound:
        raise StudyContextError(
            {
                "error": "study_covariate_decision_roster_mismatch",
                "field": "covariate_rationales",
                "unbound": unbound,
            }
        )
    if context.get("covariate_selection") != "exact" and (
        rationale_keys or temporal_keys
    ):
        raise StudyContextError(
            {
                "error": "study_covariate_decision_requires_exact_roster",
                "field": "covariate_selection",
            }
        )


def analysis_dependence_finding(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return one owner-issued conflict between cohort and inference authority.

    A stay-level model may treat rows as independent only when the configured
    cohort removes repeat ICU stays. If repeat stays are explicitly retained,
    ordinary model-based and heteroskedasticity-robust variance do not close
    within-patient dependence; the only currently expressible closure is a
    patient-clustered estimator. This function chooses no design. It prevents
    prose or an executor limitation from weakening a user-owned commitment.
    """

    raw_design = context.get("analysis_design")
    design = raw_design if isinstance(raw_design, dict) else {}
    raw_cohort = context.get("cohort")
    cohort = raw_cohort if isinstance(raw_cohort, dict) else {}
    if (
        not design
        or design.get("analysis_unit") != "icu_stay"
        or cohort.get("exclude_readmissions") is not False
        or design.get("variance_estimator") == "none_counts_only"
    ):
        return None
    if (
        design.get("variance_estimator") == "cluster_robust"
        and design.get("cluster_unit") == "patient"
    ):
        return None
    return {
        "error": "study_repeated_stay_dependence_unaddressed",
        "field": "analysis_design",
        "analysis_unit": "icu_stay",
        "exclude_readmissions": False,
        "required_design": {
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        },
        "alternative": (
            "set cohort.exclude_readmissions=true in a new "
            "user-authorized revision"
        ),
    }


def materialization_window_finding(
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return a conflict between Web wording and the executable outer window.

    ``time_window`` is the physical feature-materialization window.  The
    current cohort materializer interprets its numeric offsets from ICU
    admission.  A phenotype's clinical time zero (for example an event onset)
    comes from the concept owner, while a whole-stay outcome keeps its own
    outcome semantics.  Treating either of those as this physical anchor would
    silently execute different science from the conversation.

    Missing fields are allowed while a conversation is incomplete.  This
    finding covers only an explicit, unsupported commitment so legacy partial
    contexts remain readable and the workflow can ask for the missing slot.
    """

    raw_window = context.get("time_window")
    window = raw_window if isinstance(raw_window, dict) else {}
    raw_anchor = str(window.get("anchor") or "").strip()
    if not raw_anchor:
        return None
    normalized = re.sub(
        r"\s+",
        " ",
        raw_anchor.lower().replace("-", " ").replace("_", " "),
    ).strip()
    if normalized in {"icu admission", "admission"}:
        return None
    return {
        "error": "study_materialization_window_anchor_unsupported",
        "field": "time_window.anchor",
        "configured_anchor": raw_anchor,
        "supported_anchor": "icu_admission",
        "window_role": "outer_observation_window",
        "clinical_definition_anchor_owner": "concept_clinical_contract",
        "outcome_horizon_owner": "execution_concepts.outcome",
    }


def _validate_analysis_dependence_contract(context: Dict[str, Any]) -> None:
    finding = analysis_dependence_finding(context)
    if finding is not None:
        raise StudyContextError(finding)


def _validate_materialization_window_contract(context: Dict[str, Any]) -> None:
    finding = materialization_window_finding(context)
    if finding is not None:
        raise StudyContextError(finding)


def list_contexts() -> Dict[str, Any]:
    with _LOCK:
        raw = _read_raw()
        contexts = _contexts_from_raw(raw)
        listed = contexts[:_MAX_LISTED_CONTEXTS]
        active_id = raw.get("active_id")
        if active_id and not any(row.get("id") == active_id for row in listed):
            active_row = next(
                (row for row in contexts if row.get("id") == active_id), None
            )
            if active_row is not None and listed:
                listed[-1] = active_row
        return {
            "contexts": listed,
            "active_id": active_id,
        }


def get_context(context_id: str) -> Optional[Dict[str, Any]]:
    clean_id = _identifier(context_id, field="id")
    with _LOCK:
        for context in _contexts_from_raw(_read_raw()):
            if context.get("id") == clean_id:
                return context
    return None


def existing_context_ids(context_ids: List[str]) -> set[str]:
    """Return existing identifiers with one bounded store read."""
    clean_ids = {_identifier(value, field="id") for value in context_ids}
    if not clean_ids:
        return set()
    with _LOCK:
        return {
            str(context.get("id"))
            for context in _contexts_from_raw(_read_raw())
            if context.get("id") in clean_ids
        }


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
    _server_literature_authority_write: bool = False,
) -> Dict[str, Any]:
    patch = _sanitize_patch(
        raw_context,
        allow_literature_authority=_server_literature_authority_write,
    )
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
        if current is not None and "literature_authority" not in patch:
            binding = current.get("literature_authority")
            binding = binding if isinstance(binding, dict) else {}
            authority_schema = str(
                binding.get("schema_version") or _LITERATURE_AUTHORITY_V3
            )
            current_scope = literature_search_scope_sha256(
                current,
                schema_version=authority_schema,
            )
            proposed_scope = literature_search_scope_sha256(
                {**current, **patch},
                schema_version=authority_schema,
            )
            if proposed_scope != current_scope:
                # A Web search receipt is authority only for the exact typed
                # retrieval scope that produced its queries. Clear it in the
                # same atomic write when that scope changes so a later Plan
                # can never inherit candidates retrieved for another topic.
                patch["literature_authority"] = {}
        if patch or current is None:
            context.update(patch)
            _validate_covariate_decision_contract(context)
            _validate_analysis_dependence_contract(context)
            if "time_window" in patch:
                _validate_materialization_window_contract(context)
            context["revision"] = current_revision + 1
            context["updated_at"] = timestamp
        contexts = [row for row in contexts if row.get("id") != context_id]
        contexts.insert(0, context)
        active_id = context_id if active else raw.get("active_id")
        if len(contexts) > _MAX_STORED_CONTEXTS:
            raise StudyContextError(
                {
                    "error": "study_context_store_capacity_reached",
                    "max_contexts": _MAX_STORED_CONTEXTS,
                }
            )
        _write_raw(
            {
                "schema_version": 1,
                "updated_at": timestamp,
                "active_id": active_id,
                "contexts": contexts,
            }
        )
        return context


def bind_literature_authority(
    context_id: str,
    authority: Dict[str, Any],
    *,
    expected_revision: int,
) -> Dict[str, Any]:
    """Attach one server-issued literature receipt with revision CAS.

    Browser and model callers cannot populate this field through the generic
    metadata route.  The literature-authority owner creates the binding; this
    StudyContext owner alone commits it to the scientific configuration.
    """

    return upsert_context(
        {
            "id": _identifier(context_id, field="id"),
            "literature_authority": authority,
        },
        active=True,
        expected_revision=expected_revision,
        require_revision=True,
        lifecycle_write=False,
        _server_literature_authority_write=True,
    )


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
                "crossdb_selection",
                "cohort",
                "modules",
                "outcome",
                "primary_exposure",
                "covariates",
                "covariate_selection",
                "covariate_rationales",
                "covariate_temporal_roles",
                "execution_concepts",
                "analysis_design",
                "sensitivity_specs",
                "time_window",
                "comparator",
                "export_format",
                "analysis_goal",
                "current_stage",
                "last_route",
                "active_job_id",
                "confirmations",
                "idea_handoff",
                "literature_authority",
            )
        },
    }


def _scientific_fields_sha256(
    context: Dict[str, Any],
    *,
    fields: tuple[str, ...],
) -> str:
    sanitized = _sanitize_patch(
        {key: context.get(key) for key in fields},
        allow_literature_authority=True,
    )
    encoded = json.dumps(
        sanitized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def literature_search_scope_sha256(
    context: Dict[str, Any],
    *,
    schema_version: str = _LITERATURE_AUTHORITY_V3,
) -> str:
    """Digest the typed scope that generated one Web literature search.

    Version 2 bound retrieval to the entire scientific configuration.  That
    made the normal conversation order unusable: a user could search the
    exposure/outcome literature, then confirm a covariance estimator or time
    window, and that unrelated planning choice erased the search receipt.

    Version 3 binds exactly the fields used to compile the PubMed query:
    question, display exposure/outcome, and their execution concepts.  The
    Research Agent still re-screens every returned record against its sealed
    full ResearchContext; this narrower digest preserves *retrieval* evidence,
    not a prior comparator, novelty, eligibility, or method decision.

    ``schema_version`` remains explicit so already-issued v2 receipts can be
    verified with their historical scope contract.
    """

    if schema_version == _LITERATURE_AUTHORITY_V2:
        return _scientific_fields_sha256(
            context,
            fields=_LITERATURE_SCOPE_FIELDS_V2,
        )
    if schema_version != _LITERATURE_AUTHORITY_V3:
        raise StudyContextError(
            {
                "error": "literature_authority_schema_invalid",
                "schema_version": schema_version,
            }
        )
    execution = context.get("execution_concepts")
    execution = execution if isinstance(execution, dict) else {}
    scope = {
        "question": context.get("question"),
        "primary_exposure": context.get("primary_exposure"),
        "outcome": context.get("outcome"),
        "execution_concepts": {
            "primary_exposure": execution.get("primary_exposure"),
            "outcome": execution.get("outcome"),
        },
    }
    sanitized = _sanitize_patch(scope, allow_literature_authority=True)
    encoded = json.dumps(
        sanitized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scientific_configuration_sha256(context: Dict[str, Any]) -> str:
    """Digest only fields that can change a Research Agent scientific run.

    Lifecycle coordinates such as stage, route, revision, and active job are
    intentionally excluded.  A plan can therefore survive host bookkeeping,
    but not a changed question, source, cohort, analysis design, or evidence
    authority.
    """

    return _scientific_fields_sha256(
        context,
        fields=_SCIENTIFIC_CONFIGURATION_FIELDS,
    )


__all__ = [
    "StudyContextError",
    "analysis_dependence_finding",
    "materialization_window_finding",
    "bind_literature_authority",
    "build_agent_context_binding",
    "clear_active_job_if",
    "get_active_context",
    "get_context",
    "handoff_context",
    "list_contexts",
    "normalize_path",
    "normalize_analysis_design",
    "normalize_sensitivity_specs",
    "normalize_execution_concepts",
    "validate_context_update",
    "literature_search_scope_sha256",
    "scientific_configuration_sha256",
    "upsert_context",
]
