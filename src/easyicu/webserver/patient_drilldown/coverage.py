"""Feature-coverage index for bounded Patient Review.

This owner separates three facts that the UI must not conflate:

* a concept exists in the EasyICU catalog;
* an export materialized a typed column for that concept;
* the exported column contains at least one non-null observation.

Parquet null counts are read from row-group metadata, so even very large exports
can expose all 288 concept statuses without loading patient rows.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, Mapping

from easyicu.concept import catalog as concept_catalog
from easyicu.webserver import review_labels


SCHEMA_VERSION = "easyicu.patient_feature_coverage/1"
TIME_COLUMNS = (
    "charttime",
    "time",
    "datetime",
    "timestamp",
    "starttime",
    "endtime",
    "hour",
    "measuredat_minutes",
    "observationoffset",
)
_CACHE_LIMIT = 8
_CACHE: OrderedDict[str, tuple[tuple[Any, ...], Dict[str, Any]]] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def build_feature_coverage(
    export_path: Path, description: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return a row-free 288-concept coverage index for one export."""

    root = export_path.expanduser()
    signature = _source_signature(root, description.get("files") or [])
    key = str(root.resolve())
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == signature:
            _CACHE.move_to_end(key)
            return deepcopy(cached[1])

    payload = _build_feature_coverage(root, description)
    with _CACHE_LOCK:
        _CACHE[key] = (signature, payload)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return deepcopy(payload)


def apply_to_module_profiles(
    profiles: Iterable[Mapping[str, Any]], coverage: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    """Merge export-wide feature facts into existing module profile rows."""

    coverage_by_module = {
        str(row.get("module") or ""): row
        for row in coverage.get("modules") or []
        if isinstance(row, Mapping)
    }
    out: list[Dict[str, Any]] = []
    for profile in profiles:
        row = dict(profile)
        module = str(row.get("module") or "")
        module_coverage = coverage_by_module.get(module)
        if module_coverage:
            summary = module_coverage.get("summary") or {}
            observed = int(summary.get("observed") or 0)
            trajectory = int(summary.get("trajectory_candidates") or 0)
            row.update(
                {
                    "catalog_feature_count": int(summary.get("definitions") or 0),
                    "export_observed_features": observed,
                    "trajectory_candidate_features": trajectory,
                    "export_static_observed_features": max(0, observed - trajectory),
                    "materialized_features": int(summary.get("materialized") or 0),
                    "all_null_features": int(summary.get("all_null") or 0),
                    "unsupported_features": int(summary.get("unsupported") or 0),
                    "unknown_observation_features": int(
                        summary.get("materialized_unknown") or 0
                    ),
                    "feature_coverage_scope": "export_column_statistics",
                }
            )
        out.append(row)
    return out


def clear_cache() -> None:
    """Clear process-local metadata cache (used by focused tests)."""

    with _CACHE_LOCK:
        _CACHE.clear()


def _build_feature_coverage(
    root: Path, description: Mapping[str, Any]
) -> Dict[str, Any]:
    files_by_module = {
        str(item.get("module") or ""): dict(item)
        for item in description.get("files") or []
        if isinstance(item, Mapping) and item.get("module")
    }
    unavailable = _structurally_unavailable(root)
    modules: list[Dict[str, Any]] = []
    all_features: list[Dict[str, Any]] = []

    for module, concept_ids in concept_catalog.CONCEPT_GROUPS_INTERNAL.items():
        item = files_by_module.get(module)
        columns = [str(value) for value in ((item or {}).get("columns") or [])]
        time_column = next((name for name in TIME_COLUMNS if name in columns), None)
        stats = _column_stats(root, item, concept_ids) if item else {}
        features: list[Dict[str, Any]] = []
        for concept_id in concept_ids:
            metadata = concept_catalog.CONCEPT_DICTIONARY.get(concept_id) or ()
            structural = unavailable.get(concept_id)
            materialized = concept_id in columns
            column = stats.get(concept_id) or {}
            non_null_count = column.get("non_null_count")
            numeric = column.get("numeric")
            if structural:
                status = "structurally_unavailable"
                reason_code = str(
                    structural.get("reason_code") or "feature_structurally_unavailable"
                )
            elif not materialized:
                status = "not_materialized"
                reason_code = "feature_not_materialized"
            elif non_null_count is None:
                status = "materialized_unknown"
                reason_code = "column_statistics_unavailable"
            elif int(non_null_count) <= 0:
                status = "all_null"
                reason_code = "sample_has_no_observation"
            else:
                status = "observed"
                reason_code = None
            trajectory_candidate = bool(
                status == "observed"
                and numeric is True
                and time_column
                and int(non_null_count or 0) >= 2
            )
            feature = {
                "feature": concept_id,
                "module": module,
                "name": str(metadata[0] if metadata else concept_id),
                "name_zh": str(
                    metadata[1]
                    if len(metadata) > 1 and metadata[1]
                    else (metadata[0] if metadata else concept_id)
                ),
                "unit": str(metadata[2] if len(metadata) > 2 else ""),
                "status": status,
                "reason_code": reason_code,
                "materialized": materialized,
                "non_null_count": (
                    int(non_null_count) if non_null_count is not None else None
                ),
                "statistics_scope": str(
                    column.get("statistics_scope") or "schema_only"
                ),
                "numeric": numeric,
                "time_indexed": bool(time_column),
                "time_column": time_column,
                "trajectory_candidate": trajectory_candidate,
                "loadable": status in {"observed", "materialized_unknown"},
            }
            if structural:
                feature["supported_databases"] = list(
                    structural.get("supported_databases") or []
                )
            features.append(feature)
            all_features.append(feature)

        modules.append(
            {
                "module": module,
                "label": _module_label(module, 0),
                "label_zh": _module_label(module, 1),
                "file": (item or {}).get("file"),
                "rows": int((item or {}).get("rows") or 0),
                "time_column": time_column,
                "summary": _feature_summary(features),
                "features": features,
            }
        )

    summary = _feature_summary(all_features)
    summary["modules"] = len(modules)
    return {
        "schema_version": SCHEMA_VERSION,
        "database": description.get("database"),
        "summary": summary,
        "modules": modules,
        "provenance": {
            "computed_from": [
                "easyicu_concept_catalog",
                "export_manifest",
                "export_column_schema",
                "parquet_row_group_null_statistics",
            ],
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
            "payload_scope": "export_wide_feature_coverage_metadata",
        },
    }


def _feature_summary(features: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    rows = list(features)
    return {
        "definitions": len(rows),
        "materialized": sum(bool(row.get("materialized")) for row in rows),
        "observed": sum(row.get("status") == "observed" for row in rows),
        "all_null": sum(row.get("status") == "all_null" for row in rows),
        "not_materialized": sum(
            row.get("status") == "not_materialized" for row in rows
        ),
        "unsupported": sum(
            row.get("status") == "structurally_unavailable" for row in rows
        ),
        "materialized_unknown": sum(
            row.get("status") == "materialized_unknown" for row in rows
        ),
        "trajectory_candidates": sum(
            bool(row.get("trajectory_candidate")) for row in rows
        ),
        "loadable": sum(bool(row.get("loadable")) for row in rows),
    }


def _module_label(module: str, index: int) -> str:
    return review_labels.module_label(module, "zh" if index else "en")


def _structurally_unavailable(root: Path) -> Dict[str, Dict[str, Any]]:
    manifest = root / "_manifest.json"
    if not manifest.exists():
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    availability = payload.get("concept_availability") or {}
    rows = availability.get("structurally_unavailable") or []
    return {
        str(row.get("concept_id")): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("concept_id")
    }


def _column_stats(
    root: Path,
    item: Mapping[str, Any] | None,
    concept_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    if not item or not item.get("file"):
        return {}
    path = root / str(item["file"])
    if path.suffix.lower() != ".parquet":
        return {}
    try:
        import pyarrow.parquet as pq
        import pyarrow.types as pat

        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        index_by_name = {name: index for index, name in enumerate(schema.names)}
        out: Dict[str, Dict[str, Any]] = {}
        for concept_id in concept_ids:
            column_index = index_by_name.get(concept_id)
            if column_index is None:
                continue
            field_type = schema.field(concept_id).type
            numeric = bool(
                pat.is_integer(field_type)
                or pat.is_floating(field_type)
                or pat.is_decimal(field_type)
            )
            known = True
            non_null_count = 0
            for row_group_index in range(parquet.metadata.num_row_groups):
                row_group = parquet.metadata.row_group(row_group_index)
                statistics = row_group.column(column_index).statistics
                if statistics is None or statistics.null_count is None:
                    known = False
                    break
                non_null_count += int(row_group.num_rows) - int(statistics.null_count)
            out[concept_id] = {
                "non_null_count": non_null_count if known else None,
                "numeric": numeric,
                "statistics_scope": (
                    "parquet_row_group_exact" if known else "parquet_schema_only"
                ),
            }
        return out
    except Exception:
        return {}


def _source_signature(
    root: Path, files: Iterable[Mapping[str, Any]]
) -> tuple[Any, ...]:
    signature: list[Any] = []
    manifest = root / "_manifest.json"
    signature.append(_file_signature(manifest))
    for item in files:
        module = str(item.get("module") or "")
        if module not in concept_catalog.CONCEPT_GROUPS_INTERNAL:
            continue
        signature.append(_file_signature(root / str(item.get("file") or "")))
    return tuple(signature)


def _file_signature(path: Path) -> tuple[Any, ...]:
    try:
        stat = path.stat()
        return (
            path.name,
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
            int(stat.st_dev),
            int(stat.st_ino),
        )
    except OSError:
        return (path.name, None, None, None, None, None)


__all__ = [
    "SCHEMA_VERSION",
    "TIME_COLUMNS",
    "apply_to_module_profiles",
    "build_feature_coverage",
    "clear_cache",
]
