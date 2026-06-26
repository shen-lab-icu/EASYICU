"""Metadata-only advanced filters for the native Data Extraction screen.

The endpoints backed by this module describe and preview filters over a
registered EasyICU export source. They intentionally return module/file
metadata and aggregate coverage only, never row-level values.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store

_ID_COLUMNS = {
    "stay_id",
    "subject_id",
    "hadm_id",
    "patient_id",
    "patientunitstayid",
    "patienthealthsystemstayid",
    "uniquepid",
}

_SUPPORTED_FILTERS = {
    "database",
    "modules",
    "required_columns",
    "min_rows",
    "max_rows",
    "min_coverage_pct",
    "quality_statuses",
}

_UNSUPPORTED_FILTERS = {
    "age_at_admission": "The export manifest does not carry a validated age-distribution index.",
    "minimum_icu_los": "LOS cohort cuts require row-level cohort recomputation before extraction.",
    "observation_window": "Observation windows must be applied by the extraction job, not inferred from an export manifest.",
    "sepsis3_positive_only": "Sepsis-3 cohort restriction needs a validated cohort-builder pass.",
    "exclude_readmissions": "Readmission exclusion is not encoded in the current export manifest.",
    "patient_list": "Identifier-list filtering is intentionally not accepted by this metadata endpoint.",
    "icd_cohort": "ICD cohort filters need source diagnosis tables and are not represented in module exports.",
}


def filter_options(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return bounded filter options for the active or explicit registered source."""
    source, desc = _resolve_registered_source(body)
    modules = _module_options(desc)
    row_counts = [
        m["row_count"] for m in modules if isinstance(m.get("row_count"), int)
    ]
    coverages = [
        m["coverage_pct"]
        for m in modules
        if isinstance(m.get("coverage_pct"), (int, float))
    ]
    database = desc.get("database") or source.get("database") or "unknown"
    cohort_size = (desc.get("summary") or {}).get("stays")
    total_rows = (desc.get("summary") or {}).get("total_rows")

    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source": _source_provenance(source, desc),
        "provenance": {
            "computed_from": [
                "source_registry",
                "export_manifest",
                "file_schemas",
                "identifier_column_counts",
            ],
            "row_count_basis": "manifest_rows_with_file_metadata_fallback",
            "coverage_basis": "module_identifier_count_over_cohort_denominator",
        },
        "privacy": {
            "raw_rows_returned": False,
            "identifier_values_returned": False,
        },
        "summary": {
            "cohort_size": cohort_size,
            "modules": len(modules),
            "file_count": len(desc.get("files") or []),
            "total_rows": total_rows,
        },
        "options": {
            "databases": [{"value": database, "label": str(database).upper()}],
            "sources": [_source_provenance(source, desc)],
            "modules": modules,
            "quality_statuses": sorted({m["quality_status"] for m in modules}),
            "row_count_range": _range(row_counts),
            "coverage_pct_range": _range(coverages),
            "columns": _column_options(modules),
        },
        "filters": {
            "supported": [
                {"id": "database", "status": "supported", "basis": "manifest.database"},
                {
                    "id": "modules",
                    "status": "supported",
                    "basis": "manifest.files.module",
                },
                {
                    "id": "required_columns",
                    "status": "supported",
                    "basis": "file_schemas_without_identifier_columns",
                },
                {
                    "id": "min_rows",
                    "status": "supported",
                    "basis": "manifest.rows_or_file_metadata",
                },
                {
                    "id": "max_rows",
                    "status": "supported",
                    "basis": "manifest.rows_or_file_metadata",
                },
                {
                    "id": "min_coverage_pct",
                    "status": "supported",
                    "basis": "bounded_identifier_coverage",
                },
                {
                    "id": "quality_statuses",
                    "status": "supported",
                    "basis": "coverage_thresholds",
                },
            ],
            "unsupported": [
                {"id": key, "status": "unsupported", "reason": reason}
                for key, reason in _UNSUPPORTED_FILTERS.items()
            ],
        },
    }


def filter_preview(body: Dict[str, Any]) -> Dict[str, Any]:
    """Apply supported metadata filters. Unsupported requests fail closed."""
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    unsupported = _requested_unsupported(filters)
    if unsupported:
        return {
            "ok": False,
            "error": "unsupported_filter",
            "unsupported": unsupported,
            "supported_filters": sorted(_SUPPORTED_FILTERS),
        }

    options = filter_options(body)
    modules = list((options.get("options") or {}).get("modules") or [])
    database_filter = filters.get("database")
    if database_filter:
        wanted = {str(v).lower() for v in _list(database_filter)}
        current = str(options["source"].get("database") or "unknown").lower()
        if current not in wanted:
            modules = []

    module_filter = {str(v) for v in _list(filters.get("modules"))}
    if module_filter:
        modules = [m for m in modules if str(m.get("module")) in module_filter]

    required_columns = {str(v).lower() for v in _list(filters.get("required_columns"))}
    if required_columns:
        modules = [
            m
            for m in modules
            if required_columns <= {str(c).lower() for c in m.get("columns") or []}
        ]

    min_rows = _number(filters.get("min_rows"))
    if min_rows is not None:
        modules = [m for m in modules if (m.get("row_count") or 0) >= min_rows]

    max_rows = _number(filters.get("max_rows"))
    if max_rows is not None:
        modules = [m for m in modules if (m.get("row_count") or 0) <= max_rows]

    min_coverage = _number(filters.get("min_coverage_pct"))
    if min_coverage is not None:
        modules = [
            m
            for m in modules
            if isinstance(m.get("coverage_pct"), (int, float))
            and m["coverage_pct"] >= min_coverage
        ]

    quality_filter = {
        str(v) for v in _list(filters.get("quality_statuses")) if str(v) != "all"
    }
    if quality_filter:
        modules = [m for m in modules if str(m.get("quality_status")) in quality_filter]

    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source": options["source"],
        "provenance": options["provenance"],
        "privacy": options["privacy"],
        "applied_filters": _clean_applied_filters(filters),
        "matched_modules": modules,
        "match_count": len(modules),
        "aggregate": {
            "cohort_size": (options.get("summary") or {}).get("cohort_size"),
            "matched_rows": sum(int(m.get("row_count") or 0) for m in modules),
            "matched_file_count": len(modules),
        },
    }


def _resolve_registered_source(
    body: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    registry = source_store.load_registry()
    sources = [s for s in registry.get("sources") or [] if isinstance(s, dict)]
    requested = body.get("source_path") or body.get("path")
    if requested:
        norm = _norm_path(str(requested))
        source = next(
            (s for s in sources if _norm_path(str(s.get("path") or "")) == norm), None
        )
        if source is None:
            raise ExtractionFilterError(
                {
                    "error": "source_not_registered",
                    "path_hash": _hash(norm),
                }
            )
    else:
        active = registry.get("active_path")
        if not active:
            raise ExtractionFilterError({"error": "no_active_export"})
        active_norm = _norm_path(str(active))
        source = next(
            (s for s in sources if _norm_path(str(s.get("path") or "")) == active_norm),
            None,
        )
        if source is None:
            raise ExtractionFilterError(
                {
                    "error": "active_source_not_registered",
                    "path_hash": _hash(active_norm),
                }
            )

    desc = dataio.describe_export_source(str(source.get("path") or ""))
    if not desc.get("ok"):
        raise ExtractionFilterError(
            {"error": "invalid_export", "detail": desc.get("error")}
        )
    return source, desc


def _module_options(desc: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = Path(str(desc.get("path") or "")).expanduser()
    cohort_size = (desc.get("summary") or {}).get("stays")
    out: List[Dict[str, Any]] = []
    for item in desc.get("files") or []:
        file_name = str(item.get("file") or "")
        if not file_name:
            continue
        columns = [str(c) for c in item.get("columns") or []]
        safe_columns = [c for c in columns if not _is_identifier_column(c)]
        coverage_pct, covered = _coverage(path / file_name, cohort_size)
        module = str(item.get("module") or Path(file_name).stem.split("__", 1)[0])
        out.append(
            {
                "module": module,
                "metric_kind": dataio._presence_rate_kind(module) or "coverage",
                "table": Path(file_name).stem,
                "file": file_name,
                "row_count": int(item.get("rows") or 0),
                "columns": safe_columns,
                "hidden_identifier_columns": len(columns) - len(safe_columns),
                "coverage_pct": coverage_pct,
                "covered_entities": covered,
                "coverage_denominator": cohort_size,
                "quality_status": _quality_status(module, coverage_pct),
            }
        )
    return out


def _coverage(path: Path, cohort_size: Any) -> Tuple[float | None, int | None]:
    if not isinstance(cohort_size, int) or cohort_size <= 0:
        return None, None
    ids = dataio._read_stay_ids(
        path
    )  # reads one identifier column only, not full frames.
    if ids is None:
        return None, None
    covered = len(ids)
    return round(min(covered, cohort_size) / cohort_size * 100, 1), covered


def _quality_status(module: str, coverage_pct: float | None) -> str:
    if coverage_pct is None:
        return "unknown"
    if dataio._is_presence_rate_module(module):
        return "neutral"
    if coverage_pct >= 80:
        return "ok"
    if coverage_pct >= 50:
        return "warn"
    return "bad"


def _source_provenance(source: Dict[str, Any], desc: Dict[str, Any]) -> Dict[str, Any]:
    path = str(desc.get("path") or source.get("path") or "")
    return {
        "id": source.get("id"),
        "label": source.get("label") or Path(path).name or "local",
        "path_hash": _hash(path),
        "database": desc.get("database") or source.get("database"),
        "generated": desc.get("generated") or source.get("generated"),
    }


def _requested_unsupported(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    unsupported: List[Dict[str, Any]] = []
    for key, value in filters.items():
        if not _truthy_request(value):
            continue
        if key in _UNSUPPORTED_FILTERS:
            unsupported.append({"id": key, "reason": _UNSUPPORTED_FILTERS[key]})
        elif key not in _SUPPORTED_FILTERS:
            unsupported.append(
                {"id": key, "reason": "Unknown filter for this endpoint."}
            )
    return unsupported


def _clean_applied_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in filters.items()
        if k in _SUPPORTED_FILTERS and _truthy_request(v)
    }


def _column_options(modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for module in modules:
        for column in module.get("columns") or []:
            counts[column] = counts.get(column, 0) + 1
    return [
        {"name": name, "module_count": count}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :80
        ]
    ]


def _range(values: List[Any]) -> Dict[str, Any]:
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return {"min": None, "max": None}
    return {"min": min(numeric), "max": max(numeric)}


def _list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy_request(value: Any) -> bool:
    if value in (None, False, "", [], {}):
        return False
    return True


def _is_identifier_column(column: str) -> bool:
    return column.strip().lower() in _ID_COLUMNS


def _norm_path(raw: str) -> str:
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    return str(path)


def _hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


class ExtractionFilterError(Exception):
    def __init__(self, detail: Dict[str, Any]):
        super().__init__(str(detail.get("error") or "extraction_filter_error"))
        self.detail = detail
