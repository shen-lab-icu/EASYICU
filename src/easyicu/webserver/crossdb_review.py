"""Bounded Cross-DB Review aggregates for the native FastAPI UI.

Stage18 compares two or more registered EasyICU exports using cohort-level
aggregates only. Matched cohorts, row-level filters, p-values/SMDs, and formal
cross-database claims remain fail-closed until the numeric evidence audit gate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from easyicu.webserver import cohort_review
from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store

_REQUIRED_CORE_MODULES = {"demographics", "outcome"}
_OPTIONAL_COMPARISON_MODULES = {"sepsis3_sofa2", "sofa2_score", "vitals"}
_UNSUPPORTED_FILTERS = {
    "row_level_filters": "Cross-DB Stage18 accepts registered-source aggregates only.",
    "age_at_admission": "Age cuts need audited row-level cohort construction.",
    "minimum_icu_los": "LOS filters must recompute denominators before comparison.",
    "sepsis3_positive_only": "Event-restricted cohorts need a dedicated cohort-builder pass.",
    "matched_cohort": "Matched cohorts are blocked until the numeric evidence audit gate.",
    "propensity_match": "Matched analyses are not implemented in the native Cross-DB aggregate path.",
    "patient_list": "Identifier-list filtering is intentionally not accepted by this endpoint.",
}
_UNSUPPORTED_STATISTICS = {
    "p_value": "Inferential statistics are withheld until the numeric evidence audit gate.",
    "p-values": "Inferential statistics are withheld until the numeric evidence audit gate.",
    "smd": "Standardized mean differences need audited row-level group construction.",
    "confidence_interval": "Confidence intervals need an audited statistical backend.",
    "matched_analysis": "Matched analyses are outside Stage18 Cross-DB Review.",
    "paired_sofa_reclassification": "Paired SOFA reclassification is blocked before the numeric audit gate.",
}


def crossdb_review_summary(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return native Cross-DB descriptive aggregates for registered exports."""
    _reject_unsupported_request(body)
    requested_sources = _resolve_registered_sources(body)

    cohort_payloads: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for source in requested_sources:
        safe_source = _safe_registered_source(source)
        try:
            cohort_payloads.append(cohort_review.cohort_review_summary({"source_path": source["path"]}))
        except cohort_review.CohortReviewError as exc:
            errors.append({
                "source": safe_source,
                "error": (exc.detail or {}).get("error") or "cohort_summary_failed",
                "detail": _safe_error_detail(exc.detail),
            })
    if errors:
        raise CrossdbReviewError({
            "error": "invalid_export",
            "sources": [_safe_registered_source(source) for source in requested_sources],
            "errors": errors,
            "privacy": _privacy_payload(),
        })

    sources = [
        _source_aggregate(source, payload)
        for source, payload in zip(requested_sources, cohort_payloads)
    ]
    module_sets = [set(source.get("modules") or []) for source in sources]
    shared_modules = sorted(set.intersection(*module_sets)) if module_sets else []
    all_modules = sorted(set.union(*module_sets)) if module_sets else []
    compatibility_gate = _compatibility_gate(sources, shared_modules, all_modules)
    blocked_features = _blocked_features()

    if compatibility_gate["status"] != "compatible":
        raise CrossdbReviewError({
            "error": "crossdb_incompatible",
            "mode": "real",
            "demo": False,
            "source_count": len(sources),
            "sources": sources,
            "shared_modules": shared_modules,
            "all_modules": all_modules,
            "compatibility_gate": compatibility_gate,
            "blocked_features": blocked_features,
            "privacy": _privacy_payload(),
        })

    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source_count": len(sources),
        "sources": sources,
        "rows": _comparison_rows(sources, compatibility_gate),
        "availability": _module_availability(sources, all_modules),
        "shared_modules": shared_modules,
        "all_modules": all_modules,
        "compatibility_gate": compatibility_gate,
        "provenance": {
            "computed_from": [
                "source_registry",
                "export_manifest",
                "bounded_column_reads",
                "cohort_level_aggregates",
            ],
            "payload_scope": "cross_database_aggregate_only",
            "inference": "blocked_until_numeric_evidence_gate",
        },
        "privacy": _privacy_payload(),
        "blocked_features": blocked_features,
    }


def _reject_unsupported_request(body: Dict[str, Any]) -> None:
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    requested_filters = [
        {"id": str(key), "reason": _UNSUPPORTED_FILTERS.get(str(key), "Cross-DB Stage18 does not accept row-level filters.")}
        for key, value in filters.items()
        if _truthy_request(value)
    ]
    if _truthy_request(body.get("matched_cohort")):
        requested_filters.append({"id": "matched_cohort", "reason": _UNSUPPORTED_FILTERS["matched_cohort"]})
    comparison = body.get("comparison")
    if isinstance(comparison, dict) and _truthy_request(comparison.get("matched_cohort")):
        requested_filters.append({"id": "matched_cohort", "reason": _UNSUPPORTED_FILTERS["matched_cohort"]})
    if requested_filters:
        raise CrossdbReviewError({
            "error": "unsupported_filter",
            "unsupported": requested_filters,
            "supported_scope": "registered_source_crossdb_aggregates_only",
        })

    stats = body.get("statistics") or body.get("stats") or []
    if isinstance(stats, str):
        stats = [stats]
    requested_stats = [
        {
            "id": str(item),
            "reason": _UNSUPPORTED_STATISTICS.get(str(item), "Requested statistic is not supported by the Stage18 aggregate endpoint."),
        }
        for item in stats
        if _truthy_request(item)
    ]
    if requested_stats:
        raise CrossdbReviewError({
            "error": "unsupported_statistic",
            "unsupported": requested_stats,
            "supported_scope": "descriptive_crossdb_aggregate_only",
        })


def _resolve_registered_sources(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    registry = source_store.load_registry()
    sources = [s for s in registry.get("sources") or [] if isinstance(s, dict) and s.get("ok")]
    by_path = {_norm_path(str(s.get("path") or "")): s for s in sources if s.get("path")}
    requested = body.get("paths") or body.get("source_paths")

    if requested is not None:
        if not isinstance(requested, list):
            raise CrossdbReviewError({"error": "paths_must_be_list"})
        selected: List[Dict[str, Any]] = []
        seen = set()
        for raw in requested:
            norm = _norm_path(str(raw or ""))
            if not norm or norm in seen:
                continue
            seen.add(norm)
            source = by_path.get(norm)
            if source is None:
                raise CrossdbReviewError({"error": "source_not_registered", "path_hash": _hash(norm)})
            selected.append(source)
    else:
        selected = []
        seen = set()
        for raw in registry.get("crossdb_paths") or []:
            norm = _norm_path(str(raw or ""))
            if not norm or norm in seen:
                continue
            seen.add(norm)
            source = by_path.get(norm)
            if source is not None:
                selected.append(source)

    if len(selected) < 2:
        raise CrossdbReviewError({
            "error": "need_two_exports",
            "source_count": len(selected),
            "sources": [_safe_registered_source(source) for source in selected],
            "privacy": _privacy_payload(),
        })
    return selected


def _source_aggregate(source: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    safe_source = dict(payload.get("source") or _safe_registered_source(source))
    summary = payload.get("summary") or {}
    coverage = payload.get("coverage") or []
    quality = payload.get("quality") or {}
    modules = sorted({str(row.get("module")) for row in coverage if row.get("module")})
    if not modules:
        modules = sorted(str(module) for module in (source.get("modules") or []) if module)
    module_coverage = {
        str(row.get("module")): {
            "rows": int(row.get("rows") or 0),
            "coverage_pct": row.get("coverage_pct"),
            "quality_status": row.get("quality_status") or "unknown",
        }
        for row in coverage
        if row.get("module")
    }
    return {
        "id": safe_source.get("id"),
        "label": safe_source.get("label"),
        "path_hash": safe_source.get("path_hash") or _hash(str(source.get("path") or "")),
        "database": safe_source.get("database"),
        "generated": safe_source.get("generated"),
        "summary": {
            "cohort_size": summary.get("cohort_size"),
            "modules": summary.get("modules"),
            "file_count": summary.get("file_count"),
            "total_records": summary.get("total_records"),
            "mortality_pct": summary.get("mortality_pct"),
            "age_mean": (summary.get("age") or {}).get("mean"),
            "age_median": (summary.get("age") or {}).get("median"),
            "female_pct": (summary.get("sex") or {}).get("female_pct"),
            "sofa2_median": (summary.get("sofa2") or {}).get("median"),
            "los_median": (summary.get("los_icu_days") or {}).get("median"),
            "sepsis_pct": summary.get("sepsis_pct"),
            "coverage_median_pct": quality.get("median_coverage_pct"),
            "quality_watchlist_count": quality.get("watchlist_count"),
        },
        "modules": modules,
        "module_coverage": module_coverage,
        "quality": {
            "modules_ok": quality.get("modules_ok"),
            "modules_warn": quality.get("modules_warn"),
            "modules_bad": quality.get("modules_bad"),
            "modules_neutral": quality.get("modules_neutral"),
            "modules_unknown": quality.get("modules_unknown"),
            "watchlist_count": quality.get("watchlist_count"),
            "median_coverage_pct": quality.get("median_coverage_pct"),
        },
    }


def _compatibility_gate(
    sources: List[Dict[str, Any]],
    shared_modules: List[str],
    all_modules: List[str],
) -> Dict[str, Any]:
    shared = set(shared_modules)
    checks: List[Dict[str, Any]] = []
    reasons: List[Dict[str, Any]] = []

    enough_sources = len(sources) >= 2
    checks.append({"id": "source_count", "passed": enough_sources, "value": len(sources), "minimum": 2})
    if not enough_sources:
        reasons.append({"id": "need_two_exports", "detail": "At least two registered exports are required."})

    denominators = [
        {"label": source.get("label"), "cohort_size": (source.get("summary") or {}).get("cohort_size")}
        for source in sources
    ]
    denominator_ok = all(isinstance(row["cohort_size"], (int, float)) and row["cohort_size"] > 0 for row in denominators)
    checks.append({"id": "denominator_present", "passed": denominator_ok, "sources": denominators})
    if not denominator_ok:
        reasons.append({"id": "missing_denominator", "sources": denominators})

    missing_core = sorted(_REQUIRED_CORE_MODULES - shared)
    checks.append({
        "id": "core_modules_shared",
        "passed": not missing_core,
        "required_modules": sorted(_REQUIRED_CORE_MODULES),
        "shared_modules": shared_modules,
        "missing_modules": missing_core,
    })
    if missing_core:
        reasons.append({
            "id": "core_modules_not_shared",
            "missing_shared_modules": missing_core,
            "sources": [
                {
                    "label": source.get("label"),
                    "missing_core_modules": sorted(_REQUIRED_CORE_MODULES - set(source.get("modules") or [])),
                }
                for source in sources
            ],
        })

    comparable_metrics = [
        "cohort_size",
        "modules",
        "total_records",
        "age_mean",
        "age_median",
        "female_pct",
        "mortality_pct",
        "coverage_median_pct",
    ]
    if "sepsis3_sofa2" in shared:
        comparable_metrics.append("sepsis_pct")
    if "sofa2_score" in shared:
        comparable_metrics.append("sofa2_median")
    if "outcome" in shared:
        comparable_metrics.append("los_median")

    warnings: List[Dict[str, Any]] = []
    missing_optional = sorted((_OPTIONAL_COMPARISON_MODULES & set(all_modules)) - shared)
    if missing_optional:
        warnings.append({
            "id": "optional_modules_not_shared",
            "modules": missing_optional,
            "effect": "dependent descriptive metrics are omitted from comparison rows",
        })

    compatible = all(check["passed"] for check in checks)
    return {
        "status": "compatible" if compatible else "incompatible",
        "comparison_mode": "descriptive_only",
        "matched_cohort": False,
        "matched_cohort_ready": False,
        "descriptive_only": True,
        "inferential_statistics_allowed": False,
        "claim_level": "preview_not_reportable",
        "checks": checks,
        "reasons": reasons,
        "warnings": warnings,
        "comparable_metrics": comparable_metrics if compatible else [],
    }


def _comparison_rows(sources: List[Dict[str, Any]], gate: Dict[str, Any]) -> List[Dict[str, Any]]:
    comparable = set(gate.get("comparable_metrics") or [])
    rows: List[Dict[str, Any]] = []
    for key, label, digits, dependency in [
        ("cohort_size", "Cohort size", 0, "demographics"),
        ("modules", "Modules", 0, None),
        ("total_records", "Records", 0, None),
        ("age_mean", "Mean age", 1, "demographics"),
        ("age_median", "Median age", 1, "demographics"),
        ("female_pct", "Female %", 1, "demographics"),
        ("mortality_pct", "Mortality %", 1, "outcome"),
        ("los_median", "Median ICU LOS", 1, "outcome"),
        ("sepsis_pct", "Sepsis-3 %", 1, "sepsis3_sofa2"),
        ("sofa2_median", "Median SOFA-2", 1, "sofa2_score"),
        ("coverage_median_pct", "Median module coverage %", 1, None),
    ]:
        if dependency and key not in comparable:
            continue
        values = [(source.get("summary") or {}).get(key) for source in sources]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        delta = round(max(numeric) - min(numeric), digits) if len(numeric) >= 2 else None
        rows.append({
            "key": key,
            "label": label,
            "values": values,
            "delta": delta,
            "comparison": "descriptive_range",
        })
    return rows


def _module_availability(sources: List[Dict[str, Any]], all_modules: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for module in all_modules:
        values = []
        present_count = 0
        coverage_values: List[float] = []
        for source in sources:
            module_info = (source.get("module_coverage") or {}).get(module)
            present = module_info is not None
            if present:
                present_count += 1
                coverage = module_info.get("coverage_pct")
                if isinstance(coverage, (int, float)):
                    coverage_values.append(float(coverage))
            values.append({
                "source": source.get("label"),
                "present": present,
                "coverage_pct": module_info.get("coverage_pct") if module_info else None,
                "quality_status": module_info.get("quality_status") if module_info else "missing",
            })
        out.append({
            "module": module,
            "present_count": present_count,
            "source_count": len(sources),
            "shared": present_count == len(sources),
            "median_coverage_pct": dataio._median(coverage_values),
            "values": values,
        })
    return out


def _safe_registered_source(source: Dict[str, Any]) -> Dict[str, Any]:
    raw_path = str(source.get("path") or "")
    return {
        "id": source.get("id"),
        "label": source.get("label") or Path(raw_path).name or "local",
        "path_hash": _hash(_norm_path(raw_path)),
        "database": source.get("database"),
        "generated": source.get("generated"),
    }


def _safe_error_detail(detail: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(detail, dict):
        return {}
    return {
        key: value
        for key, value in detail.items()
        if key not in {"source", "selected", "entities", "groups", "coverage", "summary"}
    }


def _privacy_payload() -> Dict[str, Any]:
    return {
        "raw_rows_returned": False,
        "direct_identifiers_returned": False,
        "patient_rows_persisted": False,
        "secrets_returned": False,
    }


def _blocked_features() -> List[Dict[str, Any]]:
    return [
        {
            "id": "row_level_filters",
            "status": "blocked",
            "reason": "Native Cross-DB Stage18 compares registered export aggregates only.",
        },
        {
            "id": "inferential_statistics",
            "status": "blocked",
            "reason": "No p-values, SMDs, or confidence intervals before the numeric evidence audit gate.",
        },
        {
            "id": "matched_cohort",
            "status": "blocked",
            "reason": "Matched cohorts need audited row-level construction and are not implemented here.",
        },
        {
            "id": "reportable_claims",
            "status": "blocked",
            "reason": "Cross-DB preview remains analysis-only until Stage19 numeric evidence audit.",
        },
    ]


def _truthy_request(value: Any) -> bool:
    if value in (None, False, "", [], {}):
        return False
    return True


def _norm_path(raw: str) -> str:
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    return str(path)


def _hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


class CrossdbReviewError(Exception):
    def __init__(self, detail: Dict[str, Any]):
        super().__init__(str(detail.get("error") or "crossdb_review_error"))
        self.detail = detail
