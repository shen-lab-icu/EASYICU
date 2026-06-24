"""Bounded Cohort Review aggregates for the native FastAPI UI.

This endpoint is the Stage17 Cohort Review parity path. It consumes the active
registered EasyICU export and returns cohort-level aggregates only. Row-level
filters, inferential statistics, matched cohorts, and paired SOFA
reclassification stay fail-closed until their backend contracts exist.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store

_READ_MODULES = ("demographics", "outcome", "sofa2_score", "sepsis3_sofa2")
_MODULE_COLUMNS = {
    "demographics": (
        "stay_id",
        "age",
        "age_at_admission",
        "sex",
        "gender",
    ),
    "outcome": (
        "stay_id",
        "death",
        "mortality",
        "hospital_mortality",
        "in_hospital_mortality",
        "los_icu",
        "icu_los",
        "los_days",
    ),
    "sofa2_score": (
        "stay_id",
        "sofa2",
        "sofa2_total",
        "sofa_score",
        "score",
    ),
    "sepsis3_sofa2": (
        "stay_id",
        "sep3_sofa2",
        "sepsis3_sofa2",
        "sepsis3",
        "sepsis",
    ),
}
_AGE_COLUMNS = ("age", "age_at_admission")
_SEX_COLUMNS = ("sex", "gender")
_DEATH_COLUMNS = ("death", "mortality", "hospital_mortality", "in_hospital_mortality")
_LOS_COLUMNS = ("los_icu", "icu_los", "los_days")
_SOFA_COLUMNS = ("sofa2", "sofa2_total", "sofa_score", "score")
_SEPSIS_COLUMNS = ("sep3_sofa2", "sepsis3_sofa2", "sepsis3", "sepsis")
_UNSUPPORTED_FILTERS = {
    "age_at_admission": "Row-level cohort cuts require a validated cohort-builder pass.",
    "minimum_icu_los": "LOS filters must recompute the cohort denominator before review.",
    "observation_window": "Observation windows are extraction-time logic, not Cohort Review metadata.",
    "sepsis3_positive_only": "Event-restricted cohorts need a dedicated cohort-builder pass.",
    "exclude_readmissions": "Readmission exclusion is not encoded in module-export metadata.",
    "patient_list": "Identifier-list filtering is intentionally not accepted by this aggregate endpoint.",
    "custom_threshold": "Custom group thresholds require row-level validation before display.",
    "matched_cohort": "Matched cohorts are Cross-DB/evidence-gate work, not Stage17 Cohort Review.",
}
_UNSUPPORTED_STATISTICS = {
    "p_value": "Inferential statistics are withheld until the numeric evidence audit gate is implemented.",
    "p-values": "Inferential statistics are withheld until the numeric evidence audit gate is implemented.",
    "smd": "Standardized mean differences need audited row-level group construction.",
    "confidence_interval": "Confidence intervals need an audited statistical backend.",
    "matched_analysis": "Matched analyses are outside Stage17 Cohort Review.",
}


def cohort_review_summary(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return real Cohort Review aggregates for one registered export."""
    _reject_unsupported_request(body)
    source, desc = _resolve_registered_source(body)
    path = Path(str(desc.get("path") or source.get("path") or "")).expanduser()
    frames = {
        module: _read_module_frame(path, desc, module)
        for module in _READ_MODULES
    }

    demo = frames.get("demographics")
    if demo is None or getattr(demo, "empty", True):
        fallback = _fallback_entity_frame(path, desc)
        if fallback is None or getattr(fallback, "empty", True):
            raise CohortReviewError({"error": "no_entity_denominator"})
        demo = fallback

    demo = demo.copy()
    demo["stay_id"] = demo["stay_id"].map(dataio._norm_id)
    demo = demo[demo["stay_id"].astype(bool)].drop_duplicates("stay_id")
    if demo.empty:
        raise CohortReviewError({"error": "no_entity_denominator"})

    entity_ids = [str(value) for value in demo["stay_id"].tolist()]
    entity_set = set(entity_ids)
    outcome = dataio._filter_by_stay(frames.get("outcome"), entity_set)
    sofa2 = dataio._filter_by_stay(frames.get("sofa2_score"), entity_set)
    sepsis = dataio._filter_by_stay(frames.get("sepsis3_sofa2"), entity_set)

    death_col = _first_column(outcome, _DEATH_COLUMNS)
    los_col = _first_column(outcome, _LOS_COLUMNS)
    sofa_col = _first_column(sofa2, _SOFA_COLUMNS)
    sepsis_col = _first_column(sepsis, _SEPSIS_COLUMNS)

    death_by_entity = dataio._stay_bool(outcome, death_col, missing_false=True) if death_col else {}
    los_by_entity = dataio._stay_numeric(outcome, los_col, "median") if los_col else {}
    sofa_by_entity = dataio._stay_numeric(sofa2, sofa_col, "max") if sofa_col else {}
    sepsis_by_entity = dataio._stay_bool(sepsis, sepsis_col, missing_false=True) if sepsis_col else {}
    if outcome is not None and not outcome.empty and death_col:
        for entity_id in entity_ids:
            death_by_entity.setdefault(entity_id, False)
    if sepsis is not None and not sepsis.empty and sepsis_col:
        for entity_id in entity_ids:
            sepsis_by_entity.setdefault(entity_id, False)

    age_col = _first_column(demo, _AGE_COLUMNS)
    sex_col = _first_column(demo, _SEX_COLUMNS)
    age_by_entity = _entity_numeric(demo, age_col) if age_col else {}
    sex_values = list(demo[sex_col]) if sex_col else []
    coverage = _coverage_payload(path, desc)
    quality = _quality_summary(coverage)
    mortality = _bool_summary(entity_ids, death_by_entity, true_label="deceased", false_label="survived")
    sepsis_summary = _bool_summary(entity_ids, sepsis_by_entity, true_label="positive", false_label="nonpositive")
    age_summary = _numeric_summary(age_by_entity.values())
    los_summary = _numeric_summary(los_by_entity.values())
    sofa_summary = _numeric_summary(sofa_by_entity.values())

    summary = {
        "cohort_size": len(entity_ids),
        "entities": len(entity_ids),
        "modules": int((desc.get("summary") or {}).get("modules") or len({f.get("module") for f in desc.get("files") or [] if f.get("module")})),
        "file_count": int((desc.get("summary") or {}).get("file_count") or len(desc.get("files") or [])),
        "total_records": int((desc.get("summary") or {}).get("total_rows") or 0),
        "mortality": mortality,
        "mortality_pct": mortality.get("pct"),
        "age": age_summary,
        "sex": _sex_summary(sex_values),
        "sofa2": {**sofa_summary, "bins": _sofa_bins(sofa_by_entity.values())},
        "los_icu_days": los_summary,
        "sepsis3": sepsis_summary,
        "sepsis_pct": sepsis_summary.get("pct"),
    }

    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source": _source_provenance(source, desc),
        "provenance": {
            "computed_from": [
                "source_registry",
                "export_manifest",
                "bounded_column_reads",
                "cohort_level_aggregates",
            ],
            "payload_scope": "cohort_aggregate_only",
            "inference": "blocked_until_numeric_evidence_gate",
        },
        "privacy": {
            "raw_rows_returned": False,
            "direct_identifiers_returned": False,
            "patient_rows_persisted": False,
            "secrets_returned": False,
        },
        "summary": summary,
        "groups": _group_payload(
            entity_ids=entity_ids,
            age_by_entity=age_by_entity,
            sex_by_entity=_entity_text(demo, sex_col) if sex_col else {},
            death_by_entity=death_by_entity,
            los_by_entity=los_by_entity,
            sepsis_by_entity=sepsis_by_entity,
        ),
        "coverage": coverage,
        "quality": quality,
        "table_one": {
            "status": "blocked",
            "reason": "P-values, SMDs, and row-level baseline tables require the numeric evidence audit gate.",
            "inferential_statistics_allowed": False,
        },
        "sofa_reclassification": {
            "status": "blocked",
            "reason": "Paired SOFA-1/SOFA-2 reclassification needs a bounded paired-score backend; Stage17 exposes SOFA-2 aggregate only.",
            "paired_backend_ready": False,
        },
        "blocked_features": [
            {
                "id": "row_level_filters",
                "status": "blocked",
                "reason": "Cohort Review accepts only registered-source aggregate review in Stage17.",
            },
            {
                "id": "inferential_statistics",
                "status": "blocked",
                "reason": "No p-values, SMDs, or confidence intervals before the numeric evidence audit gate.",
            },
            {
                "id": "matched_cohort",
                "status": "blocked",
                "reason": "Matched cohorts belong to Cross-DB parity and audit-gated analysis.",
            },
            {
                "id": "paired_sofa_reclassification",
                "status": "blocked",
                "reason": "Requires paired SOFA-1/SOFA-2 records and a dedicated bounded API.",
            },
        ],
    }


def _reject_unsupported_request(body: Dict[str, Any]) -> None:
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    requested_filters = [
        {"id": str(key), "reason": _UNSUPPORTED_FILTERS.get(str(key), "Cohort Review Stage17 does not accept row-level filters.")}
        for key, value in filters.items()
        if _truthy_request(value)
    ]
    if requested_filters:
        raise CohortReviewError({
            "error": "unsupported_filter",
            "unsupported": requested_filters,
            "supported_scope": "registered_source_cohort_aggregates_only",
        })

    stats = body.get("statistics") or body.get("stats") or []
    if isinstance(stats, str):
        stats = [stats]
    requested_stats = [
        {
            "id": str(item),
            "reason": _UNSUPPORTED_STATISTICS.get(str(item), "Requested statistic is not supported by the Stage17 aggregate endpoint."),
        }
        for item in stats
        if _truthy_request(item)
    ]
    if requested_stats:
        raise CohortReviewError({
            "error": "unsupported_statistic",
            "unsupported": requested_stats,
            "supported_scope": "descriptive_aggregate_only",
        })

    grouping = body.get("grouping") or body.get("comparison")
    if isinstance(grouping, dict) and grouping.get("mode") == "custom":
        raise CohortReviewError({
            "error": "unsupported_grouping",
            "unsupported": [{"id": "custom", "reason": _UNSUPPORTED_FILTERS["custom_threshold"]}],
            "supported_scope": "fixed_descriptive_group_splits_only",
        })


def _resolve_registered_source(body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    registry = source_store.load_registry()
    sources = [s for s in registry.get("sources") or [] if isinstance(s, dict)]
    requested = body.get("source_path") or body.get("path")
    if requested:
        norm = _norm_path(str(requested))
        source = next((s for s in sources if _norm_path(str(s.get("path") or "")) == norm), None)
        if source is None:
            raise CohortReviewError({"error": "source_not_registered", "path_hash": _hash(norm)})
    else:
        active = registry.get("active_path")
        if not active:
            raise CohortReviewError({"error": "no_active_export"})
        active_norm = _norm_path(str(active))
        source = next((s for s in sources if _norm_path(str(s.get("path") or "")) == active_norm), None)
        if source is None:
            raise CohortReviewError({"error": "active_source_not_registered", "path_hash": _hash(active_norm)})

    desc = dataio.describe_export_source(str(source.get("path") or ""))
    if not desc.get("ok"):
        raise CohortReviewError({"error": "invalid_export", "detail": desc.get("error")})
    return source, desc


def _read_module_frame(path: Path, desc: Dict[str, Any], module: str) -> Any:
    file_meta = next((f for f in desc.get("files") or [] if f.get("module") == module), None)
    if not file_meta:
        return None
    columns = [c for c in _MODULE_COLUMNS[module] if c in (file_meta.get("columns") or [])]
    if "stay_id" not in columns:
        return None
    return _read_selected_columns(path / str(file_meta.get("file") or ""), columns)


def _fallback_entity_frame(path: Path, desc: Dict[str, Any]) -> Any:
    file_meta = next((f for f in desc.get("files") or [] if "stay_id" in (f.get("columns") or [])), None)
    if not file_meta:
        return None
    return _read_selected_columns(path / str(file_meta.get("file") or ""), ["stay_id"])


def _read_selected_columns(path: Path, columns: List[str]) -> Any:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix == ".xlsx":
        return pd.read_excel(path, usecols=columns)
    return pd.read_csv(path, usecols=columns)


def _coverage_payload(path: Path, desc: Dict[str, Any]) -> List[Dict[str, Any]]:
    cohort_size = (desc.get("summary") or {}).get("stays")
    out: List[Dict[str, Any]] = []
    for item in desc.get("files") or []:
        module = str(item.get("module") or "")
        if not module:
            continue
        covered = _covered_entities(path, item, cohort_size)
        coverage = round(covered / cohort_size * 100, 1) if isinstance(cohort_size, int) and cohort_size else None
        status = _quality_status(module, coverage)
        out.append({
            "module": module,
            "rows": int(item.get("rows") or 0),
            "column_count": len(item.get("columns") or []),
            "covered_entities": covered,
            "coverage_pct": coverage,
            "quality_status": status,
        })
    return out


def _covered_entities(path: Path, item: Dict[str, Any], cohort_size: Any) -> int | None:
    if not isinstance(cohort_size, int) or cohort_size <= 0:
        return None
    file_name = str(item.get("file") or "")
    if not file_name:
        return None
    ids = dataio._read_stay_ids(path / file_name)
    if ids is None:
        return None
    return min(len(ids), cohort_size)


def _quality_summary(coverage: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"ok": 0, "warn": 0, "bad": 0, "neutral": 0, "unknown": 0}
    values: List[float] = []
    for row in coverage:
        status = str(row.get("quality_status") or "unknown")
        if status not in counts:
            status = "unknown"
        counts[status] += 1
        pct = row.get("coverage_pct")
        if isinstance(pct, (int, float)):
            values.append(float(pct))
    return {
        "modules_ok": counts["ok"],
        "modules_warn": counts["warn"],
        "modules_bad": counts["bad"],
        "modules_neutral": counts["neutral"],
        "modules_unknown": counts["unknown"],
        "watchlist_count": counts["warn"] + counts["bad"],
        "median_coverage_pct": dataio._median(values),
    }


def _group_payload(
    *,
    entity_ids: List[str],
    age_by_entity: Dict[str, float],
    sex_by_entity: Dict[str, str],
    death_by_entity: Dict[str, bool],
    los_by_entity: Dict[str, float],
    sepsis_by_entity: Dict[str, bool],
) -> Dict[str, Any]:
    supported = [
        _group_from_bool("survival", "Survived vs Deceased", entity_ids, death_by_entity, false_name="Survived", true_name="Deceased"),
        _group_from_age(entity_ids, age_by_entity),
        _group_from_sex(entity_ids, sex_by_entity),
        _group_from_los(entity_ids, los_by_entity),
        _group_from_bool("sepsis", "Sepsis vs Non-sepsis", entity_ids, sepsis_by_entity, false_name="Non-sepsis", true_name="Sepsis"),
    ]
    supported = [row for row in supported if row is not None]
    return {
        "comparison_mode": "descriptive_only",
        "inferential_statistics_allowed": False,
        "supported": supported,
        "blocked": [
            {
                "id": "custom_threshold",
                "status": "blocked",
                "reason": "Custom thresholds require audited row-level cohort construction.",
            },
            {
                "id": "p_value_smd",
                "status": "blocked",
                "reason": "Inferential statistics are withheld until the numeric evidence audit gate.",
            },
            {
                "id": "matched_cohort",
                "status": "blocked",
                "reason": "Matched cohort logic is not part of Stage17 Cohort Review.",
            },
        ],
    }


def _group_from_bool(
    group_id: str,
    label: str,
    entity_ids: List[str],
    mapping: Dict[str, bool],
    *,
    false_name: str,
    true_name: str,
) -> Dict[str, Any] | None:
    if not mapping:
        return None
    true_count = sum(1 for entity_id in entity_ids if mapping.get(entity_id) is True)
    false_count = sum(1 for entity_id in entity_ids if mapping.get(entity_id) is False)
    unknown = len(entity_ids) - true_count - false_count
    return {
        "id": group_id,
        "label": label,
        "status": "supported",
        "basis": "registered_export_aggregate",
        "groups": [
            {"label": false_name, "count": false_count, "pct": _pct(false_count, len(entity_ids))},
            {"label": true_name, "count": true_count, "pct": _pct(true_count, len(entity_ids))},
            {"label": "Unknown", "count": unknown, "pct": _pct(unknown, len(entity_ids))},
        ],
        "inferential_statistics_allowed": False,
    }


def _group_from_age(entity_ids: List[str], age_by_entity: Dict[str, float]) -> Dict[str, Any] | None:
    if not age_by_entity:
        return None
    younger = sum(1 for entity_id in entity_ids if entity_id in age_by_entity and age_by_entity[entity_id] < 65)
    older = sum(1 for entity_id in entity_ids if entity_id in age_by_entity and age_by_entity[entity_id] >= 65)
    unknown = len(entity_ids) - younger - older
    return {
        "id": "age",
        "label": "Age Groups",
        "status": "supported",
        "basis": "age_threshold_65_descriptive",
        "groups": [
            {"label": "<65", "count": younger, "pct": _pct(younger, len(entity_ids))},
            {"label": ">=65", "count": older, "pct": _pct(older, len(entity_ids))},
            {"label": "Unknown", "count": unknown, "pct": _pct(unknown, len(entity_ids))},
        ],
        "inferential_statistics_allowed": False,
    }


def _group_from_sex(entity_ids: List[str], sex_by_entity: Dict[str, str]) -> Dict[str, Any] | None:
    if not sex_by_entity:
        return None
    female = sum(1 for entity_id in entity_ids if _sex_bucket(sex_by_entity.get(entity_id)) == "female")
    male = sum(1 for entity_id in entity_ids if _sex_bucket(sex_by_entity.get(entity_id)) == "male")
    unknown = len(entity_ids) - female - male
    return {
        "id": "sex",
        "label": "Female vs Male",
        "status": "supported",
        "basis": "sex_metadata_descriptive",
        "groups": [
            {"label": "Female", "count": female, "pct": _pct(female, len(entity_ids))},
            {"label": "Male", "count": male, "pct": _pct(male, len(entity_ids))},
            {"label": "Unknown", "count": unknown, "pct": _pct(unknown, len(entity_ids))},
        ],
        "inferential_statistics_allowed": False,
    }


def _group_from_los(entity_ids: List[str], los_by_entity: Dict[str, float]) -> Dict[str, Any] | None:
    values = [v for v in los_by_entity.values() if v is not None]
    if not values:
        return None
    threshold = dataio._median(values)
    if threshold is None:
        return None
    short = sum(1 for entity_id in entity_ids if entity_id in los_by_entity and los_by_entity[entity_id] <= threshold)
    long = sum(1 for entity_id in entity_ids if entity_id in los_by_entity and los_by_entity[entity_id] > threshold)
    unknown = len(entity_ids) - short - long
    return {
        "id": "los",
        "label": "Short vs Long Stay",
        "status": "supported",
        "basis": "median_los_descriptive_split",
        "threshold": threshold,
        "groups": [
            {"label": "Short/median", "count": short, "pct": _pct(short, len(entity_ids))},
            {"label": "Long", "count": long, "pct": _pct(long, len(entity_ids))},
            {"label": "Unknown", "count": unknown, "pct": _pct(unknown, len(entity_ids))},
        ],
        "inferential_statistics_allowed": False,
    }


def _entity_numeric(frame: Any, column: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if frame is None or frame.empty or "stay_id" not in frame.columns or column not in frame.columns:
        return out
    for _, row in frame.iterrows():
        entity_id = dataio._norm_id(row.get("stay_id"))
        value = dataio._num(row.get(column))
        if entity_id and value is not None:
            out[entity_id] = value
    return out


def _entity_text(frame: Any, column: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if frame is None or frame.empty or "stay_id" not in frame.columns or column not in frame.columns:
        return out
    for _, row in frame.iterrows():
        entity_id = dataio._norm_id(row.get("stay_id"))
        value = dataio._clean(row.get(column))
        if entity_id and value:
            out[entity_id] = value
    return out


def _numeric_summary(values: Iterable[Any]) -> Dict[str, Any]:
    vals = sorted(v for v in (dataio._num(v) for v in values) if v is not None)
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": round(sum(vals) / len(vals), 2),
        "median": dataio._median(vals),
        "min": vals[0],
        "max": vals[-1],
    }


def _bool_summary(
    entity_ids: List[str],
    mapping: Dict[str, bool],
    *,
    true_label: str,
    false_label: str,
) -> Dict[str, Any]:
    true_count = sum(1 for entity_id in entity_ids if mapping.get(entity_id) is True)
    false_count = sum(1 for entity_id in entity_ids if mapping.get(entity_id) is False)
    unknown = len(entity_ids) - true_count - false_count
    return {
        "count": true_count,
        "pct": _pct(true_count, len(entity_ids)),
        f"{true_label}_count": true_count,
        f"{false_label}_count": false_count,
        "unknown_count": unknown,
    }


def _sex_summary(values: List[Any]) -> Dict[str, Any]:
    buckets = [_sex_bucket(value) for value in values]
    buckets = [b for b in buckets if b]
    total = len(buckets)
    female = sum(1 for b in buckets if b == "female")
    male = sum(1 for b in buckets if b == "male")
    unknown = len(values) - female - male
    return {
        "count": len(values),
        "female_count": female,
        "female_pct": _pct(female, len(values)),
        "male_count": male,
        "male_pct": _pct(male, len(values)),
        "unknown_count": unknown,
    }


def _sofa_bins(values: Iterable[Any]) -> List[Dict[str, Any]]:
    vals = [v for v in (dataio._num(v) for v in values) if v is not None]
    bins = [
        ("0-5", lambda v: v <= 5),
        ("6-8", lambda v: 6 <= v <= 8),
        ("9-11", lambda v: 9 <= v <= 11),
        (">=12", lambda v: v >= 12),
    ]
    total = len(vals)
    return [
        {"label": label, "count": sum(1 for value in vals if predicate(value)), "pct": _pct(sum(1 for value in vals if predicate(value)), total)}
        for label, predicate in bins
    ]


def _first_column(frame: Any, candidates: Tuple[str, ...]) -> str | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    cols = set(getattr(frame, "columns", []))
    return next((column for column in candidates if column in cols), None)


def _quality_status(module: str, coverage_pct: float | None) -> str:
    if coverage_pct is None:
        return "unknown"
    if module in dataio._EVENT_PRESENCE_MODULES:
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


def _sex_bucket(value: Any) -> str | None:
    clean = dataio._clean(value)
    if not clean:
        return None
    text = clean.strip().lower()
    if text in {"f", "female", "woman"} or "female" in text:
        return "female"
    if text in {"m", "male", "man"} or text == "male":
        return "male"
    return None


def _pct(count: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(count / denominator * 100, 1)


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


class CohortReviewError(Exception):
    def __init__(self, detail: Dict[str, Any]):
        super().__init__(str(detail.get("error") or "cohort_review_error"))
        self.detail = detail
