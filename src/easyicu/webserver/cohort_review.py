"""Bounded Cohort Review aggregates for the native FastAPI UI.

This endpoint is the Stage17+ Cohort Review parity path. It consumes the active
registered EasyICU export and returns cohort-level aggregates only. Row-level
filters, inferential statistics, and matched cohorts stay fail-closed until
their backend contracts exist.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store

_READ_MODULES = ("demographics", "outcome", "sofa1_score", "sofa2_score", "sepsis3_sofa2")
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
        "icu_mortality",
        "icu_death",
        "death_icu",
        "mort_28d",
        "mortality_28d",
        "death_28d",
        "los_icu",
        "icu_los",
        "los_hosp",
        "hospital_los",
        "los_hospital",
        "hosp_los",
        "los_days",
        "days_to_death_28d",
        "time_to_death_28d",
        "followup_days_28d",
        "survival_time_28d",
    ),
    "sofa1_score": (
        "stay_id",
        "charttime",
        "time",
        "timestamp",
        "sofa1",
        "sofa1_total",
        "sofa",
        "sofa_total",
        "sofa_score",
        "score",
    ),
    "sofa2_score": (
        "stay_id",
        "charttime",
        "time",
        "timestamp",
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
_HOSP_DEATH_COLUMNS = ("hospital_mortality", "in_hospital_mortality", "death", "mortality")
_HOSP_LOS_COLUMNS = ("los_hosp", "hospital_los", "los_hospital", "hosp_los", "los_days")
_ICU_DEATH_COLUMNS = ("icu_death", "death_icu", "icu_mortality")
_ICU_LOS_COLUMNS = ("los_icu", "icu_los")
_MORT28_COLUMNS = ("mort_28d", "mortality_28d", "death_28d")
_MORT28_TIME_COLUMNS = ("days_to_death_28d", "time_to_death_28d", "followup_days_28d", "survival_time_28d")
_SOFA1_COLUMNS = ("sofa1", "sofa1_total", "sofa", "sofa_total", "sofa_score", "score")
_SOFA2_COLUMNS = ("sofa2", "sofa2_total", "sofa_score", "score")
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
    "p_value": "Generic Table One or group-comparison p-values are withheld; survival log-rank is available only in the audited KM module when timed outcomes exist.",
    "p-values": "Generic Table One or group-comparison p-values are withheld; survival log-rank is available only in the audited KM module when timed outcomes exist.",
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
    sofa1 = dataio._filter_by_stay(frames.get("sofa1_score"), entity_set)
    sofa2 = dataio._filter_by_stay(frames.get("sofa2_score"), entity_set)
    sepsis = dataio._filter_by_stay(frames.get("sepsis3_sofa2"), entity_set)

    death_col = _first_column(outcome, _DEATH_COLUMNS)
    los_col = _first_column(outcome, _LOS_COLUMNS)
    sofa1_col = _first_column(sofa1, _SOFA1_COLUMNS)
    sofa_col = _first_column(sofa2, _SOFA2_COLUMNS)
    sepsis_col = _first_column(sepsis, _SEPSIS_COLUMNS)

    death_by_entity = dataio._stay_bool(outcome, death_col, missing_false=True) if death_col else {}
    los_by_entity = dataio._stay_numeric(outcome, los_col, "median") if los_col else {}
    sofa1_by_entity = dataio._stay_numeric(sofa1, sofa1_col, "max") if sofa1_col else {}
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
    sofa_reclassification = _sofa_reclassification_payload(
        entity_ids=entity_ids,
        sofa1_by_entity=sofa1_by_entity,
        sofa2_by_entity=sofa_by_entity,
        has_sofa1_module=sofa1 is not None and not getattr(sofa1, "empty", True) and bool(sofa1_col),
        has_sofa2_module=sofa2 is not None and not getattr(sofa2, "empty", True) and bool(sofa_col),
    )
    blocked_features = [
        {
            "id": "row_level_filters",
            "status": "blocked",
            "reason": "Cohort Review accepts only registered-source aggregate review in Stage17.",
        },
        {
            "id": "inferential_statistics",
            "status": "blocked",
            "reason": "Generic Table One/group p-values, SMDs, and confidence intervals remain blocked; survival log-rank is scoped to the KM module when timed outcomes exist.",
        },
        {
            "id": "matched_cohort",
            "status": "blocked",
            "reason": "Matched cohorts belong to Cross-DB parity and audit-gated analysis.",
        },
    ]
    if sofa_reclassification.get("status") != "ready":
        blocked_features.append({
            "id": "paired_sofa_reclassification",
            "status": "blocked",
            "reason": str(sofa_reclassification.get("reason") or "Paired SOFA-1/SOFA-2 reclassification is not available for this export."),
        })
    survival_analysis = _survival_analysis_payload(
        outcome=outcome,
        entity_ids=entity_ids,
        age_by_entity=age_by_entity,
        sex_by_entity=_entity_text(demo, sex_col) if sex_col else {},
        sofa_by_entity=sofa_by_entity,
        sepsis_by_entity=sepsis_by_entity,
    )

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
                "survival_aggregate_if_time_to_event_available",
            ],
            "payload_scope": "cohort_aggregate_only",
            "inference": "descriptive_plus_exploratory_logrank_when_time_to_event_available",
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
            sofa_by_entity=sofa_by_entity,
            sepsis_by_entity=sepsis_by_entity,
        ),
        "coverage": coverage,
        "quality": quality,
        "table_one": {
            "status": "blocked",
            "reason": "Table One p-values, SMDs, and row-level baseline tables require the numeric evidence audit gate. Survival log-rank is scoped to the audited KM module when time-to-event data exist.",
            "inferential_statistics_allowed": False,
        },
        "survival_analysis": survival_analysis,
        "sofa_reclassification": sofa_reclassification,
        "blocked_features": blocked_features,
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
    sofa_by_entity: Dict[str, float],
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
    for row in supported:
        row["profile"] = _group_profile(
            row.get("id"),
            entity_ids=entity_ids,
            age_by_entity=age_by_entity,
            sex_by_entity=sex_by_entity,
            death_by_entity=death_by_entity,
            los_by_entity=los_by_entity,
            sofa_by_entity=sofa_by_entity,
            sepsis_by_entity=sepsis_by_entity,
        )
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


def _group_profile(
    group_id: Any,
    *,
    entity_ids: List[str],
    age_by_entity: Dict[str, float],
    sex_by_entity: Dict[str, str],
    death_by_entity: Dict[str, bool],
    los_by_entity: Dict[str, float],
    sofa_by_entity: Dict[str, float],
    sepsis_by_entity: Dict[str, bool],
) -> Dict[str, Any]:
    groups = _group_members(
        str(group_id or ""),
        entity_ids=entity_ids,
        age_by_entity=age_by_entity,
        sex_by_entity=sex_by_entity,
        death_by_entity=death_by_entity,
        los_by_entity=los_by_entity,
        sepsis_by_entity=sepsis_by_entity,
    )
    columns = [label for label, _ in groups]
    member_sets = [members for _, members in groups]
    return {
        "status": "descriptive_aggregate_only",
        "columns": columns,
        "inferential_statistics_allowed": False,
        "rows": [
            {"metric": "N", "kind": "count", "values": [len(members) for members in member_sets]},
            {"metric": "Mortality %", "kind": "percent", "values": [_bool_pct(members, death_by_entity) for members in member_sets]},
            {"metric": "Female %", "kind": "percent", "values": [_female_pct(members, sex_by_entity) for members in member_sets]},
            {"metric": "Median age", "kind": "numeric", "unit": "years", "values": [_median_for(members, age_by_entity) for members in member_sets]},
            {"metric": "Median SOFA-2", "kind": "numeric", "values": [_median_for(members, sofa_by_entity) for members in member_sets]},
            {"metric": "Median ICU LOS", "kind": "numeric", "unit": "days", "values": [_median_for(members, los_by_entity) for members in member_sets]},
            {"metric": "Sepsis-3 %", "kind": "percent", "values": [_bool_pct(members, sepsis_by_entity) for members in member_sets]},
        ],
    }


def _group_members(
    group_id: str,
    *,
    entity_ids: List[str],
    age_by_entity: Dict[str, float],
    sex_by_entity: Dict[str, str],
    death_by_entity: Dict[str, bool],
    los_by_entity: Dict[str, float],
    sepsis_by_entity: Dict[str, bool],
) -> List[Tuple[str, List[str]]]:
    if group_id == "survival":
        return _bool_member_groups(entity_ids, death_by_entity, false_name="Survived", true_name="Deceased")
    if group_id == "age":
        return [
            ("<65", [entity_id for entity_id in entity_ids if entity_id in age_by_entity and age_by_entity[entity_id] < 65]),
            (">=65", [entity_id for entity_id in entity_ids if entity_id in age_by_entity and age_by_entity[entity_id] >= 65]),
            ("Unknown", [entity_id for entity_id in entity_ids if entity_id not in age_by_entity]),
        ]
    if group_id == "sex":
        return [
            ("Female", [entity_id for entity_id in entity_ids if _sex_bucket(sex_by_entity.get(entity_id)) == "female"]),
            ("Male", [entity_id for entity_id in entity_ids if _sex_bucket(sex_by_entity.get(entity_id)) == "male"]),
            ("Unknown", [entity_id for entity_id in entity_ids if _sex_bucket(sex_by_entity.get(entity_id)) not in {"female", "male"}]),
        ]
    if group_id == "los":
        values = [value for value in los_by_entity.values() if value is not None]
        threshold = dataio._median(values)
        if threshold is None:
            return [("Known", [entity_id for entity_id in entity_ids if entity_id in los_by_entity]), ("Unknown", [entity_id for entity_id in entity_ids if entity_id not in los_by_entity])]
        return [
            ("Short/median", [entity_id for entity_id in entity_ids if entity_id in los_by_entity and los_by_entity[entity_id] <= threshold]),
            ("Long", [entity_id for entity_id in entity_ids if entity_id in los_by_entity and los_by_entity[entity_id] > threshold]),
            ("Unknown", [entity_id for entity_id in entity_ids if entity_id not in los_by_entity]),
        ]
    if group_id == "sepsis":
        return _bool_member_groups(entity_ids, sepsis_by_entity, false_name="Non-sepsis", true_name="Sepsis")
    return [("Cohort", list(entity_ids))]


def _bool_member_groups(
    entity_ids: List[str],
    mapping: Dict[str, bool],
    *,
    false_name: str,
    true_name: str,
) -> List[Tuple[str, List[str]]]:
    return [
        (false_name, [entity_id for entity_id in entity_ids if mapping.get(entity_id) is False]),
        (true_name, [entity_id for entity_id in entity_ids if mapping.get(entity_id) is True]),
        ("Unknown", [entity_id for entity_id in entity_ids if mapping.get(entity_id) not in {False, True}]),
    ]


def _median_for(entity_ids: List[str], mapping: Dict[str, float]) -> float | None:
    values = [mapping[entity_id] for entity_id in entity_ids if entity_id in mapping and mapping[entity_id] is not None]
    return dataio._median(values)


def _bool_pct(entity_ids: List[str], mapping: Dict[str, bool]) -> float | None:
    if not entity_ids:
        return None
    count = sum(1 for entity_id in entity_ids if mapping.get(entity_id) is True)
    return _pct(count, len(entity_ids))


def _female_pct(entity_ids: List[str], mapping: Dict[str, str]) -> float | None:
    if not entity_ids:
        return None
    count = sum(1 for entity_id in entity_ids if _sex_bucket(mapping.get(entity_id)) == "female")
    return _pct(count, len(entity_ids))


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


def _survival_analysis_payload(
    *,
    outcome: Any,
    entity_ids: List[str],
    age_by_entity: Dict[str, float],
    sex_by_entity: Dict[str, str],
    sofa_by_entity: Dict[str, float],
    sepsis_by_entity: Dict[str, bool],
) -> Dict[str, Any]:
    specs = [
        {
            "id": "hospital_death",
            "label": "Hospital mortality",
            "event_candidates": _HOSP_DEATH_COLUMNS,
            "time_candidates": _HOSP_LOS_COLUMNS,
            "time_label": "Hospital LOS / follow-up days",
        },
        {
            "id": "icu_death",
            "label": "ICU mortality",
            "event_candidates": _ICU_DEATH_COLUMNS,
            "time_candidates": _ICU_LOS_COLUMNS,
            "time_label": "ICU LOS / follow-up days",
        },
        {
            "id": "mort_28d",
            "label": "28-day mortality",
            "event_candidates": _MORT28_COLUMNS,
            "time_candidates": _MORT28_TIME_COLUMNS,
            "time_label": "Days to 28-day death/censoring",
        },
    ]
    outcomes = [
        _survival_outcome_option(outcome, spec, entity_ids)
        for spec in specs
    ]
    group_options = _survival_group_options(
        entity_ids=entity_ids,
        age_by_entity=age_by_entity,
        sex_by_entity=sex_by_entity,
        sofa_by_entity=sofa_by_entity,
        sepsis_by_entity=sepsis_by_entity,
    )
    curves: List[Dict[str, Any]] = []
    for option in outcomes:
        if option.get("status") != "ready":
            continue
        event_col = str(option.get("event_column") or "")
        time_col = str(option.get("time_column") or "")
        event_by_entity = dataio._stay_bool(outcome, event_col, missing_false=True) if event_col else {}
        for entity_id in entity_ids:
            event_by_entity.setdefault(entity_id, False)
        time_by_entity = dataio._stay_numeric(outcome, time_col, "max") if time_col else {}
        for group in group_options:
            if group.get("status") != "ready":
                continue
            curve = _survival_curve_payload(
                outcome_option=option,
                group_option=group,
                event_by_entity=event_by_entity,
                time_by_entity=time_by_entity,
            )
            if curve:
                curves.append(curve)

    ready_outcomes = [row for row in outcomes if row.get("status") == "ready"]
    ready_groups = [row for row in group_options if row.get("status") == "ready"]
    status = "ready" if ready_outcomes and ready_groups and curves else "blocked"
    reason = None
    if status != "ready":
        if not ready_outcomes:
            reason = "No outcome has both an event column and a time-to-event/censoring column in this export."
        elif not ready_groups:
            reason = "No supported two-group split is available for this cohort."
        else:
            reason = "No survival curve could be computed from the available timed records."

    default_outcome = next((row["id"] for row in ready_outcomes if row["id"] == "hospital_death"), None)
    default_outcome = default_outcome or (ready_outcomes[0]["id"] if ready_outcomes else None)
    default_group = next((row["id"] for row in ready_groups if row["id"] == "sepsis"), None)
    default_group = default_group or (ready_groups[0]["id"] if ready_groups else None)
    return {
        "status": status,
        "reason": reason,
        "mode": "kaplan_meier_aggregate",
        "scope": "exploratory_unadjusted",
        "reportable": False,
        "time_unit": "days",
        "default_outcome": default_outcome,
        "default_group": default_group,
        "outcomes": outcomes,
        "group_options": [
            {key: value for key, value in row.items() if key != "_members"}
            for row in group_options
        ],
        "curves": curves,
        "notes": [
            "Kaplan-Meier/log-rank requires both an event indicator and a time-to-event or censoring time.",
            "A binary mortality flag alone is shown as blocked rather than converted into a synthetic time axis.",
            "Log-rank is unadjusted and exploratory; manuscript use still needs the evidence-bound agent gate.",
        ],
    }


def _survival_outcome_option(outcome: Any, spec: Dict[str, Any], entity_ids: List[str]) -> Dict[str, Any]:
    base = {
        "id": spec["id"],
        "label": spec["label"],
        "status": "blocked",
        "event_column": None,
        "time_column": None,
        "time_label": spec["time_label"],
        "usable_entities": 0,
        "event_count": 0,
    }
    if outcome is None or getattr(outcome, "empty", True):
        return {**base, "reason": "Outcome module is not present in the registered export."}

    event_col = _first_column(outcome, spec["event_candidates"])
    if not event_col:
        return {
            **base,
            "reason": f"No event column found for {spec['label']}.",
            "expected_event_columns": list(spec["event_candidates"]),
        }
    time_col = _first_column(outcome, spec["time_candidates"])
    if not time_col:
        return {
            **base,
            "event_column": event_col,
            "reason": f"{spec['label']} is available only as an event flag; KM/log-rank needs time-to-event or censoring time.",
            "expected_time_columns": list(spec["time_candidates"]),
        }

    event_by_entity = dataio._stay_bool(outcome, event_col, missing_false=True)
    for entity_id in entity_ids:
        event_by_entity.setdefault(entity_id, False)
    time_by_entity = dataio._stay_numeric(outcome, time_col, "max")
    usable = [
        entity_id
        for entity_id in entity_ids
        if dataio._num(time_by_entity.get(entity_id)) is not None and float(time_by_entity[entity_id]) >= 0
    ]
    event_count = sum(1 for entity_id in usable if event_by_entity.get(entity_id) is True)
    if len(usable) < 2:
        return {
            **base,
            "event_column": event_col,
            "time_column": time_col,
            "usable_entities": len(usable),
            "event_count": event_count,
            "reason": "Fewer than two cohort entities have valid survival time values.",
        }
    return {
        **base,
        "status": "ready",
        "reason": None,
        "event_column": event_col,
        "time_column": time_col,
        "usable_entities": len(usable),
        "event_count": event_count,
    }


def _survival_group_options(
    *,
    entity_ids: List[str],
    age_by_entity: Dict[str, float],
    sex_by_entity: Dict[str, str],
    sofa_by_entity: Dict[str, float],
    sepsis_by_entity: Dict[str, bool],
) -> List[Dict[str, Any]]:
    raw_groups = [
        {
            "id": "sepsis",
            "label": "Sepsis vs Non-sepsis",
            "basis": "sepsis3_sofa2_event_module",
            "_members": _bool_member_groups(entity_ids, sepsis_by_entity, false_name="Non-sepsis", true_name="Sepsis"),
        },
        {
            "id": "age",
            "label": "Age <65 vs >=65",
            "basis": "age_threshold_65_descriptive",
            "_members": [
                ("<65", [entity_id for entity_id in entity_ids if entity_id in age_by_entity and age_by_entity[entity_id] < 65]),
                (">=65", [entity_id for entity_id in entity_ids if entity_id in age_by_entity and age_by_entity[entity_id] >= 65]),
            ],
        },
        {
            "id": "sex",
            "label": "Female vs Male",
            "basis": "sex_metadata_descriptive",
            "_members": [
                ("Female", [entity_id for entity_id in entity_ids if _sex_bucket(sex_by_entity.get(entity_id)) == "female"]),
                ("Male", [entity_id for entity_id in entity_ids if _sex_bucket(sex_by_entity.get(entity_id)) == "male"]),
            ],
        },
        _survival_sofa_group(entity_ids, sofa_by_entity),
    ]
    out: List[Dict[str, Any]] = []
    for row in raw_groups:
        if not row:
            continue
        members = [(label, ids) for label, ids in row.get("_members", []) if label != "Unknown"]
        nonempty = [(label, ids) for label, ids in members if ids]
        status = "ready" if len(nonempty) >= 2 else "blocked"
        out.append({
            **row,
            "status": status,
            "reason": None if status == "ready" else "This split does not produce two non-empty groups in the current cohort.",
            "groups": [
                {"label": label, "count": len(ids)}
                for label, ids in members
            ],
            "_members": members,
        })
    return out


def _survival_sofa_group(entity_ids: List[str], sofa_by_entity: Dict[str, float]) -> Dict[str, Any] | None:
    values = [value for value in sofa_by_entity.values() if value is not None]
    threshold = dataio._median(values)
    if threshold is None:
        return None
    return {
        "id": "sofa2",
        "label": f"SOFA-2 <= {threshold:g} vs > {threshold:g}",
        "basis": "median_sofa2_descriptive_split",
        "threshold": threshold,
        "_members": [
            (f"SOFA-2 <= {threshold:g}", [entity_id for entity_id in entity_ids if entity_id in sofa_by_entity and sofa_by_entity[entity_id] <= threshold]),
            (f"SOFA-2 > {threshold:g}", [entity_id for entity_id in entity_ids if entity_id in sofa_by_entity and sofa_by_entity[entity_id] > threshold]),
        ],
    }


def _survival_curve_payload(
    *,
    outcome_option: Dict[str, Any],
    group_option: Dict[str, Any],
    event_by_entity: Dict[str, bool],
    time_by_entity: Dict[str, float],
) -> Dict[str, Any] | None:
    group_records = []
    for label, members in group_option.get("_members", []):
        records = []
        for entity_id in members:
            time_value = dataio._num(time_by_entity.get(entity_id))
            if time_value is None or time_value < 0:
                continue
            records.append((float(time_value), bool(event_by_entity.get(entity_id) is True)))
        if records:
            km = _km_group_payload(label, records)
            group_records.append({"label": label, "records": records, "payload": km})
    if len(group_records) < 2:
        return None

    risk_times = _risk_times([record for row in group_records for record in row["records"]])
    logrank = _logrank_payload(group_records[0]["records"], group_records[1]["records"], group_records[0]["label"], group_records[1]["label"])
    return {
        "outcome_id": outcome_option.get("id"),
        "group_id": group_option.get("id"),
        "status": "ready",
        "label": f"{outcome_option.get('label')} by {group_option.get('label')}",
        "event_column": outcome_option.get("event_column"),
        "time_column": outcome_option.get("time_column"),
        "time_label": outcome_option.get("time_label"),
        "time_unit": "days",
        "scope": "exploratory_unadjusted",
        "reportable": False,
        "groups": [row["payload"] for row in group_records],
        "logrank": logrank,
        "number_at_risk": {
            "times": risk_times,
            "rows": [
                {
                    "label": row["label"],
                    "values": [sum(1 for time_value, _event in row["records"] if time_value >= t) for t in risk_times],
                }
                for row in group_records
            ],
        },
    }


def _km_group_payload(label: str, records: List[Tuple[float, bool]]) -> Dict[str, Any]:
    events = sum(1 for _time_value, event in records if event)
    points = _km_points(records)
    return {
        "label": label,
        "n": len(records),
        "events": events,
        "censored": len(records) - events,
        "median_survival": _km_median(points),
        "points": _thin_points(points),
    }


def _km_points(records: List[Tuple[float, bool]]) -> List[Dict[str, Any]]:
    if not records:
        return [{"time": 0, "survival": 100.0, "at_risk": 0, "events": 0}]
    event_times = sorted({time_value for time_value, event in records if event})
    max_time = max(time_value for time_value, _event in records)
    survival = 1.0
    points = [{"time": 0, "survival": 100.0, "at_risk": len(records), "events": 0}]
    for time_value in event_times:
        at_risk = sum(1 for obs_time, _event in records if obs_time >= time_value)
        events = sum(1 for obs_time, event in records if event and obs_time == time_value)
        if at_risk > 0:
            survival *= max(0.0, 1.0 - events / at_risk)
        points.append({
            "time": _round_time(time_value),
            "survival": round(survival * 100, 1),
            "at_risk": at_risk,
            "events": events,
        })
    if max_time and (not points or points[-1]["time"] != _round_time(max_time)):
        points.append({
            "time": _round_time(max_time),
            "survival": points[-1]["survival"],
            "at_risk": sum(1 for obs_time, _event in records if obs_time >= max_time),
            "events": 0,
        })
    return points


def _km_median(points: List[Dict[str, Any]]) -> float | None:
    for point in points:
        survival = dataio._num(point.get("survival"))
        if survival is not None and survival <= 50:
            return dataio._num(point.get("time"))
    return None


def _thin_points(points: List[Dict[str, Any]], max_points: int = 80) -> List[Dict[str, Any]]:
    if len(points) <= max_points:
        return points
    keep_indexes = {0, len(points) - 1}
    step = (len(points) - 1) / (max_points - 1)
    for i in range(max_points):
        keep_indexes.add(round(i * step))
    return [points[i] for i in sorted(keep_indexes)]


def _risk_times(records: List[Tuple[float, bool]]) -> List[float]:
    if not records:
        return [0]
    max_time = max(time_value for time_value, _event in records)
    base = [0, 1, 3, 7, 14, 28]
    times = [float(t) for t in base if t <= max_time]
    rounded_max = _round_time(max_time)
    if max_time > 0 and rounded_max not in times:
        times.append(rounded_max)
    if not times:
        times = [0, _round_time(max_time)]
    return times


def _logrank_payload(
    records_a: List[Tuple[float, bool]],
    records_b: List[Tuple[float, bool]],
    label_a: str,
    label_b: str,
) -> Dict[str, Any]:
    event_times = sorted({time_value for time_value, event in records_a + records_b if event})
    observed_a = expected_a = variance_a = 0.0
    total_events = 0
    for time_value in event_times:
        n_a = sum(1 for obs_time, _event in records_a if obs_time >= time_value)
        n_b = sum(1 for obs_time, _event in records_b if obs_time >= time_value)
        d_a = sum(1 for obs_time, event in records_a if event and obs_time == time_value)
        d_b = sum(1 for obs_time, event in records_b if event and obs_time == time_value)
        n_total = n_a + n_b
        d_total = d_a + d_b
        if n_total <= 1 or d_total <= 0:
            continue
        observed_a += d_a
        expected_a += d_total * (n_a / n_total)
        variance_a += (n_a * n_b * d_total * (n_total - d_total)) / ((n_total ** 2) * (n_total - 1))
        total_events += d_total
    if variance_a <= 0 or total_events <= 0:
        return {
            "status": "blocked",
            "reason": "Log-rank requires observed events in at least one timed risk set.",
            "groups": [label_a, label_b],
            "df": 1,
        }
    chi_square = ((observed_a - expected_a) ** 2) / variance_a
    p_value = math.erfc(math.sqrt(max(0.0, chi_square) / 2))
    return {
        "status": "ready",
        "test": "logrank",
        "groups": [label_a, label_b],
        "df": 1,
        "observed_events_first_group": round(observed_a, 3),
        "expected_events_first_group": round(expected_a, 3),
        "chi_square": round(chi_square, 4),
        "p_value": round(p_value, 6),
        "interpretation": "exploratory_unadjusted_not_reportable",
    }


def _round_time(value: float) -> float:
    rounded = round(float(value), 2)
    return int(rounded) if float(rounded).is_integer() else rounded


def _sofa_reclassification_payload(
    *,
    entity_ids: List[str],
    sofa1_by_entity: Dict[str, float],
    sofa2_by_entity: Dict[str, float],
    has_sofa1_module: bool,
    has_sofa2_module: bool,
) -> Dict[str, Any]:
    if not has_sofa1_module or not has_sofa2_module:
        missing = []
        if not has_sofa1_module:
            missing.append("sofa1_score")
        if not has_sofa2_module:
            missing.append("sofa2_score")
        return {
            "status": "blocked",
            "reason": "Paired SOFA-1/SOFA-2 reclassification requires both score modules in the registered export.",
            "paired_backend_ready": False,
            "missing_modules": missing,
            "supported_scope": "aggregate_only_when_both_modules_exist",
        }

    pairs = [
        (sofa1_by_entity[entity_id], sofa2_by_entity[entity_id])
        for entity_id in entity_ids
        if entity_id in sofa1_by_entity and entity_id in sofa2_by_entity
    ]
    if not pairs:
        return {
            "status": "blocked",
            "reason": "SOFA-1 and SOFA-2 modules are present, but no paired aggregate scores exist for the current cohort.",
            "paired_backend_ready": False,
            "missing_modules": [],
            "supported_scope": "aggregate_only_when_pairs_exist",
        }

    deltas = [round(sofa2 - sofa1, 3) for sofa1, sofa2 in pairs]
    up = sum(1 for delta in deltas if delta > 0)
    down = sum(1 for delta in deltas if delta < 0)
    same = sum(1 for delta in deltas if delta == 0)
    paired_count = len(pairs)
    matrix = _sofa_transition_matrix(pairs)
    return {
        "status": "ready",
        "mode": "worst_icu",
        "paired_backend_ready": True,
        "payload_scope": "paired_score_aggregate_only",
        "inferential_statistics_allowed": False,
        "paired_count": paired_count,
        "coverage_pct": _pct(paired_count, len(entity_ids)),
        "direction_counts": {
            "up": {"count": up, "pct": _pct(up, paired_count)},
            "down": {"count": down, "pct": _pct(down, paired_count)},
            "same": {"count": same, "pct": _pct(same, paired_count)},
        },
        "delta_summary": _numeric_summary(deltas),
        "severity_bins": [row["label"] for row in matrix],
        "transition_matrix": matrix,
        "mode_options": [
            {
                "id": "worst_icu",
                "label": "Worst ICU score",
                "status": "ready",
                "reason": "Uses bounded per-entity maximum SOFA-1 and SOFA-2 scores.",
            },
            {
                "id": "first24h",
                "label": "First 24h",
                "status": "blocked",
                "reason": "Requires an audited time-window pairing backend before display.",
            },
            {
                "id": "time_aligned",
                "label": "Time-aligned",
                "status": "blocked",
                "reason": "Requires an audited timestamp alignment backend before display.",
            },
        ],
    }


def _sofa_transition_matrix(pairs: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    labels = ["0-5", "6-8", "9-11", ">=12"]
    total = len(pairs)
    rows: List[Dict[str, Any]] = []
    for source_label in labels:
        cells = []
        row_total = 0
        for target_label in labels:
            count = sum(
                1
                for sofa1, sofa2 in pairs
                if _sofa_severity_label(sofa1) == source_label and _sofa_severity_label(sofa2) == target_label
            )
            row_total += count
            cells.append({"label": target_label, "count": count, "pct": _pct(count, total)})
        rows.append({"label": source_label, "count": row_total, "cells": cells})
    return rows


def _sofa_severity_label(value: Any) -> str:
    score = dataio._num(value)
    if score is None:
        return "unknown"
    if score <= 5:
        return "0-5"
    if score <= 8:
        return "6-8"
    if score <= 11:
        return "9-11"
    return ">=12"


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
