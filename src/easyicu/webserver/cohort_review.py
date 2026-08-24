"""Bounded Cohort Review aggregates for the native FastAPI UI.

This endpoint is the Stage17+ Cohort Review parity path. It consumes the active
registered EasyICU export and returns cohort-level aggregates only. Row-level
filters, inferential statistics, and matched cohorts stay fail-closed until
their backend contracts exist.
"""

from __future__ import annotations

import hashlib
import copy
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from easyicu.webserver import dataio
from easyicu.webserver import entity_ids as entity_id_contract
from easyicu.webserver import sources as source_store
from easyicu.webserver import review_labels
from easyicu.webserver import prepared_frames

_READ_MODULES = (
    "demographics",
    "outcome",
    "sofa1_score",
    "sofa2_score",
    "sepsis3_sofa2",
)
_COVERAGE_UNIQUE_STAY_SCAN_ROW_LIMIT = 1_000_000
_INTERACTIVE_TIME_INDEXED_READ_ROW_LIMIT = 2_000_000
_INTERACTIVE_SKIP_MODULES = {"sofa1_score", "sofa2_score"}
_SURVIVAL_INTERACTIVE_ENTITY_LIMIT = 250_000
_SURVIVAL_DEFAULT_WINDOW_DAYS = 30.0
_SURVIVAL_28D_WINDOW_DAYS = 28.0
_SUMMARY_CACHE_MAX = 8
_SUMMARY_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_MAX_COMPARE_FEATURES = 48
_FEATURE_RESERVED_COLUMNS = {
    "stay_id",
    "subject_id",
    "hadm_id",
    "icustay_id",
    "patientunitstayid",
    "charttime",
    "time",
    "timestamp",
    "starttime",
    "endtime",
}
_DEFAULT_COMPARE_FEATURES = (
    "blood_gas:lact",
    "vitals:hr",
    "vitals:map",
    "respiratory:pafi",
    "chemistry:crea",
    "chemistry:bun",
    "hematology:wbc",
    "renal:aki_stage",
    "vasopressors:norepi_equiv",
    "ventilator:peep",
)
_TREATMENT_PROFILE_GROUPS = (
    {
        "id": "vasopressors",
        "label": "Vasopressors",
        "label_zh": "血管活性药物",
        "modules": ("vasopressors", "vasopressor"),
    },
    {
        "id": "ventilation",
        "label": "Mechanical ventilation",
        "label_zh": "机械通气",
        "modules": ("ventilator", "ventilation"),
    },
    {
        "id": "respiratory",
        "label": "Respiratory support / gas exchange",
        "label_zh": "呼吸支持 / 血气",
        "modules": ("respiratory", "blood_gas", "blood_gas_analysis"),
    },
    {
        "id": "renal",
        "label": "Renal support / urine output",
        "label_zh": "肾脏支持 / 尿量",
        "modules": ("renal", "renal_urine_output", "urine_output"),
    },
    {
        "id": "medications",
        "label": "Other ICU medications",
        "label_zh": "其他 ICU 用药",
        "modules": ("other_medications", "medications"),
    },
)
_DIAGNOSIS_PROFILE_GROUP = {
    "id": "diagnosis_comorbidity",
    "label": "Diagnoses / comorbidities",
    "label_zh": "诊断 / 共病",
    "modules": ("diagnoses", "diagnosis", "icd", "comorbidity", "comorbidities"),
}
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
_ADM_COLUMNS = ("adm", "admission_type", "adm_type", "admission_location")
_DEATH_COLUMNS = ("death", "mortality", "hospital_mortality", "in_hospital_mortality")
_LOS_COLUMNS = ("los_icu", "icu_los", "los_days")
_HOSP_DEATH_COLUMNS = (
    "hospital_mortality",
    "in_hospital_mortality",
    "death",
    "mortality",
)
_HOSP_LOS_COLUMNS = ("los_hosp", "hospital_los", "los_hospital", "hosp_los", "los_days")
_ICU_DEATH_COLUMNS = ("icu_death", "death_icu", "icu_mortality")
_ICU_LOS_COLUMNS = ("los_icu", "icu_los")
_MORT28_COLUMNS = ("mort_28d", "mortality_28d", "death_28d")
_MORT28_TIME_COLUMNS = (
    "days_to_death_28d",
    "time_to_death_28d",
    "followup_days_28d",
    "survival_time_28d",
)
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
    requested_feature_ids = _requested_feature_ids(body)
    cache_key = _summary_cache_key(path, desc, requested_feature_ids)
    cached = _SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    frames = {
        module: _read_module_frame(path, desc, module) for module in _READ_MODULES
    }

    demo = frames.get("demographics")
    if demo is None or getattr(demo, "empty", True):
        fallback = _fallback_entity_frame(path, desc)
        if fallback is None or getattr(fallback, "empty", True):
            raise CohortReviewError({"error": "no_entity_denominator"})
        demo = fallback

    demo = demo.copy()
    demo["stay_id"] = demo["stay_id"].map(entity_id_contract.normalize_entity_id)
    demo = demo[demo["stay_id"].astype(bool)].drop_duplicates("stay_id")
    if demo.empty:
        raise CohortReviewError({"error": "no_entity_denominator"})

    entity_ids = [str(value) for value in demo["stay_id"].tolist()]
    entity_set = set(entity_ids)
    outcome = _filter_by_entity(frames.get("outcome"), entity_set)
    sofa1 = _filter_by_entity(frames.get("sofa1_score"), entity_set)
    sofa2 = _filter_by_entity(frames.get("sofa2_score"), entity_set)
    sepsis = _filter_by_entity(frames.get("sepsis3_sofa2"), entity_set)

    death_col = _first_column(outcome, _DEATH_COLUMNS)
    los_col = _first_column(outcome, _LOS_COLUMNS)
    sofa1_col = _first_column(sofa1, _SOFA1_COLUMNS)
    sofa_col = _first_column(sofa2, _SOFA2_COLUMNS)
    sepsis_col = _first_column(sepsis, _SEPSIS_COLUMNS)

    death_by_entity = (
        dataio._stay_bool(outcome, death_col, missing_false=True) if death_col else {}
    )
    los_by_entity = dataio._stay_numeric(outcome, los_col, "median") if los_col else {}
    sofa1_by_entity = dataio._stay_numeric(sofa1, sofa1_col, "max") if sofa1_col else {}
    sofa_by_entity = dataio._stay_numeric(sofa2, sofa_col, "max") if sofa_col else {}
    sepsis_by_entity = (
        dataio._stay_bool(sepsis, sepsis_col, missing_false=True) if sepsis_col else {}
    )
    if outcome is not None and not outcome.empty and death_col:
        for entity_id in entity_ids:
            death_by_entity.setdefault(entity_id, False)
    if sepsis is not None and not sepsis.empty and sepsis_col:
        for entity_id in entity_ids:
            sepsis_by_entity.setdefault(entity_id, False)

    age_col = _first_column(demo, _AGE_COLUMNS)
    sex_col = _first_column(demo, _SEX_COLUMNS)
    adm_col = _first_column(demo, _ADM_COLUMNS)
    age_by_entity = _entity_numeric(demo, age_col) if age_col else {}
    sex_values = list(demo[sex_col]) if sex_col else []
    adm_values = list(demo[adm_col]) if adm_col else []
    coverage = _coverage_payload(path, desc)
    quality = _quality_summary(coverage)
    feature_catalog = _feature_catalog(desc, coverage)
    selected_feature_ids = _selected_feature_ids(feature_catalog, requested_feature_ids)
    selected_feature_profiles = _selected_feature_profiles(
        path, desc, entity_set, feature_catalog, selected_feature_ids
    )
    mortality = _bool_summary(
        entity_ids, death_by_entity, true_label="deceased", false_label="survived"
    )
    sepsis_summary = _bool_summary(
        entity_ids, sepsis_by_entity, true_label="positive", false_label="nonpositive"
    )
    age_summary = _numeric_summary(age_by_entity.values())
    los_summary = _numeric_summary(los_by_entity.values())
    sofa_summary = _numeric_summary(sofa_by_entity.values())

    summary = {
        "cohort_size": len(entity_ids),
        "entities": len(entity_ids),
        "modules": int(
            (desc.get("summary") or {}).get("modules")
            or len(
                {f.get("module") for f in desc.get("files") or [] if f.get("module")}
            )
        ),
        "file_count": int(
            (desc.get("summary") or {}).get("file_count")
            or len(desc.get("files") or [])
        ),
        "total_records": int((desc.get("summary") or {}).get("total_rows") or 0),
        "mortality": mortality,
        "mortality_pct": mortality.get("pct"),
        "age": {
            **age_summary,
            "bins": _value_bins(age_by_entity.values(), _AGE_BIN_SPECS),
        },
        "sex": _sex_summary(sex_values),
        "admission": _category_summary(adm_values),
        "sofa2": {**sofa_summary, "bins": _sofa_bins(sofa_by_entity.values())},
        "los_icu_days": {
            **los_summary,
            "bins": _value_bins(los_by_entity.values(), _LOS_BIN_SPECS),
        },
        "sepsis3": sepsis_summary,
        "sepsis_pct": sepsis_summary.get("pct"),
    }
    summary["clinical_profile"] = _clinical_profile_payload(summary, coverage, quality)
    sofa_reclassification = _sofa_reclassification_payload(
        entity_ids=entity_ids,
        sofa1_by_entity=sofa1_by_entity,
        sofa2_by_entity=sofa_by_entity,
        has_sofa1_module=sofa1 is not None
        and not getattr(sofa1, "empty", True)
        and bool(sofa1_col),
        has_sofa2_module=sofa2 is not None
        and not getattr(sofa2, "empty", True)
        and bool(sofa_col),
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
        blocked_features.append(
            {
                "id": "paired_sofa_reclassification",
                "status": "blocked",
                "reason": str(
                    sofa_reclassification.get("reason")
                    or "Paired SOFA-1/SOFA-2 reclassification is not available for this export."
                ),
            }
        )
    survival_analysis = _survival_analysis_payload(
        outcome=outcome,
        entity_ids=entity_ids,
        age_by_entity=age_by_entity,
        sex_by_entity=_entity_text(demo, sex_col) if sex_col else {},
        sofa_by_entity=sofa_by_entity,
        sepsis_by_entity=sepsis_by_entity,
    )

    payload = {
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
                "selected_feature_aggregates",
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
            selected_features=selected_feature_profiles,
        ),
        "feature_catalog": _feature_catalog_payload(
            feature_catalog, selected_feature_ids
        ),
        "feature_selection": _feature_selection_payload(
            feature_catalog, selected_feature_ids, requested_feature_ids
        ),
        "selected_feature_distributions": _selected_feature_distributions(
            entity_ids, selected_feature_profiles
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
    _SUMMARY_CACHE[cache_key] = copy.deepcopy(payload)
    while len(_SUMMARY_CACHE) > _SUMMARY_CACHE_MAX:
        _SUMMARY_CACHE.pop(next(iter(_SUMMARY_CACHE)))
    return payload


def _summary_cache_key(
    path: Path, desc: Dict[str, Any], requested_feature_ids: Tuple[str, ...] | None
) -> Tuple[Any, ...]:
    summary = desc.get("summary") or {}
    manifest_identities: List[Tuple[Any, ...]] = []
    for name in ("_manifest.json", "easyicu_export_manifest.json"):
        manifest_path = path / name
        if manifest_path.exists():
            try:
                stat = manifest_path.stat()
                manifest_identities.append(
                    (
                        name,
                        int(stat.st_size),
                        int(stat.st_mtime_ns),
                        int(stat.st_ctime_ns),
                        int(stat.st_dev),
                        int(stat.st_ino),
                    )
                )
            except OSError:
                manifest_identities.append((name, "unreadable"))
    file_identities: List[Tuple[Any, ...]] = []
    for item in desc.get("files") or []:
        if not isinstance(item, dict) or not item.get("file"):
            continue
        relative = str(item["file"])
        source_file = path / relative
        try:
            stat = source_file.stat()
            file_identities.append(
                (
                    relative,
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    int(stat.st_ctime_ns),
                    int(stat.st_dev),
                    int(stat.st_ino),
                    str(item.get("parquet_sha256") or ""),
                )
            )
        except OSError:
            file_identities.append((relative, "unreadable"))
    return (
        str(path.resolve() if path.exists() else path),
        tuple(manifest_identities),
        tuple(file_identities),
        desc.get("generated"),
        summary.get("stays"),
        summary.get("modules"),
        summary.get("file_count"),
        summary.get("total_rows"),
        requested_feature_ids or ("__default_features__",),
    )


def _reject_unsupported_request(body: Dict[str, Any]) -> None:
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    requested_filters = [
        {
            "id": str(key),
            "reason": _UNSUPPORTED_FILTERS.get(
                str(key), "Cohort Review Stage17 does not accept row-level filters."
            ),
        }
        for key, value in filters.items()
        if _truthy_request(value)
    ]
    if requested_filters:
        raise CohortReviewError(
            {
                "error": "unsupported_filter",
                "unsupported": requested_filters,
                "supported_scope": "registered_source_cohort_aggregates_only",
            }
        )

    stats = body.get("statistics") or body.get("stats") or []
    if isinstance(stats, str):
        stats = [stats]
    requested_stats = [
        {
            "id": str(item),
            "reason": _UNSUPPORTED_STATISTICS.get(
                str(item),
                "Requested statistic is not supported by the Stage17 aggregate endpoint.",
            ),
        }
        for item in stats
        if _truthy_request(item)
    ]
    if requested_stats:
        raise CohortReviewError(
            {
                "error": "unsupported_statistic",
                "unsupported": requested_stats,
                "supported_scope": "descriptive_aggregate_only",
            }
        )

    grouping = body.get("grouping") or body.get("comparison")
    if isinstance(grouping, dict) and grouping.get("mode") == "custom":
        raise CohortReviewError(
            {
                "error": "unsupported_grouping",
                "unsupported": [
                    {"id": "custom", "reason": _UNSUPPORTED_FILTERS["custom_threshold"]}
                ],
                "supported_scope": "fixed_descriptive_group_splits_only",
            }
        )


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
            raise CohortReviewError(
                {"error": "source_not_registered", "path_hash": _hash(norm)}
            )
    else:
        active = registry.get("active_path")
        if not active:
            raise CohortReviewError({"error": "no_active_export"})
        active_norm = _norm_path(str(active))
        source = next(
            (s for s in sources if _norm_path(str(s.get("path") or "")) == active_norm),
            None,
        )
        if source is None:
            raise CohortReviewError(
                {
                    "error": "active_source_not_registered",
                    "path_hash": _hash(active_norm),
                }
            )

    desc = dataio.describe_export_source(str(source.get("path") or ""))
    if not desc.get("ok"):
        raise CohortReviewError(
            {"error": "invalid_export", "detail": desc.get("error")}
        )
    return source, desc


def _skip_expensive_whole_module_read(file_meta: Dict[str, Any], file_path: Path) -> bool:
    """Cohort Statistics reads whole modules, so a huge CSV score table is out.

    Parquet is exempt: the reader projects the wanted columns without
    materialising the rest of the file.
    """

    return (
        str(file_meta.get("module") or "") in _INTERACTIVE_SKIP_MODULES
        and int(file_meta.get("rows") or 0) > _INTERACTIVE_TIME_INDEXED_READ_ROW_LIMIT
        and file_path.suffix.lower() != ".parquet"
    )


def _read_module_frame(path: Path, desc: Dict[str, Any], module: str) -> Any:
    return prepared_frames.read_module_frame(
        path,
        desc,
        module,
        _MODULE_COLUMNS[module],
        skip=_skip_expensive_whole_module_read,
    )


def _fallback_entity_frame(path: Path, desc: Dict[str, Any]) -> Any:
    return prepared_frames.fallback_entity_frame(path, desc)


def _read_selected_columns(path: Path, columns: List[str]) -> Any:
    return prepared_frames.read_selected_columns(path, columns)


def _filter_by_entity(frame: Any, entity_ids: set[str]) -> Any:
    if frame is None or frame.empty or "stay_id" not in frame.columns:
        return frame
    normalized = _normalized_stay_id_series(frame["stay_id"])
    mask = normalized.isin(entity_ids)
    tmp = frame.loc[mask].copy()
    tmp["stay_id"] = normalized.loc[mask].astype(str)
    return tmp


def _normalized_stay_id_series(series: Any) -> Any:
    import pandas as pd
    from pandas.api import types as pd_types

    if pd_types.is_integer_dtype(series):
        return series.astype("string").fillna("")

    if pd_types.is_float_dtype(series):
        normalized = series.astype("string").fillna("")
        whole_number = series.notna() & ((series % 1) == 0)
        if whole_number.any():
            normalized.loc[whole_number] = (
                series.loc[whole_number].astype("Int64").astype("string")
            )
        return normalized.fillna("")

    normalized = series.astype("string").fillna("")
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        whole_number = numeric.notna() & ((numeric % 1) == 0)
    except TypeError:
        return normalized
    if whole_number.any():
        normalized.loc[whole_number] = (
            numeric.loc[whole_number].astype("Int64").astype("string")
        )
    return normalized.fillna("")


def _coverage_payload(path: Path, desc: Dict[str, Any]) -> List[Dict[str, Any]]:
    cohort_size = (desc.get("summary") or {}).get("stays")
    out: List[Dict[str, Any]] = []
    for item in desc.get("files") or []:
        module = str(item.get("module") or "")
        if not module:
            continue
        rows = int(item.get("rows") or 0)
        coverage_basis = "unique_entity_intersection"
        skipped_reason = None
        file_path = path / str(item.get("file") or "")
        if (
            rows > _COVERAGE_UNIQUE_STAY_SCAN_ROW_LIMIT
            and file_path.suffix.lower() != ".parquet"
        ):
            covered = None
            coverage_basis = "metadata_row_count_only"
            skipped_reason = "unique_stay_scan_skipped_large_module"
        else:
            covered = _covered_entities(path, item, cohort_size)
        coverage = (
            round(covered / cohort_size * 100, 1)
            if isinstance(covered, int) and isinstance(cohort_size, int) and cohort_size
            else None
        )
        status = _quality_status(module, coverage)
        row = {
            "module": module,
            "metric_kind": dataio._presence_rate_kind(module) or "coverage",
            "rows": rows,
            "column_count": len(item.get("columns") or []),
            "covered_entities": covered,
            "coverage_pct": coverage,
            "coverage_basis": coverage_basis,
            "quality_status": status,
        }
        if skipped_reason:
            row["skipped_reason"] = skipped_reason
        out.append(row)
    return out


def _covered_entities(path: Path, item: Dict[str, Any], cohort_size: Any) -> int | None:
    if not isinstance(cohort_size, int) or cohort_size <= 0:
        return None
    file_name = str(item.get("file") or "")
    if not file_name:
        return None
    file_path = path / file_name
    if file_path.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            entity_column = entity_id_contract.resolve_entity_id_column(
                item.get("columns") or []
            )
            if not entity_column:
                return None
            frame = pd.read_parquet(file_path, columns=[entity_column])
            frame = entity_id_contract.canonicalize_entity_frame(
                frame, entity_column
            )
            return min(int(frame["stay_id"].dropna().nunique()), cohort_size)
        except Exception:
            return None
    entity_column = entity_id_contract.resolve_entity_id_column(
        item.get("columns") or []
    )
    if not entity_column:
        return None
    try:
        frame = _read_selected_columns(file_path, [entity_column])
        frame = entity_id_contract.canonicalize_entity_frame(frame, entity_column)
        return min(int(frame["stay_id"].dropna().nunique()), cohort_size)
    except Exception:
        return None


def _requested_feature_ids(body: Dict[str, Any]) -> Tuple[str, ...] | None:
    raw = body.get("selected_features")
    if raw is None:
        raw = (
            (body.get("feature_selection") or {}).get("selected_features")
            if isinstance(body.get("feature_selection"), dict)
            else None
        )
    if not isinstance(raw, list):
        return None
    out: List[str] = []
    seen: set[str] = set()
    for value in raw:
        feature_id = str(value or "").strip()
        if not feature_id or ":" not in feature_id or feature_id in seen:
            continue
        seen.add(feature_id)
        out.append(feature_id)
        if len(out) >= _MAX_COMPARE_FEATURES:
            break
    return tuple(out)


def _feature_catalog(
    desc: Dict[str, Any], coverage: List[Dict[str, Any]]
) -> Dict[str, Any]:
    coverage_by_module = {
        str(row.get("module")): row for row in coverage if row.get("module")
    }
    modules: List[Dict[str, Any]] = []
    features_by_id: Dict[str, Dict[str, Any]] = {}
    for item in desc.get("files") or []:
        module = str(item.get("module") or "")
        if not module:
            continue
        columns = [
            str(column)
            for column in item.get("columns") or []
            if _is_compare_feature_column(str(column))
        ]
        coverage_row = coverage_by_module.get(module) or {}
        feature_rows = [
            {
                "id": _feature_id(module, column),
                "module": module,
                "column": column,
                "label": _feature_label(column),
                "kind": _feature_kind_hint(column),
                "rows": int(item.get("rows") or 0),
                "coverage_pct": coverage_row.get("coverage_pct"),
                "quality_status": coverage_row.get("quality_status"),
            }
            for column in columns
        ]
        for feature in feature_rows:
            features_by_id[feature["id"]] = feature
        modules.append(
            {
                "module": module,
                "label": _module_label(module),
                "rows": int(item.get("rows") or 0),
                "feature_count": len(feature_rows),
                "coverage_pct": coverage_row.get("coverage_pct"),
                "quality_status": coverage_row.get("quality_status"),
                "features": feature_rows,
            }
        )
    modules.sort(key=lambda row: row["module"])
    return {
        "modules": modules,
        "features_by_id": features_by_id,
        "total_modules": len(modules),
        "total_features": len(features_by_id),
    }


def _feature_catalog_payload(
    catalog: Dict[str, Any], selected_feature_ids: List[str]
) -> Dict[str, Any]:
    selected = set(selected_feature_ids)
    modules = []
    for module in catalog.get("modules") or []:
        features = [
            {key: value for key, value in feature.items() if key != "rows"}
            | {"selected": feature.get("id") in selected}
            for feature in module.get("features") or []
        ]
        modules.append(
            {
                "module": module.get("module"),
                "label": module.get("label"),
                "rows": module.get("rows"),
                "feature_count": module.get("feature_count"),
                "coverage_pct": module.get("coverage_pct"),
                "quality_status": module.get("quality_status"),
                "selected_count": sum(
                    1 for feature in features if feature.get("selected")
                ),
                "features": features,
            }
        )
    return {
        "total_modules": int(catalog.get("total_modules") or 0),
        "total_features": int(catalog.get("total_features") or 0),
        "max_selected_features": _MAX_COMPARE_FEATURES,
        "modules": modules,
    }


def _feature_selection_payload(
    catalog: Dict[str, Any],
    selected_feature_ids: List[str],
    requested_feature_ids: Tuple[str, ...] | None,
) -> Dict[str, Any]:
    features_by_id = catalog.get("features_by_id") or {}
    selected = [
        features_by_id[feature_id]
        for feature_id in selected_feature_ids
        if feature_id in features_by_id
    ]
    default_ids = [
        feature_id
        for feature_id in _DEFAULT_COMPARE_FEATURES
        if feature_id in features_by_id
    ]
    return {
        "mode": "requested" if requested_feature_ids is not None else "default",
        "selected_count": len(selected),
        "available_count": int(catalog.get("total_features") or 0),
        "module_count": int(catalog.get("total_modules") or 0),
        "max_selected_features": _MAX_COMPARE_FEATURES,
        "default_ids": default_ids,
        "selected": [
            {
                "id": feature.get("id"),
                "module": feature.get("module"),
                "column": feature.get("column"),
                "label": feature.get("label"),
                "kind": feature.get("kind"),
            }
            for feature in selected
        ],
        "ignored": [
            feature_id
            for feature_id in (requested_feature_ids or ())
            if feature_id not in features_by_id
        ],
    }


def _selected_feature_ids(
    catalog: Dict[str, Any], requested_feature_ids: Tuple[str, ...] | None
) -> List[str]:
    features_by_id = catalog.get("features_by_id") or {}
    raw_ids = (
        list(requested_feature_ids)
        if requested_feature_ids is not None
        else list(_DEFAULT_COMPARE_FEATURES)
    )
    out: List[str] = []
    seen: set[str] = set()
    for feature_id in raw_ids:
        if feature_id in features_by_id and feature_id not in seen:
            seen.add(feature_id)
            out.append(feature_id)
        if len(out) >= _MAX_COMPARE_FEATURES:
            break
    if out or requested_feature_ids is not None:
        return out
    return [
        feature_id
        for feature_id in list(features_by_id)[: min(8, _MAX_COMPARE_FEATURES)]
    ]


def _selected_feature_profiles(
    path: Path,
    desc: Dict[str, Any],
    entity_set: set[str],
    catalog: Dict[str, Any],
    selected_feature_ids: List[str],
) -> List[Dict[str, Any]]:
    features_by_id = catalog.get("features_by_id") or {}
    selected = [
        features_by_id[feature_id]
        for feature_id in selected_feature_ids
        if feature_id in features_by_id
    ]
    by_module: Dict[str, List[Dict[str, Any]]] = {}
    for feature in selected:
        by_module.setdefault(str(feature.get("module") or ""), []).append(feature)
    out: List[Dict[str, Any]] = []
    files_by_module = {
        str(item.get("module") or ""): item
        for item in desc.get("files") or []
        if item.get("module")
    }
    for module, features in by_module.items():
        item = files_by_module.get(module)
        if not item:
            continue
        file_path = path / str(item.get("file") or "")
        rows = int(item.get("rows") or 0)
        if (
            rows > _COVERAGE_UNIQUE_STAY_SCAN_ROW_LIMIT
            and file_path.suffix.lower() != ".parquet"
        ):
            for feature in features:
                out.append(
                    {
                        "id": feature["id"],
                        "module": module,
                        "column": feature["column"],
                        "label": feature["label"],
                        "kind": "blocked",
                        "aggregation": "blocked_large_non_parquet",
                        "mapping": {},
                        "reason": "Large non-Parquet module requires an audited background aggregate before interactive comparison.",
                    }
                )
            continue
        entity_column = entity_id_contract.resolve_entity_id_column(
            item.get("columns") or []
        )
        if not entity_column:
            continue
        columns = [entity_column] + [
            str(feature["column"])
            for feature in features
            if str(feature["column"]) in (item.get("columns") or [])
        ]
        if len(columns) <= 1:
            continue
        try:
            feature_frame = _read_selected_columns(file_path, columns)
            feature_frame = entity_id_contract.canonicalize_entity_frame(
                feature_frame, entity_column
            )
            frame = _filter_by_entity(
                feature_frame, entity_set
            )
        except Exception as exc:
            for feature in features:
                out.append(
                    {
                        "id": feature["id"],
                        "module": module,
                        "column": feature["column"],
                        "label": feature["label"],
                        "kind": "blocked",
                        "aggregation": "read_failed",
                        "mapping": {},
                        "reason": f"Could not read selected feature column: {type(exc).__name__}",
                    }
                )
            continue
        for feature in features:
            column = str(feature.get("column") or "")
            if column not in getattr(frame, "columns", []):
                continue
            out.append(_selected_feature_profile(frame, feature))
    return out


def _selected_feature_profile(frame: Any, feature: Dict[str, Any]) -> Dict[str, Any]:
    column = str(feature.get("column") or "")
    kind = _infer_feature_kind(frame, column)
    if kind == "binary":
        # The Cohort owner already defines missing hospital-death event rows as
        # non-events when it builds the mortality summary above.  Reuse that
        # exact concept-specific semantic for the selected-feature view so the
        # same export cannot report 10/10 mortality coverage in one panel and
        # six unknown outcomes in another.  Other binary concepts remain
        # fail-closed because their missingness may not mean a negative event.
        missing_false = (
            str(feature.get("module") or "").strip().lower() == "outcome"
            and column.strip().lower() in _DEATH_COLUMNS
        )
        mapping = dataio._stay_bool(frame, column, missing_false=missing_false)
        aggregation = "entity_any_positive_pct"
    elif kind == "numeric":
        mapping = dataio._stay_numeric(frame, column, "median")
        aggregation = "entity_median"
    else:
        mapping = _stay_present(frame, column)
        kind = "presence"
        aggregation = "entity_nonmissing_pct"
    return {
        "id": feature.get("id"),
        "module": feature.get("module"),
        "column": column,
        "label": feature.get("label") or column,
        "kind": kind,
        "aggregation": aggregation,
        "mapping": mapping,
        "reason": None,
    }


def _selected_feature_distributions(
    entity_ids: List[str], profiles: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Project selected features as bounded stay-level descriptive histograms."""

    total = len(entity_ids)
    out: List[Dict[str, Any]] = []
    for profile in profiles[:_MAX_COMPARE_FEATURES]:
        mapping = profile.get("mapping")
        mapping = mapping if isinstance(mapping, dict) else {}
        common = {
            "id": profile.get("id"),
            "module": profile.get("module"),
            "column": profile.get("column"),
            "label": profile.get("label"),
            "kind": profile.get("kind"),
            "aggregation": profile.get("aggregation"),
            "denominator": total,
            "observed": len(mapping),
            "observed_pct": _pct(len(mapping), total),
        }
        if profile.get("kind") == "numeric":
            values = sorted(
                value
                for value in (dataio._num(item) for item in mapping.values())
                if value is not None
            )
            bins: List[Dict[str, Any]] = []
            if values:
                low, high = values[0], values[-1]
                if low == high:
                    bins = [{"low": low, "high": high, "count": len(values)}]
                else:
                    width = (high - low) / 8
                    counts = [0] * 8
                    for value in values:
                        index = min(7, int((value - low) / width))
                        counts[index] += 1
                    bins = [
                        {
                            "low": round(low + index * width, 4),
                            "high": round(low + (index + 1) * width, 4),
                            "count": count,
                        }
                        for index, count in enumerate(counts)
                    ]
            out.append({**common, "summary": _numeric_summary(values), "bins": bins})
        elif profile.get("kind") == "binary":
            positive = sum(value is True for value in mapping.values())
            negative = sum(value is False for value in mapping.values())
            out.append(
                {
                    **common,
                    "categories": [
                        {"label": "Positive", "count": positive},
                        {"label": "Negative", "count": negative},
                        {"label": "Unknown", "count": max(0, total - positive - negative)},
                    ],
                }
            )
        else:
            present = len(mapping)
            out.append(
                {
                    **common,
                    "categories": [
                        {"label": "Observed", "count": present},
                        {"label": "Missing", "count": max(0, total - present)},
                    ],
                }
            )
    return out


def _infer_feature_kind(frame: Any, column: str) -> str:
    if (
        frame is None
        or getattr(frame, "empty", True)
        or column not in getattr(frame, "columns", [])
    ):
        return "presence"
    import pandas as pd

    values = frame[column].dropna()
    if values.empty:
        return "presence"
    sample = values.head(5000)
    numeric = pd.to_numeric(sample, errors="coerce")
    numeric_ratio = int(numeric.notna().sum()) / len(sample)
    if numeric_ratio >= 0.85:
        unique_values = {float(value) for value in numeric.dropna().unique()[:20]}
        if unique_values and unique_values.issubset({0.0, 1.0}):
            return "binary"
        return "numeric"
    flags = [dataio._truthy(value) for value in sample]
    known_flags = [value for value in flags if value is not None]
    if known_flags and len(known_flags) / len(sample) >= 0.85:
        return "binary"
    return "presence"


def _stay_present(frame: Any, column: str) -> Dict[str, bool]:
    if (
        frame is None
        or getattr(frame, "empty", True)
        or "stay_id" not in frame.columns
        or column not in frame.columns
    ):
        return {}
    out: Dict[str, bool] = {}
    for entity_id, vals in frame.groupby("stay_id")[column]:
        present = any(dataio._clean(value) is not None for value in vals)
        if present:
            out[str(entity_id)] = True
    return out


def _is_compare_feature_column(column: str) -> bool:
    lower = column.strip().lower()
    if lower in _FEATURE_RESERVED_COLUMNS:
        return False
    if lower.endswith("_id") or lower.endswith("id"):
        return False
    return bool(lower)


def _feature_id(module: str, column: str) -> str:
    return f"{module}:{column}"


def _module_label(module: str, lang: str = "en") -> str:
    return review_labels.module_label(module, lang)


def _feature_label(column: str) -> str:
    return (
        column.replace("_", " ").upper()
        if len(column) <= 4
        else column.replace("_", " ").title()
    )


def _feature_kind_hint(column: str) -> str:
    lower = column.lower()
    if lower.startswith(("is_", "has_")) or lower.endswith(
        ("_ind", "_positive", "_failure", "_event", "_tx")
    ):
        return "binary"
    if lower in {
        "death",
        "aki",
        "rrt",
        "abx",
        "susp_inf",
        "mech_vent",
        "vent_ind",
        "vaso_ind",
    }:
        return "binary"
    if lower in {"sex", "gender", "adm", "avpu"}:
        return "categorical"
    return "numeric"


def _quality_summary(coverage: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"ok": 0, "warn": 0, "bad": 0, "neutral": 0, "unknown": 0}
    values: List[float] = []
    for row in coverage:
        status = str(row.get("quality_status") or "unknown")
        if status not in counts:
            status = "unknown"
        counts[status] += 1
        pct = row.get("coverage_pct")
        if isinstance(pct, (int, float)) and status not in {"neutral", "unknown"}:
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
    selected_features: List[Dict[str, Any]],
) -> Dict[str, Any]:
    supported = [
        _group_from_bool(
            "survival",
            "Survived vs Deceased",
            entity_ids,
            death_by_entity,
            false_name="Survived",
            true_name="Deceased",
        ),
        _group_from_age(entity_ids, age_by_entity),
        _group_from_sex(entity_ids, sex_by_entity),
        _group_from_los(entity_ids, los_by_entity),
        _group_from_bool(
            "sepsis",
            "Sepsis vs Non-sepsis",
            entity_ids,
            sepsis_by_entity,
            false_name="Non-sepsis",
            true_name="Sepsis",
        ),
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
            selected_features=selected_features,
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
    selected_features: List[Dict[str, Any]],
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
            {
                "metric": "N",
                "kind": "count",
                "values": [len(members) for members in member_sets],
            },
            {
                "metric": "Mortality %",
                "kind": "percent",
                "values": [
                    _bool_pct(members, death_by_entity) for members in member_sets
                ],
            },
            {
                "metric": "Female %",
                "kind": "percent",
                "values": [
                    _female_pct(members, sex_by_entity) for members in member_sets
                ],
            },
            {
                "metric": "Median age",
                "kind": "numeric",
                "unit": "years",
                "values": [
                    _median_for(members, age_by_entity) for members in member_sets
                ],
            },
            {
                "metric": "Median SOFA-2",
                "kind": "numeric",
                "values": [
                    _median_for(members, sofa_by_entity) for members in member_sets
                ],
            },
            {
                "metric": "Median ICU LOS",
                "kind": "numeric",
                "unit": "days",
                "values": [
                    _median_for(members, los_by_entity) for members in member_sets
                ],
            },
            {
                "metric": "Sepsis-3 %",
                "kind": "percent",
                "values": [
                    _bool_pct(members, sepsis_by_entity) for members in member_sets
                ],
            },
        ]
        + _selected_feature_rows(member_sets, selected_features),
    }


def _selected_feature_rows(
    member_sets: List[List[str]], selected_features: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for feature in selected_features:
        kind = str(feature.get("kind") or "")
        mapping = feature.get("mapping") or {}
        label = str(
            feature.get("label") or feature.get("column") or feature.get("id") or ""
        )
        module = str(feature.get("module") or "")
        if kind == "blocked":
            rows.append(
                {
                    "metric": label,
                    "feature_id": feature.get("id"),
                    "module": module,
                    "column": feature.get("column"),
                    "kind": "blocked",
                    "values": [None for _members in member_sets],
                    "status": "blocked",
                    "aggregation": feature.get("aggregation"),
                    "reason": feature.get("reason"),
                }
            )
            continue
        if kind == "binary":
            rows.append(
                {
                    "metric": f"{label} %",
                    "feature_id": feature.get("id"),
                    "module": module,
                    "column": feature.get("column"),
                    "kind": "percent",
                    "values": [_bool_pct(members, mapping) for members in member_sets],
                    "status": "selected_feature",
                    "aggregation": feature.get("aggregation"),
                }
            )
        elif kind == "numeric":
            rows.append(
                {
                    "metric": f"Median {label}",
                    "feature_id": feature.get("id"),
                    "module": module,
                    "column": feature.get("column"),
                    "kind": "numeric",
                    "values": [
                        _median_for(members, mapping) for members in member_sets
                    ],
                    "status": "selected_feature",
                    "aggregation": feature.get("aggregation"),
                }
            )
        else:
            rows.append(
                {
                    "metric": f"{label} available %",
                    "feature_id": feature.get("id"),
                    "module": module,
                    "column": feature.get("column"),
                    "kind": "percent",
                    "values": [_bool_pct(members, mapping) for members in member_sets],
                    "status": "selected_feature",
                    "aggregation": feature.get("aggregation"),
                }
            )
    return rows


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
        return _bool_member_groups(
            entity_ids, death_by_entity, false_name="Survived", true_name="Deceased"
        )
    if group_id == "age":
        return [
            (
                "<65",
                [
                    entity_id
                    for entity_id in entity_ids
                    if entity_id in age_by_entity and age_by_entity[entity_id] < 65
                ],
            ),
            (
                ">=65",
                [
                    entity_id
                    for entity_id in entity_ids
                    if entity_id in age_by_entity and age_by_entity[entity_id] >= 65
                ],
            ),
            (
                "Unknown",
                [
                    entity_id
                    for entity_id in entity_ids
                    if entity_id not in age_by_entity
                ],
            ),
        ]
    if group_id == "sex":
        return [
            (
                "Female",
                [
                    entity_id
                    for entity_id in entity_ids
                    if _sex_bucket(sex_by_entity.get(entity_id)) == "female"
                ],
            ),
            (
                "Male",
                [
                    entity_id
                    for entity_id in entity_ids
                    if _sex_bucket(sex_by_entity.get(entity_id)) == "male"
                ],
            ),
            (
                "Unknown",
                [
                    entity_id
                    for entity_id in entity_ids
                    if _sex_bucket(sex_by_entity.get(entity_id))
                    not in {"female", "male"}
                ],
            ),
        ]
    if group_id == "los":
        values = [value for value in los_by_entity.values() if value is not None]
        threshold = dataio._median(values)
        if threshold is None:
            return [
                (
                    "Known",
                    [
                        entity_id
                        for entity_id in entity_ids
                        if entity_id in los_by_entity
                    ],
                ),
                (
                    "Unknown",
                    [
                        entity_id
                        for entity_id in entity_ids
                        if entity_id not in los_by_entity
                    ],
                ),
            ]
        return [
            (
                "Short/median",
                [
                    entity_id
                    for entity_id in entity_ids
                    if entity_id in los_by_entity
                    and los_by_entity[entity_id] <= threshold
                ],
            ),
            (
                "Long",
                [
                    entity_id
                    for entity_id in entity_ids
                    if entity_id in los_by_entity
                    and los_by_entity[entity_id] > threshold
                ],
            ),
            (
                "Unknown",
                [
                    entity_id
                    for entity_id in entity_ids
                    if entity_id not in los_by_entity
                ],
            ),
        ]
    if group_id == "sepsis":
        return _bool_member_groups(
            entity_ids, sepsis_by_entity, false_name="Non-sepsis", true_name="Sepsis"
        )
    return [("Cohort", list(entity_ids))]


def _bool_member_groups(
    entity_ids: List[str],
    mapping: Dict[str, bool],
    *,
    false_name: str,
    true_name: str,
) -> List[Tuple[str, List[str]]]:
    return [
        (
            false_name,
            [entity_id for entity_id in entity_ids if mapping.get(entity_id) is False],
        ),
        (
            true_name,
            [entity_id for entity_id in entity_ids if mapping.get(entity_id) is True],
        ),
        (
            "Unknown",
            [
                entity_id
                for entity_id in entity_ids
                if mapping.get(entity_id) not in {False, True}
            ],
        ),
    ]


def _median_for(entity_ids: List[str], mapping: Dict[str, float]) -> float | None:
    values = [
        mapping[entity_id]
        for entity_id in entity_ids
        if entity_id in mapping and mapping[entity_id] is not None
    ]
    return dataio._median(values)


def _bool_pct(entity_ids: List[str], mapping: Dict[str, bool]) -> float | None:
    if not entity_ids:
        return None
    count = sum(1 for entity_id in entity_ids if mapping.get(entity_id) is True)
    return _pct(count, len(entity_ids))


def _female_pct(entity_ids: List[str], mapping: Dict[str, str]) -> float | None:
    if not entity_ids:
        return None
    count = sum(
        1 for entity_id in entity_ids if _sex_bucket(mapping.get(entity_id)) == "female"
    )
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
            {
                "label": false_name,
                "count": false_count,
                "pct": _pct(false_count, len(entity_ids)),
            },
            {
                "label": true_name,
                "count": true_count,
                "pct": _pct(true_count, len(entity_ids)),
            },
            {
                "label": "Unknown",
                "count": unknown,
                "pct": _pct(unknown, len(entity_ids)),
            },
        ],
        "inferential_statistics_allowed": False,
    }


def _group_from_age(
    entity_ids: List[str], age_by_entity: Dict[str, float]
) -> Dict[str, Any] | None:
    if not age_by_entity:
        return None
    younger = sum(
        1
        for entity_id in entity_ids
        if entity_id in age_by_entity and age_by_entity[entity_id] < 65
    )
    older = sum(
        1
        for entity_id in entity_ids
        if entity_id in age_by_entity and age_by_entity[entity_id] >= 65
    )
    unknown = len(entity_ids) - younger - older
    return {
        "id": "age",
        "label": "Age Groups",
        "status": "supported",
        "basis": "age_threshold_65_descriptive",
        "groups": [
            {"label": "<65", "count": younger, "pct": _pct(younger, len(entity_ids))},
            {"label": ">=65", "count": older, "pct": _pct(older, len(entity_ids))},
            {
                "label": "Unknown",
                "count": unknown,
                "pct": _pct(unknown, len(entity_ids)),
            },
        ],
        "inferential_statistics_allowed": False,
    }


def _group_from_sex(
    entity_ids: List[str], sex_by_entity: Dict[str, str]
) -> Dict[str, Any] | None:
    if not sex_by_entity:
        return None
    female = sum(
        1
        for entity_id in entity_ids
        if _sex_bucket(sex_by_entity.get(entity_id)) == "female"
    )
    male = sum(
        1
        for entity_id in entity_ids
        if _sex_bucket(sex_by_entity.get(entity_id)) == "male"
    )
    unknown = len(entity_ids) - female - male
    return {
        "id": "sex",
        "label": "Female vs Male",
        "status": "supported",
        "basis": "sex_metadata_descriptive",
        "groups": [
            {"label": "Female", "count": female, "pct": _pct(female, len(entity_ids))},
            {"label": "Male", "count": male, "pct": _pct(male, len(entity_ids))},
            {
                "label": "Unknown",
                "count": unknown,
                "pct": _pct(unknown, len(entity_ids)),
            },
        ],
        "inferential_statistics_allowed": False,
    }


def _group_from_los(
    entity_ids: List[str], los_by_entity: Dict[str, float]
) -> Dict[str, Any] | None:
    values = [v for v in los_by_entity.values() if v is not None]
    if not values:
        return None
    threshold = dataio._median(values)
    if threshold is None:
        return None
    short = sum(
        1
        for entity_id in entity_ids
        if entity_id in los_by_entity and los_by_entity[entity_id] <= threshold
    )
    long = sum(
        1
        for entity_id in entity_ids
        if entity_id in los_by_entity and los_by_entity[entity_id] > threshold
    )
    unknown = len(entity_ids) - short - long
    return {
        "id": "los",
        "label": "Short vs Long Stay",
        "status": "supported",
        "basis": "median_los_descriptive_split",
        "threshold": threshold,
        "groups": [
            {
                "label": "Short/median",
                "count": short,
                "pct": _pct(short, len(entity_ids)),
            },
            {"label": "Long", "count": long, "pct": _pct(long, len(entity_ids))},
            {
                "label": "Unknown",
                "count": unknown,
                "pct": _pct(unknown, len(entity_ids)),
            },
        ],
        "inferential_statistics_allowed": False,
    }


def _entity_numeric(frame: Any, column: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if (
        frame is None
        or frame.empty
        or "stay_id" not in frame.columns
        or column not in frame.columns
    ):
        return out
    import pandas as pd

    entity_ids = frame["stay_id"].map(entity_id_contract.normalize_entity_id)
    values = pd.to_numeric(frame[column], errors="coerce")
    for entity_id, value in zip(entity_ids, values):
        if entity_id and value is not None:
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            out[entity_id] = float(value)
    return out


def _entity_text(frame: Any, column: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if (
        frame is None
        or frame.empty
        or "stay_id" not in frame.columns
        or column not in frame.columns
    ):
        return out
    entity_ids = frame["stay_id"].map(entity_id_contract.normalize_entity_id)
    for entity_id, raw_value in zip(entity_ids, frame[column]):
        value = dataio._clean(raw_value)
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


def _category_summary(values: List[Any], *, top_n: int = 6) -> Dict[str, Any]:
    """Count a categorical column (e.g. admission type) into a bounded bar payload.

    Returns ``label`` / ``count`` / ``pct`` rows shaped like ``_value_bins`` so the
    frontend renders it through the same bar renderer. Low-frequency categories
    beyond ``top_n`` collapse into one ``Other`` bucket so the chart stays bounded.
    """
    labels = [
        str(v).strip()
        for v in values
        if str(v).strip() and str(v).strip().lower() != "nan"
    ]
    total = len(labels)
    if not total:
        return {"count": 0, "bins": []}
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    bins = [
        {"label": label, "count": count, "pct": _pct(count, total)}
        for label, count in ordered[:top_n]
    ]
    remainder = sum(count for _, count in ordered[top_n:])
    if remainder:
        bins.append(
            {"label": "Other", "count": remainder, "pct": _pct(remainder, total)}
        )
    return {"count": total, "distinct": len(counts), "bins": bins}


def _clinical_profile_payload(
    summary: Dict[str, Any],
    coverage: List[Dict[str, Any]],
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    """Clinically interpretable cohort phenotype, aggregate-only.

    The payload intentionally avoids proxy visualizations such as age x LOS
    crosstabs. It reports dimensions clinicians usually inspect first and
    clearly marks unavailable diagnosis/treatment modules instead of inventing
    them from unrelated columns.
    """

    cohort_size = int(summary.get("cohort_size") or 0)
    treatment_items = [
        _module_profile_item(spec, coverage, cohort_size)
        for spec in _TREATMENT_PROFILE_GROUPS
    ]
    diagnosis_item = _module_profile_item(
        _DIAGNOSIS_PROFILE_GROUP, coverage, cohort_size
    )
    coverage_items = _coverage_profile_items(coverage, quality, cohort_size)
    domains = [
        {
            "id": "demographics",
            "label": "Demographics",
            "label_zh": "人口统计",
            "status": "ready" if (summary.get("age") or {}).get("count") else "partial",
            "items": [
                _numeric_profile_item(
                    "age",
                    "Median age",
                    "年龄中位数",
                    summary.get("age") or {},
                    "years",
                    "岁",
                ),
                _pct_profile_item(
                    "female",
                    "Female",
                    "女性",
                    (summary.get("sex") or {}).get("female_pct"),
                    (summary.get("sex") or {}).get("female_count"),
                    cohort_size,
                ),
                _category_profile_item(
                    "admission",
                    "Admission source",
                    "入院来源",
                    summary.get("admission") or {},
                ),
            ],
        },
        {
            "id": "severity_outcome",
            "label": "Severity and outcomes",
            "label_zh": "严重程度与结局",
            "status": "ready",
            "items": [
                _numeric_profile_item(
                    "sofa2",
                    "Worst SOFA-2",
                    "最严重 SOFA-2",
                    summary.get("sofa2") or {},
                    "points",
                    "分",
                ),
                _pct_profile_item(
                    "sepsis3",
                    "Sepsis-3 positive",
                    "Sepsis-3 阳性",
                    summary.get("sepsis_pct"),
                    (summary.get("sepsis3") or {}).get("positive_count"),
                    cohort_size,
                    kind="event_rate",
                ),
                _pct_profile_item(
                    "mortality",
                    "Hospital mortality",
                    "院内死亡",
                    summary.get("mortality_pct"),
                    (summary.get("mortality") or {}).get("deceased_count"),
                    cohort_size,
                    kind="event_rate",
                ),
                _numeric_profile_item(
                    "los_icu",
                    "ICU length of stay",
                    "ICU 住院时长",
                    summary.get("los_icu_days") or {},
                    "days",
                    "天",
                ),
            ],
        },
        {
            "id": "treatments",
            "label": "Treatments and organ support",
            "label_zh": "治疗暴露与器官支持",
            "status": _domain_status(treatment_items),
            "items": treatment_items,
        },
        {
            "id": "diagnosis",
            "label": "Diagnoses and comorbidities",
            "label_zh": "诊断与共病",
            "status": diagnosis_item.get("status"),
            "items": [diagnosis_item],
        },
        {
            "id": "data_completeness",
            "label": "Data completeness",
            "label_zh": "数据覆盖",
            "status": "ready" if coverage_items else "unavailable",
            "items": coverage_items,
        },
    ]
    return {
        "status": "aggregate_only",
        "payload_scope": "cohort_aggregate_only_no_patient_rows",
        "domains": domains,
        "notes": [
            {
                "label": "No patient rows",
                "label_zh": "不返回患者行",
                "text": "Treatment and diagnosis cards use module-level entity coverage; detailed rows remain in the local export only.",
                "text_zh": "治疗和诊断卡片使用模块级实体覆盖；明细行仍只保留在本地导出中。",
            }
        ],
    }


def _numeric_profile_item(
    item_id: str,
    label: str,
    label_zh: str,
    payload: Dict[str, Any],
    unit: str,
    unit_zh: str,
) -> Dict[str, Any]:
    count = int(payload.get("count") or 0)
    return {
        "id": item_id,
        "label": label,
        "label_zh": label_zh,
        "kind": "numeric",
        "status": "ready" if count else "unavailable",
        "value": payload.get("median"),
        "value_label": "median",
        "value_label_zh": "中位数",
        "unit": unit,
        "unit_zh": unit_zh,
        "count": count,
        "min": payload.get("min"),
        "max": payload.get("max"),
    }


def _pct_profile_item(
    item_id: str,
    label: str,
    label_zh: str,
    pct: Any,
    count: Any,
    denominator: int,
    *,
    kind: str = "proportion",
) -> Dict[str, Any]:
    count_int = int(count or 0) if isinstance(count, (int, float)) else None
    return {
        "id": item_id,
        "label": label,
        "label_zh": label_zh,
        "kind": kind,
        "status": "ready" if isinstance(pct, (int, float)) else "unavailable",
        "pct": pct,
        "count": count_int,
        "denominator": denominator,
    }


def _category_profile_item(
    item_id: str, label: str, label_zh: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    bins = payload.get("bins") or []
    return {
        "id": item_id,
        "label": label,
        "label_zh": label_zh,
        "kind": "category",
        "status": "ready" if bins else "unavailable",
        "count": int(payload.get("count") or 0),
        "distinct": int(payload.get("distinct") or 0),
        "bins": bins[:4],
    }


def _module_profile_item(
    spec: Dict[str, Any], coverage: List[Dict[str, Any]], cohort_size: int
) -> Dict[str, Any]:
    module_names = {str(name) for name in spec.get("modules") or ()}
    rows = [
        row
        for row in coverage
        if str(row.get("module") or "") in module_names
    ]
    if not rows:
        return {
            "id": spec.get("id"),
            "label": spec.get("label"),
            "label_zh": spec.get("label_zh"),
            "kind": "module_coverage",
            "status": "unavailable",
            "reason": "module_not_in_current_export",
            "reason_zh": "当前导出未包含对应模块",
            "modules": list(spec.get("modules") or ()),
        }
    covered_values = [
        int(row["covered_entities"])
        for row in rows
        if isinstance(row.get("covered_entities"), int)
    ]
    covered = max(covered_values) if covered_values else None
    return {
        "id": spec.get("id"),
        "label": spec.get("label"),
        "label_zh": spec.get("label_zh"),
        "kind": "module_coverage",
        "status": "ready" if covered is not None else "schema_only",
        "pct": _pct(covered, cohort_size) if covered is not None else None,
        "count": covered,
        "denominator": cohort_size,
        "rows": sum(int(row.get("rows") or 0) for row in rows),
        "column_count": sum(int(row.get("column_count") or 0) for row in rows),
        "modules": [str(row.get("module") or "") for row in rows],
        "coverage_basis": "max_unique_entity_intersection_across_modules"
        if covered is not None
        else "schema_only",
    }


def _coverage_profile_items(
    coverage: List[Dict[str, Any]],
    quality: Dict[str, Any],
    cohort_size: int,
) -> List[Dict[str, Any]]:
    items = [
        {
            "id": "modules_ok",
            "label": "Modules ready",
            "label_zh": "覆盖良好模块",
            "kind": "count",
            "status": "ready",
            "count": int(quality.get("modules_ok") or 0),
            "denominator": len(coverage),
        },
        {
            "id": "watchlist",
            "label": "Coverage watchlist",
            "label_zh": "覆盖率关注项",
            "kind": "count",
            "status": "ready",
            "count": int(quality.get("watchlist_count") or 0),
            "denominator": len(coverage),
        },
    ]
    weakest = sorted(
        [
            row
            for row in coverage
            if isinstance(row.get("coverage_pct"), (int, float))
            and str(row.get("quality_status") or "") not in {"neutral"}
        ],
        key=lambda row: float(row.get("coverage_pct") or 0),
    )[:3]
    for row in weakest:
        items.append(
            {
                "id": f"coverage_{row.get('module')}",
                "label": _module_label(str(row.get("module") or "")),
                "label_zh": _module_label(str(row.get("module") or ""), "zh"),
                "kind": "module_coverage",
                "status": str(row.get("quality_status") or "unknown"),
                "pct": row.get("coverage_pct"),
                "count": row.get("covered_entities"),
                "denominator": cohort_size,
                "rows": int(row.get("rows") or 0),
                "modules": [str(row.get("module") or "")],
            }
        )
    return items


def _domain_status(items: List[Dict[str, Any]]) -> str:
    statuses = {str(item.get("status") or "") for item in items}
    if "ready" in statuses or "schema_only" in statuses:
        return "partial" if "unavailable" in statuses else "ready"
    return "unavailable"


def _value_bins(values: Iterable[Any], specs: List[tuple]) -> List[Dict[str, Any]]:
    """Bin numeric values into a labelled histogram payload.

    ``specs`` is an ordered list of ``(label, predicate)`` pairs. The returned
    rows mirror ``_sofa_bins`` (``label`` / ``count`` / ``pct``) so the frontend
    renders every distribution chart through one bar renderer.
    """
    vals = [v for v in (dataio._num(v) for v in values) if v is not None]
    total = len(vals)
    out: List[Dict[str, Any]] = []
    for label, predicate in specs:
        count = sum(1 for value in vals if predicate(value))
        out.append({"label": label, "count": count, "pct": _pct(count, total)})
    return out


_AGE_BIN_SPECS: List[tuple] = [
    ("<40", lambda v: v < 40),
    ("40-59", lambda v: 40 <= v < 60),
    ("60-74", lambda v: 60 <= v < 75),
    (">=75", lambda v: v >= 75),
]
_LOS_BIN_SPECS: List[tuple] = [
    ("<2d", lambda v: v < 2),
    ("2-5d", lambda v: 2 <= v < 5),
    ("5-10d", lambda v: 5 <= v < 10),
    (">=10d", lambda v: v >= 10),
]


def _sofa_bins(values: Iterable[Any]) -> List[Dict[str, Any]]:
    return _value_bins(
        values,
        [
            ("0-5", lambda v: v <= 5),
            ("6-8", lambda v: 6 <= v <= 8),
            ("9-11", lambda v: 9 <= v <= 11),
            (">=12", lambda v: v >= 12),
        ],
    )


def _survival_analysis_payload(
    *,
    outcome: Any,
    entity_ids: List[str],
    age_by_entity: Dict[str, float],
    sex_by_entity: Dict[str, str],
    sofa_by_entity: Dict[str, float],
    sepsis_by_entity: Dict[str, bool],
) -> Dict[str, Any]:
    if len(entity_ids) > _SURVIVAL_INTERACTIVE_ENTITY_LIMIT:
        return {
            "status": "blocked",
            "reason": "Current export is loaded, but the cohort is above the interactive KM preview limit; continue with an audited local analysis job on this same export.",
            "mode": "kaplan_meier_aggregate",
            "scope": "exploratory_unadjusted",
            "reportable": False,
            "time_unit": "days",
            "default_outcome": None,
            "default_group": None,
            "outcomes": [],
            "group_options": [],
            "curves": [],
            "notes": [
                f"Interactive survival is limited to {_SURVIVAL_INTERACTIVE_ENTITY_LIMIT:,} entities to keep the local UI responsive.",
                "The current export remains active; no re-import is required for an audited local analysis job.",
            ],
        }
    specs = [
        {
            "id": "hospital_death",
            "label": "Hospital mortality",
            "event_candidates": _HOSP_DEATH_COLUMNS,
            "time_candidates": _HOSP_LOS_COLUMNS,
            "time_label": "Hospital LOS / follow-up days",
            "display_horizon_days": _SURVIVAL_DEFAULT_WINDOW_DAYS,
            "window_label": "30-day display window",
        },
        {
            "id": "icu_death",
            "label": "ICU mortality",
            "event_candidates": _ICU_DEATH_COLUMNS,
            "time_candidates": _ICU_LOS_COLUMNS,
            "time_label": "ICU LOS / follow-up days",
            "display_horizon_days": _SURVIVAL_DEFAULT_WINDOW_DAYS,
            "window_label": "30-day display window",
        },
        {
            "id": "mort_28d",
            "label": "28-day mortality",
            "event_candidates": _MORT28_COLUMNS,
            "time_candidates": _MORT28_TIME_COLUMNS,
            "time_label": "Days to 28-day death/censoring",
            "display_horizon_days": _SURVIVAL_28D_WINDOW_DAYS,
            "window_label": "28-day window",
            "fixed_horizon_event": True,
        },
    ]
    outcomes = [_survival_outcome_option(outcome, spec, entity_ids) for spec in specs]
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
        event_by_entity = (
            dataio._stay_bool(outcome, event_col, missing_false=False)
            if event_col
            else {}
        )
        time_by_entity = (
            dataio._stay_numeric(outcome, time_col, "max") if time_col else {}
        )
        event_by_entity, time_by_entity = _windowed_survival_vectors(
            event_by_entity,
            time_by_entity,
            horizon_days=dataio._num(option.get("display_horizon_days")),
            fixed_horizon_event=bool(option.get("fixed_horizon_event")),
        )
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
            reason = (
                "No survival curve could be computed from the available timed records."
            )

    default_outcome = next(
        (row["id"] for row in ready_outcomes if row["id"] == "mort_28d"), None
    )
    default_outcome = default_outcome or next(
        (row["id"] for row in ready_outcomes if row["id"] == "hospital_death"), None
    )
    default_outcome = default_outcome or (
        ready_outcomes[0]["id"] if ready_outcomes else None
    )
    default_group = next(
        (row["id"] for row in ready_groups if row["id"] == "sepsis"), None
    )
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
            "Hospital mortality is displayed on a 30-day visualization window by default; events after the window are censored at the window boundary.",
            "28-day mortality requires a dedicated fixed-horizon event flag and follow-up/death time; hospital mortality and LOS are not sufficient.",
            "Log-rank is unadjusted and exploratory; manuscript use still needs the evidence-bound agent gate.",
        ],
    }


def _survival_outcome_option(
    outcome: Any, spec: Dict[str, Any], entity_ids: List[str]
) -> Dict[str, Any]:
    base = {
        "id": spec["id"],
        "label": spec["label"],
        "status": "blocked",
        "event_column": None,
        "time_column": None,
        "time_label": spec["time_label"],
        "usable_entities": 0,
        "event_count": 0,
        "event_summary": {
            "status": "missing",
            "reason": "Outcome module is not present in the registered export.",
        },
        "fixed_horizon_event": bool(spec.get("fixed_horizon_event")),
    }
    if outcome is None or getattr(outcome, "empty", True):
        return {
            **base,
            "reason": "Outcome module is not present in the registered export.",
        }

    event_col = _first_column(outcome, spec["event_candidates"])
    if not event_col:
        if spec["id"] == "icu_death":
            return {
                **base,
                "reason": "ICU mortality is unavailable because this export does not include an ICU-specific event column.",
                "event_summary": {
                    "status": "missing",
                    "reason": "ICU-specific event column is not present in the registered export.",
                },
                "expected_event_columns": list(spec["event_candidates"]),
                "expected_time_columns": list(spec["time_candidates"]),
            }
        return {
            **base,
            "reason": f"No event column found for {spec['label']}.",
            "event_summary": {
                "status": "missing",
                "reason": f"No event column found for {spec['label']}.",
            },
            "expected_event_columns": list(spec["event_candidates"]),
        }
    time_col = _first_column(outcome, spec["time_candidates"])
    event_summary = _survival_event_summary(
        outcome,
        entity_ids,
        event_col=event_col,
        time_col=time_col,
        spec=spec,
    )
    if not time_col:
        if spec["id"] == "icu_death":
            return {
                **base,
                "event_column": event_col,
                "event_summary": event_summary,
                "reason": "ICU mortality event rate is available, but KM/log-rank needs ICU-specific time columns.",
                "expected_time_columns": list(spec["time_candidates"]),
            }
        return {
            **base,
            "event_column": event_col,
            "event_summary": event_summary,
            "reason": f"{spec['label']} is available only as an event flag; KM/log-rank needs time-to-event or censoring time.",
            "expected_time_columns": list(spec["time_candidates"]),
        }

    event_by_entity = dataio._stay_bool(outcome, event_col, missing_false=False)
    time_by_entity = dataio._stay_numeric(outcome, time_col, "max")
    event_by_entity, time_by_entity = _windowed_survival_vectors(
        event_by_entity,
        time_by_entity,
        horizon_days=dataio._num(spec.get("display_horizon_days")),
        fixed_horizon_event=bool(spec.get("fixed_horizon_event")),
    )
    usable = [
        entity_id
        for entity_id in entity_ids
        if entity_id in event_by_entity
        and dataio._num(time_by_entity.get(entity_id)) is not None
        and float(time_by_entity[entity_id]) >= 0
    ]
    event_count = sum(
        1 for entity_id in usable if event_by_entity.get(entity_id) is True
    )
    if len(usable) < 2:
        return {
            **base,
            "event_column": event_col,
            "time_column": time_col,
            "event_summary": event_summary,
            "usable_entities": len(usable),
            "event_count": event_count,
            "reason": "Fewer than two cohort entities have both an observed event flag and a valid survival time.",
        }
    return {
        **base,
        "status": "ready",
        "reason": None,
        "event_column": event_col,
        "time_column": time_col,
        "event_summary": event_summary,
        "usable_entities": len(usable),
        "event_count": event_count,
        "display_horizon_days": dataio._num(spec.get("display_horizon_days")),
        "window_label": spec.get("window_label"),
    }


def _survival_event_summary(
    outcome: Any,
    entity_ids: List[str],
    *,
    event_col: str,
    time_col: str | None,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    event_by_entity = dataio._stay_bool(outcome, event_col, missing_false=False)
    excluded_inconsistent = 0
    if time_col and bool(spec.get("fixed_horizon_event")):
        time_by_entity = dataio._stay_numeric(outcome, time_col, "max")
        horizon_days = dataio._num(spec.get("display_horizon_days"))
        excluded_inconsistent = _fixed_horizon_inconsistency_count(
            event_by_entity,
            time_by_entity,
            horizon_days=horizon_days,
        )
        event_by_entity, _ = _windowed_survival_vectors(
            event_by_entity,
            time_by_entity,
            horizon_days=horizon_days,
            fixed_horizon_event=True,
        )
    denominator_ids = [
        entity_id for entity_id in entity_ids if entity_id in event_by_entity
    ]
    basis = (
        "fixed_horizon_event_and_followup"
        if bool(spec.get("fixed_horizon_event")) and time_col
        else "event_flag"
    )
    fixed_horizon = bool(spec.get("fixed_horizon_event"))
    time_label = spec.get("window_label") if time_col and fixed_horizon else None
    time_column = time_col if bool(spec.get("fixed_horizon_event")) else None
    denominator = len(denominator_ids)
    event_count = sum(
        1 for entity_id in denominator_ids if event_by_entity.get(entity_id) is True
    )
    pct = round(event_count / denominator * 100, 1) if denominator else None
    return {
        "status": "available" if denominator else "missing",
        "basis": basis,
        "event_column": event_col,
        "time_column": time_column,
        "time_window_label": time_label,
        "denominator": denominator,
        "event_count": event_count,
        "event_rate_pct": pct,
        **(
            {"excluded_inconsistent_entities": excluded_inconsistent}
            if bool(spec.get("fixed_horizon_event")) and time_col
            else {}
        ),
    }


def _fixed_horizon_inconsistency_count(
    event_by_entity: Dict[str, bool],
    time_by_entity: Dict[str, float],
    *,
    horizon_days: float | None,
) -> int:
    if horizon_days is None or horizon_days <= 0:
        return 0
    inconsistent = 0
    for entity_id, event in event_by_entity.items():
        time_value = dataio._num(time_by_entity.get(entity_id))
        if time_value is None or time_value < 0:
            continue
        if (event is True and time_value > horizon_days) or (
            event is not True and time_value < horizon_days
        ):
            inconsistent += 1
    return inconsistent


def _windowed_survival_vectors(
    event_by_entity: Dict[str, bool],
    time_by_entity: Dict[str, float],
    *,
    horizon_days: float | None,
    fixed_horizon_event: bool = False,
) -> Tuple[Dict[str, bool], Dict[str, float]]:
    windowed_events: Dict[str, bool] = {}
    windowed_times: Dict[str, float] = {}
    for entity_id, raw_time in time_by_entity.items():
        if entity_id not in event_by_entity:
            continue
        time_value = dataio._num(raw_time)
        if time_value is None or time_value < 0:
            continue
        if horizon_days is None or horizon_days <= 0:
            windowed_events[entity_id] = event_by_entity[entity_id] is True
            windowed_times[entity_id] = float(time_value)
            continue
        if fixed_horizon_event and (
            (event_by_entity[entity_id] is True and time_value > horizon_days)
            or (event_by_entity[entity_id] is not True and time_value < horizon_days)
        ):
            # A fixed-horizon flag already states whether the event occurred by
            # the horizon. A later "true" event or an earlier "false" censoring
            # time contradicts that flag; treating either as survival would
            # silently change the endpoint. Keep the entity unknown instead.
            continue
        in_window = time_value <= horizon_days
        windowed_events[entity_id] = bool(
            event_by_entity[entity_id] is True and in_window
        )
        windowed_times[entity_id] = min(float(time_value), float(horizon_days))
    return windowed_events, windowed_times


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
            "_members": _bool_member_groups(
                entity_ids,
                sepsis_by_entity,
                false_name="Non-sepsis",
                true_name="Sepsis",
            ),
        },
        {
            "id": "age",
            "label": "Age <65 vs >=65",
            "basis": "age_threshold_65_descriptive",
            "_members": [
                (
                    "<65",
                    [
                        entity_id
                        for entity_id in entity_ids
                        if entity_id in age_by_entity and age_by_entity[entity_id] < 65
                    ],
                ),
                (
                    ">=65",
                    [
                        entity_id
                        for entity_id in entity_ids
                        if entity_id in age_by_entity and age_by_entity[entity_id] >= 65
                    ],
                ),
            ],
        },
        {
            "id": "sex",
            "label": "Female vs Male",
            "basis": "sex_metadata_descriptive",
            "_members": [
                (
                    "Female",
                    [
                        entity_id
                        for entity_id in entity_ids
                        if _sex_bucket(sex_by_entity.get(entity_id)) == "female"
                    ],
                ),
                (
                    "Male",
                    [
                        entity_id
                        for entity_id in entity_ids
                        if _sex_bucket(sex_by_entity.get(entity_id)) == "male"
                    ],
                ),
            ],
        },
        _survival_sofa_group(entity_ids, sofa_by_entity),
    ]
    out: List[Dict[str, Any]] = []
    for row in raw_groups:
        if not row:
            continue
        members = [
            (label, ids) for label, ids in row.get("_members", []) if label != "Unknown"
        ]
        nonempty = [(label, ids) for label, ids in members if ids]
        status = "ready" if len(nonempty) >= 2 else "blocked"
        out.append(
            {
                **row,
                "status": status,
                "reason": (
                    None
                    if status == "ready"
                    else "This split does not produce two non-empty groups in the current cohort."
                ),
                "groups": [
                    {"label": label, "count": len(ids)} for label, ids in members
                ],
                "_members": members,
            }
        )
    return out


def _survival_sofa_group(
    entity_ids: List[str], sofa_by_entity: Dict[str, float]
) -> Dict[str, Any] | None:
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
            (
                f"SOFA-2 <= {threshold:g}",
                [
                    entity_id
                    for entity_id in entity_ids
                    if entity_id in sofa_by_entity
                    and sofa_by_entity[entity_id] <= threshold
                ],
            ),
            (
                f"SOFA-2 > {threshold:g}",
                [
                    entity_id
                    for entity_id in entity_ids
                    if entity_id in sofa_by_entity
                    and sofa_by_entity[entity_id] > threshold
                ],
            ),
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
            if entity_id not in event_by_entity:
                continue
            time_value = dataio._num(time_by_entity.get(entity_id))
            if time_value is None or time_value < 0:
                continue
            records.append(
                (float(time_value), bool(event_by_entity.get(entity_id) is True))
            )
        if records:
            km = _km_group_payload(label, records)
            group_records.append({"label": label, "records": records, "payload": km})
    if len(group_records) < 2:
        return None

    risk_times = _risk_times(
        [record for row in group_records for record in row["records"]]
    )
    logrank = _logrank_payload(
        group_records[0]["records"],
        group_records[1]["records"],
        group_records[0]["label"],
        group_records[1]["label"],
    )
    return {
        "outcome_id": outcome_option.get("id"),
        "group_id": group_option.get("id"),
        "status": "ready",
        "label": f"{outcome_option.get('label')} by {group_option.get('label')}",
        "event_column": outcome_option.get("event_column"),
        "time_column": outcome_option.get("time_column"),
        "time_label": outcome_option.get("time_label"),
        "time_unit": "days",
        "display_horizon_days": outcome_option.get("display_horizon_days"),
        "window_label": outcome_option.get("window_label"),
        "derived_from": outcome_option.get("derived_from"),
        "scope": "exploratory_unadjusted",
        "reportable": False,
        "groups": [row["payload"] for row in group_records],
        "logrank": logrank,
        "number_at_risk": {
            "times": risk_times,
            "rows": [
                {
                    "label": row["label"],
                    "values": [
                        sum(
                            1
                            for time_value, _event in row["records"]
                            if time_value >= t
                        )
                        for t in risk_times
                    ],
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
        # Marks are computed from the unthinned curve so a tick sits on the
        # step the subject actually left from.
        "censor_marks": _censor_marks(records, points),
    }


def _censor_marks(
    records: List[Tuple[float, bool]],
    points: List[Dict[str, Any]],
    max_marks: int = 40,
) -> List[Dict[str, Any]]:
    """Times where a subject left follow-up without the event, placed on the curve.

    A Kaplan-Meier curve without censoring marks hides how much of a flat
    stretch is real follow-up and how much is attrition, so the marks are part
    of reading the curve rather than decoration. ``_km_points`` only emits a
    point at event times, so the censoring times are not otherwise recoverable
    from the payload.

    Each mark carries the survival value of the step it belongs to — the last
    point at or before the censoring time — so the renderer places ticks
    without re-deriving the step function. Disclosure is the same class as the
    ``points`` and ``number_at_risk`` this payload already returns: aggregate
    times and counts, never a subject row.
    """

    censor_times = sorted(
        {_round_time(time_value) for time_value, event in records if not event}
    )
    if not censor_times or not points:
        return []

    marks: List[Dict[str, Any]] = []
    index = 0
    survival = points[0].get("survival", 100.0)
    for time_value in censor_times:
        while index < len(points) and points[index]["time"] <= time_value:
            survival = points[index].get("survival", survival)
            index += 1
        marks.append({"time": time_value, "survival": survival})

    if len(marks) <= max_marks:
        return marks
    step = (len(marks) - 1) / (max_marks - 1)
    keep = sorted({0, len(marks) - 1} | {round(i * step) for i in range(max_marks)})
    return [marks[i] for i in keep]


def _km_points(records: List[Tuple[float, bool]]) -> List[Dict[str, Any]]:
    if not records:
        return [{"time": 0, "survival": 100.0, "at_risk": 0, "events": 0}]
    total_by_time: Counter[float] = Counter()
    events_by_time: Counter[float] = Counter()
    for time_value, event in records:
        total_by_time[time_value] += 1
        if event:
            events_by_time[time_value] += 1
    event_times = sorted(events_by_time)
    max_time = max(total_by_time)
    survival = 1.0
    points = [{"time": 0, "survival": 100.0, "at_risk": len(records), "events": 0}]
    at_risk_by_time: Dict[float, int] = {}
    running = 0
    for time_value in sorted(total_by_time, reverse=True):
        running += total_by_time[time_value]
        at_risk_by_time[time_value] = running
    for time_value in event_times:
        at_risk = at_risk_by_time.get(time_value, 0)
        events = events_by_time[time_value]
        if at_risk > 0:
            survival *= max(0.0, 1.0 - events / at_risk)
        points.append(
            {
                "time": _round_time(time_value),
                "survival": round(survival * 100, 1),
                "at_risk": at_risk,
                "events": events,
            }
        )
    if max_time and (not points or points[-1]["time"] != _round_time(max_time)):
        points.append(
            {
                "time": _round_time(max_time),
                "survival": points[-1]["survival"],
                "at_risk": at_risk_by_time.get(max_time, 0),
                "events": 0,
            }
        )
    return points


def _km_median(points: List[Dict[str, Any]]) -> float | None:
    for point in points:
        survival = dataio._num(point.get("survival"))
        if survival is not None and survival <= 50:
            return dataio._num(point.get("time"))
    return None


def _thin_points(
    points: List[Dict[str, Any]], max_points: int = 80
) -> List[Dict[str, Any]]:
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
    base = [0, 1, 3, 7, 14, 28, 30]
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
    total_a: Counter[float] = Counter()
    total_b: Counter[float] = Counter()
    events_a: Counter[float] = Counter()
    events_b: Counter[float] = Counter()
    for time_value, event in records_a:
        total_a[time_value] += 1
        if event:
            events_a[time_value] += 1
    for time_value, event in records_b:
        total_b[time_value] += 1
        if event:
            events_b[time_value] += 1
    event_times = sorted(set(events_a) | set(events_b))
    risk_a = _risk_count_map(total_a, event_times)
    risk_b = _risk_count_map(total_b, event_times)
    observed_a = expected_a = variance_a = 0.0
    total_events = 0
    for time_value in event_times:
        n_a = risk_a.get(time_value, 0)
        n_b = risk_b.get(time_value, 0)
        d_a = events_a.get(time_value, 0)
        d_b = events_b.get(time_value, 0)
        n_total = n_a + n_b
        d_total = d_a + d_b
        if n_total <= 1 or d_total <= 0:
            continue
        observed_a += d_a
        expected_a += d_total * (n_a / n_total)
        variance_a += (n_a * n_b * d_total * (n_total - d_total)) / (
            (n_total**2) * (n_total - 1)
        )
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
        "p_value": p_value,
        "p_value_label": _scientific_p_value_label(p_value),
        "interpretation": "exploratory_unadjusted_not_reportable",
    }


def _scientific_p_value_label(value: float | None) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    if value == 0:
        return "0"
    if value < 0:
        return None
    if value >= 0.001:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    exponent = math.floor(math.log10(value))
    mantissa = value / (10**exponent)
    return f"{mantissa:.4g} × 10^{exponent}"


def _risk_count_map(
    total_by_time: Counter[float], event_times: List[float]
) -> Dict[float, int]:
    """Return n-at-risk for arbitrary event times without scanning records repeatedly."""
    observed_times = sorted(total_by_time, reverse=True)
    out: Dict[float, int] = {}
    running = 0
    idx = 0
    for event_time in sorted(event_times, reverse=True):
        while idx < len(observed_times) and observed_times[idx] >= event_time:
            running += total_by_time[observed_times[idx]]
            idx += 1
        out[event_time] = running
    return out


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
    exact_matrix = _sofa_exact_transition_matrix(pairs)
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
        "exact_score_bins": [str(value) for value in range(25)],
        "exact_score_matrix": exact_matrix,
        "score_scale": {
            "min": 0,
            "max": 24,
            "unit": "SOFA points",
            "aggregation": "nearest_integer_clamped_0_24",
        },
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
    counts: Counter[Tuple[str, str]] = Counter()
    row_totals: Counter[str] = Counter()
    for sofa1, sofa2 in pairs:
        source_label = _sofa_severity_label(sofa1)
        target_label = _sofa_severity_label(sofa2)
        if source_label not in labels or target_label not in labels:
            continue
        counts[(source_label, target_label)] += 1
        row_totals[source_label] += 1
    rows: List[Dict[str, Any]] = []
    for source_label in labels:
        cells = [
            {
                "label": target_label,
                "count": counts[(source_label, target_label)],
                "pct": _pct(counts[(source_label, target_label)], total),
            }
            for target_label in labels
        ]
        row_total = row_totals[source_label]
        rows.append({"label": source_label, "count": row_total, "cells": cells})
    return rows


def _sofa_exact_transition_matrix(pairs: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    total = len(pairs)
    counts: Counter[Tuple[int, int]] = Counter()
    row_totals: Counter[int] = Counter()
    for sofa1, sofa2 in pairs:
        source_score = _sofa_score_index(sofa1)
        target_score = _sofa_score_index(sofa2)
        if source_score is None or target_score is None:
            continue
        counts[(source_score, target_score)] += 1
        row_totals[source_score] += 1
    rows: List[Dict[str, Any]] = []
    for source_score in range(25):
        cells = [
            {
                "label": str(target_score),
                "count": counts[(source_score, target_score)],
                "pct": _pct(counts[(source_score, target_score)], total),
            }
            for target_score in range(25)
        ]
        rows.append(
            {
                "label": str(source_score),
                "count": row_totals[source_score],
                "cells": cells,
            }
        )
    return rows


def _sofa_score_index(value: Any) -> int | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return max(0, min(24, int(round(score))))


def _sofa_severity_label(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(score):
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
