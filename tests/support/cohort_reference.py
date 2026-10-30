"""Frozen pre-optimization readers; callable only by synthetic parity tests.

The payload assembler stays production-owned; these original reader functions
provide an independent pandas oracle for every replaced Parquet reduction.
"""

from __future__ import annotations
from pathlib import Path
from collections import Counter
import math
from typing import Any, Dict, Iterable, List, Tuple
from easyicu.webserver import (
    cohort_review as owner,
    prepared_frames,
    entity_ids as entity_id_contract,
)
from easyicu.webserver import dataio

from easyicu.webserver.cohort_review import (
    _MODULE_COLUMNS,
    _skip_expensive_whole_module_read,
    _COVERAGE_UNIQUE_STAY_SCAN_ROW_LIMIT,
    _read_selected_columns,
    _filter_by_entity,
    _selected_feature_profile,
    _MAX_COMPARE_FEATURES,
    _pct,
    _group_from_bool,
    _group_from_age,
    _group_from_sex,
    _group_from_los,
    _group_members,
    _bool_pct,
    _female_pct,
    _median_for,
    _km_median,
    _thin_points,
    _round_time,
    _risk_count_map,
    _SURVIVAL_INTERACTIVE_ENTITY_LIMIT,
    _HOSP_DEATH_COLUMNS,
    _HOSP_LOS_COLUMNS,
    _SURVIVAL_DEFAULT_WINDOW_DAYS,
    _ICU_DEATH_COLUMNS,
    _ICU_LOS_COLUMNS,
    _MORT28_COLUMNS,
    _MORT28_TIME_COLUMNS,
    _SURVIVAL_28D_WINDOW_DAYS,
    _survival_group_options,
    _scientific_p_value_label,
    _first_column,
    _quality_status,
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


def _covered_entities(
    path: Path, item: Dict[str, Any], cohort_ids: set[str] | None
) -> int | None:
    if not cohort_ids:
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
            frame = entity_id_contract.canonicalize_entity_frame(frame, entity_column)
            module_ids = {
                value
                for value in frame["stay_id"]
                .map(entity_id_contract.normalize_entity_id)
                .dropna()
                .astype(str)
                if value
            }
            return len(module_ids & cohort_ids)
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
        module_ids = {
            value
            for value in frame["stay_id"]
            .map(entity_id_contract.normalize_entity_id)
            .dropna()
            .astype(str)
            if value
        }
        return len(module_ids & cohort_ids)
    except Exception:
        return None


def _selected_feature_profiles(
    path: Path,
    desc: Dict[str, Any],
    entity_set: set[str],
    catalog: Dict[str, Any],
    selected_feature_ids: List[str],
    *,
    coverage_counts=None,
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
            frame = _filter_by_entity(feature_frame, entity_set)
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
                        {
                            "label": "Unknown",
                            "count": max(0, total - positive - negative),
                        },
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
    rows: List[Dict[str, Any]] = [
        {
            "metric": "N",
            "kind": "count",
            "values": [len(members) for members in member_sets],
        }
    ]
    # Core profile rows are projections of actual source mappings.  Do not
    # manufacture a 0% row when the corresponding module is absent: a
    # run-scoped package may intentionally carry a different selected exposure
    # (for example sep3_sofa1_max) without the legacy sepsis3_sofa2 module.
    core_rows = (
        (
            bool(death_by_entity),
            "Mortality %",
            "percent",
            [_bool_pct(members, death_by_entity, known_only=True) for members in member_sets],
            None,
        ),
        (
            bool(sex_by_entity),
            "Female %",
            "percent",
            [_female_pct(members, sex_by_entity) for members in member_sets],
            None,
        ),
        (
            bool(age_by_entity),
            "Median age",
            "numeric",
            [_median_for(members, age_by_entity) for members in member_sets],
            "years",
        ),
        (
            bool(sofa_by_entity),
            "Median SOFA-2",
            "numeric",
            [_median_for(members, sofa_by_entity) for members in member_sets],
            None,
        ),
        (
            bool(los_by_entity),
            "Median ICU LOS",
            "numeric",
            [_median_for(members, los_by_entity) for members in member_sets],
            "days",
        ),
        (
            bool(sepsis_by_entity),
            "Sepsis-3 %",
            "percent",
            [_bool_pct(members, sepsis_by_entity) for members in member_sets],
            None,
        ),
    )
    for available, metric, kind, values, unit in core_rows:
        if not available:
            continue
        row: Dict[str, Any] = {"metric": metric, "kind": kind, "values": values}
        if unit:
            row["unit"] = unit
        rows.append(row)
    rows.extend(_selected_feature_rows(member_sets, selected_features))
    return {
        "status": "descriptive_aggregate_only",
        "columns": columns,
        "inferential_statistics_allowed": False,
        "rows": rows,
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
                    "values": [
                        _bool_pct(members, mapping, known_only=(module == "outcome" and str(feature.get("column") or "").lower() in owner._DEATH_COLUMNS))
                        for members in member_sets
                    ],
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


def _coverage_payload(
    path: Path, desc: Dict[str, Any], *, precomputed=None
) -> List[Dict[str, Any]]:
    files = desc.get("files") or []
    cohort_ids: set[str] | None = None
    cohort_ids_resolved = False

    def resolve_cohort_ids() -> set[str] | None:
        # The cohort id set costs one stay-key column read. Resolve it only
        # when a module is actually scanned so the large-module metadata path
        # keeps its no-scan contract.
        nonlocal cohort_ids, cohort_ids_resolved
        if not cohort_ids_resolved:
            cohort_ids = dataio._fast_stay_ids(path, files)
            cohort_ids_resolved = True
        return cohort_ids

    out: List[Dict[str, Any]] = []
    for item in files:
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
            covered = _covered_entities(path, item, resolve_cohort_ids())
        # ``covered`` is an intersection with the resolved cohort id set, so
        # that set is the only consistent denominator.
        cohort_size = len(cohort_ids) if cohort_ids else None
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


def _entity_numeric(
    frame: Any, column: str, *, normalized_ids=False
) -> Dict[str, float]:
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


def _entity_text(frame: Any, column: str, *, normalized_ids=False) -> Dict[str, str]:
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


def install_reference(monkeypatch):
    for name in (
        "_read_module_frame",
        "_fallback_entity_frame",
        "_covered_entities",
        "_selected_feature_profiles",
        "_group_payload",
        "_group_profile",
        "_selected_feature_rows",
        "_selected_feature_distributions",
        "_numeric_summary",
        "_sex_bucket",
        "_windowed_survival_vectors",
        "_fixed_horizon_inconsistency_count",
        "_survival_curve_payload",
        "_survival_analysis_payload",
        "_km_group_payload",
        "_censor_marks",
        "_km_points",
        "_risk_times",
        "_logrank_payload",
        "_survival_outcome_option",
        "_survival_event_summary",
    ):
        monkeypatch.setattr(owner, name, globals()[name])

    for name in ("_coverage_payload", "_entity_numeric", "_entity_text"):
        monkeypatch.setattr(owner, name, globals()[name])
