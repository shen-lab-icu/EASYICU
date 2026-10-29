"""Audited preparation primitives and legacy deterministic missing-data strategies."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..contracts.model_terms import PreparationRefusal

MissingStrategy = Literal["complete_case", "mean_imputation", "median_imputation"]


def apply_missing_strategy(df, strategy: MissingStrategy | str):
    """Apply a narrow, deterministic missing-data strategy to a dataframe."""

    strategy = str(strategy or "complete_case")
    if strategy == "complete_case":
        return df.dropna().copy()
    if strategy not in {"mean_imputation", "median_imputation"}:
        raise ValueError(f"unsupported missing strategy: {strategy}")
    out = df.copy()
    numeric_columns = list(out.select_dtypes(include="number").columns)
    for column in numeric_columns:
        if strategy == "mean_imputation":
            fill_value = out[column].mean()
        else:
            fill_value = out[column].median()
        out[column] = out[column].fillna(fill_value)
    return out.dropna().copy()


PUBLIC_DATABASES = ("mimic", "miiv", "eicu", "aumc", "hirid", "sic")


def source_has_raw_values(status, database):
    """Only the selected source may authorize a raw-value operation."""
    from easyicu.databases.profiles import normalize_database_key

    try:
        key = normalize_database_key(database)
    except KeyError:
        return False
    return bool(status.get(key, {}).get("raw_values_available", False))


def database_preparation_status(provenance, concept_id):
    """Report only captured dictionary facts, separately for all six databases.

    A code-defined concept or derived score has no numeric raw-value authority.
    Absent snapshots remain unavailable, rather than inheriting another DB's
    units or conversion rules. These are rules applied by extraction, not new
    conversions to apply a second time to an analysis cohort.
    """
    result = {
        db: {
            "status": "not_recorded",
            "raw_values_available": False,
            "time_values_available": False,
            "unit_conversions": [],
            "canonical_unit": None,
        }
        for db in PUBLIC_DATABASES
    }
    if provenance is None:
        return result
    for snapshot in provenance.snapshots:
        if snapshot.source_database not in result:
            continue
        concepts = [item for item in snapshot.concepts if item.concept_id == concept_id]
        if not concepts:
            continue
        # A snapshot is a captured extraction: merge repeated entries rather
        # than granting raw-value support from a derived-only definition.
        sources = [source for item in concepts for source in item.sources]
        numeric = [
            source for source in sources if source.value_variable and not source.codes
        ]
        result[snapshot.source_database] = {
            "status": "recorded"
            if all(item.status == "recorded" for item in concepts)
            else "not_recorded",
            "raw_values_available": bool(numeric)
            and not any(item.derivation_callback for item in concepts),
            "time_values_available": any(source.time_variable for source in numeric),
            "canonical_unit": concepts[0].canonical_unit,
            "unit_conversions": [
                rule.model_dump(mode="json")
                for source in sources
                for rule in source.unit_conversions
            ],
        }
    return result


def _require_raw(frame, variable, raw_variables):
    if variable not in frame or variable not in raw_variables:
        raise PreparationRefusal(
            "preparation_raw_values_unavailable",
            "This database/input does not supply raw values for the selected concept",
        )


def apply_outliers(frame, rules, *, raw_variables):
    result, receipts = frame.copy(), []
    for rule in rules:
        _require_raw(result, rule.variable, raw_variables)
        values = pd.to_numeric(result[rule.variable], errors="coerce")
        if (result[rule.variable].notna() & values.isna()).any():
            raise PreparationRefusal(
                "preparation_variable_not_numeric",
                "An outlier rule requires numeric values",
            )
        outside = values.notna() & ~values.between(rule.lower, rule.upper)
        count = int(outside.sum())
        result[rule.variable] = (
            values.mask(outside)
            if rule.rule == "range_filter"
            else values.clip(rule.lower, rule.upper)
        )
        receipts.append(
            {
                **rule.model_dump(),
                "removed": count if rule.rule == "range_filter" else 0,
                "capped": count if rule.rule == "winsorize" else 0,
                "observed_before": int(values.notna().sum()),
            }
        )
    return result, receipts


def aggregate_windows(frame, windows, *, raw_variables):
    if not windows:
        return frame.copy(), []
    groups = {item.group_column for item in windows}
    if len(groups) != 1:
        raise PreparationRefusal(
            "preparation_window_identity_conflict",
            "Window summaries must use the same analysis unit",
        )
    group = next(iter(groups))
    needed = {group, *(item.time_column for item in windows)}
    if not needed.issubset(frame) or frame[group].isna().any():
        raise PreparationRefusal(
            "preparation_time_values_unavailable",
            "Raw times and a complete analysis-unit identity are required",
        )
    varying = {item.variable for item in windows} | {
        item.time_column for item in windows
    }
    fixed = [name for name in frame if name not in varying and name != group]
    if (
        fixed
        and frame.groupby(group, sort=False)[fixed]
        .nunique(dropna=False)
        .gt(1)
        .any()
        .any()
    ):
        raise PreparationRefusal(
            "preparation_unaggregated_values_vary",
            "Every varying repeated measurement needs a window rule",
        )
    result = frame[[group, *fixed]].drop_duplicates(group).set_index(group)
    receipts = []
    for window in windows:
        _require_raw(frame, window.variable, raw_variables)
        times = pd.to_numeric(frame[window.time_column], errors="coerce")
        if times.isna().any() or not np.isfinite(times).all():
            raise PreparationRefusal(
                "preparation_time_invalid", "Aggregation requires finite times in hours"
            )
        selected = frame.loc[times.ge(window.start_hours) & times.lt(window.end_hours)]
        selected = selected.sort_values(window.time_column, kind="stable")
        grouped = selected.groupby(group, sort=False)[window.variable]
        # first/last refer to observed measurements in temporal order.
        result[window.variable] = grouped.agg(window.function)
        receipts.append(
            {
                **window.model_dump(),
                "boundary": "start_inclusive_end_exclusive",
                "measurements_used": int(selected[window.variable].notna().sum()),
                "units_without_observation": int(result[window.variable].isna().sum()),
            }
        )
    return result.reset_index(), receipts


def standardize_columns(training, scoring, choices):
    train, score, receipts = training.copy(), scoring.copy(), []
    for choice in choices:
        if not choice.standardize:
            continue
        if choice.variable not in train or choice.variable not in score:
            raise PreparationRefusal(
                "preparation_column_missing", "A selected scaling variable is absent"
            )
        values = pd.to_numeric(train[choice.variable], errors="raise")
        mean, scale = float(values.mean()), float(values.std(ddof=0))
        if not np.isfinite([mean, scale]).all() or scale <= 0:
            raise PreparationRefusal(
                "preparation_scale_not_estimable",
                "Standardization requires positive observed training variance",
            )
        train[choice.variable] = (values - mean) / scale
        score[choice.variable] = (
            pd.to_numeric(score[choice.variable], errors="raise") - mean
        ) / scale
        receipts.append(
            {
                "variable": choice.variable,
                "method": "standardization",
                "mean": mean,
                "scale": scale,
                "fit_n": int(values.notna().sum()),
            }
        )
    return train, score, receipts


__all__ = [
    "MissingStrategy",
    "apply_missing_strategy",
    "apply_outliers",
    "aggregate_windows",
    "database_preparation_status",
    "standardize_columns",
]
