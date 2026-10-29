"""Deterministic descriptive SMDs for an explicit treatment/weight design."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..contracts.covariate_balance import (
    CovariateBalanceResult,
    CovariateBalanceRow,
    CovariateBalanceSpec,
    covariate_balance_spec_sha256,
)
from ..contracts.model_terms import level_identity_class, level_spelling
from .propensity_weighting import standardized_mean_difference


def _numeric(series: pd.Series, name: str, *, missing: bool) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if bool(invalid.any()) or bool(np.isinf(values).any()):
        raise ValueError(f"covariate_balance_invalid_numeric: {name}")
    if not missing and bool(np.isnan(values).any()):
        raise ValueError(f"covariate_balance_missing_numeric: {name}")
    return values


def _moments(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    positive = weights > 0
    n = int(values.size)
    weighted_n = int(positive.sum())
    # Scaling avoids overflow in ESS and makes it invariant to weight scale.
    scaled = weights[positive] / weights[positive].max() if weighted_n else np.array([])
    total = float(weights.sum())
    if not math.isfinite(total):
        raise ValueError("covariate_balance_weight_sum_nonfinite")
    return {
        "n": n,
        "positive": weighted_n,
        "zero": n - weighted_n,
        "weight_sum": total,
        "ess": float(scaled.sum() ** 2 / np.square(scaled).sum())
        if weighted_n
        else 0.0,
        "mean": float(np.mean(values)) if n else None,
        "variance": float(np.var(values, ddof=1)) if n > 1 else None,
        "weighted_mean": float(np.average(values[positive], weights=scaled))
        if weighted_n
        else None,
    }


def _contrast(
    mean_t: float | None, mean_c: float | None, denominator: float | None
) -> tuple[float | None, str]:
    if mean_t is None or mean_c is None or denominator is None:
        return None, "insufficient_observations"
    if denominator == 0:
        return (
            (0.0, "constant_equal")
            if mean_t == mean_c
            else (None, "zero_variance_different_means")
        )
    return (float((mean_t - mean_c) / denominator), "estimated")


def compute_covariate_balance(
    frame: pd.DataFrame, spec: CovariateBalanceSpec | Mapping[str, Any]
) -> CovariateBalanceResult:
    """One unique declared unit per row; no refitting or row deduplication."""
    spec = CovariateBalanceSpec.model_validate(spec)
    required = {
        spec.unit_id_column,
        spec.treatment_column,
        *(item.column for item in spec.covariates),
    }
    if spec.weight_column is not None:
        required.add(spec.weight_column)
    if not frame.columns.is_unique or not required.issubset(frame.columns):
        raise ValueError("covariate_balance_missing_or_duplicate_columns")
    if frame.empty:
        raise ValueError("covariate_balance_empty_input")
    ids = frame[spec.unit_id_column]
    if (
        ids.isna().any()
        or ids.astype(str).str.strip().eq("").any()
        or ids.duplicated().any()
        or ids.astype(str).duplicated().any()
    ):
        raise ValueError("covariate_balance_analysis_unit_not_unique")
    treatment = _numeric(
        frame[spec.treatment_column], spec.treatment_column, missing=False
    )
    if set(treatment.tolist()) != {0.0, 1.0}:
        raise ValueError("covariate_balance_treatment_requires_both_binary_groups")
    weights = np.ones(len(frame))
    if spec.weight_column is not None:
        weights = _numeric(frame[spec.weight_column], spec.weight_column, missing=False)
        if bool((weights < 0).any()):
            raise ValueError(
                f"covariate_balance_negative_weights: n={int((weights < 0).sum())}"
            )
        if any(not bool((weights[treatment == group] > 0).any()) for group in (0, 1)):
            raise ValueError("covariate_balance_empty_weighted_group")
    group_counts = {group: int((treatment == group).sum()) for group in (0, 1)}
    rows: list[CovariateBalanceRow] = []
    for covariate in spec.covariates:
        source = frame[covariate.column]
        if covariate.kind == "categorical":
            observed = source.notna().to_numpy()
            spellings: dict[str, str] = {}
            for value in source.dropna().tolist():
                spelling, identity_class = (
                    level_spelling(value),
                    level_identity_class(value),
                )
                if isinstance(value, str) and value != value.strip():
                    raise ValueError(
                        f"covariate_balance_noncanonical_level: {covariate.column}"
                    )
                if spelling in spellings and spellings[spelling] != identity_class:
                    raise ValueError(
                        f"covariate_balance_ambiguous_level_types: {covariate.column}"
                    )
                spellings[spelling] = identity_class
            labels = source.map(
                lambda value: level_spelling(value) if pd.notna(value) else None
            )
            if not set(labels.dropna()).issubset(set(covariate.levels)):
                raise ValueError(
                    f"covariate_balance_undeclared_levels: {covariate.column}"
                )
            components = [
                (level, labels.eq(level).to_numpy(dtype=float))
                for level in covariate.levels
            ]
        else:
            values = _numeric(source, covariate.column, missing=True)
            observed = ~np.isnan(values)
            if covariate.kind == "binary" and not set(values[observed]).issubset(
                {0.0, 1.0}
            ):
                raise ValueError(
                    f"covariate_balance_binary_invalid: {covariate.column}"
                )
            components = [(None, values)]
        for level, values in components:
            masks = {group: observed & (treatment == group) for group in (0, 1)}
            moments = {
                group: _moments(values[masks[group]], weights[masks[group]])
                for group in (0, 1)
            }
            t, c = moments[1], moments[0]
            denominator = (
                math.sqrt((t["variance"] + c["variance"]) / 2)
                if t["variance"] is not None and c["variance"] is not None
                else None
            )
            _, before_status = _contrast(t["mean"], c["mean"], denominator)
            before = standardized_mean_difference(values[masks[1]], values[masks[0]])
            after, after_status = _contrast(
                t["weighted_mean"], c["weighted_mean"], denominator
            )
            if spec.weight_column is None:
                after, after_status = None, "not_requested"
            elif not t["positive"] or not c["positive"]:
                after, after_status = None, "empty_weighted_group"
            threshold = spec.abs_smd_threshold
            rows.append(
                CovariateBalanceRow(
                    row_role="category_indicator"
                    if covariate.kind == "categorical"
                    else "covariate",
                    covariate=covariate.column,
                    kind=covariate.kind,
                    level=level,
                    n_treated=t["n"],
                    n_control=c["n"],
                    missing_treated_n=group_counts[1] - t["n"],
                    missing_control_n=group_counts[0] - c["n"],
                    positive_weight_treated_n=t["positive"],
                    positive_weight_control_n=c["positive"],
                    zero_weight_treated_n=t["zero"],
                    zero_weight_control_n=c["zero"],
                    weight_sum_treated=t["weight_sum"],
                    weight_sum_control=c["weight_sum"],
                    ess_treated=t["ess"],
                    ess_control=c["ess"],
                    mean_treated_before=t["mean"],
                    mean_control_before=c["mean"],
                    variance_treated_before=t["variance"],
                    variance_control_before=c["variance"],
                    smd_denominator=denominator,
                    smd_before=before,
                    smd_after=after,
                    mean_treated_after=t["weighted_mean"]
                    if spec.weight_column
                    else None,
                    mean_control_after=c["weighted_mean"]
                    if spec.weight_column
                    else None,
                    status_before=before_status,
                    status_after=after_status,
                    exceeds_threshold_before=abs(before) > threshold
                    if before is not None and threshold is not None
                    else None,
                    exceeds_threshold_after=abs(after) > threshold
                    if after is not None and threshold is not None
                    else None,
                )
            )
    return CovariateBalanceResult(
        specification_sha256=covariate_balance_spec_sha256(spec),
        analysis_unit=spec.analysis_unit,
        input_n=len(frame),
        treated_n=group_counts[1],
        control_n=group_counts[0],
        zero_weight_n=int((weights == 0).sum()),
        rows=tuple(rows),
    )
