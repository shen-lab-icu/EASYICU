"""Typed publication projections for exposure/outcome distribution evidence.

The deterministic distribution executor publishes two different scientific
quantities in one closed table: one prespecified risk-difference contrast and
level-specific absolute outcome risks.  A figure must project those quantities
by their typed column names.  Structural columns such as ``exposure_level`` and
``exposure_level_index`` are labels/keys, never candidate effect estimates.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import pandas as pd

from .display_labels import (
    binary_contrast_label,
    binary_scope_label,
    display_label,
    scoped_label_lookup,
)


_DISTRIBUTION_IDENTITY_COLUMNS = frozenset(
    {
        "row_role",
        "exposure_level_index",
        "exposure_level",
        "outcome_denominator",
        "outcome_rate_pct",
    }
)
_RISK_DIFFERENCE_COLUMNS = frozenset(
    {
        "risk_difference_pct",
        "risk_difference_ci_low_pct",
        "risk_difference_ci_high_pct",
        "risk_difference_reference_index",
        "risk_difference_comparison_index",
        "risk_difference_effect_measure",
    }
)


def _columns(frame: pd.DataFrame) -> dict[str, Any]:
    return {str(column).strip().lower(): column for column in frame.columns}


def _empty_association() -> pd.DataFrame:
    return pd.DataFrame(columns=["label", "estimate", "lower", "upper"])


def _single_finite_number(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    values = values.loc[values.map(math.isfinite)]
    if values.empty or values.nunique(dropna=True) != 1:
        return None
    return float(values.iloc[0])


def _single_text(series: pd.Series) -> Optional[str]:
    values = series.dropna().astype(str).str.strip()
    values = values.loc[values.ne("")]
    if values.empty or values.nunique(dropna=True) != 1:
        return None
    return str(values.iloc[0])


def _declared_confidence(frame: pd.DataFrame, columns: Mapping[str, Any]) -> Optional[float]:
    column = columns.get("confidence_level")
    if column is None:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all():
        return None
    confidence = _single_finite_number(values)
    return confidence if confidence is not None and 0.5 < confidence < 1 else None


def is_exposure_outcome_distribution_frame(frame: pd.DataFrame) -> bool:
    """Whether ``frame`` carries the typed distribution product identity."""

    return _DISTRIBUTION_IDENTITY_COLUMNS <= set(_columns(frame))


def normalise_distribution_risk_difference(
    frame: pd.DataFrame,
    *,
    primary_exposure: Optional[str] = None,
    display_labels: Optional[Mapping[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """Return the one authorized risk-difference row for Panel A.

    ``None`` means this is not the typed distribution product and lets the
    ordinary association adapter inspect it.  An empty frame means it *is* the
    typed product but lacks a complete, internally consistent contrast; callers
    must fail closed rather than reinterpret a numeric key as an estimate.
    """

    if not is_exposure_outcome_distribution_frame(frame):
        return None
    columns = _columns(frame)
    if not _RISK_DIFFERENCE_COLUMNS <= set(columns):
        return _empty_association()
    confidence = _declared_confidence(frame, columns)
    if confidence is None:
        return _empty_association()

    roles = frame[columns["row_role"]].astype(str).str.strip().str.casefold()
    levels = frame.loc[roles.eq("exposure_level")].copy()
    if len(levels) < 2:
        return _empty_association()

    effect_measure = _single_text(frame[columns["risk_difference_effect_measure"]])
    if str(effect_measure or "").strip().casefold() != "risk_difference":
        return _empty_association()
    estimate = _single_finite_number(frame[columns["risk_difference_pct"]])
    lower = _single_finite_number(frame[columns["risk_difference_ci_low_pct"]])
    upper = _single_finite_number(frame[columns["risk_difference_ci_high_pct"]])
    reference_index = _single_finite_number(
        frame[columns["risk_difference_reference_index"]]
    )
    comparison_index = _single_finite_number(
        frame[columns["risk_difference_comparison_index"]]
    )
    if None in {estimate, lower, upper, reference_index, comparison_index}:
        return _empty_association()
    assert estimate is not None and lower is not None and upper is not None
    assert reference_index is not None and comparison_index is not None
    if lower > estimate or estimate > upper:
        return _empty_association()
    if not reference_index.is_integer() or not comparison_index.is_integer():
        return _empty_association()
    if int(reference_index) == int(comparison_index):
        return _empty_association()

    level_indices = pd.to_numeric(
        levels[columns["exposure_level_index"]], errors="coerce"
    )
    reference_rows = levels.loc[level_indices.eq(int(reference_index))]
    comparison_rows = levels.loc[level_indices.eq(int(comparison_index))]
    if len(reference_rows) != 1 or len(comparison_rows) != 1:
        return _empty_association()

    reference_level = reference_rows.iloc[0][columns["exposure_level"]]
    comparison_level = comparison_rows.iloc[0][columns["exposure_level"]]
    exposure_name = str(primary_exposure or "").strip()
    exposure_column = columns.get("exposure_column")
    if not exposure_name and exposure_column is not None:
        exposure_name = _single_text(frame[exposure_column]) or ""
    declared_contrast = binary_contrast_label(exposure_name, display_labels)
    exposure_label = display_label(exposure_name or "Exposure", display_labels)
    comparison_label = _exposure_level_label(
        comparison_level, display_labels, scope=exposure_name
    )
    reference_label = _exposure_level_label(
        reference_level, display_labels, scope=exposure_name
    )

    result = pd.DataFrame(
        {
            "label": [
                declared_contrast
                or f"{exposure_label}: {comparison_label} vs {reference_label}"
            ],
            "estimate": [estimate],
            "lower": [lower],
            "upper": [upper],
            "confidence_level": [confidence],
        }
    )
    result.attrs.update(
        {
            "source_contract": "exposure_outcome_distribution",
            "effect_measure": "risk_difference_pct",
            "xlabel": "Risk difference (percentage points)",
            "header": f"Risk difference, pp ({100 * confidence:g}% CI)",
            "confidence_level": confidence,
            "null_value": 0.0,
            "ratio_scale": False,
        }
    )
    return result


def normalise_distribution_outcome_rates(
    frame: pd.DataFrame,
    *,
    display_labels: Optional[Mapping[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """Return typed level-specific absolute risks for Panel B.

    The point, interval, and denominator come from the outcome fields.  Exposure
    codes only label/order rows and therefore cannot leak into the quantitative
    axis.
    """

    if not is_exposure_outcome_distribution_frame(frame):
        return None
    columns = _columns(frame)
    confidence = _declared_confidence(frame, columns)
    required = {
        "ci_low_pct",
        "ci_high_pct",
        "exposure_level_index",
    }
    if confidence is None or not required <= set(columns):
        return pd.DataFrame(columns=["score", "rate", "lower", "upper", "n"])

    roles = frame[columns["row_role"]].astype(str).str.strip().str.casefold()
    levels = frame.loc[roles.eq("exposure_level")].copy()
    if levels.empty:
        return pd.DataFrame(columns=["score", "rate", "lower", "upper", "n"])

    exposure_name = ""
    exposure_column = columns.get("exposure_column")
    if exposure_column is not None:
        exposure_name = _single_text(frame[exposure_column]) or ""
    result = pd.DataFrame(
        {
            "score": levels[columns["exposure_level"]].map(
                lambda value: _exposure_level_label(
                    value, display_labels, scope=exposure_name
                )
            ),
            "rate": pd.to_numeric(
                levels[columns["outcome_rate_pct"]], errors="coerce"
            ),
            "lower": pd.to_numeric(levels[columns["ci_low_pct"]], errors="coerce"),
            "upper": pd.to_numeric(levels[columns["ci_high_pct"]], errors="coerce"),
            "n": pd.to_numeric(
                levels[columns["outcome_denominator"]], errors="coerce"
            ),
            "_order": pd.to_numeric(
                levels[columns["exposure_level_index"]], errors="coerce"
            ),
            "confidence_level": confidence,
        }
    ).dropna(subset=["score", "rate", "lower", "upper", "n", "_order"])
    result = result.loc[
        result["n"].gt(0)
        & result["lower"].le(result["rate"])
        & result["rate"].le(result["upper"])
        & result["lower"].ge(0)
        & result["upper"].le(100)
    ].copy()
    if result.empty:
        return pd.DataFrame(columns=["score", "rate", "lower", "upper", "n"])
    result[["rate", "lower", "upper"]] = (
        result[["rate", "lower", "upper"]] / 100.0
    )
    result = result.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    scope_label = binary_scope_label(exposure_name, display_labels)
    result.attrs.update(
        {
            "score_label": (
                f"{scope_label} status" if scope_label else "Exposure group"
            ),
            "score_is_numeric": False,
            "source_contract": "exposure_outcome_distribution",
            "confidence_level": confidence,
        }
    )
    return result


def _exposure_level_label(
    value: Any,
    display_labels: Optional[Mapping[str, str]],
    *,
    scope: Any = None,
) -> str:
    declared = scoped_label_lookup(scope, value, display_labels)
    if declared is not None:
        return declared
    token = str(value).strip().casefold()
    if token in {"0", "0.0", "false", "no", "absent", "negative"}:
        return "Unexposed"
    if token in {"1", "1.0", "true", "yes", "present", "positive"}:
        return "Exposed"
    return display_label(value, display_labels)


__all__ = [
    "is_exposure_outcome_distribution_frame",
    "normalise_distribution_outcome_rates",
    "normalise_distribution_risk_difference",
]
