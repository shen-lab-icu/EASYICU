"""Host-owned validation primitives for agent-selected descriptive inputs.

The caller owns every scientific choice: which columns to use, which levels
to declare, how to group rows, and which statistics to report.  These helpers
only make numeric conversion, closed categorical accounting, and measurement
provenance mechanically explicit and fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

_NUMERIC_CATEGORY_STRING = re.compile(
    r"[+-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z"
)


class DescriptiveInputError(ValueError):
    """Raised when a descriptive input cannot satisfy its declared contract."""

    def __init__(self, message: str, *, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = dict(audit)


@dataclass(frozen=True)
class StrictNumericInput:
    """Validated numeric values and conversion counts."""

    values: pd.Series
    audit: dict[str, int]


@dataclass(frozen=True)
class ClosedCategoricalCounts:
    """Counts for every declared level plus closed-partition audit counts."""

    table: pd.DataFrame
    audit: dict[str, int]


def _semantic_numeric(
    series: pd.Series,
    *,
    allow_boolean: bool,
) -> tuple[pd.Series, pd.Series]:
    """Convert scalar numerics while rejecting semantic dtype laundering."""

    def scalar_type_allowed(value: Any) -> bool:
        try:
            if bool(pd.isna(value)):
                return True
        except (TypeError, ValueError):
            pass
        if pd.api.types.is_bool(value):
            return allow_boolean
        if isinstance(
            value,
            (complex, date, datetime, timedelta, pd.Timestamp, pd.Timedelta),
        ):
            return False
        if pd.api.types.is_datetime64_dtype(type(value)):
            return False
        if pd.api.types.is_timedelta64_dtype(type(value)):
            return False
        return True

    semantic_valid = series.map(scalar_type_allowed).astype(bool)
    if pd.api.types.is_datetime64_any_dtype(
        series.dtype
    ) or pd.api.types.is_timedelta64_dtype(series.dtype):
        semantic_valid[:] = False
    elif pd.api.types.is_bool_dtype(series.dtype) and not allow_boolean:
        semantic_valid[:] = False
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric, semantic_valid


def strict_numeric_input(
    series: pd.Series,
    *,
    name: str | None = None,
    column: str | None = None,
) -> StrictNumericInput:
    """Convert an explicitly selected Series without hiding invalid values.

    True input missingness remains missing.  A nonmissing value that cannot be
    converted, a semantically nonnumeric value (for example a timestamp), or a
    nonfinite numeric value raises :class:`DescriptiveInputError`.  ``name`` and
    ``column`` are equivalent diagnostic labels accepted for generated-analysis
    wrappers; they never select data or relax validation.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("strict_numeric_input requires a pandas Series")
    if name is not None and column is not None and name != column:
        raise ValueError("name and column diagnostic labels disagree")
    diagnostic_label = name if name is not None else column
    if diagnostic_label is not None and (
        not isinstance(diagnostic_label, str) or not diagnostic_label.strip()
    ):
        raise ValueError("numeric input diagnostic label must be a non-empty string")

    raw_missing = series.isna()
    numeric, semantic_valid = _semantic_numeric(series, allow_boolean=False)
    coercion_invalid = ~raw_missing & numeric.isna()
    semantic_invalid = ~raw_missing & ~semantic_valid
    numeric_present = numeric.notna()
    finite = pd.Series(True, index=series.index, dtype=bool)
    if bool(numeric_present.any()):
        finite.loc[numeric_present] = np.isfinite(
            numeric.loc[numeric_present].to_numpy()
        )
    nonfinite = numeric_present & ~finite
    audit = {
        "n_total": int(len(series)),
        "raw_missing_n": int(raw_missing.sum()),
        "numeric_n": int((~raw_missing).sum()),
        "coercion_invalid_n": int(coercion_invalid.sum()),
        "semantic_invalid_n": int(semantic_invalid.sum()),
        "nonfinite_n": int(nonfinite.sum()),
    }
    if (
        audit["coercion_invalid_n"]
        or audit["semantic_invalid_n"]
        or audit["nonfinite_n"]
    ):
        label = f" {diagnostic_label!r}" if diagnostic_label is not None else ""
        raise DescriptiveInputError(
            f"numeric input{label} contains unconvertible, semantically invalid, or "
            "nonfinite values",
            audit=audit,
        )

    values = numeric.copy()
    values.loc[raw_missing] = np.nan
    values.name = series.name
    return StrictNumericInput(values=values, audit=audit)


def _canonical_level(value: Any) -> tuple[str, str]:
    if pd.api.types.is_bool(value):
        return ("boolean", "true" if bool(value) else "false")

    numeric_candidate: str | None = None
    if isinstance(value, str):
        stripped = value.strip()
        # Preserve identifier-like categories such as "01".  Only strings
        # with an unambiguous ordinary numeric spelling are canonicalized with
        # their numeric scalar equivalents.
        if _NUMERIC_CATEGORY_STRING.fullmatch(stripped):
            numeric_candidate = stripped
    elif isinstance(value, (int, float, Decimal, np.number)):
        numeric_candidate = str(value)
    if numeric_candidate:
        try:
            number = Decimal(numeric_candidate)
        except InvalidOperation:
            number = None
        if number is not None and number.is_finite():
            if number == 0:
                number = Decimal(0)
            return ("number", str(number.normalize()))

    return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))


def closed_categorical_counts(
    series: pd.Series,
    *,
    declared_levels: Sequence[Any],
) -> ClosedCategoricalCounts:
    """Count an agent-declared closed categorical level set.

    Numeric-equivalent representations such as ``0``, ``0.0``, and ``"0.0"``
    share one level identity.  Every nonmissing observed value must match
    exactly one declared identity.  The returned table includes zero-count
    levels but deliberately does not choose or calculate a percentage
    denominator; that scientific reporting choice remains with the Agent.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("closed_categorical_counts requires a pandas Series")
    levels = list(declared_levels)
    if not levels:
        raise ValueError("declared_levels must contain at least one level")

    declared: dict[tuple[str, str], Any] = {}
    for level in levels:
        try:
            missing = bool(pd.isna(level))
        except (TypeError, ValueError):
            missing = False
        if missing:
            raise ValueError("declared categorical levels cannot be missing")
        key = _canonical_level(level)
        if key in declared:
            raise ValueError(
                "declared categorical levels are duplicated after canonicalization"
            )
        declared[key] = level

    raw_missing = series.isna()
    observed_keys = series.loc[~raw_missing].map(_canonical_level)
    undeclared = ~observed_keys.isin(declared)
    audit = {
        "n_total": int(len(series)),
        "nonmissing_n": int((~raw_missing).sum()),
        "missing_n": int(raw_missing.sum()),
        "declared_level_n": int(len(declared)),
        "undeclared_n": int(undeclared.sum()),
    }
    if audit["undeclared_n"]:
        raise DescriptiveInputError(
            "categorical input contains undeclared nonmissing values",
            audit=audit,
        )

    counts = observed_keys.value_counts(dropna=False)
    table = pd.DataFrame(
        [
            {
                "level": level,
                "count": int(counts.get(key, 0)),
            }
            for key, level in declared.items()
        ]
    )
    audit["closed_count_n"] = int(table["count"].sum())
    return ClosedCategoricalCounts(table=table, audit=audit)


def measurement_provenance_receipt(
    frame: pd.DataFrame,
    *,
    measured_column: str,
    count_column: str,
) -> dict[str, Any]:
    """Validate one declared measured/count triad and return audit metadata.

    The receipt contains no values, row mask, or filtered frame.  It therefore
    cannot change the caller's analysis population.  Invalid flags/counts or a
    disagreement between ``measured`` and ``count > 0`` raise immediately.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("measurement_provenance_receipt requires a DataFrame")
    if measured_column == count_column:
        raise DescriptiveInputError(
            "measurement provenance roles require two distinct columns",
            audit={
                "comparison_n": 0,
                "invalid_pair_n": int(len(frame)),
                "discordant_n": 0,
                "invalid_measured_n": 0,
            },
        )
    missing_columns = [
        column
        for column in (measured_column, count_column)
        if column not in frame.columns
    ]
    if missing_columns:
        measured_missing = measured_column not in frame.columns
        raise DescriptiveInputError(
            f"measurement provenance columns missing: {missing_columns}",
            audit={
                "comparison_n": 0,
                "invalid_pair_n": int(len(frame)),
                "discordant_n": 0,
                "invalid_measured_n": int(len(frame)) if measured_missing else 0,
            },
        )

    measured, measured_semantic_valid = _semantic_numeric(
        frame[measured_column],
        allow_boolean=True,
    )
    count, count_semantic_valid = _semantic_numeric(
        frame[count_column],
        allow_boolean=False,
    )
    valid_measured = measured_semantic_valid & measured.isin([0, 1])
    count_finite = pd.Series(False, index=frame.index, dtype=bool)
    count_present = count.notna()
    if bool(count_present.any()):
        count_finite.loc[count_present] = np.isfinite(
            count.loc[count_present].to_numpy()
        )
    valid_count = pd.Series(False, index=frame.index, dtype=bool)
    count_candidates = count_semantic_valid & count_present & count_finite
    if bool(count_candidates.any()):
        candidate_values = count.loc[count_candidates]
        try:
            valid_count.loc[count_candidates] = candidate_values.ge(
                0
            ) & candidate_values.mod(1).eq(0)
        except (TypeError, ValueError):
            # A semantically unusual extension/scalar dtype must become a
            # typed fail-closed audit, never leak a raw pandas/numpy exception.
            valid_count.loc[count_candidates] = False
    valid_pair = valid_measured & valid_count
    discordant = pd.Series(False, index=frame.index, dtype=bool)
    if bool(valid_pair.any()):
        discordant.loc[valid_pair] = measured.loc[valid_pair].astype(bool) != count.loc[
            valid_pair
        ].gt(0)
    audit = {
        "comparison_n": int(valid_pair.sum()),
        "invalid_pair_n": int((~valid_pair).sum()),
        "discordant_n": int(discordant.sum()),
        "invalid_measured_n": int((~valid_measured).sum()),
    }
    if audit["invalid_pair_n"] or audit["discordant_n"]:
        raise DescriptiveInputError(
            "measurement provenance is invalid or discordant",
            audit=audit,
        )

    return {
        "measured_column": measured_column,
        "count_column": count_column,
        "status": "checked",
        "comparison_n": audit["comparison_n"],
        "invalid_pair_n": 0,
        "discordant_n": 0,
        "role": "audit_only",
    }


__all__ = [
    "ClosedCategoricalCounts",
    "DescriptiveInputError",
    "StrictNumericInput",
    "closed_categorical_counts",
    "measurement_provenance_receipt",
    "strict_numeric_input",
]
