"""Host-owned execution for a Planner-declared grouped Table 1.

The Planner owns the grouping variable, closed levels, row variables, summary
families, and comparison tests through :class:`TableOneSpec`.  This module only
executes that closed declaration and emits auditable long-form source data.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from ..schema import TableOneSpec, TableOneVariableSpec


class TableOneContractError(ValueError):
    """The data cannot satisfy the Planner-owned Table 1 design."""


@dataclass(frozen=True)
class StandardizedDifferenceResult:
    """One binary-group standardized difference and its audit status."""

    value: float | None
    status: str

    @property
    def absolute_value(self) -> float | None:
        return None if self.value is None else abs(self.value)


def standardized_difference_from_moments(
    *,
    reference_location: float | None,
    reference_variance: float | None,
    comparison_location: float | None,
    comparison_variance: float | None,
    empty_group: bool = False,
) -> StandardizedDifferenceResult:
    """Return ``comparison - reference`` over the equal-weight pooled SD.

    Continuous variables supply sample means and variances. Categorical rows
    supply level proportions and Bernoulli variances. The shared kernel keeps
    the deterministic executor and output gate on one statistical definition.
    """

    if empty_group:
        return StandardizedDifferenceResult(None, "not_testable_empty_group")
    moments = (
        reference_location,
        reference_variance,
        comparison_location,
        comparison_variance,
    )
    if any(value is None for value in moments):
        return StandardizedDifferenceResult(
            None, "not_testable_insufficient_group_values"
        )
    reference_location = float(reference_location)
    reference_variance = float(reference_variance)
    comparison_location = float(comparison_location)
    comparison_variance = float(comparison_variance)
    if not all(
        math.isfinite(value)
        for value in (
            reference_location,
            reference_variance,
            comparison_location,
            comparison_variance,
        )
    ):
        raise TableOneContractError(
            "Table 1 standardized-difference moments must be finite"
        )
    if reference_variance < 0 or comparison_variance < 0:
        raise TableOneContractError(
            "Table 1 standardized-difference variances must be non-negative"
        )
    difference = comparison_location - reference_location
    pooled_variance = (reference_variance + comparison_variance) / 2.0
    if pooled_variance == 0.0:
        if difference == 0.0:
            return StandardizedDifferenceResult(0.0, "computed")
        return StandardizedDifferenceResult(None, "undefined_zero_pooled_variance")
    value = difference / math.sqrt(pooled_variance)
    if not math.isfinite(value):
        raise TableOneContractError("Table 1 standardized difference must be finite")
    return StandardizedDifferenceResult(value, "computed")


def table_one_spec_sha256(spec: TableOneSpec | dict[str, Any]) -> str:
    """Return the canonical digest of one Planner-owned Table 1 design."""

    contract = (
        spec if isinstance(spec, TableOneSpec) else TableOneSpec.model_validate(spec)
    )
    payload = json.dumps(
        contract.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _token(value: Any) -> str:
    value = _python_scalar(value)
    if isinstance(value, bool):
        scalar_type = "bool"
        scalar_value = value
    elif isinstance(value, int):
        scalar_type = "number"
        scalar_value = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise TableOneContractError("Table 1 levels must be finite JSON scalars")
        scalar_type = "number"
        scalar_value = int(value) if value.is_integer() else value
    elif isinstance(value, str):
        scalar_type = "str"
        scalar_value = value
    else:
        raise TableOneContractError("Table 1 levels must be JSON scalar values")
    return json.dumps(
        {"type": scalar_type, "value": scalar_value},
        sort_keys=True,
        separators=(",", ":"),
    )


def _closed_masks(
    series: pd.Series, levels: Iterable[Any], *, label: str
) -> list[pd.Series]:
    declared = list(levels)
    declared_tokens = {_token(value) for value in declared}
    observed_tokens = {_token(value) for value in series.dropna().unique().tolist()}
    unexpected = sorted(observed_tokens - declared_tokens)
    if unexpected:
        raise TableOneContractError(
            f"{label} contains values outside the Planner-declared closed levels"
        )
    return [
        series.map(
            lambda value, level=level: (
                _token(value) == _token(level) if pd.notna(value) else False
            )
        )
        for level in declared
    ]


def _numeric_values(series: pd.Series, *, label: str) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(series.dtype):
        raise TableOneContractError(f"{label} must be numeric; coercion is not allowed")
    values = series.dropna().to_numpy(dtype=float)
    if values.size and not np.isfinite(values).all():
        raise TableOneContractError(f"{label} contains non-finite values")
    return values


def _numeric_test(
    groups: list[np.ndarray], spec: TableOneVariableSpec
) -> tuple[float | None, str]:
    if any(values.size == 0 for values in groups):
        return None, "not_testable_empty_group"
    if spec.test == "welch_t_or_anova":
        if any(values.size < 2 for values in groups):
            raise TableOneContractError(
                f"Welch comparison for {spec.name!r} requires at least two values per group"
            )
        if len(groups) == 2:
            result = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            return float(result.pvalue), "welch_t"
        result = stats.f_oneway(*groups, equal_var=False)
        return float(result.pvalue), "welch_anova"
    if len(groups) == 2:
        result = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        return float(result.pvalue), "mann_whitney_u"
    result = stats.kruskal(*groups)
    return float(result.pvalue), "kruskal_wallis"


def _categorical_test(
    series: pd.Series,
    group_masks: list[pd.Series],
    spec: TableOneVariableSpec,
) -> tuple[float | None, str]:
    level_masks = _closed_masks(series, spec.levels, label=spec.name)
    table = np.asarray(
        [
            [int((level_mask & group_mask).sum()) for group_mask in group_masks]
            for level_mask in level_masks
        ],
        dtype=int,
    )
    active = table[table.sum(axis=1) > 0]
    if bool((table.sum(axis=0) == 0).any()):
        return None, "not_testable_empty_group"
    if active.shape[0] < 2 or active.shape[1] < 2:
        return None, "not_testable_no_variation"
    chi = stats.chi2_contingency(active, correction=False)
    if active.shape == (2, 2) and bool((chi.expected_freq < 5).any()):
        result = stats.fisher_exact(active, alternative="two-sided")
        return float(result.pvalue), "fisher_exact"
    return float(chi.pvalue), "chi_square"


def _p_value(
    frame: pd.DataFrame,
    group_masks: list[pd.Series],
    variable: TableOneVariableSpec,
) -> tuple[float | None, str]:
    series = frame[variable.name]
    if variable.summary == "count_percent":
        p_value, test_name = _categorical_test(series, group_masks, variable)
    else:
        groups = [
            _numeric_values(series[mask], label=variable.name) for mask in group_masks
        ]
        p_value, test_name = _numeric_test(groups, variable)
    if p_value is None:
        return None, test_name
    if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
        raise TableOneContractError(
            f"Table 1 test for {variable.name!r} returned an invalid p-value"
        )
    return p_value, test_name


def _base_row(
    *,
    variable_order: int,
    variable: TableOneVariableSpec,
    group: str,
    group_order: int,
    denominator_n: int,
    nonmissing_n: int,
    missing_n: int,
    p_value: float | None,
    test_name: str,
    contract_sha256: str,
    standardized_difference: StandardizedDifferenceResult,
    reference_group: str | None,
    comparison_group: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "easyicu.table_one_result/2",
        "contract_sha256": contract_sha256,
        "variable_order": variable_order,
        "variable": variable.name,
        "variable_type": variable.variable_kind,
        "category": None,
        "group": group,
        "group_order": group_order,
        "denominator_n": denominator_n,
        "nonmissing_n": nonmissing_n,
        "missing_n": missing_n,
        "missing_pct": 100.0 * missing_n / denominator_n if denominator_n else None,
        "count": None,
        "percentage": None,
        "mean": None,
        "sd": None,
        "median": None,
        "q25": None,
        "q75": None,
        "p_value": p_value,
        "test_name": test_name,
        "standardized_mean_difference": standardized_difference.value,
        "absolute_standardized_mean_difference": (
            standardized_difference.absolute_value
        ),
        "standardized_difference_status": standardized_difference.status,
        "standardized_difference_reference_group": reference_group,
        "standardized_difference_comparison_group": comparison_group,
    }


def _numeric_standardized_difference(
    series: pd.Series,
    group_masks: list[pd.Series],
) -> StandardizedDifferenceResult:
    if len(group_masks) != 2:
        return StandardizedDifferenceResult(None, "not_applicable_more_than_two_groups")
    groups = [_numeric_values(series[mask], label=series.name) for mask in group_masks]
    if any(values.size == 0 for values in groups):
        return standardized_difference_from_moments(
            reference_location=None,
            reference_variance=None,
            comparison_location=None,
            comparison_variance=None,
            empty_group=True,
        )
    if any(values.size < 2 for values in groups):
        return standardized_difference_from_moments(
            reference_location=None,
            reference_variance=None,
            comparison_location=None,
            comparison_variance=None,
        )
    return standardized_difference_from_moments(
        reference_location=float(np.mean(groups[0])),
        reference_variance=float(np.var(groups[0], ddof=1)),
        comparison_location=float(np.mean(groups[1])),
        comparison_variance=float(np.var(groups[1], ddof=1)),
    )


def _categorical_standardized_differences(
    series: pd.Series,
    group_masks: list[pd.Series],
    variable: TableOneVariableSpec,
) -> dict[str, StandardizedDifferenceResult]:
    if len(group_masks) != 2:
        result = StandardizedDifferenceResult(
            None, "not_applicable_more_than_two_groups"
        )
        return {_token(level): result for level in variable.levels}
    nonmissing = [int(series[mask].notna().sum()) for mask in group_masks]
    if any(value == 0 for value in nonmissing):
        result = standardized_difference_from_moments(
            reference_location=None,
            reference_variance=None,
            comparison_location=None,
            comparison_variance=None,
            empty_group=True,
        )
        return {_token(level): result for level in variable.levels}
    results: dict[str, StandardizedDifferenceResult] = {}
    for level in variable.levels:
        level_token = _token(level)
        proportions = [
            float(
                series[mask]
                .dropna()
                .map(lambda value: _token(value) == level_token)
                .mean()
            )
            for mask in group_masks
        ]
        results[level_token] = standardized_difference_from_moments(
            reference_location=proportions[0],
            reference_variance=proportions[0] * (1.0 - proportions[0]),
            comparison_location=proportions[1],
            comparison_variance=proportions[1] * (1.0 - proportions[1]),
        )
    return results


def build_grouped_table_one(
    frame: pd.DataFrame,
    spec: TableOneSpec | dict[str, Any],
) -> pd.DataFrame:
    """Execute one exact grouped Table 1 and return long-form source data."""

    contract = (
        spec if isinstance(spec, TableOneSpec) else TableOneSpec.model_validate(spec)
    )
    contract_sha256 = table_one_spec_sha256(contract)
    required = {contract.group_by, *(item.name for item in contract.variables)}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise TableOneContractError(
            f"Table 1 input columns are missing: {missing_columns}"
        )
    group_series = frame[contract.group_by]
    if bool(group_series.isna().any()):
        raise TableOneContractError(
            "Table 1 grouping variable contains missing values under fail_closed policy"
        )
    group_masks = _closed_masks(
        group_series, contract.group_levels, label=contract.group_by
    )
    if any(not bool(mask.any()) for mask in group_masks):
        raise TableOneContractError("A Planner-declared Table 1 group is empty")
    display_groups: list[tuple[str, pd.Series]] = [
        ("Overall", pd.Series(True, index=frame.index, dtype=bool)),
        *[
            (str(_python_scalar(level)), mask)
            for level, mask in zip(contract.group_levels, group_masks, strict=True)
        ],
    ]

    rows: list[dict[str, Any]] = []
    for variable_order, variable in enumerate(contract.variables, start=1):
        p_value, test_name = _p_value(frame, group_masks, variable)
        series = frame[variable.name]
        reference_group = (
            str(_python_scalar(contract.group_levels[0]))
            if len(contract.group_levels) == 2
            else None
        )
        comparison_group = (
            str(_python_scalar(contract.group_levels[1]))
            if len(contract.group_levels) == 2
            else None
        )
        numeric_smd = (
            None
            if variable.summary == "count_percent"
            else _numeric_standardized_difference(series, group_masks)
        )
        categorical_smd = (
            _categorical_standardized_differences(series, group_masks, variable)
            if variable.summary == "count_percent"
            else {}
        )
        for group_order, (group_label, mask) in enumerate(display_groups):
            grouped = series[mask]
            denominator_n = int(grouped.shape[0])
            missing_n = int(grouped.isna().sum())
            nonmissing_n = denominator_n - missing_n
            base = _base_row(
                variable_order=variable_order,
                variable=variable,
                group=group_label,
                group_order=group_order,
                denominator_n=denominator_n,
                nonmissing_n=nonmissing_n,
                missing_n=missing_n,
                p_value=p_value,
                test_name=test_name,
                contract_sha256=contract_sha256,
                standardized_difference=(
                    numeric_smd
                    if numeric_smd is not None
                    else StandardizedDifferenceResult(
                        None, "not_applicable_more_than_two_groups"
                    )
                ),
                reference_group=reference_group,
                comparison_group=comparison_group,
            )
            if variable.summary == "count_percent":
                for level, level_mask in zip(
                    variable.levels,
                    _closed_masks(grouped, variable.levels, label=variable.name),
                    strict=True,
                ):
                    row = dict(base)
                    count = int(level_mask.sum())
                    row.update(
                        category=str(_python_scalar(level)),
                        count=count,
                        percentage=(
                            100.0 * count / nonmissing_n if nonmissing_n else None
                        ),
                    )
                    smd = categorical_smd[_token(level)]
                    row.update(
                        standardized_mean_difference=smd.value,
                        absolute_standardized_mean_difference=smd.absolute_value,
                        standardized_difference_status=smd.status,
                    )
                    rows.append(row)
                continue
            values = _numeric_values(grouped, label=variable.name)
            row = dict(base)
            if values.size:
                # Mean/SD remain in the source-data contract even when the
                # reader-facing summary is median/IQR, because they are the
                # auditable moments underlying the SMD.
                row["mean"] = float(np.mean(values))
                row["sd"] = float(np.std(values, ddof=1)) if values.size > 1 else None
                if variable.summary in {"median_iqr", "both"}:
                    row["median"] = float(np.median(values))
                    row["q25"] = float(np.quantile(values, 0.25))
                    row["q75"] = float(np.quantile(values, 0.75))
            rows.append(row)
    return pd.DataFrame.from_records(rows)


__all__ = [
    "StandardizedDifferenceResult",
    "TableOneContractError",
    "build_grouped_table_one",
    "standardized_difference_from_moments",
    "table_one_spec_sha256",
]
