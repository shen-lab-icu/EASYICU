"""Host-owned execution for a Planner-declared grouped Table 1.

The Planner owns the grouping variable, closed levels, row variables, summary
families, and comparison tests through :class:`TableOneSpec`.  This module only
executes that closed declaration and emits auditable long-form source data.
"""

from __future__ import annotations

import json
import hashlib
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from ..schema import TableOneSpec, TableOneVariableSpec


class TableOneContractError(ValueError):
    """The data cannot satisfy the Planner-owned Table 1 design."""


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
    if isinstance(value, float) and not math.isfinite(value):
        raise TableOneContractError("Table 1 levels must be finite JSON scalars")
    if not isinstance(value, (str, bool, int, float)):
        raise TableOneContractError("Table 1 levels must be JSON scalar values")
    return json.dumps(
        {"type": type(value).__name__, "value": value},
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
) -> tuple[float, str]:
    if any(values.size == 0 for values in groups):
        raise TableOneContractError(
            f"Table 1 variable {spec.name!r} has an empty comparison group"
        )
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
) -> tuple[float, str]:
    level_masks = _closed_masks(series, spec.levels, label=spec.name)
    table = np.asarray(
        [
            [int((level_mask & group_mask).sum()) for group_mask in group_masks]
            for level_mask in level_masks
        ],
        dtype=int,
    )
    active = table[table.sum(axis=1) > 0]
    if active.shape[0] < 2 or active.shape[1] < 2:
        raise TableOneContractError(
            f"Table 1 variable {spec.name!r} lacks two comparable levels/groups"
        )
    chi = stats.chi2_contingency(active, correction=False)
    if active.shape == (2, 2) and bool((chi.expected_freq < 5).any()):
        result = stats.fisher_exact(active, alternative="two-sided")
        return float(result.pvalue), "fisher_exact"
    return float(chi.pvalue), "chi_square"


def _p_value(
    frame: pd.DataFrame,
    group_masks: list[pd.Series],
    variable: TableOneVariableSpec,
) -> tuple[float, str]:
    series = frame[variable.name]
    if variable.summary == "count_percent":
        p_value, test_name = _categorical_test(series, group_masks, variable)
    else:
        groups = [
            _numeric_values(series[mask], label=variable.name) for mask in group_masks
        ]
        p_value, test_name = _numeric_test(groups, variable)
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
    p_value: float,
    test_name: str,
    contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "easyicu.table_one_result/1",
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
    }


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
                    rows.append(row)
                continue
            values = _numeric_values(grouped, label=variable.name)
            row = dict(base)
            if values.size:
                if variable.summary in {"mean_sd", "both"}:
                    row["mean"] = float(np.mean(values))
                    row["sd"] = (
                        float(np.std(values, ddof=1)) if values.size > 1 else None
                    )
                if variable.summary in {"median_iqr", "both"}:
                    row["median"] = float(np.median(values))
                    row["q25"] = float(np.quantile(values, 0.25))
                    row["q75"] = float(np.quantile(values, 0.75))
            rows.append(row)
    return pd.DataFrame.from_records(rows)


__all__ = [
    "TableOneContractError",
    "build_grouped_table_one",
    "table_one_spec_sha256",
]
