"""Deterministic output gate for a Planner-owned grouped Table 1."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..authority.table_one_binding import (
    bind_table_one_execution_spec,
    table_one_execution_spec,
)
from ..methods.table_one import table_one_spec_sha256
from ..schema import AnalysisStep, ValidationFinding

_REQUIRED_COLUMNS = frozenset(
    {
        "schema_version",
        "contract_sha256",
        "variable",
        "variable_type",
        "category",
        "group",
        "denominator_n",
        "nonmissing_n",
        "missing_n",
        "missing_pct",
        "count",
        "percentage",
        "p_value",
        "test_name",
    }
)
_TESTS = {
    "welch_t_or_anova": {"welch_t", "welch_anova"},
    "mann_whitney_or_kruskal": {"mann_whitney_u", "kruskal_wallis"},
    "chi_square_with_fisher_exact_for_sparse_2x2": {
        "chi_square",
        "fisher_exact",
    },
}
_NOT_TESTABLE = {"not_testable_empty_group", "not_testable_no_variation"}


def _error(
    step: AnalysisStep, reason: str, message: str, **detail: Any
) -> ValidationFinding:
    return ValidationFinding(
        validator="table_one_contract",
        severity="error",
        message=message,
        detail={"step_id": step.step_id, "reason": reason, **detail},
    )


def table_one_output_findings(
    *,
    step: AnalysisStep,
    out_dir: Path | None,
) -> list[ValidationFinding]:
    """Verify grouped structure, tests, and missingness denominators."""

    spec = table_one_execution_spec(step)
    if spec is None:
        return []
    if out_dir is None:
        return [
            _error(
                step,
                "output_directory_missing",
                "Table 1 output directory is unavailable.",
            )
        ]
    path = Path(out_dir) / "table_one.csv"
    if not path.is_file():
        return [
            _error(
                step,
                "table_one_missing",
                "The declared grouped Table 1 file is missing.",
            )
        ]
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        return [
            _error(
                step,
                "table_one_unreadable",
                f"The grouped Table 1 is unreadable: {exc}",
            )
        ]
    missing_columns = sorted(_REQUIRED_COLUMNS - set(table.columns))
    if missing_columns:
        return [
            _error(
                step,
                "table_one_schema_incomplete",
                "The grouped Table 1 lacks required source-data columns.",
                missing_columns=missing_columns,
            )
        ]
    findings: list[ValidationFinding] = []
    expected_digest = table_one_spec_sha256(spec)
    observed_digests = set(table["contract_sha256"].dropna().astype(str))
    if observed_digests != {expected_digest}:
        findings.append(
            _error(
                step,
                "table_one_contract_digest_mismatch",
                "Table 1 rows are not bound to the Planner-owned design.",
            )
        )
    expected_groups = {"Overall", *(str(value) for value in spec.group_levels)}
    for variable in spec.variables:
        rows = table[table["variable"].astype(str).eq(variable.name)]
        if rows.empty:
            findings.append(
                _error(
                    step,
                    "table_one_variable_missing",
                    f"Table 1 omits declared variable {variable.name!r}.",
                    variable=variable.name,
                )
            )
            continue
        observed_groups = set(rows["group"].dropna().astype(str))
        if observed_groups != expected_groups:
            findings.append(
                _error(
                    step,
                    "table_one_groups_incomplete",
                    f"Table 1 groups for {variable.name!r} do not match the declaration.",
                    variable=variable.name,
                )
            )
        p_values = pd.to_numeric(rows["p_value"], errors="coerce").dropna().unique()
        test_names = set(rows["test_name"].dropna().astype(str))
        group_nonmissing = rows.groupby("group", dropna=False)["nonmissing_n"].first()
        empty_comparison_group = any(
            float(value) == 0
            for group, value in group_nonmissing.items()
            if str(group) != "Overall" and pd.notna(value)
        )
        declared_empty_group = test_names == {"not_testable_empty_group"}
        declared_no_variation = test_names == {"not_testable_no_variation"}
        overall_rows = rows[rows["group"].astype(str).eq("Overall")]
        active_overall_categories = (
            pd.to_numeric(overall_rows["count"], errors="coerce").gt(0).sum()
        )
        proven_no_variation = (
            variable.summary == "count_percent" and active_overall_categories < 2
        )
        if declared_empty_group:
            valid_p_value = len(p_values) == 0 and empty_comparison_group
        elif declared_no_variation:
            valid_p_value = len(p_values) == 0 and proven_no_variation
        else:
            valid_p_value = (
                len(p_values) == 1
                and math.isfinite(float(p_values[0]))
                and 0 <= float(p_values[0]) <= 1
            )
        if not valid_p_value:
            findings.append(
                _error(
                    step,
                    "table_one_p_value_invalid",
                    f"Table 1 requires one bounded P value for {variable.name!r}.",
                    variable=variable.name,
                )
            )
        if test_names - (_TESTS[variable.test] | _NOT_TESTABLE):
            findings.append(
                _error(
                    step,
                    "table_one_test_mismatch",
                    f"Table 1 uses an undeclared test for {variable.name!r}.",
                    variable=variable.name,
                )
            )
        for group_name, group_rows in rows.groupby("group", dropna=False):
            denominator = pd.to_numeric(group_rows["denominator_n"], errors="coerce")
            nonmissing = pd.to_numeric(group_rows["nonmissing_n"], errors="coerce")
            missing = pd.to_numeric(group_rows["missing_n"], errors="coerce")
            if (
                denominator.nunique(dropna=False) != 1
                or nonmissing.nunique(dropna=False) != 1
                or missing.nunique(dropna=False) != 1
            ):
                findings.append(
                    _error(
                        step,
                        "table_one_denominator_inconsistent",
                        f"Table 1 denominator fields vary within {variable.name!r}/{group_name!r}.",
                        variable=variable.name,
                        group=str(group_name),
                    )
                )
                continue
            denominator_n = float(denominator.iloc[0])
            nonmissing_n = float(nonmissing.iloc[0])
            missing_n = float(missing.iloc[0])
            expected_missing_pct = (
                100.0 * missing_n / denominator_n if denominator_n else math.nan
            )
            observed_missing = pd.to_numeric(group_rows["missing_pct"], errors="coerce")
            if (
                denominator_n != nonmissing_n + missing_n
                or observed_missing.isna().any()
                or not all(
                    math.isclose(
                        float(value), expected_missing_pct, rel_tol=0, abs_tol=1e-8
                    )
                    for value in observed_missing
                )
            ):
                findings.append(
                    _error(
                        step,
                        "table_one_missingness_denominator_invalid",
                        f"Table 1 missingness is not missing_n/group denominator for {variable.name!r}/{group_name!r}.",
                        variable=variable.name,
                        group=str(group_name),
                    )
                )
            if variable.summary == "count_percent":
                counts = pd.to_numeric(group_rows["count"], errors="coerce")
                if counts.isna().any() or float(counts.sum()) != nonmissing_n:
                    findings.append(
                        _error(
                            step,
                            "table_one_category_partition_invalid",
                            f"Table 1 categories do not partition non-missing {variable.name!r}/{group_name!r} rows.",
                            variable=variable.name,
                            group=str(group_name),
                        )
                    )
    return findings


__all__ = ["table_one_output_findings"]
