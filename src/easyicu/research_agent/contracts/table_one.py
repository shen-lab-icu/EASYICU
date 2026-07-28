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
from ..methods.table_one import standardized_difference_from_moments
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
        "mean",
        "sd",
        "p_value",
        "test_name",
        "standardized_mean_difference",
        "absolute_standardized_mean_difference",
        "standardized_difference_status",
        "standardized_difference_reference_group",
        "standardized_difference_comparison_group",
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


def _smd_rows_findings(
    *,
    step: AnalysisStep,
    variable: Any,
    rows: pd.DataFrame,
    group_levels: list[Any],
) -> list[ValidationFinding]:
    """Rebuild one variable's SMD from its emitted source-data moments."""

    if len(group_levels) != 2:
        expected_status = "not_applicable_more_than_two_groups"
        if (
            rows["standardized_mean_difference"].notna().any()
            or rows["absolute_standardized_mean_difference"].notna().any()
            or rows["standardized_difference_reference_group"].notna().any()
            or rows["standardized_difference_comparison_group"].notna().any()
            or set(rows["standardized_difference_status"].dropna().astype(str))
            != {expected_status}
            or rows["standardized_difference_status"].isna().any()
        ):
            return [
                _error(
                    step,
                    "table_one_standardized_difference_invalid",
                    f"Table 1 must mark multi-group SMD as not applicable for {variable.name!r}.",
                    variable=variable.name,
                )
            ]
        return []

    reference_group = str(group_levels[0])
    comparison_group = str(group_levels[1])
    reference_values = (
        rows["standardized_difference_reference_group"].dropna().astype(str)
    )
    comparison_values = (
        rows["standardized_difference_comparison_group"].dropna().astype(str)
    )
    if (
        len(reference_values) != len(rows)
        or set(reference_values) != {reference_group}
        or len(comparison_values) != len(rows)
        or set(comparison_values) != {comparison_group}
    ):
        return [
            _error(
                step,
                "table_one_standardized_difference_contrast_invalid",
                f"Table 1 SMD contrast is not bound to the declared group order for {variable.name!r}.",
                variable=variable.name,
            )
        ]

    strata: list[tuple[str | None, pd.DataFrame]]
    if variable.summary == "count_percent":
        strata = [
            (
                str(level),
                rows[rows["category"].astype(str).eq(str(level))],
            )
            for level in variable.levels
        ]
    else:
        strata = [(None, rows)]

    findings: list[ValidationFinding] = []
    for category, stratum in strata:
        reference = stratum[stratum["group"].astype(str).eq(reference_group)]
        comparison = stratum[stratum["group"].astype(str).eq(comparison_group)]
        if len(reference) != 1 or len(comparison) != 1:
            findings.append(
                _error(
                    step,
                    "table_one_standardized_difference_basis_invalid",
                    f"Table 1 lacks one exact row per SMD comparison group for {variable.name!r}.",
                    variable=variable.name,
                    category=category,
                )
            )
            continue
        reference_nonmissing = pd.to_numeric(
            reference["nonmissing_n"], errors="coerce"
        ).iloc[0]
        comparison_nonmissing = pd.to_numeric(
            comparison["nonmissing_n"], errors="coerce"
        ).iloc[0]
        if not (
            math.isfinite(float(reference_nonmissing))
            and math.isfinite(float(comparison_nonmissing))
        ):
            findings.append(
                _error(
                    step,
                    "table_one_standardized_difference_basis_invalid",
                    f"Table 1 SMD group counts are invalid for {variable.name!r}.",
                    variable=variable.name,
                    category=category,
                )
            )
            continue
        empty_group = reference_nonmissing == 0 or comparison_nonmissing == 0
        if variable.summary == "count_percent":
            reference_percentage = pd.to_numeric(
                reference["percentage"], errors="coerce"
            ).iloc[0]
            comparison_percentage = pd.to_numeric(
                comparison["percentage"], errors="coerce"
            ).iloc[0]
            if empty_group:
                expected = standardized_difference_from_moments(
                    reference_location=None,
                    reference_variance=None,
                    comparison_location=None,
                    comparison_variance=None,
                    empty_group=True,
                )
            elif not (
                math.isfinite(float(reference_percentage))
                and math.isfinite(float(comparison_percentage))
            ):
                findings.append(
                    _error(
                        step,
                        "table_one_standardized_difference_basis_invalid",
                        f"Table 1 category percentages are invalid for {variable.name!r}.",
                        variable=variable.name,
                        category=category,
                    )
                )
                continue
            else:
                reference_proportion = float(reference_percentage) / 100.0
                comparison_proportion = float(comparison_percentage) / 100.0
                expected = standardized_difference_from_moments(
                    reference_location=reference_proportion,
                    reference_variance=(
                        reference_proportion * (1.0 - reference_proportion)
                    ),
                    comparison_location=comparison_proportion,
                    comparison_variance=(
                        comparison_proportion * (1.0 - comparison_proportion)
                    ),
                )
        else:
            reference_mean = pd.to_numeric(reference["mean"], errors="coerce").iloc[0]
            comparison_mean = pd.to_numeric(comparison["mean"], errors="coerce").iloc[0]
            reference_sd = pd.to_numeric(reference["sd"], errors="coerce").iloc[0]
            comparison_sd = pd.to_numeric(comparison["sd"], errors="coerce").iloc[0]
            if empty_group:
                expected = standardized_difference_from_moments(
                    reference_location=None,
                    reference_variance=None,
                    comparison_location=None,
                    comparison_variance=None,
                    empty_group=True,
                )
            elif reference_nonmissing < 2 or comparison_nonmissing < 2:
                expected = standardized_difference_from_moments(
                    reference_location=None,
                    reference_variance=None,
                    comparison_location=None,
                    comparison_variance=None,
                )
            elif not all(
                math.isfinite(float(value))
                for value in (
                    reference_mean,
                    comparison_mean,
                    reference_sd,
                    comparison_sd,
                )
            ):
                findings.append(
                    _error(
                        step,
                        "table_one_standardized_difference_basis_invalid",
                        f"Table 1 mean/SD moments are invalid for {variable.name!r}.",
                        variable=variable.name,
                    )
                )
                continue
            else:
                expected = standardized_difference_from_moments(
                    reference_location=float(reference_mean),
                    reference_variance=float(reference_sd) ** 2,
                    comparison_location=float(comparison_mean),
                    comparison_variance=float(comparison_sd) ** 2,
                )

        statuses = set(stratum["standardized_difference_status"].dropna().astype(str))
        values = pd.to_numeric(stratum["standardized_mean_difference"], errors="coerce")
        absolute_values = pd.to_numeric(
            stratum["absolute_standardized_mean_difference"], errors="coerce"
        )
        valid = (
            not stratum.empty
            and not stratum["standardized_difference_status"].isna().any()
            and statuses == {expected.status}
        )
        if expected.value is None:
            valid = valid and values.isna().all() and absolute_values.isna().all()
        else:
            valid = (
                valid
                and not values.isna().any()
                and not absolute_values.isna().any()
                and all(
                    math.isclose(
                        float(value), expected.value, rel_tol=1e-12, abs_tol=1e-12
                    )
                    for value in values
                )
                and all(
                    math.isclose(
                        float(value),
                        expected.absolute_value,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for value in absolute_values
                )
            )
        if not valid:
            findings.append(
                _error(
                    step,
                    "table_one_standardized_difference_invalid",
                    f"Table 1 SMD does not reconcile to emitted group moments for {variable.name!r}.",
                    variable=variable.name,
                    category=category,
                )
            )
    return findings


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
    observed_schema_versions = set(table["schema_version"].dropna().astype(str))
    if observed_schema_versions != {"easyicu.table_one_result/2"}:
        findings.append(
            _error(
                step,
                "table_one_schema_version_mismatch",
                "The grouped Table 1 does not use the current result schema.",
            )
        )
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
        findings.extend(
            _smd_rows_findings(
                step=step,
                variable=variable,
                rows=rows,
                group_levels=list(spec.group_levels),
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
