"""Step 2 -- the fixed standard analysis over a loaded landmark cohort.

Every estimate here comes from an existing owner kernel:

* the adjusted model and every sensitivity refit run through
  ``run_adjusted_association_from_env`` (the host's statsmodels owner with
  patient-level cluster-robust covariance);
* absolute risks use ``wilson_interval``; ordered trends use
  ``cochran_armitage_trend`` and ``jonckheere_terpstra_trend``;
* the functional-form check uses ``rcs_basis`` for the spline columns and the
  same ``fit_estimator`` the adjusted model uses, plus ``rcs_fit`` /
  ``nonlinearity_wald_test`` for the joint nonlinearity diagnostic;
* Table 1 is ``build_grouped_table_one`` under the repeated-units schema.

The script decides nothing scientific.  It reads the specification, runs the
declared roster in a fixed order, and records every denominator and every
failed refit instead of dropping it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ....contracts.dependence import (
    PatientGroupResolutionError,
    PlannedDependenceRequirement,
    resolve_patient_groups,
)
from ....contracts.model_terms import level_spelling
from ....execution.model_matrix import compile_model_terms
from ....execution.runners.adjusted_association_executor import (
    AdjustedAssociationError,
    run_adjusted_association_from_env,
)
from ....methods.ordered_trends import (
    cochran_armitage_trend,
    jonckheere_terpstra_trend,
    wilson_interval,
)
from ....methods.rcs_dose_response import (
    RCSError,
    nonlinearity_wald_test,
    rcs_basis,
    rcs_fit,
)
from ....methods.table_one import TableOneContractError, build_grouped_table_one
from ....numeric_scalars import coerce_optional_finite_float as _finite
from ....robustness.estimators import fit_estimator
from ..spec import LandmarkCategoricalSpec
from .load_cohort import LoadedCohort, spell_levels

ANALYSIS_TOKEN = "✓ Analysis completed successfully!"
METHOD_FAMILY = "statsmodels_logit_mle"
PRIMARY_VARIANT_ID = "primary"


class AnalysisContractError(RuntimeError):
    """The declared primary analysis could not be completed as specified."""


@dataclass
class AnalysisResult:
    """All tables and the headline ledger of one skill run."""

    spec: LandmarkCategoricalSpec
    cohort: LoadedCohort
    kernel_dir: Path
    cohort_flow: pd.DataFrame
    exposure_level_counts: pd.DataFrame
    measurement_audit: pd.DataFrame
    table_one: pd.DataFrame
    absolute_risk: pd.DataFrame
    adjusted_association_estimates: pd.DataFrame
    adjusted_association_coefficients: pd.DataFrame
    adjusted_trend: pd.DataFrame
    ordinal_trend_tests: pd.DataFrame
    secondary_outcome_summary: pd.DataFrame
    association_sensitivity_grid: pd.DataFrame
    functional_form_sensitivity: pd.DataFrame
    robustness_summary: pd.DataFrame
    primary_summary: dict[str, Any]
    key_metrics: dict[str, Any] = field(default_factory=dict)
    caveat_flags: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "cohort_flow": self.cohort_flow,
            "exposure_level_counts": self.exposure_level_counts,
            "measurement_audit": self.measurement_audit,
            "table_one": self.table_one,
            "absolute_risk": self.absolute_risk,
            "adjusted_association_estimates": self.adjusted_association_estimates,
            "adjusted_association_coefficients": self.adjusted_association_coefficients,
            "adjusted_trend": self.adjusted_trend,
            "ordinal_trend_tests": self.ordinal_trend_tests,
            "secondary_outcome_summary": self.secondary_outcome_summary,
            "association_sensitivity_grid": self.association_sensitivity_grid,
            "functional_form_sensitivity": self.functional_form_sensitivity,
            "robustness_summary": self.robustness_summary,
        }

    def primary_row(self) -> pd.Series:
        estimates = self.adjusted_association_estimates
        rows = estimates.loc[estimates["is_primary_contrast"].astype(bool)]
        if len(rows) != 1:
            raise AnalysisContractError("exactly one primary contrast row is required")
        return rows.iloc[0]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _cluster_groups(
    frame: pd.DataFrame, dependence: PlannedDependenceRequirement | None
) -> tuple[Optional[pd.Series], str]:
    if dependence is None:
        return None, "model_based"
    series = frame[dependence.group_source]
    try:
        resolved = resolve_patient_groups(
            series.astype("object").tolist(), requirement=dependence
        )
    except PatientGroupResolutionError as exc:
        raise AnalysisContractError(str(exc)) from exc
    groups = pd.Series(list(resolved.groups), index=frame.index, dtype="object")
    return groups, dependence.variance_estimator


def _known_levels(frame: pd.DataFrame, column: str) -> pd.Series:
    return spell_levels(frame[column])


def _level_counts(
    frame: pd.DataFrame, column: str, spec: LandmarkCategoricalSpec
) -> list[dict[str, Any]]:
    spelled = _known_levels(frame, column)
    outcome = pd.to_numeric(frame[spec.outcome], errors="coerce")
    rows: list[dict[str, Any]] = []
    for level in [*spec.exposure_levels, spec.unknown_level_label]:
        mask = spelled.eq("") if level == spec.unknown_level_label else spelled.eq(level)
        n = int(mask.sum())
        events = int(outcome[mask].eq(1.0).sum()) if n else 0
        rows.append(
            {
                "exposure_column": column,
                "level": level,
                "is_unknown": level == spec.unknown_level_label,
                "n": n,
                "n_events": events,
                "share_of_landmark": (n / len(frame)) if len(frame) else float("nan"),
            }
        )
    return rows


def _absolute_risk_rows(
    frame: pd.DataFrame, spec: LandmarkCategoricalSpec
) -> list[dict[str, Any]]:
    outcome = pd.to_numeric(frame[spec.outcome], errors="coerce")
    levels = frame[LoadedCohort.level_column]
    rows: list[dict[str, Any]] = []
    for level in [*spec.exposure_levels, spec.unknown_level_label]:
        mask = levels.eq(level)
        n = int(mask.sum())
        events = int(outcome[mask].eq(1.0).sum()) if n else 0
        row: dict[str, Any] = {
            "level": level,
            "is_unknown": level == spec.unknown_level_label,
            "in_primary_model": level != spec.unknown_level_label,
            "n": n,
            "n_events": events,
            "risk": None,
            "risk_ci_low": None,
            "risk_ci_high": None,
            "ci_method": None,
        }
        if n:
            interval = wilson_interval(events, n)
            row.update(
                {
                    "risk": interval.estimate,
                    "risk_ci_low": interval.ci_low,
                    "risk_ci_high": interval.ci_high,
                    "ci_method": interval.method,
                }
            )
        rows.append(row)
    return rows


def _secondary_summary_rows(
    frame: pd.DataFrame, spec: LandmarkCategoricalSpec
) -> list[dict[str, Any]]:
    outcome = pd.to_numeric(frame[spec.outcome], errors="coerce")
    levels = frame[LoadedCohort.level_column]
    rows: list[dict[str, Any]] = []
    for secondary in spec.secondary_outcomes:
        values = pd.to_numeric(frame[secondary.name], errors="coerce")
        populations = [("all_landmark_rows", pd.Series(True, index=frame.index))]
        if secondary.report_survivors_separately:
            populations.append(("survivors_only", outcome.eq(0.0)))
        for population, population_mask in populations:
            for level in [*spec.exposure_levels, spec.unknown_level_label]:
                mask = levels.eq(level) & population_mask
                subset = values[mask]
                evaluable = subset.dropna()
                rows.append(
                    {
                        "outcome": secondary.name,
                        "unit": secondary.unit,
                        "population": population,
                        "level": level,
                        "is_unknown": level == spec.unknown_level_label,
                        "n_total": int(mask.sum()),
                        "n_evaluable": int(evaluable.shape[0]),
                        "n_missing": int(subset.isna().sum()),
                        "median": _finite(evaluable.median()) if len(evaluable) else None,
                        "q25": _finite(evaluable.quantile(0.25)) if len(evaluable) else None,
                        "q75": _finite(evaluable.quantile(0.75)) if len(evaluable) else None,
                    }
                )
    return rows


def _table_one(frame: pd.DataFrame, spec: LandmarkCategoricalSpec) -> pd.DataFrame:
    """Table 1 by exposure level under the repeated-units (no p-value) schema."""

    working = frame.copy()
    group_column = "__table_one_group__"
    levels = working[LoadedCohort.level_column]
    working[group_column] = levels.where(
        levels.ne(spec.unknown_level_label), None
    )
    # The Table 1 owner matches declared levels by typed JSON token, so every
    # categorical row variable is spelled the same way the levels are declared.
    categorical_columns = [
        item.name for item in spec.covariates if item.coding != "continuous"
    ] + [spec.outcome]
    for column in categorical_columns:
        spelled = spell_levels(working[column])
        working[column] = spelled.where(spelled.ne(""), None)
    variables: list[dict[str, Any]] = []
    for covariate in spec.covariates:
        if covariate.coding == "continuous":
            variables.append(
                {
                    "name": covariate.name,
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "none_descriptive_smd_only",
                }
            )
        else:
            variables.append(
                {
                    "name": covariate.name,
                    "variable_kind": "categorical",
                    "summary": "count_percent",
                    "test": "none_descriptive_smd_only",
                    "levels": [level_spelling(item) for item in (covariate.levels or [])],
                }
            )
    variables.append(
        {
            "name": spec.outcome,
            "variable_kind": "categorical",
            "summary": "count_percent",
            "test": "none_descriptive_smd_only",
            "levels": ["0", "1"],
        }
    )
    for secondary in spec.secondary_outcomes:
        variables.append(
            {
                "name": secondary.name,
                "variable_kind": "continuous",
                "summary": "median_iqr",
                "test": "none_descriptive_smd_only",
            }
        )
    contract = {
        "schema_version": "easyicu.table_one/2",
        "group_by": group_column,
        "group_levels": list(spec.exposure_levels),
        "variables": variables,
        "include_overall": True,
        "missing_group_policy": "exclude_and_report",
        "p_values_required": False,
        "p_value_adjustment": "not_applicable_repeated_units",
    }
    try:
        table = build_grouped_table_one(working, contract)
    except TableOneContractError as exc:
        raise AnalysisContractError(f"Table 1 could not be built: {exc}") from exc
    table = table.copy()
    table["group_by"] = spec.exposure
    return table


def _primary_contrast_column(spec: LandmarkCategoricalSpec, exposure: str) -> str:
    return f"{exposure}__is_{spec.primary_contrast_level}"


def _fit_variant(
    *,
    variant_id: str,
    axis: str,
    frame: pd.DataFrame,
    spec: LandmarkCategoricalSpec,
    exposure: str,
    restriction: str,
    kernel_dir: Path,
) -> tuple[dict[str, Any], Optional[dict[str, Any]], Optional[pd.DataFrame]]:
    """Run the declared model on one variant and return its primary-contrast row."""

    out_dir = kernel_dir / variant_id
    row: dict[str, Any] = {
        "variant_id": variant_id,
        "axis": axis,
        "exposure_column": exposure,
        "restriction": restriction,
        "n_rows_offered": int(len(frame)),
        "n": None,
        "n_events": None,
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "effect_scale": "odds_ratio",
        "variance_estimator": None,
        "cluster_count": None,
        "fit_status": "failed",
        "note": "",
    }
    try:
        summary = run_adjusted_association_from_env(
            requirement_id=variant_id,
            exposure=exposure,
            outcome=spec.outcome,
            covariates=spec.covariate_names(),
            model_terms=[term.model_dump(mode="json") for term in spec.model_terms(exposure=exposure)],
            estimator_kind="logistic",
            analysis_set="source_aware",
            analysis_role="primary" if variant_id == PRIMARY_VARIANT_ID else "sensitivity",
            method_family=METHOD_FAMILY,
            primary_contrast_level=spec.primary_contrast_level,
            dependence=spec.dependence,
            typed_cohort_input=None,
            frame=frame,
            cohort_path=None,
            emit_step_summary=False,
            output_dir=out_dir,
        )
    except AdjustedAssociationError as exc:
        row["note"] = str(exc)
        return row, None, None
    estimates = pd.read_csv(out_dir / "adjusted_association_estimates.csv")
    primary = estimates.loc[estimates["is_primary_contrast"].astype(bool)]
    if len(primary) != 1:
        row["note"] = "the kernel did not mark exactly one primary contrast"
        return row, summary, estimates
    primary_row = primary.iloc[0]
    row.update(
        {
            "n": int(summary["n_total"]),
            "n_events": int(summary["n_events"]) if summary.get("n_events") is not None else None,
            "estimate": _finite(primary_row["estimate"]),
            "ci_low": _finite(primary_row["ci_low"]),
            "ci_high": _finite(primary_row["ci_high"]),
            "variance_estimator": summary.get("variance_estimator"),
            "cluster_count": summary.get("cluster_count"),
            "fit_status": "fitted",
            "note": str(summary["model_contracts"][0].get("separation_detected") and "separation_detected" or ""),
        }
    )
    return row, summary, estimates


def _adjusted_trend(
    frame: pd.DataFrame, spec: LandmarkCategoricalSpec, kernel_dir: Path
) -> pd.DataFrame:
    out_dir = kernel_dir / "adjusted_trend"
    terms = [spec.ordinal_trend_term(), *(item.model_term() for item in spec.covariates)]
    record: dict[str, Any] = {
        "requirement_id": "adjusted_trend_per_level_increment",
        "exposure": spec.exposure,
        "coding": "ordinal_linear_declared_level_index",
        "levels": "|".join(spec.exposure_levels),
        "n": None,
        "n_events": None,
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "effect_scale": "odds_ratio_per_level_increment",
        "fit_status": "failed",
        "note": "",
    }
    try:
        summary = run_adjusted_association_from_env(
            requirement_id=record["requirement_id"],
            exposure=spec.exposure,
            outcome=spec.outcome,
            covariates=spec.covariate_names(),
            model_terms=[term.model_dump(mode="json") for term in terms],
            estimator_kind="logistic",
            analysis_set="source_aware",
            analysis_role="secondary",
            method_family=METHOD_FAMILY,
            primary_contrast_level=None,
            dependence=spec.dependence,
            typed_cohort_input=None,
            frame=frame,
            cohort_path=None,
            emit_step_summary=False,
            output_dir=out_dir,
        )
    except AdjustedAssociationError as exc:
        record["note"] = str(exc)
        return pd.DataFrame([record])
    record.update(
        {
            "n": int(summary["n_total"]),
            "n_events": int(summary["n_events"]) if summary.get("n_events") is not None else None,
            "estimate": _finite(summary["primary_estimate"]),
            "ci_low": _finite(summary["primary_ci_low"]),
            "ci_high": _finite(summary["primary_ci_high"]),
            "fit_status": "fitted",
        }
    )
    return pd.DataFrame([record])


def _ordinal_trend_tests(
    frame: pd.DataFrame, spec: LandmarkCategoricalSpec
) -> pd.DataFrame:
    levels = frame[LoadedCohort.level_column]
    outcome = pd.to_numeric(frame[spec.outcome], errors="coerce")
    rows: list[dict[str, Any]] = []
    totals = [int(levels.eq(level).sum()) for level in spec.exposure_levels]
    events = [
        int(outcome[levels.eq(level)].eq(1.0).sum()) for level in spec.exposure_levels
    ]
    family: list[dict[str, Any]] = []
    try:
        trend = cochran_armitage_trend(events, totals, group_order=list(spec.exposure_levels))
        rows.append(
            {
                "test_id": "death_trend_cochran_armitage",
                "outcome": spec.outcome,
                "population": "exposure_known_landmark_rows",
                "in_holm_family": True,
                **trend.as_dict(),
            }
        )
        family.append(rows[-1])
    except ValueError as exc:
        rows.append(
            {
                "test_id": "death_trend_cochran_armitage",
                "outcome": spec.outcome,
                "population": "exposure_known_landmark_rows",
                "in_holm_family": True,
                "test_name": "Cochran-Armitage trend test",
                "p_value": None,
                "note": str(exc),
            }
        )
    for secondary in spec.secondary_outcomes:
        values = pd.to_numeric(frame[secondary.name], errors="coerce")
        populations = [("exposure_known_rows", pd.Series(True, index=frame.index), True)]
        if secondary.report_survivors_separately:
            populations.append(("exposure_known_survivors", outcome.eq(0.0), False))
        for population, population_mask, in_family in populations:
            mask = population_mask & values.notna() & levels.ne(spec.unknown_level_label)
            test_id = f"{secondary.name}_trend_jonckheere_terpstra_{population}"
            try:
                result = jonckheere_terpstra_trend(
                    values[mask].to_numpy(dtype=float),
                    levels[mask].tolist(),
                    group_order=list(spec.exposure_levels),
                )
                rows.append(
                    {
                        "test_id": test_id,
                        "outcome": secondary.name,
                        "population": population,
                        "in_holm_family": in_family,
                        **result.as_dict(),
                    }
                )
                if in_family:
                    family.append(rows[-1])
            except ValueError as exc:
                rows.append(
                    {
                        "test_id": test_id,
                        "outcome": secondary.name,
                        "population": population,
                        "in_holm_family": in_family,
                        "test_name": "Jonckheere-Terpstra trend test",
                        "p_value": None,
                        "note": str(exc),
                    }
                )
    table = pd.DataFrame(rows)
    table["holm_family_size"] = len(family)
    table["p_value_holm"] = np.nan
    family_p = [row["p_value"] for row in family if row.get("p_value") is not None]
    if family_p:
        adjusted = multipletests(family_p, method="holm")[1]
        adjusted_iter = iter(float(value) for value in adjusted)
        for index, row in table.iterrows():
            if bool(row["in_holm_family"]) and pd.notna(row.get("p_value")):
                table.at[index, "p_value_holm"] = next(adjusted_iter)
    table["p_value_holm"] = pd.to_numeric(table["p_value_holm"], errors="coerce")
    return table


def _measurement_audit(
    frame: pd.DataFrame, spec: LandmarkCategoricalSpec
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outcome = pd.to_numeric(frame[spec.outcome], errors="coerce")
    for column in [spec.exposure, *spec.alternate_exposures]:
        spelled = _known_levels(frame, column)
        known = spelled.ne("")
        rows.append(
            {
                "audit_item": "exposure_ascertainment",
                "column": column,
                "role": "primary_exposure" if column == spec.exposure else "alternate_exposure",
                "n_landmark": int(len(frame)),
                "n_known": int(known.sum()),
                "n_unknown": int((~known).sum()),
                "share_unknown": float((~known).mean()) if len(frame) else None,
                "events_known": int(outcome[known].eq(1.0).sum()),
                "events_unknown": int(outcome[~known].eq(1.0).sum()),
                "event_rate_known": _finite(outcome[known].eq(1.0).mean()) if known.any() else None,
                "event_rate_unknown": _finite(outcome[~known].eq(1.0).mean()) if (~known).any() else None,
                "value_summary": "; ".join(
                    f"{level}={int(spelled.eq(level).sum())}" for level in spec.exposure_levels
                ),
            }
        )
    for covariate in spec.covariates:
        series = frame[covariate.name]
        rows.append(
            {
                "audit_item": "covariate_completeness",
                "column": covariate.name,
                "role": "covariate",
                "n_landmark": int(len(frame)),
                "n_known": int(series.notna().sum()),
                "n_unknown": int(series.isna().sum()),
                "share_unknown": float(series.isna().mean()) if len(frame) else None,
                "events_known": None,
                "events_unknown": None,
                "event_rate_known": None,
                "event_rate_unknown": None,
                "value_summary": "",
            }
        )
    for column in spec.measurement_audit_columns:
        if column not in frame.columns:
            rows.append(
                {
                    "audit_item": "measurement_process",
                    "column": column,
                    "role": "declared_but_absent",
                    "n_landmark": int(len(frame)),
                    "n_known": 0,
                    "n_unknown": int(len(frame)),
                    "share_unknown": 1.0,
                    "events_known": None,
                    "events_unknown": None,
                    "event_rate_known": None,
                    "event_rate_unknown": None,
                    "value_summary": "column absent from cohort",
                }
            )
            continue
        series = frame[column]
        distinct = series.dropna().unique()
        if len(distinct) <= 12:
            counts = series.value_counts(dropna=False)
            summary = "; ".join(
                f"{('missing' if (isinstance(k, float) and math.isnan(k)) else level_spelling(k))}={int(v)}"
                for k, v in counts.items()
            )
        else:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            summary = (
                f"median={numeric.median():.3g}; q25={numeric.quantile(0.25):.3g}; "
                f"q75={numeric.quantile(0.75):.3g}"
                if len(numeric)
                else "non-numeric high-cardinality column"
            )
        rows.append(
            {
                "audit_item": "measurement_process",
                "column": column,
                "role": "measurement_audit",
                "n_landmark": int(len(frame)),
                "n_known": int(series.notna().sum()),
                "n_unknown": int(series.isna().sum()),
                "share_unknown": float(series.isna().mean()) if len(frame) else None,
                "events_known": None,
                "events_unknown": None,
                "event_rate_known": None,
                "event_rate_unknown": None,
                "value_summary": summary,
            }
        )
    return pd.DataFrame(rows)


def _functional_form(
    frame: pd.DataFrame,
    spec: LandmarkCategoricalSpec,
    primary: dict[str, Any],
) -> pd.DataFrame:
    """Refit the primary model with one continuous covariate as an RCS."""

    rows: list[dict[str, Any]] = []
    primary_or = _finite(primary.get("estimate"))
    for covariate in spec.functional_form_covariates:
        row: dict[str, Any] = {
            "covariate": covariate,
            "spline": "restricted_cubic_spline",
            "n_knots": int(spec.functional_form_knots),
            "knots": "",
            "n": None,
            "n_events": None,
            "primary_contrast_or": None,
            "primary_contrast_ci_low": None,
            "primary_contrast_ci_high": None,
            "primary_or_linear_covariate": primary_or,
            "log_or_delta_vs_linear": None,
            "nonlinearity_wald_statistic": None,
            "nonlinearity_df": None,
            "nonlinearity_p_value": None,
            "nonlinearity_covariance": "model_based_without_clustering",
            "variance_estimator": None,
            "fit_status": "not_estimable",
            "note": "",
        }
        needed = [
            spec.outcome,
            spec.exposure,
            *spec.covariate_names(),
        ]
        if spec.dependence is not None:
            needed.append(spec.dependence.group_source)
        complete = frame.dropna(subset=list(dict.fromkeys(needed)))
        try:
            compiled = compile_model_terms(
                complete,
                terms=spec.model_terms(exclude_covariates=(covariate,)),
                exposure=spec.exposure,
            )
            basis = rcs_basis(
                complete[covariate].to_numpy(dtype=float),
                n_knots=spec.functional_form_knots,
            )
        except (RCSError, ValueError) as exc:
            row["note"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        spline = pd.DataFrame(
            np.asarray(basis.matrix, dtype=float),
            index=complete.index,
            columns=[
                covariate if name == "x" else f"{covariate}__rcs_{name}"
                for name in basis.column_names
            ],
        )
        design = pd.concat([compiled.design, spline], axis=1)
        source_by_column = dict(compiled.source_by_design_column)
        for name in spline.columns:
            source_by_column[name] = covariate
        groups, variance = _cluster_groups(complete, spec.dependence)
        y = pd.to_numeric(complete[spec.outcome], errors="coerce")
        result = fit_estimator(
            cohort=None,
            X=design,
            y=y,
            kind="logistic",
            term=_primary_contrast_column(spec, spec.exposure),
            source_by_design_column=source_by_column,
            variance_estimator=variance,
            cluster_groups=groups,
        )
        row.update(
            {
                "knots": "|".join(f"{value:.6g}" for value in basis.knots),
                "n": int(result.n),
                "n_events": result.n_events,
                "variance_estimator": result.variance_estimator,
            }
        )
        if not result.converged or _finite(result.point_estimate) is None:
            row["note"] = result.notes or "spline refit did not converge"
            rows.append(row)
            continue
        estimate = _finite(result.point_estimate)
        row.update(
            {
                "primary_contrast_or": estimate,
                "primary_contrast_ci_low": _finite(result.ci_low),
                "primary_contrast_ci_high": _finite(result.ci_high),
                "log_or_delta_vs_linear": (
                    math.log(estimate) - math.log(primary_or)
                    if estimate and primary_or and estimate > 0 and primary_or > 0
                    else None
                ),
                "fit_status": "fitted",
            }
        )
        try:
            spline_fit = rcs_fit(
                y.to_numpy(dtype=float),
                basis,
                covariates=compiled.design.to_numpy(dtype=float),
                covariate_names=list(compiled.design.columns),
                family="binomial",
            )
            wald = nonlinearity_wald_test(spline_fit)
            row.update(
                {
                    "nonlinearity_wald_statistic": _finite(wald.statistic),
                    "nonlinearity_df": int(wald.df),
                    "nonlinearity_p_value": _finite(wald.p_value),
                }
            )
        except (RCSError, ValueError) as exc:
            row["note"] = f"nonlinearity test not estimable: {exc}"
        rows.append(row)
    columns = [
        "covariate", "spline", "n_knots", "knots", "n", "n_events",
        "primary_contrast_or", "primary_contrast_ci_low", "primary_contrast_ci_high",
        "primary_or_linear_covariate", "log_or_delta_vs_linear",
        "nonlinearity_wald_statistic", "nonlinearity_df", "nonlinearity_p_value",
        "nonlinearity_covariance", "variance_estimator", "fit_status", "note",
    ]
    return pd.DataFrame(rows, columns=columns)


def _sensitivity_grid(
    cohort: LoadedCohort,
    spec: LandmarkCategoricalSpec,
    kernel_dir: Path,
    primary_row: dict[str, Any],
) -> pd.DataFrame:
    known = cohort.known()
    rows: list[dict[str, Any]] = [dict(primary_row)]
    for column in spec.alternate_exposures:
        spelled = _known_levels(cohort.frame, column)
        subset = cohort.frame.loc[spelled.ne("")]
        row, _summary, _table = _fit_variant(
            variant_id=f"alternate_exposure__{column}",
            axis="exposure_definition",
            frame=subset,
            spec=spec,
            exposure=column,
            restriction=f"rows with known {column}",
            kernel_dir=kernel_dir,
        )
        rows.append(row)
    if spec.first_stay_column:
        flags = spell_levels(known[spec.first_stay_column])
        subset = known.loc[flags.isin(["1", "true"])]
        row, _summary, _table = _fit_variant(
            variant_id="first_stay_only",
            axis="repeated_stays",
            frame=subset,
            spec=spec,
            exposure=spec.exposure,
            restriction=f"{spec.first_stay_column} in {{1, true}}",
            kernel_dir=kernel_dir,
        )
        rows.append(row)
    grid = pd.DataFrame(rows)
    primary_or = _finite(primary_row.get("estimate"))
    deltas: list[Optional[float]] = []
    consistent: list[Optional[bool]] = []
    for estimate in grid["estimate"].tolist():
        value = _finite(estimate)
        if value is None or primary_or is None or value <= 0 or primary_or <= 0:
            deltas.append(None)
            consistent.append(None)
            continue
        deltas.append(math.log(value) - math.log(primary_or))
        consistent.append((value > 1.0) == (primary_or > 1.0))
    grid["log_or_delta_vs_primary"] = deltas
    grid["direction_consistent_with_primary"] = consistent
    return grid


def _robustness_summary(
    result_rows: pd.DataFrame,
    functional_form: pd.DataFrame,
    trend: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in result_rows.iterrows():
        rows.append(
            {
                "analysis_id": row["variant_id"],
                "family": row["axis"],
                "n": row["n"],
                "n_events": row["n_events"],
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "effect_scale": row["effect_scale"],
                "fit_status": row["fit_status"],
                "direction_consistent_with_primary": row.get(
                    "direction_consistent_with_primary"
                ),
                "note": row.get("note", ""),
            }
        )
    for _, row in functional_form.iterrows():
        estimate = _finite(row["primary_contrast_or"])
        primary = _finite(row["primary_or_linear_covariate"])
        rows.append(
            {
                "analysis_id": f"functional_form__{row['covariate']}",
                "family": "functional_form",
                "n": row["n"],
                "n_events": row["n_events"],
                "estimate": estimate,
                "ci_low": _finite(row["primary_contrast_ci_low"]),
                "ci_high": _finite(row["primary_contrast_ci_high"]),
                "effect_scale": "odds_ratio",
                "fit_status": row["fit_status"],
                "direction_consistent_with_primary": (
                    ((estimate > 1.0) == (primary > 1.0))
                    if estimate is not None and primary is not None
                    else None
                ),
                "note": row["note"],
            }
        )
    for _, row in trend.iterrows():
        rows.append(
            {
                "analysis_id": row["requirement_id"],
                "family": "ordinal_trend",
                "n": row["n"],
                "n_events": row["n_events"],
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "effect_scale": row["effect_scale"],
                "fit_status": row["fit_status"],
                "direction_consistent_with_primary": None,
                "note": row["note"],
            }
        )
    return pd.DataFrame(rows)


def _flags_and_metrics(
    cohort: LoadedCohort,
    spec: LandmarkCategoricalSpec,
    primary_summary: dict[str, Any],
    primary_row: dict[str, Any],
    absolute_risk: pd.DataFrame,
    grid: pd.DataFrame,
    functional_form: pd.DataFrame,
    trend: pd.DataFrame,
    ordinal_tests: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = primary_summary["model_contracts"][0]
    n_fit = int(primary_summary["n_total"])
    n_events = int(primary_summary["n_events"])
    design_columns = [c for c in contract.get("design_columns", []) if c != "const"]
    n_parameters = len(design_columns)
    epv = (n_events / n_parameters) if n_parameters else None
    n_known = cohort.n_exposure_known
    n_dropped = n_known - n_fit
    complete_case_share_dropped = (n_dropped / n_known) if n_known else None
    unknown_share = (cohort.n_exposure_unknown / cohort.n_landmark) if cohort.n_landmark else None
    reference = absolute_risk.loc[absolute_risk["level"].eq(spec.reference_level)].iloc[0]
    sparse_levels = absolute_risk.loc[
        absolute_risk["in_primary_model"].astype(bool)
        & (absolute_risk["n_events"].astype(int) < spec.sparse_level_events)
    ]["level"].tolist()
    known_levels = absolute_risk.loc[absolute_risk["in_primary_model"].astype(bool)]
    empty_levels = known_levels.loc[known_levels["n"].astype(int) == 0]["level"].tolist()
    grid_consistency = grid.loc[
        grid["variant_id"].ne(PRIMARY_VARIANT_ID) & grid["fit_status"].eq("fitted")
    ]["direction_consistent_with_primary"].tolist()
    failed_variants = grid.loc[grid["fit_status"].ne("fitted")]["variant_id"].tolist()
    ff_not_estimable = functional_form.loc[
        functional_form["fit_status"].ne("fitted")
    ]["covariate"].tolist()
    flags: dict[str, Any] = {
        "epv": epv,
        "epv_below_minimum": (epv is not None and epv < spec.epv_minimum),
        "epv_minimum": spec.epv_minimum,
        "separation_detected": bool(contract.get("separation_detected")),
        "complete_case_share_dropped": complete_case_share_dropped,
        "complete_case_warning": (
            complete_case_share_dropped is not None
            and complete_case_share_dropped > spec.complete_case_warning_share
        ),
        "complete_case_warning_share": spec.complete_case_warning_share,
        "unknown_exposure_share": unknown_share,
        "unknown_exposure_share_high": (
            unknown_share is not None and unknown_share >= spec.unknown_exposure_warning_share
        ),
        "unknown_exposure_warning_share": spec.unknown_exposure_warning_share,
        "reference_group_n": int(reference["n"]),
        "small_reference_group": int(reference["n"]) < spec.small_reference_group_n,
        "small_reference_group_n": spec.small_reference_group_n,
        "sparse_exposure_levels": sparse_levels,
        "empty_exposure_levels": empty_levels,
        "sensitivity_direction_consistent": (
            all(bool(value) for value in grid_consistency) if grid_consistency else None
        ),
        "failed_sensitivity_variants": failed_variants,
        "functional_form_not_estimable": ff_not_estimable,
        "landmark_hours": float(spec.landmark_hours),
        "claim_ceiling": spec.claim_ceiling,
    }
    death_trend = ordinal_tests.loc[ordinal_tests["test_id"].eq("death_trend_cochran_armitage")]
    trend_row = trend.iloc[0] if len(trend) else None
    key_metrics: dict[str, Any] = {
        "skill": spec.schema_version,
        "title": spec.title,
        "n_source": cohort.n_source,
        "n_landmark": cohort.n_landmark,
        "n_exposure_known": cohort.n_exposure_known,
        "n_exposure_unknown": cohort.n_exposure_unknown,
        "n_fit": n_fit,
        "n_events_fit": n_events,
        "n_dropped_missing_covariates": n_dropped,
        "cluster_count": primary_summary.get("cluster_count"),
        "variance_estimator": primary_summary.get("variance_estimator"),
        "exposure": spec.exposure,
        "reference_level": spec.reference_level,
        "primary_contrast_level": spec.primary_contrast_level,
        "primary_or": _finite(primary_row.get("estimate")),
        "primary_or_ci_low": _finite(primary_row.get("ci_low")),
        "primary_or_ci_high": _finite(primary_row.get("ci_high")),
        "adjusted_trend_or_per_level": _finite(trend_row["estimate"]) if trend_row is not None else None,
        "adjusted_trend_ci_low": _finite(trend_row["ci_low"]) if trend_row is not None else None,
        "adjusted_trend_ci_high": _finite(trend_row["ci_high"]) if trend_row is not None else None,
        "death_trend_p_value": _finite(death_trend["p_value"].iloc[0]) if len(death_trend) else None,
        "death_trend_p_value_holm": _finite(death_trend["p_value_holm"].iloc[0]) if len(death_trend) else None,
        "covariates": ";".join(spec.covariate_names()),
        "n_model_parameters": n_parameters,
        "epv": epv,
        "epv_below_minimum": flags["epv_below_minimum"],
        "separation_detected": flags["separation_detected"],
        "complete_case_warning": flags["complete_case_warning"],
        "unknown_exposure_share": unknown_share,
        "unknown_exposure_share_high": flags["unknown_exposure_share_high"],
        "small_reference_group": flags["small_reference_group"],
        "sparse_exposure_levels": "|".join(sparse_levels),
        "sensitivity_direction_consistent": flags["sensitivity_direction_consistent"],
        "failed_sensitivity_variants": "|".join(failed_variants),
        "functional_form_not_estimable": "|".join(ff_not_estimable),
        "landmark_hours": float(spec.landmark_hours),
        "claim_ceiling": spec.claim_ceiling,
    }
    return flags, key_metrics


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run_analysis(
    cohort: LoadedCohort,
    *,
    work_dir: str | Path,
    verbose: bool = True,
) -> AnalysisResult:
    """Run the fixed standard workflow and return every table plus the ledger."""

    spec = cohort.spec
    kernel_dir = Path(work_dir) / "_kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    frame = cohort.frame
    known = cohort.known()

    if verbose:
        print("\n=== Running fixed-landmark categorical association ===\n")
        print("1. Exposure ascertainment, absolute risks and Table 1...")
    level_counts = pd.DataFrame(
        [
            row
            for column in [spec.exposure, *spec.alternate_exposures]
            for row in _level_counts(frame, column, spec)
        ]
    )
    measurement_audit = _measurement_audit(frame, spec)
    absolute_risk = pd.DataFrame(_absolute_risk_rows(frame, spec))
    secondary_summary = pd.DataFrame(_secondary_summary_rows(frame, spec))
    table_one = _table_one(frame, spec)

    if verbose:
        print("2. Fitting the declared adjusted model (primary contrast "
              f"{spec.primary_contrast_level} vs {spec.reference_level})...")
    primary_row, primary_summary, primary_estimates = _fit_variant(
        variant_id=PRIMARY_VARIANT_ID,
        axis="primary",
        frame=known,
        spec=spec,
        exposure=spec.exposure,
        restriction="exposure known at landmark",
        kernel_dir=kernel_dir,
    )
    if primary_summary is None or primary_estimates is None or primary_row["fit_status"] != "fitted":
        raise AnalysisContractError(
            "the declared primary model could not be fitted: " + str(primary_row["note"])
        )
    coefficients = pd.read_csv(
        kernel_dir / PRIMARY_VARIANT_ID / "adjusted_association_coefficients.csv"
    )

    if verbose:
        print("3. Adjusted per-level trend and ordered trend tests...")
    adjusted_trend = _adjusted_trend(known, spec, kernel_dir)
    ordinal_tests = _ordinal_trend_tests(frame, spec)

    if verbose:
        print("4. Prespecified sensitivity refits (definitions, repeated stays)...")
    grid = _sensitivity_grid(cohort, spec, kernel_dir, primary_row)

    if verbose:
        print("5. Functional-form checks (restricted cubic splines)...")
    functional_form = _functional_form(known, spec, primary_row)

    robustness = _robustness_summary(grid, functional_form, adjusted_trend)
    flags, key_metrics = _flags_and_metrics(
        cohort,
        spec,
        primary_summary,
        primary_row,
        absolute_risk,
        grid,
        functional_form,
        adjusted_trend,
        ordinal_tests,
    )

    result = AnalysisResult(
        spec=spec,
        cohort=cohort,
        kernel_dir=kernel_dir,
        cohort_flow=cohort.flow.copy(),
        exposure_level_counts=level_counts,
        measurement_audit=measurement_audit,
        table_one=table_one,
        absolute_risk=absolute_risk,
        adjusted_association_estimates=primary_estimates,
        adjusted_association_coefficients=coefficients,
        adjusted_trend=adjusted_trend,
        ordinal_trend_tests=ordinal_tests,
        secondary_outcome_summary=secondary_summary,
        association_sensitivity_grid=grid,
        functional_form_sensitivity=functional_form,
        robustness_summary=robustness,
        primary_summary=primary_summary,
        key_metrics=key_metrics,
        caveat_flags=flags,
        notes=list(cohort.notes),
    )
    if verbose:
        _print_headline(result)
    return result


def _print_headline(result: AnalysisResult) -> None:
    metrics = result.key_metrics
    flags = result.caveat_flags
    print()
    print(ANALYSIS_TOKEN)
    print(
        f"  Landmark rows: {metrics['n_landmark']} "
        f"(exposure known {metrics['n_exposure_known']}, unknown {metrics['n_exposure_unknown']})"
    )
    print(
        f"  Primary model: n={metrics['n_fit']}, events={metrics['n_events_fit']}, "
        f"clusters={metrics['cluster_count']}, EPV={metrics['epv']:.1f}"
        if metrics.get("epv") is not None
        else f"  Primary model: n={metrics['n_fit']}, events={metrics['n_events_fit']}"
    )
    print(
        f"  OR {metrics['primary_contrast_level']} vs {metrics['reference_level']}: "
        f"{metrics['primary_or']:.3f} (95% CI {metrics['primary_or_ci_low']:.3f}-"
        f"{metrics['primary_or_ci_high']:.3f})"
    )
    if metrics.get("adjusted_trend_or_per_level") is not None:
        print(
            f"  Adjusted OR per level increment: {metrics['adjusted_trend_or_per_level']:.3f} "
            f"(95% CI {metrics['adjusted_trend_ci_low']:.3f}-{metrics['adjusted_trend_ci_high']:.3f})"
        )
    raised = [
        name
        for name in (
            "epv_below_minimum",
            "separation_detected",
            "complete_case_warning",
            "unknown_exposure_share_high",
            "small_reference_group",
        )
        if flags.get(name)
    ]
    if flags.get("sparse_exposure_levels"):
        raised.append("sparse_exposure_levels=" + ",".join(flags["sparse_exposure_levels"]))
    if flags.get("sensitivity_direction_consistent") is False:
        raised.append("sensitivity_direction_inconsistent")
    print("  Caveat flags raised: " + (", ".join(raised) if raised else "none"))


__all__ = [
    "ANALYSIS_TOKEN",
    "METHOD_FAMILY",
    "PRIMARY_VARIANT_ID",
    "AnalysisContractError",
    "AnalysisResult",
    "run_analysis",
]
