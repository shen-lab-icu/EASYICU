"""Deterministic executor for one digest-bound landmark spline association."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any, Mapping, Optional

from ...authority.current_case_scientific_runtime import (
    LandmarkSplineRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.capability_ids import LANDMARK_SPLINE_ANALYSIS_KIND
from ...contracts.dependence import resolve_patient_groups
from ...contracts.host_scaffold import HostScaffoldedScript
from ...schema import AnalysisPlan, AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input


def _product_path(out_dir: Path, product: str) -> Path:
    kind, separator, name = product.partition(":")
    if separator != ":" or not name:
        raise ValueError(f"invalid landmark spline product: {product!r}")
    extension = ".csv" if kind == "table" else ".json"
    return out_dir / f"{name}{extension}"


def landmark_spline_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: LandmarkSplineRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    if authority is None:
        return False
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSplineRuntimeAuthority):
        return False
    return sealed.governed_step(plan) == step


def landmark_spline_executor_scaffold(
    step: AnalysisStep,
    *,
    authority: LandmarkSplineRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> HostScaffoldedScript:
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSplineRuntimeAuthority):
        raise TypeError("landmark executor requires landmark spline authority")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    typed_cohort_input = sole_typed_cohort_input(step)
    if typed_cohort_input is None:
        raise ValueError("landmark spline step requires one typed cohort input")
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope, frame_name="frame"
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    prologue = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.landmark_spline_executor import (
            run_landmark_spline_association,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_step_cohort_frame,
        )

        typed_cohort_input = {typed_cohort_input!r}
        authority = json.loads({json.dumps(authority_json)})
        frame, cohort_path = load_step_cohort_frame(
            typed_cohort_input=typed_cohort_input,
        )
        """
    ).strip()
    if receipt_code:
        prologue += "\n\n" + receipt_code.strip()
    prologue += "\n\n" + textwrap.dedent(
        f"""
        summary = run_landmark_spline_association(
            frame=frame,
            authority=authority,
            runtime_projection_sha256={runtime_projection_sha256!r},
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            source_cohort=cohort_path,
        )
        """
    ).strip()
    epilogue = []
    if receipt_code:
        epilogue.append('summary["plausibility_audit"] = plausibility_audit')
    epilogue.extend(
        [
            'out_dir = Path(os.environ["STEP_OUT_DIR"])',
            '(out_dir / "step_summary.json").write_text(',
            "    json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),",
            '    encoding="utf-8",',
            ")",
            "print(json.dumps(summary, ensure_ascii=False, allow_nan=False))",
        ]
    )
    return HostScaffoldedScript(prologue=prologue, body="", epilogue="\n".join(epilogue))


def landmark_spline_executor_code(
    step: AnalysisStep,
    *,
    authority: LandmarkSplineRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> str:
    return landmark_spline_executor_scaffold(
        step,
        authority=authority,
        runtime_projection_sha256=runtime_projection_sha256,
        plausibility_scope=plausibility_scope,
    ).assembled()


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("landmark spline fit produced a non-finite value")
    return number


def _adjustment_design(frame, authority: LandmarkSplineRuntimeAuthority):
    import pandas as pd

    pieces = []
    for column in authority.required_adjustment_columns:
        source = frame[column]
        if column in authority.categorical_adjustment_columns:
            encoded = pd.get_dummies(
                source.astype("string"), prefix=column, drop_first=True, dtype=float
            )
            encoded.loc[source.isna(), :] = float("nan")
            if encoded.empty:
                raise ValueError(
                    f"categorical adjustment {column!r} has fewer than two levels"
                )
            pieces.append(encoded)
        else:
            pieces.append(
                pd.DataFrame(
                    {column: pd.to_numeric(source, errors="coerce")},
                    index=frame.index,
                )
            )
    return pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=frame.index)


def _fit_binomial_model(*, sm, outcome, design, source_frame, authority):
    """Fit one authority-bound model with the declared covariance estimator."""

    model = sm.GLM(outcome.astype(float), design, family=sm.families.Binomial())
    dependence = authority.dependence
    if dependence is None:
        return model.fit(maxiter=200, disp=0), None
    group_values = source_frame.loc[design.index, dependence.group_source]
    if bool(group_values.isna().any()):
        raise ValueError("signed landmark cluster group contains missing values")
    resolved = resolve_patient_groups(group_values.tolist(), requirement=dependence)
    fit = model.fit(
        maxiter=200,
        disp=0,
        cov_type="cluster",
        cov_kwds={"groups": list(resolved.groups)},
    )
    return fit, resolved.cluster_count


def run_landmark_spline_association(
    *,
    frame: Any,
    authority: LandmarkSplineRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    out_dir: Path,
    source_cohort: Any = None,
) -> dict[str, Any]:
    """Fit exactly the signed 24-hour RCS primary and linear sensitivity."""

    import numpy as np
    import pandas as pd
    import patsy
    import statsmodels.api as sm
    from scipy.special import expit, logit
    from scipy.stats import chi2

    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSplineRuntimeAuthority):
        raise TypeError("landmark executor received the wrong authority kind")
    if len(str(runtime_projection_sha256)) != 64:
        raise ValueError("runtime projection digest is required")
    missing = sorted(set(sealed.required_columns) - set(frame.columns))
    if missing:
        raise ValueError("signed landmark input lacks columns: " + ", ".join(missing))

    working = frame[list(sealed.required_columns)].copy()
    for column in (
        sealed.exposure_column,
        sealed.outcome_column,
        sealed.outcome_time_column,
        sealed.observation_duration_column,
    ):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    outcome_values = set(working[sealed.outcome_column].dropna().unique().tolist())
    if not outcome_values.issubset({0, 1}):
        raise ValueError("signed landmark outcome is not binary")
    event_without_time = working[sealed.outcome_column].eq(1) & working[
        sealed.outcome_time_column
    ].isna()
    if bool(event_without_time.any()):
        raise ValueError(
            "signed landmark population cannot verify event timing for every death"
        )

    observation_threshold = sealed.observation_threshold
    alive_at_landmark = working[sealed.outcome_column].eq(0) | working[
        sealed.outcome_time_column
    ].gt(sealed.landmark_hours)
    under_observation = working[sealed.observation_duration_column].ge(
        observation_threshold
    )
    valid_exposure = working[sealed.exposure_column].notna()
    primary_mask = alive_at_landmark & under_observation & valid_exposure
    primary = working.loc[primary_mask].copy()
    adjustment = _adjustment_design(primary, sealed)
    outcome = primary[sealed.outcome_column]
    exposure = primary[sealed.exposure_column]
    model_frame = pd.concat(
        [exposure.rename("__exposure"), outcome.rename("__outcome"), adjustment],
        axis=1,
    ).dropna()
    if len(model_frame) < 30 or model_frame["__outcome"].nunique() != 2:
        raise ValueError("signed landmark primary population is not estimable")

    quantiles = model_frame["__exposure"].quantile(
        list(sealed.spline_knot_quantiles)
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(quantiles)) or not np.all(np.diff(quantiles) > 0):
        raise ValueError("signed landmark spline knots are not distinct")
    lower, reference, upper = [float(value) for value in quantiles]
    spline = patsy.dmatrix(
        (
            "cr(x, knots=(middle,), lower_bound=lower, upper_bound=upper, "
            "constraints='center') - 1"
        ),
        {
            "x": model_frame["__exposure"].to_numpy(dtype=float),
            "middle": reference,
            "lower": lower,
            "upper": upper,
        },
        return_type="dataframe",
    )
    spline.columns = [f"exposure_rcs_{index + 1}" for index in range(spline.shape[1])]
    spline.index = model_frame.index
    design = pd.concat(
        [spline, model_frame.drop(columns=["__exposure", "__outcome"])], axis=1
    ).astype(float)
    design = sm.add_constant(design, has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("signed landmark spline design is rank deficient")
    fit, primary_cluster_count = _fit_binomial_model(
        sm=sm,
        outcome=model_frame["__outcome"],
        design=design,
        source_frame=working,
        authority=sealed,
    )
    if not bool(getattr(fit, "converged", False)):
        raise ValueError("signed landmark spline model did not converge")

    curve_values = np.linspace(lower, upper, sealed.curve_points)
    # Publish the displayed complete-case exposure distribution on the same
    # grid as both model curves.  This lets the figure show where the cohort
    # actually contributes information without reopening patient rows or
    # inventing density from the fitted curve.  Values outside the prespecified
    # 10th--90th percentile display range remain outside the strip and the
    # denominator is recorded explicitly.
    density_edges = np.empty(len(curve_values) + 1, dtype=float)
    density_edges[1:-1] = (curve_values[:-1] + curve_values[1:]) / 2.0
    density_edges[0] = curve_values[0]
    density_edges[-1] = curve_values[-1]
    displayed_exposure = model_frame.loc[
        model_frame["__exposure"].between(lower, upper, inclusive="both"),
        "__exposure",
    ].to_numpy(dtype=float)
    density_counts, _ = np.histogram(displayed_exposure, bins=density_edges)
    density_display_n = int(density_counts.sum())
    if density_display_n <= 0:
        raise ValueError("signed landmark curve range has no exposure observations")
    density_fractions = density_counts.astype(float) / density_display_n
    curve_basis = patsy.build_design_matrices(
        [spline.design_info],
        {
            "x": curve_values,
            "middle": reference,
            "lower": lower,
            "upper": upper,
        },
    )[0]
    reference_basis = np.asarray(
        patsy.build_design_matrices(
            [spline.design_info],
            {
                "x": [reference],
                "middle": reference,
                "lower": lower,
                "upper": upper,
            },
        )[0]
    )[0]
    beta = fit.params.iloc[1 : 1 + spline.shape[1]].to_numpy(dtype=float)
    covariance = fit.cov_params().iloc[
        1 : 1 + spline.shape[1], 1 : 1 + spline.shape[1]
    ].to_numpy(dtype=float)
    curve_rows = []
    absolute_risk_rows = []
    fit_parameters = fit.params.to_numpy(dtype=float)
    fit_covariance = fit.cov_params().to_numpy(dtype=float)
    standardized_design = design.to_numpy(dtype=float, copy=True)
    for grid_index, (x_value, basis_row) in enumerate(
        zip(curve_values, np.asarray(curve_basis))
    ):
        delta = np.asarray(basis_row, dtype=float) - reference_basis
        log_or = float(delta @ beta)
        variance = max(float(delta @ covariance @ delta), 0.0)
        se = math.sqrt(variance)
        curve_rows.append(
            {
                "exposure": sealed.exposure_column,
                "exposure_value": _finite(x_value),
                "reference_exposure_value": _finite(reference),
                "adjusted_odds_ratio": _finite(math.exp(log_or)),
                "ci_low": _finite(math.exp(log_or - 1.96 * se)),
                "ci_high": _finite(math.exp(log_or + 1.96 * se)),
                "exposure_density_n": int(density_counts[grid_index]),
                "exposure_density_fraction": _finite(
                    density_fractions[grid_index]
                ),
                "exposure_density_display_n": density_display_n,
                "exposure_density_population_n": int(len(model_frame)),
                "exposure_density_scope": (
                    "primary_complete_case_within_curve_range"
                ),
            }
        )
        standardized_design[:, 1 : 1 + spline.shape[1]] = np.asarray(
            basis_row, dtype=float
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            linear_predictor = standardized_design @ fit_parameters
        if not np.isfinite(linear_predictor).all():
            raise ValueError(
                "standardized absolute-risk linear predictor is non-finite"
            )
        probabilities = expit(linear_predictor)
        standardized_risk = float(probabilities.mean())
        gradient = (
            probabilities[:, None]
            * (1.0 - probabilities[:, None])
            * standardized_design
        ).mean(axis=0)
        risk_variance = max(float(gradient @ fit_covariance @ gradient), 0.0)
        risk_se = math.sqrt(risk_variance)
        bounded_risk = min(max(standardized_risk, 1e-12), 1.0 - 1e-12)
        logit_se = risk_se / (bounded_risk * (1.0 - bounded_risk))
        absolute_risk_rows.append(
            {
                "exposure": sealed.exposure_column,
                "exposure_value": _finite(x_value),
                "reference_exposure_value": _finite(reference),
                "adjusted_absolute_risk": _finite(standardized_risk),
                "ci_low": _finite(expit(logit(bounded_risk) - 1.96 * logit_se)),
                "ci_high": _finite(expit(logit(bounded_risk) + 1.96 * logit_se)),
                "standardization_n": int(len(model_frame)),
                "events": int(model_frame["__outcome"].sum()),
                "standardization_method": "marginal_over_primary_complete_case_covariates",
                "exposure_density_n": int(density_counts[grid_index]),
                "exposure_density_fraction": _finite(
                    density_fractions[grid_index]
                ),
                "exposure_density_display_n": density_display_n,
                "exposure_density_population_n": int(len(model_frame)),
                "exposure_density_scope": (
                    "primary_complete_case_within_curve_range"
                ),
            }
        )

    contrast_rows = [
        row
        for row in curve_rows
        if math.isclose(row["exposure_value"], lower)
        or math.isclose(row["exposure_value"], upper)
    ]
    linear_design = pd.concat(
        [
            model_frame[["__exposure"]].rename(
                columns={"__exposure": sealed.exposure_column}
            ),
            model_frame.drop(columns=["__exposure", "__outcome"]),
        ],
        axis=1,
    ).astype(float)
    linear_design = sm.add_constant(linear_design, has_constant="add")
    linear_fit, linear_cluster_count = _fit_binomial_model(
        sm=sm,
        outcome=model_frame["__outcome"],
        design=linear_design,
        source_frame=working,
        authority=sealed,
    )
    if linear_cluster_count != primary_cluster_count:
        raise ValueError("signed landmark primary and linear cluster populations drifted")
    if not bool(getattr(linear_fit, "converged", False)):
        raise ValueError("signed landmark linear sensitivity did not converge")
    coefficient = _finite(linear_fit.params[sealed.exposure_column])
    standard_error = _finite(linear_fit.bse[sealed.exposure_column])
    additional_parameters = int(fit.df_model - linear_fit.df_model)
    if additional_parameters <= 0:
        raise ValueError("signed spline model does not extend the linear model")
    likelihood_ratio = _finite(max(2.0 * (fit.llf - linear_fit.llf), 0.0))
    nonlinearity_p_value = _finite(
        chi2.sf(likelihood_ratio, additional_parameters)
    )
    sample_size = int(len(model_frame))
    spline_bic = _finite(-2.0 * fit.llf + len(fit.params) * math.log(sample_size))
    linear_bic = _finite(
        -2.0 * linear_fit.llf + len(linear_fit.params) * math.log(sample_size)
    )

    definition_rows = [
        {
            "exposure_column": sealed.exposure_column,
            "is_primary_definition": True,
            "exposure_increment": sealed.linear_sensitivity_per_unit,
            "adjusted_odds_ratio": _finite(math.exp(coefficient)),
            "ci_low": _finite(math.exp(coefficient - 1.96 * standard_error)),
            "ci_high": _finite(math.exp(coefficient + 1.96 * standard_error)),
            "n": sample_size,
            "events": int(model_frame["__outcome"].sum()),
        }
    ]
    for exposure_column in sealed.alternative_exposure_columns:
        definition_population = working.loc[
            alive_at_landmark
            & under_observation
            & working[exposure_column].notna()
        ].copy()
        definition_adjustment = _adjustment_design(definition_population, sealed)
        definition_frame = pd.concat(
            [
                definition_population[exposure_column].rename("__exposure"),
                definition_population[sealed.outcome_column].rename("__outcome"),
                definition_adjustment,
            ],
            axis=1,
        ).dropna()
        if len(definition_frame) < 30 or definition_frame["__outcome"].nunique() != 2:
            raise ValueError(
                f"alternative exposure definition {exposure_column!r} is not estimable"
            )
        definition_design = pd.concat(
            [
                definition_frame[["__exposure"]].rename(
                    columns={"__exposure": exposure_column}
                ),
                definition_frame.drop(columns=["__exposure", "__outcome"]),
            ],
            axis=1,
        ).astype(float)
        definition_design = sm.add_constant(definition_design, has_constant="add")
        if np.linalg.matrix_rank(definition_design.to_numpy()) != (
            definition_design.shape[1]
        ):
            raise ValueError(
                f"alternative exposure definition {exposure_column!r} is rank deficient"
            )
        definition_fit, _definition_cluster_count = _fit_binomial_model(
            sm=sm,
            outcome=definition_frame["__outcome"],
            design=definition_design,
            source_frame=working,
            authority=sealed,
        )
        if not bool(getattr(definition_fit, "converged", False)):
            raise ValueError(
                f"alternative exposure definition {exposure_column!r} did not converge"
            )
        definition_coefficient = _finite(definition_fit.params[exposure_column])
        definition_standard_error = _finite(definition_fit.bse[exposure_column])
        definition_rows.append(
            {
                "exposure_column": exposure_column,
                "is_primary_definition": False,
                "exposure_increment": sealed.linear_sensitivity_per_unit,
                "adjusted_odds_ratio": _finite(math.exp(definition_coefficient)),
                "ci_low": _finite(
                    math.exp(definition_coefficient - 1.96 * definition_standard_error)
                ),
                "ci_high": _finite(
                    math.exp(definition_coefficient + 1.96 * definition_standard_error)
                ),
                "n": int(len(definition_frame)),
                "events": int(definition_frame["__outcome"].sum()),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = _product_path(out_dir, sealed.curve_product)
    pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
    contrasts_path = _product_path(out_dir, sealed.downstream_parent_product)
    pd.DataFrame(contrast_rows).to_csv(contrasts_path, index=False)
    sensitivity_path = _product_path(out_dir, sealed.linear_sensitivity_product)
    pd.DataFrame(
        [
            {
                "exposure_increment": sealed.linear_sensitivity_per_unit,
                "adjusted_odds_ratio": _finite(math.exp(coefficient)),
                "ci_low": _finite(math.exp(coefficient - 1.96 * standard_error)),
                "ci_high": _finite(math.exp(coefficient + 1.96 * standard_error)),
                "n": sample_size,
                "events": int(model_frame["__outcome"].sum()),
                "linear_aic": _finite(linear_fit.aic),
                "spline_aic": _finite(fit.aic),
                "linear_bic": linear_bic,
                "spline_bic": spline_bic,
                "likelihood_ratio_statistic": likelihood_ratio,
                "additional_spline_parameters": additional_parameters,
                "nonlinearity_p_value": nonlinearity_p_value,
            }
        ]
    ).to_csv(sensitivity_path, index=False)
    definition_path = None
    if sealed.exposure_definition_sensitivity_product is not None:
        definition_path = _product_path(
            out_dir, sealed.exposure_definition_sensitivity_product
        )
        pd.DataFrame(definition_rows).to_csv(definition_path, index=False)
    absolute_risk_path = None
    if sealed.adjusted_absolute_risk_product is not None:
        absolute_risk_path = _product_path(
            out_dir, sealed.adjusted_absolute_risk_product
        )
        pd.DataFrame(absolute_risk_rows).to_csv(absolute_risk_path, index=False)

    landmark_eligible = alive_at_landmark & under_observation
    population_flow_rows = [
        {
            "stage": "source_cohort",
            "n": int(len(working)),
            "excluded_from_previous": 0,
            "population_rule": "all_digest_bound_source_rows",
        },
        {
            "stage": "alive_and_under_observation_at_landmark",
            "n": int(landmark_eligible.sum()),
            "excluded_from_previous": int(len(working) - landmark_eligible.sum()),
            "population_rule": "verified_alive_and_observed_at_landmark",
        },
        {
            "stage": "valid_exposure_primary_population",
            "n": int(primary_mask.sum()),
            "excluded_from_previous": int(
                landmark_eligible.sum() - primary_mask.sum()
            ),
            "population_rule": "landmark_population_with_valid_exposure",
        },
        {
            "stage": "complete_case_model_population",
            "n": int(len(model_frame)),
            "excluded_from_previous": int(primary_mask.sum() - len(model_frame)),
            "population_rule": "primary_population_with_complete_model_terms",
        },
    ]
    population_flow_path = None
    if sealed.population_flow_product is not None:
        population_flow_path = _product_path(out_dir, sealed.population_flow_product)
        pd.DataFrame(population_flow_rows).to_csv(population_flow_path, index=False)

    variable_opportunity_path = None
    variable_opportunity_summary = None
    if sealed.variable_opportunity_sensitivity_product is not None:
        variable_population = working.loc[
            valid_exposure & working[sealed.outcome_column].notna()
        ].copy()
        variable_adjustment = _adjustment_design(variable_population, sealed)
        variable_frame = pd.concat(
            [
                variable_population[sealed.exposure_column].rename("__exposure"),
                variable_population[sealed.outcome_column].rename("__outcome"),
                variable_adjustment,
            ],
            axis=1,
        ).dropna()
        variable_design = pd.concat(
            [
                variable_frame[["__exposure"]].rename(
                    columns={"__exposure": sealed.exposure_column}
                ),
                variable_frame.drop(columns=["__exposure", "__outcome"]),
            ],
            axis=1,
        ).astype(float)
        variable_design = sm.add_constant(variable_design, has_constant="add")
        if np.linalg.matrix_rank(variable_design.to_numpy()) != variable_design.shape[1]:
            raise ValueError("variable-opportunity sensitivity is rank deficient")
        variable_fit, _variable_cluster_count = _fit_binomial_model(
            sm=sm,
            outcome=variable_frame["__outcome"],
            design=variable_design,
            source_frame=working,
            authority=sealed,
        )
        if not bool(getattr(variable_fit, "converged", False)):
            raise ValueError("variable-opportunity sensitivity did not converge")
        variable_coefficient = _finite(
            variable_fit.params[sealed.exposure_column]
        )
        variable_standard_error = _finite(
            variable_fit.bse[sealed.exposure_column]
        )
        variable_opportunity_summary = {
            "population_rule": "all_rows_with_observed_exposure_and_complete_model_terms",
            "interpretation": "secondary_variable_opportunity_association_not_landmark_equivalent",
            "exposure_increment": sealed.linear_sensitivity_per_unit,
            "adjusted_odds_ratio": _finite(math.exp(variable_coefficient)),
            "ci_low": _finite(
                math.exp(variable_coefficient - 1.96 * variable_standard_error)
            ),
            "ci_high": _finite(
                math.exp(variable_coefficient + 1.96 * variable_standard_error)
            ),
            "n": int(len(variable_frame)),
            "events": int(variable_frame["__outcome"].sum()),
            "early_event_at_or_before_landmark_n": int(
                (
                    working[sealed.outcome_column].eq(1)
                    & working[sealed.outcome_time_column].le(sealed.landmark_hours)
                ).sum()
            ),
            "icu_observation_shorter_than_landmark_n": int(
                working[sealed.observation_duration_column]
                .lt(observation_threshold)
                .sum()
            ),
        }
        variable_opportunity_path = _product_path(
            out_dir, sealed.variable_opportunity_sensitivity_product
        )
        pd.DataFrame([variable_opportunity_summary]).to_csv(
            variable_opportunity_path, index=False
        )
    receipt = {
        "schema_version": (
            "easyicu.landmark_spline_runtime_receipt/3"
            if sealed.schema_version.endswith("/3")
            else (
                "easyicu.landmark_spline_runtime_receipt/2"
                if sealed.schema_version.endswith("/2")
                else "easyicu.landmark_spline_runtime_receipt/1"
            )
        ),
        "protocol_content_sha256": sealed.protocol_content_sha256,
        "execution_contract_sha256": sealed.execution_contract_sha256,
        "runtime_projection_sha256": runtime_projection_sha256,
        "landmark_hours": sealed.landmark_hours,
        "population_rule": "alive_and_under_observation_at_landmark_with_valid_exposure",
        "spline_knot_quantiles": list(sealed.spline_knot_quantiles),
        "observed_knots": [lower, reference, upper],
        "adjustment_columns": list(sealed.required_adjustment_columns),
        "primary_population_n": int(primary_mask.sum()),
        "complete_case_n": int(len(model_frame)),
        "events": int(model_frame["__outcome"].sum()),
        "functional_form_comparison": {
            "comparison": "restricted_cubic_spline_vs_linear",
            "likelihood_ratio_statistic": likelihood_ratio,
            "degrees_of_freedom": additional_parameters,
            "p_value": nonlinearity_p_value,
            "linear_aic": _finite(linear_fit.aic),
            "spline_aic": _finite(fit.aic),
            "linear_bic": linear_bic,
            "spline_bic": spline_bic,
        },
        "interpretation": sealed.interpretation,
    }
    if not sealed.schema_version.endswith("/1"):
        receipt["population_flow"] = population_flow_rows
        receipt["adjusted_absolute_risk"] = {
            "method": "marginal_standardization_over_primary_complete_case_covariates",
            "interval": "delta_method_logit_scale_95_percent_confidence_interval",
            "grid_rows": len(absolute_risk_rows),
        }
    if sealed.schema_version.endswith("/3"):
        assert sealed.dependence is not None
        receipt["variance_estimator"] = sealed.dependence.variance_estimator
        receipt["cluster_unit"] = sealed.dependence.cluster_unit
        receipt["cluster_group_source"] = sealed.dependence.group_source
        receipt["cluster_group_derivation"] = sealed.dependence.group_derivation
        receipt["cluster_group_delimiter"] = sealed.dependence.delimiter
        receipt["cluster_count"] = primary_cluster_count
    if variable_opportunity_summary is not None:
        receipt["variable_opportunity_sensitivity"] = variable_opportunity_summary
    receipt_path = _product_path(out_dir, sealed.receipt_product)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    output_files = {
        sealed.curve_product: curve_path.name,
        sealed.downstream_parent_product: contrasts_path.name,
        sealed.linear_sensitivity_product: sensitivity_path.name,
        sealed.receipt_product: receipt_path.name,
    }
    if (
        definition_path is not None
        and sealed.exposure_definition_sensitivity_product is not None
    ):
        output_files[sealed.exposure_definition_sensitivity_product] = (
            definition_path.name
        )
    if absolute_risk_path is not None and sealed.adjusted_absolute_risk_product:
        output_files[sealed.adjusted_absolute_risk_product] = absolute_risk_path.name
    if population_flow_path is not None and sealed.population_flow_product:
        output_files[sealed.population_flow_product] = population_flow_path.name
    if (
        variable_opportunity_path is not None
        and sealed.variable_opportunity_sensitivity_product
    ):
        output_files[sealed.variable_opportunity_sensitivity_product] = (
            variable_opportunity_path.name
        )
    return {
        "status": "ok",
        "analysis_family": "association",
        "interpretation_class": "descriptive_prognostic_association",
        "scientific_runtime_receipt": receipt,
        "source_cohort": (
            Path(source_cohort).name if source_cohort is not None else None
        ),
        "n_total": int(len(frame)),
        "n_primary_population": int(primary_mask.sum()),
        "n_complete_case": int(len(model_frame)),
        "n_events": int(model_frame["__outcome"].sum()),
        "variance_estimator": (
            sealed.dependence.variance_estimator
            if sealed.dependence is not None
            else "model_based"
        ),
        "cluster_count": primary_cluster_count,
        "output_files": output_files,
    }


__all__ = [
    "LANDMARK_SPLINE_ANALYSIS_KIND",
    "landmark_spline_executor_code",
    "landmark_spline_executor_owns_step",
    "run_landmark_spline_association",
]
