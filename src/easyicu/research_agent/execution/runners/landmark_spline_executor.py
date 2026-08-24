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
from ...contracts.host_scaffold import HostScaffoldedScript
from ...schema import AnalysisPlan, AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input

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
    if not outcome_values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError("signed landmark outcome is not binary")
    event_without_time = working[sealed.outcome_column].eq(1) & working[
        sealed.outcome_time_column
    ].isna()
    if bool(event_without_time.any()):
        raise ValueError(
            "signed landmark population cannot verify event timing for every death"
        )

    observation_threshold = sealed.landmark_hours / 24.0
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
    spline.columns = [f"lactate_rcs_{index + 1}" for index in range(spline.shape[1])]
    spline.index = model_frame.index
    design = pd.concat(
        [spline, model_frame.drop(columns=["__exposure", "__outcome"])], axis=1
    ).astype(float)
    design = sm.add_constant(design, has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("signed landmark spline design is rank deficient")
    fit = sm.GLM(
        model_frame["__outcome"].astype(float),
        design,
        family=sm.families.Binomial(),
    ).fit(maxiter=200, disp=0)
    if not bool(getattr(fit, "converged", False)):
        raise ValueError("signed landmark spline model did not converge")

    curve_values = np.linspace(lower, upper, sealed.curve_points)
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
    for x_value, basis_row in zip(curve_values, np.asarray(curve_basis)):
        delta = np.asarray(basis_row, dtype=float) - reference_basis
        log_or = float(delta @ beta)
        variance = max(float(delta @ covariance @ delta), 0.0)
        se = math.sqrt(variance)
        curve_rows.append(
            {
                "lactate_mmol_l": _finite(x_value),
                "reference_lactate_mmol_l": _finite(reference),
                "adjusted_odds_ratio": _finite(math.exp(log_or)),
                "ci_low": _finite(math.exp(log_or - 1.96 * se)),
                "ci_high": _finite(math.exp(log_or + 1.96 * se)),
            }
        )

    contrast_rows = [
        row
        for row in curve_rows
        if math.isclose(row["lactate_mmol_l"], lower)
        or math.isclose(row["lactate_mmol_l"], upper)
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
    linear_fit = sm.GLM(
        model_frame["__outcome"].astype(float),
        linear_design,
        family=sm.families.Binomial(),
    ).fit(maxiter=200, disp=0)
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

    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "e2_landmark_rcs_curve.csv"
    pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
    contrasts_path = out_dir / "e2_landmark_rcs_contrasts.csv"
    pd.DataFrame(contrast_rows).to_csv(contrasts_path, index=False)
    sensitivity_path = out_dir / "e2_linear_sensitivity.csv"
    pd.DataFrame(
        [
            {
                "per_mmol_l": sealed.linear_sensitivity_per_unit,
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
    receipt = {
        "schema_version": "easyicu.landmark_spline_runtime_receipt/1",
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
    receipt_path = out_dir / "e2_scientific_runtime_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
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
        "output_files": {
            "table:e2_landmark_rcs_curve": curve_path.name,
            "table:e2_landmark_rcs_contrasts": contrasts_path.name,
            "table:e2_linear_sensitivity": sensitivity_path.name,
            "log:e2_scientific_runtime_receipt": receipt_path.name,
        },
    }


__all__ = [
    "LANDMARK_SPLINE_ANALYSIS_KIND",
    "landmark_spline_executor_code",
    "landmark_spline_executor_owns_step",
    "run_landmark_spline_association",
]
