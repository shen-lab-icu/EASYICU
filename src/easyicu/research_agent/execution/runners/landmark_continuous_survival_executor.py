"""Deterministic fixed-landmark survival suite for one continuous exposure.

The caller-reviewed runtime authority
(``LandmarkContinuousSurvivalRuntimeAuthority``) owns every scientific
coordinate.  This module executes the sealed risk-set rule, a descriptive
Table 1 and Kaplan-Meier curves by exposure tertile, the adjusted Cox model
per exposure unit with its interval model, the restricted cubic spline check
of the linear term, and the proportional-hazards audit.  It contains no case
identifier and no model-editable code.  The composite figure is rendered by
``landmark_continuous_survival_figure`` from the tables written here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional

from ...authority.continuous_survival_scientific_claims import (
    CONTINUOUS_SURVIVAL_REPORTING_KEY,
    CONTINUOUS_SURVIVAL_REPORTING_SCHEMA_VERSION,
    PER_UNIT_HAZARD_RATIO_CLAIM_ID,
    interval_per_unit_hazard_ratio_claim_id,
)
from ...authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from ...authority.landmark_continuous_survival_runtime import (
    CONTINUOUS_SURVIVAL_PLAN_METHOD,
    LandmarkContinuousSurvivalRuntimeAuthority,
)
from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...authority.prespecified_rule_outcomes import RULE_OUTCOME_SCHEMA_VERSION
from ...contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
    LandmarkContinuousSurvivalDesign,
    executed_method_design_payload,
)
from ...contracts.host_scaffold import HostScaffoldedScript
from ...contracts.manuscript_result_structure import PRIMARY_RESULT_HEADINGS_BY_FAMILY
from ...contracts.manuscript_tables import (
    MANUSCRIPT_TABLE_SCHEMA_VERSION,
    MANUSCRIPT_TABLES_KEY,
    validate_manuscript_table_declarations,
)
from ...numeric_scalars import coerce_finite_float
from ...schema import AnalysisPlan, AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input

LANDMARK_CONTINUOUS_SURVIVAL_ANALYSIS_KIND = CONTINUOUS_SURVIVAL_PLAN_METHOD
#: The fewest rows and events an adjusted Cox fit of this suite rests on.
_MINIMUM_MODEL_ROWS = 100
_MINIMUM_MODEL_EVENTS = 10
#: The descriptive groups: value tertiles of the risk set, right-closed.
_TERTILE_PREFIXES = ("t1", "t2", "t3")
_TERTILE_NAMES = ("Lowest tertile", "Middle tertile", "Highest tertile")
#: The model column of the spline's one nonlinear term.
_SPLINE_TERM = "__exposure_rcs_s1"
_SPLINE_METHOD = "restricted_cubic_spline_likelihood_ratio_test"
_FILES = {
    "table_one": "continuous_landmark_table_one.csv",
    "risk_set": "continuous_landmark_risk_set_flow.csv",
    "km": "continuous_landmark_km_curve.csv",
    "cox": "continuous_landmark_cox_summary.csv",
    "ph": "continuous_landmark_ph_diagnostics.csv",
    "time_varying": "continuous_landmark_time_varying_cox_summary.csv",
    "spline": "continuous_landmark_spline_curve.csv",
    "measurement": "continuous_landmark_measurement_audit.csv",
    "analysis": "continuous_landmark_analysis_cohort.parquet",
    "receipt": "continuous_landmark_survival_runtime_receipt.json",
}


def _sealed(
    authority: LandmarkContinuousSurvivalRuntimeAuthority | Mapping[str, Any] | None,
) -> LandmarkContinuousSurvivalRuntimeAuthority | None:
    if authority is None:
        return None
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkContinuousSurvivalRuntimeAuthority):
        return None
    return sealed


def landmark_continuous_survival_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: LandmarkContinuousSurvivalRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    sealed = _sealed(authority)
    return sealed is not None and sealed.governed_step(plan) == step


def landmark_continuous_survival_executor_scaffold(
    step: AnalysisStep,
    *,
    authority: LandmarkContinuousSurvivalRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> HostScaffoldedScript:
    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("continuous survival executor requires its sealed authority")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    typed_input = sole_typed_cohort_input(step)
    if typed_input is None:
        raise ValueError("continuous survival suite requires one typed cohort input")
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope, frame_name="analysis_frame"
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    prologue = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.landmark_continuous_survival_executor import (
            run_landmark_continuous_survival_suite,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_typed_input,
            run_dir_from_env,
        )

        typed_cohort_input = {typed_input!r}
        authority = json.loads({json.dumps(authority_json)})
        bound = load_typed_input(
            input_key=typed_cohort_input,
            run_dir=run_dir_from_env(),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).resolve(),
            expected_evidence_kind="table",
            exclusive=True,
        )
        analysis_frame = bound.frame
        """
    ).strip()
    if receipt_code:
        prologue += "\n\n" + receipt_code.strip()
    prologue += (
        "\n\n"
        + textwrap.dedent(
            f"""
        summary = run_landmark_continuous_survival_suite(
            frame=analysis_frame,
            authority=authority,
            runtime_projection_sha256={runtime_projection_sha256!r},
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            input_product=bound.input_key,
            input_evidence_id=bound.evidence_id,
            input_sha256=bound.sha256,
        )
        """
        ).strip()
    )
    epilogue: list[str] = []
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
    return HostScaffoldedScript(
        prologue=prologue, body="", epilogue="\n".join(epilogue)
    )


def landmark_continuous_survival_executor_code(
    step: AnalysisStep,
    *,
    authority: LandmarkContinuousSurvivalRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> str:
    return landmark_continuous_survival_executor_scaffold(
        step,
        authority=authority,
        runtime_projection_sha256=runtime_projection_sha256,
        plausibility_scope=plausibility_scope,
    ).assembled()


def _canonical_frame_sha256(frame: Any) -> str:
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reader_words(text: str) -> str:
    """Plain reader words for a table caption or label (no markup characters)."""

    cleaned = re.sub(r"[{}\[\]<>`\\|*_#]+", " ", str(text))
    return " ".join(cleaned.split())[:200] or "Unnamed"


def _cox_fit(
    model_frame: Any, *, duration_col: str, event_col: str
) -> tuple[Any, Optional[str]]:
    """One Cox fit of a complete-case frame, and why it is no result.

    lifelines reports separation and non-convergence as warnings and still
    returns coefficients; that is no result.  The second value is the first
    such warning, else ``None``.
    """

    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitter = CoxPHFitter().fit(
            model_frame, duration_col=duration_col, event_col=event_col
        )
    nonconvergence = [
        str(item.message).split(". ", 1)[0]
        for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    return fitter, (nonconvergence[0] if nonconvergence else None)


def _tertile_groups(values: Any) -> tuple[Any, float, float]:
    """Right-closed value tertiles of the risk set's exposure, numbered 1 to 3.

    A cutpoint shared by two tertiles leaves one empty: the exposure then has
    too few distinct values for three groups, and the suite refuses rather
    than describe two groups as three.
    """

    import numpy as np
    import pandas as pd

    low, high = (float(value) for value in values.quantile([1.0 / 3.0, 2.0 / 3.0]))
    groups = np.where(values.le(low), 1, np.where(values.le(high), 2, 3))
    counts = np.bincount(groups, minlength=4)[1:]
    if (counts == 0).any():
        raise ValueError(
            "continuous survival exposure tertiles are tied; the exposure has too "
            "few distinct values for three descriptive groups"
        )
    return pd.Series(groups, index=values.index, dtype=int), low, high


def _tertile_labels(
    sealed: LandmarkContinuousSurvivalRuntimeAuthority, low: float, high: float
) -> dict[int, str]:
    unit = f" {sealed.exposure_unit}" if sealed.exposure_unit else ""
    return {
        1: f"{_TERTILE_NAMES[0]} ({low:.4g}{unit} or less)",
        2: f"{_TERTILE_NAMES[1]} (above {low:.4g} to {high:.4g}{unit})",
        3: f"{_TERTILE_NAMES[2]} (above {high:.4g}{unit})",
    }


def _table_one(frame: Any, groups: Any, sealed: LandmarkContinuousSurvivalRuntimeAuthority):
    """Characteristics by exposure tertile; the SMD compares the highest with the lowest."""

    import numpy as np
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for column in sealed.table_one_columns:
        source = frame[column]
        if column in sealed.categorical_adjustment_columns:
            for level in sorted(str(value) for value in source.dropna().unique()):
                row: dict[str, Any] = {
                    "variable": column,
                    "level": level,
                    "summary_type": "categorical_n_percent",
                }
                proportions: dict[int, float] = {}
                for group, prefix in enumerate(_TERTILE_PREFIXES, start=1):
                    subset = source.loc[groups.eq(group)]
                    denominator = int(len(subset))
                    count = int(subset.astype("string").eq(level).sum())
                    proportion = count / denominator if denominator else float("nan")
                    row[f"{prefix}_n"] = count
                    row[f"{prefix}_denominator"] = denominator
                    row[f"{prefix}_percent"] = (
                        100.0 * proportion if math.isfinite(proportion) else None
                    )
                    proportions[group] = proportion
                pooled = (proportions[1] + proportions[3]) / 2.0
                spread = math.sqrt(pooled * (1.0 - pooled))
                row["standardized_mean_difference"] = (
                    (proportions[3] - proportions[1]) / spread
                    if spread > 0 and math.isfinite(spread)
                    else None
                )
                rows.append(row)
            continue
        numeric = pd.to_numeric(source, errors="coerce")
        row = {"variable": column, "level": "", "summary_type": "continuous_mean_sd"}
        means: dict[int, float] = {}
        variances: dict[int, float] = {}
        for group, prefix in enumerate(_TERTILE_PREFIXES, start=1):
            values = numeric.loc[groups.eq(group)].dropna()
            row[f"{prefix}_n"] = int(len(values))
            row[f"{prefix}_mean"] = float(values.mean()) if len(values) else None
            row[f"{prefix}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            row[f"{prefix}_median"] = float(values.median()) if len(values) else None
            row[f"{prefix}_q1"] = float(values.quantile(0.25)) if len(values) else None
            row[f"{prefix}_q3"] = float(values.quantile(0.75)) if len(values) else None
            means[group] = float(values.mean()) if len(values) else float("nan")
            variances[group] = float(values.var(ddof=1)) if len(values) > 1 else float("nan")
        pooled_sd = math.sqrt(np.nanmean([variances[1], variances[3]]))
        row["standardized_mean_difference"] = (
            (means[3] - means[1]) / pooled_sd
            if pooled_sd > 0 and math.isfinite(pooled_sd)
            else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _km_table(analysis: Any, groups: Any, labels: Mapping[int, str], *, sealed):
    import pandas as pd

    from ...figures.base import km_estimate

    rows: list[dict[str, Any]] = []
    for group in (1, 2, 3):
        subset = analysis.loc[groups.eq(group)]
        estimate = km_estimate(
            subset[sealed.derived_time_column], subset[sealed.derived_event_column]
        )
        for time, survival, at_risk in zip(
            estimate["time"], estimate["survival"], estimate["at_risk"]
        ):
            rows.append(
                {
                    "exposure_group": group,
                    "exposure_group_label": labels[group],
                    "time_from_landmark_days": float(time),
                    "survival_probability": float(survival),
                    "at_risk": int(at_risk),
                    "group_n": int(estimate["n"]),
                    "group_events": int(estimate["n_events"]),
                }
            )
    return pd.DataFrame(rows)


def _model_frame(analysis: Any, sealed: LandmarkContinuousSurvivalRuntimeAuthority):
    """The complete-case model frame and its covariates, exposure first."""

    import pandas as pd

    numeric = [
        column
        for column in sealed.adjustment_columns
        if column not in sealed.categorical_adjustment_columns
    ]
    pieces = [
        analysis[
            [
                sealed.derived_time_column,
                sealed.derived_event_column,
                sealed.exposure_column,
            ]
        ],
        analysis[numeric].apply(pd.to_numeric, errors="coerce"),
    ]
    for column in sealed.categorical_adjustment_columns:
        encoded = pd.get_dummies(
            analysis[column].astype("string"), prefix=column, drop_first=True, dtype=float
        )
        encoded.loc[analysis[column].isna(), :] = float("nan")
        if encoded.empty:
            raise ValueError(
                f"continuous survival categorical column {column!r} has no contrast"
            )
        pieces.append(encoded)
    model_frame = pd.concat(pieces, axis=1).dropna().astype(float)
    if (
        len(model_frame) < _MINIMUM_MODEL_ROWS
        or int(model_frame[sealed.derived_event_column].sum()) < _MINIMUM_MODEL_EVENTS
    ):
        raise ValueError("continuous survival complete-case model is not estimable")
    covariates = [
        column
        for column in model_frame.columns
        if column not in {sealed.derived_time_column, sealed.derived_event_column}
    ]
    return model_frame, covariates


def _cox_table(fitter: Any):
    import pandas as pd

    summary = fitter.summary.reset_index().rename(columns={"covariate": "term"})
    if "term" not in summary.columns:
        summary = summary.rename(columns={summary.columns[0]: "term"})
    return pd.DataFrame(
        {
            "term": summary["term"].astype(str),
            "coefficient": summary["coef"].astype(float),
            "standard_error": summary["se(coef)"].astype(float),
            "hazard_ratio": summary["exp(coef)"].astype(float),
            "ci_low": summary["exp(coef) lower 95%"].astype(float),
            "ci_high": summary["exp(coef) upper 95%"].astype(float),
            "p_value": summary["p"].astype(float),
        }
    )


def _proportional_hazards_decision(
    model_frame: Any,
    *,
    sealed: LandmarkContinuousSurvivalRuntimeAuthority,
    covariates: list[str],
) -> tuple[Any, float, float, bool]:
    """The sealed PH rule: the exposure term's or the global test below alpha."""

    from ...methods.ph_schoenfeld import ph_test

    table = ph_test(
        model_frame,
        duration_col=sealed.derived_time_column,
        event_col=sealed.derived_event_column,
        covariates=covariates,
        time_transform="km",
    )
    terms = table["covariate"].astype(str)
    global_rows = table.loc[terms.eq("global"), "p_value"]
    exposure_rows = table.loc[terms.eq(sealed.exposure_column), "p_value"]
    if len(global_rows) != 1 or len(exposure_rows) != 1:
        raise ValueError("continuous survival PH audit lacks global or exposure result")
    global_p = coerce_finite_float(global_rows.iloc[0], label="global PH p")
    exposure_p = coerce_finite_float(exposure_rows.iloc[0], label="exposure PH p")
    rejected = min(global_p, exposure_p) < sealed.proportional_hazards_alpha
    return table, global_p, exposure_p, rejected


def _spline_check(
    model_frame: Any,
    *,
    sealed: LandmarkContinuousSurvivalRuntimeAuthority,
    linear_fitter: Any,
) -> tuple[Any, dict[str, Any]]:
    """The spline check of the linear term and the curves the figure draws.

    Harrell's three knots sit at the 10th, 50th and 90th percentiles of the
    modelled exposure.  The spline model adds the one nonlinear term to the
    linear model, so the two are nested and a likelihood-ratio test on one
    degree of freedom compares them.  Both curves are hazard ratios against
    the median, over the 10th to 90th percentile.  The check is reported; it
    chooses no estimate.  Tied knots or a spline fit without a result leave
    the linear curve alone and say why.
    """

    import numpy as np
    import pandas as pd
    from scipy.stats import chi2

    from ...methods.rcs_dose_response import rcs_basis

    exposure = model_frame[sealed.exposure_column]
    knots = [float(value) for value in exposure.quantile(list(sealed.spline_knot_quantiles))]
    reference = float(exposure.median())
    low, high = (float(value) for value in exposure.quantile(list(sealed.curve_quantile_range)))
    points = np.linspace(low, high, sealed.curve_points)
    linear_beta = float(linear_fitter.params_[sealed.exposure_column])
    linear_variance = float(
        linear_fitter.variance_matrix_.loc[sealed.exposure_column, sealed.exposure_column]
    )
    linear_log = linear_beta * (points - reference)
    linear_se = np.abs(points - reference) * math.sqrt(linear_variance)
    curve = pd.DataFrame(
        {
            "exposure_value": points,
            "reference_value": reference,
            "linear_hazard_ratio": np.exp(linear_log),
            "linear_ci_low": np.exp(linear_log - 1.96 * linear_se),
            "linear_ci_high": np.exp(linear_log + 1.96 * linear_se),
        }
    )
    not_estimable = None
    if any(later <= earlier for earlier, later in zip(knots, knots[1:])):
        not_estimable = "tied_knots"
    else:
        basis = rcs_basis(exposure.to_numpy(dtype=float), knots=knots)
        spline_frame = model_frame.copy()
        spline_frame[_SPLINE_TERM] = [row[1] for row in basis.matrix]
        spline_fitter, nonconvergence = _cox_fit(
            spline_frame,
            duration_col=sealed.derived_time_column,
            event_col=sealed.derived_event_column,
        )
        statistic = 2.0 * (
            float(spline_fitter.log_likelihood_) - float(linear_fitter.log_likelihood_)
        )
        if nonconvergence is not None or not math.isfinite(statistic):
            not_estimable = "spline_model_not_estimable"
    if not_estimable is not None:
        for column in ("spline_hazard_ratio", "spline_ci_low", "spline_ci_high"):
            curve[column] = None
        curve["spline_status"] = not_estimable
        return curve, {
            "method": _SPLINE_METHOD,
            "status": "not_estimable",
            "reason": not_estimable,
            "knot_percentiles": [100.0 * value for value in sealed.spline_knot_quantiles],
        }
    evaluation = rcs_basis(np.append(points, reference), knots=knots)
    nonlinear = np.asarray([row[1] for row in evaluation.matrix], dtype=float)
    terms = [sealed.exposure_column, _SPLINE_TERM]
    beta = spline_fitter.params_[terms].to_numpy(dtype=float)
    covariance = spline_fitter.variance_matrix_.loc[terms, terms].to_numpy(dtype=float)
    contrasts = np.column_stack((points - reference, nonlinear[:-1] - nonlinear[-1]))
    spline_log = contrasts @ beta
    spline_se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", contrasts, covariance, contrasts), 0.0))
    curve["spline_hazard_ratio"] = np.exp(spline_log)
    curve["spline_ci_low"] = np.exp(spline_log - 1.96 * spline_se)
    curve["spline_ci_high"] = np.exp(spline_log + 1.96 * spline_se)
    curve["spline_status"] = "estimated"
    statistic = max(statistic, 0.0)
    if not np.isfinite(curve[["spline_hazard_ratio", "spline_ci_low", "spline_ci_high"]].to_numpy(dtype=float)).all():
        raise ValueError("continuous survival spline curve is non-finite")
    return curve, {
        "method": _SPLINE_METHOD,
        "status": "estimated",
        "knot_percentiles": [100.0 * value for value in sealed.spline_knot_quantiles],
        "knots": knots,
        "reference_value": reference,
        "likelihood_ratio_statistic": statistic,
        "degrees_of_freedom": 1,
        "p_value": float(chi2.sf(statistic, 1)),
    }


def _manuscript_tables(
    sealed: LandmarkContinuousSurvivalRuntimeAuthority,
    analysis: Any,
    groups: Any,
    labels: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Declare the suite's Table 1 and risk-set accounting as reader tables."""

    sizes = groups.value_counts()
    events = analysis[sealed.derived_event_column].groupby(groups).sum()

    def counts(group: int) -> dict[str, Any]:
        n, deaths = int(sizes.get(group, 0)), int(events.get(group, 0))
        return {"n": n, "events": deaths, "events_percent": 100.0 * deaths / n if n else 0.0}

    exposure = _reader_words(sealed.exposure_label)
    declarations = [
        {
            "schema_version": MANUSCRIPT_TABLE_SCHEMA_VERSION,
            "product": sealed.table_one_product,
            "caption": _reader_words(
                f"Characteristics of the {sealed.analysis_unit_label} in the landmark "
                f"analysis cohort, by tertile of {exposure}"
            ),
            "body": {
                "layout": "grouped_summary",
                "groups": [
                    {"prefix": prefix, "label": _reader_words(labels[group]), **counts(group)}
                    for group, prefix in enumerate(_TERTILE_PREFIXES, start=1)
                ],
                "events_label": f"Deaths by day {sealed.endpoint_horizon_days:g}, n (%)",
            },
            "notes": [
                "Tertiles divide the landmark analysis cohort by the recorded exposure "
                "value; they describe the cohort and enter no model.",
                "Categorical percentages use every record of the tertile as the "
                "denominator, so levels need not sum to 100% when a value is missing.",
                "Continuous variables are summarized over their recorded values.",
                "The standardized mean difference compares the highest with the lowest "
                "tertile; it is not a significance test.",
                "Deaths are counted from the landmark to the end of follow-up.",
            ],
        },
        {
            "schema_version": MANUSCRIPT_TABLE_SCHEMA_VERSION,
            "product": sealed.risk_set_product,
            "caption": "Risk-set accounting from the source cohort to the landmark analysis cohort",
            "body": {
                "layout": "stage_flow",
                "stage_labels": {
                    "source_rows": "Source cohort",
                    "valid_fixed_horizon_endpoint": (
                        f"Valid {sealed.endpoint_horizon_days:g}-day endpoint"
                    ),
                    "alive_and_observed_at_landmark": (
                        f"Alive and under observation at the {sealed.landmark_hours:g}-hour landmark"
                    ),
                    "landmark_analysis_population": (
                        f"Landmark analysis cohort with a recorded {exposure} value"
                    ),
                },
            },
            "notes": [
                "The last stage excludes records without a recorded exposure value "
                "in its window.",
                "Excluded counts are the records removed since the stage before.",
            ],
        },
    ]
    return [
        declaration.model_dump(mode="json")
        for declaration in validate_manuscript_table_declarations(declarations)
    ]


def _measurement_audit_table(
    sealed: LandmarkContinuousSurvivalRuntimeAuthority, source: Any, landmark: Any
):
    """Availability of every sealed source column, in the source and at the landmark."""

    import pandas as pd

    roles = {
        sealed.exposure_column: "exposure",
        sealed.event_column: "event",
        sealed.followup_time_column: "followup_time",
        **{column: "adjustment" for column in sealed.adjustment_columns},
    }
    return pd.DataFrame(
        [
            {
                "column": column,
                "column_role": roles[column],
                "source_n": int(len(source)),
                "source_missing_n": int(source[column].isna().sum()),
                "landmark_population_n": int(len(landmark)),
                "landmark_missing_n": int(landmark[column].isna().sum()),
            }
            for column in sealed.required_columns
        ]
    )


def build_continuous_survival_manuscript_projection(
    *,
    interval_count: int,
    proportional_hazards_rejected: bool,
    functional_form_estimated: bool,
) -> dict[str, object]:
    """Build the reporting projection owned by the signed continuous suite.

    The per-unit hazard ratios and the PH decision are host scientific claims
    compiled from the envelope; host placement reports each in the survival
    results.  The projection adds the primary tokens to the abstract Results
    and one neutral numeric sentence on the spline check, which has no claim
    type: its likelihood-ratio statistic and degrees of freedom, not a p value
    the numeric binder could not trace below 0.001.
    """

    if interval_count <= 1:
        raise ValueError("continuous survival projection requires intervals")
    abstract = {"kind": "abstract_label", "label": "Results"}
    survival = {
        "kind": "markdown_heading",
        "label": PRIMARY_RESULT_HEADINGS_BY_FAMILY["survival"],
    }
    primary_claims = (
        tuple(
            interval_per_unit_hazard_ratio_claim_id(position)
            for position in range(1, interval_count + 1)
        )
        if proportional_hazards_rejected
        else (PER_UNIT_HAZARD_RATIO_CLAIM_ID,)
    )
    spline = (
        [
            {
                "claim_id": "restricted_cubic_spline_check",
                "targets": [survival],
                "fragments": [
                    {
                        "text": (
                            "The likelihood-ratio test of a restricted cubic spline of the "
                            "exposure against its linear term gave a chi-square statistic of "
                        )
                    },
                    {
                        "numeric_path": "functional_form.likelihood_ratio_statistic",
                        "format_spec": ".3f",
                    },
                    {"text": " on "},
                    {
                        "numeric_path": "functional_form.degrees_of_freedom",
                        "format_spec": ".0f",
                    },
                    {"text": " degree of freedom."},
                ],
            }
        ]
        if functional_form_estimated
        else []
    )
    return {
        "schema_version": "easyicu.manuscript_projection/2",
        "claims": [
            *spline,
            *(
                {
                    "claim_id": f"abstract_{claim_id}",
                    "targets": [abstract],
                    "scientific_claim_id": claim_id,
                }
                for claim_id in primary_claims
            ),
        ],
    }


def _executed_design(sealed: LandmarkContinuousSurvivalRuntimeAuthority) -> dict[str, Any]:
    """The design this run applied, read from the sealed contract it executed."""

    start, end = sealed.exposure_window_hours
    return executed_method_design_payload(
        LandmarkContinuousSurvivalDesign(
            schema_version=EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
            design_kind="landmark_continuous_survival",
            time_origin=sealed.endpoint_time_origin,
            landmark_hours=float(sealed.landmark_hours),
            endpoint_horizon_days=float(sealed.endpoint_horizon_days),
            exposure_window_start_hours=float(start),
            exposure_window_end_hours=float(end),
            exposure_window_summary=sealed.exposure_window_summary,
            exposure_increment=float(sealed.exposure_increment),
            exposure_unit=sealed.exposure_unit,
            n_adjustment_covariates=len(sealed.adjustment_columns),
            effect_model="cox_proportional_hazards_efron_ties",
            interval_method=sealed.uncertainty_method,
            proportional_hazards_test="schoenfeld_residuals",
            proportional_hazards_alpha=float(sealed.proportional_hazards_alpha),
            time_varying_cutpoints_days=[
                float(cut) for cut in sealed.time_varying_interval_cutpoints_days
            ],
            spline_knot_percentiles=[
                100.0 * value for value in sealed.spline_knot_quantiles
            ],
            descriptive_grouping=sealed.descriptive_grouping,
        )
    )


def run_landmark_continuous_survival_suite(
    *,
    frame: Any,
    authority: LandmarkContinuousSurvivalRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    out_dir: Path,
    input_product: str,
    input_evidence_id: str,
    input_sha256: str,
) -> dict[str, Any]:
    """Execute the exact sealed continuous-exposure landmark survival suite."""

    import numpy as np
    import pandas as pd

    from ...methods.time_varying_cox import fit_piecewise_time_varying_cox

    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("continuous survival runner received the wrong authority kind")
    if len(str(runtime_projection_sha256)) != 64:
        raise ValueError("continuous survival runtime projection digest is required")
    missing = sorted(set(sealed.required_columns) - set(frame.columns))
    if missing:
        raise ValueError("continuous survival input lacks columns: " + ", ".join(missing))

    working = frame[list(sealed.required_columns)].copy()
    for column in (
        sealed.exposure_column,
        sealed.event_column,
        sealed.followup_time_column,
        *(
            column
            for column in sealed.adjustment_columns
            if column not in sealed.categorical_adjustment_columns
        ),
    ):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    exposure = working[sealed.exposure_column]
    event = working[sealed.event_column]
    followup = working[sealed.followup_time_column]
    horizon = float(sealed.endpoint_horizon_days)
    endpoint_valid = (
        event.isin([0, 1])
        & followup.notna()
        & np.isfinite(followup)
        & followup.ge(0)
        & followup.le(horizon)
        & (event.eq(1) | followup.ge(horizon))
    )
    landmark_days = sealed.landmark_hours / 24.0
    alive_at_landmark = followup.gt(landmark_days)
    exposure_recorded = exposure.notna() & np.isfinite(exposure)
    eligible = endpoint_valid & alive_at_landmark & exposure_recorded
    analysis = working.loc[eligible].copy()
    analysis[sealed.derived_event_column] = event.loc[eligible].astype(int)
    analysis[sealed.derived_time_column] = followup.loc[eligible] - landmark_days
    if len(analysis) < _MINIMUM_MODEL_ROWS:
        raise ValueError("continuous survival risk set is too small to model")
    if int(analysis[sealed.derived_event_column].sum()) < _MINIMUM_MODEL_EVENTS:
        raise ValueError("continuous survival risk set has insufficient event support")

    groups, tertile_low, tertile_high = _tertile_groups(analysis[sealed.exposure_column])
    labels = _tertile_labels(sealed, tertile_low, tertile_high)
    risk_rows = [
        ("source_rows", len(working)),
        ("valid_fixed_horizon_endpoint", int(endpoint_valid.sum())),
        ("alive_and_observed_at_landmark", int((endpoint_valid & alive_at_landmark).sum())),
        ("landmark_analysis_population", len(analysis)),
    ]
    risk_flow = pd.DataFrame(
        [
            {
                "stage_order": index + 1,
                "stage": stage,
                "count": int(count),
                "source_denominator": int(len(working)),
                "percent_of_source": 100.0 * count / len(working) if len(working) else None,
                "excluded_since_prior_stage": (
                    0 if index == 0 else int(risk_rows[index - 1][1] - count)
                ),
            }
            for index, (stage, count) in enumerate(risk_rows)
        ]
    )
    table_one = _table_one(analysis, groups, sealed)
    km_table = _km_table(analysis, groups, labels, sealed=sealed)

    model_frame, covariates = _model_frame(analysis, sealed)
    fitter, nonconvergence = _cox_fit(
        model_frame,
        duration_col=sealed.derived_time_column,
        event_col=sealed.derived_event_column,
    )
    if nonconvergence is not None:
        raise ValueError(f"continuous survival Cox model did not converge: {nonconvergence}")
    cox_table = _cox_table(fitter)
    primary_rows = cox_table.loc[cox_table["term"].eq(sealed.exposure_column)]
    if len(primary_rows) != 1:
        raise ValueError("continuous survival Cox result lacks one exposure row")
    primary_row = primary_rows.iloc[0].to_dict()
    for name in ("hazard_ratio", "ci_low", "ci_high", "standard_error", "p_value"):
        coerce_finite_float(primary_row[name], label=f"continuous survival {name}")

    ph_table, global_p, exposure_p, ph_violation = _proportional_hazards_decision(
        model_frame, sealed=sealed, covariates=covariates
    )
    if ph_violation:
        ph_status = (
            "violation_report_only"
            if sealed.proportional_hazards_policy == "report_only"
            else "violation_block_paper_authorization"
        )
    else:
        ph_status = "not_rejected"
    ph_table = ph_table.copy()
    ph_table["declared_alpha"] = sealed.proportional_hazards_alpha
    ph_table["handling_policy"] = sealed.proportional_hazards_policy
    ph_table["ph_status"] = ph_status
    ph_table["paper_authorization_allowed"] = not ph_violation

    time_varying_table = fit_piecewise_time_varying_cox(
        model_frame,
        duration_col=sealed.derived_time_column,
        event_col=sealed.derived_event_column,
        covariates=covariates,
        interval_cutpoints=sealed.time_varying_interval_cutpoints_days,
        exposure_col=sealed.exposure_column,
    )
    exposure_intervals = time_varying_table.loc[time_varying_table["is_exposure"]]
    if len(exposure_intervals) != len(sealed.time_varying_interval_cutpoints_days) + 1:
        raise ValueError("continuous survival interval model lacks every exposure interval")

    spline_curve, functional_form = _spline_check(
        model_frame, sealed=sealed, linear_fitter=fitter
    )
    functional_form_estimated = functional_form["status"] == "estimated"

    # When the prespecified PH test rejects, the constant per-unit hazard ratio
    # is not a result: it stays a diagnostic row of the Cox table and never
    # becomes a summary leaf the manuscript could bind.
    constant = (
        {}
        if ph_violation
        else {
            "hazard_ratio": float(primary_row["hazard_ratio"]),
            "ci_low": float(primary_row["ci_low"]),
            "ci_high": float(primary_row["ci_high"]),
        }
    )
    reportable = {
        "schema_version": CONTINUOUS_SURVIVAL_REPORTING_SCHEMA_VERSION,
        "execution_owner": "landmark_continuous_survival_executor_v1",
        "interpretation_ceiling": "descriptive_prognostic_association_not_causal",
        "exposure": sealed.exposure_column,
        "outcome": sealed.event_column,
        "analysis_unit": sealed.analysis_unit_label,
        "landmark_hours": float(sealed.landmark_hours),
        "exposure_increment": float(sealed.exposure_increment),
        "exposure_unit": sealed.exposure_unit,
        "adjustment_columns": list(sealed.adjustment_columns),
        **({"adjusted_hazard_ratio_per_unit": dict(constant)} if constant else {}),
        "constant_hazard_ratio_authorized": not ph_violation,
        "proportional_hazards_status": ph_status,
        "proportional_hazards_test": {
            "schema_version": RULE_OUTCOME_SCHEMA_VERSION,
            "rule": "proportional_hazards_test",
            "diagnostic": "schoenfeld_residual_test",
            "alpha": float(sealed.proportional_hazards_alpha),
            "global_p_value": global_p,
            "exposure_p_value": exposure_p,
            "disposition": (
                "assumption_rejected" if ph_violation else "assumption_not_rejected"
            ),
        },
        "time_varying_adjusted_association": {
            "method": sealed.time_varying_effect_method,
            "adjustment_columns": list(sealed.adjustment_columns),
            "intervals": [
                {
                    "start_days": float(row.interval_start_days),
                    "end_days": float(row.interval_end_days),
                    "hazard_ratio": float(row.hazard_ratio),
                    "ci_low": float(row.ci_low),
                    "ci_high": float(row.ci_high),
                    "p_value": float(row.p_value),
                }
                for row in exposure_intervals.itertuples(index=False)
            ],
        },
        "functional_form": functional_form,
        "manuscript_projection": build_continuous_survival_manuscript_projection(
            interval_count=len(exposure_intervals),
            proportional_hazards_rejected=ph_violation,
            functional_form_estimated=functional_form_estimated,
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: out_dir / name for key, name in _FILES.items()}
    table_one.to_csv(paths["table_one"], index=False)
    risk_flow.to_csv(paths["risk_set"], index=False)
    km_table.to_csv(paths["km"], index=False)
    cox_table.to_csv(paths["cox"], index=False)
    ph_table.to_csv(paths["ph"], index=False)
    time_varying_table.to_csv(paths["time_varying"], index=False)
    spline_curve.to_csv(paths["spline"], index=False)
    _measurement_audit_table(sealed, working, analysis).to_csv(paths["measurement"], index=False)
    analysis.to_parquet(paths["analysis"], index=False)

    counts = {
        "n_source": int(len(working)),
        "n_landmark_population": int(len(analysis)),
        "n_complete_case": int(len(model_frame)),
        # The whole risk set's events, Kaplan-Meier's; ``n_events`` counts
        # the complete-case models' events.
        "n_events_landmark_population": int(analysis[sealed.derived_event_column].sum()),
        "n_events": int(model_frame[sealed.derived_event_column].sum()),
    }
    receipt = {
        "schema_version": "easyicu.landmark_continuous_survival_runtime_receipt/1",
        "protocol_content_sha256": sealed.protocol_content_sha256,
        "execution_contract_sha256": sealed.execution_contract_sha256,
        "runtime_projection_sha256": runtime_projection_sha256,
        "input_product": input_product,
        "input_evidence_id": input_evidence_id,
        "input_sha256": input_sha256,
        "analysis_frame_sha256": _canonical_frame_sha256(model_frame),
        "exposure_tertile_cutpoints": [tertile_low, tertile_high],
        "proportional_hazards_status": ph_status,
        "functional_form_status": (
            "estimated" if functional_form_estimated else str(functional_form["reason"])
        ),
        "paper_authorization_allowed": not ph_violation,
        "interpretation": sealed.interpretation,
        "analysis_only": True,
        "human_attestation_required": True,
    }
    paths["receipt"].write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    missingness_measurement_audit = {
        "source_missing_n_by_column": {
            column: int(working[column].isna().sum()) for column in sealed.required_columns
        },
        "landmark_missing_n_by_model_column": {
            column: int(analysis[column].isna().sum()) for column in sealed.adjustment_columns
        },
    }
    output_files = {
        sealed.table_one_product: paths["table_one"].name,
        sealed.risk_set_product: paths["risk_set"].name,
        sealed.km_product: paths["km"].name,
        sealed.cox_product: paths["cox"].name,
        sealed.ph_product: paths["ph"].name,
        sealed.time_varying_cox_product: paths["time_varying"].name,
        sealed.spline_product: paths["spline"].name,
        sealed.measurement_audit_product: paths["measurement"].name,
        sealed.receipt_product: paths["receipt"].name,
    }
    # The per-step numeric cap registers headline roots first and then this
    # order: the design's numbers bind as one Methods fact, the envelope's
    # estimates and the counts follow, and the audit comes last.
    return {
        "status": "ok",
        "analysis_family": "survival",
        "analysis_role": "primary",
        "deterministic_standard_analysis": LANDMARK_CONTINUOUS_SURVIVAL_ANALYSIS_KIND,
        "interpretation_class": "descriptive_prognostic_association",
        EXECUTED_METHOD_DESIGN_KEY: _executed_design(sealed),
        CONTINUOUS_SURVIVAL_REPORTING_KEY: reportable,
        **counts,
        "typed_cohort_input": input_product,
        "input_evidence_id": input_evidence_id,
        "input_sha256": input_sha256,
        "effect_measure": sealed.effect_measure,
        "proportional_hazards_status": ph_status,
        "paper_authorization_allowed": False,
        "analysis_only": True,
        "human_attestation_required": True,
        "analysis_cohort_file": paths["analysis"].name,
        "scientific_runtime_receipt": receipt,
        MANUSCRIPT_TABLES_KEY: _manuscript_tables(sealed, analysis, groups, labels),
        "missingness_measurement_audit": missingness_measurement_audit,
        "output_files": output_files,
    }


__all__ = [
    "LANDMARK_CONTINUOUS_SURVIVAL_ANALYSIS_KIND",
    "build_continuous_survival_manuscript_projection",
    "landmark_continuous_survival_executor_code",
    "landmark_continuous_survival_executor_owns_step",
    "landmark_continuous_survival_executor_scaffold",
    "run_landmark_continuous_survival_suite",
]
