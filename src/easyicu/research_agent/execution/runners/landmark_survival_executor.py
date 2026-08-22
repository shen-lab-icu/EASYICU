"""Deterministic fixed-landmark survival suite.

The caller-reviewed runtime authority owns every scientific coordinate. This
module only executes the sealed risk-set rule, descriptive Table 1, Kaplan-
Meier curves, adjusted Cox model, proportional-hazards audit and source-backed
composite figure. It contains no case identifier and no model-editable code.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import textwrap
from pathlib import Path
from typing import Any, Mapping, Optional

from ...authority.current_case_scientific_runtime import (
    LandmarkSurvivalRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.host_scaffold import HostScaffoldedScript
from ...schema import AnalysisPlan, AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input

LANDMARK_SURVIVAL_ANALYSIS_KIND = "signed_landmark_survival_suite"


def landmark_survival_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: LandmarkSurvivalRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    if authority is None:
        return False
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSurvivalRuntimeAuthority):
        return False
    return sealed.governed_step(plan) == step


def landmark_survival_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: LandmarkSurvivalRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    if authority is None:
        return False
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSurvivalRuntimeAuthority):
        return False
    return sealed.governed_figure_step(plan) == step


def landmark_survival_executor_scaffold(
    step: AnalysisStep,
    *,
    authority: LandmarkSurvivalRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> HostScaffoldedScript:
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSurvivalRuntimeAuthority):
        raise TypeError("landmark survival executor requires its sealed authority")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    typed_input = sole_typed_cohort_input(step)
    if typed_input is None:
        raise ValueError("landmark survival suite requires one typed cohort input")
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

        from easyicu.research_agent.execution.runners.landmark_survival_executor import (
            run_landmark_survival_suite,
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
    prologue += "\n\n" + textwrap.dedent(
        f"""
        summary = run_landmark_survival_suite(
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
    return HostScaffoldedScript(prologue=prologue, body="", epilogue="\n".join(epilogue))


def landmark_survival_executor_code(
    step: AnalysisStep,
    *,
    authority: LandmarkSurvivalRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> str:
    return landmark_survival_executor_scaffold(
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


def _finite_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"landmark survival {label} is non-finite")
    return number


def _table_one(frame: Any, sealed: LandmarkSurvivalRuntimeAuthority):
    import numpy as np
    import pandas as pd

    group = sealed.derived_exposure_column
    rows: list[dict[str, Any]] = []
    for column in sealed.table_one_columns:
        source = frame[column]
        if column in sealed.categorical_adjustment_columns:
            levels = sorted(str(value) for value in source.dropna().unique())
            for level in levels:
                row: dict[str, Any] = {
                    "variable": column,
                    "level": level,
                    "summary_type": "categorical_n_percent",
                }
                proportions: dict[int, float] = {}
                for exposure_value, label in ((0, "unexposed"), (1, "exposed")):
                    subset = source.loc[frame[group].eq(exposure_value)]
                    denominator = int(len(subset))
                    count = int(subset.astype("string").eq(level).sum())
                    proportion = count / denominator if denominator else float("nan")
                    row[f"{label}_n"] = count
                    row[f"{label}_denominator"] = denominator
                    row[f"{label}_percent"] = (
                        100.0 * proportion if math.isfinite(proportion) else None
                    )
                    proportions[exposure_value] = proportion
                p0, p1 = proportions[0], proportions[1]
                pooled = (p0 + p1) / 2.0
                denominator = math.sqrt(pooled * (1.0 - pooled))
                row["standardized_mean_difference"] = (
                    (p1 - p0) / denominator
                    if denominator > 0 and math.isfinite(denominator)
                    else None
                )
                rows.append(row)
            continue
        numeric = pd.to_numeric(source, errors="coerce")
        row = {
            "variable": column,
            "level": "",
            "summary_type": "continuous_mean_sd",
        }
        means: dict[int, float] = {}
        variances: dict[int, float] = {}
        for exposure_value, label in ((0, "unexposed"), (1, "exposed")):
            values = numeric.loc[frame[group].eq(exposure_value)].dropna()
            row[f"{label}_n"] = int(len(values))
            row[f"{label}_mean"] = float(values.mean()) if len(values) else None
            row[f"{label}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
            row[f"{label}_median"] = (
                float(values.median()) if len(values) else None
            )
            row[f"{label}_q1"] = (
                float(values.quantile(0.25)) if len(values) else None
            )
            row[f"{label}_q3"] = (
                float(values.quantile(0.75)) if len(values) else None
            )
            means[exposure_value] = float(values.mean()) if len(values) else float("nan")
            variances[exposure_value] = (
                float(values.var(ddof=1)) if len(values) > 1 else float("nan")
            )
        pooled_sd = math.sqrt(np.nanmean([variances[0], variances[1]]))
        row["standardized_mean_difference"] = (
            (means[1] - means[0]) / pooled_sd
            if pooled_sd > 0 and math.isfinite(pooled_sd)
            else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _render_figure(
    *,
    km_table: Any,
    cox_row: Mapping[str, Any],
    risk_flow: Any,
    sealed: LandmarkSurvivalRuntimeAuthority,
    out_dir: Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import NullFormatter, NullLocator

    from ...figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    fig = plt.figure(figsize=(183 / 25.4, 118 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.5, 1.0),
        height_ratios=(1.0, 0.9),
        left=0.09,
        right=0.975,
        top=0.93,
        bottom=0.12,
        wspace=0.42,
        hspace=0.52,
    )
    ax_km = fig.add_subplot(grid[:, 0])
    ax_hr = fig.add_subplot(grid[0, 1])
    ax_flow = fig.add_subplot(grid[1, 1])
    labels = {
        0: sealed.comparator_group_label,
        1: sealed.exposed_group_label,
    }
    colors = {0: palette["blue"], 1: palette["red"]}
    for value in (0, 1):
        group = km_table.loc[km_table["exposure_group"].eq(value)]
        ax_km.step(
            group["time_from_landmark_days"],
            group["survival_probability"],
            where="post",
            color=colors[value],
            linewidth=1.5,
            label=labels[value],
        )
    ax_km.set_ylim(0.0, 1.02)
    ax_km.set_xlim(0.0, sealed.endpoint_horizon_days - sealed.landmark_hours / 24.0)
    ax_km.set_xlabel(f"Days after the {sealed.landmark_hours:g}-hour landmark")
    ax_km.set_ylabel("Survival probability")
    ax_km.set_title("Landmark Kaplan-Meier survival", loc="left")
    ax_km.legend(loc="lower left", fontsize=6.4)
    ax_km.grid(axis="y", color=palette["neutral_light"], linewidth=0.55)
    add_panel_label(ax_km, "A", x=-0.09)

    hazard_ratio = float(cox_row["hazard_ratio"])
    ci_low = float(cox_row["ci_low"])
    ci_high = float(cox_row["ci_high"])
    ax_hr.errorbar(
        hazard_ratio,
        0,
        xerr=np.array([[hazard_ratio - ci_low], [ci_high - hazard_ratio]]),
        fmt="o",
        color=palette["blue"],
        capsize=3,
        linewidth=1.2,
    )
    ax_hr.axvline(1.0, color=palette["neutral"], linestyle="--", linewidth=0.8)
    ax_hr.set_xscale("log")
    lower_limit = min(ci_low * 0.9, 0.9)
    upper_limit = max(ci_high * 1.1, 1.1)
    ax_hr.set_xlim(lower_limit, upper_limit)
    tick_candidates = (0.25, 0.5, 1.0, 2.0, 4.0)
    ticks = [tick for tick in tick_candidates if lower_limit <= tick <= upper_limit]
    ax_hr.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    ax_hr.xaxis.set_minor_locator(NullLocator())
    ax_hr.xaxis.set_minor_formatter(NullFormatter())
    ax_hr.set_yticks([0], [sealed.exposed_group_label])
    ax_hr.set_xlabel("Adjusted hazard ratio (95% CI)")
    ax_hr.set_title("Adjusted Cox association", loc="left")
    ax_hr.text(
        0.03,
        0.08,
        f"HR {hazard_ratio:.2f} ({ci_low:.2f}–{ci_high:.2f})",
        transform=ax_hr.transAxes,
        fontsize=6.4,
        ha="left",
        va="bottom",
    )
    add_panel_label(ax_hr, "B", x=-0.16, y=1.05)

    display = risk_flow.tail(4).copy()
    display_labels = {
        "valid_fixed_horizon_endpoint": (
            f"Valid {sealed.endpoint_horizon_days:g}-day endpoint"
        ),
        "alive_and_observed_at_landmark": (
            f"Alive/observed at {sealed.landmark_hours:g} h"
        ),
        "exposure_status_and_timing_supported": "Exposure timing supported",
        "landmark_analysis_population": "Landmark analysis population",
    }
    y = np.arange(len(display))
    ax_flow.barh(y, display["count"], color=palette["teal"])
    ax_flow.set_yticks(
        y,
        [display_labels.get(value, value.replace("_", " ")) for value in display["stage"]],
        fontsize=5.8,
    )
    for row_index, count in enumerate(display["count"].astype(int)):
        ax_flow.text(
            count,
            row_index,
            f"  {count:,}",
            va="center",
            ha="left",
            fontsize=5.8,
        )
    ax_flow.set_xlim(0, max(display["count"]) * 1.18)
    ax_flow.invert_yaxis()
    ax_flow.set_xlabel(f"{sealed.analysis_unit_label} (n)")
    ax_flow.set_title("Risk-set accounting", loc="left")
    add_panel_label(ax_flow, "C", x=-0.16, y=1.05)

    contract = make_figure_contract(
        figure_id="landmark_survival_suite",
        core_claim=(
            "Post-landmark survival and the adjusted hazard-ratio association "
            "are shown only after explicit exposure-timing and risk-set exclusions."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Landmark Kaplan-Meier survival",
                "role": "temporal_absolute_risk",
                "chart_type": "kaplan_meier_curve",
                "claim": "Absolute post-landmark survival is displayed by the frozen incident-exposure groups.",
                "evidence_ids": [],
                "review_risk": "This is an observational landmark comparison and does not identify a causal exposure effect.",
            },
            {
                "panel_id": "B",
                "title": "Adjusted Cox association",
                "role": "survival_effect",
                "chart_type": "hazard_ratio_forest",
                "claim": "The adjusted hazard ratio quantifies the prespecified descriptive prognostic association.",
                "evidence_ids": [],
                "review_risk": "Interpretation depends on proportional-hazards diagnostics and residual confounding remains possible.",
            },
            {
                "panel_id": "C",
                "title": "Risk-set accounting",
                "role": "cohort_accounting",
                "chart_type": "cohort_flow",
                "claim": "The analytic denominator is traceable through endpoint, landmark and exposure-timing gates.",
                "evidence_ids": [],
                "review_risk": "Excluded prevalent or timing-unknown exposure rows define the supported estimand boundary.",
            },
        ],
        export_formats=("svg", "pdf", "png"),
        source_data=("landmark_km_curve.csv", "landmark_cox_summary.csv", "landmark_risk_set_flow.csv"),
        statistics_note=(
            "Kaplan-Meier estimates use the post-landmark clock. The Cox model "
            "reports a Wald 95% confidence interval and a Schoenfeld residual audit."
        ),
        image_integrity_note="All plotted values are rendered from digest-bound upstream result tables.",
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "landmark_survival_suite",
        contract=contract,
        formats=("svg", "pdf", "png"),
        dpi=300,
    )
    plt.close(fig)
    return outputs


def run_landmark_survival_suite(
    *,
    frame: Any,
    authority: LandmarkSurvivalRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    out_dir: Path,
    input_product: str,
    input_evidence_id: str,
    input_sha256: str,
) -> dict[str, Any]:
    """Execute the exact sealed landmark survival suite."""

    import numpy as np
    import pandas as pd
    from lifelines import CoxPHFitter

    from ...figures.base import km_estimate
    from ...methods.ph_schoenfeld import ph_test

    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSurvivalRuntimeAuthority):
        raise TypeError("landmark survival runner received the wrong authority kind")
    if len(str(runtime_projection_sha256)) != 64:
        raise ValueError("landmark survival runtime projection digest is required")
    missing = sorted(set(sealed.required_columns) - set(frame.columns))
    if missing:
        raise ValueError("landmark survival input lacks columns: " + ", ".join(missing))

    working = frame[list(sealed.required_columns)].copy()
    numeric_columns = {
        sealed.exposure_status_column,
        sealed.exposure_onset_column,
        sealed.event_column,
        sealed.followup_time_column,
        *(column for column in sealed.adjustment_columns if column not in sealed.categorical_adjustment_columns),
    }
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    exposure_status = working[sealed.exposure_status_column]
    exposure_onset = working[sealed.exposure_onset_column]
    event = working[sealed.event_column]
    followup = working[sealed.followup_time_column]
    endpoint_valid = (
        event.isin([0, 1])
        & followup.notna()
        & np.isfinite(followup)
        & followup.ge(0)
        & followup.le(float(sealed.endpoint_horizon_days))
        & (
            event.eq(1)
            | followup.ge(float(sealed.endpoint_horizon_days))
        )
    )
    landmark_days = sealed.landmark_hours / 24.0
    alive_at_landmark = followup.gt(landmark_days)
    status_valid = exposure_status.isin([0, 1])
    timing_known = exposure_status.eq(0) | exposure_onset.notna()
    prevalent = exposure_status.eq(1) & exposure_onset.le(
        float(sealed.prevalent_exposure_cutoff_hours)
    )
    incident = (
        exposure_status.eq(1)
        & exposure_onset.gt(float(sealed.prevalent_exposure_cutoff_hours))
        & exposure_onset.le(float(sealed.exposure_window_hours[1]))
    )
    exposure_supported = exposure_status.eq(0) | incident
    eligible_mask = (
        endpoint_valid
        & alive_at_landmark
        & status_valid
        & timing_known
        & ~prevalent
        & exposure_supported
    )
    analysis = working.loc[eligible_mask].copy()
    analysis[sealed.derived_exposure_column] = incident.loc[eligible_mask].astype(int)
    analysis[sealed.derived_event_column] = event.loc[eligible_mask].astype(int)
    analysis[sealed.derived_time_column] = followup.loc[eligible_mask] - landmark_days
    if len(analysis) < 100 or analysis[sealed.derived_exposure_column].nunique() != 2:
        raise ValueError("landmark survival risk set lacks an estimable exposure contrast")
    if int(analysis[sealed.derived_event_column].sum()) < 10:
        raise ValueError("landmark survival risk set has insufficient event support")

    risk_rows = [
        ("source_rows", len(working)),
        ("valid_fixed_horizon_endpoint", int(endpoint_valid.sum())),
        ("alive_and_observed_at_landmark", int((endpoint_valid & alive_at_landmark).sum())),
        ("exposure_status_and_timing_supported", int((endpoint_valid & alive_at_landmark & status_valid & timing_known).sum())),
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

    table_one = _table_one(analysis, sealed)
    km_rows: list[dict[str, Any]] = []
    for exposure_value in (0, 1):
        subset = analysis.loc[analysis[sealed.derived_exposure_column].eq(exposure_value)]
        estimate = km_estimate(
            subset[sealed.derived_time_column], subset[sealed.derived_event_column]
        )
        for time, survival, at_risk in zip(
            estimate["time"], estimate["survival"], estimate["at_risk"]
        ):
            km_rows.append(
                {
                    "exposure_group": exposure_value,
                    "time_from_landmark_days": float(time),
                    "survival_probability": float(survival),
                    "at_risk": int(at_risk),
                    "group_n": int(estimate["n"]),
                    "group_events": int(estimate["n_events"]),
                }
            )
    km_table = pd.DataFrame(km_rows)

    model_source = analysis[
        [
            sealed.derived_time_column,
            sealed.derived_event_column,
            sealed.derived_exposure_column,
            *sealed.adjustment_columns,
        ]
    ].copy()
    categorical_sources = list(sealed.categorical_adjustment_columns)
    numeric_adjustments = [
        column
        for column in sealed.adjustment_columns
        if column not in sealed.categorical_adjustment_columns
    ]
    for column in numeric_adjustments:
        model_source[column] = pd.to_numeric(model_source[column], errors="coerce")
    pieces = [
        model_source[
            [
                sealed.derived_time_column,
                sealed.derived_event_column,
                sealed.derived_exposure_column,
                *numeric_adjustments,
            ]
        ]
    ]
    for column in categorical_sources:
        encoded = pd.get_dummies(
            model_source[column].astype("string"),
            prefix=column,
            drop_first=True,
            dtype=float,
        )
        encoded.loc[model_source[column].isna(), :] = float("nan")
        if encoded.empty:
            raise ValueError(f"landmark survival categorical column {column!r} has no contrast")
        pieces.append(encoded)
    model_frame = pd.concat(pieces, axis=1).dropna().astype(float)
    if len(model_frame) < 100 or int(model_frame[sealed.derived_event_column].sum()) < 10:
        raise ValueError("landmark survival complete-case model is not estimable")
    covariates = [
        column
        for column in model_frame.columns
        if column not in {sealed.derived_time_column, sealed.derived_event_column}
    ]
    fitter = CoxPHFitter()
    fitter.fit(
        model_frame,
        duration_col=sealed.derived_time_column,
        event_col=sealed.derived_event_column,
    )
    summary = fitter.summary.reset_index().rename(columns={"covariate": "term"})
    if "term" not in summary.columns:
        summary = summary.rename(columns={summary.columns[0]: "term"})
    cox_table = pd.DataFrame(
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
    primary_rows = cox_table.loc[
        cox_table["term"].eq(sealed.derived_exposure_column)
    ]
    if len(primary_rows) != 1:
        raise ValueError("landmark survival Cox result lacks one exposure row")
    primary_row = primary_rows.iloc[0].to_dict()
    for name in ("hazard_ratio", "ci_low", "ci_high", "standard_error", "p_value"):
        _finite_float(primary_row[name], label=name)

    ph_table = ph_test(
        model_frame,
        duration_col=sealed.derived_time_column,
        event_col=sealed.derived_event_column,
        covariates=covariates,
        time_transform="km",
    )
    global_rows = ph_table.loc[ph_table["covariate"].astype(str).eq("global")]
    exposure_rows = ph_table.loc[
        ph_table["covariate"].astype(str).eq(sealed.derived_exposure_column)
    ]
    if len(global_rows) != 1 or len(exposure_rows) != 1:
        raise ValueError("landmark survival PH audit lacks global or exposure result")
    global_p = _finite_float(global_rows["p_value"].iloc[0], label="global PH p")
    exposure_p = _finite_float(exposure_rows["p_value"].iloc[0], label="exposure PH p")
    ph_violation = min(global_p, exposure_p) < sealed.proportional_hazards_alpha
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

    out_dir.mkdir(parents=True, exist_ok=True)
    table_one_path = out_dir / "landmark_table_one.csv"
    risk_path = out_dir / "landmark_risk_set_flow.csv"
    km_path = out_dir / "landmark_km_curve.csv"
    cox_path = out_dir / "landmark_cox_summary.csv"
    ph_path = out_dir / "landmark_ph_diagnostics.csv"
    analysis_path = out_dir / "landmark_analysis_cohort.parquet"
    table_one.to_csv(table_one_path, index=False)
    risk_flow.to_csv(risk_path, index=False)
    km_table.to_csv(km_path, index=False)
    cox_table.to_csv(cox_path, index=False)
    ph_table.to_csv(ph_path, index=False)
    analysis.to_parquet(analysis_path, index=False)
    receipt = {
        "schema_version": "easyicu.landmark_survival_runtime_receipt/1",
        "protocol_content_sha256": sealed.protocol_content_sha256,
        "execution_contract_sha256": sealed.execution_contract_sha256,
        "runtime_projection_sha256": runtime_projection_sha256,
        "input_product": input_product,
        "input_evidence_id": input_evidence_id,
        "input_sha256": input_sha256,
        "analysis_frame_sha256": _canonical_frame_sha256(model_frame),
        "landmark_hours": sealed.landmark_hours,
        "endpoint_horizon_days": sealed.endpoint_horizon_days,
        "prevalent_exposure_action": sealed.prevalent_exposure_action,
        "adjustment_columns": list(sealed.adjustment_columns),
        "n_source": int(len(working)),
        "n_landmark_population": int(len(analysis)),
        "n_complete_case": int(len(model_frame)),
        "n_events": int(model_frame[sealed.derived_event_column].sum()),
        "hazard_ratio": float(primary_row["hazard_ratio"]),
        "ci_low": float(primary_row["ci_low"]),
        "ci_high": float(primary_row["ci_high"]),
        "ph_global_p_value": global_p,
        "ph_exposure_p_value": exposure_p,
        "ph_status": ph_status,
        "paper_authorization_allowed": not ph_violation,
        "interpretation": sealed.interpretation,
        "analysis_only": True,
        "human_attestation_required": True,
    }
    receipt_path = out_dir / "landmark_survival_runtime_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    output_files = {
        sealed.table_one_product: table_one_path.name,
        sealed.risk_set_product: risk_path.name,
        sealed.km_product: km_path.name,
        sealed.cox_product: cox_path.name,
        sealed.ph_product: ph_path.name,
        sealed.receipt_product: receipt_path.name,
    }
    return {
        "status": "ok",
        "analysis_family": "survival",
        "analysis_role": "primary",
        "deterministic_standard_analysis": LANDMARK_SURVIVAL_ANALYSIS_KIND,
        "interpretation_class": "descriptive_prognostic_association",
        "typed_cohort_input": input_product,
        "input_evidence_id": input_evidence_id,
        "input_sha256": input_sha256,
        "n_source": int(len(working)),
        "n_landmark_population": int(len(analysis)),
        "n_complete_case": int(len(model_frame)),
        "n_events": int(model_frame[sealed.derived_event_column].sum()),
        "hazard_ratio": float(primary_row["hazard_ratio"]),
        "hazard_ratio_ci_low": float(primary_row["ci_low"]),
        "hazard_ratio_ci_high": float(primary_row["ci_high"]),
        "proportional_hazards_status": ph_status,
        "paper_authorization_allowed": False,
        "analysis_only": True,
        "human_attestation_required": True,
        "analysis_cohort_file": analysis_path.name,
        "scientific_runtime_receipt": receipt,
        "output_files": output_files,
    }


def run_landmark_survival_figure(
    *,
    km_table: Any,
    cox_table: Any,
    risk_flow: Any,
    source_paths: Mapping[str, Path],
    authority: LandmarkSurvivalRuntimeAuthority | Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Render only from the three digest-bound result tables it declares."""

    import pandas as pd

    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSurvivalRuntimeAuthority):
        raise TypeError("landmark survival figure received the wrong authority kind")
    primary = cox_table.loc[
        cox_table["term"].astype(str).eq(sealed.derived_exposure_column)
    ]
    if len(primary) != 1:
        raise ValueError("landmark survival figure lacks one primary Cox row")
    out_dir.mkdir(parents=True, exist_ok=True)
    source_filename_by_product = {
        sealed.km_product: "landmark_km_curve.csv",
        sealed.cox_product: "landmark_cox_summary.csv",
        sealed.risk_set_product: "landmark_risk_set_flow.csv",
    }
    copied_sources: list[str] = []
    for product in sealed.figure_input_products:
        source = Path(source_paths[product]).resolve()
        destination = out_dir / source_filename_by_product[product]
        shutil.copyfile(source, destination)
        if (
            hashlib.sha256(source.read_bytes()).digest()
            != hashlib.sha256(destination.read_bytes()).digest()
        ):
            raise ValueError("landmark survival figure source changed while copying")
        copied_sources.append(destination.name)
    outputs = _render_figure(
        km_table=pd.DataFrame(km_table),
        cox_row=primary.iloc[0].to_dict(),
        risk_flow=pd.DataFrame(risk_flow),
        sealed=sealed,
        out_dir=out_dir,
    )
    figure_file = outputs.get("svg") or outputs.get("png")
    if figure_file is None:
        raise ValueError("landmark survival figure export is missing")
    return {
        "status": "ok",
        "rendering_only": True,
        "deterministic_standard_analysis": "signed_landmark_survival_figure",
        "source_data_files": copied_sources,
        "figure_assets": {key: value.name for key, value in outputs.items()},
        "output_files": {sealed.figure_product: figure_file.name},
    }


def landmark_survival_figure_executor_code(
    step: AnalysisStep,
    *,
    authority: LandmarkSurvivalRuntimeAuthority | Mapping[str, Any],
) -> str:
    """Return the host-owned renderer for the sealed survival result tables."""

    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSurvivalRuntimeAuthority):
        raise TypeError("landmark survival figure requires its sealed authority")
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.landmark_survival_executor import (
            run_landmark_survival_figure,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_typed_input,
            run_dir_from_env,
        )

        authority = json.loads({json.dumps(authority_json)})
        input_products = {sealed.figure_input_products!r}
        bindings = {{
            product: load_typed_input(
                input_key=product,
                run_dir=run_dir_from_env(),
                resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).resolve(),
                expected_evidence_kind="table",
                require_consumption_contract=True,
            )
            for product in input_products
        }}
        summary = run_landmark_survival_figure(
            km_table=bindings[{sealed.km_product!r}].frame,
            cox_table=bindings[{sealed.cox_product!r}].frame,
            risk_flow=bindings[{sealed.risk_set_product!r}].frame,
            source_paths={{key: value.path for key, value in bindings.items()}},
            authority=authority,
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
        )
        (Path(os.environ["STEP_OUT_DIR"]) / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        """
    ).strip()


__all__ = [
    "LANDMARK_SURVIVAL_ANALYSIS_KIND",
    "landmark_survival_executor_code",
    "landmark_survival_executor_owns_step",
    "landmark_survival_figure_executor_code",
    "landmark_survival_figure_executor_owns_step",
    "run_landmark_survival_figure",
    "run_landmark_survival_suite",
]
