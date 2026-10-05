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
import re
import shutil
import textwrap
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional

from ...authority.current_case_scientific_runtime import (
    LandmarkSurvivalRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...authority.prespecified_rule_outcomes import RULE_OUTCOME_SCHEMA_VERSION
from ...authority.survival_scientific_claims import (
    CONSTANT_HAZARD_RATIO_CLAIM_ID,
    SURVIVAL_REPORTING_SCHEMA_VERSION,
    interval_hazard_ratio_claim_id,
)
from ...contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
    LandmarkSurvivalDesign,
    executed_method_design_payload,
)
from ...contracts.host_scaffold import HostScaffoldedScript
from ...contracts.manuscript_tables import (
    MANUSCRIPT_TABLE_SCHEMA_VERSION,
    MANUSCRIPT_TABLES_KEY,
    validate_manuscript_table_declarations,
)
from ...contracts.manuscript_result_structure import PRIMARY_RESULT_HEADINGS_BY_FAMILY
from ...schema import AnalysisPlan, AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input

LANDMARK_SURVIVAL_ANALYSIS_KIND = "signed_landmark_survival_suite"
#: The figure half of the same sealed authority. Named here, beside the
#: suite it renders, so the selector spells it the way its five siblings in
#: the ``isinstance(sealed_current, ...)`` chain are spelled -- every other
#: authority-gated kind is a module constant, and this one arrived as a bare
#: literal, which is what made
#: ``test_report_names_every_owner_the_selector_consults`` single it out.
LANDMARK_SURVIVAL_FIGURE_ANALYSIS_KIND = "signed_landmark_survival_figure"


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
    prologue += (
        "\n\n"
        + textwrap.dedent(
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


def _landmark_finite_float(value: Any, *, label: str) -> float:
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
            row[f"{label}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            row[f"{label}_median"] = float(values.median()) if len(values) else None
            row[f"{label}_q1"] = float(values.quantile(0.25)) if len(values) else None
            row[f"{label}_q3"] = float(values.quantile(0.75)) if len(values) else None
            means[exposure_value] = (
                float(values.mean()) if len(values) else float("nan")
            )
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


def _reader_words(text: str) -> str:
    """Plain reader words for a table caption or label (no markup characters)."""

    cleaned = re.sub(r"[{}\[\]<>`\\|*_#]+", " ", str(text))
    return " ".join(cleaned.split())[:200] or "Unnamed"


def _manuscript_tables(
    sealed: LandmarkSurvivalRuntimeAuthority, analysis: Any
) -> list[dict[str, Any]]:
    """Declare the suite's Table 1 and risk-set accounting as reader tables.

    The reporting owner formats the two products' recorded cells under these
    words; it computes nothing.  Every group and stage word comes from the
    sealed contract, so a reader sees the groups the suite compared.
    """

    group = analysis[sealed.derived_exposure_column]
    sizes = group.value_counts()
    events = analysis[sealed.derived_event_column].groupby(group).sum()
    landmark = f"{sealed.landmark_hours:g}"
    # A suite signed before the onset representation timed the exposure by
    # its first record of any value; its words stay as they were.
    present_onset = sealed.exposure_onset_representation == "first_truthy_event_time"

    def counts(value: int) -> dict[str, Any]:
        n, deaths = int(sizes.get(value, 0)), int(events.get(value, 0))
        return {"n": n, "events": deaths, "events_percent": 100.0 * deaths / n if n else 0.0}

    declarations = [
        {
            "schema_version": MANUSCRIPT_TABLE_SCHEMA_VERSION,
            "product": sealed.table_one_product,
            "caption": _reader_words(
                f"Characteristics of the {sealed.analysis_unit_label} in the landmark "
                "analysis cohort, by exposure group"
            ),
            "body": {
                "layout": "grouped_summary",
                "groups": [
                    {
                        "prefix": "unexposed",
                        "label": _reader_words(sealed.comparator_group_label),
                        **counts(0),
                    },
                    {
                        "prefix": "exposed",
                        "label": _reader_words(sealed.exposed_group_label),
                        **counts(1),
                    },
                ],
                # Every suite endpoint is a fixed-horizon death.
                "events_label": f"Deaths by day {sealed.endpoint_horizon_days:g}, n (%)",
            },
            "notes": [
                "Categorical percentages use every record of the group as the denominator, "
                "so levels need not sum to 100% when a value is missing.",
                "Continuous variables are summarized over their recorded values.",
                "The standardized mean difference compares the exposed with the comparator "
                "group; it is not a significance test.",
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
                        f"Alive and under observation at the {landmark}-hour landmark"
                    ),
                    "exposure_status_and_timing_supported": (
                        "Exposure status and time of first present record available"
                        if present_onset
                        else "Exposure status and first recorded time available"
                    ),
                    "landmark_analysis_population": "Landmark analysis cohort",
                },
            },
            "notes": [
                "The last stage excludes exposed records whose exposure was first recorded "
                + ("as present " if present_onset else "")
                + f"at or before hour {sealed.prevalent_exposure_cutoff_hours:g} or after hour "
                f"{sealed.exposure_window_hours[1]:g}.",
                "Excluded counts are the records removed since the stage before.",
            ],
        },
    ]
    return [
        declaration.model_dump(mode="json")
        for declaration in validate_manuscript_table_declarations(declarations)
    ]


def _measurement_audit_table(
    sealed: LandmarkSurvivalRuntimeAuthority, source: Any, landmark: Any
):
    """Availability of every sealed source column, in the source and at the landmark.

    The same counts the receipt's ``missingness_measurement_audit`` records,
    published as the suite's data-quality product: the complete-case model
    drops a landmark row that lacks any adjustment column.
    """

    import pandas as pd

    roles = {
        sealed.exposure_status_column: "exposure_status",
        sealed.exposure_onset_column: "exposure_onset",
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


_PH_ROW_INCHES = 0.11
_FLOW_PANEL_MIN_INCHES = 1.2


def _fit_ph_panel_rows(fig: Any, *, ax_flow: Any, ax_ph: Any, rows: int) -> None:
    """Give every proportional-hazards term a legible row in panel d.

    The panel lists one row per model term, and a categorical covariate adds
    one per level, so a fixed panel height overlaps its labels once a model
    has a few covariates.  The height comes from the risk-set panel above it
    while that panel keeps its minimum; beyond that the figure grows.
    """

    height = fig.get_figheight()
    flow, ph = ax_flow.get_position(), ax_ph.get_position()
    need = rows * _PH_ROW_INCHES - ph.height * height
    if need <= 0:
        return
    spare = max(flow.height * height - _FLOW_PANEL_MIN_INCHES, 0.0)
    taken = min(need, spare)
    grow = need - taken
    if grow > 0:
        # Every axes keeps its absolute size and moves up; the new strip at
        # the bottom extends panel d downward.
        new_height = height + grow
        for axes in fig.axes:
            box = axes.get_position()
            axes.set_position([
                box.x0, (box.y0 * height + grow) / new_height,
                box.width, box.height * height / new_height,
            ])
        fig.set_figheight(new_height)
        height = new_height
        flow, ph = ax_flow.get_position(), ax_ph.get_position()
    ax_flow.set_position([
        flow.x0, flow.y0 + taken / height, flow.width, flow.height - taken / height,
    ])
    ax_ph.set_position([
        ph.x0, ph.y0 - grow / height, ph.width, ph.height + need / height,
    ])


def _fit_row_label_to_gutter(fig: Any, *, ax: Any, neighbour: Any, label: str) -> None:
    """Name a one-row estimate without reaching into the panel on its left.

    The row label is the exposure group's display label, whose length is the
    study's; unwrapped, a long one runs across the neighbouring panel's data.
    It is wrapped at word boundaries until its drawn extent clears that panel.
    """

    for width in (len(label), 28, 22, 18, 15, 12, 9):
        if width > len(label):
            continue
        text = label if width == len(label) else textwrap.fill(label, width=width)
        ax.set_yticks([0], [text])
        fig.canvas.draw()
        left_edge = min(tick.get_window_extent().x0 for tick in ax.get_yticklabels())
        if left_edge >= neighbour.get_window_extent().x1 + 2.0:
            return


def _reader_legend(*, headline_hr_authorized: bool, promotes_time_varying: bool) -> str:
    """The figure's source-bound legend: what each drawn panel shows, and no value.

    Panel (b) follows the signed PH decision as it is drawn.  The manuscript
    projects this legend; a figure without one cannot enter a manuscript.
    """

    if headline_hr_authorized:
        estimate = (
            "(b) The adjusted Cox hazard ratio with its confidence interval; the "
            "prespecified proportional-hazards test did not reject the assumption."
        )
    elif promotes_time_varying:
        estimate = (
            "(b) Adjusted interval-specific hazard ratios with their confidence "
            "intervals from the prespecified extended Cox model, drawn instead of "
            "one constant hazard ratio because the proportional-hazards assumption "
            "was rejected."
        )
    else:
        estimate = (
            "(b) The unadjusted restricted-mean survival-time difference with its "
            "confidence interval, drawn instead of a hazard ratio because the "
            "proportional-hazards assumption was rejected."
        )
    return " ".join((
        "(a) Unadjusted Kaplan-Meier survival after the landmark by exposure "
        "group, with the number at risk below.",
        estimate,
        "(c) Risk-set accounting from the source records through the endpoint, "
        "landmark and exposure-timing gates to the analysis population.",
        "(d) Schoenfeld residual tests of the proportional-hazards assumption "
        "for each model term and globally, the global test Bonferroni-adjusted "
        "over the terms. The dashed line marks the prespecified alpha against "
        "which the exposure term and the global test were judged; the other "
        "terms, in lighter bars, are not adjusted for multiplicity and enter the "
        "decision only through the global test.",
        "Every value is drawn from the suite's registered result tables; the "
        "figure fits no model of its own.",
    ))


def _render_figure(
    *,
    km_table: Any,
    cox_row: Mapping[str, Any],
    rmst_table: Any | None,
    time_varying_table: Any | None,
    risk_flow: Any,
    ph_table: Any,
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
    from ...figures.display_labels import display_label

    palette = apply_publication_style()
    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        2,
        width_ratios=(1.42, 1.0),
        height_ratios=(1.0, 0.85, 0.85, 0.46),
        left=0.09,
        right=0.975,
        top=0.93,
        bottom=0.10,
        wspace=0.38,
        hspace=0.72,
    )
    ax_km = fig.add_subplot(grid[:3, 0])
    ax_risk = fig.add_subplot(grid[3, 0], sharex=ax_km)
    ax_hr = fig.add_subplot(grid[0, 1])
    ax_flow = fig.add_subplot(grid[1:3, 1])
    ax_ph = fig.add_subplot(grid[3, 1])
    _fit_ph_panel_rows(fig, ax_flow=ax_flow, ax_ph=ax_ph, rows=len(ph_table))
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
    ax_km.set_title("Unadjusted landmark Kaplan-Meier survival", loc="left")
    ax_km.legend(loc="lower left", fontsize=6.4)
    add_panel_label(ax_km, "a", x=-0.09, fontsize=8.0)

    horizon = sealed.endpoint_horizon_days - sealed.landmark_hours / 24.0
    risk_times = np.linspace(0.0, horizon, 5)
    risk_rows: list[list[str]] = []
    for value in (0, 1):
        group = km_table.loc[km_table["exposure_group"].eq(value)].sort_values(
            "time_from_landmark_days"
        )
        counts: list[str] = []
        for time_point in risk_times:
            eligible = group.loc[group["time_from_landmark_days"].le(time_point)]
            counts.append(
                str(
                    int(
                        (eligible.iloc[-1] if not eligible.empty else group.iloc[0])[
                            "at_risk"
                        ]
                    )
                )
            )
        risk_rows.append(counts)
    ax_risk.axis("off")
    # The table stops below the "Number at risk" heading so the heading never
    # sits on the first column's time label.
    table = ax_risk.table(
        cellText=risk_rows,
        rowLabels=[labels[0], labels[1]],
        colLabels=[f"{value:g}" for value in risk_times],
        cellLoc="center",
        rowLoc="right",
        bbox=[0.0, 0.0, 1.0, 0.74],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.7)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.0)
    ax_risk.text(
        -0.02,
        0.98,
        "Number at risk",
        transform=ax_risk.transAxes,
        fontsize=6.2,
        fontweight="bold",
        va="top",
    )

    ph_statuses = {
        str(value).strip()
        for value in ph_table.get("ph_status", [])
        if str(value).strip()
    }
    authorization_values = {
        bool(value) for value in ph_table.get("paper_authorization_allowed", [])
    }
    ph_violation = any(value.startswith("violation_") for value in ph_statuses)
    headline_hr_authorized = not ph_violation and authorization_values != {False}

    hazard_ratio = float(cox_row["hazard_ratio"])
    ci_low = float(cox_row["ci_low"])
    ci_high = float(cox_row["ci_high"])
    promotes_time_varying = bool(
        not headline_hr_authorized
        and time_varying_table is not None
        and len(time_varying_table) > 0
    )
    if headline_hr_authorized:
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
        _fit_row_label_to_gutter(
            fig, ax=ax_hr, neighbour=ax_km, label=sealed.exposed_group_label
        )
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
    elif promotes_time_varying:
        exposure_rows = time_varying_table.loc[
            time_varying_table["is_exposure"].astype(bool)
        ].sort_values("interval_index")
        if len(exposure_rows) != len(sealed.time_varying_interval_cutpoints_days) + 1:
            raise ValueError(
                "landmark survival figure lacks every time-varying exposure interval"
            )
        estimates = exposure_rows["hazard_ratio"].to_numpy(dtype=float)
        lows = exposure_rows["ci_low"].to_numpy(dtype=float)
        highs = exposure_rows["ci_high"].to_numpy(dtype=float)
        positions = np.arange(len(exposure_rows))
        ax_hr.errorbar(
            estimates,
            positions,
            xerr=np.vstack((estimates - lows, highs - estimates)),
            fmt="o",
            color=palette["blue"],
            capsize=3,
            linewidth=1.2,
        )
        ax_hr.axvline(1.0, color=palette["neutral"], linestyle="--", linewidth=0.8)
        ax_hr.set_xscale("log")
        ax_hr.set_yticks(
            positions,
            [
                f"{row.interval_start_days:g}–{row.interval_end_days:g} d"
                for row in exposure_rows.itertuples()
            ],
        )
        ax_hr.invert_yaxis()
        ax_hr.set_xlabel("Adjusted interval-specific HR (95% CI)")
        ax_hr.set_title("Time-varying adjusted association", loc="left")
    else:
        if rmst_table is None or len(rmst_table) != 1:
            raise ValueError(
                "landmark survival requires one RMST contrast when PH is rejected"
            )
        rmst_row = rmst_table.iloc[0]
        difference = float(rmst_row["rmst_difference_days"])
        difference_low = float(rmst_row["ci_low"])
        difference_high = float(rmst_row["ci_high"])
        ax_hr.errorbar(
            difference,
            0,
            xerr=np.array(
                [[difference - difference_low], [difference_high - difference]]
            ),
            fmt="o",
            color=palette["blue"],
            capsize=3,
            linewidth=1.2,
        )
        ax_hr.axvline(0.0, color=palette["neutral"], linestyle="--", linewidth=0.8)
        span = max(abs(difference_low), abs(difference_high), 0.25)
        ax_hr.set_xlim(-1.15 * span, 1.15 * span)
        _fit_row_label_to_gutter(
            fig, ax=ax_hr, neighbour=ax_km, label=sealed.exposed_group_label
        )
        ax_hr.set_xlabel("RMST difference, days (95% CI)")
        ax_hr.set_title("PH-free survival contrast", loc="left")
        ax_hr.text(
            0.03,
            0.08,
            f"RMST difference {difference:.2f} d "
            f"({difference_low:.2f} to {difference_high:.2f})",
            transform=ax_hr.transAxes,
            fontsize=6.2,
            ha="left",
            va="bottom",
        )
    add_panel_label(ax_hr, "b", x=-0.16, y=1.05, fontsize=8.0)

    # Every stage, from the source records the legend starts at.
    display = risk_flow.copy()
    display_labels = {
        "source_rows": "Source records",
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
        [
            display_labels.get(value, value.replace("_", " "))
            for value in display["stage"]
        ],
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
    unit = sealed.analysis_unit_label
    ax_flow.set_xlabel(f"{unit[:1].upper()}{unit[1:]} (n)")
    ax_flow.set_title("Risk-set accounting", loc="left")
    add_panel_label(ax_flow, "c", x=-0.16, y=1.05, fontsize=8.0)

    ph_display = ph_table.copy()
    required_ph_columns = {"covariate", "p_value", "declared_alpha"}
    if not required_ph_columns.issubset(ph_display.columns):
        missing = sorted(required_ph_columns - set(ph_display.columns))
        raise ValueError(
            "landmark survival PH diagnostics lack columns: " + ", ".join(missing)
        )
    # The signed rule judges the exposure term and the Bonferroni global test
    # at alpha; the other terms enter it only through the global test.
    decisive_terms = {"global", sealed.derived_exposure_column}
    if not decisive_terms.issubset(set(ph_display["covariate"].astype(str))):
        raise ValueError(
            "landmark survival PH diagnostics lack the global or exposure test"
        )
    ph_values = np.asarray(ph_display["p_value"], dtype=float)
    alpha_values = np.asarray(ph_display["declared_alpha"], dtype=float)
    if (
        len(ph_values) == 0
        or not np.isfinite(ph_values).all()
        or np.any(ph_values <= 0)
        or np.any(ph_values > 1)
        or not np.isfinite(alpha_values).all()
        or np.any(alpha_values <= 0)
        or np.any(alpha_values >= 1)
        or not np.allclose(alpha_values, alpha_values[0])
    ):
        raise ValueError(
            "landmark survival PH diagnostics require finite p values and one "
            "declared alpha in (0, 1)"
        )
    ph_display["p_value"] = np.maximum(ph_values, np.finfo(float).tiny)
    ph_display["neg_log10_p"] = -np.log10(ph_display["p_value"])
    ph_display = ph_display.sort_values("neg_log10_p", ascending=True)
    ph_labels = [
        "Global" if value == "global" else display_label(value)
        for value in ph_display["covariate"]
    ]
    y_ph = np.arange(len(ph_display))
    alpha = float(alpha_values[0])
    ax_ph.barh(
        y_ph,
        ph_display["neg_log10_p"],
        color=[
            palette["orange"] if term in decisive_terms else palette["orange_soft"]
            for term in ph_display["covariate"].astype(str)
        ],
    )
    ax_ph.axvline(
        -np.log10(alpha),
        color=palette["neutral"],
        linestyle="--",
        linewidth=0.8,
    )
    ax_ph.set_yticks(y_ph, ph_labels, fontsize=5.3)
    ax_ph.set_xlabel(r"Schoenfeld test $-\log_{10}(p)$")
    ax_ph.set_title("Proportional-hazards diagnostics", loc="left")
    add_panel_label(ax_ph, "d", x=-0.16, y=1.05, fontsize=8.0)

    # Plan-time panel promises live on the sealed authority; each rendered
    # panel repeats its role and exact digest-bound sources so the
    # end-of-execute join can bind them without prose inference.
    panel_sources = {
        panel.panel_id: list(panel.source_products)
        for panel in sealed.figure_panel_templates()
    }

    def _panel_metadata(panel_id: str) -> dict[str, Any]:
        return {"placement": "main", "source_products": panel_sources[panel_id]}

    contract = make_figure_contract(
        figure_id="landmark_survival_suite",
        core_claim=(
            "Post-landmark survival and risk-set accounting are shown with the "
            "signed proportional-hazards decision; a constant Cox effect is "
            + (
                "shown only because the assumption was not rejected."
                if headline_hr_authorized
                else "withheld because the assumption was rejected."
            )
        ),
        panels=[
            {
                "panel_id": "a",
                "title": "Unadjusted landmark Kaplan-Meier survival",
                "role": "temporal_absolute_risk",
                "chart_type": "kaplan_meier_curve",
                "claim": "Unadjusted absolute post-landmark survival is displayed by the frozen incident-exposure groups.",
                "evidence_ids": [],
                "review_risk": "This is an observational landmark comparison and does not identify a causal exposure effect.",
                "metadata": _panel_metadata("a"),
            },
            {
                "panel_id": "b",
                "title": (
                    "Adjusted Cox association"
                    if headline_hr_authorized
                    else (
                        "Time-varying adjusted association"
                        if promotes_time_varying
                        else "PH-free survival contrast"
                    )
                ),
                "role": ("survival_effect"),
                "chart_type": (
                    "hazard_ratio_forest"
                    if headline_hr_authorized
                    else (
                        "time_varying_hazard_ratio_forest"
                        if promotes_time_varying
                        else "rmst_difference_forest"
                    )
                ),
                "claim": (
                    "The adjusted hazard ratio quantifies the prespecified descriptive prognostic association."
                    if headline_hr_authorized
                    else (
                        "The prespecified extended Cox model reports adjusted interval-specific associations instead of one constant hazard ratio."
                        if promotes_time_varying
                        else "The unadjusted restricted-mean survival-time difference gives a PH-free descriptive group contrast over the post-landmark horizon."
                    )
                ),
                "evidence_ids": [],
                "review_risk": (
                    "Interpretation depends on proportional-hazards diagnostics and residual confounding remains possible."
                    if headline_hr_authorized
                    else (
                        "Interval-specific hazard ratios remain observational sensitivities and do not establish a causal effect of the exposure."
                        if promotes_time_varying
                        else "The RMST contrast is unadjusted and observational; do not recover or quote the source-table hazard ratio as a constant headline effect."
                    )
                ),
                "metadata": _panel_metadata("b"),
            },
            {
                "panel_id": "c",
                "title": "Risk-set accounting",
                "role": "cohort_accounting",
                "chart_type": "cohort_flow",
                "claim": "The analytic denominator is traceable through endpoint, landmark and exposure-timing gates.",
                "evidence_ids": [],
                "review_risk": "Excluded prevalent or timing-unknown exposure rows define the supported estimand boundary.",
                "metadata": _panel_metadata("c"),
            },
            {
                "panel_id": "d",
                "title": "Proportional-hazards diagnostics",
                "role": "diagnostics",
                "chart_type": "schoenfeld_plot",
                "claim": "Schoenfeld-residual tests disclose whether the fitted Cox proportional-hazards assumption is rejected.",
                "evidence_ids": [],
                "review_risk": "A diagnostic p value does not repair non-proportional hazards; the signed handling policy still governs reportability.",
                "metadata": _panel_metadata("d"),
            },
        ],
        export_formats=("svg", "pdf", "png"),
        source_data=(
            "landmark_km_curve.csv",
            "landmark_cox_summary.csv",
            "landmark_risk_set_flow.csv",
            "landmark_ph_diagnostics.csv",
            *(("landmark_rmst_summary.csv",) if rmst_table is not None else ()),
            *(
                ("landmark_time_varying_cox_summary.csv",)
                if time_varying_table is not None
                else ()
            ),
        ),
        statistics_note=(
            "Kaplan-Meier estimates are unadjusted and use the post-landmark clock; "
            "their direction can differ from the covariate-adjusted Cox estimate. "
            "The Cox model reports a Wald 95% confidence interval and a Schoenfeld "
            "residual audit. The constant hazard-ratio estimate is replaced by "
            "prespecified interval-specific coefficients when available and an "
            "unadjusted restricted-mean survival-time difference whenever the "
            "signed PH policy rejects reportability."
        ),
        image_integrity_note="All plotted values are rendered from digest-bound upstream result tables.",
        reader_caption=_reader_legend(
            headline_hr_authorized=headline_hr_authorized,
            promotes_time_varying=promotes_time_varying,
        ),
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


def build_survival_manuscript_projection(
    *, interval_count: int, proportional_hazards_rejected: bool
) -> dict[str, object]:
    """Build the reporting projection owned by the signed survival executor.

    The hazard ratios and the PH decision are host scientific claims compiled
    from the suite's reporting envelope; host claim placement reports each in
    the survival results and restores the Conclusion from the primary one.
    The projection adds what no placement reaches: the primary hazard-ratio
    tokens in the abstract Results.  The restricted-mean contrast has no
    claim type, so it is one neutral numeric sentence, worded in the strict
    Results grammar, in the abstract Results and the survival results.  Like
    the claims, it reads as an estimate with its confidence interval: a p value
    below 0.001 has no display the numeric binder can trace.
    """

    if interval_count <= 0:
        raise ValueError("survival manuscript projection requires intervals")
    abstract = {"kind": "abstract_label", "label": "Results"}
    survival = {
        "kind": "markdown_heading",
        "label": PRIMARY_RESULT_HEADINGS_BY_FAMILY["survival"],
    }
    rmst_fragments = [
        {"text": "The unadjusted restricted mean survival time at a horizon of "},
        # The horizon is the endpoint less the landmark, so a landmark that is
        # not a whole day gives a fractional horizon; print it as Methods does.
        {"numeric_path": "rmst.tau_days_from_landmark", "format_spec": ".6g"},
        {"text": " days was "},
        {"numeric_path": "rmst.exposed_rmst_days", "format_spec": ".3f"},
        {"text": " days in the exposed group and "},
        {"numeric_path": "rmst.comparator_rmst_days", "format_spec": ".3f"},
        {"text": " days in the comparator group, a difference of "},
        {"numeric_path": "rmst.difference_days", "format_spec": ".3f"},
        {"text": " days (95% CI, "},
        {"numeric_path": "rmst.ci_low", "format_spec": ".3f"},
        {"text": " to "},
        {"numeric_path": "rmst.ci_high", "format_spec": ".3f"},
        {"text": ")."},
    ]
    primary_claims = (
        tuple(interval_hazard_ratio_claim_id(position) for position in range(1, interval_count + 1))
        if proportional_hazards_rejected
        else (CONSTANT_HAZARD_RATIO_CLAIM_ID,)
    )
    return {
        "schema_version": "easyicu.manuscript_projection/2",
        "claims": [
            {
                "claim_id": "restricted_mean_survival_contrast",
                "targets": [abstract, survival],
                "fragments": rmst_fragments,
            },
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


def _executed_survival_design(sealed: LandmarkSurvivalRuntimeAuthority) -> dict[str, Any]:
    """The design this run applied, read from the sealed contract it executed."""

    followup_days = float(sealed.endpoint_horizon_days) - float(sealed.landmark_hours) / 24.0
    return executed_method_design_payload(
        LandmarkSurvivalDesign(
            schema_version=EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
            design_kind="landmark_survival",
            time_origin=sealed.endpoint_time_origin,
            landmark_hours=float(sealed.landmark_hours),
            endpoint_horizon_days=float(sealed.endpoint_horizon_days),
            prevalent_exposure_cutoff_hours=float(sealed.prevalent_exposure_cutoff_hours),
            exposure_window_end_hours=float(sealed.exposure_window_hours[1]),
            n_adjustment_covariates=len(sealed.adjustment_columns),
            effect_model="cox_proportional_hazards_efron_ties",
            interval_method=sealed.uncertainty_method,
            proportional_hazards_test="schoenfeld_residuals",
            proportional_hazards_alpha=float(sealed.proportional_hazards_alpha),
            time_varying_cutpoints_days=[
                float(cut) for cut in sealed.time_varying_interval_cutpoints_days
            ],
            rmst_horizon_days=followup_days if sealed.rmst_product is not None else None,
            exposure_onset_representation=sealed.exposure_onset_representation,
        )
    )


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
    from lifelines.exceptions import ConvergenceWarning

    from ...figures.base import km_estimate
    from ...methods.ph_schoenfeld import ph_test
    from ...methods.rmst import rmst, rmst_difference
    from ...methods.time_varying_cox import fit_piecewise_time_varying_cox

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
        *(
            column
            for column in sealed.adjustment_columns
            if column not in sealed.categorical_adjustment_columns
        ),
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
        & (event.eq(1) | followup.ge(float(sealed.endpoint_horizon_days)))
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
        raise ValueError(
            "landmark survival risk set lacks an estimable exposure contrast"
        )
    if int(analysis[sealed.derived_event_column].sum()) < 10:
        raise ValueError("landmark survival risk set has insufficient event support")

    missingness_measurement_audit = {
        "source_n": int(len(working)),
        "landmark_population_n": int(len(analysis)),
        "source_missing_n_by_column": {
            column: int(working[column].isna().sum())
            for column in sealed.required_columns
        },
        "landmark_missing_n_by_model_column": {
            column: int(analysis[column].isna().sum())
            for column in sealed.adjustment_columns
        },
    }

    risk_rows = [
        ("source_rows", len(working)),
        ("valid_fixed_horizon_endpoint", int(endpoint_valid.sum())),
        (
            "alive_and_observed_at_landmark",
            int((endpoint_valid & alive_at_landmark).sum()),
        ),
        (
            "exposure_status_and_timing_supported",
            int(
                (endpoint_valid & alive_at_landmark & status_valid & timing_known).sum()
            ),
        ),
        ("landmark_analysis_population", len(analysis)),
    ]
    risk_flow = pd.DataFrame(
        [
            {
                "stage_order": index + 1,
                "stage": stage,
                "count": int(count),
                "source_denominator": int(len(working)),
                "percent_of_source": (
                    100.0 * count / len(working) if len(working) else None
                ),
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
        subset = analysis.loc[
            analysis[sealed.derived_exposure_column].eq(exposure_value)
        ]
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
            raise ValueError(
                f"landmark survival categorical column {column!r} has no contrast"
            )
        pieces.append(encoded)
    model_frame = pd.concat(pieces, axis=1).dropna().astype(float)
    if (
        len(model_frame) < 100
        or int(model_frame[sealed.derived_event_column].sum()) < 10
    ):
        raise ValueError("landmark survival complete-case model is not estimable")
    covariates = [
        column
        for column in model_frame.columns
        if column not in {sealed.derived_time_column, sealed.derived_event_column}
    ]
    # lifelines reports separation and non-convergence as warnings and still
    # returns coefficients, a separated term's diverging; that is no result.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitter = CoxPHFitter()
        fitter.fit(
            model_frame,
            duration_col=sealed.derived_time_column,
            event_col=sealed.derived_event_column,
        )
    nonconvergence = [
        str(item.message).split(". ", 1)[0]
        for item in caught if issubclass(item.category, ConvergenceWarning)
    ]
    if nonconvergence:
        raise ValueError(
            f"landmark survival Cox model did not converge: {nonconvergence[0]}"
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
    primary_rows = cox_table.loc[cox_table["term"].eq(sealed.derived_exposure_column)]
    if len(primary_rows) != 1:
        raise ValueError("landmark survival Cox result lacks one exposure row")
    primary_row = primary_rows.iloc[0].to_dict()
    for name in ("hazard_ratio", "ci_low", "ci_high", "standard_error", "p_value"):
        _landmark_finite_float(primary_row[name], label=name)

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
    global_p = _landmark_finite_float(
        global_rows["p_value"].iloc[0], label="global PH p"
    )
    exposure_p = _landmark_finite_float(
        exposure_rows["p_value"].iloc[0], label="exposure PH p"
    )
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

    rmst_table = None
    if sealed.rmst_product is not None:
        tau = float(sealed.endpoint_horizon_days) - landmark_days
        comparator = analysis.loc[analysis[sealed.derived_exposure_column].eq(0)]
        exposed_group = analysis.loc[analysis[sealed.derived_exposure_column].eq(1)]
        comparator_rmst = rmst(
            comparator[sealed.derived_time_column],
            comparator[sealed.derived_event_column],
            tau,
        )
        exposed_rmst = rmst(
            exposed_group[sealed.derived_time_column],
            exposed_group[sealed.derived_event_column],
            tau,
        )
        contrast = rmst_difference(
            analysis[sealed.derived_time_column],
            analysis[sealed.derived_event_column],
            analysis[sealed.derived_exposure_column],
            tau,
        )
        contrast_ci_low, contrast_ci_high = contrast["ci"]
        rmst_table = pd.DataFrame(
            [
                {
                    "contrast": (
                        f"{sealed.exposed_group_label} versus "
                        f"{sealed.comparator_group_label}"
                    ),
                    "tau_days_from_landmark": tau,
                    "exposed_rmst_days": exposed_rmst.rmst,
                    "exposed_rmst_ci_low": exposed_rmst.ci_low,
                    "exposed_rmst_ci_high": exposed_rmst.ci_high,
                    "comparator_rmst_days": comparator_rmst.rmst,
                    "comparator_rmst_ci_low": comparator_rmst.ci_low,
                    "comparator_rmst_ci_high": comparator_rmst.ci_high,
                    "rmst_difference_days": -float(contrast["diff"]),
                    "standard_error": float(contrast["se_diff"]),
                    "ci_low": -float(contrast_ci_high),
                    "ci_high": -float(contrast_ci_low),
                    "p_value": float(contrast["p_value"]),
                    "adjustment": "unadjusted_kaplan_meier_plugin",
                }
            ]
        )

    time_varying_table = None
    exposure_intervals = None
    if sealed.time_varying_effect_method is not None:
        time_varying_table = fit_piecewise_time_varying_cox(
            model_frame,
            duration_col=sealed.derived_time_column,
            event_col=sealed.derived_event_column,
            covariates=covariates,
            interval_cutpoints=sealed.time_varying_interval_cutpoints_days,
            exposure_col=sealed.derived_exposure_column,
        )
        exposure_intervals = time_varying_table.loc[time_varying_table["is_exposure"]]
        if len(exposure_intervals) != (
            len(sealed.time_varying_interval_cutpoints_days) + 1
        ):
            raise ValueError(
                "landmark survival time-varying result lacks every exposure interval"
            )

    # When the prespecified PH test rejects, the constant hazard ratio is not a
    # result: it stays a diagnostic row of the Cox table and never becomes a
    # summary leaf, which would make it bindable in the manuscript.
    constant_hazard_ratio = (
        {}
        if ph_violation
        else {
            "hazard_ratio": float(primary_row["hazard_ratio"]),
            "ci_low": float(primary_row["ci_low"]),
            "ci_high": float(primary_row["ci_high"]),
        }
    )
    constant_hazard_ratio_envelope = (
        {"adjusted_hazard_ratio": dict(constant_hazard_ratio)}
        if constant_hazard_ratio
        else {}
    )
    reportable_survival_results = None
    if rmst_table is not None and exposure_intervals is not None:
        rmst_row = rmst_table.iloc[0]
        reportable_survival_results = {
            "schema_version": SURVIVAL_REPORTING_SCHEMA_VERSION,
            "execution_owner": "landmark_survival_executor_v1",
            "interpretation_ceiling": "descriptive_prognostic_association_not_causal",
            # Typed claim coordinates: context concepts read by their names.
            "exposure": sealed.exposure_status_column,
            "outcome": sealed.event_column,
            "analysis_unit": sealed.analysis_unit_label,
            "landmark_hours": float(sealed.landmark_hours),
            "contrast": str(rmst_row["contrast"]),
            "adjustment_columns": list(sealed.adjustment_columns),
            **constant_hazard_ratio_envelope,
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
            "rmst": {
                "method": str(rmst_row["adjustment"]),
                "tau_days_from_landmark": float(rmst_row["tau_days_from_landmark"]),
                "exposed_rmst_days": float(rmst_row["exposed_rmst_days"]),
                "comparator_rmst_days": float(rmst_row["comparator_rmst_days"]),
                "difference_days": float(rmst_row["rmst_difference_days"]),
                "ci_low": float(rmst_row["ci_low"]),
                "ci_high": float(rmst_row["ci_high"]),
                "p_value": float(rmst_row["p_value"]),
            },
            "time_varying_adjusted_association": {
                "method": str(sealed.time_varying_effect_method),
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
            "manuscript_projection": build_survival_manuscript_projection(
                interval_count=len(exposure_intervals),
                proportional_hazards_rejected=ph_violation,
            ),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    table_one_path = out_dir / "landmark_table_one.csv"
    risk_path = out_dir / "landmark_risk_set_flow.csv"
    km_path = out_dir / "landmark_km_curve.csv"
    cox_path = out_dir / "landmark_cox_summary.csv"
    ph_path = out_dir / "landmark_ph_diagnostics.csv"
    rmst_path = out_dir / "landmark_rmst_summary.csv"
    time_varying_path = out_dir / "landmark_time_varying_cox_summary.csv"
    analysis_path = out_dir / "landmark_analysis_cohort.parquet"
    table_one.to_csv(table_one_path, index=False)
    risk_flow.to_csv(risk_path, index=False)
    km_table.to_csv(km_path, index=False)
    cox_table.to_csv(cox_path, index=False)
    ph_table.to_csv(ph_path, index=False)
    if rmst_table is not None:
        rmst_table.to_csv(rmst_path, index=False)
    if time_varying_table is not None:
        time_varying_table.to_csv(time_varying_path, index=False)
    measurement_path = out_dir / "landmark_measurement_audit.csv"
    if sealed.measurement_audit_product is not None:
        _measurement_audit_table(sealed, working, analysis).to_csv(measurement_path, index=False)
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
        "missingness_measurement_audit": missingness_measurement_audit,
        "n_events": int(model_frame[sealed.derived_event_column].sum()),
        "effect_measure": sealed.effect_measure,
        "contrast": (
            f"{sealed.exposed_group_label} versus {sealed.comparator_group_label}"
        ),
        **constant_hazard_ratio,
        "ph_global_p_value": global_p,
        "ph_exposure_p_value": exposure_p,
        "ph_status": ph_status,
        "non_ph_alternative": sealed.non_ph_alternative,
        "time_varying_effect_method": sealed.time_varying_effect_method,
        "time_varying_interval_cutpoints_days": list(
            sealed.time_varying_interval_cutpoints_days
        ),
        "rmst_difference_days": (
            None
            if rmst_table is None
            else float(rmst_table.loc[0, "rmst_difference_days"])
        ),
        "paper_authorization_allowed": not ph_violation,
        "interpretation": sealed.interpretation,
        "analysis_only": True,
        "human_attestation_required": True,
    }
    receipt_path = out_dir / "landmark_survival_runtime_receipt.json"
    receipt_path.write_text(
        json.dumps(
            receipt, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
        ),
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
    if sealed.rmst_product is not None:
        output_files[sealed.rmst_product] = rmst_path.name
    if sealed.time_varying_cox_product is not None:
        output_files[sealed.time_varying_cox_product] = time_varying_path.name
    if sealed.measurement_audit_product is not None:
        output_files[sealed.measurement_audit_product] = measurement_path.name
    return {
        "status": "ok",
        "analysis_family": "survival",
        "analysis_role": "primary",
        "deterministic_standard_analysis": LANDMARK_SURVIVAL_ANALYSIS_KIND,
        "interpretation_class": "descriptive_prognostic_association",
        # The first numeric block. The per-step numeric cap keeps headline
        # roots first and then this order, so the design's few numbers bind
        # as one exact Methods fact whatever the size of the rest.
        EXECUTED_METHOD_DESIGN_KEY: _executed_survival_design(sealed),
        "typed_cohort_input": input_product,
        "input_evidence_id": input_evidence_id,
        "input_sha256": input_sha256,
        "n_source": int(len(working)),
        "n_landmark_population": int(len(analysis)),
        "n_complete_case": int(len(model_frame)),
        # The whole risk set's events, Kaplan-Meier's and the restricted
        # mean's; ``n_events`` counts the complete-case models' events.
        "n_events_landmark_population": int(analysis[sealed.derived_event_column].sum()),
        "missingness_measurement_audit": missingness_measurement_audit,
        "n_events": int(model_frame[sealed.derived_event_column].sum()),
        "effect_measure": sealed.effect_measure,
        "contrast": (
            f"{sealed.exposed_group_label} versus {sealed.comparator_group_label}"
        ),
        **(
            {
                "hazard_ratio": constant_hazard_ratio["hazard_ratio"],
                "hazard_ratio_ci_low": constant_hazard_ratio["ci_low"],
                "hazard_ratio_ci_high": constant_hazard_ratio["ci_high"],
            }
            if constant_hazard_ratio
            else {}
        ),
        "proportional_hazards_status": ph_status,
        "non_ph_alternative": sealed.non_ph_alternative,
        "time_varying_effect_method": sealed.time_varying_effect_method,
        "time_varying_interval_cutpoints_days": list(
            sealed.time_varying_interval_cutpoints_days
        ),
        "rmst_difference_days": (
            None
            if rmst_table is None
            else float(rmst_table.loc[0, "rmst_difference_days"])
        ),
        **(
            {"reportable_survival_results": reportable_survival_results}
            if reportable_survival_results is not None
            else {}
        ),
        "paper_authorization_allowed": False,
        "analysis_only": True,
        "human_attestation_required": True,
        "analysis_cohort_file": analysis_path.name,
        "scientific_runtime_receipt": receipt,
        MANUSCRIPT_TABLES_KEY: _manuscript_tables(sealed, analysis),
        "output_files": output_files,
    }


def run_landmark_survival_figure(
    *,
    km_table: Any,
    cox_table: Any,
    rmst_table: Any | None,
    risk_flow: Any,
    ph_table: Any,
    source_paths: Mapping[str, Path],
    authority: LandmarkSurvivalRuntimeAuthority | Mapping[str, Any],
    out_dir: Path,
    time_varying_table: Any | None = None,
) -> dict[str, Any]:
    """Render only from the digest-bound result tables the authority declares."""

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
        sealed.ph_product: "landmark_ph_diagnostics.csv",
    }
    if sealed.rmst_product is not None:
        source_filename_by_product[sealed.rmst_product] = "landmark_rmst_summary.csv"
    if sealed.time_varying_cox_product is not None:
        source_filename_by_product[sealed.time_varying_cox_product] = (
            "landmark_time_varying_cox_summary.csv"
        )
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
        rmst_table=None if rmst_table is None else pd.DataFrame(rmst_table),
        time_varying_table=(
            None if time_varying_table is None else pd.DataFrame(time_varying_table)
        ),
        risk_flow=pd.DataFrame(risk_flow),
        ph_table=pd.DataFrame(ph_table),
        sealed=sealed,
        out_dir=out_dir,
    )
    receipt_path = out_dir / "landmark_survival_figure_runtime_receipt.json"
    ph_statuses = {
        str(value).strip()
        for value in pd.DataFrame(ph_table).get("ph_status", pd.Series(dtype=str))
        if str(value).strip()
    }
    ph_rejected = any(status.startswith("violation_") for status in ph_statuses)
    promotes_rmst = bool(
        ph_rejected
        and time_varying_table is None
        and sealed.non_ph_alternative == "unadjusted_rmst_difference"
        and sealed.rmst_product is not None
        and rmst_table is not None
    )
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "easyicu.landmark_survival_figure_runtime_receipt/1"
                ),
                "adjustment_columns": list(sealed.adjustment_columns),
                "effect_measure": sealed.effect_measure,
                "promoted_adjustment_columns": (
                    [] if promotes_rmst else list(sealed.adjustment_columns)
                ),
                "promoted_effect_measure": (
                    "restricted_mean_survival_time_difference"
                    if promotes_rmst
                    else (
                        "interval_specific_hazard_ratio"
                        if ph_rejected and time_varying_table is not None
                        else sealed.effect_measure
                    )
                ),
                "source_sha256": {
                    product: hashlib.sha256(
                        Path(source_paths[product]).read_bytes()
                    ).hexdigest()
                    for product in sealed.figure_input_products
                },
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    outputs["runtime_receipt"] = receipt_path
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
            rmst_table={f"bindings[{sealed.rmst_product!r}].frame" if sealed.rmst_product is not None else "None"},
            time_varying_table={f"bindings[{sealed.time_varying_cox_product!r}].frame" if sealed.time_varying_cox_product is not None else "None"},
            risk_flow=bindings[{sealed.risk_set_product!r}].frame,
            ph_table=bindings[{sealed.ph_product!r}].frame,
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
