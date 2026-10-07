"""Source-bound composite figure of the continuous-exposure survival suite.

The renderer reads only the digest-bound result tables the sealed authority
declares and fits nothing.  Panel (a) draws the descriptive groups its
Kaplan-Meier table records: the exposure tertiles, or the whole risk set when
the suite could not form them.  Panel (b) follows the suite's own PH decision:
the adjusted hazard-ratio curve of a model with a constant effect is drawn only
when the prespecified test did not reject the assumption, and the per-unit
hazard ratios of the interval model replace it when it did.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any, Mapping

from ...authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from ...authority.landmark_continuous_survival_runtime import (
    CONTINUOUS_SURVIVAL_FIGURE_METHOD,
    LandmarkContinuousSurvivalRuntimeAuthority,
)
from ...contracts.executed_method_design import WHOLE_RISK_SET_REASON_WORDS
from ...schema import AnalysisPlan, AnalysisStep

LANDMARK_CONTINUOUS_SURVIVAL_FIGURE_ANALYSIS_KIND = CONTINUOUS_SURVIVAL_FIGURE_METHOD
_FIGURE_STEM = "landmark_continuous_survival_suite"
#: Inches each proportional-hazards term needs to stay legible in panel d.
_PH_ROW_INCHES = 0.12


def _sealed(
    authority: LandmarkContinuousSurvivalRuntimeAuthority | Mapping[str, Any] | None,
) -> LandmarkContinuousSurvivalRuntimeAuthority | None:
    if authority is None:
        return None
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkContinuousSurvivalRuntimeAuthority):
        return None
    return sealed


def landmark_continuous_survival_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: LandmarkContinuousSurvivalRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    sealed = _sealed(authority)
    return sealed is not None and sealed.governed_figure_step(plan) == step


def _source_filenames(sealed: LandmarkContinuousSurvivalRuntimeAuthority) -> dict[str, str]:
    return {
        sealed.km_product: "continuous_landmark_km_curve.csv",
        sealed.cox_product: "continuous_landmark_cox_summary.csv",
        sealed.spline_product: "continuous_landmark_spline_curve.csv",
        sealed.time_varying_cox_product: "continuous_landmark_time_varying_cox_summary.csv",
        sealed.risk_set_product: "continuous_landmark_risk_set_flow.csv",
        sealed.ph_product: "continuous_landmark_ph_diagnostics.csv",
    }


def _whole_risk_set_reason(km_table: Any) -> str | None:
    """The reason the suite described its whole risk set; ``None`` for tertiles."""

    groupings = {str(value) for value in km_table.get("descriptive_grouping", [])}
    reasons = {
        str(value)
        for value in km_table.get("descriptive_grouping_reason", [])
        if isinstance(value, str) and value
    }
    if groupings == {"value_tertiles"} and not reasons:
        return None
    if groupings == {"whole_risk_set"} and len(reasons) == 1:
        (reason,) = reasons
        if reason in WHOLE_RISK_SET_REASON_WORDS:
            return reason
    raise ValueError(
        "continuous survival KM table lacks one stated descriptive grouping"
    )


def _ph_rejected(ph_table: Any) -> bool:
    statuses = {
        str(value).strip() for value in ph_table.get("ph_status", []) if str(value).strip()
    }
    if not statuses:
        raise ValueError("continuous survival PH diagnostics lack the signed status")
    return any(status.startswith("violation_") for status in statuses)


_RATIO_TICKS = (0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0)


def _ratio_y_axis(axes: Any, *, low: float, high: float) -> None:
    """A log hazard-ratio y axis labelled with a few plain multipliers, 1 included."""

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    if not 0 < low <= high:
        raise ValueError("continuous survival hazard-ratio bounds must be positive")
    span_low, span_high = min(low, 1.0) / 1.08, max(high, 1.0) * 1.08
    ticks = [value for value in _RATIO_TICKS if span_low <= value <= span_high]
    if len(ticks) > 5:
        ticks = sorted({ticks[0], 1.0, ticks[-1], *ticks[1:-1:2]})
    axes.set_yscale("log")
    axes.set_ylim(span_low, span_high)
    axes.yaxis.set_major_locator(FixedLocator(ticks))
    axes.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:.2f}".rstrip("0").rstrip("."))
    )
    axes.yaxis.set_minor_locator(FixedLocator([]))
    axes.yaxis.set_minor_formatter(NullFormatter())


def _reader_legend(
    *,
    ph_rejected: bool,
    spline_estimated: bool,
    exposure: str,
    unit: str | None,
    whole_risk_set_reason: str | None,
) -> str:
    """The figure's source-bound legend: what each drawn panel shows, and no value."""

    scale = f"{exposure} ({unit})" if unit else exposure
    survival = (
        "(a) Unadjusted Kaplan-Meier survival after the landmark by tertile of the "
        "exposure, with the number at risk below."
        if whole_risk_set_reason is None
        else (
            "(a) Unadjusted Kaplan-Meier survival after the landmark for the whole risk "
            "set, with the number at risk below; exposure tertiles were not formed, "
            f"because {WHOLE_RISK_SET_REASON_WORDS[whole_risk_set_reason]}."
        )
    )
    if ph_rejected:
        estimate = (
            f"(b) Adjusted hazard ratios per unit of {scale} for each follow-up "
            "interval, with their confidence intervals, from the prespecified "
            "interval model, drawn instead of one constant hazard ratio because the "
            "proportional-hazards assumption was rejected."
        )
    elif spline_estimated:
        estimate = (
            f"(b) Adjusted hazard ratio across {scale} relative to its median, from "
            "the restricted cubic spline model (solid line, shaded confidence band) "
            "and the linear model (dashed line), between the 10th and 90th "
            "percentiles of the exposure."
        )
    else:
        estimate = (
            f"(b) Adjusted hazard ratio across {scale} relative to its median from "
            "the linear model, with its confidence band, between the 10th and 90th "
            "percentiles of the exposure; the spline check had no result."
        )
    return " ".join((
        survival,
        estimate,
        "(c) Risk-set accounting from the source records through the endpoint, "
        "landmark and exposure-value gates to the analysis population.",
        "(d) Schoenfeld residual tests of the proportional-hazards assumption for "
        "each model term and globally, the global test Bonferroni-adjusted over "
        "the terms. The dashed line marks the prespecified alpha against which the "
        "exposure term and the global test were judged; the other terms, in "
        "lighter bars, enter the decision only through the global test.",
        "Every value is drawn from the suite's registered result tables; the "
        "figure fits no model of its own.",
    ))


def _render(
    *,
    km_table: Any,
    spline_table: Any,
    time_varying_table: Any,
    risk_flow: Any,
    ph_table: Any,
    sealed: LandmarkContinuousSurvivalRuntimeAuthority,
    out_dir: Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from ...figures.display_labels import display_label
    from ...figures.publication import (
        add_panel_label,
        apply_publication_style,
        configure_ratio_axis,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    whole_reason = _whole_risk_set_reason(km_table)
    group_ids = (1, 2, 3) if whole_reason is None else (0,)
    ph_rejected = _ph_rejected(ph_table)
    statuses = {str(value) for value in spline_table.get("spline_status", [])}
    if len(statuses) != 1:
        raise ValueError("continuous survival spline table lacks one status")
    spline_estimated = statuses == {"estimated"}

    # Panel d lists one row per model term; beyond what its share of the
    # page holds, the figure grows rather than overlap the labels.
    base_height = 150 / 25.4
    extra = max(0.0, len(ph_table) * _PH_ROW_INCHES - 0.36 * base_height)
    fig = plt.figure(figsize=(183 / 25.4, base_height + extra))
    outer = fig.add_gridspec(
        1, 2, width_ratios=(1.3, 1.0),
        left=0.15, right=0.975, top=0.94, bottom=0.07, wspace=0.62,
    )
    left = outer[0, 0].subgridspec(3, 1, height_ratios=(2.0, 0.78, 1.0), hspace=0.6)
    right = outer[0, 1].subgridspec(
        2, 1, height_ratios=(1.2, 1.0 + extra / (0.36 * base_height)), hspace=0.42,
    )
    ax_km = fig.add_subplot(left[0])
    ax_risk = fig.add_subplot(left[1])
    ax_flow = fig.add_subplot(left[2])
    ax_hr = fig.add_subplot(right[0])
    ax_ph = fig.add_subplot(right[1])

    horizon = sealed.endpoint_horizon_days - sealed.landmark_hours / 24.0
    colors = {
        0: palette["blue"],
        1: palette["blue"],
        2: palette["teal"],
        3: palette["red"],
    }
    labels: dict[int, str] = {}
    if set(km_table["exposure_group"].astype(int)) != set(group_ids):
        raise ValueError(
            "continuous survival KM table holds other groups than it states"
        )
    for group in group_ids:
        rows = km_table.loc[km_table["exposure_group"].eq(group)].sort_values(
            "time_from_landmark_days"
        )
        labels[group] = str(rows["exposure_group_label"].iloc[0])
        ax_km.step(
            rows["time_from_landmark_days"],
            rows["survival_probability"],
            where="post",
            color=colors[group],
            linewidth=1.4,
            label=textwrap.fill(labels[group], width=34),
        )
    ax_km.set_ylim(0.0, 1.02)
    ax_km.set_xlim(0.0, horizon)
    ax_km.set_xlabel(f"Days after the {sealed.landmark_hours:g}-hour landmark")
    ax_km.set_ylabel("Survival probability")
    km_title = "Unadjusted Kaplan-Meier survival" + (
        " by tertile" if whole_reason is None else ""
    )
    ax_km.set_title(km_title, loc="left")
    ax_km.legend(loc="lower left", fontsize=5.8)
    add_panel_label(ax_km, "a", x=-0.09, fontsize=8.0)

    risk_times = np.linspace(0.0, horizon, 5)
    risk_rows: list[list[str]] = []
    for group in group_ids:
        rows = km_table.loc[km_table["exposure_group"].eq(group)].sort_values(
            "time_from_landmark_days"
        )
        counts = []
        for time_point in risk_times:
            seen = rows.loc[rows["time_from_landmark_days"].le(time_point)]
            counts.append(str(int((seen.iloc[-1] if not seen.empty else rows.iloc[0])["at_risk"])))
        risk_rows.append(counts)
    ax_risk.axis("off")
    table = ax_risk.table(
        cellText=risk_rows,
        rowLabels=(
            ["Tertile 1", "Tertile 2", "Tertile 3"]
            if whole_reason is None
            else ["Risk set"]
        ),
        colLabels=[f"{value:g}" for value in risk_times],
        cellLoc="center",
        rowLoc="right",
        bbox=[0.0, 0.0, 1.0, 0.8],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.4)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.0)
    ax_risk.text(
        -0.02, 0.98, "Number at risk", transform=ax_risk.transAxes,
        fontsize=6.0, fontweight="bold", va="top",
    )

    unit = f" ({sealed.exposure_unit})" if sealed.exposure_unit else ""
    exposure_axis = f"{sealed.exposure_label}{unit}"
    if ph_rejected:
        rows = time_varying_table.loc[time_varying_table["is_exposure"].astype(bool)]
        statuses = {str(value) for value in rows.get("model_status", [])}
        intervals = len(sealed.time_varying_interval_cutpoints_days) + 1
        if statuses != {"estimated"} or len(rows) != intervals:
            raise ValueError("continuous survival figure lacks every interval estimate")
        rows = rows.sort_values("interval_index")
        estimates = rows["hazard_ratio"].to_numpy(dtype=float)
        lows = rows["ci_low"].to_numpy(dtype=float)
        highs = rows["ci_high"].to_numpy(dtype=float)
        positions = np.arange(len(rows))
        ax_hr.errorbar(
            estimates, positions,
            xerr=np.vstack((estimates - lows, highs - estimates)),
            fmt="o", color=palette["blue"], capsize=3, linewidth=1.2,
        )
        ax_hr.axvline(1.0, color=palette["neutral"], linestyle="--", linewidth=0.8)
        configure_ratio_axis(ax_hr, lows=lows, highs=highs)
        ax_hr.set_yticks(
            positions,
            [f"{row.interval_start_days:g}–{row.interval_end_days:g} d" for row in rows.itertuples()],
        )
        ax_hr.invert_yaxis()
        ax_hr.set_xlabel("Adjusted HR per unit (95% CI)")
        ax_hr.set_title("Time-varying adjusted association", loc="left")
    else:
        curve = spline_table.sort_values("exposure_value")
        x = curve["exposure_value"].to_numpy(dtype=float)
        bounds = [curve["linear_ci_low"], curve["linear_ci_high"]]
        if spline_estimated:
            ax_hr.fill_between(
                x,
                curve["spline_ci_low"].to_numpy(dtype=float),
                curve["spline_ci_high"].to_numpy(dtype=float),
                color=palette["blue"], alpha=0.18, linewidth=0.0,
            )
            ax_hr.plot(x, curve["spline_hazard_ratio"], color=palette["blue"], linewidth=1.5)
            bounds = [curve["spline_ci_low"], curve["spline_ci_high"], *bounds]
        else:
            ax_hr.fill_between(
                x,
                curve["linear_ci_low"].to_numpy(dtype=float),
                curve["linear_ci_high"].to_numpy(dtype=float),
                color=palette["neutral"], alpha=0.18, linewidth=0.0,
            )
        ax_hr.plot(
            x, curve["linear_hazard_ratio"], color=palette["neutral"],
            linestyle="--", linewidth=1.1,
        )
        ax_hr.axhline(1.0, color=palette["neutral"], linestyle=":", linewidth=0.8)
        lows = np.concatenate([np.asarray(item, dtype=float) for item in bounds[0::2]])
        highs = np.concatenate([np.asarray(item, dtype=float) for item in bounds[1::2]])
        _ratio_y_axis(ax_hr, low=float(np.min(lows)), high=float(np.max(highs)))
        ax_hr.set_xlim(float(x.min()), float(x.max()))
        ax_hr.set_xlabel(textwrap.fill(exposure_axis, width=48), fontsize=6.4)
        ax_hr.set_ylabel("Adjusted HR vs median (95% CI)")
        ax_hr.set_title("Adjusted hazard ratio curve", loc="left")
    add_panel_label(ax_hr, "b", x=-0.18, y=1.04, fontsize=8.0)

    stage_labels = {
        "source_rows": "Source records",
        "valid_fixed_horizon_endpoint": f"Valid {sealed.endpoint_horizon_days:g}-day endpoint",
        "alive_and_observed_at_landmark": f"Alive/observed at {sealed.landmark_hours:g} h",
        "landmark_analysis_population": "Exposure recorded",
    }
    flow = risk_flow.sort_values("stage_order")
    y = np.arange(len(flow))
    ax_flow.barh(y, flow["count"], color=palette["teal"])
    ax_flow.set_yticks(
        y,
        [
            textwrap.fill(stage_labels.get(value, str(value).replace("_", " ")), width=24)
            for value in flow["stage"]
        ],
        fontsize=5.6,
    )
    for index, count in enumerate(flow["count"].astype(int)):
        ax_flow.text(count, index, f"  {count:,}", va="center", ha="left", fontsize=5.6)
    ax_flow.set_xlim(0, max(flow["count"]) * 1.22)
    ax_flow.invert_yaxis()
    unit_label = sealed.analysis_unit_label
    ax_flow.set_xlabel(f"{unit_label[:1].upper()}{unit_label[1:]} (n)")
    ax_flow.set_title("Risk-set accounting", loc="left")
    add_panel_label(ax_flow, "c", x=-0.09, y=1.08, fontsize=8.0)

    required = {"covariate", "p_value", "declared_alpha"}
    if not required.issubset(ph_table.columns):
        raise ValueError(
            "continuous survival PH diagnostics lack columns: "
            + ", ".join(sorted(required - set(ph_table.columns)))
        )
    decisive = {"global", sealed.exposure_column}
    if not decisive.issubset(set(ph_table["covariate"].astype(str))):
        raise ValueError("continuous survival PH diagnostics lack the global or exposure test")
    p_values = np.asarray(ph_table["p_value"], dtype=float)
    alphas = np.asarray(ph_table["declared_alpha"], dtype=float)
    if (
        not np.isfinite(p_values).all()
        or np.any(p_values < 0)
        or np.any(p_values > 1)
        or not np.allclose(alphas, alphas[0])
    ):
        raise ValueError("continuous survival PH diagnostics require finite p values and one alpha")
    display = ph_table.copy()
    display["neg_log10_p"] = -np.log10(np.maximum(p_values, np.finfo(float).tiny))
    display = display.sort_values("neg_log10_p", ascending=True)
    y_ph = np.arange(len(display))
    ax_ph.barh(
        y_ph,
        display["neg_log10_p"],
        color=[
            palette["orange"] if term in decisive else palette["orange_soft"]
            for term in display["covariate"].astype(str)
        ],
    )
    ax_ph.axvline(-np.log10(float(alphas[0])), color=palette["neutral"], linestyle="--", linewidth=0.8)
    ax_ph.set_yticks(
        y_ph,
        [
            "Global" if value == "global"
            else textwrap.fill(
                sealed.exposure_label if value == sealed.exposure_column else display_label(value),
                width=22,
            )
            for value in display["covariate"].astype(str)
        ],
        fontsize=5.2,
    )
    ax_ph.set_xlabel(r"Schoenfeld test $-\log_{10}(p)$")
    ax_ph.set_title("Proportional-hazards diagnostics", loc="left")
    add_panel_label(ax_ph, "d", x=-0.18, y=1.04, fontsize=8.0)

    panel_sources = {
        panel.panel_id: list(panel.source_products)
        for panel in sealed.figure_panel_templates()
    }

    def metadata(panel_id: str) -> dict[str, Any]:
        return {"placement": "main", "source_products": panel_sources[panel_id]}

    contract = make_figure_contract(
        figure_id=_FIGURE_STEM,
        core_claim=(
            "Post-landmark survival "
            + (
                "by exposure tertile"
                if whole_reason is None
                else "of the whole risk set"
            )
            + " and risk-set accounting are "
            "shown with the signed proportional-hazards decision; a constant "
            "per-unit Cox effect is "
            + (
                "withheld because the assumption was rejected."
                if ph_rejected
                else "shown only because the assumption was not rejected."
            )
        ),
        panels=[
            {
                "panel_id": "a",
                "title": km_title,
                "role": "temporal_absolute_risk",
                "chart_type": "kaplan_meier_curve",
                "claim": (
                    "Unadjusted post-landmark survival is displayed by descriptive exposure tertile."
                    if whole_reason is None
                    else "Unadjusted post-landmark survival is displayed for the whole risk set, without exposure tertiles."
                ),
                "evidence_ids": [],
                "review_risk": (
                    "Tertiles describe the cohort; they are not the modelled exposure and do not identify a causal effect."
                    if whole_reason is None
                    else "The curve describes the whole risk set; it shows no exposure contrast and identifies no causal effect."
                ),
                "metadata": metadata("a"),
            },
            {
                "panel_id": "b",
                "title": (
                    "Time-varying adjusted association"
                    if ph_rejected
                    else "Adjusted hazard ratio curve"
                ),
                "role": "survival_effect",
                "chart_type": (
                    "time_varying_hazard_ratio_forest" if ph_rejected else "hazard_ratio_curve"
                ),
                "claim": (
                    "The prespecified interval model reports adjusted per-unit hazard ratios by follow-up interval instead of one constant estimate."
                    if ph_rejected
                    else "The adjusted hazard ratio is shown across the exposure range relative to its median, with the spline check of the linear term."
                ),
                "evidence_ids": [],
                "review_risk": (
                    "Interval-specific hazard ratios are observational and do not establish a causal effect of the exposure."
                    if ph_rejected
                    else "Interpretation depends on proportional-hazards diagnostics, and residual confounding remains possible."
                ),
                "metadata": metadata("b"),
            },
            {
                "panel_id": "c",
                "title": "Risk-set accounting",
                "role": "cohort_accounting",
                "chart_type": "cohort_flow",
                "claim": "The analytic denominator is traceable through the endpoint, landmark and exposure-value gates.",
                "evidence_ids": [],
                "review_risk": "Records without a recorded exposure value define the supported estimand boundary.",
                "metadata": metadata("c"),
            },
            {
                "panel_id": "d",
                "title": "Proportional-hazards diagnostics",
                "role": "diagnostics",
                "chart_type": "schoenfeld_plot",
                "claim": "Schoenfeld-residual tests disclose whether the Cox proportional-hazards assumption is rejected.",
                "evidence_ids": [],
                "review_risk": "A diagnostic p value does not repair non-proportional hazards; the signed handling policy still governs reportability.",
                "metadata": metadata("d"),
            },
        ],
        export_formats=("svg", "pdf", "png"),
        source_data=tuple(_source_filenames(sealed).values()),
        statistics_note=(
            "Kaplan-Meier estimates are unadjusted and use the post-landmark clock. "
            "The Cox model reports a Wald 95% confidence interval per unit of the "
            "exposure and a Schoenfeld residual audit; prespecified interval-specific "
            "estimates replace the constant estimate when the signed PH policy "
            "rejects it. The spline check is reported and chooses no estimate."
        ),
        image_integrity_note="All plotted values are rendered from digest-bound upstream result tables.",
        reader_caption=_reader_legend(
            ph_rejected=ph_rejected,
            spline_estimated=spline_estimated,
            exposure=sealed.exposure_label,
            unit=sealed.exposure_unit,
            whole_risk_set_reason=whole_reason,
        ),
    )
    # The risk-set stage names are the leftmost text; the wider export gutter
    # keeps their full extent inside the canvas.
    outputs = save_publication_figure(
        fig,
        out_dir / _FIGURE_STEM,
        contract=contract,
        formats=("svg", "pdf", "png"),
        dpi=300,
        pad_inches=0.12,
    )
    plt.close(fig)
    return outputs


def run_landmark_continuous_survival_figure(
    *,
    km_table: Any,
    spline_table: Any,
    time_varying_table: Any,
    risk_flow: Any,
    ph_table: Any,
    source_paths: Mapping[str, Path],
    authority: LandmarkContinuousSurvivalRuntimeAuthority | Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Render only from the digest-bound result tables the authority declares."""

    import pandas as pd

    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("continuous survival figure received the wrong authority kind")
    out_dir.mkdir(parents=True, exist_ok=True)
    filenames = _source_filenames(sealed)
    copied: list[str] = []
    for product in sealed.figure_input_products:
        source = Path(source_paths[product]).resolve()
        destination = out_dir / filenames[product]
        shutil.copyfile(source, destination)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(
            destination.read_bytes()
        ).digest():
            raise ValueError("continuous survival figure source changed while copying")
        copied.append(destination.name)
    ph_frame = pd.DataFrame(ph_table)
    outputs = _render(
        km_table=pd.DataFrame(km_table),
        spline_table=pd.DataFrame(spline_table),
        time_varying_table=pd.DataFrame(time_varying_table),
        risk_flow=pd.DataFrame(risk_flow),
        ph_table=ph_frame,
        sealed=sealed,
        out_dir=out_dir,
    )
    ph_rejected = _ph_rejected(ph_frame)
    receipt_path = out_dir / "landmark_continuous_survival_figure_runtime_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.landmark_continuous_survival_figure_runtime_receipt/1",
                "adjustment_columns": list(sealed.adjustment_columns),
                "effect_measure": sealed.effect_measure,
                "promoted_adjustment_columns": list(sealed.adjustment_columns),
                "promoted_effect_measure": (
                    "interval_specific_hazard_ratio_per_unit"
                    if ph_rejected
                    else sealed.effect_measure
                ),
                "source_sha256": {
                    product: hashlib.sha256(Path(source_paths[product]).read_bytes()).hexdigest()
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
        raise ValueError("continuous survival figure export is missing")
    return {
        "status": "ok",
        "rendering_only": True,
        "deterministic_standard_analysis": LANDMARK_CONTINUOUS_SURVIVAL_FIGURE_ANALYSIS_KIND,
        "source_data_files": copied,
        "figure_assets": {key: value.name for key, value in outputs.items()},
        "output_files": {sealed.figure_product: figure_file.name},
    }


def landmark_continuous_survival_figure_executor_code(
    step: AnalysisStep,
    *,
    authority: LandmarkContinuousSurvivalRuntimeAuthority | Mapping[str, Any],
) -> str:
    """Return the host-owned renderer for the sealed continuous survival tables."""

    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("continuous survival figure requires its sealed authority")
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.landmark_continuous_survival_figure import (
            run_landmark_continuous_survival_figure,
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
        summary = run_landmark_continuous_survival_figure(
            km_table=bindings[{sealed.km_product!r}].frame,
            spline_table=bindings[{sealed.spline_product!r}].frame,
            time_varying_table=bindings[{sealed.time_varying_cox_product!r}].frame,
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
    "LANDMARK_CONTINUOUS_SURVIVAL_FIGURE_ANALYSIS_KIND",
    "landmark_continuous_survival_figure_executor_code",
    "landmark_continuous_survival_figure_executor_owns_step",
    "run_landmark_continuous_survival_figure",
]
