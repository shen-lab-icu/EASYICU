"""Source-bound composite figure of the signed target trial suite.

The renderer reads only the digest-bound result tables the sealed authority
declares and fits nothing.  It leads with the protocol, as the causal figure
strategy asks (``planning.figure_strategy``): (a) the trial's clock -- ICU
admission, time zero, the grace period and the horizon -- with each strategy
and the eligibility counts; (b) the weighted cumulative risk of the outcome
under each strategy over follow-up, with the primary contrast; (c) the
balance of each confounder among the clones followed through the grace
period, without and with the weights; (d) the contrast under each weighting
the host prespecified -- the stabilized weights as estimated, truncated, and
none -- beside the weights' extremes.
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
from ...authority.target_trial_runtime import (
    TARGET_TRIAL_FIGURE_METHOD,
    TargetTrialRuntimeAuthority,
)
from ...contracts.target_trial_design import TARGET_TRIAL_STOP_THRESHOLDS
from ...numeric_scalars import coerce_optional_finite_float
from ...schema import AnalysisPlan, AnalysisStep

TARGET_TRIAL_FIGURE_ANALYSIS_KIND = TARGET_TRIAL_FIGURE_METHOD
_FIGURE_STEM = "target_trial_emulation"
_WEIGHTINGS = (
    ("stabilized", "Stabilized weights"),
    ("stabilized_truncated", "Truncated weights"),
    ("unweighted", "No weights"),
)


def _sealed(
    authority: TargetTrialRuntimeAuthority | Mapping[str, Any] | None,
) -> TargetTrialRuntimeAuthority | None:
    if authority is None:
        return None
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, TargetTrialRuntimeAuthority):
        return None
    return sealed


def target_trial_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: TargetTrialRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    sealed = _sealed(authority)
    return sealed is not None and sealed.governed_figure_step(plan) == step


def _source_filenames(sealed: TargetTrialRuntimeAuthority) -> dict[str, str]:
    return {
        sealed.protocol_product: "target_trial_protocol.csv",
        sealed.eligibility_product: "target_trial_eligibility_flow.csv",
        sealed.risk_curve_product: "target_trial_risk_curves.csv",
        sealed.effect_product: "target_trial_effect_estimates.csv",
        sealed.balance_product: "target_trial_covariate_balance.csv",
        sealed.weight_product: "target_trial_weight_diagnostics.csv",
    }


def _effect_row(effects: Any, estimand: str, weighting: str) -> Mapping[str, Any]:
    rows = effects.loc[
        effects["estimand"].eq(estimand) & effects["weighting"].eq(weighting)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"target trial effect table lacks one {estimand} row with {weighting} weights"
        )
    return rows.iloc[0].to_dict()


def _reader_legend(sealed: TargetTrialRuntimeAuthority) -> str:
    """The figure's source-bound legend: what each panel shows, and no value."""

    return " ".join(
        (
            "(a) The emulated trial's clock in hours after ICU admission: eligibility "
            "and follow-up start at time zero, the initiating strategy starts the "
            "treatment within the grace period and the deferring strategy does not; "
            "the counts follow eligibility at time zero.",
            "(b) Cumulative risk of the outcome under each strategy from time zero, "
            "from Kaplan-Meier estimates of the clones weighted by stabilized inverse "
            "probability of censoring weights, with the primary risk difference and "
            "its bootstrap percentile interval.",
            "(c) Absolute standardized mean difference of each confounder between "
            "the clones each strategy follows through the grace period and the "
            "eligible population, without (open) and with (filled) the weights; the "
            "dashed line marks the host's flag for imbalance.",
            "(d) The risk difference with stabilized weights as estimated (the "
            "primary result), with each strategy's weights truncated at "
            "prespecified percentiles, and without weights, which has no interval; "
            "the text gives the weights' largest values.",
            "Every value is drawn from the suite's registered result tables; the "
            "figure fits no model of its own. The estimates rest on the emulation's "
            "assumptions of no unmeasured confounding, positivity, correctly "
            "specified weight models and censoring at ICU exit that the baseline "
            "covariates explain.",
        )
    )


def _render(
    *,
    protocol: Any,
    eligibility: Any,
    risk_curves: Any,
    effects: Any,
    balance: Any,
    weights: Any,
    sealed: TargetTrialRuntimeAuthority,
    out_dir: Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ...figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    colors = {"initiate": palette["red"], "defer": palette["blue"]}
    labels = {"initiate": sealed.initiate_label, "defer": sealed.defer_label}
    t0, grace = sealed.time_zero_hours, sealed.grace_period_hours

    fig = plt.figure(figsize=(183 / 25.4, 130 / 25.4))
    grid = fig.add_gridspec(
        2, 2, left=0.09, right=0.98, top=0.93, bottom=0.09, wspace=0.42, hspace=0.62
    )
    ax_clock = fig.add_subplot(grid[0, 0])
    ax_risk = fig.add_subplot(grid[0, 1])
    ax_love = fig.add_subplot(grid[1, 0])
    ax_robust = fig.add_subplot(grid[1, 1])

    # (a) The trial's clock.  The axis spans ICU admission to a little past the
    # grace period; the horizon is stated, not drawn to scale.
    stages = eligibility.sort_values("stage_order")
    if list(protocol["item"])[:1] != ["eligibility"]:
        raise ValueError("target trial protocol table does not lead with eligibility")
    end = t0 + grace
    ax_clock.set_xlim(-0.5, end + max(2.0, 0.35 * end))
    ax_clock.set_ylim(-0.6, 2.6)
    ax_clock.axvspan(t0, end, color=palette["orange_soft"], zorder=0)
    for x, text in ((0.0, "ICU admission"), (t0, "Time zero"), (end, "End of grace")):
        ax_clock.axvline(x, color=palette["baseline"], linewidth=0.8)
        ax_clock.text(x, 2.45, text, ha="center", va="bottom", fontsize=5.8)
    for row, arm in ((1.6, "initiate"), (0.6, "defer")):
        ax_clock.plot(
            [t0, end + 0.3 * end], [row, row], color=colors[arm], linewidth=2.0
        )
        ax_clock.text(
            t0,
            row + 0.18,
            textwrap.shorten(labels[arm], width=34, placeholder="..."),
            fontsize=5.8,
            color=colors[arm],
        )
    first, last = int(stages["count"].iloc[0]), int(stages["count"].iloc[-1])
    ax_clock.text(
        0.0,
        -0.45,
        f"{first:,} stays in the source; {last:,} eligible at time zero; "
        f"follow-up to day {sealed.endpoint_horizon_days}",
        fontsize=5.6,
        va="bottom",
    )
    ax_clock.set_yticks([])
    ax_clock.set_xlabel("Hours after ICU admission")
    ax_clock.set_title("Target trial protocol", loc="left")
    add_panel_label(ax_clock, "a", x=-0.08, fontsize=8.0)

    # (b) Weighted cumulative risk under each strategy.
    if set(risk_curves["arm"]) != {"initiate", "defer"}:
        raise ValueError("target trial risk curves lack a strategy")
    for arm in ("initiate", "defer"):
        rows = risk_curves.loc[risk_curves["arm"].eq(arm)].sort_values(
            "hours_from_time_zero"
        )
        ax_risk.step(
            rows["hours_from_time_zero"] / 24.0,
            100.0 * rows["cumulative_risk"],
            where="post",
            color=colors[arm],
            linewidth=1.4,
            label=textwrap.fill(labels[arm], width=30),
        )
    difference = _effect_row(effects, "risk_difference", "stabilized")
    ax_risk.text(
        0.02,
        0.96,
        "Risk difference "
        f"{100.0 * float(difference['estimate']):.1f} points "
        f"({100.0 * float(difference['ci_low']):.1f} to "
        f"{100.0 * float(difference['ci_high']):.1f})",
        transform=ax_risk.transAxes,
        fontsize=5.8,
        va="top",
    )
    ax_risk.set_xlim(0.0, sealed.horizon_hours / 24.0 - t0 / 24.0)
    ax_risk.set_ylim(bottom=0.0)
    ax_risk.set_xlabel("Days from time zero")
    ax_risk.set_ylabel("Cumulative risk (%)")
    ax_risk.set_title("Weighted risk by strategy", loc="left")
    ax_risk.legend(loc="lower right", fontsize=5.6)
    add_panel_label(ax_risk, "b", x=-0.12, fontsize=8.0)

    # (c) Balance of each confounder: |SMD| without and with the weights.
    flag = float(TARGET_TRIAL_STOP_THRESHOLDS["balance_smd_flag"])
    shown = balance.dropna(subset=["smd_unweighted", "smd_weighted"])
    terms = list(dict.fromkeys(shown["label"]))
    positions = {term: index for index, term in enumerate(reversed(terms))}
    for offset, arm in ((0.15, "initiate"), (-0.15, "defer")):
        rows = shown.loc[shown["arm"].eq(arm)]
        y = [positions[label] + offset for label in rows["label"]]
        ax_love.scatter(
            rows["smd_unweighted"].abs(),
            y,
            facecolors="none",
            edgecolors=colors[arm],
            s=12,
            linewidths=0.8,
        )
        ax_love.scatter(rows["smd_weighted"].abs(), y, color=colors[arm], s=12)
    ax_love.axvline(flag, color=palette["neutral"], linestyle="--", linewidth=0.8)
    ax_love.set_yticks(range(len(terms)))
    ax_love.set_yticklabels(
        [
            textwrap.shorten(term, width=28, placeholder="...")
            for term in reversed(terms)
        ],
        fontsize=5.4,
    )
    ax_love.set_xlim(left=0.0)
    ax_love.set_xlabel("Absolute standardized mean difference")
    ax_love.set_title("Balance of confounders", loc="left")
    add_panel_label(ax_love, "c", x=-0.42, fontsize=8.0)

    # (d) The contrast under each prespecified weighting.
    for row, (weighting, name) in enumerate(reversed(_WEIGHTINGS)):
        values = _effect_row(effects, "risk_difference", weighting)
        point = 100.0 * float(values["estimate"])
        ci_low = coerce_optional_finite_float(values.get("ci_low"))
        ci_high = coerce_optional_finite_float(values.get("ci_high"))
        if ci_low is not None and ci_high is not None:
            ax_robust.plot(
                [100.0 * ci_low, 100.0 * ci_high],
                [row, row],
                color=palette["baseline"],
                linewidth=1.0,
            )
        ax_robust.scatter(
            [point],
            [row],
            color=palette["red"] if weighting == "stabilized" else palette["baseline"],
            s=16,
            zorder=3,
        )
    ax_robust.axvline(0.0, color=palette["neutral"], linewidth=0.8)
    ax_robust.set_yticks(range(len(_WEIGHTINGS)))
    ax_robust.set_yticklabels([name for _, name in reversed(_WEIGHTINGS)], fontsize=5.8)
    untruncated = weights.loc[~weights["truncated"].astype(str).str.lower().eq("true")]
    ax_robust.text(
        0.02,
        -0.32,
        "Largest weight: "
        + "; ".join(
            f"{textwrap.shorten(labels[str(row['arm'])], width=22, placeholder='...')} "
            f"{float(row['maximum']):.2f}"
            for _, row in untruncated.iterrows()
        ),
        transform=ax_robust.transAxes,
        fontsize=5.4,
        va="top",
    )
    ax_robust.set_xlabel("Risk difference (percentage points)")
    ax_robust.set_title("Weighting sensitivity", loc="left")
    add_panel_label(ax_robust, "d", x=-0.3, fontsize=8.0)

    panel_sources = {
        panel.panel_id: list(panel.source_products)
        for panel in sealed.figure_panel_templates()
    }

    def metadata(panel_id: str) -> dict[str, Any]:
        return {"placement": "main", "source_products": panel_sources[panel_id]}

    contract = make_figure_contract(
        figure_id=_FIGURE_STEM,
        core_claim=(
            "The emulated target trial's protocol, the weighted risk under each "
            "strategy with the primary contrast, the balance the weights achieve "
            "and the contrast under each prespecified weighting are shown under "
            "the emulation's assumptions."
        ),
        panels=[
            {
                "panel_id": "a",
                "title": "Target trial protocol",
                "role": "causal_protocol",
                "chart_type": "timeline_diagram",
                "claim": "Eligibility, time zero, the grace period and the strategies are explicit.",
                "evidence_ids": [],
                "review_risk": "The protocol fixes the estimand; it does not establish exchangeability.",
                "metadata": metadata("a"),
            },
            {
                "panel_id": "b",
                "title": "Weighted risk by strategy",
                "role": "causal_contrast",
                "chart_type": "effect_curve",
                "claim": "The weighted cumulative risk under each strategy and their difference are shown with a bootstrap percentile interval.",
                "evidence_ids": [],
                "review_risk": "The contrast is causal only under no unmeasured confounding, positivity, correct weight models and censoring at ICU exit explained by baseline covariates.",
                "metadata": metadata("b"),
            },
            {
                "panel_id": "c",
                "title": "Balance of confounders",
                "role": "balance_positivity",
                "chart_type": "love_plot",
                "claim": "Weighting brings the clones each strategy follows towards the eligible population on the measured confounders.",
                "evidence_ids": [],
                "review_risk": "Balance on measured confounders says nothing about unmeasured ones.",
                "metadata": metadata("c"),
            },
            {
                "panel_id": "d",
                "title": "Weighting sensitivity",
                "role": "robustness",
                "chart_type": "trimming_panel",
                "claim": "The contrast is shown with the weights as estimated, truncated and absent.",
                "evidence_ids": [],
                "review_risk": "Agreement across weightings does not remove confounding the weights do not model.",
                "metadata": metadata("d"),
            },
        ],
        export_formats=("svg", "pdf", "png"),
        source_data=tuple(_source_filenames(sealed).values()),
        statistics_note=(
            "Risks are weighted Kaplan-Meier estimates of cloned stays under "
            "stabilized inverse probability of censoring weights; intervals are "
            "bootstrap percentile intervals over resampled units, refitting every "
            "model."
        ),
        image_integrity_note="All plotted values are rendered from digest-bound upstream result tables.",
        reader_caption=_reader_legend(sealed),
    )
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


def run_target_trial_figure(
    *,
    protocol: Any,
    eligibility: Any,
    risk_curves: Any,
    effects: Any,
    balance: Any,
    weights: Any,
    source_paths: Mapping[str, Path],
    authority: TargetTrialRuntimeAuthority | Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Render only from the digest-bound result tables the authority declares."""

    import pandas as pd

    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("target trial figure received the wrong authority kind")
    out_dir.mkdir(parents=True, exist_ok=True)
    filenames = _source_filenames(sealed)
    copied: list[str] = []
    for product in sealed.figure_input_products:
        source = Path(source_paths[product]).resolve()
        destination = out_dir / filenames[product]
        shutil.copyfile(source, destination)
        if (
            hashlib.sha256(source.read_bytes()).digest()
            != hashlib.sha256(destination.read_bytes()).digest()
        ):
            raise ValueError("target trial figure source changed while copying")
        copied.append(destination.name)
    outputs = _render(
        protocol=pd.DataFrame(protocol),
        eligibility=pd.DataFrame(eligibility),
        risk_curves=pd.DataFrame(risk_curves),
        effects=pd.DataFrame(effects),
        balance=pd.DataFrame(balance),
        weights=pd.DataFrame(weights),
        sealed=sealed,
        out_dir=out_dir,
    )
    receipt_path = out_dir / "target_trial_figure_runtime_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.target_trial_figure_runtime_receipt/1",
                "adjustment_columns": [item.column for item in sealed.covariates],
                "promoted_effect_measure": "risk_difference_under_strategies",
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
        raise ValueError("target trial figure export is missing")
    return {
        "status": "ok",
        "rendering_only": True,
        "deterministic_standard_analysis": TARGET_TRIAL_FIGURE_ANALYSIS_KIND,
        "source_data_files": copied,
        "figure_assets": {key: value.name for key, value in outputs.items()},
        "output_files": {sealed.figure_product: figure_file.name},
    }


def target_trial_figure_executor_code(
    step: AnalysisStep,
    *,
    authority: TargetTrialRuntimeAuthority | Mapping[str, Any],
) -> str:
    """Return the host-owned renderer for the sealed target trial tables."""

    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("target trial figure requires its sealed authority")
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.target_trial_figure import (
            run_target_trial_figure,
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
        summary = run_target_trial_figure(
            protocol=bindings[{sealed.protocol_product!r}].frame,
            eligibility=bindings[{sealed.eligibility_product!r}].frame,
            risk_curves=bindings[{sealed.risk_curve_product!r}].frame,
            effects=bindings[{sealed.effect_product!r}].frame,
            balance=bindings[{sealed.balance_product!r}].frame,
            weights=bindings[{sealed.weight_product!r}].frame,
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
    "TARGET_TRIAL_FIGURE_ANALYSIS_KIND",
    "run_target_trial_figure",
    "target_trial_figure_executor_code",
    "target_trial_figure_executor_owns_step",
]
