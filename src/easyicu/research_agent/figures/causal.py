"""Causal-emulation publication figure.

Panels (satisfying the ``causal_emulation`` figure strategy, hero role
``causal_protocol``):

* A -- target-trial protocol panel: eligibility, time zero, strategies,
  estimand, assumptions (hero ``causal_protocol``);
* B -- covariate-balance love plot, SMD before vs after weighting
  (role ``balance_positivity``);
* C -- adjusted causal contrast with uncertainty (role ``causal_contrast``).

Source evidence: the ``covariate_balance`` SMD table the plan contract already
requires, plus an adjusted-effect table. Returns ``None`` when a covariate-
balance table or an effect estimate cannot be found, so the skill falls
through to its existing behaviour without regressing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..evidence import EvidenceStore
from ..schema import AnalysisPlan, EvidenceRecord, ResearchContext
from .base import RenderedFigure, load_table, numeric_series, resolve_column

_BALANCE_NAMES = [
    "covariate_balance",
    "balance_table",
    "smd_table",
    "love_plot_data",
    "balance",
]
_EFFECT_NAMES = [
    "causal_effect",
    "effect_estimate",
    "treatment_effect",
    "adjusted_effect",
    "ate_estimate",
    "causal_contrast",
    "primary_effect",
]


def _load_effect(
    evidence: EvidenceStore, run_dir: Path
) -> Optional[Tuple[Optional[EvidenceRecord], float, float, float, str]]:
    record, frame = load_table(evidence, run_dir, _EFFECT_NAMES)
    if frame is None:
        return None
    est_col = resolve_column(
        frame,
        [
            "estimate",
            "effect",
            "ate",
            "rd",
            "or",
            "hr",
            "rr",
            "value",
            "adjusted_effect",
        ],
    )
    if est_col is None:
        return None
    lo_col = resolve_column(
        frame, ["lower", "ci_lower", "lower_95", "conf_lower", "lcl"]
    )
    hi_col = resolve_column(
        frame, ["upper", "ci_upper", "upper_95", "conf_upper", "ucl"]
    )
    scale_col = resolve_column(frame, ["scale", "measure", "estimand"])
    try:
        est = float(numeric_series(frame, est_col).dropna().iloc[0])
    except (IndexError, ValueError):
        return None
    lo = (
        float(numeric_series(frame, lo_col).dropna().iloc[0])
        if lo_col and numeric_series(frame, lo_col).notna().any()
        else est
    )
    hi = (
        float(numeric_series(frame, hi_col).dropna().iloc[0])
        if hi_col and numeric_series(frame, hi_col).notna().any()
        else est
    )
    scale = (
        str(frame.iloc[0][scale_col])
        if scale_col
        else (est_col.upper() if est_col.lower() in ("or", "hr", "rr") else "effect")
    )
    return record, est, lo, hi, scale


def _protocol_lines(context: ResearchContext) -> List[Tuple[str, str]]:
    exposure = context.primary_exposure or "the treatment strategy"
    outcome = context.target_outcome or "the outcome"
    return [
        ("Eligibility", "Adult ICU stays meeting the stated inclusion criteria"),
        ("Time zero", "ICU admission / first eligible exposure window"),
        ("Strategies", f"{exposure} vs. comparator"),
        ("Outcome", str(outcome)),
        ("Estimand", "Adjusted effect under the stated identification assumptions"),
        ("Assumptions", "No unmeasured confounding; positivity; correct model"),
    ]


def render_causal_figure(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    evidence: EvidenceStore,
    run_dir: Path,
) -> Optional[RenderedFigure]:
    bal_record, balance = load_table(evidence, run_dir, _BALANCE_NAMES, min_rows=1)
    effect = _load_effect(evidence, run_dir)
    if balance is None or effect is None:
        return None
    effect_record, est, lo, hi, scale = effect

    cov_col = resolve_column(
        balance, ["covariate", "variable", "term", "feature", "label"]
    )
    before_col = resolve_column(
        balance,
        ["smd_unweighted", "smd_before", "smd_unadjusted", "unweighted_smd", "before"],
    )
    after_col = resolve_column(
        balance, ["smd_weighted", "smd_after", "smd_adjusted", "weighted_smd", "after"]
    )
    generic_smd = resolve_column(
        balance, ["smd", "standardized_mean_difference", "std_mean_diff"]
    )
    if cov_col is None or (before_col is None and generic_smd is None):
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from ..publication_figures import add_panel_label, apply_publication_style

    palette = apply_publication_style()
    blue = palette.get("blue", "#0F4D92")
    red = palette.get("red", "#B22222")
    neutral = palette.get("neutral", "#8F8F8F")

    fig = plt.figure(figsize=(183 / 25.4, 86 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.92, 1.12, 0.78],
        left=0.03,
        right=0.975,
        top=0.9,
        bottom=0.2,
        wspace=0.5,
    )
    ax_proto = fig.add_subplot(grid[0, 0])
    ax_love = fig.add_subplot(grid[0, 1])
    ax_eff = fig.add_subplot(grid[0, 2])

    # A -- protocol panel (hero)
    ax_proto.axis("off")
    ax_proto.set_title("Target-trial protocol", loc="left", pad=4)
    y = 0.93
    for key, val in _protocol_lines(context):
        ax_proto.text(
            0.0,
            y,
            f"{key}:",
            fontsize=6.4,
            fontweight="bold",
            color=palette.get("baseline", "#272727"),
            transform=ax_proto.transAxes,
        )
        ax_proto.text(
            0.0,
            y - 0.045,
            val,
            fontsize=5.8,
            color=palette.get("baseline", "#272727"),
            transform=ax_proto.transAxes,
            wrap=True,
        )
        y -= 0.155
    ax_proto.add_patch(
        plt.Rectangle(
            (-0.02, -0.02),
            1.04,
            1.0,
            transform=ax_proto.transAxes,
            fill=False,
            edgecolor=neutral,
            linewidth=0.7,
        )
    )
    add_panel_label(ax_proto, "A", x=-0.02, y=1.02)

    # B -- love plot
    bal = balance.copy()
    covariates = bal[cov_col].astype(str).tolist()
    yy = np.arange(len(covariates))
    if before_col is not None:
        before = numeric_series(bal, before_col).abs().to_numpy()
        ax_love.scatter(
            before, yy, s=16, color=neutral, label="Before weighting", zorder=3
        )
    if after_col is not None:
        after = numeric_series(bal, after_col).abs().to_numpy()
        ax_love.scatter(after, yy, s=16, color=blue, label="After weighting", zorder=3)
    elif generic_smd is not None and before_col is None:
        smd = numeric_series(bal, generic_smd).abs().to_numpy()
        ax_love.scatter(smd, yy, s=16, color=blue, label="SMD", zorder=3)
    ax_love.axvline(0.1, color=red, linestyle="--", linewidth=0.8)
    ax_love.set_yticks(yy, [c[:18] for c in covariates], fontsize=5.6)
    ax_love.invert_yaxis()
    ax_love.set_xlabel("|Standardised mean difference|")
    ax_love.set_title("Covariate balance", loc="left", pad=4)
    ax_love.legend(loc="lower right", fontsize=5.4)
    ax_love.grid(
        axis="x",
        color=palette.get("neutral_light", "#D8D8D8"),
        linewidth=0.5,
        alpha=0.8,
        zorder=0,
    )
    add_panel_label(ax_love, "B", x=-0.28)

    # C -- effect contrast
    ratio_scale = scale.lower() in ("or", "hr", "rr")
    null_value = 1.0 if ratio_scale else 0.0
    ax_eff.errorbar(
        est,
        0,
        xerr=np.array([[max(0.0, est - lo)], [max(0.0, hi - est)]], dtype=float),
        fmt="o",
        color=blue,
        ecolor=blue,
        elinewidth=1.2,
        capsize=3.0,
        markersize=5.0,
    )
    ax_eff.axvline(null_value, color=neutral, linestyle="--", linewidth=0.8)
    ax_eff.set_yticks(
        [0], [scale.upper() if ratio_scale else "Adjusted effect"], fontsize=6.4
    )
    ax_eff.set_ylim(-0.6, 0.6)
    if ratio_scale and lo > 0:
        ax_eff.set_xscale("log")
    ax_eff.set_xlabel(f"{scale.upper() if ratio_scale else 'Effect'} (95% CI)")
    ax_eff.set_title("Adjusted contrast", loc="left", pad=4)
    ax_eff.text(
        est,
        0.28,
        f"{est:.2f} ({lo:.2f}-{hi:.2f})",
        ha="center",
        fontsize=6.0,
        color=palette.get("baseline", "#272727"),
    )
    add_panel_label(ax_eff, "C", x=-0.24)

    core_claim = (
        "The causal question is shown as a target-trial protocol, a covariate-"
        "balance love plot, and the adjusted contrast with uncertainty, rendered "
        "from registered balance and effect evidence and bounded by the stated "
        "assumptions."
    )

    def _ids(rec: Optional[EvidenceRecord]) -> List[str]:
        return [rec.evidence_id] if rec is not None else []

    panels = [
        {
            "panel_id": "A",
            "title": "Target-trial protocol",
            "role": "causal_protocol",
            "chart_type": "protocol_table",
            "claim": "Eligibility, time zero, strategies, and estimand are stated before any effect is shown.",
            "evidence_ids": _ids(effect_record) or _ids(bal_record),
            "review_risk": "The estimate is only causal insofar as the stated assumptions hold.",
        },
        {
            "panel_id": "B",
            "title": "Covariate balance",
            "role": "balance_positivity",
            "chart_type": "love_plot",
            "claim": "Standardised mean differences before and after weighting show whether confounders are balanced.",
            "evidence_ids": _ids(bal_record),
            "review_risk": "Residual SMD above 0.1 signals incomplete balance and possible residual confounding.",
        },
        {
            "panel_id": "C",
            "title": "Adjusted contrast",
            "role": "causal_contrast",
            "chart_type": "causal_contrast_panel",
            "claim": "The adjusted effect and its interval quantify the causal contrast on its stated scale.",
            "evidence_ids": _ids(effect_record),
            "review_risk": "Report the effect conditional on balance and positivity; do not over-claim causality.",
        },
    ]

    source_frames: Dict[str, pd.DataFrame] = {"covariate_balance": bal}
    source_ids = _ids(bal_record) + _ids(effect_record)

    return RenderedFigure(
        fig=fig,
        figure_id="easyicu_causal_publication_figure",
        core_claim=core_claim,
        generation_mode="causal_publication_figure",
        panels=panels,
        source_evidence_ids=source_ids,
        source_frames=source_frames,
    )
