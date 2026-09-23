"""Step 3 -- deterministic figures for the analysis result.

Figures are drawn only from the tables held by :class:`AnalysisResult`; nothing
is recomputed here.  Each figure is written as PNG and SVG and carries a
``reader_caption`` with the definitions and denominators, so the picture keeps
axes, legend and the few numbers a reader needs while the long text lives
beside it (the workspace figure rule).
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.ticker  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .run_analysis import PRIMARY_VARIANT_ID, AnalysisResult  # noqa: E402

PLOTS_TOKEN = "✓ All plots generated successfully!"
FIGURE_DPI = 200
# Deterministic SVG element ids so two runs of the same data hash identically.
matplotlib.rcParams["svg.hashsalt"] = "easyicu-skill-package"
_ACCENT = "#1f4e79"
_MUTED = "#8a8a8a"
_UNKNOWN = "#b8860b"
#: Reader labels for the host eligibility predicates recorded in the cohort flow.
_FLOW_LABELS = {
    "source_cohort": "Source cohort",
    "nonnegative_event_time": "Non-negative event time",
    "alive_at_landmark": "Alive at landmark",
    "observed_at_landmark": "Observed at landmark",
    "exposure_known": "Exposure state at landmark",
}


def _reader_text(value: Any) -> str:
    """Spell an identifier for a reader: underscores to spaces, first letter up."""

    text = str(value or "").replace("__", ": ").replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def _flow_label(predicate: Any) -> str:
    return _FLOW_LABELS.get(str(predicate), _reader_text(predicate))


def _variant_label(analysis_id: Any, spec: Any) -> str:
    """Name a sensitivity refit the way the report names it, not by its id."""

    token = str(analysis_id or "")
    if token == PRIMARY_VARIANT_ID:
        return "Primary model"
    if token == "first_stay_only":
        return "First ICU stay only"
    if token == "adjusted_trend_per_level_increment":
        return "Per-level increment (ordinal trend)"
    head, separator, column = token.partition("__")
    if separator and head == "alternate_exposure":
        return f"Alternate definition: {spec.label(column)}"
    if separator and head == "functional_form":
        return f"Spline form: {spec.label(column)}"
    return _reader_text(token)


@dataclass(frozen=True)
class FigureArtifact:
    figure_id: str
    png_path: str
    svg_path: str
    source_tables: tuple[str, ...]
    reader_caption: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "png_path": self.png_path,
            "svg_path": self.svg_path,
            "source_tables": list(self.source_tables),
            "reader_caption": self.reader_caption,
        }


def _save(fig: "plt.Figure", out_dir: Path, stem: str) -> tuple[str, str]:
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight", metadata={"Software": "easyicu-skill"})
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None, "Creator": "easyicu-skill"})
    plt.close(fig)
    return png.name, svg.name


def _forest(
    ax: "plt.Axes",
    labels: list[str],
    estimates: list[float | None],
    lows: list[float | None],
    highs: list[float | None],
    *,
    highlight: list[bool],
    xlabel: str,
) -> None:
    y = np.arange(len(labels))[::-1]
    for position, estimate, low, high, strong in zip(y, estimates, lows, highs, highlight):
        if estimate is None or low is None or high is None:
            ax.plot([1.0], [position], marker="x", color=_MUTED, markersize=7)
            continue
        ax.plot([low, high], [position, position], color=_ACCENT if strong else _MUTED, linewidth=1.6)
        ax.plot(
            [estimate],
            [position],
            marker="D" if strong else "o",
            color=_ACCENT if strong else _MUTED,
            markersize=7 if strong else 5,
        )
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    finite = [value for value in [*lows, *highs] if value is not None and value > 0]
    if finite:
        lower = min(finite + [1.0]) / 1.4
        upper = max(finite + [1.0]) * 1.4
        ax.set_xlim(lower, upper)
        candidates = [0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0]
        ticks = [value for value in candidates if lower <= value <= upper]
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{value:g}" for value in ticks])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def plot_primary_forest(result: AnalysisResult, out_dir: Path) -> FigureArtifact:
    spec = result.spec
    table = result.adjusted_association_estimates
    labels, est, low, high, strong = [], [], [], [], []
    for level in spec.non_reference_levels:
        row = table.loc[table["exposure_level"].astype(str).eq(level)]
        if len(row) != 1:
            continue
        row = row.iloc[0]
        labels.append(f"{level} vs {spec.reference_level}")
        est.append(float(row["estimate"]))
        low.append(float(row["ci_low"]))
        high.append(float(row["ci_high"]))
        strong.append(bool(row["is_primary_contrast"]))
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.45 * max(len(labels), 2)))
    _forest(
        ax,
        labels,
        est,
        low,
        high,
        highlight=strong,
        xlabel=f"Adjusted odds ratio vs level {spec.reference_level} (95% CI, log scale)",
    )
    # The exposure is named once, in the title; rows carry only the contrast.
    ax.set_title(
        "\n".join(textwrap.wrap(f"Adjusted association by level of {spec.label(spec.exposure)}", 78)),
        loc="left",
        fontsize=11,
    )
    png, svg = _save(fig, out_dir, "figure_adjusted_association_forest")
    metrics = result.key_metrics
    caption = (
        f"Adjusted odds ratios of {spec.label(spec.outcome)} for each level of "
        f"{spec.label(spec.exposure)} versus level {spec.reference_level}, among the "
        f"{metrics['n_fit']} landmark rows with a known exposure and complete covariates "
        f"({metrics['n_events_fit']} events; covariates: {metrics['covariates']}; "
        f"{metrics['variance_estimator']} covariance across {metrics['cluster_count']} patients). "
        f"The diamond marks the prespecified primary contrast (level {spec.primary_contrast_level}). "
        f"Rows with an unknown exposure ({metrics['n_exposure_unknown']}) are not in this model. "
        f"Claim ceiling: {spec.claim_ceiling}."
    )
    return FigureArtifact(
        "adjusted_association_forest", png, svg, ("adjusted_association_estimates",), caption
    )


def plot_absolute_risk(result: AnalysisResult, out_dir: Path) -> FigureArtifact:
    spec = result.spec
    table = result.absolute_risk
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(table))
    for position, (_, row) in zip(x, table.iterrows()):
        risk = row["risk"]
        if risk is None or (isinstance(risk, float) and math.isnan(risk)):
            continue
        color = _UNKNOWN if bool(row["is_unknown"]) else _ACCENT
        ax.bar(position, float(risk), color=color, alpha=0.85, width=0.6)
        ax.errorbar(
            position,
            float(risk),
            yerr=[[float(risk) - float(row["risk_ci_low"])], [float(row["risk_ci_high"]) - float(risk)]],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['level']}\n(n={int(row['n'])})" for _, row in table.iterrows()]
    )
    ax.set_ylabel(f"{spec.label(spec.outcome)} (proportion)")
    ax.set_xlabel(spec.label(spec.exposure))
    ax.set_ylim(0, min(1.0, max(0.05, float(pd.to_numeric(table["risk_ci_high"], errors="coerce").max()) * 1.15)))
    ax.set_title("Absolute risk by exposure level (Wilson 95% CI)", loc="left", fontsize=11)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    png, svg = _save(fig, out_dir, "figure_absolute_risk_by_level")
    caption = (
        f"Observed proportion of {spec.label(spec.outcome)} by {spec.label(spec.exposure)} "
        f"among the {result.cohort.n_landmark} rows alive and observed at the "
        f"{spec.landmark_hours:g} h landmark, with Wilson score 95% intervals. The amber bar is "
        f"the '{spec.unknown_level_label}' state (exposure not evaluable; kept in the denominator, "
        f"never recoded to level {spec.reference_level}, excluded from adjusted models)."
    )
    return FigureArtifact("absolute_risk_by_level", png, svg, ("absolute_risk",), caption)


def plot_cohort_flow(result: AnalysisResult, out_dir: Path) -> FigureArtifact:
    flow = result.cohort_flow
    stages = flow.loc[flow["action"].ne("split_not_exclude")]
    labels = [_flow_label(row["predicate_kind"]) for _, row in stages.iterrows()]
    remaining = [int(row["n_remaining"]) for _, row in stages.iterrows()]
    excluded = [int(row["n_excluded"]) for _, row in stages.iterrows()]
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.55 * (len(labels) + 1)))
    y = np.arange(len(labels) + 1)[::-1]
    ax.barh(y[:-1], remaining, color=_ACCENT, alpha=0.85)
    for position, value, dropped in zip(y[:-1], remaining, excluded):
        ax.text(value, position, f"  n={value}" + (f"  (−{dropped})" if dropped else ""), va="center", fontsize=9)
    known = result.cohort.n_exposure_known
    unknown = result.cohort.n_exposure_unknown
    ax.barh(y[-1], known, color=_ACCENT, alpha=0.85)
    ax.barh(y[-1], unknown, left=known, color=_UNKNOWN, alpha=0.85)
    ax.text(known + unknown, y[-1], f"  known {known} / unknown {unknown}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([*labels, _flow_label("exposure_known")])
    ax.set_xlabel("ICU stays")
    ax.set_xlim(0, max(remaining) * 1.45)
    ax.set_title("Cohort flow to the landmark analysis population", loc="left", fontsize=11)
    png, svg = _save(fig, out_dir, "figure_cohort_flow")
    caption = (
        f"Sequential eligibility from the source cohort ({result.cohort.n_source} stays): "
        "non-negative event time, alive at the landmark, observed at the landmark; the final bar "
        "splits the landmark population by whether the exposure was evaluable. "
        "Numbers in parentheses are exclusions at each step."
    )
    return FigureArtifact("cohort_flow", png, svg, ("cohort_flow",), caption)


def plot_exposure_ascertainment(result: AnalysisResult, out_dir: Path) -> FigureArtifact:
    spec = result.spec
    audit = result.measurement_audit
    rows = audit.loc[audit["audit_item"].isin(["exposure_ascertainment", "covariate_completeness"])]
    labels = [spec.label(str(row["column"])) for _, row in rows.iterrows()]
    unknown_share = [float(row["share_unknown"] or 0.0) for _, row in rows.iterrows()]
    colors = [
        _UNKNOWN if row["audit_item"] == "exposure_ascertainment" else _MUTED
        for _, row in rows.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.5 * max(len(labels), 2)))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, unknown_share, color=colors, alpha=0.9)
    for position, share in zip(y, unknown_share):
        ax.text(share, position, f"  {share:.1%}", va="center", fontsize=9)
    ax.axvline(spec.unknown_exposure_warning_share, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(1.0 if max(unknown_share, default=0) > 0.8 else max(unknown_share, default=0.1) * 1.6, 0.15))
    ax.set_xlabel("Share of landmark rows with the value not evaluable / missing")
    ax.set_title("Ascertainment and completeness at the landmark", loc="left", fontsize=11)
    png, svg = _save(fig, out_dir, "figure_exposure_ascertainment")
    caption = (
        "Share of landmark rows whose exposure definition could not be evaluated (amber: "
        "primary and alternate exposure definitions) or whose covariate is missing (grey). "
        f"The dashed line is the reporting threshold {spec.unknown_exposure_warning_share:.0%} "
        "above which the unknown share is flagged in the report."
    )
    return FigureArtifact("exposure_ascertainment", png, svg, ("measurement_audit",), caption)


def plot_robustness(result: AnalysisResult, out_dir: Path) -> FigureArtifact:
    spec = result.spec
    summary = result.robustness_summary
    rows = summary.loc[summary["effect_scale"].eq("odds_ratio")]
    labels, est, low, high, strong = [], [], [], [], []
    for _, row in rows.iterrows():
        labels.append(_variant_label(row["analysis_id"], spec))
        est.append(None if pd.isna(row["estimate"]) else float(row["estimate"]))
        low.append(None if pd.isna(row["ci_low"]) else float(row["ci_low"]))
        high.append(None if pd.isna(row["ci_high"]) else float(row["ci_high"]))
        strong.append(str(row["analysis_id"]) == PRIMARY_VARIANT_ID)
    fig, ax = plt.subplots(figsize=(7.6, 0.9 + 0.45 * max(len(labels), 2)))
    _forest(
        ax,
        labels,
        est,
        low,
        high,
        highlight=strong,
        xlabel=f"OR, level {spec.primary_contrast_level} vs {spec.reference_level} (95% CI, log scale)",
    )
    ax.set_title("Primary contrast across prespecified sensitivity analyses", loc="left", fontsize=11)
    png, svg = _save(fig, out_dir, "figure_robustness_forest")
    caption = (
        f"Odds ratio of the prespecified primary contrast (level {spec.primary_contrast_level} vs "
        f"{spec.reference_level}) under the primary model (diamond) and each prespecified sensitivity "
        "refit: alternate exposure definitions, first-ICU-stay restriction and restricted-cubic-spline "
        "covariate forms. A cross marks a refit that could not be estimated; its reason is in "
        "robustness_summary.csv. Denominators differ by design and are listed in the same table."
    )
    return FigureArtifact("robustness_forest", png, svg, ("robustness_summary",), caption)


def plot_secondary_outcomes(result: AnalysisResult, out_dir: Path) -> list[FigureArtifact]:
    spec = result.spec
    table = result.secondary_outcome_summary
    artifacts: list[FigureArtifact] = []
    if table.empty:
        return artifacts
    for secondary in spec.secondary_outcomes:
        rows = table.loc[table["outcome"].eq(secondary.name) & table["is_unknown"].eq(False)]
        populations = list(dict.fromkeys(rows["population"].tolist()))
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        width = 0.8 / max(len(populations), 1)
        x = np.arange(len(spec.exposure_levels))
        for offset, population in enumerate(populations):
            subset = rows.loc[rows["population"].eq(population)].set_index("level").reindex(spec.exposure_levels)
            medians = pd.to_numeric(subset["median"], errors="coerce").to_numpy(dtype=float)
            q25 = pd.to_numeric(subset["q25"], errors="coerce").to_numpy(dtype=float)
            q75 = pd.to_numeric(subset["q75"], errors="coerce").to_numpy(dtype=float)
            positions = x - 0.4 + width * (offset + 0.5)
            ax.errorbar(
                positions,
                medians,
                yerr=[medians - q25, q75 - medians],
                fmt="o",
                capsize=3,
                label=population.replace("_", " "),
                color=_ACCENT if offset == 0 else _MUTED,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(list(spec.exposure_levels))
        ax.set_xlabel(spec.label(spec.exposure))
        ax.set_ylabel(f"{spec.label(secondary.name)}: median (IQR), {secondary.unit}")
        ax.legend(frameon=False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.set_title(f"{spec.label(secondary.name)} by exposure level", loc="left", fontsize=11)
        png, svg = _save(fig, out_dir, f"figure_secondary_{secondary.name}_by_level")
        caption = (
            f"Median and interquartile range of {spec.label(secondary.name)} by "
            f"{spec.label(spec.exposure)} in the landmark population, shown for all landmark rows and "
            "for survivors only because deaths shorten the stay. Denominators per level and population "
            "are in secondary_outcome_summary.csv; the ordered trend tests are in ordinal_trend_tests.csv."
        )
        artifacts.append(
            FigureArtifact(
                f"secondary_{secondary.name}_by_level", png, svg, ("secondary_outcome_summary",), caption
            )
        )
    return artifacts


def generate_all_plots(
    result: AnalysisResult,
    out_dir: str | Path,
    *,
    verbose: bool = True,
) -> dict[str, FigureArtifact]:
    """Render every standard figure as PNG + SVG and return their captions."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("\n=== Generating figures ===")
    artifacts = [
        plot_cohort_flow(result, target),
        plot_exposure_ascertainment(result, target),
        plot_absolute_risk(result, target),
        plot_primary_forest(result, target),
        plot_robustness(result, target),
        *plot_secondary_outcomes(result, target),
    ]
    figures = {artifact.figure_id: artifact for artifact in artifacts}
    if verbose:
        for artifact in artifacts:
            print(f"   Saved: {artifact.png_path} (+ .svg)")
        print(PLOTS_TOKEN)
    return figures


__all__ = ["PLOTS_TOKEN", "FigureArtifact", "generate_all_plots"]
