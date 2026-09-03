#!/usr/bin/env python3
"""Render a non-destructive presentation layer for the completed E2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, NullLocator


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 7.2
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.facecolor"] = "white"

NAVY = "#173B6C"
BLUE = "#356FB6"
BLUE_SOFT = "#DDE8F5"
TEAL = "#248A8D"
TEAL_SOFT = "#D9EFEE"
WARM = "#C96B3B"
INK = "#20252D"
MID = "#6B7280"
LIGHT = "#E7E9ED"
PALE = "#F5F7FA"

SOURCE_PATHS = (
    "figure_gallery.json",
    "steps/estimate_primary_adjusted_association/outputs/landmark_rcs_curve.csv",
    "steps/estimate_primary_adjusted_association/outputs/landmark_adjusted_absolute_risk.csv",
    "steps/11_cohort_accounting_figure/outputs/cohort_flow_source_data.csv",
    "steps/12_data_quality_figure/outputs/data_quality_measurement_process_source_data.csv",
    "steps/replay_robustness_specifications_figure/outputs/robustness_plot_source_data.csv",
    "steps/replay_robustness_specifications_figure/outputs/robustness_plot_bound_statistics_source_data.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(run_dir: Path) -> dict[str, Any]:
    paths = {relative: run_dir / relative for relative in SOURCE_PATHS}
    missing = [relative for relative, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing registered E2 source data: {missing}")
    curve = pd.read_csv(paths[SOURCE_PATHS[1]])
    risk = pd.read_csv(paths[SOURCE_PATHS[2]])
    cohort = pd.read_csv(paths[SOURCE_PATHS[3]])
    measurement = pd.read_csv(paths[SOURCE_PATHS[4]])
    robustness = pd.read_csv(paths[SOURCE_PATHS[5]])
    bound_stats = pd.read_csv(paths[SOURCE_PATHS[6]])
    if list(curve.columns) != [
        "exposure_value",
        "reference_exposure_value",
        "adjusted_odds_ratio",
        "ci_low",
        "ci_high",
    ]:
        raise ValueError("Unexpected RCS curve schema")
    if list(risk.columns)[:5] != [
        "exposure_value",
        "reference_exposure_value",
        "adjusted_absolute_risk",
        "ci_low",
        "ci_high",
    ]:
        raise ValueError("Unexpected adjusted-risk schema")
    if len(curve) < 20 or len(curve) != len(risk):
        raise ValueError("The two registered exposure grids must align")
    if not np.allclose(curve["exposure_value"], risk["exposure_value"]):
        raise ValueError("The OR and risk exposure grids do not align")
    source_n = int(cohort.iloc[0]["n_remaining"])
    eligible_n = int(cohort.iloc[-1]["n_remaining"])
    complete_case_n = int(
        bound_stats.loc[bound_stats["statistic"] == "complete_case_n", "value"].iloc[0]
    )
    event_n = int(robustness["event_n"].dropna().iloc[0])
    if not (source_n >= eligible_n >= complete_case_n > event_n > 0):
        raise ValueError("Registered E2 denominator ordering is inconsistent")
    return {
        "paths": paths,
        "curve": curve,
        "risk": risk,
        "cohort": cohort,
        "measurement": measurement,
        "robustness": robustness,
        "source_n": source_n,
        "eligible_n": eligible_n,
        "complete_case_n": complete_case_n,
        "event_n": event_n,
    }


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def quiet_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=LIGHT, linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=6.4, width=0.7, length=3)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)


def save_bundle(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, str]:
    exports: dict[str, str] = {}
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, **kwargs)
        exports[suffix] = f"presentation_figures/{path.name}"
    plt.close(fig)
    return exports


def render_primary(data: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    curve = data["curve"]
    risk = data["risk"]
    x = curve["exposure_value"].to_numpy(float)
    reference = float(curve["reference_exposure_value"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 91 / 25.4))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.88, wspace=0.31)

    ax = axes[0]
    y = curve["adjusted_odds_ratio"].to_numpy(float)
    low = curve["ci_low"].to_numpy(float)
    high = curve["ci_high"].to_numpy(float)
    ax.fill_between(x, low, high, color=BLUE_SOFT, linewidth=0, zorder=1)
    ax.plot(x, y, color=NAVY, linewidth=1.8, zorder=3)
    ax.axhline(1.0, color=MID, linestyle=(0, (3, 3)), linewidth=0.8, zorder=2)
    ax.axvline(reference, color=MID, linestyle=(0, (3, 3)), linewidth=0.8, zorder=2)
    ax.scatter(
        [reference], [1.0], s=18, color=NAVY, edgecolor="white", linewidth=0.6, zorder=4
    )
    ax.set_yscale("log")
    ax.set_ylim(min(0.7, float(low.min()) * 0.94), max(2.1, float(high.max()) * 1.04))
    ax.set_yticks([0.75, 1.0, 1.25, 1.5, 2.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.95, 5.05)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Registered exposure value", fontsize=7)
    ax.set_ylabel("Outcome odds ratio (95% CI)", fontsize=7)
    ax.set_title("Exposure–response association", loc="left", fontsize=8.2, pad=8)
    quiet_axes(ax)
    panel_label(ax, "a")
    end = curve.iloc[-1]
    ax.annotate(
        f"5.0 vs {reference:.1f}: OR {end.adjusted_odds_ratio:.2f}\n"
        f"95% CI {end.ci_low:.2f}–{end.ci_high:.2f}",
        xy=(5.0, float(end.adjusted_odds_ratio)),
        xytext=(4.92, 1.34),
        ha="right",
        va="top",
        fontsize=6.2,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": WARM, "lw": 0.8},
    )
    ax.text(
        reference + 0.05,
        float(ax.get_ylim()[0]) * 1.02,
        f"Reference {reference:.1f}",
        fontsize=5.9,
        color=MID,
        rotation=90,
        va="bottom",
    )

    ax = axes[1]
    y = 100 * risk["adjusted_absolute_risk"].to_numpy(float)
    low = 100 * risk["ci_low"].to_numpy(float)
    high = 100 * risk["ci_high"].to_numpy(float)
    ax.fill_between(x, low, high, color=TEAL_SOFT, linewidth=0, zorder=1)
    ax.plot(x, y, color=TEAL, linewidth=1.8, zorder=3)
    ax.axvline(reference, color=MID, linestyle=(0, (3, 3)), linewidth=0.8, zorder=2)
    ax.set_xlim(0.95, 5.05)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(5, 20.2)
    ax.set_yticks([6, 9, 12, 15, 18])
    ax.set_xlabel("Registered exposure value", fontsize=7)
    ax.set_ylabel("Model-standardized outcome risk (%)", fontsize=7)
    ax.set_title("Model-standardized absolute risk", loc="left", fontsize=8.2, pad=8)
    quiet_axes(ax)
    panel_label(ax, "b")
    ref_idx = int(np.abs(x - reference).argmin())
    ax.scatter(
        [x[ref_idx], x[-1]],
        [y[ref_idx], y[-1]],
        s=18,
        color=TEAL,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    ax.text(
        x[ref_idx] + 0.08,
        y[ref_idx] - 0.9,
        f"{y[ref_idx]:.1f}%",
        fontsize=6.2,
        color=INK,
    )
    ax.text(
        x[-1] - 0.08, y[-1] + 0.55, f"{y[-1]:.1f}%", fontsize=6.2, color=INK, ha="right"
    )
    fig.text(
        0.5,
        0.035,
        f"Complete-case model: n={data['complete_case_n']:,}; events={data['event_n']:,}. "
        "Adjustment set and estimand remain defined by the bound run. "
        "Observational association; not causal.",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color=MID,
    )
    exports = save_bundle(fig, out_dir, "e2_primary_result")
    return {
        "label": "Registered primary association and adjusted risk",
        "figure_id": "e2_primary_result_presentation",
        "tier": "presentation_main",
        "relative_path": exports["png"],
        "contract_path": "presentation_figures/e2_primary_result.figure_contract.json",
        "panel_count": 2,
        "panel_roles": ["primary_estimand", "clinical_interpretation"],
        "chart_types": ["exposure_response", "absolute_risk_curve"],
        "exports": exports,
        "status": "presentation_only",
        "review_recommendation": "review_first_original_run_preserved",
    }


def render_context(data: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    fig = plt.figure(figsize=(183 / 25.4, 103 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.78, 1.25],
        left=0.08,
        right=0.985,
        bottom=0.16,
        top=0.91,
        hspace=0.62,
        wspace=0.44,
    )
    ax_flow = fig.add_subplot(grid[0, :])
    ax_measure = fig.add_subplot(grid[1, 0])
    ax_robust = fig.add_subplot(grid[1, 1])

    ax_flow.set_axis_off()
    ax_flow.set_title("Cohort accounting", loc="left", fontsize=8.2, pad=8)
    panel_label(ax_flow, "a")
    stages = [
        ("Source ICU stays", data["source_n"], NAVY),
        ("Registered eligible", data["eligible_n"], BLUE),
        ("Primary complete cases", data["complete_case_n"], TEAL),
    ]
    x_positions = [0.03, 0.365, 0.70]
    for idx, ((label, count, color), xpos) in enumerate(zip(stages, x_positions)):
        width = 0.255
        rect = FancyBboxPatch(
            (xpos, 0.30),
            width,
            0.43,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            transform=ax_flow.transAxes,
            facecolor=color,
            edgecolor="none",
        )
        ax_flow.add_patch(rect)
        ax_flow.text(
            xpos + width / 2,
            0.58,
            f"{count:,}",
            transform=ax_flow.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        ax_flow.text(
            xpos + width / 2,
            0.40,
            label,
            transform=ax_flow.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=6.7,
        )
        if idx < 2:
            arrow = FancyArrowPatch(
                (xpos + width + 0.015, 0.515),
                (x_positions[idx + 1] - 0.015, 0.515),
                transform=ax_flow.transAxes,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color=MID,
            )
            ax_flow.add_patch(arrow)
    excluded_1 = data["source_n"] - data["eligible_n"]
    excluded_2 = data["eligible_n"] - data["complete_case_n"]
    if excluded_1:
        ax_flow.text(
            0.325,
            0.16,
            f"−{excluded_1:,}\nwithout registered eligibility",
            transform=ax_flow.transAxes,
            ha="center",
            va="top",
            fontsize=5.8,
            color=MID,
        )
    ax_flow.text(
        0.66,
        0.16,
        f"−{excluded_2:,}\nincomplete model covariates",
        transform=ax_flow.transAxes,
        ha="center",
        va="top",
        fontsize=5.8,
        color=MID,
    )

    measurement = data["measurement"].set_index("concept")
    # Show the registered exposure alongside the first two fully observed
    # baseline covariates. Conditional event-time rows are deliberately not
    # mixed into this measurement-opportunity panel.
    preferred_concepts = ["lact", "age", "sex"]
    concepts = [
        concept for concept in preferred_concepts if concept in measurement.index
    ]
    if not concepts:
        raise ValueError("No registered E2 measurement concepts are available")
    labels = [concept.replace("_", " ").title() for concept in concepts]
    measured = np.array(
        [
            100
            * measurement.loc[c, "measured_one_n"]
            / measurement.loc[c, "eligible_n"]
            for c in concepts
        ]
    )
    repeated = np.array(
        [
            100
            * measurement.loc[c, "repeat_measured_n"]
            / measurement.loc[c, "eligible_n"]
            for c in concepts
        ]
    )
    y = np.arange(len(labels))[::-1]
    ax_measure.barh(
        y,
        np.repeat(100.0, len(y)),
        color=PALE,
        height=0.52,
        edgecolor=LIGHT,
        linewidth=0.6,
    )
    ax_measure.barh(y, measured, color=BLUE_SOFT, height=0.52, label="Measured ≥1")
    ax_measure.scatter(repeated, y, color=NAVY, s=28, zorder=3, label="Repeated ≥2")
    for yi, measured_value, repeated_value in zip(y, measured, repeated):
        ax_measure.text(
            99.2,
            yi + 0.12,
            f"{measured_value:.1f}%",
            ha="right",
            va="center",
            fontsize=6,
            color=INK,
        )
        ax_measure.text(
            repeated_value,
            yi - 0.22,
            f"{repeated_value:.1f}%",
            ha="center",
            va="center",
            fontsize=5.8,
            color=NAVY,
        )
    ax_measure.set_xlim(0, 102)
    ax_measure.set_yticks(y, labels)
    ax_measure.set_xlabel("Share of eligible ICU stays (%)", fontsize=7)
    ax_measure.set_title("Measurement opportunity", loc="left", fontsize=8.2, pad=25)
    ax_measure.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncol=2,
        fontsize=5.8,
        handlelength=1.5,
        columnspacing=1.0,
    )
    quiet_axes(ax_measure)
    panel_label(ax_measure, "b")

    robust = data["robustness"].copy()
    order = [
        "signed_upper_boundary_contrast",
        "signed_linear_functional_form_sensitivity",
        "complete_case_primary_model",
    ]
    robust = robust.set_index("spec_id").loc[order]
    labels = [
        "Registered upper-grid\ncontrast",
        "Registered linear\nsensitivity",
        "Complete-case record*",
    ]
    colors = [NAVY, TEAL, MID]
    y = np.arange(3)[::-1]
    for idx, (yi, (_, row), color) in enumerate(zip(y, robust.iterrows(), colors)):
        ax_robust.plot(
            [row.ci_low, row.ci_high], [yi, yi], color=color, lw=1.4, zorder=2
        )
        face = "white" if idx == 2 else color
        ax_robust.scatter(
            row.point_estimate,
            yi,
            s=28,
            facecolor=face,
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
        )
        ax_robust.text(
            2.32,
            yi,
            f"{row.point_estimate:.2f} ({row.ci_low:.2f}–{row.ci_high:.2f})",
            ha="right",
            va="center",
            fontsize=5.8,
            color=INK,
        )
    ax_robust.axvline(1.0, color=MID, linestyle=(0, (3, 3)), linewidth=0.8)
    ax_robust.set_xlim(0.95, 2.35)
    ax_robust.set_xticks(np.arange(1.0, 2.3, 0.2))
    ax_robust.set_ylim(-0.65, 2.65)
    ax_robust.set_yticks(y, labels)
    ax_robust.set_xlabel("Odds ratio (95% CI)", fontsize=7)
    ax_robust.set_title("Pre-specified checks", loc="left", fontsize=8.2, pad=8)
    quiet_axes(ax_robust)
    panel_label(ax_robust, "c")
    ax_robust.text(
        0.0,
        -0.24,
        "Different estimands are shown for audit, not direct equivalence.\n* Reuses the primary fit; not an independent refit.",
        transform=ax_robust.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=MID,
    )

    exports = save_bundle(fig, out_dir, "e2_cohort_and_audit")
    return {
        "label": "Registered cohort, measurement and specification audit",
        "figure_id": "e2_cohort_audit_presentation",
        "tier": "presentation_supporting",
        "relative_path": exports["png"],
        "contract_path": "presentation_figures/e2_cohort_and_audit.figure_contract.json",
        "panel_count": 3,
        "panel_roles": ["cohort_accounting", "data_quality", "robustness"],
        "chart_types": ["cohort_flow", "measurement_coverage", "sensitivity_forest"],
        "exports": exports,
        "status": "presentation_only",
        "review_recommendation": "supporting_context_original_run_preserved",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def resolve_output_run_dir(run_dir: Path, requested: Path | None) -> Path:
    """Keep presentation outputs and their verified source bindings in one run."""

    output_run_dir = requested.resolve() if requested is not None else run_dir
    if output_run_dir != run_dir:
        raise ValueError(
            "--output-run-dir must resolve to --run-dir because the Web overlay "
            "verifies every source binding inside the displayed run directory"
        )
    return output_run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        help="Optional Web-facing run directory that contains its own figure_gallery.json.",
    )
    parser.add_argument("--storyboard", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_run_dir = resolve_output_run_dir(run_dir, args.output_run_dir)
    canonical_gallery = output_run_dir / "figure_gallery.json"
    if not canonical_gallery.is_file():
        raise FileNotFoundError(
            f"Missing Web-facing canonical figure gallery: {canonical_gallery}"
        )
    out_dir = output_run_dir / "presentation_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(run_dir)
    main_figure = render_primary(data, out_dir)
    audit_figure = render_context(data, out_dir)
    for figure in (main_figure, audit_figure):
        figure["sha256"] = sha256_file(output_run_dir / figure["relative_path"])
    source_bindings = [
        {
            "relative_path": str(path.relative_to(output_run_dir)),
            "sha256": sha256_file(path),
        }
        for relative, path in data["paths"].items()
        if relative != "figure_gallery.json"
    ]
    common = {
        "backend": "python_matplotlib",
        "authority_ceiling": "analysis_only",
        "source_bindings": source_bindings,
        "statistics_note": "All curves, intervals and counts are reproduced from registered source-data tables without model refitting or denominator changes.",
        "image_integrity_note": "No raster source image was altered. These are new code-rendered presentation views; the original run figures remain unchanged.",
    }
    write_json(
        out_dir / "e2_primary_result.figure_contract.json",
        {
            "schema_version": "easyicu.presentation-figure-contract/1",
            "figure_id": main_figure["figure_id"],
            "core_claim": "The presentation reproduces the registered exposure-response and model-standardized risk curves without refitting; interpretation remains bound to the source run.",
            "archetype": "asymmetric_mixed_modality",
            "width_mm": 183,
            "height_mm": 91,
            "panels": [
                {
                    "panel_id": "a",
                    "role": "primary_estimand",
                    "claim": "Registered restricted-cubic-spline odds ratios and 95% confidence intervals across the source display grid.",
                },
                {
                    "panel_id": "b",
                    "role": "clinical_interpretation",
                    "claim": "Registered model-standardized absolute outcome risk and 95% confidence intervals on the same exposure grid.",
                },
            ],
            **common,
        },
    )
    write_json(
        out_dir / "e2_cohort_and_audit.figure_contract.json",
        {
            "schema_version": "easyicu.presentation-figure-contract/1",
            "figure_id": audit_figure["figure_id"],
            "core_claim": "The registered result is conditional on a narrowed cohort, near-complete covariate measurement and explicitly distinct robustness estimands.",
            "archetype": "quantitative_grid",
            "width_mm": 183,
            "height_mm": 103,
            "panels": [
                {
                    "panel_id": "a",
                    "role": "cohort_accounting",
                    "claim": "Source, registered-eligible and complete-case denominators are shown without changing the sequential ledger.",
                },
                {
                    "panel_id": "b",
                    "role": "data_quality",
                    "claim": "Measured-at-least-once and repeated-measurement shares use the registered eligible denominators.",
                },
                {
                    "panel_id": "c",
                    "role": "robustness",
                    "claim": "Every registered specification row is shown, with the non-independent complete-case documentation explicitly marked.",
                },
            ],
            **common,
        },
    )
    storyboard_binding = None
    if args.storyboard is not None and args.storyboard.is_file():
        storyboard_target = out_dir / "composition_storyboard.png"
        storyboard_target.write_bytes(args.storyboard.read_bytes())
        storyboard_binding = {
            "relative_path": "presentation_figures/composition_storyboard.png",
            "sha256": sha256_file(storyboard_target),
            "role": "layout_reference_only_no_scientific_values",
        }
    gallery = {
        "schema_version": "easyicu.presentation-figure-gallery/1",
        "status": "presentation_only",
        "authority_ceiling": "analysis_only",
        "derived_from": {
            "artifact": "figure_gallery.json",
            "sha256": sha256_file(canonical_gallery),
        },
        "source_bindings": source_bindings,
        "storyboard_binding": storyboard_binding,
        "primary_count": 1,
        "supporting_count": 1,
        "figures": [main_figure, audit_figure],
        "integrity": {
            "original_run_figures_overwritten": False,
            "models_refit": False,
            "denominators_changed": False,
            "presentation_only": True,
        },
    }
    write_json(out_dir / "presentation_figure_gallery.json", gallery)
    svg_text_counts = {}
    for stem in ("e2_primary_result", "e2_cohort_and_audit"):
        svg = (out_dir / f"{stem}.svg").read_text(encoding="utf-8")
        svg_text_counts[stem] = svg.count("<text")
        if svg_text_counts[stem] < 10:
            raise ValueError(f"Editable SVG text audit failed for {stem}")
    write_json(
        out_dir / "qa_report.json",
        {
            "schema_version": "easyicu.presentation-figure-qa/1",
            "backend": "python_matplotlib",
            "source_binding_count": len(source_bindings),
            "svg_text_node_counts": svg_text_counts,
            "original_run_figures_overwritten": False,
            "authority_ceiling": "analysis_only",
            "checks": {
                "source_digests_recorded": True,
                "editable_svg_text": True,
                "svg_exported": True,
                "pdf_exported": True,
                "tiff_600_dpi_exported": True,
                "png_300_dpi_exported": True,
            },
        },
    )
    print(out_dir / "presentation_figure_gallery.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
