"""Repackage frozen Dev9 E3/M2 outputs into an article-level figure suite.

This benchmark-only renderer performs no fitting, imputation, row selection, or
scientific recomputation.  It validates and plots every row of named outputs
from one already completed analysis-only run.  It must never be used to promote
Dev9 evidence to paper-authorized status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from easyicu.research_agent.execution.runners.prediction_figure_executor import (
    run_prediction_figure,
)
from easyicu.research_agent.figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)


E3_RUN_RELATIVE = Path("e3/e3_kdigo_gradient/aware/run_20260825T024928_3b8fef")
M2_RUN_RELATIVE = Path("m2/m2_mortality_prediction/aware/run_20260825T025332_5c7922")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains non-finite values")


def _copy_source(frame: pd.DataFrame, path: Path, output: Path) -> str:
    copied = frame.copy()
    copied.insert(0, "source_sha256", _sha256(path))
    copied.insert(0, "source_file", path.name)
    copied.insert(0, "source_row_index", range(len(copied)))
    copied.to_csv(output, index=False)
    return output.name


def _save(
    fig: Any,
    *,
    out_dir: Path,
    product: str,
    height_mm: float,
    panels: list[dict[str, Any]],
    source_data: list[str],
    core_claim: str,
    statistics_note: str,
) -> list[str]:
    contract = make_figure_contract(
        figure_id=f"figure:{product}",
        core_claim=core_claim,
        archetype="asymmetric_mixed_modality",
        width_mm=183.0,
        height_mm=height_mm,
        panels=panels,
        source_data=source_data,
        statistics_note=statistics_note,
    )
    outputs = save_publication_figure(
        fig,
        out_dir / product,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
        pad_inches=0.14,
    )
    plt.close(fig)
    return [path.name for path in outputs.values()]


def _forest(
    ax: Any, frame: pd.DataFrame, labels: list[str], title: str, color: str
) -> None:
    estimates = frame["estimate"].to_numpy(dtype=float)
    lows = frame["ci_low"].to_numpy(dtype=float)
    highs = frame["ci_high"].to_numpy(dtype=float)
    positions = np.arange(len(frame))
    ax.errorbar(
        estimates,
        positions,
        xerr=np.vstack((estimates - lows, highs - estimates)),
        fmt="o",
        color=color,
        capsize=2.5,
    )
    ax.axvline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.set_title(title, loc="left", pad=10)


def _render_e3(source_run: Path, out_dir: Path) -> dict[str, Any]:
    paths = {
        "ordered": source_run
        / "steps/ordinal_trend_01/outputs/ordered_stratified_outcomes.csv",
        "adjusted": source_run
        / "steps/adjusted_association_01/outputs/adjusted_association_estimates.csv",
        "sensitivity": source_run
        / "steps/host_association_model_grid_d3cfb9c38429/outputs/e3_scientific_sensitivity.csv",
        "missingness": source_run
        / "steps/measurement_audit_01/outputs/missingness_measurement_audit.csv",
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("E3 source run is incomplete")
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    ordered = frames["ordered"].sort_values("level_order").reset_index(drop=True)
    if not ordered["row_role"].astype(str).eq("exposure_level").all():
        raise ValueError("E3 ordered outcomes contain non-level rows")
    if ordered["level_order"].tolist() != list(range(len(ordered))):
        raise ValueError("E3 ordered levels are not consecutive")
    _finite(
        ordered,
        (
            "level_n",
            "binary_n",
            "binary_event_n",
            "binary_percentage",
            "binary_ci_low",
            "binary_ci_high",
            "continuous_n",
            "continuous_median",
            "continuous_q25",
            "continuous_q75",
        ),
    )
    expected_risk = 100.0 * ordered["binary_event_n"] / ordered["binary_n"]
    if not np.allclose(expected_risk, ordered["binary_percentage"], atol=1e-8):
        raise ValueError("E3 binary risks do not reconcile to counts")
    if not (
        (ordered["continuous_q25"] <= ordered["continuous_median"])
        & (ordered["continuous_median"] <= ordered["continuous_q75"])
    ).all():
        raise ValueError("E3 LOS quantiles are reversed")

    out_dir.mkdir(parents=True, exist_ok=True)
    source_files = {
        name: _copy_source(frame, paths[name], out_dir / f"e3_{name}_source_data.csv")
        for name, frame in frames.items()
    }
    palette = apply_publication_style(font_size=7.0)
    level_labels = [f"KDIGO stage {int(value)}" for value in ordered["level_value"]]

    fig, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 82 / 25.4), constrained_layout=True
    )
    positions = np.arange(len(ordered))
    risk = ordered["binary_percentage"].to_numpy(dtype=float)
    risk_low = 100.0 * ordered["binary_ci_low"].to_numpy(dtype=float)
    risk_high = 100.0 * ordered["binary_ci_high"].to_numpy(dtype=float)
    axes[0].errorbar(
        positions,
        risk,
        yerr=np.vstack((risk - risk_low, risk_high - risk)),
        fmt="o-",
        color=palette["blue"],
        capsize=2.5,
    )
    axes[0].set_xticks(positions, level_labels, rotation=20, ha="right")
    axes[0].set_ylabel("Observed mortality risk (%)")
    axes[0].set_title("Mortality across ordered KDIGO stages", loc="left", pad=10)
    add_panel_label(axes[0], "a", x=-0.12, y=1.04, fontsize=8.0)

    median = ordered["continuous_median"].to_numpy(dtype=float)
    q25 = ordered["continuous_q25"].to_numpy(dtype=float)
    q75 = ordered["continuous_q75"].to_numpy(dtype=float)
    axes[1].errorbar(
        positions,
        median,
        yerr=np.vstack((median - q25, q75 - median)),
        fmt="o-",
        color=palette["orange"],
        capsize=2.5,
    )
    axes[1].set_xticks(positions, level_labels, rotation=20, ha="right")
    axes[1].set_ylabel("ICU length of stay (days), median (IQR)")
    axes[1].set_title("ICU length of stay across KDIGO stages", loc="left", pad=10)
    add_panel_label(axes[1], "b", x=-0.12, y=1.04, fontsize=8.0)
    figure_files = _save(
        fig,
        out_dir=out_dir,
        product="e3_main_figure_1_outcomes_by_kdigo_stage",
        height_mm=82.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Mortality across ordered KDIGO stages",
                "role": "descriptive_result",
                "article_role": "descriptive_result",
                "chart_type": "dot_interval_absolute_risk",
                "claim": "Observed mortality risk with Wilson 95% confidence intervals is shown for every ordered stage.",
                "evidence_ids": [_sha256(paths["ordered"])],
                "metadata": {"source_data": [source_files["ordered"]]},
            },
            {
                "panel_id": "b",
                "title": "ICU length of stay across KDIGO stages",
                "role": "descriptive_result",
                "article_role": "descriptive_result",
                "chart_type": "dot_interval",
                "claim": "Median ICU length of stay and interquartile range are shown for every ordered stage.",
                "evidence_ids": [_sha256(paths["ordered"])],
                "metadata": {"source_data": [source_files["ordered"]]},
            },
        ],
        source_data=[source_files["ordered"]],
        core_claim="Mortality risk and ICU length of stay are displayed separately across all ordered KDIGO stages.",
        statistics_note="All ordered-stratified rows are preserved. Mortality intervals are the source Wilson 95% confidence intervals; length of stay is the source median and IQR. No model is refit.",
    )

    adjusted = frames["adjusted"].copy()
    sensitivity = frames["sensitivity"].copy()
    _finite(adjusted, ("estimate", "ci_low", "ci_high"))
    _finite(sensitivity, ("estimate", "ci_low", "ci_high"))
    if not adjusted["fit_status"].astype(str).eq("fitted").all():
        raise ValueError("E3 adjusted association contains non-fitted rows")
    if not sensitivity["converged"].astype(str).str.casefold().eq("true").all():
        raise ValueError("E3 sensitivity table contains non-converged rows")
    fig, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 100 / 25.4), constrained_layout=True
    )
    _forest(
        axes[0],
        adjusted,
        [str(value).replace(" vs ", " vs stage ") for value in adjusted["contrast"]],
        "Primary adjusted mortality association",
        palette["blue"],
    )
    add_panel_label(axes[0], "a", x=-0.12, y=1.04, fontsize=8.0)
    _forest(
        axes[1],
        sensitivity,
        [str(value).replace("_", " ") for value in sensitivity["analysis_id"]],
        "Scientific sensitivity analyses",
        palette["orange"],
    )
    add_panel_label(axes[1], "b", x=-0.12, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="e3_main_figure_2_adjusted_and_sensitivity",
        height_mm=100.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Primary adjusted mortality association",
                "role": "primary_estimand",
                "article_role": "primary_estimand",
                "chart_type": "forest",
                "claim": "All fitted stage contrasts from the registered primary model are shown with 95% confidence intervals.",
                "evidence_ids": [_sha256(paths["adjusted"])],
                "metadata": {"source_data": [source_files["adjusted"]]},
            },
            {
                "panel_id": "b",
                "title": "Scientific sensitivity analyses",
                "role": "robustness",
                "article_role": "robustness",
                "chart_type": "sensitivity_forest",
                "claim": "Every converged registered sensitivity analysis is shown with 95% confidence intervals.",
                "evidence_ids": [_sha256(paths["sensitivity"])],
                "metadata": {"source_data": [source_files["sensitivity"]]},
            },
        ],
        source_data=[source_files["adjusted"], source_files["sensitivity"]],
        core_claim="The registered adjusted association and all converged scientific sensitivity analyses are shown without refitting.",
        statistics_note="Odds ratios and 95% confidence intervals are copied from the registered primary and sensitivity tables. The renderer performs no row selection or model fitting.",
    )

    missingness = frames["missingness"].copy()
    _finite(missingness, ("missing_n", "n_total", "missing_pct"))
    expected_missing = 100.0 * missingness["missing_n"] / missingness["n_total"]
    if not np.allclose(expected_missing, missingness["missing_pct"], atol=1e-8):
        raise ValueError("E3 missingness percentages do not reconcile to counts")
    quality = missingness.sort_values("missing_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 105 / 25.4), constrained_layout=True)
    positions = np.arange(len(quality))
    ax.barh(positions, quality["missing_pct"], color=palette["blue_soft"])
    ax.set_yticks(positions, quality["label"].astype(str).str.replace("_", " "))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing among eligible records (%)")
    ax.set_title("Measurement and missingness audit", loc="left", pad=10)
    add_panel_label(ax, "a", x=-0.12, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="e3_supplementary_figure_s1_missingness",
        height_mm=105.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Measurement and missingness audit",
                "role": "data_quality",
                "article_role": "data_quality",
                "chart_type": "availability_panel",
                "claim": "Routine measurement missingness is shown as supplementary audit evidence.",
                "evidence_ids": [_sha256(paths["missingness"])],
                "metadata": {
                    "placement": "supplementary",
                    "source_data": [source_files["missingness"]],
                },
            }
        ],
        source_data=[source_files["missingness"]],
        core_claim="Routine E3 missingness is retained as supplementary audit evidence rather than a primary-result panel.",
        statistics_note="All rows and source percentages are preserved and reconciled to counts. Missingness is not interpreted as a clinical effect.",
    )
    return {
        "source_run": str(source_run),
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "figure_files": sorted(set(figure_files)),
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    e3_run = source_root / E3_RUN_RELATIVE
    m2_run = source_root / M2_RUN_RELATIVE
    e3 = _render_e3(e3_run, output_root / "e3")
    m2_summary = run_prediction_figure(
        out_dir=output_root / "m2",
        run_dir=m2_run,
        resolved_inputs=m2_run / "resolved_inputs/prediction_figure_suite.json",
        step_id="prediction_figure_suite",
        figure_product="m2_main_figure_prediction_performance",
    )
    try:
        code_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_head = "unknown"
    manifest = {
        "schema_version": "easyicu.dev9_article_display_remediation/1",
        "code_head": code_head,
        "source_root": str(source_root),
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "provider_calls": 0,
        "scientific_recomputation": False,
        "e3": e3,
        "m2": m2_summary,
    }
    (output_root / "article_display_remediation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
