#!/usr/bin/env python3
"""Render a Nature-style multi-panel figure from an EasyICU run directory.

This helper reads an existing EasyICU run (cohort parquet + evidence tables),
recomputes the source data needed for a claim-first manuscript figure, and
exports a publication bundle (SVG/PDF/PNG/TIFF + contract + QA notes).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from easyicu.research_agent.publication_figures import (
    add_panel_label,
    apply_publication_style,
    audit_figure_contract,
    audit_publication_exports,
    make_figure_contract,
    save_publication_figure,
)


RUN_NAME_DEFAULT = "sofa2_mortality_missingness_audit"


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def _safe_missing_pct(series: pd.Series) -> float:
    return float(series.isna().mean() * 100.0)


def _build_panel_data(run_dir: Path) -> Dict[str, pd.DataFrame]:
    cohort_path = run_dir / "cohort.parquet"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Missing cohort parquet: {cohort_path}")
    cohort = pd.read_parquet(cohort_path)

    outcome_df = _load_required(
        run_dir / "steps/02_outcome_incidence/outputs/outcome_incidence.csv"
    )
    zero_df = _load_required(
        run_dir / "steps/03_missingness_audit/outputs/sofa2_zero_anomalies.csv"
    )

    score_counts = (
        cohort.groupby("sofa2", dropna=False)
        .agg(n_total=("death", "size"), n_events=("death", "sum"))
        .reset_index()
        .sort_values("sofa2")
    )
    score_counts["mortality_rate"] = score_counts["n_events"] / score_counts["n_total"]
    cis = score_counts.apply(
        lambda row: _wilson_ci(int(row["n_events"]), int(row["n_total"])), axis=1
    )
    score_counts["ci_low"] = [lo for lo, _ in cis]
    score_counts["ci_high"] = [hi for _, hi in cis]
    score_counts["mortality_pct"] = score_counts["mortality_rate"] * 100.0
    score_counts["ci_low_pct"] = score_counts["ci_low"] * 100.0
    score_counts["ci_high_pct"] = score_counts["ci_high"] * 100.0

    distribution_df = (
        cohort.groupby("sofa2", dropna=False)
        .size()
        .rename("n_stays")
        .reset_index()
        .sort_values("sofa2")
    )
    distribution_df["pct_stays"] = distribution_df["n_stays"] / len(cohort) * 100.0

    missingness_variables = [
        "vaso",
        "sofa2_liver",
        "bili",
        "sofa2",
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
    ]
    missingness_df = pd.DataFrame(
        {
            "variable": missingness_variables,
            "pct_missing": [
                _safe_missing_pct(cohort[var]) if var in cohort.columns else math.nan
                for var in missingness_variables
            ],
        }
    )
    missingness_df["label"] = missingness_df["variable"].replace(
        {
            "vaso": "Vasopressor",
            "bili": "Bilirubin",
            "sofa2_liver": "SOFA-2 liver",
            "sofa2_resp": "SOFA-2 respiratory",
            "sofa2_coag": "SOFA-2 coagulation",
            "sofa2_cardio": "SOFA-2 cardiovascular",
            "sofa2_cns": "SOFA-2 CNS",
            "sofa2_renal": "SOFA-2 renal",
            "sofa2": "SOFA-2 total",
        }
    )
    missingness_df = missingness_df.sort_values("pct_missing", ascending=True)

    zero_df = zero_df.copy()
    zero_df["label"] = zero_df["component"].replace(
        {
            "sofa2_resp": "Respiratory",
            "sofa2_coag": "Coagulation",
            "sofa2_liver": "Liver",
            "sofa2_cardio": "Cardiovascular",
            "sofa2_cns": "CNS",
            "sofa2_renal": "Renal",
        }
    )

    summary = {
        "n_total": int(outcome_df["n_total"].iloc[0]),
        "n_events": int(outcome_df["n_events"].iloc[0]),
        "outcome_rate": float(outcome_df["outcome_rate"].iloc[0]),
    }
    summary["outcome_rate_ci_low"], summary["outcome_rate_ci_high"] = _wilson_ci(
        summary["n_events"], summary["n_total"]
    )

    return {
        "cohort": cohort,
        "mortality_by_score": score_counts,
        "score_distribution": distribution_df,
        "missingness": missingness_df,
        "zero_anomalies": zero_df,
        "outcome": outcome_df,
        "summary": pd.DataFrame([summary]),
    }


def _write_source_data(source_dir: Path, panel_data: Dict[str, pd.DataFrame]) -> List[str]:
    source_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "panel_a_mortality_by_sofa2.csv": panel_data["mortality_by_score"],
        "panel_b_sofa2_distribution.csv": panel_data["score_distribution"],
        "panel_c_missingness.csv": panel_data["missingness"],
        "panel_d_zero_score_component_missingness.csv": panel_data["zero_anomalies"],
        "figure_summary.csv": panel_data["summary"],
    }
    written: List[str] = []
    for name, df in outputs.items():
        path = source_dir / name
        df.to_csv(path, index=False)
        written.append(str(path))
    return written


def _build_contract(source_files: Iterable[str], evidence_ids: Dict[str, str]):
    return make_figure_contract(
        figure_id="Figure1_sofa2_mortality_qc",
        core_claim=(
            "Early SOFA-2 severity follows a mortality gradient in the EasyICU synthetic cohort, "
            "but score-zero strata and liver/vasopressor missingness require explicit quality auditing."
        ),
        archetype="asymmetric_mixed_modality",
        width_mm=183.0,
        height_mm=122.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Mortality by SOFA-2 score",
                "role": "relationship",
                "claim": "Mortality increases across higher SOFA-2 scores, with score-zero behaviour needing separate scrutiny.",
                "evidence_ids": [
                    evidence_ids.get("mortality_by_sofa2"),
                    evidence_ids.get("outcome_incidence"),
                ],
                "review_risk": "Score-zero behaviour may reflect ascertainment artefact rather than true low severity.",
            },
            {
                "panel_id": "b",
                "title": "SOFA-2 distribution",
                "role": "overview",
                "claim": "The cohort is concentrated in low SOFA-2 strata, so severity distribution must be seen alongside mortality rates.",
                "evidence_ids": [evidence_ids.get("sofa2_distribution"), evidence_ids.get("cohort_summary")],
                "review_risk": "Distribution alone should not be interpreted as a risk estimate.",
            },
            {
                "panel_id": "c",
                "title": "Missingness burden",
                "role": "audit",
                "claim": "Liver-related variables and vasopressor exposure carry the heaviest missingness burden.",
                "evidence_ids": [evidence_ids.get("missingness_audit")],
                "review_risk": "Missingness is descriptive here and does not establish mechanism.",
            },
            {
                "panel_id": "d",
                "title": "Zero-score component audit",
                "role": "validation",
                "claim": "SOFA-2 zero strata disproportionately lose liver-component information.",
                "evidence_ids": [evidence_ids.get("zero_anomalies")],
                "review_risk": "This is a dataset quality audit, not an external clinical prevalence claim.",
            },
        ],
        source_data=list(source_files),
        statistics_note=(
            "Panel a shows mortality proportions with 95% Wilson confidence intervals. "
            "Panel b shows cohort counts by SOFA-2 score. Panels c-d report missingness percentages."
        ),
        image_integrity_note=(
            "All panels are vector or line-art exports drawn from EasyICU run tables/cohort parquet; no pixel-level image manipulation was applied."
        ),
    )


def _build_evidence_aliases(run_dir: Path) -> Dict[str, str]:
    alias_path = run_dir / "evidence" / "evidence_aliases.json"
    aliases = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
    return {
        "cohort_summary": aliases.get("cohort_summary", "table_cohort_summary"),
        "outcome_incidence": aliases.get("outcome_incidence", "table_outcome_incidence"),
        "missingness_audit": aliases.get("missingness_audit", "table_missingness_audit"),
        "zero_anomalies": aliases.get("sofa2_zero_anomalies", "table_sofa2_zero_anomalies"),
        "mortality_by_sofa2": aliases.get("mortality_by_sofa2", "figure_mortality_by_sofa2"),
        "sofa2_distribution": aliases.get("sofa2_distribution", "figure_sofa2_distribution"),
    }


def _render_figure(
    panel_data: Dict[str, pd.DataFrame],
    *,
    run_dir: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    palette = apply_publication_style(font_size=6.8)
    fig = plt.figure(figsize=(183 / 25.4, 122 / 25.4), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.35, 1.15, 0.95],
        height_ratios=[1.0, 1.0],
        wspace=0.55,
        hspace=0.6,
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1:3])

    mortality = panel_data["mortality_by_score"]
    ax_a.errorbar(
        mortality["sofa2"],
        mortality["mortality_pct"],
        yerr=[
            mortality["mortality_pct"] - mortality["ci_low_pct"],
            mortality["ci_high_pct"] - mortality["mortality_pct"],
        ],
        color=palette["blue"],
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        capsize=2.0,
    )
    ax_a.axvspan(-0.35, 0.35, color=palette["band"], zorder=0)
    ax_a.set_xlabel("SOFA-2 score")
    ax_a.set_ylabel("ICU mortality (%)")
    ax_a.set_title("Mortality gradient by SOFA-2 severity", loc="left", x=0.08, pad=6)
    ax_a.set_ylim(bottom=0)
    ax_a.text(
        0.02,
        0.98,
        "Score 0 highlighted for QC",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=palette["neutral"],
    )
    add_panel_label(ax_a, "a", x=-0.005, y=1.03, fontsize=10.5)

    dist = panel_data["score_distribution"]
    x_vals = dist["sofa2"].astype(int).to_numpy()
    ax_b.bar(
        x_vals,
        dist["n_stays"],
        color=palette["blue_soft"],
        edgecolor=palette["blue"],
        linewidth=0.5,
        width=0.82,
    )
    ax_b.set_xlabel("SOFA-2")
    ax_b.set_ylabel("Stays (n)")
    ax_b.set_title("SOFA-2 distribution", loc="left", x=0.10, pad=6)
    tick_candidates = x_vals[::2] if len(x_vals) > 8 else x_vals
    ax_b.set_xticks(tick_candidates)
    ax_b.set_xlim(x_vals.min() - 0.75, x_vals.max() + 0.75)
    summary = panel_data["summary"].iloc[0]
    ax_b.text(
        0.98,
        0.98,
        f"n={int(summary['n_total'])}\ndeaths={int(summary['n_events'])}\nrate={summary['outcome_rate']*100:.1f}%",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=palette["baseline"],
    )
    add_panel_label(ax_b, "b", x=-0.005, y=1.03, fontsize=10.5)

    missing = panel_data["missingness"]
    bar_colors = [
        palette["red_soft"] if v in {"Vasopressor", "Bilirubin", "SOFA-2 liver"} else palette["neutral_light"]
        for v in missing["label"]
    ]
    ax_c.barh(missing["label"], missing["pct_missing"], color=bar_colors, edgecolor=palette["neutral"])
    ax_c.axvline(20, color=palette["neutral"], linestyle="--", linewidth=0.8)
    ax_c.set_xlabel("Missing values (%)")
    ax_c.set_title("Key variable missingness", loc="left", x=0.08, pad=6)
    ax_c.set_xlim(0, max(80, float(missing["pct_missing"].max()) + 5))
    add_panel_label(ax_c, "c", x=-0.005, y=1.03, fontsize=10.5)

    zero = panel_data["zero_anomalies"]
    zero_colors = [
        palette["red"] if label == "Liver" else palette["teal"] for label in zero["label"]
    ]
    ax_d.barh(
        zero["label"],
        zero["pct_component_missing_at_zero"],
        color=zero_colors,
        edgecolor="none",
        height=0.62,
    )
    ax_d.set_xlabel("Missing among SOFA-2 = 0 stays (%)")
    ax_d.set_title("Score-zero component missingness", loc="left", x=0.06, pad=6)
    ax_d.set_xlim(0, max(60, float(zero["pct_component_missing_at_zero"].max()) + 5))
    ax_d.invert_yaxis()
    add_panel_label(ax_d, "d", x=-0.005, y=1.03, fontsize=10.5)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.tick_params(length=3, width=0.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    source_files = _write_source_data(source_dir, panel_data)
    contract = _build_contract(source_files, _build_evidence_aliases(run_dir=run_dir))
    contract_findings = audit_figure_contract(contract)
    paths = save_publication_figure(
        fig,
        output_dir / RUN_NAME_DEFAULT,
        contract=contract,
    )
    export_findings = audit_publication_exports(paths)
    qa_payload = {
        "contract_findings": [f.model_dump() for f in contract_findings],
        "export_findings": [f.model_dump() for f in export_findings],
    }
    (output_dir / f"{RUN_NAME_DEFAULT}.qa.json").write_text(
        json.dumps(qa_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    contract_payload = {
        "core_conclusion": contract.core_claim,
        "figure_archetype": contract.archetype,
        "target_journal_output": "Nature-style double-column figure bundle (SVG/PDF/PNG/TIFF)",
        "backend": "Python",
        "final_size_mm": {"width": contract.width_mm, "height": contract.height_mm},
        "panel_map": {panel.panel_id: panel.title for panel in contract.panels},
        "evidence_hierarchy": {
            "hero_evidence": ["panel a"],
            "validation_evidence": ["panel d"],
            "controls_or_context": ["panel b", "panel c"],
        },
        "statistics_needed": contract.statistics_note,
        "source_data_needed": source_files,
        "image_integrity_notes": contract.image_integrity_note,
        "reviewer_risk": [panel.review_risk for panel in contract.panels if panel.review_risk],
    }
    (output_dir / f"{RUN_NAME_DEFAULT}.contract.json").write_text(
        json.dumps(contract_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "nature_figure")).resolve()
    panel_data = _build_panel_data(run_dir)
    paths = _render_figure(panel_data, run_dir=run_dir, output_dir=output_dir)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
