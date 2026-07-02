"""The deterministic association forest-plot rescue must tolerate the OR/CI
column-name variants that free-model code emits (e.g. ``ci_lower``/``ci_upper``
rather than ``or_ci_low``/``or_ci_high``). Without this, a figure-only step
fails the whole run even though the parent step computed a valid odds ratio.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from easyicu.research_agent.pipeline import (
    _render_association_publication_bundle_from_prior_outputs as rescue,
    _render_sensitivity_publication_bundle_from_prior_outputs as sensitivity_rescue,
)
from easyicu.research_agent.schema import AnalysisStep


def _make_parent_step(run_dir: Path, csv_name: str, columns: dict) -> None:
    out = run_dir / "steps" / "03_association_model" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(out / csv_name, index=False)


def test_rescue_handles_ci_lower_upper_variant(tmp_path: Path):
    # free-model style column names: odds_ratio + ci_lower/ci_upper
    _make_parent_step(
        tmp_path,
        "adjusted_odds_ratios.csv",
        {
            "variable": ["const", "sepsis3", "age"],
            "odds_ratio": [0.01, 0.80, 1.03],
            "ci_lower": [0.0, 0.74, 1.01],
            "ci_upper": [0.1, 0.86, 1.05],
        },
    )
    out = tmp_path / "steps" / "03_association_model_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    rid = rescue(
        run_dir=tmp_path, current_step_id="03_association_model_figure", out_dir=out
    )
    assert rid is not None
    figs = {p.suffix for p in out.iterdir()}
    assert ".png" in figs and ".svg" in figs
    contract_path = out / "publication_figure.figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B"]
    assert (out / "publication_figure_source_data.csv").exists()
    findings = FigureContractQualityValidator().audit_contract_file(
        contract_path,
        manuscript_facing=True,
    )
    assert not any(f.severity == "error" for f in findings), findings
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="03_association_model_figure",
            intent="Render the publication figure declared by step '03_association_model'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []


def test_rescue_handles_canonical_or_ci_columns(tmp_path: Path):
    # our deterministic fallback style: or_ci_low/or_ci_high
    _make_parent_step(
        tmp_path,
        "association_results.csv",
        {
            "variable": ["sepsis3"],
            "odds_ratio": [0.80],
            "or_ci_low": [0.74],
            "or_ci_high": [0.86],
        },
    )
    out = tmp_path / "steps" / "03_fig" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    rid = rescue(run_dir=tmp_path, current_step_id="03_fig", out_dir=out)
    assert rid is not None


def test_rescue_returns_none_without_or_ci_table(tmp_path: Path):
    _make_parent_step(
        tmp_path, "prevalence.csv", {"group": ["a"], "rate": [0.3]}
    )
    out = tmp_path / "steps" / "03_fig" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    assert rescue(run_dir=tmp_path, current_step_id="03_fig", out_dir=out) is None


def test_sensitivity_rescue_writes_multipanel_contract_and_source_data(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "05_sensitivity_comparison" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "spec_id": ["primary", "alt_cohort", "risk_difference"],
            "axis": ["cohort", "cohort", "outcome"],
            "display_label": ["Primary cohort", "Alternative cohort", "Risk difference"],
            "effect_scale": ["OR", "OR", "RD"],
            "point_estimate": [1.12, 1.05, 0.03],
            "ci_low": [1.02, 0.95, 0.01],
            "ci_high": [1.24, 1.17, 0.05],
            "se": [0.05, 0.06, 0.01],
            "modeled_analytic_n": [1000, 920, 1000],
            "converged": [True, True, True],
        }
    ).to_csv(parent / "sensitivity_comparison.csv", index=False)
    out = tmp_path / "steps" / "05_sensitivity_comparison_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    rid = sensitivity_rescue(
        run_dir=tmp_path,
        current_step_id="05_sensitivity_comparison_figure",
        out_dir=out,
    )

    assert rid == "sensitivity_publication_bundle_from_parent_outputs_v1"
    contract_path = out / "sensitivity_forest.figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B", "C"]
    assert (out / "sensitivity_forest_source_data.csv").exists()
    assert FigureContractQualityValidator().audit_contract_file(
        contract_path,
        manuscript_facing=True,
    ) == []
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_sensitivity_comparison_figure",
            intent="Render the sensitivity figure declared by step '05_sensitivity_comparison'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []
