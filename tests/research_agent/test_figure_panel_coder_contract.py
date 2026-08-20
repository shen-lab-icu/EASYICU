from __future__ import annotations

from easyicu.research_agent.agents.coder import (
    _compact_repair_scope_contract,
    _declared_output_scope_contract,
)
from easyicu.research_agent.schema import AnalysisStep


def _planned_figure_step() -> AnalysisStep:
    return AnalysisStep.model_validate(
        {
            "step_id": "render",
            "planned_analysis_role": "auxiliary",
            "intent": "Render the planned result figure.",
            "method": "visualization",
            "inputs": ["table:result"],
            "expected_outputs": ["figure:result"],
            "figure_panels": [
                {
                    "panel_id": "panel_a",
                    "figure_output": "figure:result",
                    "article_role": "descriptive_result",
                    "chart_type": "grouped_bar",
                    "source_products": ["table:result"],
                }
            ],
        }
    )


def _unplanned_figure_step() -> AnalysisStep:
    return AnalysisStep.model_validate(
        {
            "step_id": "render",
            "planned_analysis_role": "auxiliary",
            "intent": "Render the declared figure from typed inputs.",
            "method": "visualization",
            "inputs": ["table:result"],
            "expected_outputs": ["figure:result"],
            "figure_panels": [],
        }
    )


def test_initial_coder_contract_preserves_exact_planned_panel_coordinates() -> None:
    contract = _declared_output_scope_contract(_planned_figure_step())

    assert "PLANNED PANEL CONTRACT (binding)" in contract
    for field in ("panel_id", "article_role", "chart_type", "source_products"):
        assert field in contract
    assert "Match its cardinality" in contract
    assert "AnalysisPlan.figure_panels is invalid" in contract


def test_repair_scope_keeps_planned_panel_cardinality_and_coordinates() -> None:
    contract = _compact_repair_scope_contract(_planned_figure_step())

    assert "Match its cardinality" in contract
    assert "copy panel_id, article_role, chart_type" in contract


def test_empty_planned_panels_authorize_truthful_runtime_composition() -> None:
    initial = _declared_output_scope_contract(_unplanned_figure_step())
    repair = _compact_repair_scope_contract(_unplanned_figure_step())

    for contract in (initial, repair):
        assert "UNPLANNED PANEL COMPOSITION (binding)" in contract
        assert "AnalysisStep.figure_panels empty" in contract
        assert "AnalysisPlan.figure_panels invalid" in contract
        assert "Runtime panels describe declared-input plots" in contract
        assert "Match its cardinality" not in contract
