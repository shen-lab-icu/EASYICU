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


def test_initial_coder_contract_preserves_exact_planned_panel_coordinates() -> None:
    contract = _declared_output_scope_contract(_planned_figure_step())

    assert "PLANNED PANEL CONTRACT (binding)" in contract
    for field in ("panel_id", "article_role", "chart_type", "source_products"):
        assert field in contract
    assert "Do not split, merge, rename, infer, or invent" in contract


def test_repair_scope_keeps_planned_panel_cardinality_and_coordinates() -> None:
    contract = _compact_repair_scope_contract(_planned_figure_step())

    assert "exact planned figure_panels cardinality" in contract
    assert "never split or merge planned panels" in contract
