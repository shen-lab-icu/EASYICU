"""A structural-accounting guard protects only the accounting tables a step draws.

The preflight's structural filter and integer findings apply to the tables
``structural_accounting_products`` names: an accounting-named input, else the
own sources of an accounting-role panel, else every table input of a step
whose own intent or outputs are accounting-shaped.  Synthetic steps only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.figure_plan import PlannedFigurePanelSpec
from easyicu.research_agent.gates.structural_accounting import (
    is_structural_accounting_name,
    structural_accounting_products,
    typed_table_products,
)
from easyicu.research_agent.schema import AnalysisStep


@pytest.mark.parametrize(
    "name",
    [
        "cohort_flow",
        "table:denominator_reconciliation",
        "Attrition by stage",
        "consort_diagram",
        "participant_funnel",
        "eligibility-accounting",
        "source availability",
        "universe_counts",
    ],
)
def test_an_accounting_name_is_recognised(name: str) -> None:
    assert is_structural_accounting_name(name)


@pytest.mark.parametrize(
    "name",
    ["", "effect_estimates", "flow", "cohort_summary", "source_data", "universe"],
)
def test_a_name_without_an_accounting_role_is_not(name: str) -> None:
    assert not is_structural_accounting_name(name)


def test_only_table_tokens_name_table_products() -> None:
    assert typed_table_products(
        ["table:Flow", "figure:forest", "selected_first", "table:", None]
    ) == {"flow"}


def _panel(panel_id: str, role: str, sources: list[str]) -> PlannedFigurePanelSpec:
    return PlannedFigurePanelSpec(
        panel_id=panel_id,
        figure_output=f"figure:{role}",
        article_role=role,
        chart_type="bar",
        source_products=sources,
    )


def _step(intent: str, inputs: list[str], panels=()) -> AnalysisStep:
    outputs = [panel.figure_output for panel in panels] or ["figure:summary"]
    return AnalysisStep(
        step_id="render",
        intent=intent,
        inputs=inputs,
        expected_outputs=outputs,
        method="visualization",
        figure_panels=list(panels),
    )


def test_an_accounting_named_input_is_protected_on_its_own() -> None:
    step = _step(
        "Render panels.",
        ["table:cohort_flow", "table:effect_estimates"],
        [_panel("eff", "primary_effect", ["table:effect_estimates"])],
    )

    assert structural_accounting_products(step) == {"cohort_flow"}


def test_an_accounting_panel_protects_only_its_own_sources() -> None:
    step = _step(
        "Render panels.",
        ["table:effect_estimates", "table:renamed_counts"],
        [
            _panel("acc", "cohort_accounting", ["table:renamed_counts"]),
            _panel("eff", "primary_effect", ["table:effect_estimates"]),
        ],
    )

    assert structural_accounting_products(step) == {"renamed_counts"}


def test_a_step_label_protects_every_table_only_without_panel_bindings() -> None:
    inputs = ["table:renamed_counts", "selected_first"]
    unbound = _step("Render the cohort accounting summary.", inputs)
    bound = _step(
        "Render the cohort accounting summary.",
        inputs,
        [_panel("eff", "primary_effect", ["table:renamed_counts"])],
    )
    unlabelled = _step("Render the summary.", inputs)

    assert structural_accounting_products(unbound) == {"renamed_counts"}
    assert structural_accounting_products(bound) == set()
    assert structural_accounting_products(unlabelled) == set()
