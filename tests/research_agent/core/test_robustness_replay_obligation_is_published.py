"""The sentence that asks for the robustness step must ask for the whole step.

E1 reached 9 of 12 steps in canary14 -- cohort, Table 1, missingness audit,
exposure-outcome distribution, the primary adjusted association and three
figures -- and then lost step 09.  Its own step_summary.json says::

    status: blocked
    blocking_reason: Executable locked robustness specifications did not emit
                     verifiable estimates: complete_case_primary_variables
    n_locked_specs: 1   n_converged_variants: 0
    warnings: ['generic deterministic robustness refitting is disabled; only a
               validated step-owned primary estimate is retained, and variants
               require exact registered primary-script replay']

The run had locked a grid the plan declared, and no step declared
``robustness_replay_spec``, which is the only route to the deterministic replay
owner -- generic refitting is deliberately off, because it would choose an
exposure, outcome or method on the plan's behalf.

The Planner is TOLD to create that step: "Add an auxiliary post-primary step
with method='robustness_sensitivity' producing table:robustness_matrix and
statistic:robustness_summary."  That sentence did not mention the field that
makes the step runnable; the field was described ninety lines earlier, behind a
warning about not claiming a replay when the step introduces new science.  So
the step gets created and left unrunnable.

E3, M1 and H2 stop at the same layer.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.core import _build_planner_user_prompt
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
    RobustnessReplaySpec,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Is the exposure associated with in-hospital death?",
        cohort=CohortDescriptor(
            cohort_name="robustness-obligation",
            database="test",
            n_patients=100,
            n_stays=100,
            id_columns=["stay_id"],
        ),
        variables=[],
    )


@pytest.fixture(scope="module")
def directive() -> str:
    return _build_planner_user_prompt(_context())


# --- the obligation is stated where the step is asked for --------------------


def test_the_robustness_step_sentence_names_the_field_that_runs_it(directive):
    marker = "method='robustness_sensitivity'"
    assert marker in directive
    tail = directive[directive.index(marker) :][:1200]

    assert "robustness_replay_spec" in tail


def test_the_consequence_of_omitting_it_is_stated_not_implied(directive):
    marker = "method='robustness_sensitivity'"
    tail = directive[directive.index(marker) :][:1200]

    assert "estimated by" in tail or "no such declaration" in tail


def test_declaring_the_grid_is_said_to_oblige_a_step(directive):
    assert "obliges exactly one step" in directive


# --- product_id is the step's own output name, not the output enum -----------


def test_the_directive_separates_product_id_from_output(directive):
    marker = "method='robustness_sensitivity'"
    tail = directive[directive.index(marker) :][:1200]

    assert "product_id" in tail
    assert "different field from `output`" in tail


def test_the_worked_example_shows_the_prefix_being_stripped(directive):
    marker = "method='robustness_sensitivity'"
    tail = directive[directive.index(marker) :][:1200]

    assert "table:robustness_matrix" in tail
    assert "'robustness_matrix'" in tail


# --- and the example the directive gives actually validates ------------------


def _step_from_the_directive() -> AnalysisStep:
    """Exactly what the two sentences together instruct: the step, and a spec
    whose product_id is that step's own declared output minus its prefix."""

    return AnalysisStep(
        step_id="09_robustness_sensitivity",
        intent="Re-estimate the locked grid without changing the estimand.",
        method="robustness_sensitivity",
        expected_outputs=[
            "table:robustness_matrix",
            "statistic:robustness_summary",
        ],
        inputs=[],
        robustness_replay_spec=RobustnessReplaySpec(
            products=[
                {"product_id": "robustness_matrix", "output": "robustness_matrix"},
                {"product_id": "robustness_summary", "output": "robustness_summary"},
            ]
        ),
    )


def test_following_the_directive_produces_a_step_the_schema_accepts():
    """If this raises, the host is instructing a plan it will then refuse."""

    step = _step_from_the_directive()

    assert step.robustness_replay_spec is not None
    assert [item.product_id for item in step.robustness_replay_spec.products] == [
        "robustness_matrix",
        "robustness_summary",
    ]


def test_using_an_output_enum_value_the_step_never_declares_is_refused():
    """The mistake the directive now warns against, and why it warns.

    A real plan named complete_case_n / membership_change /
    missingness_strategy_notes as product_ids -- all three are legal ``output``
    values, none was one of that step's declared outputs -- and the whole plan
    was rejected.
    """

    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="does not declare as outputs"):
        AnalysisStep(
            step_id="09_robustness_sensitivity",
            intent="x",
            method="robustness_sensitivity",
            expected_outputs=["table:robustness_matrix"],
            inputs=[],
            robustness_replay_spec=RobustnessReplaySpec(
                products=[
                    {"product_id": "complete_case_n", "output": "complete_case_n"},
                ]
            ),
        )
