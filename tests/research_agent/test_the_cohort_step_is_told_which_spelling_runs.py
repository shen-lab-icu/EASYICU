"""Five legal spellings, one of them executable, and the directive said so
about none of them.

The Planner is told to declare "exactly one materialised closed primary-cohort
product" and offered the whole published vocabulary -- ``artifact:``,
``dataset:``, ``table:analysis_cohort``, ``cohort:analysis_set``,
``cohort:<name>``.  All five are legal.  Only one PAIR lets the host execute
the step itself: ``_declares_host_cohort_products`` accepts
``{artifact:analysis_cohort, table:cohort_flow}`` exactly, and nothing else.

Measured 2026-08-02 over 282 recorded plans: 142 first steps declared that
exact pair; 64 more declared ``cohort:analysis_set`` + ``table:cohort_flow``,
equally endorsed by the directive and executable by nobody.  Of the 142 with a
recorded replay, 39 were still Coder-written and 15 needed repair -- that half
was a separate defect (the host refused to adopt its own typed materialization,
fixed in ``load_materialized_analysis_cohort_result``).  This file covers the
other half: a Planner that was never told which spelling buys the executor.

That is the same shape ``_closed_cohort_product_sentence`` was written to
prevent one layer up -- an offer wider than the enforcement -- so the fix is
the same: render the sentence from the constants that decide it.

The step this concerns is the most expensive one in the system to lose: over
the recorded runs it was Coder-written in 127 of 127 plans, failed in 21 for
ten unrelated reasons, and each failure killed a mean of 5.1 downstream steps
-- 59% of every cascade in the corpus.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from easyicu.research_agent.agents.core import (
    PlannerAgent,
    ReplannerAgent,
    _build_planner_user_prompt,
    _host_executed_cohort_step_sentence,
    _planner_retry_response_projection,
)
from easyicu.research_agent.agents.plan_payload import planner_science_retry_guide
from easyicu.research_agent.authority.run_input import (
    _declares_host_cohort_products,
)
from easyicu.research_agent.schema import (
    COHORT_DEFINITION_COHORT_OUTPUT,
    COHORT_DEFINITION_FLOW_OUTPUT,
    CohortDefinitionSpec,
    CohortDescriptor,
    ResearchContext,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Is the exposure associated with in-hospital death?",
        cohort=CohortDescriptor(
            cohort_name="cohort-step-spelling",
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


class _Step:
    def __init__(self, outputs):
        self.expected_outputs = list(outputs)


# ---------------------------------------------------------------------------
# The pair the directive names is the pair the host executes
# ---------------------------------------------------------------------------


def test_the_named_pair_is_the_one_the_predicate_accepts():
    """Otherwise the directive would be advertising a different contract.

    Anchored on the predicate, not on a transcribed literal: widening or
    narrowing what the host executes without moving the sentence fails here.
    """

    assert _declares_host_cohort_products(
        _Step([COHORT_DEFINITION_COHORT_OUTPUT, COHORT_DEFINITION_FLOW_OUTPUT])
    )


def test_the_recorded_alternative_spelling_is_still_unexecutable():
    """The 64-plan group. Legal, and claimed by nobody -- which is exactly why
    the Planner has to be told."""

    assert not _declares_host_cohort_products(
        _Step(["cohort:analysis_set", COHORT_DEFINITION_FLOW_OUTPUT])
    )


def test_a_third_output_alongside_the_pair_loses_the_executor():
    """The predicate is exact-set equality, so the directive's "no third
    output of any kind" is load-bearing rather than tidiness."""

    assert not _declares_host_cohort_products(
        _Step(
            [
                COHORT_DEFINITION_COHORT_OUTPUT,
                COHORT_DEFINITION_FLOW_OUTPUT,
                "table:cohort_summary",
            ]
        )
    )


def test_the_sentence_is_rendered_from_the_shared_constants():
    sentence = _host_executed_cohort_step_sentence()

    assert COHORT_DEFINITION_COHORT_OUTPUT in sentence
    assert COHORT_DEFINITION_FLOW_OUTPUT in sentence


# ---------------------------------------------------------------------------
# The Planner actually reads it
# ---------------------------------------------------------------------------


def test_the_directive_states_which_declaration_the_host_executes(directive):
    assert _host_executed_cohort_step_sentence() in directive


def test_the_directive_says_a_third_output_costs_the_executor(directive):
    """Anchored on the claim, not on one phrasing of it -- the paragraph was
    rewritten shorter once already and an exact-wording anchor failed a test
    whose property still held."""

    tail = directive[directive.index(_host_executed_cohort_step_sentence()) :][:1400]

    assert "third output" in tail
    assert "code generator" in tail


def test_the_directive_asks_for_the_spec_and_its_two_real_fields(directive):
    assert "cohort_definition_spec" in directive
    marker = "also declare `cohort_definition_spec`"
    tail = directive[directive.index(marker) :][:1200]

    assert "identity_column" in tail
    assert "eligibility_criteria" in tail
    assert "criterion_id" in tail


def test_the_directive_says_when_to_omit_the_spec(directive):
    """A step that really filters rows must keep the generated-code path; a
    directive that only advertised the reward would get the spec on steps the
    host cannot compute."""

    marker = "also declare `cohort_definition_spec`"
    tail = directive[directive.index(marker) :][:1400]

    assert "omits the spec" in tail


# ---------------------------------------------------------------------------
# The shape is shown, not only described
# ---------------------------------------------------------------------------


def test_the_json_example_carries_the_nested_criterion_shape(directive):
    """Prose conveys which choices are the Planner's; it does not convey
    whether a field nests. The distribution spec paid for that lesson once."""

    marker = '"cohort_definition_spec": {'
    assert marker in directive
    block = directive[directive.index(marker) :][:600]

    assert '"identity_column"' in block
    assert '"eligibility_criteria"' in block
    assert '"criterion_id"' in block
    assert '"description"' in block


def test_the_example_step_declares_the_executable_pair(directive):
    marker = '"step_id": "01_define_analysis_cohort"'
    assert marker in directive
    block = directive[directive.index(marker) :][:600]

    assert COHORT_DEFINITION_COHORT_OUTPUT in block
    assert COHORT_DEFINITION_FLOW_OUTPUT in block


def test_the_retry_format_reminder_lists_the_optional_spec():
    """On a structured-response retry the Planner is re-sent an explicit roster
    of the step keys; a field missing from it reads as one it may not send.

    The reminder is a literal inside ``PlannerAgent.run``, so it is read from
    the source rather than by driving a provider failure.
    """

    source = inspect.getsource(PlannerAgent.run)
    marker = "steps (array of objects"
    assert marker in source, "the format reminder moved; re-anchor this test"
    roster = source[source.index(marker) :][:900]

    assert "cohort_definition_spec" in roster


def test_the_retry_format_reminder_publishes_exact_model_term_shape():
    """A rejected model roster must be repairable without guessing aliases."""

    source = inspect.getsource(PlannerAgent.run)
    assert "_payload.planner_science_retry_guide()" in source
    assert "_payload.planner_science_retry_guide()" in inspect.getsource(
        ReplannerAgent.run
    )

    from easyicu.research_agent.planning.primary_result_contract import (
        model_terms_retry_guide,
    )

    reminder = model_terms_retry_guide()
    for required in (
        "`name`",
        "`role`",
        "`coding`",
        "`levels`",
        "`reference_level`",
        "`transform`",
        "treatment_contrast",
        "declared_level_index",
        "exposure_source",
        "covariates",
    ):
        assert required in reminder
    assert "`variable`" in reminder
    assert "`binary_indicator`" in reminder


def test_the_retry_reminder_publishes_primary_contract_applicability():
    reminder = planner_science_retry_guide()

    assert "`causal_inference`" in reminder
    assert "`survival`" in reminder
    assert "`association_study` must omit" in reminder
    assert "`model_requirements`" in reminder
    assert "`know_how_decisions`" in reminder
    assert "never `null`" in reminder


# ---------------------------------------------------------------------------
# A constant the Planner cannot choose is not dictated to it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name",
    [
        name
        for name, field in CohortDefinitionSpec.model_fields.items()
        if not field.is_required()
        and isinstance(field.default, str)
        and "_" in field.default
    ],
)
def test_the_directive_never_dictates_a_constant_it_could_misspell(
    directive, field_name
):
    """Three canonical tasks once produced no plan at all because a directive
    listed single-valued literals as ``field=value`` and the model wrote a
    plausible synonym."""

    only_value = CohortDefinitionSpec.model_fields[field_name].default

    assert f"{field_name}=" not in directive
    assert str(only_value) not in directive


# ---------------------------------------------------------------------------
# A retry must not have to rediscover it
# ---------------------------------------------------------------------------


def test_the_retry_projection_keeps_the_spec():
    payload = {
        "research_question": "q",
        "steps": [
            {
                "step_id": "01_define_analysis_cohort",
                "expected_outputs": [
                    COHORT_DEFINITION_COHORT_OUTPUT,
                    COHORT_DEFINITION_FLOW_OUTPUT,
                ],
                "method": "cohort_definition_and_attrition",
                "cohort_definition_spec": {
                    "identity_column": "stay_id",
                    "eligibility_criteria": [
                        {"criterion_id": "universe", "description": "All ICU stays."}
                    ],
                },
            }
        ],
    }

    projected = _planner_retry_response_projection(json.dumps(payload))

    assert "cohort_definition_spec" in projected
    assert "stay_id" in projected


def test_the_example_plan_still_parses_as_json_with_placeholders(directive):
    """The example is the Planner's template; a comma or brace lost while
    adding a step would teach it a malformed shape."""

    start = directive.index("Required JSON shape")
    block = directive[directive.index("{", start) :]
    depth = 0
    end = None
    for index, char in enumerate(block):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    assert end is not None, "the example object never closed"

    # Placeholders such as <one sentence> are not JSON scalars; quote-wrapped
    # ones already are, and the bare 0.95 / [] entries parse as themselves.
    payload = json.loads(re.sub(r"<[^>\"]*>", "PLACEHOLDER", block[:end]))

    step_ids = [step["step_id"] for step in payload["steps"]]
    assert step_ids[0] == "01_define_analysis_cohort"
    assert payload["steps"][0]["expected_outputs"] == [
        COHORT_DEFINITION_COHORT_OUTPUT,
        COHORT_DEFINITION_FLOW_OUTPUT,
    ]
    assert all("cohort_definition_spec" in step for step in payload["steps"])
