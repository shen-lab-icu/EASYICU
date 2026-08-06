"""The auditor sees one step and must not judge the whole plan from it.

MEASURED (h1_ventilation_survival, sweep47_A): step
``04_primary_landmark_survival_analysis`` died ``repair_failed`` on

    "Prevalent exposure is only audited, not excluded before the 24-hour
    follow-up landmark, contradicting the explicit incident-case rule."

The plan assigns that exclusion to a different step. ``01_define_analysis_cohort``
carries, in its own ``icu_rule_refs``:

    "Exclude prevalent events before the 24-hour landmark as a cohort-definition
    step, distinct from leakage and immortal-time auditing."

Step 01 had already run, and its output is step 04's input. The analysis step was
blocked for not re-applying a cohort rule the plan had given to the cohort step.

A second run (``_verify12``) lost ``06_survival_diagnostics`` to the same class.

The mechanism is structural, not a wording accident. ``LLMConceptAuditor.audit``
receives one ``AnalysisStep`` and no roster, so "this obligation was discharged
by another step" and "nobody discharged this obligation" look identical from
where it stands -- and one of those is a defect while the other is the plan
working as designed. The auditor was making a whole-plan judgement from a
step-local view, and the step it faulted was not the owner.

The roster is id/role/method only. The other steps' rule prose would be the
largest block in that prompt and would invite auditing them instead of the one
under review; what closes the gap is not their content but the fact that they
exist and own work.
"""

from __future__ import annotations

import ast
import inspect

from easyicu.research_agent.audits.validators import (
    LLMConceptAuditor,
    _concept_audit_step_roster_block,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)

_ROSTER = (
    {
        "step_id": "01_define_analysis_cohort",
        "planned_analysis_role": "auxiliary",
        "method": "cohort_definition_and_attrition",
    },
    {
        "step_id": "05_complete_case_sensitivity",
        "planned_analysis_role": "sensitivity",
        "method": "sensitivity_replay",
    },
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question=(
            "Estimate the association between mechanical ventilation and 28-day "
            "mortality with time-to-event methods."
        ),
        cohort=CohortDescriptor(
            cohort_name="c", database="miiv", n_patients=1, n_stays=1
        ),
        variables=[],
        target_outcome="death",
    )


def _audited_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="04_primary_landmark_survival_analysis",
        intent="Fit the primary landmark survival model.",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:primary_estimates"],
        method="survival_model",
    )


def _prompt(*, roster: object) -> str:
    return LLMConceptAuditor(llm=None)._prompt(  # type: ignore[arg-type]
        context=_context(),
        script_text="print(1)",
        step=_audited_step(),
        plan_step_roster=roster,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# The block itself.
# --------------------------------------------------------------------------


def test_the_other_steps_are_named_with_their_roles() -> None:
    block = _concept_audit_step_roster_block(_ROSTER)
    assert "01_define_analysis_cohort" in block
    assert "auxiliary" in block
    assert "cohort_definition_and_attrition" in block


def test_the_scope_of_the_audit_is_stated() -> None:
    block = _concept_audit_step_roster_block(_ROSTER)
    assert "auditing ONLY the step named above" in block
    # The rule, not just the scope: whose obligation an unmet requirement is.
    assert "that step's obligation" in block
    # And the condition under which it IS reportable here, so the instruction
    # cannot be read as "never report a missing requirement".
    assert "only if THIS step's own declared contract carries it" in block


def test_the_measured_rule_class_is_named_outright() -> None:
    """Cohort/eligibility/exclusion rules, because that is what was measured.

    A generic "another step may own it" leaves the auditor to decide whether a
    cohort exclusion counts, and on the recorded run it decided it did not.
    """

    block = _concept_audit_step_roster_block(_ROSTER)
    for word in ("cohort", "eligibility", "exclusion", "attrition"):
        assert word in block, word
    # The reason it cannot be this step's job: the owner already ran.
    assert "already run" in block
    assert "Do not require this script to re-apply it" in block


def test_the_roster_carries_no_rule_prose() -> None:
    """Id, role and method only.

    Including the other steps' `icu_rule_refs` would make them the largest block
    in the prompt and invite auditing them. It would also re-create the failure
    in a new shape: the auditor reading a rule it cannot see discharged.
    """

    roster = (
        {
            "step_id": "01_define_analysis_cohort",
            "planned_analysis_role": "auxiliary",
            "method": "cohort_definition_and_attrition",
            "icu_rule_refs": ["Exclude prevalent events before the landmark."],
            "intent": "Construct the landmark cohort.",
        },
    )
    block = _concept_audit_step_roster_block(roster)
    assert "Exclude prevalent events" not in block
    assert "Construct the landmark cohort" not in block


def test_a_roster_entry_without_an_id_is_dropped() -> None:
    block = _concept_audit_step_roster_block(
        ({"planned_analysis_role": "primary", "method": "m"}, *_ROSTER)
    )
    assert block.count("\n- ") == len(_ROSTER)


def test_no_roster_renders_nothing_rather_than_an_empty_claim() -> None:
    """A heading over no steps would assert this plan has one step.

    The sibling ambient-trajectory entry shipped exactly that shape -- a promise
    of completeness over an empty list -- and was reverted.
    """

    assert _concept_audit_step_roster_block(None) == ""
    assert _concept_audit_step_roster_block(()) == ""
    assert _concept_audit_step_roster_block(({"step_id": "  "},)) == ""


# --------------------------------------------------------------------------
# Delivery: the block has to reach the prompt that is actually sent.
# --------------------------------------------------------------------------


def test_the_roster_is_delivered_in_the_prompt() -> None:
    """Asserting over the helper alone does not hold this.

    A mutation deleting the one concatenation line left every helper-level
    assertion green earlier in this session; the defect spans the text and its
    delivery, so the check has to span both.
    """

    prompt = _prompt(roster=_ROSTER)
    assert "auditing ONLY the step named above" in prompt
    assert "01_define_analysis_cohort" in prompt


def test_the_prompt_is_unchanged_when_no_roster_is_supplied() -> None:
    """A single-step plan, or any caller that has no roster, pays nothing."""

    assert "auditing ONLY the step named above" not in _prompt(roster=None)


def test_the_roster_precedes_the_step_contract_it_qualifies() -> None:
    prompt = _prompt(roster=_ROSTER)
    assert prompt.index("auditing ONLY") < prompt.index(
        "Planner-declared step contract:"
    )


# --------------------------------------------------------------------------
# Wiring: the roster must come from the plan, and must exclude this step.
# --------------------------------------------------------------------------


def test_the_execute_phase_builds_the_roster_from_the_plan_without_this_step() -> None:
    """Located structurally.

    Every test above supplies its own roster, so none of them says a run ever
    builds one. Two properties matter and neither is visible from the helper:
    it comes from ``plan_result.plan.steps``, and the audited step is excluded
    -- listing it among "other steps" would tell the auditor its own obligations
    belong to somebody else.
    """

    from easyicu.research_agent.execution import phase

    sources: list[str] = []
    for node in ast.walk(ast.parse(inspect.getsource(phase))):
        if not isinstance(node, ast.keyword) or node.arg != "plan_step_roster":
            continue
        sources.append(ast.unparse(node.value))
    assert len(sources) == 1, sources
    rendered = sources[0]
    assert "plan_result.plan.steps" in rendered, rendered
    assert "step.step_id" in rendered, rendered
    assert "!=" in rendered, rendered
