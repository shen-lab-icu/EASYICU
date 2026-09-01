"""Both halves of one predicate must tolerate the same agent-authored naming.

MEASURED 2026-07-30 over 314 recorded real steps from 64 runs.

``_primary_analysis_cohort_attrition_candidate`` decides two things at once:
whether a step is told the host's canonical cohort-flow contract, and whether
its emitted products reach ``primary_analysis_cohort_integrity_findings`` --
833 lines and roughly forty checks, among them
``attrition_transitions_do_not_conserve``,
``attrition_start_counts_do_not_conserve`` and ``attrition_counts_increase``.

Its method half has always matched on tokens, and says so: "Planner method
labels are agent-authored and may use an equivalent phrase instead of one host
spelling."  Its product half required one of seven exact strings.  Product
names are authored by the same agent under the same freedom, so the asymmetry
decided coverage by spelling.

``table:attrition_flow`` was declared five times and matched none of the seven
(the set held ``attrition``, ``cohort_flow`` and ``eligibility_flow``, but not
that pair of words in that order).  Those five steps were never told the column
contract and never reached one of the forty checks.  One of them shipped this,
recorded ``ok``::

    stage_order  stage                     n_at_start  n_excluded  n_remaining
    2            eligible_analysis_cohort       94458           0        60461

A patient-flow row that does not subtract, in the table that becomes Figure 1.

The fix separates two questions the seven-name set was answering at once.
*Recognition* -- does this step claim to be a cohort construction + attrition
step? -- moves onto tokens beside the method half, so an unfamiliar spelling
still enters the gate. *Acceptance* -- is this a declaration the host will
validate? -- keeps the exact seven, so a step naming its cascade something else
is told to rename it at plan preflight instead of being handed to the Coder and
checked by nobody.

Widening both would have been the wrong repair, and the existing suite said so:
making the seven-name set tokens everywhere turned a plan declaring
``cohort_flow`` beside ``eligibility_attrition`` from an error into a pass, and
that rejection is a deliberate regression test for a plan caught only after
three Coder calls.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.declared_product import (
    _is_primary_analysis_cohort_flow_product as is_flow_product,
    _primary_analysis_cohort_attrition_candidate as is_candidate,
    primary_analysis_cohort_plan_findings,
)
from easyicu.research_agent.schema import AnalysisStep

# The seven spellings the deleted frozenset held. Every one must still match,
# or this became a narrowing instead of a widening.
_HISTORICAL = (
    "attrition",
    "attrition_by_rule",
    "cohort_attrition",
    "cohort_denominator",
    "cohort_denominators",
    "cohort_flow",
    "eligibility_flow",
)

# Declared in real recorded plans and matched by none of the seven.
_MEASURED_MISSES = (
    "attrition_flow",
    "analytic_set_attrition",
    "complete_case_attrition",
    "primary_model_complete_case_attrition_reconciled",
    "strict_complete_case_attrition_from_03b_cohort",
    "final_locked_complete_case_attrition",
    "complete_case_attrition_for_primary_model",
)

# Declared in the same corpus 25 times between them. None is an attrition
# cascade, and subjecting them to the flow contract would demand columns they
# have no reason to carry.
_MEASURED_NON_FLOW = (
    "cohort_summary",
    "cohort_prevalence_incidence",
    "cohort_reconciliation_audit",
    "strict_sofa_model_eligibility_audit",
    "final_locked_sofa_model_eligibility_audit",
)


def _cohort_step(*products: str) -> AnalysisStep:
    return AnalysisStep(
        step_id="01_define_analysis_cohort",
        intent="Build the analysis cohort and report its attrition.",
        inputs=["artifact:universe"],
        expected_outputs=["artifact:analysis_cohort", *products],
        method="cohort_definition_and_attrition",
        table_one_spec=None,
    )


@pytest.mark.parametrize("name", _HISTORICAL)
def test_every_previously_listed_spelling_still_matches(name):
    assert is_flow_product(name)


@pytest.mark.parametrize("name", _MEASURED_MISSES)
def test_the_measured_misses_now_match(name):
    assert is_flow_product(name)


@pytest.mark.parametrize("name", _MEASURED_NON_FLOW)
def test_a_summary_or_audit_is_not_an_attrition_cascade(name):
    assert not is_flow_product(name)


def test_flow_alone_is_not_enough():
    """``flow`` needs a population word; a flow of something else is not this."""

    assert not is_flow_product("data_flow")
    assert not is_flow_product("patient_state_flow")
    assert is_flow_product("cohort_flow")
    assert is_flow_product("analysis_flow")


def test_an_empty_or_unnamed_product_does_not_match():
    for value in ("", "   ", None):
        assert not is_flow_product(value)


# ---------------------------------------------------------------------------
# The whole predicate, which is what actually gates the contract
# ---------------------------------------------------------------------------


def test_the_step_that_escaped_is_now_a_candidate():
    """The exact declaration of the run that shipped a non-subtracting table."""

    assert is_candidate(_cohort_step("table:attrition_flow"))


def test_the_previously_covered_declaration_is_unchanged():
    assert is_candidate(_cohort_step("table:cohort_flow"))


def test_the_method_half_still_gates_it():
    """Widening the product half must not let an unrelated step in.

    A robustness rider that happens to emit an attrition table is excluded by
    the method half, which has always refused those tokens.
    """

    step = _cohort_step("table:attrition_flow")
    step.method = "sensitivity_cohort_definition"

    assert not is_candidate(step)


def test_a_step_with_no_flow_product_is_still_not_a_candidate():
    assert not is_candidate(_cohort_step("table:cohort_summary"))


# ---------------------------------------------------------------------------
# Entering the gate is not passing it
# ---------------------------------------------------------------------------


def _plan(step: AnalysisStep):
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        ConceptPredicate,
        TimeWindow,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    window = TimeWindow(anchor="icu_admit", start_offset_hours=0, end_offset_hours=24)
    return AnalysisPlan(
        research_question="Describe a locked ICU analysis cohort.",
        cohort=CohortDefinition(
            name="primary",
            inclusion=(
                ConceptPredicate(
                    concept_id="age",
                    time_window=window,
                    aggregation="first",
                    op=">=",
                    value=18,
                ),
            ),
        ),
        steps=[step],
    )


def test_the_escaped_declaration_is_now_refused_before_the_first_coder_call():
    """The measured outcome, and the point of the whole change.

    Verified against the recorded plan of the run that shipped the
    non-subtracting table: zero preflight findings at the previous commit, one
    error now.  Recognition decides whether a step enters the gate; the host's
    canonical vocabulary still decides whether it passes, so the Planner is
    told to rename its product instead of being handed the Coder and checked by
    nobody.
    """

    findings = primary_analysis_cohort_plan_findings(
        plan=_plan(_cohort_step("table:attrition_flow"))
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["issue"] == "primary_cohort_product_owner_ambiguous"


def test_the_canonical_declaration_still_passes_preflight():
    assert (
        primary_analysis_cohort_plan_findings(
            plan=_plan(_cohort_step("table:cohort_flow", "table:cohort_attrition"))
        )
        == []
    )
