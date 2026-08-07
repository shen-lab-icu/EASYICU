"""The step that kills the most runs had no typed declaration at all.

Measured 2026-08-02 over every recorded run: the first analysis step of a plan
was written by the Coder in **127 of 127** runs -- no deterministic owner has
ever claimed one.  It failed in 21 of them, and each failure killed a mean of
5.1 downstream steps: 108 dead steps, **59% of every cascade in the corpus**.

There is no single defect to fix.  The 21 deaths carry ten distinct reasons,
none more than twice, and eight carry no error finding at all.  That is the
signature of a compliance-heavy step being improvised afresh every run.

It is also arithmetic the host can do.  The cohort is materialised and
digest-bound before the run starts; the step reads it, declares it the analysis
set, accounts for the drop from the universe and emits an identity receipt.
Over 196 recorded attrition tables **191 excluded zero rows** -- the multi-row
flows document criteria that removed nobody.  The full-cohort E1 run that
succeeded reported ``n_universe == n_final_analysis_cohort == 94458`` with a
single flow row ``universe,94458,94458,0``.

This is the first of the three parts that made ``exposure_outcome_distribution``
work (typed spec -> Planner directive -> deterministic owner); that product now
runs with zero provider calls and zero repairs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyicu.research_agent.schema import (
    COHORT_DEFINITION_COHORT_OUTPUT,
    COHORT_DEFINITION_FLOW_OUTPUT,
    AnalysisStep,
    CohortDefinitionSpec,
)

_SPEC = {
    "analysis_set": "bound_universe",
    "identity_column": "stay_id",
    "eligibility_criteria": [
        {"criterion_id": "universe", "description": "All prepared ICU stays."}
    ],
}


def _step(**overrides):
    payload = {
        "step_id": "01_define_analysis_cohort",
        "intent": "Define the analytic cohort and report its attrition.",
        "inputs": [],
        "expected_outputs": [
            COHORT_DEFINITION_COHORT_OUTPUT,
            COHORT_DEFINITION_FLOW_OUTPUT,
        ],
        "method": "cohort_definition_and_attrition",
        "cohort_definition_spec": _SPEC,
    }
    payload.update(overrides)
    return AnalysisStep(**payload)


# ---------------------------------------------------------------------------
# The declaration exists and says what it means
# ---------------------------------------------------------------------------


def test_the_step_can_now_declare_what_it_produces():
    step = _step()

    assert step.cohort_definition_spec is not None
    assert step.cohort_definition_spec.analysis_set == "bound_universe"
    assert step.cohort_definition_spec.identity_column == "stay_id"


def test_the_analysis_set_is_a_closed_single_member():
    """The lever that keeps this from claiming steps it cannot compute.

    A study that really excludes rows cannot spell that here, so it omits the
    spec and keeps the generated-code path. Widening this is a capability
    decision; nothing should be able to do it by accident.
    """

    with pytest.raises(ValidationError, match="bound_universe"):
        CohortDefinitionSpec(analysis_set="after_exclusions", identity_column="stay_id")


def test_the_identity_column_is_declared_not_discovered():
    """An executor that picked it from column order would bind the receipt to
    whichever column happened to come first."""

    with pytest.raises(ValidationError):
        CohortDefinitionSpec(analysis_set="bound_universe", identity_column="")


def test_criterion_ids_must_be_unique():
    """The attrition table is keyed by them."""

    with pytest.raises(ValidationError, match="must be unique"):
        CohortDefinitionSpec(
            analysis_set="bound_universe",
            identity_column="stay_id",
            eligibility_criteria=[
                {"criterion_id": "adult", "description": "Age 18 or older."},
                {"criterion_id": "adult", "description": "Duplicate id."},
            ],
        )


def test_the_criteria_list_may_be_empty():
    """86 of 196 recorded flow tables carry exactly one row."""

    spec = CohortDefinitionSpec(
        analysis_set="bound_universe", identity_column="stay_id"
    )

    assert spec.eligibility_criteria == []


def test_the_spec_forbids_unknown_fields():
    """A typo in a host-ownership claim must not pass silently."""

    with pytest.raises(ValidationError):
        CohortDefinitionSpec(
            analysis_set="bound_universe",
            identity_column="stay_id",
            exclusions=[{"column": "age", "min": 18}],
        )


# ---------------------------------------------------------------------------
# It is bound to the products the step actually promises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outputs,missing",
    [
        ([COHORT_DEFINITION_COHORT_OUTPUT], COHORT_DEFINITION_FLOW_OUTPUT),
        ([COHORT_DEFINITION_FLOW_OUTPUT], COHORT_DEFINITION_COHORT_OUTPUT),
        ([], COHORT_DEFINITION_COHORT_OUTPUT),
    ],
)
def test_both_halves_must_be_promised(outputs, missing):
    """A spec on a step promising one half claims ownership of work the step
    never said it would deliver."""

    with pytest.raises(ValidationError, match="requires expected output"):
        _step(expected_outputs=outputs)


def test_a_step_without_the_spec_is_untouched():
    """127 recorded first steps have no spec; none of them may start failing."""

    step = AnalysisStep(
        step_id="01_define_analysis_cohort",
        intent="Define the analytic cohort and report its attrition.",
        inputs=[],
        expected_outputs=["cohort:analysis_set", "table:cohort_flow"],
        method="cohort_definition",
    )

    assert step.cohort_definition_spec is None


def test_the_two_product_names_are_single_spellings():
    """The recorded plans used four names for each half.

    A host owner that had to recognise all eight would be exactly the
    string-set matching this codebase keeps having to delete; the typed
    declaration is what replaces it.
    """

    assert COHORT_DEFINITION_COHORT_OUTPUT == "artifact:analysis_cohort"
    assert COHORT_DEFINITION_FLOW_OUTPUT == "table:cohort_flow"
