"""The review hands a closing survival-suite proposal to the host, not the researcher.

A survival plan that names the landmark survival suite is executable only once
the suite's signed runtime authority binds its primary step.  Until then the
review publishes the proposal's coordinates; when every one closes from a host
owner the remedy is the host compiling the study's survival design.  Naming the
suite never closes the timing design by itself.  Synthetic study (renal
replacement therapy, 90-day mortality); zero patient rows.
"""

from __future__ import annotations

from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    landmark_survival_suite_facts,
)
from tests.support.survival_proposal import (
    AGE,
    SEX,
    proposed_survival_plan,
    survival_context,
    survival_request,
    survival_spec,
)

_RUNTIME_REF = "scientific_runtime_contract:landmark_survival_suite"


def _post_baseline_context():
    # Exposure assessed over the first 24 h: it can start after ICU admission.
    context = survival_context()
    variables = [
        item.model_copy(update={"analysis_window": "icu_admission[0,24]h"})
        if item.name == "rrt" else item
        for item in context.variables
    ]
    return context.model_copy(update={"variables": variables})


def _proposal(context):
    plan, _llm = proposed_survival_plan(context, survival_spec(survival_request(context), [AGE, SEX]))
    return plan


def _primary(plan):
    return next(step for step in plan.steps if step.planned_analysis_role == "primary")


def _blockers(review):
    return {item.code: item.remediation_route for item in review.findings if item.severity == "blocker"}


def test_a_closing_proposal_is_a_runtime_gap_with_its_published_coordinates():
    context = _post_baseline_context()
    review = build_plan_scientific_review(context=context, plan=_proposal(context))

    assert _blockers(review) == {
        "SURVIVAL_LANDMARK_OWNER_NOT_SEALED": "runtime_capability",
        # Naming the suite does not close the timing design; sealing it does.
        "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED": "runtime_capability",
    }
    assert review.facts["timing_design_executable"] is False
    assert review.facts["landmark_survival_suite"] == {
        "sealed": False,
        "executable": True,
        "exposure": "rrt",
        "event_column": "mort_90d",
        "followup_column": "followup_days_90d",
        "endpoint_horizon_days": 90.0,
        "landmark_hours": 24.0,
        "covariates": ["age", "sex"],
        "covariate_rationales": {"age": AGE["clinical_rationale"], "sex": SEX["clinical_rationale"]},
        "covariate_temporal_roles": {"age": "baseline_static", "sex": "baseline_static"},
    }
    codes = {item.code for item in review.findings}
    # The suite owns the endpoint and its follow-up once the design is compiled.
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" not in codes
    assert "OUTCOME_DEFINITION_UNRESOLVED" not in codes


def test_the_signed_authority_binding_seals_the_suite_and_closes_the_timing():
    context = _post_baseline_context()
    plan = _proposal(context)
    primary = _primary(plan)
    sealed = plan.model_copy(update={"steps": [
        step.model_copy(update={"icu_rule_refs": [*step.icu_rule_refs, _RUNTIME_REF]})
        if step is primary else step
        for step in plan.steps
    ]})

    assert landmark_survival_suite_facts(context, sealed) == {"sealed": True}
    review = build_plan_scientific_review(context=context, plan=sealed)
    assert review.facts["timing_design_executable"] is True
    assert {"SURVIVAL_LANDMARK_OWNER_NOT_SEALED", "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"}.isdisjoint(
        item.code for item in review.findings
    )


def test_a_proposal_whose_coordinates_do_not_close_is_a_plan_revision():
    context = survival_context()
    plan = _proposal(context).model_copy(update={"adjustment_proposal": None})

    facts = landmark_survival_suite_facts(context, plan)
    assert facts["sealed"] is False and facts["executable"] is False
    blockers = _blockers(build_plan_scientific_review(context=context, plan=plan))
    assert blockers["SURVIVAL_LANDMARK_OWNER_NOT_SEALED"] == "agent_plan_revision"
    # Nothing will seal the suite, so it is not credited with the endpoint.
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" in blockers
    assert "OUTCOME_DEFINITION_UNRESOLVED" in blockers

    beyond = context.model_copy(update={"time_windows": []})
    assert landmark_survival_suite_facts(beyond, _proposal(context))["executable"] is False


def test_a_plan_that_does_not_name_the_suite_publishes_no_survival_facts():
    context = survival_context()
    plan = _proposal(context)
    primary = _primary(plan)
    other = plan.model_copy(update={"steps": [
        step.model_copy(update={"method": "cox_proportional_hazards"}) if step is primary else step
        for step in plan.steps
    ]})

    assert landmark_survival_suite_facts(context, other) is None
    assert "landmark_survival_suite" not in build_plan_scientific_review(context=context, plan=other).facts
