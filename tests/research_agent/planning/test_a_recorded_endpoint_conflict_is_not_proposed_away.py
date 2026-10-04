"""A recorded endpoint conflict stays the researcher's when a survival suite is proposed.

A survival plan whose proposed suite closes is credited with the requested
time-to-event endpoint: compiling the suite binds the fixed-horizon event and
its paired follow-up, so the review asks the researcher nothing about the
endpoint.  The review gave that credit even when the context builder had
recorded that the question asks for another endpoint (survival to day 28 of
the 90-day endpoint), so the conflict never reached the researcher, and the
compiled suite would have answered another question.  Synthetic study (renal
replacement therapy); zero patient rows.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    landmark_survival_suite_facts,
)
from easyicu.research_agent.research_context.builder import _enrich_target_outcome_descriptor
from tests.support.survival_proposal import (
    AGE,
    SEX,
    proposed_survival_plan,
    survival_context,
    survival_request,
    survival_spec,
)


def _context(question: str):
    context = survival_context(research_question=question)
    # The concept owner's definition of the event, as the dictionary issues it.
    variables = [
        item.model_copy(
            update={"source_concept": "mort_90d", "description": "90-day Mortality"} if item.name == "mort_90d" else {},
            deep=True,
        )
        for item in context.variables
    ]
    _enrich_target_outcome_descriptor(descriptors=variables, research_question=question, target_outcome="mort_90d")
    return context.model_copy(update={"variables": variables})


@pytest.mark.parametrize(("day", "conflict"), [(28, True), (90, False)])
def test_the_proposal_does_not_answer_a_question_about_another_endpoint(day, conflict):
    context = _context(
        "Among adult ICU stays, is renal replacement therapy associated with survival to "
        f"day {day} in a time-respecting survival analysis?"
    )
    plan, _llm = proposed_survival_plan(context, survival_spec(survival_request(context), [AGE, SEX]))
    assert landmark_survival_suite_facts(context, plan)["executable"] is True

    review = build_plan_scientific_review(context=context, plan=plan)

    unresolved = [item for item in review.findings if item.code == "OUTCOME_DEFINITION_UNRESOLVED"]
    assert bool(unresolved) is conflict
    if conflict:
        assert unresolved[0].severity == "blocker"
        assert unresolved[0].requires_user_authorization is True
