"""A question's subgroup is answered by a step that reports the outcome in each of its levels.

"Describe ICU mortality by age group (<65, >=65)": the Planner states the age
grouping as a subgroup requirement, and the descriptive family template has no
subgroup step.  The plan's primary step reports the outcome in each level of
that very grouping, which is what the question asks, so the requirement is
covered by that step.  It used to be a host-verified capability gap that
refused approval of every such plan.  A subgroup the plan does not report by
level, one that names a second grouping too, or one only an auxiliary step
reports stays a gap.

Synthetic contexts and generic wording only.
"""
from __future__ import annotations

from easyicu.research_agent.planning.question_requirements import (
    QuestionRequirement,
    judge_question_requirements,
    question_requirement_findings,
)

from .descriptive_interval_fixtures import descriptive_context, descriptive_plan


def _subgroup(*concepts: str) -> QuestionRequirement:
    return QuestionRequirement.model_validate(
        {
            "id": "r1",
            "kind": "subgroup",
            "quote": "Describe exposure and outcome distributions",
            "concepts": list(concepts),
            "coverage": "plan",
            "gap": None,
            "note": None,
        }
    )


def _judge(requirement: QuestionRequirement, plan=None):
    plan = plan or descriptive_plan(descriptive_context())
    (judged,) = judge_question_requirements(
        [requirement],
        plan=plan,
        family_template=True,
        relatives=lambda column: frozenset({column}),
        check_gap=lambda gap: ("unverifiable", "no typed evidence"),
    )
    return judged


def test_the_grouping_the_primary_step_reports_by_level_answers_the_subgroup() -> None:
    judged = _judge(_subgroup("exposure_flag"))

    assert (judged.disposition, judged.reason_code, judged.owner_step_ids) == (
        "covered",
        "typed_plan_owner_present",
        ("03_distribution",),
    )
    assert judged.verified_by_host is True
    assert question_requirement_findings([judged]) == []


def test_a_subgroup_the_plan_does_not_report_by_level_stays_a_gap() -> None:
    for requirement in (_subgroup("age_years"), _subgroup("exposure_flag", "sex_code")):
        judged = _judge(requirement)

        assert (judged.disposition, judged.reason_code) == (
            "capability_gap",
            "question_subgroup_step_unavailable_in_family_template",
        )
        (stop,) = question_requirement_findings([judged])
        assert stop.detail["approval_allowed"] is False


def test_an_auxiliary_step_reporting_the_grouping_answers_nothing() -> None:
    plan = descriptive_plan(descriptive_context())
    plan = plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"planned_analysis_role": "auxiliary"})
                for step in plan.steps
            ]
        }
    )

    judged = _judge(_subgroup("exposure_flag"), plan)

    assert judged.disposition == "capability_gap"
