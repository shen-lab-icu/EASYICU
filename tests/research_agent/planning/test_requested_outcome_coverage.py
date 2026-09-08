"""Requested endpoints survive family routing without turning summaries into models."""

import pytest

from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    planned_model_outcomes,
)
from easyicu.research_agent.schema import AnalysisPlan, ConceptDescriptor, VariableRole

from .scientific_review_fixtures import _absolute_risk_distribution_step, _context


def _multi_outcome_context():
    context = _context()
    return context.model_copy(update={
        "cohort": context.cohort.model_copy(update={
            "requested_outcome_columns": ["death", "los_icu"],
            "outcome_columns": ["death", "los_icu"],
        }),
        "variables": [*context.variables, ConceptDescriptor(
            name="los_icu", role=VariableRole.OUTCOME, dtype="float64",
        )],
    })


@pytest.mark.parametrize("family", [
    "descriptive_epidemiology", "descriptive_study", "prediction_model", "survival",
])
def test_renaming_the_family_cannot_hide_a_requested_endpoint(family):
    context = _multi_outcome_context()
    plan = AnalysisPlan(
        research_question=context.research_question, analysis_type=family,
        steps=[_absolute_risk_distribution_step()],
    )

    review = build_plan_scientific_review(context=context, plan=plan)

    missing = [finding for finding in review.findings
               if finding.code == "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE"]
    assert len(missing) == 1
    assert missing[0].severity == "blocker"
    assert "los_icu" in missing[0].message
    assert not review.approval_allowed


def test_typed_descriptive_endpoint_does_not_require_a_regression():
    context = _context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_epidemiology",
        steps=[_absolute_risk_distribution_step()],
    )

    review = build_plan_scientific_review(context=context, plan=plan)

    assert review.facts["model_covered_outcomes"] == ["death"]
    assert review.facts["missing_model_outcomes"] == []
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" not in {
        finding.code for finding in review.findings
    }


@pytest.mark.parametrize("updates", [
    {"expected_outputs": ["table:unowned_summary"]},
    {"inputs": ["exposure", "death"]},
    {"inputs": ["cohort:analysis_set", "exposure"]},
    {"planned_analysis_role": "auxiliary"},
    {"descriptive_claim": None},
])
def test_loose_inputs_or_an_invalid_distribution_do_not_cover_an_endpoint(updates):
    context = _context()
    step = _absolute_risk_distribution_step().model_copy(update=updates)
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_epidemiology",
        steps=[_absolute_risk_distribution_step()],
    ).model_copy(update={"steps": [step]})

    assert planned_model_outcomes(plan, context) == ()


def test_a_descriptive_table_does_not_replace_the_requested_association_model():
    context = _context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="association_study", steps=[_absolute_risk_distribution_step()],
    )

    review = build_plan_scientific_review(context=context, plan=plan)

    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" in {
        finding.code for finding in review.findings
    }
