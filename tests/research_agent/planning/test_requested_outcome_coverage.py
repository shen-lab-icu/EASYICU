"""Requested endpoints survive family routing without turning summaries into models."""

import pytest

from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    planned_model_outcomes,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ConceptDescriptor,
    VariableRole,
)

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


def _static_prediction_primary(**updates):
    step = AnalysisStep(
        step_id="primary_performance",
        planned_analysis_role="primary",
        intent="Fit the prespecified static model and report held-out performance.",
        inputs=["death", "age", "exposure", "artifact:analysis_cohort"],
        expected_outputs=["table:prediction_scores", "table:model_performance"],
        method="prespecified_prediction_model_discrimination_calibration",
        scientific_action_id="prediction.discrimination_calibration",
    )
    return step.model_copy(update=updates) if updates else step


def test_claimed_static_prediction_primary_covers_its_declared_outcome():
    context = _context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="prediction_model",
        steps=[_static_prediction_primary()],
    )

    assert planned_model_outcomes(plan, context) == ("death",)
    review = build_plan_scientific_review(context=context, plan=plan)
    assert review.facts["missing_model_outcomes"] == []
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" not in {
        finding.code for finding in review.findings
    }


@pytest.mark.parametrize("updates", [
    # The outcome is not part of the declared model-column prefix.
    {"inputs": ["age", "exposure", "artifact:analysis_cohort"]},
    # Supporting columns after the cohort input are not model columns.
    {"inputs": ["age", "exposure", "artifact:analysis_cohort", "death"]},
    # Without the static prediction action the step is not a claimed owner.
    {"scientific_action_id": None},
    {"planned_analysis_role": "secondary"},
])
def test_unclaimed_or_outcome_free_prediction_step_covers_nothing(updates):
    context = _context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="prediction_model",
        steps=[_static_prediction_primary(**updates)],
    )

    assert planned_model_outcomes(plan, context) == ()


def test_static_prediction_primary_does_not_cover_a_second_requested_endpoint():
    context = _multi_outcome_context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="prediction_model",
        steps=[_static_prediction_primary()],
    )

    review = build_plan_scientific_review(context=context, plan=plan)

    assert review.facts["model_covered_outcomes"] == ["death"]
    assert review.facts["missing_model_outcomes"] == ["los_icu"]
