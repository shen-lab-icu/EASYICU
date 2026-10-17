"""Safe source ceilings cannot silently change the reviewed scientific design."""

import json

import pytest

from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import AnalysisPlan

from .scientific_review_fixtures import _absolute_risk_distribution_step, _context


def _study(variance):
    context = _context()
    design = {"analysis_family": "descriptive_epidemiology", "analysis_unit": "icu_stay",
              "variance_estimator": variance}
    if variance == "cluster_robust":
        design["cluster_unit"] = "patient"
    return context.model_copy(update={
        "user_preferences": context.user_preferences.model_copy(update={
            "data_constraints": json.dumps({"analysis_design": design}) if variance else None,
        }),
    })


def _plan(context, *, counts_only=True):
    step = _absolute_risk_distribution_step()
    if counts_only:
        spec = step.exposure_outcome_distribution_spec.model_copy(update={
            "schema_version": "easyicu.exposure_outcome_distribution/3",
            "interval_method": "none_counts_only", "repeated_unit_interval_method": None,
            "confidence_level": None, "dependence": None, "risk_difference_contrast": None,
        })
        step = step.model_copy(update={"exposure_outcome_distribution_spec": spec})
    return AnalysisPlan(research_question=context.research_question,
                        analysis_type="descriptive_epidemiology", steps=[step])


@pytest.mark.parametrize("variance", ["model_based", "cluster_robust", "heteroskedasticity_robust"])
def test_counts_only_is_blocked_when_the_study_declares_uncertainty(variance):
    context = _study(variance)
    plan = _plan(context)
    original_context = context.model_dump()
    original_plan = plan.model_dump()
    review = build_plan_scientific_review(context=context, plan=plan)
    conflicts = [f for f in review.findings if f.code == "DESCRIPTIVE_VARIANCE_DESIGN_CONFLICT"]
    assert len(conflicts) == 1
    assert conflicts[0].severity == "blocker"
    assert variance in conflicts[0].message
    assert not review.approval_allowed
    assert context.model_dump() == original_context
    assert plan.model_dump() == original_plan


@pytest.mark.parametrize("variance, counts_only", [(None, True), ("none_counts_only", True),
                                                  ("model_based", False)])
def test_compatible_or_unstated_design_gets_no_false_conflict(variance, counts_only):
    context = _study(variance)
    review = build_plan_scientific_review(context=context, plan=_plan(context, counts_only=counts_only))
    assert "DESCRIPTIVE_VARIANCE_DESIGN_CONFLICT" not in {f.code for f in review.findings}


@pytest.mark.parametrize("family, role", [("association_study", "primary"),
                                          ("descriptive_epidemiology", "auxiliary")])
def test_a_supporting_count_table_does_not_replace_another_primary_design(family, role):
    context = _study("model_based")
    plan = _plan(context)
    plan = plan.model_copy(update={"analysis_type": family,
                                   "steps": [plan.steps[0].model_copy(update={
                                       "planned_analysis_role": role,
                                   })]})
    review = build_plan_scientific_review(context=context, plan=plan)
    assert "DESCRIPTIVE_VARIANCE_DESIGN_CONFLICT" not in {f.code for f in review.findings}
