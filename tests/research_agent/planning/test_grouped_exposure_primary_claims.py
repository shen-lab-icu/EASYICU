"""Synthetic metadata and skeletons: each descriptive claim names its own exposure."""

import pytest

from easyicu.research_agent.planning.capability_registry import assess_scientific_capability
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    post_baseline_exposure,
)
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

from .descriptive_interval_fixtures import descriptive_context
from .progressive_planner_fixtures import _payload

_LIMITATION = ("post_baseline_exposure_opportunity_unresolved",)
_BLOCKERS = {
    "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE",
    "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE",
}


def _study(primary="early_group"):
    base = descriptive_context(grouped=True)
    return base.model_copy(update={
        "research_question": (
            "Describe hospital outcome and recovery duration across biomarker groups."
        ),
        "primary_exposure": primary,
        "variables": [
            *base.variables,
            ConceptDescriptor(
                name="early_group", dtype="int64", role=VariableRole.OTHER,
                unit="category", unit_normalization="exposure_group", valid_range=(1, 3),
                analysis_window="icu_admission[0,24]h",
            ),
            ConceptDescriptor(
                name="baseline_group", dtype="int64", role=VariableRole.DEMOGRAPHIC,
                observed_domain={"n_unique": 2, "levels": [0, 1]},
            ),
            ConceptDescriptor(
                name="unwindowed_group", dtype="int64", role=VariableRole.OTHER,
                unit="category", unit_normalization="exposure_group", valid_range=(1, 3),
            ),
            ConceptDescriptor(
                name="recovery_hours", role=VariableRole.OUTCOME, dtype="float64",
                unit="hours", observed_domain={"n_unique": 40, "min": 1, "max": 200},
            ),
        ],
        "cohort": base.cohort.model_copy(update={
            "outcome_columns": ["outcome_flag", "recovery_hours"],
            "requested_outcome_columns": ["outcome_flag", "recovery_hours"],
        }),
    })


def _compile(context, exposure="early_group"):
    payload = _payload()
    cohort, distribution = payload["steps"][0], payload["steps"][2]
    cohort["raw_inputs"] = ["stay_id"]
    distribution.update(
        planned_analysis_role="primary", primary_exposure=exposure,
        raw_inputs=[exposure, "outcome_flag"],
    )
    summary = {
        "step_id": "04_recovery", "planned_analysis_role": "secondary",
        "module_id": "continuous_outcome_summary",
        "objective": "Describe recovery duration within the specified groups.",
        "depends_on": ["01_cohort"], "primary_exposure": exposure,
        "raw_inputs": ["recovery_hours", exposure], "outputs": [],
    }
    payload.update(
        analysis_type="descriptive_epidemiology", robustness_intents=[],
        steps=[cohort, distribution, summary],
    )
    return compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context,
    )[0]


def test_the_bound_primary_distribution_is_reportable_and_covers_both_outcomes():
    context = _study()
    plan = _compile(context)
    distribution, summary = plan.steps[1:]

    assert distribution.descriptive_claim.unresolved_limitations == _LIMITATION
    assert summary.descriptive_claim.unresolved_limitations == _LIMITATION
    assert distribution.descriptive_claim.claim_ceiling == "descriptive_only"
    spec = distribution.exposure_outcome_distribution_spec
    assert spec.interval_method == "wilson"
    assert spec.dependence.group_source == "person_key"
    assessment = assess_scientific_capability(
        analysis_type=plan.analysis_type, context=context, plan=plan,
    )
    assert assessment.capability_id == "descriptive_exposure_outcome_distribution_v1"
    assert assessment.claim_ceiling == "reportable"
    assert assessment.issue_code is None
    review = build_plan_scientific_review(
        context=context, plan=plan, require_reportable_capability=True,
    )
    assert not _BLOCKERS & {finding.code for finding in review.findings}
    assert review.status == "analysis_only"
    assert review.approval_allowed


@pytest.mark.parametrize("primary", [None, "baseline_group"])
def test_each_step_uses_its_own_post_admission_grouping(primary):
    context = _study(primary)
    plan = _compile(context)

    assert post_baseline_exposure(context) == (False, None)
    for step in plan.steps[1:]:
        assert step.descriptive_claim.unresolved_limitations == _LIMITATION
        assert step.descriptive_claim.claim_ceiling == "descriptive_only"


@pytest.mark.parametrize("grouping", ["baseline_group", "unwindowed_group"])
def test_a_baseline_step_does_not_borrow_the_primary_exposures_window(grouping):
    context = _study()
    plan = _compile(context, exposure=grouping)

    assert post_baseline_exposure(context) == (True, "icu_admission[0,24]h")
    distribution, summary = plan.steps[1:]
    assert distribution.descriptive_claim.unresolved_limitations == (
        "descriptive_study_design",
    )
    assert summary.descriptive_claim is None


@pytest.mark.parametrize(
    "primary,expected",
    [(None, (False, None)), ("baseline_group", (False, None)),
     ("early_group", (True, "icu_admission[0,24]h"))],
)
def test_the_default_classifier_keeps_the_context_primary_semantics(primary, expected):
    context = _study(primary)

    assert post_baseline_exposure(context) == expected
    assert post_baseline_exposure(context, None) == expected
    assert post_baseline_exposure(context, "early_group") == (
        True, "icu_admission[0,24]h",
    )
    assert post_baseline_exposure(context, "baseline_group") == (False, None)
    assert post_baseline_exposure(context, "") == (False, None)


@pytest.mark.parametrize("grouping", ["baseline_group", "unwindowed_group"])
def test_baseline_primary_distribution_with_intervals_is_reportable(grouping):
    context = _study(grouping)
    plan = _compile(context, exposure=grouping)
    step = plan.steps[1]
    assert step.exposure_outcome_distribution_spec.interval_method == "wilson"
    assessment = assess_scientific_capability(
        analysis_type=plan.analysis_type, context=context, plan=plan,
    )
    review = build_plan_scientific_review(
        context=context, plan=plan, require_reportable_capability=True,
    )
    blockers = _BLOCKERS & {finding.code for finding in review.findings}
    assert step.descriptive_claim.unresolved_limitations == ("descriptive_study_design",)
    assert assessment.capability_id == "descriptive_exposure_outcome_distribution_v1"
    assert assessment.claim_ceiling == "reportable" and not blockers
