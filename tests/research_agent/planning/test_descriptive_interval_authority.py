"""Synthetic descriptive plans respect one disclosed, source-bound ceiling."""

import json

import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.planning.dependence_authority import (
    DependenceAuthorityError,
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    plan_revision_blocker_codes,
    repeated_unit_design_closed,
)

from .descriptive_interval_fixtures import counts_only_plan, descriptive_context, descriptive_plan
from .progressive_planner_fixtures import _payload


@pytest.mark.parametrize("variance", ["model_based", "cluster_robust", "heteroskedasticity_robust"])
def test_source_ceiling_compiles_counts_and_discloses_the_declared_estimator(variance):
    context = descriptive_context(variance)
    original = context.model_dump(mode="json")
    plan = descriptive_plan(context)
    spec = plan.steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == "none_counts_only"
    assert spec.confidence_level is None
    assert spec.risk_difference_contrast is None
    assert spec.dependence is None
    limitation = spec.interval_limitation
    assert limitation.code == "descriptive_intervals_withheld_without_patient_grouping"
    assert limitation.declared_variance_estimator == variance
    assert limitation.reason == "repeated_stays_possible_without_verified_patient_grouping"
    bound = bind_context_dependence_authority(plan=plan, context=context)
    assert bound == plan
    review = build_plan_scientific_review(context=context, plan=bound)
    assert not [finding for finding in review.findings if finding.severity == "blocker"]
    assert review.approval_allowed
    assert not review.top_journal_candidate
    finding = next(f for f in review.findings if f.descriptive_interval_limitation is not None)
    assert finding.descriptive_interval_limitation == limitation
    assert variance.replace("_", " ") in finding.message
    authority = PlanReviewAuthority.create(plan=bound)
    complete = authority.model_dump(mode="json")["plan_payload"]
    assert complete["steps"][-1]["exposure_outcome_distribution_spec"]["interval_limitation"] == limitation.model_dump(mode="json")
    assert context.model_dump(mode="json") == original


@pytest.mark.parametrize("role", ["primary", "secondary", "sensitivity", "auxiliary"])
@pytest.mark.parametrize("variance", ["model_based", None])
def test_a_revisable_counts_only_plan_uses_the_agent_lane_for_every_role(variance, role):
    context = descriptive_context(variance, grouped=True)
    plan = counts_only_plan(context, role=role)
    bound = bind_context_dependence_authority(plan=plan, context=context)
    review = build_plan_scientific_review(context=context, plan=bound)
    spec = bound.steps[-1].exposure_outcome_distribution_spec
    if spec.interval_method == "none_counts_only":
        conflict = next(f for f in review.findings if f.code == "DESCRIPTIVE_VARIANCE_DESIGN_CONFLICT")
        assert conflict.severity == "blocker"
        assert conflict.remediation_route == "agent_plan_revision"
        assert "restore" not in conflict.remediation.casefold()
        assert "03_distribution" in conflict.message
        assert "DESCRIPTIVE_VARIANCE_DESIGN_CONFLICT" not in plan_revision_blocker_codes(review.findings)
    else:
        assert spec.interval_method == "wilson"
        assert spec.dependence.group_source == "person_key"


@pytest.mark.parametrize("grouped", [False, True])
def test_untyped_design_follows_source_authority(grouped):
    context = descriptive_context(None, grouped=grouped)
    plan = descriptive_plan(context)
    spec = plan.steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == ("wilson" if grouped else "none_counts_only")
    assert getattr(spec, "interval_limitation", None) is None
    if grouped:
        assert spec.dependence is not None
        assert spec.dependence.group_source == "person_key"
    review = build_plan_scientific_review(context=context, plan=plan)
    assert "DESCRIPTIVE_INTERVAL_DEPENDENCE_UNRESOLVED" not in {f.code for f in review.findings}


def test_independent_units_keep_the_existing_interval_design():
    context = descriptive_context(repeats=False)
    plan = descriptive_plan(context)
    spec = plan.steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == "wilson"
    assert spec.confidence_level == 0.95
    assert spec.dependence is None
    assert getattr(spec, "interval_limitation", None) is None
    assert bind_context_dependence_authority(plan=plan, context=context) == plan


@pytest.mark.parametrize("role", ["secondary", "sensitivity", "auxiliary"])
def test_uncompiled_intervals_without_authority_still_require_plan_revision(role):
    context = descriptive_context()
    plan = descriptive_plan(descriptive_context(repeats=False))
    plan = plan.model_copy(update={"steps": [
        plan.steps[0], plan.steps[-1].model_copy(update={"planned_analysis_role": role}),
    ]})
    review = build_plan_scientific_review(context=context, plan=plan)
    finding = next(f for f in review.findings if f.code == "DESCRIPTIVE_INTERVAL_DEPENDENCE_UNRESOLVED")
    assert finding.remediation_route == "agent_plan_revision"


def test_the_legacy_binder_applies_the_same_disclosed_source_ceiling():
    context = descriptive_context()
    plan = descriptive_plan(descriptive_context(repeats=False))
    bound = bind_context_dependence_authority(plan=plan, context=context)
    spec = bound.steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == "none_counts_only"
    assert spec.interval_limitation.declared_variance_estimator == "model_based"
    assert spec.risk_difference_contrast is None


def test_review_refuses_a_counts_only_ceiling_without_its_disclosure():
    context = descriptive_context()
    plan = counts_only_plan(context, role="auxiliary")
    review = build_plan_scientific_review(context=context, plan=plan)
    finding = next(f for f in review.findings if f.code == "DESCRIPTIVE_INTERVAL_LIMITATION_MISSING")
    assert finding.severity == "blocker"
    assert finding.remediation_route == "agent_plan_revision"
    repaired = bind_context_dependence_authority(plan=plan, context=context)
    assert "DESCRIPTIVE_INTERVAL_LIMITATION_MISSING" not in {
        f.code for f in build_plan_scientific_review(context=context, plan=repaired).findings
    }


def test_the_source_ceiling_overrides_a_planner_interval_request():
    context = descriptive_context()
    payload = _payload()
    payload.update(analysis_type="descriptive_epidemiology", robustness_intents=[])
    payload["steps"] = [payload["steps"][0], payload["steps"][2]]
    payload["steps"][-1].update(
        planned_analysis_role="primary", repeated_unit_interval_method="patient_cluster_robust_wald",
    )
    plan, _ = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context,
    )
    spec = plan.steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == "none_counts_only"
    assert spec.repeated_unit_interval_method is None
    assert spec.interval_limitation is not None


@pytest.mark.parametrize("variance", ["model_based", None])
def test_grouped_intervals_close_dependence_without_requiring_a_cluster_design(variance):
    context = descriptive_context(variance, grouped=True)
    plan = descriptive_plan(context)
    assert repeated_unit_design_closed(context, plan)


@pytest.mark.parametrize("variance,cluster_unit", [("cluster_robust", None), ("model_based", "patient")])
def test_descriptive_binding_keeps_the_existing_typed_design_validation(variance, cluster_unit):
    context = descriptive_context(grouped=True)
    context = context.model_copy(update={
        "user_preferences": context.user_preferences.model_copy(update={
            "data_constraints": json.dumps({"analysis_design": {
                "analysis_family": "descriptive_epidemiology", "analysis_unit": "icu_stay",
                "variance_estimator": variance, "cluster_unit": cluster_unit,
            }}),
        }),
    })
    with pytest.raises(DependenceAuthorityError):
        bind_context_dependence_authority(plan=descriptive_plan(context), context=context)
