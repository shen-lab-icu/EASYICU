"""A named subgroup requires a subgroup-capable analysis, never only Table 1."""

import json

import pytest

from easyicu.research_agent.planning.capability_gap import check_capability_gap
from easyicu.research_agent.planning.progressive_contract import ProgressiveCapabilityGap
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review, render_agent_plan_revision_contract,
    render_plan_scientific_guardrails,
)
from easyicu.research_agent.schema import UserPreferences
from .scientific_review_fixtures import _context, _plan, _traditional_table_one_step


def _stated(context=None, *, status="available", concept="age"):
    context = context or _context()
    preferences = context.user_preferences or UserPreferences()
    constraints = json.loads(preferences.data_constraints or "{}")
    constraints["stated_subgroups"] = [{
        "variable": "age", "words": "within age subgroups", "concept": concept,
        "source_status": status,
    }]
    return context.model_copy(update={
        "user_preferences": preferences.model_copy(update={
            "data_constraints": json.dumps(constraints),
        }),
    })


def _grouped_only():
    table = _traditional_table_one_step()
    table = table.model_copy(update={"table_one_spec": table.table_one_spec.model_copy(
        update={"group_by": "age"},
    )})
    return _plan().model_copy(update={"steps": [*_plan().steps, table]})


def test_table_one_grouping_does_not_answer_a_stated_subgroup():
    review = build_plan_scientific_review(context=_stated(), plan=_grouped_only())
    finding = next(item for item in review.findings if item.code == "STATED_SUBGROUP_MISSING")
    assert finding.severity == "blocker" and not review.approval_allowed
    assert finding.remediation_route == "agent_plan_revision"
    assert not finding.requires_user_authorization and "age" in finding.message
    assert "STATED_SUBGROUP_MISSING" in render_agent_plan_revision_contract(review)
    row, = review.facts["stated_subgroup_coverage"]["records"]
    assert row["status"] == "missing" and row["owner_step_ids"] == []


@pytest.mark.parametrize("inputs,covered", [(["age"], True), (["age_measured"], False), (["exposure"], False)])
def test_effect_modification_must_read_the_subgroup(inputs, covered):
    plan = _grouped_only()
    step = plan.steps[0].model_copy(update={
        "step_id": "subgroup_estimates", "planned_analysis_role": "secondary",
        "scientific_action_id": "association.effect_modification", "inputs": inputs,
    })
    plan = plan.model_copy(update={"steps": [*plan.steps, step]})
    review = build_plan_scientific_review(context=_stated(), plan=plan)
    assert review.facts["stated_subgroup_coverage"]["complete"] is covered
    assert any(item.code.startswith("STATED_SUBGROUP_") for item in review.findings) is not covered


@pytest.mark.parametrize("role,covered", [("demographic", True), ("meta", False), ("time", False)])
def test_a_derived_subgroup_value_counts_but_metadata_companions_do_not(role, covered):
    from easyicu.research_agent.schema import ConceptDescriptor

    context = _stated()
    context = context.model_copy(update={"variables": [*context.variables, ConceptDescriptor(
        name="derived_group", source_concept="age", role=role, dtype="int64",
    )]})
    plan = _grouped_only()
    step = plan.steps[0].model_copy(update={
        "step_id": "subgroup_estimates", "planned_analysis_role": "secondary",
        "scientific_action_id": "association.effect_modification", "inputs": ["derived_group"],
    })
    review = build_plan_scientific_review(context=context, plan=plan.model_copy(update={"steps": [*plan.steps, step]}))
    assert review.facts["stated_subgroup_coverage"]["complete"] is covered


def test_a_source_that_lacks_the_subgroup_stops_definitely():
    review = build_plan_scientific_review(context=_stated(status="unavailable"), plan=_grouped_only())
    finding = next(item for item in review.findings if item.code == "STATED_SUBGROUP_UNAVAILABLE")
    assert finding.severity == "blocker" and not review.approval_allowed
    assert finding.remediation_route == "runtime_capability"
    assert review.facts["stated_subgroup_coverage"]["records"][0]["status"] == "unsupported"


@pytest.mark.parametrize("status,concept", [("unresolved", None), ("unbound", "age")])
def test_an_unmatched_subgroup_is_shown_for_review_without_a_revision_loop(status, concept):
    """No concept to check: the card says so, as for an adjustment; nothing can repair it."""

    review = build_plan_scientific_review(context=_stated(status=status, concept=concept), plan=_grouped_only())
    codes = {item.code: item for item in review.findings}
    assert "STATED_SUBGROUP_MISSING" not in codes
    finding = codes["STATED_SUBGROUP_UNRESOLVED"]
    assert finding.severity == "major" and finding.remediation_route == "independent_review"
    row, = review.facts["stated_subgroup_coverage"]["records"]
    assert row["status"] == "unsupported" and row["reason_code"] == "stated_subgroup_concept_unresolved"


def test_a_family_with_neither_route_has_a_definite_capability_stop():
    plan = _grouped_only().model_copy(update={"analysis_type": "causal_inference"})
    review = build_plan_scientific_review(context=_stated(), plan=plan)
    finding = next(item for item in review.findings if item.code == "STATED_SUBGROUP_UNSUPPORTED")
    assert finding.remediation_route == "runtime_capability"
    assert finding.severity == "blocker" and not review.approval_allowed


@pytest.mark.parametrize("concept,status", [("exposure_flag", "covered"), ("sex_code", "missing")])
def test_a_descriptive_plan_answers_a_subgroup_with_its_result_in_each_level(concept, status):
    """"Describe the outcome in each group" is a descriptive answer, not a stop."""

    from easyicu.research_agent.planning.requirement_coverage import stated_subgroup_coverage
    from .descriptive_interval_fixtures import descriptive_context, descriptive_plan

    plan = descriptive_plan(descriptive_context())
    row, = stated_subgroup_coverage(_stated(descriptive_context(), concept=concept), plan).records
    assert row.status == status
    assert row.owner_step_ids == ((plan.steps[-1].step_id,) if status == "covered" else ())


def test_a_static_prediction_plan_has_no_route_to_a_subgroup_owner():
    from easyicu.research_agent.contracts.prediction_execution import PREDICTION_PRIMARY_ACTION
    from easyicu.research_agent.planning.requirement_coverage import stated_subgroup_coverage

    plan = _grouped_only()
    primary = plan.steps[0].model_copy(update={"scientific_action_id": PREDICTION_PRIMARY_ACTION})
    plan = plan.model_copy(update={"analysis_type": "prediction_model", "steps": [primary, *plan.steps[1:]]})
    row, = stated_subgroup_coverage(_stated(), plan).records
    assert (row.status, row.reason_code) == ("unsupported", "stated_subgroup_owner_unavailable")
    fairness = primary.model_copy(update={
        "step_id": "subgroup_fairness", "planned_analysis_role": "secondary",
        "scientific_action_id": "prediction.subgroup_fairness", "inputs": ["age"],
    })
    row, = stated_subgroup_coverage(_stated(), plan.model_copy(update={"steps": [*plan.steps, fairness]})).records
    assert row.status == "covered"


def test_the_planner_gets_the_requirement_and_the_existing_gap_shape():
    text = render_plan_scientific_guardrails(_stated())
    assert "STATED SUBGROUPS" in text and "within age subgroups" in text
    assert "association.effect_modification" in text
    assert "capability_gap='design_element_unsupported'" in text
    assert "capability_gap_concept" in text


@pytest.mark.parametrize("status,concept,expected", [
    ("available", "age", "unverified"), ("unavailable", "age", "verified"),
    ("available", "other", "unverified"), ("unbound", "age", "unverifiable"),
    ("unavailable", "other", "unverified"),
])
def test_a_declared_gap_is_checked_against_the_stated_subgroup(status, concept, expected):
    gap = ProgressiveCapabilityGap(requirement="design_element_unsupported", concept=concept)
    assert check_capability_gap(gap, context=_stated(status=status)).verification == expected


def test_a_subgroup_gap_for_a_reading_no_concept_matched_is_refused():
    gap = ProgressiveCapabilityGap(requirement="design_element_unsupported", concept="age")
    check = check_capability_gap(gap, context=_stated(status="unresolved", concept=None))
    assert check.verification == "unverified" and "roster column" in check.fact


@pytest.mark.parametrize("family,expected", [("causal_inference", "verified"), ("descriptive_epidemiology", "unverified")])
def test_a_declared_family_gap_is_verified_against_its_owners(family, expected):
    context = _stated()
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(
        update={"inferred_analysis_family": family},
    )})
    gap = ProgressiveCapabilityGap(requirement="design_element_unsupported", concept="age")
    check = check_capability_gap(gap, context=context)
    assert check.verification == expected
    assert ("no subgroup owner" in check.fact) is (expected == "verified")


@pytest.mark.parametrize("family,status", [("association_study", "unavailable")])
def test_a_declared_subgroup_gap_stops_planning_without_a_checkpoint(family, status):
    from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from .progressive_planner_fixtures import _context as planner_context

    context = _stated(planner_context(), status=status, concept="sex_code")
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(
        update={"inferred_analysis_family": family},
    )})
    outline = {"analysis_type": family, "cohort_objective": "Keep the declared population.",
               "rationale": "The stated subgroup cannot be analysed by this source and family.",
               "steps": [], "capability_gap": "design_element_unsupported", "capability_gap_concept": "sex_code"}
    llm = ScriptedMockLLMClient([json.dumps(outline)])
    checkpoints = []
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(context, checkpoint_callback=checkpoints.append)
    assert caught.value.reason_code == "progressive_capability_gap"
    finding, = caught.value.details["findings"]
    assert finding["concept"] == "sex_code" and finding["verification"] == "verified"
    assert not checkpoints


def test_a_revision_with_an_accepted_subgroup_table_reaches_the_outline_planner():
    from easyicu.research_agent.agents.family_spec_planner import family_spec_fallback_reason
    from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
    from easyicu.research_agent.planning.baseline_requirements import baseline_requirement_coverage
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure
    from .family_spec_fixtures import _context as family_context
    from easyicu.research_agent.planning.baseline_requirements import bind_baseline_requirements

    context = bind_baseline_requirements(_stated(family_context()), {
        "source_plan_sha256": "c" * 64,
        "tables": [{"source_step_id": "baseline_context",
                    "group_by": {"name": "age"}, "variables": [{"name": "sex"}]}],
    })
    assert family_spec_fallback_reason(
        context, analysis_types=("association_study",), resume_checkpoint=None, stop_after_outline=False,
    ) == "family_template_stated_subgroup_requires_outline"
    # The revision reaches the outline Planner with the accepted table, where a
    # gap claimed for a subgroup the family and source can answer is refused.
    outline = json.dumps({
        "analysis_type": "association_study", "cohort_objective": "Keep the declared population.",
        "rationale": "Claims the stated subgroup cannot be analysed.", "steps": [],
        "capability_gap": "design_element_unsupported", "capability_gap_concept": "age",
    })
    llm = ScriptedMockLLMClient([outline] * 6)
    with pytest.raises(StructuredResponseFailure):
        ProgressivePlannerAgent(llm).run(
            context, planner_strategy="family_spec_v1",
            planning_contract_context=render_plan_scientific_guardrails(context),
        )
    assert any("ACCEPTED BASELINE CONTENT" in message.content for message in llm.calls[0][0])
    assert any("progressive_capability_gap_claim_unverified" in message.content for message in llm.calls[1][0])
    # The fallback does not relax the accepted table: an exposure-grouped
    # substitute is still incomplete, while the original grouping is retained.
    table = _traditional_table_one_step()
    spec = table.table_one_spec.model_copy(update={"variables": [
        table.table_one_spec.variables[0].model_copy(update={"name": "sex"}),
    ]})
    for grouping, expected in [("injury_stage", "incomplete"), ("age", "complete")]:
        candidate = _plan().model_copy(update={"steps": [table.model_copy(update={
            "table_one_spec": spec.model_copy(update={"group_by": grouping}),
        })]})
        assert baseline_requirement_coverage(context, candidate)["status"] == expected


def test_an_available_subgroup_yields_before_the_template_grouping_check():
    from easyicu.research_agent.agents.family_spec_planner import family_spec_fallback_reason
    from easyicu.research_agent.planning.baseline_requirements import bind_baseline_requirements
    from .family_spec_fixtures import _context as family_context

    context = bind_baseline_requirements(_stated(family_context()), {
        "source_plan_sha256": "d" * 64,
        "tables": [{"source_step_id": "baseline_context",
                    "group_by": {"name": "age"}, "variables": [{"name": "sex"}]}],
    })
    assert family_spec_fallback_reason(
        context, analysis_types=("association_study",), resume_checkpoint=None, stop_after_outline=False,
    ) == "family_template_stated_subgroup_requires_outline"


def test_a_subgroup_action_from_another_family_cannot_cover_the_requirement():
    plan = _grouped_only()
    step = plan.steps[0].model_copy(update={
        "step_id": "wrong_family", "scientific_action_id": "prediction.subgroup_fairness",
        "planned_analysis_role": "secondary", "inputs": ["age"],
    })
    review = build_plan_scientific_review(context=_stated(), plan=plan.model_copy(update={"steps": [*plan.steps, step]}))
    assert not review.facts["stated_subgroup_coverage"]["complete"]


def test_an_auxiliary_step_cannot_answer_a_stated_subgroup():
    plan = _grouped_only()
    step = plan.steps[0].model_copy(update={
        "step_id": "auxiliary", "scientific_action_id": "association.effect_modification",
        "planned_analysis_role": "auxiliary", "inputs": ["age"],
    })
    review = build_plan_scientific_review(context=_stated(), plan=plan.model_copy(update={"steps": [*plan.steps, step]}))
    assert not review.facts["stated_subgroup_coverage"]["complete"]


def test_a_table_cannot_claim_coverage_by_borrowing_a_subgroup_action_label():
    plan = _grouped_only()
    table = plan.steps[-1].model_copy(update={
        "scientific_action_id": "association.effect_modification",
        "planned_analysis_role": "secondary",
    })
    review = build_plan_scientific_review(context=_stated(), plan=plan.model_copy(update={"steps": [table]}))
    assert not review.facts["stated_subgroup_coverage"]["complete"]


def test_a_subgroup_does_not_hide_an_adjustment_timing_gap_for_the_same_variable():
    from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
    context = _stated()
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["stated_adjustments"] = constraints["stated_subgroups"]
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(
        update={"data_constraints": json.dumps(constraints)},
    )})
    context = context.model_copy(update={
        "variables": [ConceptDescriptor(name="age", role=VariableRole.LAB, dtype="float64", analysis_window="icu_admission[0,8]h")],
        "user_preferences": context.user_preferences.model_copy(update={"landmark_hours": 4}),
    })
    gap = ProgressiveCapabilityGap(requirement="design_element_unsupported", concept="age")
    assert check_capability_gap(gap, context=context).verification == "verified"


@pytest.mark.parametrize("concept,yields", [("phenotype", False), ("sex", True)])
def test_the_descriptive_template_keeps_a_subgroup_its_own_grouping_answers(concept, yields):
    from easyicu.research_agent.agents.family_spec_planner import family_spec_fallback_reason
    from .family_spec_fixtures import _descriptive_context

    reason = family_spec_fallback_reason(
        _stated(_descriptive_context(), concept=concept), analysis_types=("descriptive_epidemiology",),
        resume_checkpoint=None, stop_after_outline=False,
    )
    assert (reason == "family_template_stated_subgroup_requires_outline") is yields


def test_only_the_template_plans_a_static_prediction_so_it_keeps_the_subgroup():
    from easyicu.research_agent.agents.family_spec_planner import family_spec_fallback_reason
    from .family_spec_fixtures import _prediction_context

    reason = family_spec_fallback_reason(
        _stated(_prediction_context(), concept="sex"), analysis_types=("prediction_model",),
        resume_checkpoint=None, stop_after_outline=False,
    )
    assert reason != "family_template_stated_subgroup_requires_outline"
