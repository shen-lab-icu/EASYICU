"""An explicit adjustment needs a primary fitted owner or a typed gap."""

import json

import pytest

from easyicu.research_agent.planning.capability_gap import check_capability_gap
from easyicu.research_agent.planning.progressive_contract import ProgressiveCapabilityGap
from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review, render_agent_plan_revision_contract,
    render_plan_scientific_guardrails,
)

from .scientific_review_fixtures import _context, _plan, _traditional_table_one_step


def _stated_context(*, source_status="available"):
    context = _context()
    return context.model_copy(update={
        "research_question": "Adjust for age when relating the exposure to death.",
        "user_preferences": context.user_preferences.model_copy(update={
            "data_constraints": json.dumps({"stated_adjustments": [{
                "variable": "age", "words": "Adjust for age", "concept": "age",
                "source_status": source_status,
            }]}),
        }),
    })


def _without_adjustment():
    plan = _plan()
    primary = plan.steps[0]
    requirement = primary.model_requirements[0].model_copy(update={
        "covariates": [], "model_terms": primary.model_requirements[0].model_terms[:1],
    })
    return plan.model_copy(update={"steps": [
        primary.model_copy(update={"model_requirements": [requirement]}),
        *plan.steps[1:], _traditional_table_one_step(),
    ]})


def test_an_omitted_adjustment_blocks_and_names_the_automatic_revision():
    review = build_plan_scientific_review(context=_stated_context(), plan=_without_adjustment())
    finding = next(item for item in review.findings if item.code == "STATED_ADJUSTMENT_MISSING")
    assert finding.severity == "blocker"
    assert finding.remediation_route == "agent_plan_revision"
    assert not finding.requires_user_authorization
    assert not review.approval_allowed
    assert "age" in finding.message
    assert "STATED_ADJUSTMENT_MISSING" in render_agent_plan_revision_contract(review)
    row, = review.facts["stated_adjustment_coverage"]["records"]
    assert row["status"] == "missing" and row["owner_step_ids"] == []


def test_a_primary_model_using_the_adjustment_covers_the_requirement():
    review = build_plan_scientific_review(context=_stated_context(), plan=_plan())
    assert not any(item.code.startswith("STATED_ADJUSTMENT_") for item in review.findings)
    row, = review.facts["stated_adjustment_coverage"]["records"]
    assert row["status"] == "covered" and row["owner_step_ids"] == ["primary_model"]


def test_an_unavailable_adjustment_is_an_explicit_unsupported_requirement():
    review = build_plan_scientific_review(
        context=_stated_context(source_status="unavailable"), plan=_without_adjustment(),
    )
    finding = next(item for item in review.findings if item.code == "STATED_ADJUSTMENT_UNAVAILABLE")
    assert finding.remediation_route == "runtime_capability"
    assert not review.approval_allowed and "age" in finding.message
    row, = review.facts["stated_adjustment_coverage"]["records"]
    assert row["status"] == "unsupported"


def test_an_unresolved_adjustment_is_shown_for_review_without_a_revision_loop():
    """No concept, no column to check: the card says so; nothing can repair it."""

    context = _stated_context()
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["stated_adjustments"][0].update(concept=None, source_status="unresolved")
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(
        update={"data_constraints": json.dumps(constraints)},
    )})

    review = build_plan_scientific_review(context=context, plan=_without_adjustment())

    codes = {item.code: item for item in review.findings}
    assert "STATED_ADJUSTMENT_MISSING" not in codes
    finding = codes["STATED_ADJUSTMENT_UNRESOLVED"]
    assert finding.severity == "major"
    assert finding.remediation_route == "independent_review"
    row, = review.facts["stated_adjustment_coverage"]["records"]
    assert row["status"] == "unsupported"
    assert row["reason_code"] == "stated_adjustment_concept_unresolved"


def test_the_planner_is_told_the_required_subset_and_how_to_declare_a_gap():
    guardrails = render_plan_scientific_guardrails(_stated_context())
    assert "required subset" in guardrails
    assert "Adjust for age" in guardrails
    assert "capability_gap='design_element_unsupported' with capability_gap_concept" in guardrails


def test_the_required_subset_allows_additional_confounders():
    plan = _plan()
    primary = plan.steps[0]
    required = primary.model_requirements[0]
    requirement = required.model_copy(update={
        "covariates": ["age", "additional_baseline"],
        "model_terms": [*required.model_terms, ModelTermSpec(
            name="additional_baseline", role="covariate", coding="continuous", transform="identity",
        )],
    })
    plan = plan.model_copy(update={"steps": [primary.model_copy(update={"model_requirements": [requirement]}), *plan.steps[1:]]})
    review = build_plan_scientific_review(context=_stated_context(), plan=plan)
    assert review.facts["stated_adjustment_coverage"]["complete"]


@pytest.mark.parametrize("change", ["secondary_only", "secondary_requirement", "inputs_only", "unfitted_roster", "measurement_companion", "one_primary_only"])
def test_mentions_other_than_every_primary_covariate_do_not_cover_adjustment(change):
    plan = _without_adjustment()
    original = _plan().steps[0]
    if change == "secondary_only":
        plan = plan.model_copy(update={"steps": [
            original.model_copy(update={"planned_analysis_role": "secondary"}), *plan.steps[1:],
        ]})
    elif change in {"secondary_requirement", "unfitted_roster"}:
        update = {"analysis_role": "secondary"} if change == "secondary_requirement" else {"model_terms": []}
        requirement = original.model_requirements[0].model_copy(update=update)
        plan = plan.model_copy(update={"steps": [original.model_copy(update={"model_requirements": [requirement]}), *plan.steps[1:]]})
    elif change == "measurement_companion":
        requirement = original.model_requirements[0].model_copy(update={
            "covariates": ["age_measured"], "model_terms": [
                original.model_requirements[0].model_terms[0],
                ModelTermSpec(name="age_measured", role="covariate", coding="continuous", transform="identity"),
            ],
        })
        plan = plan.model_copy(update={"steps": [original.model_copy(update={"model_requirements": [requirement]}), *plan.steps[1:]]})
    elif change == "one_primary_only":
        plan = plan.model_copy(update={"steps": [*plan.steps, original.model_copy(update={"step_id": "other_primary"})]})
    review = build_plan_scientific_review(context=_stated_context(), plan=plan)
    assert any(item.code == "STATED_ADJUSTMENT_MISSING" for item in review.findings)


@pytest.mark.parametrize("status,concept,expected", [
    ("unavailable", "age", "verified"), ("available", "age", "unverified"),
    ("unavailable", "not_requested", "unverified"),
])
def test_an_availability_gap_is_checked_against_the_required_source(status, concept, expected):
    gap = ProgressiveCapabilityGap(requirement="design_element_unsupported", concept=concept)
    assert check_capability_gap(gap, context=_stated_context(source_status=status)).verification == expected


def test_a_gap_for_a_reading_no_concept_matched_is_refused_with_the_roster_route():
    """An unmatched name is not a source that lacks the variable: plan with its column."""

    context = _stated_context()
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["stated_adjustments"][0].update(concept=None, source_status="unresolved")
    context = context.model_copy(update={"user_preferences": context.user_preferences.model_copy(
        update={"data_constraints": json.dumps(constraints)},
    )})
    check = check_capability_gap(
        ProgressiveCapabilityGap(requirement="design_element_unsupported", concept="age"), context=context,
    )
    assert check.verification == "unverified" and "roster column" in check.fact
    guardrails = render_plan_scientific_guardrails(context)
    assert "'unresolved' matched no source concept" in guardrails and "it is no gap" in guardrails


def test_a_design_element_gap_without_a_concept_stays_unverifiable():
    gap = ProgressiveCapabilityGap(requirement="design_element_unsupported")
    assert check_capability_gap(gap, context=_stated_context()).verification == "unverifiable"


@pytest.mark.parametrize("requirement", ["estimand_unsupported", "multiple_sources_required"])
def test_only_thresholds_and_design_element_gaps_name_a_concept(requirement):
    with pytest.raises(ValueError):
        ProgressiveCapabilityGap(requirement=requirement, concept="age")


def test_omitting_an_available_source_column_cannot_manufacture_a_source_gap():
    context = _stated_context().model_copy(update={"variables": []})
    gap = ProgressiveCapabilityGap(requirement="design_element_unsupported", concept="age")
    assert check_capability_gap(gap, context=context).verification == "unverified"


@pytest.mark.parametrize("end,expected", [(8, "verified"), (2, "unverified")])
def test_a_timing_gap_needs_a_measurement_window_after_typed_time_zero(end, expected):
    context = _stated_context()
    context = context.model_copy(update={
        "variables": [ConceptDescriptor(name="age", role=VariableRole.LAB, dtype="float64", analysis_window=f"icu_admission[0,{end}]h")],
        "user_preferences": context.user_preferences.model_copy(update={"landmark_hours": 4}),
    })
    gap = ProgressiveCapabilityGap(requirement="design_element_unsupported", concept="age")
    assert check_capability_gap(gap, context=context).verification == expected


def test_a_score_value_aggregation_covers_adjustment_but_its_companion_does_not():
    context = _stated_context()
    constraints = {"stated_adjustments": [{
        "variable": "trial score", "words": "controlling for trial score",
        "concept": "trial_score", "source_status": "available",
    }]}
    context = context.model_copy(update={
        "variables": [*context.variables,
                      ConceptDescriptor(name="trial_score_min", role=VariableRole.COMPOSITE_SCORE, dtype="float64"),
                      ConceptDescriptor(name="trial_score_measured", role=VariableRole.OTHER, dtype="bool")],
        "user_preferences": context.user_preferences.model_copy(update={"data_constraints": json.dumps(constraints)}),
    })
    plan = _plan()
    primary = plan.steps[0]
    required = primary.model_requirements[0]
    for column, covered in [("trial_score_min", True), ("trial_score_measured", False)]:
        requirement = required.model_copy(update={
            "covariates": ["age", column], "model_terms": [*required.model_terms, ModelTermSpec(
                name=column, role="covariate", coding="continuous", transform="identity",
            )],
        })
        candidate = plan.model_copy(update={"steps": [primary.model_copy(update={"model_requirements": [requirement]}), *plan.steps[1:]]})
        review = build_plan_scientific_review(context=context, plan=candidate)
        assert review.facts["stated_adjustment_coverage"]["complete"] is covered


def test_a_declared_required_adjustment_gap_records_the_stopped_outline():
    from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from .progressive_planner_fixtures import _context as planner_context, _outline_payload

    context = planner_context().model_copy(update={
        "user_preferences": _stated_context(source_status="unavailable").user_preferences,
    })
    outline = {**_outline_payload(), "steps": [],
               "capability_gap": "design_element_unsupported", "capability_gap_concept": "age"}
    llm = ScriptedMockLLMClient([json.dumps(outline)])
    llm.supports_strict_json_schema = True
    checkpoints = []
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(context, checkpoint_callback=checkpoints.append)
    assert caught.value.reason_code == "progressive_capability_gap"
    assert caught.value.details["findings"][0]["concept"] == "age"
    assert caught.value.details["findings"][0]["verification"] == "verified"
    assert len(checkpoints) == 1
    assert checkpoints[0].prompt_metrics["capability_gap"]["concept"] == "age"
    assert len(llm.calls) == 1

