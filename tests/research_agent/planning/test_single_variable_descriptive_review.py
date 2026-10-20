"""Synthetic empirical summaries keep their question and scientific boundary."""

from __future__ import annotations

import pytest

from easyicu.research_agent.execution.runners.cohort_summary_executor import (
    cohort_summary_executor_owns_step,
)
from easyicu.research_agent.planning.capability_registry import assess_scientific_capability
from easyicu.research_agent.planning.figure_plan_shaping import (
    bind_deterministic_figure_panels,
    ensure_primary_result_figure_step,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    study_endpoint_required,
)
from easyicu.research_agent.reporting.scientific_maturity import build_scientific_maturity_audit
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    PlannedModelRequirement,
    ResearchContext,
    VariableRole,
)


def summary_context() -> ResearchContext:
    return ResearchContext(
        research_question="Describe recorded temperature among synthetic ICU stays.",
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="eicu", n_stays=0, id_columns=["stay_id"],
            provenance={
                "analysis_unit": "icu_stay", "evidence_stage": "metadata_only_planning",
                "patient_rows_read": False, "patient_identity_available": False,
                "patient_grouping_status": "available_unbound",
            },
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="temperature", role=VariableRole.VITAL, dtype="float64"),
            ConceptDescriptor(name="event", role=VariableRole.OUTCOME, dtype="int64"),
        ],
    )


def summary_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question=summary_context().research_question,
        analysis_type="descriptive_epidemiology",
        steps=[AnalysisStep(
            step_id="measurement", planned_analysis_role="primary",
            intent="Summarize the observed stay-level measurement without inference.",
            method="empirical_summary_chosen_by_planner",
            scientific_action_id="descriptive.descriptive_summary",
            inputs=["temperature", "stay_id", "artifact:analysis_cohort"],
            expected_outputs=["table:cohort_summary", "table:distribution_summary"],
        )],
    )


def review(context=None, plan=None):
    context, plan = context or summary_context(), plan or summary_plan()
    return build_plan_scientific_review(
        context=context, plan=plan, require_reportable_capability=True,
        figure_strategy=build_article_figure_strategy(context, analysis_family="descriptive"),
    )


def test_measurement_without_outcome_needs_no_endpoint_in_review_or_maturity(tmp_path):
    context, plan = summary_context(), summary_plan()
    assert not study_endpoint_required(context, plan)
    assert "OUTCOME_DEFINITION_UNRESOLVED" not in {f.code for f in review().findings}
    maturity = build_scientific_maturity_audit(context=context, plan=plan, run_dir=tmp_path)
    assert "OUTCOME_DEFINITION_UNRESOLVED" not in {f.code for f in maturity.findings}


@pytest.mark.parametrize("declaration", ["target", "requested", "model", "summarized_outcome"])
def test_descriptive_endpoint_declarations_still_require_a_definition(declaration):
    context, plan = summary_context(), summary_plan()
    if declaration == "target":
        context = context.model_copy(update={"target_outcome": "event"})
    elif declaration == "requested":
        context = context.model_copy(update={
            "cohort": context.cohort.model_copy(update={"requested_outcome_columns": ["event"]}),
        })
    elif declaration == "summarized_outcome":
        plan.steps[0] = plan.steps[0].model_copy(update={
            "inputs": ["event", "artifact:analysis_cohort"],
        })
    else:
        # Typed outcome-bearing steps count even if their method/name never says outcome.
        plan.steps[0] = plan.steps[0].model_copy(update={"model_requirements": [
            PlannedModelRequirement(
                requirement_id="estimate", outcome="event", outcome_type="binary",
                method_family="logistic_regression", exposure_source="temperature",
                analysis_role="primary", analysis_set="complete_case",
            ),
        ]})
        assert study_endpoint_required(context, plan)
        return
    assert study_endpoint_required(context, plan)
    assert "OUTCOME_DEFINITION_UNRESOLVED" in {f.code for f in review(context, plan).findings}


@pytest.mark.parametrize("family", ["prediction_model", "dynamic_prediction", "survival", "association"])
def test_endpoint_families_keep_their_definition_requirement(family):
    plan = summary_plan().model_copy(update={"analysis_type": family})
    assert study_endpoint_required(summary_context(), plan)


def test_primary_summary_is_owned_and_reportable_without_promoting_arbitrary_code():
    context, plan = summary_context(), summary_plan()
    assert cohort_summary_executor_owns_step(plan.steps[0])
    assessment = assess_scientific_capability(
        analysis_type=plan.analysis_type, plan=plan, context=context,
    )
    assert assessment.capability_id == "descriptive_measurement_v1"
    assert assessment.scientific_validator_available
    assert assessment.claim_ceiling == "reportable"
    assert "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE" not in {f.code for f in review().findings}
    untyped = plan.model_copy(update={"steps": [
        plan.steps[0].model_copy(update={"scientific_action_id": None}),
    ]})
    assert assess_scientific_capability(
        analysis_type=plan.analysis_type, plan=untyped, context=context,
    ).claim_ceiling != "reportable"


@pytest.mark.parametrize("change", [
    {"expected_outputs": ["table:cohort_summary"]},
    {"scientific_capability": "association_adjusted_v1"},
    {"planned_analysis_role": "secondary"},
    {"inputs": ["temperature", "artifact:analysis_cohort", "artifact:another_cohort"]},
])
def test_primary_summary_ownership_cannot_be_borrowed_by_another_contract(change):
    assert not cohort_summary_executor_owns_step(summary_plan().steps[0].model_copy(update=change))


def test_primary_summary_needs_a_known_numeric_measurement():
    context, plan = summary_context(), summary_plan()
    unknown = plan.model_copy(update={"steps": [
        plan.steps[0].model_copy(update={"inputs": ["unbound_measurement", "artifact:analysis_cohort"]}),
    ]})
    assert assess_scientific_capability(
        analysis_type=plan.analysis_type, plan=unknown, context=context,
    ).claim_ceiling != "reportable"


def test_distribution_figure_is_a_typed_descendant_of_the_summary():
    plan, _ = ensure_primary_result_figure_step(plan=summary_plan())
    plan, _ = bind_deterministic_figure_panels(plan=plan)
    figures = [s for s in plan.steps if s.figure_panels]
    assert len(figures) == 1
    assert figures[0].inputs == ["table:distribution_summary"]
    assert figures[0].figure_panels[0].article_role == "distribution"
    facts = review(plan=plan).facts["figure_roles"]
    assert "distribution" not in facts["missing_roles"]


def test_empirical_stay_summary_does_not_claim_or_require_patient_independence(tmp_path):
    result = review()
    codes = {f.code for f in result.findings}
    assert "REPEATED_STAY_IDENTITY_UNAVAILABLE" not in codes
    assert "REPEATED_STAY_METHOD_NOT_DECLARED" not in codes
    assert "REPEATED_STAY_DEDUP_UNDECLARED" not in codes
    assert summary_context().cohort.provenance["patient_identity_available"] is False
    maturity = build_scientific_maturity_audit(
        context=summary_context(), plan=summary_plan(), run_dir=tmp_path,
    )
    assert "REPEATED_STAY_DEPENDENCE_UNRESOLVED" not in {f.code for f in maturity.findings}


def test_a_requested_outcome_the_summary_does_not_carry_is_still_a_coverage_gap():
    # A descriptive question that names an outcome keeps the coverage check: a
    # measurement summary does not answer it.
    context = summary_context()
    context = context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"requested_outcome_columns": ["event"]}),
    })
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" in {
        f.code for f in review(context=context).findings
    }


def test_two_measurements_are_not_one_empirical_summary():
    context = summary_context()
    context = context.model_copy(update={"variables": [
        *context.variables,
        ConceptDescriptor(name="heart_rate", role=VariableRole.VITAL, dtype="float64"),
    ]})
    plan = summary_plan()
    plan = plan.model_copy(update={"steps": [plan.steps[0].model_copy(update={
        "inputs": ["temperature", "heart_rate", "stay_id", "artifact:analysis_cohort"],
    })]})
    assert assess_scientific_capability(
        analysis_type=plan.analysis_type, plan=plan, context=context,
    ).claim_ceiling != "reportable"
