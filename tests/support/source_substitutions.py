"""Synthetic study and real whole-plan approval fixtures."""
from __future__ import annotations
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from easyicu.databases.research_approval import CapabilityApprovalError, require_approved_database_capabilities
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import StudyRequirements, ThresholdRequirement, assess_study
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences
from easyicu.research_agent.authority.plan_lifecycle import (
    ApprovedExecutablePlan, NormalizedPlan, ProposedPlan, persist_approved_executable_plan,
)
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.databases.research_approval import (
    bind_database_capability_review,
)
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint
from easyicu.research_agent.orchestration.scientific_plan_review_gate import persist_or_validate_scientific_plan_review
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest, HumanReviewDecision
from easyicu.research_agent.planning.scientific_review import PlanScientificReview


def context(database, *, unit="icu_stay", outcome=None, thresholds=()):
    profile = research_capability_profile(database)
    identity = next(item.column for item in profile.units if item.unit == "icu_stay")
    return ResearchContext(research_question="Estimate the association of baseline weight with recorded mortality.",
        primary_exposure="weight", target_outcome=outcome,
        cohort=CohortDescriptor(cohort_name="synthetic", database=database, n_stays=4,
            id_columns=[identity], outcome_columns=["icu_death", "death"]),
        variables=[ConceptDescriptor(name=identity, dtype="int64", role="id"),
            ConceptDescriptor(name="age", dtype="float64", role="demographic", unit="years"),
            ConceptDescriptor(name="weight", dtype="float64", role="demographic", unit="kg"),
            ConceptDescriptor(name="icu_death", dtype="boolean", role="outcome", source_concept="icu_death"),
            ConceptDescriptor(name="death", dtype="boolean", role="outcome", source_concept="death")],
        user_preferences=UserPreferences(data_constraints=json.dumps({
            "analysis_design": {"analysis_unit": unit, "variance_estimator": "model_based"},
            "database_study_requirements": StudyRequirements(family="association_study", analysis_unit=unit,
                outcome="hospital_death" if outcome == "death" else outcome,
                thresholds=thresholds).model_dump(mode="json")})))



def plan(ctx, spec=None):
    cohort = compile_population(spec or PopulationSpec(criteria=[]), ctx).cohort_definition()
    outcome = ctx.target_outcome or "icu_death"
    step = AnalysisStep(step_id="association", intent="Estimate the declared association.",
        method="adjusted_association_models", expected_outputs=["table:adjusted_association_estimates"],
        inputs=["weight", outcome], planned_analysis_role="primary", model_requirements=[{
            "requirement_id": "primary", "outcome": outcome, "outcome_type": "binary",
            "method_family": "logistic_regression", "exposure_source": "weight", "covariates": [],
            "analysis_role": "primary", "analysis_set": "complete_case"}])
    return AnalysisPlan(research_question=ctx.research_question, analysis_type="association_study",
                        cohort=cohort, steps=[step])



def _review(context, plan):
    original = PlanScientificReview(status="ready_for_approval", approval_allowed=True, top_journal_candidate=False,
                                    score=80, dimension_scores={}, context_sha256=canonical_sha256(context.model_dump(mode="json")),
                                    plan_sha256=canonical_sha256(plan.model_dump(mode="json")), literature_sha256="a" * 64,
                                    figure_strategy_sha256="b" * 64, generated_at="2026-10-26T00:00:00Z")
    return bind_database_capability_review(original, context=context, plan=plan)



def _approve(tmp_path, evidence, context, plan, review, *, review_digest=None):
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=review)
    record = evidence.get("scientific_plan_review")
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={record.evidence_id: review_digest or record.sha256})
    payload = {"plan_review_authority": authority.model_dump(mode="json")}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Review the source interpretation.",
                                        payload=payload, authority_sha256=canonical_sha256(payload))
    decision = HumanReviewDecision(review_id=request.review_id, authority_sha256=request.authority_sha256,
                                   decision="approved", reviewer="synthetic-reviewer",
                                   decided_at="2026-10-26T00:00:00Z").model_dump(mode="json")
    checkpoint = HumanReviewCheckpoint.create(
        run_id="synthetic", pipeline_config_sha256="a" * 64, environment_identity={},
        llm_signature_sha256="b" * 64, run_input_capsule_sha256="c" * 64,
        capability_activation_sha256="d" * 64, runtime_capabilities=(), runtime_bundle=None,
        requests=(request,), plan_handoff={"plan": plan.model_dump(mode="json")}, execution_coordinates={},
    ).approved(decisions=(decision,), decision_records=(decision,), decision_sha256=canonical_sha256([decision]))
    write_checkpoint(tmp_path / "human_review_checkpoint.json", checkpoint)
    normalized = NormalizedPlan.create(proposed=ProposedPlan.create(plan=plan, source="synthetic"),
                                       transformation_receipts=(), plan=plan)
    approved = ApprovedExecutablePlan.create(normalized=normalized, plan_review_authority=authority,
                                             decision_set_sha256=canonical_sha256([decision]))
    persist_approved_executable_plan(run_dir=tmp_path, evidence=evidence, approved=approved)
    return SimpleNamespace(context=context, plan=plan, evidence=evidence)
