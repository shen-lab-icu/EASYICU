"""A substitution is a digest-bound reviewed proposal, never a run fallback."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_lifecycle import (
    ApprovedExecutablePlan, NormalizedPlan, ProposedPlan, persist_approved_executable_plan,
)
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.databases.research_approval import (
    CapabilityApprovalError, bind_database_capability_review, require_approved_database_capabilities,
)
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint
from easyicu.research_agent.orchestration.scientific_plan_review_gate import persist_or_validate_scientific_plan_review
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest, HumanReviewDecision
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.schema import AnalysisPlan, CohortDescriptor, ResearchContext, UserPreferences


def _case(database="hirid"):
    context = ResearchContext(research_question="Summarize the death status.", target_outcome="death",
                              cohort=CohortDescriptor(cohort_name="synthetic", database=database, n_stays=0), variables=[])
    plan = AnalysisPlan(research_question=context.research_question, analysis_type="descriptive_epidemiology", steps=[])
    return context, plan


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


def test_review_carries_the_exact_original_substitute_reason_and_source():
    context, plan = _case()
    review = _review(context, plan)
    companion = review.facts["database_capability_review"]
    assert companion["plan_sha256"] == review.plan_sha256
    record = companion["decision"]["substitutions"][0]
    assert record["original"] == "hospital_death"
    assert record["substitute"] == "icu_death"
    assert record["reason_code"] == "mortality_meaning_differs"
    assert record["source_field"]["column"] == "discharge_status"


def test_execution_without_approval_refuses_before_any_runner(tmp_path):
    context, plan = _case()
    result = SimpleNamespace(context=context, plan=plan, evidence=EvidenceStore(tmp_path))
    with pytest.raises(CapabilityApprovalError) as error:
        require_approved_database_capabilities(result, tmp_path)
    assert error.value.code == "capability_substitution_approval_required"


@pytest.mark.parametrize("database", ["hirid", "aumc"])
def test_exact_reviewed_substitution_can_cross_the_execution_boundary(tmp_path, database):
    context, plan = _case(database)
    result = _approve(tmp_path, EvidenceStore(tmp_path), context, plan, _review(context, plan))
    require_approved_database_capabilities(result, tmp_path)


@pytest.mark.parametrize("mutation", ["omit", "source", "substitute", "original", "reason"])
def test_an_approval_for_a_different_or_absent_substitution_is_refused(tmp_path, mutation):
    context, plan = _case()
    review = _review(context, plan)
    facts = json.loads(json.dumps(review.facts))
    substitutions = facts["database_capability_review"]["decision"]["substitutions"]
    if mutation == "omit":
        substitutions.clear()
    elif mutation == "source":
        substitutions[0]["source_field"]["column"] = "synthetic_wrong_column"
    else:
        field = "reason_code" if mutation == "reason" else mutation
        substitutions[0][field] = "synthetic_changed_value"
    result = _approve(tmp_path, EvidenceStore(tmp_path), context, plan, review.model_copy(update={"facts": facts}))
    with pytest.raises(CapabilityApprovalError) as error:
        require_approved_database_capabilities(result, tmp_path)
    assert error.value.code == "capability_substitution_review_mismatch"


def test_an_exact_study_does_not_acquire_an_extra_approval_rule(tmp_path):
    context, plan = _case("miiv")
    result = SimpleNamespace(context=context, plan=plan, evidence=EvidenceStore(tmp_path))
    require_approved_database_capabilities(result, tmp_path)


@pytest.mark.parametrize("database", ["synthetic", "synthetic_export_source"])
def test_an_unknown_profile_has_no_capability_decision(database):
    from easyicu.databases.research_plan_requirements import plan_capability_decision

    context, plan = _case(database)
    assert plan_capability_decision(context, plan) is None


def test_a_registered_profile_metadata_error_is_not_an_unknown_profile(monkeypatch):
    from easyicu.databases import research_plan_requirements as owner

    def broken_metadata(database):
        raise KeyError("synthetic_missing_metadata_field")

    monkeypatch.setattr(owner, "research_capability_profile", broken_metadata)
    with pytest.raises(KeyError, match="synthetic_missing_metadata_field"):
        owner.plan_capability_decision(*_case("miiv"))


@pytest.mark.parametrize("allowed", [True, False])
def test_an_unknown_profile_keeps_the_existing_review_unchanged(allowed):
    context, plan = _case("synthetic")
    original = _review(*_case("miiv")).model_copy(update={
        "facts": {"synthetic_existing_fact": True}, "approval_allowed": allowed,
        "status": "ready_for_approval" if allowed else "changes_required",
    })
    reviewed = bind_database_capability_review(original, context=context, plan=plan)
    assert reviewed.model_dump() == original.model_dump()
    assert "database_capability_review" not in reviewed.facts


def test_an_unknown_profile_adds_no_execution_approval_requirement(tmp_path):
    context, plan = _case("synthetic")
    result = SimpleNamespace(context=context, plan=plan)
    require_approved_database_capabilities(result, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_unknown_export_concepts_are_carried_by_review_without_refusal(tmp_path):
    context, plan = _case("eicu")
    context = context.model_copy(update={"primary_exposure": "synthetic_assay"})
    review = _review(context, plan)
    decision = review.facts["database_capability_review"]["decision"]
    assert review.approval_allowed
    assert decision["status"] == "exact"
    assert decision["unassessed_concepts"] == ["synthetic_assay"]
    require_approved_database_capabilities(SimpleNamespace(context=context, plan=plan), tmp_path)


@pytest.mark.parametrize("covariates", [None, [], ["d_dimer"]])
def test_capability_projection_keeps_an_undeclared_adjustment_set_undeclared(covariates):
    from easyicu.databases.research_plan_requirements import plan_capability_decision
    from easyicu.research_agent.methods.descriptive_inputs import plan_value_columns

    context, plan = _case("eicu")
    plan = AnalysisPlan.model_validate({**plan.model_dump(mode="json"), "steps": [{
        "step_id": "synthetic_model", "intent": "Estimate the declared association.",
        "inputs": [], "method": "adjusted_association_models",
        "expected_outputs": ["table:adjusted_association_estimates"], "model_requirements": [{
            "requirement_id": "synthetic_association", "outcome": "death", "outcome_type": "binary",
            "method_family": "binary_logistic_regression", "exposure_source": "synthetic_assay",
            "analysis_role": "primary", "analysis_set": "complete_case", "covariates": covariates,
        }],
    }]})
    original = plan.model_dump(mode="json")
    assert plan_value_columns(plan) == ("death", "synthetic_assay", *(covariates or ()))
    decision = plan_capability_decision(context, plan)
    assert decision.status == ("refusal" if covariates else "exact")
    assert decision.unassessed_concepts == ("synthetic_assay",)
    assert plan.model_dump(mode="json") == original
    assert plan.steps[0].model_requirements[0].covariates == covariates


def test_missing_source_capability_cannot_be_overridden_by_approval(tmp_path):
    context, plan = _case("eicu")
    plan = plan.model_copy(update={"analysis_type": "survival"})
    review = _review(context, plan)
    assert review.approval_allowed is False
    assert review.status == "changes_required"
    with pytest.raises(CapabilityApprovalError) as error:
        require_approved_database_capabilities(SimpleNamespace(context=context, plan=plan, evidence=EvidenceStore(tmp_path)), tmp_path)
    assert error.value.code == "study_capability_refused"


def test_execute_owner_checks_substitutions_before_starting_its_services(tmp_path):
    from easyicu.research_agent.execution.phase import _prepare_execute_phase_authority

    context, plan = _case()
    result = SimpleNamespace(context=context, plan=plan, evidence=EvidenceStore(tmp_path))
    pipeline = SimpleNamespace(_execute_phase_services=lambda: pytest.fail("execution started before approval"))
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_approval_required"):
        _prepare_execute_phase_authority(pipeline, plan_result=result, run_dir=tmp_path,
                                         resume_from_step_id=None, stop_after_step_id=None)


@pytest.mark.parametrize("database", ["mimic", "mimic_demo", "miiv", "miiv_demo", "eicu", "eicu_demo", "aumc"])
def test_an_aligned_threshold_runs_only_with_its_original_requirement_in_the_approved_companion(tmp_path, database):
    context, plan = _case(database)
    original, aligned = (65, 60) if database == "aumc" else (95, 90)
    constraints = {"database_study_requirements": {
        "family": "descriptive_epidemiology",
        "thresholds": [{"concept": "age", "operator": ">=", "value": original}],
    }}
    context = context.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    plan = AnalysisPlan.model_validate({**plan.model_dump(mode="json"), "cohort": {
        "name": "synthetic", "inclusion": [{"concept_id": "age", "aggregation": "first", "op": ">=", "value": aligned,
            "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24}}],
    }})
    review = _review(context, plan)
    assert review.approval_allowed
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_approval_required"):
        require_approved_database_capabilities(SimpleNamespace(context=context, plan=plan, evidence=EvidenceStore(tmp_path)), tmp_path)
    result = _approve(tmp_path, EvidenceStore(tmp_path), context, plan, review)
    require_approved_database_capabilities(result, tmp_path)


def test_approval_does_not_execute_an_unaligned_threshold(tmp_path):
    context, plan = _case("aumc")
    plan = AnalysisPlan.model_validate({**plan.model_dump(mode="json"), "cohort": {
        "name": "synthetic", "inclusion": [{"concept_id": "age", "aggregation": "first", "op": ">=", "value": 65,
            "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24}}],
    }})
    assert not _review(context, plan).approval_allowed
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_not_applied"):
        require_approved_database_capabilities(SimpleNamespace(context=context, plan=plan, evidence=EvidenceStore(tmp_path)), tmp_path)


def test_adding_an_aligned_predicate_does_not_hide_a_remaining_unaligned_one(tmp_path):
    context, plan = _case("aumc")
    plan = AnalysisPlan.model_validate({**plan.model_dump(mode="json"), "cohort": {
        "name": "synthetic", "inclusion": [{"concept_id": "age", "aggregation": "first", "op": ">=", "value": value,
            "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24}} for value in (65, 60)],
    }})
    assert not _review(context, plan).approval_allowed
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_not_applied"):
        require_approved_database_capabilities(SimpleNamespace(context=context, plan=plan, evidence=EvidenceStore(tmp_path)), tmp_path)


@pytest.mark.parametrize("database", ["hirid", "sic"])
@pytest.mark.parametrize("effective_unit", ["hospital_admission", "icu_stay"])
def test_unit_substitution_requires_the_effective_design_and_its_approval(tmp_path, database, effective_unit):
    context, plan = _case(database)
    constraints = {"analysis_design": {"analysis_unit": effective_unit}, "database_study_requirements": {
        "family": "descriptive_epidemiology", "analysis_unit": "hospitalization", "outcome": "hospital_death",
    }}
    context = context.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    review = _review(context, plan)
    result = SimpleNamespace(context=context, plan=plan, evidence=EvidenceStore(tmp_path))
    if effective_unit == "hospital_admission":
        assert not review.approval_allowed
        with pytest.raises(CapabilityApprovalError, match="capability_substitution_not_applied"):
            require_approved_database_capabilities(result, tmp_path)
    else:
        assert review.approval_allowed
        with pytest.raises(CapabilityApprovalError, match="capability_substitution_approval_required"):
            require_approved_database_capabilities(result, tmp_path)
        require_approved_database_capabilities(_approve(tmp_path, result.evidence, context, plan, review), tmp_path)


@pytest.mark.parametrize("missing", ["human_review_checkpoint.json", "approved_executable_plan_revision_1.json"])
def test_a_review_record_alone_cannot_replace_existing_approval_authority(tmp_path, missing):
    context, plan = _case()
    result = _approve(tmp_path, EvidenceStore(tmp_path), context, plan, _review(context, plan))
    if missing.startswith("approved_executable_plan"):
        from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path

        verified_run_evidence_path(tmp_path, result.evidence.get("approved_executable_plan_revision_1")).unlink()
    else:
        (tmp_path / missing).unlink()
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_approval_required"):
        require_approved_database_capabilities(result, tmp_path)


def test_approval_must_bind_this_scientific_review_digest(tmp_path):
    context, plan = _case()
    result = _approve(tmp_path, EvidenceStore(tmp_path), context, plan, _review(context, plan), review_digest="d" * 64)
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_approval_required"):
        require_approved_database_capabilities(result, tmp_path)


def test_resume_cannot_reuse_a_review_that_lost_its_substitution(tmp_path):
    from easyicu.research_agent.orchestration.scientific_plan_review_gate import ScientificPlanReviewArtifactError

    context, plan = _case()
    review = _review(context, plan)
    evidence = EvidenceStore(tmp_path)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=review)
    with pytest.raises(ScientificPlanReviewArtifactError, match="binding_drift"):
        persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence,
                                                   current_review=review.model_copy(update={"facts": {}}), reuse_existing_review=True)


def test_the_existing_review_gate_persists_the_companion_before_approval(tmp_path, monkeypatch):
    from easyicu.research_agent.orchestration import scientific_plan_review_gate as gate

    context, plan = _case()
    original = _review(context, plan).model_copy(update={"facts": {}})
    monkeypatch.setattr(gate, "build_plan_scientific_review", lambda **kwargs: original)
    result = gate.prepare_scientific_plan_review_gate(run_dir=tmp_path, evidence=EvidenceStore(tmp_path),
        context=context, plan=plan, literature=None, figure_strategy=None,
        require_reportable_capability=False, reuse_existing_review=False, runtime_authority=None, model_retention=None)
    assert result.review.facts["database_capability_review"]["decision"]["status"] == "substitution"
    saved = PlanScientificReview.model_validate_json(result.artifact_path.read_bytes())
    assert saved.facts == result.review.facts


def test_incomplete_original_requirements_cannot_hide_the_planned_mortality_meaning():
    context, plan = _case()
    constraints = {"database_study_requirements": {"family": "descriptive_epidemiology"}}
    context = context.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    companion = _review(context, plan).facts["database_capability_review"]
    assert companion["decision"]["status"] == "substitution"
    assert companion["decision"]["substitutions"][0]["substitute"] == "icu_death"


def test_a_preserved_threshold_requires_its_replacement_predicate(tmp_path):
    context, plan = _case("miiv")
    constraints = {"database_study_requirements": {"family": "descriptive_epidemiology",
        "thresholds": [{"concept": "age", "operator": ">=", "value": 95}]}}
    context = context.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    assert not _review(context, plan).approval_allowed
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_not_applied"):
        require_approved_database_capabilities(SimpleNamespace(context=context, plan=plan, evidence=EvidenceStore(tmp_path)), tmp_path)


def test_original_requirements_do_not_hide_an_unavailable_effective_unit():
    context, plan = _case("hirid")
    constraints = {"analysis_design": {"analysis_unit": "hospital_admission"},
                   "database_study_requirements": {"family": "descriptive_epidemiology", "analysis_unit": "icu_stay"}}
    context = context.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    assert not _review(context, plan).approval_allowed


def test_the_generic_death_builder_keeps_its_actual_meaning_even_if_another_status_exists(tmp_path):
    context, plan = _case("eicu")
    constraints = {"database_study_requirements": {"family": "descriptive_epidemiology", "outcome": "icu_death"}}
    context = context.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    decision = _review(context, plan).facts["database_capability_review"]["decision"]
    assert decision["status"] == "substitution"
    assert decision["substitutions"][0]["original"] == "icu_death"
    assert decision["substitutions"][0]["substitute"] == "hospital_death"
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_approval_required"):
        require_approved_database_capabilities(SimpleNamespace(context=context, plan=plan, evidence=EvidenceStore(tmp_path)), tmp_path)
