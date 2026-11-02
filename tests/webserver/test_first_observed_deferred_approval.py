"""Synthetic durable decisions must bind the deferred plan before source reads."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority, ReviewExecutionAuthority
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision, HumanReviewRequest
from easyicu.research_agent.schema import AnalysisPlan, CohortDescriptor, ResearchContext
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.webserver.first_observed_deferred_execution import verify_approved_deferred_review


def _checkpoint(tmp_path, *, approve, blocked=False):
    plan = AnalysisPlan(research_question="synthetic", steps=[])
    plan_sha = canonical_sha256(plan.model_dump(mode="json"))
    context = ResearchContext(research_question="synthetic", cohort=CohortDescriptor(
        cohort_name="synthetic", database="eicu", n_stays=0), variables=[],
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    context_raw = context.model_dump_json().encode()
    (tmp_path / "research_context.json").write_bytes(context_raw)
    capsule = b'{}'
    (tmp_path / "run_input_capsule.json").write_bytes(capsule)
    capsule_sha = hashlib.sha256(capsule).hexdigest()
    recipe = {"schema_version": "easyicu.first-observed-pending-materialization/1",
              "status": "pending_materialization", "plan_sha256": plan_sha,
              "analysis_authorized": False, "actual_cohort_exists": False,
              "same_skeleton_compilation_verified": True,
              "context_sha256": canonical_sha256(context.model_dump(mode="json"))}
    raw = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    (tmp_path / "first_observed_deferred_recipe.json").write_bytes(raw)
    review = PlanScientificReview(status="ready_for_approval", approval_allowed=True, top_journal_candidate=False,
        score=80, dimension_scores={}, context_sha256=canonical_sha256(context.model_dump(mode="json")), plan_sha256=plan_sha,
        literature_sha256="f" * 64, figure_strategy_sha256="a" * 64, generated_at="2026-01-01T00:00:00Z")
    evidence = EvidenceStore(tmp_path)
    record = evidence.get("scientific_plan_review") or evidence.register_text(
        kind="log", description="Synthetic scientific review", filename="scientific_plan_review.json",
        text=review.model_dump_json(), evidence_id="scientific_plan_review")
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={
        "first_observed_deferred_recipe": hashlib.sha256(raw).hexdigest(), "scientific_plan_review": record.sha256,
        "research_context": hashlib.sha256(context_raw).hexdigest()},
        execution=ReviewExecutionAuthority(pipeline_config_sha256="a" * 64,
            capability_activation_sha256="d" * 64, run_input_capsule_sha256=capsule_sha))
    payload = {"plan_review_authority": authority.model_dump(mode="json")}
    if blocked:
        payload["approval_allowed"] = False
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Synthetic review", payload=payload,
                                        authority_sha256=canonical_sha256(payload))
    decision = HumanReviewDecision(review_id=request.review_id, authority_sha256=request.authority_sha256,
        decision="approved", reviewer="synthetic", decided_at="2026-01-01T00:00:00Z").model_dump(mode="json")
    digest = canonical_sha256([decision])
    checkpoint = HumanReviewCheckpoint.create(run_id=tmp_path.name, pipeline_config_sha256="a" * 64,
        environment_identity={}, llm_signature_sha256="b" * 64, run_input_capsule_sha256=capsule_sha,
        capability_activation_sha256="d" * 64, runtime_capabilities=(), runtime_bundle=None,
        requests=(request,), plan_handoff={}, execution_coordinates={})
    if approve:
        checkpoint = checkpoint.approved(decisions=(decision,), decision_records=(decision,), decision_sha256=digest)
    write_checkpoint(tmp_path / "human_review_checkpoint.json", checkpoint)
    return plan_sha, digest


def test_unapproved_or_changed_plan_cannot_reach_recipe_consumer(tmp_path):
    plan, decision = _checkpoint(tmp_path, approve=False)
    with pytest.raises(ValueError, match="approval_missing"):
        verify_approved_deferred_review(run_dir=tmp_path, approved_plan_sha256=plan, decision_set_sha256=decision)
    plan, decision = _checkpoint(tmp_path, approve=True)
    with pytest.raises(ValueError, match="plan_mismatch"):
        verify_approved_deferred_review(run_dir=tmp_path, approved_plan_sha256="0" * 64, decision_set_sha256=decision)
    assert verify_approved_deferred_review(run_dir=tmp_path, approved_plan_sha256=plan,
                                           decision_set_sha256=decision)["plan_sha256"] == plan
    (tmp_path / "first_observed_deferred_recipe.json").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_approved_deferred_review(run_dir=tmp_path, approved_plan_sha256=plan, decision_set_sha256=decision)


def test_blocked_request_cannot_be_promoted_by_a_constructed_checkpoint(tmp_path):
    plan, decision = _checkpoint(tmp_path, approve=True, blocked=True)
    with pytest.raises(ValueError, match="approval_mismatch"):
        verify_approved_deferred_review(run_dir=tmp_path, approved_plan_sha256=plan, decision_set_sha256=decision)


def test_changed_review_context_is_rejected_before_source_access(tmp_path, monkeypatch):
    from easyicu.research_agent.intake import export_package

    plan, decision = _checkpoint(tmp_path, approve=True)
    monkeypatch.setattr(export_package, 'open_export_package', lambda *_: pytest.fail('source reader was called'))
    (tmp_path / 'research_context.json').write_text('{}')
    with pytest.raises(ValueError, match='context_changed'):
        verify_approved_deferred_review(run_dir=tmp_path, approved_plan_sha256=plan, decision_set_sha256=decision)
