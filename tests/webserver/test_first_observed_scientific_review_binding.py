"""A preparation proposal must bind the actual persisted scientific review."""
from types import SimpleNamespace
import hashlib

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.webserver import first_observed_preparation as owner
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding
from tests.support.descriptive_plan import _context

_verified_scientific_review = owner._verified_scientific_review


def _case(tmp_path, *, review_plan=None, review_context=None):
    plan = AnalysisPlan(research_question="first observed ICU", steps=[
        AnalysisStep(step_id="describe", intent="Describe data")])
    context = SimpleNamespace(model_dump=lambda **_: {"question": "first observed ICU"})
    review = PlanScientificReview(
        status="changes_required", approval_allowed=False, top_journal_candidate=False,
        score=0, dimension_scores={}, context_sha256=review_context or canonical_sha256(context.model_dump()),
        plan_sha256=review_plan or canonical_sha256(plan.model_dump(mode="json")), literature_sha256="a" * 64,
        figure_strategy_sha256="b" * 64, generated_at="2026-10-30T00:00:00Z",
    )
    record = EvidenceStore(tmp_path).register_text(
        kind="log", description="scientific review", filename="review.json",
        text=review.model_dump_json(), evidence_id="scientific_plan_review",
    )
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={"scientific_plan_review": record.sha256})
    return authority, context, record


def test_persisted_review_binds_plan_and_context(tmp_path):
    authority, context, record = _case(tmp_path)
    assert _verified_scientific_review(run_dir=tmp_path, authority=authority, context=context) == record.sha256


@pytest.mark.parametrize("field", ["plan", "context"])
def test_review_of_another_plan_or_context_is_rejected(tmp_path, field):
    authority, context, _ = _case(tmp_path, **{f"review_{field}": "c" * 64})
    with pytest.raises(ValueError, match="scientific_review_mismatch"):
        _verified_scientific_review(run_dir=tmp_path, authority=authority, context=context)


def test_replaced_review_is_rejected(tmp_path):
    authority, context, record = _case(tmp_path)
    (tmp_path / record.relative_path).write_text("{}")
    with pytest.raises(ValueError, match="scientific_review_missing"):
        _verified_scientific_review(run_dir=tmp_path, authority=authority, context=context)


def test_preparation_refuses_old_review_before_source_discovery_or_patient_read(tmp_path, monkeypatch):
    context = _context()
    raw = context.model_dump_json().encode()
    (tmp_path / "research_context.json").write_bytes(raw)
    plan = AnalysisPlan(research_question=context.research_question, steps=[
        AnalysisStep(step_id="describe", intent="Describe data")])
    review = PlanScientificReview(
        status="changes_required", approval_allowed=False, top_journal_candidate=False,
        score=0, dimension_scores={}, context_sha256="c" * 64,
        plan_sha256=canonical_sha256(plan.model_dump(mode="json")), literature_sha256="a" * 64,
        figure_strategy_sha256="b" * 64, generated_at="2026-10-30T00:00:00Z",
    )
    record = EvidenceStore(tmp_path).register_text(kind="log", description="review", filename="review.json",
        text=review.model_dump_json(), evidence_id="scientific_plan_review")
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={
        "research_context": hashlib.sha256(raw).hexdigest(), "scientific_plan_review": record.sha256})
    payload = {"plan_review_authority": authority.model_dump(mode="json")}
    request = HumanReviewRequest.create(kind="scientific_stop", summary=owner._SUMMARY,
        payload=payload, authority_sha256=canonical_sha256(payload))
    checkpoint = RawPreparationCheckpoint.pending(plan_sha256=authority.plan_sha256, request=request)
    binding = RawPreparationHostBinding(project_id="project", session_id="session", study_context_id="study",
                                        run_id="run", nonce="e" * 64)
    monkeypatch.setattr(owner, "reviewed_population_spec", lambda **_: {"plan_sha256": authority.plan_sha256,
        "criteria": [{"criterion": {"kind": "first_observed_icu_single_hospital_adult", "role": "include"}}]})
    def forbidden(*args, **kwargs):
        raise AssertionError("source or patient rows accessed before review validation")
    monkeypatch.setattr(owner.sources, "load_registry", forbidden)
    monkeypatch.setattr(owner, "derive_from_verified_eicu_demo", forbidden)
    with pytest.raises(ValueError, match="scientific_review_mismatch"):
        owner.prepare_first_observed_coordinate(run_dir=tmp_path, checkpoint=checkpoint, binding=binding, scope={})
