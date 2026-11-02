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


def _execution_bound_proposal(tmp_path, capsule_mode="valid"):
    import json
    from easyicu.research_agent.authority.plan_review import ReviewExecutionAuthority
    from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint

    run = tmp_path / "run_synthetic"
    run.mkdir()
    source = {"database": "eicu_demo", "registration_id": "synthetic"}
    constraints = json.dumps({"source_identity": source})
    context = {"research_question": "Synthetic question", "cohort": {"database": "eicu_demo",
        "provenance": {"source_identity": source}}, "user_preferences": {"data_constraints": constraints}}
    context_bytes = json.dumps(context).encode()
    (run / "research_context.json").write_bytes(context_bytes)
    context_sha = hashlib.sha256(context_bytes).hexdigest()
    from easyicu.research_agent.authority.run_input import RunInputCapsule
    identity = {"question": "Synthetic question", "database": "eicu_demo", "user_preferences": {"data_constraints": constraints}}
    if capsule_mode == "wrong_question":
        identity["question"] = "Other question"
    if capsule_mode == "wrong_database":
        identity["database"] = "miiv"
    if capsule_mode == "wrong_source":
        identity["user_preferences"]["data_constraints"] = json.dumps({"source_identity": {**source, "registration_id": "other"}})
    capsule = RunInputCapsule(scientific_identity=identity,
        scientific_identity_sha256="0" * 64 if capsule_mode == "identity" else canonical_sha256(identity),
        context_sha256=context_sha, cohort_sha256="f" * 64, initial_environment={}).model_dump_json().encode()
    if capsule_mode == "malformed":
        capsule = json.dumps({"context_sha256": context_sha}).encode()
    (run / "run_input_capsule.json").write_bytes(capsule)
    if capsule_mode != "unregistered":
        EvidenceStore(run).register_text(kind="log", description="capsule", filename="capsule.json",
            text=capsule.decode(), evidence_id=("run_input_capsule_new" if capsule_mode == "alias" else "run_input_capsule"))
    execution = ReviewExecutionAuthority(pipeline_config_sha256="1" * 64,
        capability_activation_sha256="2" * 64, run_input_capsule_sha256=hashlib.sha256(capsule).hexdigest())
    plan = AnalysisPlan(research_question="Synthetic question", steps=[AnalysisStep(step_id="describe", intent="Describe")])
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={"research_context": context_sha}, execution=execution)
    payload = {"plan_review_authority": authority.model_dump(mode="json"), "approval_allowed": False}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Population requires preparation",
        payload=payload, authority_sha256=canonical_sha256(payload))
    checkpoint = HumanReviewCheckpoint.create(run_id=run.name, pipeline_config_sha256="1" * 64,
        environment_identity={}, llm_signature_sha256="3" * 64, run_input_capsule_sha256=execution.run_input_capsule_sha256,
        capability_activation_sha256="2" * 64, runtime_capabilities=(), runtime_bundle=None,
        requests=(request,), plan_handoff={}, execution_coordinates={})
    write_checkpoint(run / "human_review_checkpoint.json", checkpoint)
    return run, authority, checkpoint


def test_execution_bound_unapproved_plan_can_propose_preparation(tmp_path):
    run, authority, checkpoint = _execution_bound_proposal(tmp_path)
    assert owner._pending_execution_proposal(run, authority) == checkpoint.checkpoint_sha256
    assert authority.execution is not None  # Coordinates alone do not approve analysis.
    assert checkpoint.state == "pending" and checkpoint.consumed_decision_sha256 is None


@pytest.mark.parametrize("change", ["checkpoint", "consumed", "staged", "capsule", "other_plan", "config", "wrong_run", "forged_request_digest", "request_set"])
def test_execution_bound_preparation_rejects_stale_or_forged_review(tmp_path, change):
    import json
    from easyicu.research_agent.authority.plan_review import ReviewExecutionAuthority
    from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint

    run, authority, checkpoint = _execution_bound_proposal(tmp_path)
    if change == "checkpoint":
        path = run / "human_review_checkpoint.json"
        body = json.loads(path.read_text()); body["pipeline_config_sha256"] = "a" * 64
        path.write_text(json.dumps(body))
    elif change in {"consumed", "staged"}:
        body = checkpoint.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if change == "consumed":
            body["state"] = "consumed"
        else:
            decisions = [{"decision": "approved"}]
            checkpoint = checkpoint.decision_recorded(decisions=decisions,
                decision_records=[{"decision": "approved"}], decision_sha256=canonical_sha256(decisions))
            body = checkpoint.model_dump(mode="json", exclude={"checkpoint_sha256"})
        body["checkpoint_sha256"] = canonical_sha256(body)
        write_checkpoint(run / "human_review_checkpoint.json", HumanReviewCheckpoint.model_validate(body))
    elif change == "forged_request_digest":
        original = checkpoint.requests[0]
        forged = HumanReviewRequest.create(kind=original.kind, summary=original.summary,
            payload=original.payload, authority_sha256="f" * 64)
        body = checkpoint.model_dump(mode="json", exclude={"checkpoint_sha256"})
        body["requests"] = [forged.model_dump(mode="json")]
        body["request_set_sha256"] = canonical_sha256(body["requests"])
        body["checkpoint_sha256"] = canonical_sha256(body)
        write_checkpoint(run / "human_review_checkpoint.json", HumanReviewCheckpoint.model_validate(body))
    elif change == "request_set":
        from types import SimpleNamespace
        with pytest.raises(ValueError, match="request_set_mismatch"):
            owner._assert_pending_request_set(run, SimpleNamespace(requests=[]))
        return
    elif change == "capsule":
        (run / "run_input_capsule.json").write_text('{"context_sha256":"changed"}')
    elif change == "other_plan":
        authority = PlanReviewAuthority.create(plan=AnalysisPlan(research_question="Different", steps=[
            AnalysisStep(step_id="describe", intent="Describe")]),
            evidence_sha256=authority.evidence_sha256, execution=authority.execution)
    elif change == "config":
        execution = authority.execution.model_dump(mode="json"); execution["pipeline_config_sha256"] = "d" * 64
        authority = PlanReviewAuthority.create(plan=authority.model_dump(mode="json")["plan_payload"],
            evidence_sha256=authority.evidence_sha256, execution=ReviewExecutionAuthority.model_validate(execution))
    else:
        other = tmp_path / "different_run"; run.rename(other); run = other
    with pytest.raises(ValueError, match="first_observed_preparation_review_"):
        owner._pending_execution_proposal(run, authority)


@pytest.mark.parametrize("capsule_mode", ["malformed", "identity", "unregistered", "wrong_question", "wrong_database", "wrong_source"])
def test_execution_bound_preparation_rejects_unsealed_or_invalid_capsule(tmp_path, capsule_mode):
    run, authority, _ = _execution_bound_proposal(tmp_path, capsule_mode)
    with pytest.raises(ValueError):
        owner._pending_execution_proposal(run, authority)


def _run_bytes(root):
    return {str(path.relative_to(root)): (
                ("symlink", str(path.readlink())) if path.is_symlink()
                else ("directory", path.stat().st_mode) if path.is_dir()
                else ("file", path.stat().st_mode, path.read_bytes())
            ) for path in root.rglob("*")}


@pytest.mark.parametrize("capsule_mode", ["valid", "alias"])
def test_pending_review_reads_authority_without_repairing_missing_projection(tmp_path, capsule_mode, monkeypatch):
    run, authority, checkpoint = _execution_bound_proposal(tmp_path, capsule_mode)
    # An initialized EvidenceStore would repair this legacy projection on open.
    (run / "evidence" / "evidence_index.json").unlink()
    before = _run_bytes(run)
    def no_store_constructor(*_args, **_kwargs):
        pytest.fail("viewing a review must not initialize a mutable EvidenceStore")
    monkeypatch.setattr(EvidenceStore, "__init__", no_store_constructor)
    assert owner._pending_execution_proposal(run, authority) == checkpoint.checkpoint_sha256
    assert _run_bytes(run) == before
    assert not (run / "evidence" / "evidence_index.json").exists()


@pytest.mark.parametrize("change", ["missing_authority", "corrupt_authority", "missing_evidence", "tampered_evidence"])
def test_pending_review_rejects_invalid_evidence_without_repair_or_writes(tmp_path, change):
    run, authority, _ = _execution_bound_proposal(tmp_path)
    store = EvidenceStore(run)
    record = store.get("run_input_capsule")
    path = run / "evidence" / "evidence_authority.json"
    if change == "missing_authority":
        path.unlink()
    elif change == "corrupt_authority":
        path.write_bytes(b"{}")
    elif change == "missing_evidence":
        (run / record.relative_path).unlink()
    else:
        (run / record.relative_path).write_bytes(b"{}")
    before = _run_bytes(run)
    with pytest.raises(ValueError, match="first_observed_preparation_review_capsule_mismatch"):
        owner._pending_execution_proposal(run, authority)
    assert _run_bytes(run) == before


def test_scientific_review_snapshot_read_never_repairs_projection(tmp_path):
    authority, context, record = _case(tmp_path)
    (tmp_path / "evidence" / "evidence_index.json").unlink()
    before = _run_bytes(tmp_path)
    assert _verified_scientific_review(run_dir=tmp_path, authority=authority, context=context) == record.sha256
    assert _run_bytes(tmp_path) == before


def test_unregistered_capsule_does_not_create_evidence_directory(tmp_path):
    run, authority, _ = _execution_bound_proposal(tmp_path, "unregistered")
    assert not (run / "evidence").exists()
    before = _run_bytes(run)
    with pytest.raises(ValueError, match="first_observed_preparation_review_capsule_mismatch"):
        owner._pending_execution_proposal(run, authority)
    assert _run_bytes(run) == before


def test_corrupt_scientific_evidence_is_rejected_without_repair(tmp_path):
    authority, context, _ = _case(tmp_path)
    (tmp_path / "evidence" / "evidence_authority.json").write_text("{}")
    before = _run_bytes(tmp_path)
    with pytest.raises(ValueError, match="first_observed_preparation_scientific_review_missing"):
        _verified_scientific_review(run_dir=tmp_path, authority=authority, context=context)
    assert _run_bytes(tmp_path) == before


@pytest.mark.parametrize("malformed", [False, True])
def test_legacy_numeric_claim_validation_matches_original_store_without_writes(tmp_path, malformed):
    import json
    import shutil
    from easyicu.research_agent.authority.numeric_claim_identity import NumericClaim

    run, authority, checkpoint = _execution_bound_proposal(tmp_path)
    record = EvidenceStore(run).get("run_input_capsule")
    evidence_bytes = (run / record.relative_path).read_bytes()
    shutil.rmtree(run / "evidence")
    for path in run.glob(".easyicu_evidence_authority*"):
        path.unlink()
    evidence = run / "evidence"
    evidence.mkdir()
    target = run / record.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(evidence_bytes)
    (evidence / "evidence_index.json").write_text(json.dumps([record.model_dump(mode="json")]))
    (evidence / "evidence_aliases.json").write_text("{}")
    claim = ({"invalid": True} if malformed else NumericClaim(
        value="12", canonical=12.0, evidence_id=record.evidence_id,
        step_id="synthetic", source_field="count",
    ).to_dict())
    (evidence / "numeric_claims.json").write_text(json.dumps([claim]))
    before = _run_bytes(run)
    if malformed:
        with pytest.raises(ValueError, match="first_observed_preparation_review_capsule_mismatch"):
            owner._pending_execution_proposal(run, authority)
    else:
        assert owner._pending_execution_proposal(run, authority) == checkpoint.checkpoint_sha256
    assert _run_bytes(run) == before


@pytest.mark.parametrize("kind", ["invalid_record", "unknown_alias"])
def test_legacy_snapshot_rejects_unrelated_schema_damage_without_writes(tmp_path, kind):
    import json
    import shutil

    run, authority, _ = _execution_bound_proposal(tmp_path)
    record = EvidenceStore(run).get("run_input_capsule")
    evidence_bytes = (run / record.relative_path).read_bytes()
    shutil.rmtree(run / "evidence")
    for path in run.glob(".easyicu_evidence_authority*"):
        path.unlink()
    evidence = run / "evidence"
    evidence.mkdir()
    target = run / record.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(evidence_bytes)
    records = [record.model_dump(mode="json")]
    if kind == "invalid_record":
        records.append({"evidence_id": "another_record", "kind": "invalid"})
    (evidence / "evidence_index.json").write_text(json.dumps(records))
    (evidence / "evidence_aliases.json").write_text(json.dumps({"bad": "missing"} if kind == "unknown_alias" else {}))
    (evidence / "numeric_claims.json").write_text("[]")
    before = _run_bytes(run)
    with pytest.raises(ValueError, match="first_observed_preparation_review_capsule_mismatch"):
        owner._pending_execution_proposal(run, authority)
    assert _run_bytes(run) == before
