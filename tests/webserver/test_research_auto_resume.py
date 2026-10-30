"""Synthetic checkpoint/ledger contracts; no Provider or scientific execution."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.webserver import agent_review_recovery, provider_adapter, research_auto_resume as recovery
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError


@pytest.fixture
def bound(tmp_path, monkeypatch):
    from easyicu.webserver import settings, study_contexts

    study = {"id": "study-a", "question": "synthetic", "data_source": {"path": str(tmp_path / "export")}}
    wrapper = tmp_path / "study-a" / "run_original"
    wrapper.mkdir(parents=True)
    config = PipelineConfig(workdir=wrapper / "pipeline", runner_kind="docker")
    task_id = "web-original"
    ledger = ProviderHardStopLedger(
        path=wrapper / ".runtime" / "ledger.json", task_ids=(task_id,),
        limits=provider_adapter.web_research_agent_hard_stop_limits("full_reviewed"),
        batch_id=task_id, declaration_sha256="a" * 64,
    )
    task = ledger.start_task(task_id)
    ledger.reserve_transport_attempt(task_id=task_id, role="planner", model="synthetic", messages=[], max_tokens=4, prior_attempt_id=None)
    seed = agent_review_recovery.WebReviewRecoverySeed.create(
        wrapper_dir=str(wrapper), study=study,
        scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(study),
        provider_meta={"provider": "openai", "external": True}, provider_public={"provider": "openai", "model": "synthetic"},
        credential_source="pi_verified", budget_mode="full_reviewed",
        prepared_package_binding={"schema_version": "easyicu.web-prepared-package-binding/1", "binding_sha256": "b" * 64},
        pipeline_config=config.recovery_payload(), pipeline_config_sha256=config.canonical_digest(),
        acquisition_projection={}, hard_stop_ledger_path=str(ledger.path), hard_stop_task_id=task_id,
        hard_stop_declaration_sha256="a" * 64, created_at=1.0,
    )
    agent_review_recovery.put_recovery_seed(seed)
    descriptor = recovery.prepare_descriptor(
        study=study, project_root=tmp_path, credential_source="pi_verified",
        budget_mode="full_reviewed", external_llm_opt_in=True,
    )
    descriptor["provider_binding"] = {"wrapper_dir": str(wrapper), **{
        key: getattr(seed, key) for key in ("hard_stop_ledger_path", "hard_stop_task_id", "hard_stop_declaration_sha256")
    }}
    descriptor["_resume_attempt"] = 2
    monkeypatch.setattr(study_contexts, "get_context", lambda _id: study)
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(recovery, "_provider_environment", lambda _seed: {})
    job = SimpleNamespace(id="original", is_cancel_requested=lambda: False, emit=lambda _event: None)
    return SimpleNamespace(descriptor=descriptor, seed=seed, study=study, wrapper=wrapper, ledger=ledger, task=task, job=job)


def _checkpoint(bound, monkeypatch, state="executing", authenticated=True):
    run_id = "run_scientific"
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Synthetic plan", authority_sha256="c" * 64, payload={})
    decisions = [{"review_id": request.review_id, "authority_sha256": request.authority_sha256,
                  "decision": "approved", "reviewer": "original", "decided_at": "2026-10-23T00:00:00Z", "note": "original"}]
    records = [{"review_id": request.review_id, "decision": "approved", "reviewer_identity_source": "authenticated" if authenticated else "unauthenticated_client_claim"}]
    checkpoint = HumanReviewCheckpoint.create(
        run_id=run_id, pipeline_config_sha256=bound.seed.pipeline_config_sha256,
        environment_identity={}, llm_signature_sha256="d" * 64, run_input_capsule_sha256="e" * 64,
        capability_activation_sha256="f" * 64, runtime_capabilities=(), runtime_bundle=None,
        requests=(request,), plan_handoff={}, execution_coordinates={},
    )
    if state != "pending":
        checkpoint = checkpoint.decision_recorded(decisions=decisions, decision_records=records, decision_sha256=canonical_sha256(decisions)).decision_committed()
    if state in {"executing", "write_in_progress", "completed"}:
        checkpoint = checkpoint.execution_started()
    if state in {"write_in_progress", "completed"}:
        checkpoint = checkpoint.transitioned(state)
    write_checkpoint(bound.wrapper / "pipeline" / run_id / "human_review_checkpoint.json", checkpoint)
    record = agent_review_recovery.WebReviewRecoveryRecord.create(
        **bound.seed.model_dump(exclude={"schema_version", "seed_sha256"}), run_id=run_id,
    )
    monkeypatch.setattr(agent_review_recovery, "get_record", lambda key: record if key == run_id else None)
    bound.descriptor["execution_resume_source_run_id"] = run_id
    return checkpoint


def test_descriptor_omits_credentials_plan_and_review_inputs(bound):
    kwargs = {"project_root": bound.descriptor["project_root"], "credential_source": "pi_verified", "budget_mode": "full_reviewed", "provider_environment": {"TOKEN": "secret"}, "plan_change_request": {"text": "private plan"}}
    result = recovery.prepare_descriptor({"external_llm_opt_in": True, "reviewer": "private person", "note": "private review"}, bound.study, kwargs)
    serialized = json.dumps(result)
    assert all(value not in serialized for value in ("TOKEN", "secret", "private plan", "private person", "private review", "provider_environment"))


def test_planner_uses_new_wrapper_and_original_ended_attempt_reservation(bound, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs, research_launch_resume
    from easyicu.webserver.research_job_recovery import existing_provider_task
    from easyicu.webserver.run_record import RunDirectory

    monkeypatch.setattr(research_launch_resume, "development_resume_checkpoint", lambda **kw: {"checkpoint_sha256": "c" * 64})
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client", lambda *a, **kw: (object(), dict(bound.seed.provider_public)))
    captured = {}
    attempt = bound.ledger.snapshot()["tasks"][0]["calls"][0]["attempt_id"]
    bound.ledger.finish_transport_attempt(task_id=bound.task.task_id, attempt_id=attempt,
                                          usage=None, error_type="InternalServerError")
    before = bound.ledger.snapshot()["totals"]

    def make_runner(**kwargs):
        captured.update(kwargs)
        def run(job):
            task = existing_provider_task(job)
            assert task is not None
            assert task.task_id == bound.task.task_id
            assert task.ledger.path == bound.ledger.path
            assert task.ledger.snapshot()["totals"] == before
            assert task.ledger.snapshot()["tasks"][0]["calls"][0]["state"] == "failed_usage_unknown"
            target = RunDirectory.create(Path(kwargs["project_root"]), bound.study["id"], job.id)
            assert target.path != bound.wrapper
            assert job.id == "original-restart-2"
            return {"checkpoint_continued": True}
        return run

    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", make_runner)
    result = recovery.recover_runner(bound.descriptor, bound.job)(bound.job)
    assert result == {"checkpoint_continued": True}
    assert captured["development_resume_source_job_id"] == "original"
    assert "execution_resume_source_run_id" not in captured
    assert captured["literature_search_authorized"] is False


def test_planner_unknown_inflight_without_guard_is_never_replayed(bound, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs, research_launch_resume

    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", lambda **kw: pytest.fail("unsafe replay"))
    monkeypatch.setattr(research_launch_resume, "development_resume_checkpoint", lambda **kw: {"checkpoint_sha256": "c" * 64})
    with pytest.raises(ResearchPipelineRunError) as error:
        recovery.recover_runner(bound.descriptor, bound.job)
    assert error.value.code == "research_auto_resume_planner_provider_worker_unverified"
    assert bound.ledger.snapshot()["tasks"][0]["calls"][0]["state"] == "in_progress"


def test_probe_unknown_inflight_without_guard_is_never_replayed(bound):
    row = bound.ledger.snapshot()["tasks"][0]
    row["calls"][0]["role"] = "provider_recovery_probe"
    with pytest.raises(ResearchPipelineRunError) as error:
        recovery._verify_guarded_planner_inflight(bound.seed, row)
    assert error.value.code == "research_auto_resume_planner_provider_worker_unverified"
    assert bound.ledger.snapshot()["tasks"][0]["calls"][0]["state"] == "in_progress"


@pytest.mark.parametrize("state", ["approved_pending_execution", "executing"])
def test_existing_approved_execution_is_delegated_without_new_decision(bound, monkeypatch, state):
    from easyicu.webserver import agent_pipeline_runs

    original = _checkpoint(bound, monkeypatch, state)
    seen = []
    monkeypatch.setattr(agent_pipeline_runs, "resume_research_pipeline", lambda **kwargs: seen.append(kwargs) or {"status": "continued"})
    result = recovery.recover_runner(bound.descriptor, bound.job)(bound.job)
    assert result == {"status": "continued"}
    assert seen[0]["require_recorded_decision"] is True
    assert seen[0]["reviewer"] == seen[0]["note"] == ""
    from easyicu.research_agent.orchestration.human_review_checkpoint import load_checkpoint
    assert load_checkpoint(bound.wrapper / "pipeline" / original.run_id / "human_review_checkpoint.json", require_pending=False) == original


@pytest.mark.parametrize("state", ["pending", "write_in_progress", "completed"])
def test_missing_approval_or_uncertain_side_effect_never_resumes(bound, monkeypatch, state):
    _checkpoint(bound, monkeypatch, state)
    with pytest.raises(ResearchPipelineRunError, match="original authority") as raised:
        recovery.recover_runner(bound.descriptor, bound.job)
    assert raised.value.code == "research_auto_resume_approved_checkpoint_required"


def test_claimed_review_is_not_server_approval(bound, monkeypatch):
    _checkpoint(bound, monkeypatch, authenticated=False)
    with pytest.raises(ResearchPipelineRunError) as raised:
        recovery.recover_runner(bound.descriptor, bound.job)
    assert raised.value.code == "research_auto_resume_recorded_approval_required"


@pytest.mark.parametrize("failure", ["cancelled", "opt_out", "scope", "ledger_missing", "binding_changed", "no_checkpoint", "formal"])
def test_guard_refusals_never_construct_fresh_runner(bound, monkeypatch, failure):
    from easyicu.webserver import agent_pipeline_runs, research_launch_resume, settings
    from easyicu.research_agent.orchestration.config import PipelineConfig

    monkeypatch.setattr(agent_pipeline_runs, "make_research_pipeline_run_runner", lambda **kw: pytest.fail("fresh launch"))
    monkeypatch.setattr(research_launch_resume, "development_resume_checkpoint", lambda **kw: None)
    expected = {"cancelled": "cancelled", "opt_out": "opt_in_revoked", "scope": "configuration_superseded", "ledger_missing": "budget_ledger_missing", "binding_changed": "budget_binding_changed", "no_checkpoint": "planner_checkpoint_missing", "formal": "formal_profile_requires_protocol"}[failure]
    if failure == "cancelled":
        bound.job.is_cancel_requested = lambda: True
    elif failure == "opt_out":
        monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": False})
    elif failure == "scope":
        bound.study["question"] = "Changed scientific question"
    elif failure == "ledger_missing":
        bound.ledger.path.unlink()
    elif failure == "binding_changed":
        bound.descriptor["provider_binding"]["hard_stop_task_id"] = "different"
    elif failure == "formal":
        monkeypatch.setattr(PipelineConfig, "from_recovery_payload", lambda *a, **kw: SimpleNamespace(submission_profile_name="npj_dm_formal", workdir=bound.wrapper / "pipeline"))
    with pytest.raises(ResearchPipelineRunError) as raised:
        recovery.recover_runner(bound.descriptor, bound.job)
    assert raised.value.code == "research_auto_resume_" + expected
    if failure == "ledger_missing":
        assert not bound.ledger.path.exists()


def test_provider_identity_drift_preserves_original_accounting(bound, monkeypatch):
    from easyicu.webserver import research_launch_resume

    monkeypatch.setattr(research_launch_resume, "development_resume_checkpoint", lambda **kw: {"checkpoint_sha256": "c" * 64})
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client", lambda *a, **kw: (object(), {"provider": "openai", "model": "different"}))
    attempt = bound.ledger.snapshot()["tasks"][0]["calls"][0]["attempt_id"]
    bound.ledger.finish_transport_attempt(task_id=bound.task.task_id, attempt_id=attempt,
                                          usage=None, error_type="InternalServerError")
    before = bound.ledger.path.read_bytes()
    with pytest.raises(ResearchPipelineRunError) as raised:
        recovery.recover_runner(bound.descriptor, bound.job)
    assert raised.value.code == "research_auto_resume_provider_identity_changed"
    assert bound.ledger.path.read_bytes() == before


@pytest.mark.parametrize("state", ["pending", "executing"])
def test_real_review_owner_reuses_approval_or_refuses_missing_decision(bound, monkeypatch, state):
    from easyicu.research_agent.orchestration.workflow import HumanReviewPending
    from easyicu.webserver import agent_pipeline_runs, dataio
    from easyicu.webserver.provider_job_recovery import ProviderJobRecovery, RecoveryJob

    checkpoint = _checkpoint(bound, monkeypatch, state)
    run_dir = bound.wrapper / "pipeline" / checkpoint.run_id
    seen = []

    class Pipeline:
        def resume_human_review(self, decisions, **kwargs):
            seen.append(decisions)
            return SimpleNamespace(manifest_path=run_dir / "manifest.json")

    pending = HumanReviewPending(run_id=checkpoint.run_id, thread_id=checkpoint.run_id,
                                run_dir=str(run_dir), requests=checkpoint.requests,
                                resume_scope="durable_checkpoint", resume_pid=None)
    entry = agent_review_recovery.PendingReviewEntry(
        pipeline=Pipeline(), pending=pending, wrapper_dir=bound.wrapper,
        study=bound.study, provider={}, acquisition=SimpleNamespace(), created_at=1,
        credential_source="pi_verified", budget_mode="full_reviewed",
        prepared_package_binding=bound.seed.prepared_package_binding,
    )
    registry = agent_review_recovery.PendingReviewRegistry()
    registry.register(entry)
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", registry)
    monkeypatch.setattr(agent_pipeline_runs, "_pending_plan_authority", lambda pending: {})
    monkeypatch.setattr(agent_pipeline_runs, "_plan_has_complete_reviewable_recommendation", lambda plan: True)
    monkeypatch.setattr(agent_pipeline_runs, "_load_pending_scientific_review", lambda *args: {"approval_allowed": True})
    monkeypatch.setattr(dataio, "validate_research_pipeline_source", lambda *a, **kw: {})
    monkeypatch.setattr(agent_pipeline_runs.run_patient_grouping, "require_patient_groups_for_approval", lambda *a, **kw: None)
    monkeypatch.setattr(agent_pipeline_runs, "remove_review_recovery_record", lambda *a: None)
    monkeypatch.setattr(agent_pipeline_runs, "_remove_local_recovery", lambda *a: None)
    monkeypatch.setattr(agent_pipeline_runs, "_write_projection", lambda **kw: {"status": "complete"})
    job = RecoveryJob(ProviderJobRecovery(bound.job), bound.job.id)
    kwargs = dict(run_id=checkpoint.run_id, study_context_id=bound.study["id"], decision="approved",
                  reviewer="must not replace", note="must not replace", job=job, require_recorded_decision=True)
    if state == "pending":
        with pytest.raises(ResearchPipelineRunError) as raised:
            agent_pipeline_runs.resume_research_pipeline(**kwargs)
        assert raised.value.code == "research_pipeline_recorded_approval_required"
        assert seen == []
    else:
        result = agent_pipeline_runs.resume_research_pipeline(**kwargs)
        assert result == {"status": "complete"}
        assert seen == [[dict(item) for item in checkpoint.approved_decisions]]


def test_exhausted_original_attempt_budget_blocks_before_execution(bound, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs

    _checkpoint(bound, monkeypatch)
    limits = replace(bound.ledger.limits, max_provider_attempts_per_run=1, max_provider_attempts_per_batch=1)
    bound.ledger.path.unlink()
    ledger = ProviderHardStopLedger(path=bound.ledger.path, task_ids=(bound.task.task_id,),
                                   limits=limits, batch_id=bound.task.task_id,
                                   declaration_sha256=bound.seed.hard_stop_declaration_sha256)
    ledger.start_task(bound.task.task_id)
    ledger.reserve_transport_attempt(task_id=bound.task.task_id, role="planner", model="synthetic",
                                     messages=[], max_tokens=4, prior_attempt_id=None)
    monkeypatch.setattr(provider_adapter, "web_research_agent_hard_stop_limits", lambda mode: limits)
    monkeypatch.setattr(agent_pipeline_runs, "resume_research_pipeline", lambda **kw: pytest.fail("budget bypass"))
    with pytest.raises(ResearchPipelineRunError) as raised:
        recovery.recover_runner(bound.descriptor, bound.job)
    assert raised.value.code == "research_auto_resume_budget_exhausted"
    assert ledger.snapshot()["tasks"][0]["calls"][0]["state"] == "in_progress"


def test_durable_envelope_does_not_restart_in_process_retry_quota(bound, monkeypatch):
    from easyicu.webserver.provider_job_recovery import ProviderJobRecovery, RecoveryJob
    from easyicu.webserver.research_job_recovery import recover_review

    owner = ProviderJobRecovery(bound.job)
    job = RecoveryJob(owner, bound.job.id)
    monkeypatch.setattr(ProviderJobRecovery, "run", lambda *a: pytest.fail("reset outage quota"))
    seen = []
    wrapped = recover_review(lambda *, job: seen.append(job) or "continued")
    assert wrapped(job=job) == "continued"
    assert seen == [job]


@pytest.mark.parametrize("backend,state,allowed", [
    ("docker", "executing", True),
    ("host", "executing", False),
    ("auto", "executing", False),
    ("host", "approved_pending_execution", True),
])
def test_interrupted_worker_requires_confirmable_teardown(bound, monkeypatch, backend, state, allowed):
    from easyicu.webserver import agent_pipeline_runs

    config = PipelineConfig(workdir=bound.wrapper / "pipeline", runner_kind=backend)
    seed_values = bound.seed.model_dump(exclude={"seed_sha256"})
    seed_values.update(pipeline_config=config.recovery_payload(), pipeline_config_sha256=config.canonical_digest())
    bound.seed = agent_review_recovery.WebReviewRecoverySeed.create(**seed_values)
    monkeypatch.setattr(agent_review_recovery, "load_recovery_seed", lambda wrapper: bound.seed)
    _checkpoint(bound, monkeypatch, state)
    effects = []
    monkeypatch.setattr(agent_pipeline_runs, "resume_research_pipeline", lambda **kw: effects.append("execution") or {"status": "continued"})
    before = bound.ledger.path.read_bytes()
    if allowed:
        result = recovery.recover_runner(bound.descriptor, bound.job)(bound.job)
        assert result == {"status": "continued"}
        assert effects == ["execution"]
    else:
        with pytest.raises(ResearchPipelineRunError) as raised:
            recovery.recover_runner(bound.descriptor, bound.job)
        assert raised.value.code == "research_auto_resume_child_process_state_unverified"
        assert effects == []
        assert bound.ledger.path.read_bytes() == before


def test_quiescent_host_recovery_keeps_original_approval_and_budget(bound, monkeypatch):
    from easyicu.runtime.host_execution_lease import prepare_host_execution
    from easyicu.webserver import agent_pipeline_runs

    config = PipelineConfig(workdir=bound.wrapper / "pipeline", runner_kind="host")
    seed_values = bound.seed.model_dump(exclude={"seed_sha256"})
    seed_values.update(pipeline_config=config.recovery_payload(), pipeline_config_sha256=config.canonical_digest())
    bound.seed = agent_review_recovery.WebReviewRecoverySeed.create(**seed_values)
    monkeypatch.setattr(agent_review_recovery, "load_recovery_seed", lambda wrapper: bound.seed)
    checkpoint = _checkpoint(bound, monkeypatch)
    run_dir = bound.wrapper / "pipeline" / checkpoint.run_id
    lease = prepare_host_execution(run_dir / ".host-execution", script_sha256="a" * 64)
    lease.abort()
    lease.release()
    before = bound.ledger.snapshot()
    checkpoint_bytes = (run_dir / "human_review_checkpoint.json").read_bytes()
    calls = []

    def resume(**kwargs):
        calls.append(kwargs)
        assert kwargs["run_id"] == checkpoint.run_id
        assert kwargs["require_recorded_decision"] is True
        assert kwargs["reviewer"] == kwargs["note"] == ""
        now = bound.ledger.snapshot()
        assert now["tasks"][0]["task_id"] == before["tasks"][0]["task_id"]
        assert now["tasks"][0]["calls"] == before["tasks"][0]["calls"]
        assert now["totals"] == before["totals"]
        assert now["tasks"][0]["elapsed_seconds"] >= before["tasks"][0]["elapsed_seconds"]
        assert (run_dir / "human_review_checkpoint.json").read_bytes() == checkpoint_bytes
        return {"status": "continued"}

    monkeypatch.setattr(agent_pipeline_runs, "resume_research_pipeline", resume)
    assert recovery.recover_runner(bound.descriptor, bound.job)(bound.job) == {"status": "continued"}
    assert len(calls) == 1


def test_host_guard_is_rechecked_immediately_before_resume(bound, monkeypatch):
    from easyicu.runtime.host_execution_lease import HostExecutionLeaseError
    from easyicu.runtime import host_execution_lease
    from easyicu.webserver import agent_pipeline_runs

    config = PipelineConfig(workdir=bound.wrapper / "pipeline", runner_kind="host")
    seed_values = bound.seed.model_dump(exclude={"seed_sha256"})
    seed_values.update(pipeline_config=config.recovery_payload(), pipeline_config_sha256=config.canonical_digest())
    bound.seed = agent_review_recovery.WebReviewRecoverySeed.create(**seed_values)
    monkeypatch.setattr(agent_review_recovery, "load_recovery_seed", lambda wrapper: bound.seed)
    _checkpoint(bound, monkeypatch)
    checks = []

    def guard(path, **kwargs):
        assert kwargs["allow_missing"] is False
        checks.append(path)
        if len(checks) == 2:
            raise HostExecutionLeaseError("host_execution_unresolved")

    monkeypatch.setattr(host_execution_lease, "assert_host_execution_quiescent", guard)
    monkeypatch.setattr(agent_pipeline_runs, "resume_research_pipeline", lambda **kw: pytest.fail("uncertain child executed"))
    resume = recovery.recover_runner(bound.descriptor, bound.job)
    with pytest.raises(ResearchPipelineRunError) as raised:
        resume(bound.job)
    assert raised.value.code == "research_auto_resume_child_process_state_unverified"
    assert len(checks) == 2
