"""Synthetic outages exercise the job budget and checkpoint handoff."""

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopError,
    ProviderHardStopExceeded,
    ProviderHardStopLedger,
    ProviderHardStopLimits,
)
from easyicu.webserver.jobs import Job
from tests.support.provider_recovery import recovery as recovery_fixture

recovery = recovery_fixture


def http_failure(status=500):
    error = RuntimeError("private provider response")
    error.status_code = status
    return error




def test_outage_longer_than_two_requests_resumes_once_from_checkpoint(recovery):
    r = recovery
    calls = []

    def stopped(job):
        calls.append("initial")
        raise http_failure()

    def select(error):
        return r.owner.CheckpointContinuation(
            checkpoint={"kind": "planner", "path": "pipeline/run_synthetic/checkpoint.json", "sha256": "b" * 64},
            run=lambda job: calls.append("checkpoint") or {"status": "complete"},
        )

    assert r.state.run(stopped, select) == {"status": "complete"}
    assert calls == ["initial", "checkpoint"]
    receipt = json.loads((r.root / "provider_recovery.json").read_text())
    assert len(receipt["automatic_resumes"]) == 1
    event = receipt["automatic_resumes"][0]
    assert event["actor"] == "system" and event["cause_code"] == "provider_http_error"
    assert event["checkpoint"]["sha256"] == "b" * 64 and event["at"]
    assert r.probes[0] >= 120 and r.clock.now >= 1300
    for messages, kwargs in r.requests:
        assert [(message.role, message.content) for message in messages] == [("user", "Reply OK.")]
        assert kwargs == {"max_tokens": 16, "temperature": 0}
    calls = r.task.ledger.snapshot()["tasks"][0]["calls"]
    assert len(calls) == len(r.probes)
    assert all(call["role"] == "provider_recovery_probe" for call in calls)
    assert calls[0]["state"] == "failed_usage_unknown"
    assert calls[-1]["state"] == "completed"


def test_outage_bound_stops_without_spending_another_resume(recovery):
    r = recovery
    r.clock.recover_at = 10_000
    continuation = r.owner.CheckpointContinuation(checkpoint={"kind": "planner"}, run=lambda job: pytest.fail("outage"))
    with pytest.raises(r.owner.ProviderRecoveryExhausted) as stopped:
        r.state.run(lambda job: (_ for _ in ()).throw(http_failure()), lambda exc: continuation)
    assert stopped.value.code == "research_pipeline_provider_recovery_exhausted"
    assert r.clock.now == 1800
    assert json.loads((r.root / "provider_recovery.json").read_text())["automatic_resumes"] == []


@pytest.mark.parametrize("error", [http_failure(400), http_failure(429), ValueError("invalid contract"), RuntimeError("scientific review failed")])
def test_non_transient_failures_never_probe_or_resume(recovery, error):
    r = recovery
    with pytest.raises(type(error)) as stopped:
        r.state.run(lambda job: (_ for _ in ()).throw(error), lambda exc: pytest.fail("non-transient"))
    assert stopped.value is error
    assert r.probes == []


def test_missing_verified_checkpoint_keeps_original_failure_and_page_action(recovery):
    r = recovery
    error = http_failure()
    with pytest.raises(RuntimeError) as stopped:
        r.state.run(lambda job: (_ for _ in ()).throw(error), lambda exc: None)
    assert stopped.value is error and not r.probes


def test_a_refused_continuation_records_one_readable_sentence_without_spending_an_attempt(recovery):
    r = recovery
    code = "research_pipeline_development_resume_phase_not_replayed"

    def select(exc):
        r.state.record_continuation_refusal(code)
        return None

    with pytest.raises(RuntimeError):
        r.state.run(lambda job: (_ for _ in ()).throw(http_failure()), select)
    receipt = json.loads((r.root / "provider_recovery.json").read_text())
    assert receipt["continuation_refusal"]["code"] == code
    assert "prepare the same reviewed candidate again" in receipt["continuation_refusal"]["message"]
    assert code not in receipt["continuation_refusal"]["message"]
    assert receipt["automatic_resumes"] == [] and not r.probes
    assert sum(event.get("step") == "provider_continuation_refused" for event in r.job.events) == 1


def test_a_late_restore_refusal_is_recorded_after_the_provider_recovers(recovery):
    r = recovery
    r.clock.recover_at = 0
    error = r.owner.ResearchPipelineRunError("research_pipeline_development_resume_phase_not_replayed", "internal")

    def continued(job):
        raise error

    with pytest.raises(r.owner.ResearchPipelineRunError) as stopped:
        r.state.run(lambda job: (_ for _ in ()).throw(http_failure()),
                    lambda exc: r.owner.CheckpointContinuation({"kind": "planner", "sha256": "a" * 64}, continued))
    assert stopped.value is error
    receipt = json.loads((r.root / "provider_recovery.json").read_text())
    assert len(receipt["automatic_resumes"]) == 1
    assert receipt["continuation_refusal"]["code"] == error.code
    assert "internal" not in receipt["continuation_refusal"]["message"]


@pytest.mark.parametrize("error,code", [
    (ConnectionError("offline"), "provider_connection_failure"),
    (TimeoutError("offline"), "provider_timeout"),
    (http_failure(599), "provider_http_error"),
])
def test_connection_and_timeout_are_typed_transients(recovery, error, code):
    assert recovery.owner.transient_provider_cause(error) == code


@pytest.mark.parametrize("status", [400, 401, 429, 499, 500, 599, 600, True])
def test_shared_transport_classifier_uses_only_server_statuses(status):
    from easyicu.research_agent.providers.clients import transient_transport_failure
    result = transient_transport_failure(http_failure(status))
    assert bool(result) is (type(status) is int and 500 <= status <= 599)


@pytest.mark.parametrize("name,code", [("ReadTimeout", "provider_timeout"), ("APIConnectionError", "provider_connection_failure")])
def test_transport_class_names_require_an_sdk_owner(name, code):
    from easyicu.research_agent.providers.clients import transient_transport_failure
    sdk_error = type(name, (Exception,), {"__module__": "httpx"})("private")
    untyped = type(name, (Exception,), {"__module__": "application"})("private 503")
    assert transient_transport_failure(sdk_error) == {"reason_code": code}
    assert transient_transport_failure(untyped) is None


def test_context_and_contract_wrappers_never_hide_a_non_transient_stop(recovery):
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    r = recovery
    error = ValueError("review failed")
    error.__context__ = http_failure()
    assert r.owner.transient_provider_cause(error) is None
    error = ResearchPipelineRunError("scientific_plan_review_changes_required", "review")
    error.__cause__ = http_failure()
    assert r.owner.transient_provider_cause(error) is None
    envelope = ResearchPipelineRunError("research_pipeline_execution_failed", "failed")
    envelope.__cause__ = http_failure()
    assert r.owner.transient_provider_cause(envelope) == "provider_http_error"


def test_writer_transport_envelopes_preserve_cause_but_quality_failure_stops(recovery):
    from easyicu.research_agent.reporting.writer_only_migration import WriterOnlyMigrationError
    for code in ("WRITER_ONLY_REPAIR_FAILED_PRIOR_PRESERVED", "WRITER_ONLY_AUTHORITY_REPAIR_FAILED_PRIOR_PRESERVED"):
        error = WriterOnlyMigrationError(code=code, detail="failed")
        error.__cause__ = http_failure()
        assert recovery.owner.transient_provider_cause(error) == "provider_http_error"
    error = WriterOnlyMigrationError(code="WRITER_ONLY_QUALITY_AUDIT_FAILED_PRIOR_PRESERVED", detail="quality")
    error.__cause__ = http_failure()
    assert recovery.owner.transient_provider_cause(error) is None


def test_resume_count_is_shared_across_recurrent_outages(recovery):
    r = recovery
    r.clock.recover_at = 0
    def fail(job):
        raise http_failure()
    with pytest.raises(r.owner.ProviderRecoveryExhausted) as stopped:
        r.state.run(fail, lambda exc: r.owner.CheckpointContinuation(checkpoint={"kind": "planner"}, run=fail))
    assert stopped.value.details["reason_code"] == "automatic_resume_limit_exhausted"
    assert len(r.state.resumes) == 2 and len(r.probes) == 2


@pytest.mark.parametrize("dimension", ["attempts", "tokens", "cost", "wall"])
def test_original_budget_caps_bind_probes_and_continuation(recovery, dimension):
    r = recovery
    from dataclasses import replace
    # Freeze a new, deliberately small test ledger before the first request.
    limit = {
        "attempts": {"max_provider_attempts_per_run": 1},
        "tokens": {"max_total_tokens_per_run": 1},
        "cost": {"max_estimated_cost_usd_per_batch": 0.000001},
        "wall": {"max_wall_clock_seconds_per_task": 60},
    }[dimension]
    ledger = ProviderHardStopLedger(
        path=r.root / "small-ledger.json", task_ids=("small",),
        limits=replace(r.task.ledger.limits, **limit), batch_id="small", declaration_sha256="a" * 64,
    )
    r.state.task = ledger.start_task("small")
    with pytest.raises(ProviderHardStopExceeded):
        r.state.run(lambda job: (_ for _ in ()).throw(http_failure()),
                    lambda exc: r.owner.CheckpointContinuation(checkpoint={"kind": "planner"}, run=lambda job: pytest.fail("budget")))
    assert r.state.resumes == []
    assert len(ledger.snapshot()["tasks"][0]["calls"]) <= (1 if dimension == "attempts" else 0)
    if dimension == "wall":
        assert r.clock.now <= 60


def test_checkpoint_change_during_outage_stops_before_resume(recovery):
    r = recovery
    r.clock.recover_at = 0
    count = 0
    def select(exc):
        nonlocal count
        count += 1
        return r.owner.CheckpointContinuation(checkpoint={"sha256": str(count)}, run=lambda job: pytest.fail("changed"))
    with pytest.raises(RuntimeError, match="private provider"):
        r.state.run(lambda job: (_ for _ in ()).throw(http_failure()), select)
    assert not r.state.resumes


def test_cancellation_stops_wait_without_a_probe(recovery):
    r = recovery
    r.job.request_cancel()
    with pytest.raises(r.owner.ResearchPipelineRunError) as stopped:
        r.state.run(lambda job: (_ for _ in ()).throw(http_failure()),
                    lambda exc: r.owner.CheckpointContinuation(checkpoint={}, run=lambda job: pytest.fail("cancelled")))
    assert stopped.value.code == "research_pipeline_cancelled" and not r.probes


def test_probe_deadline_caps_the_existing_transport(recovery):
    r = recovery
    r.state.deadline = 10
    assert r.owner._ProbeTask(r.state).assert_active() == 10
    r.clock.now = 10
    with pytest.raises(r.owner.ProviderRecoveryExhausted):
        r.owner._ProbeTask(r.state).assert_active()


def test_shared_task_cannot_be_replaced(recovery):
    r = recovery
    with pytest.raises(ValueError, match="replace"):
        r.state.bind(wrapper_dir=r.root, task=object(), client=object())


def test_non_transient_probe_is_sanitized_and_never_resumes(recovery):
    r = recovery
    r.state.client = SimpleNamespace(complete_with_usage=lambda *a, **kw: (_ for _ in ()).throw(http_failure(401)))
    with pytest.raises(r.owner.ResearchPipelineRunError) as stopped:
        r.state.run(lambda job: (_ for _ in ()).throw(http_failure()),
                    lambda exc: r.owner.CheckpointContinuation(checkpoint={}, run=lambda job: pytest.fail("401")))
    assert stopped.value.code == "research_pipeline_provider_recovery_probe_failed"
    assert "private provider" not in str(stopped.value)
    assert not r.state.resumes


@pytest.mark.parametrize("stop_kind", ["provider_hard_stop", "pipeline_run"])
def test_typed_probe_stop_is_preserved_without_retry_or_resume(recovery, stop_kind):
    r = recovery
    error = (
        ProviderHardStopError("synthetic hard stop")
        if stop_kind == "provider_hard_stop"
        else r.owner.ResearchPipelineRunError("research_pipeline_cancelled", "synthetic cancellation")
    )
    probes = []

    def stop(messages, **kwargs):
        probes.append(r.clock.now)
        raise error

    r.state.client = SimpleNamespace(complete_with_usage=stop)
    with pytest.raises(type(error)) as stopped:
        r.state.run(lambda job: (_ for _ in ()).throw(http_failure()),
                    lambda exc: r.owner.CheckpointContinuation(checkpoint={}, run=lambda job: pytest.fail("stopped")))
    assert stopped.value is error
    assert probes == [r.owner.PROBE_INTERVAL_SECONDS]
    assert not r.state.resumes
    assert not any(event.get("step") == "provider_auto_resume" for event in r.job.events)


def test_approval_operation_runs_once_and_recovery_uses_the_existing_launch(recovery, monkeypatch):
    from easyicu.webserver import research_job_recovery
    r = recovery
    r.clock.recover_at = 0
    approvals = []
    continuations = []

    @research_job_recovery.recover_review
    def approve(*, job, decision):
        approvals.append(decision)
        state = job.provider_recovery
        state.bind(wrapper_dir=r.root, task=r.task, client=r.state.client)
        state.review_launch = {"execution_resume_source_run_id": "approved"}
        raise http_failure()

    def select(state, launch):
        assert launch == {"execution_resume_source_run_id": "approved"}
        def run(job):
            continuations.append(job)
            assert job.provider_recovery.task is r.task
            return {"complete": True}
        return r.owner.CheckpointContinuation(checkpoint={"kind": "execution"}, run=run)

    monkeypatch.setattr(research_job_recovery, "_continuation", select)
    assert approve(job=r.job, decision="approved") == {"complete": True}
    assert approvals == ["approved"] and len(continuations) == 1
