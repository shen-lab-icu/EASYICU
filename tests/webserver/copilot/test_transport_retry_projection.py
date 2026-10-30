"""Provider retry progress is bounded, path-free, and job-scoped."""

from easyicu.webserver.agent_pipeline_runs import _web_provider_progress_task, _web_provider_task_for_job
from easyicu.webserver.jobs import Job
from easyicu.webserver.pi_copilot.projections import project_job


def test_retry_projection_keeps_only_a_valid_opaque_call_id():
    job = Job("synthetic", "agent-run")
    for retry_id in ("a" * 32, "secret /Users/private", "A" * 32):
        job.emit({"type": "progress", "step": "provider_transport_retry",
                  "retry_phase": "waiting", "retry_id": retry_id, "current": 1})
    rows = project_job(job.snapshot())["progress"]
    assert rows[0]["retry_id"] == "a" * 32
    assert all("retry_id" not in row for row in rows[1:])


def test_retry_observers_are_independent_views_of_the_same_ledger(tmp_path):
    from tests.support.provider_recovery import _limits
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger

    ledger = ProviderHardStopLedger(path=(tmp_path / "ledger.json").resolve(),
                                   task_ids=("E1",), limits=_limits(), batch_id="test")
    original = ledger.start_task("E1")
    first, second = Job("first", "agent-run"), Job("second", "agent-run")
    a, b = _web_provider_progress_task(original, first), _web_provider_progress_task(original, second)
    assert a.ledger is b.ledger is original.ledger
    assert a.task_id == b.task_id == original.task_id
    a.observer({"phase": "waiting", "call_id": "a" * 32, "attempt": 1})
    b.observer({"phase": "started", "call_id": "b" * 32, "attempt": 2})
    assert [event["retry_id"] for event in first.snapshot()["events"]] == ["a" * 32]
    assert [event["retry_id"] for event in second.snapshot()["events"]] == ["b" * 32]
    a.observer({"phase": "waiting", "call_id": "private/path", "attempt": 1})
    assert len(first.snapshot()["events"]) == 1


def test_recovery_rebinds_exact_task_without_replacing_its_ledger(tmp_path):
    from tests.support.provider_recovery import _limits
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.webserver.provider_job_recovery import ProviderJobRecovery, RecoveryJob
    from easyicu.webserver.research_job_recovery import existing_provider_task

    ledger = ProviderHardStopLedger(path=(tmp_path / "ledger.json").resolve(),
                                   task_ids=("E1",), limits=_limits(), batch_id="test")
    original = ledger.start_task("E1")
    job = Job("job", "agent-run")
    recovery = ProviderJobRecovery(job)
    bound_job = RecoveryJob(recovery, job.id)
    first = _web_provider_task_for_job(bound_job, existing=existing_provider_task(bound_job),
                                       start=lambda: original)
    recovery.bind(wrapper_dir=tmp_path, task=first, client=object())
    def no_new_task():
        raise AssertionError("a continuation must not create a new budget")
    continued = _web_provider_task_for_job(bound_job, existing=existing_provider_task(bound_job),
                                           start=no_new_task)
    assert continued is first
    recovery.bind(wrapper_dir=tmp_path, task=continued, client=object())
    assert recovery.task is first and first.ledger is ledger
