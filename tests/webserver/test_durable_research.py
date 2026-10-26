"""Host admission, accounting references and durable transport integration."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from easyicu.webserver import durable_research as runtime
from easyicu.webserver.durable_research_store import DurableResearchError
from easyicu.webserver.jobs import JobManager


class Dispatcher:
    def __init__(self, path, callback, **kwargs):
        self.callback = callback
        self.submitted = []
        self.state = None

    def launch(self):
        pass

    def submit(self, identity):
        self.submitted.append(identity)
        return identity

    def status(self, identity):
        return self.state

    def close(self, timeout=4):
        return True


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "runtime_digest", lambda: "code")
    monkeypatch.setattr("easyicu.webserver.durable_research_store.runtime_digest", lambda: "code")
    monkeypatch.setattr(runtime.study_contexts, "clear_active_job_if", lambda *a, **k: {})
    manager = JobManager()
    manager.enable_persistence(tmp_path / "jobs.sqlite3")
    service = runtime.DurableResearchService(tmp_path, manager=manager, dispatcher_factory=Dispatcher,
                                           recover_runner=lambda payload, job: lambda job: {"gate": {"status": "passed"}})
    return tmp_path, manager, service


def restart(setup, recover=None):
    root, _, old = setup
    manager = JobManager()
    manager.enable_persistence(root / "jobs.sqlite3")
    service = runtime.DurableResearchService(root, manager=manager, dispatcher_factory=Dispatcher,
                                           recover_runner=recover or old.recover_runner)
    service.launch()
    return manager, service


def test_prepared_command_never_publishes_or_recovers(setup):
    _, _, service = setup
    job = service.reserve({"study_context_id": "study-a"})
    manager, recovered = restart(setup)
    assert recovered.dispatcher.submitted == []
    assert manager.get(job.id).snapshot()["interrupted"]
    assert recovered.execute(job.id)["status"] == "blocked"


def test_activated_outbox_republishes_same_job_and_spends_same_attempts(setup):
    _, manager, service = setup
    job = service.reserve({"study_context_id": "study-a"})
    service.activate(job.id, lambda job: pytest.fail("lost process callable must not execute"))
    manager, recovered = restart(setup)
    assert recovered.dispatcher.submitted == [job.id]
    assert manager.get(job.id).snapshot()["status"] == "running"
    assert not any(e.get("type") == "end" for e in manager.get(job.id).snapshot()["events"])
    assert recovered.execute(job.id) == {"status": "done", "job_id": job.id}
    assert recovered.store.get(job.id)["attempts"] == 1
    assert manager.get(job.id).result["gate"]["status"] == "passed"
    recovered.execute(job.id)
    assert recovered.store.get(job.id)["attempts"] == 1


@pytest.mark.parametrize("reason", ["user_cancelled", "server_shutdown"])
def test_persisted_cancellation_prevents_recovery(setup, reason):
    _, _, service = setup
    job = service.reserve({"study_context_id": "study-a"})
    service.activate(job.id, lambda job: None)
    job.request_cancel(reason)
    manager, recovered = restart(setup, lambda *a: pytest.fail("cancelled work resumed"))
    assert recovered.dispatcher.submitted == []
    assert recovered.store.get(job.id)["reason"] == "cancelled"
    assert manager.get(job.id).status == "cancelled"


def test_code_change_blocks_before_scientific_reconstruction(setup, monkeypatch):
    _, _, service = setup
    job = service.reserve({"study_context_id": "study-a"})
    service.activate(job.id, lambda job: None)
    manager, recovered = restart(setup, lambda *a: pytest.fail("changed runtime resumed"))
    monkeypatch.setattr(runtime, "runtime_digest", lambda: "new-code")
    assert recovered.execute(job.id)["status"] == "failed"
    assert manager.get(job.id).error == "research_auto_resume_runtime_changed"


def test_duplicate_callback_does_not_steal_initial_runner(setup):
    _, _, service = setup
    started, release = threading.Event(), threading.Event()
    calls = []
    def runner(job):
        calls.append(job.id)
        started.set()
        assert release.wait(3)
        return {"ok": True}
    job = service.reserve({"study_context_id": "study-a"})
    service.activate(job.id, runner)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(service.execute, job.id)
        assert started.wait(2)
        second = pool.submit(service.execute, job.id)
        release.set()
        assert first.result()["status"] == "done"
        second.result()
    assert calls == [job.id]
    assert service.store.get(job.id)["attempts"] == 1


def test_ambiguous_enqueue_keeps_admission_and_runner(setup, monkeypatch):
    _, _, service = setup
    job = service.reserve({"study_context_id": "study-a"})
    monkeypatch.setattr(service.dispatcher, "submit", lambda identity: (_ for _ in ()).throw(OSError("ambiguous")))
    service.activate(job.id, lambda job: {"ok": True})
    assert service.store.get(job.id)["status"] == "active"
    assert not job.cancel_requested
    assert service.execute(job.id)["status"] == "done"
    service.close()


def test_completed_scheduler_receipt_cannot_strand_host_job(setup):
    _, _, service = setup
    job = service.reserve({"study_context_id": "study-a"})
    service.dispatcher.state = {"status": "SUCCESS", "receipt": {"status": "failed"}}
    service.activate(job.id, lambda job: pytest.fail("already completed workflow repeated"))
    assert job.status == "failed"
    assert service.store.get(job.id)["status"] == "terminal"


def test_original_budget_binding_immutable_latest_wrapper_advances(setup):
    _, _, service = setup
    job = service.reserve({"study_context_id": "study-a"})
    service.activate(job.id, lambda job: {})
    binding = {"wrapper_dir": "/first", "hard_stop_ledger_path": "/ledger", "hard_stop_task_id": "paid-task",
               "hard_stop_declaration_sha256": "digest"}
    service.store.bind_provider(job.id, binding)
    service.store.bind_provider(job.id, {**binding, "wrapper_dir": "/second"})
    assert service.store.get(job.id)["payload"]["provider_binding"]["wrapper_dir"] == "/second"
    with pytest.raises(DurableResearchError, match="budget_binding_changed"):
        service.store.bind_provider(job.id, {**binding, "hard_stop_task_id": "fresh-budget"})


def test_recovery_attempt_limit_is_durable(setup):
    _, _, service = setup
    job = service.reserve({"study_context_id": "study-a"})
    service.activate(job.id, lambda job: pytest.fail("attempt cap bypass"))
    for expected in range(1, 5):
        assert service.store.begin_attempt(job.id) == expected
    manager, recovered = restart(setup, lambda *a: pytest.fail("attempt cap bypass"))
    assert recovered.execute(job.id)["status"] == "failed"
    assert manager.get(job.id).error == "research_auto_resume_attempt_limit"


def test_cancel_before_dispatch_releases_capacity_and_study_binding(setup, monkeypatch):
    _, manager, service = setup
    cleared = []
    monkeypatch.setattr(runtime.study_contexts, "clear_active_job_if", lambda *args, **kwargs: cleared.append(args))
    job = service.reserve({"study_context_id": "study-a"})
    service.activate(job.id, lambda job: pytest.fail("cancelled queued runner executed"))
    job.request_cancel("user_cancelled")
    assert service.execute(job.id)["status"] == "cancelled"
    assert job.status == "cancelled"
    assert not job.consumes_capacity()
    assert cleared == [("study-a", job.id)]
    assert service.store.get(job.id)["attempts"] == 0


def test_abandon_after_study_handoff_clears_only_its_binding(setup, monkeypatch):
    _, _, service = setup
    cleared = []
    monkeypatch.setattr(runtime.study_contexts, "clear_active_job_if", lambda *args, **kwargs: cleared.append(args))
    job = service.reserve({"study_context_id": "study-a"})
    service.abandon(job.id)
    assert job.status == "failed"
    assert cleared == [("study-a", job.id)]


def test_outbox_cancel_survives_a_failed_job_journal_write(setup, monkeypatch):
    from easyicu.webserver.job_journal import JobJournalError

    _, _, service = setup
    job = service.reserve({"study_context_id": "study-a"})
    service.activate(job.id, lambda job: pytest.fail("cancelled runner executed"))
    def failed(receipt):
        raise JobJournalError("synthetic_write_failure")
    monkeypatch.setattr(job, "_persist", failed)
    assert job.request_cancel("user_cancelled") is False
    assert service.store.get(job.id)["payload"]["_cancel_requested"] is True
    manager, recovered = restart(setup, lambda *a: pytest.fail("outbox cancellation lost"))
    assert recovered.dispatcher.submitted == []
    assert recovered.store.get(job.id)["reason"] == "cancelled"
    assert manager.get(job.id).cancel_requested
