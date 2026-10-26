"""Stable transport identities for a separately authorized durable executor."""
from __future__ import annotations

import threading

import pytest

from easyicu.webserver.job_journal import JobJournal, JobJournalError
from easyicu.webserver.jobs import Job, JobCapacityError, JobManager


def interrupted_manager(tmp_path, *, cancelled=False):
    path = tmp_path / "jobs.sqlite3"
    journal = JobJournal(path)
    old = Job("durable-job-1", "agent-run", persist=journal.save)
    journal.create(old.lifecycle_receipt())
    old.emit({"type": "progress", "step": "before_crash"})
    if cancelled:
        old.request_cancel()
    manager = JobManager()
    manager.enable_persistence(path)
    return manager, old


def test_reservation_is_stable_idempotent_and_has_no_runner(tmp_path):
    manager = JobManager(max_running=1)
    manager.enable_persistence(tmp_path / "jobs.sqlite3")
    job = manager.reserve("agent-run", "stable-1")
    assert manager.reserve("agent-run", "stable-1") is job
    assert job.status == "running" and job.result is None
    assert manager._journal.records()[0]["id"] == "stable-1"
    with pytest.raises(JobJournalError, match="kind_conflict"):
        manager.reserve("extraction", "stable-1")
    with pytest.raises(JobCapacityError):
        manager.reserve("agent-run", "different")
    calls = []
    assert manager.run_reserved(job.id, lambda target: calls.append(target.id)) is job
    assert job.status == "done" and calls == [job.id]
    assert manager.run_reserved(job.id, lambda target: calls.append("duplicate")) is job
    assert calls == [job.id]


def test_cancelled_reservation_never_enters_runner_and_survives_restart(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    manager = JobManager()
    manager.enable_persistence(path)
    job = manager.reserve("agent-run", "cancel-before-run")
    assert job.request_cancel("user_requested")
    called = []
    manager.run_reserved(job.id, lambda target: called.append(target.id))
    assert job.status == "cancelled" and not called
    restarted = JobManager()
    restarted.enable_persistence(path)
    assert restarted.reserve("agent-run", job.id, resume_interrupted=True).status == "cancelled"
    restarted.run_reserved(job.id, lambda target: called.append(target.id))
    assert not called


def test_reattach_reuses_identity_without_replaying_stale_terminal_event(tmp_path):
    manager, old = interrupted_manager(tmp_path)
    stopped = manager.get(old.id)
    assert stopped.status == "failed" and stopped.snapshot()["interrupted"]
    terminal_seq = stopped.snapshot()["events"][-1]["seq"]
    job = manager.reserve("agent-run", old.id, resume_interrupted=True)
    assert job.id == old.id and job.created == old.created
    assert job.status == "running" and job.error is None
    assert not job.snapshot().get("interrupted")
    events, status = job.events_since(terminal_seq)
    assert status == "running"
    assert not any(event["type"] == "end" for event in events)
    assert events[-1]["seq"] > terminal_seq
    assert events[-1]["step"] == "durable_execution_resumed"
    manager.run_reserved(job.id, lambda target: {"private": "owner-only-result"})
    assert job.status == "done"
    assert b"owner-only-result" not in manager._journal.path.read_bytes()
    again = JobManager()
    again.enable_persistence(manager._journal.path)
    assert again.get(job.id).status == "done"
    assert again.get(job.id).snapshot()["next_event_seq"] == job.snapshot()["next_event_seq"]


def test_cancel_requested_before_crash_cannot_be_reopened(tmp_path):
    manager, old = interrupted_manager(tmp_path, cancelled=True)
    job = manager.reserve("agent-run", old.id, resume_interrupted=True)
    assert job.status == "cancelled"
    assert job.cancel_requested
    called = []
    manager.run_reserved(job.id, lambda target: called.append(target.id))
    assert not called


def test_duplicate_live_claim_is_rejected_and_cancellation_reaches_owner(tmp_path):
    manager = JobManager()
    manager.enable_persistence(tmp_path / "jobs.sqlite3")
    job = manager.reserve("agent-run", "same-live-id")
    entered = threading.Event()
    stopped = threading.Event()
    calls = []
    def runner(target):
        calls.append(target.id)
        target.register_cancel_callback(stopped.set)
        entered.set()
        assert stopped.wait(3)
    thread = threading.Thread(target=manager.run_reserved, args=(job.id, runner))
    thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(JobJournalError, match="already_claimed"):
            manager.run_reserved(job.id, runner)
        assert job.request_cancel()
    finally:
        stopped.set()
        thread.join(3)
    assert not thread.is_alive()
    assert calls == [job.id]
    assert job.status == "cancelled"


def test_failed_resume_persistence_never_replaces_the_interrupted_job(tmp_path, monkeypatch):
    manager, old = interrupted_manager(tmp_path)
    interrupted = manager.get(old.id)
    def unavailable(receipt):
        raise JobJournalError("job_journal_write_failed")
    monkeypatch.setattr(manager._journal, "reopen_interrupted", unavailable)
    with pytest.raises(JobJournalError, match="write_failed"):
        manager.reserve("agent-run", old.id, resume_interrupted=True)
    assert manager.get(old.id) is interrupted
    assert interrupted.status == "failed"


def test_missing_resume_receipt_fails_closed():
    with pytest.raises(JobJournalError, match="receipt_missing"):
        JobManager().reserve("agent-run", "missing", resume_interrupted=True)


def test_terminal_success_and_owner_guard_failure_cannot_be_reopened(tmp_path):
    manager, old = interrupted_manager(tmp_path)
    job = manager.reserve("agent-run", old.id, resume_interrupted=True)
    def scope_guard(target):
        raise RuntimeError("original_authority_changed")
    manager.run_reserved(job.id, scope_guard)
    assert job.status == "failed"
    assert manager.reserve("agent-run", old.id, resume_interrupted=True) is job
    calls = []
    manager.run_reserved(job.id, lambda target: calls.append(target.id))
    assert not calls
    receipt = {**job.lifecycle_receipt(), "status": "running", "finished": None}
    with pytest.raises(JobJournalError, match="not_interrupted"):
        manager._journal.reopen_interrupted(receipt)
