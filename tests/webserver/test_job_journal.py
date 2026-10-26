"""Job lifecycle recovery is durable; execution authority is never replayed."""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from easyicu.webserver.job_journal import JobJournal, JobJournalError
from easyicu.webserver.jobs import Job, JobManager


def _finish(manager: JobManager, kind="synthetic", result=None) -> Job:
    ready = threading.Event()

    def runner(job):
        ready.wait(2)
        return result

    job = manager.submit(kind, runner)
    ready.set()
    deadline = time.monotonic() + 3
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.status == "done"
    return job


def test_receipts_survive_restart_without_serializing_payloads(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    manager = JobManager()
    manager.enable_persistence(path)
    job = _finish(manager, result={"token": "secret-not-a-receipt", "patient": "private-row"})

    restored = JobManager()
    restored.enable_persistence(path)
    recovered = restored.get(job.id)
    assert recovered is not None
    assert recovered.status == "done"
    assert recovered.result is None
    assert recovered.snapshot()["result_omitted"] is True
    assert recovered.snapshot()["recovered"] is True
    assert recovered.snapshot()["next_event_seq"] == job.snapshot()["next_event_seq"]
    assert b"secret-not-a-receipt" not in path.read_bytes()
    assert b"private-row" not in path.read_bytes()
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600

    again = JobManager()
    again.enable_persistence(path)
    assert again.get(job.id).snapshot()["next_event_seq"] == recovered.snapshot()["next_event_seq"]


def test_process_crash_restores_sse_terminal_cursor_and_never_reruns(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    marker = tmp_path / "executions.txt"
    script = """
import json, os, sys, threading
from pathlib import Path
from easyicu.webserver.jobs import JobManager
ready = threading.Event()
manager = JobManager()
manager.enable_persistence(Path(sys.argv[1]))
def runner(job):
    with Path(sys.argv[2]).open('a') as output:
        output.write('one execution\\n')
    job.emit({'type': 'progress', 'step': 'execute', 'private': 'not-persisted'})
    ready.set()
    threading.Event().wait(60)
job = manager.submit('synthetic', runner)
assert ready.wait(5)
os.write(1, json.dumps({'id': job.id, 'next': job.snapshot()['next_event_seq']}).encode())
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path), str(marker)],
        capture_output=True, text=True, check=True, timeout=15,
    )
    original = json.loads(completed.stdout)
    manager = JobManager()
    manager.enable_persistence(path)
    recovered = manager.get(original["id"])
    assert recovered.status == "failed"
    assert recovered.error == "job_interrupted_by_server_restart"
    events, status = recovered.events_since(original["next"])
    assert status == "failed"
    assert events[0]["step"] == "history_truncated"
    assert events[-1]["type"] == "end"
    assert events[-1]["seq"] >= original["next"]
    assert marker.read_text() == "one execution\n"
    assert b"not-persisted" not in path.read_bytes()
    assert recovered.consumes_capacity() is False

    again = JobManager()
    again.enable_persistence(path)
    repeated, repeated_status = again.get(original["id"]).events_since(original["next"])
    assert repeated_status == status
    assert repeated[-1] == events[-1]


def test_streaming_reserves_cursors_without_per_token_disk_writes(tmp_path):
    journal = JobJournal(tmp_path / "jobs.sqlite3")
    writes = []

    def save(receipt):
        writes.append(receipt)
        journal.save(receipt)

    job = Job("streaming-job", "pi-copilot-message", persist=save)
    journal.create(job.lifecycle_receipt())
    for number in range(600):
        job.emit({"type": "progress", "delta": str(number)})
    assert len(writes) == 3
    assert journal.records()[0]["next_event_seq"] >= job.snapshot()["next_event_seq"]
    job.finish("done")
    assert len(writes) == 4
    assert journal.records()[0]["next_event_seq"] == job.snapshot()["next_event_seq"]


def test_failed_cursor_reservation_terminates_without_reusing_delivered_ids(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    manager = JobManager()
    manager.enable_persistence(path)
    start = threading.Event()

    def disk_unavailable(receipt):
        raise JobJournalError("job_journal_write_failed")

    monkeypatch.setattr(manager._journal, "save", disk_unavailable)

    def runner(job):
        assert start.wait(2)
        for number in range(600):
            job.emit({"type": "progress", "delta": str(number)})

    job = manager.submit("synthetic", runner)
    start.set()
    deadline = time.monotonic() + 3
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.status == "failed"
    snapshot = job.snapshot()
    assert snapshot["persistence_error"] == "job_journal_unavailable"
    assert snapshot["error"] == "job_journal_unavailable"
    assert snapshot["events"][-1]["type"] == "end"
    delivered = {event["seq"] for event in snapshot["events"]}
    assert delivered == set(range(256))
    assert job._journal_cursor == 256
    assert JobJournal(path).records()[0]["next_event_seq"] == 256

    restored = JobManager()
    restored.enable_persistence(path)
    recovered = restored.get(job.id)
    recovery_events, status = recovered.events_since(max(delivered) + 1)
    assert status == "failed"
    assert recovery_events[-1]["type"] == "end"
    assert all(event["seq"] not in delivered for event in recovery_events)
    assert recovery_events[-1]["seq"] == 256


def test_cancel_callbacks_still_run_when_next_cursor_reservation_fails(tmp_path):
    journal = JobJournal(tmp_path / "jobs.sqlite3")
    job = Job("cancel-at-boundary", "synthetic", persist=journal.save)
    journal.create(job.lifecycle_receipt())
    for _ in range(255):
        job.emit({"type": "progress"})
    callback = threading.Event()
    job.register_cancel_callback(callback.set)

    def disk_unavailable(receipt):
        raise JobJournalError("job_journal_write_failed")

    job._persist = disk_unavailable
    assert job.request_cancel() is False
    assert callback.is_set()
    assert job.cancel_requested is True
    assert job.snapshot()["persistence_error"] == "job_journal_unavailable"
    assert job.snapshot()["next_event_seq"] == 255
    job.complete_from_runner()
    assert job.status == "cancelled"
    assert job.snapshot()["events"][-1]["type"] == "end"
    assert job.snapshot()["events"][-1]["seq"] == 255


def test_cancel_retry_retries_receipt_without_repeating_interrupt(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    journal = JobJournal(path)
    job = Job("retry-cancel-receipt", "synthetic", persist=journal.save)
    journal.create(job.lifecycle_receipt())
    job.emit({"type": "progress"})
    callbacks = []
    attempts = []
    job.register_cancel_callback(lambda: callbacks.append("interrupt"))

    def unavailable(receipt):
        attempts.append(receipt["cancel_requested"])
        raise JobJournalError("job_journal_write_failed")

    job._persist = unavailable
    assert job.request_cancel("user_requested") is False
    assert job.request_cancel("user_requested") is False
    assert attempts == [True, True]
    assert callbacks == ["interrupt"]
    assert journal.records()[0]["cancel_requested"] == 0
    assert sum(event["type"] == "cancel_requested" for event in job.events) == 1

    job._persist = journal.save
    assert job.request_cancel("user_requested") is True
    assert callbacks == ["interrupt"]
    assert journal.records()[0]["cancel_requested"] == 1
    restarted = JobManager()
    restarted.enable_persistence(path)
    assert restarted.get(job.id).status == "cancelled"


def test_missing_initial_cursor_cannot_leave_failed_job_running():
    def disk_unavailable(receipt):
        raise JobJournalError("job_journal_write_failed")

    job = Job("unreserved-job", "synthetic", persist=disk_unavailable)
    JobManager()._run(job, lambda target: target.emit({"type": "progress"}))
    assert job.status == "failed"
    assert job.events == []
    assert job.snapshot()["persistence_error"] == "job_journal_unavailable"


def test_scientific_failure_still_terminates_when_error_progress_cannot_persist():
    from easyicu.research_agent.planning.scientific_action_catalog import (
        ScientificActionGapError, resolve_scientific_action_request,
    )

    def disk_unavailable(receipt):
        raise JobJournalError("job_journal_write_failed")

    def runner(job):
        raise ScientificActionGapError(resolve_scientific_action_request(
            analysis_type="prediction_model", action_id="prediction.external_validation",
        ))

    job = Job("unreserved-failure", "agent-run", persist=disk_unavailable)
    JobManager()._run(job, runner)
    assert job.status == "failed"
    assert job.result["action_required"]["requires_user_confirmation"] is True
    assert job.snapshot()["persistence_error"] == "job_journal_unavailable"


def test_accepted_cancellation_wins_when_process_disappears(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    journal = JobJournal(path)
    job = Job("cancelled-job", "synthetic", persist=journal.save)
    journal.create(job.lifecycle_receipt())
    assert job.request_cancel("private text that must not be persisted")
    manager = JobManager()
    manager.enable_persistence(path)
    recovered = manager.get(job.id)
    assert recovered.status == "cancelled"
    assert recovered.cancel_requested is True
    assert recovered.snapshot()["interrupted"] is True
    assert b"private text" not in path.read_bytes()


def test_acceptance_failure_does_not_start_runner_and_corruption_is_visible(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    manager = JobManager()
    manager.enable_persistence(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE jobs")
    executed = threading.Event()
    with pytest.raises(JobJournalError, match="job_journal_write_failed"):
        manager.submit("synthetic", lambda job: executed.set())
    assert not executed.is_set()
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not a sqlite database")
    with pytest.raises(JobJournalError, match="job_journal_unavailable"):
        JobManager().enable_persistence(corrupt)


def test_same_process_startup_is_idempotent_and_cannot_switch_private_home(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    manager = JobManager()
    manager.enable_persistence(path)
    job = _finish(manager, result={"live": "result"})
    manager.enable_persistence(path)
    assert manager.get(job.id) is job
    assert "recovered" not in job.snapshot()
    with pytest.raises(JobJournalError, match="job_journal_path_changed"):
        manager.enable_persistence(tmp_path / "another-home.sqlite3")


def test_journal_refuses_symlink_and_unknown_schema_without_overwriting(tmp_path):
    original = tmp_path / "original"
    original.write_text("private unrelated file")
    link = tmp_path / "link.sqlite3"
    try:
        link.symlink_to(original)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(JobJournalError):
        JobJournal(link)
    assert original.read_text() == "private unrelated file"

    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(JobJournalError, match="job_journal_unsupported_version"):
        JobJournal(path)


def test_live_write_failure_remains_visible_without_reexecuting_operation(tmp_path, monkeypatch):
    manager = JobManager()
    manager.enable_persistence(tmp_path / "jobs.sqlite3")

    def fail_save(record):
        raise JobJournalError("job_journal_write_failed")

    monkeypatch.setattr(manager._journal, "save", fail_save)
    job = _finish(manager, result={"already_executed": True})
    assert job.snapshot()["persistence_error"] == "job_journal_unavailable"
    assert job.result == {"already_executed": True}


def test_journal_deduplicates_identity_and_keeps_terminal_receipts_immutable(tmp_path):
    journal = JobJournal(tmp_path / "jobs.sqlite3")
    job = Job("same-job", "synthetic", persist=journal.save)
    journal.create(job.lifecycle_receipt())
    with pytest.raises(JobJournalError, match="job_journal_write_failed"):
        journal.create(job.lifecycle_receipt())
    job.finish("done")
    changed = {**job.lifecycle_receipt(), "status": "running", "finished": None}
    with pytest.raises(JobJournalError, match="job_journal_terminal_receipt_changed"):
        journal.save(changed)
    assert journal.records()[0]["status"] == "done"


def test_journal_retention_is_bounded_without_deleting_active_jobs(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    journal = JobJournal(path, max_completed=2)
    active = Job("still-running", "synthetic")
    journal.create(active.lifecycle_receipt())
    for number in range(4):
        job = Job(f"finished-{number}", "synthetic", persist=journal.save)
        journal.create(job.lifecycle_receipt())
        job.finish("done")
    records = journal.records()
    assert len(records) == 3
    assert {row["id"] for row in records} == {"still-running", "finished-2", "finished-3"}


def test_lifecycle_api_reconnects_to_restored_job(tmp_path, monkeypatch):
    from easyicu.webserver import jobs
    from easyicu.webserver.routes.jobs import jobs_get, jobs_cancel

    original = JobManager()
    original.enable_persistence(tmp_path / "jobs.sqlite3")
    job = _finish(original)
    manager = JobManager()
    manager.enable_persistence(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(jobs, "MANAGER", manager)
    assert jobs_get(job.id)["status"] == "done"
    assert jobs_get(job.id)["result_omitted"] is True
    assert jobs_cancel(job.id)["cancel_request_accepted"] is False


def test_startup_loads_journal_after_lease_and_releases_lease_on_failure(monkeypatch):
    app = importlib.import_module("easyicu.webserver.app")
    jobs = importlib.import_module("easyicu.webserver.jobs")
    calls = []

    def fail_load(path):
        calls.append("journal")
        raise JobJournalError("job_journal_unavailable")

    monkeypatch.setattr(app, "acquire_single_process_lease", lambda: calls.append("lease"))
    monkeypatch.setattr(app, "release_single_process_lease", lambda: calls.append("release"))
    monkeypatch.setattr(jobs.MANAGER, "enable_persistence", fail_load)
    with pytest.raises(JobJournalError):
        app._acquire_web_deployment_lease()
    assert calls == ["lease", "journal", "release"]
