from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from easyicu.research_agent.authority import run_heartbeat as heartbeat_owner
from easyicu.research_agent.authority.run_heartbeat import (
    RUN_HEARTBEAT_SCHEMA,
    RunHeartbeatSupervisor,
    bind_active_run_heartbeat,
    finish_active_run_heartbeat,
    record_active_run_progress,
    run_heartbeat_scope,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_heartbeat_persists_phase_deadlines_and_stops_cleanly(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    with run_heartbeat_scope(run_id=run_dir.name):
        path = bind_active_run_heartbeat(
            run_dir,
            interval_seconds=0.01,
            task_timeout_seconds=120.0,
        )
        assert path is not None
        record_active_run_progress(
            stage="runner",
            message="Running standard executor script for step_06.",
            status="running",
            step_id="step_06",
            phase_timeout_seconds=30.0,
            run_id=run_dir.name,
        )
        active = _load(path)
        assert active["schema_version"] == RUN_HEARTBEAT_SCHEMA
        assert active["active"] is True
        assert active["stage"] == "runner"
        assert active["step_id"] == "step_06"
        assert active["phase_timeout_seconds"] == 30.0
        assert active["phase_deadline_at"]
        assert active["task_timeout_seconds"] == 120.0
        assert active["task_deadline_at"]

    terminal = _load(run_dir / "run_heartbeat.json")
    assert terminal["active"] is False
    assert terminal["stage_status"] == "inactive"
    assert terminal["terminal_reason"] == "call_returned"


def test_worker_thread_progress_resolves_supervisor_by_run_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_threaded"
    with run_heartbeat_scope(run_id=run_dir.name):
        path = bind_active_run_heartbeat(run_dir, interval_seconds=30.0)
        assert path is not None

        worker = threading.Thread(
            target=lambda: record_active_run_progress(
                stage="critic",
                message="Reviewing bounded scientific summary.",
                step_id="step_04",
                run_id=run_dir.name,
            )
        )
        worker.start()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        payload = _load(path)
        assert payload["stage"] == "critic"
        assert payload["step_id"] == "step_04"


def test_heartbeat_records_exception_type_without_exception_text(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_failed"
    secret = "do-not-persist-this-message"
    with pytest.raises(RuntimeError, match=secret):
        with run_heartbeat_scope(run_id=run_dir.name):
            bind_active_run_heartbeat(run_dir, interval_seconds=30.0)
            raise RuntimeError(secret)

    payload = _load(run_dir / "run_heartbeat.json")
    assert payload["active"] is False
    assert payload["terminal_reason"] == "call_failed:RuntimeError"
    assert secret not in json.dumps(payload)


def test_completed_heartbeat_is_immutable_through_scope_exit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_completed"
    with run_heartbeat_scope(run_id=run_dir.name) as supervisor:
        path = bind_active_run_heartbeat(run_dir, interval_seconds=3600.0)
        assert path is not None
        finish_active_run_heartbeat(run_id=run_dir.name)
        terminal_bytes = path.read_bytes()
        terminal = _load(path)
        assert terminal["active"] is False
        assert terminal["terminal_reason"] == "workflow_completed"
        assert supervisor._thread is not None
        assert not supervisor._thread.is_alive()

        supervisor.finish(terminal_reason="call_returned")
        supervisor.flush()
        record_active_run_progress(stage="late", message="Delayed worker callback.")
        with pytest.raises(RuntimeError, match="already finished"):
            bind_active_run_heartbeat(run_dir)
        with pytest.raises(RuntimeError, match="already finished"):
            supervisor.bind(tmp_path / "another_run")
        assert not (tmp_path / "another_run").exists()
        assert path.read_bytes() == terminal_bytes

    assert path.read_bytes() == terminal_bytes


def test_finished_unbound_heartbeat_cannot_start_later(tmp_path: Path) -> None:
    supervisor = RunHeartbeatSupervisor(run_id="unbound")
    supervisor.finish(terminal_reason="workflow_completed")
    supervisor.finish(terminal_reason="call_returned")
    supervisor.record_progress(stage="late", message="Not bound.")
    supervisor.flush()
    with pytest.raises(RuntimeError, match="already finished"):
        supervisor.bind(tmp_path / "unbound")
    assert supervisor._thread is None
    assert not (tmp_path / "unbound").exists()


def test_finish_active_heartbeat_requires_matching_current_run(tmp_path: Path) -> None:
    finish_active_run_heartbeat(run_id="outside_scope")
    with run_heartbeat_scope(run_id="current"):
        path = bind_active_run_heartbeat(tmp_path / "current", interval_seconds=3600.0)
        assert path is not None
        before = path.read_bytes()
        with pytest.raises(RuntimeError, match="different run"):
            finish_active_run_heartbeat(run_id="another_run")
        assert path.read_bytes() == before
        assert _load(path)["active"] is True


def test_finish_drains_writers_before_one_final_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start_periodic_write = threading.Event()
    write_started = threading.Event()
    release_write = threading.Event()
    queued_flush_started = threading.Event()
    finish_returned = threading.Event()
    errors: list[BaseException] = []
    writes: list[dict] = []
    supervisor = RunHeartbeatSupervisor(run_id="run_concurrent")
    original_write = heartbeat_owner._atomic_write_json

    def periodic_loop() -> None:
        try:
            assert start_periodic_write.wait(timeout=5.0)
            supervisor.flush()
        except BaseException as exc:
            errors.append(exc)

    def controlled_write(path: Path, payload: dict) -> None:
        if threading.current_thread() is supervisor._thread:
            write_started.set()
            assert release_write.wait(timeout=5.0)
        original_write(path, payload)
        writes.append(payload)

    def start_worker(action) -> threading.Thread:
        def guarded() -> None:
            try:
                action()
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=guarded, daemon=True)
        worker.start()
        return worker

    def queued_flush() -> None:
        queued_flush_started.set()
        supervisor.flush()

    def complete() -> None:
        supervisor.finish(terminal_reason="workflow_completed")
        finish_returned.set()

    monkeypatch.setattr(supervisor, "_heartbeat_loop", periodic_loop)
    monkeypatch.setattr(heartbeat_owner, "_atomic_write_json", controlled_write)
    path = supervisor.bind(tmp_path / supervisor.run_id, interval_seconds=3600.0)
    writes.clear()
    workers: list[threading.Thread] = []
    try:
        start_periodic_write.set()
        assert write_started.wait(timeout=5.0)
        workers.append(start_worker(queued_flush))
        assert queued_flush_started.wait(timeout=5.0)
        workers.append(start_worker(complete))
        assert supervisor._stop_event.wait(timeout=5.0)
        workers.append(
            start_worker(lambda: supervisor.finish(terminal_reason="call_returned"))
        )
        assert not finish_returned.is_set()
        supervisor.record_progress(stage="late", message="No longer active.")
        with pytest.raises(RuntimeError, match="already finished"):
            supervisor.bind(tmp_path / "late_binding")
        release_write.set()
    finally:
        start_periodic_write.set()
        release_write.set()
        for worker in workers:
            worker.join(timeout=5.0)
        if all(not worker.is_alive() for worker in workers):
            supervisor.finish(terminal_reason="cleanup")

    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert supervisor._thread is not None
    assert not supervisor._thread.is_alive()
    assert finish_returned.is_set()
    assert [payload["active"] for payload in writes] == [True, False]
    terminal = _load(path)
    assert terminal == writes[-1]
    assert terminal["terminal_reason"] == "workflow_completed"
    assert terminal["stage"] == "run"
    assert not (tmp_path / "late_binding").exists()


def test_final_heartbeat_write_error_remains_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = RunHeartbeatSupervisor(run_id="write_error")
    path = supervisor.bind(tmp_path / "write_error", interval_seconds=3600.0)
    before = path.read_bytes()
    attempts: list[dict] = []

    def fail_write(path: Path, payload: dict) -> None:
        attempts.append(payload)
        raise OSError("diagnostic disk write unavailable")

    monkeypatch.setattr(heartbeat_owner, "_atomic_write_json", fail_write)
    supervisor.finish(terminal_reason="workflow_completed")
    supervisor.finish(terminal_reason="call_returned")
    supervisor.flush()
    supervisor.record_progress(stage="late", message="No new write attempt.")
    assert len(attempts) == 1
    assert attempts[0]["active"] is False
    assert supervisor._last_write_error_type == "OSError"
    assert path.read_bytes() == before
    assert supervisor._thread is not None
    assert not supervisor._thread.is_alive()
