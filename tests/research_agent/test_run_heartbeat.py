from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from easyicu.research_agent.authority.run_heartbeat import (
    RUN_HEARTBEAT_SCHEMA,
    bind_active_run_heartbeat,
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
