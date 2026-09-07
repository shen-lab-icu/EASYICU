"""PHI-safe liveness receipts for one research-agent run.

The heartbeat is diagnostic supervision, not scientific authority.  It records
only bounded orchestration metadata already present in progress events and
never inspects cohort rows, prompts, generated code, or result artefacts.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


RUN_HEARTBEAT_SCHEMA = "easyicu.run_heartbeat/1"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
_MAX_MESSAGE_CHARS = 500


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


class RunHeartbeatSupervisor:
    """Periodically persist bounded liveness state for one run."""

    def __init__(self, *, run_id: str) -> None:
        self.run_id = str(run_id)
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._finish_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._path: Optional[Path] = None
        self._interval_seconds = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        self._started_at = _utc_now()
        self._started_monotonic = time.monotonic()
        self._last_progress_at = self._started_at
        self._last_progress_monotonic = self._started_monotonic
        self._stage_started_at = self._started_at
        self._stage_started_monotonic = self._started_monotonic
        self._stage = "run"
        self._stage_status = "starting"
        self._step_id: Optional[str] = None
        self._message = "Run supervision initialized."
        self._phase_timeout_seconds: Optional[float] = None
        self._task_timeout_seconds: Optional[float] = None
        self._active = True
        self._terminal_reason: Optional[str] = None
        self._sequence = 0
        self._last_write_error_type: Optional[str] = None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def bind(
        self,
        run_dir: str | Path,
        *,
        interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        task_timeout_seconds: Optional[float] = None,
    ) -> Path:
        """Bind the supervisor to a validated run directory and start it."""

        interval = float(interval_seconds)
        if interval <= 0:
            raise ValueError("heartbeat interval_seconds must be positive")
        task_timeout = (
            None if task_timeout_seconds is None else float(task_timeout_seconds)
        )
        if task_timeout is not None and task_timeout <= 0:
            raise ValueError("heartbeat task_timeout_seconds must be positive")

        target = Path(run_dir) / "run_heartbeat.json"
        with self._lock:
            if self._stop_event.is_set():
                raise RuntimeError("run heartbeat has already finished")
            if self._path is not None and self._path != target:
                raise RuntimeError("run heartbeat is already bound to another directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            self._path = target
            self._interval_seconds = interval
            self._task_timeout_seconds = task_timeout
            if self._thread is None:
                thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name=f"easyicu-heartbeat-{self.run_id}",
                    daemon=True,
                )
                thread.start()
                self._thread = thread
        self.flush()
        return target

    def record_progress(
        self,
        *,
        stage: str,
        message: str,
        status: str = "running",
        step_id: Optional[str] = None,
        phase_timeout_seconds: Optional[float] = None,
    ) -> None:
        """Record a new bounded phase coordinate without scientific payloads."""

        now = _utc_now()
        now_monotonic = time.monotonic()
        normalized_stage = str(stage or "run")[:120]
        normalized_step = str(step_id)[:200] if step_id else None
        normalized_timeout = (
            None
            if phase_timeout_seconds is None
            else float(phase_timeout_seconds)
        )
        if normalized_timeout is not None and normalized_timeout <= 0:
            normalized_timeout = None
        with self._lock:
            if self._stop_event.is_set():
                return
            if (
                normalized_stage != self._stage
                or normalized_step != self._step_id
                or str(status) != self._stage_status
            ):
                self._stage_started_at = now
                self._stage_started_monotonic = now_monotonic
            self._stage = normalized_stage
            self._stage_status = str(status or "running")[:80]
            self._step_id = normalized_step
            self._message = str(message or "")[:_MAX_MESSAGE_CHARS]
            self._phase_timeout_seconds = normalized_timeout
            self._last_progress_at = now
            self._last_progress_monotonic = now_monotonic
        self.flush()

    def finish(self, *, terminal_reason: str) -> None:
        """Drain writers and persist one final inactive receipt exactly once."""

        with self._finish_lock:
            with self._lock:
                if not self._active:
                    return
                self._stop_event.set()
                thread = self._thread
            # Never join while holding a lock a pending flush needs. Returning
            # before a writer drains would let it invalidate the run receipt.
            if thread is not None and thread is not threading.current_thread():
                thread.join()
            with self._write_lock:
                with self._lock:
                    self._active = False
                    self._terminal_reason = str(terminal_reason)[:160]
                    self._stage_status = "inactive"
                self._write_snapshot()

    def flush(self) -> None:
        """Atomically write the current snapshot when a path is bound."""

        with self._write_lock:
            if self._stop_event.is_set():
                return
            self._write_snapshot()

    def _write_snapshot(self) -> None:
        """Write under the caller's write lock, including the final snapshot."""

        with self._lock:
            path = self._path
            if path is None:
                return
            self._sequence += 1
            payload = self._snapshot_locked()
        try:
            _atomic_write_json(path, payload)
        except OSError as exc:
            # Runtime supervision is diagnostic. A transient heartbeat write
            # failure must not replace the pipeline's more precise scientific
            # or execution outcome.
            with self._lock:
                self._last_write_error_type = type(exc).__name__

    def _snapshot_locked(self) -> dict[str, Any]:
        now = _utc_now()
        now_monotonic = time.monotonic()
        process_elapsed = max(0.0, now_monotonic - self._started_monotonic)
        quiet_for = max(0.0, now_monotonic - self._last_progress_monotonic)
        stage_elapsed = max(0.0, now_monotonic - self._stage_started_monotonic)
        task_deadline = (
            self._started_at + timedelta(seconds=self._task_timeout_seconds)
            if self._task_timeout_seconds is not None
            else None
        )
        phase_deadline = (
            self._stage_started_at + timedelta(seconds=self._phase_timeout_seconds)
            if self._phase_timeout_seconds is not None
            else None
        )
        return {
            "schema_version": RUN_HEARTBEAT_SCHEMA,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "active": self._active,
            "terminal_reason": self._terminal_reason,
            "started_at": _iso(self._started_at),
            "heartbeat_at": _iso(now),
            "sequence": self._sequence,
            "process_elapsed_seconds": round(process_elapsed, 3),
            "last_progress_at": _iso(self._last_progress_at),
            "quiet_for_seconds": round(quiet_for, 3),
            "stage": self._stage,
            "stage_status": self._stage_status,
            "stage_started_at": _iso(self._stage_started_at),
            "stage_elapsed_seconds": round(stage_elapsed, 3),
            "step_id": self._step_id,
            "message": self._message,
            "phase_timeout_seconds": self._phase_timeout_seconds,
            "phase_deadline_at": _iso(phase_deadline) if phase_deadline else None,
            "phase_timeout_exceeded": bool(
                phase_deadline is not None and now >= phase_deadline
            ),
            "task_timeout_seconds": self._task_timeout_seconds,
            "task_deadline_at": _iso(task_deadline) if task_deadline else None,
            "task_timeout_exceeded": bool(
                task_deadline is not None and now >= task_deadline
            ),
            "last_write_error_type": self._last_write_error_type,
        }

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self.flush()


_ACTIVE_HEARTBEAT: ContextVar[Optional[RunHeartbeatSupervisor]] = ContextVar(
    "easyicu_active_run_heartbeat", default=None
)
_REGISTRY_LOCK = threading.Lock()
_ACTIVE_BY_RUN_ID: dict[str, RunHeartbeatSupervisor] = {}


@contextmanager
def run_heartbeat_scope(*, run_id: str) -> Iterator[RunHeartbeatSupervisor]:
    """Own one supervisor for the lifetime of a run or resume call."""

    supervisor = RunHeartbeatSupervisor(run_id=run_id)
    token = _ACTIVE_HEARTBEAT.set(supervisor)
    try:
        yield supervisor
    except BaseException as exc:
        supervisor.finish(terminal_reason=f"call_failed:{type(exc).__name__}")
        raise
    else:
        supervisor.finish(terminal_reason="call_returned")
    finally:
        with _REGISTRY_LOCK:
            if _ACTIVE_BY_RUN_ID.get(supervisor.run_id) is supervisor:
                _ACTIVE_BY_RUN_ID.pop(supervisor.run_id, None)
        _ACTIVE_HEARTBEAT.reset(token)


def bind_active_run_heartbeat(
    run_dir: str | Path,
    *,
    interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    task_timeout_seconds: Optional[float] = None,
) -> Optional[Path]:
    """Bind the current run scope, returning ``None`` outside a run."""

    supervisor = _ACTIVE_HEARTBEAT.get()
    if supervisor is None:
        return None
    path = supervisor.bind(
        run_dir,
        interval_seconds=interval_seconds,
        task_timeout_seconds=task_timeout_seconds,
    )
    with _REGISTRY_LOCK:
        _ACTIVE_BY_RUN_ID[supervisor.run_id] = supervisor
    return path


def finish_active_run_heartbeat(*, run_id: str) -> None:
    """Quiesce the current run's heartbeat before sealing its terminal tree."""

    supervisor = _ACTIVE_HEARTBEAT.get()
    if supervisor is None:
        return
    if supervisor.run_id != run_id:
        raise RuntimeError("active run heartbeat belongs to a different run")
    supervisor.finish(terminal_reason="workflow_completed")


def record_active_run_progress(
    *,
    stage: str,
    message: str,
    status: str = "running",
    step_id: Optional[str] = None,
    phase_timeout_seconds: Optional[float] = None,
    run_id: Optional[str] = None,
) -> None:
    """Update the active supervisor from main or worker execution threads."""

    supervisor = _ACTIVE_HEARTBEAT.get()
    if supervisor is None and run_id:
        with _REGISTRY_LOCK:
            supervisor = _ACTIVE_BY_RUN_ID.get(str(run_id))
    if supervisor is None:
        return
    supervisor.record_progress(
        stage=stage,
        message=message,
        status=status,
        step_id=step_id,
        phase_timeout_seconds=phase_timeout_seconds,
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "RUN_HEARTBEAT_SCHEMA",
    "RunHeartbeatSupervisor",
    "bind_active_run_heartbeat",
    "finish_active_run_heartbeat",
    "record_active_run_progress",
    "run_heartbeat_scope",
]
