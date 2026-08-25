"""In-process long-task framework for the web server (Stage 3b).

The native UI needs an explicit job model for long tasks. A job runs on a
daemon thread, appends progress events to an in-memory history, and flips to a
terminal status. The SSE endpoint in ``app.py`` replays the history then tails
live events, so a subscriber that connects late (or reconnects) still sees
every event.

This is the reusable foundation: the convert job (3b) is the first user; the
extract/export job (3c) and the research-agent run (Stage 5) drive the same
``JobManager.submit(kind, runner)`` contract.

Local-first: jobs run in the user's own server process; nothing leaves the
machine. State is intentionally in-memory — a job does not outlive the process.
"""

from __future__ import annotations

import re
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


_SCIENTIFIC_ACTION_ID = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_SCIENTIFIC_ACTION_STATUSES = frozenset(
    {"direct", "composed", "alternative", "unavailable"}
)


def _scientific_action_projection(value: Any) -> Optional[Dict[str, Any]]:
    """Validate the one browser-safe exception payload this generic owner accepts."""

    if not isinstance(value, dict) or value.get("schema_version") != (
        "easyicu.scientific_action_resolution/1"
    ):
        return None
    status = str(value.get("status") or "")
    issue_code = str(value.get("issue_code") or "")
    requested = str(value.get("requested_action_id") or "")
    detail = str(value.get("detail") or "")
    if (
        status not in _SCIENTIFIC_ACTION_STATUSES
        or not re.fullmatch(r"[a-z][a-z0-9_]{2,120}", issue_code)
        or (requested and not _SCIENTIFIC_ACTION_ID.fullmatch(requested))
        or len(detail) > 3_000
        or not isinstance(value.get("requires_user_confirmation"), bool)
    ):
        return None

    def _action_ids(key: str) -> Optional[List[str]]:
        raw = value.get(key)
        if not isinstance(raw, list) or len(raw) > 16:
            return None
        items = [str(item) for item in raw]
        return items if all(_SCIENTIFIC_ACTION_ID.fullmatch(item) for item in items) else None

    selected = _action_ids("selected_action_ids")
    alternatives = _action_ids("alternative_action_ids")
    requirements = value.get("missing_requirements")
    if (
        selected is None
        or alternatives is None
        or not isinstance(requirements, list)
        or len(requirements) > 16
        or any(len(str(item)) > 500 for item in requirements)
    ):
        return None
    return {
        "schema_version": "easyicu.scientific_action_resolution/1",
        "status": status,
        "requested_action_id": requested,
        "selected_action_ids": selected,
        "alternative_action_ids": alternatives,
        "missing_requirements": [str(item) for item in requirements],
        "issue_code": issue_code,
        "requires_user_confirmation": value["requires_user_confirmation"],
        "detail": detail,
    }


class JobCapacityError(RuntimeError):
    def __init__(self, *, max_running: int, running: int) -> None:
        self.max_running = max_running
        self.running = running
        super().__init__(f"running job limit reached ({running}/{max_running})")


class Job:
    """A single long task: progress event history + terminal status/result."""

    def __init__(self, job_id: str, kind: str) -> None:
        self._lock = threading.RLock()
        self.id = job_id
        self.kind = kind
        self.status = "running"  # running | done | failed | cancelled
        self.created = time.time()
        self.finished: Optional[float] = None
        self.events: List[Dict[str, Any]] = []
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.cancel_requested = False
        self.cancel_reason: Optional[str] = None
        self._cancel_callbacks: Dict[int, Callable[[], None]] = {}
        self._next_cancel_callback_id = 0
        # Some runners return a terminal user-facing result while an
        # uninterruptible reader finishes cooperatively in the background.
        # Such a job remains a real local-capacity consumer until that reader
        # exits, even though SSE has already published cancelled/failed.
        self.draining = False

    def _append_event_locked(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload["seq"] = len(self.events)
        self.events.append(payload)

    def emit(self, event: Dict[str, Any]) -> bool:
        """Append one non-terminal event while the job is still running."""
        with self._lock:
            if self.status != "running" or event.get("type") == "end":
                return False
            self._append_event_locked(event)
            return True

    def finish(
        self, status: str, result: Any = None, error: Optional[str] = None
    ) -> bool:
        """Record terminal state. Emits the closing ``end`` event BEFORE flipping
        ``status`` so the SSE tailer always flushes it before breaking."""
        if status not in {"done", "failed", "cancelled"}:
            raise ValueError(f"invalid terminal job status: {status}")
        with self._lock:
            if self.status != "running":
                return False
            self.result = result
            self.error = error
            self._append_event_locked(
                {"type": "end", "status": status, "result": result, "error": error}
            )
            self.finished = time.time()
            self.status = status
            self._cancel_callbacks.clear()
            return True

    def complete_from_runner(self, result: Any = None) -> bool:
        """Atomically choose ``done`` versus ``cancelled`` for a runner result."""
        with self._lock:
            status = "cancelled" if self.cancel_requested else "done"
            return self.finish(status, result=result)

    def fail_from_runner(self, error: str, result: Any = None) -> bool:
        """Atomically let an accepted cancellation win over a late failure."""
        with self._lock:
            if self.cancel_requested:
                return self.finish("cancelled")
            return self.finish("failed", result=result, error=error)

    def request_cancel(self, reason: Optional[str] = None) -> bool:
        """Mark this job cancelled and notify registered interrupt owners."""
        callbacks: List[Callable[[], None]] = []
        with self._lock:
            if self.status != "running":
                return False
            if not self.cancel_requested:
                self.cancel_requested = True
                self.cancel_reason = reason or "user_requested"
                self._append_event_locked(
                    {"type": "cancel_requested", "reason": self.cancel_reason}
                )
                callbacks = list(self._cancel_callbacks.values())
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Job %s cancellation callback failed", self.id)
        return True

    def register_cancel_callback(
        self, callback: Callable[[], None]
    ) -> Callable[[], None]:
        """Register one owner-provided interrupt and return its unregister hook.

        The callback runs at most once for the first accepted cancellation. If
        cancellation won the race before registration, it runs immediately.
        """
        callback_id: Optional[int] = None
        run_now = False
        with self._lock:
            if self.status == "running" and not self.cancel_requested:
                callback_id = self._next_cancel_callback_id
                self._next_cancel_callback_id += 1
                self._cancel_callbacks[callback_id] = callback
            elif self.status == "running" and self.cancel_requested:
                run_now = True

        if run_now:
            try:
                callback()
            except Exception:
                logger.exception(
                    "Job %s late cancellation callback failed", self.id
                )

        def unregister() -> None:
            if callback_id is None:
                return
            with self._lock:
                self._cancel_callbacks.pop(callback_id, None)

        return unregister

    def begin_draining(self) -> None:
        with self._lock:
            self.draining = True

    def end_draining(self) -> None:
        with self._lock:
            self.draining = False

    def consumes_capacity(self) -> bool:
        with self._lock:
            return self.status == "running" or self.draining

    def is_cancel_requested(self) -> bool:
        with self._lock:
            return self.cancel_requested

    def events_since(self, offset: int) -> tuple[List[Dict[str, Any]], str]:
        """Return a consistent event slice and status for the SSE tailer."""
        with self._lock:
            start = max(0, int(offset))
            return [dict(event) for event in self.events[start:]], self.status

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "created": self.created,
                "finished": self.finished,
                "events": [dict(event) for event in self.events],
                "result": self.result,
                "error": self.error,
                "cancel_requested": self.cancel_requested,
                "cancel_reason": self.cancel_reason,
                "draining": self.draining,
            }


class JobManager:
    """Owns running jobs and runs each runner on its own daemon thread."""

    def __init__(self, max_completed: int = 100, max_running: int = 8) -> None:
        if max_completed < 0:
            raise ValueError("max_completed must be non-negative")
        if max_running <= 0:
            raise ValueError("max_running must be positive")
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_completed = max_completed
        self._max_running = max_running

    def submit(self, kind: str, runner: Callable[[Job], Any]) -> Job:
        """Start ``runner(job)`` on a background thread. ``runner`` emits progress
        via ``job.emit(...)`` and may return a result dict; an exception flips the
        job to ``failed``. Returns the Job immediately (non-blocking)."""
        with self._lock:
            self._prune_completed_locked()
            running = sum(job.consumes_capacity() for job in self._jobs.values())
            if running >= self._max_running:
                raise JobCapacityError(
                    max_running=self._max_running,
                    running=running,
                )
            job = Job(uuid.uuid4().hex[:12], kind)
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, runner), daemon=True).start()
        return job

    def _run(self, job: Job, runner: Callable[[Job], Any]) -> None:
        try:
            result = runner(job)
            job.complete_from_runner(result)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the client
            message = str(exc)
            code = str(getattr(exc, "code", "") or "").strip()
            if code and re.fullmatch(r"[a-z][a-z0-9_]{2,120}", code):
                if not message.startswith(f"{code}:"):
                    message = f"{code}: {message}"
            projection = _scientific_action_projection(
                getattr(exc, "user_action_required", None)
            )
            if projection is not None:
                alternatives = [
                    str(value)
                    for value in projection.get("alternative_action_ids", [])
                    if str(value).strip()
                ]
                label = (
                    "Scientific method needs your confirmation: "
                    + ", ".join(alternatives)
                    if alternatives
                    else "Scientific method is not executable with the current inputs"
                )
                job.emit(
                    {
                        # ``progress`` is deliberately used here: both Classic
                        # Agent and Guided Pi already stream this event type, so
                        # the recoverable gap is visible without a second UI
                        # event/rendering stack.  The exact typed payload is
                        # retained for the assistant and the terminal result.
                        "type": "progress",
                        "step": "scientific_action_gap",
                        "status": "action_required",
                        "reason_code": str(projection.get("issue_code") or ""),
                        "label": label,
                        "action": projection,
                    }
                )
                job.fail_from_runner(
                    message,
                    result={"action_required": projection},
                )
                return
            job.fail_from_runner(message)
        finally:
            with self._lock:
                self._prune_completed_locked()

    def _prune_completed_locked(self) -> None:
        completed = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status != "running" and not job.consumes_capacity()
            ),
            key=lambda job: job.finished or job.created,
        )
        for job in completed[: max(0, len(completed) - self._max_completed)]:
            self._jobs.pop(job.id, None)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)


MANAGER = JobManager()
