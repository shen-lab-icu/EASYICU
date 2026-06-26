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

import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional


class Job:
    """A single long task: progress event history + terminal status/result."""

    def __init__(self, job_id: str, kind: str) -> None:
        self.id = job_id
        self.kind = kind
        self.status = "running"  # running | done | failed | cancelled
        self.created = time.time()
        self.events: List[Dict[str, Any]] = []
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.cancel_requested = False
        self.cancel_reason: Optional[str] = None

    def emit(self, event: Dict[str, Any]) -> None:
        """Append a progress event (read by the SSE tailer)."""
        event.setdefault("seq", len(self.events))
        self.events.append(event)

    def finish(
        self, status: str, result: Any = None, error: Optional[str] = None
    ) -> None:
        """Record terminal state. Emits the closing ``end`` event BEFORE flipping
        ``status`` so the SSE tailer always flushes it before breaking."""
        self.result = result
        self.error = error
        self.emit({"type": "end", "status": status, "result": result, "error": error})
        self.status = status

    def request_cancel(self, reason: Optional[str] = None) -> bool:
        """Mark this job for cooperative cancellation.

        Runners check ``cancel_requested`` between phases. We do not force-kill
        Python threads because a runner may be inside a file/database read; the
        request is still surfaced immediately over SSE so the UI is honest.
        """
        if self.status != "running":
            return False
        if not self.cancel_requested:
            self.cancel_requested = True
            self.cancel_reason = reason or "user_requested"
            self.emit({"type": "cancel_requested", "reason": self.cancel_reason})
        return True

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "events": list(self.events),
            "result": self.result,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "cancel_reason": self.cancel_reason,
        }


class JobManager:
    """Owns running jobs and runs each runner on its own daemon thread."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, runner: Callable[[Job], Any]) -> Job:
        """Start ``runner(job)`` on a background thread. ``runner`` emits progress
        via ``job.emit(...)`` and may return a result dict; an exception flips the
        job to ``failed``. Returns the Job immediately (non-blocking)."""
        job = Job(uuid.uuid4().hex[:12], kind)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, runner), daemon=True).start()
        return job

    def _run(self, job: Job, runner: Callable[[Job], Any]) -> None:
        try:
            result = runner(job)
            if job.status == "running":
                if job.cancel_requested:
                    job.finish("cancelled", result=result)
                else:
                    job.finish("done", result=result)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the client
            job.finish("failed", error=str(exc))

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)


MANAGER = JobManager()
