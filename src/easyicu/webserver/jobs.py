"""In-process long-task framework for the web server (Stage 3b).

The native UI needs an explicit job model for long tasks. A job runs on a
daemon thread, appends progress events to an in-memory history, and flips to a
terminal status. The SSE endpoint in ``app.py`` replays the history then tails
live events. History is bounded by event count and serialized bytes; a late
subscriber receives an explicit gap marker plus the retained tail. Durable run
artifacts, not this transient progress history, own scientific evidence.

This is the reusable foundation: the convert job (3b) is the first user; the
extract/export job (3c) and the research-agent run (Stage 5) drive the same
``JobManager.submit(kind, runner)`` contract.

Local-first: jobs run in the user's own server process; nothing leaves the
machine. The Web startup enables a private lifecycle journal. After a restart,
job IDs retain a terminal receipt and unfinished jobs become interrupted. The
journal never replays a runner or grants permission to resume research.
"""

from __future__ import annotations

import re
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from easyicu.webserver.job_journal import JobJournalError

if TYPE_CHECKING:
    from easyicu.webserver.host_action_contracts import HostActionTag


logger = logging.getLogger(__name__)

_JOURNAL_CURSOR_BLOCK = 256


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

    def __init__(self, job_id: str, kind: str, *, max_events: int = 1000,
                 max_history_bytes: int = 512 * 1024, max_event_bytes: int = 32 * 1024,
                 persist: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        if max_events < 1 or min(max_history_bytes, max_event_bytes) < 512:
            raise ValueError("job history requires positive count and byte budgets >= 512")
        self._lock = threading.RLock()
        self.id = job_id
        self.kind = kind
        self.status = "running"  # running | done | failed | cancelled
        self.created = time.time()
        self.finished: Optional[float] = None
        self.events: List[Dict[str, Any]] = []
        self._max_events = max_events
        self._max_history_bytes = max_history_bytes
        self._max_event_bytes = min(max_event_bytes, max_history_bytes)
        self._event_sizes: List[int] = []
        self._history_bytes = 0
        self._next_seq = 0
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
        # The host decision this job answers (``webserver.host_action_jobs``).
        # Not in the snapshot: the browser reads jobs through it, and the tag
        # names the conversation that asked.
        self._host_action: Optional[HostActionTag] = None
        self._persist = persist
        self._persistence_error = False
        self._recovered = False
        self._interrupted = False
        self._journal_cursor = 0

    def lifecycle_receipt(self) -> Dict[str, Any]:
        """Minimal metadata only; free-form payloads remain with their owners."""
        with self._lock:
            return {
                "id": self.id, "kind": self.kind, "status": self.status,
                "created": self.created, "finished": self.finished,
                "cancel_requested": self.cancel_requested,
                "next_event_seq": self._next_seq, "interrupted": self._interrupted,
            }

    def _persist_locked(self, *, reserve_cursor: bool = False) -> bool:
        if self._persist is None:
            return True
        receipt = self.lifecycle_receipt()
        if self.status == "running":
            # Persist a cursor reservation before publishing a batch. A crash
            # may leave a gap, but its terminal event cannot reuse a delivered
            # sequence number. Streaming deltas never require one fsync each.
            proposed_cursor = max(
                self._journal_cursor,
                self._next_seq + (_JOURNAL_CURSOR_BLOCK if reserve_cursor else 0),
            )
            receipt["next_event_seq"] = proposed_cursor
        try:
            self._persist(receipt)
        except JobJournalError:
            # A metadata-disk failure cannot undo an already executed operation.
            # Keep it visible, without logging potentially private DB contents.
            if not self._persistence_error:
                logger.error("Job %s lifecycle journal is unavailable", self.id)
            self._persistence_error = True
            return False
        if self.status == "running":
            self._journal_cursor = proposed_cursor
        return True

    @classmethod
    def from_lifecycle_receipt(cls, receipt: Dict[str, Any]) -> "Job":
        """Restore a status receipt, never a callable or a scientific grant."""
        job = cls(receipt["id"], receipt["kind"])
        job.created = receipt["created"]
        job.cancel_requested = bool(receipt["cancel_requested"])
        job._recovered = True
        job._interrupted = receipt["status"] == "running" or bool(receipt["interrupted"])
        job.status = receipt["status"]
        job.finished = receipt["finished"]
        if job.status == "running":
            job.status = "cancelled" if job.cancel_requested else "failed"
            job.finished = time.time()
            job._next_seq = receipt["next_event_seq"]
        else:
            # Replay the existing terminal cursor, including across repeated
            # restarts, instead of creating a new completion each time.
            job._next_seq = max(0, receipt["next_event_seq"] - 1)
        if job._interrupted:
            job.error = "job_interrupted_by_server_restart"
        elif job.status == "failed":
            job.error = "job_failed_before_server_restart"
        job._append_event_locked({
            "type": "end", "status": job.status, "result": None, "error": job.error,
            "result_omitted": True, "recovered": True,
            "timestamp": datetime.fromtimestamp(job.finished, timezone.utc).isoformat(),
            "reason_code": "job_interrupted_by_server_restart" if job._interrupted else "job_receipt_restored",
        })
        return job

    def _append_event_locked(self, event: Dict[str, Any]) -> None:
        if self._persist is not None:
            # Keep one durable slot for a terminal event. If the next batch
            # cannot be reserved, the runner can still fail visibly without
            # publishing an ID that recovery could later reuse.
            limit = self._journal_cursor - (0 if event.get("type") == "end" else 1)
            if self._next_seq >= limit and not self._persist_locked(reserve_cursor=True):
                raise JobJournalError("job_journal_unavailable")
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        try:
            timestamp = datetime.fromisoformat(str(payload["timestamp"]))
            if timestamp.tzinfo is None:
                raise ValueError("event timestamp needs a timezone")
        except (ValueError, TypeError):
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        payload["seq"] = self._next_seq
        self._next_seq += 1
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
        size = len(encoded.encode("utf-8"))
        if size > self._max_event_bytes:
            terminal = event.get("type") == "end"
            payload = {
                "type": "end" if terminal else "progress",
                "seq": payload["seq"],
                "timestamp": payload["timestamp"],
                "step": "event_payload_omitted",
                "payload_omitted": True,
                "reason_code": "job_event_size_limit",
            }
            if terminal:
                payload.update(status=event["status"], result=None, result_omitted=True)
            encoded = json.dumps(payload)
            size = len(encoded.encode("utf-8"))
        # Own a detached JSON payload: later caller mutations cannot evade limits.
        payload = json.loads(encoded)
        self.events.append(payload)
        self._event_sizes.append(size)
        self._history_bytes += size
        while len(self.events) > self._max_events or self._history_bytes > self._max_history_bytes:
            self.events.pop(0)
            self._history_bytes -= self._event_sizes.pop(0)

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
            try:
                self._append_event_locked(
                    {"type": "end", "status": status, "result": result, "error": error}
                )
            except JobJournalError:
                # Defensive fallback for an embedded Job without an initial
                # reservation: terminate the snapshot/SSE tail without emitting
                # an unsafe cursor. JobManager jobs retain the terminal slot.
                status = "cancelled" if self.cancel_requested else "failed"
                self.error = "job_journal_unavailable"
            self.finished = time.time()
            self.status = status
            self._cancel_callbacks.clear()
            self._persist_locked()
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
        """Interrupt now; acknowledge only when the cancellation receipt saves.

        A failed journal write cannot suppress physical interruption, but it
        must not promise that a future process will remember the request.
        Repeated requests retry that write without repeating owner callbacks.
        """
        callbacks: List[Callable[[], None]] = []
        with self._lock:
            if self.status != "running":
                return False
            if not self.cancel_requested:
                self.cancel_requested = True
                self.cancel_reason = reason or "user_requested"
                try:
                    self._append_event_locked(
                        {"type": "cancel_requested", "reason": self.cancel_reason}
                    )
                except JobJournalError:
                    # Cancellation owns process interruption even when its
                    # optional progress event cannot obtain a durable cursor.
                    pass
                callbacks = list(self._cancel_callbacks.values())
            persisted = self._persist_locked()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Job %s cancellation callback failed", self.id)
        return persisted

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

    def tag_host_action(self, tag: HostActionTag) -> bool:
        """Record the one host decision this job answers."""
        with self._lock:
            if self._host_action is not None:
                return self._host_action == tag
            self._host_action = tag
            return True

    @property
    def host_action(self) -> Optional[HostActionTag]:
        with self._lock:
            return self._host_action

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
            first = self.events[0]["seq"] if self.events else self._next_seq
            events = []
            if start < first:
                events.append({"type": "progress", "seq": first - 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": "history_truncated", "history_truncated": True,
                    "dropped_events": first - start,
                    "label": "Earlier progress events expired; durable artifacts remain authoritative."})
            events.extend(dict(event) for event in self.events if event["seq"] >= start)
            return events, self.status

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "created": self.created,
                "finished": self.finished,
                "events": [dict(event) for event in self.events],
                "events_dropped": self._next_seq - len(self.events),
                "next_event_seq": self._next_seq,
                "event_history_bytes": self._history_bytes,
                "result": self.result,
                "error": self.error,
                "cancel_requested": self.cancel_requested,
                "cancel_reason": self.cancel_reason,
                "draining": self.draining,
                **({"recovered": True, "result_omitted": True, "interrupted": self._interrupted}
                   if self._recovered else {}),
                **({"persistence_error": "job_journal_unavailable"} if self._persistence_error else {}),
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
        self._journal = None
        self._runner_claims: set[str] = set()

    def enable_persistence(self, path: Path) -> None:
        """Enable after acquiring the Web deployment lease, before accepting jobs."""
        from easyicu.webserver.job_journal import JobJournal, JobJournalError

        with self._lock:
            if self._journal is not None:
                if self._journal.path != Path(path):
                    raise JobJournalError("job_journal_path_changed")
                return
            if any(job.consumes_capacity() for job in self._jobs.values()):
                raise JobJournalError("job_journal_requires_idle_startup")
            journal = JobJournal(path, max_completed=self._max_completed)
            restored = {}
            for receipt in journal.records():
                job = Job.from_lifecycle_receipt(receipt)
                if receipt["status"] == "running":
                    journal.save(job.lifecycle_receipt())
                restored[job.id] = job
            # Any existing completed jobs belong to this process (e.g. embedded
            # callers before startup); do not replace their richer live result.
            restored.update(self._jobs)
            self._jobs = restored
            self._journal = journal
            self._prune_completed_locked()

    def reserve(
        self, kind: str, job_id: Optional[str] = None, *, resume_interrupted: bool = False,
    ) -> Job:
        """Persist stable acceptance without starting a callable.

        A durable dispatcher may reattach the same ID after a restart. It must
        check the original scientific authority in its runner before any action.
        Completed/cancelled receipts are returned unchanged and never rerun.
        """
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", str(kind)):
            raise JobJournalError("job_kind_invalid")
        identity = str(job_id) if job_id is not None else uuid.uuid4().hex[:12]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", identity):
            raise JobJournalError("job_id_invalid")
        with self._lock:
            existing = self._jobs.get(identity)
            if existing is not None:
                if existing.kind != kind:
                    raise JobJournalError("job_identity_kind_conflict")
                if not resume_interrupted or not existing._interrupted or existing.cancel_requested:
                    return existing
                if existing.status != "failed" or not existing._recovered:
                    raise JobJournalError("job_resume_not_interrupted")
            elif resume_interrupted:
                raise JobJournalError("job_resume_receipt_missing")
            self._prune_completed_locked()
            running = sum(job.consumes_capacity() for job in self._jobs.values())
            if running >= self._max_running:
                raise JobCapacityError(max_running=self._max_running, running=running)
            if existing is not None:
                # Use a fresh event buffer: the synthetic restart end must not
                # terminate a newly attached SSE stream. Its cursor stays spent.
                job = Job(identity, kind, persist=self._journal.save if self._journal is not None else None)
                job.created = existing.created
                job._next_seq = existing.snapshot()["next_event_seq"]
                job._journal_cursor = job._next_seq + _JOURNAL_CURSOR_BLOCK
                receipt = {**job.lifecycle_receipt(), "next_event_seq": job._journal_cursor}
                if self._journal is not None:
                    self._journal.reopen_interrupted(receipt)
                job.emit({"type": "progress", "step": "durable_execution_resumed", "reason_code": "job_execution_reattached"})
            else:
                job = Job(identity, kind, persist=self._journal.save if self._journal is not None else None)
                if self._journal is not None:
                    job._journal_cursor = _JOURNAL_CURSOR_BLOCK
                    self._journal.create({**job.lifecycle_receipt(), "next_event_seq": job._journal_cursor})
            self._jobs[job.id] = job
            return job

    def run_reserved(self, job_id: str, runner: Callable[[Job], Any]) -> Job:
        """Execute a reserved Job synchronously, suitable for a durable step.

        Only one caller may claim a running identity. A persisted cancellation
        wins before the runner starts, and terminal work is never repeated.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobJournalError("job_reservation_missing")
            if job.status != "running":
                return job
            if job_id in self._runner_claims:
                raise JobJournalError("job_runner_already_claimed")
            self._runner_claims.add(job_id)
        try:
            if job.is_cancel_requested():
                job.complete_from_runner()
            else:
                self._run(job, runner)
            return job
        finally:
            with self._lock:
                self._runner_claims.discard(job_id)
                self._prune_completed_locked()

    def submit(self, kind: str, runner: Callable[[Job], Any]) -> Job:
        """Accept a fresh local job and execute it on a background thread."""
        job = self.reserve(kind)
        threading.Thread(target=self.run_reserved, args=(job.id, runner), daemon=True).start()
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
                try:
                    job.emit(
                        {
                            # Keep the existing event type shared by both UIs;
                            # the typed payload also survives in the result.
                            "type": "progress",
                            "step": "scientific_action_gap",
                            "status": "action_required",
                            "reason_code": str(projection.get("issue_code") or ""),
                            "label": label,
                            "action": projection,
                        }
                    )
                except JobJournalError:
                    # Reporting an owner failure must still reach its terminal
                    # state when the progress cursor could not be reserved.
                    pass
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

    def cancel_all(self, reason: str = "server_shutdown") -> None:
        """Ask each runner's interrupt owner to clean up its own resources."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.request_cancel(reason)


MANAGER = JobManager()
