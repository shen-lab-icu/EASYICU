"""Bounded, per-session ordered dispatch for Pi host-tool calls.

The Pi sidecar has one stdout protocol reader.  That reader must never execute
host tools itself: a slow tool would otherwise block events and responses for
every open session.  This owner keeps execution off the reader while preserving
the ordering contract inside each session.
"""

from __future__ import annotations

import collections
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional

from .contracts import PiCopilotError


@dataclass(frozen=True)
class HostToolOutcome:
    """Sanitized result written back to the Pi sidecar by one writer thread."""

    result: Optional[dict] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error_code is None


class HostToolDispatchRejected(RuntimeError):
    """A tool request could not enter the bounded dispatcher."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(eq=False)
class _HostToolTask:
    session_id: str
    execute: Callable[[], dict]
    respond: Callable[[HostToolOutcome], None]
    terminal: bool = False


_STOP = object()


class HostToolDispatcher:
    """Run tools concurrently across sessions and serially within a session."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        max_pending: int = 64,
        on_response_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._max_pending = int(max_pending)
        self._on_response_error = on_response_error
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._accepting = True
        self._pending_count = 0
        self._queued_by_session: Dict[str, Deque[_HostToolTask]] = {}
        self._active_by_session: Dict[str, _HostToolTask] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers),
            thread_name_prefix="easyicu-pi-host-tool",
        )
        # At most max_pending outcomes can exist, because a capacity slot is
        # released only after its response has been serialized.
        self._outcomes: queue.Queue[object] = queue.Queue(maxsize=max_pending + 1)
        self._writer_thread = threading.Thread(
            target=self._write_outcomes,
            name="easyicu-pi-host-tool-writer",
            daemon=True,
        )
        self._writer_thread.start()

    @property
    def closed(self) -> bool:
        with self._lock:
            return not self._accepting

    def submit(
        self,
        *,
        session_id: str,
        execute: Callable[[], dict],
        respond: Callable[[HostToolOutcome], None],
    ) -> None:
        """Accept one task without blocking the protocol reader."""

        task = _HostToolTask(
            session_id=str(session_id),
            execute=execute,
            respond=respond,
        )
        start_now = False
        with self._lock:
            if not self._accepting:
                raise HostToolDispatchRejected(
                    "pi_host_tool_dispatcher_closed",
                    "The EasyICU host-tool dispatcher is closed.",
                )
            if self._pending_count >= self._max_pending:
                raise HostToolDispatchRejected(
                    "pi_host_tool_dispatcher_full",
                    "The EasyICU host-tool queue is full; the request was not executed.",
                )
            self._pending_count += 1
            if task.session_id in self._active_by_session:
                self._queued_by_session.setdefault(
                    task.session_id, collections.deque()
                ).append(task)
            else:
                self._active_by_session[task.session_id] = task
                start_now = True
        if start_now:
            self._start_task(task)

    def _start_task(self, task: _HostToolTask) -> None:
        try:
            self._executor.submit(self._execute_task, task)
        except RuntimeError:
            # Executor shutdown raced with scheduling.  Preserve a structured
            # fail-closed response rather than stranding the sidecar request.
            self._offer_outcome(
                task,
                HostToolOutcome(
                    error_code="pi_host_tool_dispatcher_closed",
                    error_message="The EasyICU host-tool dispatcher closed before execution.",
                ),
            )

    def _execute_task(self, task: _HostToolTask) -> None:
        try:
            result = task.execute()
            if not isinstance(result, dict):
                raise PiCopilotError(
                    "pi_host_tool_result_invalid",
                    "The EasyICU host tool returned an invalid result.",
                    status_code=500,
                )
            outcome = HostToolOutcome(result=dict(result))
        except PiCopilotError as exc:
            outcome = HostToolOutcome(
                error_code=exc.code,
                error_message=exc.message,
            )
        except Exception:
            outcome = HostToolOutcome(
                error_code="pi_host_tool_failed",
                error_message=(
                    "The EasyICU host tool failed without exposing traceback text."
                ),
            )
        self._offer_outcome(task, outcome)

    def _offer_outcome(self, task: _HostToolTask, outcome: HostToolOutcome) -> None:
        with self._lock:
            if task.terminal:
                return
            task.terminal = True
        # Capacity is guaranteed by max_pending, so this never waits on the
        # protocol reader or a host-tool worker.
        self._outcomes.put_nowait((task, outcome))

    def _write_outcomes(self) -> None:
        while True:
            item = self._outcomes.get()
            if item is _STOP:
                return
            task, outcome = item
            try:
                task.respond(outcome)
            except Exception as exc:
                if self._on_response_error is not None:
                    try:
                        self._on_response_error(exc)
                    except Exception:
                        pass
            finally:
                self._finish_task(task)

    def _finish_task(self, task: _HostToolTask) -> None:
        next_task: Optional[_HostToolTask] = None
        with self._lock:
            if self._active_by_session.get(task.session_id) is task:
                self._active_by_session.pop(task.session_id, None)
                queued = self._queued_by_session.get(task.session_id)
                if queued:
                    next_task = queued.popleft()
                    self._active_by_session[task.session_id] = next_task
                    if not queued:
                        self._queued_by_session.pop(task.session_id, None)
            self._pending_count -= 1
            if self._pending_count == 0:
                self._idle.notify_all()
        if next_task is not None:
            self._start_task(next_task)

    def wait_until_idle(self, timeout: float) -> bool:
        """Wait only for tests/orderly shutdown; submissions remain accepted."""

        with self._idle:
            return self._idle.wait_for(
                lambda: self._pending_count == 0,
                timeout=max(0.0, float(timeout)),
            )

    def shutdown(self, *, timeout: float = 2.0) -> None:
        """Reject future work and fail queued/in-flight requests closed."""

        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
            tasks = list(self._active_by_session.values())
            for queued in self._queued_by_session.values():
                tasks.extend(queued)
            self._queued_by_session.clear()
        closed = HostToolOutcome(
            error_code="pi_host_tool_dispatcher_closed",
            error_message="The EasyICU host-tool dispatcher closed before completion.",
        )
        for task in tasks:
            self._offer_outcome(task, closed)
        self.wait_until_idle(timeout)
        self._executor.shutdown(wait=False, cancel_futures=True)
        try:
            self._outcomes.put_nowait(_STOP)
        except queue.Full:
            # If a response writer is still draining, let the daemon exit with
            # the gateway process.  Every accepted task was already failed.
            return
        self._writer_thread.join(timeout=max(0.0, float(timeout)))


__all__ = [
    "HostToolDispatchRejected",
    "HostToolDispatcher",
    "HostToolOutcome",
]
