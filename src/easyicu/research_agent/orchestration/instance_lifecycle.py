"""Fail-fast lifecycle ownership for one pipeline object.

The run-directory writer lock owns one ``run_id`` across processes.  It cannot
protect mutable fields on a :class:`ResearchAgentPipeline` object when two
fresh runs receive different ids in the same process.  This module owns that
smaller boundary: exactly one run or resume call may use an instance, and a
human-review pause keeps the instance reserved until that live handoff is
resolved.

The internal mutex is held only while changing the four-state lifecycle.  It
is never held while a pipeline executes or while a human decides, so callers
fail immediately instead of waiting behind a long-running analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal, Optional


LifecycleState = Literal["idle", "running", "paused", "resuming"]
LifecycleOperation = Literal["run", "resume"]


class PipelineInstanceLifecycleError(RuntimeError):
    """Base error for an invalid pipeline-instance lifecycle transition."""

    reason_code = "pipeline_instance_lifecycle_invalid"


class PipelineInstanceLifecycleBusy(PipelineInstanceLifecycleError):
    """Raised immediately when another lifecycle already owns the instance."""

    reason_code = "pipeline_instance_lifecycle_busy"


@dataclass(frozen=True)
class PipelineInstanceLifecycleToken:
    """Unforgeable-by-convention coordinate for one active method call."""

    generation: int
    operation: LifecycleOperation


@dataclass(frozen=True)
class PipelineInstanceLifecycleSnapshot:
    """Read-only diagnostic view; it grants no lifecycle authority."""

    state: LifecycleState
    paused_run_id: Optional[str]
    active_operation: Optional[LifecycleOperation]


class PipelineInstanceLifecycleLease:
    """Own the mutable lifecycle of one ``ResearchAgentPipeline`` instance."""

    def __init__(self) -> None:
        self._guard = Lock()
        self._state: LifecycleState = "idle"
        self._paused_run_id: Optional[str] = None
        self._active_token: Optional[PipelineInstanceLifecycleToken] = None
        self._generation = 0

    def snapshot(self) -> PipelineInstanceLifecycleSnapshot:
        """Return a stable diagnostic snapshot without exposing mutable state."""

        with self._guard:
            token = self._active_token
            return PipelineInstanceLifecycleSnapshot(
                state=self._state,
                paused_run_id=self._paused_run_id,
                active_operation=token.operation if token is not None else None,
            )

    def begin_run(
        self, *, pending_review_run_id: Optional[str] = None
    ) -> PipelineInstanceLifecycleToken:
        """Claim the instance for one run, or fail immediately.

        ``pending_review_run_id`` lets a hot-reloaded or test-constructed
        pipeline adopt an already-live pause before evaluating the new call.
        Normal production pauses are already represented by ``_state``.
        """

        with self._guard:
            if self._state == "idle" and pending_review_run_id:
                self._state = "paused"
                self._paused_run_id = str(pending_review_run_id)
            if self._state != "idle":
                raise self._busy_error(requested="run")
            return self._activate(operation="run", state="running")

    def begin_resume(
        self, *, pending_review_run_id: Optional[str]
    ) -> PipelineInstanceLifecycleToken:
        """Claim the exact paused lifecycle for resume, or fail immediately."""

        with self._guard:
            if self._state == "idle":
                if not pending_review_run_id:
                    raise PipelineInstanceLifecycleError(
                        "no human review is pending on this pipeline instance; "
                        "resume is same_process and must answer the pause returned "
                        "by run()"
                    )
                # Compatibility adoption for a pause restored into the same
                # process before this lease owner existed.  This is a state
                # transition, not a bypass: the resume immediately owns it.
                self._state = "paused"
                self._paused_run_id = str(pending_review_run_id)
            if self._state != "paused":
                raise self._busy_error(requested="resume")
            if not pending_review_run_id:
                raise PipelineInstanceLifecycleError(
                    "pipeline lifecycle is paused but the human-review handoff "
                    "is missing"
                )
            if str(pending_review_run_id) != self._paused_run_id:
                raise PipelineInstanceLifecycleError(
                    "human-review handoff does not match the run reserved by "
                    f"this pipeline lifecycle: expected {self._paused_run_id!r}, "
                    f"got {str(pending_review_run_id)!r}"
                )
            return self._activate(operation="resume", state="resuming")

    def hold_for_review(
        self,
        token: PipelineInstanceLifecycleToken,
        *,
        run_id: str,
    ) -> None:
        """End an active call while retaining ownership for human review."""

        with self._guard:
            self._require_active(token)
            self._state = "paused"
            self._paused_run_id = str(run_id)
            self._active_token = None

    def release(self, token: PipelineInstanceLifecycleToken) -> None:
        """Release a completed or terminally failed call back to ``idle``."""

        with self._guard:
            self._require_active(token)
            self._state = "idle"
            self._paused_run_id = None
            self._active_token = None

    def _activate(
        self, *, operation: LifecycleOperation, state: LifecycleState
    ) -> PipelineInstanceLifecycleToken:
        self._generation += 1
        token = PipelineInstanceLifecycleToken(
            generation=self._generation,
            operation=operation,
        )
        self._state = state
        self._active_token = token
        return token

    def _require_active(self, token: PipelineInstanceLifecycleToken) -> None:
        if self._active_token != token or self._state not in {"running", "resuming"}:
            raise PipelineInstanceLifecycleError(
                "pipeline instance lifecycle token is stale or no longer active"
            )

    def _busy_error(self, *, requested: LifecycleOperation) -> Exception:
        if self._state == "paused":
            return PipelineInstanceLifecycleBusy(
                "pipeline instance is paused for human review"
                + (
                    f" (run_id={self._paused_run_id!r})"
                    if self._paused_run_id
                    else ""
                )
                + f"; cannot start {requested} until that live handoff is resumed"
            )
        active = (
            self._active_token.operation
            if self._active_token is not None
            else self._state
        )
        return PipelineInstanceLifecycleBusy(
            "pipeline instance lifecycle is already active "
            f"({active}); concurrent {requested} calls are not allowed"
        )


__all__ = [
    "PipelineInstanceLifecycleBusy",
    "PipelineInstanceLifecycleError",
    "PipelineInstanceLifecycleLease",
    "PipelineInstanceLifecycleSnapshot",
    "PipelineInstanceLifecycleToken",
]
