"""Safe UI projections for orchestration lifecycle events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..authority.run_heartbeat import record_active_run_progress
from ..authority.runtime_artifacts import AuditLogger
from ..providers.structured_retry import StructuredRetryProgress


class NonFatalProgressCallbackError(RuntimeError):
    """Mark a UI projection failure as safe to ignore by orchestration."""


class ProgressControlSignal(RuntimeError):
    """Base for typed host cancellation or orchestration control signals."""


class ResumableProgressChannel:
    """Per-run progress transport that can be rebound after a review pause.

    The channel owns only UI/audit transport.  Replacing its callback cannot
    replace the workflow, plan, evidence store, or digest-bound invokers that
    the human approved.
    """

    def __init__(
        self,
        callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self._callback = callback
        self._audit_logger: Optional[AuditLogger] = None

    def bind_audit_logger(self, audit_logger: AuditLogger) -> None:
        self._audit_logger = audit_logger

    def replace_callback(
        self,
        callback: Optional[Callable[[dict[str, Any]], None]],
    ) -> None:
        self._callback = callback

    def emit(self, stage: str, message: str, **extra: Any) -> None:
        status = str(extra.get("status", "running"))
        step_id = str(extra.get("step_id")) if extra.get("step_id") else None
        record_active_run_progress(
            stage=stage,
            message=message,
            status=status,
            step_id=step_id,
            phase_timeout_seconds=extra.get("phase_timeout_seconds"),
            run_id=(str(extra.get("run_id")) if extra.get("run_id") else None),
        )
        if self._audit_logger is not None:
            try:
                self._audit_logger.emit(
                    phase=stage,
                    event=message,
                    status=status,
                    step_id=step_id,
                    detail={
                        key: value
                        for key, value in extra.items()
                        if key not in {"status", "step_id"}
                    },
                )
            except Exception:
                pass
        if self._callback is None:
            return
        payload_extra = dict(extra)
        payload = {
            "stage": stage,
            "message": message,
            "status": payload_extra.pop("status", "running"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload_extra,
        }
        try:
            self._callback(payload)
        except ProgressControlSignal:
            raise
        except Exception:
            pass


def planner_retry_progress_callback(
    emit_progress: Callable[..., None],
    *,
    run_id: str,
) -> Callable[[StructuredRetryProgress], None]:
    """Translate structured Planner attempts into bounded host progress."""

    def callback(event: StructuredRetryProgress) -> None:
        attempt = int(event.attempt or 0)
        total = int(event.total_attempts or 0)
        if event.phase == "started":
            label = f"Generating plan draft {attempt}/{total}."
            status = "running"
        elif event.phase == "rejected":
            label = (
                f"Plan draft {attempt}/{total} did not satisfy the scientific "
                + ("contract; retrying." if attempt < total else "contract.")
            )
            status = "running" if attempt < total else "error"
        elif event.phase == "accepted":
            label = f"Plan draft {attempt}/{total} passed contract validation."
            status = "complete"
        else:  # pragma: no cover - Literal contract guards this path
            return
        emit_progress(
            "planning",
            label,
            current=attempt,
            total=total,
            status=status,
            run_id=run_id,
        )

    return callback


__all__ = [
    "NonFatalProgressCallbackError",
    "ProgressControlSignal",
    "ResumableProgressChannel",
    "planner_retry_progress_callback",
]
