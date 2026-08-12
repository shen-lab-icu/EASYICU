"""Safe UI projections for orchestration lifecycle events."""

from __future__ import annotations

from typing import Callable

from ..providers.structured_retry import StructuredRetryProgress


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


__all__ = ["planner_retry_progress_callback"]
