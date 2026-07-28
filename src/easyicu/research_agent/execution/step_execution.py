"""Mechanical execution boundary for one already-locked analysis script.

Scientific choices, runner selection, replay authority, validation, and evidence
publication remain with their existing owners.  This module only preserves the
backend cleanup contract and performs one sandbox call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..contracts.runtime import RunResult


@dataclass(frozen=True)
class LockedStepExecutionRequest:
    """Host-locked inputs for one mechanical backend invocation."""

    step_id: str
    code: str
    resolved_inputs_path: Optional[Path]
    output_dir: Path


class StepExecutor:
    """Execute locked code without owning scientific or authority decisions."""

    def __init__(self, *, clear_output_dir: Callable[[Path], None]) -> None:
        self._clear_output_dir = clear_output_dir

    @staticmethod
    def runner_timeout(runner: Any, fallback_seconds: float) -> float:
        """Return the timeout actually enforced by the selected runner."""

        return float(getattr(runner, "timeout_seconds", fallback_seconds))

    def execute(
        self,
        *,
        runner: Any,
        request: LockedStepExecutionRequest,
    ) -> "RunResult":
        if not bool(getattr(runner, "manages_output_cleanup", False)):
            self._clear_output_dir(request.output_dir)
        return runner.run(
            step_id=request.step_id,
            code=request.code,
            resolved_inputs_path=request.resolved_inputs_path,
        )


__all__ = ["LockedStepExecutionRequest", "StepExecutor"]
