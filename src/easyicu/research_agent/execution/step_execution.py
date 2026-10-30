"""Mechanical execution boundary for one already-locked analysis script.

Scientific choices, runner selection, replay authority, validation, and evidence
publication remain with their existing owners.  This module only preserves the
backend cleanup contract and performs one sandbox call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from ..contracts.runtime import RunResult


class CodeOrigin(str, Enum):
    """Producer of these exact bytes, never inferred from the step's method."""

    HOST_RENDERED_EXECUTOR = "host_rendered_executor"
    MODEL_GENERATED = "model_generated"
    REPAIR = "repair"
    RECOVERED_CAPSULE = "recovered_capsule"


class HostCodeRefused(ValueError):
    """Untrusted code has no execution authority on the desktop runner."""

    reason_code = "host_runner_refuses_model_code"

    def __init__(self, detail: str = "") -> None:
        super().__init__(f"{self.reason_code}: {detail}")
        self.easyicu_safe_diagnostic = {
            "owner": "easyicu.execution.runtime_v1",
            "reason_code": self.reason_code,
            "runner_kind": "host",
        }


def initial_code_origin(runner_kind: str, step_id: str, host_code: Optional[str]) -> CodeOrigin:
    if host_code is not None:
        return CodeOrigin.HOST_RENDERED_EXECUTOR
    if runner_kind == "host":
        raise HostCodeRefused(f"no freshly rendered executor for {step_id}")
    return CodeOrigin.MODEL_GENERATED


def host_runtime_available(probe_timeout_seconds: float) -> bool:
    """Probe the app interpreter's typed-executor imports without a cohort."""
    from ..contracts.method_packages import BASELINE_PACKAGES

    names = [*BASELINE_PACKAGES, "psutil"]
    source = (
        "import importlib.util, sys; "
        f"sys.exit(0 if all(importlib.util.find_spec(n) for n in {names!r}) else 1)"
    )
    try:
        command = [sys.executable, "-c", source]
        frozen = bool(getattr(sys, "frozen", False))
        if frozen:
            from easyicu.runtime.host_worker import frozen_worker_command

            command = frozen_worker_command(
                sys.executable, "capabilities", "--names", json.dumps(names),
            )
        result = subprocess.run(
            command, capture_output=True,
            timeout=max(0.1, probe_timeout_seconds), check=False,
        )
        return result.returncode == 0 and (not frozen or set(names).issubset(json.loads(result.stdout)))
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return False


@dataclass(frozen=True)
class LockedStepExecutionRequest:
    """Host-locked inputs for one mechanical backend invocation."""

    step_id: str
    code: str
    resolved_inputs_path: Optional[Path]
    output_dir: Path
    code_origin: CodeOrigin = CodeOrigin.MODEL_GENERATED
    wave_authority: Optional[Any] = None


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
        """Execute one attempt into a directory this call owns.

        An executed attempt fully replaces ``output_dir``: a
        cleanup-managing runner stages an attempt-owned directory and swaps
        it into place, and every other runner is cleared here first.  So the
        contents of ``output_dir`` belong to the most recent executed attempt.
        A host provenance refusal happens before cleanup and preserves that
        evidence; its result explicitly forbids collecting those old outputs.

        Callers must therefore not pre-emptively clear the directory when
        *scheduling* a retry.  A retry that never reaches execution -- a
        repaired draft quarantined by a pre-execution gate, say -- would
        otherwise leave the step reporting a ``RunResult`` whose outputs the
        host had already deleted.
        """

        wave_kwargs = {}
        if request.wave_authority is not None:
            from .runner import (
                _code_requests_robustness_authority_snapshot,
                require_wave_artifact_authority,
            )

            if _code_requests_robustness_authority_snapshot(request.code):
                if not getattr(runner, "supports_wave_artifact_authority", False):
                    raise ValueError("runner cannot receive the required wave artifact authority")
                require_wave_artifact_authority(request.wave_authority)
                wave_kwargs["wave_authority"] = request.wave_authority
        if not bool(getattr(runner, "manages_output_cleanup", False)):
            self._clear_output_dir(request.output_dir)
        provenance = (
            {"code_origin": request.code_origin}
            if getattr(runner, "requires_code_origin", False) else {}
        )
        provenance.update(wave_kwargs)
        return runner.run(
            step_id=request.step_id,
            code=request.code,
            resolved_inputs_path=request.resolved_inputs_path,
            **provenance,
        )


__all__ = ["CodeOrigin", "HostCodeRefused", "LockedStepExecutionRequest", "StepExecutor"]
