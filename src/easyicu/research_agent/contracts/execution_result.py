"""Dependency-neutral runner results and terminal codes shared with reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


class RunnerFailureCode(str, Enum):
    """Host-issued runner diagnostics that generated code cannot mint."""

    ISOLATION_BACKEND_UNAVAILABLE = "isolation_backend_unavailable"
    HOST_RUNNER_REFUSES_MODEL_CODE = "host_runner_refuses_model_code"


@dataclass
class RunResult:
    """Everything captured from one generated-code execution."""

    step_id: str
    script_path: Path
    cwd: Path
    out_dir: Path
    stdout: str
    stderr: str
    returncode: int
    duration_seconds: float
    artefacts: List[Path] = field(default_factory=list)
    timed_out: bool = False
    requested_network_policy: str = "none"
    effective_isolation: str = "unknown"
    isolation_degraded: bool = False
    isolation_degradation_reason: Optional[str] = None
    runtime_provenance: Dict[str, object] = field(default_factory=dict)
    # False means callers must not scan or hash anything under ``out_dir``.
    outputs_safe_to_collect: bool = True
    runner_log_path: Optional[Path] = None
    # This control-plane value is assigned by the runner from a trusted backend
    # probe.  stdout/stderr/run.log are child-influenced diagnostic surfaces and
    # must never be used as substitutes for it.
    runner_failure_code: Optional[RunnerFailureCode] = None

    @property
    def succeeded(self) -> bool:
        return (
            self.returncode == 0 and not self.timed_out and self.outputs_safe_to_collect
        )


class RuntimeFailureClass(str, Enum):
    """Closed runtime-failure classes that change orchestration."""

    PLAN_DATA_CONTRACT = "plan_data_contract"
    EXECUTION_TIMEOUT = "execution_timeout"
    ISOLATION_BACKEND_UNAVAILABLE = "isolation_backend_unavailable"
    DETERMINISTIC_MODEL_NOT_ESTIMABLE = "deterministic_model_not_estimable"


DECLARED_EXPOSURE_LEVEL_EMPTY = "declared_exposure_level_empty"


def runtime_failure_codes(record: Mapping[str, Any]) -> dict[str, str]:
    """Project the host-classified empty-level stop from a failed step record."""

    if (record.get("runtime_failure_class") == RuntimeFailureClass.PLAN_DATA_CONTRACT.value
            and record.get("runtime_failure_reason") == DECLARED_EXPOSURE_LEVEL_EMPTY
            and record.get("status") == "contract_failed"):
        return {"reason_code": DECLARED_EXPOSURE_LEVEL_EMPTY}
    return {}


STANDARD_EXECUTOR_FAILURE_REASONS = frozenset({
    "executor_runtime_failure", "executor_output_contract_failed", "missing_executor_outputs",
})


def standard_executor_failure_codes(record: Mapping[str, Any]) -> dict[str, str]:
    """Carry host terminal failures when the executor has no scientific stop.

    These codes describe execution, not a rejected scientific design. Raw
    exceptions and generated output prose never cross this projection.
    """

    reason = record.get("standard_executor_terminal_reason")
    if (
        record.get("status") == "deterministic_standard_blocked"
        and record.get("deterministic_standard_analysis")
        and isinstance(reason, str)
        and reason in STANDARD_EXECUTOR_FAILURE_REASONS
    ):
        return {"reason_code": "deterministic_standard_executor_failed_closed", "cause_code": reason}
    return {}


__all__ = [
    "RunnerFailureCode", "RunResult", "RuntimeFailureClass",
    "DECLARED_EXPOSURE_LEVEL_EMPTY", "STANDARD_EXECUTOR_FAILURE_REASONS",
    "runtime_failure_codes", "standard_executor_failure_codes",
]
