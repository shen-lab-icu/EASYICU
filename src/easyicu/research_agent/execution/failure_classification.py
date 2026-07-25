"""Fail-closed classification for trusted runtime contract failures.

Runtime diagnostics normally belong to Coder repair.  Two situations are
different, and both stop the step without spending an LLM code-repair attempt:

* A missing closed comparison group — changing Python cannot restore rows
  removed by an upstream cohort definition.  The two canonical Table 1
  diagnostics are treated as plan/data-contract contradictions.
* A wall-clock timeout — the script was killed mid-run, so the diagnostic the
  repairer would read is a truncated log with no traceback and no error.  The
  Coder cannot see that the wall clock, not the code, ended the step, so it
  hunts a defect that may not exist and the step retries the same overlong
  computation until the repair budget is gone.  Repair is the wrong instrument
  for "correct but too slow": the remedies are a registered deterministic
  executor (which carries its own, much larger bounded timeout) or an
  explicitly raised ``timeout_seconds`` — both operator decisions, not
  something a rewritten script can reach on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ..contracts.runtime import ValidationFinding


class RuntimeFailureClass(str, Enum):
    """Closed runtime-failure classes that change orchestration."""

    PLAN_DATA_CONTRACT = "plan_data_contract"
    EXECUTION_TIMEOUT = "execution_timeout"


_EMPTY_CLOSED_COMPARISON_SIGNATURES = (
    "Planner-declared Table 1 groups are absent from",
    "A Planner-declared Table 1 group is empty",
)


@dataclass(frozen=True)
class RuntimeFailureDecision:
    """Terminal fail-closed payload consumed by execute-phase bookkeeping."""

    finding: ValidationFinding
    step_updates: Mapping[str, Any]
    progress_message: str


def classify_runtime_failure(
    *,
    run_log: str,
    timed_out: bool,
    step_id: str,
    returncode: int,
    timeout_seconds: float | None = None,
    deterministic_executor_used: bool = False,
) -> RuntimeFailureDecision | None:
    """Return a fail-closed class without exposing diagnostic literals.

    A timeout is classified on the timeout alone and never on the partial log:
    a script killed mid-run may have emitted any prefix of any signature, so
    reading that prefix would attribute the wrong cause.  The class suppresses
    Coder repair; like every class here it never authorizes evidence and never
    upgrades a failed step.
    """

    if timed_out:
        failure_class = RuntimeFailureClass.EXECUTION_TIMEOUT
        limit = None if timeout_seconds is None else float(timeout_seconds)
        return RuntimeFailureDecision(
            finding=ValidationFinding(
                validator="runtime_execution_timeout",
                severity="error",
                message=(
                    "The step exceeded its execution wall clock and was killed "
                    "mid-run. It failed closed without requesting Coder repair: "
                    "the surviving diagnostic is a truncated log with no "
                    "traceback, so a rewrite would be aimed at an unidentified "
                    "defect while the same overlong computation consumed the "
                    "remaining repair budget. Route this analysis to a "
                    "registered deterministic executor or raise the configured "
                    "timeout."
                ),
                detail={
                    "step_id": step_id,
                    "failure_class": failure_class.value,
                    "returncode": returncode,
                    "timeout_seconds": limit,
                    "deterministic_executor_used": bool(deterministic_executor_used),
                },
            ),
            step_updates={
                "status": "execution_failed",
                "diagnostic_only": True,
                "runtime_failure_class": failure_class.value,
                "runtime_repair_route": "fail_closed",
                "llm_repair_used": False,
                "timed_out": True,
                "execution_timeout_seconds": limit,
            },
            progress_message=(
                f"Execution timed out for {step_id}; Coder repair was not "
                "authorized."
            ),
        )
    if any(signature in run_log for signature in _EMPTY_CLOSED_COMPARISON_SIGNATURES):
        failure_class = RuntimeFailureClass.PLAN_DATA_CONTRACT
        return RuntimeFailureDecision(
            finding=ValidationFinding(
                validator="runtime_plan_data_contract",
                severity="error",
                message=(
                    "The runtime data cannot satisfy the Planner-declared closed "
                    "comparison. The step failed closed without requesting Coder "
                    "repair."
                ),
                detail={
                    "step_id": step_id,
                    "failure_class": failure_class.value,
                    "returncode": returncode,
                },
            ),
            step_updates={
                "status": "contract_failed",
                "diagnostic_only": True,
                "runtime_failure_class": failure_class.value,
                "runtime_repair_route": "fail_closed",
                "llm_repair_used": False,
            },
            progress_message=(
                f"Runtime plan/data contract failed for {step_id}; "
                "Coder repair was not authorized."
            ),
        )
    return None
