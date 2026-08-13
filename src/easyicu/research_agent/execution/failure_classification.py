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
* An unavailable isolation backend — generated Python never started, so asking
  the Coder to rewrite it cannot affect the host/Docker/sandbox boundary.  This
  is an operator environment failure and must remain distinct from a code
  defect.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ..contracts.runtime import ValidationFinding
from ..contracts.execution_result import RunnerFailureCode


class RuntimeFailureClass(str, Enum):
    """Closed runtime-failure classes that change orchestration."""

    PLAN_DATA_CONTRACT = "plan_data_contract"
    EXECUTION_TIMEOUT = "execution_timeout"
    ISOLATION_BACKEND_UNAVAILABLE = "isolation_backend_unavailable"
    DETERMINISTIC_MODEL_NOT_ESTIMABLE = "deterministic_model_not_estimable"


_EMPTY_CLOSED_COMPARISON_SIGNATURES = (
    "Planner-declared Table 1 groups are absent from",
    "A Planner-declared Table 1 group is empty",
)

#: The host's OWN model fitter reporting that the declared model cannot be
#: estimated on this data. Both halves are required: the typed error the
#: deterministic owner raises, and a phrase naming estimability rather than
#: any other reason that owner can refuse for.
_DETERMINISTIC_MODEL_ERROR = "AdjustedAssociationError"
_NOT_ESTIMABLE_SIGNATURES = (
    "did not converge",
    "could not be fitted as declared",
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
    runner_failure_code: RunnerFailureCode | str | None = None,
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
    if runner_failure_code == RunnerFailureCode.ISOLATION_BACKEND_UNAVAILABLE:
        failure_class = RuntimeFailureClass.ISOLATION_BACKEND_UNAVAILABLE
        return RuntimeFailureDecision(
            finding=ValidationFinding(
                validator="runtime_isolation_backend_unavailable",
                severity="error",
                message=(
                    "The generated script was not launched because no approved "
                    "execution-isolation backend was available. The step failed "
                    "closed without requesting Coder repair: changing analysis "
                    "code cannot start Docker or make the host sandbox accept "
                    "the configured interpreter. Restore an approved runner "
                    "backend, then retry the run."
                ),
                detail={
                    "step_id": step_id,
                    "failure_class": failure_class.value,
                    "returncode": returncode,
                },
            ),
            step_updates={
                "status": "execution_environment_failed",
                "diagnostic_only": True,
                "runtime_failure_class": failure_class.value,
                "runtime_repair_route": "fail_closed",
                "llm_repair_used": False,
            },
            progress_message=(
                f"Execution isolation was unavailable for {step_id}; Coder "
                "repair was not authorized."
            ),
        )
    if (
        deterministic_executor_used
        and _DETERMINISTIC_MODEL_ERROR in run_log
        and any(signature in run_log for signature in _NOT_ESTIMABLE_SIGNATURES)
    ):
        # Gated on ``deterministic_executor_used`` on purpose. The same words in
        # an agent-written script describe code the Coder CAN fix; here they
        # describe the host's own fitter reporting that the declared model is
        # not estimable on this data, which no rewrite of that script can
        # change. Measured over every recorded run: 3 steps hit this, all 3 spent
        # LLM repairs on the host's own script, and one of those repairs invented
        # a keyword argument the host does not accept -- turning a statistical
        # outcome into `TypeError: ... unexpected keyword argument 'fit_kwargs'`
        # and losing the real reason. Same shape as EXECUTION_TIMEOUT above:
        # correct host code, unfixable-by-rewrite cause, repair budget burned.
        failure_class = RuntimeFailureClass.DETERMINISTIC_MODEL_NOT_ESTIMABLE
        return RuntimeFailureDecision(
            finding=ValidationFinding(
                validator="runtime_deterministic_model_not_estimable",
                severity="error",
                message=(
                    "The host's own model fitter reported that the declared "
                    "model cannot be estimated on this data. The step failed "
                    "closed without requesting Coder repair: the script is "
                    "host-owned and correct, so a rewrite can only damage it, "
                    "and estimability is a property of the declared model and "
                    "the cohort. Re-declare the model — fewer or differently "
                    "coded covariates, a wider analysis set, or a different "
                    "method family — rather than editing the executor."
                ),
                detail={
                    "step_id": step_id,
                    "failure_class": failure_class.value,
                    "returncode": returncode,
                    "deterministic_executor_used": True,
                },
            ),
            step_updates={
                "status": "execution_failed",
                "diagnostic_only": True,
                "runtime_failure_class": failure_class.value,
                "runtime_repair_route": "fail_closed",
                "llm_repair_used": False,
            },
            progress_message=(
                f"The declared model for {step_id} is not estimable; Coder "
                "repair was not authorized."
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
