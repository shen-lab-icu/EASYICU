"""Fail-closed classification for trusted runtime contract failures.

Runtime diagnostics normally belong to Coder repair.  A missing closed
comparison group is different: changing Python cannot restore rows removed by
an upstream cohort definition.  Treat the two canonical Table 1 diagnostics as
plan/data-contract contradictions so the execute phase stops without spending
an LLM code-repair attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ..contracts.runtime import ValidationFinding


class RuntimeFailureClass(str, Enum):
    """Closed runtime-failure classes that change orchestration."""

    PLAN_DATA_CONTRACT = "plan_data_contract"


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
) -> RuntimeFailureDecision | None:
    """Return a fail-closed class without exposing diagnostic literals.

    A timeout remains a runtime failure even if a partial log happens to
    contain one of the signatures.  The signatures only suppress Coder repair;
    they never authorize evidence or upgrade a failed step.
    """

    if timed_out:
        return None
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
