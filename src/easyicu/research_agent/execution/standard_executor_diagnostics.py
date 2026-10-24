"""Structured failure attribution for deterministic standard executors."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts.executor_stop import recorded_executor_stop
from ..schema import ValidationFinding

__all__ = [
    "STANDARD_EXECUTOR_FAILURE_REASONS", "standard_executor_failure_codes",
    "standard_executor_failure_finding",
]

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


def standard_executor_failure_finding(
    *,
    step_record: Mapping[str, Any],
    step_id: str,
    reason: str,
    failure_phase: str,
    executor_errors: Any = None,
) -> ValidationFinding:
    """Attribute a standard-executor failure to its actual implementation.

    A registered stop the executor named (``contracts.executor_stop``) is part
    of the attribution: its reason code and lower-layer cause code.
    """

    analysis_kind = str(
        step_record.get("deterministic_standard_analysis")
        or "unknown_standard_executor"
    )
    selection_reason = str(
        step_record.get("deterministic_standard_selection_reason")
        or "unknown_selection_reason"
    )
    detail: dict[str, Any] = {
        "step_id": step_id,
        "issue_code": "deterministic_standard_executor_failed_closed",
        "failure_phase": failure_phase,
        "analysis_kind": analysis_kind,
        "selection_reason": selection_reason,
        "reason": reason,
        "executor_errors": executor_errors,
    }
    stop = recorded_executor_stop(step_record)
    if stop is not None:
        detail["executor_stop_reason_code"] = stop[0]
        if stop[1] is not None:
            detail["executor_stop_cause_code"] = stop[1]
    return ValidationFinding(
        validator="deterministic_standard_executor",
        severity="error",
        message=(
            f"The planner-scoped deterministic standard executor "
            f"{analysis_kind!r} failed closed during {failure_phase}; no Coder "
            "repair or method substitution was attempted."
        ),
        detail=detail,
    )
