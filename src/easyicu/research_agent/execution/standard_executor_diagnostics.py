"""Structured failure attribution for deterministic standard executors."""

from __future__ import annotations

from typing import Any, Mapping

from ..schema import ValidationFinding

__all__ = ["standard_executor_failure_finding"]


def standard_executor_failure_finding(
    *,
    step_record: Mapping[str, Any],
    step_id: str,
    reason: str,
    failure_phase: str,
    executor_errors: Any = None,
) -> ValidationFinding:
    """Attribute a standard-executor failure to its actual implementation."""

    analysis_kind = str(
        step_record.get("deterministic_standard_analysis")
        or "unknown_standard_executor"
    )
    selection_reason = str(
        step_record.get("deterministic_standard_selection_reason")
        or "unknown_selection_reason"
    )
    return ValidationFinding(
        validator="deterministic_standard_executor",
        severity="error",
        message=(
            f"The planner-scoped deterministic standard executor "
            f"{analysis_kind!r} failed closed during {failure_phase}; no Coder "
            "repair or method substitution was attempted."
        ),
        detail={
            "step_id": step_id,
            "issue_code": "deterministic_standard_executor_failed_closed",
            "failure_phase": failure_phase,
            "analysis_kind": analysis_kind,
            "selection_reason": selection_reason,
            "reason": reason,
            "executor_errors": executor_errors,
        },
    )
