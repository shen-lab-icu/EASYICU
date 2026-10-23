"""Bind fresh executor provenance before any initial-code fallback is selected.

This gate records a terminal refusal rather than allowing missing executor code
to reach a Coder, repair, or recovered capsule on the desktop backend.
"""

from __future__ import annotations

from typing import Any, Callable, ContextManager

from ..contracts.runtime import ValidationFinding
from .failure_classification import record_host_code_refusal


def resolve_initial_code_origin(
    *,
    runner_kind: str,
    host_code: str | None,
    step_id: str,
    step_record: dict[str, Any],
    findings: list[ValidationFinding],
    shared_lock: ContextManager[Any],
    per_step_records: list[dict[str, Any]],
    append_terminal: Callable[[list[dict[str, Any]], dict[str, Any]], Any],
    flush_manifest: Callable[[], Any],
) -> bool:
    """Bind fresh provenance, or persist a terminal refusal before any coder call."""
    from .step_execution import HostCodeRefused, initial_code_origin

    try:
        step_record["code_origin"] = initial_code_origin(runner_kind, step_id, host_code).value
    except HostCodeRefused:
        record_host_code_refusal(
            step_id=step_id, step_record=step_record, findings=findings,
            shared_lock=shared_lock, per_step_records=per_step_records,
            append_terminal=append_terminal, flush_manifest=flush_manifest,
        )
        return False
    return True


def standard_executor_finding(step_id: str, analysis_kind: str) -> ValidationFinding:
    """Describe a selected typed executor without granting code authority."""
    return ValidationFinding(
        validator="coder",
        severity="info",
        message=(
            "Using the deterministic calculator for the complete "
            "Planner-owned standard-executor specification in "
            f"step {step_id}."
        ),
        detail={"step_id": step_id, "analysis_kind": analysis_kind},
    )
