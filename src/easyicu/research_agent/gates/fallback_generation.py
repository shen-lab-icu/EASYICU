"""What a fallback-generation step may count as, and what it must prove (C-F9).

Owner
-----
A step produced in fallback generation mode counts as a primary estimate
only when the reviewed plan allows it on that exact step, and it must prove
that it passed the method-compatibility gate
(:func:`.method_compatibility.fallback_method_compatibility_findings`).
Readiness asks this module both questions.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


def is_fallback_generation(record: Mapping[str, Any]) -> bool:
    """Return True when a step record was produced in fallback generation mode."""

    return str(record.get("generation_mode") or "").strip().lower() == "fallback"


def fallback_primary_allowed(plan: Any, step_id: str) -> bool:
    """Return True only when the plan explicitly allows fallback as primary."""

    steps = getattr(plan, "steps", None)
    if not isinstance(steps, list):
        return False
    for step in steps:
        candidate_id = getattr(step, "step_id", None)
        if isinstance(step, Mapping):
            candidate_id = step.get("step_id")
        if str(candidate_id or "") != str(step_id or ""):
            continue
        flag = getattr(step, "allow_fallback_as_primary", None)
        if isinstance(step, Mapping):
            flag = step.get("allow_fallback_as_primary")
        return bool(flag) is True
    return False


def primary_records_for_readiness(
    per_step_records: Any, plan: Any = None
) -> list[dict[str, Any]]:
    """Filter step records for primary estimation: downgrade fallbacks.

    C-F9: ``generation_mode=fallback`` steps do not count as primary unless
    the plan explicitly allows it via
    ``AnalysisStep.allow_fallback_as_primary``. Downgraded steps remain
    visible in execution records; they are only excluded from the primary
    headline binding.
    """

    filtered: list[dict[str, Any]] = []
    for record in per_step_records or []:
        if not isinstance(record, dict):
            continue
        if is_fallback_generation(record) and not fallback_primary_allowed(
            plan, str(record.get("step_id") or "")
        ):
            continue
        filtered.append(record)
    return filtered


def executed_gate_approved_code(record: Mapping[str, Any]) -> bool:
    """Whether a step ran the exact script the pre-execution code gate approved.

    The candidate loop records ``concept_approved_code_sha256`` only after
    :func:`~easyicu.research_agent.gates.concept.deterministic_code_gate_findings`
    returned no error for that script, and that gate runs the same
    :func:`~easyicu.research_agent.gates.method_compatibility.detect_forbidden_pattern_usage`
    matrix as :func:`fallback_method_compatibility_findings`. Outputs whose
    executed digest differs from the approved one are rejected before they are
    registered, so equal digests are the compatibility evidence of a record
    that does not carry its code, such as a persisted or resumed record.
    """

    executed = str(record.get("executed_code_sha256") or "").strip().lower()
    approved = str(record.get("concept_approved_code_sha256") or "").strip().lower()
    return re.fullmatch(r"[0-9a-f]{64}", executed) is not None and executed == approved


def fallback_method_compatibility_errors(
    *,
    per_step_records: Any,
    context: Any,
    plan: Any = None,
) -> list[Any]:
    """Force fallback products through the method-compatibility gate (C-F9).

    Every fallback-generation record must have passed
    :func:`fallback_method_compatibility_findings`. When the executed code is
    available on the record it is scanned. Otherwise the record needs either a
    ``method_compatibility_checked`` marker or proof that it executed the
    exact script the pre-execution code gate approved; without either it
    fails closed. Plan-allowed fallback primaries are still checked — the flag
    only controls primary counting, never gate bypass.
    """

    from ..contracts.runtime import ValidationFinding as _ValidationFinding
    from .method_compatibility import fallback_method_compatibility_findings

    errors: list[Any] = []
    for record in per_step_records or []:
        if not isinstance(record, dict):
            continue
        if not is_fallback_generation(record):
            continue
        summary = record.get("step_summary")
        summary_map = summary if isinstance(summary, Mapping) else {}
        checked = bool(
            record.get("method_compatibility_checked")
            or summary_map.get("method_compatibility_checked")
            or executed_gate_approved_code(record)
        )
        code = record.get("code") or record.get("executed_code") or ""
        if not isinstance(code, str):
            code = ""
        violations: list[dict[str, object]] = []
        if code and context is not None and hasattr(context, "variables"):
            try:
                plan_step = None
                steps = getattr(plan, "steps", None)
                if isinstance(steps, list):
                    for candidate in steps:
                        cid = (
                            candidate.get("step_id")
                            if isinstance(candidate, Mapping)
                            else getattr(candidate, "step_id", None)
                        )
                        if str(cid or "") == str(record.get("step_id") or ""):
                            plan_step = (
                                candidate
                                if not isinstance(candidate, Mapping)
                                else None
                            )
                            break
                violations = fallback_method_compatibility_findings(
                    code=code, context=context, step=plan_step
                )
            except Exception:
                violations = []
        if violations:
            errors.append(
                _ValidationFinding(
                    validator="method_compatibility",
                    severity="error",
                    message=(
                        f"Fallback step {record.get('step_id')} produced "
                        "method-incompatible code; fallback products must pass "
                        "the method_compatibility gate."
                    ),
                    detail={
                        "step_id": str(record.get("step_id") or ""),
                        "generation_mode": "fallback",
                        "violations": violations,
                    },
                )
            )
        elif not checked and not code:
            errors.append(
                _ValidationFinding(
                    validator="method_compatibility",
                    severity="error",
                    message=(
                        f"Fallback step {record.get('step_id')} has no "
                        "method_compatibility evidence; fallback products must "
                        "pass the method_compatibility gate."
                    ),
                    detail={
                        "step_id": str(record.get("step_id") or ""),
                        "generation_mode": "fallback",
                    },
                )
            )
    return errors


__all__ = [
    "executed_gate_approved_code",
    "fallback_method_compatibility_errors",
    "fallback_primary_allowed",
    "is_fallback_generation",
    "primary_records_for_readiness",
]
