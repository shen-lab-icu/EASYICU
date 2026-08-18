"""Cross-step validation for stable scientific-capability declarations."""

from __future__ import annotations

from typing import Any, Sequence

from .association_execution import (
    ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID,
    association_binary_sensitivity_plan_verdict,
)
from .capability_ids import capability_family


def validate_scientific_capability_id(value: Any) -> str | None:
    """Normalize one declared id and reject values outside the stable vocabulary."""

    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if capability_family(cleaned) is None:
        raise ValueError(
            "scientific_capability_unknown: "
            f"{cleaned!r} is not in the stable capability vocabulary"
        )
    return cleaned


def validate_scientific_capability_declarations(steps: Sequence[Any]) -> None:
    """Reject unknown ids and incomplete inherited sensitivity contracts."""

    for step in steps:
        declared_id = validate_scientific_capability_id(step.scientific_capability)
        if (
            declared_id == ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID
            and step.planned_analysis_role == "sensitivity"
        ):
            verdict = association_binary_sensitivity_plan_verdict(
                step,
                plan_steps=steps,
            )
            if not verdict.claimed:
                raise ValueError(f"{verdict.reason_code}: {verdict.reason}")


__all__ = [
    "validate_scientific_capability_declarations",
    "validate_scientific_capability_id",
]
