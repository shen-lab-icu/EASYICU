"""Stable user-facing selectors for Agent-owned plan step identifiers."""

from __future__ import annotations

import re
from typing import Optional

from ..schema import AnalysisPlan


def _resolve_plan_step_selector(
    plan: AnalysisPlan,
    requested: Optional[str],
    *,
    option_name: str,
) -> Optional[str]:
    """Resolve an id, ordinal, or unique typed-product checkpoint."""

    if requested is None:
        return None
    if requested == "@first":
        if not plan.steps:
            raise ValueError(f"{option_name}='@first' requires a non-empty plan")
        return str(plan.steps[0].step_id)
    index_match = re.fullmatch(r"@index:([1-9][0-9]*)", requested)
    if index_match is not None:
        one_based_index = int(index_match.group(1))
        if one_based_index > len(plan.steps):
            raise ValueError(
                f"{option_name}={requested!r} exceeds the active plan's "
                f"{len(plan.steps)} step(s)."
            )
        return str(plan.steps[one_based_index - 1].step_id)
    if requested.startswith("@product:"):
        product = requested.removeprefix("@product:")
        owners = [
            str(step.step_id)
            for step in plan.steps
            if product in {str(value) for value in step.expected_outputs}
        ]
        if len(owners) != 1:
            raise ValueError(
                f"{option_name}={requested!r} requires exactly one declared "
                f"producer; observed {owners!r}."
            )
        return owners[0]
    return requested


def resolve_stop_after_step_selector(
    plan: AnalysisPlan,
    requested: Optional[str],
) -> Optional[str]:
    """Resolve a user-facing stop-after selector."""

    return _resolve_plan_step_selector(
        plan,
        requested,
        option_name="stop_after_step_id",
    )


def resolve_resume_from_step_selector(
    plan: AnalysisPlan,
    requested: Optional[str],
) -> Optional[str]:
    """Resolve a user-facing resume-from selector before plan migration."""

    return _resolve_plan_step_selector(
        plan,
        requested,
        option_name="resume_from_step_id",
    )


__all__ = [
    "resolve_resume_from_step_selector",
    "resolve_stop_after_step_selector",
]
