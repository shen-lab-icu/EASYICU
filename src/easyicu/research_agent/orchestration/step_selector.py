"""Stable user-facing selectors for Agent-owned plan step identifiers."""

from __future__ import annotations

import re
from typing import Optional

from ..schema import AnalysisPlan


def resolve_stop_after_step_selector(
    plan: AnalysisPlan,
    requested: Optional[str],
) -> Optional[str]:
    """Resolve an id, ordinal, or unique typed-product checkpoint."""

    if requested is None:
        return None
    if requested == "@first":
        if not plan.steps:
            raise ValueError("stop_after_step_id='@first' requires a non-empty plan")
        return str(plan.steps[0].step_id)
    index_match = re.fullmatch(r"@index:([1-9][0-9]*)", requested)
    if index_match is not None:
        one_based_index = int(index_match.group(1))
        if one_based_index > len(plan.steps):
            raise ValueError(
                f"stop_after_step_id={requested!r} exceeds the active plan's "
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
                f"stop_after_step_id={requested!r} requires exactly one declared "
                f"producer; observed {owners!r}."
            )
        return owners[0]
    return requested


__all__ = ["resolve_stop_after_step_selector"]
