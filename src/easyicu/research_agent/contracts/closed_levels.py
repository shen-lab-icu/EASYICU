"""Dependency-neutral validation for closed JSON-scalar level sets."""

from __future__ import annotations

import math
from typing import Any, Sequence


def validate_closed_scalar_levels(values: Sequence[Any], *, label: str) -> list[Any]:
    """Return a copy of one finite, typed-unique JSON-scalar level set."""

    tokens: list[tuple[str, str]] = []
    for value in values:
        if not isinstance(value, (str, bool, int, float)):
            raise ValueError(f"{label} must contain only JSON scalar values")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite values")
        tokens.append((type(value).__name__, repr(value)))
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"{label} must contain unique typed values")
    return list(values)


__all__ = ["validate_closed_scalar_levels"]
