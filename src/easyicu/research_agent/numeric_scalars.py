"""Dependency-neutral scalar coercion shared by execution and figure owners."""

from __future__ import annotations

import math
from typing import Any


def coerce_optional_finite_float(value: Any) -> float | None:
    """Return a coercible finite float, or ``None`` for invalid/non-finite input."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def coerce_finite_float(value: Any, *, label: str = "value") -> float:
    """Return one finite float or raise a stable scalar-contract error."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


__all__ = ["coerce_finite_float", "coerce_optional_finite_float"]
