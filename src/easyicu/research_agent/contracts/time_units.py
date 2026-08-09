"""Closed time-unit vocabulary for scientific execution contracts."""

from __future__ import annotations

from typing import Any, Literal, Optional


CanonicalTimeUnit = Literal["minutes", "hours", "days"]

_ALIASES: dict[str, CanonicalTimeUnit] = {
    "minute": "minutes",
    "minutes": "minutes",
    "min": "minutes",
    "mins": "minutes",
    "hour": "hours",
    "hours": "hours",
    "hr": "hours",
    "hrs": "hours",
    "day": "days",
    "days": "days",
}


def canonical_time_unit(value: Any) -> Optional[CanonicalTimeUnit]:
    """Resolve a declared unit token; return ``None`` for unknown units.

    This function never inspects a column name or numeric magnitude. It only
    normalises a unit value already present in host-owned metadata.
    """

    token = str(value or "").strip().casefold()
    return _ALIASES.get(token)


__all__ = ["CanonicalTimeUnit", "canonical_time_unit"]
