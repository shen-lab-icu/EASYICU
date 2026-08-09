"""Publication-facing p-value representation without mutating raw evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping

MIN_NONZERO_REPORTED_P = 1e-300

_EXACT_P_VALUE_FIELDS = frozenset(
    {
        "adjusted_p",
        "adjusted_p_value",
        "p",
        "p_val",
        "p_value",
        "pvalue",
        "primary_p_value",
        "raw_p",
        "raw_p_value",
        "unadjusted_p",
        "unadjusted_p_value",
    }
)


def is_p_value_field(name: str) -> bool:
    """Return whether a field is a numeric p-value rather than a p-value flag."""

    normalized = str(name or "").strip().lower()
    if normalized.startswith("statistic:"):
        normalized = normalized.split(":", 1)[1]
    if normalized.endswith(("_reporting", "_bounded")):
        return False
    return normalized in _EXACT_P_VALUE_FIELDS or normalized.endswith(
        ("_p_value", "_pvalue", "_p_val")
    )


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def publication_p_value(value: Any) -> tuple[Any, str | None, bool]:
    """Return a bounded writer value, optional label, and underflow flag.

    Zero remains untouched in stored tables and evidence.  Only the compact
    writer handoff substitutes a conservative positive bound, preventing prose
    such as ``p=0`` while preserving the raw artifact for audit.
    """

    numeric = _numeric(value)
    if numeric is None or numeric > 0.0:
        return value, None, False
    if numeric < 0.0:
        return value, None, False
    return MIN_NONZERO_REPORTED_P, "p < 1e-300", True


def prepare_p_values_for_writer(values: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a digest row and add explicit reporting metadata for underflow."""

    prepared = dict(values)
    for key, value in list(values.items()):
        if not is_p_value_field(key):
            continue
        bounded, reporting, underflow = publication_p_value(value)
        if not underflow:
            continue
        prepared[key] = bounded
        prepared[f"{key}_reporting"] = reporting
        prepared[f"{key}_source_underflow"] = True
    return prepared


def render_claim_value_for_writer(
    *,
    source_field: str,
    value: Any,
    canonical: Any | None = None,
) -> str:
    """Render one secondary claim without exposing ``p=0`` to the writer."""

    if is_p_value_field(source_field):
        bounded, reporting, underflow = publication_p_value(value)
        if underflow:
            return (
                f"{bounded} (reporting={reporting}; "
                "source_underflow=true)"
            )
    if canonical is None:
        return str(value)
    return f"{value} (canonical={canonical})"


__all__ = [
    "MIN_NONZERO_REPORTED_P",
    "is_p_value_field",
    "prepare_p_values_for_writer",
    "publication_p_value",
    "render_claim_value_for_writer",
]
