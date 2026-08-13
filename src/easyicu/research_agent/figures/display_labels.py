"""Planner-owned display labels for deterministic publication figures."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional


def _normalise_display_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def label_lookup(
    value: Any, display_labels: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Return the Planner-owned label for an exact/normalized identifier."""

    if not display_labels:
        return None
    raw = str(value or "").strip()
    exact = display_labels.get(raw)
    if exact is not None and str(exact).strip():
        return str(exact).strip()
    normalized = _normalise_display_key(raw)
    if not normalized:
        return None
    for key, label in display_labels.items():
        if _normalise_display_key(key) == normalized and str(label).strip():
            return str(label).strip()
    return None


def display_label(
    value: Any, display_labels: Optional[Mapping[str, str]] = None
) -> str:
    """Render a declared label, otherwise apply case-neutral title casing."""

    declared = label_lookup(value, display_labels)
    if declared is not None:
        return declared
    token = str(value or "").strip()
    if not token:
        return "Value"
    return re.sub(r"[_-]+", " ", token).strip().title()


__all__ = ["display_label", "label_lookup"]
