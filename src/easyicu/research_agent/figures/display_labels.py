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


def _binary_level(value: Any) -> Optional[int]:
    token = str("" if value is None else value).strip().casefold()
    if token in {"0", "0.0", "false", "no", "n", "absent", "negative"}:
        return 0
    if token in {"1", "1.0", "true", "yes", "y", "present", "positive"}:
        return 1
    return None


def scoped_label_lookup(
    scope: Any,
    value: Any,
    display_labels: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return a Planner label declared as ``<scope>=<level>``.

    Binary aliases are matched deliberately (for example ``1`` and ``1.0``),
    while the scope still requires an exact normalized identifier.  This keeps
    a level label bound to its variable instead of letting a generic ``0`` or
    ``1`` label leak across unrelated panels.
    """

    scope_key = _normalise_display_key(scope)
    level = _binary_level(value)
    if not scope_key or level is None or not display_labels:
        return None
    for raw_key, raw_label in display_labels.items():
        key = str(raw_key or "").strip()
        if "=" not in key:
            continue
        raw_scope, raw_level = key.rsplit("=", 1)
        label = str(raw_label or "").strip()
        if (
            _normalise_display_key(raw_scope) == scope_key
            and _binary_level(raw_level) == level
            and label
        ):
            return label
    return None


def binary_contrast_label(
    scope: Any,
    display_labels: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return the Planner-owned positive-versus-reference contrast label."""

    reference = scoped_label_lookup(scope, 0, display_labels)
    comparison = scoped_label_lookup(scope, 1, display_labels)
    if reference and comparison and reference != comparison:
        return f"{comparison} vs {reference}"
    return None


def binary_scope_label(
    scope: Any,
    display_labels: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return a shared Planner-authored prefix for a complete binary pair.

    ``"Marker A absent"`` and ``"Marker A present"`` yield ``"Marker A"``.
    If the two labels do not share a word prefix, no scope is inferred.
    """

    reference = scoped_label_lookup(scope, 0, display_labels)
    comparison = scoped_label_lookup(scope, 1, display_labels)
    if not reference or not comparison:
        return None
    reference_words = reference.split()
    comparison_words = comparison.split()
    shared: list[str] = []
    for left, right in zip(reference_words, comparison_words):
        if left.casefold() != right.casefold():
            break
        shared.append(left)
    return " ".join(shared) or None


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


__all__ = [
    "binary_contrast_label",
    "binary_scope_label",
    "display_label",
    "label_lookup",
    "scoped_label_lookup",
]
