"""Planner-owned display labels for deterministic publication figures.

Figure titles, axes and captions are written in English, so every label a
figure shows must be English too.  A Planner label written in another script
(for example in the language of the user's question) is not figure text.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

_FOREIGN_SCRIPT_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯豈-﫿]")


def _normalise_display_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def _figure_text(label: Any) -> Optional[str]:
    text = str(label or "").strip()
    if not text or _FOREIGN_SCRIPT_RE.search(text):
        return None
    return text


def figure_language_labels(
    display_labels: Optional[Mapping[str, str]],
    descriptions: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Keep only labels a figure can show in its own (English) text.

    A variable label in another script is replaced by that variable's English
    source description, first letter capitalised.  A level label
    (``variable=level``) or a variable without an English description keeps
    no label, so the renderer's case-neutral rendering of the key applies.
    """

    english = {
        str(name): " ".join(str(text or "").split())
        for name, text in (descriptions or {}).items()
    }
    result: dict[str, str] = {}
    for key, label in (display_labels or {}).items():
        text = _figure_text(label)
        if text is not None:
            result[str(key)] = text
            continue
        name, separator, _level = str(key).partition("=")
        description = _figure_text(english.get(name))
        if not separator and description is not None:
            result[str(key)] = description[:1].upper() + description[1:]
    return result


def label_lookup(
    value: Any, display_labels: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Return the Planner-owned label for an exact/normalized identifier."""

    if not display_labels:
        return None
    raw = str(value or "").strip()
    exact = _figure_text(display_labels.get(raw))
    if exact is not None:
        return exact
    normalized = _normalise_display_key(raw)
    if not normalized:
        return None
    for key, label in display_labels.items():
        text = _figure_text(label)
        if _normalise_display_key(key) == normalized and text is not None:
            return text
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

    Exact named/ordinal categories are supported. Binary aliases are matched
    deliberately (for example ``1`` and ``1.0``),
    while the scope still requires an exact normalized identifier.  This keeps
    a level label bound to its variable instead of letting a generic ``0`` or
    ``1`` label leak across unrelated panels.
    """

    scope_key = _normalise_display_key(scope)
    # Exact scoped categories need not be binary (e.g. an ordinal stage or a
    # named treatment). Never borrow a global level label from another field.
    raw_value = str(value).strip()
    exact = _figure_text((display_labels or {}).get(f"{scope}={raw_value}"))
    if exact is not None:
        return exact
    level = _binary_level(value)
    if not scope_key or level is None or not display_labels:
        return None
    for raw_key, raw_label in display_labels.items():
        key = str(raw_key or "").strip()
        if "=" not in key:
            continue
        raw_scope, raw_level = key.rsplit("=", 1)
        label = _figure_text(raw_label)
        if (
            _normalise_display_key(raw_scope) == scope_key
            and _binary_level(raw_level) == level
            and label is not None
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
    words = re.sub(r"[_-]+", " ", token).strip().split()
    clinical_tokens = {
        "aki": "AKI",
        "bili": "Bilirubin",
        "bun": "BUN",
        "cardio": "Cardiovascular",
        "charlson": "Charlson index",
        "cns": "CNS",
        "coag": "Coagulation",
        "crea": "Creatinine",
        "hr": "Heart rate",
        "icu": "ICU",
        "kdigo": "KDIGO",
        "lact": "Lactate",
        "map": "MAP",
        "na": "Sodium",
        "ph": "pH",
        "plt": "Platelets",
        "resp": "Respiratory",
        "rrt": "RRT",
        "sep3": "Sepsis-3",
        "sofa": "SOFA",
        "sofa2": "SOFA",
        "spo2": "SpO2",
        "susp": "Suspected",
        "temp": "Temperature",
        "wbc": "WBC",
    }
    rendered = [clinical_tokens.get(word.casefold(), word.title()) for word in words]
    return " ".join(rendered)


__all__ = [
    "binary_contrast_label",
    "binary_scope_label",
    "display_label",
    "figure_language_labels",
    "label_lookup",
    "scoped_label_lookup",
]
