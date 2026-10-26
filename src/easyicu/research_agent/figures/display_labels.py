"""Planner-owned display labels for deterministic publication figures.

Figure titles, axes and captions are written in English, so every label a
figure shows must be English too.  A Planner label written in another script
(for example in the language of the user's question) is not figure text.
"""

from __future__ import annotations

import math
import numbers
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional

from ..contracts.exposure_group_rules import ExposureGroupRuleError, read_grouping_rules

_FOREIGN_SCRIPT_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯豈-﫿]")
_CLINICAL_ACRONYMS = {
    token.casefold(): token
    for token in ("AKI", "APACHE", "BUN", "CI", "CNS", "HR", "ICU", "KDIGO", "MAP",
                  "OR", "RR", "RRT", "SOFA", "SpO2", "WBC", "pH")
}


@dataclass(frozen=True)
class _ClinicalBound:
    measure: str
    operator: str
    value: Decimal
    unit: str


def _clinical_interval(bounds: list[_ClinicalBound]) -> str | None:
    if not bounds or len({(bound.measure, bound.unit) for bound in bounds}) != 1:
        return None
    lower = max(
        (b for b in bounds if b.operator in {">", "≥"}),
        key=lambda b: (b.value, b.operator == ">"), default=None,
    )
    upper = min(
        (b for b in bounds if b.operator in {"<", "≤"}),
        key=lambda b: (b.value, b.operator != "<"), default=None,
    )
    if lower and upper:
        if lower.value >= upper.value:
            return None
        return (
            f"{lower.operator if lower.operator == '>' else ''}{lower.value.normalize():f}"
            f"–{upper.operator}{upper.value.normalize():f}"
        )
    bound = lower or upper
    return f"{bound.operator}{bound.value.normalize():f}"


def grouped_axis_labels(
    grouping: Mapping[str, Any], labels: Mapping[str, str],
) -> dict[str, str]:
    """Factor a shared measure only from the pinned, ordered grouping rules.

    Each level intersects its threshold with the complements of *all* earlier
    thresholds. Mixed summaries/units, unmeasured levels and empty intervals
    keep their full labels; matching prose alone never authorizes factoring.
    """
    parameters = grouping.get("parameters", {})
    try:
        rules = read_grouping_rules(parameters)
    except ExposureGroupRuleError:
        return {}
    levels = grouping.get("levels", [])
    if (
        rules.concept != grouping.get("concept")
        or rules.scale != grouping.get("scale")
        or rules.unmeasured != "exclude"
        or len(levels) != len(rules.codes)
        or {level["group"]: level["code"] for level in levels} != dict(rules.codes)
    ):
        return {}
    identities = {
        (group["rule"]["summary"], group["rule"].get("unit"))
        for group in parameters["groups"] if isinstance(group["rule"], Mapping)
    }
    if len(identities) != 1:
        return {}
    summary, unit = next(iter(identities))
    if not isinstance(unit, str) or not unit.strip():
        return {}
    operators = {"<": "<", "<=": "≤", ">": ">", ">=": "≥"}
    inverse = {"<": "≥", "≤": ">", ">": "≤", "≥": "<"}
    earlier: list[_ClinicalBound] = []
    result = {}
    scope = grouping["variable"]
    for group in parameters["groups"]:
        rule = group["rule"]
        current = ([] if rule == "otherwise" else [_ClinicalBound(
            summary, operators[rule["op"]], Decimal(str(rule["value"])), unit,
        )])
        interval = _clinical_interval([*earlier, *current])
        if interval is None:
            return {}
        result[f"{scope}={rules.codes[group['id']]}"] = interval
        if current:
            bound = current[0]
            earlier.append(_ClinicalBound(summary, inverse[bound.operator], bound.value, unit))
    measure = {"max": "Maximum", "min": "Minimum", "mean": "Mean", "first": "First", "value": "Value"}[summary]
    name = display_label(rules.concept, labels)
    if not name[:2].isupper():
        name = name[:1].lower() + name[1:]
    result[scope] = f"{measure} {name} ({unit})"
    return result


def concise_clinical_level(label: str) -> str:
    """Shorten only the host's exact scalar-predicate grammar.

    Intersect typed bounds on the same summary and unit. Different summaries,
    units, arbitrary prose and empty intersections retain their full wording.
    In particular, a minimum/maximum combination is not a numeric interval.
    """
    operators = {"below": "<", "at most": "≤", "above": ">", "at least": "≥"}
    inverse = {"<": "≥", "≤": ">", ">": "≤", "≥": "<"}
    text = " ".join(label.split())
    if text.startswith("Neither ") and " nor " in text:
        clauses = [(part, True) for part in text[8:].split(" nor ")]
    elif ", without " in text:
        first, second = text.split(", without ", 1)
        clauses = [(first, False), (second, True)]
    else:
        clauses = [(text, False)]
    bounds = []
    for clause, negate in clauses:
        match = re.fullmatch(
            r"(maximum|minimum|first|last|mean|median) "
            r"(below|at most|above|at least) (-?\d+(?:\.\d+)?) ([^,]+)",
            clause, re.IGNORECASE,
        )
        if match is None:
            return label
        measure, operator, value, unit = match.groups()
        op = operators[operator.lower()]
        bounds.append(_ClinicalBound(
            measure.capitalize(), inverse[op] if negate else op, Decimal(value), unit,
        ))
    interval = _clinical_interval(bounds)
    return f"{bounds[0].measure} {interval} {bounds[0].unit}" if interval else label


def wrap_axis_tick_labels(ax: Any, coordinate: str, *, width: float | None = None) -> None:
    """Wrap complete labels at measured pixel widths, without reducing type."""
    from .presentation import wrap_figure_label

    ticks = ax.get_xticklabels() if coordinate == "x" else ax.get_yticklabels()
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    if width is None:
        width = ax.get_window_extent(renderer).width / max(len(ticks), 1) - 12
    for tick in ticks:
        tick.set_fontsize(max(7.5, tick.get_fontsize()))
        tick.set_text(wrap_figure_label(
            tick.get_text(), renderer=renderer, font=tick.get_fontproperties(),
            width=max(width, 24),
        ))
    locations = ax.get_xticks() if coordinate == "x" else ax.get_yticks()
    setter = ax.set_xticks if coordinate == "x" else ax.set_yticks
    setter(locations, [tick.get_text() for tick in ticks], fontsize=7.5)


def _normalise_display_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def _sentence_case(text: str) -> str:
    """Lowercase ordinary label words while retaining clinical notation."""
    def word_case(match: re.Match[str]) -> str:
        word = match.group()
        if word.casefold() in _CLINICAL_ACRONYMS:
            return _CLINICAL_ACRONYMS[word.casefold()]
        # Keep unknown acronyms and mixed-case notation (e.g. CRP, eGFR).
        if word.isupper() or (not word.islower() and not word.istitle()):
            return word
        return word.capitalize() if match.start() == 0 else word.lower()

    return re.sub(r"[A-Za-z][A-Za-z0-9]*", word_case, text)


def _figure_text(label: Any) -> Optional[str]:
    text = str(label or "").strip()
    if not text or _FOREIGN_SCRIPT_RE.search(text):
        return None
    text = "In-hospital mortality" if text.casefold() in {
        "in hospital mortality", "in-hospital mortality",
    } else text
    return _sentence_case(text)


def figure_language_labels(
    display_labels: Optional[Mapping[str, str]],
    descriptions: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Keep only labels a figure can show in its own (English) text.

    A variable label in another script is replaced by that variable's English
    source description in sentence case, preserving clinical notation. A sealed
    grouping's descriptions also carry its English level names (``variable=level``),
    including ones the Planner did not label.
    """

    english = {
        str(name): " ".join(str(text or "").split())
        for name, text in (descriptions or {}).items()
    }
    result: dict[str, str] = {
        name: text for name, description in english.items()
        if "=" in name and (text := _figure_text(description)) is not None
    }
    for name in tuple(result):
        scope = name.partition("=")[0]
        if (scope_text := _figure_text(english.get(scope))) is not None:
            result.setdefault(scope, scope_text)
    for key, label in (display_labels or {}).items():
        text = _figure_text(label)
        if text is not None:
            result[str(key)] = text
            continue
        description = _figure_text(english.get(str(key)))
        if description is not None:
            result[str(key)] = description
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


def _integral_code(value: Any) -> Optional[str]:
    """``"2"`` for a real number without a fraction (``2.0``, ``numpy.int64(2)``).

    Text stays as written (``"2.0"`` is a label, not a number), a boolean is
    no level code, and NaN or infinity names no level.
    """

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        return None
    return str(int(number))


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
    # A level code read back as a number (2.0) names the level "2".
    code = _integral_code(value)
    if code is not None and code != raw_value:
        exact = _figure_text((display_labels or {}).get(f"{scope}={code}"))
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
    """Render sentence-case labels without changing their clinical meaning."""

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
    rendered = [clinical_tokens.get(word.casefold(), word) for word in words]
    return _figure_text(" ".join(rendered)) or "Value"


__all__ = [
    "binary_contrast_label",
    "binary_scope_label",
    "display_label",
    "figure_language_labels",
    "grouped_axis_labels",
    "label_lookup",
    "scoped_label_lookup",
]
