"""Fail-safe interpretation of metadata-bound mortality horizons."""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Optional, Sequence

from ..schema import ResearchContext


_MORTALITY_COLUMN_NAMES = {
    "death",
    "mortality",
    "death_icu",
    "icu_death",
    "icu_mortality",
    "death_hosp",
    "hospital_death",
    "hospital_mortality",
    "hospital_expire_flag",
    "death_28d",
    "mortality_28d",
    "death_30d",
    "mortality_30d",
}
_MORTALITY_EVENT_TIME_NAMES = {
    "death_time",
    "time_to_death",
    "days_to_death",
    "hours_to_death",
    "event_time",
    "event_time_hours",
    "followup_time",
    "followup_time_hours",
    "follow_up_time",
    "censor_time",
    "censoring_time",
    "event_observed",
}


def _script_tree(script_text: str) -> Optional[ast.AST]:
    try:
        return ast.parse(script_text or "")
    except (SyntaxError, TypeError, ValueError):
        return None


def _script_column_references(tree: ast.AST) -> set[str]:
    """Collect column-shaped references, excluding free-text metadata."""

    references: set[str] = set()
    tracked = _MORTALITY_COLUMN_NAMES | _MORTALITY_EVENT_TIME_NAMES
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            token = node.value.strip().lower()
            if token in tracked:
                references.add(token)
        if isinstance(node, ast.Name) and node.id.lower() in tracked:
            references.add(node.id.lower())
        if isinstance(node, ast.Subscript):
            value = node.slice
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                token = value.value.strip().lower()
                if token in tracked:
                    references.add(token)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in {"get", "pop"} or not node.args:
                continue
            value = node.args[0]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                token = value.value.strip().lower()
                if token in tracked:
                    references.add(token)
    return references


def _script_uses_bound_outcome(*, script_text: str, outcome: str) -> bool:
    tree = _script_tree(script_text)
    return bool(
        tree is not None
        and str(outcome or "").strip().lower() in _script_column_references(tree)
    )


def _phrase_is_negated(text: str, start: int) -> bool:
    clause = re.split(r"[.;!?\n]", text[max(0, start - 100) : start])[-1]
    negations = list(
        re.finditer(
            r"\b(?:not|never|without|no|do not|does not|is not|are not)\b",
            clause,
        )
    )
    if not negations:
        return False
    tail = clause[negations[-1].end() :]
    return "but" not in tail.split()


def _has_asserted_label(text: str, patterns: Sequence[str]) -> bool:
    lowered = str(text or "").lower()
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            if not _phrase_is_negated(lowered, match.start()):
                return True
    return False


def _script_has_conflicting_mortality_semantics(
    *,
    script_text: str,
    outcome: str,
    source: str,
) -> bool:
    """Return true for real alternate outcomes, labels, or event-time logic."""

    tree = _script_tree(script_text)
    if tree is None:
        return True
    references = _script_column_references(tree)
    if references.intersection(_MORTALITY_EVENT_TIME_NAMES):
        return True
    allowed_column = str(outcome or "").strip().lower()
    if any(
        reference in _MORTALITY_COLUMN_NAMES and reference != allowed_column
        for reference in references
    ):
        return True

    conflicting_labels = {
        "icu_mortality": (
            r"\b(?:in[- ]?hospital|hospital) mortality\b",
            r"\b(?:28|30)[- ]?day mortality\b",
            r"\bfixed[- ](?:window|horizon) mortality\b",
        ),
        "hospital_mortality": (
            r"\bicu mortality\b",
            r"\b(?:28|30)[- ]?day mortality\b",
            r"\bfixed[- ](?:window|horizon) mortality\b",
        ),
        "mortality_28d": (
            r"\bicu mortality\b",
            r"\b(?:in[- ]?hospital|hospital) mortality\b",
            r"\b30[- ]?day mortality\b",
        ),
        "mortality_30d": (
            r"\bicu mortality\b",
            r"\b(?:in[- ]?hospital|hospital) mortality\b",
            r"\b28[- ]?day mortality\b",
        ),
    }
    string_literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    if any(
        _has_asserted_label(value, conflicting_labels[source])
        for value in string_literals
    ):
        return True

    # Preserve errors when a fixed horizon is constructed from the bound
    # mortality flag and a time/length expression, even without a label.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.BinOp, ast.BoolOp)):
            continue
        expression = ast.unparse(node).lower()
        if (
            re.search(r"\b(?:death|mortality)\b", expression)
            and re.search(
                r"\b(?:time|followup|follow_up|los(?:_icu)?|days?|hours?)\b",
                expression,
            )
            and re.search(r"\b(?:28|30|672|720)(?:\.0)?\b", expression)
        ):
            return True
    return False


def _script_copies_named_full_stay_window(
    *, context: ResearchContext, script_text: str, outcome: str
) -> bool:
    full_stay = next(
        (
            window
            for window in context.time_windows
            if re.sub(r"[^a-z0-9]+", "_", window.name.lower()).strip("_")
            == "full_stay"
        ),
        None,
    )
    if full_stay is None:
        return False
    tree = _script_tree(script_text)
    if tree is None:
        return False

    def _contains_bound_window(value: Any) -> bool:
        if isinstance(value, dict):
            concept = str(
                value.get("concept_id")
                or value.get("column")
                or value.get("target")
                or ""
            ).strip()
            window = value.get("time_window")
            if concept == outcome and isinstance(window, dict):
                try:
                    start = float(
                        window.get("start_offset_hours", window.get("start_hours"))
                    )
                    end = float(
                        window.get("end_offset_hours", window.get("end_hours"))
                    )
                except (TypeError, ValueError):
                    pass
                else:
                    if math.isclose(start, full_stay.start_hours) and math.isclose(
                        end, full_stay.end_hours
                    ):
                        return True
            return any(_contains_bound_window(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(_contains_bound_window(item) for item in value)
        return False

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Dict, ast.List, ast.Tuple)):
            continue
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            continue
        if _contains_bound_window(value):
            return True
    return False


def _finding_claims_mortality_horizon_mismatch(text: str) -> bool:
    lowered = str(text or "").lower()
    return bool(
        re.search(r"\b(?:death|mortality)\b", lowered)
        and any(
            token in lowered
            for token in (
                "fixed-window",
                "fixed window",
                "fixed-horizon",
                "fixed horizon",
                "follow-up horizon",
                "28-day",
                "30-day",
                "720 hour",
                "720-hour",
                "0–720",
                "0-720",
                "event-time",
                "time filter",
            )
        )
    )


__all__ = [
    "_finding_claims_mortality_horizon_mismatch",
    "_script_copies_named_full_stay_window",
    "_script_has_conflicting_mortality_semantics",
    "_script_uses_bound_outcome",
]
