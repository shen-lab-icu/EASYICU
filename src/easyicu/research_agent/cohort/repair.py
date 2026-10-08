"""Translate a plan's prose 纳排 into typed CTAS predicates so the framework
can materialise and enforce the analysis cohort.

Why this exists
---------------
Bench-style runs disable the deterministic planner fallback
(``enable_deterministic_planner_fallback=False``) to measure the real hosted
model honestly. A weak model then commonly emits a probe-only initial plan and
grows the real plan via the replanner — a plan that carries a
``01_cohort_definition`` step but leaves ``plan.cohort`` structurally empty:
the 纳排 lives only in the step's prose ``intent``. ``materialize_locked_\
analysis_cohort`` then no-ops (``no_definition``) and every downstream step
silently runs on the unfiltered universe (E1 run12).

This module extracts the inclusion/exclusion criteria the agent **already
stated in prose**, grounds them in the universe's actual columns, and returns a
typed :class:`CohortDefinition` the materialiser can apply. It only
*translates* the agent's stated criteria — it never invents 纳排. That keeps the
L2-autonomy boundary intact: the framework enforces the agent's cohort, it does
not impose one.

A predicate's ``time_window`` states the window its criterion reads.  The
builder filters a column as it was summarized, and reads an event over a
finite window by the event's own time (``cohort.schema``).  So a window the
prose states is carried, and checked against the column like any plan's.  A
criterion the prose states without a window reads its column as the column
was materialized: the column's own window, the whole stay for an event the
stay records whole (an outcome such as death), and a first-24 h default only
for a column the context records no window for (a value fixed at admission,
which no window summarizes).  The aggregation is audit metadata: each
predicate names its column directly.  An extractor that leaves out a window
the prose states makes the criterion read its column's window; the prompt
asks for the window, and nothing else can recover it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Sequence

from .schema import (
    CohortDefinition,
    CohortSchemaError,
    ConceptPredicate,
    TimeWindow,
    cohort_concept_id_scope,
    validate_cohort_definition,
)
from ..providers.protocol import LLMClient, LLMMessage
from ..research_context.materialization_window import context_column_windows
from ..research_context.stay_events import whole_stay_event_columns
from ..providers.factory import authorized_complete

# Operators ``build_cohort._apply_op`` actually implements.
_SUPPORTED_OPS = (
    ">=",
    "<=",
    ">",
    "<",
    "==",
    "!=",
    "in",
    "not_in",
    "missing",
    "not_missing",
)

# The window of a column the context records no window for (a value fixed at
# admission, which no window summarizes), and the audit aggregation every
# predicate records.
_DEFAULT_TIME_WINDOW = {
    "anchor": "icu_admit",
    "start_offset_hours": 0,
    "end_offset_hours": 24,
}
_DEFAULT_AGGREGATION = "first"

_SYSTEM = (
    "You translate an already-written cohort-definition step into typed "
    "inclusion/exclusion predicates. You do NOT invent criteria: translate only "
    "what the prose explicitly states. If the prose states no concrete, "
    "column-checkable criterion, return an empty inclusion list."
)

_NO_FILTER_INTENT = re.compile(
    r"\b(?:preserv(?:e|ing)|retain(?:ing)?)\s+(?:the\s+)?"
    r"(?:full|entire|complete)\s+(?:denominator|cohort|population)\b|"
    r"\b(?:all|every)\s+(?:supplied|provided|available|input)\s+"
    r"(?:icu\s+)?(?:stay|stays|row|rows|record|records)\b",
    re.IGNORECASE,
)
_EXPLICIT_FILTER_INTENT = re.compile(
    r"(?:>=|<=|==|!=|(?<!-)[<>](?!=))|"
    r"\b(?:includ(?:e|es|ed|ing)|exclud(?:e|es|ed|ing)|eligib(?:le|ility)|"
    r"only|minimum|maximum|at\s+least|at\s+most|greater\s+than|less\s+than|"
    r"adult|aged?|missing|non[- ]?missing|recorded|required?)\b",
    re.IGNORECASE,
)


def _explicitly_unfiltered_cohort(prose: str) -> bool:
    """Return whether prose declares a full denominator with no eligibility."""

    return bool(_NO_FILTER_INTENT.search(prose)) and not bool(
        _EXPLICIT_FILTER_INTENT.search(prose)
    )


def _user_prompt(*, cohort_prose: str, universe_columns: Sequence[str]) -> str:
    cols = ", ".join(sorted(str(c) for c in universe_columns))
    ops = ", ".join(_SUPPORTED_OPS)
    return (
        "COHORT-DEFINITION STEP PROSE (the analysis-population criteria the "
        "agent already chose):\n"
        f"{cohort_prose.strip()}\n\n"
        "AVAILABLE PER-STAY COLUMNS (use these exact names as concept_id; do "
        "not reference any column not in this list):\n"
        f"{cols}\n\n"
        f"ALLOWED OPERATORS: {ops}\n\n"
        "Return ONLY a JSON object of this shape (no prose, no code fence):\n"
        '{"inclusion": [{"concept_id": "<column>", "op": "<operator>", '
        '"value": <number|string|list|null>}], "exclusion": [...]}\n\n'
        "Rules:\n"
        "- One predicate per explicitly-stated criterion (e.g. 'adults' over an "
        "`age` column -> {concept_id: age, op: >=, value: 18}).\n"
        "- Only use concept_id values that appear verbatim in AVAILABLE "
        "COLUMNS. Drop any criterion you cannot map to a listed column.\n"
        "- When the prose states the time window a criterion reads (e.g. "
        "'within 24 h of ICU admission'), give it as \"time_window\": "
        '{"anchor": "icu_admission", "start_offset_hours": <hours>, '
        '"end_offset_hours": <hours>}; otherwise omit time_window. Omit '
        "aggregation.\n"
        '- If nothing maps, return {"inclusion": [], "exclusion": []}.'
    )


def _strip_fence(text: str) -> str:
    text = text.strip()
    if "```" in text:
        # keep the content between the first pair of fences, else drop fence lines
        parts = text.split("```")
        # parts like ['', 'json\n{...}', ''] -> take the largest brace-bearing chunk
        candidates = [p for p in parts if "{" in p and "}" in p]
        if candidates:
            text = max(candidates, key=len)
            # drop a leading language tag line (e.g. "json")
            if "\n" in text and "{" not in text.split("\n", 1)[0]:
                text = text.split("\n", 1)[1]
    return text.strip()


def _loads_json_object(text: str) -> Optional[dict]:
    text = _strip_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _column_windows(context: Any, columns: set[str]) -> dict[str, dict]:
    """The window each column is read over when the prose states none."""

    if context is None:
        return {}
    whole_stay = whole_stay_event_columns(context)
    windows = context_column_windows(context)
    found: dict[str, dict] = {}
    for column in columns:
        window = windows.get(column)
        if window is not None and window.anchor is not None:
            found[column] = {
                "anchor": window.anchor,
                "start_offset_hours": window.start_hours,
                "end_offset_hours": window.end_hours,
            }
        elif column in whole_stay:
            found[column] = {
                "anchor": "icu_admission",
                "start_offset_hours": 0,
                "end_offset_hours": "inf",
            }
    return found


class _StatedWindowUnreadable(ValueError):
    """The translator gave a criterion a time window that names no window."""


def _predicate_from_minimal(
    item: Any,
    *,
    columns: set[str],
    column_windows: Mapping[str, dict] | None = None,
) -> Optional[ConceptPredicate]:
    if not isinstance(item, dict):
        return None
    concept_id = str(item.get("concept_id") or "").strip()
    op = str(item.get("op") or "").strip()
    if concept_id not in columns or op not in _SUPPORTED_OPS:
        return None
    stated = item.get("time_window")
    window = stated or (column_windows or {}).get(concept_id) or _DEFAULT_TIME_WINDOW
    aggregation = str(item.get("aggregation") or _DEFAULT_AGGREGATION)
    try:
        time_window = TimeWindow.from_dict(window)
    except CohortSchemaError as exc:
        if stated:
            # Dropping the criterion would apply the others alone, a wider
            # cohort than the prose states.
            raise _StatedWindowUnreadable(str(exc)) from exc
        return None
    value = item.get("value", None)
    if op in {"missing", "not_missing"}:
        value = None
    return ConceptPredicate(
        concept_id=concept_id,
        time_window=time_window,
        aggregation=aggregation,
        op=op,
        value=value,
    )


def extract_cohort_definition_from_prose(
    *,
    cohort_prose: str,
    universe_columns: Sequence[str],
    llm: LLMClient,
    name: str = "primary",
    context: Any = None,
) -> Optional[CohortDefinition]:
    """Return a validated :class:`CohortDefinition` from the cohort step prose,
    or ``None`` when nothing column-checkable can be extracted.

    The result is grounded: every predicate's ``concept_id`` is one of
    ``universe_columns`` and its operator is one ``build_cohort`` implements.
    A criterion the prose states without a window reads its column over the
    window ``context`` records for it (``_column_windows``).  A window the
    translator states that names no window (a bound that is no number, an end
    not after its start, a missing anchor) fails the whole translation: the
    other criteria alone would select a wider cohort than the prose states.
    Pre-materialised columns are visible only inside a local validation scope;
    extraction never widens the process registry.
    """
    if not (cohort_prose or "").strip() or not universe_columns:
        return None
    if _explicitly_unfiltered_cohort(cohort_prose):
        return None
    columns = {str(c) for c in universe_columns}
    column_windows = _column_windows(context, columns)
    try:
        raw = authorized_complete(
            llm,
            [
                LLMMessage(role="system", content=_SYSTEM),
                LLMMessage(
                    role="user",
                    content=_user_prompt(
                        cohort_prose=cohort_prose, universe_columns=columns
                    ),
                ),
            ],
            max_tokens=800,
            temperature=0.0,
        )
    except Exception:
        return None
    data = _loads_json_object(raw or "")
    if data is None:
        return None

    # Predicate construction itself validates concept ids, so the scope must
    # cover construction as well as the final definition check. These are
    # actual columns in this run's materialized universe, not dictionary claims
    # and never process-global registrations.
    with cohort_concept_id_scope(columns):
        inclusion = []
        exclusion = []
        try:
            for kind, found in (("inclusion", inclusion), ("exclusion", exclusion)):
                for item in data.get(kind) or []:
                    pred = _predicate_from_minimal(
                        item, columns=columns, column_windows=column_windows
                    )
                    if pred is not None:
                        found.append(pred)
        except _StatedWindowUnreadable:
            return None

        if not (inclusion or exclusion):
            return None

        definition = CohortDefinition(
            name=name,
            inclusion=tuple(inclusion),
            exclusion=tuple(exclusion),
        )
        try:
            validate_cohort_definition(definition)
        except CohortSchemaError:
            return None
    return definition
