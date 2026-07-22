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

The CTAS ``time_window`` / ``aggregation`` of each predicate are audit metadata
only; ``build_cohort`` filters by ``concept_id``/``op``/``value`` against
already-materialised per-stay columns. So a missing window/aggregation is
filled with a first-24h default rather than rejected.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from .schema import (
    CohortDefinition,
    CohortSchemaError,
    ConceptPredicate,
    TimeWindow,
    register_cohort_concept_ids,
    validate_cohort_definition,
)
from ..providers.protocol import LLMClient, LLMMessage
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

# Audit-only defaults for a per-stay first-24h summary column. build_cohort
# ignores these when filtering; they exist so the predicate validates and the
# locked definition records a window/aggregation.
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
        "- Omit time_window/aggregation; the framework fills audit defaults.\n"
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


def _predicate_from_minimal(
    item: Any, *, columns: set[str]
) -> Optional[ConceptPredicate]:
    if not isinstance(item, dict):
        return None
    concept_id = str(item.get("concept_id") or "").strip()
    op = str(item.get("op") or "").strip()
    if concept_id not in columns or op not in _SUPPORTED_OPS:
        return None
    window = item.get("time_window") or _DEFAULT_TIME_WINDOW
    aggregation = str(item.get("aggregation") or _DEFAULT_AGGREGATION)
    try:
        time_window = TimeWindow.from_dict(window)
    except CohortSchemaError:
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
) -> Optional[CohortDefinition]:
    """Return a validated :class:`CohortDefinition` from the cohort step prose,
    or ``None`` when nothing column-checkable can be extracted.

    The result is grounded: every predicate's ``concept_id`` is one of
    ``universe_columns`` and its operator is one ``build_cohort`` implements.
    ``register_cohort_concept_ids`` is called so these pre-materialised columns
    pass predicate validation.
    """
    if not (cohort_prose or "").strip() or not universe_columns:
        return None
    columns = {str(c) for c in universe_columns}
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

    inclusion = []
    for item in data.get("inclusion") or []:
        pred = _predicate_from_minimal(item, columns=columns)
        if pred is not None:
            inclusion.append(pred)
    exclusion = []
    for item in data.get("exclusion") or []:
        pred = _predicate_from_minimal(item, columns=columns)
        if pred is not None:
            exclusion.append(pred)

    if not (inclusion or exclusion):
        return None

    definition = CohortDefinition(
        name=name,
        inclusion=tuple(inclusion),
        exclusion=tuple(exclusion),
    )
    # Allow these pre-materialised universe columns through predicate validation
    # (they are not dictionary concepts).
    register_cohort_concept_ids(columns)
    try:
        validate_cohort_definition(definition)
    except CohortSchemaError:
        return None
    return definition
