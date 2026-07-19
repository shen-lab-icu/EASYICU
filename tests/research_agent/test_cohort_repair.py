"""Unit tests for ``cohort.repair.extract_cohort_definition_from_prose``.

It translates a cohort step's prose 纳排 into typed CTAS predicates, grounded in
the universe's actual columns. It must only emit predicates over real columns
with implemented operators, and return ``None`` when nothing maps (so the caller
falls back to the auditable contract error rather than enforcing an invented
cohort).
"""

from __future__ import annotations

import json

from easyicu.research_agent.cohort.repair import (
    extract_cohort_definition_from_prose,
)


class _StubLLM:
    name = "stub"

    def __init__(self, reply: str):
        self._reply = reply

    def complete(self, messages, *, max_tokens=2048, temperature=0.2):
        return self._reply


_COLUMNS = ["stay_id", "age", "los_icu", "sofa2", "death"]


def test_extracts_grounded_predicates():
    llm = _StubLLM(
        json.dumps(
            {
                "inclusion": [
                    {"concept_id": "age", "op": ">=", "value": 18},
                    {"concept_id": "los_icu", "op": ">=", "value": 1},
                ],
                "exclusion": [],
            }
        )
    )
    definition = extract_cohort_definition_from_prose(
        cohort_prose="adults with ICU LoS >= 1 day",
        universe_columns=_COLUMNS,
        llm=llm,
    )
    assert definition is not None
    assert [p.concept_id for p in definition.inclusion] == ["age", "los_icu"]
    assert [p.op for p in definition.inclusion] == [">=", ">="]
    # audit defaults filled
    assert definition.inclusion[0].aggregation == "first"
    assert definition.inclusion[0].time_window.anchor == "icu_admit"


def test_drops_predicates_on_unknown_columns():
    llm = _StubLLM(
        json.dumps(
            {
                "inclusion": [
                    {"concept_id": "age", "op": ">=", "value": 18},
                    {"concept_id": "not_a_real_column", "op": ">=", "value": 1},
                ]
            }
        )
    )
    definition = extract_cohort_definition_from_prose(
        cohort_prose="adults", universe_columns=_COLUMNS, llm=llm
    )
    assert definition is not None
    assert [p.concept_id for p in definition.inclusion] == ["age"]


def test_rejects_unsupported_operator():
    llm = _StubLLM(
        json.dumps({"inclusion": [{"concept_id": "age", "op": "between", "value": [18, 90]}]})
    )
    definition = extract_cohort_definition_from_prose(
        cohort_prose="adults", universe_columns=_COLUMNS, llm=llm
    )
    assert definition is None  # nothing valid remained


def test_returns_none_on_empty_or_garbage():
    assert (
        extract_cohort_definition_from_prose(
            cohort_prose="adults", universe_columns=_COLUMNS, llm=_StubLLM("not json")
        )
        is None
    )
    assert (
        extract_cohort_definition_from_prose(
            cohort_prose="adults",
            universe_columns=_COLUMNS,
            llm=_StubLLM(json.dumps({"inclusion": [], "exclusion": []})),
        )
        is None
    )
    # no prose / no columns → no LLM call, no definition
    assert (
        extract_cohort_definition_from_prose(
            cohort_prose="", universe_columns=_COLUMNS, llm=_StubLLM("{}")
        )
        is None
    )


def test_missing_operator_needs_no_value():
    llm = _StubLLM(
        json.dumps({"inclusion": [{"concept_id": "sofa2", "op": "not_missing"}]})
    )
    definition = extract_cohort_definition_from_prose(
        cohort_prose="patients with a recorded SOFA-2",
        universe_columns=_COLUMNS,
        llm=llm,
    )
    assert definition is not None
    assert definition.inclusion[0].op == "not_missing"
    assert definition.inclusion[0].value is None
