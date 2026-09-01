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
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient


def _stub(reply: str) -> ScriptedMockLLMClient:
    return ScriptedMockLLMClient([reply])


_COLUMNS = ["stay_id", "age", "los_icu", "sofa2", "death"]


def test_extracts_grounded_predicates():
    llm = _stub(
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
    llm = _stub(
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
    llm = _stub(
        json.dumps(
            {"inclusion": [{"concept_id": "age", "op": "between", "value": [18, 90]}]}
        )
    )
    definition = extract_cohort_definition_from_prose(
        cohort_prose="adults", universe_columns=_COLUMNS, llm=llm
    )
    assert definition is None  # nothing valid remained


def test_returns_none_on_empty_or_garbage():
    assert (
        extract_cohort_definition_from_prose(
            cohort_prose="adults", universe_columns=_COLUMNS, llm=_stub("not json")
        )
        is None
    )
    assert (
        extract_cohort_definition_from_prose(
            cohort_prose="adults",
            universe_columns=_COLUMNS,
            llm=_stub(json.dumps({"inclusion": [], "exclusion": []})),
        )
        is None
    )
    # no prose / no columns → no LLM call, no definition
    assert (
        extract_cohort_definition_from_prose(
            cohort_prose="", universe_columns=_COLUMNS, llm=_stub("{}")
        )
        is None
    )


def test_full_denominator_prose_cannot_be_laundered_into_a_filter():
    llm = _stub(
        json.dumps(
            {
                "inclusion": [
                    {"concept_id": "sofa2", "op": ">=", "value": 0},
                ],
                "exclusion": [],
            }
        )
    )

    definition = extract_cohort_definition_from_prose(
        cohort_prose=(
            "Construct the analysis cohort from all supplied ICU stays while "
            "preserving the full denominator and both exposure levels."
        ),
        universe_columns=_COLUMNS,
        llm=llm,
    )

    assert definition is None
    assert llm.calls == []


def test_explicit_eligibility_is_not_hidden_by_denominator_reporting():
    llm = _stub(
        json.dumps(
            {
                "inclusion": [
                    {"concept_id": "age", "op": ">=", "value": 18},
                ],
                "exclusion": [],
            }
        )
    )

    definition = extract_cohort_definition_from_prose(
        cohort_prose=(
            "Include adults age >= 18 and preserve the full denominator for "
            "attrition reporting."
        ),
        universe_columns=_COLUMNS,
        llm=llm,
    )

    assert definition is not None
    assert definition.inclusion[0].concept_id == "age"
    assert len(llm.calls) == 1


def test_missing_operator_needs_no_value():
    llm = _stub(
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


def test_extraction_does_not_leak_universe_columns_into_process_state():
    """Extraction must answer a question, not permanently widen validation.

    ``extract_cohort_definition_from_prose`` used to call
    ``register_cohort_concept_ids(columns)`` -- a permanent process-wide
    registration -- so that the definition it had just built could pass
    ``validate_cohort_definition``. Every caller therefore leaked its universe
    columns into every later validation in the same process.

    Measured 2026-08-18 at ``4868315``: running this module left
    ``{age, death, los_icu, sofa2, stay_id}`` registered for the rest of the
    process, and ``tests/research_agent/test_plan_lifecycle_authority.py``
    then stopped raising on an unsealed materialized ``stay_id`` -- two
    fail-closed assertions passing only because nothing had touched cohort
    repair first. That is the failure this test exists to keep dead: the
    minimal ordered reproduction was exactly these two files, in this order.

    The run-lifetime registration the execution layer genuinely needs is now
    made explicitly by ``execution/phase_support.py``, which owns the run.
    """

    from easyicu.research_agent.planning import cohort_contract

    cohort_contract.clear_cohort_concept_ids()
    assert not cohort_contract.concept_id_exists("stay_id")

    llm = _stub(
        json.dumps(
            {
                "inclusion": [{"concept_id": "age", "op": ">=", "value": 18}],
                "exclusion": [],
            }
        )
    )
    definition = extract_cohort_definition_from_prose(
        cohort_prose="adults",
        universe_columns=_COLUMNS,
        llm=llm,
    )

    # The validation still had to succeed -- the scope is what makes the
    # pre-materialised columns visible *while* validating.
    assert definition is not None
    assert [pred.concept_id for pred in definition.inclusion] == ["age"]

    # ... and none of it survives the call. Only ``stay_id`` is checkable
    # through the public reader -- age/los_icu/sofa2/death are genuine
    # dictionary concepts, so they answer True either way and would make this
    # assertion vacuous. ``stay_id`` is the pre-materialised column, which is
    # exactly the id whose leak silenced the lifecycle guards.
    assert not cohort_contract.concept_id_exists(
        "stay_id"
    ), "stay_id leaked into process-global cohort concept state"
    # The extra registry is the leak surface itself: assert it is untouched so
    # a future dictionary addition cannot make the check above vacuous too.
    assert cohort_contract._EXTRA_COHORT_CONCEPT_IDS == set()
