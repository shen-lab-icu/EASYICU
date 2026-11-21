"""A recorded condition the question restricts its patients to reaches the cohort.

A question such as "Among adult ICU stays with AKI, ..." restricts the study to
a condition EasyICU records.  The host passes it to planning as a constraint
(``stated_population_conditions``: the question's words and the concepts that
record it).  A plan whose cohort neither applies nor lists it goes back to the
Planner with the reason; when the Planner's attempts are spent, the host states
it itself: present on the recorded concept over the window it was summarized
in, missing excluded.  The cohort records that the host stated it, for the plan
card.  A Planner that states it is not overridden, and nothing happens without
the host's constraint or when the caller binds every input row (S1757).
Fixtures are generic: the condition is AKI, not the case that found the gap.
"""

from __future__ import annotations

import copy
import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure

from easyicu.research_agent.planning.progressive_compiler import (
    ProgressivePlanCompileError,
    require_stated_population_conditions,
)
from easyicu.research_agent.planning.progressive_contract import ProgressiveCohortIntent
from easyicu.research_agent.planning.stated_population import (
    StatedCondition,
    host_statements,
    next_criterion_ids,
    stated_conditions,
)
from easyicu.research_agent.schema import ConceptDescriptor, ResearchContext, VariableRole

from tests.research_agent.planning import family_spec_fixtures as fixtures
from tests.research_agent.planning.family_spec_fixtures import _context, _request, _run, _spec_payload
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _progressive_context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

_CODE = "progressive_stated_population_condition_missing"
_QUESTION_WORDS = "Among adult ICU stays with AKI,"


@pytest.fixture(autouse=True)
def _aki_label(monkeypatch):
    monkeypatch.setitem(fixtures.LABELS, "aki_max", "AKI in the first 24 h")


def _aki_context(*, stated: bool = True, quote: str = "AKI") -> ResearchContext:
    context = _context()
    question = context.research_question.replace("Among adult ICU stays,", _QUESTION_WORDS)
    assert question != context.research_question
    constraints = json.loads(context.user_preferences.data_constraints)
    if stated:
        constraints["stated_population_conditions"] = [{"quote": quote, "concepts": ["aki"]}]
    aki = ConceptDescriptor(
        name="aki_max", role=VariableRole.OTHER, dtype="int64",
        observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
        source_concept="aki", unit_normalization="window_presence_max",
        analysis_window="icu_admission[0,24]h",
    )
    return context.model_copy(update={
        "research_question": question,
        "variables": [*context.variables, aki],
        "user_preferences": context.user_preferences.model_copy(
            update={"data_constraints": json.dumps(constraints)}
        ),
    })


def _predicate(concept: str, op: str, number: float, *, aggregation: str = "first") -> dict:
    return {
        "concept_id": concept, "anchor": "icu_admission", "start_offset_hours": 0,
        "end_offset_hours": 24.0, "aggregation": aggregation, "op": op,
        "value": {"mode": "number", "string_value": None, "number_value": number,
                  "boolean_value": None, "string_list": [], "number_list": []},
    }


_OLDER = {"criteria": [{"criterion": "older adults", "concept_ids": ["age"]}],
          "inclusion": [_predicate("age", ">=", 65)], "exclusion": []}
_TYPED_ADULT = {"criteria": [], "inclusion": [], "exclusion": [], "population_spec": {"criteria": [
    {"id": "c1", "quote": "adult", "source": "question", "role": "include",
     "kind": "age_years", "min_years": 18, "max_years": None},
]}}
_AKI_WORDS_ONLY = {"criteria": [], "inclusion": [], "exclusion": [], "population_spec": {"criteria": [
    {"id": "c1", "quote": "adult", "source": "question", "role": "include",
     "kind": "age_years", "min_years": 18, "max_years": None},
    {"id": "c2", "quote": "AKI", "source": "question", "role": "include",
     "kind": "not_typed", "why": "no recorded variable named in the request"},
]}}
_WITH_AKI = {"criteria": [{"criterion": "with AKI", "concept_ids": ["aki"]}],
             "inclusion": [_predicate("aki", "==", 1, aggregation="max")], "exclusion": []}


def _inclusion(result) -> list[tuple]:
    return [(p.concept_id, p.op, float(p.value), p.aggregation) for p in result.output.cohort.inclusion]


def _feedback(llm) -> list[bool]:
    return [_CODE in call[0][-1].content for call in llm.calls[1:]]


@pytest.mark.parametrize("population", [None, _OLDER, _TYPED_ADULT], ids=["none", "legacy", "typed"])
def test_a_condition_the_planner_leaves_out_twice_is_stated_by_the_host(population) -> None:
    context = _aki_context()
    payload = json.dumps(_spec_payload(_request(context), population=population))

    llm, result = _run(context, [payload] * 4)

    assert len(llm.calls) == 3 and _feedback(llm) == [True, True]
    assert "'AKI' (recorded by aki)" in llm.calls[1][0][-1].content
    assert result.output.cohort.selection_mode == "predicate_filtered"
    inclusion = _inclusion(result)
    assert ("aki", "==", 1.0, "max") in inclusion
    assert ("age", ">=", 18.0, "first") in inclusion, "the template's adult bound is kept"
    stated = {"quote": "AKI", "concept": "aki", "start_hours": 0.0, "end_hours": 24.0}
    cohort = result.facts.skeleton.cohort
    assert [item.model_dump(mode="json") for item in cohort.host_stated_population] == [stated]
    assert result.facts.prompt_metrics["host_stated_population_conditions"] == [stated]


def test_a_planner_that_states_the_condition_when_asked_again_keeps_its_own_cohort() -> None:
    context = _aki_context()
    request = _request(context)
    without = json.dumps(_spec_payload(request, population=None))
    with_aki = json.dumps(_spec_payload(request, population=_WITH_AKI))

    llm, result = _run(context, [without, with_aki])

    assert len(llm.calls) == 2 and _feedback(llm) == [True]
    assert ("aki", "==", 1.0, "max") in _inclusion(result)
    assert not result.facts.skeleton.cohort.host_stated_population
    assert "host_stated_population_conditions" not in result.facts.prompt_metrics


@pytest.mark.parametrize("context", [
    pytest.param(lambda: _aki_context(stated=False), id="no host constraint"),
    pytest.param(lambda: _aki_context(quote="sepsis"), id="words not in the question"),
])
def test_without_a_stated_condition_nothing_is_asked_or_added(context) -> None:
    built = context()
    llm, result = _run(built, [json.dumps(_spec_payload(_request(built), population=None))])

    assert len(llm.calls) == 1
    assert all(item[0] != "aki" for item in _inclusion(result))
    assert not result.facts.skeleton.cohort.host_stated_population


def test_the_progressive_foundation_guard_reads_predicates_columns_and_typed_words() -> None:
    context = _aki_context()
    base = {"name": "stays", "selection_mode": "predicate_filtered",
            "inclusion": [_predicate("age", ">=", 18)], "exclusion": []}
    words_only = {**base, "population_criteria": [{"criterion": "stays with AKI", "concept_ids": []}]}
    for unaccounted in (base, words_only):
        with pytest.raises(ProgressivePlanCompileError) as refused:
            require_stated_population_conditions(
                ProgressiveCohortIntent.model_validate(unaccounted), context=context
            )
        assert refused.value.reason_code == _CODE

    for accounted in (
        {**base, "inclusion": [*base["inclusion"], _predicate("aki_max", "==", 1, aggregation="max")]},
        {**base, "population_criteria": [{"criterion": "stays with AKI", "concept_ids": ["age"]}]},
    ):
        require_stated_population_conditions(ProgressiveCohortIntent.model_validate(accounted), context=context)


def test_the_host_states_each_condition_over_its_recorded_window() -> None:
    context = _aki_context()
    conditions = stated_conditions(context, allowed=["aki", "age"])
    assert conditions == (StatedCondition(quote="AKI", concepts=("aki",)),)
    assert stated_conditions(context, allowed=["age"]) == ()

    (statement,) = host_statements(context, conditions)
    assert (statement.anchor, statement.start_hours, statement.end_hours) == ("icu_admission", 0.0, 24.0)
    assert statement.typed_criterion("c2") == {
        "id": "c2", "quote": "AKI", "source": "question", "role": "include",
        "kind": "condition_present", "concepts_all_of": ["aki"],
        "window": {"start_hours": 0.0, "end_hours": 24.0}, "missing_policy": "exclude",
    }
    assert next_criterion_ids(["c1", "c3"], 2) == ["c2", "c4"]



def _progressive_aki_context() -> ResearchContext:
    context = _progressive_context()
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints["stated_population_conditions"] = [{"quote": "AKI", "concepts": ["aki"]}]
    aki = ConceptDescriptor(
        name="aki_max", role=VariableRole.OTHER, dtype="int64",
        observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
        source_concept="aki", unit_normalization="window_presence_max",
        analysis_window="icu_admission[0,24]h",
    )
    return context.model_copy(update={
        "research_question": "Among ICU patients with AKI, e" + context.research_question[1:],
        "variables": [*context.variables, aki],
        "user_preferences": context.user_preferences.model_copy(
            update={"data_constraints": json.dumps(constraints)}
        ),
    })


def _progressive_run(*foundations: dict):
    llm = ScriptedMockLLMClient([
        json.dumps(_outline_payload()),
        *(json.dumps(item) for item in foundations),
        *(json.dumps(item) for item in _materialization_payloads()),
    ])
    return llm, ProgressivePlannerAgent(llm)


def test_a_progressive_foundation_without_the_condition_is_asked_again_then_stops_with_its_reason() -> None:
    unfiltered = _foundation_payload()
    filtered = copy.deepcopy(unfiltered)
    filtered["foundation"]["cohort"] = {
        "name": "stays with AKI", "selection_mode": "predicate_filtered", "exclusion": [],
        "inclusion": [_predicate("aki_max", "==", 1, aggregation="max")],
    }

    llm, agent = _progressive_run(unfiltered, filtered)
    plan = agent.run(_progressive_aki_context())
    assert _CODE in llm.calls[2][0][-1].content
    assert [(p.concept_id, p.op, float(p.value)) for p in plan.cohort.inclusion] == [("aki_max", "==", 1.0)]

    llm, agent = _progressive_run(unfiltered, unfiltered)
    with pytest.raises(StructuredResponseFailure, match=_CODE):
        agent.run(_progressive_aki_context())


def test_a_planner_cannot_write_the_host_stated_record() -> None:
    marked = _foundation_payload()
    marked["foundation"]["cohort"]["host_stated_population"] = [
        {"quote": "AKI", "concept": "aki", "start_hours": 0, "end_hours": 24},
    ]
    llm, agent = _progressive_run(marked, marked)
    with pytest.raises(StructuredResponseFailure, match="progressive_host_stated_population_from_planner"):
        agent.run(_progressive_aki_context())


def test_a_statement_the_family_cannot_hold_leaves_the_plan_to_review(monkeypatch) -> None:
    """S1758: no stop; the plan keeps the Planner's cohort and the card names the condition."""

    from easyicu.research_agent.agents import family_spec_planner
    from easyicu.webserver.pi_copilot.plan_decisions import unselected_question_conditions

    monkeypatch.setattr(family_spec_planner, "_with_host_stated_population",
                        lambda spec, missing, *, context: (spec, ()))
    context = _aki_context()
    payload = json.dumps(_spec_payload(_request(context), population=_OLDER))

    llm, result = _run(context, [payload] * 4)

    assert len(llm.calls) == 3 and _feedback(llm) == [True, True]
    assert all(item[0] != "aki" for item in _inclusion(result))
    assert not result.facts.skeleton.cohort.host_stated_population
    assert result.facts.prompt_metrics["host_stated_population_unplaced"] == [
        {"quote": "AKI", "concepts": ["aki"], "reason": "family_spec_population_full"},
    ]
    study = {"question": context.research_question}
    plan = result.output.model_dump(mode="json")
    assert unselected_question_conditions(study, plan) == ["AKI"]


def test_a_condition_the_planner_only_quotes_is_stated_by_the_host_in_its_place() -> None:
    """Quoting an offered condition as not_typed does not apply it (UX rc_150 sepsis)."""

    context = _aki_context()
    payload = json.dumps(_spec_payload(_request(context), population=_AKI_WORDS_ONLY))

    llm, result = _run(context, [payload] * 4)

    assert len(llm.calls) == 3 and _feedback(llm) == [True, True]
    assert ("aki", "==", 1.0, "max") in _inclusion(result)
    cohort = result.facts.skeleton.cohort
    assert [item.concept for item in cohort.host_stated_population] == ["aki"]
    criteria = cohort.population_spec["criteria"]
    assert [(item["id"], item["kind"]) for item in criteria] == [
        ("c1", "age_years"), ("c2", "condition_present"),
    ], "the host's statement takes the quoted criterion's place"
