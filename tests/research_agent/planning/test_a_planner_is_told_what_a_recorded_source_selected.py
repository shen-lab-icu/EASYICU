"""A Planner is told what the source export's recorded selection applied.

The foundation contract says the input rows are every row of the source
cohort and that only the contracts the data authority shows are already
applied.  Shown empty contract lists, a Planner still listed the population
the study names as a criterion "defined by the supplied source-cohort
eligibility", gave it no concepts although allowed concepts expressed it, and
applied only an age bound; the plan named its cohort for that population.

When the source export records its selection, the contract now says what it
applied: nothing, or only the contracts shown (with any concept-derived
population), so no other condition, treatment or age restriction selected
the rows; the study's wording is an intent, and a criterion without concepts
is applied by nothing.  An empty contract list alone proves nothing: a
legacy preset or a prepared package may have selected rows it never
declared, so these sentences describe a recorded selection only (an
unrecorded selection and a package's declaration have their own,
``test_a_planner_is_told_how_the_source_selection_is_known``).  Nothing is
claimed when the context has no record or its concept-population record
cannot be read, or when the caller binds the cohort.  Fixtures are generic.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    foundation_shape_contract,
    source_selection_statement,
)
from easyicu.research_agent.planning.progressive_contract import ProgressiveCohortIntent
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import ResearchContext, UserPreferences

from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

_SELECTED_NO_ONE = (
    "The source export records its selection, and it applied no inclusion or "
    "exclusion contract and no concept-derived population: no condition, "
    "treatment or age restriction has selected these input rows."
)
_ONLY_THE_CONTRACTS = (
    "The source export records its selection: only the inclusion and exclusion "
    "contracts shown in the data authority"
)
_RECORDED = {"recorded": True, "host_applied": []}
_WINDOW = {"definition": "sepsis3", "window_end_hours": 24}


def _with(context: ResearchContext, **cohort) -> ResearchContext:
    return context.model_copy(update={"cohort": context.cohort.model_copy(update=cohort)})


def _with_constraints(context: ResearchContext, constraints: dict) -> ResearchContext:
    preferences = context.user_preferences or UserPreferences()
    return context.model_copy(
        update={
            "user_preferences": preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )


def test_a_recorded_selection_says_what_it_applied() -> None:
    context = _context()
    recorded = _with_constraints(context, {"source_selection": _RECORDED})

    assert source_selection_statement(recorded) == "none"
    contracted = _with(recorded, inclusion_criteria=["age >= 18"])
    assert source_selection_statement(contracted) == "contracts"
    windowed = _with_constraints(
        context, {"source_selection": _RECORDED, "concept_cohort_window": _WINDOW}
    )
    assert source_selection_statement(windowed) == "contracts"
    # The study's own wording, copied into the criteria by an older context,
    # is an intent, not a criterion the export applied.
    worded = _with_constraints(
        _with(context, inclusion_criteria=["Adults with a named syndrome"]),
        {"source_selection": _RECORDED, "cohort": {"label": "Adults with a named syndrome"}},
    )
    assert source_selection_statement(worded) == "none"
    # Empty lists alone prove nothing: a context without a record is not
    # described, and an unrecorded selection is not described as recorded.
    assert source_selection_statement(context) is None
    assert source_selection_statement(_with(context, inclusion_criteria=["age >= 18"])) is None
    for marker in ({"recorded": False, "host_applied": []}, {"recorded": "yes"}):
        assert source_selection_statement(
            _with_constraints(context, {"source_selection": marker})
        ) == "unrecorded"
    # An unreadable record claims nothing either way.
    unreadable = _with_constraints(
        context, {"source_selection": _RECORDED, "concept_cohort_window": {"definition": "x"}}
    )
    assert source_selection_statement(unreadable) is None


@pytest.mark.parametrize(
    ("kwargs", "shown"),
    [
        ({"host_cohort": None, "source_selection": "none"}, _SELECTED_NO_ONE),
        ({"host_cohort": None, "source_selection": "contracts"}, _ONLY_THE_CONTRACTS),
        ({"host_cohort": None, "source_selection": None}, None),
        (
            {
                "host_cohort": None,
                "required_cohort_selection_mode": "predicate_filtered",
                "required_cohort_name": "source_cohort",
                "source_selection": "none",
            },
            _SELECTED_NO_ONE,
        ),
        (
            {
                "host_cohort": None,
                "required_cohort_selection_mode": "all_input_rows",
                "required_cohort_name": "source_cohort",
                "source_selection": "none",
            },
            None,
        ),
        (
            {
                "host_cohort": ProgressiveCohortIntent(
                    name="source_cohort", selection_mode="all_input_rows"
                ),
                "source_selection": "contracts",
            },
            None,
        ),
    ],
    ids=["none", "contracts", "no_record", "required_filter", "all_rows", "caller_bound"],
)
def test_the_contract_says_so_only_where_the_planner_states_a_population(
    kwargs: dict, shown: str | None
) -> None:
    contract = foundation_shape_contract(
        outline_sha256="a" * 64, cohort_concept_ids=("age_years",), **kwargs
    )

    for sentence in (_SELECTED_NO_ONE, _ONLY_THE_CONTRACTS):
        assert (sentence in contract) is (sentence == shown)
    assert ("a criterion without concepts is applied by nothing" in contract) is (
        shown is not None
    )


@pytest.mark.parametrize(
    ("constraints", "update", "shown"),
    [
        ({"source_selection": _RECORDED}, {}, _SELECTED_NO_ONE),
        ({"source_selection": _RECORDED}, {"inclusion_criteria": ["age >= 18"]}, _ONLY_THE_CONTRACTS),
        (None, {}, None),
    ],
    ids=["selected_no_one", "contracted", "no_record"],
)
def test_the_planner_states_it_in_the_prompt_and_on_a_retry(
    constraints: dict | None, update: dict, shown: str | None
) -> None:
    context = _with(_context(), **update)
    if constraints is not None:
        context = _with_constraints(context, constraints)
    unapplied = _foundation_payload()
    unapplied["foundation"]["cohort"] = {
        "name": "Adults",
        "selection_mode": "all_input_rows",
        "inclusion": [],
        "exclusion": [],
        "population_criteria": [{"criterion": "adults", "concept_ids": ["age_years"]}],
    }
    responses = [_outline_payload(), unapplied, _foundation_payload(), *_materialization_payloads()]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])

    ProgressivePlannerAgent(llm).run(context)

    first = "\n".join(message.content for message in llm.calls[1][0])
    retry = llm.calls[2][0][-1].content
    assert "PROGRESSIVE PLAN-FOUNDATION AUTHORITY" in first
    assert "progressive_population_criterion_unapplied" in retry
    for prompt in (first, retry):
        for sentence in (_SELECTED_NO_ONE, _ONLY_THE_CONTRACTS):
            assert (sentence in prompt) is (sentence == shown)
