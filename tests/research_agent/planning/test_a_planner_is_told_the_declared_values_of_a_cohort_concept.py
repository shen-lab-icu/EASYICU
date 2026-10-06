"""A Planner writing cohort predicates is told each concept's declared values.

The foundation is where a Planner writes the cohort's predicates, and its data
authority describes observed values by shape only.  A run planned on a
metadata-only schema observes none, so a Planner once read "SOFA >=2" in a 0/1
diagnosis flag's description and required the flag to be at least 2: a
predicate no row meets, which nothing caught before the plan was locked.  The
contract now lists the declared value set of each allowed cohort concept that
has one, from the same authority the data cards, the compiler and the review
read (``declared_domain_for_variable``), and says to compare with those values
rather than with a number from a description.  A level observed in the cohort
is never listed.  Fixtures are generic.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    declared_cohort_concept_domains,
    foundation_shape_contract,
)
from easyicu.research_agent.planning.progressive_contract import ProgressiveCohortIntent
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import ConceptDescriptor, ResearchContext, VariableRole

from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

_SENTENCE = "Declared value sets of allowed cohort concepts"
_PRINCIPLE = "never with a number from the concept's description"


def _with(context: ResearchContext, *variables: ConceptDescriptor) -> ResearchContext:
    return context.model_copy(update={"variables": [*context.variables, *variables]})


def _declared_context() -> ResearchContext:
    return _with(
        _context(),
        # A declared ordinal scale.
        ConceptDescriptor(
            name="organ_score",
            role=VariableRole.ORDINAL_SCORE,
            dtype="int64",
            is_ordinal=True,
            ordinal_levels=[0, 1, 2, 3, 4],
        ),
        # A declared ordinal range wider than the listed length.
        ConceptDescriptor(
            name="wide_scale",
            role=VariableRole.ORDINAL_SCORE,
            dtype="int64",
            is_ordinal=True,
            valid_range=[3.0, 30.0],
        ),
        # A logical event status the concept dictionary declares.
        ConceptDescriptor(
            name="vent_ind",
            role=VariableRole.OTHER,
            dtype="float64",
            source_concept="vent_ind",
        ),
        # A count of the same concept holds another quantity.
        ConceptDescriptor(
            name="vent_ind_n",
            role=VariableRole.OTHER,
            dtype="float64",
            source_concept="vent_ind",
            unit_normalization="window_nonnull_count",
        ),
    )


def _contract(domains, **kwargs) -> str:
    return foundation_shape_contract(
        outline_sha256="a" * 64,
        host_cohort=kwargs.pop("host_cohort", None),
        cohort_concept_ids=kwargs.pop(
            "cohort_concept_ids", ("organ_score", "wide_scale", "vent_ind")
        ),
        cohort_concept_domains=domains,
        **kwargs,
    )


def test_each_declared_set_comes_from_the_domain_authority() -> None:
    context = _declared_context()
    ids = (
        "organ_score",
        "wide_scale",
        "vent_ind",
        "vent_ind_n",
        "sex_code",
        "exposure_flag",
        "not_a_column",
    )

    assert declared_cohort_concept_domains(context, ids) == {
        "organ_score": [0, 1, 2, 3, 4],
        "wide_scale": list(range(3, 31)),
        "vent_ind": [0, 1],
    }


def test_the_contract_lists_them_beside_the_allowed_ids() -> None:
    domains = declared_cohort_concept_domains(
        _declared_context(), ("organ_score", "wide_scale", "vent_ind")
    )
    contract = _contract({**domains, "unlisted_concept": [0, 1]})

    listed = (
        '{"organ_score":[0,1,2,3,4],'
        '"wide_scale":{"integers_from":3,"to":30},'
        '"vent_ind":[0,1]}'
    )
    assert f"{_SENTENCE} (each concept owner's declaration, not values observed in any row): {listed}." in contract
    assert _PRINCIPLE in contract
    # The sentence follows the ids it describes.
    assert contract.index("Allowed cohort concept ids") < contract.index(_SENTENCE)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"required_cohort_selection_mode": "all_input_rows", "required_cohort_name": "all"},
        {
            "host_cohort": ProgressiveCohortIntent(
                name="host", selection_mode="all_input_rows"
            )
        },
    ],
    ids=["no_declared_set", "all_input_rows", "caller_bound"],
)
def test_nothing_is_listed_where_no_predicate_is_written_or_none_is_declared(
    kwargs,
) -> None:
    domains = {} if not kwargs else {"organ_score": [0, 1, 2, 3, 4]}
    contract = _contract(domains, **kwargs)

    assert _SENTENCE not in contract
    assert _PRINCIPLE not in contract


def test_the_planner_states_them_in_its_foundation_prompt() -> None:
    context = _with(
        _context(),
        ConceptDescriptor(
            name="organ_score",
            role=VariableRole.ORDINAL_SCORE,
            dtype="int64",
            is_ordinal=True,
            ordinal_levels=[0, 1, 2, 3, 4],
        ),
    )
    responses = [_outline_payload(), _foundation_payload(), *_materialization_payloads()]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])

    ProgressivePlannerAgent(llm).run(context)

    prompt = "\n".join(message.content for message in llm.calls[1][0])
    assert "PROGRESSIVE PLAN-FOUNDATION AUTHORITY" in prompt
    # Only the declared scale is listed; the observed levels of the other
    # columns are never named here.
    assert f'{_SENTENCE} (each concept owner\'s declaration, not values observed in any row): {{"organ_score":[0,1,2,3,4]}}.' in prompt


def test_a_retried_foundation_is_reminded_of_them() -> None:
    context = _with(
        _context(),
        ConceptDescriptor(
            name="organ_score",
            role=VariableRole.ORDINAL_SCORE,
            dtype="int64",
            is_ordinal=True,
            ordinal_levels=[0, 1, 2, 3, 4],
        ),
    )
    responses = [
        json.dumps(_outline_payload()),
        "{}",
        json.dumps(_foundation_payload()),
        *(json.dumps(item) for item in _materialization_payloads()),
    ]
    llm = ScriptedMockLLMClient(responses)

    ProgressivePlannerAgent(llm).run(context)

    feedback = llm.calls[2][0][-1].content
    assert f'{_SENTENCE} (each concept owner\'s declaration, not values observed in any row): {{"organ_score":[0,1,2,3,4]}}.' in feedback
