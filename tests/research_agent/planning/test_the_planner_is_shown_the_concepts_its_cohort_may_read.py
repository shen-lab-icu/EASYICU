"""A Planner that chooses its cohort is shown the concepts the cohort may read.

A Provider without strict JSON schema never receives the structured-output
schema that enumerates the allowed cohort concepts. Its Planner was told to
"copy an allowed cohort concept id" from a list it could not see, and the
template it copied showed empty inclusion, exclusion and population lists: it
named the cohort for the study's population and kept every input row.  The
text contract now lists the allowed cohort concepts, shows a population
criterion item, and says all_input_rows fits only when no stated restriction
is left to apply.  A cohort the caller binds, or a mode that keeps every input
row, states no population and is shown no concepts.  Fixtures are generic.
"""

from __future__ import annotations

import json
import re

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    foundation_shape_contract,
)
from easyicu.research_agent.planning.progressive_compiler import (
    progressive_cohort_concept_ids,
)
from easyicu.research_agent.planning.progressive_contract import ProgressiveCohortIntent
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

_CONCEPTS = ("age_years", "shock_flag", "heart_rate")
_LISTED = (
    "Allowed cohort concept ids (a predicate's concept_id and a criterion's "
    'concept_ids copy one of these exactly): ["age_years", "shock_flag", "heart_rate"].'
)
_CRITERION_ITEM = (
    '"population_criteria":[{"criterion":"<2-160 characters>",'
    '"concept_ids":["<copy an allowed cohort concept id>"]}]'
)
_LISTING = re.compile(
    r"Allowed cohort concept ids \(a predicate's concept_id and a criterion's "
    r"concept_ids copy one of these exactly\): (\[[^\]]*\])\."
)


def _contract(**kwargs) -> str:
    return foundation_shape_contract(
        outline_sha256="a" * 64, cohort_concept_ids=_CONCEPTS, **kwargs
    )


def test_a_planner_choosing_its_cohort_is_shown_the_allowed_concepts() -> None:
    contract = _contract(host_cohort=None)

    assert _LISTED in contract
    assert _CRITERION_ITEM in contract
    assert '"inclusion":[],"exclusion":[]' in contract
    assert (
        "all_input_rows keeps both lists empty, and fits only when no "
        "restriction the study states is left for these predicates to apply"
    ) in contract


def test_a_required_filter_is_shown_the_allowed_concepts_and_a_criterion() -> None:
    contract = _contract(
        host_cohort=None,
        required_cohort_selection_mode="predicate_filtered",
        required_cohort_name="source_cohort",
    )

    assert _LISTED in contract
    assert _CRITERION_ITEM in contract
    assert '"selection_mode":"predicate_filtered"' in contract


def test_a_mode_that_keeps_every_row_states_no_population() -> None:
    contract = _contract(
        host_cohort=None,
        required_cohort_selection_mode="all_input_rows",
        required_cohort_name="source_cohort",
    )

    # The text mirrors the schema: the bound mode exactly, and no items.
    assert '"selection_mode":"all_input_rows"' in contract
    assert '"selection_mode":"<all_input_rows|predicate_filtered>"' not in contract
    assert '"inclusion":[],"exclusion":[],"population_criteria":[]' in contract
    assert "Allowed cohort concept ids" not in contract
    assert "population_criteria must be a JSON array" not in contract


def test_a_cohort_the_caller_binds_is_shown_no_concepts() -> None:
    contract = _contract(
        host_cohort=ProgressiveCohortIntent(name="source_cohort", selection_mode="all_input_rows"),
    )

    assert '"population_criteria":[]' in contract
    assert "Allowed cohort concept ids" not in contract


def test_an_empty_roster_lists_nothing() -> None:
    contract = foundation_shape_contract(outline_sha256="a" * 64, host_cohort=None)

    assert "Allowed cohort concept ids" not in contract
    assert _CRITERION_ITEM in contract


def test_the_planner_without_strict_schema_sends_the_list_again_on_a_retry() -> None:
    context = _context()
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

    expected = list(
        progressive_cohort_concept_ids(context, [item.name for item in context.variables])
    )
    assert "age_years" in expected
    first = "\n".join(message.content for message in llm.calls[1][0])
    retry = llm.calls[2][0][-1].content
    assert "PROGRESSIVE PLAN-FOUNDATION AUTHORITY" in first
    assert "progressive_population_criterion_unapplied" in retry
    for prompt in (first, retry):
        listed = _LISTING.findall(prompt)
        assert listed, "the allowed cohort concepts are not listed"
        assert json.loads(listed[-1]) == expected
