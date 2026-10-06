"""A Planner is told how the source export's selection is known.

The foundation contract described the source's selection only when the
export's contract recorded it.  The Web caller now records how the host knows
it (``data_constraints.source_selection.basis``): by the export's contract,
by a prepared package that declares itself the study's cohort (which the
launch accepts), or not at all.  When nothing records the selection, the
contract tells the Planner to apply each restriction the question or the
study's cohort wording names, never attributing one to the source's
eligibility.  When a package declares itself the study's cohort, it tells the
Planner not to restate the study's restrictions through another definition,
which would narrow the rows to an intersection nobody chose, and to apply only
what the question adds.  A context without a record (the CLI, a benchmark) is
not described.  Fixtures are generic.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    foundation_shape_contract,
    source_selection_statement,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import ResearchContext, UserPreferences

from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

_UNRECORDED = (
    "Nothing records which rows the source export selected, so no restriction "
    "is the source's eligibility"
)
_DECLARED = "The source package declares itself this study's prepared cohort"
_NOT_RESTATED = "Do not restate a restriction that wording names, through any definition"
_INTENT = (
    "The study's cohort wording states whom it intends to include, not a "
    "restriction already applied"
)
_RECORDED = "The source export records its selection"


def _with_selection(context: ResearchContext, record: object) -> ResearchContext:
    preferences = context.user_preferences or UserPreferences()
    return context.model_copy(
        update={
            "user_preferences": preferences.model_copy(
                update={"data_constraints": json.dumps({"source_selection": record})}
            )
        }
    )


def _contract(statement: str | None, **kwargs) -> str:
    return foundation_shape_contract(
        outline_sha256="a" * 64,
        host_cohort=None,
        cohort_concept_ids=("age_years",),
        source_selection=statement,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("record", "statement"),
    [
        ({"basis": "export_contract", "host_applied": []}, "none"),
        ({"basis": "unrecorded", "host_applied": []}, "unrecorded"),
        ({"basis": "package_declaration", "host_applied": []}, "declared"),
        # A record written before the basis field is read by its flag.
        ({"recorded": True, "host_applied": []}, "none"),
        ({"recorded": False, "host_applied": []}, "unrecorded"),
        # An unknown basis is never taken for a contract or a declaration.
        ({"basis": "another_basis", "host_applied": []}, "unrecorded"),
    ],
    ids=["contract", "unrecorded", "declared", "old_recorded", "old_unrecorded", "unknown"],
)
def test_the_statement_follows_how_the_selection_is_known(record, statement) -> None:
    assert source_selection_statement(_with_selection(_context(), record)) == statement


def test_a_context_without_a_record_is_not_described() -> None:
    assert source_selection_statement(_context()) is None
    contract = _contract(None)
    for sentence in (_UNRECORDED, _DECLARED, _RECORDED, _INTENT):
        assert sentence not in contract


@pytest.mark.parametrize(
    ("statement", "shown", "hidden"),
    [
        ("unrecorded", (_UNRECORDED, _INTENT), (_DECLARED, _NOT_RESTATED, _RECORDED)),
        # Under a declaration the wording is what selected the rows, so it is
        # not called an intent.
        ("declared", (_DECLARED, _NOT_RESTATED), (_UNRECORDED, _RECORDED, _INTENT)),
    ],
)
def test_the_contract_states_it_where_the_planner_states_a_population(
    statement, shown, hidden
) -> None:
    contract = _contract(statement)
    for sentence in shown:
        assert sentence in contract
    for sentence in hidden:
        assert sentence not in contract
    # A caller that binds every input row is told neither.
    bound = _contract(
        statement,
        required_cohort_selection_mode="all_input_rows",
        required_cohort_name="source_cohort",
    )
    for sentence in (*shown, *hidden):
        assert sentence not in bound


@pytest.mark.parametrize(
    ("basis", "shown"),
    [("unrecorded", _UNRECORDED), ("package_declaration", _DECLARED)],
)
def test_the_planner_states_it_in_its_foundation_prompt(basis: str, shown: str) -> None:
    context = _with_selection(_context(), {"basis": basis, "host_applied": []})
    responses = [_outline_payload(), _foundation_payload(), *_materialization_payloads()]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])

    ProgressivePlannerAgent(llm).run(context)

    prompt = "\n".join(message.content for message in llm.calls[1][0])
    assert "PROGRESSIVE PLAN-FOUNDATION AUTHORITY" in prompt
    assert shown in prompt
    for other in (_UNRECORDED, _DECLARED, _RECORDED):
        assert (other in prompt) is (other == shown)
