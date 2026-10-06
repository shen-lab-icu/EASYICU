"""A row identifier states no population.

Five stored plans, written before the Foundation listed the cohort concepts,
kept every input row as ``predicate_filtered`` through a single predicate over
the stay's own identifier: ``patientunitstayid`` not missing, or a count of
``stay_id`` of at least one.  Every row has its identifier, so such a
predicate keeps every row or an arbitrary subset of them, and the cohort
claims a restriction that it does not apply.  The Foundation transport and
the family request now offer the cohort concepts without the row identifiers,
and the compiler refuses a predicate or a criterion over one.  The roster
that seals and re-parses a plan still holds every column.  Fixtures are
generic.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.progressive_compiler import (
    cohort_identity_columns,
    progressive_cohort_concept_ids,
    progressive_population_concept_ids,
    validate_progressive_foundation,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanFoundation,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)

from tests.research_agent.planning import family_spec_fixtures
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _planner_context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Among older adults, is an exposure associated with death?",
        cohort=CohortDescriptor(
            cohort_name="source_cohort",
            database="synthetic",
            n_stays=0,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            # An identifier column that is not the row key identifies rows too.
            ConceptDescriptor(name="patient_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="age_years", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
            ConceptDescriptor(name="exposure_flag", role=VariableRole.INTERVENTION, dtype="int64"),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="float64"),
        ],
        target_outcome="death",
    )


def _value(mode: str, number: float | None = None) -> dict[str, Any]:
    return {
        "mode": mode,
        "string_value": None,
        "number_value": number,
        "boolean_value": None,
        "string_list": [],
        "number_list": [],
    }


def _predicate(concept: str, aggregation: str, op: str, value: dict[str, Any]) -> dict:
    return {
        "concept_id": concept,
        "anchor": "icu_admission",
        "start_offset_hours": 0,
        "end_offset_hours": 24,
        "aggregation": aggregation,
        "op": op,
        "value": value,
    }


_OLDER = _predicate("age_years", "first", ">=", _value("number", 65))


def _validate(cohort: dict[str, Any]) -> None:
    validate_progressive_foundation(
        ProgressivePlanFoundation.model_validate(
            {
                "cohort": cohort,
                "display_labels": [],
                "robustness_intents": [],
                "know_how_decisions": [],
            }
        ),
        context=_context(),
        analysis_type="descriptive_epidemiology",
    )


@pytest.mark.parametrize(
    ("side", "predicate"),
    [
        # The two forms the stored plans used.
        ("inclusion", _predicate("stay_id", "first", "not_missing", _value("none"))),
        ("inclusion", _predicate("stay_id", "count", ">=", _value("number", 1))),
        # Picking rows by their id selects no population either.
        ("exclusion", _predicate("stay_id", "first", "==", _value("number", 7))),
    ],
)
def test_a_predicate_over_the_row_identifier_is_refused(
    side: str, predicate: dict[str, Any]
) -> None:
    cohort = {
        "name": "Every stay",
        "selection_mode": "predicate_filtered",
        "inclusion": [_OLDER],
        "exclusion": [],
    }
    cohort[side] = [*cohort[side], predicate]

    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate(cohort)

    index = len(cohort[side]) - 1
    assert caught.value.reason_code == "progressive_cohort_predicate_on_identity"
    assert caught.value.path == f"cohort.{side}[{index}].concept_id"
    assert "all_input_rows" in str(caught.value)


def test_a_criterion_over_the_row_identifier_names_no_allowed_concept() -> None:
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate(
            {
                "name": "ICU stays",
                "selection_mode": "all_input_rows",
                "inclusion": [],
                "exclusion": [],
                "population_criteria": [{"criterion": "ICU stays", "concept_ids": ["stay_id"]}],
            }
        )

    assert caught.value.reason_code == "progressive_population_concept_unavailable"
    assert caught.value.path == "cohort.population_criteria[0].concept_ids"


def test_a_population_over_its_concepts_still_compiles() -> None:
    _validate(
        {
            "name": "Older adults",
            "selection_mode": "predicate_filtered",
            "inclusion": [_OLDER],
            "exclusion": [],
            "population_criteria": [{"criterion": "older adults", "concept_ids": ["age_years"]}],
        }
    )


def test_the_offer_leaves_out_the_identifier_and_the_seal_keeps_it() -> None:
    context = _context()
    names = tuple(variable.name for variable in context.variables)

    offered = progressive_population_concept_ids(context, names)

    assert cohort_identity_columns(context) == {"stay_id", "patient_id"}
    assert not {"stay_id", "patient_id"} & set(offered)
    assert {"age_years", "exposure_flag", "death"} <= set(offered)
    # A stored plan over the identifier still parses under its sealed roster.
    assert "stay_id" in progressive_cohort_concept_ids(context, names)
    # The cohort's key columns count whatever role a variable of that name has.
    keyed = context.model_copy(
        update={"cohort": context.cohort.model_copy(update={"id_columns": ["age_years"]})}
    )
    assert "age_years" in cohort_identity_columns(keyed)


def _concept_enums(node: Any) -> list[list[str]]:
    found: list[list[str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"concept_id", "concept_ids"} and isinstance(value, dict):
                enum = value.get("enum") or (value.get("items") or {}).get("enum")
                if isinstance(enum, list):
                    found.append(enum)
            found.extend(_concept_enums(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_concept_enums(value))
    return found


def test_the_foundation_prompt_and_schema_offer_no_row_identifier() -> None:
    planner_context = _planner_context()
    context = planner_context.model_copy(
        update={
            "variables": [
                ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
                *planner_context.variables,
            ]
        }
    )
    responses = [_outline_payload(), _foundation_payload(), *_materialization_payloads()]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True

    ProgressivePlannerAgent(llm).run(context)

    messages, kwargs = llm.calls[1]
    prompt = "\n".join(message.content for message in messages)
    start = prompt.index("Allowed cohort concept ids")
    allowed = prompt[start : prompt.index("]", start) + 1]
    assert "age_years" in allowed and "stay_id" not in allowed
    enums = _concept_enums(json.loads(kwargs["structured_output"].schema_json))
    assert enums and all("stay_id" not in enum for enum in enums)
    assert any("age_years" in enum for enum in enums)


def test_a_retried_foundation_is_offered_no_row_identifier_either() -> None:
    planner_context = _planner_context()
    context = planner_context.model_copy(
        update={
            "variables": [
                ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
                *planner_context.variables,
            ]
        }
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
    start = feedback.index("Allowed cohort concept ids")
    allowed = feedback[start : feedback.index("]", start) + 1]
    assert "age_years" in allowed and "stay_id" not in allowed


def test_a_family_request_offers_no_row_identifier() -> None:
    context = family_spec_fixtures._context()
    variables = select_progressive_variables(context)
    request = build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=variables,
        allowed_literature_citation_keys=family_spec_fixtures.ALLOWED_CITATIONS,
        direct_comparator_literature_keys=family_spec_fixtures.DIRECT_COMPARATORS,
        comparison_literature_keys=family_spec_fixtures.DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode="predicate_filtered",
        cohort_concept_ids=progressive_cohort_concept_ids(context, variables),
    )

    assert "stay_id" in progressive_cohort_concept_ids(context, variables)
    assert request.population_concepts
    assert "stay_id" not in request.population_concepts
