"""A population the question names is stated, and applied by the predicates.

A Planner that chose the cohort itself named it for the question's population
("adult ... ICU stays") and still kept every input row, although the allowed
cohort concepts expressed that population; a prompt rule alone did not change
that.  The foundation now states each restriction the question places on whom
the study includes as a typed criterion with the allowed concepts that express
it, and the host refuses a foundation whose predicates read none of a listed
criterion's concepts.  A criterion that no allowed concept expresses is stated
with none.  A plan on a sealed trajectory design that states no population says
that it clusters every input row, so the plan a researcher approves says what
the plan that handed the question over said.  Fixtures are generic.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from easyicu.research_agent.agents.progressive_payload import (
    progressive_foundation_structured_output_request,
)
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    foundation_shape_contract,
)
from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    normalize_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.planning.progressive_compiler import (
    validate_progressive_foundation,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveCohortIntent,
    ProgressivePlanCompileError,
    ProgressivePlanFoundation,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    remediation_route_for_finding,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)

from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _association_context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

_DISCLOSURE = "TRAJECTORY_DESIGN_STATES_NO_POPULATION"
_QUESTION = "Among older adults with shock, which heart-rate patterns occur in the first day?"


def _context() -> ResearchContext:
    return ResearchContext(
        research_question=_QUESTION,
        cohort=CohortDescriptor(
            cohort_name="source_cohort",
            database="synthetic",
            n_stays=0,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="age_years", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
            ConceptDescriptor(name="shock_flag", role=VariableRole.OTHER, dtype="float64"),
            ConceptDescriptor(name="heart_rate", role=VariableRole.VITAL, dtype="float64"),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="float64"),
        ],
        target_outcome="death",
    )


def _predicate(concept: str, op: str, number: float, aggregation: str = "max") -> dict:
    return {
        "concept_id": concept,
        "anchor": "icu_admission",
        "start_offset_hours": 0,
        "end_offset_hours": 24,
        "aggregation": aggregation,
        "op": op,
        "value": {
            "mode": "number",
            "string_value": None,
            "number_value": number,
            "boolean_value": None,
            "string_list": [],
            "number_list": [],
        },
    }


_OLDER = _predicate("age_years", ">=", 65, "first")
_SHOCK = _predicate("shock_flag", "==", 1)
_CRITERIA = [
    {"criterion": "older adults", "concept_ids": ["age_years"]},
    {"criterion": "with shock", "concept_ids": ["shock_flag"]},
]


def _foundation(cohort: dict[str, Any]) -> ProgressivePlanFoundation:
    return ProgressivePlanFoundation.model_validate(
        {"cohort": cohort, "display_labels": [], "robustness_intents": [], "know_how_decisions": []}
    )


def _validate(cohort: dict[str, Any]) -> None:
    validate_progressive_foundation(
        _foundation(cohort), context=_context(), analysis_type="descriptive_epidemiology"
    )


def test_a_stated_population_applied_by_its_predicates_is_accepted() -> None:
    _validate(
        {
            "name": "Older adults with shock",
            "selection_mode": "predicate_filtered",
            "inclusion": [_OLDER, _SHOCK],
            "exclusion": [],
            "population_criteria": _CRITERIA,
        }
    )


@pytest.mark.parametrize(
    ("selection", "unapplied"),
    [
        # Every input row kept although both criteria have concepts.
        ({"selection_mode": "all_input_rows", "inclusion": []}, 0),
        # Age applied, shock stated but no predicate reads it.
        ({"selection_mode": "predicate_filtered", "inclusion": [_OLDER]}, 1),
    ],
)
def test_a_stated_criterion_that_no_predicate_reads_is_refused(
    selection: dict[str, Any], unapplied: int
) -> None:
    cohort = {
        "name": "Older adults with shock",
        "exclusion": [],
        "population_criteria": _CRITERIA,
        **selection,
    }

    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate(cohort)

    assert caught.value.reason_code == "progressive_population_criterion_unapplied"
    assert caught.value.path == f"cohort.population_criteria[{unapplied}]"
    assert _CRITERIA[unapplied]["criterion"] in str(caught.value)


def test_an_exclusion_predicate_applies_a_criterion_too() -> None:
    _validate(
        {
            "name": "Stays without shock",
            "selection_mode": "predicate_filtered",
            "inclusion": [],
            "exclusion": [_SHOCK],
            "population_criteria": [{"criterion": "without shock", "concept_ids": ["shock_flag"]}],
        }
    )


def test_a_criterion_no_allowed_concept_expresses_is_stated_without_concepts() -> None:
    _validate(
        {
            "name": "Every input row",
            "selection_mode": "all_input_rows",
            "inclusion": [],
            "exclusion": [],
            "population_criteria": [{"criterion": "after cardiac surgery", "concept_ids": []}],
        }
    )


def test_a_criterion_names_only_allowed_cohort_concepts() -> None:
    cohort = {
        "name": "Older adults",
        "selection_mode": "predicate_filtered",
        "inclusion": [_OLDER],
        "exclusion": [],
        "population_criteria": [{"criterion": "older adults", "concept_ids": ["age_years", "frailty_index"]}],
    }

    with pytest.raises(ProgressivePlanCompileError) as caught:
        _validate(cohort)

    assert caught.value.reason_code == "progressive_population_concept_unavailable"
    assert caught.value.path == "cohort.population_criteria[0].concept_ids"


def test_an_intent_that_states_no_criteria_keeps_its_sealed_shape() -> None:
    intent = ProgressiveCohortIntent(name="primary", selection_mode="all_input_rows")

    assert intent.model_dump(mode="json") == {
        "name": "primary",
        "selection_mode": "all_input_rows",
        "inclusion": [],
        "exclusion": [],
    }
    with pytest.raises(ValueError):
        ProgressiveCohortIntent.model_validate(
            {
                "name": "primary",
                "selection_mode": "all_input_rows",
                "population_criteria": [
                    {"criterion": "Adults", "concept_ids": []},
                    {"criterion": "adults", "concept_ids": []},
                ],
            }
        )


def test_the_planner_is_asked_again_when_its_foundation_leaves_a_criterion_unapplied() -> None:
    criteria = [{"criterion": "adults", "concept_ids": ["age_years"]}]
    unapplied = _foundation_payload()
    unapplied["foundation"]["cohort"] = {
        "name": "Adults",
        "selection_mode": "all_input_rows",
        "inclusion": [],
        "exclusion": [],
        "population_criteria": criteria,
    }
    applied = _foundation_payload()
    applied["foundation"]["cohort"] = {
        "name": "Adults",
        "selection_mode": "predicate_filtered",
        "inclusion": [_predicate("age_years", ">=", 18, "first")],
        "exclusion": [],
        "population_criteria": criteria,
    }
    responses = [_outline_payload(), unapplied, applied, *_materialization_payloads()]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True

    plan = ProgressivePlannerAgent(llm).run(_association_context())

    assert "progressive_population_criterion_unapplied" in llm.calls[2][0][-1].content
    assert plan.cohort is not None
    assert plan.cohort.selection_mode == "predicate_filtered"
    assert [item.concept_id for item in plan.cohort.inclusion] == ["age_years"]


def _schema(**kwargs: Any) -> dict:
    request = progressive_foundation_structured_output_request(
        outline_sha256="a" * 64,
        variable_names=["age_years", "shock_flag", "heart_rate"],
        cohort_concept_ids=["age_years", "shock_flag", "heart_rate"],
        **kwargs,
    )
    return json.loads(request.schema_json)


def test_the_transport_offers_criteria_over_the_allowed_concepts_only_when_the_planner_chooses() -> None:
    free = _schema()
    criterion = free["$defs"]["ProgressivePopulationCriterion"]

    assert criterion["properties"]["concept_ids"]["items"]["enum"] == [
        "age_years",
        "shock_flag",
        "heart_rate",
    ]
    assert set(criterion["required"]) == {"criterion", "concept_ids"}
    assert criterion["additionalProperties"] is False
    cohort = free["$defs"]["ProgressiveCohortIntent"]["properties"]
    assert "maxItems" not in cohort["population_criteria"] or cohort["population_criteria"]["maxItems"] > 0

    bound = _schema(required_cohort_selection_mode="all_input_rows", required_cohort_name="source_cohort")
    assert bound["$defs"]["ProgressiveCohortIntent"]["properties"]["population_criteria"]["maxItems"] == 0


def test_the_free_contract_asks_for_the_criteria_and_a_bound_cohort_states_none() -> None:
    free = foundation_shape_contract(outline_sha256="a" * 64, host_cohort=None)
    bound = foundation_shape_contract(
        outline_sha256="a" * 64,
        host_cohort=ProgressiveCohortIntent(name="source_cohort", selection_mode="all_input_rows"),
    )

    assert "in population_criteria, in the words that state it" in free
    assert "the host refuses a foundation in which no predicate reads the concepts" in free
    assert '"criterion":"<2-160 characters>"' in free
    assert '"population_criteria":[]' in bound
    assert "the host refuses a foundation" not in bound


def _signed_plan(population: dict[str, Any] | None):
    design = normalize_trajectory_design(
        {
            "coordinate_concepts": ["sofa2_resp", "sofa2_cardio"],
            "window_end_hours": 24,
            "grid_width_hours": 4,
            **({"population": population} if population else {}),
        }
    )
    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(
            load_trajectory_design(design), protocol_content_sha256="1" * 64
        )
    )
    return authority.development_execution_only_plan(research_question=_QUESTION)


def _review(population: dict[str, Any] | None):
    context = ResearchContext(
        research_question=_QUESTION,
        cohort=CohortDescriptor(
            cohort_name="source_cohort", database="synthetic", n_stays=0,
            id_columns=["stay_id"], outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name=name, role=role, dtype="float64")
            for name, role in {
                "stay_id": VariableRole.ID,
                "age": VariableRole.DEMOGRAPHIC,
                "sofa2_resp": VariableRole.ORDINAL_SCORE,
                "sofa2_cardio": VariableRole.ORDINAL_SCORE,
                "death": VariableRole.OUTCOME,
            }.items()
        ],
        target_outcome="death",
    )
    return build_plan_scientific_review(
        context=context, plan=_signed_plan(population), literature=None
    )


def test_a_sealed_trajectory_design_without_a_population_says_it_clusters_every_row() -> None:
    review = _review(None)

    [finding] = [item for item in review.findings if item.code == _DISCLOSURE]
    assert finding.severity == "minor"
    assert "clusters every input row of the source cohort" in finding.message
    assert remediation_route_for_finding(finding) == "study_authority_change"
    assert finding.requires_user_authorization is False


def test_a_sealed_trajectory_design_with_a_population_needs_no_disclosure() -> None:
    # A sealed design's predicates name dictionary concepts.
    older = {
        "concept_id": "age",
        "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24},
        "aggregation": "first",
        "op": ">=",
        "value": 65,
    }

    review = _review({"inclusion": [older]})

    assert not [item for item in review.findings if item.code == _DISCLOSURE]
