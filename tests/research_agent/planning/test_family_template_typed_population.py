"""Family templates expose the shared population owner, without inventing criteria.

The desktop failure used concept-only family planning, so a first ICU within
each hospitalization was reported as inexpressible despite its typed owner.
These fixtures contain metadata and scripted responses only, never patient rows.
"""

from tests.support.typed_family_population import context, typed_population, payload
from copy import deepcopy
import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import (
    family_spec_response_shape, family_spec_structured_output_request, family_spec_user_prompt,
)
from easyicu.research_agent.planning.family_spec.contract import (
    FamilySpecError, spec_from_mapping, validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.landmark_categorical_template import (
    stated_population_sentence,
)
from easyicu.research_agent.planning.family_spec import build_descriptive_skeleton, build_phenotyping_skeleton
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from tests.research_agent.planning.family_spec_fixtures import _descriptive_context, _request, _run
from tests.research_agent.planning.family_spec_fixtures import _descriptive_payload
from tests.support.hospitalization import receipt


def test_plain_and_strict_routes_offer_the_same_host_owned_kind():
    request = _request(context())
    shape = family_spec_response_shape(request)
    schema = json.loads(family_spec_structured_output_request(request).schema_json)
    assert "first_icu_stay_per_hospitalization" in shape
    assert "each patient's first ICU" in shape
    kinds = schema["$defs"]["PopulationSpec"]["properties"]["criteria"]["items"]["anyOf"]
    assert {"$ref": "#/$defs/FirstIcuStayPerHospitalization"} in kinds
    assert "concept" not in schema["$defs"]["FirstIcuStayPerHospitalization"]["properties"]
    assert schema["$defs"]["Measurement"]["properties"]["concept"]["enum"] == request.population_concepts
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    assert "require_all_ordinals_one_through_selected_icu" in prompt
    assert "SDU/Step down" in prompt
    assert "not proof that this input is already restricted" in prompt
    assert "use definition/definition_only" in prompt
    assert "independently requested analyses still need their own coverage" in prompt


def test_family_template_preserves_the_typed_spec_and_waits_for_extraction():
    ctx = context(); request = _request(ctx)
    spec = spec_from_mapping(payload(request))
    validate_family_plan_spec(spec, request)
    intent = build_descriptive_skeleton(request, spec).foundation.foundation.cohort
    assert intent.population_spec == typed_population()["population_spec"]
    compiled = compile_population(PopulationSpec.model_validate(intent.population_spec), ctx,
                                  time_zero_hours=24)
    assert [(row.criterion.kind, row.disposition) for row in compiled.criteria] == [
        ("age_years", "applied_by_plan"),
        ("first_icu_stay_per_hospitalization", "requires_extraction"),
    ]
    assert "not applied, as no available concept" not in stated_population_sentence(spec.population, "en")
    assert "compilation and review" in stated_population_sentence(spec.population, "en")
    llm, result = _run(ctx, [json.dumps(payload(request))])
    assert len(llm.calls) == 1
    assert result.output.cohort.inclusion[0].concept_id == "age"
    assert list(result.output.cohort.unapplied_population_criteria) == [
        "first ICU within each hospitalization",
    ]


def test_new_source_receipt_applies_only_the_hospitalization_kind():
    ctx = context()
    raw = receipt()
    coordinates = raw.pop("authority_coordinates")
    proof = {**raw, **coordinates}
    assert "authority_coordinates" not in proof
    ctx.cohort.provenance["first_icu_stay_per_hospitalization_restriction"] = proof
    typed = typed_population()["population_spec"]
    compiled = compile_population(PopulationSpec.model_validate(typed), ctx, time_zero_hours=24)
    assert compiled.criteria[1].disposition == "applied_by_source"
    typed["criteria"][1]["kind"] = "first_icu_stay"
    assert compile_population(PopulationSpec.model_validate(typed), ctx, time_zero_hours=24).criteria[1].disposition == "requires_extraction"


@pytest.mark.parametrize("change", ["omit_age", "weaken_age", "unknown_concept", "legacy_mixed"])
def test_a_typed_spec_cannot_drop_host_bounds_or_bypass_concept_scope(change):
    request = _request(context()); raw = payload(request)
    population = raw["population"]
    if change == "omit_age":
        population["population_spec"]["criteria"].pop(0)
    elif change == "weaken_age":
        population["population_spec"]["criteria"][0]["min_years"] = 17
    elif change == "unknown_concept":
        population["population_spec"]["criteria"].append({
            "id": "c3", "kind": "condition_present", "source": "question",
            "role": "include", "quote": "described by group", "window": None,
            "concepts_all_of": ["unoffered_flag"],
        })
    else:
        population["criteria"] = [{"criterion": "other condition", "concept_ids": []}]
    with pytest.raises((FamilySpecError, ValueError)):
        validate_family_plan_spec(spec_from_mapping(raw), request)


def test_typing_is_an_explicit_model_choice_not_inferred_from_question():
    request = _request(context())
    old = {"criteria": [{"criterion": "first ICU within each hospitalization", "concept_ids": []}],
           "inclusion": [], "exclusion": []}
    spec = spec_from_mapping(payload(request, deepcopy(old)))
    validate_family_plan_spec(spec, request)
    assert build_descriptive_skeleton(request, spec).foundation.foundation.cohort.population_spec is None
    assert "not applied" in stated_population_sentence(spec.population, "en")


def test_a_bound_all_rows_request_does_not_offer_a_new_population():
    request = _request(context(), cohort_mode="all_input_rows")
    assert "first_icu_stay_per_hospitalization" not in family_spec_response_shape(request)
    with pytest.raises(FamilySpecError, match="population_not_applicable"):
        validate_family_plan_spec(spec_from_mapping(payload(request)), request)


def test_phenotyping_passes_the_complete_typed_population_to_the_shared_owner():
    from .family_spec_fixtures import _phenotyping_context, _phenotyping_payload

    ctx = _phenotyping_context()
    request = _request(ctx)
    response = _phenotyping_payload(request, features=["hr_max", "lactate_max", "map_min"],
                                    baseline=["age"], membership=None)
    response["population"] = typed_population()
    spec = spec_from_mapping(response)
    validate_family_plan_spec(spec, request)
    intent = build_phenotyping_skeleton(request, spec).foundation.foundation.cohort
    assert intent.population_spec == typed_population()["population_spec"]
    compiled = compile_population(PopulationSpec.model_validate(intent.population_spec), ctx, time_zero_hours=24)
    assert [item.disposition for item in compiled.criteria] == ["applied_by_plan", "requires_extraction"]


@pytest.mark.parametrize("filter_kind", ["risk_set", "membership"])
def test_typed_bridge_refuses_unrepresented_template_filters(filter_kind):
    request = _request(context())
    response = payload(request)
    if filter_kind == "risk_set":
        request = request.model_copy(update={"prediction_time_hours": 12.0})
        response["request_sha256"] = request.request_sha256
    else:
        from .family_spec_fixtures import _phenotyping_context, _phenotyping_payload

        request = _request(_phenotyping_context())
        response = _phenotyping_payload(request, features=["hr_max", "lactate_max", "map_min"],
                                        baseline=["age"], membership="phenotype_flag")
        response["population"] = typed_population()
        response["cohort_membership_column"] = "phenotype_flag"

    spec = spec_from_mapping(response)
    with pytest.raises(FamilySpecError, match="template_filter_unrepresented"):
        validate_family_plan_spec(spec, request)


@pytest.mark.parametrize("maximum", [None, 80])
def test_a_complete_population_cannot_drop_or_weaken_a_host_upper_age_bound(maximum):
    request = _request(context()).model_copy(update={"age_max": 79.0})
    response = payload(request)
    response["population"]["population_spec"]["criteria"][0]["max_years"] = maximum
    with pytest.raises(FamilySpecError, match="typed_bound_missing"):
        validate_family_plan_spec(spec_from_mapping(response), request)


@pytest.mark.parametrize("criterion", [
    {"kind": "condition_present", "concepts_all_of": ["not_offered"], "window": None},
    {"kind": "event_absent", "concept": "not_offered", "window": None},
    {"kind": "measurement", "concept": "not_offered", "summary": "max",
     "window": {"start_hours": 0, "end_hours": 12}, "op": ">", "value": 5, "unit": None},
])
def test_every_concept_bearing_kind_is_checked_at_the_public_boundary(criterion):
    request = _request(context())
    response = payload(request)
    response["population"]["population_spec"]["criteria"].append({
        "id": "c3", "source": "question", "role": "include", "quote": "described by group", **criterion,
    })
    with pytest.raises(FamilySpecError, match="concept_unavailable"):
        validate_family_plan_spec(spec_from_mapping(response), request)


def test_each_prompt_route_includes_one_typed_template_and_strict_schema_offers_it():
    request = _request(context())
    for shape in ("", family_spec_response_shape(request)):
        prompt = family_spec_user_prompt(request, variable_descriptions={}, response_shape=shape)
        assert prompt.count("Typed population response template") == 1
    schema = json.loads(family_spec_structured_output_request(request).schema_json)
    properties = schema["properties"]["population"]["anyOf"][0]["properties"]
    assert properties["population_spec"]["anyOf"] == [{"$ref": "#/$defs/PopulationSpec"}, {"type": "null"}]
