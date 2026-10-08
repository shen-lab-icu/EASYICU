"""A family-template plan applies the population its study states.

A family template built its cohort only from the study's typed age and stay
bounds ("prose criteria are not authority").  The family Planner never saw the
study's own cohort wording and its spec had no population, so a study whose
population was stated only in words analyzed every row the bounds kept, and
nothing said so.  A family that applies its plan's own cohort by a time zero is
now offered the run's cohort concepts and the study's wording.  The Planner
states the population as criteria and the predicates applying them, each
decided by time zero; the template adds them to the typed bounds, and the plan
says which population it applies.  A reviewed candidate that filtered its
cohort by a stated population binds the next pass to a filtered cohort; with
no typed bound, only the population the Planner states again can filter it,
so the host offers that route instead of refusing it, and asks again when the
answer states none.  Fixtures are generic.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from easyicu.research_agent.agents.family_spec_planner import (
    family_spec_response_shape,
    family_spec_structured_output_request,
    family_spec_user_prompt,
)
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.family_spec.contract import (
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    LANDMARK_SURVIVAL_FAMILY_ID,
    SOURCE_FEASIBILITY_FAMILY_ID,
    FamilySpecError,
    population_required,
    spec_from_mapping,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.landmark_categorical_template import (
    _cohort_intent,
    stated_population_sentence,
)
from easyicu.research_agent.planning.family_spec.phenotyping_template import (
    _cohort_intent as _phenotype_cohort_intent,
)
from easyicu.research_agent.planning.family_spec.request import _bind_population_authority
from easyicu.research_agent.planning.progressive_compiler import (
    progressive_cohort_concept_ids,
    progressive_population_concept_ids,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import ResearchContext

from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _context,
    _descriptive_context,
    _phenotyping_context,
    _phenotyping_payload,
    _prediction_context,
    _prediction_payload,
    _run,
    _spec_payload,
)
from tests.support.survival_proposal import (
    ALLOWED as SURVIVAL_CITATIONS,
    AGE,
    SEX,
    survival_context,
    survival_spec,
)

_WORDING = {
    "label": "Older adults with a high comorbidity burden",
    "review": "Stays of older adults whose comorbidity burden is high.",
    "exclusion_statement": "Stays transferred in from another ICU.",
}


def _with_cohort(context: ResearchContext, **fields: Any) -> ResearchContext:
    """Set the study's cohort fields, as the Web study's cohort does."""

    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    cohort = {**constraints.get("cohort", {}), **fields}
    constraints["cohort"] = {key: value for key, value in cohort.items() if value is not None}
    return context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )


def _offered_request(
    context: ResearchContext,
    *,
    cohort_mode: str | None = "predicate_filtered",
    concepts: bool = True,
):
    """The request the family Planner seals, with the run's cohort concepts."""

    variables = select_progressive_variables(context)
    return build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=variables,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode=cohort_mode,
        cohort_concept_ids=progressive_cohort_concept_ids(context, variables) if concepts else (),
    )


def _predicate(
    concept: str, op: str, number: float, *, end: float = 24.0, anchor: str = "icu_admission"
) -> dict:
    return {
        "concept_id": concept,
        "anchor": anchor,
        "start_offset_hours": 0,
        "end_offset_hours": end,
        "aggregation": "first",
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


_OLDER = {"criterion": "older adults", "concept_ids": ["age"]}
_BURDEN = {"criterion": "a high comorbidity burden", "concept_ids": ["comorbidity_index"]}
_POPULATION = {
    "criteria": [_OLDER, _BURDEN],
    "inclusion": [_predicate("age", ">=", 65), _predicate("comorbidity_index", ">=", 3)],
    "exclusion": [],
}


def _spec(request, population: dict | None):
    return spec_from_mapping(_spec_payload(request, population=population))


def test_a_family_applying_its_own_cohort_is_offered_the_population_authority() -> None:
    context = _with_cohort(_context(), **_WORDING)
    request = _offered_request(context, cohort_mode=None)
    variables = select_progressive_variables(context)

    # The request offers the roster without its row identifiers.
    assert request.population_concepts == list(
        progressive_population_concept_ids(context, variables)
    )
    assert {"age", "comorbidity_index"} <= set(request.population_concepts)
    assert request.study_cohort_wording == _WORDING
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    authority = json.loads(prompt.split("population you write):\n", 1)[1].split("\n\n", 1)[0])
    assert authority == {
        "time_zero_hours_after_icu_admission": 24.0,
        "already_applied": {"age_min": 18.0},
        "study_cohort_wording": _WORDING,
        "population_concepts": request.population_concepts,
        "population_required": False,
    }


def test_a_cohort_the_caller_binds_or_a_run_without_concepts_is_offered_nothing() -> None:
    context = _with_cohort(_context(), **_WORDING)
    bound = _offered_request(context, cohort_mode="all_input_rows")
    unoffered = build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
    )

    for request in (bound, unoffered):
        assert request.population_concepts == [] and request.study_cohort_wording == {}
        # Neither field enters the digest, so these requests keep their identity.
        dumped = request.model_dump(mode="json")
        assert "population_concepts" not in dumped and "study_cohort_wording" not in dumped
        assert "Population authority" not in family_spec_user_prompt(
            request, variable_descriptions={}
        )
        assert '"population"' not in family_spec_response_shape(request)
        schema = json.loads(family_spec_structured_output_request(request).schema_json)
        assert "population" not in schema["properties"]


@pytest.mark.parametrize(
    "update",
    [
        # A sealed suite's cohort is its study design's.
        {"family_id": FIXED_WINDOW_TRAJECTORY_FAMILY_ID},
        {"family_id": LANDMARK_SURVIVAL_FAMILY_ID, "sealed_suite": object()},
        # Nothing is decided by a time zero.
        {"family_id": SOURCE_FEASIBILITY_FAMILY_ID},
        {"landmark_hours": None, "observation_window_hours": None},
    ],
)
def test_a_family_that_does_not_apply_its_own_cohort_by_time_zero_is_offered_nothing(
    update: dict,
) -> None:
    context = _with_cohort(_context(), **_WORDING)
    request = _offered_request(context, cohort_mode=None).model_copy(
        update={"population_concepts": [], "study_cohort_wording": {}, **update}
    )

    bound = _bind_population_authority(
        context,
        request,
        cohort_concept_ids=["age", "comorbidity_index"],
        required_primary_cohort_selection_mode=None,
    )

    assert bound is request


def test_a_stated_population_reading_offered_concepts_by_time_zero_is_accepted() -> None:
    request = _offered_request(_context())

    validate_family_plan_spec(_spec(request, _POPULATION), request)
    # A criterion that no offered concept expresses is stated, not applied.
    unexpressed = {"criteria": [{"criterion": "after cardiac surgery", "concept_ids": []}],
                   "inclusion": [], "exclusion": []}
    validate_family_plan_spec(_spec(request, unexpressed), request)


@pytest.mark.parametrize(
    ("population", "reason", "path"),
    [
        (
            {"criteria": [{"criterion": "frail adults", "concept_ids": ["frailty_index"]}],
             "inclusion": [], "exclusion": []},
            "family_spec_population_concept_unavailable",
            "population.criteria[0].concept_ids",
        ),
        (
            {"criteria": [_OLDER], "inclusion": [_predicate("frailty_index", ">=", 1)],
             "exclusion": []},
            "family_spec_population_concept_unavailable",
            "population.inclusion[0].concept_id",
        ),
        (
            {"criteria": [_OLDER],
             "inclusion": [_predicate("age", ">=", 65), _predicate("sex", "==", 1)],
             "exclusion": []},
            "family_spec_population_predicate_unstated",
            "population.inclusion[1]",
        ),
        (
            {"criteria": [_OLDER],
             "inclusion": [_predicate("age", ">=", 65, anchor="hospital_admission")],
             "exclusion": []},
            "family_spec_population_anchor_unavailable",
            "population.inclusion[0].anchor",
        ),
        (
            {"criteria": [_OLDER], "inclusion": [],
             "exclusion": [_predicate("age", "<", 65, end=48.0)]},
            "family_spec_population_after_time_zero",
            "population.exclusion[0].end_offset_hours",
        ),
        (
            {"criteria": [_OLDER, _BURDEN], "inclusion": [_predicate("age", ">=", 65)],
             "exclusion": []},
            "family_spec_population_criterion_unapplied",
            "population.criteria[1]",
        ),
    ],
)
def test_a_stated_population_outside_its_authority_is_refused(
    population: dict, reason: str, path: str
) -> None:
    request = _offered_request(_context())

    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(_spec(request, population), request)

    assert caught.value.reason_code == reason
    assert caught.value.path == path


def test_a_criterion_without_its_predicate_may_lose_its_concepts_only_for_its_window() -> None:
    # A concept that expresses a criterion over another window than the one
    # it states cannot apply it: the criterion then names no concepts.
    request = _offered_request(_context())
    population = {
        "criteria": [_OLDER, _BURDEN],
        "inclusion": [_predicate("age", ">=", 65)],
        "exclusion": [],
    }

    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(_spec(request, population), request)

    assert (
        "give it no concepts only when no offered concept expresses it over the "
        "window the criterion states"
    ) in str(caught.value)


def test_a_population_where_none_is_offered_is_refused() -> None:
    request = _offered_request(_context(), cohort_mode="all_input_rows")

    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(_spec(request, _POPULATION), request)

    assert caught.value.reason_code == "family_spec_population_not_applicable"


def test_the_template_adds_the_stated_population_to_the_typed_bounds() -> None:
    request = _offered_request(_context())
    spec = _spec(
        request,
        {**_POPULATION, "exclusion": [_predicate("first_stay_flag", "==", 0)],
         "criteria": [*_POPULATION["criteria"],
                      {"criterion": "first ICU stays", "concept_ids": ["first_stay_flag"]}]},
    )

    intent = _cohort_intent(request, spec.population)

    assert intent.selection_mode == "predicate_filtered"
    assert [(item.concept_id, item.op, item.value.number_value) for item in intent.inclusion] == [
        ("age", ">=", 18.0),
        ("age", ">=", 65.0),
        ("comorbidity_index", ">=", 3.0),
    ]
    assert [item.concept_id for item in intent.exclusion] == ["first_stay_flag"]
    assert [item.criterion for item in intent.population_criteria] == [
        "older adults",
        "a high comorbidity burden",
        "first ICU stays",
    ]


def test_a_stated_population_filters_a_cohort_without_typed_bounds() -> None:
    request = _offered_request(_with_cohort(_context(), age_min=None), cohort_mode=None)
    assert request.cohort_selection_mode == "all_input_rows"

    restricted = _cohort_intent(request, _spec(request, _POPULATION).population)
    stated_only = _cohort_intent(
        request,
        _spec(
            request,
            {"criteria": [{"criterion": "after cardiac surgery", "concept_ids": []}],
             "inclusion": [], "exclusion": []},
        ).population,
    )

    assert restricted.selection_mode == "predicate_filtered"
    assert [item.concept_id for item in restricted.inclusion] == ["age", "comorbidity_index"]
    assert stated_only.selection_mode == "all_input_rows"
    assert [item.criterion for item in stated_only.population_criteria] == [
        "after cardiac surgery"
    ]


def test_the_phenotype_cohort_adds_the_stated_population_after_its_membership() -> None:
    context = _phenotyping_context()
    request = _offered_request(context)
    assert "age" in request.population_concepts
    payload = _phenotyping_payload(
        request, features=["hr_max", "lactate_max", "map_min"], baseline=["age"],
        membership="phenotype_flag",
    )
    payload["population"] = {
        "criteria": [_OLDER],
        "inclusion": [_predicate("age", ">=", 65, end=request.cohort_time_zero_hours)],
        "exclusion": [],
    }
    spec = spec_from_mapping(payload)
    validate_family_plan_spec(spec, request)

    intent = _phenotype_cohort_intent(request, spec)

    assert [item.concept_id for item in intent.inclusion][0] == "phenotype_flag"
    assert [(item.concept_id, item.op) for item in intent.inclusion][-1] == ("age", ">=")
    assert [item.criterion for item in intent.population_criteria] == ["older adults"]


def test_the_plan_names_the_population_it_applies_and_what_it_cannot() -> None:
    request = _offered_request(_context())
    population = _spec(
        request,
        {**_POPULATION,
         "criteria": [*_POPULATION["criteria"],
                      {"criterion": "after cardiac surgery", "concept_ids": []}]},
    ).population

    english = stated_population_sentence(population, "en")
    chinese = stated_population_sentence(population, "zh")

    assert english == (
        " The plan applies the population the study states: older adults; a high "
        "comorbidity burden. Stated but not applied, as no available concept expresses it: "
        "after cardiac surgery."
    )
    assert chinese == (
        "计划施加研究陈述的人群：older adults；a high comorbidity burden。"
        "研究陈述、但没有可用概念表达而未施加的条件：after cardiac surgery。"
    )
    assert stated_population_sentence(None, "en") == ""


def test_the_strict_schema_offers_the_population_over_the_offered_concepts() -> None:
    request = _offered_request(_context())
    schema = json.loads(family_spec_structured_output_request(request).schema_json)

    assert "population" in schema["required"]
    population = schema["properties"]["population"]["anyOf"][0]["properties"]
    criterion = population["criteria"]["items"]["properties"]
    predicate = population["inclusion"]["items"]["properties"]
    assert criterion["concept_ids"]["items"]["enum"] == request.population_concepts
    assert predicate["concept_id"]["enum"] == request.population_concepts
    assert predicate["anchor"]["enum"] == ["icu_admission"]
    assert '"population": null, or an object' in family_spec_response_shape(request)


def test_the_planner_states_the_population_and_is_asked_again_when_one_is_unapplied() -> None:
    context = _with_cohort(_context(), **_WORDING)
    request = _offered_request(context)
    unapplied = _spec_payload(
        request,
        population={"criteria": [_OLDER, _BURDEN], "inclusion": [_predicate("age", ">=", 65)],
                    "exclusion": []},
    )
    applied = _spec_payload(request, population=_POPULATION)

    llm, result = _run(context, [json.dumps(unapplied), json.dumps(applied)])

    first = "\n".join(message.content for message in llm.calls[0][0])
    assert "Population authority" in first and _WORDING["label"] in first
    assert "family_spec_population_criterion_unapplied" in llm.calls[1][0][-1].content
    cohort = result.output.cohort
    assert cohort is not None and cohort.selection_mode == "predicate_filtered"
    assert [(item.concept_id, item.op, item.value) for item in cohort.inclusion] == [
        ("age", ">=", 18.0),
        ("age", ">=", 65.0),
        ("comorbidity_index", ">=", 3.0),
    ]
    reviewable = list(result.output.design_selection.selected.reviewable_plan)
    assert "older adults; a high comorbidity burden" in reviewable[0]


def _descriptive_payload(request) -> dict:
    keys = [*request.required_reader_label_keys, *request.level_label_keys, "age", "sex"]
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "baseline_variables": ["age", "sex"],
        "reader_display_labels": [
            {"key": key, "value": f"Reader label for {key}"} for key in dict.fromkeys(keys)
        ],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    f"Compare this description with {key} on population, exposure definition, "
                    "time zero, and estimand without copying its design or claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Descriptive family: baseline variables from host-timed candidates.",
    }


@pytest.mark.parametrize(
    ("context_factory", "payload"),
    [
        (_descriptive_context, _descriptive_payload),
        (
            _prediction_context,
            lambda request: _prediction_payload(
                request, features=["age", "sex", "hr_max", "lactate_max", "map_min"]
            ),
        ),
    ],
    ids=["descriptive", "prediction"],
)
def test_each_family_cohort_applies_the_stated_population(context_factory, payload) -> None:
    context = context_factory()
    request = _offered_request(context, cohort_mode=None)
    end = request.cohort_time_zero_hours
    spec = payload(request)
    spec["population"] = {
        "criteria": [_OLDER],
        "inclusion": [_predicate("age", ">=", 65, end=end)],
        "exclusion": [],
    }

    _llm, result = _run(
        context, [json.dumps(spec)], required_primary_cohort_selection_mode=None
    )

    cohort = result.output.cohort
    assert cohort is not None and cohort.selection_mode == "predicate_filtered"
    assert [(item.concept_id, item.op, item.value) for item in cohort.inclusion][-1] == (
        "age",
        ">=",
        65.0,
    )
    reviewable = list(result.output.design_selection.selected.reviewable_plan)
    assert "The plan applies the population the study states: older adults." in reviewable[0]
    assert not reviewable[0].startswith("All input rows")


def test_a_proposed_survival_suite_applies_the_stated_population() -> None:
    context = survival_context()
    variables = select_progressive_variables(context)
    request = build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=variables,
        allowed_literature_citation_keys=SURVIVAL_CITATIONS,
        cohort_concept_ids=progressive_cohort_concept_ids(context, variables),
    )
    assert request.proposed_suite is not None and "age" in request.population_concepts
    spec = survival_spec(request, [AGE, SEX])
    spec["population"] = {
        "criteria": [_OLDER],
        "inclusion": [_predicate("age", ">=", 65, end=request.cohort_time_zero_hours)],
        "exclusion": [],
    }
    llm = ScriptedMockLLMClient([json.dumps(spec)])

    result = ProgressivePlannerAgent(llm).run_attempt(
        context,
        planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=SURVIVAL_CITATIONS,
        direct_comparator_literature_keys=[],
        enforce_article_contract=True,
        article_contract_context=context,
        planning_contract_context="",
        required_primary_cohort_selection_mode=None,
    )

    cohort = result.output.cohort
    assert cohort is not None and cohort.selection_mode == "predicate_filtered"
    assert [(item.concept_id, item.op, item.value) for item in cohort.inclusion][-1] == (
        "age",
        ">=",
        65.0,
    )
    reviewable = list(result.output.design_selection.selected.reviewable_plan)
    assert "The plan applies the population the study states: older adults." in reviewable[0]


# A reviewed candidate whose stated population filtered its cohort binds the
# next pass to a predicate-filtered cohort; these contexts have no typed bound.
# A prediction cohort always has one, its prediction time.
_UNBOUNDED = [
    pytest.param(lambda: _with_cohort(_context(), age_min=None), id="landmark"),
    pytest.param(_descriptive_context, id="descriptive"),
]


@pytest.mark.parametrize("context_factory", _UNBOUNDED)
def test_a_caller_bound_filter_without_typed_bounds_is_offered_the_population(
    context_factory,
) -> None:
    request = _offered_request(context_factory())

    assert request.cohort_selection_mode == "predicate_filtered"
    assert (request.age_min, request.age_max, request.minimum_icu_hours) == (None, None, None)
    assert "age" in request.population_concepts
    assert population_required(request)
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    authority = json.loads(prompt.split("population you write):\n", 1)[1].split("\n\n", 1)[0])
    assert authority["population_required"] is True
    assert authority["already_applied"] == {}


@pytest.mark.parametrize("context_factory", _UNBOUNDED)
def test_without_concepts_a_caller_bound_filter_is_refused_before_any_call(
    context_factory,
) -> None:
    with pytest.raises(FamilySpecError) as caught:
        _offered_request(context_factory(), concepts=False)

    assert caught.value.reason_code == "family_spec_cohort_predicate_unavailable"
    assert caught.value.path == "cohort"


def test_only_a_stated_predicate_can_filter_a_caller_bound_cohort() -> None:
    request = _offered_request(_descriptive_context())
    end = request.cohort_time_zero_hours
    unexpressed = {"criteria": [{"criterion": "after cardiac surgery", "concept_ids": []}],
                   "inclusion": [], "exclusion": []}
    applied = {"criteria": [_OLDER], "inclusion": [_predicate("age", ">=", 65, end=end)],
               "exclusion": []}

    for population in (None, unexpressed):
        payload = {**_descriptive_payload(request), "population": population}
        with pytest.raises(FamilySpecError) as caught:
            validate_family_plan_spec(spec_from_mapping(payload), request)
        assert caught.value.reason_code == "family_spec_population_required"
        assert caught.value.path == "population"
    validate_family_plan_spec(
        spec_from_mapping({**_descriptive_payload(request), "population": applied}), request
    )
    # A typed bound filters the cohort by itself; the population stays optional.
    bounded = _offered_request(_context())
    assert not population_required(bounded)
    validate_family_plan_spec(_spec(bounded, None), bounded)


def test_a_phenotyping_membership_flag_may_filter_a_caller_bound_cohort() -> None:
    context = _phenotyping_context()
    # The membership flag is the spec's to choose, so nothing is refused early.
    assert _offered_request(context, concepts=False).population_concepts == []
    request = _offered_request(context)
    assert population_required(request)
    features = ["hr_max", "lactate_max", "map_min"]

    member = _phenotyping_payload(
        request, features=features, baseline=["age"], membership="phenotype_flag"
    )
    validate_family_plan_spec(spec_from_mapping(member), request)
    unfiltered = _phenotyping_payload(request, features=features, baseline=["age"], membership=None)
    with pytest.raises(FamilySpecError) as caught:
        validate_family_plan_spec(spec_from_mapping(unfiltered), request)
    assert caught.value.reason_code == "family_spec_population_required"


def test_the_next_pass_states_the_population_its_reviewed_candidate_filtered_by() -> None:
    context = _with_cohort(_descriptive_context(), **_WORDING)
    request = _offered_request(context)
    stated = {**_descriptive_payload(request), "population": {
        "criteria": [_OLDER],
        "inclusion": [_predicate("age", ">=", 65, end=request.cohort_time_zero_hours)],
        "exclusion": [],
    }}

    llm, result = _run(
        context,
        [json.dumps(_descriptive_payload(request)), json.dumps(stated)],
        required_primary_cohort_selection_mode="predicate_filtered",
    )

    assert "family_spec_population_required" in llm.calls[1][0][-1].content
    cohort = result.output.cohort
    assert cohort is not None and cohort.selection_mode == "predicate_filtered"
    assert [(item.concept_id, item.op, item.value) for item in cohort.inclusion] == [
        ("age", ">=", 65.0)
    ]
