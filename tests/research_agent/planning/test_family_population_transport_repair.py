"""Replay the release3 transport failure shapes using synthetic planning metadata."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.family_spec_planner import (
    family_spec_response_shape,
    family_spec_structured_output_request, family_spec_user_prompt, parse_family_plan_spec,
)
from easyicu.research_agent.agents.progressive_planner import _MAX_STEP_PARSE_RETRIES
from easyicu.research_agent.providers.structured_retry import StructuredResponseFailure
from easyicu.research_agent.planning.family_spec.contract import FamilySpecError
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from tests.research_agent.planning.family_spec_fixtures import _request, _run
from tests.support.typed_family_population import context, payload


def test_three_recorded_failure_shapes_remain_refused():
    request = _request(context())
    good = payload(request)
    sequence = deepcopy(good)
    sequence["population"]["population_spec"] = [good["population"]["population_spec"]]
    with pytest.raises(ValidationError) as rejected:
        parse_family_plan_spec(json.dumps(sequence), request)
    assert rejected.value.errors()[0]["loc"] == ("population", "population_spec")
    invalid_id = deepcopy(good)
    invalid_id["population"]["population_spec"]["criteria"][0]["id"] = "first_icu"
    with pytest.raises(ValidationError) as rejected:
        parse_family_plan_spec(json.dumps(invalid_id), request)
    assert rejected.value.errors()[0]["loc"][-1] == "id"
    missing_bound = deepcopy(good)
    missing_bound["population"]["population_spec"]["criteria"].pop(0)
    with pytest.raises(FamilySpecError) as rejected:
        parse_family_plan_spec(json.dumps(missing_bound), request)
    assert rejected.value.reason_code == "family_spec_population_typed_bound_missing"


def test_shown_nested_template_accepts_the_same_study_without_contract_relaxation():
    ctx = context(); request = _request(ctx)
    contract = family_spec_response_shape(request)
    # Read exactly the outer object a non-schema model is shown. The model still
    # chooses its criteria and replaces the documented placeholders.
    template = next(json.loads(line) for line in contract.splitlines() if line.startswith('{"population":'))
    assert isinstance(template["population"]["population_spec"], dict)
    assert template["population"]["population_spec"]["criteria"][0]["id"] == "c1"
    response = payload(request)
    template["population"]["population_spec"]["criteria"] = response["population"]["population_spec"]["criteria"]
    response.update(template)
    parsed = parse_family_plan_spec(json.dumps(response), request)
    assert len(parsed.population.population_spec.criteria) == 2
    llm, result = _run(ctx, [json.dumps(response)])
    assert len(llm.calls) == 1
    assert any(p.concept_id == "age" and p.value == 18 for p in result.output.cohort.inclusion)
    assert list(result.output.cohort.unapplied_population_criteria) == ["first ICU within each hospitalization"]


def test_pending_bound_projection_follows_the_request_in_both_transports():
    request = _request(context()).model_copy(update={
        "age_min": 21.0, "age_max": 79.0, "minimum_icu_hours": 36.0,
    })
    expected = [
        {"kind": "age_years", "min_years": 21.0, "max_years": 79.0},
        {"kind": "icu_stay_hours", "min_hours": 36.0, "max_hours": None},
    ]
    for text in (family_spec_response_shape(request),
                 family_spec_user_prompt(request, variable_descriptions={})):
        shown = text.split("pending_typed_population_bounds: ", 1)[1].splitlines()[0]
        assert json.loads(shown) == expected
        assert '"c1", "c2", ..., "c99"' in text
        assert "it is never an array" in text
        assert "template no longer supplies predicates" in text
    strict = json.loads(family_spec_structured_output_request(request).schema_json)
    assert strict["$defs"]["AgeYears"]["properties"]["id"]["pattern"] == r"^c[1-9][0-9]?$"


def test_pending_bounds_are_not_automatically_inserted_into_a_response():
    request = _request(context())
    response = payload(request)
    response["population"]["population_spec"]["criteria"] = []
    before = deepcopy(response)
    with pytest.raises(FamilySpecError, match="typed_bound_missing"):
        parse_family_plan_spec(json.dumps(response), request)
    assert response == before


def test_upstream_selection_and_caller_bound_rows_do_not_add_pending_bounds():
    request = _request(context())
    source_only = request.model_copy(update={
        "age_min": None, "age_max": None, "minimum_icu_hours": None,
        "source_selection_recorded": True,
        "source_applied_inclusion": ["age range: 65 to *"],
    })
    caller_bound = request.model_copy(update={"caller_binds_all_input_rows": True})
    all_rows = _request(context(), cohort_mode="all_input_rows")
    for case in (source_only, caller_bound):
        text = family_spec_user_prompt(case, variable_descriptions={})
        pending = text.split("pending_typed_population_bounds: ", 1)[1].splitlines()[0]
        assert json.loads(pending) == []
    assert "pending_typed_population_bounds: " not in family_spec_user_prompt(
        all_rows, variable_descriptions={}
    )


def test_a_null_upper_bound_is_shown_as_json_null():
    request = _request(context())
    text = family_spec_user_prompt(request, variable_descriptions={})
    pending = json.loads(text.split("pending_typed_population_bounds: ", 1)[1].splitlines()[0])
    assert pending == [{"kind": "age_years", "min_years": 18.0, "max_years": None}]
    assert "At least one bound must be a number" in text
    assert "Do not write null as a string" in text


@pytest.mark.parametrize("field", ["review", "exclusion_statement"])
def test_displayed_study_wording_is_a_quote_source_but_not_applied_proof(field):
    ctx = context().model_copy(update={
        "research_question": "Describe group mortality within the study's stated population.",
    })
    constraints = json.loads(ctx.user_preferences.data_constraints)
    constraints["cohort"][field] = "age 18 or older; first ICU within each hospitalization"
    ctx.user_preferences.data_constraints = json.dumps(constraints)
    request = _request(ctx)
    response = payload(request)
    for criterion in response["population"]["population_spec"]["criteria"]:
        criterion["source"] = "study_wording"
    llm, result = _run(ctx, [json.dumps(response)])
    assert len(llm.calls) == 1
    # The first-hospital restriction still waits for an extraction receipt.
    assert list(result.output.cohort.unapplied_population_criteria) == ["first ICU within each hospitalization"]


def test_a_quote_absent_from_all_actual_study_text_remains_rejected():
    ctx = context().model_copy(update={"research_question": "Describe group mortality."})
    request = _request(ctx)
    response = payload(request)
    response["population"]["population_spec"]["criteria"][0]["source"] = "study_wording"
    with pytest.raises(StructuredResponseFailure) as rejected:
        _run(ctx, [json.dumps(response)] * (_MAX_STEP_PARSE_RETRIES + 1))
    cause = rejected.value.__cause__
    assert isinstance(cause, ProgressivePlanCompileError)
    assert cause.reason_code == "progressive_population_quote_not_verbatim"
    assert len(rejected.value.easyicu_structured_attempt_metadata) == _MAX_STEP_PARSE_RETRIES + 1


def test_a_complete_typed_population_retries_its_quote_without_losing_the_host_bounds():
    ctx = context()
    request = _request(ctx)
    good = payload(request)
    bad = deepcopy(good)
    bad["population"]["population_spec"]["criteria"][0]["quote"] = "adults older than eighteen"
    llm, result = _run(ctx, [json.dumps(bad), json.dumps(good)])
    assert len(llm.calls) == 2
    feedback = llm.calls[1][0][-1].content
    assert "progressive_population_quote_not_verbatim" in feedback
    assert "cohort.population_spec.criteria[0].quote" in feedback
    assert any(p.concept_id == "age" and p.value == 18 for p in result.output.cohort.inclusion)
    assert list(result.output.cohort.unapplied_population_criteria) == ["first ICU within each hospitalization"]
