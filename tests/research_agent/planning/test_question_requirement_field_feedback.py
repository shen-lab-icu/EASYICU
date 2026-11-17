"""Cross-field scientific refusals identify the field the planner must fix."""

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.family_spec_planner import parse_family_plan_spec
from easyicu.research_agent.planning.question_requirements import (
    QuestionRequirement, question_requirements_schema, question_requirements_shape,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import call_llm_with_structured_retry
from easyicu.research_agent.providers.structured_diagnostics import safe_validation_issues

from .family_spec_fixtures import _context, _request, _spec_payload


@pytest.mark.parametrize("patch,field,allowed", [
    ({"kind": "estimand", "clauses": ["Report age"]}, "clauses", "analysis"),
    ({"clauses": ["Report age", "Report age"]}, "clauses", "distinct"),
    ({"clauses": ["Report mortality"]}, "clauses", "verbatim"),
    ({"clauses": [""]}, "clauses", "verbatim"),
    ({"clauses": ["age"]}, "clauses", "complete"),
])
def test_cross_field_refusal_locates_the_field_and_allowed_shape(patch, field, allowed):
    payload = {"id": "r1", "kind": "analysis", "quote": "Report age", "concepts": ["age"],
               "coverage": "plan", **patch}
    with pytest.raises(ValidationError) as refused:
        QuestionRequirement.model_validate(payload)
    issue = refused.value.errors()[0]
    assert issue["loc"] == (field,)
    assert allowed in issue["msg"]
    assert safe_validation_issues(refused.value)[0]["location"] == [field]


def test_family_retry_names_field_and_correction_without_dropping_the_requirement():
    request = _request(_context())
    bad = _spec_payload(request, question_requirements=[{
        "id": "r1", "kind": "estimand", "quote": "adult ICU stays", "concepts": ["age"],
        "coverage": "plan", "note": None,
        "effect_measure": None, "clauses": ["adult ICU stays"], "analysis_exposures": [],
        "deliverable_kinds": [], "gap": None,
    }])
    good = json.loads(json.dumps(bad))
    good["question_requirements"][0]["clauses"] = []
    llm = ScriptedMockLLMClient([json.dumps(bad), json.dumps(good)])
    result = call_llm_with_structured_retry(
        llm, [], parser=lambda raw: parse_family_plan_spec(raw, request),
        max_retries=1, role="family_spec_planner",
    )
    assert len(llm.calls) == 2 and result.question_requirements[0].kind == "estimand"
    feedback = llm.calls[1][0][-1].content
    assert "question_requirements.0.clauses" in feedback
    assert "[]" in feedback and "kind=analysis" in feedback


def test_transport_and_word_contract_state_the_cross_field_shapes():
    schema = question_requirements_schema(["age"], effect_measures=["odds_ratio"])
    properties = schema["items"]["properties"]
    assert "analysis" in properties["clauses"]["description"]
    shape = question_requirements_shape(["age"], effect_measures=["odds_ratio"])
    assert "clauses=[]" in shape and "analysis" in shape
