"""A refused typed specification keeps safe schema coordinates through wrappers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from easyicu.research_agent.agents.exposure_grouping_planner import ask_exposure_groupings
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_diagnostics import (
    infer_validation_stage,
    safe_validation_issues,
)
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
    safe_structured_attempt_metadata,
)


def _grouping(scale="ordinal"):
    return {
        "id": "x1", "concept": "marker_a",
        "window": {"start_hours": 0, "end_hours": 18}, "scale": scale,
        "groups": [
            {"id": "g1", "label": "low", "rule": {
                "summary": "min", "op": "<", "value": 1.5, "unit": "units",
            }},
            {"id": "g2", "label": "high only", "rule": {
                "summary": "max", "op": ">", "value": 4.0, "unit": "units",
            }},
            {"id": "g3", "label": "remaining measured values", "rule": "otherwise"},
        ],
        "unmeasured": {"handling": "own_group", "label": "not measured"},
        "reference": "g3", "contrast": "g1",
        "quote": "low, high only, and remaining measured values", "source": "question",
    }


def _ask(responses, events=None):
    context = SimpleNamespace(
        research_question=(
            "Compare the first 18 hours of marker A as low, high only, "
            "and remaining measured values."
        ),
        cohort=SimpleNamespace(inclusion_criteria=(), exclusion_criteria=()),
    )
    client = ScriptedMockLLMClient([json.dumps({"groupings": [g]}) for g in responses])
    result = ask_exposure_groupings(
        client, context=context, sources=(),
        progress_callback=events.append if events is not None else None,
    )
    return result, client


def test_wrapped_grouping_rejections_keep_coordinates_in_terminal_receipt():
    with pytest.raises(StructuredResponseFailure) as raised:
        _ask([_grouping(), _grouping(), _grouping()])

    rows = safe_structured_attempt_metadata(raised.value.attempts)
    assert len(rows) == 3
    for row in rows:
        assert row["validation_stage"] == "schema_validation"
        assert row["validation_issues"][0]["location"] == ["groupings", 0]
        assert row["validation_issues"][0]["issue_type"] == "value_error"
    rendered = json.dumps(rows)
    for private_value in ("marker_a", "high only", "not measured", "ordinal"):
        assert private_value not in rendered


def test_nominal_recovery_preserves_the_measurement_rules_and_question():
    events = []
    result, client = _ask([_grouping(), _grouping("nominal")], events)
    spec = result.groupings.groupings[0]
    assert spec.scale == "nominal"
    assert [g.rule.summary for g in spec.groups[:2]] == ["min", "max"]
    assert [g.rule.value for g in spec.groups[:2]] == [1.5, 4.0]
    assert len(client.calls) == 2
    assert client.calls[0][0][0:2] == client.calls[1][0][0:2]
    rejected = next(event for event in events if event.phase == "rejected")
    assert rejected.validation_stage == "schema_validation"
    assert rejected.validation_issues[0]["location"] == ["groupings", 0]


class _Payload(BaseModel):
    confidence_level: float


def _schema_error():
    try:
        _Payload.model_validate({"confidence_level": "private-response-token"})
    except ValidationError as exc:
        return exc
    raise AssertionError("synthetic payload must be rejected")


def test_only_explicit_causes_supply_validation_coordinates():
    wrapper = ValueError("private-wrapper-message")
    wrapper.__cause__ = _schema_error()
    assert infer_validation_stage(wrapper) == "schema_validation"
    issues = safe_validation_issues(wrapper)
    assert issues == [{
        "location": ["confidence_level"], "issue_type": "type_error",
        "input_shape": {"kind": "string", "sentinel": "other"},
    }]
    assert "private" not in json.dumps(issues)
    wrapper.__context__ = wrapper.__cause__
    wrapper.__cause__ = None
    assert safe_validation_issues(wrapper) == []
    assert infer_validation_stage(wrapper) is None


def test_cyclic_or_deep_causes_do_not_hang_diagnostics():
    first, second = ValueError("one"), ValueError("two")
    first.__cause__, second.__cause__ = second, first
    assert safe_validation_issues(first) == []
    assert infer_validation_stage(first) is None
    chain = _schema_error()
    for _ in range(12):
        wrapper = ValueError("wrapper")
        wrapper.__cause__ = chain
        chain = wrapper
    assert safe_validation_issues(chain) == []


def test_broken_error_enumeration_does_not_hide_an_explicit_schema_cause():
    class Broken(ValueError):
        def errors(self):
            raise RuntimeError("private-helper-error")

    wrapper = Broken("private-wrapper")
    wrapper.__cause__ = _schema_error()
    assert safe_validation_issues(wrapper)[0]["location"] == ["confidence_level"]
