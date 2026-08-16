"""Strict Planner transport-schema ownership and representation regressions."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.plan_payload import (
    PlannerStructuredOutputSchemaError,
    decode_planner_transport_payload,
    planner_structured_output_request,
)
from easyicu.research_agent.contracts.capability_ids import CAPABILITY_FAMILIES
from easyicu.research_agent.agents.core import PlannerAgent
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext


def _walk_schema(node):
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        yield node
        for child in properties.values():
            yield from _walk_schema(child)
    definitions = node.get("$defs")
    if isinstance(definitions, dict):
        for child in definitions.values():
            yield from _walk_schema(child)
    for key in ("items", "not", "if", "then", "else"):
        child = node.get(key)
        if isinstance(child, dict):
            yield from _walk_schema(child)
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                yield from _walk_schema(child)


def test_planner_transport_schema_is_closed_compact_and_deterministic():
    request = planner_structured_output_request()
    schema = json.loads(request.schema_json)

    assert request.name == "easyicu_analysis_plan_v1"
    assert request.strict is True
    assert request.payload_bytes < 30_000
    assert len(request.authority_sha256) == 64
    for object_schema in _walk_schema(schema):
        properties = object_schema["properties"]
        assert set(object_schema["required"]) == set(properties)
        assert object_schema["additionalProperties"] is False

    # A schema property named like stripped metadata must remain a property.
    criterion = schema["$defs"]["CohortEligibilityCriterion"]
    assert criterion["properties"]["description"]["type"] == "string"
    assert "description" not in criterion["properties"]["description"]

    capability = schema["$defs"]["AnalysisStep"]["properties"][
        "scientific_capability"
    ]
    assert capability == {
        "anyOf": [
            {"type": "string", "enum": sorted(CAPABILITY_FAMILIES)},
            {"type": "null"},
        ]
    }

    # Callers receive a fresh wire mapping; mutation cannot alter authority.
    first = request.to_openai_response_format()
    first["json_schema"]["schema"]["properties"].clear()
    second = request.to_openai_response_format()
    assert second["json_schema"]["schema"]["properties"]


def test_transport_projection_only_decodes_labels_and_null_placeholders():
    source = {
        "research_question": "q",
        "display_labels": [
            {"key": "x", "value": "Exposure"},
            {"key": "y", "value": "Outcome"},
        ],
        "robustness_specs": [
            {
                "spec_id": "cc",
                "missing_override": {
                    "strategy": "complete_case",
                    "variables": ["x", "y"],
                    "audit_flags": None,
                },
                "outcome_override": {
                    "column": None,
                    "concept_id": None,
                    "target": None,
                    "event_time_column": None,
                    "time_column": None,
                    "aggregation": None,
                },
            }
        ],
        "steps": [
            {
                "literature_citation_keys": ["strobe_2007"],
                "literature_design_bindings": [
                    {"citation_key": "sterne_missing_data_2009"}
                ],
            }
        ],
    }

    decoded = decode_planner_transport_payload(source)

    assert decoded["display_labels"] == {"x": "Exposure", "y": "Outcome"}
    assert decoded["robustness_specs"][0]["missing_override"] == {
        "strategy": "complete_case",
        "variables": ["x", "y"],
    }
    assert decoded["robustness_specs"][0]["outcome_override"] is None
    assert decoded["steps"][0]["literature_citation_keys"] == [
        "strobe_2007",
        "sterne_missing_data_2009",
    ]
    assert isinstance(source["display_labels"], list), "input must not be mutated"
    assert source["steps"][0]["literature_citation_keys"] == ["strobe_2007"]


def test_transport_projection_rejects_duplicate_display_label_keys():
    with pytest.raises(PlannerStructuredOutputSchemaError, match="repeats key"):
        decode_planner_transport_payload(
            {
                "display_labels": [
                    {"key": "x", "value": "first"},
                    {"key": "x", "value": "second"},
                ]
            }
        )


def test_strict_planner_prompt_does_not_duplicate_the_wire_schema_example():
    context = ResearchContext(
        research_question="Describe the ICU cohort.",
        cohort=CohortDescriptor(
            cohort_name="strict-probe",
            database="synthetic",
            n_patients=10,
            n_stays=10,
            id_columns=["stay_id"],
        ),
        variables=[],
    )

    ordinary = PlannerAgent.request_messages(context)[1].content
    strict = PlannerAgent.request_messages(
        context, strict_transport_schema=True
    )[1].content

    assert "Required JSON shape (truncated example)" in ordinary
    assert "Required JSON shape (truncated example)" not in strict
    assert "HOST-ENFORCED STRICT JSON SCHEMA" in strict
    assert "RESEARCH CONTEXT:" in strict
    assert "schema_version='easyicu.exposure_outcome_distribution/2'" in strict
    assert "interval_method='wilson'" in strict
    assert "repeated_unit_interval_method='patient_cluster_robust_wald'" in strict
    assert "even while dependence is null before host binding" in strict
    assert len(strict.encode("utf-8")) < len(ordinary.encode("utf-8"))


def test_planner_schema_capability_and_request_survive_runtime_wrapper_chain(tmp_path):
    """Exercise the production envelope -> hard-stop -> meter composition."""

    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedger,
        ProviderHardStopLimits,
    )
    from easyicu.research_agent.providers.capabilities import (
        llm_supports_strict_json_schema,
    )
    from easyicu.research_agent.providers.cost import CostMeter, MeteredClient
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.research_agent.replication.envelope import (
        ReproEnvelope,
        ReproRecordingClient,
    )

    observed = {}

    class _StrictLeaf:
        name = "strict-leaf"
        _model = "strict-model"
        supports_strict_json_schema = True

        def complete_with_usage(
            self,
            messages,
            *,
            max_tokens=2048,
            temperature=0.2,
            structured_output=None,
        ):
            observed["structured_output"] = structured_output
            return "{}", {
                "prompt_tokens": 11,
                "completion_tokens": 1,
                "actual_model": self._model,
            }

    limits = ProviderHardStopLimits(
        max_provider_attempts_per_run=2,
        max_provider_attempts_per_batch=2,
        max_total_tokens_per_run=1_000_000,
        max_total_tokens_per_batch=1_000_000,
        max_estimated_cost_usd_per_batch=10.0,
        max_wall_clock_seconds_per_task=60.0,
        input_cost_usd_per_million_tokens=1.0,
        output_cost_usd_per_million_tokens=2.0,
    )
    ledger = ProviderHardStopLedger(
        path=tmp_path / "provider_progress.json",
        task_ids=("E1",),
        limits=limits,
        batch_id="wrapper-chain",
    )
    envelope = ReproEnvelope(run_id="wrapper-chain")
    recorded = ReproRecordingClient(
        _StrictLeaf(), role="planner", envelope=envelope
    )
    stopped = HardStopClient(
        recorded, role="planner", task=ledger.start_task("E1")
    )
    client = MeteredClient(
        stopped,
        role="planner",
        meter=CostMeter(runtime_dir=tmp_path / "cost-runtime"),
    )
    request = planner_structured_output_request()

    assert llm_supports_strict_json_schema(client) is True
    response, usage = client.complete_with_usage(
        [LLMMessage(role="user", content="return an object")],
        max_tokens=8,
        temperature=0.0,
        structured_output=request,
    )

    assert response == "{}"
    assert usage["total_tokens"] == 12
    assert observed["structured_output"] is request
    assert len(envelope.calls) == 1
