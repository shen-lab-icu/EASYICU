"""Persist HTTP status without provider bodies, URLs, headers, or messages."""

import json
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_pipeline_runs


def test_outer_http_failure_takes_precedence_over_contextual_compiler_failure():
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressivePlanCompileError,
    )

    try:
        raise ProgressivePlanCompileError(
            "progressive_typed_product_specs_invalid",
            "an earlier bounded repair failed",
            step_id="baseline_context",
            step_index=1,
            path="typed_product_specs",
        )
    except ProgressivePlanCompileError:
        transport = RuntimeError("provider body must stay private")
        transport.response = SimpleNamespace(
            status_code=503,
            headers={"Authorization": "secret"},
            text="secret",
            url="https://private.example/patient",
        )
        try:
            raise transport
        except RuntimeError as failure:
            assert agent_pipeline_runs._safe_pipeline_typed_failure(failure) == {
                "owner": "easyicu.providers.http_transport_v1",
                "reason_code": "provider_http_error",
                "status_code": 503,
            }
            assert agent_pipeline_runs._pipeline_failure_code(
                failure,
                budget_mode="planner_canary",
            ) == "research_pipeline_planner_provider_unavailable"


def test_prompt_budget_failure_publishes_its_own_measurements():
    """A size gate that hides its numbers cannot be acted on.

    ``progressive_prompt_budget_exceeded`` aborted plan generation twice while
    the Web diagnostic carried only "the host compiler rejected the bounded
    Planner repairs": the request size and the envelope it crossed were already
    computed at the check and then thrown away, so the only way to learn them
    was to fail again. Integers under a canonical key are the one part of that
    request that is safe to publish.
    """
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressivePlanCompileError,
    )

    failure = ProgressivePlanCompileError(
        "progressive_prompt_budget_exceeded",
        "initial request uses 96000 bytes; limit=90000",
        path="planner_request",
        metrics={
            "request_bytes": 96000,
            "byte_limit": 90000,
            "message_bytes": 88000,
            "schema_bytes": 8000,
            # Anything that is not a canonical, bounded, non-negative integer
            # stays out of a projection that is allowed to reach a browser.
            "PromptText": "patient row 12345678",
            "negative": -5,
            "huge": 10 ** 30,
            "floaty": 1.5,
        },
    )

    projected = agent_pipeline_runs._safe_pipeline_typed_failure(failure)

    assert projected == {
        "owner": "easyicu.planning.progressive_compiler_v1",
        "reason_code": "progressive_prompt_budget_exceeded",
        "path": "planner_request",
        "metrics": {
            "request_bytes": 96000,
            "byte_limit": 90000,
            "message_bytes": 88000,
            "schema_bytes": 8000,
        },
    }
    assert "12345678" not in json.dumps(projected)
    assert "96000 bytes" not in json.dumps(projected)
    assert agent_pipeline_runs._pipeline_failure_code(failure) == (
        "research_pipeline_progressive_compile_failed"
    )


@pytest.mark.parametrize("status", [401, 429, 503])
def test_pipeline_failure_keeps_typed_http_status_across_exception_chain(tmp_path, status):
    secret = "sk-secret-provider-message-and-patient-fragment"
    transport = RuntimeError(secret)
    transport.response = SimpleNamespace(
        status_code=status, headers={"Authorization": secret}, text=secret,
        url="https://private.example/patient",
    )
    wrapper = RuntimeError("outer " + secret)
    wrapper.__cause__ = transport
    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path, exc=wrapper, code="research_pipeline_execution_failed",
    )
    payload = json.loads((tmp_path / relative).read_text())
    assert payload["failure_type"] == "provider_http"
    assert payload["typed_failure"] == {
        "owner": "easyicu.providers.http_transport_v1",
        "reason_code": "provider_http_error",
        "status_code": status,
    }
    rendered = json.dumps(payload)
    assert secret not in rendered
    assert "private.example" not in rendered
    assert "Authorization" not in rendered
    assert payload["secrets_recorded"] is False


def test_untyped_http_wording_does_not_invent_a_status():
    failure = RuntimeError("HTTP status 503")
    assert agent_pipeline_runs._safe_pipeline_typed_failure(failure) == {}


def test_existing_typed_owner_takes_precedence_over_transport_cause():
    failure = RuntimeError("private compiler detail")
    failure.easyicu_safe_diagnostic = {
        "owner": "easyicu.planning.progressive_compiler_v1",
        "reason_code": "progressive_step_invalid",
    }
    failure.__cause__ = RuntimeError("private provider detail")
    failure.__cause__.status_code = 503
    assert agent_pipeline_runs._safe_pipeline_typed_failure(failure) == {
        "owner": "easyicu.planning.progressive_compiler_v1",
        "reason_code": "progressive_step_invalid",
    }
