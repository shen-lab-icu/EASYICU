"""Persist HTTP status without provider bodies, URLs, headers, or messages."""

import json
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_pipeline_runs


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
