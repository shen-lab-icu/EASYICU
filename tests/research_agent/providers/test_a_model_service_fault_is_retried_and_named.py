"""B1 (UX walk 2026-11-19): a Codex turn failure is typed, retried only when transient, and named in the receipt."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.agents.progressive_payload import progressive_outline_structured_output_request
from easyicu.research_agent.providers import codex_app_server, llm
from easyicu.research_agent.providers.codex_app_server import CodexAppServerError
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.providers.strict_json_schema import (
    StrictJsonSchemaError,
    assert_closed_json_schema,
)
from tests.support.provider_app_server import _authorized_client, _FakeRuntime

_INVALID_SCHEMA = {
    "message": json.dumps({
        "type": "error",
        "error": {"message": "Invalid schema", "type": "invalid_request_error",
                  "param": "text.format.schema", "code": "invalid_json_schema"},
        "status": 400,
    }),
    "codexErrorInfo": "other",
}
_STREAM_DROPPED = {"message": "stream disconnected", "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": None}}}


def _scripted_runtime(failures: list[dict[str, Any] | None]) -> type:
    """Each new App Server process fails its turn with the next scripted error."""

    class _Scripted(_FakeRuntime):
        def wait_for_notification(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            completed = super().wait_for_notification(*args, **kwargs)
            error = failures.pop(0) if failures else None
            if error is not None:
                completed["params"]["turn"].update(status="failed", items=[], error=error)
            return completed

    return _Scripted


def _ask(client: Any) -> tuple[str, Any]:
    return client.complete_with_usage([LLMMessage(role="user", content="Answer.")])


def test_a_dropped_model_stream_is_retried_on_a_fresh_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeRuntime.instances.clear()
    monkeypatch.setattr(codex_app_server, "CodexAppServerRuntime", _scripted_runtime([_STREAM_DROPPED]))
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    client = _authorized_client(tmp_path)

    text, _usage = _ask(client)

    assert json.loads(text) == {"answer": "ok"}
    assert len(_FakeRuntime.instances) == 2
    assert client.last_transport_retries == 1


def test_a_persistent_model_service_fault_stops_after_bounded_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeRuntime.instances.clear()
    monkeypatch.setattr(codex_app_server, "CodexAppServerRuntime", _scripted_runtime([_STREAM_DROPPED] * 5))
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)

    with pytest.raises(CodexAppServerError) as raised:
        _ask(_authorized_client(tmp_path))

    assert len(_FakeRuntime.instances) == 1 + len(codex_app_server.CODEX_TRANSIENT_BACKOFF_SECONDS)
    assert raised.value.easyicu_safe_diagnostic["error_kind"] == "responseStreamDisconnected"
    assert codex_app_server.codex_failure_is_transient(raised.value)


def test_a_rejected_request_is_not_retried_and_keeps_its_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeRuntime.instances.clear()
    monkeypatch.setattr(codex_app_server, "CodexAppServerRuntime", _scripted_runtime([_INVALID_SCHEMA]))

    with pytest.raises(CodexAppServerError) as raised:
        _ask(_authorized_client(tmp_path))

    assert len(_FakeRuntime.instances) == 1
    assert raised.value.easyicu_safe_diagnostic == {
        "owner": "easyicu.providers.codex_app_server_v1",
        "reason_code": "codex_app_server_turn_failed",
        "error_kind": "other",
        "provider_error_code": "invalid_json_schema",
        "http_status": 400,
    }
    assert not codex_app_server.codex_failure_is_transient(raised.value)


@pytest.mark.parametrize(("design_cards", "citations"), [((), ("c1",)), (None, ()), (("c1",), ("c1",))])
def test_an_outline_without_design_cards_is_a_valid_strict_schema(design_cards: Any, citations: Any) -> None:
    request = progressive_outline_structured_output_request(
        analysis_types=["descriptive_epidemiology"], variable_names=["age"], scientific_action_ids=["a1"],
        allowed_literature_citation_keys=citations, design_card_citation_keys=design_cards,
    )
    schema = json.loads(request.transport_schema_json)

    assert_closed_json_schema(schema)
    decisions = schema["$defs"]["ResearchDesignCandidate"]["properties"]["literature_design_decisions"]
    assert isinstance(decisions.get("items"), dict)


def test_the_strict_owner_refuses_an_array_without_items() -> None:
    with pytest.raises(StrictJsonSchemaError, match="must define items"):
        assert_closed_json_schema({
            "type": "object", "properties": {"a": {"type": "array", "maxItems": 0}},
            "required": ["a"], "additionalProperties": False,
        })
