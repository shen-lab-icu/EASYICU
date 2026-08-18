from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

# The native Anthropic adapter is an OPTIONAL provider surface: `anthropic`
# ships in the `webapp` / `agentic` extras, not in `[project.dependencies]`.
# research_agent_ci.yml deliberately installs the MINIMUM runtime stack, so a
# hard import here fails that job for a dependency it is designed not to have.
# Skipping under the minimum stack is correct; NEVER skipping silently
# everywhere is what matters -- the `anthropic adapter` job in
# research_agent_ci.yml installs `.[agentic]` and asserts the SDK imports
# before running this file, so the coverage cannot quietly evaporate.
pytest.importorskip("anthropic")


class _FakeMessages:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected Anthropic transport call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _response(
    text: str = '{"status":"ready"}',
    *,
    stop_reason: str = "end_turn",
) -> object:
    return SimpleNamespace(
        model="claude-sonnet-4-5-actual",
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[object],
) -> tuple[list[dict[str, Any]], _FakeMessages]:
    import anthropic

    constructor_calls: list[dict[str, Any]] = []
    messages = _FakeMessages(responses)

    class FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            constructor_calls.append(kwargs)
            self.messages = messages

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    return constructor_calls, messages


def _build_client(
    *,
    strict: bool = True,
    max_retries: int = 0,
):
    from easyicu.research_agent.providers.factory import build_provider_client

    return build_provider_client(
        provider="anthropic",
        model="claude-sonnet-4-5",
        request_timeout=17.0,
        title="EasyICU Anthropic test",
        environment={
            "ANTHROPIC_API_KEY": "test-private-key",
            "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        },
        max_retries=max_retries,
        supports_strict_json_schema=strict,
        allow_environment_overrides=False,
    )


def test_direct_anthropic_client_is_denied_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers.llm import AnthropicMessagesClient
    from easyicu.research_agent.providers.protocol import LLMMessage

    constructors, messages = _install_fake_sdk(monkeypatch, [_response()])
    client = AnthropicMessagesClient(
        model="claude-sonnet-4-5",
        api_key="test-private-key",
        allow_environment_overrides=False,
    )

    with pytest.raises(PermissionError, match="factory-minted"):
        client.complete([LLMMessage(role="user", content="synthetic")])

    assert constructors[0]["base_url"] == "https://api.anthropic.com"
    assert messages.calls == []


def test_factory_anthropic_client_uses_native_messages_and_output_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
    )
    from easyicu.research_agent.providers.protocol import (
        LLMMessage,
        StructuredOutputRequest,
    )

    constructor_calls, messages = _install_fake_sdk(monkeypatch, [_response()])
    client = _build_client(strict=True)
    structured = StructuredOutputRequest.from_schema(
        name="synthetic_contract",
        schema={
            "type": "object",
            "properties": {"status": {"const": "ready"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    )

    text, usage = client.complete_with_usage(
        [
            LLMMessage(role="system", content="Return JSON only."),
            LLMMessage(role="user", content="Synthetic request."),
        ],
        max_tokens=128,
        temperature=0,
        structured_output=structured,
    )

    assert text == '{"status":"ready"}'
    assert constructor_calls == [
        {
            "api_key": "test-private-key",
            "base_url": "https://api.anthropic.com",
            "timeout": 17.0,
            "max_retries": 0,
        }
    ]
    call = messages.calls[0]
    assert call["system"] == "Return JSON only."
    assert call["messages"] == [
        {"role": "user", "content": "Synthetic request."}
    ]
    assert call["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"status": {"const": "ready"}},
                "required": ["status"],
                "additionalProperties": False,
            },
        }
    }
    assert usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "actual_model": "claude-sonnet-4-5-actual",
    }
    manifest = provider_authorization_manifest(client)
    assert manifest["clients"][0]["provider"] == "anthropic"
    assert manifest["clients"][0]["transport_policy"]["transport"] == (
        "anthropic_messages"
    )
    assert "test-private-key" not in repr(manifest)


def test_anthropic_refusal_is_terminal_without_transport_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers.protocol import LLMMessage, ProviderRefusal

    _constructors, messages = _install_fake_sdk(
        monkeypatch,
        [_response("I cannot comply.", stop_reason="refusal")],
    )
    client = _build_client(max_retries=3)

    with pytest.raises(ProviderRefusal) as refusal:
        client.complete([LLMMessage(role="user", content="Synthetic request.")])

    assert refusal.value.finish_reason == "content_filter"
    assert refusal.value.transport_attempts == 1
    assert refusal.value.usage_summary["total_tokens"] == 18
    assert len(messages.calls) == 1


def test_anthropic_strict_schema_requires_explicit_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers.protocol import (
        LLMMessage,
        StructuredOutputCapabilityError,
        StructuredOutputRequest,
    )

    _constructors, messages = _install_fake_sdk(monkeypatch, [_response()])
    client = _build_client(strict=False)
    structured = StructuredOutputRequest.from_schema(
        name="synthetic_contract",
        schema={"type": "object"},
    )

    with pytest.raises(StructuredOutputCapabilityError):
        client.complete(
            [LLMMessage(role="user", content="Synthetic request.")],
            structured_output=structured,
        )

    assert messages.calls == []


def test_anthropic_transport_identity_mutation_fails_before_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers.protocol import LLMMessage

    _constructors, messages = _install_fake_sdk(monkeypatch, [_response()])
    client = _build_client()
    client._resolved_base_url = "https://attacker.example"

    with pytest.raises(PermissionError, match="factory-minted"):
        client.complete([LLMMessage(role="user", content="Synthetic request.")])

    assert messages.calls == []


def test_anthropic_client_is_exported_from_research_agent() -> None:
    from easyicu.research_agent import AnthropicMessagesClient
    from easyicu.research_agent.providers.llm import (
        AnthropicMessagesClient as CanonicalClient,
    )

    assert AnthropicMessagesClient is CanonicalClient
