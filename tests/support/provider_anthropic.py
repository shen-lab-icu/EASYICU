"""Shared synthetic fixtures for provider and resume contracts."""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import pytest


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

