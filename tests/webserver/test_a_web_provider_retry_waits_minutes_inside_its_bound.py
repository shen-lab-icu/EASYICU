"""A Web provider retry waits minutes, inside the reviewed two-request bound.

A transient 5xx that the local provider proxy relays lasts minutes, so the
Web Research Agent's one retry waits about three minutes (180 s ± 20%) inside
a five-minute window instead of five seconds.  The two-request bound, the
status allowlist and the frozen environment are unchanged; the schedule
joins the public transport metadata and the client's reviewed transport
policy.

Fake provider SDK and credentials; no provider is called.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.research_agent.providers.transport_retry import TransportRetrySchedule
from easyicu.webserver import provider_adapter

_SCHEDULE = TransportRetrySchedule((180.0,), jitter_fraction=0.2, window_seconds=300.0)


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provider_adapter,
        "_load_external_credentials",
        lambda *_args, **_kwargs: {
            "provider": "openai",
            "api_key": "test-private-provider-key",
            "base_url": "http://127.0.0.1:8317/v1/chat/completions",
            "model": "test-local-model",
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
            "model_env": "OPENAI_MODEL",
            "auth_header": "x-api-key",
        },
    )


def test_the_web_provider_waits_minutes_before_its_one_retry(
    monkeypatch: pytest.MonkeyPatch, credentials: None
) -> None:
    captured: dict[str, Any] = {}
    import easyicu.research_agent.providers as providers

    def fake_builder(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(providers, "build_provider_client", fake_builder)

    _client, public = provider_adapter.build_research_agent_provider_client(
        {"provider": "openai", "external": True},
    )

    assert captured["max_retries"] == 1
    assert captured["retryable_http_status_codes"] == (500, 502, 503, 504)
    assert captured["allow_environment_overrides"] is False
    assert captured["retry_schedule"] == _SCHEDULE
    assert public["transport_max_attempts"] == 2
    assert public["transport_retry_delays_seconds"] == [180.0]
    assert public["transport_retry_jitter_fraction"] == 0.2
    assert public["transport_retry_window_seconds"] == 300.0


def test_the_web_client_s_reviewed_policy_states_the_same_schedule(
    monkeypatch: pytest.MonkeyPatch, credentials: None
) -> None:
    # The reviewed client types are read from the llm module, which the web
    # app has loaded before it builds a client.
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
    )

    transport = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))
    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_kwargs: transport)
    )

    client, public = provider_adapter.build_research_agent_provider_client(
        {"provider": "openai", "external": True},
    )
    policy = provider_authorization_manifest(client)["clients"][0]["transport_policy"]

    assert policy["schema_version"] == "easyicu.provider_transport_policy/4"
    for key in (
        "transport_max_attempts",
        "transport_retry_delays_seconds",
        "transport_retry_jitter_fraction",
        "transport_retry_window_seconds",
    ):
        assert policy[key] == public[key]
