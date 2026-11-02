"""Shared synthetic fixtures for provider and resume contracts."""
from __future__ import annotations
import sys
from types import SimpleNamespace


def _mock_transport_client(
    monkeypatch,
    client_type,
    *,
    model: str,
    completions=None,
    max_retries=0,
    retryable_http_status_codes=None,
    allow_environment_overrides=True,
    supports_strict_json_schema=False,
):
    # These tests exercise the concrete adapter against an in-memory fake SDK,
    # but the adapter itself must go through its real constructor so provider
    # authority cannot be minted for an ``object.__new__`` pseudo-instance.
    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=completions or SimpleNamespace())
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **_kwargs: transport),
    )
    from easyicu.research_agent.providers.factory import build_provider_client

    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setenv("EASYICU_LLM_MAX_RETRIES", str(max_retries))
    return build_provider_client(
        provider="openai",
        model=model,
        base_url_override="http://127.0.0.1:8787/v1",
        request_timeout=1.0,
        title="EasyICU provider adapter test",
        client_cls=client_type,
        max_retries=max_retries,
        retryable_http_status_codes=retryable_http_status_codes,
        supports_strict_json_schema=supports_strict_json_schema,
        allow_environment_overrides=allow_environment_overrides,
    )

