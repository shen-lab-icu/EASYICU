"""Canonical provider construction and cross-entrypoint security regressions."""

from __future__ import annotations

from typing import Any

import pytest

_PROVIDER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "EASYICU_ALLOW_EXTERNAL_LLM",
)


@pytest.fixture(autouse=True)
def _clean_provider_environment(monkeypatch):
    for name in _PROVIDER_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8787/v1", True),
        ("https://LOCALHOST./v1", True),
        ("http://127.0.0.1:8787/v1", True),
        ("http://127.42.0.9/v1", True),
        ("http://[::1]:8787/v1", True),
        ("http://0.0.0.0:8787/v1", False),
        ("http://localhost.example/v1", False),
        ("https://attacker.example/collect?next=localhost", False),
        ("ftp://127.0.0.1/v1", False),
        ("http://127.0.0.1:not-a-port/v1", False),
        ("127.0.0.1:8787/v1", False),
        (None, False),
    ],
)
def test_loopback_url_classification_is_parsed_and_strict(ra, base_url, expected):
    from easyicu.research_agent.providers import is_loopback_openai_base_url

    assert is_loopback_openai_base_url(base_url) is expected


def _install_entrypoint_recorders(monkeypatch, ra):
    import easyicu.research_agent.mcp_server as mcp
    import tools.run_discovery_to_manuscript as discovery
    import tools.run_research_agent_bench as benchmark

    seen: dict[str, dict[str, Any]] = {}

    def recorder(label: str):
        class RecordingClient:
            def __init__(self, **kwargs):
                seen[label] = kwargs

        return RecordingClient

    monkeypatch.setattr(mcp, "OpenAIClient", recorder("mcp"))
    monkeypatch.setattr(discovery, "OpenAIClient", recorder("discovery"))
    monkeypatch.setattr(ra, "OpenAIClient", recorder("benchmark"))
    return mcp, discovery, benchmark, seen


def _build_all_three(mcp, discovery, benchmark, *, provider: str):
    model = "openai/gpt-oss-120b:free"
    timeout = 17.0
    mcp_client, mcp_error = mcp._build_run_llm(
        {
            "provider": provider,
            "model": model,
            "request_timeout": timeout,
        }
    )
    assert mcp_error is None
    assert mcp_client is not None
    discovery._build_data_foundation_llm(
        provider=provider,
        model=model,
        request_timeout=timeout,
    )
    benchmark._make_llm(
        provider=provider,
        model=model,
        request_timeout=timeout,
    )


def test_openrouter_contract_is_consistent_across_all_three_entries(ra, monkeypatch):
    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-secret")
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")

    _build_all_three(mcp, discovery, benchmark, provider="openrouter")

    expected_titles = {
        "mcp": "EasyICU research-agent MCP",
        "discovery": "EasyICU discovery-to-manuscript",
        "benchmark": "EasyICU research-agent benchmark",
    }
    assert set(seen) == set(expected_titles)
    for entry, kwargs in seen.items():
        assert kwargs["api_key"] == "router-secret"
        assert kwargs["base_url"] == "https://router.example/v1"
        assert kwargs["model"] == "openai/gpt-oss-120b:free"
        assert kwargs["request_timeout"] == 17.0
        assert kwargs["extra_headers"] == {
            "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
            "X-Title": expected_titles[entry],
        }


def test_loopback_openai_never_forwards_paid_secrets_from_any_entry(ra, monkeypatch):
    from easyicu.research_agent.providers import LOCAL_OPENAI_DUMMY_API_KEY

    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "paid-openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "paid-openrouter-secret")

    _build_all_three(mcp, discovery, benchmark, provider="openai")

    assert set(seen) == {"mcp", "discovery", "benchmark"}
    for kwargs in seen.values():
        assert kwargs["base_url"] == "http://127.0.0.1:8787/v1"
        assert kwargs["api_key"] == LOCAL_OPENAI_DUMMY_API_KEY
        assert kwargs["api_key"] not in {
            "paid-openai-secret",
            "paid-openrouter-secret",
        }


def test_loopback_opt_in_forwards_real_openai_key_to_trusted_proxy(ra, monkeypatch):
    # A TRUSTED authenticating loopback proxy (e.g. Codex Tools on :8787) that
    # validates the client key needs the real key. With the explicit opt-in set,
    # the factory forwards OPENAI_API_KEY instead of the dummy so the proxy does
    # not 401. This is off by default (see the security test above).
    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-proxy-key")
    monkeypatch.setenv("EASYICU_TRUST_LOOPBACK_PROXY_KEY", "1")

    _build_all_three(mcp, discovery, benchmark, provider="openai")

    assert set(seen) == {"mcp", "discovery", "benchmark"}
    for kwargs in seen.values():
        assert kwargs["base_url"] == "http://127.0.0.1:8787/v1"
        assert kwargs["api_key"] == "sk-real-proxy-key"


def test_loopback_opt_in_does_not_forward_real_key_to_per_request_override(
    ra, monkeypatch
):
    # Regression (security): the opt-in trusts ONE operator-configured proxy
    # (server-owned OPENAI_BASE_URL). An untrusted per-request ``base_url``
    # override can name ANY loopback port -- including a listener a local caller
    # controls -- so it must NEVER receive the real key even with the opt-in set.
    # Otherwise a local caller could steer the paid secret to a port it owns and
    # harvest it from the Authorization header.
    import easyicu.research_agent.mcp_server as mcp
    from easyicu.research_agent.providers import LOCAL_OPENAI_DUMMY_API_KEY

    constructed: list[dict[str, Any]] = []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-proxy-key")
    monkeypatch.setenv("EASYICU_TRUST_LOOPBACK_PROXY_KEY", "1")
    monkeypatch.setattr(
        mcp, "OpenAIClient", lambda **kwargs: constructed.append(kwargs)
    )

    client, error = mcp._build_run_llm(
        {
            "provider": "openai",
            "model": "model",
            # Attacker-chosen loopback port supplied via the per-request override.
            "base_url": "http://127.0.0.1:9999/v1",
        }
    )

    assert error is None
    assert constructed, "client should still be constructed for a loopback override"
    assert constructed[0]["base_url"] == "http://127.0.0.1:9999/v1"
    assert constructed[0]["api_key"] == LOCAL_OPENAI_DUMMY_API_KEY
    assert constructed[0]["api_key"] != "sk-real-proxy-key"


def test_loopback_opt_in_without_key_still_uses_dummy(ra, monkeypatch):
    # Opt-in set but no real key present (true no-auth vLLM): still the dummy,
    # never an empty/None credential.
    from easyicu.research_agent.providers import LOCAL_OPENAI_DUMMY_API_KEY

    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EASYICU_TRUST_LOOPBACK_PROXY_KEY", "1")

    _build_all_three(mcp, discovery, benchmark, provider="openai")

    for kwargs in seen.values():
        assert kwargs["api_key"] == LOCAL_OPENAI_DUMMY_API_KEY


@pytest.mark.parametrize(
    ("provider", "available_key", "missing_key"),
    [
        ("openai", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        ("openrouter", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    ],
)
def test_provider_keys_are_not_interchangeable_across_entrypoints(
    ra,
    monkeypatch,
    provider,
    available_key,
    missing_key,
):
    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(monkeypatch, ra)
    monkeypatch.setenv(available_key, "wrong-provider-secret")
    if provider == "openai":
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    _client, error = mcp._build_run_llm(
        {"provider": provider, "model": "model", "request_timeout": 3.0}
    )
    assert error["error_code"] == "llm_configuration_required"
    assert missing_key in error["error"]

    expected_cli_error = f"{missing_key} is required for --provider {provider}"
    with pytest.raises(SystemExit, match=expected_cli_error):
        discovery._build_data_foundation_llm(
            provider=provider,
            model="model",
            request_timeout=3.0,
        )
    with pytest.raises(SystemExit, match=expected_cli_error):
        benchmark._make_llm(
            provider=provider,
            model="model",
            request_timeout=3.0,
        )
    assert seen == {}


def test_mcp_rejects_wildcard_bind_address_as_per_call_loopback(ra, monkeypatch):
    import easyicu.research_agent.mcp_server as mcp

    constructed = []
    monkeypatch.setenv("OPENAI_API_KEY", "paid-openai-secret")
    monkeypatch.setattr(
        mcp,
        "OpenAIClient",
        lambda **kwargs: constructed.append(kwargs),
    )

    client, error = mcp._build_run_llm(
        {
            "provider": "openai",
            "model": "model",
            "base_url": "http://0.0.0.0:8787/v1",
        }
    )

    assert client is None
    assert error["error_code"] == "llm_configuration_invalid"
    assert "loopback" in error["error"]
    assert constructed == []


def test_external_provider_requires_explicit_operator_authorization(ra, monkeypatch):
    import easyicu.research_agent.mcp_server as mcp

    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-not-authorized")
    client, error = mcp._build_run_llm(
        {"provider": "openai", "model": "gpt-test", "request_timeout": 5}
    )

    assert client is None
    assert error == {
        "error": (
            "configuration_error: external LLM transport is disabled; set "
            "EASYICU_ALLOW_EXTERNAL_LLM=1 only after approving the exact "
            "provider endpoint and outbound data policy"
        ),
        "error_code": "llm_external_transport_not_authorized",
    }


def test_factory_authorization_records_exact_nonsecret_endpoint():
    from easyicu.research_agent.providers.factory import (
        build_provider_client,
        provider_authorization_manifest,
    )

    class RecordingClient:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

    client = build_provider_client(
        provider="openrouter",
        model="provider/model",
        request_timeout=10,
        title="test",
        client_cls=RecordingClient,
        environment={
            "OPENROUTER_API_KEY": "secret-never-persisted",
            "OPENROUTER_BASE_URL": "https://router.example/v1",
            "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        },
    )
    payload = provider_authorization_manifest(client)

    assert payload["schema_version"] == "easyicu.provider_authorization_manifest/1"
    assert payload["clients"] == [
        {
            "provider": "openrouter",
            "model": "provider/model",
            "base_url": "https://router.example/v1",
            "destination": "external",
            "authorization_mode": "operator_env",
            "authorization_sha256": client.__easyicu_provider_authorization__.authorization_sha256,
        }
    ]
    assert "secret-never-persisted" not in str(payload)


def test_unknown_provider_is_unmanaged_external_and_never_local_exempt():
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
        provider_transport_destination,
    )

    class CustomForwarder:
        name = "custom-forwarder"

    client = CustomForwarder()

    assert provider_transport_destination(client) == "external"
    assert provider_authorization_manifest(client) == {
        "schema_version": "easyicu.provider_authorization_manifest/1",
        "clients": [
            {
                "provider": "custom-forwarder",
                "model": "",
                "base_url": "",
                "destination": "external",
                "authorization_mode": "unmanaged",
                "authorization_sha256": "",
            }
        ],
    }
