"""Canonical provider construction and cross-entrypoint security regressions."""

from __future__ import annotations

from typing import Any

import pytest


_PROVIDER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
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


def test_openrouter_contract_is_consistent_across_all_three_entries(
    ra, monkeypatch
):
    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(
        monkeypatch, ra
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-secret")

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


def test_loopback_openai_never_forwards_paid_secrets_from_any_entry(
    ra, monkeypatch
):
    from easyicu.research_agent.providers import LOCAL_OPENAI_DUMMY_API_KEY

    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(
        monkeypatch, ra
    )
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
    mcp, discovery, benchmark, seen = _install_entrypoint_recorders(
        monkeypatch, ra
    )
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
