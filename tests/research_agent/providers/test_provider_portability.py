from __future__ import annotations

from typing import Any

import pytest


class _RecordingClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).calls.append(dict(kwargs))


@pytest.fixture(autouse=True)
def _clear_recording_client() -> None:
    _RecordingClient.calls.clear()


def test_deepseek_uses_its_own_identity_on_openai_compatible_transport() -> None:
    from easyicu.research_agent.providers.factory import (
        build_provider_client,
        provider_authorization_for_configuration,
    )

    environment = {
        "DEEPSEEK_API_KEY": "test-deepseek-key",
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        # A stale local-Luna setting belongs to the OpenAI/custom profile and
        # must not change DeepSeek's official Authorization transport.
        "EASYICU_OPENAI_AUTH_HEADER": "x-api-key",
        "EASYICU_TRUST_LOOPBACK_PROXY_KEY": "1",
    }
    build_provider_client(
        provider="deepseek",
        model="deepseek-v4-flash",
        request_timeout=17,
        title="test",
        client_cls=_RecordingClient,
        environment=environment,
        supports_strict_json_schema=False,
        allow_environment_overrides=False,
    )

    assert len(_RecordingClient.calls) == 1
    call = _RecordingClient.calls[0]
    assert call["api_key"] == "test-deepseek-key"
    assert call["base_url"] == "https://api.deepseek.com"
    assert call["model"] == "deepseek-v4-flash"
    assert call["supports_strict_json_schema"] is False
    manifest = provider_authorization_for_configuration(
        provider="deepseek",
        model="deepseek-v4-flash",
        environment=environment,
    )
    assert manifest["clients"][0]["provider"] == "deepseek"
    assert manifest["clients"][0]["base_url"] == "https://api.deepseek.com"
    assert manifest["clients"][0]["destination"] == "external"


def test_custom_provider_requires_server_owned_endpoint_and_credential() -> None:
    from easyicu.research_agent.providers.factory import (
        MISSING_PROVIDER_BASE_URL,
        MISSING_PROVIDER_KEY,
        ProviderConfigurationError,
        build_provider_client,
    )

    with pytest.raises(ProviderConfigurationError) as missing_base:
        build_provider_client(
            provider="custom",
            model="custom-model",
            request_timeout=17,
            title="test",
            client_cls=_RecordingClient,
            environment={"EASYICU_LLM_API_KEY": "test-key"},
        )
    assert missing_base.value.issue == MISSING_PROVIDER_BASE_URL

    with pytest.raises(ProviderConfigurationError) as missing_key:
        build_provider_client(
            provider="custom",
            model="custom-model",
            request_timeout=17,
            title="test",
            client_cls=_RecordingClient,
            environment={
                "EASYICU_LLM_BASE_URL": "https://provider.example/v1",
                "EASYICU_ALLOW_EXTERNAL_LLM": "1",
            },
        )
    assert missing_key.value.issue == MISSING_PROVIDER_KEY


def test_custom_provider_does_not_infer_strict_json_schema_capability() -> None:
    from easyicu.research_agent.providers.factory import build_provider_client

    build_provider_client(
        provider="custom",
        model="custom-model",
        request_timeout=17,
        title="test",
        client_cls=_RecordingClient,
        environment={
            "EASYICU_LLM_API_KEY": "test-key",
            "EASYICU_LLM_BASE_URL": "https://provider.example/v1",
            "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        },
    )

    assert _RecordingClient.calls[0]["supports_strict_json_schema"] is False


def test_custom_provider_preserves_exact_operator_endpoint_and_identity() -> None:
    from easyicu.research_agent.providers.factory import (
        build_provider_client,
        provider_authorization_for_configuration,
    )

    environment = {
        "EASYICU_LLM_API_KEY": "test-custom-key",
        "CUSTOM_BASE_URL": "https://relay.example/v1",
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
    }
    build_provider_client(
        provider="custom",
        model="vendor-model",
        request_timeout=17,
        title="test",
        client_cls=_RecordingClient,
        environment=environment,
    )

    call = _RecordingClient.calls[0]
    assert call["base_url"] == "https://relay.example/v1"
    assert call["model"] == "vendor-model"
    manifest = provider_authorization_for_configuration(
        provider="custom",
        model="vendor-model",
        environment=environment,
    )
    assert manifest["clients"][0]["provider"] == "custom"
    assert manifest["clients"][0]["base_url"] == "https://relay.example/v1"


def test_benchmark_snapshot_keeps_selected_provider_coordinates(monkeypatch) -> None:
    from tools import run_research_agent_bench as benchmark

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-provider-boundary")
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setenv("EASYICU_OPENAI_AUTH_HEADER", "x-api-key")
    monkeypatch.setenv("EASYICU_TRUST_LOOPBACK_PROXY_KEY", "1")

    snapshot = benchmark._provider_environment_snapshot(
        provider="deepseek",
        provider_base_url="https://api.deepseek.com",
    )

    assert snapshot["DEEPSEEK_API_KEY"] == "test-deepseek-key"
    assert snapshot["DEEPSEEK_BASE_URL"] == "https://api.deepseek.com"
    assert "OPENAI_API_KEY" not in snapshot
    assert "EASYICU_OPENAI_AUTH_HEADER" not in snapshot
    assert "EASYICU_TRUST_LOOPBACK_PROXY_KEY" not in snapshot
    assert snapshot["EASYICU_ALLOW_EXTERNAL_LLM"] == "1"


def test_codex_account_configuration_matches_constructed_authority(monkeypatch) -> None:
    import shutil

    from easyicu.research_agent.providers.factory import (
        provider_authorization_for_configuration,
        provider_authorization_manifest,
    )
    from easyicu.research_agent.providers.llm import build_llm_client

    monkeypatch.setattr(shutil, "which", lambda command: f"/usr/bin/{command}")
    environment = {
        "PATH": "/usr/bin",
        "HOME": "/private/user",
        "CODEX_HOME": "/private/codex",
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        "OPENAI_API_KEY": "must-not-reach-account-transport",
    }
    client = build_llm_client(
        prefer="codex",
        model="gpt-5.6-luna",
        allow_mock=False,
        ladder=["codex"],
        request_timeout=37,
        environment=environment,
    ).client
    constructed = provider_authorization_manifest(client)
    configured = provider_authorization_for_configuration(
        provider="codex",
        model="gpt-5.6-luna",
        request_timeout=37,
        transport_max_attempts=1,
        environment=environment,
    )

    assert configured == constructed
    assert "OPENAI_API_KEY" not in client._subprocess_environment


def test_codex_account_environment_model_has_one_authority_identity(monkeypatch) -> None:
    import shutil

    from easyicu.research_agent.providers.factory import (
        provider_authorization_for_configuration,
        provider_authorization_manifest,
    )
    from easyicu.research_agent.providers.llm import build_llm_client

    monkeypatch.setattr(shutil, "which", lambda command: f"/usr/bin/{command}")
    environment = {
        "PATH": "/usr/bin",
        "HOME": "/private/user",
        "CODEX_HOME": "/private/codex",
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        "EASYICU_CODEX_MODEL": "account-configured-model",
    }
    client = build_llm_client(
        prefer="codex",
        allow_mock=False,
        ladder=["codex"],
        environment=environment,
    ).client

    assert provider_authorization_for_configuration(
        provider="codex",
        model="",
        request_timeout=120,
        transport_max_attempts=1,
        environment=environment,
    ) == provider_authorization_manifest(client)


def test_benchmark_snapshot_for_codex_account_contains_no_api_key(monkeypatch) -> None:
    from tools import run_research_agent_bench as benchmark

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/private/user")
    monkeypatch.setenv("CODEX_HOME", "/private/codex")
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-account-transport")

    snapshot = benchmark._provider_environment_snapshot(
        provider="codex",
        provider_base_url="cli://codex",
    )

    assert snapshot["CODEX_HOME"] == "/private/codex"
    assert snapshot["EASYICU_ALLOW_EXTERNAL_LLM"] == "1"
    assert "OPENAI_API_KEY" not in snapshot


def test_benchmark_codex_user_session_uses_app_server_identity(tmp_path) -> None:
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
    )
    from tools import run_research_agent_bench as benchmark

    binding = "a" * 64
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "CODEX_HOME": str(tmp_path / "codex"),
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        "EASYICU_CODEX_SESSION_SHA256": binding,
        "OPENAI_API_KEY": "must-not-cross-user-account-boundary",
    }

    client = benchmark._make_llm(
        provider="codex",
        model="gpt-5.6-luna",
        request_timeout=37,
        planner_strict_json_schema=True,
        provider_environment=environment,
    )
    manifest = provider_authorization_manifest(client)
    authority = manifest["clients"][0]

    assert authority["provider"] == "codex-app-server"
    assert authority["model"] == "gpt-5.6-luna"
    assert authority["base_url"] == f"app-server://stdio/session/{binding}"
    assert authority["authorization_mode"] == "account_session"
    assert authority["transport_policy"] == {
        "schema_version": "easyicu.provider_transport_policy/3",
        "transport": "codex_app_server_account",
        "request_timeout_seconds": 37.0,
        "request_hard_timeout_seconds": 37.0,
        "transport_max_attempts": 1,
        "retryable_http_status_codes": None,
        "stream_enabled": False,
        "strict_json_schema_enabled": True,
        "reasoning_effort": "low",
    }
    assert "OPENAI_API_KEY" not in client._subprocess_environment


def test_benchmark_codex_user_session_is_development_only() -> None:
    from tools import run_research_agent_bench as benchmark

    binding = "a" * 64
    common = {
        "provider": "codex",
        "model": "gpt-5.6-luna",
        "multiple_models_requested": False,
        "explicit_provider_base_url": None,
        "reasoning_effort_profile": "provider_default",
        "transport_max_attempts": 1,
        "stream_enabled": False,
    }

    assert benchmark._validated_development_codex_session_binding(
        binding,
        development_diagnostic=True,
        formal_authority_requested=False,
        **common,
    ) == binding
    with pytest.raises(SystemExit, match="development-only"):
        benchmark._validated_development_codex_session_binding(
            binding,
            development_diagnostic=False,
            formal_authority_requested=False,
            **common,
        )
    with pytest.raises(SystemExit, match="development-only"):
        benchmark._validated_development_codex_session_binding(
            binding,
            development_diagnostic=True,
            formal_authority_requested=True,
            **common,
        )


def test_benchmark_account_provider_uses_cli_default_model(monkeypatch) -> None:
    from tools import run_research_agent_bench as benchmark

    monkeypatch.setenv("EASYICU_HOSTED_DEFAULT_MODEL", "api-only-model")

    assert benchmark._default_model_for_provider("codex") == "cli-default"
    assert benchmark._default_model_for_provider("deepseek") == "api-only-model"

    monkeypatch.setenv("EASYICU_CODEX_MODEL", "account-configured-model")
    assert benchmark._default_model_for_provider("codex") == (
        "account-configured-model"
    )


def test_benchmark_native_api_uses_provider_specific_model_or_fails_closed(
    monkeypatch,
) -> None:
    from tools import run_research_agent_bench as benchmark

    monkeypatch.delenv("EASYICU_HOSTED_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("EASYICU_LLM_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    with pytest.raises(SystemExit, match="--model is required"):
        benchmark._default_model_for_provider("anthropic")

    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-configured-model")
    assert benchmark._default_model_for_provider("anthropic") == (
        "claude-configured-model"
    )


def test_benchmark_anthropic_uses_native_messages_transport(monkeypatch) -> None:
    # Anthropic is an optional webapp/agentic adapter. The dedicated CI lane
    # installs and asserts the SDK before exercising the real adapter; a
    # minimum local/core environment should report that absence as a skip,
    # not as a product regression.
    anthropic = pytest.importorskip("anthropic")

    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
    )
    from tools import run_research_agent_bench as benchmark

    constructor_calls: list[dict[str, Any]] = []

    class _FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            constructor_calls.append(dict(kwargs))
            self.messages = object()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    client = benchmark._make_llm(
        provider="anthropic",
        model="claude-sonnet-4-5",
        request_timeout=17.0,
        planner_strict_json_schema=True,
        provider_environment={
            "ANTHROPIC_API_KEY": "test-private-key",
            "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        },
    )

    assert constructor_calls == [
        {
            "api_key": "test-private-key",
            "base_url": "https://api.anthropic.com",
            "timeout": 17.0,
            "max_retries": 0,
        }
    ]
    manifest = provider_authorization_manifest(client)
    assert manifest["clients"][0]["provider"] == "anthropic"
    assert manifest["clients"][0]["transport_policy"]["transport"] == (
        "anthropic_messages"
    )
    assert manifest["clients"][0]["transport_policy"][
        "strict_json_schema_enabled"
    ] is True


def test_benchmark_anthropic_rejects_openai_reasoning_profile() -> None:
    from tools import run_research_agent_bench as benchmark

    with pytest.raises(SystemExit, match="provider_default"):
        benchmark._make_llm(
            provider="anthropic",
            model="claude-sonnet-4-5",
            request_timeout=17.0,
            reasoning_effort_profile="adaptive_v1",
            provider_environment={
                "ANTHROPIC_API_KEY": "test-private-key",
                "EASYICU_ALLOW_EXTERNAL_LLM": "1",
            },
        )
