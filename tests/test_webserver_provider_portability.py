from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any

import pytest

from easyicu.webserver import provider_adapter


SESSION_SHA256 = "b" * 64


def _codex_user_environment(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "user-home"
    codex_home = tmp_path / "user-codex-home"
    home.mkdir()
    codex_home.mkdir()
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "CODEX_HOME": str(codex_home),
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        "EASYICU_CODEX_SESSION_SHA256": SESSION_SHA256,
        "EASYICU_CODEX_MODEL": "gpt-5.6-luna",
        "OPENAI_API_KEY": "must-not-cross-account-boundary",
        "DEEPSEEK_API_KEY": "must-not-cross-account-boundary",
    }


class _ReadyCodexAppServer:
    instances: list["_ReadyCodexAppServer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def __enter__(self) -> "_ReadyCodexAppServer":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def request(self, method: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        assert method == "account/read"
        return {
            "account": {
                "type": "chatgpt",
                "email": "user@example.test",
                "planType": "plus",
            }
        }


@pytest.mark.parametrize(
    ("provider", "key_env", "base_env", "endpoint", "sdk_base", "model", "transport"),
    [
        (
            "deepseek",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/chat/completions",
            "https://api.deepseek.com",
            "deepseek-v4-flash",
            "openai_chat_completions",
        ),
        (
            "custom",
            "EASYICU_LLM_API_KEY",
            "CUSTOM_BASE_URL",
            "https://gateway.example/v1/chat/completions",
            "https://gateway.example/v1",
            "vendor-model",
            "openai_chat_completions",
        ),
        (
            "anthropic",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com",
            "claude-sonnet-4-5",
            "anthropic_messages",
        ),
    ],
)
def test_web_full_research_agent_bridge_accepts_catalogued_compatible_providers(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key_env: str,
    base_env: str,
    endpoint: str,
    sdk_base: str,
    model: str,
    transport: str,
) -> None:
    private_key = f"test-{provider}-private-key"
    monkeypatch.setattr(
        provider_adapter,
        "_load_external_credentials",
        lambda *_args, **_kwargs: {
            "provider": provider,
            "api_key": private_key,
            "base_url": endpoint,
            "model": model,
            "api_key_env": key_env,
            "base_url_env": base_env,
            "model_env": f"{provider.upper()}_MODEL",
            "auth_header": "x-api-key" if provider == "anthropic" else "authorization",
            "transport": transport,
        },
    )
    captured: dict[str, Any] = {}
    import easyicu.research_agent.providers as providers

    def fake_builder(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(providers, "build_provider_client", fake_builder)

    _client, public = provider_adapter.build_research_agent_provider_client(
        {"provider": provider, "external": True}
    )

    assert captured["provider"] == provider
    assert captured["model"] == model
    assert captured["environment"][key_env] == private_key
    assert captured["environment"][base_env] == sdk_base
    assert captured["environment"]["EASYICU_ALLOW_EXTERNAL_LLM"] == "1"
    assert public["provider"] == provider
    assert public["client_constructed"] is True
    assert private_key not in json.dumps(public)


def test_web_deepseek_readiness_uses_catalog_defaults_without_exposing_values() -> None:
    readiness = provider_adapter.provider_readiness(
        "deepseek",
        ai_enabled=True,
        environ={
            "DEEPSEEK_API_KEY": "test-private-key",
            "DEEPSEEK_MODEL": "deepseek-v4-flash",
            "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
        },
    )

    assert readiness["ready"] is True
    assert readiness["credential_source"] == "DEEPSEEK_API_KEY"
    assert readiness["model_source"] == "DEEPSEEK_MODEL"
    assert readiness["base_url_source"] == "provider_default"
    assert readiness["secrets_returned"] is False
    assert "test-private-key" not in json.dumps(readiness)


def test_web_deepseek_ignores_stale_local_proxy_auth_header() -> None:
    credentials = provider_adapter._load_external_credentials(
        "deepseek",
        environ={
            "DEEPSEEK_API_KEY": "test-private-key",
            "DEEPSEEK_MODEL": "deepseek-v4-flash",
            "EASYICU_OPENAI_AUTH_HEADER": "x-api-key",
            "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
        },
    )

    assert credentials["auth_header"] == "authorization"


def test_web_anthropic_credentials_use_native_messages_endpoint() -> None:
    credentials = provider_adapter._load_external_credentials(
        "anthropic",
        environ={
            "ANTHROPIC_API_KEY": "test-private-key",
            "ANTHROPIC_MODEL": "claude-sonnet-4-5",
            "EASYICU_OPENAI_AUTH_HEADER": "authorization",
            "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
        },
    )

    assert credentials["base_url"] == "https://api.anthropic.com/v1/messages"
    assert credentials["transport"] == "anthropic_messages"
    assert credentials["auth_header"] == "x-api-key"
    assert credentials["model"] == "claude-sonnet-4-5"


@pytest.mark.parametrize("provider", ["claude", "gemini"])
def test_web_exposes_no_claude_or_gemini_account_provider(provider: str) -> None:
    readiness = provider_adapter.provider_readiness(
        provider,
        ai_enabled=True,
        environ={"EASYICU_DISABLE_PROVIDER_ENV_FILE": "1"},
    )

    assert provider_adapter.is_cli_account_provider(provider) is False
    assert readiness["ready"] is False
    assert readiness["error"] == "research_agent_provider_unsupported"


def test_web_writes_deepseek_config_with_private_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "provider.env"
    monkeypatch.setattr(provider_adapter, "_DEFAULT_PROVIDER_ENV_FILE", target)

    result = provider_adapter.write_provider_config(
        "deepseek",
        api_key="test-private-key",
        base_url="",
        model="deepseek-v4-flash",
    )

    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert result["env_file"]["loaded_keys"] == [
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
    ]
    assert "test-private-key" not in json.dumps(result)
    assert "DEEPSEEK_API_KEY=test-private-key" in target.read_text(encoding="utf-8")


def test_web_codex_readiness_uses_account_session_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.providers import codex_app_server

    monkeypatch.setattr(
        codex_app_server,
        "CodexAppServerRuntime",
        _ReadyCodexAppServer,
    )
    environment = _codex_user_environment(tmp_path)
    readiness = provider_adapter.provider_readiness(
        "codex",
        ai_enabled=True,
        environ=environment,
    )

    assert readiness["ready"] is True
    assert readiness["authentication_mode"] == "chatgpt_account"
    assert readiness["authentication_verified"] is True
    assert readiness["session_binding_sha256"] == SESSION_SHA256
    assert readiness["model"] == "gpt-5.6-luna"
    assert readiness["model_source"] == "EASYICU_CODEX_MODEL"
    assert readiness["base_url_source"] == "codex_app_server"
    assert readiness["credential_env_candidates"] == []
    assert "must-not-cross-account-boundary" not in json.dumps(readiness)
    runtime_environment = _ReadyCodexAppServer.instances[-1].kwargs["environment"]
    assert runtime_environment["HOME"] == environment["HOME"]
    assert runtime_environment["CODEX_HOME"] == environment["CODEX_HOME"]
    assert "OPENAI_API_KEY" not in runtime_environment
    assert "DEEPSEEK_API_KEY" not in runtime_environment


def test_web_account_pipeline_environment_excludes_unrelated_api_keys(
    tmp_path: Path,
) -> None:
    environment = _codex_user_environment(tmp_path)

    selected = provider_adapter.account_provider_environment(
        "codex",
        environ=environment,
    )

    assert selected["HOME"] == environment["HOME"]
    assert selected["CODEX_HOME"] == environment["CODEX_HOME"]
    assert selected["EASYICU_CODEX_SESSION_SHA256"] == SESSION_SHA256
    assert selected["EASYICU_ALLOW_EXTERNAL_LLM"] == "1"
    assert "OPENAI_API_KEY" not in selected
    assert "DEEPSEEK_API_KEY" not in selected


def test_web_full_pipeline_accepts_only_matching_codex_user_auth_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    from easyicu.webserver.routes import agent as agent_route

    expected = {
        "HOME": "/private/user-session",
        "CODEX_HOME": "/private/user-session/.codex",
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        "EASYICU_CODEX_SESSION_SHA256": SESSION_SHA256,
    }
    monkeypatch.setattr(
        provider_adapter,
        "account_provider_environment",
        lambda _provider, *, environ: dict(environ or {}),
    )

    resolved = agent_route._provider_environment_for_agent_run(
        credential_source="codex_user_auth",
        engine="research_agent_pipeline",
        run_type="full",
        external_llm_opt_in=True,
        llm_provider="codex",
        account_environment=expected,
    )
    assert resolved == expected

    with pytest.raises(HTTPException) as wrong_provider:
        agent_route._provider_environment_for_agent_run(
            credential_source="codex_user_auth",
            engine="research_agent_pipeline",
            run_type="full",
            external_llm_opt_in=True,
            llm_provider="deepseek",
            account_environment=expected,
        )
    assert wrong_provider.value.detail == {
        "error": "codex_user_auth_provider_required"
    }


@pytest.mark.parametrize(
    ("source", "provider", "expected"),
    [
        ("codex_user_auth", "codex", "codex_user_auth"),
        ("pi_verified", "openai", "pi_verified"),
    ],
)
def test_pipeline_credential_source_is_bound_to_provider_family(
    source: str,
    provider: str,
    expected: str,
) -> None:
    from easyicu.webserver import agent_pipeline_runs

    assert agent_pipeline_runs._validated_pipeline_credential_source(
        source,
        provider={"provider": provider},
    ) == expected


def test_pipeline_credential_source_rejects_cross_family_reuse() -> None:
    from easyicu.webserver import agent_pipeline_runs

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as api_as_account:
        agent_pipeline_runs._validated_pipeline_credential_source(
            "codex_user_auth",
            provider={"provider": "deepseek"},
        )
    assert api_as_account.value.code == (
        "research_pipeline_codex_user_auth_provider_required"
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as account_as_api:
        agent_pipeline_runs._validated_pipeline_credential_source(
            "pi_verified",
            provider={"provider": "codex"},
        )
    assert account_as_api.value.code == (
        "research_pipeline_codex_user_auth_required"
    )


def test_web_codex_client_uses_only_reviewed_account_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.providers import codex_app_server
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
    )
    from easyicu.research_agent.providers.llm import CodexAppServerLLMClient

    monkeypatch.setattr(
        codex_app_server,
        "CodexAppServerRuntime",
        _ReadyCodexAppServer,
    )
    environment = _codex_user_environment(tmp_path)

    client, public = provider_adapter.build_research_agent_provider_client(
        {
            "provider": "codex",
            "external": True,
            "ai_enabled": True,
            "provider_gate_order": [],
        },
        environ=environment,
    )

    assert isinstance(client, CodexAppServerLLMClient)
    manifest = provider_authorization_manifest(client)
    assert manifest["clients"][0]["provider"] == "codex-app-server"
    assert manifest["clients"][0]["base_url"].endswith(
        f"/session/{SESSION_SHA256}"
    )
    assert client._subprocess_environment["CODEX_HOME"] == environment["CODEX_HOME"]
    assert "OPENAI_API_KEY" not in client._subprocess_environment
    assert public["authentication_mode"] == "chatgpt_account"
    assert public["transport_max_attempts"] == 1
    assert public["request_idle_timeout_seconds"] == 240.0
    assert public["request_hard_timeout_seconds"] == 1800.0
    assert client._reasoning_effort == "medium"
    assert public["reasoning_effort"] == "medium"
    assert public["reasoning_effort_source"] == "easyicu_account_research_default"
    assert public["progress_resets_idle_timeout"] is True
    assert public["strict_json_schema_enabled"] is True
    assert "must-not-cross-account-boundary" not in json.dumps(public)


def test_web_codex_scaffold_uses_app_server_output_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers.mocks import ExternalCaptureMockLLMClient

    client = ExternalCaptureMockLLMClient(
        [
            json.dumps(
                {
                    "agent_plan": {"steps": []},
                    "manuscript_draft": {"claims": [], "sentences": []},
                }
            )
        ]
    )

    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        lambda *_args, **_kwargs: (
            client,
            {
                "provider": "codex",
                "model": "gpt-5.6-luna",
                "strict_json_schema_enabled": True,
                "provider_gate_order": ["account_session_checked"],
            },
        ),
    )

    result = provider_adapter.generate_bound_provider_payload(
        provider_meta={"provider": "codex", "external": True},
        run_id="run_1",
        study_id="study_1",
        question="Synthetic question",
        summary={},
        cohort={},
        quality=[],
        environ={"EASYICU_DISABLE_PROVIDER_ENV_FILE": "1"},
    )

    _messages, captured = client.calls[0]
    assert captured["structured_output"].name == "easyicu_agent_run"
    assert result["agent_plan"]["run_id"] == "run_1"
    assert result["provider"]["json_format_style"] == (
        "codex_app_server_output_schema"
    )
    assert result["provider"]["external_calls"] == 1


def test_web_anthropic_scaffold_uses_native_typed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers.mocks import ExternalCaptureMockLLMClient

    client = ExternalCaptureMockLLMClient(
        [
            json.dumps(
                {
                    "agent_plan": {"steps": []},
                    "manuscript_draft": {"claims": [], "sentences": []},
                }
            )
        ]
    )
    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        lambda *_args, **_kwargs: (
            client,
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "strict_json_schema_enabled": True,
                "provider_gate_order": ["credentials_loaded"],
            },
        ),
    )

    result = provider_adapter.generate_bound_provider_payload(
        provider_meta={"provider": "anthropic", "external": True},
        run_id="run_1",
        study_id="study_1",
        question="Synthetic question",
        summary={},
        cohort={},
        quality=[],
        environ={"EASYICU_DISABLE_PROVIDER_ENV_FILE": "1"},
    )

    _messages, captured = client.calls[0]
    assert captured["structured_output"].name == "easyicu_agent_run"
    assert result["provider"]["json_format_style"] == "anthropic_output_config"
    assert result["provider"]["external_calls"] == 1
