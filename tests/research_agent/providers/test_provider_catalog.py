from __future__ import annotations


def test_catalog_separates_provider_identity_from_openai_compatible_transport() -> None:
    from easyicu.research_agent.providers.capabilities import (
        ANTHROPIC_MESSAGES,
        DEFAULT_ANTHROPIC_BASE_URL,
        DEFAULT_DEEPSEEK_BASE_URL,
        OPENAI_CHAT_COMPLETIONS,
        SUPPORTED_PROVIDER_NAMES,
        provider_profile,
    )

    assert SUPPORTED_PROVIDER_NAMES == (
        "openai",
        "openrouter",
        "deepseek",
        "custom",
        "anthropic",
    )
    deepseek = provider_profile("DeepSeek")
    assert deepseek is not None
    assert deepseek.name == "deepseek"
    assert deepseek.transport == OPENAI_CHAT_COMPLETIONS
    assert deepseek.api_key_env_names[0] == "DEEPSEEK_API_KEY"
    assert deepseek.base_url_env_names[0] == "DEEPSEEK_BASE_URL"
    assert deepseek.model_env_names[0] == "DEEPSEEK_MODEL"
    assert deepseek.default_base_url == DEFAULT_DEEPSEEK_BASE_URL
    assert deepseek.supports_auth_header_override is False

    custom = provider_profile("custom")
    assert custom is not None
    assert custom.default_base_url is None
    assert custom.supports_auth_header_override is True
    assert custom.api_key_env_names == ("EASYICU_LLM_API_KEY",)
    assert custom.base_url_env_names == (
        "CUSTOM_BASE_URL",
        "EASYICU_LLM_BASE_URL",
    )
    assert provider_profile("native-anthropic") is None
    anthropic = provider_profile("anthropic")
    assert anthropic is not None
    assert anthropic.transport == ANTHROPIC_MESSAGES
    assert anthropic.default_base_url == DEFAULT_ANTHROPIC_BASE_URL
    assert anthropic.api_key_env_names[0] == "ANTHROPIC_API_KEY"


def test_catalog_environment_resolution_uses_provider_specific_then_generic() -> None:
    from easyicu.research_agent.providers.capabilities import provider_profile

    deepseek = provider_profile("deepseek")
    assert deepseek is not None
    key_source, key = deepseek.api_key(
        {
            "DEEPSEEK_API_KEY": "provider-specific",
            "EASYICU_LLM_API_KEY": "generic-fallback",
        }
    )
    assert (key_source, key) == ("DEEPSEEK_API_KEY", "provider-specific")

    key_source, key = deepseek.api_key({"EASYICU_LLM_API_KEY": "generic-fallback"})
    assert (key_source, key) == ("EASYICU_LLM_API_KEY", "generic-fallback")


def test_anthropic_endpoint_shaped_configuration_resolves_to_sdk_root() -> None:
    from easyicu.research_agent.providers.factory import resolve_provider_base_url

    for configured in (
        "https://api.anthropic.com/v1",
        "https://api.anthropic.com/v1/messages",
    ):
        assert resolve_provider_base_url(
            "anthropic",
            environment={"ANTHROPIC_BASE_URL": configured},
        ) == "https://api.anthropic.com"


def test_account_cli_catalog_keeps_login_transport_separate_from_api_providers() -> None:
    from easyicu.research_agent.providers.capabilities import (
        REGISTERED_CLI_BACKEND_NAMES,
        SUPPORTED_CLI_ACCOUNT_NAMES,
        cli_account_profile,
        provider_profile,
    )

    assert SUPPORTED_CLI_ACCOUNT_NAMES == ("codex",)
    assert REGISTERED_CLI_BACKEND_NAMES == ("codex", "claude")
    codex = cli_account_profile("Codex")
    assert codex is not None
    assert codex.provider_identity == "codex-cli"
    assert codex.endpoint_identity == "cli://codex"
    assert codex.status_argv == ("codex", "login", "status")
    assert codex.supports_strict_json_schema is True
    claude = cli_account_profile("claude")
    assert claude is not None
    assert claude.supports_strict_json_schema is False
    assert cli_account_profile("gemini") is None
    assert provider_profile("codex") is None
