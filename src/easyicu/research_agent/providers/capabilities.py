"""Dependency-neutral capability probes for configured LLM clients.

This module owns provider capability discovery without importing a concrete
transport implementation. Gates can therefore ask whether an injected client
supports an optional capability without acquiring the production provider (or
its offline mock fallback) as a transitive dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
ANTHROPIC_MESSAGES = "anthropic_messages"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"


@dataclass(frozen=True)
class ProviderProfile:
    """Immutable configuration coordinates for one provider identity."""

    name: str
    transport: str
    api_key_env_names: tuple[str, ...]
    base_url_env_names: tuple[str, ...]
    model_env_names: tuple[str, ...]
    default_base_url: Optional[str] = None
    sends_attribution_headers: bool = False
    supports_auth_header_override: bool = False

    def first_environment_value(
        self,
        environment: Mapping[str, str],
        names: tuple[str, ...],
    ) -> tuple[Optional[str], str]:
        for name in names:
            value = str(environment.get(name) or "").strip()
            if value:
                return name, value
        return None, ""

    def api_key(self, environment: Mapping[str, str]) -> tuple[Optional[str], str]:
        return self.first_environment_value(environment, self.api_key_env_names)

    def base_url(self, environment: Mapping[str, str]) -> tuple[Optional[str], str]:
        name, value = self.first_environment_value(
            environment,
            self.base_url_env_names,
        )
        return name, value or str(self.default_base_url or "")

    def model(self, environment: Mapping[str, str]) -> tuple[Optional[str], str]:
        return self.first_environment_value(environment, self.model_env_names)


@dataclass(frozen=True)
class CLIAccountProfile:
    """Reviewed local CLI transport backed by an operator-owned login session."""

    name: str
    executable: str
    provider_identity: str
    endpoint_identity: str
    status_argv: tuple[str, ...] | None
    model_env_names: tuple[str, ...]
    supports_strict_json_schema: bool

    def model(self, environment: Mapping[str, str]) -> tuple[Optional[str], str]:
        """Resolve only backend-specific model overrides."""

        for name in self.model_env_names:
            value = str(environment.get(name) or "").strip()
            if value:
                return name, value
        return None, ""


@dataclass(frozen=True)
class CLIAccountReadiness:
    """Sanitized result of one bounded local account-session probe."""

    backend: str
    provider_identity: str
    executable_present: bool
    status_check_supported: bool
    authentication_verified: Optional[bool]
    launch_ready: bool
    reason_code: str
    subprocess_calls: int


_CLI_ACCOUNT_PROFILES = {
    "codex": CLIAccountProfile(
        name="codex",
        executable="codex",
        provider_identity="codex-cli",
        endpoint_identity="cli://codex",
        status_argv=("codex", "login", "status"),
        model_env_names=("EASYICU_CODEX_MODEL", "CODEX_MODEL"),
        supports_strict_json_schema=True,
    ),
    "claude": CLIAccountProfile(
        name="claude",
        executable="claude",
        provider_identity="claude-cli",
        endpoint_identity="cli://claude",
        # Claude Code supports account login, but its cross-version CLI does
        # not expose one stable, documented non-interactive status contract.
        status_argv=None,
        model_env_names=("EASYICU_CLAUDE_MODEL", "CLAUDE_MODEL"),
        supports_strict_json_schema=False,
    ),
}


# Only Codex is a user-facing account provider. Claude remains an internal,
# backwards-compatible agentic-coder backend; Research Agent users select the
# native Anthropic API instead of inheriting Claude Code's local configuration.
REGISTERED_CLI_BACKEND_NAMES = tuple(_CLI_ACCOUNT_PROFILES)
SUPPORTED_CLI_ACCOUNT_NAMES = ("codex",)


def cli_account_profile(backend: str) -> Optional[CLIAccountProfile]:
    """Return one reviewed account-backed CLI profile, if registered."""

    return _CLI_ACCOUNT_PROFILES.get(str(backend or "").strip().lower())


_PROFILES = {
    "openai": ProviderProfile(
        name="openai",
        transport=OPENAI_CHAT_COMPLETIONS,
        api_key_env_names=("OPENAI_API_KEY", "EASYICU_LLM_API_KEY"),
        base_url_env_names=("OPENAI_BASE_URL", "EASYICU_LLM_BASE_URL"),
        model_env_names=("OPENAI_MODEL", "EASYICU_LLM_MODEL"),
        default_base_url=DEFAULT_OPENAI_BASE_URL,
        supports_auth_header_override=True,
    ),
    "openrouter": ProviderProfile(
        name="openrouter",
        transport=OPENAI_CHAT_COMPLETIONS,
        api_key_env_names=("OPENROUTER_API_KEY", "EASYICU_LLM_API_KEY"),
        base_url_env_names=("OPENROUTER_BASE_URL", "EASYICU_LLM_BASE_URL"),
        model_env_names=("OPENROUTER_MODEL", "EASYICU_LLM_MODEL"),
        default_base_url=DEFAULT_OPENROUTER_BASE_URL,
        sends_attribution_headers=True,
    ),
    "deepseek": ProviderProfile(
        name="deepseek",
        transport=OPENAI_CHAT_COMPLETIONS,
        api_key_env_names=("DEEPSEEK_API_KEY", "EASYICU_LLM_API_KEY"),
        base_url_env_names=("DEEPSEEK_BASE_URL", "EASYICU_LLM_BASE_URL"),
        model_env_names=("DEEPSEEK_MODEL", "EASYICU_LLM_MODEL"),
        default_base_url=DEFAULT_DEEPSEEK_BASE_URL,
    ),
    "custom": ProviderProfile(
        name="custom",
        transport=OPENAI_CHAT_COMPLETIONS,
        api_key_env_names=("EASYICU_LLM_API_KEY",),
        base_url_env_names=("CUSTOM_BASE_URL", "EASYICU_LLM_BASE_URL"),
        model_env_names=("CUSTOM_MODEL", "EASYICU_LLM_MODEL"),
        supports_auth_header_override=True,
    ),
    "anthropic": ProviderProfile(
        name="anthropic",
        transport=ANTHROPIC_MESSAGES,
        api_key_env_names=("ANTHROPIC_API_KEY", "EASYICU_LLM_API_KEY"),
        base_url_env_names=("ANTHROPIC_BASE_URL", "EASYICU_LLM_BASE_URL"),
        model_env_names=("ANTHROPIC_MODEL", "EASYICU_LLM_MODEL"),
        default_base_url=DEFAULT_ANTHROPIC_BASE_URL,
    ),
}


SUPPORTED_PROVIDER_NAMES = tuple(_PROFILES)


def provider_profile(provider: str) -> Optional[ProviderProfile]:
    """Return the reviewed profile for *provider*, or ``None`` if unknown."""

    return _PROFILES.get(str(provider or "").strip().lower())


def model_looks_vision_capable(model: str) -> bool:
    """Return the conservative name-based default for vision support."""

    lowered = (model or "").strip().lower()
    if not lowered:
        return False
    positive_tokens = (
        "gpt-4o",
        "omni",
        "vision",
        "gemini",
        "qwen-vl",
        "qwen2.5-vl",
        "vl-",
        "pixtral",
        "llava",
        "molmo",
        "internvl",
    )
    negative_tokens = (
        "coder",
        "instruct",
        "reasoner",
        "embedding",
        "rerank",
        "whisper",
        "audio",
    )
    if any(token in lowered for token in negative_tokens):
        return False
    return any(token in lowered for token in positive_tokens)


def llm_supports_vision(client: Any) -> bool:
    """Best-effort capability probe for optional figure-VLM review.

    Unknown clients fail closed. Wrappers and routers may expose their children
    through ``for_role`` or ``iter_clients``; the probe follows those public
    seams without importing or naming any concrete provider class.
    """

    if client is None:
        return False
    if hasattr(client, "supports_vision"):
        advertised = getattr(client, "supports_vision")
        try:
            return bool(advertised() if callable(advertised) else advertised)
        except Exception:
            return False
    if hasattr(client, "for_role"):
        try:
            analyzer_client = client.for_role("analyzer")
        except Exception:
            analyzer_client = None
        if analyzer_client is not None:
            return llm_supports_vision(analyzer_client)
    if hasattr(client, "iter_clients"):
        try:
            return any(llm_supports_vision(child) for child in client.iter_clients())
        except Exception:
            return False
    if hasattr(client, "complete_with_images"):
        model = getattr(client, "_model", None)
        if model is None:
            return True
        return model_looks_vision_capable(str(model))
    return False


def llm_supports_strict_json_schema(client: Any) -> bool:
    """Return only an explicitly advertised strict-schema capability.

    Name-based guessing is unsafe for OpenAI-compatible relays: two endpoints
    serving the same model name may implement different response-format
    subsets. Wrappers expose the leaf advertisement through their existing
    delegation seam; routers are resolved to the Planner role.
    """

    if client is None:
        return False
    if hasattr(client, "supports_strict_json_schema"):
        advertised = getattr(client, "supports_strict_json_schema")
        try:
            return bool(advertised() if callable(advertised) else advertised)
        except Exception:
            return False
    if hasattr(client, "for_role"):
        try:
            planner_client = client.for_role("planner")
        except Exception:
            planner_client = None
        if planner_client is not None:
            return llm_supports_strict_json_schema(planner_client)
    return False


__all__ = [
    "ANTHROPIC_MESSAGES",
    "CLIAccountReadiness",
    "CLIAccountProfile",
    "DEFAULT_ANTHROPIC_BASE_URL",
    "DEFAULT_DEEPSEEK_BASE_URL",
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OPENROUTER_BASE_URL",
    "OPENAI_CHAT_COMPLETIONS",
    "ProviderProfile",
    "REGISTERED_CLI_BACKEND_NAMES",
    "SUPPORTED_CLI_ACCOUNT_NAMES",
    "SUPPORTED_PROVIDER_NAMES",
    "cli_account_profile",
    "llm_supports_strict_json_schema",
    "llm_supports_vision",
    "model_looks_vision_capable",
    "provider_profile",
]
