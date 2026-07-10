"""Canonical construction helpers for external LLM providers."""

from .factory import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENROUTER_BASE_URL,
    LOCAL_OPENAI_DUMMY_API_KEY,
    ProviderConfigurationError,
    build_provider_client,
    is_loopback_openai_base_url,
    resolve_provider_base_url,
)

__all__ = [
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OPENROUTER_BASE_URL",
    "LOCAL_OPENAI_DUMMY_API_KEY",
    "ProviderConfigurationError",
    "build_provider_client",
    "is_loopback_openai_base_url",
    "resolve_provider_base_url",
]
