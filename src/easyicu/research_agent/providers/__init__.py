"""Provider contracts plus lazy construction helpers.

The factory imports the concrete production client from :mod:`.llm`, while
that module imports the provider-neutral protocol from this package. Keeping
factory exports lazy prevents package initialization from turning that legal
layering into a runtime import cycle.
"""

from __future__ import annotations

from typing import Any

_FACTORY_EXPORTS = frozenset(
    {
        "LOCAL_OPENAI_DUMMY_API_KEY",
        "ProviderConfigurationError",
        "build_provider_client",
        "is_loopback_openai_base_url",
        "resolve_provider_base_url",
    }
)
_CAPABILITY_EXPORTS = frozenset(
    {
        "ANTHROPIC_MESSAGES",
        "DEFAULT_ANTHROPIC_BASE_URL",
        "DEFAULT_DEEPSEEK_BASE_URL",
        "DEFAULT_OPENAI_BASE_URL",
        "DEFAULT_OPENROUTER_BASE_URL",
        "CLIAccountReadiness",
        "CLIAccountProfile",
        "ProviderProfile",
        "REGISTERED_CLI_BACKEND_NAMES",
        "SUPPORTED_CLI_ACCOUNT_NAMES",
        "SUPPORTED_PROVIDER_NAMES",
        "cli_account_profile",
        "provider_profile",
    }
)
__all__ = sorted(_FACTORY_EXPORTS | _CAPABILITY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _CAPABILITY_EXPORTS:
        from . import capabilities as capabilities_module

        value = getattr(capabilities_module, name)
    elif name in _FACTORY_EXPORTS:
        from . import factory as factory_module

        value = getattr(factory_module, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
