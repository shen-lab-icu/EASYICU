"""Dependency-neutral wire authentication contract for model providers.

The Pi setup screen, native Web provider bridge, and Research Agent provider
factory all consume this small contract.  It deliberately carries only a
non-secret header mode; credentials remain owned by their existing private
stores and are never serialized through this module.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping


OPENAI_AUTH_HEADER_ENV = "EASYICU_OPENAI_AUTH_HEADER"
PROVIDER_AUTH_HEADER_UNSUPPORTED = "provider_auth_header_unsupported"


class OpenAIAuthHeader(str, Enum):
    """Closed set of credential headers supported by OpenAI-compatible APIs."""

    AUTHORIZATION = "authorization"
    X_API_KEY = "x-api-key"


class ProviderAuthContractError(ValueError):
    """Raised when an untrusted auth-header mode is outside the closed set."""

    def __init__(self, value: object) -> None:
        self.code = PROVIDER_AUTH_HEADER_UNSUPPORTED
        self.value = str(value or "")
        super().__init__(self.code)


def normalize_openai_auth_header(value: object) -> OpenAIAuthHeader:
    """Return one typed header mode, defaulting to standard Bearer auth."""

    if isinstance(value, OpenAIAuthHeader):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return OpenAIAuthHeader.AUTHORIZATION
    try:
        return OpenAIAuthHeader(text)
    except ValueError as exc:
        raise ProviderAuthContractError(value) from exc


def pi_openai_auth_header(*, provider: str, api_transport: str) -> OpenAIAuthHeader:
    """Compile one Pi provider preset into its OpenAI-compatible wire mode."""

    normalized_provider = str(provider or "").strip().lower()
    normalized_transport = str(api_transport or "").strip().lower()
    if normalized_provider == "easyicu-local" and normalized_transport in {
        "openai-completions",
        "openai-responses",
    }:
        return OpenAIAuthHeader.X_API_KEY
    return OpenAIAuthHeader.AUTHORIZATION


def credential_headers(
    api_key: str,
    *,
    mode: OpenAIAuthHeader | str,
) -> Mapping[str, str]:
    """Build the credential header without changing or persisting the key."""

    resolved = normalize_openai_auth_header(mode)
    if resolved is OpenAIAuthHeader.X_API_KEY:
        return {"x-api-key": api_key}
    return {"Authorization": f"Bearer {api_key}"}


__all__ = [
    "OPENAI_AUTH_HEADER_ENV",
    "PROVIDER_AUTH_HEADER_UNSUPPORTED",
    "OpenAIAuthHeader",
    "ProviderAuthContractError",
    "credential_headers",
    "normalize_openai_auth_header",
    "pi_openai_auth_header",
]
