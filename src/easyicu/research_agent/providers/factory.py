"""Single provider contract shared by MCP, discovery, and benchmark entrypoints.

The factory deliberately owns credential selection.  In particular, a local
OpenAI-compatible endpoint receives a fixed non-secret credential even when
real OpenAI or OpenRouter keys are present in the process environment.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlsplit

from ..llm import OpenAIClient, openrouter_reasoning_extra_body

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LOCAL_OPENAI_DUMMY_API_KEY = "easyicu-local-noauth"
EASYICU_HTTP_REFERER = "https://github.com/shen-lab-icu/easyicu"

MISSING_OPENAI_KEY = "missing_openai_key"
MISSING_OPENROUTER_KEY = "missing_openrouter_key"
INVALID_OPENAI_BASE_URL_OVERRIDE = "invalid_openai_base_url_override"
OPENROUTER_BASE_URL_OVERRIDE = "openrouter_base_url_override"
UNSUPPORTED_PROVIDER = "unsupported_provider"

_BASE_URL_UNSET = object()


class ProviderConfigurationError(ValueError):
    """A provider setup failure that adapters can render in their own shape."""

    def __init__(self, issue: str, provider: str) -> None:
        self.issue = issue
        self.provider = provider
        if issue == MISSING_OPENAI_KEY:
            message = "OPENAI_API_KEY is required for --provider openai"
        elif issue == MISSING_OPENROUTER_KEY:
            message = "OPENROUTER_API_KEY is required for --provider openrouter"
        elif issue == INVALID_OPENAI_BASE_URL_OVERRIDE:
            message = (
                "a per-call base_url is permitted only for a parsed loopback "
                "HTTP(S) endpoint"
            )
        elif issue == OPENROUTER_BASE_URL_OVERRIDE:
            message = (
                "provider=openrouter does not accept a per-call base_url; "
                "configure OPENROUTER_BASE_URL on the server"
            )
        else:
            message = f"Unsupported provider: {provider}"
        super().__init__(message)


def is_loopback_openai_base_url(base_url: Optional[str]) -> bool:
    """Return whether *base_url* is an HTTP(S) endpoint on loopback only.

    This parses the URL and classifies its hostname with :mod:`ipaddress`.
    Substring matches are intentionally insufficient: wildcard bind addresses
    such as ``0.0.0.0`` and hostnames merely containing ``localhost`` are not
    loopback destinations.
    """

    try:
        parsed = urlsplit(str(base_url or "").strip())
        hostname = parsed.hostname
        # Accessing ``port`` also rejects malformed ports rather than treating
        # an invalid endpoint with a loopback-looking hostname as trusted.
        parsed.port
    except (TypeError, ValueError):
        return False
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        return False
    normalized_host = hostname.rstrip(".").lower()
    if normalized_host == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized_host).is_loopback
    except ValueError:
        return False


def resolve_provider_base_url(
    provider: str,
    *,
    environment: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve the non-secret backend URL used for run provenance."""

    env = os.environ if environment is None else environment
    normalized_provider = str(provider or "").strip().lower()
    if normalized_provider == "openrouter":
        return env.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL
    if normalized_provider == "openai":
        return env.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
    return "unknown"


def build_provider_client(
    *,
    provider: str,
    model: str,
    request_timeout: float,
    title: str,
    client_cls: Callable[..., Any] = OpenAIClient,
    environment: Optional[Mapping[str, str]] = None,
    base_url_override: object = _BASE_URL_UNSET,
) -> Any:
    """Build an OpenAI-compatible client under the canonical key policy.

    ``base_url_override`` represents an untrusted per-request override.  It is
    accepted only for a parsed loopback OpenAI endpoint and is never accepted
    for OpenRouter.  Server-owned environment URLs remain configurable, but
    every non-loopback OpenAI destination requires ``OPENAI_API_KEY``.
    """

    env = os.environ if environment is None else environment
    normalized_provider = str(provider or "").strip().lower()
    has_override = base_url_override is not _BASE_URL_UNSET

    if normalized_provider == "openrouter":
        if has_override:
            raise ProviderConfigurationError(
                OPENROUTER_BASE_URL_OVERRIDE,
                normalized_provider,
            )
        api_key = env.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ProviderConfigurationError(
                MISSING_OPENROUTER_KEY,
                normalized_provider,
            )
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "base_url": resolve_provider_base_url(
                normalized_provider,
                environment=env,
            ),
            "request_timeout": float(request_timeout),
            "extra_headers": {
                "HTTP-Referer": EASYICU_HTTP_REFERER,
                "X-Title": title,
            },
        }
        extra_body = openrouter_reasoning_extra_body(model)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        return client_cls(**kwargs)

    if normalized_provider == "openai":
        if has_override:
            base_url = str(base_url_override or "")
            if not is_loopback_openai_base_url(base_url):
                raise ProviderConfigurationError(
                    INVALID_OPENAI_BASE_URL_OVERRIDE,
                    normalized_provider,
                )
        else:
            base_url = str(env.get("OPENAI_BASE_URL") or "")

        api_key = env.get("OPENAI_API_KEY")
        loopback = is_loopback_openai_base_url(base_url)
        if not loopback and not api_key:
            raise ProviderConfigurationError(
                MISSING_OPENAI_KEY,
                normalized_provider,
            )
        kwargs = {
            "model": model,
            "request_timeout": float(request_timeout),
            "api_key": LOCAL_OPENAI_DUMMY_API_KEY if loopback else api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return client_cls(**kwargs)

    raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized_provider)
