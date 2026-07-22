"""Single provider contract shared by MCP, discovery, and benchmark entrypoints.

The factory deliberately owns credential selection.  In particular, a local
OpenAI-compatible endpoint receives a fixed non-secret credential even when
real OpenAI or OpenRouter keys are present in the process environment.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlsplit

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LOCAL_OPENAI_DUMMY_API_KEY = "easyicu-local-noauth"
EASYICU_HTTP_REFERER = "https://github.com/shen-lab-icu/easyicu"

# Opt-in: forward the real OPENAI_API_KEY to a LOOPBACK OpenAI-compatible
# endpoint. Default OFF -- a loopback endpoint receives only the non-secret
# dummy key, so an untrusted local server cannot harvest a paid secret. Set this
# to a truthy value ONLY when the loopback endpoint is a TRUSTED authenticating
# proxy that requires the real key (e.g. the local Codex Tools proxy on :8787,
# which now validates the client key and 401s the dummy). vLLM / Ollama ignore
# the key either way, so this is a no-op for a true no-auth local server.
TRUST_LOOPBACK_PROXY_KEY_ENV = "EASYICU_TRUST_LOOPBACK_PROXY_KEY"
ALLOW_EXTERNAL_LLM_ENV = "EASYICU_ALLOW_EXTERNAL_LLM"


def _loopback_forwards_real_key(env: Mapping[str, str]) -> bool:
    """True when the operator has opted in to forwarding the real key to a
    trusted loopback proxy (see ``TRUST_LOOPBACK_PROXY_KEY_ENV``)."""
    raw = str(env.get(TRUST_LOOPBACK_PROXY_KEY_ENV, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


MISSING_OPENAI_KEY = "missing_openai_key"
MISSING_OPENROUTER_KEY = "missing_openrouter_key"
INVALID_OPENAI_BASE_URL_OVERRIDE = "invalid_openai_base_url_override"
OPENROUTER_BASE_URL_OVERRIDE = "openrouter_base_url_override"
UNSUPPORTED_PROVIDER = "unsupported_provider"
EXTERNAL_LLM_NOT_AUTHORIZED = "external_llm_not_authorized"

_BASE_URL_UNSET = object()


def _openrouter_reasoning_extra_body(model: str) -> Optional[dict[str, Any]]:
    """Return the small provider-specific reasoning override, if proven safe."""

    lowered = str(model or "").strip().lower()
    if not lowered or "gpt-oss" in lowered:
        return None
    if any(token in lowered for token in ("glm", "qwen", "deepseek", "r1")):
        return {"reasoning": {"effort": "none", "exclude": True}}
    return None


@dataclass(frozen=True)
class ProviderAuthorization:
    """Non-secret provider destination authority minted by this factory."""

    provider: str
    model: str
    base_url: str
    destination: str
    authorization_mode: str
    authorization_sha256: str

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        model: str,
        base_url: str,
        destination: str,
        authorization_mode: str,
    ) -> "ProviderAuthorization":
        if destination not in {"external", "local"}:
            raise ValueError("provider destination must be external or local")
        payload = {
            "schema": "easyicu.provider_authorization/1",
            "provider": str(provider),
            "model": str(model),
            "base_url": str(base_url),
            "destination": destination,
            "authorization_mode": str(authorization_mode),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return cls(
            provider=payload["provider"],
            model=payload["model"],
            base_url=payload["base_url"],
            destination=destination,
            authorization_mode=payload["authorization_mode"],
            authorization_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )


def _attach_provider_authorization(
    client: Any,
    authorization: ProviderAuthorization,
) -> Any:
    """Attach factory-minted provenance; fail if a client cannot carry it."""

    try:
        setattr(client, "__easyicu_provider_authorization__", authorization)
    except Exception as exc:  # pragma: no cover - custom-client boundary
        # Test/configuration probes historically use a plain mapping as a
        # constructor spy. It is not an executable provider and cannot carry
        # runtime provenance; keep that introspection surface compatible.
        if client is None or isinstance(client, Mapping):
            return client
        raise ProviderConfigurationError(
            UNSUPPORTED_PROVIDER,
            authorization.provider,
        ) from exc
    return client


def provider_transport_destination(client: Any) -> str:
    """Classify provider transport without reading credentials.

    Direct non-loopback :class:`OpenAIClient` instances remain external even
    before all legacy constructors have migrated into this factory.
    """

    if bool(getattr(client, "__easyicu_mock_client__", False)):
        return "mock"
    authorization = getattr(client, "__easyicu_provider_authorization__", None)
    if isinstance(authorization, ProviderAuthorization):
        return authorization.destination
    if bool(getattr(client, "__easyicu_openai_transport__", False)):
        base_url = getattr(client, "_resolved_base_url", None)
        return "local" if is_loopback_openai_base_url(base_url) else "external"
    # Unknown adapters are never trusted merely because they are in-process:
    # they may forward requests to an arbitrary remote service. Treat them as
    # external so repair prompts receive the structured outbound envelope, and
    # keep their authorization mode unmanaged below. A truly local transport
    # must be factory-attested (for example, a loopback OpenAI endpoint).
    return "external"


def provider_authorization_manifest(client: Any) -> dict[str, Any]:
    """Return exact non-secret endpoint authorization for run provenance."""

    clients: list[Any]
    if isinstance(getattr(client, "_clients", None), (list, tuple)):
        clients = list(client._clients)
    elif hasattr(client, "iter_clients"):
        clients = list(client.iter_clients())
    else:
        clients = [client]
    records: list[dict[str, Any]] = []
    for item in clients:
        authorization = getattr(item, "__easyicu_provider_authorization__", None)
        if isinstance(authorization, ProviderAuthorization):
            records.append(
                {
                    "provider": authorization.provider,
                    "model": authorization.model,
                    "base_url": authorization.base_url,
                    "destination": authorization.destination,
                    "authorization_mode": authorization.authorization_mode,
                    "authorization_sha256": authorization.authorization_sha256,
                }
            )
            continue
        if bool(getattr(item, "__easyicu_mock_client__", False)):
            records.append(
                {
                    "provider": "mock",
                    "model": "mock",
                    "base_url": "",
                    "destination": "mock",
                    "authorization_mode": "mock_exempt",
                    "authorization_sha256": "",
                }
            )
            continue
        base_url = str(getattr(item, "_resolved_base_url", None) or "")
        destination = provider_transport_destination(item)
        records.append(
            {
                "provider": str(getattr(item, "name", type(item).__name__)),
                "model": str(getattr(item, "_model", "") or ""),
                "base_url": base_url,
                "destination": destination,
                "authorization_mode": "unmanaged",
                "authorization_sha256": "",
            }
        )
    unique = {
        json.dumps(record, sort_keys=True, separators=(",", ":")): record
        for record in records
    }
    return {
        "schema_version": "easyicu.provider_authorization_manifest/1",
        "clients": [unique[key] for key in sorted(unique)],
    }


def provider_authorization_for_configuration(
    *,
    provider: str,
    model: str,
    environment: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Mint non-secret identity coordinates without constructing a client."""

    env = os.environ if environment is None else environment
    normalized = str(provider or "").strip().lower()
    if normalized == "mock":
        return {
            "schema_version": "easyicu.provider_authorization_manifest/1",
            "clients": [
                {
                    "provider": "mock",
                    "model": "mock",
                    "base_url": "",
                    "destination": "mock",
                    "authorization_mode": "mock_exempt",
                    "authorization_sha256": "",
                }
            ],
        }
    if normalized not in {"openai", "openrouter"}:
        raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized)
    base_url = resolve_provider_base_url(normalized, environment=env)
    loopback = normalized == "openai" and is_loopback_openai_base_url(base_url)
    if not loopback and not _external_llm_allowed(env):
        raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, normalized)
    authorization = ProviderAuthorization.create(
        provider=normalized,
        model=model,
        base_url=base_url,
        destination="local" if loopback else "external",
        authorization_mode="local_exempt" if loopback else "operator_env",
    )
    return {
        "schema_version": "easyicu.provider_authorization_manifest/1",
        "clients": [
            {
                "provider": authorization.provider,
                "model": authorization.model,
                "base_url": authorization.base_url,
                "destination": authorization.destination,
                "authorization_mode": authorization.authorization_mode,
                "authorization_sha256": authorization.authorization_sha256,
            }
        ],
    }


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
        elif issue == EXTERNAL_LLM_NOT_AUTHORIZED:
            message = (
                "external LLM transport is disabled; set "
                f"{ALLOW_EXTERNAL_LLM_ENV}=1 only after approving the exact "
                "provider endpoint and outbound data policy"
            )
        else:
            message = f"Unsupported provider: {provider}"
        super().__init__(message)


def _external_llm_allowed(env: Mapping[str, str]) -> bool:
    raw = str(env.get(ALLOW_EXTERNAL_LLM_ENV, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


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
    client_cls: Callable[..., Any],
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
        if not _external_llm_allowed(env):
            raise ProviderConfigurationError(
                EXTERNAL_LLM_NOT_AUTHORIZED,
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
        extra_body = _openrouter_reasoning_extra_body(model)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        client = client_cls(**kwargs)
        return _attach_provider_authorization(
            client,
            ProviderAuthorization.create(
                provider=normalized_provider,
                model=model,
                base_url=str(kwargs["base_url"]),
                destination="external",
                authorization_mode="operator_env",
            ),
        )

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
        if not loopback and not _external_llm_allowed(env):
            raise ProviderConfigurationError(
                EXTERNAL_LLM_NOT_AUTHORIZED,
                normalized_provider,
            )
        # Loopback endpoints get the non-secret dummy key by default so an
        # untrusted local server cannot harvest a paid secret. When the operator
        # explicitly opts in (trusted authenticating proxy, e.g. Codex Tools on
        # :8787) AND a real key is present, forward the real key so the proxy
        # accepts the request instead of 401-ing the dummy.
        #
        # The opt-in trusts ONE operator-configured endpoint: the SERVER-OWNED
        # ``OPENAI_BASE_URL``. A per-request ``base_url_override`` is untrusted
        # (``has_override``) -- it can name ANY loopback port, including a listener
        # a local caller controls -- so it always receives the dummy key even under
        # the opt-in. Without this guard the flag (meant to trust one proxy) would
        # let a request-controlled override steer the paid secret to an
        # attacker-chosen loopback port and harvest it from the Authorization header.
        if loopback:
            trusted_loopback = (not has_override) and _loopback_forwards_real_key(env)
            loopback_key = (
                api_key
                if (api_key and trusted_loopback)
                else LOCAL_OPENAI_DUMMY_API_KEY
            )
        else:
            loopback_key = api_key
        kwargs = {
            "model": model,
            "request_timeout": float(request_timeout),
            "api_key": loopback_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        client = client_cls(**kwargs)
        resolved_url = str(base_url or DEFAULT_OPENAI_BASE_URL)
        return _attach_provider_authorization(
            client,
            ProviderAuthorization.create(
                provider=normalized_provider,
                model=model,
                base_url=resolved_url,
                destination="local" if loopback else "external",
                authorization_mode=("local_exempt" if loopback else "operator_env"),
            ),
        )

    raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized_provider)
