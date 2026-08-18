"""Single provider contract shared by MCP, discovery, and benchmark entrypoints.

The factory deliberately owns credential selection.  In particular, a local
OpenAI-compatible endpoint receives a fixed non-secret credential even when
real OpenAI or OpenRouter keys are present in the process environment.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional, Sequence

from easyicu.provider_auth import (
    OPENAI_AUTH_HEADER_ENV,
    OpenAIAuthHeader,
    ProviderAuthContractError,
    normalize_openai_auth_header,
)

from .capabilities import (
    ANTHROPIC_MESSAGES,
    OPENAI_CHAT_COMPLETIONS,
    SUPPORTED_PROVIDER_NAMES,
    cli_account_profile,
    provider_profile,
    user_account_profile,
)

# Client introspection lives in its own dependency-neutral owner so that
# layers which only inspect a client do not depend on client construction.
from .client_trust import (  # noqa: F401  (re-exported for existing callers)
    ALLOW_EXTERNAL_LLM_ENV,
    EXTERNAL_LLM_NOT_AUTHORIZED,
    INVALID_OPENAI_BASE_URL_OVERRIDE,
    INVALID_PROVIDER_BASE_URL_OVERRIDE,
    MISSING_OPENAI_KEY,
    MISSING_OPENROUTER_KEY,
    MISSING_PROVIDER_BASE_URL,
    MISSING_PROVIDER_KEY,
    OPENROUTER_BASE_URL_OVERRIDE,
    UNSUPPORTED_PROVIDER,
    _CONSTRUCTED_CLIENTS,
    _ConstructedClientRecord,
    _attach_provider_authorization,
    _caller_is_exact_constructor,
    _constructed_client_record,
    _mark_reviewed_transport_constructed,
    _new_trusted_record,
    _refresh_reviewed_transport_dispatch,
    _register_external_capture_test_client,
    _register_loopback_provider_client,
    _remember_trusted_client,
    register_offline_test_client,
    ProviderConfigurationError,
    _callable_code,
    _callable_contract,
    _callable_contracts_match,
    _callables_match_record,
    _canonical_endpoint,
    _class_callable,
    _dispatch_matches_record,
    _reviewed_dispatch_identity,
    _transport_matches_authorization,
    authorized_complete,
    authorized_complete_with_images,
    require_provider_client_authorization,
    ProviderAuthorization,
    _CallableContract,
    _TRUSTED_CLIENTS,
    _TRUSTED_CLIENTS_LOCK,
    _TrustedClientRecord,
    _configured_cli_transport_policy,
    _configured_transport_policy,
    _is_reviewed_client_type,
    _provider_transport_policy,
    _safe_instance_vars,
    _trusted_client_record,
    _valid_provider_authorization,
    is_loopback_openai_base_url,
    provider_authorization_manifest,
    provider_client_is_offline,
    provider_transport_destination,
)

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


def _loopback_forwards_real_key(env: Mapping[str, str]) -> bool:
    """True when the operator has opted in to forwarding the real key to a
    trusted loopback proxy (see ``TRUST_LOOPBACK_PROXY_KEY_ENV``)."""
    raw = str(env.get(TRUST_LOOPBACK_PROXY_KEY_ENV, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


OPENAI_AUTH_HEADER_NOT_AUTHORIZED = "openai_auth_header_not_authorized"

_BASE_URL_UNSET = object()


def _openai_auth_header(
    env: Mapping[str, str],
    *,
    provider: str = "openai",
) -> OpenAIAuthHeader:
    profile = provider_profile(provider)
    if profile is not None and not profile.supports_auth_header_override:
        return OpenAIAuthHeader.AUTHORIZATION
    try:
        return normalize_openai_auth_header(env.get(OPENAI_AUTH_HEADER_ENV))
    except ProviderAuthContractError as exc:
        raise ProviderConfigurationError(
            OPENAI_AUTH_HEADER_NOT_AUTHORIZED,
            provider,
        ) from exc


def _openai_authorization_mode(
    env: Mapping[str, str],
    *,
    provider: str = "openai",
    loopback: bool,
    has_override: bool,
    api_key: Optional[str],
) -> tuple[OpenAIAuthHeader, str]:
    """Validate one wire header against the trusted loopback authority."""

    header = _openai_auth_header(env, provider=provider)
    if header is OpenAIAuthHeader.X_API_KEY:
        if (
            not loopback
            or has_override
            or not api_key
            or not _loopback_forwards_real_key(env)
        ):
            raise ProviderConfigurationError(
                OPENAI_AUTH_HEADER_NOT_AUTHORIZED,
                provider,
            )
        return header, "local_x_api_key"
    return header, "local_exempt" if loopback else "operator_env"














































def provider_client_is_mockish(client: Any) -> bool:
    """Classify only factory-registered, intact offline graphs as mockish."""

    stack = [client]
    seen: set[int] = set()
    found_leaf = False
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        record = _trusted_client_record(current)
        if (
            record is None
            or not _callables_match_record(current, record)
            or not _dispatch_matches_record(current, record)
        ):
            return False
        if record.kind == "offline":
            found_leaf = True
            continue
        if record.kind != "wrapper" or record.children_getter is None:
            return False
        try:
            children = tuple(record.children_getter())
        except Exception:
            return False
        if tuple(id(child) for child in children) != record.child_ids:
            return False
        stack.extend(children)
    return found_leaf




def _register_provider_wrapper(
    wrapper: Any,
    *,
    children_getter: Callable[[], Sequence[Any]],
) -> Any:
    """Register one reviewed wrapper and bind its complete child graph."""

    children = tuple(children_getter())
    if not children or any(child is None or child is wrapper for child in children):
        raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, type(wrapper).__name__)
    _remember_trusted_client(
        wrapper,
        _new_trusted_record(
            wrapper,
            kind="wrapper",
            authorization=None,
            child_ids=tuple(id(child) for child in children),
            children_getter=children_getter,
        ),
    )
    return wrapper


def _openrouter_reasoning_extra_body(model: str) -> Optional[dict[str, Any]]:
    """Return the small provider-specific reasoning override, if proven safe."""

    lowered = str(model or "").strip().lower()
    if not lowered or "gpt-oss" in lowered:
        return None
    if any(token in lowered for token in ("glm", "qwen", "deepseek", "r1")):
        return {"reasoning": {"effort": "none", "exclude": True}}
    return None






def authorize_provider_client(
    client: Any,
    *,
    provider: str,
    model: str,
    base_url: str,
    destination: str,
    environment: Optional[Mapping[str, str]] = None,
) -> Any:
    """Attach explicit endpoint authority to a non-OpenAI provider adapter.

    This is the only supported bridge for adapters such as the local coding
    agent CLI.  Unknown custom clients are not implicitly trusted merely
    because they implement ``complete``.
    """

    env = os.environ if environment is None else environment
    if destination not in {"external", "local"}:
        raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, provider)
    is_openai = _is_reviewed_client_type(client, "OpenAIClient")
    is_anthropic = _is_reviewed_client_type(client, "AnthropicMessagesClient")
    is_cli = _is_reviewed_client_type(client, "CLIAgentLLMClient")
    is_codex_user = _is_reviewed_client_type(client, "CodexAppServerLLMClient")
    if not (is_openai or is_anthropic or is_cli or is_codex_user):
        raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
    if is_openai:
        profile_definition = provider_profile(provider)
        if (
            profile_definition is None
            or profile_definition.transport != OPENAI_CHAT_COMPLETIONS
        ):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
        live_base_url = str(getattr(client, "_resolved_base_url", None) or "")
        live_model = str(getattr(client, "_model", "") or "")
        if _canonical_endpoint(live_base_url) != _canonical_endpoint(base_url):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
        if live_model != str(model):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
        if (destination == "local") != is_loopback_openai_base_url(live_base_url):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
    elif is_anthropic:
        profile_definition = provider_profile(provider)
        live_base_url = str(getattr(client, "_resolved_base_url", None) or "")
        live_model = str(getattr(client, "_model", "") or "")
        if (
            profile_definition is None
            or profile_definition.transport != ANTHROPIC_MESSAGES
            or _canonical_endpoint(live_base_url) != _canonical_endpoint(base_url)
            or live_model != str(model)
            or destination != "external"
        ):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
    elif is_cli:
        backend = str(getattr(client, "_backend", "") or "")
        profile_definition = cli_account_profile(backend)
        live_model = str(getattr(client, "_model", "") or "") or "cli-default"
        if (
            profile_definition is None
            or str(provider) != profile_definition.provider_identity
            or str(base_url) != profile_definition.endpoint_identity
            or str(model) != live_model
        ):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
        if destination == "local":
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
    else:
        profile_definition = user_account_profile("codex")
        live_model = str(getattr(client, "_model", "") or "") or "account-default"
        live_endpoint = str(getattr(client, "_endpoint_identity", "") or "")
        session_binding = str(
            getattr(client, "_session_binding_sha256", "") or ""
        )
        if (
            profile_definition is None
            or str(provider) != profile_definition.provider_identity
            or str(base_url) != live_endpoint
            or str(model) != live_model
            or not session_binding
            or not live_endpoint.endswith("/session/" + session_binding)
            or destination != "external"
        ):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
    if destination == "external" and not _external_llm_allowed(env):
        raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
    authorization = ProviderAuthorization.create(
        provider=str(provider),
        model=str(model),
        base_url=str(base_url),
        destination=destination,
        authorization_mode=(
            "account_session"
            if is_cli or is_codex_user
            else ("operator_env" if destination == "external" else "local_exempt")
        ),
    )
    return _attach_provider_authorization(client, authorization)


























def provider_authorization_for_configuration(
    *,
    provider: str,
    model: str,
    environment: Optional[Mapping[str, str]] = None,
    reasoning_effort_profile: str = "provider_default",
    request_timeout: float = 120.0,
    transport_max_attempts: int = 9,
    retryable_http_status_codes: Optional[Sequence[int]] = None,
    stream_enabled: bool = False,
    supports_strict_json_schema: bool = False,
) -> dict[str, Any]:
    """Mint non-secret identity coordinates without constructing a client."""

    env = os.environ if environment is None else environment
    profile = str(reasoning_effort_profile or "").strip().lower()
    if profile not in {"provider_default", "adaptive_v1"}:
        raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, profile)
    normalized = str(provider or "").strip().lower()
    transport_policy = _configured_transport_policy(
        request_timeout=request_timeout,
        transport_max_attempts=transport_max_attempts,
        retryable_http_status_codes=retryable_http_status_codes,
        stream_enabled=stream_enabled,
        supports_strict_json_schema=supports_strict_json_schema,
    )
    if normalized == "mock":
        return {
            "schema_version": "easyicu.provider_authorization_manifest/3",
            "reasoning_effort_profile": profile,
            "clients": [
                {
                    "provider": "mock",
                    "model": "mock",
                    "base_url": "",
                    "destination": "mock",
                    "authorization_mode": "mock_exempt",
                    "authorization_sha256": "",
                    "transport_policy": {
                        "schema_version": "easyicu.provider_transport_policy/2",
                        "transport": "offline",
                        "request_timeout_seconds": None,
                        "transport_max_attempts": 0,
                        "retryable_http_status_codes": None,
                        "stream_enabled": False,
                        "strict_json_schema_enabled": False,
                    },
                }
            ],
        }
    cli_profile = cli_account_profile(normalized)
    if cli_profile is not None:
        if profile != "provider_default":
            raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized)
        if stream_enabled or transport_max_attempts != 1:
            raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized)
        if not _external_llm_allowed(env):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, normalized)
        _model_source, configured_model = cli_profile.model(env)
        selected_model = (
            str(model or "").strip()
            or str(configured_model or "").strip()
            or "cli-default"
        )
        authorization = ProviderAuthorization.create(
            provider=cli_profile.provider_identity,
            model=selected_model,
            base_url=cli_profile.endpoint_identity,
            destination="external",
            authorization_mode="account_session",
        )
        return {
            "schema_version": "easyicu.provider_authorization_manifest/3",
            "reasoning_effort_profile": profile,
            "clients": [
                {
                    "provider": authorization.provider,
                    "model": authorization.model,
                    "base_url": authorization.base_url,
                    "destination": authorization.destination,
                    "authorization_mode": authorization.authorization_mode,
                    "authorization_sha256": authorization.authorization_sha256,
                    "transport_policy": _configured_cli_transport_policy(
                        request_timeout=request_timeout,
                        supports_strict_json_schema=(
                            cli_profile.supports_strict_json_schema
                        ),
                    ),
                }
            ],
        }
    profile_definition = provider_profile(normalized)
    if profile_definition is None:
        raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized)
    if profile == "adaptive_v1" and profile_definition.transport != OPENAI_CHAT_COMPLETIONS:
        raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized)
    transport_policy = _configured_transport_policy(
        request_timeout=request_timeout,
        transport_max_attempts=transport_max_attempts,
        retryable_http_status_codes=retryable_http_status_codes,
        stream_enabled=stream_enabled,
        supports_strict_json_schema=supports_strict_json_schema,
        transport=(
            "anthropic_messages"
            if profile_definition.transport == ANTHROPIC_MESSAGES
            else "openai_compatible"
        ),
    )
    base_url = resolve_provider_base_url(normalized, environment=env)
    if not base_url or base_url == "unknown":
        raise ProviderConfigurationError(MISSING_PROVIDER_BASE_URL, normalized)
    loopback = (
        profile_definition.transport == OPENAI_CHAT_COMPLETIONS
        and normalized != "openrouter"
        and is_loopback_openai_base_url(base_url)
    )
    if not loopback and not _external_llm_allowed(env):
        raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, normalized)
    authorization_mode = "operator_env"
    if (
        profile_definition.transport == OPENAI_CHAT_COMPLETIONS
        and normalized != "openrouter"
    ):
        _key_name, api_key = profile_definition.api_key(env)
        _header, authorization_mode = _openai_authorization_mode(
            env,
            provider=normalized,
            loopback=loopback,
            has_override=False,
            api_key=api_key,
        )
    authorization = ProviderAuthorization.create(
        provider=normalized,
        model=model,
        base_url=base_url,
        destination="local" if loopback else "external",
        authorization_mode=authorization_mode,
    )
    return {
        "schema_version": "easyicu.provider_authorization_manifest/3",
        "reasoning_effort_profile": profile,
        "clients": [
            {
                "provider": authorization.provider,
                "model": authorization.model,
                "base_url": authorization.base_url,
                "destination": authorization.destination,
                "authorization_mode": authorization.authorization_mode,
                "authorization_sha256": authorization.authorization_sha256,
                "transport_policy": transport_policy,
            }
        ],
    }




def _external_llm_allowed(env: Mapping[str, str]) -> bool:
    raw = str(env.get(ALLOW_EXTERNAL_LLM_ENV, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}




def resolve_provider_base_url(
    provider: str,
    *,
    environment: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve the non-secret backend URL used for run provenance."""

    env = os.environ if environment is None else environment
    profile_definition = provider_profile(provider)
    if profile_definition is None:
        return "unknown"
    _source, base_url = profile_definition.base_url(env)
    if profile_definition.transport == ANTHROPIC_MESSAGES:
        # The Anthropic SDK appends ``/v1/messages`` itself. Accept the two
        # endpoint-shaped values operators commonly paste, but bind and pass
        # only the SDK root so the request cannot become ``/v1/v1/messages``.
        normalized = str(base_url or "").rstrip("/")
        for suffix in ("/v1/messages", "/v1"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        base_url = normalized
    return base_url or "unknown"


def build_provider_client(
    *,
    provider: str,
    model: str,
    request_timeout: float,
    title: str,
    client_cls: Optional[Callable[..., Any]] = None,
    environment: Optional[Mapping[str, str]] = None,
    base_url_override: object = _BASE_URL_UNSET,
    extra_body: Optional[Mapping[str, Any]] = None,
    max_retries: int = 8,
    retryable_http_status_codes: Optional[Sequence[int]] = None,
    stream_enabled: Optional[bool] = None,
    supports_strict_json_schema: bool = False,
    allow_environment_overrides: bool = True,
) -> Any:
    """Build a reviewed API client under the canonical key policy.

    ``base_url_override`` represents an untrusted per-request override.  It is
    accepted only for a parsed loopback OpenAI-compatible endpoint and is never
    accepted for OpenRouter. Server-owned environment URLs remain configurable,
    while each external provider resolves credentials through its reviewed
    profile. Strict JSON Schema capability is explicit and is never inferred
    from the provider name.
    """

    env = os.environ if environment is None else environment
    normalized_provider = str(provider or "").strip().lower()
    has_override = base_url_override is not _BASE_URL_UNSET
    profile_definition = provider_profile(normalized_provider)
    if profile_definition is None:
        raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized_provider)

    if profile_definition.transport == ANTHROPIC_MESSAGES:
        if has_override:
            raise ProviderConfigurationError(
                INVALID_PROVIDER_BASE_URL_OVERRIDE,
                normalized_provider,
            )
        base_url = resolve_provider_base_url(
            normalized_provider,
            environment=env,
        )
        if not base_url or base_url == "unknown":
            raise ProviderConfigurationError(
                MISSING_PROVIDER_BASE_URL,
                normalized_provider,
            )
        _key_name, api_key = profile_definition.api_key(env)
        if not api_key:
            raise ProviderConfigurationError(
                MISSING_PROVIDER_KEY,
                normalized_provider,
            )
        if not _external_llm_allowed(env):
            raise ProviderConfigurationError(
                EXTERNAL_LLM_NOT_AUTHORIZED,
                normalized_provider,
            )
        if stream_enabled:
            raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized_provider)
        if extra_body:
            raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized_provider)
        selected_client_cls = client_cls
        if selected_client_cls is None:
            from .clients import AnthropicMessagesClient

            selected_client_cls = AnthropicMessagesClient
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "request_timeout": float(request_timeout),
            "max_retries": int(max_retries),
            "stream_enabled": False,
            "supports_strict_json_schema": bool(supports_strict_json_schema),
            "allow_environment_overrides": bool(allow_environment_overrides),
        }
        if retryable_http_status_codes is not None:
            kwargs["retryable_http_status_codes"] = tuple(
                retryable_http_status_codes
            )
        client = selected_client_cls(**kwargs)
        return _attach_provider_authorization(
            client,
            ProviderAuthorization.create(
                provider=normalized_provider,
                model=model,
                base_url=base_url,
                destination="external",
                authorization_mode="operator_env",
            ),
        )

    if normalized_provider == "openrouter":
        if has_override:
            raise ProviderConfigurationError(
                OPENROUTER_BASE_URL_OVERRIDE,
                normalized_provider,
            )
        _key_name, api_key = profile_definition.api_key(env)
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
            "max_retries": int(max_retries),
            "stream_enabled": stream_enabled,
            "supports_strict_json_schema": bool(supports_strict_json_schema),
            "allow_environment_overrides": bool(allow_environment_overrides),
            "extra_headers": {
                "HTTP-Referer": EASYICU_HTTP_REFERER,
                "X-Title": title,
            },
        }
        if retryable_http_status_codes is not None:
            kwargs["retryable_http_status_codes"] = tuple(retryable_http_status_codes)
        provider_extra_body = _openrouter_reasoning_extra_body(model)
        merged_extra_body = dict(provider_extra_body or {})
        merged_extra_body.update(dict(extra_body or {}))
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body
        selected_client_cls = client_cls
        if selected_client_cls is None:
            from .clients import OpenAIClient

            selected_client_cls = OpenAIClient
        client = selected_client_cls(**kwargs)
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

    if (
        normalized_provider in SUPPORTED_PROVIDER_NAMES
        and profile_definition.transport == OPENAI_CHAT_COMPLETIONS
    ):
        if has_override:
            base_url = str(base_url_override or "")
            if not is_loopback_openai_base_url(base_url):
                raise ProviderConfigurationError(
                    (
                        INVALID_OPENAI_BASE_URL_OVERRIDE
                        if normalized_provider == "openai"
                        else INVALID_PROVIDER_BASE_URL_OVERRIDE
                    ),
                    normalized_provider,
                )
        else:
            base_url = resolve_provider_base_url(
                normalized_provider,
                environment=env,
            )
        if not base_url or base_url == "unknown":
            raise ProviderConfigurationError(
                MISSING_PROVIDER_BASE_URL,
                normalized_provider,
            )

        _key_name, api_key = profile_definition.api_key(env)
        loopback = is_loopback_openai_base_url(base_url)
        auth_header, authorization_mode = _openai_authorization_mode(
            env,
            provider=normalized_provider,
            loopback=loopback,
            has_override=has_override,
            api_key=api_key,
        )
        if not loopback and not api_key:
            raise ProviderConfigurationError(
                (
                    MISSING_OPENAI_KEY
                    if normalized_provider == "openai"
                    else MISSING_PROVIDER_KEY
                ),
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
            "max_retries": int(max_retries),
            "stream_enabled": stream_enabled,
            "supports_strict_json_schema": bool(supports_strict_json_schema),
            "allow_environment_overrides": bool(allow_environment_overrides),
        }
        if retryable_http_status_codes is not None:
            kwargs["retryable_http_status_codes"] = tuple(retryable_http_status_codes)
        if base_url:
            kwargs["base_url"] = base_url
        if extra_body:
            kwargs["extra_body"] = dict(extra_body)
        if auth_header is OpenAIAuthHeader.X_API_KEY:
            assert loopback_key is not None
            kwargs["extra_headers"] = {"x-api-key": loopback_key}
        selected_client_cls = client_cls
        if selected_client_cls is None:
            from .clients import OpenAIClient

            selected_client_cls = OpenAIClient
        client = selected_client_cls(**kwargs)
        resolved_url = str(base_url)
        return _attach_provider_authorization(
            client,
            ProviderAuthorization.create(
                provider=normalized_provider,
                model=model,
                base_url=resolved_url,
                destination="local" if loopback else "external",
                authorization_mode=authorization_mode,
            ),
        )

    raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, normalized_provider)
