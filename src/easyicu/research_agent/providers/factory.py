"""Single provider contract shared by MCP, discovery, and benchmark entrypoints.

The factory deliberately owns credential selection.  In particular, a local
OpenAI-compatible endpoint receives a fixed non-secret credential even when
real OpenAI or OpenRouter keys are present in the process environment.
"""

from __future__ import annotations

import hashlib
import inspect
import ipaddress
import json
import math
import os
import sys
import threading
import weakref
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.parse import urlsplit

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
MISSING_PROVIDER_KEY = "missing_provider_key"
MISSING_PROVIDER_BASE_URL = "missing_provider_base_url"
INVALID_OPENAI_BASE_URL_OVERRIDE = "invalid_openai_base_url_override"
INVALID_PROVIDER_BASE_URL_OVERRIDE = "invalid_provider_base_url_override"
OPENROUTER_BASE_URL_OVERRIDE = "openrouter_base_url_override"
UNSUPPORTED_PROVIDER = "unsupported_provider"
EXTERNAL_LLM_NOT_AUTHORIZED = "external_llm_not_authorized"
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


@dataclass(frozen=True)
class _CallableContract:
    client_type: type[Any] | None = None
    complete_impl: object | None = None
    complete_with_usage_impl: object | None = None
    complete_with_images_impl: object | None = None
    rebuild_impl: object | None = None
    getattribute_impl: object | None = None
    complete_code: object | None = None
    complete_with_usage_code: object | None = None
    complete_with_images_code: object | None = None
    rebuild_code: object | None = None


@dataclass(frozen=True)
class _TrustedClientRecord:
    kind: str
    authorization: Optional[ProviderAuthorization]
    child_ids: tuple[int, ...] = ()
    children_getter: Optional[Callable[[], Sequence[Any]]] = None
    constructor_impl: object | None = None
    callable_contract: _CallableContract | None = None
    dispatch_identity: tuple[Any, ...] = ()


@dataclass(frozen=True)
class _ConstructedClientRecord:
    client_type: type[Any]
    constructor_impl: object
    callable_contract: _CallableContract
    dispatch_identity: tuple[Any, ...]


_TRUSTED_CLIENTS: dict[
    int, tuple[weakref.ReferenceType[Any], _TrustedClientRecord]
] = {}
_TRUSTED_CLIENTS_LOCK = threading.RLock()
_CONSTRUCTED_CLIENTS: dict[
    int, tuple[weakref.ReferenceType[Any], _ConstructedClientRecord]
] = {}


def _class_callable(client_type: type[Any], name: str) -> object | None:
    value = inspect.getattr_static(client_type, name, None)
    return value if callable(value) else None


def _callable_code(value: object | None) -> object | None:
    if isinstance(value, (classmethod, staticmethod)):
        value = value.__func__
    return getattr(value, "__code__", None)


def _safe_instance_vars(client: Any) -> Mapping[str, Any]:
    try:
        value = object.__getattribute__(client, "__dict__")
    except (AttributeError, TypeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _reviewed_dispatch_identity(client: Any) -> tuple[Any, ...]:
    """Bind the concrete transport object used by a reviewed adapter."""

    if _is_reviewed_client_type(client, "OpenAIClient"):
        instance_vars = _safe_instance_vars(client)
        local_noauth = bool(instance_vars.get("_local_noauth_mode", False))
        attribute = "_local_http_client" if local_noauth else "_client"
        transport = instance_vars.get(attribute)
        if transport is None:
            raise ProviderConfigurationError(
                EXTERNAL_LLM_NOT_AUTHORIZED,
                "openai",
            )
        extra_body = instance_vars.get("_extra_body", {})
        try:
            extra_body_identity = json.dumps(
                extra_body or {},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            raise ProviderConfigurationError(
                EXTERNAL_LLM_NOT_AUTHORIZED,
                "openai",
            ) from None
        return (
            "openai",
            local_noauth,
            attribute,
            id(transport),
            type(transport),
            extra_body_identity,
            str(instance_vars.get("_completion_token_parameter", "")),
            float(instance_vars.get("_request_timeout", 0.0)),
            bool(instance_vars.get("_stream_enabled", False)),
            bool(instance_vars.get("supports_strict_json_schema", False)),
            int(instance_vars.get("_max_retries", 0)),
            (
                None
                if instance_vars.get("_retryable_http_status_codes") is None
                else tuple(
                    sorted(instance_vars.get("_retryable_http_status_codes") or ())
                )
            ),
        )
    if _is_reviewed_client_type(client, "AnthropicMessagesClient"):
        instance_vars = _safe_instance_vars(client)
        transport = instance_vars.get("_client")
        if transport is None:
            raise ProviderConfigurationError(
                EXTERNAL_LLM_NOT_AUTHORIZED,
                "anthropic",
            )
        return (
            "anthropic",
            id(transport),
            type(transport),
            str(instance_vars.get("_model", "")),
            _canonical_endpoint(instance_vars.get("_resolved_base_url", "")),
            float(instance_vars.get("_request_timeout", 0.0)),
            bool(instance_vars.get("_stream_enabled", False)),
            bool(instance_vars.get("supports_strict_json_schema", False)),
            int(instance_vars.get("_max_retries", 0)),
            (
                None
                if instance_vars.get("_retryable_http_status_codes") is None
                else tuple(
                    sorted(instance_vars.get("_retryable_http_status_codes") or ())
                )
            ),
        )
    if _is_reviewed_client_type(client, "CLIAgentLLMClient"):
        instance_vars = _safe_instance_vars(client)
        return (
            "cli",
            str(instance_vars.get("_backend", "")),
            str(instance_vars.get("_command", "")),
            str(instance_vars.get("_model", "")),
            float(instance_vars.get("_timeout", 0.0)),
            bool(instance_vars.get("supports_strict_json_schema", False)),
            str(instance_vars.get("_subprocess_environment_sha256", "")),
        )
    if _is_reviewed_client_type(client, "CodexAppServerLLMClient"):
        instance_vars = _safe_instance_vars(client)
        return (
            "codex-app-server",
            str(instance_vars.get("_model", "")),
            float(instance_vars.get("_timeout", 0.0)),
            str(instance_vars.get("_endpoint_identity", "")),
            str(instance_vars.get("_session_binding_sha256", "")),
            str(instance_vars.get("_subprocess_environment_sha256", "")),
        )
    return ()


def _callable_contract(client: Any) -> _CallableContract:
    client_type = type(client)
    instance_vars = _safe_instance_vars(client)
    protected_names = (
        "complete",
        "complete_with_usage",
        "complete_with_images",
        "_rebuild_openai_client",
    )
    if any(name in instance_vars for name in protected_names):
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            client_type.__name__,
        )
    complete_impl = _class_callable(client_type, "complete")
    if complete_impl is None:
        raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, client_type.__name__)
    getattribute_impl = inspect.getattr_static(client_type, "__getattribute__", None)
    if getattribute_impl is not object.__getattribute__:
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            client_type.__name__,
        )
    usage_impl = _class_callable(client_type, "complete_with_usage")
    image_impl = _class_callable(client_type, "complete_with_images")
    rebuild_impl = _class_callable(client_type, "_rebuild_openai_client")
    return _CallableContract(
        client_type=client_type,
        complete_impl=complete_impl,
        complete_with_usage_impl=usage_impl,
        complete_with_images_impl=image_impl,
        rebuild_impl=rebuild_impl,
        getattribute_impl=getattribute_impl,
        complete_code=_callable_code(complete_impl),
        complete_with_usage_code=_callable_code(usage_impl),
        complete_with_images_code=_callable_code(image_impl),
        rebuild_code=_callable_code(rebuild_impl),
    )


def _callable_contracts_match(
    live: _CallableContract,
    recorded: _CallableContract | None,
) -> bool:
    """Compare every dispatch callable and code object by identity."""

    if recorded is None:
        return False
    return all(
        getattr(live, field) is getattr(recorded, field)
        for field in _CallableContract.__dataclass_fields__
    )


def _caller_is_exact_constructor(client: Any, *, skip: int = 0) -> bool:
    constructor = inspect.getattr_static(type(client), "__init__", None)
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    for _ in range(skip):
        caller = caller.f_back if caller is not None else None
    return bool(
        callable(constructor)
        and caller is not None
        and getattr(constructor, "__code__", None) is caller.f_code
    )


def _mark_reviewed_transport_constructed(client: Any) -> Any:
    """Record one exact reviewed adapter only from its real constructor."""

    if not (
        _is_reviewed_client_type(client, "OpenAIClient")
        or _is_reviewed_client_type(client, "AnthropicMessagesClient")
        or _is_reviewed_client_type(client, "CLIAgentLLMClient")
        or _is_reviewed_client_type(client, "CodexAppServerLLMClient")
    ) or not _caller_is_exact_constructor(client, skip=1):
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            type(client).__name__,
        )
    contract = _callable_contract(client)
    client_type = type(client)
    constructor_impl = inspect.getattr_static(client_type, "__init__")
    ident = id(client)
    reference = weakref.ref(
        client,
        lambda _ref, key=ident, registry=_CONSTRUCTED_CLIENTS: registry.pop(key, None),
    )
    with _TRUSTED_CLIENTS_LOCK:
        _CONSTRUCTED_CLIENTS[ident] = (
            reference,
            _ConstructedClientRecord(
                client_type=client_type,
                constructor_impl=constructor_impl,
                callable_contract=contract,
                dispatch_identity=_reviewed_dispatch_identity(client),
            ),
        )
    return client


def _constructed_client_record(client: Any) -> Optional[_ConstructedClientRecord]:
    with _TRUSTED_CLIENTS_LOCK:
        stored = _CONSTRUCTED_CLIENTS.get(id(client))
    if stored is None or stored[0]() is not client:
        return None
    return stored[1]


def _new_trusted_record(
    client: Any,
    *,
    kind: str,
    authorization: Optional[ProviderAuthorization],
    child_ids: tuple[int, ...] = (),
    children_getter: Optional[Callable[[], Sequence[Any]]] = None,
    construction: Optional[_ConstructedClientRecord] = None,
) -> _TrustedClientRecord:
    contract = _callable_contract(client)
    client_type = type(client)
    return _TrustedClientRecord(
        kind=kind,
        authorization=authorization,
        child_ids=child_ids,
        children_getter=children_getter,
        constructor_impl=(
            construction.constructor_impl
            if construction
            else inspect.getattr_static(client_type, "__init__", None)
        ),
        callable_contract=contract,
        dispatch_identity=(
            construction.dispatch_identity if construction is not None else ()
        ),
    )


def _callables_match_record(client: Any, record: _TrustedClientRecord) -> bool:
    try:
        contract = _callable_contract(client)
    except ProviderConfigurationError:
        return False
    return bool(
        inspect.getattr_static(type(client), "__init__", None)
        is record.constructor_impl
        and _callable_contracts_match(contract, record.callable_contract)
    )


def _dispatch_matches_record(client: Any, record: _TrustedClientRecord) -> bool:
    if not record.dispatch_identity:
        return True
    try:
        return _reviewed_dispatch_identity(client) == record.dispatch_identity
    except ProviderConfigurationError:
        return False


def _refresh_reviewed_transport_dispatch(client: Any) -> None:
    """Rotate a bound OpenAI transport only from its reviewed rebuild method."""

    ident = id(client)
    with _TRUSTED_CLIENTS_LOCK:
        constructed = _CONSTRUCTED_CLIENTS.get(ident)
        trusted = _TRUSTED_CLIENTS.get(ident)
    if constructed is None or constructed[0]() is not client:
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            type(client).__name__,
        )
    construction = constructed[1]
    try:
        live_contract = _callable_contract(client)
    except ProviderConfigurationError as exc:
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            type(client).__name__,
        ) from exc
    caller = inspect.currentframe().f_back
    if (
        not _is_reviewed_client_type(client, "OpenAIClient")
        or not _callable_contracts_match(
            live_contract,
            construction.callable_contract,
        )
        or live_contract.rebuild_impl is None
        or live_contract.rebuild_code is None
        or caller is None
        or construction.callable_contract.rebuild_impl is not live_contract.rebuild_impl
        or construction.callable_contract.rebuild_code is not live_contract.rebuild_code
        or caller.f_code is not construction.callable_contract.rebuild_code
    ):
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            type(client).__name__,
        )
    dispatch_identity = _reviewed_dispatch_identity(client)
    with _TRUSTED_CLIENTS_LOCK:
        current_constructed = _CONSTRUCTED_CLIENTS.get(ident)
        current_trusted = _TRUSTED_CLIENTS.get(ident)
        if (
            current_constructed is None
            or current_constructed != constructed
            or current_trusted != trusted
        ):
            raise ProviderConfigurationError(
                EXTERNAL_LLM_NOT_AUTHORIZED,
                type(client).__name__,
            )
        _CONSTRUCTED_CLIENTS[ident] = (
            constructed[0],
            replace(construction, dispatch_identity=dispatch_identity),
        )
        if trusted is not None and trusted[0]() is client:
            _TRUSTED_CLIENTS[ident] = (
                trusted[0],
                replace(trusted[1], dispatch_identity=dispatch_identity),
            )


def _remember_trusted_client(client: Any, record: _TrustedClientRecord) -> None:
    ident = id(client)
    try:
        reference = weakref.ref(
            client,
            lambda _ref, key=ident, registry=_TRUSTED_CLIENTS: registry.pop(key, None),
        )
    except TypeError as exc:
        raise ProviderConfigurationError(
            UNSUPPORTED_PROVIDER,
            type(client).__name__,
        ) from exc
    with _TRUSTED_CLIENTS_LOCK:
        _TRUSTED_CLIENTS[ident] = (reference, record)


def _trusted_client_record(client: Any) -> Optional[_TrustedClientRecord]:
    with _TRUSTED_CLIENTS_LOCK:
        stored = _TRUSTED_CLIENTS.get(id(client))
    if stored is None or stored[0]() is not client:
        return None
    return stored[1]


def register_offline_test_client(client: Any) -> Any:
    """Register one exact built-in, no-transport mock type.

    Arbitrary objects cannot turn themselves into trusted transports by calling
    this function.  The reviewed mock classes live in ``providers.mocks`` and
    are compared by class identity, not by a marker or spoofable module/name.
    """

    module = sys.modules.get("easyicu.research_agent.providers.mocks")
    allowed = tuple(
        candidate
        for candidate in (
            getattr(module, "MockLLMClient", None) if module else None,
            getattr(module, "ScriptedMockLLMClient", None) if module else None,
            getattr(module, "ScriptedVisionMockLLMClient", None) if module else None,
            (
                getattr(module, "BudgetAwareScriptedMockLLMClient", None)
                if module
                else None
            ),
            (getattr(module, "PatternScriptedMockLLMClient", None) if module else None),
        )
        if isinstance(candidate, type)
    )
    if (
        not allowed
        or type(client) not in allowed
        or not _caller_is_exact_constructor(client, skip=1)
    ):
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            type(client).__name__,
        )

    _remember_trusted_client(
        client,
        _new_trusted_record(client, kind="offline", authorization=None),
    )
    return client


def _register_external_capture_test_client(client: Any) -> Any:
    """Register the exact built-in non-network external capture mock."""

    module = sys.modules.get("easyicu.research_agent.providers.mocks")
    capture_type = (
        getattr(module, "ExternalCaptureMockLLMClient", None) if module else None
    )
    if (
        not isinstance(capture_type, type)
        or type(client) is not capture_type
        or not _caller_is_exact_constructor(client, skip=1)
    ):
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            type(client).__name__,
        )
    _remember_trusted_client(
        client,
        _new_trusted_record(client, kind="external_capture", authorization=None),
    )
    return client


def provider_client_is_offline(client: Any) -> bool:
    """Return whether *client* has explicit identity-bound offline trust."""

    record = _trusted_client_record(client)
    return bool(record is not None and record.kind == "offline")


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


def _register_loopback_provider_client(
    client: Any,
    *,
    model: str,
    base_url: str,
) -> Any:
    if not is_loopback_openai_base_url(base_url):
        raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, "openai")
    return _attach_provider_authorization(
        client,
        ProviderAuthorization.create(
            provider="openai",
            model=str(model),
            base_url=str(base_url),
            destination="local",
            authorization_mode="local_exempt",
        ),
    )


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
    """Attach provenance only to one reviewed transport implementation.

    ``build_provider_client`` accepts a constructor parameter so entrypoint
    tests can inspect the non-secret configuration passed to an adapter.  That
    testing seam must not turn an arbitrary constructor into an authorized
    transport.  Unknown instances are therefore returned unmodified and will
    be rejected by :func:`require_provider_client_authorization` before any
    prompt delivery.
    """

    if client is None or isinstance(client, Mapping):
        return client
    if not (
        _is_reviewed_client_type(client, "OpenAIClient")
        or _is_reviewed_client_type(client, "AnthropicMessagesClient")
        or _is_reviewed_client_type(client, "CLIAgentLLMClient")
        or _is_reviewed_client_type(client, "CodexAppServerLLMClient")
    ):
        return client

    construction = _constructed_client_record(client)
    if construction is None or construction.client_type is not type(client):
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            authorization.provider,
        )
    try:
        live_contract = _callable_contract(client)
    except ProviderConfigurationError:
        live_contract = None
    if not (
        live_contract is not None
        and inspect.getattr_static(type(client), "__init__", None)
        is construction.constructor_impl
        and _callable_contracts_match(
            live_contract,
            construction.callable_contract,
        )
        and _reviewed_dispatch_identity(client) == construction.dispatch_identity
    ):
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            authorization.provider,
        )
    try:
        setattr(client, "__easyicu_provider_authorization__", authorization)
    except Exception as exc:  # pragma: no cover - custom-client boundary
        raise ProviderConfigurationError(
            UNSUPPORTED_PROVIDER,
            authorization.provider,
        ) from exc
    _remember_trusted_client(
        client,
        _new_trusted_record(
            client,
            kind="transport",
            authorization=authorization,
            construction=construction,
        ),
    )
    return client


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


def _valid_provider_authorization(value: object) -> bool:
    if not isinstance(value, ProviderAuthorization):
        return False
    if value.destination == "external" and value.authorization_mode not in {
        "operator_env",
        "account_session",
    }:
        return False
    if value.destination == "local" and value.authorization_mode not in {
        "local_exempt",
        "local_x_api_key",
    }:
        return False
    expected = ProviderAuthorization.create(
        provider=value.provider,
        model=value.model,
        base_url=value.base_url,
        destination=value.destination,
        authorization_mode=value.authorization_mode,
    )
    return value == expected


def _is_reviewed_client_type(client: Any, name: str) -> bool:
    module = sys.modules.get("easyicu.research_agent.providers.llm")
    reviewed = getattr(module, name, None) if module else None
    return isinstance(reviewed, type) and type(client) is reviewed


def _canonical_endpoint(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        return ""
    if (
        scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return ""
    default_port = (scheme == "https" and port in {None, 443}) or (
        scheme == "http" and port in {None, 80}
    )
    authority = hostname if default_port else f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or ""
    return f"{scheme}://{authority}{path}"


def _transport_matches_authorization(
    client: Any,
    authorization: ProviderAuthorization,
) -> bool:
    if _is_reviewed_client_type(client, "OpenAIClient"):
        live_base_url = str(getattr(client, "_resolved_base_url", None) or "")
        live_model = str(getattr(client, "_model", "") or "")
        if live_model != authorization.model:
            return False
        if _canonical_endpoint(live_base_url) != _canonical_endpoint(
            authorization.base_url
        ):
            return False
        is_loopback = is_loopback_openai_base_url(live_base_url)
        if (authorization.destination == "local") != is_loopback:
            return False
        live_auth_header = str(
            getattr(client, "_provider_auth_header_mode", "authorization")
            or "authorization"
        )
        expected_auth_header = (
            "x-api-key"
            if authorization.authorization_mode == "local_x_api_key"
            else "authorization"
        )
        return live_auth_header == expected_auth_header
    if _is_reviewed_client_type(client, "AnthropicMessagesClient"):
        live_base_url = str(getattr(client, "_resolved_base_url", None) or "")
        live_model = str(getattr(client, "_model", "") or "")
        profile_definition = provider_profile(authorization.provider)
        return (
            profile_definition is not None
            and profile_definition.transport == ANTHROPIC_MESSAGES
            and live_model == authorization.model
            and _canonical_endpoint(live_base_url)
            == _canonical_endpoint(authorization.base_url)
            and authorization.destination == "external"
            and authorization.authorization_mode == "operator_env"
        )
    if _is_reviewed_client_type(client, "CLIAgentLLMClient"):
        backend = str(getattr(client, "_backend", "") or "")
        profile_definition = cli_account_profile(backend)
        live_model = str(getattr(client, "_model", "") or "") or "cli-default"
        return (
            profile_definition is not None
            and authorization.provider == profile_definition.provider_identity
            and authorization.base_url == profile_definition.endpoint_identity
            and authorization.model == live_model
            and authorization.destination == "external"
            and authorization.authorization_mode == "account_session"
        )
    if _is_reviewed_client_type(client, "CodexAppServerLLMClient"):
        profile_definition = user_account_profile("codex")
        live_model = str(getattr(client, "_model", "") or "") or "account-default"
        live_endpoint = str(getattr(client, "_endpoint_identity", "") or "")
        session_binding = str(
            getattr(client, "_session_binding_sha256", "") or ""
        )
        return (
            profile_definition is not None
            and authorization.provider == profile_definition.provider_identity
            and authorization.base_url == live_endpoint
            and authorization.model == live_model
            and authorization.destination == "external"
            and authorization.authorization_mode == "account_session"
            and bool(session_binding)
            and live_endpoint.endswith("/session/" + session_binding)
        )
    return False


def require_provider_client_authorization(client: Any) -> None:
    """Deny unmanaged provider graphs before any prompt reaches ``complete``.

    Routers, fallback clients, and transparent cost/repro wrappers are walked
    recursively. Every transport leaf must be an explicit offline mock or carry
    factory-minted, internally consistent destination authority.
    """

    stack = [client]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        record = _trusted_client_record(current)
        if record is None:
            raise ProviderConfigurationError(
                EXTERNAL_LLM_NOT_AUTHORIZED,
                str(getattr(current, "name", type(current).__name__)),
            )
        if not _callables_match_record(current, record) or not _dispatch_matches_record(
            current, record
        ):
            raise ProviderConfigurationError(
                EXTERNAL_LLM_NOT_AUTHORIZED,
                type(current).__name__,
            )
        if record.kind in {"offline", "external_capture"}:
            continue
        if (
            record.kind == "transport"
            and _valid_provider_authorization(record.authorization)
            and _transport_matches_authorization(current, record.authorization)
        ):
            continue
        if record.kind == "wrapper" and record.children_getter is not None:
            try:
                children = tuple(record.children_getter())
            except Exception as exc:
                raise ProviderConfigurationError(
                    UNSUPPORTED_PROVIDER,
                    type(current).__name__,
                ) from exc
            if tuple(id(child) for child in children) != record.child_ids:
                raise ProviderConfigurationError(
                    EXTERNAL_LLM_NOT_AUTHORIZED,
                    type(current).__name__,
                )
            stack.extend(children)
            continue
        raise ProviderConfigurationError(
            EXTERNAL_LLM_NOT_AUTHORIZED,
            str(getattr(current, "name", type(current).__name__)),
        )


def authorized_complete(client: Any, messages: Any, **kwargs: Any) -> str:
    """Deliver a prompt only after the entire provider graph is authorized."""

    require_provider_client_authorization(client)
    complete = object.__getattribute__(client, "complete")
    return complete(messages, **kwargs)


def authorized_complete_with_images(client: Any, **kwargs: Any) -> str:
    """Deliver image prompts only after provider-graph authorization."""

    require_provider_client_authorization(client)
    complete_with_images = object.__getattribute__(client, "complete_with_images")
    return complete_with_images(**kwargs)


def provider_transport_destination(client: Any) -> str:
    """Classify provider transport without reading credentials.

    Direct non-loopback :class:`OpenAIClient` instances remain external even
    before all legacy constructors have migrated into this factory.
    """

    record = _trusted_client_record(client)
    if record is not None and record.kind == "offline":
        return "mock"
    if record is not None and record.kind == "external_capture":
        return "external"
    if record is not None and record.kind == "wrapper" and record.children_getter:
        destinations = {
            provider_transport_destination(child) for child in record.children_getter()
        }
        if "external" in destinations:
            return "external"
        if destinations == {"mock"}:
            return "mock"
        if destinations and destinations <= {"mock", "local"}:
            return "local"
    if record is not None and _valid_provider_authorization(record.authorization):
        assert record.authorization is not None
        return record.authorization.destination
    if bool(getattr(client, "__easyicu_openai_transport__", False)):
        base_url = getattr(client, "_resolved_base_url", None)
        return "local" if is_loopback_openai_base_url(base_url) else "external"
    if bool(getattr(client, "__easyicu_anthropic_transport__", False)):
        return "external"
    # Unknown adapters are never trusted merely because they are in-process:
    # they may forward requests to an arbitrary remote service. Treat them as
    # external so repair prompts receive the structured outbound envelope, and
    # keep their authorization mode unmanaged below. A truly local transport
    # must be factory-attested (for example, a loopback OpenAI endpoint).
    return "external"


def _configured_transport_policy(
    *,
    request_timeout: float,
    transport_max_attempts: int,
    retryable_http_status_codes: Optional[Sequence[int]],
    stream_enabled: bool,
    supports_strict_json_schema: bool,
    transport: str = "openai_compatible",
) -> dict[str, Any]:
    timeout = float(request_timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("request_timeout must be finite and positive")
    if not isinstance(transport_max_attempts, int) or isinstance(
        transport_max_attempts, bool
    ):
        raise ValueError("transport_max_attempts must be a positive integer")
    total_attempts = transport_max_attempts
    if total_attempts <= 0:
        raise ValueError("transport_max_attempts must be a positive integer")
    statuses: Optional[list[int]] = None
    if retryable_http_status_codes is not None:
        statuses = []
        for raw_status in retryable_http_status_codes:
            if not isinstance(raw_status, int) or isinstance(raw_status, bool):
                raise ValueError("retryable HTTP statuses must be integers")
            status = raw_status
            if status < 100 or status > 599:
                raise ValueError("retryable HTTP statuses must be in 100..599")
            statuses.append(status)
        statuses = sorted(set(statuses))
    return {
        "schema_version": "easyicu.provider_transport_policy/2",
        "transport": str(transport),
        "request_timeout_seconds": timeout,
        "transport_max_attempts": total_attempts,
        "retryable_http_status_codes": statuses,
        "stream_enabled": bool(stream_enabled),
        "strict_json_schema_enabled": bool(supports_strict_json_schema),
    }


def _configured_cli_transport_policy(
    *,
    request_timeout: float,
    supports_strict_json_schema: bool,
) -> dict[str, Any]:
    timeout = float(request_timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("request_timeout must be finite and positive")
    return {
        "schema_version": "easyicu.provider_transport_policy/2",
        "transport": "cli_account",
        "request_timeout_seconds": timeout,
        "transport_max_attempts": 1,
        "retryable_http_status_codes": None,
        "stream_enabled": False,
        "strict_json_schema_enabled": bool(supports_strict_json_schema),
    }


def _provider_transport_policy(
    client: Any,
    *,
    record: Optional[_TrustedClientRecord],
) -> dict[str, Any]:
    if record is not None and record.kind == "offline":
        return {
            "schema_version": "easyicu.provider_transport_policy/2",
            "transport": "offline",
            "request_timeout_seconds": None,
            "transport_max_attempts": 0,
            "retryable_http_status_codes": None,
            "stream_enabled": False,
            "strict_json_schema_enabled": False,
        }
    if _is_reviewed_client_type(client, "OpenAIClient"):
        instance_vars = _safe_instance_vars(client)
        return _configured_transport_policy(
            request_timeout=float(instance_vars.get("_request_timeout", 0.0)),
            transport_max_attempts=(
                1 + max(0, int(instance_vars.get("_max_retries", 0)))
            ),
            retryable_http_status_codes=instance_vars.get(
                "_retryable_http_status_codes"
            ),
            stream_enabled=bool(instance_vars.get("_stream_enabled", False)),
            supports_strict_json_schema=bool(
                instance_vars.get("supports_strict_json_schema", False)
            ),
        )
    if _is_reviewed_client_type(client, "AnthropicMessagesClient"):
        instance_vars = _safe_instance_vars(client)
        return _configured_transport_policy(
            request_timeout=float(instance_vars.get("_request_timeout", 0.0)),
            transport_max_attempts=(
                1 + max(0, int(instance_vars.get("_max_retries", 0)))
            ),
            retryable_http_status_codes=instance_vars.get(
                "_retryable_http_status_codes"
            ),
            stream_enabled=False,
            supports_strict_json_schema=bool(
                instance_vars.get("supports_strict_json_schema", False)
            ),
            transport="anthropic_messages",
        )
    if _is_reviewed_client_type(client, "CLIAgentLLMClient"):
        instance_vars = _safe_instance_vars(client)
        return _configured_cli_transport_policy(
            request_timeout=float(instance_vars.get("_timeout", 0.0)),
            supports_strict_json_schema=bool(
                instance_vars.get("supports_strict_json_schema", False)
            ),
        )
    return {
        "schema_version": "easyicu.provider_transport_policy/2",
        "transport": "unmanaged",
        "request_timeout_seconds": None,
        "transport_max_attempts": None,
        "retryable_http_status_codes": None,
        "stream_enabled": None,
        "strict_json_schema_enabled": None,
    }


def provider_authorization_manifest(client: Any) -> dict[str, Any]:
    """Return exact non-secret endpoint authorization for run provenance."""

    reasoning_effort_profile = str(
        getattr(client, "_reasoning_effort_profile", "provider_default")
        or "provider_default"
    )
    clients: list[Any] = []
    stack = [client]
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        record = _trusted_client_record(item)
        if record is not None and record.kind == "wrapper" and record.children_getter:
            stack.extend(record.children_getter())
        else:
            clients.append(item)
    records: list[dict[str, Any]] = []
    for item in clients:
        record = _trusted_client_record(item)
        authorization = record.authorization if record is not None else None
        transport_policy = _provider_transport_policy(item, record=record)
        if isinstance(authorization, ProviderAuthorization):
            records.append(
                {
                    "provider": authorization.provider,
                    "model": authorization.model,
                    "base_url": authorization.base_url,
                    "destination": authorization.destination,
                    "authorization_mode": authorization.authorization_mode,
                    "authorization_sha256": authorization.authorization_sha256,
                    "transport_policy": transport_policy,
                }
            )
            continue
        if record is not None and record.kind == "offline":
            records.append(
                {
                    "provider": "mock",
                    "model": "mock",
                    "base_url": "",
                    "destination": "mock",
                    "authorization_mode": "mock_exempt",
                    "authorization_sha256": "",
                    "transport_policy": transport_policy,
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
                "transport_policy": transport_policy,
            }
        )
    unique = {
        json.dumps(record, sort_keys=True, separators=(",", ":")): record
        for record in records
    }
    return {
        "schema_version": "easyicu.provider_authorization_manifest/3",
        "reasoning_effort_profile": reasoning_effort_profile,
        "clients": [unique[key] for key in sorted(unique)],
    }


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


class ProviderConfigurationError(ValueError):
    """A provider setup failure that adapters can render in their own shape."""

    def __init__(self, issue: str, provider: str) -> None:
        self.issue = issue
        self.provider = provider
        if issue == MISSING_OPENAI_KEY:
            message = "OPENAI_API_KEY is required for --provider openai"
        elif issue == MISSING_OPENROUTER_KEY:
            message = "OPENROUTER_API_KEY is required for --provider openrouter"
        elif issue == MISSING_PROVIDER_KEY:
            profile_definition = provider_profile(provider)
            names = (
                profile_definition.api_key_env_names
                if profile_definition is not None
                else ()
            )
            expected = " or ".join(names) or "a provider API key"
            message = f"{expected} is required for --provider {provider}"
        elif issue == MISSING_PROVIDER_BASE_URL:
            profile_definition = provider_profile(provider)
            names = (
                profile_definition.base_url_env_names
                if profile_definition is not None
                else ()
            )
            expected = " or ".join(names) or "a provider base URL"
            message = f"{expected} is required for --provider {provider}"
        elif issue == INVALID_OPENAI_BASE_URL_OVERRIDE:
            message = (
                "a per-call base_url is permitted only for a parsed loopback "
                "HTTP(S) endpoint"
            )
        elif issue == INVALID_PROVIDER_BASE_URL_OVERRIDE:
            message = (
                "a per-call base_url is permitted only for a parsed loopback "
                f"HTTP(S) endpoint for provider={provider}"
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
    if parsed.username is not None or parsed.password is not None:
        return False
    if parsed.query or parsed.fragment:
        return False
    if parsed.path not in {"", "/", "/v1", "/v1/"}:
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
            from .llm import AnthropicMessagesClient

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
            from .llm import OpenAIClient

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
            from .llm import OpenAIClient

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
