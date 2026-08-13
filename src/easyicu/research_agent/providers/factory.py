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
OPENAI_AUTH_HEADER_NOT_AUTHORIZED = "openai_auth_header_not_authorized"

_BASE_URL_UNSET = object()


def _openai_auth_header(env: Mapping[str, str]) -> OpenAIAuthHeader:
    try:
        return normalize_openai_auth_header(env.get(OPENAI_AUTH_HEADER_ENV))
    except ProviderAuthContractError as exc:
        raise ProviderConfigurationError(
            OPENAI_AUTH_HEADER_NOT_AUTHORIZED,
            "openai",
        ) from exc


def _openai_authorization_mode(
    env: Mapping[str, str],
    *,
    loopback: bool,
    has_override: bool,
    api_key: Optional[str],
) -> tuple[OpenAIAuthHeader, str]:
    """Validate one wire header against the trusted loopback authority."""

    header = _openai_auth_header(env)
    if header is OpenAIAuthHeader.X_API_KEY:
        if (
            not loopback
            or has_override
            or not api_key
            or not _loopback_forwards_real_key(env)
        ):
            raise ProviderConfigurationError(
                OPENAI_AUTH_HEADER_NOT_AUTHORIZED,
                "openai",
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
        )
    if _is_reviewed_client_type(client, "CLIAgentLLMClient"):
        instance_vars = _safe_instance_vars(client)
        return (
            "cli",
            str(instance_vars.get("_backend", "")),
            str(instance_vars.get("_command", "")),
            str(instance_vars.get("_model", "")),
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
        or _is_reviewed_client_type(client, "CLIAgentLLMClient")
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
        or _is_reviewed_client_type(client, "CLIAgentLLMClient")
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
    is_cli = _is_reviewed_client_type(client, "CLIAgentLLMClient")
    if not (is_openai or is_cli):
        raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
    if is_openai:
        if str(provider) != "openai":
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
        live_base_url = str(getattr(client, "_resolved_base_url", None) or "")
        live_model = str(getattr(client, "_model", "") or "")
        if _canonical_endpoint(live_base_url) != _canonical_endpoint(base_url):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
        if live_model != str(model):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
        if (destination == "local") != is_loopback_openai_base_url(live_base_url):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
    else:
        backend = str(getattr(client, "_backend", "") or "")
        live_model = str(getattr(client, "_model", "") or "") or "cli-default"
        if (
            backend not in {"codex", "claude"}
            or str(provider) != f"{backend}-cli"
            or str(base_url) != f"cli://{backend}"
            or str(model) != live_model
        ):
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
        if destination == "local":
            raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
    if destination == "external" and not _external_llm_allowed(env):
        raise ProviderConfigurationError(EXTERNAL_LLM_NOT_AUTHORIZED, provider)
    authorization = ProviderAuthorization.create(
        provider=str(provider),
        model=str(model),
        base_url=str(base_url),
        destination=destination,
        authorization_mode=(
            "operator_env" if destination == "external" else "local_exempt"
        ),
    )
    return _attach_provider_authorization(client, authorization)


def _valid_provider_authorization(value: object) -> bool:
    if not isinstance(value, ProviderAuthorization):
        return False
    if value.destination == "external" and value.authorization_mode != "operator_env":
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
    if _is_reviewed_client_type(client, "CLIAgentLLMClient"):
        backend = str(getattr(client, "_backend", "") or "")
        live_model = str(getattr(client, "_model", "") or "") or "cli-default"
        return (
            backend in {"codex", "claude"}
            and authorization.provider == f"{backend}-cli"
            and authorization.base_url == f"cli://{backend}"
            and authorization.model == live_model
            and authorization.destination == "external"
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
    # Unknown adapters are never trusted merely because they are in-process:
    # they may forward requests to an arbitrary remote service. Treat them as
    # external so repair prompts receive the structured outbound envelope, and
    # keep their authorization mode unmanaged below. A truly local transport
    # must be factory-attested (for example, a loopback OpenAI endpoint).
    return "external"


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
        if record is not None and record.kind == "offline":
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
        "schema_version": "easyicu.provider_authorization_manifest/2",
        "reasoning_effort_profile": reasoning_effort_profile,
        "clients": [unique[key] for key in sorted(unique)],
    }


def provider_authorization_for_configuration(
    *,
    provider: str,
    model: str,
    environment: Optional[Mapping[str, str]] = None,
    reasoning_effort_profile: str = "provider_default",
) -> dict[str, Any]:
    """Mint non-secret identity coordinates without constructing a client."""

    env = os.environ if environment is None else environment
    profile = str(reasoning_effort_profile or "").strip().lower()
    if profile not in {"provider_default", "adaptive_v1"}:
        raise ProviderConfigurationError(UNSUPPORTED_PROVIDER, profile)
    normalized = str(provider or "").strip().lower()
    if normalized == "mock":
        return {
            "schema_version": "easyicu.provider_authorization_manifest/2",
            "reasoning_effort_profile": profile,
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
    authorization_mode = "operator_env"
    if normalized == "openai":
        _header, authorization_mode = _openai_authorization_mode(
            env,
            loopback=loopback,
            has_override=False,
            api_key=env.get("OPENAI_API_KEY"),
        )
    authorization = ProviderAuthorization.create(
        provider=normalized,
        model=model,
        base_url=base_url,
        destination="local" if loopback else "external",
        authorization_mode=authorization_mode,
    )
    return {
        "schema_version": "easyicu.provider_authorization_manifest/2",
        "reasoning_effort_profile": profile,
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
    extra_body: Optional[Mapping[str, Any]] = None,
    max_retries: int = 8,
    stream_enabled: Optional[bool] = None,
    allow_environment_overrides: bool = True,
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
            "max_retries": int(max_retries),
            "stream_enabled": stream_enabled,
            "allow_environment_overrides": bool(allow_environment_overrides),
            "extra_headers": {
                "HTTP-Referer": EASYICU_HTTP_REFERER,
                "X-Title": title,
            },
        }
        provider_extra_body = _openrouter_reasoning_extra_body(model)
        merged_extra_body = dict(provider_extra_body or {})
        merged_extra_body.update(dict(extra_body or {}))
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body
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
        auth_header, authorization_mode = _openai_authorization_mode(
            env,
            loopback=loopback,
            has_override=has_override,
            api_key=api_key,
        )
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
            "max_retries": int(max_retries),
            "stream_enabled": stream_enabled,
            "allow_environment_overrides": bool(allow_environment_overrides),
        }
        if base_url:
            kwargs["base_url"] = base_url
        if extra_body:
            kwargs["extra_body"] = dict(extra_body)
        if auth_header is OpenAIAuthHeader.X_API_KEY:
            assert loopback_key is not None
            kwargs["extra_headers"] = {"x-api-key": loopback_key}
        client = client_cls(**kwargs)
        resolved_url = str(base_url or DEFAULT_OPENAI_BASE_URL)
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
