"""Bind metadata subprocess inputs to the provider adapter's exact client."""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import math
import secrets
import threading
import weakref
from typing import Any, Mapping

_KEY = secrets.token_bytes(32)
_LOCK = threading.RLock()
_CLIENTS: dict[int, tuple[weakref.ReferenceType[Any], bytes, float]] = {}
_WRAPPERS: dict[int, tuple[weakref.ReferenceType[Any], bytes, int, int, int, str, str]] = {}
_PUBLIC_FIELDS = (
    "provider", "model", "credential_fingerprint", "endpoint_fingerprint",
    "credential_header", "base_url_endpoint", "session_binding_sha256",
)


def _mac(provider: Mapping[str, Any], environment: Mapping[str, Any], public: Mapping[str, Any]) -> bytes:
    try:
        data = json.dumps({"provider": dict(provider), "environment": dict(environment),
                           "public": {key: public.get(key) for key in _PUBLIC_FIELDS}},
                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata_provider_binding_invalid") from exc
    return hmac.new(_KEY, data, hashlib.sha256).digest()


def remember_adapter_client(client: Any, *, provider: Mapping[str, Any],
                            environment: Mapping[str, Any] | None, public: Mapping[str, Any]) -> None:
    """Only the adapter that constructed the client can issue this binding."""
    from easyicu.webserver import provider_adapter

    caller = inspect.currentframe().f_back
    if caller is None or caller.f_code is not provider_adapter.build_research_agent_provider_client.__code__:
        raise ValueError("metadata_provider_binding_owner_invalid")
    if environment is None:
        return
    digest = _mac(provider, environment, public)
    timeout = float(public.get("request_timeout_seconds") or 0)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("metadata_provider_binding_invalid")
    with _LOCK:
        _CLIENTS[id(client)] = (weakref.ref(client), digest, timeout)


def bind_wrapper(wrapper: Any, *, timeout: float) -> None:
    """Compare against adapter-issued authority before trusting this wrapper."""
    from easyicu.research_agent.providers.factory import require_provider_client_authorization
    from easyicu.research_agent.providers.client_trust import _trusted_client_record

    inner = wrapper._inner
    require_provider_client_authorization(inner)
    if wrapper._environment.get("EASYICU_DISABLE_PROVIDER_ENV_FILE") != "1":
        raise ValueError("metadata_provider_file_fallback_forbidden")
    record = _trusted_client_record(inner)
    if (record is None or record.kind != "transport" or record.authorization is None
            or record.authorization.destination != "external" or wrapper._provider.get("external") is not True):
        raise ValueError("metadata_provider_binding_authority_invalid")
    digest = _mac(wrapper._provider, wrapper._environment, wrapper._public)
    with _LOCK:
        entry = _CLIENTS.get(id(inner))
        if (entry is None or entry[0]() is not inner or not hmac.compare_digest(entry[1], digest)
                or timeout > entry[2]):
            raise ValueError("metadata_provider_binding_mismatch")
        task = wrapper._task
        ledger = task.ledger
        path, task_id = str(ledger.path), str(task.task_id)
        if not task_id or not path or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("metadata_provider_binding_invalid")
        full = hmac.new(_KEY, digest + repr(float(timeout)).encode(), hashlib.sha256).digest()
        _WRAPPERS[id(wrapper)] = (weakref.ref(wrapper), full, id(inner), id(task), id(ledger), path, task_id)


def verify_wrapper(wrapper: Any) -> dict[str, Any]:
    """Recheck the live dispatch inputs before budget or worker side effects."""
    from easyicu.research_agent.providers.factory import require_provider_client_authorization
    from easyicu.webserver.metadata_provider_concurrency import account_identity

    with _LOCK:
        entry = _WRAPPERS.get(id(wrapper))
    if entry is None or entry[0]() is not wrapper:
        raise ValueError("metadata_provider_binding_missing")
    _reference, expected, inner_id, task_id, ledger_id, path, task_name = entry
    task = wrapper._task
    if (id(wrapper._inner) != inner_id or id(task) != task_id or id(task.ledger) != ledger_id
            or str(task.ledger.path) != path or str(task.task_id) != task_name
            or account_identity(wrapper._public) != wrapper._account
            or wrapper._model != str(wrapper._public.get("model") or "metadata-provider")
            or wrapper.name != "metadata_provider_guarded"
            or wrapper.provider_attempt_budget_aware is not True
            or wrapper.requires_call_scoped_receipt is not True):
        raise ValueError("metadata_provider_binding_mismatch")
    try:
        snapshot = json.loads(json.dumps({"provider": wrapper._provider, "environment": wrapper._environment,
                                          "public": wrapper._public, "timeout": wrapper._timeout},
                                         allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata_provider_binding_invalid") from exc
    if snapshot["environment"].get("EASYICU_DISABLE_PROVIDER_ENV_FILE") != "1":
        raise ValueError("metadata_provider_file_fallback_forbidden")
    digest = _mac(snapshot["provider"], snapshot["environment"], snapshot["public"])
    actual = hmac.new(_KEY, digest + repr(float(snapshot["timeout"])).encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, actual):
        raise ValueError("metadata_provider_binding_mismatch")
    require_provider_client_authorization(wrapper._inner)
    return {**snapshot, "task": task, "account": wrapper._account}
