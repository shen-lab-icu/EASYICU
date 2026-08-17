"""External provider adapter for native FastAPI agent runs.

The adapter is intentionally narrow:

- credentials are read only after provider gate opt-ins pass;
- credentials are never returned in provider metadata or persisted artifacts;
- prompts contain bounded aggregate summaries only, never patient rows;
- model output must be JSON and still goes through STRICT evidence audit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional

if TYPE_CHECKING:
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLimits,
    )

from easyicu.provider_auth import (
    OPENAI_AUTH_HEADER_ENV,
    OpenAIAuthHeader,
    ProviderAuthContractError,
    credential_headers,
    normalize_openai_auth_header,
)
from easyicu.research_agent.providers.capabilities import (
    ANTHROPIC_MESSAGES,
    OPENAI_CHAT_COMPLETIONS,
    SUPPORTED_USER_ACCOUNT_NAMES,
    provider_profile,
    user_account_profile,
)
from easyicu.webserver import state_paths
from easyicu.webserver import agent_outputs
from easyicu.webserver.provider_url_security import (
    ProviderUrlSecurityError,
    validate_credential_endpoint,
)

_MAX_EXTERNAL_CALLS_PER_RUN = 1
_DEFAULT_MAX_OUTPUT_TOKENS = 1200
_MIN_MAX_OUTPUT_TOKENS = 128
_ABSOLUTE_MAX_OUTPUT_TOKENS = 4000
_DEFAULT_PROVIDER_ENV_FILE = state_paths.state_root() / "provider.env"
_DEFAULT_RESEARCH_AGENT_REQUEST_TIMEOUT = 240.0
_DEFAULT_CODEX_APP_SERVER_TURN_HARD_TIMEOUT = 1_800.0
_DEFAULT_CODEX_APP_SERVER_REASONING_EFFORT = "medium"
_LOOPBACK_RESEARCH_AGENT_REQUEST_TIMEOUT = 480.0
_WEB_RESEARCH_AGENT_TRANSIENT_HTTP_STATUS_CODES = (500, 502, 503, 504)
_WEB_RESEARCH_AGENT_MAX_PROVIDER_ATTEMPTS = 192
_WEB_RESEARCH_AGENT_MAX_TOTAL_TOKENS = 2_000_000
_WEB_RESEARCH_AGENT_MAX_ESTIMATED_COST_USD = 100.0
_WEB_RESEARCH_AGENT_MAX_WALL_CLOCK_SECONDS = 21_600.0
_WEB_RESEARCH_AGENT_INPUT_COST_PER_MILLION = 10.0
_WEB_RESEARCH_AGENT_OUTPUT_COST_PER_MILLION = 30.0
# One acquisition request plus the Planner's five bounded structured attempts.
# The local OpenAI-compatible Provider exposes a 128k output ceiling, so each
# transport must reserve that ceiling before it can start even though actual
# completions are much smaller. These aggregate ceilings fund exactly those six
# worst-case reservations; the old 24/250k/$10 tuple advertised attempts that
# its token and cost stops made impossible after two Planner responses.
_WEB_PLANNER_CANARY_MAX_PROVIDER_ATTEMPTS = 6
_WEB_PLANNER_CANARY_MAX_TOTAL_TOKENS = 1_200_000
_WEB_PLANNER_CANARY_MAX_ESTIMATED_COST_USD = 30.0
_WEB_PLANNER_CANARY_MAX_WALL_CLOCK_SECONDS = 1_800.0


class ProviderAdapterError(ValueError):
    """Raised when a connected provider cannot be used safely."""

    def __init__(self, detail: Dict[str, Any]) -> None:
        super().__init__(str(detail.get("error") or "provider_adapter_error"))
        self.detail = detail


def web_research_agent_hard_stop_limits(
    mode: str = "full_reviewed",
) -> "ProviderHardStopLimits":
    """Return the explicit reviewed stop-loss for one Web pipeline launch mode."""

    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLimits,
    )

    selected = str(mode or "").strip().lower()
    if selected == "planner_canary":
        attempts = _WEB_PLANNER_CANARY_MAX_PROVIDER_ATTEMPTS
        tokens = _WEB_PLANNER_CANARY_MAX_TOTAL_TOKENS
        cost = _WEB_PLANNER_CANARY_MAX_ESTIMATED_COST_USD
        wall_clock = _WEB_PLANNER_CANARY_MAX_WALL_CLOCK_SECONDS
    elif selected == "full_reviewed":
        attempts = _WEB_RESEARCH_AGENT_MAX_PROVIDER_ATTEMPTS
        tokens = _WEB_RESEARCH_AGENT_MAX_TOTAL_TOKENS
        cost = _WEB_RESEARCH_AGENT_MAX_ESTIMATED_COST_USD
        wall_clock = _WEB_RESEARCH_AGENT_MAX_WALL_CLOCK_SECONDS
    else:
        raise ValueError(f"Unknown Web Research Agent budget mode: {mode!r}")
    return ProviderHardStopLimits(
        max_provider_attempts_per_run=attempts,
        max_provider_attempts_per_batch=attempts,
        max_total_tokens_per_run=tokens,
        max_total_tokens_per_batch=tokens,
        max_estimated_cost_usd_per_batch=cost,
        max_wall_clock_seconds_per_task=wall_clock,
        input_cost_usd_per_million_tokens=(_WEB_RESEARCH_AGENT_INPUT_COST_PER_MILLION),
        output_cost_usd_per_million_tokens=(
            _WEB_RESEARCH_AGENT_OUTPUT_COST_PER_MILLION
        ),
    )


def require_external_credentials(
    provider_meta: Dict[str, Any],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return sanitized provider metadata after checking env credentials."""
    if not provider_meta.get("external"):
        return provider_meta
    provider = str(provider_meta.get("provider") or "").strip().lower()
    if is_user_account_provider(provider):
        return _require_user_account_session(provider_meta, environ=environ)
    try:
        credentials = _load_external_credentials(
            str(provider_meta.get("provider") or ""), environ=environ
        )
    except ProviderAdapterError as exc:
        raise ProviderAdapterError({**provider_meta, **exc.detail}) from exc
    updated = dict(provider_meta)
    updated.update(_credential_public_metadata(credentials))
    updated["provider_gate"] = "external_provider_credentials_ready"
    updated.setdefault("provider_gate_order", []).append("credentials_loaded")
    return updated


def _user_account_environment(
    provider: str,
    environ: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    if environ is None:
        raise ProviderAdapterError(
            {
                "error": "codex_auth_session_required",
                "provider": provider,
                "secrets_returned": False,
            }
        )
    source = environ
    from easyicu.research_agent.providers.subprocess_env import (
        build_provider_subprocess_env,
    )

    selected = build_provider_subprocess_env(
        provider,
        environment=source,
        required_keys=(
            "EASYICU_ALLOW_EXTERNAL_LLM",
            "EASYICU_CODEX_SESSION_SHA256",
            "EASYICU_CODEX_MODEL",
            "CODEX_MODEL",
        ),
    )
    binding = str(selected.get("EASYICU_CODEX_SESSION_SHA256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", binding):
        raise ProviderAdapterError(
            {
                "error": "codex_auth_session_required",
                "provider": provider,
                "secrets_returned": False,
            }
        )
    if not selected.get("HOME") or not selected.get("CODEX_HOME"):
        raise ProviderAdapterError(
            {
                "error": "codex_auth_isolated_home_required",
                "provider": provider,
                "secrets_returned": False,
            }
        )
    selected["EASYICU_ALLOW_EXTERNAL_LLM"] = "1"
    return selected


def is_user_account_provider(provider: str) -> bool:
    """Return whether *provider* is a reviewed per-user account transport."""

    return str(provider or "").strip().lower() in SUPPORTED_USER_ACCOUNT_NAMES


def is_cli_account_provider(provider: str) -> bool:
    """Compatibility alias; public Web accounts are no longer host CLI logins."""

    return is_user_account_provider(provider)


def account_provider_environment(
    provider: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Compile the bounded subprocess environment for a Web Pipeline run.

    The returned mapping contains only the reviewed user-account variables and
    the canonical external-LLM opt-in bit. API keys for unrelated providers are
    deliberately excluded by :mod:`providers.subprocess_env`.
    """

    normalized = str(provider or "").strip().lower()
    if not is_user_account_provider(normalized):
        raise ProviderAdapterError(
            {
                "error": "research_pipeline_codex_user_auth_provider_required",
                "provider": normalized,
                "secrets_returned": False,
            }
        )
    return _user_account_environment(normalized, environ)


def _user_account_status(
    provider: str,
    *,
    ai_enabled: bool,
    environ: Optional[Mapping[str, str]],
) -> Dict[str, Any]:
    profile = user_account_profile(provider) if is_user_account_provider(provider) else None
    if profile is None:
        raise ProviderAdapterError(
            {
                "error": "research_agent_provider_unsupported",
                "provider": provider,
                "secrets_returned": False,
            }
        )
    if environ is None:
        return {
            "provider": provider,
            "provider_identity": profile.provider_identity,
            "external": True,
            "ai_enabled": bool(ai_enabled),
            "authentication_mode": "chatgpt_account",
            "authentication_verified": False,
            "account_session_present": False,
            "account_session_status": "codex_auth_session_required",
            "credential_env_candidates": [],
            "credential_present": False,
            "credential_source": None,
            "base_url_env_candidates": [],
            "base_url_present": True,
            "base_url_source": "codex_app_server",
            "base_url_validation": "codex_app_server",
            "base_url_rejection_reason": None,
            "model_env_candidates": list(profile.model_env_names),
            "model_present": True,
            "model": "account-default",
            "model_source": "account_default",
            "ready": False,
            "missing": ["codex_user_auth"],
            "env_file": {
                "enabled": False,
                "status": "not_used_user_account",
                "present": False,
                "loaded_keys": [],
                "secrets_returned": False,
            },
            "secrets_returned": False,
            "client_constructed": False,
            "network_calls": 0,
            "subprocess_calls": 0,
        }
    source = _user_account_environment(provider, environ)
    model_source, model = profile.model(source)
    from easyicu.research_agent.providers.codex_app_server import (
        CodexAppServerError,
        CodexAppServerRuntime,
    )

    account = None
    status_code = "codex_auth_login_required"
    try:
        runtime_cwd = Path(str(source["HOME"])) / "app-server-readiness"
        with CodexAppServerRuntime(
            environment=source,
            cwd=runtime_cwd,
            request_timeout=15.0,
        ) as runtime:
            account = runtime.request(
                "account/read", {"refreshToken": False}, timeout=15.0
            ).get("account")
    except CodexAppServerError as exc:
        status_code = exc.code
    verified = bool(isinstance(account, Mapping) and account.get("type") == "chatgpt")
    if verified:
        status_code = "codex_auth_ready"
    missing: List[str] = []
    if not ai_enabled:
        missing.append("ai_enabled")
    if not verified:
        missing.append("codex_user_auth")
    binding = str(source.get("EASYICU_CODEX_SESSION_SHA256") or "")
    return {
        "provider": provider,
        "provider_identity": profile.provider_identity,
        "external": True,
        "ai_enabled": bool(ai_enabled),
        "authentication_mode": "chatgpt_account",
        "authentication_verified": verified,
        "account_session_present": verified,
        "account_session_status": status_code,
        "session_binding_sha256": binding,
        "status_check_supported": True,
        "credential_env_candidates": [],
        "credential_present": verified,
        "credential_source": "codex_user_auth" if verified else None,
        "base_url_env_candidates": [],
        "base_url_present": True,
        "base_url_source": "codex_app_server",
        "base_url_validation": "codex_app_server",
        "base_url_rejection_reason": None,
        "model_env_candidates": list(profile.model_env_names),
        "model_present": True,
        "model": model or "account-default",
        "model_source": model_source or "account_default",
        "ready": bool(ai_enabled and verified),
        "missing": missing,
        "env_file": {
            "enabled": False,
            "status": "not_used_user_account",
            "present": False,
            "loaded_keys": [],
            "secrets_returned": False,
        },
        "secrets_returned": False,
        "client_constructed": False,
        "network_calls": 0,
        "subprocess_calls": 1,
    }


def _require_user_account_session(
    provider_meta: Dict[str, Any],
    *,
    environ: Optional[Mapping[str, str]],
) -> Dict[str, Any]:
    provider = str(provider_meta.get("provider") or "").strip().lower()
    status = _user_account_status(
        provider,
        ai_enabled=bool(provider_meta.get("ai_enabled")),
        environ=environ,
    )
    if not status["ready"]:
        raise ProviderAdapterError(
            {
                **provider_meta,
                **status,
                "error": str(status["account_session_status"]),
                "blocked_by": "codex_user_auth",
            }
        )
    updated = dict(provider_meta)
    updated.update(status)
    updated["credentials_attempted"] = True
    updated["credentials_loaded"] = True
    profile = user_account_profile(provider)
    assert profile is not None  # guarded by _user_account_status
    endpoint = f"{profile.endpoint_identity}/session/{status['session_binding_sha256']}"
    updated["endpoint_fingerprint"] = _fingerprint(endpoint)
    updated["base_url_endpoint"] = "codex_app_server"
    updated["provider_gate"] = "external_provider_account_ready"
    updated.setdefault("provider_gate_order", []).append("user_account_checked")
    return updated


def build_research_agent_provider_client(
    provider_meta: Dict[str, Any],
    *,
    request_timeout: Optional[float] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Construct the governed Research Agent client without exposing a key.

    The native Web provider file and the Research Agent provider factory have
    deliberately separate responsibilities.  This adapter is their only
    bridge: it revalidates the private credential and endpoint, passes them
    directly to the factory in memory, and returns only the live client plus
    non-secret provenance metadata.  Callers must never serialize the client.
    """

    if not provider_meta.get("external"):
        raise ProviderAdapterError(
            {
                "error": "research_agent_external_provider_required",
                "secrets_returned": False,
            }
        )
    requested_provider = str(provider_meta.get("provider") or "").strip().lower()
    account_profile = (
        user_account_profile(requested_provider)
        if is_user_account_provider(requested_provider)
        else None
    )
    if account_profile is not None:
        account = _require_user_account_session(provider_meta, environ=environ)
        account_environment = _user_account_environment(requested_provider, environ)
        selected_model = str(account.get("model") or "account-default")
        effective_timeout = (
            float(request_timeout)
            if request_timeout is not None
            else _DEFAULT_RESEARCH_AGENT_REQUEST_TIMEOUT
        )
        try:
            from easyicu.research_agent.providers.factory import (
                authorize_provider_client,
            )
            from easyicu.research_agent.providers.llm import (
                CodexAppServerLLMClient,
            )

            client = CodexAppServerLLMClient(
                model=(
                    None if selected_model == "account-default" else selected_model
                ),
                request_timeout=effective_timeout,
                turn_hard_timeout=_DEFAULT_CODEX_APP_SERVER_TURN_HARD_TIMEOUT,
                reasoning_effort=_DEFAULT_CODEX_APP_SERVER_REASONING_EFFORT,
                environment=account_environment,
            )
            endpoint = (
                f"{account_profile.endpoint_identity}/session/"
                f"{account['session_binding_sha256']}"
            )
            client = authorize_provider_client(
                client,
                provider=account_profile.provider_identity,
                model=selected_model,
                base_url=endpoint,
                destination="external",
                environment=account_environment,
            )
        except Exception as exc:
            raise ProviderAdapterError(
                {
                    "error": "research_agent_provider_client_failed",
                    "provider": requested_provider,
                    "reason": type(exc).__name__,
                    "secrets_returned": False,
                }
            ) from exc
        public = dict(account)
        public.update(
            {
                "client": (
                    "easyicu.research_agent.providers.llm."
                    "CodexAppServerLLMClient"
                ),
                "client_constructed": True,
                "provider_gate": "research_agent_provider_ready",
                "request_timeout_seconds": effective_timeout,
                "request_idle_timeout_seconds": effective_timeout,
                "request_hard_timeout_seconds": (
                    _DEFAULT_CODEX_APP_SERVER_TURN_HARD_TIMEOUT
                ),
                "reasoning_effort": _DEFAULT_CODEX_APP_SERVER_REASONING_EFFORT,
                "reasoning_effort_source": "easyicu_account_research_default",
                "progress_resets_idle_timeout": True,
                "transport_max_attempts": 1,
                "retryable_http_status_codes": [],
                "strict_json_schema_enabled": bool(
                    account_profile.supports_strict_json_schema
                ),
                "provider_hard_stop": {
                    "required": True,
                    "schema_version": "easyicu.web-provider-hard-stop-policy/1",
                },
                "secrets_returned": False,
            }
        )
        return client, public
    credentials = _load_external_credentials(
        str(provider_meta.get("provider") or ""), environ=environ
    )
    provider = str(credentials.get("provider") or "").strip().lower()
    profile = provider_profile(provider)
    if profile is None or profile.transport not in {
        OPENAI_CHAT_COMPLETIONS,
        ANTHROPIC_MESSAGES,
    }:
        raise ProviderAdapterError(
            {
                "error": "research_agent_provider_unsupported",
                "provider": provider,
                "secrets_returned": False,
            }
        )
    endpoint = str(credentials["base_url"])
    base_url = _provider_sdk_base_url(endpoint, transport=profile.transport)
    key_name = profile.api_key_env_names[0]
    base_name = profile.base_url_env_names[0]
    provider_environment = {
        key_name: credentials["api_key"],
        base_name: base_url,
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
    }
    try:
        from easyicu.research_agent.providers import build_provider_client
        from easyicu.research_agent.providers.factory import (
            TRUST_LOOPBACK_PROXY_KEY_ENV,
            is_loopback_openai_base_url,
        )

        loopback = bool(
            profile.transport == OPENAI_CHAT_COMPLETIONS
            and is_loopback_openai_base_url(base_url)
        )
        if loopback:
            # The endpoint and credential were both loaded from the server-owned
            # provider configuration after explicit Web opt-in. Tell the shared
            # factory this one configured loopback proxy is allowed to receive
            # its own authentication token rather than the no-auth dummy key.
            provider_environment[TRUST_LOOPBACK_PROXY_KEY_ENV] = "1"
        if profile.transport == OPENAI_CHAT_COMPLETIONS:
            provider_environment[OPENAI_AUTH_HEADER_ENV] = str(
                credentials.get("auth_header")
                or OpenAIAuthHeader.AUTHORIZATION.value
            )
        effective_timeout = (
            float(request_timeout)
            if request_timeout is not None
            else (
                _LOOPBACK_RESEARCH_AGENT_REQUEST_TIMEOUT
                if loopback
                else _DEFAULT_RESEARCH_AGENT_REQUEST_TIMEOUT
            )
        )
        client = build_provider_client(
            provider=provider,
            model=credentials["model"],
            request_timeout=effective_timeout,
            title="EasyICU Web Research Agent",
            environment=provider_environment,
            # One extra transport attempt, hence two total requests.  Freeze
            # environment overrides so EASYICU_LLM_MAX_RETRIES cannot silently
            # widen the Web job's reviewed bound.  This allowlist is the whole
            # Web retry policy: non-HTTP failures and other status codes fail
            # closed without a transport replay.
            max_retries=1,
            retryable_http_status_codes=(
                _WEB_RESEARCH_AGENT_TRANSIENT_HTTP_STATUS_CODES
            ),
            allow_environment_overrides=False,
        )
    except Exception as exc:
        raise ProviderAdapterError(
            {
                "error": "research_agent_provider_client_failed",
                "provider": provider,
                "reason": type(exc).__name__,
                "secrets_returned": False,
            }
        ) from exc
    public = dict(provider_meta)
    public.update(_credential_public_metadata(credentials))
    public.update(
        {
            "client": (
                "easyicu.research_agent.AnthropicMessagesClient"
                if profile.transport == ANTHROPIC_MESSAGES
                else "easyicu.research_agent.OpenAIClient"
            ),
            "client_constructed": True,
            "provider_gate": "research_agent_provider_ready",
            "request_timeout_seconds": effective_timeout,
            "transport_max_attempts": 2,
            "strict_json_schema_enabled": bool(
                getattr(client, "supports_strict_json_schema", False)
            ),
            "retryable_http_status_codes": list(
                _WEB_RESEARCH_AGENT_TRANSIENT_HTTP_STATUS_CODES
            ),
            "provider_hard_stop": {
                "required": True,
                "schema_version": "easyicu.web-provider-hard-stop-policy/1",
            },
            "secrets_returned": False,
        }
    )
    return client, public


def generate_bound_provider_payload(
    *,
    provider_meta: Dict[str, Any],
    run_id: str,
    study_id: str,
    question: Optional[str],
    summary: Dict[str, Any],
    cohort: Dict[str, Any],
    quality: List[Dict[str, Any]],
    output_artifacts: Optional[Dict[str, Dict[str, Any]]] = None,
    transport: Optional[
        Callable[[Dict[str, Any], Dict[str, str]], Dict[str, Any]]
    ] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Call one governed API or account provider and return bounded artifacts."""
    requested_provider = str(provider_meta.get("provider") or "").strip().lower()
    profile = provider_profile(requested_provider)
    governed_client_transport = bool(
        is_user_account_provider(requested_provider)
        or (profile is not None and profile.transport == ANTHROPIC_MESSAGES)
    )
    if governed_client_transport:
        if transport is not None:
            raise ProviderAdapterError(
                {"error": "governed_provider_transport_override_unsupported"}
            )
        client, provider_public = build_research_agent_provider_client(
            provider_meta,
            environ=environ,
        )
        max_output_tokens = _max_output_tokens(environ=environ)
        request = _build_chat_request(
            provider=requested_provider,
            run_id=run_id,
            study_id=study_id,
            question=question,
            summary=summary,
            cohort=cohort,
            quality=quality,
            output_artifacts=output_artifacts or {},
            model=str(provider_public.get("model") or "account-default"),
            max_output_tokens=max_output_tokens,
            json_format_style="chat",
        )
        try:
            from easyicu.research_agent.providers.protocol import (
                LLMMessage,
                StructuredOutputRequest,
            )
            from easyicu.research_agent.providers.factory import (
                authorized_complete,
                require_provider_client_authorization,
            )

            messages = [
                LLMMessage(
                    role=str(message.get("role") or "user"),
                    content=str(message.get("content") or ""),
                )
                for message in request["messages"]
            ]
            structured_output = None
            if bool(provider_public.get("strict_json_schema_enabled")):
                structured_output = StructuredOutputRequest.from_schema(
                    name="easyicu_agent_run",
                    schema=_agent_payload_json_schema(
                        request["easyicu_policy"]["allowed_evidence_ids"]
                    ),
                )
            require_provider_client_authorization(client)
            complete_with_usage = getattr(client, "complete_with_usage", None)
            if callable(complete_with_usage):
                text, call_usage = complete_with_usage(
                    messages,
                    max_tokens=max_output_tokens,
                    temperature=0,
                    structured_output=structured_output,
                )
            else:
                text = authorized_complete(
                    client,
                    messages,
                    max_tokens=max_output_tokens,
                    temperature=0,
                    structured_output=structured_output,
                )
                call_usage = {}
        except Exception as exc:
            raise ProviderAdapterError(
                {
                    "error": "external_provider_call_failed",
                    "provider": requested_provider,
                    "reason": type(exc).__name__,
                    "secrets_returned": False,
                }
            ) from exc
        payload = _coerce_provider_payload(
            _parse_json_object(text),
            run_id=run_id,
            study_id=study_id,
            question=question,
        )
        provider_update = dict(provider_public)
        strict_enabled = bool(provider_public.get("strict_json_schema_enabled"))
        if strict_enabled and is_user_account_provider(requested_provider):
            json_transport = "codex_app_server_output_schema"
        elif strict_enabled and profile is not None:
            json_transport = "anthropic_output_config"
        else:
            json_transport = "prompted_json"
        provider_update.update(
            {
                "external_calls": int(provider_meta.get("external_calls") or 0) + 1,
                "max_external_calls_per_run": _MAX_EXTERNAL_CALLS_PER_RUN,
                "max_output_tokens": max_output_tokens,
                "json_format_style": json_transport,
                "provider_gate": "external_provider_ready",
                "provider_gate_order": [
                    *list(provider_public.get("provider_gate_order") or []),
                    "client_constructed",
                    "external_call_completed",
                ],
                "usage": _public_llm_usage(call_usage),
            }
        )
        request["easyicu_policy"]["json_format_style"] = provider_update[
            "json_format_style"
        ]
        return {
            "agent_plan": payload["agent_plan"],
            "manuscript_draft": payload["manuscript_draft"],
            "provider": provider_update,
            "request_policy": request["easyicu_policy"],
        }
    credentials = _load_external_credentials(
        requested_provider, environ=environ
    )
    max_output_tokens = _max_output_tokens(environ=environ)
    json_format_style = _json_format_style(environ=environ)
    request = _build_chat_request(
        provider=str(provider_meta.get("provider") or ""),
        run_id=run_id,
        study_id=study_id,
        question=question,
        summary=summary,
        cohort=cohort,
        quality=quality,
        output_artifacts=output_artifacts or {},
        model=credentials["model"],
        max_output_tokens=max_output_tokens,
        json_format_style=json_format_style,
    )
    auth_header = normalize_openai_auth_header(credentials.get("auth_header"))
    if auth_header is OpenAIAuthHeader.X_API_KEY:
        from easyicu.research_agent.providers.factory import (
            is_loopback_openai_base_url,
        )

        endpoint = str(credentials.get("base_url") or "")
        suffix = "/chat/completions"
        auth_base_url = (
            endpoint[: -len(suffix)] if endpoint.endswith(suffix) else endpoint
        )
        if not is_loopback_openai_base_url(auth_base_url):
            raise ProviderAdapterError(
                {
                    "error": "provider_x_api_key_requires_loopback",
                    "secrets_returned": False,
                }
            )
    headers = {
        **credential_headers(credentials["api_key"], mode=auth_header),
        "Content-Type": "application/json",
    }
    if transport is None:
        response = _post_chat_completion(
            url=credentials["base_url"],
            request=request,
            headers=headers,
            timeout=45,
        )
    else:
        response = transport(request, headers)
    payload = _coerce_provider_payload(
        response, run_id=run_id, study_id=study_id, question=question
    )
    provider_update = dict(provider_meta)
    provider_update.update(_credential_public_metadata(credentials))
    provider_update.update(
        {
            "client": "OpenAICompatibleChat",
            "client_constructed": True,
            "external_calls": int(provider_meta.get("external_calls") or 0) + 1,
            "max_external_calls_per_run": _MAX_EXTERNAL_CALLS_PER_RUN,
            "max_output_tokens": max_output_tokens,
            "json_format_style": json_format_style,
            "provider_gate": "external_provider_ready",
            "provider_gate_order": [
                *list(provider_meta.get("provider_gate_order") or []),
                "client_constructed",
                "external_call_completed",
            ],
            "usage": _public_usage(response),
        }
    )
    return {
        "agent_plan": payload["agent_plan"],
        "manuscript_draft": payload["manuscript_draft"],
        "provider": provider_update,
        "request_policy": request["easyicu_policy"],
    }


def provider_readiness(
    provider: str,
    *,
    ai_enabled: bool = False,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return sanitized provider readiness without constructing clients."""
    provider_text = str(provider or "openai").strip() or "openai"
    if is_user_account_provider(provider_text):
        status = _user_account_status(
            provider_text,
            ai_enabled=ai_enabled,
            environ=environ,
        )
        status["limits"] = {
            "max_external_calls_per_run": _MAX_EXTERNAL_CALLS_PER_RUN,
            "max_output_tokens": _max_output_tokens(environ=environ),
            "json_format_style": (
                "codex_app_server_output_schema"
                if bool(
                    user_account_profile(provider_text)
                    and user_account_profile(provider_text).supports_strict_json_schema
                )
                else "prompted_json"
            ),
        }
        return status
    if not _is_offline_provider(provider_text) and provider_profile(provider_text) is None:
        return {
            "provider": provider_text,
            "external": True,
            "ai_enabled": bool(ai_enabled),
            "ready": False,
            "missing": ["supported_provider"],
            "error": "research_agent_provider_unsupported",
            "secrets_returned": False,
            "client_constructed": False,
            "network_calls": 0,
        }
    env, env_file = _provider_env(environ=environ)
    external = not _is_offline_provider(provider_text)
    key_names = _api_key_env_names(provider_text)
    base_names = _base_url_env_names(provider_text)
    model_names = _model_env_names(provider_text)
    key_name, api_key = _first_env(env, key_names)
    base_name, base_url = _first_env(env, base_names)
    default_base = _default_base_url(provider_text)
    model_name, model = _first_env(env, model_names)
    has_base_url = bool(base_url or default_base)
    base_url_validation = (
        "provider_default" if default_base and not base_url else "missing"
    )
    base_url_rejection_reason: Optional[str] = None
    if base_url:
        try:
            validate_provider_base_url(base_url)
        except ProviderAdapterError as exc:
            base_url_validation = "rejected"
            base_url_rejection_reason = str(exc.detail.get("reason") or "rejected")
        else:
            base_url_validation = "validated"
    base_url_safe = has_base_url and base_url_validation != "rejected"
    missing: List[str] = []
    if external and not ai_enabled:
        missing.append("ai_enabled")
    if external and not api_key:
        missing.append("credential")
    if external and not has_base_url:
        missing.append("base_url")
    if external and has_base_url and not base_url_safe:
        missing.append("base_url_safety")
    if external and not model:
        missing.append("model")
    if external and env_file.get("status") == "insecure_permissions":
        missing.append("env_file_permissions")
    return {
        "provider": provider_text,
        "external": external,
        "ai_enabled": bool(ai_enabled),
        "credential_env_candidates": key_names,
        "credential_present": bool(api_key),
        "credential_source": key_name if api_key else None,
        "base_url_env_candidates": base_names,
        "base_url_present": has_base_url,
        "base_url_source": (
            base_name if base_url else ("provider_default" if default_base else None)
        ),
        "base_url_validation": base_url_validation,
        "base_url_rejection_reason": base_url_rejection_reason,
        "model_env_candidates": model_names,
        "model_present": bool(model),
        "model_source": model_name if model else None,
        "ready": bool(
            (not external)
            or (
                ai_enabled
                and api_key
                and base_url_safe
                and model
                and env_file.get("status") != "insecure_permissions"
            )
        ),
        "missing": missing,
        "env_file": env_file,
        "limits": {
            "max_external_calls_per_run": _MAX_EXTERNAL_CALLS_PER_RUN,
            "max_output_tokens": _max_output_tokens(environ=env),
            "json_format_style": _json_format_style(environ=env),
        },
        "secrets_returned": False,
        "client_constructed": False,
        "network_calls": 0,
    }


def write_provider_config(
    provider: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: str = "",
    json_format_style: str = "",
    force: bool = True,
) -> Dict[str, Any]:
    """Write the private provider env file and return sanitized metadata."""
    provider_text = str(provider or "openai").strip() or "openai"
    if provider_profile(provider_text) is None:
        raise ProviderAdapterError(
            {
                "error": "research_agent_provider_unsupported",
                "provider": provider_text,
                "secrets_returned": False,
            }
        )
    api_key = str(api_key or "").strip()
    base_url = str(base_url or "").strip()
    model = str(model or "").strip()
    if not api_key:
        raise ProviderAdapterError(
            {
                "error": "external_provider_api_key_required",
                "secrets_returned": False,
            }
        )
    if not model:
        raise ProviderAdapterError(
            {
                "error": "external_provider_model_required",
                "secrets_returned": False,
            }
        )
    if not base_url and not _default_base_url(provider_text):
        raise ProviderAdapterError(
            {
                "error": "external_provider_base_url_required",
                "secrets_returned": False,
            }
        )
    entries: Dict[str, str] = {
        _api_key_env_names(provider_text)[0]: api_key,
        _model_env_names(provider_text)[0]: model,
    }
    if base_url:
        validate_provider_base_url(base_url)
        entries[_base_url_env_names(provider_text)[0]] = base_url
    max_tokens = str(max_tokens or "").strip()
    if max_tokens:
        entries["EASYICU_LLM_MAX_TOKENS"] = max_tokens
    json_format_style = str(json_format_style or "").strip().lower()
    if json_format_style in {"chat", "responses", "both"}:
        entries["EASYICU_LLM_JSON_FORMAT_STYLE"] = json_format_style
    path = _DEFAULT_PROVIDER_ENV_FILE
    if path.exists() and not force:
        raise ProviderAdapterError(
            {
                "error": "external_provider_env_file_exists",
                "env_file": str(path),
                "secrets_returned": False,
            }
        )
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    lines = [
        "# EasyICU private external-provider config",
        "# Created by the local EasyICU FastAPI UI.",
        "# Keep this file mode 0600. Do not commit it.",
    ]
    for key, value in entries.items():
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise ProviderAdapterError(
                {
                    "error": "external_provider_invalid_env_key",
                    "env_key": key,
                    "secrets_returned": False,
                }
            )
        lines.append(f"{key}={_quote_env_value(value)}")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    return {
        "ok": True,
        "provider": provider_text,
        "env_file": {
            "enabled": True,
            "status": "written",
            "present": True,
            "configured": "default",
            "mode": "0600",
            "loaded_keys": sorted(entries),
            "secrets_returned": False,
        },
        "secrets_returned": False,
    }


def _load_external_credentials(
    provider: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    provider = _normalize_provider(provider)
    profile = provider_profile(provider)
    if profile is None:
        raise ProviderAdapterError(
            {
                "error": "research_agent_provider_unsupported",
                "provider": provider,
                "secrets_returned": False,
            }
        )
    provider_transport = (
        profile.transport
    )
    env, env_file = _provider_env(environ=environ)
    key_names = _api_key_env_names(provider)
    key_name, api_key = _first_env(env, key_names)
    base_name, base_url = _first_env(env, _base_url_env_names(provider))
    if not base_url:
        base_url = _default_base_url(provider)
    else:
        # Re-checked here, not only where the UI writes it: the env file is an
        # ordinary file, and this is the last point before the key is sent.
        validate_provider_base_url(base_url)
    base_url = _provider_request_url(base_url, transport=provider_transport)
    model_name, model = _first_env(env, _model_env_names(provider))
    if provider_transport == ANTHROPIC_MESSAGES:
        auth_header_value = "x-api-key"
    elif profile is not None and not profile.supports_auth_header_override:
        # Provider-specific profiles own their authentication contract. In
        # particular, a stale local-Luna x-api-key setting must never alter an
        # official DeepSeek request or its public provenance receipt.
        auth_header_value = OpenAIAuthHeader.AUTHORIZATION.value
    else:
        try:
            auth_header_value = normalize_openai_auth_header(
                env.get(OPENAI_AUTH_HEADER_ENV)
            ).value
        except ProviderAuthContractError as exc:
            raise ProviderAdapterError(
                {
                    "error": exc.code,
                    "blocked_by": "external_provider_credentials",
                    "secrets_returned": False,
                }
            ) from exc
    attempted = {
        "credentials_attempted": True,
        "credentials_loaded": False,
        "client_constructed": False,
        "credential_env_candidates": key_names,
        "env_file": env_file,
    }
    if env_file.get("status") == "insecure_permissions":
        raise ProviderAdapterError(
            {
                **attempted,
                "error": "external_provider_env_file_permissions",
                "blocked_by": "external_provider_credentials",
            }
        )
    if not api_key:
        raise ProviderAdapterError(
            {
                **attempted,
                "error": "external_provider_credentials_required",
                "blocked_by": "external_provider_credentials",
            }
        )
    if not base_url:
        raise ProviderAdapterError(
            {
                **attempted,
                "error": "external_provider_base_url_required",
                "blocked_by": "external_provider_credentials",
                "credential_source": key_name,
            }
        )
    if not model:
        raise ProviderAdapterError(
            {
                **attempted,
                "error": "external_provider_model_required",
                "blocked_by": "external_provider_credentials",
                "credential_source": key_name,
            }
        )
    return {
        "provider": provider,
        "api_key": api_key,
        "api_key_env": key_name,
        "base_url": base_url,
        "base_url_env": base_name or "provider_default",
        "model": model,
        "model_env": model_name,
        "auth_header": auth_header_value,
        "transport": provider_transport,
    }


def _credential_public_metadata(credentials: Dict[str, str]) -> Dict[str, Any]:
    transport = str(
        credentials.get("transport") or OPENAI_CHAT_COMPLETIONS
    ).strip()
    return {
        "credentials_attempted": True,
        "credentials_loaded": True,
        "credential_source": credentials["api_key_env"],
        "credential_fingerprint": _fingerprint(credentials["api_key"]),
        "endpoint_fingerprint": _fingerprint(credentials["base_url"]),
        "base_url_configured": True,
        "base_url_endpoint": (
            "anthropic_messages"
            if transport == ANTHROPIC_MESSAGES
            else "chat_completions"
        ),
        "base_url_source": credentials["base_url_env"],
        "model": credentials["model"],
        "model_source": credentials["model_env"],
        "credential_header": str(
            credentials.get("auth_header") or OpenAIAuthHeader.AUTHORIZATION.value
        ),
        "client_constructed": False,
    }


def _api_key_env_names(provider: str) -> List[str]:
    normalized = _normalize_provider(provider)
    profile = provider_profile(normalized)
    if profile is not None:
        return list(profile.api_key_env_names)
    if normalized == "openai":
        return ["OPENAI_API_KEY", "EASYICU_LLM_API_KEY"]
    if normalized == "openrouter":
        return ["OPENROUTER_API_KEY", "EASYICU_LLM_API_KEY"]
    if normalized == "anthropic":
        return ["ANTHROPIC_API_KEY", "EASYICU_LLM_API_KEY"]
    if normalized == "custom":
        return ["EASYICU_LLM_API_KEY"]
    return [f"{_env_token(normalized)}_API_KEY", "EASYICU_LLM_API_KEY"]


def validate_provider_base_url(base_url: str) -> str:
    """Refuse a provider URL this host should not send an API key to.

    The configured base URL is where the ``Authorization: Bearer <key>``
    header goes, and the request is issued by the server, from inside the
    network the server sits in. An unchecked value therefore buys two things
    at once: an SSRF probe into whatever the host can reach, and delivery of
    the operator's key to an address of the caller's choosing. ``requests``
    strips the auth header across a host change, so the redirect risk here is
    reach rather than credentials — redirects are refused anyway, because a
    destination that was checked and a destination that is contacted should be
    the same one.

    Plaintext ``http`` is allowed only to loopback, which is how the local
    model proxies used for benchmarking are addressed.

    What this deliberately does not claim: resolving a name here does not bind
    it for later. A name that answers publicly now and privately at request
    time is not caught by this check alone, which is why it runs again when
    credentials are loaded.
    """

    try:
        return validate_credential_endpoint(base_url)
    except ProviderUrlSecurityError as exc:
        error = (
            "external_provider_base_url_required"
            if exc.reason == "missing"
            else "external_provider_base_url_rejected"
        )
        raise ProviderAdapterError(
            {
                "error": error,
                **({"reason": exc.reason} if exc.reason != "missing" else {}),
                "secrets_returned": False,
            }
        ) from exc


def _base_url_env_names(provider: str) -> List[str]:
    normalized = _normalize_provider(provider)
    profile = provider_profile(normalized)
    if profile is not None:
        return list(profile.base_url_env_names)
    return [f"{_env_token(normalized)}_BASE_URL", "EASYICU_LLM_BASE_URL"]


def _model_env_names(provider: str) -> List[str]:
    normalized = _normalize_provider(provider)
    profile = provider_profile(normalized)
    if profile is not None:
        return list(profile.model_env_names)
    return [f"{_env_token(normalized)}_MODEL", "EASYICU_LLM_MODEL"]


def _default_base_url(provider: str) -> str:
    normalized = _normalize_provider(provider)
    profile = provider_profile(normalized)
    if profile is None or not profile.default_base_url:
        return ""
    return str(profile.default_base_url)


def _provider_request_url(value: str, *, transport: str) -> str:
    if transport == ANTHROPIC_MESSAGES:
        return _anthropic_messages_url(value)
    return _chat_completions_url(value)


def _provider_sdk_base_url(value: str, *, transport: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if transport == ANTHROPIC_MESSAGES:
        suffix = "/v1/messages"
    else:
        suffix = "/chat/completions"
    return text[: -len(suffix)] if text.endswith(suffix) else text


def _anthropic_messages_url(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    if text.endswith("/v1/messages"):
        return text
    if text.endswith("/v1"):
        return text + "/messages"
    return text + "/v1/messages"


def _chat_completions_url(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    if text.endswith("/chat/completions"):
        return text
    return text + "/chat/completions"


def _first_env(env: Mapping[str, str], names: List[str]) -> tuple[Optional[str], str]:
    for name in names:
        value = str(env.get(name) or "").strip()
        if value:
            return name, value
    return None, ""


def _build_chat_request(
    *,
    provider: str,
    run_id: str,
    study_id: str,
    question: Optional[str],
    summary: Dict[str, Any],
    cohort: Dict[str, Any],
    quality: List[Dict[str, Any]],
    output_artifacts: Dict[str, Dict[str, Any]],
    model: str,
    max_output_tokens: int,
    json_format_style: str,
) -> Dict[str, Any]:
    valid_evidence = [
        "run_context.json",
        "cohort_summary.json",
        *agent_outputs.OUTPUT_ARTIFACT_NAMES,
        "quality_gate.json",
    ]
    bounded_context = {
        "run_id": run_id,
        "study_id": study_id,
        "question": question,
        "summary": summary,
        "cohort": cohort,
        "quality": quality,
        "output_artifacts": {
            name: output_artifacts.get(name)
            for name in agent_outputs.OUTPUT_ARTIFACT_NAMES
            if name in output_artifacts
        },
        "valid_evidence_ids": valid_evidence,
    }
    system = (
        "You are generating a locked EasyICU analysis-only draft scaffold. "
        "Use only the bounded aggregate context. Do not invent patient rows. "
        "Return exactly one JSON object with this shape and no sectioned "
        "manuscript keys: "
        '{"agent_plan":{"steps":[{"id":"step_001","title":"...",'
        '"evidence_ids":["run_context.json"]}]},'
        '"manuscript_draft":{"claims":[{"id":"claim_001",'
        '"text":"...","evidence_ids":["cohort_summary.json"]}],'
        '"sentences":[{"id":"sentence_001","text":"...",'
        '"evidence_ids":["quality_gate.json"]}]}}. '
        "agent_plan must be an object, not an array. Do not return title, "
        "abstract, introduction, methods, or results sections. Every claim "
        "and sentence must include evidence_ids drawn only from "
        "valid_evidence_ids."
    )
    user = json.dumps(bounded_context, ensure_ascii=False, sort_keys=True)
    request = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "easyicu_policy": {
            "provider": provider,
            "bounded_aggregate_snapshot_only": True,
            "patient_rows_excluded": True,
            "allowed_evidence_ids": valid_evidence,
            "max_external_calls_per_run": _MAX_EXTERNAL_CALLS_PER_RUN,
            "max_output_tokens": max_output_tokens,
            "json_format_style": json_format_style,
        },
    }
    schema = _agent_payload_json_schema(valid_evidence)
    if json_format_style == "responses":
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "easyicu_agent_run",
                "schema": schema,
                "strict": True,
            }
        }
    elif json_format_style == "both":
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "easyicu_agent_run",
                "schema": schema,
                "strict": True,
            },
        }
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "easyicu_agent_run",
                "schema": schema,
                "strict": True,
            }
        }
    else:
        request["response_format"] = {"type": "json_object"}
    return request


def _agent_payload_json_schema(valid_evidence: List[str]) -> Dict[str, Any]:
    evidence_ids = {
        "type": "array",
        "minItems": 1,
        "items": {"type": "string", "enum": valid_evidence},
    }
    evidence_bound_record = {
        "type": "object",
        "additionalProperties": True,
        "required": ["id", "text", "evidence_ids"],
        "properties": {
            "id": {"type": "string"},
            "text": {"type": "string"},
            "evidence_ids": evidence_ids,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["agent_plan", "manuscript_draft"],
        "properties": {
            "agent_plan": {
                "type": "object",
                "additionalProperties": True,
                "required": ["steps"],
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    }
                },
            },
            "manuscript_draft": {
                "type": "object",
                "additionalProperties": True,
                "required": ["claims", "sentences"],
                "properties": {
                    "claims": {"type": "array", "items": evidence_bound_record},
                    "sentences": {"type": "array", "items": evidence_bound_record},
                },
            },
        },
    }


def _post_chat_completion(
    *,
    url: str,
    request: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
) -> Dict[str, Any]:
    import requests

    safe_request = {k: v for k, v in request.items() if k != "easyicu_policy"}
    # No redirects: the address that was checked must be the address that is
    # contacted. A 3xx would otherwise move the request to a host nothing
    # validated, which is the whole point of validating it.
    response = requests.post(
        url,
        json=safe_request,
        headers=headers,
        timeout=timeout,
        allow_redirects=False,
    )
    if response.is_redirect or response.is_permanent_redirect:
        raise ProviderAdapterError(
            {
                "error": "external_provider_redirect_refused",
                "status_code": response.status_code,
            }
        )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ProviderAdapterError({"error": "external_provider_response_not_object"})
    return data


def _coerce_provider_payload(
    response: Dict[str, Any],
    *,
    run_id: str,
    study_id: str,
    question: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    if "agent_plan" in response and "manuscript_draft" in response:
        payload = response
    else:
        content = _extract_message_content(response)
        payload = _parse_json_object(content)
    if not isinstance(payload, dict):
        raise ProviderAdapterError({"error": "external_provider_payload_not_object"})
    plan = payload.get("agent_plan")
    draft = payload.get("manuscript_draft")
    if isinstance(plan, list):
        plan = {"steps": plan}
    if not isinstance(plan, dict) or not isinstance(draft, dict):
        raise ProviderAdapterError(
            {"error": "external_provider_payload_missing_artifacts"}
        )
    plan.setdefault("run_id", run_id)
    plan.setdefault("study_id", study_id)
    plan.setdefault("execution", "external_provider_scaffold")
    draft.setdefault("run_id", run_id)
    draft.setdefault("study_id", study_id)
    draft.setdefault("question", question)
    draft.setdefault("status", "locked_until_human_signoff")
    draft.setdefault("claims", [])
    draft.setdefault("sentences", [])
    return {"agent_plan": plan, "manuscript_draft": draft}


def _extract_message_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderAdapterError(
            {"error": "external_provider_response_missing_choices"}
        )
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ProviderAdapterError(
            {"error": "external_provider_response_missing_content"}
        )
    return content


def _parse_json_object(content: str) -> Dict[str, Any]:
    text = content.strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderAdapterError(
            {
                "error": "external_provider_response_json_invalid",
                "message": str(exc),
            }
        ) from exc
    if not isinstance(parsed, dict):
        raise ProviderAdapterError(
            {"error": "external_provider_response_json_not_object"}
        )
    return parsed


def _public_usage(response: Dict[str, Any]) -> Dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _public_llm_usage(value: object) -> Dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    public: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = source.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            public[key] = item
    actual_model = source.get("actual_model")
    if (
        isinstance(actual_model, str)
        and actual_model.strip()
        and len(actual_model.strip()) <= 256
        and actual_model.strip().isprintable()
    ):
        public["actual_model"] = actual_model.strip()
    return public


def _fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]


def _normalize_provider(provider: str) -> str:
    return str(provider or "").strip().lower() or "custom"


def _is_offline_provider(provider: str) -> bool:
    return _normalize_provider(provider) in {
        "mock",
        "offline",
        "none",
        "local",
        "disabled",
    }


def _env_token(provider: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", provider.upper()).strip("_") or "EASYICU_LLM"


def _max_output_tokens(*, environ: Optional[Mapping[str, str]] = None) -> int:
    env, _ = _provider_env(environ=environ)
    raw = str(env.get("EASYICU_LLM_MAX_TOKENS") or "").strip()
    if not raw:
        return _DEFAULT_MAX_OUTPUT_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_OUTPUT_TOKENS
    return max(_MIN_MAX_OUTPUT_TOKENS, min(_ABSOLUTE_MAX_OUTPUT_TOKENS, value))


def _json_format_style(*, environ: Optional[Mapping[str, str]] = None) -> str:
    env, _ = _provider_env(environ=environ)
    value = str(env.get("EASYICU_LLM_JSON_FORMAT_STYLE") or "").strip().lower()
    if value in {"responses", "text"}:
        return "responses"
    if value in {"both", "dual"}:
        return "both"
    return "chat"


def _provider_env(
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[Dict[str, str], Dict[str, Any]]:
    source = os.environ if environ is None else environ
    base = {str(k): str(v) for k, v in source.items()}
    if _truthy(base.get("EASYICU_DISABLE_PROVIDER_ENV_FILE")):
        return base, {
            "enabled": False,
            "status": "disabled",
            "present": False,
            "loaded_keys": [],
            "secrets_returned": False,
        }
    configured = str(base.get("EASYICU_LLM_ENV_FILE") or "").strip()
    path = Path(configured).expanduser() if configured else _DEFAULT_PROVIDER_ENV_FILE
    meta: Dict[str, Any] = {
        "enabled": True,
        "status": "missing",
        "present": False,
        "configured": "custom" if configured else "default",
        "loaded_keys": [],
        "secrets_returned": False,
    }
    if not path.exists():
        return base, meta
    meta["present"] = True
    if not path.is_file():
        meta["status"] = "not_file"
        return base, meta
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        meta["status"] = "insecure_permissions"
        return base, meta
    parsed = _parse_env_file(path)
    for key, value in parsed.items():
        base.setdefault(key, value)
    meta["status"] = "loaded"
    meta["loaded_keys"] = sorted(parsed)
    return base, meta


def _parse_env_file(path: Path) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            continue
        parsed[key] = _unquote_env_value(value.strip())
    return parsed


def _quote_env_value(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:@+=,-]+", str(value or "")):
        return str(value)
    return json.dumps(str(value or ""))


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
