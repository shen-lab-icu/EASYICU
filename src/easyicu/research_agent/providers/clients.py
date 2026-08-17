"""Direct HTTP provider client implementations.

Owner: the concrete OpenAI-compatible and Anthropic Messages clients — what
actually speaks to a provider endpoint. This module is a leaf: it knows how to
make a call, not which backend to pick or how to build one, so :mod:`.factory`
(construction) and :mod:`.llm` (backend selection and fallback) can both depend
on it without depending on each other.

Split out of ``llm.py`` in 2026-08: ``factory`` needed these classes as its
defaults, which made ``factory`` import ``llm`` and closed an import cycle
through the offline mock floor.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..authority.provider_budget import (
    consume_active_transport_attempt,
)
from ..authority.secret_redaction import (
    debug_capture_enabled,
    redact_debug_value,
    redact_text_secrets,
)
from .capabilities import (
    model_looks_vision_capable,
)
from .protocol import (
    LLMMessage,
    ProviderRefusal,
    StructuredOutputCapabilityError,
    StructuredOutputRequest,
)


_PROVIDER_CALL_RECEIPT: ContextVar[Optional[ProviderCallReceipt]] = ContextVar(
    "easyicu_provider_call_receipt",
    default=None,
)


LLM_DEBUG_FIELD_CHARS = 4000


_CLOSED_PROVIDER_FINISH_REASONS = frozenset(
    {
        "stop",
        "length",
        "tool_calls",
        "function_call",
        "content_filter",
        "cancelled",
        "error",
    }
)


def safe_provider_finish_reason(value: Any) -> Optional[str]:
    """Return a closed diagnostic category, never a provider-supplied token."""

    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in _CLOSED_PROVIDER_FINISH_REASONS else "other"


@dataclass(frozen=True)
class ProviderCallReceipt:
    """Call-scoped, response-free metadata for the just-finished transport."""

    finish_reason: Optional[str]
    usage_summary: Tuple[Tuple[str, int], ...]
    transport_attempts: int


def clear_provider_call_receipt() -> None:
    """Clear the current call context before invoking another provider."""

    _PROVIDER_CALL_RECEIPT.set(None)


def current_provider_call_receipt() -> Optional[ProviderCallReceipt]:
    """Return metadata from this context's most recent provider call only."""

    return _PROVIDER_CALL_RECEIPT.get()


def _record_provider_call_receipt(
    *,
    finish_reason: Any,
    usage: Optional[Dict[str, Any]],
    transport_attempts: int,
) -> None:
    safe_usage: list[tuple[str, int]] = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = (usage or {}).get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            safe_usage.append((key, int(value)))
    _PROVIDER_CALL_RECEIPT.set(
        ProviderCallReceipt(
            finish_reason=safe_provider_finish_reason(finish_reason),
            usage_summary=tuple(safe_usage),
            transport_attempts=max(1, int(transport_attempts)),
        )
    )


def _truncated_debug_text(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= LLM_DEBUG_FIELD_CHARS:
        return text
    return f"{text[:LLM_DEBUG_FIELD_CHARS]}… [{len(text)} chars total]"


def _truncated_debug_messages(messages: Any) -> Any:
    """Bound each debug-dumped message so one call cannot write megabytes."""

    if not isinstance(messages, (list, tuple)):
        return _truncated_debug_text(messages)
    bounded = []
    for message in messages:
        if isinstance(message, dict):
            bounded.append(
                {
                    key: (
                        _truncated_debug_text(value)
                        if isinstance(value, str)
                        else value
                    )
                    for key, value in message.items()
                }
            )
        else:
            bounded.append(_truncated_debug_text(message))
    return bounded


def _strip_reasoning_blocks(text: str) -> str:
    """Remove private reasoning blocks from OpenAI-compatible model output."""
    if not text:
        return ""
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.I | re.S)
    cleaned = re.sub(r"<think\b[^>]*>.*$", "", cleaned, flags=re.I | re.S)
    return cleaned.strip()


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Pull a ``Retry-After`` hint out of an OpenRouter / OpenAI exception.

    OpenRouter wraps the upstream provider's headers inside
    ``exc.response.headers`` AND repeats the value inside the JSON body as
    ``metadata.retry_after_seconds``. We probe both. Returns seconds (float)
    or None if we can't find a hint.
    """
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            hdr = getattr(resp, "headers", None) or {}
            ra = hdr.get("Retry-After") or hdr.get("retry-after")
            if ra is not None:
                seconds = float(ra)
                if math.isfinite(seconds) and seconds >= 0:
                    return seconds
    except Exception:
        pass
    # Fallback: parse out of the str(exc) which usually includes the JSON.
    s = str(exc)
    m = re.search(r"retry_after_seconds['\"]?\s*[:=]\s*([0-9.]+)", s)
    if m:
        try:
            seconds = float(m.group(1))
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
        except Exception:
            pass
    m = re.search(r"Retry-After['\"]?\s*[:=]?\s*['\"]?([0-9.]+)", s)
    if m:
        try:
            seconds = float(m.group(1))
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
        except Exception:
            pass
    return None


_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


def _provider_http_status_code(exc: Exception) -> Optional[int]:
    """Return a provider HTTP status without depending on one SDK class."""

    status_code = _structured_provider_http_status_code(exc)
    if status_code is not None:
        return status_code

    match = re.search(
        r"\b(?:http(?:\s+status)?|status(?:\s+code)?|error\s+code)"
        r"\s*[:=]?\s*(408|409|429|500|502|503|504)\b",
        f"{type(exc).__name__}: {exc}",
        flags=re.I,
    )
    return int(match.group(1)) if match else None


def _structured_provider_http_status_code(exc: Exception) -> Optional[int]:
    """Read an HTTP status only from typed exception/response attributes.

    A strict host retry policy must never promote arbitrary exception text into
    transport authority: validator text, provider echoes, and even user input
    can contain strings such as ``status code 500``.
    """

    try:
        response = getattr(exc, "response", None)
    except Exception:
        response = None
    for candidate in (exc, response):
        if candidate is None:
            continue
        try:
            value = getattr(candidate, "status_code", None)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and 100 <= value <= 599
            ):
                return value
        except Exception:
            pass
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return _provider_http_status_code(exc) == 429 or any(
        token in text
        for token in (
            "ratelimit",
            "rate limit",
            "rate-limit",
            "rate limited",
            "rate-limited",
            "too many requests",
        )
    )


def _is_transient_connection_error(exc: Exception) -> bool:
    """Recognise connection/timeout failures across OpenAI and HTTP clients."""

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    if any(
        token in name
        for token in (
            "apiconnectionerror",
            "connecterror",
            "connectionerror",
            "connecttimeout",
            "readtimeout",
            "pooltimeout",
            "remoteprotocolerror",
        )
    ):
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "connection error",
            "connection aborted",
            "connection refused",
            "connection reset",
            "connection timed out",
            "server disconnected",
            "remote protocol error",
        )
    )


def _is_retryable_transport_error(exc: Exception) -> bool:
    status_code = _provider_http_status_code(exc)
    if status_code in _TRANSIENT_HTTP_STATUS_CODES:
        return True
    if _is_rate_limit_error(exc) or _is_transient_connection_error(exc):
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "internal server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "request timeout",
            "temporarily unavailable",
            "overloaded",
        )
    )


def _is_local_openai_compatible_base_url(base_url: Optional[str]) -> bool:
    from .client_trust import is_loopback_openai_base_url

    return is_loopback_openai_base_url(base_url)


def _no_keepalive_limits():
    """httpx limits that DISABLE connection reuse for the local proxy.

    The shared :8787 proxy (Codex Tools / cli-proxy-api) rotates its upstream
    key mid-run. A POOLED httpx connection stays bound to the now-stale upstream
    and 401s indefinitely ("Invalid proxy api key"), while a FRESH connection
    binds to the current upstream and succeeds -- which is exactly why a curl
    probe (new socket every call) returns 200 at the same instant a long-lived
    pooled client 401s. Setting ``max_keepalive_connections=0`` makes every
    request open a fresh connection like curl, so the poisoned-pool failure mode
    cannot arise. Returns None if httpx is unavailable (caller omits the arg).
    """
    try:
        import httpx  # type: ignore

        return httpx.Limits(max_keepalive_connections=0)
    except Exception:
        return None


def _response_namespace_from_payload(payload: Dict[str, Any]) -> Any:
    choices: List[Any] = []
    for raw_choice in payload.get("choices") or []:
        raw_message = raw_choice.get("message") or {}
        message = SimpleNamespace(**raw_message)
        choices.append(
            SimpleNamespace(
                message=message,
                finish_reason=raw_choice.get("finish_reason"),
            )
        )
    usage = payload.get("usage")
    usage_ns = SimpleNamespace(**usage) if isinstance(usage, dict) else None
    # Keep relay-supplied model provenance with the normalised response.  The
    # local no-auth path otherwise used to discard the only evidence that a
    # hosted relay had substituted a fallback model.
    return SimpleNamespace(
        choices=choices,
        usage=usage_ns,
        model=payload.get("model"),
        easyicu_model_provenance=payload.get("easyicu_model_provenance"),
    )


def _response_namespace_from_stream(stream: Any) -> Any:
    """Collect an OpenAI chat-completion stream into the normal response shape.

    Streaming is transport-only: downstream agents still receive the same final
    string and usage metadata when the provider supplies it.  The stream is
    always closed, including when iteration raises, so retries do not leak a
    socket to a local OpenAI-compatible proxy.
    """

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    refusal_parts: List[str] = []
    finish_reason: Optional[str] = None
    usage = None
    response_model = None
    saw_choice = False
    try:
        for chunk in stream:
            chunk_model = getattr(chunk, "model", None)
            if isinstance(chunk_model, str) and chunk_model.strip():
                response_model = chunk_model.strip()
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            for choice in getattr(chunk, "choices", None) or []:
                saw_choice = True
                choice_finish = getattr(choice, "finish_reason", None)
                if choice_finish is not None:
                    finish_reason = choice_finish
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str):
                    content_parts.append(content)
                refusal = getattr(delta, "refusal", None)
                if isinstance(refusal, str):
                    refusal_parts.append(refusal)
                for attr in ("reasoning_content", "reasoning"):
                    value = getattr(delta, attr, None)
                    if isinstance(value, str):
                        reasoning_parts.append(value)
                        break
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    if not saw_choice:
        return SimpleNamespace(choices=[], usage=usage, model=response_model)
    reasoning = "".join(reasoning_parts)
    message = SimpleNamespace(
        content="".join(content_parts),
        reasoning_content=reasoning or None,
        reasoning=reasoning or None,
        refusal="".join(refusal_parts) or None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, model=response_model)


class OpenAIClient:
    """Thin wrapper around ``openai>=1.40.0`` chat completions.

    External transports must be created through
    :func:`easyicu.research_agent.providers.factory.build_provider_client`.
    Direct construction remains available for the factory and local transport
    tests, but an unmanaged external instance is rejected before any message is
    serialized or sent.

        pipeline = ResearchAgentPipeline(llm=llm, ...)

    The class deliberately does not bundle prompt templates.  Set
    ``EASYICU_LLM_STREAM=1`` to use transport-level streaming for long
    OpenAI-compatible responses; downstream agents still receive one final
    response string.
    """

    name = "openai"
    __easyicu_openai_transport__ = True
    provider_attempt_budget_aware = True

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: float = 120.0,
        max_retries: int = 8,
        retryable_http_status_codes: Optional[Sequence[int]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        supports_vision: Optional[bool] = None,
        supports_strict_json_schema: bool = False,
        stream_enabled: Optional[bool] = None,
        allow_environment_overrides: bool = True,
    ) -> None:
        # 🔧 2026-07-10: allow env overrides so a flaky SHARED local proxy (the
        # cli-proxy-api / Codex Tools instance that intermittently rotates its key
        # or drops the connection) can be given a longer per-call timeout and a
        # bigger retry budget without a code change:
        #   EASYICU_LLM_TIMEOUT=<seconds>   EASYICU_LLM_MAX_RETRIES=<retries>
        if allow_environment_overrides:
            request_timeout = float(
                os.environ.get("EASYICU_LLM_TIMEOUT") or request_timeout
            )
            max_retries = int(os.environ.get("EASYICU_LLM_MAX_RETRIES") or max_retries)
        if stream_enabled is None:
            stream_enabled = (
                str(os.environ.get("EASYICU_LLM_STREAM", "") or "").strip().lower()
                in {"1", "true", "yes", "on"}
                if allow_environment_overrides
                else False
            )
        kwargs: Dict[str, Any] = {}
        # Accept either OPENAI_API_KEY (vanilla) or OPENROUTER_API_KEY so
        # users don't have to alias the variable themselves.
        env_key = api_key
        if not env_key and allow_environment_overrides:
            env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
                "OPENROUTER_API_KEY"
            )
        resolved_base_url = base_url or (
            os.environ.get("OPENAI_BASE_URL") if allow_environment_overrides else None
        )
        if env_key:
            kwargs["api_key"] = env_key
        # macOS system proxies (Clash, Surge, etc.) silently break
        # localhost calls because httpx (used by the OpenAI SDK)
        # respects the system proxy even when the user has
        # ``localhost`` in the proxy exception list. Detect a local
        # base_url and inject a non-proxying httpx client so vLLM /
        # llama.cpp / Ollama just work.
        self._client = None
        self._local_http_client = None
        self._local_noauth_mode = bool(
            resolved_base_url
            and _is_local_openai_compatible_base_url(resolved_base_url)
            and not env_key
        )
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        if _is_local_openai_compatible_base_url(resolved_base_url):
            try:
                import httpx  # type: ignore

                # ``trust_env=False`` tells httpx to ignore HTTP_PROXY /
                # HTTPS_PROXY env vars *and* the macOS system proxy
                # configuration. Without this, Clash / Surge / Shadow-
                # rocket route localhost traffic to their listener
                # which returns 503 because vLLM is not configured as
                # an upstream.
                _limits = _no_keepalive_limits()
                _client_kw = dict(trust_env=False, timeout=request_timeout)
                if _limits is not None:
                    _client_kw["limits"] = _limits
                local_http_client = httpx.Client(**_client_kw)
                kwargs["http_client"] = local_http_client
                if self._local_noauth_mode and resolved_base_url:
                    self._local_http_client = httpx.Client(
                        base_url=resolved_base_url.rstrip("/"),
                        **_client_kw,
                    )
            except Exception:
                # Fall back to setting the env vars; the SDK's default
                # client picks them up.
                os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
                os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")
        # The explicit recovery loop in ``complete`` is the sole retry owner.
        # Leaving the OpenAI SDK's internal retries enabled multiplies every
        # outer attempt and makes a bounded repair look arbitrarily slow.
        kwargs["max_retries"] = 0
        # OpenRouter recommends — and some providers require — a
        # ``HTTP-Referer`` / ``X-Title`` header for analytics. Pass them
        # to the SDK as default headers when supplied.
        if extra_headers:
            kwargs["default_headers"] = dict(extra_headers)
        self._provider_auth_header_mode = (
            "x-api-key"
            if any(
                str(key).strip().lower() == "x-api-key" for key in (extra_headers or {})
            )
            else "authorization"
        )
        # Stash the params needed to REBUILD the client with a fresh httpx
        # connection pool on a transient proxy-401 (see _rebuild_openai_client):
        # the shared :8787 proxy rotates its upstream key, and a POOLED httpx
        # connection stays bound to the stale upstream -> 401s forever, while a
        # NEW connection binds to the current upstream -> 200. Exclude the live
        # ``http_client`` (a fresh one is created on rebuild).
        self._client_base_kwargs = {
            k: v for k, v in kwargs.items() if k != "http_client"
        }
        self._needs_local_http_client = _is_local_openai_compatible_base_url(
            resolved_base_url
        )
        self._resolved_base_url = resolved_base_url
        self._request_timeout = request_timeout
        if not self._local_noauth_mode:
            try:
                from openai import OpenAI  # type: ignore
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised only when SDK missing
                raise ImportError(
                    "OpenAIClient requires the 'openai' package. Install with `pip install openai`."
                ) from exc
            self._client = OpenAI(**kwargs)
        self._model = model
        self._completion_token_parameter = _completion_token_parameter_name(model)
        self._timeout = request_timeout
        # ``max_retries`` means retries *after* the initial request, matching
        # both its public name and the structured-response retry contract.  A
        # historical implementation treated it as total attempts, which made
        # ``max_retries=1`` issue only one request and forced callers to encode
        # an off-by-one workaround.  Keep one explicit total-attempt variable
        # in the loop below instead of letting the two meanings drift again.
        self._max_retries = max(0, int(max_retries))
        if retryable_http_status_codes is None:
            self._retryable_http_status_codes = None
        else:
            normalized_statuses: set[int] = set()
            for raw_code in retryable_http_status_codes:
                if not isinstance(raw_code, int) or isinstance(raw_code, bool):
                    raise ValueError("retryable HTTP statuses must be integers")
                code = raw_code
                if code < 100 or code > 599:
                    raise ValueError("retryable HTTP statuses must be in 100..599")
                normalized_statuses.add(code)
            self._retryable_http_status_codes = frozenset(normalized_statuses)
        self._stream_enabled = bool(stream_enabled)
        self._allow_environment_overrides = bool(allow_environment_overrides)
        self._extra_body = dict(extra_body or {})
        self.supports_vision = (
            bool(supports_vision)
            if supports_vision is not None
            else model_looks_vision_capable(model)
        )
        self.supports_strict_json_schema = bool(supports_strict_json_schema)
        if _model_looks_like_qwen3(model):
            self._extra_body.setdefault("enable_thinking", False)
            chat_kwargs = self._extra_body.get("chat_template_kwargs")
            if not isinstance(chat_kwargs, dict):
                chat_kwargs = {}
            chat_kwargs.setdefault("enable_thinking", False)
            self._extra_body["chat_template_kwargs"] = chat_kwargs
        # OpenRouter reasoning-model suppression: DeepSeek V4 Flash,
        # GLM, Qwen-thinking etc. dump chain-of-thought into content
        # unless we explicitly exclude it. Auto-apply when the
        # base_url looks like OpenRouter and the model is a known
        # thinking family.
        if resolved_base_url and "openrouter" in (resolved_base_url or "").lower():
            reasoning_body = openrouter_reasoning_extra_body(model)
            if reasoning_body:
                for k, v in reasoning_body.items():
                    self._extra_body.setdefault(k, v)
        # Mint construction/loopback authority only after every
        # model-dependent dispatch option is finalized.  Recording earlier
        # makes reviewed Qwen/OpenRouter clients look mutated on first use.
        from .client_trust import _mark_reviewed_transport_constructed

        _mark_reviewed_transport_constructed(self)
        if _is_local_openai_compatible_base_url(resolved_base_url):
            from .client_trust import _register_loopback_provider_client

            _register_loopback_provider_client(
                self,
                model=model,
                base_url=str(resolved_base_url),
            )

    def _require_outbound_authorization(self) -> None:
        """Reject unmanaged external transports before serializing messages."""

        from .client_trust import require_provider_client_authorization

        try:
            require_provider_client_authorization(self)
        except Exception as exc:
            raise PermissionError(
                "external OpenAI-compatible calls require factory-minted "
                "provider authorization"
            ) from exc

    def _rebuild_openai_client(self) -> None:
        """Recreate the OpenAI client with a FRESH httpx connection pool.

        The shared local proxy (:8787) rotates its upstream key; a POOLED httpx
        connection stays bound to the STALE upstream and 401s indefinitely, while
        a NEW connection binds to the current upstream and succeeds (verified live:
        a fresh probe got 200 on the same backend/key/instant that a long-lived
        pooled client got 401). Called from the transient-proxy-401 retry branch so
        the next attempt uses a fresh pool instead of hammering the dead connection.
        Best-effort: on any failure it keeps the existing client.
        """
        # Recreate the raw local http client (used by the no-auth POST path).
        if getattr(self, "_local_http_client", None) is not None:
            try:
                import httpx  # type: ignore

                if self._resolved_base_url:
                    _limits = _no_keepalive_limits()
                    _kw = dict(
                        base_url=self._resolved_base_url.rstrip("/"),
                        trust_env=False,
                        timeout=getattr(self, "_request_timeout", self._timeout),
                    )
                    if _limits is not None:
                        _kw["limits"] = _limits
                    self._local_http_client = httpx.Client(**_kw)
            except Exception:
                pass
        if getattr(self, "_client", None) is None:
            from .client_trust import _refresh_reviewed_transport_dispatch

            _refresh_reviewed_transport_dispatch(self)
            return
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return
        kwargs = dict(getattr(self, "_client_base_kwargs", {}) or {})
        if getattr(self, "_needs_local_http_client", False):
            try:
                import httpx  # type: ignore

                _limits = _no_keepalive_limits()
                _kw = dict(
                    trust_env=False,
                    timeout=getattr(self, "_request_timeout", self._timeout),
                )
                if _limits is not None:
                    _kw["limits"] = _limits
                kwargs["http_client"] = httpx.Client(**_kw)
            except Exception:
                pass
        try:
            new_client = OpenAI(**kwargs)
        except Exception:
            from .client_trust import _refresh_reviewed_transport_dispatch

            _refresh_reviewed_transport_dispatch(self)
            return
        # Swap in the fresh client with a plain reference assignment (atomic in
        # CPython). Do NOT close the old client synchronously: the single
        # OpenAIClient is shared across the writer's 8-way ThreadPoolExecutor
        # (agents/core.py), and a peer thread may be mid-request on the old client.
        # Closing its httpx pool here tears that in-flight request down -- which
        # itself surfaces as a "connection reset" that re-enters this same
        # transient branch and triggers cascading rebuilds across all writers.
        # The old pool is reclaimed by GC once no thread holds it; rebuilds are
        # rare and the client is per-run, so the transient leak is bounded.
        self._client = new_client
        from .client_trust import _refresh_reviewed_transport_dispatch

        _refresh_reviewed_transport_dispatch(self)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        structured_output: Optional[StructuredOutputRequest] = None,
    ) -> str:
        content, _usage = self.complete_with_usage(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            structured_output=structured_output,
        )
        return content

    def complete_with_usage(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        structured_output: Optional[StructuredOutputRequest] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Return text and usage from the same provider response.

        The tuple is call-scoped: concurrent callers never have to read the
        shared compatibility attribute ``last_usage`` to attribute cost.
        """
        # Clear compatibility and call-scoped metadata before authorization/
        # transport so a failed call can never inherit the preceding call's
        # receipt, usage, or finish reason.
        clear_provider_call_receipt()
        # Clear compatibility metadata before authorization/transport so a
        # failed call can never inherit the preceding successful call's usage
        # or finish reason in a diagnostic projection.
        self.last_usage = None
        self.last_finish_reason = None
        self.last_transport_attempts = 0
        self._require_outbound_authorization()
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]
        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "temperature": temperature,
            "timeout": self._timeout,
        }
        create_kwargs[self._completion_token_parameter] = int(max_tokens)
        if seed is not None:
            # OpenAI / OpenRouter / most OpenAI-compatible providers
            # accept a ``seed`` integer for deterministic(-ish) output.
            # Providers that ignore it still succeed; the envelope
            # records the requested value regardless so reviewers can
            # see user intent even when the provider does not honour
            # it.
            create_kwargs["seed"] = int(seed)
        # top_p is intentionally optional. When unset we do NOT pass it
        # to the API so the provider default applies; the envelope
        # records ``requested_top_p=None`` which a reviewer can read as
        # "provider default" rather than "unknown".
        if top_p is not None:
            create_kwargs["top_p"] = float(top_p)
        if structured_output is not None:
            if not self.supports_strict_json_schema:
                raise StructuredOutputCapabilityError(
                    "OpenAI-compatible client was not configured for strict JSON Schema"
                )
            if "response_format" in self._extra_body:
                raise StructuredOutputCapabilityError(
                    "response_format is ambiguous between extra_body and the "
                    "typed structured-output request"
                )
            create_kwargs["response_format"] = (
                structured_output.to_openai_response_format()
            )
        if self._extra_body:
            create_kwargs["extra_body"] = self._extra_body
        # Manual back-off for 503 / overloaded errors. SDK retries are disabled
        # in the constructor, so this is the single transport retry owner.
        import time as _time

        last_exc: Optional[Exception] = None
        import json as _json

        def _do_call():
            hard_stop_remaining = consume_active_transport_attempt()
            transport_kwargs = dict(create_kwargs)
            if hard_stop_remaining is not None:
                transport_kwargs["timeout"] = min(
                    float(transport_kwargs["timeout"]),
                    float(hard_stop_remaining),
                )
            if getattr(self, "_local_noauth_mode", False):
                if self._local_http_client is None:
                    raise RuntimeError("Local no-auth HTTP client was not initialized.")
                payload = {
                    "model": self._model,
                    "messages": chat_messages,
                    "temperature": temperature,
                }
                payload[self._completion_token_parameter] = int(max_tokens)
                if seed is not None:
                    payload["seed"] = int(seed)
                if top_p is not None:
                    payload["top_p"] = float(top_p)
                if structured_output is not None:
                    payload["response_format"] = (
                        structured_output.to_openai_response_format()
                    )
                if self._extra_body:
                    payload.update(self._extra_body)
                post_kwargs: Dict[str, Any] = {"json": payload}
                if hard_stop_remaining is not None:
                    post_kwargs["timeout"] = transport_kwargs["timeout"]
                resp = self._local_http_client.post(
                    "/chat/completions",
                    **post_kwargs,
                )
                resp.raise_for_status()
                data = resp.json()
                return _response_namespace_from_payload(data)
            if self._stream_enabled:
                # Do not send ``stream_options`` unconditionally: several local
                # OpenAI-compatible proxies accept SSE streaming but reject the
                # optional include-usage extension.  Usage is still collected
                # when a provider includes it in any chunk; otherwise the
                # existing MeteredClient heuristic remains the fallback.
                stream = self._client.chat.completions.create(  # type: ignore[union-attr,arg-type]
                    **transport_kwargs,
                    stream=True,
                )
                resp = _response_namespace_from_stream(stream)
            else:
                resp = self._client.chat.completions.create(**transport_kwargs)  # type: ignore[union-attr,arg-type]
            # Eager validation of the envelope so transient null-choices/null-
            # message responses surface here and are caught by the retry loop
            # below, rather than crashing the caller with `'NoneType' object
            # is not subscriptable` later.
            _choices = getattr(resp, "choices", None)
            if not _choices:
                raise RuntimeError(
                    "LLM_TRANSIENT_NO_CHOICES: provider returned no choices "
                    f"(finish_reason={getattr(resp, 'finish_reason', None)}, "
                    f"model={self._model})"
                )
            _first = _choices[0]
            if getattr(_first, "message", None) is None:
                raise RuntimeError(
                    "LLM_TRANSIENT_NO_MESSAGE: provider returned a choice "
                    f"without `.message` (model={self._model})"
                )
            return resp

        # 🔧 2026-05-17: bump retry budget from 4 → 8 retries (up to 9 total
        # attempts) so persistent
        # free-tier upstream rate-limit storms (Venice provider for llama-3.3-70b
        # observed ~30s Retry-After headers repeating) can't tip the run into
        # uncaught RateLimitError. Also honor the provider's Retry-After when
        # present in the exception body.
        # The public value is an *additional retry* count.  Total attempts are
        # therefore initial request + retries.  A zero budget still issues the
        # initial request exactly once.
        manual_attempts = 1 + max(0, int(getattr(self, "_max_retries", 8)))

        def _record_transport_failure(exc: BaseException, attempts: int) -> None:
            self.last_transport_attempts = int(attempts)
            # The structured-retry owner may receive this exception instead of
            # a response.  Attach only a count -- never request/response data.
            try:
                setattr(exc, "easyicu_transport_attempts", int(attempts))
            except Exception:
                pass

        def _retryable_for_this_client(exc: Exception) -> bool:
            # When a caller supplies a status allowlist (the Web Research Agent
            # does), it is the whole retry policy: connection errors, malformed
            # envelopes, 408/409/429, and message-text guesses do not get an
            # implicit retry.  The default preserves the broader CLI policy.
            if self._retryable_http_status_codes is not None:
                return (
                    _structured_provider_http_status_code(exc)
                    in self._retryable_http_status_codes
                )
            return _is_retryable_transport_error(exc)

        def _sleep_before_retry(seconds: float, attempt_index: int) -> None:
            # Only the attempt count decides. This also consulted the step
            # allowance, which was coherent while that allowance cancelled
            # retries; once it stopped doing so (2026-08-04) the check
            # survived only to suppress the backoff of a retry that was going
            # to happen regardless -- i.e. to hammer a failing endpoint
            # hardest exactly when the step was nearly out of budget.
            if attempt_index + 1 < manual_attempts:
                _time.sleep(seconds)

        for attempt in range(manual_attempts):
            self.last_transport_attempts = attempt + 1
            try:
                resp = _do_call()
                break
            except _json.JSONDecodeError as exc:
                # 🔧 2026-05-16: OpenRouter / free-tier providers occasionally
                # stream a chunk that the openai SDK can't parse as JSON
                # (rate-limit notice mid-stream, truncated SSE, etc.). Treat
                # as transient: short backoff, retry. Pilot run
                # run_20260516T123840_cc32d5 lost step 08_model_validation to
                # exactly this.
                _record_transport_failure(exc, attempt + 1)
                if self._retryable_http_status_codes is not None:
                    raise
                last_exc = exc
                _sleep_before_retry(2.0 * (attempt + 1), attempt)
                continue
            except Exception as exc:  # noqa: BLE001
                _record_transport_failure(exc, attempt + 1)
                msg = str(exc).lower()
                transient_proxy_auth = "invalid proxy api key" in msg or (
                    "401" in msg and "proxy" in msg
                )
                transient_connection = _is_transient_connection_error(exc)
                if _retryable_for_this_client(exc) or (
                    self._retryable_http_status_codes is None and transient_proxy_auth
                ):
                    last_exc = exc
                    if transient_connection or transient_proxy_auth:
                        # A fresh connection pool is required when the local
                        # proxy rotates its upstream key or drops a pooled
                        # connection. The SDK itself owns no retries.
                        self._rebuild_openai_client()
                    # Respect provider-supplied Retry-After (e.g. Venice's
                    # ~30 s for llama-3.3-70b:free). Fall back to a quadratic
                    # backoff so consecutive failures don't hammer the
                    # endpoint (5s, 20s, 45s, 80s, ...).
                    _retry_after = _extract_retry_after(exc)
                    if _retry_after is not None:
                        backoff = float(_retry_after) + 2.0
                    else:
                        backoff = 5.0 * (attempt + 1) ** 2
                    _sleep_before_retry(min(backoff, 120.0), attempt)
                    continue
                # JSON parse errors sometimes surface wrapped in other
                # exception classes (e.g. APIError). Catch by message text
                # as a safety net so we still get the backoff path.
                msg = str(exc).lower()
                if self._retryable_http_status_codes is None and (
                    "expecting value" in msg or "json" in msg and "decode" in msg
                ):
                    last_exc = exc
                    _sleep_before_retry(2.0 * (attempt + 1), attempt)
                    continue
                # Our own LLM_TRANSIENT_* envelope failures from _do_call
                # (null choices / null message) are retryable too.
                if self._retryable_http_status_codes is None and (
                    "llm_transient_no_choices" in msg
                    or "llm_transient_no_message" in msg
                ):
                    last_exc = exc
                    _sleep_before_retry(2.0 * (attempt + 1), attempt)
                    continue
                raise
        else:
            if last_exc is not None:
                raise last_exc
        # T3.2 cost tracking: stash the SDK's reported usage so a wrapping
        # ``MeteredClient`` can pull authoritative token counts instead of
        # falling back to the chars/4 heuristic. Defensive: not every
        # provider populates ``usage`` on every response.
        call_usage: Optional[Dict[str, Any]] = None
        actual_model: Optional[str] = None
        relay_provenance: Optional[Dict[str, Any]] = None
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                call_usage = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        getattr(usage, "completion_tokens", 0) or 0
                    ),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
            response_model = getattr(resp, "model", None)
            if isinstance(response_model, str) and response_model.strip():
                actual_model = response_model.strip()
                if call_usage is None:
                    call_usage = {}
                call_usage["actual_model"] = actual_model
            response_provenance = getattr(resp, "easyicu_model_provenance", None)
            if isinstance(response_provenance, dict):
                relay_provenance = dict(response_provenance)
                if call_usage is None:
                    call_usage = {}
                call_usage["model_provenance"] = relay_provenance
        except Exception:
            call_usage = None
        # Compatibility only; cost attribution uses the call-scoped return.
        self.last_usage = call_usage

        # T1.3 — robust content extraction. Reasoning-tuned models
        # (GLM-4.5, DeepSeek-R1, o1-style, Qwen3) often leave ``content``
        # empty and put the answer in ``reasoning`` / ``reasoning_content``,
        # OR embed the entire output (including the answer) inside
        # <think>…</think> tags with nothing after the closing tag.
        # OpenRouter typically surfaces reasoning under ``reasoning``;
        # z.ai's native API uses ``reasoning_content``. Fall through
        # the common attributes and finally scan the message dump.
        #
        # IMPORTANT: strip <think> blocks BEFORE the empty-content check so
        # that a response like "<think>…</think>" (no trailing answer text,
        # produced by Qwen3 in default thinking mode) correctly falls through
        # to the fallback chain rather than being treated as non-empty.
        # (choices/message validated upstream in _do_call → retried with
        # backoff. By the time we get here resp.choices[0].message is non-None.)
        choice = resp.choices[0]
        self.last_finish_reason = getattr(choice, "finish_reason", None)
        msg = choice.message
        raw_refusal = getattr(msg, "refusal", None)
        refusal_reason = (
            redact_text_secrets(str(raw_refusal).strip())
            if raw_refusal is not None and str(raw_refusal).strip()
            else ""
        )
        if refusal_reason:
            refusal_reason = _truncated_debug_text(refusal_reason)
            _record_provider_call_receipt(
                finish_reason=self.last_finish_reason,
                usage=call_usage,
                transport_attempts=self.last_transport_attempts,
            )
            receipt = current_provider_call_receipt()
            raise ProviderRefusal(
                refusal_reason,
                finish_reason=receipt.finish_reason if receipt is not None else None,
                usage_summary=(
                    dict(receipt.usage_summary) if receipt is not None else None
                ),
                transport_attempts=(
                    receipt.transport_attempts
                    if receipt is not None
                    else self.last_transport_attempts
                ),
            )
        raw_msg_content = (getattr(msg, "content", None) or "").strip()
        content = _strip_reasoning_blocks(raw_msg_content)
        if not content:
            for attr in ("reasoning_content", "reasoning"):
                val = getattr(msg, attr, None)
                if isinstance(val, str) and val.strip():
                    content = val.strip()
                    break
        if not content:
            # Last-resort: walk the SDK's model_dump() and pick the
            # longest non-trivial string field. Catches providers that
            # use ``thinking`` or other vendor-specific keys.
            try:
                dump = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)  # type: ignore[arg-type]
                best = ""
                for k, v in (dump or {}).items():
                    if k in {"role", "refusal", "annotations"}:
                        continue
                    if isinstance(v, str) and len(v.strip()) > len(best):
                        best = v.strip()
                if best:
                    content = _strip_reasoning_blocks(best)
            except Exception:
                pass
        if not content and raw_msg_content:
            # Qwen3 / thinking-mode last-ditch: the model emitted only a
            # <think>…</think> block, or an unclosed <think> prefix, with no
            # trailing answer text. Extract
            # the inner reasoning so the downstream parser at least receives
            # non-empty text (it may still fail JSON parsing, but the error
            # message will contain useful information instead of len=0).
            m = re.search(r"<think\b[^>]*>(.*?)</think>", raw_msg_content, re.I | re.S)
            if m:
                content = m.group(1).strip()
            else:
                m = re.search(r"<think\b[^>]*>(.*)$", raw_msg_content, re.I | re.S)
                if m:
                    content = m.group(1).strip()

        # Optional debug dump — ``EASYICU_LLM_DEBUG=1 …`` writes one
        # JSON file per call so the user can inspect what the model
        # actually returned (finish_reason, raw message, prompt).
        if debug_capture_enabled(os.environ.get("EASYICU_LLM_DEBUG")) and (
            configured_debug_dir := str(
                os.environ.get("EASYICU_LLM_DEBUG_DIR") or ""
            ).strip()
        ):
            try:
                from datetime import datetime
                from pathlib import Path

                log_dir = Path(configured_debug_dir)
                log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                # The dump contains the full prompt: the research question,
                # variable definitions, cohort description and every method
                # detail the agent reasoned over. That is study-sensitive even
                # though it is not patient-level, so the directory is owner-only
                # and each file is written 0600.
                try:
                    os.chmod(log_dir, 0o700)
                except OSError:
                    pass
                ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
                payload = redact_debug_value(
                    {
                        # Do not collapse configured/requested and actual
                        # response models. A hosted relay may have fallen back
                        # after the request left this client.
                        "requested_model": self._model,
                        "actual_model": actual_model,
                        "model_provenance": relay_provenance,
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "prompt_messages": _truncated_debug_messages(chat_messages),
                        "raw_message_head": _truncated_debug_text(
                            msg.model_dump_json()
                            if hasattr(msg, "model_dump_json")
                            else str(msg)
                        ),
                        "extracted_content_head": content[:1200],
                        "extracted_content_chars": len(content),
                        "note": (
                            "Truncated and recursively redacted debug dump. "
                            "Contains study design detail; keep it out of shared "
                            "or synced directories."
                        ),
                    }
                )
                target = log_dir / f"{ts}.json"
                descriptor = os.open(
                    target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(
                        payload, handle, indent=2, ensure_ascii=False, default=str
                    )
            except Exception:
                pass

        _record_provider_call_receipt(
            finish_reason=self.last_finish_reason,
            usage=call_usage,
            transport_attempts=self.last_transport_attempts,
        )
        return content, call_usage

    def complete_with_images(
        self,
        *,
        prompt: str,
        image_paths: Sequence[Path],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Run a multimodal chat-completions request against image files.

        Optional: normal agents use ``complete(...)``. ``VLMVisualQAAdapter``
        checks for the method via the ``llm_supports_vision`` probe and falls
        back to text-only review when a provider does not support image inputs.
        Lives on OpenAIClient (not FallbackLLMClient) because it needs
        ``self._client`` / ``self._model`` / ``self._timeout`` / ``self._extra_body``.
        """
        clear_provider_call_receipt()
        self.last_usage = None
        self.last_finish_reason = None
        self.last_transport_attempts = 0
        self._require_outbound_authorization()
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            p = Path(path)
            mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                }
            )
        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "timeout": self._timeout,
        }
        create_kwargs[self._completion_token_parameter] = int(max_tokens)
        if self._extra_body:
            create_kwargs["extra_body"] = self._extra_body
        hard_stop_remaining = consume_active_transport_attempt()
        if hard_stop_remaining is not None:
            create_kwargs["timeout"] = min(
                float(create_kwargs["timeout"]),
                float(hard_stop_remaining),
            )
        self.last_transport_attempts = 1
        resp = self._client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.last_usage = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        getattr(usage, "completion_tokens", 0) or 0
                    ),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
            else:
                self.last_usage = None
        except Exception:
            self.last_usage = None
        choice = resp.choices[0]
        self.last_finish_reason = getattr(choice, "finish_reason", None)
        _record_provider_call_receipt(
            finish_reason=self.last_finish_reason,
            usage=self.last_usage,
            transport_attempts=self.last_transport_attempts,
        )
        msg = choice.message
        return _strip_reasoning_blocks((getattr(msg, "content", None) or "").strip())


def _anthropic_finish_reason(value: Any) -> Optional[str]:
    """Map Anthropic stop reasons into the closed provider vocabulary."""

    normalized = str(value or "").strip().lower()
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
        "pause_turn": "stop",
    }.get(normalized, "other" if normalized else None)


class AnthropicMessagesClient:
    """Native Anthropic Messages API adapter with explicit schema authority."""

    name = "anthropic"
    __easyicu_anthropic_transport__ = True
    provider_attempt_budget_aware = True

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: float = 120.0,
        max_retries: int = 0,
        retryable_http_status_codes: Optional[Sequence[int]] = None,
        supports_strict_json_schema: bool = False,
        stream_enabled: Optional[bool] = False,
        allow_environment_overrides: bool = True,
        **unsupported: Any,
    ) -> None:
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported Anthropic transport options: {names}")
        if stream_enabled:
            raise ValueError("Anthropic Messages streaming is not enabled in this adapter")
        if allow_environment_overrides:
            request_timeout = float(
                os.environ.get("EASYICU_LLM_TIMEOUT") or request_timeout
            )
            max_retries = int(os.environ.get("EASYICU_LLM_MAX_RETRIES") or max_retries)
        timeout = float(request_timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        resolved_key = str(
            api_key
            or (os.environ.get("ANTHROPIC_API_KEY") if allow_environment_overrides else "")
            or ""
        ).strip()
        if not resolved_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        resolved_base_url = str(
            base_url
            or (
                os.environ.get("ANTHROPIC_BASE_URL")
                if allow_environment_overrides
                else ""
            )
            or "https://api.anthropic.com"
        ).rstrip("/")
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as exc:  # pragma: no cover - SDK-missing environment
            raise ImportError(
                "AnthropicMessagesClient requires the 'anthropic' package. "
                "Install EasyICU with the agentic or webapp extra."
            ) from exc
        self._client = Anthropic(
            api_key=resolved_key,
            base_url=resolved_base_url,
            timeout=timeout,
            max_retries=0,
        )
        self._model = str(model or "").strip()
        if not self._model:
            raise ValueError("Anthropic model is required")
        self._resolved_base_url = resolved_base_url
        self._request_timeout = timeout
        self._timeout = timeout
        self._max_retries = max(0, int(max_retries))
        if retryable_http_status_codes is None:
            self._retryable_http_status_codes = None
        else:
            statuses: set[int] = set()
            for raw_status in retryable_http_status_codes:
                if not isinstance(raw_status, int) or isinstance(raw_status, bool):
                    raise ValueError("retryable HTTP statuses must be integers")
                if raw_status < 100 or raw_status > 599:
                    raise ValueError("retryable HTTP statuses must be in 100..599")
                statuses.add(raw_status)
            self._retryable_http_status_codes = frozenset(statuses)
        self._stream_enabled = False
        self._allow_environment_overrides = bool(allow_environment_overrides)
        self.supports_strict_json_schema = bool(supports_strict_json_schema)
        self.supports_vision = False
        self.last_usage: Optional[Dict[str, Any]] = None
        self.last_finish_reason: Optional[str] = None
        self.last_transport_attempts = 0
        from .client_trust import _mark_reviewed_transport_constructed

        _mark_reviewed_transport_constructed(self)

    def _require_outbound_authorization(self) -> None:
        from .client_trust import require_provider_client_authorization

        try:
            require_provider_client_authorization(self)
        except Exception as exc:
            raise PermissionError(
                "external Anthropic calls require factory-minted provider authorization"
            ) from exc

    @staticmethod
    def _wire_messages(
        messages: Sequence[LLMMessage],
    ) -> tuple[str, list[dict[str, str]]]:
        system_parts: list[str] = []
        wire: list[dict[str, str]] = []
        for message in messages:
            role = str(message.role or "").strip().lower()
            content = str(message.content or "")
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            if role not in {"user", "assistant"}:
                raise ValueError(f"unsupported Anthropic message role: {role!r}")
            if wire and wire[-1]["role"] == role:
                wire[-1]["content"] += "\n\n" + content
            else:
                wire.append({"role": role, "content": content})
        if not wire:
            raise ValueError("Anthropic Messages requires at least one user/assistant message")
        return "\n\n".join(system_parts), wire

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        structured_output: Optional[StructuredOutputRequest] = None,
    ) -> str:
        text, _usage = self.complete_with_usage(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            structured_output=structured_output,
        )
        return text

    def complete_with_usage(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        structured_output: Optional[StructuredOutputRequest] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        del seed  # The native Messages API does not expose a seed parameter.
        clear_provider_call_receipt()
        self.last_usage = None
        self.last_finish_reason = None
        self.last_transport_attempts = 0
        self._require_outbound_authorization()
        system, wire_messages = self._wire_messages(messages)
        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": wire_messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if system:
            create_kwargs["system"] = system
        if top_p is not None:
            create_kwargs["top_p"] = float(top_p)
        if structured_output is not None:
            if not self.supports_strict_json_schema:
                raise StructuredOutputCapabilityError(
                    "Anthropic client was not configured for strict JSON Schema"
                )
            create_kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": json.loads(structured_output.schema_json),
                }
            }

        import time as _time

        attempts = 1 + self._max_retries
        last_exc: Optional[Exception] = None
        response: Any = None
        for attempt in range(attempts):
            self.last_transport_attempts = attempt + 1
            hard_stop_remaining = consume_active_transport_attempt()
            transport_kwargs = dict(create_kwargs)
            if hard_stop_remaining is not None:
                transport_kwargs["timeout"] = min(
                    self._timeout,
                    float(hard_stop_remaining),
                )
            try:
                response = self._client.messages.create(**transport_kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                try:
                    setattr(exc, "easyicu_transport_attempts", attempt + 1)
                except Exception:
                    pass
                status = _structured_provider_http_status_code(exc)
                retryable = (
                    status in self._retryable_http_status_codes
                    if self._retryable_http_status_codes is not None
                    else _is_retryable_transport_error(exc)
                )
                if not retryable or attempt + 1 >= attempts:
                    raise
                retry_after = _extract_retry_after(exc)
                backoff = (
                    float(retry_after) + 1.0
                    if retry_after is not None
                    else min(2.0 * (attempt + 1) ** 2, 30.0)
                )
                _time.sleep(backoff)
        else:  # pragma: no cover - loop either breaks or raises
            if last_exc is not None:
                raise last_exc

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        call_usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        actual_model = str(getattr(response, "model", "") or "").strip()
        if actual_model:
            call_usage["actual_model"] = actual_model
        self.last_usage = call_usage
        raw_stop_reason = getattr(response, "stop_reason", None)
        self.last_finish_reason = _anthropic_finish_reason(raw_stop_reason)
        text_parts: list[str] = []
        for block in list(getattr(response, "content", None) or []):
            if str(getattr(block, "type", "") or "") == "text":
                value = getattr(block, "text", None)
                if isinstance(value, str) and value:
                    text_parts.append(value)
        text = "\n".join(text_parts).strip()
        _record_provider_call_receipt(
            finish_reason=self.last_finish_reason,
            usage=call_usage,
            transport_attempts=self.last_transport_attempts,
        )
        if str(raw_stop_reason or "").strip().lower() == "refusal":
            raise ProviderRefusal(
                _truncated_debug_text(
                    redact_text_secrets(text or "Anthropic provider refusal")
                ),
                finish_reason=self.last_finish_reason,
                usage_summary=call_usage,
                transport_attempts=self.last_transport_attempts,
            )
        if not text:
            error = RuntimeError("Anthropic provider returned no text content")
            setattr(error, "easyicu_transport_attempts", self.last_transport_attempts)
            raise error
        return text, call_usage


def _model_looks_like_qwen3(model: str) -> bool:
    lowered = (model or "").strip().lower()
    return lowered.startswith("qwen3") or "/qwen3" in lowered or "qwen3-" in lowered


def _completion_token_parameter_name(model: str) -> str:
    """Return the output-cap field honored by the selected model family.

    GPT-5 and OpenAI ``o`` reasoning models use ``max_completion_tokens``.
    Several compatible gateways silently accept but ignore the legacy
    ``max_tokens`` field for those models, which defeats both output-size and
    cost controls. Other compatible providers still require ``max_tokens``.
    """

    leaf = (model or "").strip().lower().rsplit("/", 1)[-1]
    if leaf.startswith("gpt-5") or re.match(r"^o[134](?:-|$)", leaf):
        return "max_completion_tokens"
    return "max_tokens"


def openrouter_reasoning_extra_body(model: str) -> Optional[Dict[str, Any]]:
    """Return provider-specific reasoning controls only for models that need them.

    OpenRouter free models are not uniform here:

    * some reasoning-heavy families (notably GLM / Qwen / DeepSeek-R1 style
      endpoints) benefit from suppressing reasoning so the usable answer is
      not truncated inside ``message.reasoning``;
    * other endpoints (notably GPT-OSS free) reject requests that try to
      disable reasoning because reasoning is mandatory on that route.

    Keep the default conservative: only attach the extra_body when the model
    family is known to benefit from it.
    """
    lowered = (model or "").strip().lower()
    if not lowered:
        return None
    if "gpt-oss" in lowered:
        return None
    if any(token in lowered for token in ("glm", "qwen", "deepseek", "r1")):
        return {"reasoning": {"effort": "none", "exclude": True}}
    return None
