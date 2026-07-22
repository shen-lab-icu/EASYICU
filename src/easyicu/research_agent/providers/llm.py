"""LLM client abstraction for the research agent layer.

Two design rules:

1. **Offline fixtures are separate from production providers.** The
   deterministic mock client lives in :mod:`.mocks`, so importing this module
   does not initialize the large canned-response layer used by tests.

2. **No SDK is imported until used.** ``OpenAIClient`` lazy-imports
   ``openai``; if it is not installed the user gets a clear
   ImportError only when they actually try to invoke the model. This
   keeps ``import easyicu.research_agent`` cheap.

Adding another provider (Anthropic, Ollama, vLLM, ...) is a matter
of writing one class with a ``complete(messages, **kwargs) -> str``
method. The pipeline never imports a specific provider.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from ..authority.provider_budget import (
    active_provider_retry_available,
    consume_active_transport_attempt,
)
from .protocol import LLMClient, LLMMessage


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

    for candidate in (exc, getattr(exc, "response", None)):
        if candidate is None:
            continue
        value = getattr(candidate, "status_code", None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(
        r"\b(?:http(?:\s+status)?|status(?:\s+code)?|error\s+code)"
        r"\s*[:=]?\s*(408|409|429|500|502|503|504)\b",
        f"{type(exc).__name__}: {exc}",
        flags=re.I,
    )
    return int(match.group(1)) if match else None


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
    lowered = (base_url or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in ("localhost", "127.0.0.1", "0.0.0.0"))


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
    return SimpleNamespace(choices=choices, usage=usage_ns)


def _response_namespace_from_stream(stream: Any) -> Any:
    """Collect an OpenAI chat-completion stream into the normal response shape.

    Streaming is transport-only: downstream agents still receive the same final
    string and usage metadata when the provider supplies it.  The stream is
    always closed, including when iteration raises, so retries do not leak a
    socket to a local OpenAI-compatible proxy.
    """

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    finish_reason: Optional[str] = None
    usage = None
    saw_choice = False
    try:
        for chunk in stream:
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
        return SimpleNamespace(choices=[], usage=usage)
    reasoning = "".join(reasoning_parts)
    message = SimpleNamespace(
        content="".join(content_parts),
        reasoning_content=reasoning or None,
        reasoning=reasoning or None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# OpenAI client (optional — only imported on first use)
# ---------------------------------------------------------------------------


class OpenAIClient:
    """Thin wrapper around ``openai>=1.0`` chat completions.

    Usage::

        from easyicu.research_agent import OpenAIClient

        # OpenAI proper
        llm = OpenAIClient(model="gpt-4o-mini")

        # OpenRouter (free tier) — anything OpenAI-compatible works the
        # same way; the ``base_url`` is the only knob that differs.
        llm = OpenAIClient(
            model="google/gemini-2.0-flash-exp:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            extra_headers={"HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
                           "X-Title": "EasyICU research-agent"},
        )

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
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        supports_vision: Optional[bool] = None,
    ) -> None:
        # 🔧 2026-07-10: allow env overrides so a flaky SHARED local proxy (the
        # cli-proxy-api / Codex Tools instance that intermittently rotates its key
        # or drops the connection) can be given a longer per-call timeout and a
        # bigger retry budget without a code change:
        #   EASYICU_LLM_TIMEOUT=<seconds>   EASYICU_LLM_MAX_RETRIES=<attempts>
        request_timeout = float(
            os.environ.get("EASYICU_LLM_TIMEOUT") or request_timeout
        )
        max_retries = int(os.environ.get("EASYICU_LLM_MAX_RETRIES") or max_retries)
        kwargs: Dict[str, Any] = {}
        # Accept either OPENAI_API_KEY (vanilla) or OPENROUTER_API_KEY so
        # users don't have to alias the variable themselves.
        env_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
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
        self._timeout = request_timeout
        self._max_retries = int(max_retries)
        self._extra_body = dict(extra_body or {})
        self.supports_vision = (
            bool(supports_vision)
            if supports_vision is not None
            else _model_looks_vision_capable(model)
        )
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

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        content, _usage = self.complete_with_usage(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
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
    ) -> tuple[str, Optional[Dict[str, int]]]:
        """Return text and usage from the same provider response.

        The tuple is call-scoped: concurrent callers never have to read the
        shared compatibility attribute ``last_usage`` to attribute cost.
        """
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]
        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self._timeout,
        }
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
        if self._extra_body:
            create_kwargs["extra_body"] = self._extra_body
        # Manual back-off for 503 / overloaded errors. SDK retries are disabled
        # in the constructor, so this is the single transport retry owner.
        import time as _time

        last_exc: Optional[Exception] = None
        import json as _json

        def _do_call():
            consume_active_transport_attempt()
            if getattr(self, "_local_noauth_mode", False):
                if self._local_http_client is None:
                    raise RuntimeError("Local no-auth HTTP client was not initialized.")
                payload = {
                    "model": self._model,
                    "messages": chat_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if seed is not None:
                    payload["seed"] = int(seed)
                if top_p is not None:
                    payload["top_p"] = float(top_p)
                if self._extra_body:
                    payload.update(self._extra_body)
                resp = self._local_http_client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return _response_namespace_from_payload(data)
            stream_enabled = str(
                os.environ.get("EASYICU_LLM_STREAM", "") or ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            if stream_enabled:
                # Do not send ``stream_options`` unconditionally: several local
                # OpenAI-compatible proxies accept SSE streaming but reject the
                # optional include-usage extension.  Usage is still collected
                # when a provider includes it in any chunk; otherwise the
                # existing MeteredClient heuristic remains the fallback.
                stream = self._client.chat.completions.create(  # type: ignore[union-attr,arg-type]
                    **create_kwargs,
                    stream=True,
                )
                resp = _response_namespace_from_stream(stream)
            else:
                resp = self._client.chat.completions.create(**create_kwargs)  # type: ignore[union-attr,arg-type]
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

        # 🔧 2026-05-17: bump retry budget from 4 → 8 attempts so persistent
        # free-tier upstream rate-limit storms (Venice provider for llama-3.3-70b
        # observed ~30s Retry-After headers repeating) can't tip the run into
        # uncaught RateLimitError. Also honor the provider's Retry-After when
        # present in the exception body.
        # ``_max_retries`` is the manual attempt budget used by this outer
        # provider-recovery loop.  Honour explicit small budgets (notably the
        # experiment setting ``EASYICU_LLM_MAX_RETRIES=0``) instead of silently
        # restoring the historical eight-attempt floor.  Even a zero budget
        # must issue the initial request once; it simply disables another
        # manual attempt after a transient failure.
        manual_attempts = max(1, int(getattr(self, "_max_retries", 8)))

        def _sleep_before_retry(seconds: float, attempt_index: int) -> None:
            if (
                attempt_index + 1 < manual_attempts
                and active_provider_retry_available()
            ):
                _time.sleep(seconds)

        for attempt in range(manual_attempts):
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
                last_exc = exc
                _sleep_before_retry(2.0 * (attempt + 1), attempt)
                continue
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                transient_proxy_auth = "invalid proxy api key" in msg or (
                    "401" in msg and "proxy" in msg
                )
                transient_connection = _is_transient_connection_error(exc)
                if _is_retryable_transport_error(exc) or transient_proxy_auth:
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
                if "expecting value" in msg or "json" in msg and "decode" in msg:
                    last_exc = exc
                    _sleep_before_retry(2.0 * (attempt + 1), attempt)
                    continue
                # Our own LLM_TRANSIENT_* envelope failures from _do_call
                # (null choices / null message) are retryable too.
                if (
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
        call_usage: Optional[Dict[str, int]] = None
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
        if os.environ.get("EASYICU_LLM_DEBUG"):
            try:
                from datetime import datetime
                from pathlib import Path

                log_dir = Path(
                    os.environ.get("EASYICU_LLM_DEBUG_DIR")
                    or "./research_output/llm_debug"
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
                payload = {
                    "model": self._model,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "prompt_messages": chat_messages,
                    "raw_message": (
                        msg.model_dump() if hasattr(msg, "model_dump") else str(msg)
                    ),
                    "extracted_content_head": content[:1200],
                    "extracted_content_chars": len(content),
                }
                (log_dir / f"{ts}.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            except Exception:
                pass

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
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self._timeout,
        }
        if self._extra_body:
            create_kwargs["extra_body"] = self._extra_body
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
        msg = choice.message
        return _strip_reasoning_blocks((getattr(msg, "content", None) or "").strip())


def _model_looks_like_qwen3(model: str) -> bool:
    lowered = (model or "").strip().lower()
    return lowered.startswith("qwen3") or "/qwen3" in lowered or "qwen3-" in lowered


def _model_looks_vision_capable(model: str) -> bool:
    lowered = (model or "").strip().lower()
    if not lowered:
        return False
    positive_tokens = (
        "gpt-4o",
        "omni",
        "vision",
        "gemini",
        "qwen-vl",
        "qwen2.5-vl",
        "vl-",
        "pixtral",
        "llava",
        "molmo",
        "internvl",
    )
    negative_tokens = (
        "coder",
        "instruct",
        "reasoner",
        "embedding",
        "rerank",
        "whisper",
        "audio",
    )
    if any(token in lowered for token in negative_tokens):
        return False
    return any(token in lowered for token in positive_tokens)


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


def _retryable_provider_error(exc: Exception) -> bool:
    if _is_retryable_transport_error(exc):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "temporarily",
            "provider returned error",
            "retry after",
        )
    )


def _client_counts_transport_attempts(client: Any) -> bool:
    """Detect transport-aware clients through common transparent wrappers."""

    seen: set[int] = set()
    current = client
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if bool(getattr(current, "provider_attempt_budget_aware", False)):
            return True
        current = getattr(current, "_inner", None)
    return False


class FallbackLLMClient:
    """Try several compatible clients in order until one succeeds.

    This is primarily used for free-tier OpenRouter deployments where a
    single upstream model might be temporarily rate-limited even though
    alternative free models remain available.
    """

    provider_attempt_budget_aware = True

    def __init__(
        self,
        *clients: Any,
        name: Optional[str] = None,
    ) -> None:
        self._clients = [client for client in clients if client is not None]
        if not self._clients:
            raise ValueError("FallbackLLMClient requires at least one child client.")
        self.name = (
            name
            or "fallback("
            + " -> ".join(
                getattr(
                    client, "_model", getattr(client, "name", type(client).__name__)
                )
                for client in self._clients
            )
            + ")"
        )
        self.last_usage = None
        self.last_finish_reason = None
        self.last_client_name = None

    def complete(
        self,
        messages: Sequence["LLMMessage"],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        out, _usage = self.complete_with_usage(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
        )
        return out

    def complete_with_usage(
        self,
        messages: Sequence["LLMMessage"],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> tuple[str, Optional[Dict[str, int]]]:
        """Return usage from the same successful fallback call, when available."""
        errors: List[str] = []
        last_exc: Optional[Exception] = None
        for client in self._clients:
            try:
                if not _client_counts_transport_attempts(client):
                    consume_active_transport_attempt()
                # Forward top_p only to clients that accept it (OpenAI-
                # compatible); legacy clients keep their previous
                # 3-kwarg signature.
                import inspect as _inspect

                child_complete_with_usage = getattr(client, "complete_with_usage", None)
                child_method = (
                    child_complete_with_usage
                    if callable(child_complete_with_usage)
                    else client.complete
                )
                try:
                    _params = _inspect.signature(child_method).parameters
                    _accepts_seed = "seed" in _params
                    _accepts_top_p = "top_p" in _params
                except (TypeError, ValueError):
                    _accepts_seed = False
                    _accepts_top_p = False
                _kwargs: Dict[str, Any] = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if _accepts_seed and seed is not None:
                    _kwargs["seed"] = seed
                if _accepts_top_p and top_p is not None:
                    _kwargs["top_p"] = top_p
                if callable(child_complete_with_usage):
                    out, raw_usage = child_method(messages, **_kwargs)
                    usage = dict(raw_usage) if isinstance(raw_usage, dict) else None
                else:
                    out = child_method(messages, **_kwargs)
                    usage = None
                self.last_usage = dict(usage) if usage is not None else None
                self.last_finish_reason = getattr(client, "last_finish_reason", None)
                self.last_client_name = getattr(
                    client, "_model", getattr(client, "name", type(client).__name__)
                )
                return out, usage
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised via tests with fake clients
                last_exc = exc
                errors.append(
                    f"{getattr(client, '_model', getattr(client, 'name', type(client).__name__))}: {exc}"
                )
                if not _retryable_provider_error(exc):
                    raise
        if last_exc is not None:
            raise RuntimeError(
                "All fallback LLM clients failed after retryable provider errors: "
                + " | ".join(errors)
            ) from last_exc
        raise RuntimeError("FallbackLLMClient had no usable clients.")

    def complete_with_images(
        self,
        *,
        prompt: str,
        image_paths: Sequence[Path],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Delegate multimodal review to the first child client that supports it.

        The real implementation lives on :class:`OpenAIClient` (where
        ``_client`` / ``_model`` / ``_timeout`` exist). Previously this method
        was defined here and referenced ``self._model`` — attributes
        FallbackLLMClient never sets — so every image-QA call under a fallback
        wrapper raised ``AttributeError`` (swallowed by the visual-QA adapter
        into a spurious warning, and the actual review never ran). Delegate to
        the first vision-capable child; if none exists, degrade to a text-only
        completion so the caller still gets a usable response instead of a crash.
        """
        for client in self._clients:
            if client is not self and hasattr(client, "complete_with_images"):
                return client.complete_with_images(
                    prompt=prompt,
                    image_paths=image_paths,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
        return self.complete(
            [LLMMessage(role="user", content=prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
        )


# ---------------------------------------------------------------------------
# Per-agent LLM router (T2.3 — different tool, different brain)
# ---------------------------------------------------------------------------


_ROUTER_ROLES = ("planner", "coder", "analyzer", "writer", "literature")


class LLMRouter:
    """Per-role LLM client mapping.

    The four research-agent agents have very different needs:

    * **Planner** must emit valid JSON matching the AnalysisPlan schema.
      A frontier model usually wins here; a small model often emits
      malformed JSON that even our hardened parser cannot recover.
    * **Coder** writes the largest *output* (code + plot calls). A
      mid-tier model that is fast and cheap is the sweet spot.
    * **Analyzer** runs the shortest prompt of all (one paragraph
      input, four sentences out). The cheapest available model is
      usually fine.
    * **Writer** is brief (≈ 600 tokens) but needs to follow the
      ``{evidence:<id>}`` format precisely. A mid-tier model is
      typically enough.
    * **Literature** is optional; the offline curated registry is the
      default, but the agent can be wired through this router too.

    Running everything on the same model wastes money and rate limit.
    The :class:`LLMRouter` lets the pipeline use a different
    :class:`LLMClient` per role::

        router = LLMRouter(
            default=OpenAIClient(model="gpt-4o-mini"),
            planner=OpenAIClient(model="gpt-4o"),
            analyzer=OpenAIClient(model="gpt-4o-mini"),
        )
        pipeline = ResearchAgentPipeline(workdir=..., llm=router)

    Backwards compatibility: passing a plain :class:`LLMClient`
    (``MockLLMClient``, ``OpenAIClient``, …) to
    :class:`ResearchAgentPipeline` continues to work because the
    pipeline asks the router for ``for_role(role)`` only when the
    object actually has the method.
    """

    name = "router"

    def __init__(
        self,
        *,
        default: Optional[Any] = None,
        planner: Optional[Any] = None,
        coder: Optional[Any] = None,
        analyzer: Optional[Any] = None,
        writer: Optional[Any] = None,
        literature: Optional[Any] = None,
    ) -> None:
        self._default = default
        self._roles: Dict[str, Optional[Any]] = {
            "planner": planner,
            "coder": coder,
            "analyzer": analyzer,
            "writer": writer,
            "literature": literature,
        }
        if default is None and all(v is None for v in self._roles.values()):
            raise ValueError(
                "LLMRouter needs at least one client. Pass a `default=` "
                "and/or any subset of role-specific clients."
            )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def for_role(self, role: str) -> Any:
        """Return the client to use for ``role``.

        Falls back to the ``default`` client when a role-specific
        client is not configured. Raises ``KeyError`` if neither is
        available.
        """
        if role not in self._roles:
            raise KeyError(
                f"unknown role {role!r}; expected one of {list(self._roles)}"
            )
        client = self._roles[role] or self._default
        if client is None:
            raise KeyError(f"LLMRouter has no client for role {role!r} and no default.")
        return client

    def iter_clients(self):
        """Yield every distinct underlying client.

        Used by the pipeline to bind ``ResearchContext`` onto every
        :class:`MockLLMClient` reachable through the router so the
        canned responses pick up the cohort that's actually being
        analysed.
        """
        seen = set()
        for client in (self._default, *self._roles.values()):
            if client is None:
                continue
            ident = id(client)
            if ident in seen:
                continue
            seen.add(ident)
            yield client

    # ------------------------------------------------------------------
    # Pass-through ``complete``
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: Sequence["LLMMessage"],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
    ) -> str:
        """Route to the default client.

        This bridge exists so a router can be passed to legacy code
        paths that haven't been updated to call :meth:`for_role`.
        Prefer ``router.for_role(role).complete(...)`` in new code.
        """
        if self._default is None:
            raise RuntimeError(
                "LLMRouter.complete() called but no `default` client is "
                "configured; use ``router.for_role(role).complete(...)`` "
                "or set ``default=...`` at construction."
            )
        import inspect as _inspect

        try:
            _accepts_top_p = (
                "top_p" in _inspect.signature(self._default.complete).parameters
            )
        except (TypeError, ValueError):
            _accepts_top_p = False
        if _accepts_top_p and top_p is not None:
            return self._default.complete(
                messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )
        return self._default.complete(
            messages, max_tokens=max_tokens, temperature=temperature
        )


class CLIAgentLLMClient:
    """Drive a local, pre-authenticated coding-agent CLI as a text backend.

    This is the *altitude-1* integration of a local coding agent (Codex CLI /
    Claude Code CLI) into the research-agent framework: the CLI satisfies the
    same ``LLMClient`` protocol every other provider does, so any role
    (planner / coder / analyzer / writer / critic) can use it via
    ``llm.complete(messages) -> str``.

    What this is **not**: it does NOT delegate execution or evidence binding to
    the CLI. Generated code still runs inside the instrumented Safe Analytical
    Runtime and every number is still bound as a ``NumericClaim`` there, exactly
    as with :class:`OpenAIClient`. Letting the CLI itself run the analysis loop
    (the "Coder/repair" delegation) is a separate, larger change that has to
    re-route results back through the evidence pipeline — deliberately out of
    scope here.

    Safety posture (mirrors the webapp ``copilot.cli_agent`` twin): the CLI is
    invoked in text-only / read-only-sandbox mode, in a throwaway cwd, with no
    tool-write permission. It is still a real external model call, so it must be
    used behind the same opt-in the other real providers sit behind.

    The CLIs do not honour ``max_tokens`` / ``temperature`` / ``seed`` /
    ``top_p``; those are accepted for protocol compatibility and ignored.
    """

    _SUPPORTED = {"codex", "claude"}

    def __init__(
        self,
        backend: str = "codex",
        model: Optional[str] = None,
        request_timeout: float = 180.0,
    ) -> None:
        backend = str(backend or "").strip().lower()
        if backend not in self._SUPPORTED:
            raise ValueError(
                f"Unknown CLI backend {backend!r}; expected one of {sorted(self._SUPPORTED)}."
            )
        self._backend = backend
        self._command = backend  # executable name == backend name
        self._model = (model or "").strip()  # "" => CLI default
        self._timeout = float(request_timeout)
        self.name = f"{backend}-cli"

    @staticmethod
    def _flatten(messages: Sequence[LLMMessage]) -> tuple[str, str]:
        system_parts: List[str] = []
        convo_parts: List[str] = []
        for m in messages:
            content = str(getattr(m, "content", "") or "").strip()
            if not content:
                continue
            role = str(getattr(m, "role", "user") or "user").strip().lower()
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                convo_parts.append(f"Assistant:\n{content}")
            else:
                convo_parts.append(f"User:\n{content}")
        return "\n\n".join(system_parts), "\n\n".join(convo_parts)

    def _build_argv(self, system: str, cwd: str) -> List[str]:
        model = self._model
        if self._backend == "claude":
            argv = [self._command, "-p", "--output-format", "text"]
            if model:
                argv += ["--model", model]
            if system:
                argv += ["--append-system-prompt", system]
            # print mode + default permissions: any tool call needing approval
            # is auto-denied (no interactive approver) => stays a text generator.
            return argv
        # codex
        argv = [
            self._command,
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-C",
            cwd,
        ]
        if model:
            argv += ["-m", model]
        return argv

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        **_ignored: Any,
    ) -> str:
        import shutil
        import subprocess
        import tempfile

        if not shutil.which(self._command):
            raise RuntimeError(
                f"The '{self._command}' CLI is not installed or not on PATH."
            )
        system, conversation = self._flatten(messages)
        with tempfile.TemporaryDirectory(prefix="easyicu-research-cli-") as cwd:
            argv = self._build_argv(system, cwd)
            if self._backend == "codex":
                # codex has no system flag; fold system into the prompt.
                prompt = (
                    f"{system}\n\n{conversation}".strip() if system else conversation
                )
            else:
                prompt = conversation
            try:
                proc = subprocess.run(
                    argv,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=cwd,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"{self._command} timed out after {self._timeout:.0f}s."
                ) from exc
            except OSError as exc:
                raise RuntimeError(f"Failed to launch {self._command}: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"{self._command} exited with code {proc.returncode}: {detail[:500]}"
            )
        text = _strip_reasoning_blocks((proc.stdout or "").strip())
        if not text:
            raise RuntimeError(f"{self._command} returned an empty response.")
        return text


@dataclass
class LLMClientSelection:
    """The outcome of :func:`build_llm_client`'s capability ladder.

    Recorded so a run envelope can show *what brain actually ran* and why it
    fell back — reviewers should never have to guess whether a result came
    from a local coding-agent CLI, an API model, or the offline mock.
    """

    client: Any
    backend: str  # what was actually built ("codex" / "openai" / "mock" ...)
    requested: str  # what the caller preferred
    fell_back: bool  # True when backend != requested
    reason: str  # human-readable explanation
    ladder: List[str]  # the order that was tried


# Backends served by a local coding-agent CLI vs. an OpenAI-compatible API.
_CLI_BACKENDS = {"codex", "claude"}
_API_BACKENDS = {"openai", "openrouter", "custom"}

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def cli_backend_available(backend: str) -> bool:
    """True when the CLI executable backing *backend* is installed on PATH."""
    import shutil

    if backend not in _CLI_BACKENDS:
        return False
    return shutil.which(backend) is not None


def _api_key_present(api_key: Optional[str]) -> bool:
    return bool(
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )


def _backend_available(
    backend: str, *, api_key: Optional[str], allow_mock: bool
) -> bool:
    if backend in _CLI_BACKENDS:
        return cli_backend_available(backend)
    if backend in _API_BACKENDS:
        return _api_key_present(api_key)
    if backend == "mock":
        return allow_mock
    return False


def _construct_backend(
    backend: str,
    *,
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    extra_headers: Optional[Dict[str, str]],
) -> Any:
    if backend in _CLI_BACKENDS:
        return CLIAgentLLMClient(backend=backend, model=model or None)
    if backend in _API_BACKENDS:
        from .factory import build_provider_client

        resolved_base = base_url
        if backend == "openrouter" and not resolved_base:
            resolved_base = _OPENROUTER_BASE_URL
        provider_environment = dict(os.environ)
        if api_key:
            provider_environment[
                "OPENROUTER_API_KEY" if backend == "openrouter" else "OPENAI_API_KEY"
            ] = api_key
        if resolved_base:
            provider_environment[
                "OPENROUTER_BASE_URL" if backend == "openrouter" else "OPENAI_BASE_URL"
            ] = resolved_base
        return build_provider_client(
            provider=backend,
            model=model or "gpt-4o-mini",
            request_timeout=120.0,
            title=str((extra_headers or {}).get("X-Title") or "EasyICU LLM selector"),
            client_cls=OpenAIClient,
            environment=provider_environment,
        )
    if backend == "mock":
        from .mocks import MockLLMClient

        return MockLLMClient()
    raise ValueError(f"Unknown LLM backend: {backend!r}")


def build_llm_client(
    prefer: str = "codex",
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    allow_mock: bool = True,
    ladder: Optional[Sequence[str]] = None,
) -> LLMClientSelection:
    """Build the best available LLM client, degrading gracefully.

    The whole point of this factory is concern #1: **a local coding-agent CLI
    (Codex / Claude Code) must be an *optional* engine, never a dependency.**
    Not everyone has it installed, so selection walks a capability ladder and
    returns the first engine that is actually usable:

        prefer (e.g. "codex")  ->  "openai"  ->  "openrouter"  ->  "mock"

    - CLI backends are available iff their executable is on ``PATH``.
    - API backends are available iff an API key is supplied / in the env.
    - ``mock`` is always available (unless ``allow_mock=False``) and is the
      guaranteed floor: the pipeline still runs end-to-end with zero external
      agent, exactly as design rule #1 in this module requires.

    The returned :class:`LLMClientSelection` records what actually ran and why,
    so the choice is auditable rather than silent.
    """
    requested = str(prefer or "").strip().lower() or "codex"
    if ladder is None:
        default_chain = [requested, "openai", "openrouter", "mock"]
        # de-duplicate while preserving order
        seen: set[str] = set()
        chain: List[str] = []
        for name in default_chain:
            if name and name not in seen:
                seen.add(name)
                chain.append(name)
    else:
        chain = [str(name).strip().lower() for name in ladder if str(name).strip()]
    if not allow_mock:
        chain = [name for name in chain if name != "mock"]

    for backend in chain:
        if not _backend_available(backend, api_key=api_key, allow_mock=allow_mock):
            continue
        client = _construct_backend(
            backend,
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_headers=extra_headers,
        )
        fell_back = backend != requested
        if not fell_back:
            reason = f"using requested backend {backend!r}"
        elif backend in _CLI_BACKENDS:
            reason = (
                f"requested {requested!r} unavailable; using CLI backend {backend!r}"
            )
        elif backend == "mock":
            reason = (
                f"requested {requested!r} unavailable and no API key configured; "
                "fell back to the offline mock"
            )
        else:
            reason = f"requested {requested!r} unavailable; fell back to {backend!r}"
        return LLMClientSelection(
            client=client,
            backend=backend,
            requested=requested,
            fell_back=fell_back,
            reason=reason,
            ladder=chain,
        )

    raise RuntimeError(
        f"No usable LLM backend in ladder {chain!r}. "
        "Install a coding-agent CLI (codex/claude), configure an API key, "
        "or allow the offline mock."
    )


def resolve_role_client(llm: Any, role: str) -> Any:
    """Return the client to use for ``role``.

    If ``llm`` exposes ``for_role`` (i.e. it is an :class:`LLMRouter`),
    we delegate; otherwise the same ``llm`` is returned for every role
    — preserving the pre-T2.3 single-client semantics.
    """
    if llm is None:
        return None
    if hasattr(llm, "for_role"):
        return llm.for_role(role)
    return llm


def llm_supports_vision(client: Any) -> bool:
    """Best-effort capability probe for optional figure-VLM review.

    The pipeline uses this only to decide whether vision-based QA
    should be enabled automatically. It stays intentionally
    conservative: unknown clients default to ``False`` unless they
    explicitly advertise ``supports_vision`` or expose a
    ``complete_with_images`` method without a contradicting model
    heuristic.
    """

    if client is None:
        return False
    if hasattr(client, "supports_vision"):
        advertised = getattr(client, "supports_vision")
        try:
            return bool(advertised() if callable(advertised) else advertised)
        except Exception:
            return False
    if hasattr(client, "for_role"):
        try:
            analyzer_client = client.for_role("analyzer")
        except Exception:
            analyzer_client = None
        if analyzer_client is not None:
            return llm_supports_vision(analyzer_client)
    if hasattr(client, "iter_clients"):
        try:
            return any(llm_supports_vision(child) for child in client.iter_clients())
        except Exception:
            return False
    if hasattr(client, "complete_with_images"):
        model = getattr(client, "_model", None)
        if model is None:
            return True
        return _model_looks_vision_capable(str(model))
    return False


def llm_is_mockish(client: Any) -> bool:
    """Return true when ``client`` is effectively a mock/offline stub."""

    if client is None:
        return False
    if getattr(client, "__easyicu_mock_client__", False) is True:
        return True
    if hasattr(client, "for_role"):
        try:
            analyzer_client = client.for_role("analyzer")
        except Exception:
            analyzer_client = None
        if analyzer_client is not None:
            return llm_is_mockish(analyzer_client)
    if hasattr(client, "iter_clients"):
        try:
            children = list(client.iter_clients())
        except Exception:
            children = []
        if children:
            return all(llm_is_mockish(child) for child in children)
    lowered = " ".join(
        str(part).lower()
        for part in (
            type(client).__name__,
            getattr(client, "name", ""),
            getattr(client, "_model", ""),
        )
    )
    return "mock" in lowered


__all__ = [
    "LLMMessage",
    "LLMClient",
    "OpenAIClient",
    "CLIAgentLLMClient",
    "LLMRouter",
    "LLMClientSelection",
    "build_llm_client",
    "cli_backend_available",
    "llm_is_mockish",
    "llm_supports_vision",
    "openrouter_reasoning_extra_body",
    "resolve_role_client",
]
