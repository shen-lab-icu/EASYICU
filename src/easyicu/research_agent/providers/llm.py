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
from contextvars import ContextVar
import hashlib
import json
import math
import mimetypes
import os
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..authority.provider_budget import (
    consume_active_provider_handoff,
    consume_active_transport_attempt,
)
from ..authority.secret_redaction import (
    debug_capture_enabled,
    redact_debug_value,
    redact_text_secrets,
)
from .capabilities import (
    CLIAccountReadiness,
    REGISTERED_CLI_BACKEND_NAMES,
    SUPPORTED_CLI_ACCOUNT_NAMES,
    SUPPORTED_PROVIDER_NAMES,
    cli_account_profile,
    llm_supports_vision,
    model_looks_vision_capable,
    provider_profile,
    user_account_profile,
)
from .protocol import (
    LLMClient,
    LLMMessage,
    ProviderRefusal,
    StructuredOutputCapabilityError,
    StructuredOutputRequest,
)

# HTTP client implementations live in their own leaf owner so that the
# factory can build them without depending on this selection ladder.
from .clients import (  # noqa: F401  (re-exported for existing callers)
    _PROVIDER_CALL_RECEIPT,
    AnthropicMessagesClient,
    LLM_DEBUG_FIELD_CHARS,
    OpenAIClient,
    ProviderCallReceipt,
    _CLOSED_PROVIDER_FINISH_REASONS,
    _TRANSIENT_HTTP_STATUS_CODES,
    _anthropic_finish_reason,
    _completion_token_parameter_name,
    _extract_retry_after,
    _is_local_openai_compatible_base_url,
    _is_rate_limit_error,
    _is_retryable_transport_error,
    _is_transient_connection_error,
    _model_looks_like_qwen3,
    _no_keepalive_limits,
    _provider_http_status_code,
    _record_provider_call_receipt,
    _response_namespace_from_payload,
    _response_namespace_from_stream,
    _strip_reasoning_blocks,
    _structured_provider_http_status_code,
    _truncated_debug_messages,
    _truncated_debug_text,
    clear_provider_call_receipt,
    current_provider_call_receipt,
    openrouter_reasoning_extra_body,
    safe_provider_finish_reason,
)

#: Per-field ceiling for the optional LLM debug dump. A full prompt is tens of
#: kilobytes, and an unbounded per-call dump fills the disk of a machine that
#: is already tight on space during a long run.












































# ---------------------------------------------------------------------------
# OpenAI client (optional — only imported on first use)
# ---------------------------------------------------------------------------














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


def client_counts_transport_attempts(client: Any) -> bool:
    """Detect transport-aware clients through common transparent wrappers."""

    seen: set[int] = set()
    current = client
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if bool(getattr(current, "provider_attempt_budget_aware", False)):
            return True
        current = getattr(current, "_inner", None)
    return False


# Private compatibility alias for older call sites and tests. New wrappers use
# the public name so the pre-transport accounting contract is explicit.
_client_counts_transport_attempts = client_counts_transport_attempts


class FallbackLLMClient:
    """Try several compatible clients in order until one succeeds.

    This is primarily used for free-tier OpenRouter deployments where a
    single upstream model might be temporarily rate-limited even though
    alternative free models remain available.
    """

    provider_attempt_budget_aware = True

    @property
    def supports_strict_json_schema(self) -> bool:
        """Advertise only when every possible fallback can honor the schema."""

        from .capabilities import llm_supports_strict_json_schema

        return bool(self._clients) and all(
            llm_supports_strict_json_schema(client) for client in self._clients
        )

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
        from .factory import _register_provider_wrapper

        _register_provider_wrapper(self, children_getter=lambda: tuple(self._clients))

    def complete(
        self,
        messages: Sequence["LLMMessage"],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        structured_output: Optional[StructuredOutputRequest] = None,
    ) -> str:
        out, _usage = self.complete_with_usage(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            structured_output=structured_output,
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
        structured_output: Optional[StructuredOutputRequest] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Return usage from the same successful fallback call, when available."""
        errors: List[str] = []
        last_exc: Optional[Exception] = None
        for client in self._clients:
            try:
                if not client_counts_transport_attempts(client):
                    consume_active_provider_handoff()
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
                    _accepts_structured_output = "structured_output" in _params or any(
                        parameter.kind is _inspect.Parameter.VAR_KEYWORD
                        for parameter in _params.values()
                    )
                except (TypeError, ValueError):
                    _accepts_seed = False
                    _accepts_top_p = False
                    _accepts_structured_output = False
                _kwargs: Dict[str, Any] = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if _accepts_seed and seed is not None:
                    _kwargs["seed"] = seed
                if _accepts_top_p and top_p is not None:
                    _kwargs["top_p"] = top_p
                if structured_output is not None:
                    if not _accepts_structured_output:
                        raise StructuredOutputCapabilityError(
                            "fallback child cannot honor the strict structured-output "
                            "request"
                        )
                    _kwargs["structured_output"] = structured_output
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
                if not client_counts_transport_attempts(client):
                    consume_active_provider_handoff()
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


REASONING_EFFORT_PROFILE_PROVIDER_DEFAULT = "provider_default"
REASONING_EFFORT_PROFILE_ADAPTIVE_V1 = "adaptive_v1"
_REASONING_EFFORT_PROFILES: Dict[str, Dict[str, str]] = {
    REASONING_EFFORT_PROFILE_PROVIDER_DEFAULT: {},
    REASONING_EFFORT_PROFILE_ADAPTIVE_V1: {
        "planner": "medium",
        "coder": "medium",
        "analyzer": "low",
        "writer": "low",
        "literature": "low",
        "repair": "high",
    },
}
_ROUTER_ROLES = (
    "planner",
    "coder",
    "analyzer",
    "writer",
    "literature",
    "repair",
)


def reasoning_effort_by_role(profile: str) -> Dict[str, str]:
    """Return one reviewed per-role reasoning profile.

    ``provider_default`` sends no override. ``adaptive_v1`` is deliberately
    explicit so benchmark authority and per-call evidence can distinguish it
    from a proxy-wide default that may change outside EasyICU.
    """

    normalized = str(profile or "").strip().lower()
    if normalized not in _REASONING_EFFORT_PROFILES:
        raise ValueError(
            f"unknown reasoning effort profile {profile!r}; expected one of "
            f"{sorted(_REASONING_EFFORT_PROFILES)}"
        )
    return dict(_REASONING_EFFORT_PROFILES[normalized])


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
    * **Repair** is isolated from initial code generation so a configured
      profile can spend more reasoning only after a validated failure.

    Running everything on the same model wastes money and rate limit.
    The :class:`LLMRouter` lets the pipeline use a different
    :class:`LLMClient` per role::

        router = LLMRouter(
            default=factory_minted_default_client,
            planner=factory_minted_planner_client,
            analyzer=factory_minted_analyzer_client,
        )
        pipeline = ResearchAgentPipeline(workdir=..., llm=router)

    External clients must be created through
    :func:`easyicu.research_agent.providers.factory.build_provider_client`;
    unmanaged direct clients are rejected before transport.

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
        repair: Optional[Any] = None,
        reasoning_effort_profile: str = REASONING_EFFORT_PROFILE_PROVIDER_DEFAULT,
    ) -> None:
        profile = str(reasoning_effort_profile or "").strip().lower()
        expected_efforts = reasoning_effort_by_role(profile)
        self._default = default
        self._roles: Dict[str, Optional[Any]] = {
            "planner": planner,
            "coder": coder,
            "analyzer": analyzer,
            "writer": writer,
            "literature": literature,
            "repair": repair,
        }
        self._reasoning_effort_profile = profile
        if default is None and all(v is None for v in self._roles.values()):
            raise ValueError(
                "LLMRouter needs at least one client. Pass a `default=` "
                "and/or any subset of role-specific clients."
            )
        for role, expected_effort in expected_efforts.items():
            client = self._roles.get(role) or self._default
            extra_body = getattr(client, "_extra_body", None)
            reasoning = (
                extra_body.get("reasoning") if isinstance(extra_body, dict) else None
            )
            actual_effort = (
                reasoning.get("effort") if isinstance(reasoning, dict) else None
            )
            if actual_effort != expected_effort:
                raise ValueError(
                    f"reasoning profile {profile!r} requires {role}="
                    f"{expected_effort!r}, observed {actual_effort!r}"
                )
        from .factory import _register_provider_wrapper

        _register_provider_wrapper(
            self, children_getter=lambda: tuple(self.iter_clients())
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
        structured_output: Optional[StructuredOutputRequest] = None,
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
            _parameters = _inspect.signature(self._default.complete).parameters
            _accepts_top_p = "top_p" in _parameters
            _accepts_structured_output = "structured_output" in _parameters or any(
                parameter.kind is _inspect.Parameter.VAR_KEYWORD
                for parameter in _parameters.values()
            )
        except (TypeError, ValueError):
            _accepts_top_p = False
            _accepts_structured_output = False
        kwargs: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if _accepts_top_p and top_p is not None:
            kwargs["top_p"] = top_p
        if structured_output is not None:
            if not _accepts_structured_output:
                raise StructuredOutputCapabilityError(
                    "router default client cannot honor strict structured output"
                )
            kwargs["structured_output"] = structured_output
        return self._default.complete(messages, **kwargs)


class CLIAgentLLMClient:
    """Drive a local, pre-authenticated coding-agent CLI as a text backend.

    This is the *altitude-1* integration of the user-facing Codex account
    transport (plus one legacy internal Claude Code seam) into the
    research-agent framework: the adapter satisfies the same ``LLMClient``
    protocol every other provider does, so any role
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
    Codex does, however, support a strict final ``--output-schema`` contract,
    which this adapter materializes inside the throwaway working directory.
    """

    _SUPPORTED = set(REGISTERED_CLI_BACKEND_NAMES)

    def __init__(
        self,
        backend: str = "codex",
        model: Optional[str] = None,
        request_timeout: float = 180.0,
        environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        backend = str(backend or "").strip().lower()
        if backend not in self._SUPPORTED:
            raise ValueError(
                f"Unknown CLI backend {backend!r}; expected one of {sorted(self._SUPPORTED)}."
            )
        self._backend = backend
        self._model = (model or "").strip()  # "" => CLI default
        self._timeout = float(request_timeout)
        profile = cli_account_profile(backend)
        assert profile is not None  # guarded by _SUPPORTED
        self._command = profile.executable
        self.supports_strict_json_schema = bool(profile.supports_strict_json_schema)
        from .subprocess_env import build_provider_subprocess_env

        subprocess_environment = build_provider_subprocess_env(
            backend,
            environment=environment,
        )
        self._subprocess_environment = MappingProxyType(subprocess_environment)
        self._subprocess_environment_sha256 = hashlib.sha256(
            json.dumps(
                subprocess_environment,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        self.name = f"{backend}-cli"
        from .client_trust import _mark_reviewed_transport_constructed

        _mark_reviewed_transport_constructed(self)

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

    def _build_argv(
        self,
        system: str,
        cwd: str,
        *,
        output_schema_path: Optional[str] = None,
    ) -> List[str]:
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
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
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
        if output_schema_path:
            argv += ["--output-schema", output_schema_path]
        return argv

    def _require_outbound_authorization(self) -> None:
        """Reject direct/unmanaged CLI transports before process launch."""

        from .client_trust import require_provider_client_authorization

        try:
            require_provider_client_authorization(self)
        except Exception as exc:
            raise PermissionError(
                "external CLI calls require factory-minted provider authorization"
            ) from exc

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        structured_output: Optional[StructuredOutputRequest] = None,
        **_ignored: Any,
    ) -> str:
        import shutil
        import subprocess
        import tempfile

        self._require_outbound_authorization()
        if structured_output is not None:
            if not self.supports_strict_json_schema:
                raise StructuredOutputCapabilityError(
                    f"{self._backend} CLI does not advertise strict JSON Schema"
                )
            if not isinstance(structured_output, StructuredOutputRequest):
                raise TypeError("structured_output must be StructuredOutputRequest")
        if not shutil.which(self._command):
            raise RuntimeError(
                f"The '{self._command}' CLI is not installed or not on PATH."
            )
        system, conversation = self._flatten(messages)
        with tempfile.TemporaryDirectory(prefix="easyicu-research-cli-") as cwd:
            output_schema_path: Optional[str] = None
            if structured_output is not None:
                schema_path = Path(cwd) / "final-output.schema.json"
                schema_path.write_text(
                    structured_output.schema_json,
                    encoding="utf-8",
                )
                schema_path.chmod(0o600)
                output_schema_path = str(schema_path)
            argv = self._build_argv(
                system,
                cwd,
                output_schema_path=output_schema_path,
            )
            if self._backend == "codex":
                # Codex has no system flag; fold system into the prompt.
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
                    env=dict(self._subprocess_environment),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"{self._command} timed out after {self._timeout:.0f}s."
                ) from exc
            except OSError as exc:
                raise RuntimeError(f"Failed to launch {self._command}: {exc}") from exc
        if proc.returncode != 0:
            detail = redact_text_secrets((proc.stderr or proc.stdout or "").strip())
            raise RuntimeError(
                f"{self._command} exited with code {proc.returncode}: {detail[:500]}"
            )
        raw_output = (proc.stdout or "").strip()
        text = _strip_reasoning_blocks(raw_output)
        if not text:
            raise RuntimeError(f"{self._command} returned an empty response.")
        return text


_CODEX_TURN_PROGRESS_METHODS = frozenset(
    {
        "turn/started",
        "turn/plan/updated",
        "item/started",
        "item/completed",
        "item/agentMessage/delta",
        "item/plan/delta",
        "item/reasoning/summaryPartAdded",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/textDelta",
        "rawResponseItem/completed",
        "rawResponse/completed",
        "thread/tokenUsage/updated",
        "model/rerouted",
        "model/verification",
        "model/safetyBuffering/updated",
    }
)

_CODEX_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)


def _codex_turn_progress_notification(
    notification: Mapping[str, Any],
    *,
    thread_id: str,
    turn_id: str,
) -> bool:
    if notification.get("method") not in _CODEX_TURN_PROGRESS_METHODS:
        return False
    params = notification.get("params")
    return bool(
        isinstance(params, Mapping)
        and params.get("threadId") == thread_id
        and params.get("turnId") == turn_id
    )


class CodexAppServerLLMClient:
    """Use one isolated user's managed ChatGPT login through App Server.

    Unlike :class:`CLIAgentLLMClient`, this transport never inspects the host
    operator's Codex login. The Web session owner must provide an isolated
    ``HOME``/``CODEX_HOME`` pair plus its non-secret binding digest.
    """

    supports_strict_json_schema = True
    provider_attempt_budget_aware = True

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        request_timeout: float = 180.0,
        turn_hard_timeout: float | None = None,
        reasoning_effort: str | None = None,
        environment: Mapping[str, str],
    ) -> None:
        profile = user_account_profile("codex")
        assert profile is not None
        from .subprocess_env import build_provider_subprocess_env

        selected = build_provider_subprocess_env(
            "codex",
            environment=environment,
            required_keys=(
                "EASYICU_ALLOW_EXTERNAL_LLM",
                "EASYICU_CODEX_SESSION_SHA256",
            ),
        )
        session_sha256 = str(
            selected.get("EASYICU_CODEX_SESSION_SHA256") or ""
        ).strip()
        if not re.fullmatch(r"[0-9a-f]{64}", session_sha256):
            raise ValueError("codex_auth_session_binding_required")
        if not selected.get("HOME") or not selected.get("CODEX_HOME"):
            raise ValueError("codex_auth_isolated_home_required")
        self._backend = "codex"
        self._model = str(model or "").strip()
        timeout = float(request_timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        hard_timeout = (
            timeout if turn_hard_timeout is None else float(turn_hard_timeout)
        )
        if not math.isfinite(hard_timeout) or hard_timeout <= 0:
            raise ValueError("turn_hard_timeout must be finite and positive")
        normalized_effort = (
            None
            if reasoning_effort is None
            else str(reasoning_effort).strip().lower()
        )
        if normalized_effort is not None and normalized_effort not in (
            _CODEX_REASONING_EFFORTS
        ):
            raise ValueError("reasoning_effort is not supported by Codex App Server")
        self._timeout = max(0.1, timeout)
        self._turn_hard_timeout = max(0.1, hard_timeout)
        self._reasoning_effort = normalized_effort
        self._command = profile.executable
        self._session_binding_sha256 = session_sha256
        self._endpoint_identity = (
            f"{profile.endpoint_identity}/session/{session_sha256}"
        )
        self._subprocess_environment = MappingProxyType(selected)
        self._subprocess_environment_sha256 = hashlib.sha256(
            json.dumps(
                selected,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        self.last_usage: Optional[Dict[str, int]] = None
        self.last_finish_reason: Optional[str] = None
        self.last_model: Optional[str] = None
        self.name = "codex-app-server"
        from .client_trust import _mark_reviewed_transport_constructed

        _mark_reviewed_transport_constructed(self)

    def _require_outbound_authorization(self) -> None:
        from .client_trust import require_provider_client_authorization

        try:
            require_provider_client_authorization(self)
        except Exception as exc:
            raise PermissionError(
                "Codex App Server calls require factory-minted user authorization"
            ) from exc

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        structured_output: Optional[StructuredOutputRequest] = None,
        **_ignored: Any,
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
        **_ignored: Any,
    ) -> tuple[str, Optional[Dict[str, int]]]:
        del max_tokens, temperature, seed, top_p
        import tempfile

        from .codex_app_server import CodexAppServerError, CodexAppServerRuntime

        self._require_outbound_authorization()
        if structured_output is not None and not isinstance(
            structured_output, StructuredOutputRequest
        ):
            raise TypeError("structured_output must be StructuredOutputRequest")
        system, conversation = CLIAgentLLMClient._flatten(messages)
        prompt = f"{system}\n\n{conversation}".strip() if system else conversation
        if not prompt:
            raise ValueError("Codex App Server requires a non-empty prompt")
        with tempfile.TemporaryDirectory(prefix="easyicu-codex-turn-") as cwd:
            runtime = CodexAppServerRuntime(
                environment=self._subprocess_environment,
                cwd=Path(cwd),
                executable=self._command,
                request_timeout=min(self._timeout, 30.0),
                experimental_api=True,
            )
            with runtime:
                account = runtime.request(
                    "account/read",
                    {"refreshToken": True},
                    timeout=min(self._timeout, 30.0),
                ).get("account")
                if not isinstance(account, Mapping) or account.get("type") != "chatgpt":
                    raise RuntimeError("codex_auth_chatgpt_login_required")
                thread_params: Dict[str, Any] = {
                    "cwd": cwd,
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "ephemeral": True,
                    "dynamicTools": [],
                    "environments": [],
                    "runtimeWorkspaceRoots": [cwd],
                    "developerInstructions": (
                        "Act only as a text-generation backend. Do not invoke tools, "
                        "inspect files, or execute commands; answer from the supplied "
                        "message and return only the requested final response."
                    ),
                }
                if self._model:
                    thread_params["model"] = self._model
                thread_result = runtime.request(
                    "thread/start",
                    thread_params,
                    timeout=min(self._timeout, 30.0),
                )
                thread = thread_result.get("thread")
                if not isinstance(thread, Mapping) or not thread.get("id"):
                    raise RuntimeError("codex_app_server_thread_start_invalid")
                self.last_model = str(thread_result.get("model") or "") or None
                turn_params: Dict[str, Any] = {
                    "threadId": str(thread["id"]),
                    "input": [{"type": "text", "text": prompt}],
                    "approvalPolicy": "never",
                    "sandboxPolicy": {
                        "type": "readOnly",
                        "networkAccess": False,
                    },
                    "environments": [],
                    "runtimeWorkspaceRoots": [cwd],
                }
                if self._reasoning_effort is not None:
                    turn_params["effort"] = self._reasoning_effort
                if structured_output is not None:
                    turn_params["outputSchema"] = json.loads(
                        structured_output.schema_json
                    )
                notification_start = runtime.notification_count
                attempt_started = time.monotonic()
                hard_stop_remaining = consume_active_transport_attempt()
                turn_start_timeout = min(self._timeout, 30.0)
                if hard_stop_remaining is not None:
                    turn_start_timeout = min(
                        turn_start_timeout,
                        float(hard_stop_remaining),
                    )
                turn_result = runtime.request(
                    "turn/start",
                    turn_params,
                    timeout=turn_start_timeout,
                )
                turn = turn_result.get("turn")
                if not isinstance(turn, Mapping) or not turn.get("id"):
                    raise RuntimeError("codex_app_server_turn_start_invalid")
                turn_idle_timeout = self._timeout
                turn_hard_timeout = self._turn_hard_timeout
                if hard_stop_remaining is not None:
                    elapsed = max(0.0, time.monotonic() - attempt_started)
                    remaining = float(hard_stop_remaining) - elapsed
                    if remaining <= 0:
                        raise CodexAppServerError(
                            "codex_auth_notification_hard_timeout"
                        )
                    turn_idle_timeout = min(turn_idle_timeout, remaining)
                    turn_hard_timeout = min(turn_hard_timeout, remaining)
                thread_id = str(thread["id"])
                turn_id = str(turn["id"])
                completed = runtime.wait_for_notification(
                    lambda item: item.get("method") == "turn/completed"
                    and (item.get("params") or {}).get("threadId") == thread_id
                    and ((item.get("params") or {}).get("turn") or {}).get("id")
                    == turn_id,
                    after=notification_start,
                    timeout=turn_idle_timeout,
                    hard_timeout=turn_hard_timeout,
                    progress_predicate=lambda item: _codex_turn_progress_notification(
                        item,
                        thread_id=thread_id,
                        turn_id=turn_id,
                    ),
                )
                completed_turn = (completed.get("params") or {}).get("turn") or {}
                status = str(completed_turn.get("status") or "")
                if status != "completed":
                    error = completed_turn.get("error") or {}
                    error_kind = error.get("codexErrorInfo") if isinstance(error, Mapping) else None
                    self.last_finish_reason = "error"
                    raise RuntimeError(
                        "codex_app_server_turn_failed:"
                        + str(error_kind or status or "unknown")[:120]
                    )
                text = ""
                for item in completed_turn.get("items") or []:
                    if isinstance(item, Mapping) and item.get("type") == "agentMessage":
                        text = str(item.get("text") or "").strip()
                notifications = runtime.notifications_since(notification_start)
                if not text:
                    text = "".join(
                        str((item.get("params") or {}).get("delta") or "")
                        for item in notifications
                        if item.get("method") == "item/agentMessage/delta"
                        and (item.get("params") or {}).get("turnId") == turn_id
                    ).strip()
                if not text:
                    raise RuntimeError("codex_app_server_empty_response")
                usage: Optional[Dict[str, int]] = None
                for item in notifications:
                    if item.get("method") != "thread/tokenUsage/updated":
                        continue
                    params = item.get("params") or {}
                    if params.get("turnId") != turn_id:
                        continue
                    last = (params.get("tokenUsage") or {}).get("last") or {}
                    if isinstance(last, Mapping):
                        usage = {
                            "prompt_tokens": max(0, int(last.get("inputTokens") or 0)),
                            "completion_tokens": max(
                                0, int(last.get("outputTokens") or 0)
                            ),
                            "total_tokens": max(0, int(last.get("totalTokens") or 0)),
                        }
                self.last_usage = dict(usage) if usage is not None else None
                self.last_finish_reason = "stop"
                return text, usage


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
_CLI_BACKENDS = set(SUPPORTED_CLI_ACCOUNT_NAMES)
_API_BACKENDS = set(SUPPORTED_PROVIDER_NAMES)


def cli_backend_available(backend: str) -> bool:
    """True when the CLI executable backing *backend* is installed on PATH."""
    import shutil

    if backend not in _CLI_BACKENDS:
        return False
    return shutil.which(backend) is not None


def probe_cli_account_readiness(
    backend: str,
    *,
    environment: Optional[Mapping[str, str]] = None,
    timeout: float = 5.0,
) -> CLIAccountReadiness:
    """Check one local CLI/account boundary without returning command output.

    Codex exposes a stable non-interactive login-status command, so its account
    session can be verified before a research prompt is sent. Other reviewed
    account CLIs have no equivalent stable cross-version status command; an
    installed executable is therefore launch-ready with authentication
    explicitly unverified. Its first real call still fails closed if the user
    has not logged in.
    """

    import shutil
    import subprocess
    import tempfile

    normalized = str(backend or "").strip().lower()
    profile = cli_account_profile(normalized)
    if profile is None:
        raise ValueError(f"Unknown CLI account backend: {backend!r}")
    from .subprocess_env import build_provider_subprocess_env

    safe_environment = build_provider_subprocess_env(
        normalized,
        environment=environment,
    )
    executable = shutil.which(
        profile.executable,
        path=safe_environment.get("PATH"),
    )
    if executable is None:
        return CLIAccountReadiness(
            backend=normalized,
            provider_identity=profile.provider_identity,
            executable_present=False,
            status_check_supported=profile.status_argv is not None,
            authentication_verified=False,
            launch_ready=False,
            reason_code="cli_executable_missing",
            subprocess_calls=0,
        )
    if profile.status_argv is None:
        return CLIAccountReadiness(
            backend=normalized,
            provider_identity=profile.provider_identity,
            executable_present=True,
            status_check_supported=False,
            authentication_verified=None,
            launch_ready=True,
            reason_code="cli_login_status_unavailable",
            subprocess_calls=0,
        )
    bounded_timeout = max(0.1, min(float(timeout), 30.0))
    try:
        with tempfile.TemporaryDirectory(prefix="easyicu-cli-status-") as cwd:
            result = subprocess.run(
                list(profile.status_argv),
                capture_output=True,
                text=True,
                timeout=bounded_timeout,
                cwd=cwd,
                env=safe_environment,
            )
    except subprocess.TimeoutExpired:
        reason_code = "cli_login_status_timeout"
        verified = False
    except OSError:
        reason_code = "cli_login_status_failed"
        verified = False
    else:
        verified = result.returncode == 0
        reason_code = "cli_account_ready" if verified else "cli_login_required"
    return CLIAccountReadiness(
        backend=normalized,
        provider_identity=profile.provider_identity,
        executable_present=True,
        status_check_supported=True,
        authentication_verified=verified,
        launch_ready=verified,
        reason_code=reason_code,
        subprocess_calls=1,
    )


def _api_key_present(
    backend: str,
    api_key: Optional[str],
    *,
    environment: Mapping[str, str],
) -> bool:
    profile = provider_profile(backend)
    return bool(
        api_key
        or (
            profile is not None
            and any(environment.get(name) for name in profile.api_key_env_names)
        )
    )


def _external_llm_authorized(environment: Mapping[str, str]) -> bool:
    """Whether this environment carries the canonical external-LLM opt-in.

    ``CLAUDE.md`` makes ``ai_optin.check_external_llm_opt_in`` the canonical
    gate for any path that may issue a real LLM call, and the CLIs that own the
    user-facing prompt call it before stamping ``EASYICU_ALLOW_EXTERNAL_LLM``
    into the environment they hand this factory. Reading that stamp here is
    what makes the gate structural rather than a convention every future caller
    has to remember.
    """

    from .client_trust import ALLOW_EXTERNAL_LLM_ENV

    raw = str(environment.get(ALLOW_EXTERNAL_LLM_ENV, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _backend_available(
    backend: str,
    *,
    api_key: Optional[str],
    allow_mock: bool,
    environment: Mapping[str, str],
) -> bool:
    if backend in _CLI_BACKENDS:
        return cli_backend_available(backend) and _external_llm_authorized(environment)
    if backend in _API_BACKENDS:
        # Deliberately does NOT also require the opt-in stamp. An API backend
        # requested without authorization must reach ``build_provider_client``
        # and fail closed there with the precise
        # ``EXTERNAL_LLM_NOT_AUTHORIZED`` ProviderConfigurationError. Screening
        # it out here instead would silently degrade an explicitly requested
        # real provider to the mock floor -- trading an actionable error for a
        # wrong answer, which is exactly the fallback this codebase forbids.
        # ``test_api_backend_without_opt_in_fails_closed_and_is_never_downgraded``
        # locks that behaviour.
        return _api_key_present(
            backend,
            api_key,
            environment=environment,
        )
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
    request_timeout: float,
    environment: Mapping[str, str],
) -> Any:
    if backend in _CLI_BACKENDS:
        from .factory import authorize_provider_client

        profile = cli_account_profile(backend)
        assert profile is not None  # guarded by _CLI_BACKENDS
        _model_source, configured_model = profile.model(environment)
        resolved_model = str(model or configured_model or "").strip()
        client = CLIAgentLLMClient(
            backend=backend,
            model=resolved_model or None,
            request_timeout=request_timeout,
            environment=environment,
        )
        return authorize_provider_client(
            client,
            provider=profile.provider_identity,
            model=resolved_model or "cli-default",
            base_url=profile.endpoint_identity,
            destination="external",
            environment=environment,
        )
    if backend in _API_BACKENDS:
        from .factory import build_provider_client

        profile = provider_profile(backend)
        if profile is None:  # pragma: no cover - guarded by _API_BACKENDS
            raise ValueError(f"Unknown LLM backend: {backend!r}")
        provider_environment = dict(environment)
        if api_key:
            provider_environment[profile.api_key_env_names[0]] = api_key
        if base_url:
            provider_environment[profile.base_url_env_names[0]] = base_url
        _model_source, configured_model = profile.model(provider_environment)
        resolved_model = model or configured_model
        if not resolved_model:
            if backend == "openai":
                resolved_model = "gpt-4o-mini"
            else:
                raise ValueError(
                    f"An explicit model is required for LLM backend {backend!r}"
                )
        return build_provider_client(
            provider=backend,
            model=resolved_model,
            request_timeout=request_timeout,
            title=str((extra_headers or {}).get("X-Title") or "EasyICU LLM selector"),
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
    request_timeout: float = 120.0,
    environment: Optional[Mapping[str, str]] = None,
) -> LLMClientSelection:
    """Build the best available LLM client, degrading gracefully.

    The whole point of this factory is concern #1: **a local coding-agent CLI
    (Codex) must be an *optional* engine, never a dependency.**
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
    selected_environment = os.environ if environment is None else environment
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
        if not _backend_available(
            backend,
            api_key=api_key,
            allow_mock=allow_mock,
            environment=selected_environment,
        ):
            continue
        client = _construct_backend(
            backend,
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_headers=extra_headers,
            request_timeout=request_timeout,
            environment=selected_environment,
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
        "Install the Codex CLI, configure an API key, "
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


def llm_is_mockish(client: Any) -> bool:
    """Return true only for a factory-registered intact offline graph."""

    if client is None:
        return False
    from .factory import provider_client_is_mockish

    return provider_client_is_mockish(client)


__all__ = [
    "LLMMessage",
    "LLMClient",
    "OpenAIClient",
    "AnthropicMessagesClient",
    "CLIAgentLLMClient",
    "LLMRouter",
    "LLMClientSelection",
    "build_llm_client",
    "cli_backend_available",
    "llm_is_mockish",
    "llm_supports_vision",
    "openrouter_reasoning_extra_body",
    "reasoning_effort_by_role",
    "resolve_role_client",
]
