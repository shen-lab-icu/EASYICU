"""LLM reproducibility envelope (O20).

A small, opt-in layer that records, for every LLM call the pipeline
makes, enough information to re-drive the same prompts against the
same model with the same sampling settings:

* sha256 of the prompt (every ``LLMMessage`` joined deterministically);
* sha256 of the returned text;
* provider / model string, temperature, max_tokens, requested seed;
* role the call was made under (planner / coder / analyzer / writer /
  literature / reviewer / ...).

The records are persisted to ``reproducibility_envelope.json`` in the
run directory, registered in the :class:`EvidenceStore`, and a
compact summary dict lands in ``AnalysisManifest.reproducibility``.

Design constraints
------------------

* **Opt-in, but cheap.** Even with envelope tracking on, every call
  still goes through to the inner client; we just hash the input and
  output strings. No additional LLM calls, no SDK creep.
* **Provider-agnostic.** ``OpenAIClient`` passes a ``seed`` to the
  OpenAI Chat Completions API when supplied. Providers that ignore
  the field (Anthropic, local models, OpenRouter free tier) are
  still valid; the envelope records the *requested* seed regardless.
* **Honest about determinism.** Temperature > 0 and provider-level
  non-determinism mean that byte-identical replay is never guaranteed
  with hosted APIs. The envelope records enough metadata for a
  reviewer to detect and reason about drift; it does not claim to
  make it impossible.
* **Composable with :mod:`cost`.** ``ReproRecordingClient`` wraps an
  :class:`LLMClient` transparently; wrapping a ``MeteredClient``
  (or wrapping the envelope client with a ``MeteredClient``) both
  work. In the pipeline the envelope sits *outside* the metered
  client so prompt/response hashes cover the exact string the agent
  sent / received, not a possibly-truncated version.

Nothing in this module imports an LLM SDK.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ENVELOPE_SCHEMA_VERSION = "easyicu.reproducibility_envelope/3"


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _canonical_messages(messages: Sequence[Any]) -> str:
    """Deterministically serialise a message list for hashing.

    Each :class:`LLMMessage` has ``role`` and ``content``; we join them
    with a clear separator that cannot appear inside a role name, so
    two different message lists cannot collide by accident.
    """
    parts: List[str] = []
    for m in messages:
        role = getattr(m, "role", None) or "user"
        content = getattr(m, "content", None) or ""
        parts.append(f"<<<{role}>>>\n{content}")
    return "\n<<<END>>>\n".join(parts)


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def sha256_messages(messages: Sequence[Any]) -> str:
    return sha256_text(_canonical_messages(messages))


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------


_SAFE_ENV_VARS = (
    "EASYICU_RUNNER_IMAGE",
    "EASYICU_DOCKER_EXECUTABLE",
    "EASYICU_LLM_DEBUG_DIR",
    "EASYICU_HOSTED_DEFAULT_MODEL",
    "PYTHONHASHSEED",
    "TZ",
    "LANG",
    "LC_ALL",
    "MPLBACKEND",
)


def build_environment_snapshot() -> Dict[str, Any]:
    """Return a provenance snapshot of the Python process.

    Kept intentionally tiny and PHI-safe: no arbitrary environment
    variables, no working directory, no user name. Only values that
    affect how deterministic Python code runs.
    """
    snap = {
        "python_version": sys.version.split(" ")[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    env_subset = {k: os.environ.get(k) for k in _SAFE_ENV_VARS if os.environ.get(k)}
    if env_subset:
        snap["env"] = env_subset
    # Version-pin markers for libraries the generated code relies on.
    for lib in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib"):
        try:
            mod = __import__(lib)
            snap[f"{lib}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            continue
    return snap


# ---------------------------------------------------------------------------
# Call record + envelope
# ---------------------------------------------------------------------------


@dataclass
class ReproCallRecord:
    """A single ``LLMClient.complete`` invocation fingerprint.

    ``prompt_sha256`` / ``response_sha256`` make it possible to detect
    drift between runs without storing the prompt text itself (keeping
    the envelope small and avoiding any chance of echoing PHI). If a
    user wants to keep the full text, ``preview`` holds the first
    ``preview_max_chars`` characters of each.
    """

    timestamp: str
    role: Optional[str]
    client_name: str
    model: str
    temperature: float
    max_tokens: int
    requested_seed: Optional[int]
    prompt_sha256: str
    response_sha256: str
    prompt_chars: int
    response_chars: int
    reasoning_effort: Optional[str] = None
    elapsed_ms: Optional[float] = None
    # Envelope schema /2: requested_top_p records the top_p value the
    # caller asked for. ``None`` means the caller did not set top_p and
    # the provider's default applies (typically 1.0 for OpenAI-compatible
    # APIs). Recorded explicitly so a reviewer can distinguish "we
    # didn't override top_p" from "top_p value is missing".
    requested_top_p: Optional[float] = None
    prompt_preview: Optional[str] = None
    response_preview: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        payload = {
            "timestamp": self.timestamp,
            "role": self.role,
            "client_name": self.client_name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "requested_seed": self.requested_seed,
            "requested_top_p": self.requested_top_p,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "reasoning_effort": self.reasoning_effort,
            "elapsed_ms": self.elapsed_ms,
        }
        if self.prompt_preview is not None:
            payload["prompt_preview"] = self.prompt_preview
        if self.response_preview is not None:
            payload["response_preview"] = self.response_preview
        return payload


@dataclass
class ReproEnvelope:
    """Collector for :class:`ReproCallRecord` plus a run-level envelope.

    Construct one per pipeline run. Pass it to any
    :class:`ReproRecordingClient`. After the run finishes, call
    :meth:`to_manifest_summary` to get a compact summary suitable for
    ``AnalysisManifest.reproducibility`` and :meth:`to_disk` to
    persist the full envelope.
    """

    run_id: str
    seed: Optional[int] = None
    preview_max_chars: int = 280
    include_previews: bool = False
    calls: List[ReproCallRecord] = field(default_factory=list)
    env_snapshot: Dict[str, Any] = field(default_factory=build_environment_snapshot)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        role: Optional[str],
        client_name: str,
        model: str,
        temperature: float,
        max_tokens: int,
        requested_seed: Optional[int],
        messages: Sequence[Any],
        response: str,
        requested_top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        elapsed_ms: Optional[float] = None,
    ) -> ReproCallRecord:
        prompt_canonical = _canonical_messages(messages)
        prompt_preview: Optional[str] = None
        response_preview: Optional[str] = None
        if self.include_previews:
            prompt_preview = prompt_canonical[: self.preview_max_chars]
            response_preview = (response or "")[: self.preview_max_chars]
        rec = ReproCallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            role=role,
            client_name=client_name,
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            requested_seed=int(requested_seed) if requested_seed is not None else None,
            requested_top_p=(
                float(requested_top_p) if requested_top_p is not None else None
            ),
            prompt_sha256=sha256_text(prompt_canonical),
            response_sha256=sha256_text(response or ""),
            prompt_chars=len(prompt_canonical),
            response_chars=len(response or ""),
            reasoning_effort=reasoning_effort,
            elapsed_ms=(float(elapsed_ms) if elapsed_ms is not None else None),
            prompt_preview=prompt_preview,
            response_preview=response_preview,
        )
        self.calls.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Summary / persistence
    # ------------------------------------------------------------------

    def to_manifest_summary(self) -> Dict[str, Any]:
        """Compact summary suitable for ``AnalysisManifest.reproducibility``."""
        by_role: Dict[str, Dict[str, Any]] = {}
        by_model: Dict[str, Dict[str, Any]] = {}
        temperatures = set()
        seeds = set()
        top_ps = set()
        top_p_was_unset = False
        reasoning_efforts = set()
        elapsed_ms_total = 0.0
        for r in self.calls:
            role_key = r.role or "unrouted"
            rb = by_role.setdefault(
                role_key,
                {"n_calls": 0, "prompt_sha256s": [], "response_sha256s": []},
            )
            rb["n_calls"] += 1
            rb["prompt_sha256s"].append(r.prompt_sha256)
            rb["response_sha256s"].append(r.response_sha256)
            mb = by_model.setdefault(r.model, {"n_calls": 0})
            mb["n_calls"] += 1
            temperatures.add(r.temperature)
            seeds.add(r.requested_seed)
            if r.requested_top_p is None:
                top_p_was_unset = True
            else:
                top_ps.add(r.requested_top_p)
            if r.reasoning_effort is not None:
                reasoning_efforts.add(r.reasoning_effort)
            if r.elapsed_ms is not None:
                elapsed_ms_total += r.elapsed_ms
        summary = {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "n_calls": len(self.calls),
            "prompt_sha256s": [r.prompt_sha256 for r in self.calls],
            "response_sha256s": [r.response_sha256 for r in self.calls],
            "temperatures": sorted(t for t in temperatures if t is not None),
            "requested_seeds": sorted([s for s in seeds if s is not None]) or [],
            "requested_top_ps": sorted(top_ps),
            "top_p_used_provider_default": top_p_was_unset,
            "reasoning_efforts": sorted(reasoning_efforts),
            "recorded_elapsed_ms_total": round(elapsed_ms_total, 3),
            "by_role": by_role,
            "by_model": by_model,
            "env_snapshot": dict(self.env_snapshot),
        }
        return summary

    def to_disk(self, path: Path) -> Path:
        payload = {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "env_snapshot": dict(self.env_snapshot),
            "calls": [c.to_json() for c in self.calls],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Transparent client wrapper
# ---------------------------------------------------------------------------


class ReproRecordingClient:
    """Wraps any :class:`LLMClient` so each ``complete`` is fingerprinted.

    Preserves the ``LLMClient`` protocol: agents that already accept an
    ``LLMClient`` keep working unchanged. When the inner client exposes
    call-scoped usage through ``complete_with_usage``, that usage is returned
    with the same response so :class:`MeteredClient` never has to read shared
    mutable ``last_usage`` state.

    ``seed`` is forwarded to the inner client's ``complete`` via the
    kwarg of the same name *only* when the inner client's signature
    accepts it (OpenAI/OpenRouter does, Anthropic 1.x does not). On
    clients that don't accept seed, the envelope still records the
    *requested* seed so a reviewer can see the user's intent.
    """

    def __init__(
        self,
        inner: Any,
        *,
        role: Optional[str],
        envelope: ReproEnvelope,
        seed: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> None:
        self._inner = inner
        self._role = role
        self._envelope = envelope
        self._seed = seed
        self._model_override = model_override
        from ..providers.factory import _register_provider_wrapper

        _register_provider_wrapper(self, children_getter=lambda: (self._inner,))

    # Protocol attribute some callers read.
    @property
    def name(self) -> str:
        return getattr(self._inner, "name", "recording")

    def _resolve_model(self) -> str:
        if self._model_override:
            return self._model_override
        for attr in ("_model", "model", "name"):
            val = getattr(self._inner, attr, None)
            if isinstance(val, str) and val:
                return val
        return "unknown"

    def _resolve_reasoning_effort(self) -> Optional[str]:
        extra_body = getattr(self._inner, "_extra_body", None)
        if not isinstance(extra_body, dict):
            return None
        reasoning = extra_body.get("reasoning")
        if not isinstance(reasoning, dict):
            return None
        effort = reasoning.get("effort")
        return str(effort) if effort is not None else None

    def _forward_complete(
        self,
        messages: Sequence[Any],
        *,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float] = None,
    ) -> str:
        """Call ``inner.complete``, forwarding ``seed=``/``top_p=`` if accepted."""
        import inspect

        try:
            sig = inspect.signature(self._inner.complete)
            params = sig.parameters
            accepts_seed = "seed" in params
            accepts_top_p = "top_p" in params
        except (TypeError, ValueError):
            accepts_seed = False
            accepts_top_p = False
        kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if accepts_seed and self._seed is not None:
            kwargs["seed"] = self._seed
        if accepts_top_p and top_p is not None:
            kwargs["top_p"] = top_p
        return self._inner.complete(messages, **kwargs)

    def complete(
        self,
        messages,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
    ) -> str:
        response, _usage = self.complete_with_usage(
            messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p
        )
        return response

    def complete_with_usage(
        self,
        messages,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
    ) -> tuple[str, Optional[Dict[str, int]]]:
        """Record one response and return usage owned by that exact call."""
        import inspect

        complete_with_usage = getattr(self._inner, "complete_with_usage", None)
        usage: Optional[Dict[str, int]] = None
        started = time.monotonic()
        if callable(complete_with_usage):
            try:
                params = inspect.signature(complete_with_usage).parameters
                accepts_seed = "seed" in params
                accepts_top_p = "top_p" in params
            except (TypeError, ValueError):
                accepts_seed = False
                accepts_top_p = False
            kwargs: Dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if accepts_seed and self._seed is not None:
                kwargs["seed"] = self._seed
            if accepts_top_p and top_p is not None:
                kwargs["top_p"] = top_p
            response, raw_usage = complete_with_usage(messages, **kwargs)
            if isinstance(raw_usage, dict):
                usage = {
                    str(key): int(value)
                    for key, value in raw_usage.items()
                    if isinstance(value, (int, float))
                }
        else:
            response = self._forward_complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        self._envelope.record(
            role=self._role,
            client_name=getattr(self._inner, "name", type(self._inner).__name__),
            model=self._resolve_model(),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            requested_seed=self._seed,
            requested_top_p=top_p,
            messages=messages,
            response=response,
            reasoning_effort=self._resolve_reasoning_effort(),
            elapsed_ms=elapsed_ms,
        )
        # Compatibility only. Cost attribution uses the returned call-scoped
        # value above and never reads this shared attribute.
        self.last_usage = dict(usage) if usage is not None else None
        if hasattr(self._inner, "last_finish_reason"):
            try:
                self.last_finish_reason = getattr(self._inner, "last_finish_reason")
            except Exception:
                pass
        return response, usage

    # LLMRouter compatibility: if someone wraps a router, still route.
    def for_role(self, role: str):
        if hasattr(self._inner, "for_role"):
            child = self._inner.for_role(role)
            return ReproRecordingClient(
                child,
                role=role,
                envelope=self._envelope,
                seed=self._seed,
                model_override=self._model_override,
            )
        return self

    def iter_clients(self):
        if hasattr(self._inner, "iter_clients"):
            return self._inner.iter_clients()
        return iter([self._inner])


# ---------------------------------------------------------------------------
# Role resolver helper (mirrors metered_role_resolver)
# ---------------------------------------------------------------------------


def envelope_role_resolver(
    llm: Any, envelope: ReproEnvelope, *, seed: Optional[int] = None
):
    """Return a ``role_resolver(role)`` that wraps each per-role client.

    Designed to be composable with :func:`easyicu.research_agent.providers.cost.metered_role_resolver`.
    The typical composition used in the pipeline is::

        resolver = envelope_role_resolver(llm, envelope, seed=seed)
        # inside the pipeline, if cost tracking is on, wrap again:
        resolver = metered_role_resolver(resolver, meter)

    The order matters: recording must see the exact prompt / response
    strings, so it sits closest to the inner client.
    """
    from ..providers.llm import resolve_role_client

    def _resolve(role: str):
        inner = resolve_role_client(llm, role)
        return ReproRecordingClient(
            inner,
            role=role,
            envelope=envelope,
            seed=seed,
        )

    return _resolve


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "ReproCallRecord",
    "ReproEnvelope",
    "ReproRecordingClient",
    "build_environment_snapshot",
    "envelope_role_resolver",
    "sha256_messages",
    "sha256_text",
]
