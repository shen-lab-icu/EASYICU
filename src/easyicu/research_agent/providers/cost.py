"""LLM cost tracking (T3.2).

A small, opt-in metering layer that wraps any :class:`LLMClient` and
records prompt/completion token counts and (when a price table is
available) an estimated USD cost into a :class:`CostMeter`. The
pipeline appends these records to ``AnalysisManifest.cost_records``
so the paper can quote per-role spend without trusting an LLM-emitted
number.

Design constraints:

* **Opt-in.** Default pipeline behaviour is unchanged. Cost tracking
  activates only when ``ResearchAgentPipeline(enable_cost_tracking=True)``.
* **Provider-agnostic.** Real clients (OpenAI / OpenRouter) return usage with
  the same call through ``complete_with_usage``. Clients without that
  call-scoped API fall back to a transparent ``chars / 4`` heuristic — and the record is marked
  ``is_heuristic=True`` so reviewers can tell.
* **No SDK creep.** This module never imports ``openai`` or any provider SDK.
* **Cheap to test.** ``MeteredClient`` is a plain ``LLMClient``
  proxy; tests can exercise it with the mock client.

Built-in price table is conservative — populated only with
publicly-listed prices from major providers as of 2025; adding more
models should be a one-line PR. Numbers are USD per **million** input/
output tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..schema import CostRecord

# Approx 4 chars per token for English / code mix (OpenAI's own rule of
# thumb). Used only when the inner client does not report usage.
_CHARS_PER_TOKEN = 4

# (prompt USD/1M tokens, completion USD/1M tokens) — order matters.
_DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    # OpenAI (cached as of mid-2025; treat as approximate).
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3-mini": (1.10, 4.40),
    # Anthropic via API gateway (configured providers).
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-3-opus-latest": (15.00, 75.00),
    # Common OpenRouter free / cheap aliases (zero-cost rows kept so
    # the meter still records a row even when cost is exactly $0).
    "google/gemini-2.0-flash-exp:free": (0.0, 0.0),
    "meta-llama/llama-3.1-8b-instruct:free": (0.0, 0.0),
    "openai/gpt-oss-120b:free": (0.0, 0.0),
    "openai/gpt-oss-20b:free": (0.0, 0.0),
    # DeepSeek (the reliability-baseline / discovery models for the
    # EasyICU evaluation protocol). APPROXIMATE published API rates —
    # CONFIRM against the current DeepSeek pricing page before quoting in
    # the manuscript; token counts are recorded exactly regardless, and a
    # precise table can always be passed via ``cost_price_table``.
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-v4-flash": (0.27, 1.10),  # APPROX — confirm on pricing page
    "deepseek-v4-pro": (0.55, 2.19),  # APPROX — confirm on pricing page
}


# ---------------------------------------------------------------------------
# Meter
# ---------------------------------------------------------------------------


@dataclass
class CostMeter:
    """Append-only sink for :class:`CostRecord` rows.

    Construct one per pipeline run; pass it to any
    :class:`MeteredClient` you create. After the run finishes, read
    ``meter.records`` and ``meter.summary()`` to populate the manifest
    and the run report.
    """

    price_table: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(_DEFAULT_PRICES)
    )
    records: List[CostRecord] = field(default_factory=list)
    runtime_dir: Optional[Path] = None
    _receipt_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Recover completed call records from durable receipts on resume."""

        if self.records or self.runtime_dir is None:
            return
        receipt_payloads = self._transport_receipt_payloads()
        recovered: List[CostRecord] = []
        for payload in receipt_payloads:
            usage = payload.get("usage")
            if payload.get("state") != "completed" or not isinstance(usage, dict):
                continue
            try:
                prompt_tokens = max(0, int(usage.get("prompt_tokens") or 0))
                completion_tokens = max(
                    0,
                    int(usage.get("completion_tokens") or 0),
                )
                model = str(payload.get("model") or "unknown")
                timestamp = payload.get("started_at")
                recovered.append(
                    CostRecord(
                        timestamp=timestamp,
                        role=(
                            str(payload["role"])
                            if payload.get("role") is not None
                            else None
                        ),
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=max(
                            prompt_tokens + completion_tokens,
                            int(usage.get("total_tokens") or 0),
                        ),
                        estimated_cost_usd=self.estimate_cost(
                            model,
                            prompt_tokens,
                            completion_tokens,
                        ),
                        is_heuristic=bool(usage.get("is_heuristic")),
                    )
                )
            except (TypeError, ValueError):
                continue
        if recovered:
            self.records.extend(recovered)
            return
        legacy = Path(self.runtime_dir).parent / "cost_records.json"
        if not legacy.exists():
            return
        try:
            raw_records = json.loads(legacy.read_text(encoding="utf-8"))
            if isinstance(raw_records, list):
                self.records.extend(
                    CostRecord.model_validate(item)
                    for item in raw_records
                    if isinstance(item, dict)
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return

    def _transport_receipt_payloads(self) -> List[Dict[str, Any]]:
        if self.runtime_dir is None:
            return []
        receipt_dir = Path(self.runtime_dir) / "provider_transport_receipts"
        if not receipt_dir.is_dir():
            return []
        payloads: List[Dict[str, Any]] = []
        for path in sorted(receipt_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
        return sorted(
            payloads,
            key=lambda item: (
                str(item.get("started_at") or ""),
                str(item.get("call_id") or ""),
            ),
        )

    @staticmethod
    def _atomic_write_receipt(path: Path, payload: Dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        raw = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        old_mask = None
        if hasattr(signal, "pthread_sigmask"):
            old_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                {signal.SIGINT, signal.SIGTERM},
            )
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short transport-receipt write")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            if old_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

    def begin_transport(
        self,
        *,
        role: Optional[str],
        model: str,
        messages: Sequence[Any],
        max_tokens: int,
        temperature: float,
    ) -> Optional[tuple[Path, Dict[str, Any]]]:
        """Persist an in-progress PHI-safe receipt before provider delivery."""

        if self.runtime_dir is None:
            return None
        receipt_dir = Path(self.runtime_dir) / "provider_transport_receipts"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(receipt_dir, 0o700)
        call_id = uuid.uuid4().hex
        request_hasher = hashlib.sha256()
        prompt_bytes = 0
        for message in messages:
            role_value = str(getattr(message, "role", "") or "")
            content = str(getattr(message, "content", "") or "")
            encoded = content.encode("utf-8")
            prompt_bytes += len(encoded)
            request_hasher.update(role_value.encode("utf-8"))
            request_hasher.update(b"\0")
            request_hasher.update(encoded)
            request_hasher.update(b"\0")
        # A role client serves several consumers, so the role alone cannot say
        # which one produced this call. Without it an over-budget prompt is
        # visible but unattributable, which is how the analyzer envelope stayed
        # broken. None means the caller is not on a budgeted role.
        from .prompt_budget import active_prompt_consumer

        payload: Dict[str, Any] = {
            "schema_version": "easyicu.provider_transport_receipt/1",
            "call_id": call_id,
            "state": "in_progress",
            "role": role,
            "consumer": active_prompt_consumer.get(),
            "model": model,
            "request_sha256": request_hasher.hexdigest(),
            "prompt_bytes": prompt_bytes,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "error_type": None,
            "usage": None,
            "response_sha256": None,
        }
        path = receipt_dir / f"{call_id}.json"
        with self._receipt_lock:
            self._atomic_write_receipt(path, payload)
        return path, payload

    def finish_transport(
        self,
        receipt: Optional[tuple[Path, Dict[str, Any]]],
        *,
        state: str,
        error_type: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        response: Optional[str] = None,
    ) -> None:
        """Terminalize a transport receipt without persisting prompt/response."""

        if receipt is None:
            return
        path, original = receipt
        payload = dict(original)
        payload.update(
            {
                "state": str(state),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error_type": str(error_type) if error_type else None,
                "usage": dict(usage) if usage is not None else None,
                "response_sha256": (
                    hashlib.sha256(str(response).encode("utf-8")).hexdigest()
                    if response is not None
                    else None
                ),
            }
        )
        with self._receipt_lock:
            self._atomic_write_receipt(path, payload)

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Optional[float]:
        """Return USD cost or ``None`` if the model is not in the price table."""
        prices = self.price_table.get(model)
        if not prices:
            return None
        p_per_1m, c_per_1m = prices
        return (prompt_tokens * p_per_1m + completion_tokens * c_per_1m) / 1_000_000.0

    def record(
        self,
        *,
        role: Optional[str],
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        is_heuristic: bool = False,
    ) -> CostRecord:
        rec = CostRecord(
            role=role,
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens) + int(completion_tokens),
            estimated_cost_usd=self.estimate_cost(
                model, prompt_tokens, completion_tokens
            ),
            is_heuristic=is_heuristic,
        )
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _receipt_usage_accounting(self) -> Dict[str, Any]:
        reported = {
            "n_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
        heuristic = dict(reported)
        unknown = {
            "n_calls": 0,
            "states": {},
        }
        conservative_tokens = 0
        conservative_cost = 0.0
        for payload in self._transport_receipt_payloads():
            usage = payload.get("usage")
            model = str(payload.get("model") or "unknown")
            if isinstance(usage, dict):
                prompt = max(0, int(usage.get("prompt_tokens") or 0))
                completion = max(0, int(usage.get("completion_tokens") or 0))
                total = max(
                    prompt + completion,
                    int(usage.get("total_tokens") or 0),
                )
                bucket = heuristic if bool(usage.get("is_heuristic")) else reported
                bucket["n_calls"] += 1
                bucket["prompt_tokens"] += prompt
                bucket["completion_tokens"] += completion
                bucket["total_tokens"] += total
                cost = self.estimate_cost(model, prompt, completion)
                if cost is not None:
                    bucket["estimated_cost_usd"] += cost
                if not bool(usage.get("is_heuristic")):
                    conservative_tokens += total
                    conservative_cost += cost or 0.0
                    continue
            unknown["n_calls"] += 1
            state = str(payload.get("state") or "unknown")
            unknown["states"][state] = unknown["states"].get(state, 0) + 1
            prompt_reserve = max(0, int(payload.get("prompt_bytes") or 0)) + 4096
            completion_reserve = max(0, int(payload.get("max_tokens") or 0))
            conservative_tokens += prompt_reserve + completion_reserve
            cost = self.estimate_cost(model, prompt_reserve, completion_reserve)
            conservative_cost += cost or 0.0
        for bucket in (reported, heuristic):
            bucket["estimated_cost_usd"] = round(
                float(bucket["estimated_cost_usd"]),
                12,
            )
        unknown["states"] = dict(sorted(unknown["states"].items()))
        return {
            "provider_reported": reported,
            "heuristic": heuristic,
            "usage_unknown": unknown,
            "conservative_upper_bound": {
                "total_tokens": conservative_tokens,
                "estimated_cost_usd": round(conservative_cost, 12),
                "source": "transport_receipt_fallback",
            },
        }

    def summary(
        self,
        *,
        hard_stop_accounting: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        usage_accounting = self._receipt_usage_accounting()
        if isinstance(hard_stop_accounting, dict):
            for key in (
                "provider_reported",
                "usage_unknown",
                "conservative_upper_bound",
            ):
                value = hard_stop_accounting.get(key)
                if isinstance(value, dict):
                    usage_accounting[key] = dict(value)
        if not self.records:
            return {
                "n_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "by_role": {},
                "by_model": {},
                "any_heuristic": False,
                "usage_accounting": usage_accounting,
            }
        by_role: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "n_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        )
        by_model: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "n_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        )
        any_heuristic = False
        for r in self.records:
            for bucket in (by_role[r.role or "unrouted"], by_model[r.model]):
                bucket["n_calls"] += 1
                bucket["prompt_tokens"] += r.prompt_tokens
                bucket["completion_tokens"] += r.completion_tokens
                bucket["total_tokens"] += r.total_tokens
                if r.estimated_cost_usd is not None:
                    bucket["cost_usd"] += r.estimated_cost_usd
            if r.is_heuristic:
                any_heuristic = True
        return {
            "n_calls": len(self.records),
            "total_prompt_tokens": sum(r.prompt_tokens for r in self.records),
            "total_completion_tokens": sum(r.completion_tokens for r in self.records),
            "total_tokens": sum(r.total_tokens for r in self.records),
            "total_cost_usd": sum((r.estimated_cost_usd or 0.0) for r in self.records),
            "by_role": {k: dict(v) for k, v in by_role.items()},
            "by_model": {k: dict(v) for k, v in by_model.items()},
            "any_heuristic": any_heuristic,
            "usage_accounting": usage_accounting,
        }


# ---------------------------------------------------------------------------
# MeteredClient — a transparent proxy that records cost on every call
# ---------------------------------------------------------------------------


class MeteredClient:
    """Wraps an :class:`LLMClient` to record token usage to a :class:`CostMeter`.

    The wrapper preserves the ``LLMClient`` protocol so any agent that
    already accepts an LLMClient continues to work unchanged. Token
    counts come from a call-scoped ``complete_with_usage`` result when present;
    otherwise we fall back to a transparent ``chars/4`` heuristic and mark the
    record so reviewers can tell. Shared mutable ``last_usage`` is never read.
    """

    name = "metered"

    def __init__(
        self,
        inner: Any,
        *,
        role: Optional[str],
        meter: CostMeter,
        model_override: Optional[str] = None,
    ) -> None:
        self._inner = inner
        self._role = role
        self._meter = meter
        self._model_override = model_override
        from .factory import _register_provider_wrapper

        _register_provider_wrapper(self, children_getter=lambda: (self._inner,))

    # The protocol methods the agents call.

    def complete(
        self, messages, *, max_tokens: int = 2048, temperature: float = 0.2
    ) -> str:
        model = self._model_override or _identify_model(self._inner)
        receipt = self._meter.begin_transport(
            role=self._role,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            complete_with_usage = getattr(self._inner, "complete_with_usage", None)
            if callable(complete_with_usage):
                result, usage = complete_with_usage(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                result = self._inner.complete(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # A shared ``last_usage`` attribute is not call-scoped and cannot be
                # read safely under concurrent role calls. Legacy providers use the
                # transparent heuristic until they implement ``complete_with_usage``.
                usage = None
        except BaseException as exc:
            self._meter.finish_transport(
                receipt,
                state=(
                    "cancelled"
                    if isinstance(exc, (KeyboardInterrupt, SystemExit))
                    else "failed"
                ),
                error_type=type(exc).__name__,
            )
            raise
        if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            is_heuristic = False
        else:
            prompt_chars = sum(len(m.content or "") for m in messages)
            completion_chars = len(result or "")
            prompt_tokens = max(1, prompt_chars // _CHARS_PER_TOKEN)
            completion_tokens = max(1, completion_chars // _CHARS_PER_TOKEN)
            is_heuristic = True

        self._meter.record(
            role=self._role,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_heuristic=is_heuristic,
        )
        self._meter.finish_transport(
            receipt,
            state="completed",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "is_heuristic": is_heuristic,
            },
            response=result,
        )
        return result

    # Mirror commonly-touched attributes so existing duck-typing keeps
    # working (e.g. MockLLMClient.context).

    @property
    def context(self):  # pragma: no cover - delegating attribute
        return getattr(self._inner, "context", None)

    @context.setter
    def context(self, value) -> None:  # pragma: no cover - delegating attribute
        try:
            self._inner.context = value
        except Exception:
            pass

    def __getattr__(self, item):  # pragma: no cover - delegating attribute
        return getattr(self._inner, item)


def _identify_model(client: Any) -> str:
    """Best-effort 'what model is this?' for the cost record."""
    for attr in ("_model", "model", "name"):
        v = getattr(client, attr, None)
        if isinstance(v, str) and v:
            return v
    return type(client).__name__


# ---------------------------------------------------------------------------
# Convenience: wrap a router's role lookups with the meter in one go
# ---------------------------------------------------------------------------


def metered_role_resolver(llm: Any, meter: CostMeter):
    """Return a callable ``resolver(role) → MeteredClient`` for the pipeline.

    The resolver delegates to the existing ``resolve_role_client`` so
    a single :class:`LLMClient`, an :class:`LLMRouter`, or a mock all
    work — every returned client is a :class:`MeteredClient` that
    records to the same meter.
    """
    from .llm import resolve_role_client

    def resolver(role: str):
        base = resolve_role_client(llm, role)
        if base is None:
            return None
        if isinstance(base, MeteredClient):
            # Don't stack meters on top of each other.
            return base
        return MeteredClient(base, role=role, meter=meter)

    return resolver


__all__ = [
    "CostMeter",
    "MeteredClient",
    "metered_role_resolver",
]
