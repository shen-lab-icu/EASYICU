"""Development-only call, token, and wall-clock budget for Planner."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PlannerEfficiencyLimits:
    """Hard limits for one development Planner run."""

    max_calls: int
    max_reported_tokens: int
    max_wall_seconds: float

    def __post_init__(self) -> None:
        if self.max_calls <= 0:
            raise ValueError("Planner efficiency max_calls must be positive")
        if self.max_reported_tokens <= 0:
            raise ValueError(
                "Planner efficiency max_reported_tokens must be positive"
            )
        if self.max_wall_seconds <= 0:
            raise ValueError(
                "Planner efficiency max_wall_seconds must be positive"
            )


class PlannerEfficiencyBudgetExhausted(RuntimeError):
    """The next Planner request would exceed its development envelope."""

    code = "planner_efficiency_budget_exhausted"
    reason_code = code
    owner = "easyicu.providers.planner_efficiency_budget_v1"

    def __init__(self, *, reason: str, snapshot: Mapping[str, Any]) -> None:
        self.reason = str(reason)
        self.snapshot = dict(snapshot)
        self.easyicu_safe_diagnostic = {
            "owner": self.owner,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "calls": int(self.snapshot.get("calls") or 0),
            "reported_tokens": int(
                self.snapshot.get("reported_tokens") or 0
            ),
            "elapsed_seconds": float(
                self.snapshot.get("elapsed_seconds") or 0.0
            ),
            "limits": dict(self.snapshot.get("limits") or {}),
        }
        super().__init__(f"{self.reason_code}:{self.reason}")


class PlannerEfficiencyBudgetClient:
    """Count every Planner transport call and its provider-reported usage."""

    name = "planner_efficiency_budget"

    def __init__(self, inner: Any, *, limits: PlannerEfficiencyLimits) -> None:
        self._inner = inner
        self._limits = limits
        self._started_at = time.monotonic()
        self._calls = 0
        self._reported_tokens = 0
        self._usage_available = True
        self._lock = threading.Lock()
        from .factory import _register_provider_wrapper

        _register_provider_wrapper(self, children_getter=lambda: (self._inner,))

    def iter_clients(self):
        """Expose the wrapped client to provider trust inspection."""

        inner_iter = getattr(self._inner, "iter_clients", None)
        if callable(inner_iter):
            yield from inner_iter()
        else:
            yield self._inner

    @property
    def limits(self) -> PlannerEfficiencyLimits:
        return self._limits

    def efficiency_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": "easyicu.planner_efficiency_budget/1",
                "owner": "easyicu.providers.planner_efficiency_budget_v1",
                "calls": self._calls,
                "reported_tokens": self._reported_tokens,
                "elapsed_seconds": round(
                    max(0.0, time.monotonic() - self._started_at), 6
                ),
                "usage_available": self._usage_available,
                "limits": asdict(self._limits),
            }

    def _raise(self, reason: str) -> None:
        raise PlannerEfficiencyBudgetExhausted(
            reason=reason,
            snapshot=self.efficiency_snapshot(),
        )

    def _before_call(self) -> None:
        with self._lock:
            elapsed = max(0.0, time.monotonic() - self._started_at)
            if elapsed >= self._limits.max_wall_seconds:
                reason = "wall_clock_limit"
            elif self._calls >= self._limits.max_calls:
                reason = "call_limit"
            elif not self._usage_available:
                reason = "provider_usage_unavailable"
            elif self._reported_tokens >= self._limits.max_reported_tokens:
                reason = "reported_token_limit"
            else:
                self._calls += 1
                return
        self._raise(reason)

    def _record_usage(self, usage: Any) -> None:
        reported: int | None = None
        if isinstance(usage, Mapping):
            try:
                candidate = int(usage.get("total_tokens"))
            except (TypeError, ValueError, OverflowError):
                candidate = -1
            if candidate >= 0:
                reported = candidate
        with self._lock:
            if reported is None:
                self._usage_available = False
            else:
                self._reported_tokens += reported

    def complete(
        self,
        messages: Sequence[Any],
        **kwargs: Any,
    ) -> str:
        response, _usage = self.complete_with_usage(messages, **kwargs)
        return response

    def complete_with_usage(
        self,
        messages: Sequence[Any],
        **kwargs: Any,
    ) -> tuple[str, Mapping[str, Any] | None]:
        self._before_call()
        complete_with_usage = getattr(self._inner, "complete_with_usage", None)
        if callable(complete_with_usage):
            response, usage = complete_with_usage(messages, **kwargs)
        else:
            response = self._inner.complete(messages, **kwargs)
            usage = None
        self._record_usage(usage)
        return response, usage

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_planner_efficiency_budget(
    inner: Any,
    *,
    max_calls: int | None,
    max_reported_tokens: int | None,
    max_wall_seconds: float | None,
) -> Any:
    """Apply the development Planner envelope when it is configured."""

    if max_calls is None:
        return inner
    return PlannerEfficiencyBudgetClient(
        inner,
        limits=PlannerEfficiencyLimits(
            max_calls=int(max_calls),
            max_reported_tokens=int(max_reported_tokens or 0),
            max_wall_seconds=float(max_wall_seconds or 0.0),
        ),
    )


__all__ = [
    "PlannerEfficiencyBudgetClient",
    "PlannerEfficiencyBudgetExhausted",
    "PlannerEfficiencyLimits",
    "wrap_planner_efficiency_budget",
]
