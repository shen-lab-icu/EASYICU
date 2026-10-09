"""The waits a caller states between provider transport attempts.

Owner
-----
This module owns one transport rule: after a failure the client's status
allowlist retries, how long a reviewed client waits before its next attempt,
and when it stops waiting.  The allowlist still decides which failures are
retried; a schedule decides only the waits.  A client given no schedule keeps
its historical backoff.

A schedule names one wait per retry, each drawn within ``±jitter_fraction``
of its delay; a provider's Retry-After lengthens a wait, never shortens it.
No retry is attempted when even its shortest wait (the low edge of its jitter
band, or the Retry-After) would end more than ``window_seconds`` after the
first attempt began; a drawn wait is cut to end at the window, so jitter moves
when a retry begins, never whether.  Nor is one attempted when its wait would
use the hard-stop wall clock that remained at the last reservation: that
reservation would be refused, and waiting first would only delay the refusal.
The client records why it stopped on the raised exception
(``easyicu_transport_retry_exhausted``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Optional

RETRY_ATTEMPTS_EXHAUSTED = "transport_retry_attempts_exhausted"
RETRY_WINDOW_EXHAUSTED = "transport_retry_window_exhausted"
RETRY_WALL_CLOCK_EXHAUSTED = "transport_retry_wall_clock_exhausted"
#: Every reason ``easyicu_transport_retry_exhausted`` can name.
RETRY_STOP_REASONS = frozenset(
    {RETRY_ATTEMPTS_EXHAUSTED, RETRY_WINDOW_EXHAUSTED, RETRY_WALL_CLOCK_EXHAUSTED}
)
TRANSPORT_RETRY_POLICY_SCHEMA = "easyicu.provider_transport_policy/4"


@dataclass(frozen=True)
class TransportRetrySchedule:
    """One wait per retry, a jitter fraction and an optional total window."""

    delays_seconds: tuple[float, ...]
    jitter_fraction: float = 0.2
    window_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        delays = tuple(float(value) for value in self.delays_seconds)
        if not delays or not all(math.isfinite(d) and d > 0 for d in delays):
            raise ValueError("transport retry delays must be finite and positive")
        jitter = float(self.jitter_fraction)
        if not (math.isfinite(jitter) and 0 <= jitter < 1):
            raise ValueError("transport retry jitter must be in [0, 1)")
        window = self.window_seconds
        if window is not None and not (math.isfinite(float(window)) and window > 0):
            raise ValueError("transport retry window must be finite and positive")
        object.__setattr__(self, "delays_seconds", delays)
        object.__setattr__(self, "jitter_fraction", jitter)
        if window is not None:
            object.__setattr__(self, "window_seconds", float(window))

    @property
    def max_retries(self) -> int:
        return len(self.delays_seconds)

    def policy_fields(self) -> dict[str, Any]:
        """The schedule as the transport policy records it."""

        return {
            "transport_retry_delays_seconds": list(self.delays_seconds),
            "transport_retry_jitter_fraction": self.jitter_fraction,
            "transport_retry_window_seconds": self.window_seconds,
        }

    def next_wait(
        self,
        *,
        retry_index: int,
        since_start: float,
        since_reservation: float,
        wall_clock_remaining: Optional[float],
        retry_after: Optional[float],
        draw: Callable[[], float],
    ) -> tuple[Optional[float], str]:
        """The wait before retry ``retry_index`` (from 0), or why none follows.

        ``since_start`` is measured from the first attempt's start and
        ``since_reservation`` from the reservation that reported
        ``wall_clock_remaining``.
        """

        if retry_index >= len(self.delays_seconds):
            return None, RETRY_ATTEMPTS_EXHAUSTED
        delay = self.delays_seconds[retry_index]
        shortest = delay * (1 - self.jitter_fraction)
        if retry_after is not None:
            shortest = max(shortest, float(retry_after))
        window = self.window_seconds
        if window is not None and since_start + shortest > window:
            return None, RETRY_WINDOW_EXHAUSTED
        drawn = min(1.0, max(0.0, float(draw())))
        wait = delay * (1 + self.jitter_fraction * (2 * drawn - 1))
        if retry_after is not None:
            wait = max(wait, float(retry_after))
        if window is not None:
            wait = min(wait, window - since_start)
        if (
            wall_clock_remaining is not None
            and since_reservation + wait >= wall_clock_remaining
        ):
            return None, RETRY_WALL_CLOCK_EXHAUSTED
        return wait, ""


def mark_retry_exhausted(exc: BaseException, reason: str) -> None:
    """Record on ``exc`` why the schedule stopped retrying (never raises)."""

    try:
        exc.easyicu_transport_retry_exhausted = reason  # type: ignore[attr-defined]
    except Exception:
        pass


__all__ = [
    "RETRY_ATTEMPTS_EXHAUSTED",
    "RETRY_STOP_REASONS",
    "RETRY_WALL_CLOCK_EXHAUSTED",
    "RETRY_WINDOW_EXHAUSTED",
    "TRANSPORT_RETRY_POLICY_SCHEMA",
    "TransportRetrySchedule",
    "mark_retry_exhausted",
]
