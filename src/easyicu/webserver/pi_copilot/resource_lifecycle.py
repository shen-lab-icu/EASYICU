"""Process-tree memory admission for the local Copilot Web lifecycle."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

import psutil

from .contracts import PiCopilotError


def _bounded_int(
    value: object,
    *,
    fallback: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


@dataclass(frozen=True)
class WebMemoryPolicy:
    soft_rss_mb: int
    emergency_rss_mb: int

    @classmethod
    def from_environment(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        *,
        total_memory_mb: Optional[float] = None,
    ) -> "WebMemoryPolicy":
        source = os.environ if environ is None else environ
        detected_total = (
            psutil.virtual_memory().total / (1024**2)
            if total_memory_mb is None
            else float(total_memory_mb)
        )
        total_mb = max(1024, int(detected_total))
        default_soft = max(1024, min(4096, int(total_mb * 0.15)))
        default_emergency = max(
            default_soft + 512,
            min(6144, int(total_mb * 0.25)),
        )
        soft = _bounded_int(
            source.get("EASYICU_WEB_SOFT_RSS_MB"),
            fallback=default_soft,
            minimum=512,
            maximum=1_000_000,
        )
        emergency = _bounded_int(
            source.get("EASYICU_WEB_EMERGENCY_RSS_MB"),
            fallback=default_emergency,
            minimum=soft + 128,
            maximum=1_000_000,
        )
        return cls(soft_rss_mb=soft, emergency_rss_mb=emergency)


def process_tree_rss_mb(pid: Optional[int] = None) -> float:
    """Return current RSS for the Web process and all live descendants."""

    try:
        root = psutil.Process(os.getpid() if pid is None else int(pid))
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError, ValueError):
        return 0.0
    total = 0
    for process in processes:
        try:
            total += int(process.memory_info().rss)
        except (psutil.Error, OSError):
            continue
    return round(total / (1024**2), 1)


class WebMemoryAdmission:
    """Reject new work only at the emergency process-tree high-water mark."""

    def __init__(self, policy: Optional[WebMemoryPolicy] = None) -> None:
        self.policy = policy or WebMemoryPolicy.from_environment()

    def status(self, *, rss_mb: Optional[float] = None) -> dict[str, object]:
        observed = process_tree_rss_mb() if rss_mb is None else max(0.0, float(rss_mb))
        pressure = (
            "emergency"
            if observed >= self.policy.emergency_rss_mb
            else ("soft" if observed >= self.policy.soft_rss_mb else "normal")
        )
        return {
            "process_tree_rss_mb": round(observed, 1),
            "soft_rss_mb": self.policy.soft_rss_mb,
            "emergency_rss_mb": self.policy.emergency_rss_mb,
            "pressure": pressure,
        }

    def require_capacity(self, *, rss_mb: Optional[float] = None) -> dict[str, object]:
        status = self.status(rss_mb=rss_mb)
        if status["pressure"] == "emergency":
            raise PiCopilotError(
                "pi_web_memory_pressure",
                "EasyICU is under memory pressure; retry after current work finishes.",
                status_code=429,
                details=status,
            )
        return status


__all__ = [
    "WebMemoryAdmission",
    "WebMemoryPolicy",
    "process_tree_rss_mb",
]
