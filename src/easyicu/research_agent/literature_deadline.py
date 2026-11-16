"""One wall-clock budget for the pre-plan literature stage.

Each source request already has a per-request timeout, but a search issues
many serial requests, so per-request limits alone let a degraded network
hold plan generation for most of an hour (pilot1, 2026-11-16). The literature
agent sets one deadline for the stage on every live client. Once it has
passed, a client makes no further request and records the reason as a
transport failure, so the bundle's provenance shows which sources were cut
short; the stage itself never reports a complete search it did not make.
"""

from __future__ import annotations

import time
from typing import Optional

LITERATURE_STAGE_SECONDS = 300.0


class LiteratureDeadlineExceeded(RuntimeError):
    """The literature stage's wall-clock budget is spent; no request was made."""


def stage_deadline(seconds: float = LITERATURE_STAGE_SECONDS) -> float:
    return time.monotonic() + float(seconds)


def request_timeout(deadline: Optional[float], timeout: float) -> float:
    """The per-request timeout, shortened to what remains of the stage budget."""
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise LiteratureDeadlineExceeded(
            f"literature stage deadline of {LITERATURE_STAGE_SECONDS:g} s reached before this request"
        )
    return min(float(timeout), remaining)


__all__ = [
    "LITERATURE_STAGE_SECONDS",
    "LiteratureDeadlineExceeded",
    "request_timeout",
    "stage_deadline",
]
