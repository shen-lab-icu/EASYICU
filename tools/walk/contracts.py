"""Versioned, terminal results shared by the report and resume receipts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum

from .privacy import metrics_projection, token

SCHEMA_VERSION = "easyicu.walk/1"
PHASES = ("environment", "mapping", "catalogue", "literature", "ideas", "workflows", "interactions")


class Status(str, Enum):
    PASSED = "passed"
    REFUSED = "refused-with-reason"
    FAILED = "failed"
    TIMED_OUT = "timed-out"
    SKIPPED = "skipped-with-reason"


@dataclass(frozen=True)
class ItemResult:
    phase: str
    item_id: str
    status: Status
    reason_code: str
    started_at: float
    elapsed_seconds: float
    metrics: dict

    def __post_init__(self):
        object.__setattr__(self, "status", Status(self.status))
        if self.phase not in PHASES or not self.item_id:
            raise ValueError("invalid_item_identity")
        if self.status != Status.PASSED and not self.reason_code:
            raise ValueError("result_reason_required")
        if any(not math.isfinite(v) or v < 0 for v in (self.started_at, self.elapsed_seconds)):
            raise ValueError("invalid_result_timing")
        object.__setattr__(self, "item_id", token(self.item_id, "redacted"))
        object.__setattr__(self, "reason_code", token(self.reason_code, "") if not self.reason_code else token(self.reason_code))
        object.__setattr__(self, "metrics", metrics_projection(self.metrics))

    def to_dict(self) -> dict:
        return {**asdict(self), "status": self.status.value}


class WalkStop(Exception):
    def __init__(self, status: Status, code: str, metrics: dict | None = None):
        self.status, self.code, self.metrics = status, token(code), metrics or {}
        super().__init__(self.code)
