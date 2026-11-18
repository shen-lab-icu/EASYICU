"""Attempt accounting for repairs whose decisions belong to a separate gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

DEFAULT_REPAIR_ATTEMPTS = 3


@dataclass(frozen=True)
class RepairFeedback:
    code: str
    path: str
    facts: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BoundedRepair:
    """Keep the typed input and disposition for every attempt.

    Callers validate candidates with their own gates. A resolved receipt only
    describes the supplied findings, and never authorizes a plan or report.
    """

    def __init__(self, limit: int = DEFAULT_REPAIR_ATTEMPTS):
        if not 0 <= limit <= DEFAULT_REPAIR_ATTEMPTS:
            raise ValueError("repair budget exceeds the shared limit")
        self.receipt: dict[str, Any] = {"max_attempts": limit, "attempts": []}
        self._active: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return len(self.receipt["attempts"]) < self.receipt["max_attempts"]

    @property
    def active(self) -> bool:
        return self._active is not None

    def begin(self, feedback: Sequence[RepairFeedback], **coordinates: Any) -> dict[str, Any]:
        if not self.available or self.active or not feedback:
            raise ValueError("a repair requires typed feedback and an available attempt")
        row = {
            **coordinates,
            "attempt": len(self.receipt["attempts"]) + 1,
            "feedback": [item.as_dict() for item in feedback],
            "status": "running",
        }
        self.receipt["attempts"].append(row)
        self._active = row
        return row

    def finish(
        self,
        status: Literal["resolved", "unresolved", "rejected", "hard_stop"],
        *,
        findings: Sequence[RepairFeedback] = (),
        regressions: Sequence[RepairFeedback] = (),
        **coordinates: Any,
    ) -> bool:
        """Return whether a candidate may replace its predecessor."""
        if self._active is None:
            raise ValueError("no repair attempt is active")
        if regressions and status != "hard_stop":
            status = "rejected"
        self._active.update(
            **coordinates, status=status,
            findings=[item.as_dict() for item in (*findings, *regressions)],
        )
        self._active = None
        return status in {"resolved", "unresolved"}
