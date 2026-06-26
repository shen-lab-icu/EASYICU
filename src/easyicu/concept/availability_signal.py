"""Runtime availability signals emitted by concept loading.

These records explain why a concept did or did not produce usable rows for a
specific database.  They deliberately separate structural availability from
true data missingness so downstream agents do not turn unmapped concepts or
missing source tables into misleading "100% missing" claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


ConceptAvailabilityReason = Literal[
    "mapped_present",
    "data_missing",
    "source_unavailable",
    "unmapped",
]
ConceptAvailabilityStatus = Literal["full", "degraded", "blocked"]


_REASON_TO_STATUS: dict[ConceptAvailabilityReason, ConceptAvailabilityStatus] = {
    "mapped_present": "full",
    "data_missing": "degraded",
    "source_unavailable": "blocked",
    "unmapped": "blocked",
}


@dataclass(frozen=True)
class ConceptAvailabilityRecord:
    """Availability of one concept/database pair from the runtime load path."""

    concept: str
    database: str
    reason: ConceptAvailabilityReason
    status: ConceptAvailabilityStatus = field(init=False)
    n_rows: Optional[int] = None
    sources_defined: tuple[str, ...] = ()
    missing_tables: tuple[str, ...] = ()
    note: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _REASON_TO_STATUS[self.reason])


def status_for_availability_reason(
    reason: ConceptAvailabilityReason,
) -> ConceptAvailabilityStatus:
    """Map a fine-grained runtime reason onto the existing RA status terms."""

    return _REASON_TO_STATUS[reason]
