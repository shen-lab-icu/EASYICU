"""Runtime availability signals emitted by concept loading.

These records explain why a concept did or did not produce usable rows for a
specific database.  They deliberately separate structural availability from
true data missingness so downstream agents do not turn unmapped concepts or
missing source tables into misleading "100% missing" claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


ConceptAvailabilityReason = Literal[
    "mapped_present",
    "data_missing",
    "source_unavailable",
    "unmapped",
]
ConceptAvailabilityStatus = Literal["full", "degraded", "blocked"]
ConceptEvaluationStatus = Literal["evaluated", "not_evaluable"]
ConceptEvaluationReason = Literal[
    "no_antibiotics_observed",
    "required_source_table_missing",
    "required_component_missing",
    "antibiotic_time_index_missing",
    "component_time_index_missing",
    "producer_returned_no_physical_output",
]


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
    evaluation_status: Optional[ConceptEvaluationStatus] = None
    reason_code: Optional[ConceptEvaluationReason] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _REASON_TO_STATUS[self.reason])


def table_availability(table: object, concept: str) -> Optional[ConceptAvailabilityRecord]:
    """Read a producer receipt preserved by pandas copies and table wrappers."""
    frame = getattr(table, "data", table)
    payload = getattr(frame, "attrs", {}).get("easyicu_availability")
    if not isinstance(payload, dict) or payload.get("concept") != concept:
        return None
    fields = {key: value for key, value in payload.items() if key != "status"}
    return ConceptAvailabilityRecord(**fields)


def set_table_availability(table: object, record: ConceptAvailabilityRecord) -> None:
    """Keep receipts JSON serializable, including when pandas writes Parquet."""
    frame = getattr(table, "data", table)
    frame.attrs["easyicu_availability"] = asdict(record)


def status_for_availability_reason(
    reason: ConceptAvailabilityReason,
) -> ConceptAvailabilityStatus:
    """Map a fine-grained runtime reason onto the existing RA status terms."""

    return _REASON_TO_STATUS[reason]
