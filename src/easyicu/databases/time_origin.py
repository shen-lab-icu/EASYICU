"""Time-origin names for producer-owned native database clocks.

These labels describe configured source semantics; a label alone is neither
evidence that an export consumed the source nor approval to substitute clocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

SIC_CARE_UNIT_ORIGIN = "first_intensive_or_intermediate_care_transfer"
ICU_ADMISSION_ORIGIN = "icu_admission"


@dataclass(frozen=True, slots=True)
class SourceTimeAxisBinding:
    """A host-established source-to-export identity, not a plan approval.

    A caller must verify the registered source identity and the plan approval
    before creating this binding. Its only role here is to pin the physical
    source and axis while reading a particular immutable export package.
    """

    source_identity_sha256: str
    export_authority_sha256: str
    manifest_sha256: str
    selected_origin: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
               for value in (self.source_identity_sha256, self.export_authority_sha256,
                             self.manifest_sha256)):
            raise ValueError("source time-axis binding requires canonical digests")
        if self.selected_origin != SIC_CARE_UNIT_ORIGIN:
            raise ValueError("unsupported source time-axis selection")


def native_export_time_origin(database: str) -> str:
    """Describe the clock used by the native export producer for this database."""
    from .profiles import normalize_database_key

    try:
        key = normalize_database_key(database)
    except KeyError:
        key = str(database).strip().lower()
    return SIC_CARE_UNIT_ORIGIN if key == "sic" else ICU_ADMISSION_ORIGIN
