"""Typed host authority for patient-grouped cohort materialization.

The Research Agent may request patient-clustered inference, but it must never
guess a physical identifier column or read a private stay-to-patient mapping.
This immutable value is compiled by the host after it verifies that mapping;
the acquisition layer only forwards it to the deterministic materializer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class PatientGroupingBinding:
    """One digest-bound private mapping used only during materialization."""

    mapping_path: Path
    mapping_sha256: str
    mapping_stay_column: str
    mapping_patient_column: str
    output_identity_column: str = "patient_stay_id"
    authority_coordinates: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.mapping_path).expanduser()
        if not path.is_absolute():
            raise ValueError("patient grouping mapping path must be absolute")
        if _SHA256.fullmatch(str(self.mapping_sha256 or "")) is None:
            raise ValueError("patient grouping mapping sha256 is invalid")
        for label, value in (
            ("mapping_stay_column", self.mapping_stay_column),
            ("mapping_patient_column", self.mapping_patient_column),
            ("output_identity_column", self.output_identity_column),
        ):
            if _COLUMN.fullmatch(str(value or "")) is None:
                raise ValueError(f"{label} is invalid")
        if self.mapping_stay_column == self.mapping_patient_column:
            raise ValueError("patient grouping mapping columns must be distinct")
        object.__setattr__(self, "mapping_path", path)
        object.__setattr__(
            self,
            "authority_coordinates",
            MappingProxyType(dict(self.authority_coordinates)),
        )

    def materializer_kwargs(self) -> dict[str, object]:
        """Return the exact public materializer coordinates."""

        return {
            "replacement_identity_path": self.mapping_path,
            "replacement_identity_sha256": self.mapping_sha256,
            "replacement_identity_stay_column": self.mapping_stay_column,
            "replacement_identity_patient_column": self.mapping_patient_column,
            "output_identity_column": self.output_identity_column,
            "identity_authority_coordinates": dict(self.authority_coordinates),
        }


__all__ = ["PatientGroupingBinding"]
