"""Typed host authority for patient-grouped cohort materialization.

The Research Agent may request patient-clustered inference, but it must never
guess a physical identifier column or read a private stay-to-patient mapping.
This immutable value is compiled by the host after it verifies that mapping;
the acquisition layer only forwards it to the deterministic materializer.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STAY_COLUMN = "stay_id"
_PRIVATE_GROUP_COLUMN = "__private_patient_group"


class PatientGroupingError(ValueError):
    """A private stay-to-patient grouping cannot be used safely."""


@dataclass(frozen=True, slots=True)
class VerifiedPatientGrouping:
    """A verified local-only stay-to-patient grouping table.

    ``frame`` deliberately remains an internal runtime object.  Callers must
    derive opaque local cluster indices before they pass anything to a model,
    persist an artifact, or construct a public receipt.
    """

    frame: pd.DataFrame
    file_size: int


def _exact_int_series(values: pd.Series, *, label: str) -> pd.Series:
    """Decode private structural identifiers without lossy coercion."""

    def parse(value: object) -> int:
        if isinstance(value, (bool, np.bool_)) or pd.isna(value):
            raise PatientGroupingError(f"{label} must be an exact integer")
        if isinstance(value, (int, np.integer)):
            parsed = int(value)
        elif isinstance(value, str):
            if re.fullmatch(r"-?(0|[1-9][0-9]*)", value) is None or value == "-0":
                raise PatientGroupingError(
                    f"{label} string must be a canonical integer"
                )
            parsed = int(value)
        elif isinstance(value, (float, np.floating)):
            numeric = float(value)
            if (
                not np.isfinite(numeric)
                or not numeric.is_integer()
                or abs(numeric) >= 2**53
            ):
                raise PatientGroupingError(
                    f"{label} floating value is not exactly representable"
                )
            parsed = int(numeric)
        else:
            raise PatientGroupingError(f"{label} must be an exact integer")
        if parsed < np.iinfo(np.int64).min or parsed > np.iinfo(np.int64).max:
            raise PatientGroupingError(f"{label} exceeds int64 bounds")
        return parsed

    return pd.Series(
        (parse(value) for value in values.tolist()),
        index=values.index,
        dtype="int64",
        name=values.name,
    )


def load_verified_patient_grouping(
    binding: "PatientGroupingBinding",
) -> VerifiedPatientGrouping:
    """Read one digest-bound mapping through a stable local file descriptor.

    The returned frame is intentionally private: it contains a raw patient
    grouping key and is for in-process host computation only.  It has no
    serialization helper and must never be projected into provider context or
    public receipts.
    """

    path = binding.mapping_path
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.suffix.lower() != ".parquet"
    ):
        raise PatientGroupingError(
            "patient grouping mapping must be an absolute regular Parquet file"
        )

    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PatientGroupingError(
                "patient grouping mapping must be a regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != binding.mapping_sha256:
                raise PatientGroupingError("patient grouping mapping digest mismatch")
            handle.seek(0)
            table = pd.read_parquet(
                handle,
                columns=[
                    binding.mapping_stay_column,
                    binding.mapping_patient_column,
                ],
                engine="pyarrow",
            )
            after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise PatientGroupingError("patient grouping mapping changed while being read")
    except PatientGroupingError:
        raise
    except (OSError, ValueError) as exc:
        raise PatientGroupingError("cannot read patient grouping mapping") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    mapping = table.rename(
        columns={
            binding.mapping_stay_column: _STAY_COLUMN,
            binding.mapping_patient_column: _PRIVATE_GROUP_COLUMN,
        }
    )
    mapping[_STAY_COLUMN] = _exact_int_series(
        mapping[_STAY_COLUMN], label="patient grouping stay identity"
    )
    mapping[_PRIVATE_GROUP_COLUMN] = _exact_int_series(
        mapping[_PRIVATE_GROUP_COLUMN], label="patient grouping patient identity"
    )
    if mapping[_STAY_COLUMN].duplicated().any():
        raise PatientGroupingError(
            "patient grouping mapping contains duplicate stay identifiers"
        )
    return VerifiedPatientGrouping(
        frame=mapping.loc[:, [_STAY_COLUMN, _PRIVATE_GROUP_COLUMN]].copy(),
        file_size=int(before.st_size),
    )


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


__all__ = [
    "PatientGroupingBinding",
    "PatientGroupingError",
    "VerifiedPatientGrouping",
    "load_verified_patient_grouping",
]
