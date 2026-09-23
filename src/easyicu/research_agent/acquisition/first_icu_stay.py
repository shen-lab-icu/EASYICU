"""Typed host authority for a first-ICU-stay-per-patient restriction.

A study may keep only each patient's first ICU stay when the host can prove,
from an official stay table it already binds by digest, which stay came first.
This module owns that proof: the fail-closed ordering rules, the private
coordinate the materializer reads, and the aggregate receipt a reader sees.
The coordinate carries stay identifiers and a flag, never a patient key, and
it covers every stay of the source, so a stay whose patient was first admitted
outside a study's export is never promoted to a first stay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from .patient_grouping import (
    PatientGroupingError,
    exact_integer_series,
    read_digest_bound_parquet,
)


FIRST_ICU_STAY_SCHEMA = "easyicu.derived_first_icu_stay/1"
COORDINATE_STAY_COLUMN = "stay_id"
COORDINATE_FLAG_COLUMN = "first_icu_stay"
ORDER_RULE = "earliest_order_time_per_patient"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FirstIcuStayError(ValueError):
    """A first-ICU-stay coordinate cannot be derived or used safely."""

    def __init__(self, message: str, *, code: str = "first_icu_stay_coordinate_invalid"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class DerivedFirstIcuStay:
    """Every official stay, flagged when it is its patient's first ICU stay.

    ``frame`` holds no patient key, but it is still private host state: it
    lists the source's stays.  ``receipt`` is aggregate-only.
    """

    frame: pd.DataFrame
    receipt: Mapping[str, object]


def derive_first_icu_stay(
    identity_table: pd.DataFrame,
    *,
    stay_column: str,
    patient_column: str,
    order_column: str,
    identity_table_name: str,
) -> DerivedFirstIcuStay:
    """Mark each patient's first ICU stay in one verified official table.

    The order is the table's own admission time.  A patient with one stay
    needs no order.  For a patient with several, a missing or unparseable time,
    or two stays sharing the earliest time, makes the whole coordinate
    unusable: breaking the tie by stay number, or dropping that patient, would
    silently change the population.
    """

    missing = [
        column
        for column in (stay_column, patient_column, order_column)
        if column not in identity_table.columns
    ]
    if missing:
        raise FirstIcuStayError(
            "official stay table lacks its declared columns: " + ", ".join(missing),
            code="first_icu_stay_identity_columns_missing",
        )
    try:
        stays = exact_integer_series(
            identity_table[stay_column], label="official stay identity"
        )
    except PatientGroupingError as exc:
        raise FirstIcuStayError(
            str(exc), code="first_icu_stay_stay_identity_invalid"
        ) from exc
    if bool(stays.duplicated().any()):
        raise FirstIcuStayError(
            "official stay table repeats a stay identifier",
            code="first_icu_stay_stay_identity_duplicate",
        )
    patients = identity_table[patient_column].astype("string")
    if bool(patients.isna().any()) or bool(patients.str.strip().eq("").any()):
        raise FirstIcuStayError(
            "official patient identifiers must all be present and non-empty",
            code="first_icu_stay_patient_identifier_missing",
        )
    frame = pd.DataFrame(
        {
            "stay": stays.to_numpy(dtype="int64"),
            "patient": patients.to_numpy(),
            "order": pd.to_datetime(
                identity_table[order_column], errors="coerce"
            ).to_numpy(),
        }
    )
    repeated = frame.groupby("patient", sort=False)["stay"].transform("size").gt(1)
    if bool(frame.loc[repeated, "order"].isna().any()):
        raise FirstIcuStayError(
            "a patient with repeated stays has a stay without a usable admission time",
            code="first_icu_stay_order_time_missing",
        )
    first = pd.Series(True, index=frame.index)
    if bool(repeated.any()):
        several = frame.loc[repeated]
        earliest = several.groupby("patient", sort=False)["order"].transform("min")
        first.loc[repeated] = several["order"].eq(earliest)
        per_patient = first.loc[repeated].groupby(several["patient"], sort=False).sum()
        if bool(per_patient.gt(1).any()):
            raise FirstIcuStayError(
                "two stays of one patient share the earliest admission time",
                code="first_icu_stay_order_tied",
            )
    coordinate = (
        pd.DataFrame(
            {
                COORDINATE_STAY_COLUMN: frame["stay"].to_numpy(dtype="int64"),
                COORDINATE_FLAG_COLUMN: first.to_numpy(dtype=bool),
            }
        )
        .sort_values(COORDINATE_STAY_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )
    first_stays = int(coordinate[COORDINATE_FLAG_COLUMN].sum())
    receipt = {
        "schema_version": FIRST_ICU_STAY_SCHEMA,
        "scope": "source_global",
        "identity_table": str(identity_table_name),
        "stay_column": str(stay_column),
        "patient_identifier_column": str(patient_column),
        "order_column": str(order_column),
        "order_rule": ORDER_RULE,
        "tie_policy": "fail_closed",
        "missing_order_policy": "fail_closed_when_patient_has_repeated_stays",
        "stays": int(len(coordinate)),
        "patients": int(frame["patient"].nunique()),
        "first_icu_stays": first_stays,
        "non_first_icu_stays": int(len(coordinate)) - first_stays,
        "identifier_values_returned": False,
    }
    return DerivedFirstIcuStay(frame=coordinate, receipt=receipt)


@dataclass(frozen=True, slots=True)
class FirstIcuStayBinding:
    """One digest-bound private first-stay coordinate used by materialization."""

    coordinate_path: Path
    coordinate_sha256: str
    authority_coordinates: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.coordinate_path).expanduser()
        if not path.is_absolute():
            raise ValueError("first ICU stay coordinate path must be absolute")
        if _SHA256.fullmatch(str(self.coordinate_sha256 or "")) is None:
            raise ValueError("first ICU stay coordinate sha256 is invalid")
        object.__setattr__(self, "coordinate_path", path)
        object.__setattr__(
            self,
            "authority_coordinates",
            MappingProxyType(dict(self.authority_coordinates)),
        )

    def materializer_kwargs(self) -> dict[str, object]:
        """Return the exact materializer coordinates."""

        return {
            "first_icu_stay_path": self.coordinate_path,
            "first_icu_stay_sha256": self.coordinate_sha256,
            "first_icu_stay_authority_coordinates": dict(self.authority_coordinates),
        }


def load_verified_first_icu_stay(
    path: Path, *, expected_sha256: str
) -> pd.DataFrame:
    """Read one digest-bound coordinate as ``stay_id`` and its first-stay flag."""

    table, _size = read_digest_bound_parquet(
        Path(path),
        expected_sha256=expected_sha256,
        columns=[COORDINATE_STAY_COLUMN, COORDINATE_FLAG_COLUMN],
        label="first ICU stay coordinate",
        error=FirstIcuStayError,
    )
    try:
        stays = exact_integer_series(
            table[COORDINATE_STAY_COLUMN], label="first ICU stay coordinate stay"
        )
    except PatientGroupingError as exc:
        raise FirstIcuStayError(str(exc)) from exc
    if bool(stays.duplicated().any()):
        raise FirstIcuStayError("first ICU stay coordinate repeats a stay identifier")
    flags = table[COORDINATE_FLAG_COLUMN]
    if not pd.api.types.is_bool_dtype(flags) or bool(flags.isna().any()):
        raise FirstIcuStayError("first ICU stay flags must be boolean")
    return pd.DataFrame(
        {
            COORDINATE_STAY_COLUMN: stays.to_numpy(dtype="int64"),
            COORDINATE_FLAG_COLUMN: flags.to_numpy(dtype=bool),
        }
    )


__all__ = [
    "COORDINATE_FLAG_COLUMN",
    "COORDINATE_STAY_COLUMN",
    "DerivedFirstIcuStay",
    "FIRST_ICU_STAY_SCHEMA",
    "FirstIcuStayBinding",
    "FirstIcuStayError",
    "derive_first_icu_stay",
    "load_verified_first_icu_stay",
]
