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


SCHEMA = "easyicu.derived_first_icu_stay_per_hospitalization/1"
RESTRICTION_SCHEMA = "easyicu.first_icu_stay_per_hospitalization_restriction/1"
FLAG = "first_icu_stay_per_hospitalization"
REASON = "first_icu_stay_per_hospitalization_reason"
SELECTION_RULE = "minimum_declared_icu_unitvisitnumber_per_hospitalization"
ICU_UNIT_TYPES = (
    "MICU", "SICU", "Med-Surg ICU", "CCU-CTICU", "CTICU", "CSICU", "Cardiac ICU",
    "Cardiovascular ICU", "Neuro ICU", "CCU", "ICU", "VICU", "Virtual ICU",
    "Mobile ICU", "Vent ICU", "Burn-Trauma ICU",
)
NON_ICU_UNIT_TYPES = ("SDU/Step down",)
SELECTION_POLICY = {
    "selection_rule": SELECTION_RULE,
    "missing_hospitalization_policy": "exclude_rows_with_receipt",
    "invalid_or_duplicate_ordinal_policy": "exclude_hospitalization_with_receipt",
    "unknown_unit_type_policy": "exclude_hospitalization_with_receipt",
    "prior_unit_coverage_policy": "require_all_ordinals_one_through_selected_icu",
    "non_icu_unit_policy": "exclude_declared_non_icu_units",
    "icu_unit_types": list(ICU_UNIT_TYPES), "non_icu_unit_types": list(NON_ICU_UNIT_TYPES),
    "establishes_patient_independence": False,
}
UNKNOWN_REASONS = frozenset({
    "hospitalization_identifier_missing_or_invalid",
    "hospitalization_unit_ordinal_missing_or_invalid",
    "hospitalization_unit_ordinal_duplicate",
    "hospitalization_prior_unit_coverage_incomplete",
    "hospitalization_unit_type_missing_or_unknown",
})


class FirstIcuStayPerHospitalizationError(ValueError):
    """The declared hospital-level source coordinate cannot be verified."""


@dataclass(frozen=True, slots=True)
class DerivedFirstIcuStayPerHospitalization:
    frame: pd.DataFrame
    receipt: Mapping[str, object]


def _positive_integers(values: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(values, errors="coerce")
    valid = numbers.notna() & numbers.gt(0) & numbers.le(2**53 - 1) & numbers.mod(1).eq(0)
    valid &= ~values.map(lambda item: isinstance(item, bool))
    return numbers.where(valid).astype("Int64")


def derive_first_icu_stay_per_hospitalization(table: pd.DataFrame) -> DerivedFirstIcuStayPerHospitalization:
    """Keep the earliest declared ICU in each unambiguous hospitalization.

    Missing/invalid hospital identifiers exclude their rows. Any missing,
    invalid or duplicated ordinal excludes that entire hospitalization. An
    ICU candidate needs all prior ordinals present and classified: a later
    observed ICU is never promoted across unseen earlier units. Missing or
    unknown unit types exclude the hospital. Duplicate/missing stay identity
    refuses the whole coordinate. The unit taxonomy is explicit in the receipt.
    """
    columns = {"patientunitstayid", "patienthealthsystemstayid", "unitvisitnumber", "unittype"}
    if not columns <= set(table):
        raise FirstIcuStayPerHospitalizationError("hospital first-stay source columns are missing")
    try:
        stays = exact_integer_series(table["patientunitstayid"], label="official ICU stay identity")
    except PatientGroupingError as exc:
        raise FirstIcuStayPerHospitalizationError(str(exc)) from exc
    if stays.duplicated().any() or stays.le(0).any():
        raise FirstIcuStayPerHospitalizationError("official ICU stay identities must be unique and positive")
    frame = pd.DataFrame({
        "stay_id": stays.to_numpy(dtype="int64"),
        "hospital": _positive_integers(table["patienthealthsystemstayid"]).to_numpy(),
        "ordinal": _positive_integers(table["unitvisitnumber"]).to_numpy(),
        "unit_type": table["unittype"].astype("string").str.strip().str.casefold().to_numpy(),
    })
    reason = pd.Series("", index=frame.index, dtype="string")
    reason.loc[frame["hospital"].isna()] = "hospitalization_identifier_missing_or_invalid"
    groups = frame.loc[frame["hospital"].notna()]
    invalid = set(groups.loc[groups["ordinal"].isna(), "hospital"])
    reason.loc[frame["hospital"].isin(invalid)] = "hospitalization_unit_ordinal_missing_or_invalid"
    duplicate = groups.duplicated(["hospital", "ordinal"], keep=False)
    duplicate_groups = set(groups.loc[duplicate, "hospital"])
    reason.loc[reason.eq("") & frame["hospital"].isin(duplicate_groups)] = "hospitalization_unit_ordinal_duplicate"
    icu = frame["unit_type"].isin({value.casefold() for value in ICU_UNIT_TYPES})
    non_icu = frame["unit_type"].isin({value.casefold() for value in NON_ICU_UNIT_TYPES})
    unknown_type_groups = set(frame.loc[~(icu | non_icu) & frame["hospital"].notna(), "hospital"])
    reason.loc[reason.eq("") & frame["hospital"].isin(unknown_type_groups)] = "hospitalization_unit_type_missing_or_unknown"
    candidate = frame.loc[icu].groupby("hospital")["ordinal"].min()
    first_icu_ordinal = frame["hospital"].map(candidate)
    preceding = frame.loc[frame["ordinal"].lt(first_icu_ordinal)].groupby("hospital").size()
    prior_count = frame["hospital"].map(preceding).fillna(0)
    gap = first_icu_ordinal.notna() & prior_count.ne(first_icu_ordinal - 1)
    reason.loc[reason.eq("") & gap] = "hospitalization_prior_unit_coverage_incomplete"
    eligible = reason.eq("")
    reason.loc[eligible & non_icu] = "non_icu_unit"
    reason.loc[eligible & icu & frame["ordinal"].eq(first_icu_ordinal)] = "kept"
    reason.loc[eligible & icu & ~frame["ordinal"].eq(first_icu_ordinal)] = "non_first_icu_stay"
    coordinate = pd.DataFrame({
        "stay_id": frame["stay_id"], FLAG: reason.eq("kept"), REASON: reason,
    }).sort_values("stay_id", kind="mergesort").reset_index(drop=True)
    counts = {str(key): int(value) for key, value in reason.value_counts().items() if key != "kept"}
    receipt = {
        "schema_version": SCHEMA, "scope": "source_global", "identity_table": "patient",
        "hospitalization_identifier_column": "patienthealthsystemstayid",
        "order_column": "unitvisitnumber", "unit_type_column": "unittype",
        **SELECTION_POLICY,
        "stays": int(len(frame)), "hospitalizations": int(groups["hospital"].nunique()),
        "first_icu_stays": int(coordinate[FLAG].sum()),
        "non_first_icu_stays": counts.get("non_first_icu_stay", 0),
        "non_icu_stays": counts.get("non_icu_unit", 0),
        "unknown_eligibility_stays": sum(counts.get(key, 0) for key in UNKNOWN_REASONS),
        "excluded_reason_counts": counts, "identifier_values_returned": False,
    }
    return DerivedFirstIcuStayPerHospitalization(coordinate, receipt)


def load_verified_first_icu_stay_per_hospitalization(path: Path, *, expected_sha256: str) -> pd.DataFrame:
    frame, _ = read_digest_bound_parquet(
        path, expected_sha256=expected_sha256, columns=["stay_id", FLAG, REASON],
        label="first ICU stay per hospitalization", error=FirstIcuStayPerHospitalizationError,
    )
    try:
        frame["stay_id"] = exact_integer_series(frame["stay_id"], label="coordinate stay")
    except PatientGroupingError as exc:
        raise FirstIcuStayPerHospitalizationError(str(exc)) from exc
    if frame["stay_id"].duplicated().any() or frame["stay_id"].le(0).any():
        raise FirstIcuStayPerHospitalizationError("coordinate stay identities must be unique and positive")
    if not pd.api.types.is_bool_dtype(frame[FLAG]) or frame[FLAG].isna().any():
        raise FirstIcuStayPerHospitalizationError("coordinate first-stay flags must be boolean")
    if not frame[REASON].isin({"kept", "non_first_icu_stay", "non_icu_unit", *UNKNOWN_REASONS}).all():
        raise FirstIcuStayPerHospitalizationError("coordinate has an undeclared exclusion reason")
    if not frame[FLAG].eq(frame[REASON].eq("kept")).all():
        raise FirstIcuStayPerHospitalizationError("coordinate flags disagree with exclusion reasons")
    return frame


@dataclass(frozen=True, slots=True)
class FirstIcuStayPerHospitalizationBinding:
    coordinate_path: Path
    coordinate_sha256: str
    authority_coordinates: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(self.coordinate_path).expanduser()
        if not path.is_absolute() or re.fullmatch(r"[0-9a-f]{64}", self.coordinate_sha256) is None:
            raise ValueError("hospital first-stay coordinate requires an absolute path and SHA-256")
        object.__setattr__(self, "coordinate_path", path)
        object.__setattr__(self, "authority_coordinates", MappingProxyType(dict(self.authority_coordinates)))

    def materializer_kwargs(self) -> dict[str, object]:
        return {
            "first_icu_stay_per_hospitalization_path": self.coordinate_path,
            "first_icu_stay_per_hospitalization_sha256": self.coordinate_sha256,
            "first_icu_stay_per_hospitalization_authority_coordinates": dict(self.authority_coordinates),
        }


def restrict_to_first_icu_stay_per_hospitalization(
    cohort: pd.DataFrame, *, coordinate_path: Path, coordinate_sha256: str,
    authority_coordinates: Mapping[str, object] | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    coordinate = load_verified_first_icu_stay_per_hospitalization(
        coordinate_path, expected_sha256=coordinate_sha256,
    )
    if {FLAG, REASON} & set(cohort):
        raise FirstIcuStayPerHospitalizationError("hospital first-stay coordinate would overwrite cohort columns")
    joined = cohort.merge(coordinate, on="stay_id", how="left", validate="one_to_one", sort=False)
    if joined[FLAG].isna().any():
        raise FirstIcuStayPerHospitalizationError("hospital first-stay coordinate does not cover all cohort stays")
    keep = joined[FLAG].astype(bool).to_numpy()
    result = cohort.loc[keep].reset_index(drop=True)
    counts = {str(key): int(value) for key, value in joined.loc[~keep, REASON].value_counts().items()}
    return result, {
        "schema_version": RESTRICTION_SCHEMA, "coordinate_sha256": coordinate_sha256,
        "stays_before": int(len(cohort)), "stays_after": int(len(result)),
        "non_first_icu_stays_removed": counts.get("non_first_icu_stay", 0),
        "non_icu_stays_removed": counts.get("non_icu_unit", 0),
        "unknown_eligibility_stays_removed": sum(counts.get(key, 0) for key in UNKNOWN_REASONS),
        "excluded_reason_counts": counts, "authority_coordinates": dict(authority_coordinates or {}),
    }
