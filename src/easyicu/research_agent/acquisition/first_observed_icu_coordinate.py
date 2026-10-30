"""Private reader for the conservative source-visible eICU subset.

This primitive does not create a coordinate, approve a scope, or authorize a
research run. Its caller must first verify a host-reviewed preparation grant,
source completeness, and the full research plan. The coordinate is never a
substitute for a patient-wide or lifetime first-ICU assertion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .eicu_first_observed_candidate import FLAG_COLUMN, REASON_COLUMN
from .patient_grouping import PatientGroupingError, exact_integer_series, read_digest_bound_parquet


class FirstObservedCoordinateError(ValueError):
    """An outcome-blind coordinate is malformed or no longer matches its bytes."""


_REASONS = frozenset({
    "kept", "non_selected_unit", "hospitalization_identifier_missing_or_invalid",
    "hospitalization_identifier_shared_across_patients", "multiple_observed_hospitalizations",
    "first_icu_order_or_type_unverifiable", "age_unknown_or_invalid", "under_18",
})


@dataclass(frozen=True, slots=True)
class FirstObservedIcuCoordinate:
    """Host-private bytes and exact source/review digests, not an approval."""

    path: Path
    sha256: str
    archive_sha256: str
    patient_table_sha256: str
    export_manifest_sha256: str
    plan_sha256: str
    preparation_checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise ValueError("first-observed coordinate path must be absolute")
        for name in ("sha256", "archive_sha256", "patient_table_sha256", "export_manifest_sha256", "plan_sha256", "preparation_checkpoint_sha256"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"invalid first-observed {name}")
        object.__setattr__(self, "path", Path(self.path))


def restrict_to_first_observed_icu(cohort: pd.DataFrame, *, coordinate: FirstObservedIcuCoordinate) -> pd.DataFrame:
    """Return source-visible selected stays, or fail on any uncovered stay.

    The result and coordinate remain host-private. No exclusion counts or
    row-level data are emitted; a separate disclosure policy must prepare any
    cohort-flow report, including suppression of small/inferable cells.
    """
    if {FLAG_COLUMN, REASON_COLUMN} & set(cohort):
        raise FirstObservedCoordinateError("coordinate would overwrite cohort columns")
    try:
        identities = exact_integer_series(cohort["stay_id"], label="cohort stay")
        rows, _ = read_digest_bound_parquet(
            coordinate.path, expected_sha256=coordinate.sha256,
            columns=["stay_id", FLAG_COLUMN, REASON_COLUMN],
            label="first observed ICU coordinate", error=FirstObservedCoordinateError,
        )
        rows["stay_id"] = exact_integer_series(rows["stay_id"], label="coordinate stay")
    except (KeyError, PatientGroupingError) as exc:
        raise FirstObservedCoordinateError("stay identities are invalid") from exc
    if identities.isna().any() or identities.le(0).any() or identities.duplicated().any():
        raise FirstObservedCoordinateError("cohort stays must be unique and positive")
    if rows["stay_id"].isna().any() or rows["stay_id"].le(0).any() or rows["stay_id"].duplicated().any():
        raise FirstObservedCoordinateError("coordinate stays must be unique and positive")
    if not pd.api.types.is_bool_dtype(rows[FLAG_COLUMN]) or rows[FLAG_COLUMN].isna().any():
        raise FirstObservedCoordinateError("coordinate selection must be boolean")
    if not rows[REASON_COLUMN].isin(_REASONS).all() or not rows[FLAG_COLUMN].eq(rows[REASON_COLUMN].eq("kept")).all():
        raise FirstObservedCoordinateError("coordinate reasons disagree with selection")
    ordered = pd.DataFrame({"stay_id": identities.to_numpy()}).merge(rows, on="stay_id", how="left", validate="one_to_one", sort=False)
    if ordered[FLAG_COLUMN].isna().any():
        raise FirstObservedCoordinateError("coordinate does not cover every cohort stay")
    return cohort.loc[ordered[FLAG_COLUMN].to_numpy(dtype=bool)].reset_index(drop=True)
