"""Resolve cohort row, ICU-stay, and patient granularity without guessing.

This owner module intentionally recognises only explicit, well-known patient
identifiers.  A stay identifier can establish the analysis-unit identity, but
it must never be relabelled as a patient identifier merely because it is the
first ID column in a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import pandas as pd

from ..contracts.dependence import (
    PlannedDependenceRequirement,
    PatientGroupResolutionError,
    resolve_patient_groups,
)

__all__ = [
    "CohortGranularity",
    "format_patient_count",
    "resolve_cohort_granularity",
]


_PATIENT_ID_NAMES = frozenset(
    {
        "patient_id",
        "patientid",
        "person_id",
        "subject_id",
        "unique_pid",
        "uniquepid",
    }
)
_STAY_ID_NAMES = frozenset(
    {
        "admissionid",
        "icustay_id",
        "patient_stay_id",
        "patientunitstayid",
        "stay_id",
    }
)


@dataclass(frozen=True)
class CohortGranularity:
    """Immutable, dependency-neutral analysis-unit receipt."""

    analysis_unit: Literal["icu_stay", "row"]
    row_count: int
    stay_id_columns: tuple[str, ...]
    patient_id_columns: tuple[str, ...]
    patient_grouping_source: str | None
    n_patients: int | None

    @property
    def patient_identity_available(self) -> bool:
        return self.n_patients is not None

    def provenance(self) -> dict[str, object]:
        return {
            "analysis_unit": self.analysis_unit,
            "analysis_row_count": self.row_count,
            "stay_id_columns": list(self.stay_id_columns),
            "patient_id_columns": list(self.patient_id_columns),
            "patient_identity_available": self.patient_identity_available,
            "n_patients_source": (
                self.patient_id_columns[0]
                if self.patient_id_columns
                else (
                    f"{self.patient_grouping_source}:prefix_before_:s"
                    if self.patient_grouping_source
                    else "unavailable"
                )
            ),
        }


def format_patient_count(n_patients: int | None) -> str:
    """Render unavailable patient identity without relabeling ICU stays."""

    return (
        f"{n_patients:,}"
        if n_patients is not None
        else "unavailable (no patient identifier)"
    )


def resolve_cohort_granularity(
    *,
    frame: pd.DataFrame,
    id_columns: Sequence[str],
    replacement_row_identity: Mapping[str, object] | None = None,
) -> CohortGranularity:
    """Resolve granularity from exact identifier roles.

    Unknown IDs remain generic IDs.  They are not promoted to patient identity
    because doing so would turn a row or ICU-stay count into a false patient
    count.
    """

    available = tuple(column for column in id_columns if column in frame.columns)
    patient_ids = tuple(
        column
        for column in available
        if column.strip().lower() in _PATIENT_ID_NAMES
    )
    stay_ids = tuple(
        column
        for column in available
        if column.strip().lower() in _STAY_ID_NAMES
    )
    patient_grouping_source: str | None = None
    n_patients = (
        int(frame[patient_ids[0]].nunique(dropna=True)) if patient_ids else None
    )
    if not patient_ids and replacement_row_identity is not None:
        source = replacement_row_identity.get("output_identity_column")
        derivation = replacement_row_identity.get("patient_group_derivation")
        if (
            isinstance(source, str)
            and source in available
            and derivation
            == {"algorithm": "prefix_before_:s", "delimiter": ":s"}
        ):
            patient_grouping_source = source
            # Metadata-only planning catalogs intentionally expose the verified
            # grouping coordinate but contain zero patient rows.  They can
            # authorize a clustered design without making a patient-count claim;
            # execution cohorts below still have to resolve every value.
            if frame.empty:
                return CohortGranularity(
                    analysis_unit="icu_stay" if stay_ids else "row",
                    row_count=0,
                    stay_id_columns=stay_ids,
                    patient_id_columns=patient_ids,
                    patient_grouping_source=patient_grouping_source,
                    n_patients=None,
                )
            try:
                resolved = resolve_patient_groups(
                    frame[source].tolist(),
                    requirement=PlannedDependenceRequirement(
                        group_source=source,
                        group_derivation="prefix_before_delimiter",
                        delimiter=":s",
                    ),
                )
            except PatientGroupResolutionError as exc:
                raise ValueError(
                    "verified replacement patient grouping is invalid"
                ) from exc
            n_patients = resolved.cluster_count
    return CohortGranularity(
        analysis_unit="icu_stay" if stay_ids else "row",
        row_count=int(len(frame)),
        stay_id_columns=stay_ids,
        patient_id_columns=patient_ids,
        patient_grouping_source=patient_grouping_source,
        n_patients=n_patients,
    )
