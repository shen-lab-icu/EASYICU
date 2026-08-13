"""Typed covariance contract shared by planning, execution, and review.

The scientific choice to account for repeated observations belongs to the
study/plan authority.  The executor only consumes the exact grouping source
and deterministic derivation declared here; it never discovers a patient id
from prose or column-name heuristics at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlannedDependenceRequirement(BaseModel):
    """One exact cluster-robust covariance requirement for a planned model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.planned_dependence/1"] = (
        "easyicu.planned_dependence/1"
    )
    variance_estimator: Literal["cluster_robust"] = "cluster_robust"
    cluster_unit: Literal["patient"] = "patient"
    group_source: str = Field(min_length=1, max_length=128)
    group_derivation: Literal["identity", "prefix_before_delimiter"]
    delimiter: str | None = Field(default=None, min_length=1, max_length=16)

    @model_validator(mode="after")
    def _closed_group_derivation(self) -> "PlannedDependenceRequirement":
        source = str(self.group_source or "").strip()
        if source != self.group_source:
            raise ValueError("group_source must be a normalized exact column name")
        if self.group_derivation == "identity" and self.delimiter is not None:
            raise ValueError("identity group derivation must not declare a delimiter")
        if self.group_derivation == "prefix_before_delimiter":
            delimiter = str(self.delimiter or "")
            if not delimiter or delimiter.strip() != delimiter:
                raise ValueError(
                    "prefix_before_delimiter requires a normalized delimiter"
                )
        return self


class PatientGroupResolutionError(ValueError):
    """One declared patient-group value cannot be resolved exactly."""


@dataclass(frozen=True)
class ResolvedPatientGroups:
    """Immutable group vector shared by every covariance consumer.

    Keeping this derivation beside the typed declaration prevents two
    executors from silently assigning the same source values to different
    patients.  Missingness remains the caller's dataframe responsibility; this
    owner handles only the exact, non-missing values it is given.
    """

    groups: tuple[str, ...]
    cluster_count: int


def resolve_patient_groups(
    values: Iterable[Any],
    *,
    requirement: PlannedDependenceRequirement,
) -> ResolvedPatientGroups:
    """Derive one exact patient grouping under the declared contract.

    Identity preserves both Python type and value: integer ``1`` and text
    ``"1"`` are not silently merged.  Prefix derivation accepts only original
    strings; coercing arbitrary values to text would broaden the signed data
    contract differently in different executors.
    """

    resolved: list[str] = []
    delimiter = str(requirement.delimiter or "")
    for value in values:
        if requirement.group_derivation == "identity":
            if isinstance(value, str) and not value.strip():
                raise PatientGroupResolutionError(
                    "declared cluster group identity contains a blank string"
                )
            resolved.append(f"{type(value).__name__}:{value!r}")
            continue
        if not isinstance(value, str) or delimiter not in value:
            raise PatientGroupResolutionError(
                "declared cluster group prefix identity is not an original "
                "delimited string"
            )
        patient, stay = value.split(delimiter, 1)
        if not patient or not stay or not patient.strip() or not stay.strip():
            raise PatientGroupResolutionError(
                "declared cluster group prefix identity has an empty patient or "
                "stay component"
            )
        resolved.append(patient)

    cluster_count = len(set(resolved))
    if cluster_count < 2:
        raise PatientGroupResolutionError(
            "cluster-robust covariance requires at least two patient groups"
        )
    return ResolvedPatientGroups(
        groups=tuple(resolved),
        cluster_count=cluster_count,
    )


__all__ = [
    "PatientGroupResolutionError",
    "PlannedDependenceRequirement",
    "ResolvedPatientGroups",
    "resolve_patient_groups",
]
