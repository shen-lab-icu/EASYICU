"""Descriptive size contract for an existing, frozen cluster assignment table."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..canonical_json import canonical_sha256
from .cohort_product_keys import is_closed_cohort_product_key

CLUSTER_SIZE_ACTION = "phenotyping.cluster_sizes"
CLUSTER_SIZE_KIND = "cluster_sizes"
CLUSTER_SIZE_OUTPUTS = ("table:cluster_sizes", "log:cluster_sizes_receipt")


def cluster_sizes_cohort_input(step: Any) -> str | None:
    """Identify the single cohort among this action's several typed sources."""
    cohorts = [key for key in step.inputs if is_closed_cohort_product_key(key)]
    return cohorts[0] if len(cohorts) == 1 else None


class ClusterSizeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["easyicu.cluster_sizes/1"] = "easyicu.cluster_sizes/1"
    unit_id_column: str = Field(min_length=1)
    analysis_unit: Literal["subject", "encounter"]
    assignment_source: Literal[
        "table:phenotype_assignments", "table:cluster_assignments"
    ]
    assignment_unit_id_column: str = Field(min_length=1)
    label_column: str = Field(min_length=1)
    expected_clusters: tuple[str, ...] = Field(min_length=2)
    missing_label_policy: Literal["error", "exclude"]
    proportion_denominator: Literal["assigned_units"] = "assigned_units"
    near_empty_n: int | None = Field(default=None, ge=1, strict=True)
    near_empty_fraction: float | None = Field(default=None, gt=0, lt=1)

    @field_validator("unit_id_column", "assignment_unit_id_column", "label_column")
    @classmethod
    def _exact_column(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("cluster-size coordinates must be nonblank exact names")
        return value

    @model_validator(mode="after")
    def _closed_roster(self) -> "ClusterSizeSpec":
        if self.label_column == self.assignment_unit_id_column:
            raise ValueError("assignment identity and label columns must differ")
        if len(set(self.expected_clusters)) != len(self.expected_clusters):
            raise ValueError("expected clusters must be unique")
        if any(not value or value != value.strip() for value in self.expected_clusters):
            raise ValueError("expected clusters require nonblank exact spellings")
        return self


class ClusterSizeRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    row_role: Literal["cluster"] = "cluster"
    cluster: str
    n_units: int = Field(ge=0)
    proportion_assigned: float = Field(ge=0, le=1)
    proportion_all_units: float = Field(ge=0, le=1)
    is_empty: bool
    is_near_empty: bool | None


class ClusterSizeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["easyicu.cluster_sizes_result/1"] = (
        "easyicu.cluster_sizes_result/1"
    )
    specification_sha256: str
    analysis_unit: Literal["subject", "encounter"]
    input_n: int = Field(ge=1)
    assignment_rows_n: int = Field(ge=0)
    assigned_n: int = Field(ge=1)
    missing_label_n: int = Field(ge=0)
    unmatched_unit_n: int = Field(ge=0)
    matched_missing_label_n: int = Field(ge=0)
    observed_cluster_n: int = Field(ge=1)
    degenerate_single_cluster: bool
    rows: tuple[ClusterSizeRow, ...] = Field(min_length=2)
    claim_ceiling: Literal["analysis_only"] = "analysis_only"

    @model_validator(mode="after")
    def _counts_reconcile(self) -> "ClusterSizeResult":
        observed = sum(row.n_units > 0 for row in self.rows)
        if (
            self.assigned_n + self.missing_label_n != self.input_n
            or self.unmatched_unit_n + self.matched_missing_label_n != self.missing_label_n
            or self.assignment_rows_n + self.unmatched_unit_n != self.input_n
            or sum(row.n_units for row in self.rows) != self.assigned_n
        ):
            raise ValueError(
                "cluster counts and missing labels must reconcile with the unit roster"
            )
        if observed != self.observed_cluster_n or self.degenerate_single_cluster != (
            observed == 1
        ):
            raise ValueError(
                "cluster degeneracy must agree with observed cluster counts"
            )
        if len({row.cluster for row in self.rows}) != len(self.rows):
            raise ValueError("cluster result rows must be unique")
        for row in self.rows:
            if (
                row.is_empty != (row.n_units == 0)
                or row.proportion_assigned != row.n_units / self.assigned_n
                or row.proportion_all_units != row.n_units / self.input_n
            ):
                raise ValueError(
                    "cluster proportions and empty flags must agree with their denominators"
                )
        return self


def cluster_size_spec_sha256(spec: ClusterSizeSpec) -> str:
    return canonical_sha256(spec.model_dump(mode="json"))
