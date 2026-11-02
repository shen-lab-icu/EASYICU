"""Reviewed standalone diagnostic contracts for balance and cluster sizes.

Each action retains its own specification, outputs and statistical meaning.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator, model_validator

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


BALANCE_ACTION = "causal_emulation.covariate_balance"
BALANCE_KIND = "covariate_balance"
BALANCE_OUTPUTS = ("table:covariate_balance", "log:covariate_balance_receipt")


class BalanceCovariate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    column: str = Field(min_length=1)
    kind: Literal["continuous", "binary", "categorical"]
    levels: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _closed_levels(self) -> "BalanceCovariate":
        if self.column != self.column.strip():
            raise ValueError("balance column must not have surrounding whitespace")
        if self.kind == "categorical":
            if len(self.levels) < 2 or len(set(self.levels)) != len(self.levels):
                raise ValueError(
                    "categorical balance requires at least two unique levels"
                )
            if any(not level or level != level.strip() for level in self.levels):
                raise ValueError("categorical levels must be nonblank exact spellings")
        elif self.levels:
            raise ValueError("only categorical covariates declare levels")
        return self


class CovariateBalanceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["easyicu.covariate_balance/1"] = (
        "easyicu.covariate_balance/1"
    )
    unit_id_column: str = Field(min_length=1)
    analysis_unit: Literal["subject", "encounter"]
    treatment_column: str = Field(min_length=1)
    covariates: tuple[BalanceCovariate, ...] = Field(min_length=1)
    weight_column: str | None = Field(default=None, min_length=1)
    missing_policy: Literal["per_covariate_complete_case"] = (
        "per_covariate_complete_case"
    )
    denominator: Literal["unweighted_pooled_sample_variance"] = (
        "unweighted_pooled_sample_variance"
    )
    weight_policy: Literal["nonnegative_zero_excluded"] = "nonnegative_zero_excluded"
    abs_smd_threshold: float | None = Field(default=None, gt=0)

    @field_validator("unit_id_column", "treatment_column", "weight_column")
    @classmethod
    def _exact_column(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("balance coordinates must be nonblank exact column names")
        return value

    @model_validator(mode="after")
    def _distinct_columns(self) -> "CovariateBalanceSpec":
        columns = [self.unit_id_column, self.treatment_column]
        if self.weight_column is not None:
            columns.append(self.weight_column)
        columns.extend(item.column for item in self.covariates)
        if len(columns) != len(set(columns)):
            raise ValueError(
                "balance identity, treatment, weight and covariates must differ"
            )
        return self


class CovariateBalanceRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    row_role: Literal["covariate", "category_indicator"]
    covariate: str
    kind: Literal["continuous", "binary", "categorical"]
    level: str | None
    n_treated: int = Field(ge=0)
    n_control: int = Field(ge=0)
    missing_treated_n: int = Field(ge=0)
    missing_control_n: int = Field(ge=0)
    positive_weight_treated_n: int = Field(ge=0)
    positive_weight_control_n: int = Field(ge=0)
    zero_weight_treated_n: int = Field(ge=0)
    zero_weight_control_n: int = Field(ge=0)
    weight_sum_treated: float = Field(ge=0)
    weight_sum_control: float = Field(ge=0)
    ess_treated: float = Field(ge=0)
    ess_control: float = Field(ge=0)
    mean_treated_before: float | None
    mean_control_before: float | None
    variance_treated_before: float | None
    variance_control_before: float | None
    smd_denominator: float | None
    smd_before: float | None
    smd_after: float | None
    mean_treated_after: float | None
    mean_control_after: float | None
    status_before: Literal[
        "estimated",
        "constant_equal",
        "zero_variance_different_means",
        "insufficient_observations",
    ]
    status_after: Literal[
        "estimated",
        "constant_equal",
        "zero_variance_different_means",
        "insufficient_observations",
        "not_requested",
        "empty_weighted_group",
    ]
    exceeds_threshold_before: bool | None
    exceeds_threshold_after: bool | None

    @model_validator(mode="after")
    def _weight_counts_reconcile(self) -> "CovariateBalanceRow":
        if (
            self.positive_weight_treated_n + self.zero_weight_treated_n
            != self.n_treated
            or self.positive_weight_control_n + self.zero_weight_control_n
            != self.n_control
        ):
            raise ValueError("balance complete-case weight counts must reconcile")
        if self.positive_weight_treated_n == 0 and (
            self.weight_sum_treated != 0 or self.ess_treated != 0
        ):
            raise ValueError("an empty weighted treated group has zero mass and ESS")
        if self.positive_weight_control_n == 0 and (
            self.weight_sum_control != 0 or self.ess_control != 0
        ):
            raise ValueError("an empty weighted control group has zero mass and ESS")
        return self


class CovariateBalanceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["easyicu.covariate_balance_result/1"] = (
        "easyicu.covariate_balance_result/1"
    )
    specification_sha256: str
    analysis_unit: Literal["subject", "encounter"]
    input_n: int = Field(ge=1)
    treated_n: int = Field(ge=1)
    control_n: int = Field(ge=1)
    zero_weight_n: int = Field(ge=0)
    rows: tuple[CovariateBalanceRow, ...] = Field(min_length=1)
    claim_ceiling: Literal["analysis_only"] = "analysis_only"

    @model_validator(mode="after")
    def _counts_reconcile(self) -> "CovariateBalanceResult":
        if (
            self.treated_n + self.control_n != self.input_n
            or self.zero_weight_n > self.input_n
        ):
            raise ValueError("balance group counts must reconcile with the unit roster")
        for row in self.rows:
            if (
                row.n_treated + row.missing_treated_n != self.treated_n
                or row.n_control + row.missing_control_n != self.control_n
            ):
                raise ValueError(
                    "balance covariate missingness must reconcile with group counts"
                )
        return self


def covariate_balance_spec_sha256(spec: CovariateBalanceSpec) -> str:
    return canonical_sha256(spec.model_dump(mode="json"))


def validate_descriptive_auxiliary_steps(step: Any) -> None:
    """Enforce the declared balance and cluster-size step contracts."""
    for spec, action, outputs, columns in (
        (step.covariate_balance_spec, BALANCE_ACTION, BALANCE_OUTPUTS,
         {step.covariate_balance_spec.unit_id_column, step.covariate_balance_spec.treatment_column,
          *(item.column for item in step.covariate_balance_spec.covariates),
          *([step.covariate_balance_spec.weight_column] if step.covariate_balance_spec.weight_column else [])}
         if step.covariate_balance_spec else set()),
        (step.cluster_size_spec, CLUSTER_SIZE_ACTION, CLUSTER_SIZE_OUTPUTS,
         {step.cluster_size_spec.unit_id_column} if step.cluster_size_spec else set()),
    ):
        if spec is None:
            continue
        if step.scientific_action_id != action or step.planned_analysis_role not in {"secondary", "auxiliary"}:
            raise ValueError(f"{action} requires its exact auxiliary or secondary action")
        if set(step.expected_outputs) != set(outputs) or len(step.expected_outputs) != len(outputs):
            raise ValueError(f"{action} requires exactly its declared table and receipt outputs")
        if not columns.issubset(step.inputs):
            raise ValueError(f"{action} variables must be explicit step inputs")


_WIRE = {"anyOf": [{"type": "string"}, {"type": "null"}]}


class DiagnosticPlanFields(BaseModel):
    covariate_balance_spec: Annotated[CovariateBalanceSpec | None, WithJsonSchema(_WIRE)] = Field(
        default=None, exclude_if=lambda value: value is None,
        description="JSON CovariateBalanceSpec; units, treatment, covariates, optional fixed weights and explicit missing/denominator policy.",
    )
    cluster_size_spec: Annotated[ClusterSizeSpec | None, WithJsonSchema(_WIRE)] = Field(
        default=None, exclude_if=lambda value: value is None,
        description="JSON ClusterSizeSpec; cohort identity, frozen assignment source/identity/label, closed labels and missing-label policy.",
    )

    @field_validator("covariate_balance_spec", "cluster_size_spec", mode="before")
    @classmethod
    def _decode(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value
