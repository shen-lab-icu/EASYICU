"""Typed result receipt for the deterministic landmark-spline owner."""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LandmarkSplineFunctionalFormReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison: Literal["restricted_cubic_spline_vs_linear"]
    likelihood_ratio_statistic: float = Field(ge=0)
    degrees_of_freedom: int = Field(ge=1)
    p_value: float = Field(ge=0, le=1)
    linear_aic: float
    spline_aic: float
    linear_bic: float
    spline_bic: float

    @model_validator(mode="after")
    def _finite_statistics(self) -> "LandmarkSplineFunctionalFormReceipt":
        values = (
            self.likelihood_ratio_statistic,
            self.p_value,
            self.linear_aic,
            self.spline_aic,
            self.linear_bic,
            self.spline_bic,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("landmark spline diagnostics must be finite")
        return self


class LandmarkSplinePopulationFlowRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal[
        "source_cohort",
        "alive_and_under_observation_at_landmark",
        "valid_exposure_primary_population",
        "complete_case_model_population",
    ]
    n: int = Field(ge=0)
    excluded_from_previous: int = Field(ge=0)
    population_rule: str = Field(min_length=1)


class LandmarkSplineAbsoluteRiskReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "marginal_standardization_over_primary_complete_case_covariates"
    ]
    interval: Literal[
        "delta_method_logit_scale_95_percent_confidence_interval"
    ]
    grid_rows: int = Field(ge=5)


class LandmarkSplineVariableOpportunityReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    population_rule: Literal[
        "all_rows_with_observed_exposure_and_complete_model_terms"
    ]
    interpretation: Literal[
        "secondary_variable_opportunity_association_not_landmark_equivalent"
    ]
    exposure_increment: float = Field(gt=0)
    adjusted_odds_ratio: float = Field(gt=0)
    ci_low: float = Field(gt=0)
    ci_high: float = Field(gt=0)
    n: int = Field(ge=30)
    events: int = Field(ge=1)
    early_event_at_or_before_landmark_n: int = Field(ge=0)
    icu_observation_shorter_than_landmark_n: int = Field(ge=0)

    @model_validator(mode="after")
    def _coherent_interval(self) -> "LandmarkSplineVariableOpportunityReceipt":
        if not self.ci_low <= self.adjusted_odds_ratio <= self.ci_high:
            raise ValueError("variable-opportunity interval does not contain estimate")
        if self.events >= self.n:
            raise ValueError("variable-opportunity population lacks non-events")
        return self


class LandmarkSplineRuntimeReceipt(BaseModel):
    """Evidence that the signed host owner fitted the declared landmark model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "easyicu.landmark_spline_runtime_receipt/1",
        "easyicu.landmark_spline_runtime_receipt/2",
        "easyicu.landmark_spline_runtime_receipt/3",
    ]
    protocol_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    landmark_hours: int = Field(gt=0)
    population_rule: str = Field(min_length=1)
    spline_knot_quantiles: tuple[float, float, float]
    observed_knots: tuple[float, float, float]
    adjustment_columns: tuple[str, ...] = Field(min_length=1)
    primary_population_n: int = Field(ge=30)
    complete_case_n: int = Field(ge=30)
    events: int = Field(ge=1)
    functional_form_comparison: LandmarkSplineFunctionalFormReceipt
    population_flow: tuple[LandmarkSplinePopulationFlowRow, ...] | None = None
    adjusted_absolute_risk: LandmarkSplineAbsoluteRiskReceipt | None = None
    variable_opportunity_sensitivity: (
        LandmarkSplineVariableOpportunityReceipt | None
    ) = None
    variance_estimator: Literal["cluster_robust"] | None = None
    cluster_unit: Literal["patient"] | None = None
    cluster_group_source: str | None = None
    cluster_group_derivation: Literal[
        "identity", "prefix_before_delimiter"
    ] | None = None
    cluster_group_delimiter: str | None = None
    cluster_count: int | None = Field(default=None, ge=2)
    interpretation: Literal["descriptive_prognostic_association_not_causal"]

    @model_validator(mode="after")
    def _coherent_population_and_knots(self) -> "LandmarkSplineRuntimeReceipt":
        if self.complete_case_n > self.primary_population_n:
            raise ValueError("complete-case population exceeds landmark population")
        if self.events >= self.complete_case_n:
            raise ValueError("landmark outcome must contain events and non-events")
        if len(set(self.adjustment_columns)) != len(self.adjustment_columns):
            raise ValueError("landmark adjustment columns must be unique")
        if not all(math.isfinite(value) for value in self.observed_knots):
            raise ValueError("observed landmark knots must be finite")
        if not (
            0
            < self.spline_knot_quantiles[0]
            < self.spline_knot_quantiles[1]
            < self.spline_knot_quantiles[2]
            < 1
        ):
            raise ValueError("landmark knot quantiles must be ordered inside (0, 1)")
        if not (
            self.observed_knots[0] < self.observed_knots[1] < self.observed_knots[2]
        ):
            raise ValueError("observed landmark knots must be strictly increasing")
        if self.schema_version.endswith("/1"):
            if (
                self.population_flow is not None
                or self.adjusted_absolute_risk is not None
                or self.variable_opportunity_sensitivity is not None
            ):
                raise ValueError("landmark receipt v1 cannot carry v2 reporting fields")
        else:
            if self.population_flow is None or self.adjusted_absolute_risk is None:
                raise ValueError(
                    "landmark receipt v2 requires population flow and adjusted risk"
                )
            counts = [row.n for row in self.population_flow]
            if len(counts) != 4 or any(
                later > earlier for earlier, later in zip(counts, counts[1:])
            ):
                raise ValueError("landmark population flow must be four nested stages")
            if counts[-2] != self.primary_population_n or counts[-1] != self.complete_case_n:
                raise ValueError("landmark population flow disagrees with model receipt")
        cluster_fields = (
            self.variance_estimator,
            self.cluster_unit,
            self.cluster_group_source,
            self.cluster_group_derivation,
            self.cluster_count,
        )
        if self.schema_version.endswith("/3"):
            if any(value is None for value in cluster_fields):
                raise ValueError(
                    "landmark receipt v3 requires cluster-robust execution evidence"
                )
            if (
                self.cluster_group_derivation == "identity"
                and self.cluster_group_delimiter is not None
            ):
                raise ValueError("identity cluster grouping cannot declare a delimiter")
            if (
                self.cluster_group_derivation == "prefix_before_delimiter"
                and not self.cluster_group_delimiter
            ):
                raise ValueError("prefix cluster grouping requires a delimiter")
        elif any(value is not None for value in (*cluster_fields, self.cluster_group_delimiter)):
            raise ValueError("landmark receipt v1/v2 cannot claim clustered covariance")
        return self


def landmark_spline_runtime_receipt_valid(summary: Any) -> bool:
    if not isinstance(summary, Mapping) or summary.get("status") != "ok":
        return False
    try:
        LandmarkSplineRuntimeReceipt.model_validate(
            summary.get("scientific_runtime_receipt")
        )
    except Exception:
        return False
    return True


__all__ = [
    "LandmarkSplineFunctionalFormReceipt",
    "LandmarkSplineRuntimeReceipt",
    "landmark_spline_runtime_receipt_valid",
]
