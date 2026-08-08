"""Planner-owned contracts for optional deterministic post-analysis work."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EValueConversionSpec(BaseModel):
    """Exact evidence/population binding for OR-to-RR E-value conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    effect_scale: Literal["odds_ratio"] = "odds_ratio"
    conversion_method: Literal["zhang_yu_observed_baseline_risk"] = (
        "zhang_yu_observed_baseline_risk"
    )
    baseline_risk_evidence_id: str = "outcome_rate"
    baseline_risk_column: str
    population_column: str
    baseline_population: str
    point_estimate_transformation: Literal["zhang_yu"] = "zhang_yu"
    interval_transformation: Literal["endpointwise_zhang_yu"] = "endpointwise_zhang_yu"
    null_crossing_rule: Literal["ci_bound_evalue_one"] = "ci_bound_evalue_one"

    @field_validator(
        "baseline_risk_evidence_id",
        "baseline_risk_column",
        "population_column",
        "baseline_population",
    )
    @classmethod
    def _nonempty_coordinate(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError(
                "E-value evidence and population coordinates must be non-empty"
            )
        return cleaned


class SubgroupAnalysisSpec(BaseModel):
    """Exact, pre-specified subgroup axes and multiplicity family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_model_requirement_id: str
    predictor: str
    outcome: str
    subgroup_columns: List[str] = Field(min_length=1)
    adjustment_covariates: List[str] = Field(default_factory=list)
    continuous_binning: Literal["quantile"] = "quantile"
    continuous_buckets: int = Field(default=4, ge=2, le=10)
    minimum_axis_n: int = Field(default=50, ge=30)
    minimum_stratum_n: int = Field(default=30, ge=20)
    effect_scale: Literal["odds_ratio"] = "odds_ratio"
    multiplicity_family_id: str

    @field_validator(
        "primary_model_requirement_id",
        "predictor",
        "outcome",
        "multiplicity_family_id",
    )
    @classmethod
    def _nonempty_string(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("subgroup contract coordinates must be non-empty")
        return cleaned

    @field_validator("subgroup_columns")
    @classmethod
    def _unique_subgroups(cls, values: List[str]) -> List[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("subgroup_columns must be non-empty and unique")
        return cleaned

    @model_validator(mode="after")
    def _current_kernel_has_no_adjusted_model(self) -> "SubgroupAnalysisSpec":
        if self.adjustment_covariates:
            raise ValueError(
                "subgroup_adjustment_unsupported: the current deterministic kernel "
                "implements only an explicitly unadjusted model"
            )
        if self.predictor == self.outcome or self.predictor in self.subgroup_columns:
            raise ValueError("predictor, outcome, and subgroup axes must be distinct")
        return self


__all__ = ["EValueConversionSpec", "SubgroupAnalysisSpec"]
