"""Planner-owned primary-result contract for causal and survival studies.

This module is dependency-neutral with respect to orchestration and execution.
The Planner declares the scientific target; family-specific executors and
gates consume this public value object without importing one another.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .model_tokens import normalise_model_contract_token


class FamilyPrimaryResultRequirement(BaseModel):
    """Headline-result coordinates shared by causal and survival families."""

    model_config = ConfigDict(extra="forbid")

    analysis_family: Literal["causal_inference", "survival"]
    exposure_source: str
    outcome: str
    expected_result_product: str
    input_product: Optional[str] = None
    estimator: str
    effect_scale: str
    uncertainty_method: str
    population: str
    estimand: Optional[str] = None
    treatment: Optional[str] = None
    comparator: Optional[str] = None
    adjustment_strategy: Optional[str] = None
    overlap_diagnostic: Optional[str] = None
    time_origin: Optional[str] = None
    time_column: Optional[str] = None
    event_column: Optional[str] = None
    event_definition: Optional[str] = None
    censoring_strategy: Optional[str] = None
    competing_risk_strategy: Optional[str] = None
    time_horizon: Optional[str] = None
    effect_measure: Optional[str] = None
    proportional_hazards_diagnostic: Optional[str] = None
    covariates: Optional[List[str]] = None
    exposure_encoding: Optional[Literal["numeric_linear"]] = "numeric_linear"
    missing_data_policy: Optional[Literal["complete_case"]] = "complete_case"
    time_unit: Optional[Literal["minutes", "hours", "days"]] = None
    event_value: Optional[int] = None
    time_horizon_value: Optional[float] = Field(default=None, gt=0)

    @field_validator(
        "exposure_source",
        "outcome",
        "expected_result_product",
        "estimator",
        "effect_scale",
        "uncertainty_method",
        "population",
    )
    @classmethod
    def _require_nonblank_contract_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("family primary-result contract fields must be non-empty")
        return text

    @field_validator("covariates")
    @classmethod
    def _validate_survival_covariates(
        cls, value: Optional[List[str]]
    ) -> Optional[List[str]]:
        if value is None:
            return None
        names = [str(item or "").strip() for item in value]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("family primary-result covariates must be unique and nonblank")
        return names

    @model_validator(mode="after")
    def _require_family_specific_scientific_fields(
        self,
    ) -> "FamilyPrimaryResultRequirement":
        product = str(self.expected_result_product).strip()
        if not product.startswith("table:"):
            raise ValueError(
                "family primary-result expected_result_product must be a typed table"
            )

        if self.analysis_family == "causal_inference":
            missing = [
                name
                for name in (
                    "estimand",
                    "treatment",
                    "comparator",
                    "adjustment_strategy",
                    "overlap_diagnostic",
                )
                if not str(getattr(self, name) or "").strip()
            ]
            if missing:
                raise ValueError(
                    "causal primary-result contract requires " + ", ".join(missing)
                )
        else:
            missing = [
                name
                for name in (
                    "time_origin",
                    "time_column",
                    "event_column",
                    "event_definition",
                    "censoring_strategy",
                    "competing_risk_strategy",
                    "time_horizon",
                    "effect_measure",
                    "input_product",
                    "exposure_encoding",
                    "missing_data_policy",
                    "time_unit",
                )
                if not str(getattr(self, name) or "").strip()
            ]
            if self.covariates is None:
                missing.append("covariates")
            if self.event_value is None:
                missing.append("event_value")
            if self.time_horizon_value is None:
                missing.append("time_horizon_value")
            if missing:
                raise ValueError(
                    "survival primary-result contract requires " + ", ".join(missing)
                )
            estimator = normalise_model_contract_token(self.estimator)
            if "cox" in estimator and not str(
                self.proportional_hazards_diagnostic or ""
            ).strip():
                raise ValueError(
                    "a Cox primary-result contract requires "
                    "proportional_hazards_diagnostic"
                )
            if self.outcome in (self.covariates or []):
                raise ValueError("survival covariates must not contain the outcome")
            if self.exposure_source in (self.covariates or []):
                raise ValueError("survival covariates must not contain the exposure")
            if self.time_column in (self.covariates or []) or self.event_column in (
                self.covariates or []
            ):
                raise ValueError(
                    "survival covariates must not contain time/event columns"
                )
        return self


__all__ = ["FamilyPrimaryResultRequirement"]
