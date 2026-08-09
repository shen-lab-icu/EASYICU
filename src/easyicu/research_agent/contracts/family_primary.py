"""Planner-owned primary-result contract for causal and survival studies.

This module is dependency-neutral with respect to orchestration and execution.
The Planner declares the scientific target; family-specific executors and
gates consume this public value object without importing one another.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .model_terms import ModelTermSpec, validate_model_term_roster
from .model_tokens import (
    SURVIVAL_COX_ESTIMATOR,
    SURVIVAL_PH_DIAGNOSTIC,
    canonical_survival_estimator,
    canonical_survival_ph_diagnostic,
)


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
    proportional_hazards_alpha: Optional[float] = Field(default=None, gt=0, lt=1)
    #: What the plan asks the host to do when the PH assumption is rejected.
    #:
    #: ``human_review`` was removed from this vocabulary on 2026-08-07. It named
    #: a workflow that did not exist: a PH violation under that policy produced
    #: ``violation_human_review`` and ``paper_authorization_allowed=False``, and
    #: nothing anywhere converted that into a review request, bound a reviewer
    #: decision to the PH/result digests, or resumed the publication gate. Its
    #: observable behaviour was identical to ``block_paper_authorization``, so
    #: the contract advertised a review capability the system did not have.
    #: A plan that wants human adjudication declares
    #: ``block_paper_authorization``, which is what actually happens.
    #:
    #: Neither remaining value lets the plan authorize its own paper: see
    #: ``SurvivalAnalysisReceipt`` for why ``report_only`` is a disclosure
    #: setting rather than an authorization.
    proportional_hazards_policy: Optional[
        Literal["report_only", "block_paper_authorization"]
    ] = None
    covariates: Optional[List[str]] = None
    model_terms: Optional[List[ModelTermSpec]] = None
    exposure_encoding: Optional[Literal["numeric_linear", "declared_model_terms"]] = (
        None
    )
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

    @field_validator("estimator", mode="before")
    @classmethod
    def _canonical_estimator(cls, value: object) -> str:
        token = canonical_survival_estimator(value)
        if not token:
            raise ValueError("family primary-result contract fields must be non-empty")
        return token

    @field_validator("proportional_hazards_diagnostic", mode="before")
    @classmethod
    def _canonical_ph_diagnostic(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        token = canonical_survival_ph_diagnostic(value)
        return token or None

    @field_validator("covariates")
    @classmethod
    def _validate_survival_covariates(
        cls, value: Optional[List[str]]
    ) -> Optional[List[str]]:
        if value is None:
            return None
        names = [str(item or "").strip() for item in value]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "family primary-result covariates must be unique and nonblank"
            )
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
            if self.model_terms is not None:
                _, adjustment_terms = validate_model_term_roster(
                    terms=self.model_terms,
                    exposure=self.exposure_source,
                    covariates=self.covariates,
                )
                declared_covariates = [item.name for item in adjustment_terms]
                if self.covariates is None:
                    self.covariates = declared_covariates
                if self.exposure_encoding not in (None, "declared_model_terms"):
                    raise ValueError(
                        "a survival model_terms contract requires "
                        "exposure_encoding='declared_model_terms'"
                    )
                self.exposure_encoding = "declared_model_terms"
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
                    "model_terms",
                )
                if not str(getattr(self, name) or "").strip()
            ]
            if self.covariates is None:
                missing.append("covariates")
            if self.event_value is None:
                missing.append("event_value")
            if self.time_horizon_value is None:
                missing.append("time_horizon_value")
            if self.estimator == SURVIVAL_COX_ESTIMATOR:
                if self.proportional_hazards_diagnostic is None:
                    missing.append("proportional_hazards_diagnostic")
                if self.proportional_hazards_alpha is None:
                    missing.append("proportional_hazards_alpha")
                if self.proportional_hazards_policy is None:
                    missing.append("proportional_hazards_policy")
            if missing:
                raise ValueError(
                    "survival primary-result contract requires " + ", ".join(missing)
                )
            if (
                self.estimator == SURVIVAL_COX_ESTIMATOR
                and self.proportional_hazards_diagnostic != SURVIVAL_PH_DIAGNOSTIC
            ):
                raise ValueError(
                    "the implemented Cox primary-result contract requires the "
                    f"exact diagnostic {SURVIVAL_PH_DIAGNOSTIC!r}"
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
