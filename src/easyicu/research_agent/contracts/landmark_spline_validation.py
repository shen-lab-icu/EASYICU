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


class LandmarkSplineRuntimeReceipt(BaseModel):
    """Evidence that the signed host owner fitted the declared landmark model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.landmark_spline_runtime_receipt/1"]
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
