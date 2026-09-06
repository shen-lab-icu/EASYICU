"""Exact variable and basis choices for an RCS-versus-linear sensitivity.

This plan contract is not an execution capability. The owning estimator must
still validate its population, adjustment set, dependence, and source inputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RCS_LINEAR_SENSITIVITY_METHODS = frozenset({
    "restricted_cubic_spline_sensitivity", "linear_per_unit_sensitivity",
})


class FunctionalFormSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.functional_form/1"] = "easyicu.functional_form/1"
    target_column: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    comparison: Literal["restricted_cubic_spline_vs_linear"] = "restricted_cubic_spline_vs_linear"
    knot_quantiles: tuple[float, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _ordered_knots(self) -> "FunctionalFormSpec":
        lower, middle, upper = self.knot_quantiles
        if not 0 < lower < middle < upper < 1:
            raise ValueError("functional-form knot quantiles must be strictly ordered inside (0, 1)")
        return self
