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

# The deterministic display owner resolves its specification-contrast panel from
# bound input keys alone and never sees the plan, so a functional-form effect
# product is recognizable there only by carrying one of these tokens next to its
# `_exposure_curve`/`_exposure_contrasts` suffix.
DISPLAY_SENSITIVITY_TOKENS = ("robustness", "sensitivity")


def functional_form_products(diagnostic_product: str, *, include_effects: bool) -> tuple[str, ...]:
    """Keep the planned comparison name and attach its model-effect products."""

    kind, separator, name = diagnostic_product.partition(":")
    if kind != "table" or not separator or not name or any(c in name for c in "/\\"):
        raise ValueError("functional-form comparison requires a named table product")
    if not include_effects:
        return (diagnostic_product,)
    # Naming the effect products after the planner's diagnostic alone would make
    # the article figure's contrast panel depend on how that diagnostic happened
    # to be worded. Append the token when it is absent so the display boundary
    # resolves the same product for every reviewed functional-form sensitivity.
    stem = (
        name
        if any(token in name for token in DISPLAY_SENSITIVITY_TOKENS)
        else f"{name}_sensitivity"
    )
    return (diagnostic_product, f"table:{stem}_exposure_curve", f"table:{stem}_exposure_contrasts")


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
