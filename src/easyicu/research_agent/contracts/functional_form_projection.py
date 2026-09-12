"""Exact linear-specification projection of one sealed landmark sensitivity row.

The signed primary owner already fitted the nested linear model on the reviewed
landmark population and sealed its per-unit odds ratio and Wald interval.
Restating that same fitted line on the primary exposure grid changes only the
contrast coordinates, so this contract forbids an independent refit and carries
the single linear exposure term with its own variance.

It supplies no new scientific authority: its inputs must still pass the typed
artifact loader, and its coordinates must match the reviewed runtime, the parent
step, and the sealed primary contrasts it is bound to.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dependence import PlannedDependenceRequirement
from .functional_form import FunctionalFormSpec
from .functional_form_effects import FunctionalFormSource


class FunctionalFormProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["easyicu.functional_form_projection/1"] = "easyicu.functional_form_projection/1"
    execution_owner: Literal["landmark_spline_functional_form_executor_v1"] = "landmark_spline_functional_form_executor_v1"
    status: Literal["ok"] = "ok"
    analysis_role: Literal["sensitivity"] = "sensitivity"
    independent_refit: Literal[False] = False
    projected_specification: Literal["linear_per_unit_exposure_term"] = "linear_per_unit_exposure_term"
    step_id: str = Field(min_length=1)
    spec_id: str = Field(min_length=1)
    form: FunctionalFormSpec
    exposure: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    outcome_time_column: str = Field(min_length=1)
    adjustment_columns: tuple[str, ...]
    categorical_adjustment_columns: tuple[str, ...]
    landmark_hours: float = Field(gt=0)
    observation_duration_column: str = Field(min_length=1)
    observation_duration_unit: Literal["hours", "days"]
    population_rule: Literal["alive_and_under_observation_at_landmark_with_valid_exposure"] = "alive_and_under_observation_at_landmark_with_valid_exposure"
    interpretation: Literal["descriptive_prognostic_association_not_causal"] = "descriptive_prognostic_association_not_causal"
    runtime_projection_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    execution_contract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    protocol_content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    n: int = Field(ge=30, strict=True)
    events: int = Field(gt=0, strict=True)
    # Inherited variance policy of the sealed row. A projection refits nothing,
    # so it cannot and must not claim its own reproduced cluster count.
    dependence: PlannedDependenceRequirement | None
    exposure_knots: tuple[float, float, float]
    reference: float
    curve_points: int = Field(ge=5, strict=True)
    exposure_increment: float = Field(gt=0)
    effect_scale: Literal["odds_ratio"] = "odds_ratio"
    interval_method: Literal["wald_log_odds"] = "wald_log_odds"
    confidence_level: Literal[0.95] = 0.95
    parameter_columns: tuple[str, ...]
    parameter_values: tuple[float, ...]
    covariance_matrix: tuple[tuple[float, ...], ...]
    primary_contrasts: tuple[tuple[float, float, float, float, float], ...]
    primary_exposure_nonlinearity_p_value: float = Field(ge=0, le=1)
    input_bindings: tuple[FunctionalFormSource, ...]

    @model_validator(mode="after")
    def _coherent(self):
        if self.events >= self.n:
            raise ValueError("functional-form projection needs events and non-events")
        if self.form.target_column != self.exposure:
            raise ValueError("a projection restates the primary exposure, never another term")
        if self.exposure in self.categorical_adjustment_columns:
            raise ValueError("functional-form projection exposure must stay continuous")
        if len(set(self.adjustment_columns)) != len(self.adjustment_columns):
            raise ValueError("functional-form projection adjustment set repeats a column")
        lower, middle, upper = self.exposure_knots
        if not lower < middle < upper:
            raise ValueError("functional-form projection knots must be distinct and ordered")
        if self.reference != middle:
            raise ValueError("functional-form projection reference changed from the primary median")
        if self.parameter_columns != (self.exposure,):
            raise ValueError("a projection carries exactly the sealed linear exposure term")
        if len(self.parameter_values) != 1 or len(self.covariance_matrix) != 1:
            raise ValueError("functional-form projection parameter dimensions disagree")
        if len(self.covariance_matrix[0]) != 1 or self.covariance_matrix[0][0] < 0:
            raise ValueError("functional-form projection variance is invalid")
        if len(self.primary_contrasts) != 2:
            raise ValueError("functional-form projection requires both source primary contrasts")
        for row in self.primary_contrasts:
            value, reference, estimate, low, high = row
            if reference != self.reference or not 0 < low <= estimate <= high:
                raise ValueError("functional-form projection primary contrast identity is invalid")
        if tuple(row[0] for row in self.primary_contrasts) != (lower, upper):
            raise ValueError("functional-form projection contrast coordinates changed")
        if len(self.input_bindings) != 2 or len({item.input_key for item in self.input_bindings}) != 2:
            raise ValueError("a projection binds exactly the two sealed primary tables, not a cohort")
        return self


__all__ = ["FunctionalFormProjection"]
