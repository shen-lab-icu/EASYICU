"""Sealed identity of a covariate-form refit's primary-exposure effects.

This contract travels in the registered tables and step summary. It supplies
no new scientific authority: its inputs must still pass the typed artifact
loader and its model choices must match the reviewed runtime and parent step.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dependence import PlannedDependenceRequirement
from .functional_form import FunctionalFormSpec


class FunctionalFormSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_key: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    loaded: Literal[True]
    row_count: int = Field(ge=1, strict=True)


class FunctionalFormEffects(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["easyicu.functional_form_effects/1"] = "easyicu.functional_form_effects/1"
    execution_owner: Literal["landmark_spline_functional_form_executor_v1"] = "landmark_spline_functional_form_executor_v1"
    status: Literal["ok"] = "ok"
    analysis_role: Literal["sensitivity"] = "sensitivity"
    independent_refit: Literal[True] = True
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
    model_rows_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    n: int = Field(ge=30, strict=True)
    events: int = Field(gt=0, strict=True)
    dependence: PlannedDependenceRequirement | None
    cluster_count: int | None = Field(default=None, ge=2, strict=True)
    reproduced_primary_cluster_count: int | None = Field(default=None, ge=2, strict=True)
    exposure_knots: tuple[float, float, float]
    target_knots: tuple[float, float, float]
    reference: float
    curve_points: int = Field(ge=5, strict=True)
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
            raise ValueError("functional-form outcome needs events and non-events")
        if self.form.target_column not in self.adjustment_columns or self.form.target_column in self.categorical_adjustment_columns:
            raise ValueError("functional-form target must be a continuous adjustment term")
        if self.form.target_column == self.exposure or len(set(self.adjustment_columns)) != len(self.adjustment_columns):
            raise ValueError("functional-form target cannot replace the primary exposure")
        if bool(self.dependence) != (self.cluster_count is not None):
            raise ValueError("functional-form dependence and cluster count disagree")
        if self.cluster_count is not None and self.cluster_count > self.n:
            raise ValueError("functional-form clusters exceed observations")
        if self.reproduced_primary_cluster_count != self.cluster_count:
            raise ValueError("functional-form refit and reproduced primary clusters differ")
        for knots in (self.exposure_knots, self.target_knots):
            if not knots[0] < knots[1] < knots[2]:
                raise ValueError("functional-form knots must be distinct and ordered")
        if self.reference != self.exposure_knots[1]:
            raise ValueError("functional-form reference changed from the primary median")
        k = len(self.parameter_columns)
        if k < 3 or len(set(self.parameter_columns)) != k or len(self.parameter_values) != k or len(self.covariance_matrix) != k:
            raise ValueError("functional-form parameter dimensions disagree")
        for i, row in enumerate(self.covariance_matrix):
            if len(row) != k or row[i] < 0:
                raise ValueError("functional-form covariance dimensions or diagonal invalid")
        if any(abs(self.covariance_matrix[i][j] - self.covariance_matrix[j][i]) > 1e-8 for i in range(k) for j in range(k)):
            raise ValueError("functional-form covariance is not symmetric")
        if len(self.primary_contrasts) != 2:
            raise ValueError("functional-form requires both source primary contrasts")
        for row in self.primary_contrasts:
            value, reference, estimate, lower, upper = row
            if reference != self.reference or not 0 < lower <= estimate <= upper:
                raise ValueError("functional-form primary contrast identity is invalid")
        if tuple(row[0] for row in self.primary_contrasts) != (self.exposure_knots[0], self.exposure_knots[2]):
            raise ValueError("functional-form primary contrast coordinates changed")
        if len(self.input_bindings) != 3 or len({item.input_key for item in self.input_bindings}) != 3:
            raise ValueError("functional-form requires the exact cohort and two parent bindings")
        return self
