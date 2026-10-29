"""Closed declarations for descriptive performance of frozen binary predictions.

The host still owns source authorization and plan approval. Neither this
declaration nor its result grants a clinical or normative fairness judgment.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..canonical_json import canonical_sha256
from .closed_levels import validate_closed_scalar_levels
from .prediction_execution import PREDICTION_SCORES_PRODUCT
from .prediction_validation import CalibrationStatus

PREDICTION_SUBGROUP_ACTION = "prediction.subgroup_fairness"
PREDICTION_SUBGROUP_PRODUCT = "table:subgroup_performance"
PREDICTION_SUBGROUP_OUTPUTS = (PREDICTION_SUBGROUP_PRODUCT,)
_METHOD = "prediction_subgroup_performance"


class PredictionSubgroupError(ValueError):
    """Stable refusal without including patient values in the diagnostic."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {message}")


class PredictionSubgroupAxis(BaseModel):
    """A predeclared closed category set; None explicitly includes missingness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str = Field(min_length=1)
    levels: tuple[Any, ...] = Field(min_length=1, max_length=20)

    @field_validator("column")
    @classmethod
    def _column(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("subgroup column must be canonical")
        return value

    @field_validator("levels")
    @classmethod
    def _levels(cls, values: tuple[Any, ...]) -> tuple[Any, ...]:
        validate_closed_scalar_levels(
            [value for value in values if value is not None], label="subgroup levels"
        )
        if sum(value is None for value in values) > 1:
            raise ValueError("the missing subgroup level can be declared only once")
        if any(isinstance(value, str) and not value.strip() for value in values):
            raise ValueError("blank subgroup levels are not supported; declare None")
        return values


class PredictionSubgroupSpec(BaseModel):
    """One frozen score artifact, one evaluation split and explicit subgroups."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["easyicu.prediction_subgroup/1"] = "easyicu.prediction_subgroup/1"
    prediction_kind: Literal["binary_probability"] = "binary_probability"
    unit_id_column: str = "unit_id"
    subject_id_column: str = "subject_id"
    split_column: str = "split"
    outcome_column: str = "outcome"
    probability_column: str = "probability"
    evaluation_split: str = "validation"
    analysis_unit: Literal["subject", "encounter"]
    subgroup_input_key: str
    subgroup_unit_id_column: str
    subgroup_key_encoding: Literal["exact", "prediction_typed_scalar"] = "exact"
    axes: tuple[PredictionSubgroupAxis, ...] = Field(min_length=1, max_length=4)
    include_intersections: bool = True
    missing_value_policy: Literal["error", "complete_case"] = "error"
    minimum_group_n: int = Field(default=20, ge=2, strict=True)
    minimum_class_n: int = Field(default=5, ge=1, strict=True)
    calibration_bins: int = Field(default=10, ge=2, le=20, strict=True)
    calibration_logit_epsilon: float = Field(default=1e-6, gt=0, lt=0.5)
    uncertainty: Literal["point_estimates_only"] = "point_estimates_only"

    @field_validator(
        "unit_id_column", "subject_id_column", "split_column", "outcome_column",
        "probability_column", "evaluation_split", "subgroup_unit_id_column",
        "subgroup_input_key",
    )
    @classmethod
    def _coordinate(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("subgroup coordinates must be nonempty and canonical")
        return value

    @model_validator(mode="after")
    def _closed(self) -> PredictionSubgroupSpec:
        kind, separator, product = self.subgroup_input_key.partition(":")
        if (not separator or not product or kind not in {"cohort", "table", "dataset", "artifact"}
                or self.subgroup_input_key == PREDICTION_SCORES_PRODUCT):
            raise ValueError("subgroup membership requires a separate typed table input")
        identities = {self.unit_id_column, self.subject_id_column}
        values = {self.split_column, self.outcome_column, self.probability_column}
        columns = [axis.column for axis in self.axes]
        if (len(values) != 3 or identities & values or len(set(columns)) != len(columns)
                or set(columns) & (identities | values | {self.subgroup_unit_id_column})):
            raise ValueError("subgroup axes, values and identity coordinates must be distinct")
        if self.include_intersections and math.prod(len(axis.levels) for axis in self.axes) > 200:
            raise ValueError("the declared full intersection exceeds 200 cells")
        return self


class PredictionSubgroupPlanFields(BaseModel):
    """Typed planner option compiled into the existing sealed method field."""

    prediction_subgroup_spec: PredictionSubgroupSpec | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )

    @field_validator("prediction_subgroup_spec", mode="before")
    @classmethod
    def _wire(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value


def prediction_subgroup_spec_sha256(spec: PredictionSubgroupSpec | Mapping[str, Any]) -> str:
    return canonical_sha256(PredictionSubgroupSpec.model_validate(spec).model_dump(mode="json"))


def prediction_subgroup_method(spec: PredictionSubgroupSpec) -> str:
    parsed = PredictionSubgroupSpec.model_validate(spec)
    return f"{_METHOD}({json.dumps(parsed.model_dump(mode='json'), sort_keys=True, ensure_ascii=False, separators=(',', ':'))})"


def prediction_subgroup_spec_for_step(step: object) -> PredictionSubgroupSpec | None:
    method = str(getattr(step, "method", "") or "")
    prefix = f"{_METHOD}("
    if not method.startswith(prefix):
        return None
    if getattr(step, "scientific_action_id", None) != PREDICTION_SUBGROUP_ACTION:
        raise ValueError("prediction_subgroup_spec belongs only to prediction.subgroup_fairness")
    try:
        if not method.endswith(")"):
            raise ValueError("unclosed subgroup declaration")
        return PredictionSubgroupSpec.model_validate_json(method[len(prefix):-1])
    except ValueError as exc:
        raise PredictionSubgroupError("prediction_subgroup_declaration_invalid", "a closed typed declaration is required") from exc


def prediction_subgroup_step_matches(step: object) -> bool:
    spec = prediction_subgroup_spec_for_step(step)
    return bool(
        getattr(step, "scientific_action_id", None) == PREDICTION_SUBGROUP_ACTION
        and spec is not None
        and tuple(getattr(step, "inputs", ())) == (PREDICTION_SCORES_PRODUCT, spec.subgroup_input_key)
        and tuple(getattr(step, "expected_outputs", ())) == PREDICTION_SUBGROUP_OUTPUTS
        and getattr(step, "planned_analysis_role", None) in {"secondary", "auxiliary"}
        and not any(getattr(step, name, None) for name in (
            "model_requirements", "table_one_spec", "cohort_definition_spec", "measurement_audit_spec",
            "robustness_replay_spec", "trajectory_stability_spec", "family_primary_result_requirement",
        ))
    )


class PredictionSubgroupPerformance(BaseModel):
    """One retained cell, including unavailable metrics and their denominators."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    scope: Literal["overall", "marginal", "intersection"]
    group_values: dict[str, Any]
    input_n: int = Field(ge=0)
    evaluation_n: int = Field(ge=0)
    excluded_n: int = Field(ge=0)
    missing_outcome_n: int = Field(ge=0)
    missing_probability_n: int = Field(ge=0)
    event_n: int = Field(ge=0)
    non_event_n: int = Field(ge=0)
    subject_n: int = Field(ge=0)
    repeated_subject_n: int = Field(ge=0)
    sample_status: Literal["adequate_for_declared_counts", "small_sample", "empty"]
    event_rate: float | None = Field(default=None, ge=0, le=1)
    mean_predicted_probability: float | None = Field(default=None, ge=0, le=1)
    brier_score: float | None = Field(default=None, ge=0, le=1)
    auroc: float | None = Field(default=None, ge=0, le=1)
    auroc_status: Literal["estimated", "not_estimable_single_class", "not_estimable_empty"]
    calibration_status: CalibrationStatus | Literal["not_estimable_single_class", "not_estimable_empty"]
    calibration_intercept: float | None = None
    calibration_slope: float | None = None
    clipped_probability_n: int = Field(default=0, ge=0)
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reconcile(self) -> PredictionSubgroupPerformance:
        if self.event_n + self.non_event_n != self.evaluation_n or self.evaluation_n + self.excluded_n != self.input_n:
            raise ValueError("subgroup denominators do not reconcile")
        if not max(self.missing_outcome_n, self.missing_probability_n) <= self.excluded_n <= self.missing_outcome_n + self.missing_probability_n:
            raise ValueError("complete-case exclusion counts do not reconcile")
        if self.subject_n > self.evaluation_n or self.repeated_subject_n > self.subject_n:
            raise ValueError("subgroup identity counts do not reconcile")
        if self.clipped_probability_n > self.evaluation_n:
            raise ValueError("clipped probability count exceeds the evaluation count")
        if (self.auroc_status == "estimated") != (self.auroc is not None):
            raise ValueError("AUROC availability must agree with its status")
        rates = (self.event_rate, self.mean_predicted_probability, self.brier_score)
        if self.evaluation_n:
            if any(value is None for value in rates) or not math.isclose(self.event_rate, self.event_n / self.evaluation_n):
                raise ValueError("nonempty subgroup requires its defined rates and Brier score")
            if (self.auroc_status == "estimated") != bool(self.event_n and self.non_event_n):
                raise ValueError("AUROC requires both outcome classes")
        elif any(value is not None for value in rates) or self.sample_status != "empty":
            raise ValueError("empty groups must retain unavailable metrics")
        calibration = (self.calibration_intercept, self.calibration_slope)
        if self.calibration_status == "estimated":
            if any(value is None for value in calibration):
                raise ValueError("estimated calibration requires intercept and slope")
        elif any(value is not None for value in calibration):
            raise ValueError("unavailable calibration must not publish coefficients")
        return self


class PredictionSubgroupFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: Literal[
        "prediction_subgroup_result_schema_invalid", "prediction_subgroup_result_mismatch",
    ]
    message: str


class PredictionSubgroupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["easyicu.prediction_subgroup_result/1"] = "easyicu.prediction_subgroup_result/1"
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_split: str
    analysis_unit: Literal["subject", "encounter"]
    input_n: int = Field(ge=1)
    evaluation_input_n: int = Field(ge=1)
    evaluation_unit_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: tuple[PredictionSubgroupPerformance, ...] = Field(min_length=2)
    uncertainty: Literal["point_estimates_only"] = "point_estimates_only"
    interpretation: Literal["descriptive_subgroup_performance_no_fairness_judgment"] = "descriptive_subgroup_performance_no_fairness_judgment"
    authority_scope: Literal["analysis_only"] = "analysis_only"
    paper_authorization_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _overall(self) -> PredictionSubgroupResult:
        if self.rows[0].scope != "overall" or self.rows[0].input_n != self.evaluation_input_n:
            raise ValueError("the first result must retain the complete evaluation split")
        return self
