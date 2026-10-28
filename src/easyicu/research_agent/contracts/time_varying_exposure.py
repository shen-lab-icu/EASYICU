"""Plan-owned time-varying association and g-method specifications.

This contract chooses no columns or missingness strategy for a caller. It is
reviewed with the plan, then shared by acquisition and deterministic execution.
The running-maximum Cox specification describes an association. The separate
g-method specification requires explicit causal assumptions and source-owned
person-period histories; neither creates publication authority.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..canonical_json import canonical_json, canonical_sha256


TIME_VARYING_EXPOSURE_METHOD = "time_varying_exposure_model"
TIME_VARYING_EXPOSURE_CAPABILITY = "association_time_varying_exposure_v1"
TIME_VARYING_ANALYSIS_KIND = "signed_time_varying_exposure_cox"
TIME_VARYING_INPUT_METADATA_KEY = "easyicu.time_varying_input"
TIME_VARYING_MODEL_EXPOSURE_COLUMNS = (
    "exposure_running_max_when_observed",
    "exposure_unmeasured_indicator",
)


class BinaryBaselineEncoding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["binary_indicator"]
    output_column: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    positive_level: str = Field(min_length=1)
    negative_level: str = Field(min_length=1)
    unknown_or_missing_policy: Literal["reject"]

    @model_validator(mode="after")
    def _distinct_levels(self) -> "BinaryBaselineEncoding":
        if self.positive_level == self.negative_level:
            raise ValueError("binary baseline levels must differ")
        return self


class TimeVaryingExposureSpecification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exposure_concept: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    exposure_summary: Literal["running_max_of_direct_measurements"]
    exposure_window_hours: Literal[24]
    followup: Literal["hospital_death_or_discharge_from_icu_admission"]
    missingness_policy: Literal["observed_state_indicator"]
    baseline_columns: tuple[str, ...]
    baseline_categorical_encodings: dict[str, BinaryBaselineEncoding] = Field(
        default_factory=dict
    )
    interpretation: Literal["descriptive_time_updated_association_not_causal"]

    @model_validator(mode="after")
    def _closed_baseline(self) -> "TimeVaryingExposureSpecification":
        if len(self.baseline_columns) != len(set(self.baseline_columns)):
            raise ValueError("baseline columns must be unique")
        if not set(self.baseline_categorical_encodings).issubset(self.baseline_columns):
            raise ValueError("categorical encodings must belong to the baseline roster")
        if self.exposure_concept in self.baseline_columns:
            raise ValueError(
                "time-updated exposure cannot also be a baseline adjustment"
            )
        if any(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None
            for value in self.baseline_columns
        ):
            raise ValueError("baseline columns must be canonical identifiers")
        reserved = {
            "analysis_stay_index",
            "analysis_cluster_index",
            "hospital_death",
            "patient_stay_id",
            "interval_start_hours",
            "interval_stop_hours",
        }
        if reserved.intersection(self.model_covariates) or len(
            self.model_covariates
        ) != len(set(self.model_covariates)):
            raise ValueError(
                "model columns collide with each other or runtime coordinates"
            )
        return self

    @property
    def model_covariates(self) -> tuple[str, ...]:
        return (
            *TIME_VARYING_MODEL_EXPOSURE_COLUMNS,
            *(
                self.baseline_categorical_encodings[column].output_column
                if column in self.baseline_categorical_encodings
                else column
                for column in self.baseline_columns
            ),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


__all__ = [
    "BinaryBaselineEncoding",
    "TimeVaryingExposureSpecification",
    "TIME_VARYING_EXPOSURE_METHOD",
    "TIME_VARYING_EXPOSURE_CAPABILITY",
    "TIME_VARYING_INPUT_METADATA_KEY",
    "TIME_VARYING_MODEL_EXPOSURE_COLUMNS",
    "TIME_VARYING_ANALYSIS_KIND",
]


G_METHOD_ACTION = "causal_emulation.g_methods"
G_METHOD_CAPABILITY = "causal_g_methods_v1"
G_METHOD_OUTPUTS = (
    "table:g_method_estimates", "table:g_method_weight_diagnostics",
    "log:g_method_receipt", "report:g_method_checklist",
)
G_METHOD_SOURCE_KEY = "easyicu.g_methods_source"
Column = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Lag = Annotated[int, Field(ge=0, le=24, strict=True)]


class GMethodError(ValueError):
    """Stable refusal; no alternative baseline estimator is authorized."""

    def __init__(self, code: str, message: str, *, period: int | None = None):
        super().__init__(message)
        self.code, self.period = code, period
        self.easyicu_safe_diagnostic = {"owner": G_METHOD_CAPABILITY, "reason_code": code, "period": period}


class GMethodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    estimator: Literal["longitudinal_msm", "point_gformula"]
    database: str = Field(min_length=1)
    identity_column: Column
    period_column: Column
    treatment_column: Column
    outcome_column: Column
    confounder_history: tuple[Column, ...] = Field(min_length=1, max_length=24)
    baseline_columns: tuple[Column, ...] = Field(max_length=24)
    treatment_lags: tuple[Lag, ...] = Field(max_length=24)
    confounder_lags: tuple[Lag, ...] = Field(max_length=24)
    intervals: int = Field(ge=1, le=24, strict=True)
    interval_hours: float = Field(gt=0)
    time_zero: Literal["icu_admission"] = "icu_admission"
    censoring_policy: Literal["complete_followup", "stabilized_ipcw"]
    censoring_column: Column | None
    ps_bounds: tuple[float, float]
    truncation_quantiles: tuple[float, float] | None
    n_bootstrap: Annotated[int, Field(ge=20, le=10000, strict=True)] | None
    random_seed: int = Field(ge=0, strict=True)
    require_exchangeability: Literal[True]
    require_correct_specification: Literal[True]
    require_consistency: Literal[True]

    @model_validator(mode="after")
    def _closed(self):
        names = [self.identity_column, self.period_column, self.treatment_column,
                 self.outcome_column, *self.confounder_history, *self.baseline_columns]
        if self.censoring_column:
            names.append(self.censoring_column)
        if len(names) != len(set(names)):
            raise ValueError("g-method source columns must have distinct roles")
        for lags in (self.treatment_lags, self.confounder_lags):
            if tuple(sorted(set(lags))) != lags:
                raise ValueError("history lags must be sorted and unique")
        if not 0 < self.ps_bounds[0] < self.ps_bounds[1] < 1:
            raise ValueError("propensity bounds must lie strictly inside (0, 1)")
        if self.truncation_quantiles is not None and not (
            0 <= self.truncation_quantiles[0] < self.truncation_quantiles[1] <= 1
        ):
            raise ValueError("truncation quantiles must satisfy 0 <= low < high <= 1")
        if (self.censoring_policy == "stabilized_ipcw") != (self.censoring_column is not None):
            raise ValueError("IPCW requires an explicit end-of-interval censoring indicator")
        if self.estimator == "longitudinal_msm":
            if self.n_bootstrap is not None:
                raise ValueError("MSM uses a person-cluster sandwich; full weight-estimation bootstrap is not implemented")
            if self.intervals < 2 or not self.treatment_lags or 0 in self.treatment_lags:
                raise ValueError("MSM requires multiple periods and positive treatment-history lags")
            if not self.confounder_lags or self.confounder_lags[0] != 0:
                raise ValueError("MSM requires current pretreatment confounders")
            if max((*self.treatment_lags, *self.confounder_lags)) >= self.intervals:
                raise ValueError("history lags must occur within the declared follow-up")
        elif (self.intervals != 1 or self.treatment_lags or self.confounder_lags != (0,)
              or self.censoring_policy != "complete_followup" or self.truncation_quantiles is not None
              or self.n_bootstrap is None):
            raise ValueError("point g-formula supports one interval, baseline confounders and no censoring/truncation")
        reserved = {"intercept", "treatment", *(f"treatment_lag_{lag}" for lag in self.treatment_lags),
                    *(f"period_{period}" for period in range(1, self.intervals))}
        if reserved.intersection(self.baseline_columns):
            raise ValueError("baseline stabilizers collide with generated outcome terms")
        return self

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.identity_column, self.period_column, self.treatment_column,
                self.outcome_column, *self.confounder_history, *self.baseline_columns,
                *((self.censoring_column,) if self.censoring_column else ()))

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


def g_method_serialization(spec: GMethodSpec) -> str:
    validated = GMethodSpec.model_validate(spec.model_dump())
    return f"g_methods({canonical_json(validated.model_dump(mode='json'))})"


def g_method_for_step(step: object) -> GMethodSpec | None:
    method = str(getattr(step, "method", "") or "")
    action = getattr(step, "scientific_action_id", None)
    if not method.startswith("g_methods(") and action != G_METHOD_ACTION:
        return None
    try:
        if not method.startswith("g_methods(") or not method.endswith(")") or action != G_METHOD_ACTION:
            raise ValueError("declare the exact g-method action and typed method")
        return GMethodSpec.model_validate_json(method[len("g_methods("):-1])
    except (ValueError, ValidationError) as exc:
        raise GMethodError("g_methods_declaration_required", "Declare the complete reviewed g-method specification.") from exc


__all__ += ["G_METHOD_ACTION", "G_METHOD_CAPABILITY", "G_METHOD_OUTPUTS", "G_METHOD_SOURCE_KEY", "GMethodError", "GMethodSpec", "g_method_serialization", "g_method_for_step"]
