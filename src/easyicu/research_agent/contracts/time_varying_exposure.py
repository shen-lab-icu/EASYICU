"""Plan-owned specification for an early running-maximum exposure model.

This contract chooses no columns or missingness strategy for a caller. It is
reviewed with the plan, then shared by acquisition and the deterministic Cox
executor. It describes an association, not a causal or publication authority.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical_json import canonical_sha256


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
