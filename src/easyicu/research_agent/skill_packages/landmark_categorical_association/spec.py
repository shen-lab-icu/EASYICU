"""Typed specification for the fixed-landmark categorical association skill.

The specification is the only channel through which scientific choices enter
the scripts.  It names the exposure and its closed level set, the binary
outcome and its event-time/follow-up columns, the landmark, the exact
covariate roster, the patient-level dependence contract and the prespecified
sensitivity axes.  The scripts never infer any of these from the data.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...contracts.dependence import PlannedDependenceRequirement
from ...contracts.model_terms import ModelTermSpec, level_spelling

SKILL_ID = "landmark_categorical_association"
SKILL_VERSION = "easyicu.skill.landmark_categorical_association/1"

CovariateCoding = Literal["continuous", "binary", "categorical"]


class CovariateSpec(BaseModel):
    """Exactly how one adjustment variable enters the primary model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    coding: CovariateCoding
    levels: Optional[List[str]] = None
    reference_level: Optional[str] = None
    label: Optional[str] = None

    @model_validator(mode="after")
    def _closed_levels(self) -> "CovariateSpec":
        if self.coding == "continuous":
            if self.levels is not None or self.reference_level is not None:
                raise ValueError(
                    f"continuous covariate {self.name!r} must not declare levels"
                )
            return self
        levels = [level_spelling(item) for item in (self.levels or [])]
        if len(levels) < 2 or len(levels) != len(set(levels)):
            raise ValueError(
                f"covariate {self.name!r} needs at least two unique declared levels"
            )
        if self.coding == "binary" and len(levels) != 2:
            raise ValueError(f"binary covariate {self.name!r} needs exactly two levels")
        if self.reference_level is None or level_spelling(self.reference_level) not in levels:
            raise ValueError(
                f"covariate {self.name!r} reference_level must be one of its levels"
            )
        return self

    def model_term(self) -> ModelTermSpec:
        if self.coding == "continuous":
            return ModelTermSpec(
                name=self.name, role="covariate", coding="continuous", transform="identity"
            )
        return ModelTermSpec(
            name=self.name,
            role="covariate",
            coding=self.coding,
            levels=list(self.levels or []),
            reference_level=self.reference_level,
            transform="treatment_contrast",
        )


class SecondaryOutcomeSpec(BaseModel):
    """One prespecified continuous secondary outcome summarised by exposure level."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    label: Optional[str] = None
    #: Deaths shorten a length of stay; the two denominators are reported
    #: side by side instead of choosing one silently.
    report_survivors_separately: bool = True


class LandmarkCategoricalSpec(BaseModel):
    """Complete, reviewable design of one fixed-landmark categorical association."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.skill.landmark_categorical_association/1"] = (
        SKILL_VERSION
    )
    title: str = Field(min_length=1)
    identity_column: str = Field(min_length=1)

    exposure: str = Field(min_length=1)
    exposure_levels: List[str] = Field(min_length=2)
    reference_level: str
    primary_contrast_level: str
    exposure_label: Optional[str] = None
    #: How a row whose exposure is missing is described.  It is never recoded to
    #: the reference level; it stays in every descriptive denominator and leaves
    #: every model.
    unknown_level_label: str = "unknown"

    outcome: str = Field(min_length=1)
    outcome_label: Optional[str] = None
    event_time_column: str = Field(min_length=1)
    observation_duration_column: str = Field(min_length=1)
    observation_duration_unit: Literal["hours", "days"] = "hours"
    landmark_hours: float = Field(gt=0.0)

    covariates: List[CovariateSpec] = Field(min_length=1)
    dependence: Optional[PlannedDependenceRequirement] = None

    alternate_exposures: List[str] = Field(default_factory=list)
    first_stay_column: Optional[str] = None
    functional_form_covariates: List[str] = Field(default_factory=list)
    functional_form_knots: int = Field(default=4, ge=3, le=7)
    secondary_outcomes: List[SecondaryOutcomeSpec] = Field(default_factory=list)
    measurement_audit_columns: List[str] = Field(default_factory=list)

    #: Reporting thresholds.  They are declared here so the report can quote the
    #: rule it applied rather than a number with no origin.
    epv_minimum: float = Field(default=10.0, gt=0.0)
    complete_case_warning_share: float = Field(default=0.20, gt=0.0, lt=1.0)
    unknown_exposure_warning_share: float = Field(default=0.10, gt=0.0, lt=1.0)
    small_reference_group_n: int = Field(default=50, ge=1)
    sparse_level_events: int = Field(default=5, ge=1)

    display_labels: dict[str, str] = Field(default_factory=dict)
    claim_ceiling: Literal["analysis_only"] = "analysis_only"

    @field_validator("reference_level", "primary_contrast_level", mode="before")
    @classmethod
    def _spell_levels(cls, value: Any) -> str:
        spelled = level_spelling(value)
        if not spelled:
            raise ValueError("exposure reference/contrast levels must be non-empty")
        return spelled

    @field_validator("exposure_levels", mode="before")
    @classmethod
    def _closed_exposure_levels(cls, values: List[Any]) -> List[str]:
        levels = [level_spelling(item) for item in values]
        if any(not item for item in levels):
            raise ValueError("exposure levels must be non-empty")
        if len(levels) != len(set(levels)):
            raise ValueError("exposure levels must be unique")
        return levels

    @model_validator(mode="after")
    def _closed_design(self) -> "LandmarkCategoricalSpec":
        reference = self.reference_level
        primary = self.primary_contrast_level
        if reference not in self.exposure_levels:
            raise ValueError("reference_level must be one of exposure_levels")
        if primary not in self.exposure_levels or primary == reference:
            raise ValueError(
                "primary_contrast_level must be a non-reference exposure level"
            )
        names = [item.name for item in self.covariates]
        if len(names) != len(set(names)):
            raise ValueError("covariate names must be unique")
        reserved = {
            self.exposure,
            self.outcome,
            self.event_time_column,
            self.observation_duration_column,
            self.identity_column,
        }
        if reserved & set(names):
            raise ValueError("a covariate cannot double as exposure, outcome or time")
        for column in self.functional_form_covariates:
            spec = next((item for item in self.covariates if item.name == column), None)
            if spec is None or spec.coding != "continuous":
                raise ValueError(
                    f"functional_form covariate {column!r} must be a declared "
                    "continuous covariate"
                )
        if len(set(self.alternate_exposures)) != len(self.alternate_exposures):
            raise ValueError("alternate_exposures must be unique")
        if self.exposure in self.alternate_exposures:
            raise ValueError("the primary exposure is not an alternate exposure")
        if self.dependence is not None and self.dependence.group_source != self.identity_column:
            raise ValueError(
                "dependence.group_source must be the declared identity column"
            )
        return self

    # -- derived contracts -------------------------------------------------

    @property
    def non_reference_levels(self) -> tuple[str, ...]:
        return tuple(
            level for level in self.exposure_levels if level != self.reference_level
        )

    def exposure_term(self, *, exposure: Optional[str] = None) -> ModelTermSpec:
        return ModelTermSpec(
            name=exposure or self.exposure,
            role="exposure",
            coding="categorical",
            levels=list(self.exposure_levels),
            reference_level=self.reference_level,
            transform="treatment_contrast",
        )

    def ordinal_trend_term(self) -> ModelTermSpec:
        return ModelTermSpec(
            name=self.exposure,
            role="exposure",
            coding="ordinal_linear",
            levels=list(self.exposure_levels),
            reference_level=None,
            transform="declared_level_index",
        )

    def model_terms(
        self,
        *,
        exposure: Optional[str] = None,
        exclude_covariates: tuple[str, ...] = (),
    ) -> list[ModelTermSpec]:
        terms = [self.exposure_term(exposure=exposure)]
        terms.extend(
            item.model_term()
            for item in self.covariates
            if item.name not in exclude_covariates
        )
        return terms

    def covariate_names(self) -> list[str]:
        return [item.name for item in self.covariates]

    def label(self, column: str) -> str:
        if column in self.display_labels:
            return self.display_labels[column]
        if column == self.exposure and self.exposure_label:
            return self.exposure_label
        if column == self.outcome and self.outcome_label:
            return self.outcome_label
        for item in self.covariates:
            if item.name == column and item.label:
                return item.label
        for item in self.secondary_outcomes:
            if item.name == column and item.label:
                return item.label
        return column

    def required_columns(self) -> list[str]:
        columns = [
            self.identity_column,
            self.exposure,
            self.outcome,
            self.event_time_column,
            self.observation_duration_column,
            *self.covariate_names(),
            *self.alternate_exposures,
            *[item.name for item in self.secondary_outcomes],
        ]
        if self.first_stay_column:
            columns.append(self.first_stay_column)
        return list(dict.fromkeys(columns))

    def optional_columns(self) -> list[str]:
        return list(dict.fromkeys(self.measurement_audit_columns))

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


__all__ = [
    "SKILL_ID",
    "SKILL_VERSION",
    "CovariateSpec",
    "LandmarkCategoricalSpec",
    "SecondaryOutcomeSpec",
]
