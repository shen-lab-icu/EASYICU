"""Host coordinates for explicit, unrouted longitudinal family templates.

Signed suites retain the actual authority, including its digest and product
roster. Dynamic prediction is the existing analysis-only Coder composition;
its reviewed kernels construct features/labels and evaluate supplied scores.
No coordinates here select a family from a research question.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...authority.target_trial_runtime import TargetTrialRuntimeAuthority
from ...authority.time_varying_runtime import TimeVaryingRuntimeAuthority

TARGET_TRIAL_FAMILY_ID = "target_trial_emulation"
TIME_VARYING_SURVIVAL_FAMILY_ID = "time_varying_exposure_survival"
DYNAMIC_LANDMARK_FAMILY_ID = "dynamic_landmark_prediction"
LONGITUDINAL_FAMILY_IDS = frozenset({
    TARGET_TRIAL_FAMILY_ID, TIME_VARYING_SURVIVAL_FAMILY_ID, DYNAMIC_LANDMARK_FAMILY_ID,
})
Column = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")]


class DynamicLandmarkCoordinates(BaseModel):
    """Prespecified times and a patient-grouped, numeric sklearn Pipeline.

    Times are hours after ICU admission. Features use [landmark-lookback,
    landmark); labels use [landmark, landmark+horizon), as the kernels do.
    The Coder fits the specified Pipeline separately per landmark/horizon.
    The metric kernel supplies AUROC, Brier and calibration bins, without CIs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    trajectory_bound: Literal[True]
    trajectory_time_unit: Literal["hours"]
    trajectory_time_origin: Literal["icu_admission"]
    patient_column: Column
    event_time_column: Column
    followup_end_column: Column
    feature_concepts: list[Column] = Field(min_length=1, max_length=16)
    landmark_hours: list[float] = Field(min_length=2, max_length=12)
    lookback_hours: float = Field(gt=0)
    horizon_hours: list[float] = Field(min_length=1, max_length=6)
    aggregations: list[Literal["last", "mean", "min", "max"]] = Field(min_length=1)
    split_unit: Literal["patient"] = "patient"
    validation_fraction: float = Field(default=0.25, gt=0, lt=1)
    seed: int = Field(default=1729, ge=0, le=2**32 - 1)
    model_pipeline: Literal["median_standardized_logistic"] = "median_standardized_logistic"
    calibration_bins: int = Field(default=10, ge=2, le=100)
    evidence_ceiling: Literal["analysis_only"] = "analysis_only"

    @field_validator("landmark_hours", "horizon_hours")
    @classmethod
    def _ordered_times(cls, values: list[float], info) -> list[float]:
        landmark = info.field_name == "landmark_hours"
        if any(not math.isfinite(value) or (value < 0 if landmark else value <= 0) for value in values):
            raise ValueError("landmarks must be nonnegative and horizons positive, all finite")
        if values != sorted(set(values)):
            raise ValueError("times must be distinct and increasing")
        return values

    @field_validator("feature_concepts", "aggregations")
    @classmethod
    def _unique(cls, values):
        if len(values) != len(set(values)):
            raise ValueError("feature concepts and aggregations must be unique")
        return values

    @model_validator(mode="after")
    def _separate_roles(self):
        reserved = {self.patient_column, self.event_time_column, self.followup_end_column}
        if len(reserved) != 3 or reserved.intersection(self.feature_concepts):
            raise ValueError("patient, event time, follow-up and features must have distinct roles")
        return self

    @property
    def source_columns(self) -> tuple[str, ...]:
        return (self.patient_column, self.event_time_column, self.followup_end_column, *self.feature_concepts)


def longitudinal_request_problem(request: Any) -> tuple[str, str, str] | None:
    """Return a typed refusal's code, explanation and path, without routing."""

    fields = {
        TARGET_TRIAL_FAMILY_ID: "sealed_target_trial",
        TIME_VARYING_SURVIVAL_FAMILY_ID: "sealed_time_varying",
        DYNAMIC_LANDMARK_FAMILY_ID: "dynamic_landmark",
    }
    present = [name for name in fields.values() if getattr(request, name) is not None]
    wanted = fields.get(request.family_id)
    if present != ([wanted] if wanted else []):
        return ("family_spec_longitudinal_coordinates", "The family requires exactly its own host coordinates.", wanted or "family_id")
    if wanted is None:
        return None
    if request.cohort_selection_mode != "all_input_rows" or any(value is not None for value in request.typed_cohort_bounds):
        return ("family_spec_longitudinal_population", "Use the bound input cohort; the longitudinal owner applies its own risk sets.", "cohort_selection_mode")
    # These coordinates belong to existing templates and would otherwise be
    # silently ignored by a new owner. Keep the explicit surface narrow.
    unsupported = (
        "sealed_suite", "proposed_suite", "sealed_continuous_suite", "proposed_continuous_suite",
        "sealed_trajectory", "sealed_feasibility", "landmark_hours", "landmark_spec_id",
        "event_time_column", "observation_duration_column", "observation_duration_unit",
        "prediction_death_time", "concept_cohort_definition", "concept_cohort_window_end_hours",
        "feature_candidates", "accepted_feature_groups", "accepted_baseline_rows", "membership_candidates",
        "secondary_continuous_outcome", "subgroup_columns", "alternate_exposures", "first_stay",
        "complete_case_spec_id", "functional_form_spec_ids", "population_concepts", "study_population_occurrence",
        "benchmark_candidates", "benchmark_omissions", "distribution_missing_policies",
        "distribution_missingness_unobserved", "caller_binds_all_input_rows", "counts_only",
        "exposure_companion_columns", "observation_window_hours",
    )
    for name in unsupported:
        if getattr(request, name):
            return ("family_spec_longitudinal_coordinate_unsupported", "This coordinate has no owner in this template.", name)
    if request.execution_tier != "article":
        return ("family_spec_longitudinal_coordinate_unsupported", "This template has no first-result subset.", "execution_tier")
    if request.outcome_levels != ["0", "1"] or request.event_level_index != 1:
        return ("family_spec_longitudinal_endpoint", "The longitudinal owners require the declared 0/1 event coding.", "outcome_levels")
    if request.exposure_levels or request.exposure_is_ordered or request.reference_level_index or request.primary_contrast_level_index:
        return ("family_spec_longitudinal_exposure", "These owners do not select categorical contrasts.", "exposure_levels")
    owner = getattr(request, wanted)
    if request.family_id == TARGET_TRIAL_FAMILY_ID:
        expected = ("causal_inference", owner.unit_id_column, owner.event_column, "none", "")
        columns = owner.required_columns
        roster = [item.column for item in owner.covariates]
        clustered = owner.resampling_unit == "patient"
    elif request.family_id == TIME_VARYING_SURVIVAL_FAMILY_ID:
        expected = ("association_study", owner.identity_column, owner.outcome_column, "continuous", owner.exposure_column)
        columns, roster, clustered = owner.required_columns, list(owner.specification.baseline_columns), True
        if request.requires_visualization:
            return ("family_spec_longitudinal_product_unowned", "The time-varying owner has no figure, PH, absolute-risk or competing-event product.", "requires_visualization")
    else:
        expected = ("dynamic_prediction", "stay_id", request.outcome, "none", "")
        columns, roster, clustered = ("stay_id", request.outcome, *owner.source_columns), [], True
        if not request.outcome or request.outcome in owner.source_columns or "stay_id" in owner.source_columns:
            return ("family_spec_longitudinal_endpoint", "Identity, outcome and longitudinal source roles must be distinct.", "dynamic_landmark")
    actual = (request.analysis_type, request.identity_column, request.outcome, request.exposure_kind, request.primary_exposure)
    if actual != expected or request.cluster_unit != ("patient" if clustered else None):
        return ("family_spec_longitudinal_owner_mismatch", "The request must preserve the owner's family, identity, endpoint, exposure and dependence.", wanted)
    if request.adjustment_selection != "exact" or request.exact_roster != roster:
        return ("family_spec_longitudinal_roster_mismatch", "The host coordinates own the adjustment roster.", "exact_roster")
    if not set(columns).issubset(request.variable_roster):
        return ("family_spec_longitudinal_source_missing", "The request lacks source columns required by its owner.", "variable_roster")
    if len(request.variable_roster) > 24:
        return ("family_spec_roster_exceeds_design", "The design contract permits at most 24 source variables.", "variable_roster")
    if not set(request.measurement_audit_columns).issubset(request.variable_roster):
        return ("family_spec_longitudinal_source_missing", "Measurement audit columns must be in the bound roster.", "measurement_audit_columns")
    return None


__all__ = [
    "DYNAMIC_LANDMARK_FAMILY_ID", "DynamicLandmarkCoordinates", "LONGITUDINAL_FAMILY_IDS",
    "TARGET_TRIAL_FAMILY_ID", "TIME_VARYING_SURVIVAL_FAMILY_ID",
    "TargetTrialRuntimeAuthority", "TimeVaryingRuntimeAuthority", "longitudinal_request_problem",
]
