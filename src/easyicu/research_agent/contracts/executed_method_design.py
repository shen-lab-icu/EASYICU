"""An owner's receipt of the design it executed.

A signed owner applies a design fixed before the run: a time grid read from
its anchor, an eligibility minimum, a model and its selection rule.  The run
context's time window describes what the host materialized for the cohort,
not what a longitudinal owner read, and the strict Methods grammar admits a
numeric design detail only as an exact host fact.  An owner therefore states
its executed design as a closed, versioned block (``executed_method_design``
in its step summary); the reporting owner renders it and never infers it.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_serializer,
    model_validator,
)

from ..methods.time_varying_cox import TimeVaryingNotEstimableReason

EXECUTED_METHOD_DESIGN_KEY = "executed_method_design"
EXECUTED_METHOD_DESIGN_SCHEMA_VERSION = "easyicu.executed_method_design/1"

#: Which exposure tertile a risk set left empty, and so why a continuous suite
#: described the whole risk set.
WholeRiskSetReason = Literal[
    "upper_tertile_cutpoint_at_maximum",
    "no_value_between_tertile_cutpoints",
    "lower_tertile_cutpoint_at_maximum",
]
#: The one reader clause for each reason; a table note, a figure legend and
#: the Methods state it alike.
WHOLE_RISK_SET_REASON_WORDS: dict[str, str] = {
    "upper_tertile_cutpoint_at_maximum": (
        "the upper tertile cutpoint of the exposure was its largest recorded value, "
        "so no record lay above it"
    ),
    "no_value_between_tertile_cutpoints": (
        "no recorded value of the exposure lay between its two tertile cutpoints"
    ),
    "lower_tertile_cutpoint_at_maximum": (
        "the lower tertile cutpoint of the exposure was its largest recorded value, "
        "so no record lay above it"
    ),
}

#: The spread of the modelled exposure a continuous suite read its reporting
#: step from: the interquartile range, or, for a heaped exposure whose
#: narrower spreads are zero, the range between its tenth and ninetieth
#: percentiles, then its range.
ExposureIncrementSpread = Literal[
    "interquartile_range", "central_eighty_percent_range", "range"
]
#: Reader words for each spread, without digits: the Methods bind every
#: number they state.
EXPOSURE_INCREMENT_SPREAD_WORDS: dict[str, str] = {
    "interquartile_range": "interquartile range",
    "central_eighty_percent_range": (
        "range between its tenth and ninetieth percentiles"
    ),
    "range": "range",
}


def is_round_exposure_step(value: Any) -> bool:
    """Whether ``value`` is one, two or five times a power of ten."""

    if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
        return False
    digits = Decimal(repr(value)).normalize().as_tuple().digits
    return len(digits) == 1 and digits[0] in (1, 2, 5)


def exposure_step_text(value: float) -> str:
    """The step as readers see it: its decimal digits, never an exponent."""

    return format(Decimal(repr(float(value))).normalize(), "f")


class _ExecutedDesign(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.executed_method_design/1"]


class FixedWindowRepresentationDesign(_ExecutedDesign):
    """Per-window summaries on a fixed grid, and the eligibility minimum."""

    design_kind: Literal["fixed_window_representation"]
    anchor: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    window_start_hours: int
    window_end_hours: int
    window_width_hours: int = Field(gt=0)
    n_windows: int = Field(ge=1)
    window_aggregation: Literal["max"]
    minimum_observed_windows: int = Field(ge=1)
    #: What makes a window count toward the minimum: any SOFA-2 coordinate with
    #: an owner-available value, directly observed or carried forward by the
    #: SOFA-2 owner.  A design written before this field omits it.
    window_evidence: Literal["any_available_sofa2_score"] | None = None

    @model_serializer(mode="wrap")
    def _preserve_unstated_window_evidence(self, handler):
        payload = handler(self)
        if self.window_evidence is None:
            payload.pop("window_evidence", None)
        return payload

    @model_validator(mode="after")
    def _grid_tiles_the_window(self) -> "FixedWindowRepresentationDesign":
        if (
            self.window_end_hours <= self.window_start_hours
            or self.window_end_hours - self.window_start_hours
            != self.n_windows * self.window_width_hours
        ):
            raise ValueError("executed time grid does not tile its window")
        if self.minimum_observed_windows > self.n_windows:
            raise ValueError("the window minimum exceeds the grid")
        return self


class LatentClassModelDesign(_ExecutedDesign):
    """The class model, its coordinate scale, and its selection rule."""

    design_kind: Literal["latent_class_model"]
    model_family: Literal[
        "latent_class_diagonal_gaussian_mixture", "latent_class_mixed_mode"
    ]
    coordinate_scaling: Literal[
        "pooled_coordinate_wise_z_score", "continuous_coordinate_wise_z_score"
    ]
    candidate_class_counts: list[int]
    selection_criterion: Literal["bic"]
    minimum_class_fraction: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _grid_is_increasing(self) -> "LatentClassModelDesign":
        counts = self.candidate_class_counts
        if (
            len(counts) < 2
            or counts[0] < 2
            or any(later <= earlier for earlier, later in zip(counts, counts[1:]))
        ):
            raise ValueError("candidate class counts must be an increasing grid from 2")
        # The mixed-mode model keeps ordinal levels; only a Gaussian-only model
        # z-scores every coordinate.
        if (self.model_family == "latent_class_mixed_mode") != (
            self.coordinate_scaling == "continuous_coordinate_wise_z_score"
        ):
            raise ValueError("the class model and its coordinate scaling disagree")
        return self


class LandmarkSurvivalDesign(_ExecutedDesign):
    """The landmark risk set, the model, its diagnostic and its alternatives.

    Times are hours or days from the time origin; the cutpoints and the
    restricted-mean horizon are days after the landmark, the scale the
    survival model runs on.
    """

    design_kind: Literal["landmark_survival"]
    time_origin: str = Field(min_length=1, max_length=80)
    landmark_hours: float = Field(gt=0)
    endpoint_horizon_days: float = Field(gt=0)
    prevalent_exposure_cutoff_hours: float
    exposure_window_end_hours: float = Field(gt=0)
    n_adjustment_covariates: int = Field(ge=0)
    effect_model: Literal["cox_proportional_hazards_efron_ties"]
    interval_method: Literal["wald_95_ci"]
    proportional_hazards_test: Literal["schoenfeld_residuals"]
    proportional_hazards_alpha: float = Field(gt=0.0, lt=1.0)
    time_varying_cutpoints_days: list[float]
    rmst_horizon_days: float | None = Field(default=None, gt=0)
    #: How the exposure was timed: by its first record as present.  A design
    #: written before this field omits it; its suite timed the exposure by its
    #: first record of any value.
    exposure_onset_representation: Literal["first_truthy_event_time"] | None = None
    #: The prevalence-definition sensitivity analysis: each fit also excluded
    #: exposed records first recorded by one of these hours.  A design without
    #: that analysis omits the field.
    prevalence_sensitivity_cutoffs_hours: list[float] | None = None
    #: Why the prespecified interval model had no estimate; a design whose
    #: interval model was estimated, or that has none, omits it, as one
    #: written before it does.
    interval_model_not_estimable_reason: TimeVaryingNotEstimableReason | None = None

    @model_serializer(mode="wrap")
    def _preserve_unstated_fields(self, handler):
        payload = handler(self)
        for name in (
            "exposure_onset_representation",
            "prevalence_sensitivity_cutoffs_hours",
            "interval_model_not_estimable_reason",
        ):
            if getattr(self, name) is None:
                payload.pop(name, None)
        return payload

    @model_validator(mode="after")
    def _times_are_ordered(self) -> "LandmarkSurvivalDesign":
        followup_days = self.endpoint_horizon_days - self.landmark_hours / 24.0
        cutpoints = self.time_varying_cutpoints_days
        if followup_days <= 0:
            raise ValueError("the landmark is not before the endpoint horizon")
        if not (
            self.prevalent_exposure_cutoff_hours
            < self.exposure_window_end_hours
            <= self.landmark_hours
        ):
            raise ValueError("the exposure window does not close by the landmark")
        if any(
            later <= earlier for earlier, later in zip(cutpoints, cutpoints[1:])
        ) or any(not 0 < cut < followup_days for cut in cutpoints):
            raise ValueError("time-varying cutpoints must increase within follow-up")
        if self.rmst_horizon_days is not None and abs(
            self.rmst_horizon_days - followup_days
        ) > 1e-9:
            raise ValueError("the restricted-mean horizon is not the follow-up end")
        if self.interval_model_not_estimable_reason is not None and not cutpoints:
            raise ValueError("only a prespecified interval model can be not estimable")
        hours = self.prevalence_sensitivity_cutoffs_hours
        if hours is not None and (
            not hours
            or any(later <= earlier for earlier, later in zip(hours, hours[1:]))
            or not self.prevalent_exposure_cutoff_hours < hours[0]
            or hours[-1] >= self.exposure_window_end_hours
        ):
            raise ValueError(
                "sensitivity cutoffs must increase between the cutoff and the window end"
            )
        return self


class LandmarkContinuousSurvivalDesign(_ExecutedDesign):
    """The landmark risk set of a continuous exposure, its model and its checks.

    The exposure is one window summary recorded by the landmark, modelled per
    ``exposure_increment`` units of the source's scale: one, two or five
    times a power of ten, read from the spread ``exposure_increment_spread``
    names.  Times are hours or days from the time origin; the cutpoints are
    days after the landmark.
    """

    design_kind: Literal["landmark_continuous_survival"]
    time_origin: str = Field(min_length=1, max_length=80)
    landmark_hours: float = Field(gt=0)
    endpoint_horizon_days: float = Field(gt=0)
    exposure_window_start_hours: float = Field(ge=0)
    exposure_window_end_hours: float = Field(gt=0)
    exposure_window_summary: Literal["max", "min", "mean", "first"]
    exposure_increment: float = Field(gt=0)
    exposure_increment_spread: ExposureIncrementSpread
    exposure_unit: str | None = Field(min_length=1, max_length=40)
    n_adjustment_covariates: int = Field(ge=0)
    effect_model: Literal["cox_proportional_hazards_efron_ties"]
    interval_method: Literal["wald_95_ci"]
    proportional_hazards_test: Literal["schoenfeld_residuals"]
    proportional_hazards_alpha: float = Field(gt=0.0, lt=1.0)
    time_varying_cutpoints_days: list[float] = Field(min_length=1)
    spline_knot_percentiles: list[float]
    #: The prespecified alpha of the spline check of the linear term.
    functional_form_alpha: float = Field(gt=0.0, lt=1.0)
    #: The groups Table 1 and the Kaplan-Meier curves described: the sealed
    #: value tertiles, or the whole risk set when a tertile would be empty.
    descriptive_grouping: Literal["value_tertiles", "whole_risk_set"]
    descriptive_grouping_reason: WholeRiskSetReason | None = None
    #: Why the prespecified interval model had no estimate; a design whose
    #: interval model was estimated omits it, as one written before it does.
    interval_model_not_estimable_reason: TimeVaryingNotEstimableReason | None = None

    @model_serializer(mode="wrap")
    def _preserve_unstated_fields(self, handler):
        payload = handler(self)
        for name in (
            "descriptive_grouping_reason",
            "interval_model_not_estimable_reason",
        ):
            if getattr(self, name) is None:
                payload.pop(name, None)
        return payload

    @model_validator(mode="after")
    def _times_are_ordered(self) -> "LandmarkContinuousSurvivalDesign":
        followup_days = self.endpoint_horizon_days - self.landmark_hours / 24.0
        cutpoints = self.time_varying_cutpoints_days
        if followup_days <= 0:
            raise ValueError("the landmark is not before the endpoint horizon")
        if not (
            self.exposure_window_start_hours
            < self.exposure_window_end_hours
            <= self.landmark_hours
        ):
            raise ValueError("the exposure window does not close by the landmark")
        if any(
            later <= earlier for earlier, later in zip(cutpoints, cutpoints[1:])
        ) or any(not 0 < cut < followup_days for cut in cutpoints):
            raise ValueError("time-varying cutpoints must increase within follow-up")
        if self.spline_knot_percentiles != [10.0, 50.0, 90.0]:
            raise ValueError("the spline knots are the 10th, 50th and 90th percentiles")
        if not is_round_exposure_step(self.exposure_increment):
            raise ValueError(
                "the exposure step is one, two or five times a power of ten"
            )
        if (self.descriptive_grouping == "whole_risk_set") != (
            self.descriptive_grouping_reason is not None
        ):
            raise ValueError(
                "the whole risk set is described for a stated reason, and only then"
            )
        return self


ExecutedMethodDesign = Annotated[
    Union[
        FixedWindowRepresentationDesign,
        LatentClassModelDesign,
        LandmarkSurvivalDesign,
        LandmarkContinuousSurvivalDesign,
    ],
    Field(discriminator="design_kind"),
]
_DESIGN_ADAPTER: TypeAdapter[Any] = TypeAdapter(ExecutedMethodDesign)


def validate_executed_method_design(payload: object) -> Any:
    """Parse one owner's design block into its closed type (fail closed)."""

    return _DESIGN_ADAPTER.validate_python(payload)


def executed_method_design_payload(design: Any) -> dict[str, Any]:
    return design.model_dump(mode="json")


__all__ = [
    "EXECUTED_METHOD_DESIGN_KEY",
    "EXECUTED_METHOD_DESIGN_SCHEMA_VERSION",
    "EXPOSURE_INCREMENT_SPREAD_WORDS",
    "WHOLE_RISK_SET_REASON_WORDS",
    "ExecutedMethodDesign",
    "ExposureIncrementSpread",
    "FixedWindowRepresentationDesign",
    "LandmarkContinuousSurvivalDesign",
    "LandmarkSurvivalDesign",
    "LatentClassModelDesign",
    "WholeRiskSetReason",
    "executed_method_design_payload",
    "exposure_step_text",
    "is_round_exposure_step",
    "validate_executed_method_design",
]
