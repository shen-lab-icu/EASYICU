"""Host claims for the hazard ratios of the continuous survival suite.

The continuous-exposure landmark suite reports its association per one
readable step of the exposure: ``exposure_increment``, one, two or five times
a power of ten of the source's scale.  Two prespecified rules choose the
result.  When the proportional-hazards test rejects, the per-step hazard
ratios of the interval model are the result; the data must allow them, or the
suite fails.  Otherwise, when the spline check rejects a linear term, the
spline's hazard ratios at two percentiles of the exposure relative to its
median are the result; else one per-step hazard ratio is.  The interval model
is then a secondary model the data may leave without an estimate, and the
envelope says why.  The suite opts in with a versioned
``easyicu.continuous_survival_reporting/2`` envelope under the same key as the
binary suite's; the two schemas never read each other.  Every sentence is the
fixed association or rule-outcome template over typed fields.
"""

from __future__ import annotations

from decimal import Decimal
import math
from typing import Annotated, Any, Literal, Mapping, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from ..contracts.executed_method_design import (
    exposure_step_text,
    is_round_exposure_step,
)
from ..methods.time_varying_cox import TimeVaryingNotEstimableReason
from .prespecified_rule_outcomes import (
    RULE_OUTCOME_SCHEMA_VERSION,
    RULE_OUTCOMES_KEY,
    FunctionalFormNotAssessableReason,
    FunctionalFormTestOutcome,
    ProportionalHazardsTestOutcome,
    derive_rule_outcome_claim_payloads,
    rule_outcome_payload,
)

CONTINUOUS_SURVIVAL_REPORTING_KEY = "reportable_survival_results"
CONTINUOUS_SURVIVAL_REPORTING_SCHEMA_VERSION = "easyicu.continuous_survival_reporting/2"
PER_UNIT_HAZARD_RATIO_CLAIM_ID = "adjusted_hazard_ratio_per_unit"
#: The estimate the suite's two rules leave as its result.
PrimaryEstimate = Literal[
    "per_step_hazard_ratio",
    "spline_percentile_contrasts",
    "interval_per_step_hazard_ratios",
]


def interval_per_unit_hazard_ratio_claim_id(position: int) -> str:
    """The claim id of the ``position``-th (1-based) follow-up interval."""

    return f"interval_{position}_adjusted_hazard_ratio_per_unit"


def spline_contrast_claim_id(percentile: float) -> str:
    """The claim id of the spline's hazard ratio at one percentile of the exposure."""

    return f"spline_hazard_ratio_at_percentile_{percentile:g}"


def primary_claim_ids(
    primary_estimate: str,
    *,
    interval_count: int,
    contrast_percentiles: Sequence[float],
) -> tuple[str, ...]:
    """The claims that state the suite's result, in reporting order."""

    if primary_estimate == "per_step_hazard_ratio":
        return (PER_UNIT_HAZARD_RATIO_CLAIM_ID,)
    if primary_estimate == "spline_percentile_contrasts":
        if len(contrast_percentiles) != 2:
            raise ValueError("the spline result is two percentile contrasts")
        return tuple(spline_contrast_claim_id(value) for value in contrast_percentiles)
    if primary_estimate == "interval_per_step_hazard_ratios":
        if interval_count <= 1:
            raise ValueError("the interval result needs every follow-up interval")
        return tuple(
            interval_per_unit_hazard_ratio_claim_id(position)
            for position in range(1, interval_count + 1)
        )
    raise ValueError(f"unknown continuous survival result {primary_estimate!r}")


def _plain(value: float) -> str:
    """Four significant digits in plain decimals, as readers and the binder read them."""

    if value == 0.0:
        return "0"
    places = 3 - math.floor(math.log10(abs(value)))
    return format(Decimal(repr(round(value, places))).normalize(), "f")


def _ordinal(value: float) -> str:
    number = int(value)
    suffix = (
        "th"
        if 10 <= number % 100 <= 20
        else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    )
    return f"{number}{suffix}"


class _PerStepHazardRatio(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    hazard_ratio: float = Field(gt=0.0)
    ci_low: float = Field(gt=0.0)
    ci_high: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _interval_contains_estimate(self) -> "_PerStepHazardRatio":
        if not self.ci_low <= self.hazard_ratio <= self.ci_high:
            raise ValueError("a hazard-ratio interval must contain its estimate")
        return self

    @property
    def direction(self) -> str:
        if self.ci_low > 1.0:
            return "positive"
        if self.ci_high < 1.0:
            return "negative"
        return "no_clear_association"


class _IntervalPerStepHazardRatio(_PerStepHazardRatio):
    start_days: float = Field(ge=0.0)
    end_days: float
    p_value: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _interval_has_width(self) -> "_IntervalPerStepHazardRatio":
        if self.end_days <= self.start_days:
            raise ValueError("a follow-up interval must end after it starts")
        return self


class _IntervalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["estimated"]
    method: Literal["piecewise_time_varying_cox"]
    adjustment_columns: list[str]
    intervals: list[_IntervalPerStepHazardRatio] = Field(min_length=2)

    @model_validator(mode="after")
    def _intervals_tile_the_follow_up(self) -> "_IntervalModel":
        if self.intervals[0].start_days != 0.0 or any(
            later.start_days != earlier.end_days
            for earlier, later in zip(self.intervals, self.intervals[1:])
        ):
            raise ValueError("time-varying intervals must tile follow-up from the landmark")
        return self


class _IntervalModelNotEstimable(BaseModel):
    """A prespecified interval model the data left without an estimate, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["not_estimable"]
    reason: TimeVaryingNotEstimableReason
    method: Literal["piecewise_time_varying_cox"]
    adjustment_columns: list[str]


class _SplineContrast(_PerStepHazardRatio):
    """The spline model's hazard ratio at one percentile of the exposure against its median."""

    percentile: float = Field(gt=0.0, lt=100.0)
    exposure_value: float
    reference_value: float

    @model_validator(mode="after")
    def _contrast_has_two_values(self) -> "_SplineContrast":
        if self.exposure_value == self.reference_value:
            raise ValueError("a spline contrast compares two different exposure values")
        return self


class _FunctionalFormCheck(BaseModel):
    """The likelihood-ratio test of a restricted cubic spline against the linear term.

    It is judged at the prespecified alpha.  The test of the spline terms
    against no exposure term answers whether there is an association at all;
    it is absent when the model without the exposure had no result.  The
    percentile contrasts are present exactly when they are the suite's result.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    method: Literal["restricted_cubic_spline_likelihood_ratio_test"]
    status: Literal["estimated"]
    knot_percentiles: list[float]
    knots: list[float] = Field(min_length=3, max_length=3)
    reference_value: float
    likelihood_ratio_statistic: float = Field(ge=0.0)
    degrees_of_freedom: Literal[1]
    p_value: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(gt=0.0, lt=1.0)
    disposition: Literal["linearity_rejected", "linearity_not_rejected"]
    overall_likelihood_ratio_statistic: float | None = Field(default=None, ge=0.0)
    overall_degrees_of_freedom: Literal[2] | None = None
    overall_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    contrasts: list[_SplineContrast] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _omit_unstated_fields(self, handler):
        payload = handler(self)
        for name in (
            "overall_likelihood_ratio_statistic",
            "overall_degrees_of_freedom",
            "overall_p_value",
        ):
            if getattr(self, name) is None:
                payload.pop(name, None)
        if not self.contrasts:
            payload.pop("contrasts", None)
        return payload

    @model_validator(mode="after")
    def _check_is_coherent(self) -> "_FunctionalFormCheck":
        if self.knot_percentiles != [10.0, 50.0, 90.0] or any(
            later <= earlier for earlier, later in zip(self.knots, self.knots[1:])
        ):
            raise ValueError("the spline knots are increasing values at 10/50/90")
        expected = (
            "linearity_rejected"
            if self.p_value < self.alpha
            else "linearity_not_rejected"
        )
        if self.disposition != expected:
            raise ValueError("the spline check's disposition contradicts its p value")
        overall = (
            self.overall_likelihood_ratio_statistic,
            self.overall_degrees_of_freedom,
            self.overall_p_value,
        )
        if any(value is None for value in overall) and any(
            value is not None for value in overall
        ):
            raise ValueError(
                "the overall association test states all of its fields or none"
            )
        if self.contrasts:
            if [item.percentile for item in self.contrasts] != [
                self.knot_percentiles[0],
                self.knot_percentiles[-1],
            ]:
                raise ValueError(
                    "the spline contrasts sit at the boundary knot percentiles"
                )
            if any(
                item.reference_value != self.reference_value for item in self.contrasts
            ):
                raise ValueError(
                    "the spline contrasts share the check's median reference"
                )
        return self


class _FunctionalFormNotEstimable(BaseModel):
    """A spline check without a result: its knots tied, or its fit failed."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["restricted_cubic_spline_likelihood_ratio_test"]
    status: Literal["not_estimable"]
    reason: FunctionalFormNotAssessableReason
    knot_percentiles: list[float]
    alpha: float = Field(gt=0.0, lt=1.0)
    disposition: Literal["not_assessable"]


class ContinuousSurvivalReporting(BaseModel):
    """The suite's reporting envelope; fields the claims do not read pass through."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.continuous_survival_reporting/2"]
    execution_owner: Literal["landmark_continuous_survival_executor_v1"]
    interpretation_ceiling: Literal["descriptive_prognostic_association_not_causal"]
    exposure: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    analysis_unit: str = Field(min_length=1)
    landmark_hours: float = Field(gt=0.0)
    exposure_increment: float = Field(gt=0.0)
    exposure_unit: str | None = None
    adjustment_columns: list[str]
    primary_estimate: PrimaryEstimate
    # Present exactly when the per-step estimate is the result.
    adjusted_hazard_ratio_per_unit: _PerStepHazardRatio | None = None
    #: The PH test let hazard ratios constant over follow-up stand.
    constant_hazard_ratio_authorized: bool
    proportional_hazards_status: str = Field(min_length=1)
    proportional_hazards_test: ProportionalHazardsTestOutcome
    time_varying_adjusted_association: Annotated[
        Union[_IntervalModel, _IntervalModelNotEstimable],
        Field(discriminator="status"),
    ]
    functional_form: Annotated[
        Union[_FunctionalFormCheck, _FunctionalFormNotEstimable],
        Field(discriminator="status"),
    ]
    manuscript_projection: dict[str, Any]

    @model_validator(mode="after")
    def _estimate_follows_from_the_rules(self) -> "ContinuousSurvivalReporting":
        if not is_round_exposure_step(self.exposure_increment):
            raise ValueError(
                "the exposure step is one, two or five times a power of ten"
            )
        rejected = self.proportional_hazards_test.disposition == "assumption_rejected"
        if self.constant_hazard_ratio_authorized == rejected:
            raise ValueError(
                "constant hazard ratios are authorized exactly when the PH test does not reject"
            )
        if self.proportional_hazards_status.startswith("violation_") != rejected:
            raise ValueError("the PH status contradicts the PH test outcome")
        if rejected:
            expected = "interval_per_step_hazard_ratios"
        elif self.linearity_rejected:
            expected = "spline_percentile_contrasts"
        else:
            expected = "per_step_hazard_ratio"
        if self.primary_estimate != expected:
            raise ValueError(
                "the result follows from the PH test, then from the spline check"
            )
        if (self.adjusted_hazard_ratio_per_unit is not None) != (
            expected == "per_step_hazard_ratio"
        ):
            raise ValueError(
                "the per-step hazard ratio is reported exactly when it is the result"
            )
        if bool(self.spline_contrasts) != (expected == "spline_percentile_contrasts"):
            raise ValueError(
                "the spline contrasts are reported exactly when they are the result"
            )
        if rejected and self.time_varying_adjusted_association.status != "estimated":
            raise ValueError(
                "a rejected PH test leaves the interval estimates as the result, so "
                "they must exist"
            )
        if (
            self.time_varying_adjusted_association.adjustment_columns
            != self.adjustment_columns
        ):
            raise ValueError("the interval model must adjust for the sealed covariates")
        if len(set(self.adjustment_columns)) != len(self.adjustment_columns) or any(
            not column.strip() for column in self.adjustment_columns
        ):
            raise ValueError("survival adjustment columns must be unique and non-empty")
        return self

    @property
    def linearity_rejected(self) -> bool:
        return (
            getattr(self.functional_form, "disposition", None) == "linearity_rejected"
        )

    @property
    def interval_estimates(self) -> tuple[_IntervalPerStepHazardRatio, ...]:
        """The interval model's per-step hazard ratios; none when it had no estimate."""

        model = self.time_varying_adjusted_association
        return tuple(model.intervals) if isinstance(model, _IntervalModel) else ()

    @property
    def spline_contrasts(self) -> tuple[_SplineContrast, ...]:
        """The spline's percentile contrasts; present only when they are the result."""

        check = self.functional_form
        return tuple(check.contrasts) if isinstance(check, _FunctionalFormCheck) else ()

    @property
    def adjusted(self) -> bool:
        """Whether the models adjusted for any covariate."""

        return bool(self.adjustment_columns)

    @property
    def per_step_words(self) -> str:
        """``per-50 U/L``: the step as a premodifier of the ratio's name.

        The numeric binder reads a number after "hazard ratio" in a
        sentence as a ratio; the step is not one, so it comes first.
        """

        step = exposure_step_text(self.exposure_increment)
        if self.exposure_unit is None:
            return f"per-{step}-unit"
        return f"per-{step} {self.exposure_unit}"

    def functional_form_outcome(self) -> FunctionalFormTestOutcome:
        """The spline check as the prespecified rule outcome a host claim states."""

        check = self.functional_form
        estimated = isinstance(check, _FunctionalFormCheck)
        return FunctionalFormTestOutcome(
            schema_version=RULE_OUTCOME_SCHEMA_VERSION,
            rule="functional_form_test",
            diagnostic="restricted_cubic_spline_likelihood_ratio_test",
            alpha=check.alpha,
            nonlinearity_p_value=check.p_value if estimated else None,
            disposition=check.disposition,
            not_assessable_reason=None if estimated else check.reason,
            primary_estimate=self.primary_estimate,
        )


def continuous_survival_reporting_requests_claims(summary: Mapping[str, Any]) -> bool:
    reporting = summary.get(CONTINUOUS_SURVIVAL_REPORTING_KEY)
    return (
        isinstance(reporting, Mapping)
        and reporting.get("schema_version")
        == CONTINUOUS_SURVIVAL_REPORTING_SCHEMA_VERSION
    )


def continuous_survival_claim_ids(reporting: ContinuousSurvivalReporting) -> tuple[str, ...]:
    """The hazard-ratio claims, in reporting order; the rule claims follow them."""

    primary = primary_claim_ids(
        reporting.primary_estimate,
        interval_count=len(reporting.interval_estimates),
        contrast_percentiles=[item.percentile for item in reporting.spline_contrasts],
    )
    if reporting.primary_estimate == "interval_per_step_hazard_ratios":
        return primary
    return primary + tuple(
        interval_per_unit_hazard_ratio_claim_id(position)
        for position in range(1, len(reporting.interval_estimates) + 1)
    )


def derive_continuous_survival_claim_payloads(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compile the suite's hazard ratios and its two rule outcomes into claims.

    The association claims come first, so a Conclusion that reads the first
    primary claim reads an association rather than a test that chose it.
    """

    if summary.get("status") != "ok":
        raise ValueError("survival claims require a completed owner summary")
    reporting = ContinuousSurvivalReporting.model_validate(
        summary[CONTINUOUS_SURVIVAL_REPORTING_KEY]
    )
    adjusted = "adjusted " if reporting.adjusted else ""
    unit = f" {reporting.exposure_unit}" if reporting.exposure_unit else ""
    common = {
        "claim_type": "association",
        "exposure": reporting.exposure,
        "outcome": reporting.outcome,
        "population": (
            f"the {reporting.analysis_unit} alive and under observation at the "
            f"{reporting.landmark_hours:g}-hour landmark"
        ),
        "status": "supported",
        "adjusted_for": list(reporting.adjustment_columns),
    }

    def association(
        claim_id: str, estimate: _PerStepHazardRatio, *, estimand: str, role: str
    ) -> dict[str, Any]:
        return {
            **common,
            "claim_id": claim_id,
            "direction": estimate.direction,
            "estimand": estimand,
            "analysis_role": role,
            "point_estimate": estimate.hazard_ratio,
            "interval_lower": estimate.ci_low,
            "interval_upper": estimate.ci_high,
        }

    payloads: list[dict[str, Any]] = []
    if reporting.adjusted_hazard_ratio_per_unit is not None:
        payloads.append(
            association(
                PER_UNIT_HAZARD_RATIO_CLAIM_ID,
                reporting.adjusted_hazard_ratio_per_unit,
                estimand=(
                    f"{reporting.per_step_words} {adjusted}hazard ratio over the "
                    "post-landmark follow-up"
                ),
                role="primary",
            )
        )
    for contrast in reporting.spline_contrasts:
        payloads.append(
            association(
                spline_contrast_claim_id(contrast.percentile),
                contrast,
                # The exposure values precede the ratio's name, as the step does.
                estimand=(
                    f"{_ordinal(contrast.percentile)}-percentile-versus-median "
                    f"({_plain(contrast.exposure_value)} versus "
                    f"{_plain(contrast.reference_value)}{unit}) {adjusted}hazard "
                    "ratio from the restricted cubic spline model, over the "
                    "post-landmark follow-up"
                ),
                role="primary",
            )
        )
    # When the PH test rejects, the interval estimates are the result;
    # otherwise they are the prespecified secondary model.  A rejected linear
    # term leaves each a per-step summary of an average log-linear trend.
    interval_role = (
        "primary"
        if reporting.primary_estimate == "interval_per_step_hazard_ratios"
        else "secondary"
    )
    trend = (
        ", an average log-linear trend within the interval"
        if reporting.linearity_rejected
        else ""
    )
    for position, interval in enumerate(reporting.interval_estimates, start=1):
        payloads.append(
            association(
                interval_per_unit_hazard_ratio_claim_id(position),
                interval,
                estimand=(
                    f"{reporting.per_step_words} {adjusted}hazard ratio for days "
                    f"{interval.start_days:g} to {interval.end_days:g} after the "
                    f"landmark{trend}"
                ),
                role=interval_role,
            )
        )
    payloads.extend(
        derive_rule_outcome_claim_payloads(
            {
                "status": "ok",
                RULE_OUTCOMES_KEY: [
                    rule_outcome_payload(reporting.proportional_hazards_test),
                    rule_outcome_payload(reporting.functional_form_outcome()),
                ],
            }
        )
    )
    return payloads


__all__ = [
    "CONTINUOUS_SURVIVAL_REPORTING_KEY",
    "CONTINUOUS_SURVIVAL_REPORTING_SCHEMA_VERSION",
    "ContinuousSurvivalReporting",
    "PER_UNIT_HAZARD_RATIO_CLAIM_ID",
    "PrimaryEstimate",
    "continuous_survival_claim_ids",
    "continuous_survival_reporting_requests_claims",
    "derive_continuous_survival_claim_payloads",
    "interval_per_unit_hazard_ratio_claim_id",
    "primary_claim_ids",
    "spline_contrast_claim_id",
]
