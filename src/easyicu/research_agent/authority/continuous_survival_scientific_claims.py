"""Host claims for the per-unit hazard ratios of the continuous survival suite.

The continuous-exposure landmark suite reports one adjusted hazard ratio per
unit of the exposure when its prespecified proportional-hazards test does not
reject the assumption, and the per-unit hazard ratios of its prespecified
interval model whenever the data allow them.  Without them the envelope says
why; a rejected test then leaves the suite without a result, so it fails.  It opts in with a versioned
``easyicu.continuous_survival_reporting/1`` envelope under the same key as the
binary suite's; the two schemas never read each other.  The reported estimate
follows from the envelope's own test result, and every sentence is the fixed
association or rule-outcome template over typed fields.

The spline check of the linear term is reported, never claimed: it chooses no
estimate.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..methods.time_varying_cox import TimeVaryingNotEstimableReason
from .prespecified_rule_outcomes import (
    RULE_OUTCOMES_KEY,
    ProportionalHazardsTestOutcome,
    derive_rule_outcome_claim_payloads,
    rule_outcome_payload,
)

CONTINUOUS_SURVIVAL_REPORTING_KEY = "reportable_survival_results"
CONTINUOUS_SURVIVAL_REPORTING_SCHEMA_VERSION = "easyicu.continuous_survival_reporting/1"
PER_UNIT_HAZARD_RATIO_CLAIM_ID = "adjusted_hazard_ratio_per_unit"


def interval_per_unit_hazard_ratio_claim_id(position: int) -> str:
    """The claim id of the ``position``-th (1-based) follow-up interval."""

    return f"interval_{position}_adjusted_hazard_ratio_per_unit"


class _PerUnitHazardRatio(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    hazard_ratio: float = Field(gt=0.0)
    ci_low: float = Field(gt=0.0)
    ci_high: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _interval_contains_estimate(self) -> "_PerUnitHazardRatio":
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


class _IntervalPerUnitHazardRatio(_PerUnitHazardRatio):
    start_days: float = Field(ge=0.0)
    end_days: float
    p_value: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _interval_has_width(self) -> "_IntervalPerUnitHazardRatio":
        if self.end_days <= self.start_days:
            raise ValueError("a follow-up interval must end after it starts")
        return self


class _IntervalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["estimated"]
    method: Literal["piecewise_time_varying_cox"]
    adjustment_columns: list[str]
    intervals: list[_IntervalPerUnitHazardRatio] = Field(min_length=2)

    @model_validator(mode="after")
    def _intervals_tile_the_follow_up(self) -> "_IntervalModel":
        if self.intervals[0].start_days != 0.0 or any(
            later.start_days != earlier.end_days
            for earlier, later in zip(self.intervals, self.intervals[1:])
        ):
            raise ValueError("time-varying intervals must tile follow-up from the landmark")
        return self


class _FunctionalFormCheck(BaseModel):
    """The likelihood-ratio test of a restricted cubic spline against the linear term."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    method: Literal["restricted_cubic_spline_likelihood_ratio_test"]
    status: Literal["estimated"]
    knot_percentiles: list[float]
    knots: list[float] = Field(min_length=3, max_length=3)
    reference_value: float
    likelihood_ratio_statistic: float = Field(ge=0.0)
    degrees_of_freedom: Literal[1]
    p_value: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _knots_are_the_frozen_percentiles(self) -> "_FunctionalFormCheck":
        if self.knot_percentiles != [10.0, 50.0, 90.0] or any(
            later <= earlier for earlier, later in zip(self.knots, self.knots[1:])
        ):
            raise ValueError("the spline knots are increasing values at 10/50/90")
        return self


class _FunctionalFormNotEstimable(BaseModel):
    """A spline check without a result: its knots tied, or its fit failed."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["restricted_cubic_spline_likelihood_ratio_test"]
    status: Literal["not_estimable"]
    reason: Literal["tied_knots", "spline_model_not_estimable"]
    knot_percentiles: list[float]


class _IntervalModelNotEstimable(BaseModel):
    """A prespecified interval model the data left without an estimate, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["not_estimable"]
    reason: TimeVaryingNotEstimableReason
    method: Literal["piecewise_time_varying_cox"]
    adjustment_columns: list[str]


class ContinuousSurvivalReporting(BaseModel):
    """The suite's reporting envelope; fields the claims do not read pass through."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.continuous_survival_reporting/1"]
    execution_owner: Literal["landmark_continuous_survival_executor_v1"]
    interpretation_ceiling: Literal["descriptive_prognostic_association_not_causal"]
    exposure: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    analysis_unit: str = Field(min_length=1)
    landmark_hours: float = Field(gt=0.0)
    exposure_increment: float = Field(gt=0.0)
    exposure_unit: str | None = None
    adjustment_columns: list[str]
    # Present exactly when the constant estimate is a result.
    adjusted_hazard_ratio_per_unit: _PerUnitHazardRatio | None = None
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
    def _estimate_follows_from_the_test(self) -> "ContinuousSurvivalReporting":
        rejected = self.proportional_hazards_test.disposition == "assumption_rejected"
        if self.constant_hazard_ratio_authorized == rejected:
            raise ValueError(
                "a constant hazard ratio is authorized exactly when the PH test does not reject"
            )
        if self.constant_hazard_ratio_authorized != (
            self.adjusted_hazard_ratio_per_unit is not None
        ):
            raise ValueError(
                "the constant per-unit hazard ratio is reported exactly when authorized"
            )
        if self.proportional_hazards_status.startswith("violation_") != rejected:
            raise ValueError("the PH status contradicts the PH test outcome")
        if rejected and self.time_varying_adjusted_association.status != "estimated":
            raise ValueError(
                "a rejected PH test leaves the interval estimates as the result, so "
                "they must exist"
            )
        if self.time_varying_adjusted_association.adjustment_columns != self.adjustment_columns:
            raise ValueError("the interval model must adjust for the sealed covariates")
        if len(set(self.adjustment_columns)) != len(self.adjustment_columns) or any(
            not column.strip() for column in self.adjustment_columns
        ):
            raise ValueError("survival adjustment columns must be unique and non-empty")
        return self

    @property
    def interval_estimates(self) -> tuple[_IntervalPerUnitHazardRatio, ...]:
        """The interval model's per-unit hazard ratios; none when it had no estimate."""

        model = self.time_varying_adjusted_association
        return tuple(model.intervals) if isinstance(model, _IntervalModel) else ()

    @property
    def per_unit_words(self) -> str:
        """``per 1-mmol/L increase``-style words for the estimand."""

        step = f"{self.exposure_increment:g}"
        if self.exposure_unit is None:
            return f"per {step}-unit increase in the exposure"
        return f"per {step} {self.exposure_unit} increase in the exposure"


def continuous_survival_reporting_requests_claims(summary: Mapping[str, Any]) -> bool:
    reporting = summary.get(CONTINUOUS_SURVIVAL_REPORTING_KEY)
    return (
        isinstance(reporting, Mapping)
        and reporting.get("schema_version")
        == CONTINUOUS_SURVIVAL_REPORTING_SCHEMA_VERSION
    )


def continuous_survival_claim_ids(reporting: ContinuousSurvivalReporting) -> tuple[str, ...]:
    """The hazard-ratio claims, in reporting order; the PH rule claim follows them."""

    constant = (
        (PER_UNIT_HAZARD_RATIO_CLAIM_ID,)
        if reporting.constant_hazard_ratio_authorized
        else ()
    )
    return constant + tuple(
        interval_per_unit_hazard_ratio_claim_id(position)
        for position in range(1, len(reporting.interval_estimates) + 1)
    )


def derive_continuous_survival_claim_payloads(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compile the suite's per-unit hazard ratios and PH decision into claims.

    The association claims come first, so a Conclusion that reads the first
    primary claim reads an association rather than the test that chose it.
    """

    if summary.get("status") != "ok":
        raise ValueError("survival claims require a completed owner summary")
    reporting = ContinuousSurvivalReporting.model_validate(
        summary[CONTINUOUS_SURVIVAL_REPORTING_KEY]
    )
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
        claim_id: str, estimate: _PerUnitHazardRatio, *, estimand: str, role: str
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
        payloads.append(association(
            PER_UNIT_HAZARD_RATIO_CLAIM_ID, reporting.adjusted_hazard_ratio_per_unit,
            estimand=(
                f"adjusted hazard ratio {reporting.per_unit_words} over the "
                "post-landmark follow-up"
            ),
            role="primary",
        ))
    # When the assumption is rejected the interval estimates replace the
    # constant one; otherwise they are the prespecified secondary model.
    interval_role = "secondary" if reporting.constant_hazard_ratio_authorized else "primary"
    for position, interval in enumerate(reporting.interval_estimates, start=1):
        payloads.append(association(
            interval_per_unit_hazard_ratio_claim_id(position), interval,
            estimand=(
                f"adjusted hazard ratio {reporting.per_unit_words} for days "
                f"{interval.start_days:g} to {interval.end_days:g} after the landmark"
            ),
            role=interval_role,
        ))
    payloads.extend(derive_rule_outcome_claim_payloads({
        "status": "ok",
        RULE_OUTCOMES_KEY: [rule_outcome_payload(reporting.proportional_hazards_test)],
    }))
    return payloads


__all__ = [
    "CONTINUOUS_SURVIVAL_REPORTING_KEY",
    "CONTINUOUS_SURVIVAL_REPORTING_SCHEMA_VERSION",
    "ContinuousSurvivalReporting",
    "PER_UNIT_HAZARD_RATIO_CLAIM_ID",
    "continuous_survival_claim_ids",
    "continuous_survival_reporting_requests_claims",
    "derive_continuous_survival_claim_payloads",
    "interval_per_unit_hazard_ratio_claim_id",
]
