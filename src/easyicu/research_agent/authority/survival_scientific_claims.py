"""Host claims for the hazard ratios the signed landmark survival suite reports.

The suite reports one constant adjusted hazard ratio when its prespecified
proportional-hazards test does not reject the assumption, and adjusted
interval-specific hazard ratios from its prespecified extended Cox model
when the data allow them.  An interval model the data left without an
estimate is reported with its reason; beside a rejected test, where the
interval estimates are the result, it is refused.  Without host claims the
strict Results grammar admits none of these estimates, and a survival
manuscript's Conclusion has no claim to read.

The suite opts in with a versioned ``easyicu.survival_reporting/2`` envelope.
Its coordinates are typed here, the reported estimate follows from the
envelope's own test result, and every sentence is the fixed association or
rule-outcome template over those fields.  An older ``/1`` envelope stays
readable and claims nothing.

A suite that declares the prevalence-definition sensitivity analysis adds its
re-fits to the envelope.  Each estimable fit repeats the primary estimand on
the restricted risk set and is claimed as a sensitivity analysis, never as the
answer.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..methods.time_varying_cox import TimeVaryingNotEstimableReason
from .prespecified_rule_outcomes import (
    RULE_OUTCOMES_KEY,
    ProportionalHazardsTestOutcome,
    derive_rule_outcome_claim_payloads,
    rule_outcome_payload,
)

SURVIVAL_REPORTING_KEY = "reportable_survival_results"
SURVIVAL_REPORTING_SCHEMA_VERSION = "easyicu.survival_reporting/2"
CONSTANT_HAZARD_RATIO_CLAIM_ID = "adjusted_hazard_ratio"


def interval_hazard_ratio_claim_id(position: int) -> str:
    """The claim id of the ``position``-th (1-based) follow-up interval."""

    return f"interval_{position}_adjusted_hazard_ratio"


def prevalence_sensitivity_claim_id(hours: float, position: int | None = None) -> str:
    """The claim id of one sensitivity re-fit: its constant or ``position``-th interval estimate."""

    token = f"{hours:g}".replace(".", "p")
    estimate = (
        CONSTANT_HAZARD_RATIO_CLAIM_ID
        if position is None
        else interval_hazard_ratio_claim_id(position)
    )
    return f"prevalence_cutoff_{token}h_{estimate}"


class _HazardRatio(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    hazard_ratio: float = Field(gt=0.0)
    ci_low: float = Field(gt=0.0)
    ci_high: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _interval_contains_estimate(self) -> "_HazardRatio":
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


class _IntervalHazardRatio(_HazardRatio):
    start_days: float = Field(ge=0.0)
    end_days: float
    p_value: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _interval_has_width(self) -> "_IntervalHazardRatio":
        if self.end_days <= self.start_days:
            raise ValueError("a follow-up interval must end after it starts")
        return self


class _TimeVaryingAssociation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    method: Literal["piecewise_time_varying_cox"]
    adjustment_columns: list[str]
    intervals: list[_IntervalHazardRatio] = Field(min_length=1)

    @model_validator(mode="after")
    def _intervals_tile_the_follow_up(self) -> "_TimeVaryingAssociation":
        if self.intervals[0].start_days != 0.0 or any(
            later.start_days != earlier.end_days
            for earlier, later in zip(self.intervals, self.intervals[1:])
        ):
            raise ValueError("time-varying intervals must tile follow-up from the landmark")
        return self


class _TimeVaryingNotEstimable(BaseModel):
    """A prespecified interval model the data left without an estimate, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["not_estimable"]
    reason: TimeVaryingNotEstimableReason
    method: Literal["piecewise_time_varying_cox"]
    adjustment_columns: list[str]


def _interval_estimates(
    association: "_TimeVaryingAssociation | _TimeVaryingNotEstimable",
) -> list[_IntervalHazardRatio]:
    """The interval estimates; none when the model was not estimable."""

    if isinstance(association, _TimeVaryingAssociation):
        return list(association.intervals)
    return []


class _PrevalenceSensitivityFit(BaseModel):
    """One re-fit that also excluded exposed records first recorded by ``hours``.

    It carries the primary's estimand, which the envelope checks: the constant
    hazard ratio when that is authorized, else every interval's.  A fit the
    risk set could not support carries neither and is reported without an
    estimate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    prevalent_exposure_cutoff_hours: float = Field(gt=0.0)
    n_analysis: int = Field(ge=0)
    n_events: int = Field(ge=0)
    adjusted_hazard_ratio: _HazardRatio | None = None
    interval_hazard_ratios: list[_HazardRatio] | None = None


class _PrevalenceSensitivity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    axis: Literal["prevalence_definition"]
    fits: list[_PrevalenceSensitivityFit] = Field(min_length=1)

    @model_validator(mode="after")
    def _hours_increase(self) -> "_PrevalenceSensitivity":
        hours = [fit.prevalent_exposure_cutoff_hours for fit in self.fits]
        if any(later <= earlier for earlier, later in zip(hours, hours[1:])):
            raise ValueError("sensitivity fits must follow increasing cutoffs")
        return self


class SurvivalReporting(BaseModel):
    """The suite's reporting envelope; fields the claims do not read pass through."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.survival_reporting/2"]
    execution_owner: Literal["landmark_survival_executor_v1"]
    interpretation_ceiling: Literal["descriptive_prognostic_association_not_causal"]
    exposure: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    analysis_unit: str = Field(min_length=1)
    landmark_hours: float = Field(gt=0.0)
    contrast: str = Field(min_length=1)
    adjustment_columns: list[str]
    # Present exactly when the constant estimate is a result; an envelope
    # signed before the PH fence may still carry it beside a rejected test.
    adjusted_hazard_ratio: _HazardRatio | None = None
    constant_hazard_ratio_authorized: bool
    proportional_hazards_status: str = Field(min_length=1)
    proportional_hazards_test: ProportionalHazardsTestOutcome
    rmst: dict[str, Any]
    time_varying_adjusted_association: Union[
        _TimeVaryingAssociation, _TimeVaryingNotEstimable
    ]
    manuscript_projection: dict[str, Any]
    prevalence_definition_sensitivity: _PrevalenceSensitivity | None = None

    @model_validator(mode="after")
    def _estimate_follows_from_the_test(self) -> "SurvivalReporting":
        rejected = self.proportional_hazards_test.disposition == "assumption_rejected"
        if self.constant_hazard_ratio_authorized == rejected:
            raise ValueError(
                "a constant hazard ratio is authorized exactly when the PH test does not reject"
            )
        if self.constant_hazard_ratio_authorized and self.adjusted_hazard_ratio is None:
            raise ValueError("an authorized constant hazard ratio must be reported")
        if self.proportional_hazards_status.startswith("violation_") != rejected:
            raise ValueError("the PH status contradicts the PH test outcome")
        if self.time_varying_adjusted_association.adjustment_columns != self.adjustment_columns:
            raise ValueError("the interval model must adjust for the sealed covariates")
        if rejected and not isinstance(
            self.time_varying_adjusted_association, _TimeVaryingAssociation
        ):
            raise ValueError(
                "a rejected PH test makes the interval estimates the result"
            )
        if len(set(self.adjustment_columns)) != len(self.adjustment_columns) or any(
            not column.strip() for column in self.adjustment_columns
        ):
            raise ValueError("survival adjustment columns must be unique and non-empty")
        intervals = len(_interval_estimates(self.time_varying_adjusted_association))
        for fit in (
            self.prevalence_definition_sensitivity.fits
            if self.prevalence_definition_sensitivity is not None
            else ()
        ):
            if (
                fit.adjusted_hazard_ratio is not None
                and not self.constant_hazard_ratio_authorized
            ) or (
                fit.interval_hazard_ratios is not None
                and (
                    self.constant_hazard_ratio_authorized
                    or len(fit.interval_hazard_ratios) != intervals
                )
            ):
                raise ValueError("a sensitivity fit must repeat the primary estimand")
        return self


def survival_reporting_requests_claims(summary: Mapping[str, Any]) -> bool:
    reporting = summary.get(SURVIVAL_REPORTING_KEY)
    return (
        isinstance(reporting, Mapping)
        and reporting.get("schema_version") == SURVIVAL_REPORTING_SCHEMA_VERSION
    )


def survival_claim_ids(reporting: SurvivalReporting) -> tuple[str, ...]:
    """The hazard-ratio claims, in reporting order; the PH rule claim follows them."""

    constant = (
        (CONSTANT_HAZARD_RATIO_CLAIM_ID,)
        if reporting.constant_hazard_ratio_authorized
        else ()
    )
    intervals = _interval_estimates(reporting.time_varying_adjusted_association)
    return constant + tuple(
        interval_hazard_ratio_claim_id(position)
        for position in range(1, len(intervals) + 1)
    ) + tuple(claim_id for claim_id, _estimate, _hours, _estimand in _sensitivity_estimates(reporting))


def _sensitivity_estimates(reporting: SurvivalReporting):
    """Each estimable re-fit's estimates: claim id, estimate, hours and estimand."""

    sensitivity = reporting.prevalence_definition_sensitivity
    primary_intervals = _interval_estimates(reporting.time_varying_adjusted_association)
    for fit in sensitivity.fits if sensitivity is not None else ():
        hours = fit.prevalent_exposure_cutoff_hours
        if fit.adjusted_hazard_ratio is not None:
            yield (
                prevalence_sensitivity_claim_id(hours),
                fit.adjusted_hazard_ratio,
                hours,
                "adjusted hazard ratio over the post-landmark follow-up",
            )
        for position, (estimate, interval) in enumerate(
            zip(fit.interval_hazard_ratios or (), primary_intervals), start=1
        ):
            yield (
                prevalence_sensitivity_claim_id(hours, position),
                estimate,
                hours,
                f"adjusted hazard ratio for days {interval.start_days:g} to "
                f"{interval.end_days:g} after the landmark",
            )


def derive_survival_claim_payloads(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compile the suite's hazard ratios and PH decision into claim payloads.

    The association claims come first, so a Conclusion that reads the first
    primary claim reads an association rather than the test that chose it.
    """

    if summary.get("status") != "ok":
        raise ValueError("survival claims require a completed owner summary")
    reporting = SurvivalReporting.model_validate(summary[SURVIVAL_REPORTING_KEY])
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

    def association(claim_id: str, estimate: _HazardRatio, *, estimand: str, role: str):
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
    if reporting.constant_hazard_ratio_authorized:
        assert reporting.adjusted_hazard_ratio is not None
        payloads.append(association(
            CONSTANT_HAZARD_RATIO_CLAIM_ID, reporting.adjusted_hazard_ratio,
            estimand="adjusted hazard ratio over the post-landmark follow-up",
            role="primary",
        ))
    # When the assumption is rejected the interval estimates replace the
    # constant one; otherwise they are the prespecified secondary model.
    interval_role = "secondary" if reporting.constant_hazard_ratio_authorized else "primary"
    for position, interval in enumerate(
        _interval_estimates(reporting.time_varying_adjusted_association), start=1
    ):
        payloads.append(association(
            interval_hazard_ratio_claim_id(position), interval,
            estimand=(
                f"adjusted hazard ratio for days {interval.start_days:g} to "
                f"{interval.end_days:g} after the landmark"
            ),
            role=interval_role,
        ))
    # A re-fit on a restricted risk set is a sensitivity analysis of the
    # prevalence definition: its population says which records it excluded.
    for claim_id, estimate, hours, estimand in _sensitivity_estimates(reporting):
        payloads.append({
            **association(claim_id, estimate, estimand=estimand, role="sensitivity"),
            "population": (
                f"{common['population']}, excluding exposed records first recorded "
                f"at or before hour {hours:g}"
            ),
        })
    payloads.extend(derive_rule_outcome_claim_payloads({
        "status": "ok",
        RULE_OUTCOMES_KEY: [rule_outcome_payload(reporting.proportional_hazards_test)],
    }))
    return payloads


__all__ = [
    "CONSTANT_HAZARD_RATIO_CLAIM_ID",
    "SURVIVAL_REPORTING_KEY",
    "SURVIVAL_REPORTING_SCHEMA_VERSION",
    "SurvivalReporting",
    "derive_survival_claim_payloads",
    "interval_hazard_ratio_claim_id",
    "prevalence_sensitivity_claim_id",
    "survival_claim_ids",
    "survival_reporting_requests_claims",
]
