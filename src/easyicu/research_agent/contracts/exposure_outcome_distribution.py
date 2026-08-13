"""Typed exposure-by-outcome distribution and absolute-risk contracts.

This owner is dependency-neutral: planning declares the scientific design,
the host binds any repeated-unit authority, and execution consumes the same
immutable Pydantic models.  ``research_agent.schema`` re-exports the models
for compatibility with existing callers.
"""

from __future__ import annotations

import math
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .closed_levels import typed_level_key, validate_closed_scalar_levels
from .dependence import PlannedDependenceRequirement


def _closed_levels(values: List[Any], *, label: str) -> List[Any]:
    return validate_closed_scalar_levels(values, label=label)


class ExposureOutcomeRiskDifferenceContrast(BaseModel):
    """One prespecified unadjusted absolute-risk contrast.

    The order is explicit because ``comparison - reference`` and its inverse
    answer different questions.  The executor must not choose the two levels
    from their sort order, observed frequency, or clinical-looking labels.
    Covariance is intentionally not declared here: the host binds the exact
    repeated-unit authority on the enclosing distribution specification.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.exposure_outcome_risk_difference/1"] = (
        "easyicu.exposure_outcome_risk_difference/1"
    )
    reference_exposure_level: Any
    comparison_exposure_level: Any
    effect_measure: Literal["risk_difference"] = "risk_difference"
    interval_method: Literal["linear_probability_wald"] = (
        "linear_probability_wald"
    )

    @field_validator("reference_exposure_level", "comparison_exposure_level")
    @classmethod
    def _closed_contrast_level(cls, value: Any) -> Any:
        return _closed_levels(
            [value], label="exposure_outcome risk-difference contrast level"
        )[0]

    @model_validator(mode="after")
    def _distinct_levels(self) -> "ExposureOutcomeRiskDifferenceContrast":
        if typed_level_key(self.reference_exposure_level) == typed_level_key(
            self.comparison_exposure_level
        ):
            raise ValueError(
                "risk-difference comparison and reference levels must differ"
            )
        return self


class ExposureOutcomeDistributionSpec(BaseModel):
    """Planner-owned exposure-by-outcome distribution design.

    The host executes this declaration but never decides which column is the
    exposure, which is the outcome, which outcome value counts as the event,
    whose rows form each denominator, or how the interval is built. Those are
    scientific choices; an executor that infers them from column names, from
    input ordering, or from prose has taken a decision that belongs to the
    Planner.

    Three fields carry most of the scientific weight:

    ``outcome_levels`` closes the outcome. Without it an outcome value the
    study never declared -- a ``2`` in a column believed to be 0/1, or a
    ``"yes"`` in a column declared numerically -- is observed, matches no
    event, and is therefore counted as a *non-event*. That silently deflates
    every rate in the table and nothing downstream can detect it. With the set
    closed, an undeclared observed value stops the step instead.

    ``denominator_policy`` and ``missing_outcome_policy`` together decide what
    an unobserved outcome means. Treating missingness as "the event did not
    happen" is legitimate only when absence really is structural (no death
    record because the patient lived), and that is a claim about the data
    source, not a default an executor may take.

    ``level_match_policy`` decides whether a declared number may match the same
    number stored as text. Prepared exports differ, so this is real, but it is
    declared rather than assumed -- and no policy ever lets a boolean answer a
    numeric level.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.exposure_outcome_distribution/2"] = (
        "easyicu.exposure_outcome_distribution/2"
    )
    exposure: str
    exposure_levels: List[Any] = Field(min_length=2)
    outcome: str
    outcome_levels: List[Any] = Field(
        min_length=2,
        description=(
            "The closed set of observed outcome values the study recognises. "
            "Any other non-missing value stops the step rather than being "
            "counted as a non-event."
        ),
    )
    outcome_positive_value: Any = Field(
        description=(
            "The exact observed value that counts as the event. Declared "
            "because a binary outcome is not always encoded 1/0, and guessing "
            "silently inverts every rate in the table. Must be one of "
            "outcome_levels, by type as well as by value."
        ),
    )
    level_match_policy: Literal["exact_typed", "numeric_string_equivalent"] = Field(
        description=(
            "'exact_typed' matches a declared level only against values of the "
            "same kind. 'numeric_string_equivalent' additionally treats a "
            "number and its exact text spelling as the same value, for exports "
            "that store codes as strings. Neither policy lets a boolean match "
            "a numeric level."
        ),
    )
    denominator_policy: Literal["all_declared_rows", "observed_outcome_rows"]
    missing_exposure_policy: Literal["fail_closed", "exclude_from_denominator"] = Field(
        default="fail_closed",
        description=(
            "What a row with no observed exposure means. 'fail_closed' stops "
            "the step. 'exclude_from_denominator' is complete-case on the "
            "exposure: those rows leave the table and their count travels in "
            "it, so the denominator change is visible rather than inferred. "
            "There is deliberately NO option to pool them into a level -- an "
            "unobserved exposure is not the reference and not any other "
            "category, and encoding it as one reports a stay under a stage "
            "nobody recorded."
        ),
    )
    missing_outcome_policy: Literal[
        "fail_closed",
        "exclude_from_denominator",
        "structural_absence_is_non_event",
    ] = Field(
        description=(
            "What an unobserved outcome means. 'fail_closed' refuses any "
            "missing outcome. 'exclude_from_denominator' is complete-case and "
            "requires denominator_policy='observed_outcome_rows'. "
            "'structural_absence_is_non_event' asserts that absence encodes "
            "'the event did not occur' and requires "
            "denominator_policy='all_declared_rows'."
        ),
    )
    undeclared_outcome_policy: Literal["fail_closed"] = "fail_closed"
    interval_method: Literal["wilson"] = Field(
        default="wilson",
        description=(
            "Marginal proportion interval for independent rows. If the host "
            "binds a patient-cluster dependence authority, execution "
            "deterministically replaces Wilson with patient-cluster-robust "
            "Wald intervals for exposure prevalence and all absolute risks; "
            "the effective method and covariance travel in the result."
        ),
    )
    repeated_unit_interval_method: Literal["patient_cluster_robust_wald"] = Field(
        default="patient_cluster_robust_wald",
        description=(
            "Typed effective marginal-interval projection used exactly when "
            "the host has bound dependence. Together with dependence and "
            "interval_method this makes the runtime choice part of the "
            "digest-bound specification rather than a silent executor switch. "
            "The patient-cluster sandwich Wald projection is bounded to the "
            "0-100% probability scale; risk-difference intervals are not."
        ),
    )
    risk_difference_contrast: Optional[ExposureOutcomeRiskDifferenceContrast] = Field(
        default=None,
        description=(
            "Optional prespecified unadjusted absolute-risk difference, always "
            "comparison minus reference. Its point estimate is the difference "
            "of the two declared outcome risks. The Wald interval uses HC1 "
            "covariance for independent rows or the host-bound patient-cluster "
            "covariance when dependence is present."
        ),
    )
    dependence: Optional[PlannedDependenceRequirement] = Field(
        default=None,
        description=(
            "Exact host-bound repeated-unit covariance authority for every "
            "marginal proportion and, when declared, the risk-difference "
            "contrast. Null means independent-row marginal intervals and HC1 "
            "risk-difference covariance. The executor never infers grouping "
            "from free text, column names, or duplicated-looking values."
        ),
    )
    confidence_level: float = Field(
        gt=0.5,
        lt=1.0,
        description=(
            "Planner-owned two-sided confidence level for every interval in "
            "the product. Declared rather than defaulted so the executor never "
            "hard-codes a coverage the study did not choose."
        ),
    )

    @field_validator("exposure_levels")
    @classmethod
    def _closed_exposure_levels(cls, values: List[Any]) -> List[Any]:
        return _closed_levels(
            values, label="exposure_outcome_distribution exposure_levels"
        )

    @field_validator("outcome_levels")
    @classmethod
    def _closed_outcome_levels(cls, values: List[Any]) -> List[Any]:
        return _closed_levels(
            values, label="exposure_outcome_distribution outcome_levels"
        )

    @field_validator("outcome_positive_value")
    @classmethod
    def _closed_positive_value(cls, value: Any) -> Any:
        return _closed_levels(
            [value], label="exposure_outcome_distribution outcome_positive_value"
        )[0]

    @field_validator("confidence_level")
    @classmethod
    def _finite_confidence_level(cls, value: float) -> float:
        if not math.isfinite(float(value)):
            raise ValueError(
                "exposure_outcome_distribution confidence_level must be finite"
            )
        return float(value)

    @model_validator(mode="after")
    def _closed_design(self) -> "ExposureOutcomeDistributionSpec":
        self.exposure = str(self.exposure or "").strip()
        self.outcome = str(self.outcome or "").strip()
        if not self.exposure:
            raise ValueError("exposure_outcome_distribution exposure must be non-empty")
        if not self.outcome:
            raise ValueError("exposure_outcome_distribution outcome must be non-empty")
        if self.exposure == self.outcome:
            raise ValueError(
                "exposure_outcome_distribution exposure and outcome must differ"
            )
        declared = {typed_level_key(value) for value in self.outcome_levels}
        if typed_level_key(self.outcome_positive_value) not in declared:
            raise ValueError(
                "exposure_outcome_distribution outcome_positive_value must be one "
                "of outcome_levels, matched by type as well as by value: a "
                "positive value outside the closed set would make every "
                "remaining level a non-event by omission"
            )
        if self.risk_difference_contrast is not None:
            exposure_levels = {
                typed_level_key(value) for value in self.exposure_levels
            }
            contrast_levels = {
                typed_level_key(
                    self.risk_difference_contrast.reference_exposure_level
                ),
                typed_level_key(
                    self.risk_difference_contrast.comparison_exposure_level
                ),
            }
            if not contrast_levels.issubset(exposure_levels):
                raise ValueError(
                    "risk-difference reference and comparison levels must both "
                    "belong to exposure_levels, matched by type and value"
                )
        if (
            self.missing_outcome_policy == "exclude_from_denominator"
            and self.denominator_policy != "observed_outcome_rows"
        ):
            raise ValueError(
                "exposure_outcome_distribution missing_outcome_policy="
                "'exclude_from_denominator' is complete-case analysis and "
                "requires denominator_policy='observed_outcome_rows'"
            )
        if (
            self.missing_outcome_policy == "structural_absence_is_non_event"
            and self.denominator_policy != "all_declared_rows"
        ):
            raise ValueError(
                "exposure_outcome_distribution missing_outcome_policy="
                "'structural_absence_is_non_event' keeps unobserved rows in the "
                "denominator and requires denominator_policy='all_declared_rows'"
            )
        return self


__all__ = [
    "ExposureOutcomeDistributionSpec",
    "ExposureOutcomeRiskDifferenceContrast",
    "typed_level_key",
]
