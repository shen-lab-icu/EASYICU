"""Typed, case-neutral authority for prespecified sensitivity analyses.

Study configuration owns *which* robustness questions must be executed.  The
Planner still chooses a valid step layout, but prose in ``analysis_goal`` is
not execution authority.  This dependency-neutral contract is shared by the
Web StudyContext boundary and :class:`~easyicu.research_agent.schema.UserPreferences`.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


SensitivityAxis = Literal[
    "timing",
    "repeated_stays",
    "functional_form",
    "missing_data",
    "cohort",
    "outcome_definition",
]
SensitivityStrategy = Literal[
    "landmark",
    "time_varying",
    "alternate_window",
    "first_stay",
    "non_readmission_restriction",
    "cluster_robust",
    "mixed_effects",
    "restricted_cubic_spline",
    "fractional_polynomial",
    "categorical",
    "complete_case",
    "multiple_imputation",
    "inverse_probability_weighting",
    "alternate_eligibility",
    "alternate_definition",
]

_STRATEGIES_BY_AXIS: dict[str, frozenset[str]] = {
    "timing": frozenset({"landmark", "time_varying", "alternate_window"}),
    "repeated_stays": frozenset(
        {
            "first_stay",
            "non_readmission_restriction",
            "cluster_robust",
            "mixed_effects",
        }
    ),
    "functional_form": frozenset(
        {"restricted_cubic_spline", "fractional_polynomial", "categorical"}
    ),
    "missing_data": frozenset(
        {"complete_case", "multiple_imputation", "inverse_probability_weighting"}
    ),
    "cohort": frozenset({"alternate_eligibility"}),
    "outcome_definition": frozenset({"alternate_definition"}),
}

EXECUTABLE_METHODS_BY_STRATEGY: dict[str, frozenset[str]] = {
    "landmark": frozenset(
        {"signed_landmark_restricted_cubic_spline", "landmark_analysis"}
    ),
    "time_varying": frozenset({"time_varying_exposure_model", "clone_censor_weight"}),
    "alternate_window": frozenset({"alternate_window_analysis"}),
    "first_stay": frozenset({"one_stay_per_patient_association", "first_stay_association"}),
    "non_readmission_restriction": frozenset({"non_readmission_restriction"}),
    "cluster_robust": frozenset({"cluster_robust_association"}),
    "mixed_effects": frozenset({"mixed_effects_association", "mixed_effects_regression"}),
    "restricted_cubic_spline": frozenset(
        {"signed_landmark_restricted_cubic_spline", "restricted_cubic_spline_sensitivity"}
    ),
    "fractional_polynomial": frozenset({"fractional_polynomial_sensitivity"}),
    "categorical": frozenset({"categorical_functional_form_sensitivity"}),
    "complete_case": frozenset({"complete_case_sensitivity"}),
    "multiple_imputation": frozenset({"multiple_imputation_sensitivity"}),
    "inverse_probability_weighting": frozenset(
        {"inverse_probability_weighting_sensitivity"}
    ),
    "alternate_eligibility": frozenset({"alternate_eligibility_sensitivity"}),
    "alternate_definition": frozenset({"alternate_outcome_definition_sensitivity"}),
}


class PrespecifiedSensitivitySpec(BaseModel):
    """One immutable, user-reviewed robustness commitment.

    ``execution_variables`` contains owner-issued source concept identifiers,
    never display labels.  It can therefore enlarge materialization without
    silently enlarging the primary adjustment set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    axis: SensitivityAxis
    strategy: SensitivityStrategy
    execution_variables: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    landmark_hours: float | None = Field(default=None, gt=0, le=24 * 365)
    require_alive_at_landmark: bool = False
    exclude_negative_event_times: bool = False

    @model_validator(mode="after")
    def _closed_strategy(self) -> "PrespecifiedSensitivitySpec":
        if self.strategy not in _STRATEGIES_BY_AXIS[self.axis]:
            raise ValueError(
                f"strategy {self.strategy!r} is not valid for axis {self.axis!r}"
            )
        variables = tuple(str(value or "").strip() for value in self.execution_variables)
        if any(not value for value in variables) or len(variables) != len(set(variables)):
            raise ValueError("execution_variables must be unique non-empty identifiers")
        if any(
            len(value) > 80
            or not value[0].isalnum()
            or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in value)
            for value in variables
        ):
            raise ValueError("execution_variables must be owner-issued identifiers")
        if self.strategy == "landmark":
            if self.landmark_hours is None:
                raise ValueError("landmark sensitivity requires landmark_hours")
        elif self.landmark_hours is not None:
            raise ValueError("landmark_hours is valid only for landmark sensitivity")
        if (self.require_alive_at_landmark or self.exclude_negative_event_times) and (
            self.strategy != "landmark"
        ):
            raise ValueError("landmark eligibility flags require landmark strategy")
        if self.strategy in {
            "non_readmission_restriction",
            "first_stay",
            "restricted_cubic_spline",
            "fractional_polynomial",
            "categorical",
            "complete_case",
            "multiple_imputation",
            "inverse_probability_weighting",
            "alternate_definition",
        } and not variables:
            raise ValueError(f"{self.strategy} sensitivity requires execution_variables")
        object.__setattr__(self, "execution_variables", variables)
        return self


def normalize_prespecified_sensitivities(
    value: Any,
) -> tuple[PrespecifiedSensitivitySpec, ...]:
    """Validate and de-duplicate a bounded sensitivity authority list."""

    if value in (None, (), []):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("sensitivity_specs must be a list")
    if len(value) > 16:
        raise ValueError("sensitivity_specs may contain at most 16 items")
    rows: list[PrespecifiedSensitivitySpec] = []
    seen: set[str] = set()
    for raw in value:
        if isinstance(raw, PrespecifiedSensitivitySpec):
            row = raw
        elif isinstance(raw, Mapping):
            row = PrespecifiedSensitivitySpec.model_validate(dict(raw))
        else:
            raise ValueError("each sensitivity_specs item must be an object")
        if row.spec_id in seen:
            raise ValueError("sensitivity_specs spec_id values must be unique")
        seen.add(row.spec_id)
        rows.append(row)
    return tuple(rows)


__all__ = [
    "EXECUTABLE_METHODS_BY_STRATEGY",
    "PrespecifiedSensitivitySpec",
    "SensitivityAxis",
    "SensitivityStrategy",
    "normalize_prespecified_sensitivities",
]
