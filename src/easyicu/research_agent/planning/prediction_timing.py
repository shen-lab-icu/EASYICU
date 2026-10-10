"""Whether a static prediction predicts from what is known at its prediction time.

Owner
-----
This module owns one question about a plan's static prediction primary (the
step the static prediction owner claims, ``prediction.discrimination_calibration``):
when it predicts, whether its cohort keeps only the stays still at risk then,
and which of its predictors the host cannot prove observed by then.  The
scientific review turns the answer into its finding.  Only the prediction
family template states a prediction time, its risk set and the predictors
observed by it, so an outline the Progressive Planner composes stops at a
static prediction primary instead of compiling one without them
(``outline_action_rules.static_prediction_outline_stop``).

- The prediction time is the end of the host-bound outer feature window
  (``adjustment_authority.host_outer_feature_window_end_hours``), the time the
  prediction template predicts at.  Without one the model predicts at ICU
  admission, when only owner-declared baseline demographics are known.
- A predictor is proven observed by the prediction time only as
  ``adjustment_authority.host_window_bound_roles`` proves it, over the roles
  a fitted feature may hold (``WINDOW_DESCRIPTION_ROLES``): the reading the
  prediction template offers its features by.  No other rule is written here.
- The risk set is the cohort's inclusion keeping only the stays still in the
  ICU after the prediction time, read as the time-zero rule reads an ICU
  length-of-stay predicate (``cohort_eligibility.icu_stay_kept_after_hours``).
  The template's further exclusion of a death recorded before the prediction
  time depends on the export's death-time record and is not re-checked here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..contracts.prediction_execution import (
    static_prediction_execution_verdict,
    static_prediction_features,
    static_prediction_model_columns,
)
from .adjustment_authority import (
    WINDOW_DESCRIPTION_ROLES,
    host_outer_feature_window_end_hours,
    host_window_bound_roles,
)
from .cohort_eligibility import icu_stay_kept_after_hours
from .dependence_authority import context_patient_group_authority

_PROVEN = frozenset({"at_or_before_time_zero", "baseline_static"})


@dataclass(frozen=True)
class PredictionTimingFacts:
    """What one static prediction primary predicts from, against its prediction time."""

    step_id: str
    #: Hours after ICU admission; ``None`` predicts at ICU admission.
    prediction_time_hours: Optional[float]
    #: The latest hour the cohort keeps only stays still in the ICU after.
    stays_kept_after_hours: Optional[float]
    #: Predictors the host cannot prove observed by the prediction time.
    unproven_predictors: tuple[str, ...]

    @property
    def risk_set_kept(self) -> bool:
        """Whether only the stays still in the ICU after the prediction time are analyzed."""

        if self.prediction_time_hours is None:
            return True
        return (
            self.stays_kept_after_hours is not None
            and self.stays_kept_after_hours >= self.prediction_time_hours
        )

    @property
    def proven(self) -> bool:
        return self.risk_set_kept and not self.unproven_predictors


def static_prediction_timing_facts(context: Any, plan: Any) -> tuple[PredictionTimingFacts, ...]:
    """The timing facts of each static prediction primary of ``plan``, in step order."""

    steps = [
        step
        for step in getattr(plan, "steps", None) or ()
        if static_prediction_execution_verdict(step).claimed
    ]
    if not steps:
        return ()
    prediction_time = host_outer_feature_window_end_hours(context)
    proven = {
        name
        for name, role in host_window_bound_roles(
            context,
            reference_hours=prediction_time,
            dynamic_roles=WINDOW_DESCRIPTION_ROLES,
        ).items()
        if role in _PROVEN
    }
    cohort = getattr(plan, "cohort", None)
    kept_after = (
        icu_stay_kept_after_hours(
            context, inclusion=[predicate.to_dict() for predicate in cohort.inclusion]
        )
        if cohort is not None
        else None
    )
    group = context_patient_group_authority(context)
    outcome = str(getattr(context, "target_outcome", "") or "").strip()
    return tuple(
        PredictionTimingFacts(
            step_id=str(step.step_id),
            prediction_time_hours=prediction_time,
            stays_kept_after_hours=kept_after,
            unproven_predictors=tuple(
                name
                for name in static_prediction_features(
                    static_prediction_model_columns(step),
                    outcome=outcome,
                    group_source=group.group_source if group is not None else None,
                )
                if name not in proven
            ),
        )
        for step in steps
    )


__all__ = ["PredictionTimingFacts", "static_prediction_timing_facts"]
