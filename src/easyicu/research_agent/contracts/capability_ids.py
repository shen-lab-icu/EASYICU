"""Stable scientific-capability vocabulary, independent of planner runtime."""

from __future__ import annotations

from typing import Final

from .time_varying_exposure import TIME_VARYING_EXPOSURE_CAPABILITY

LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID: Final = "association_landmark_spline_v1"
LANDMARK_SPLINE_ANALYSIS_KIND: Final = "signed_landmark_spline_association"
PHENOTYPING_CLUSTER_CAPABILITY_ID: Final = "phenotyping_cluster_v1"
PHENOTYPING_ANALYSIS_KIND: Final = "cross_sectional_phenotyping"
SOURCE_FEASIBILITY_NON_USE_CAPABILITY_ID: Final = (
    "source_feasibility_non_use_v1"
)
SOURCE_FEASIBILITY_ANALYSIS_KIND: Final = "signed_source_feasibility_fail_closed"
SIGNED_TRAJECTORY_PHENOTYPING_CAPABILITY_ID: Final = (
    "trajectory_signed_phenotyping_v1"
)

CAPABILITY_FAMILIES: Final[dict[str, str]] = {
    "survival_time_to_event_v1": "time_to_event",
    "causal_target_trial_v1": "causal_emulation",
    SOURCE_FEASIBILITY_NON_USE_CAPABILITY_ID: "causal_emulation",
    "association_ordinal_trend_v1": "association",
    "association_adjusted_v1": "association",
    LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID: "association",
    TIME_VARYING_EXPOSURE_CAPABILITY: "association",
    "association_freeform_v1": "association",
    "prediction_risk_model_v1": "prediction",
    "dynamic_prediction_landmark_v1": "prediction",
    PHENOTYPING_CLUSTER_CAPABILITY_ID: "phenotyping",
    SIGNED_TRAJECTORY_PHENOTYPING_CAPABILITY_ID: "phenotyping",
    "descriptive_measurement_v1": "descriptive",
    "descriptive_exposure_outcome_distribution_v1": "descriptive",
}


def capability_family(capability_id: str | None) -> str | None:
    """Return the stable family for a known id, else ``None``."""

    return CAPABILITY_FAMILIES.get(str(capability_id or "").strip())


__all__ = [
    "CAPABILITY_FAMILIES",
    "LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID",
    "LANDMARK_SPLINE_ANALYSIS_KIND",
    "PHENOTYPING_CLUSTER_CAPABILITY_ID",
    "PHENOTYPING_ANALYSIS_KIND",
    "SOURCE_FEASIBILITY_ANALYSIS_KIND",
    "SOURCE_FEASIBILITY_NON_USE_CAPABILITY_ID",
    "SIGNED_TRAJECTORY_PHENOTYPING_CAPABILITY_ID",
    "capability_family",
]
