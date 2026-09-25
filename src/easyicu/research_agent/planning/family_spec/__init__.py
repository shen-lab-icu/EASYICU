"""Family-spec planning: the Planner fills a small typed spec, the host compiles.

Owner
-----
This package owns the ``family_spec_v1`` planner strategy for method families
that have a host template (today: the fixed-landmark categorical and
continuous/spline association families, the descriptive exposure–outcome
family, the cross-sectional phenotyping family, the static binary
prediction family, the sealed fixed-landmark survival suite, the sealed
fixed-window trajectory suite, and the sealed fail-closed source-feasibility
decision).  The Planner's only output is a
:class:`~.contract.FamilyPlanSpec` — the scientific decisions a statistician
makes for the family (adjustment set with rationales, reader labels, how each
screened comparator is applied).  Every executable coordinate (steps, products,
owners, denominators, sensitivity wiring) is projected deterministically from
the sealed :class:`~easyicu.research_agent.schema.ResearchContext` by the
family template and then compiled and validated by the unchanged Progressive
v2 validators and compiler.  No plan gate is weakened: a template that cannot
compile fails closed with the compiler's own reason code.

Boundaries
----------
* The template never selects covariates, exposures, outcomes, or sensitivity
  axes: covariates come from the spec (or the exact user roster), everything
  else from typed StudyContext facts.
* Timing authority for Planner-selected covariates is the host's
  :func:`~easyicu.research_agent.planning.adjustment_authority.host_proven_temporal_roles`;
  a rationale never widens it.
* This package must not import agents, orchestration, pipeline, or webserver.
"""

from __future__ import annotations

from .contract import (
    DESCRIPTIVE_FAMILY_ID,
    FAMILY_SPEC_SCHEMA_VERSION,
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    LANDMARK_CATEGORICAL_FAMILY_ID,
    LANDMARK_SPLINE_FAMILY_ID,
    LANDMARK_SURVIVAL_FAMILY_ID,
    PHENOTYPING_FAMILY_ID,
    PREDICTION_FAMILY_ID,
    SOURCE_FEASIBILITY_FAMILY_ID,
    AcceptedFeatureGroup,
    AdjustmentCandidate,
    FamilyPlanSpec,
    FamilySpecError,
    FamilySpecRequest,
    SealedFeasibilityCoordinates,
    SealedSuiteCoordinates,
    SealedTrajectoryCoordinates,
    SpecCovariateDecision,
    validate_family_plan_spec,
)
from .request import (
    build_family_spec_request,
    exposure_companion_columns,
    family_template_id_for_context,
    sealed_feasibility_coordinates,
    sealed_survival_suite_coordinates,
    sealed_trajectory_suite_coordinates,
)
from .landmark_categorical_template import (
    FamilySkeletonDraft,
    build_landmark_association_skeleton,
    build_landmark_categorical_skeleton,
    keeps_unmeasured_covariate_rows,
)
from .descriptive_template import build_descriptive_skeleton
from .feasibility_template import build_source_feasibility_skeleton
from .phenotyping_template import build_phenotyping_skeleton
from .prediction_template import build_prediction_skeleton
from .survival_template import build_landmark_survival_skeleton
from .trajectory_template import build_fixed_window_trajectory_skeleton

__all__ = [
    "DESCRIPTIVE_FAMILY_ID",
    "FAMILY_SPEC_SCHEMA_VERSION",
    "FIXED_WINDOW_TRAJECTORY_FAMILY_ID",
    "LANDMARK_CATEGORICAL_FAMILY_ID",
    "LANDMARK_SPLINE_FAMILY_ID",
    "LANDMARK_SURVIVAL_FAMILY_ID",
    "PHENOTYPING_FAMILY_ID",
    "PREDICTION_FAMILY_ID",
    "SOURCE_FEASIBILITY_FAMILY_ID",
    "AcceptedFeatureGroup",
    "AdjustmentCandidate",
    "FamilyPlanSpec",
    "FamilySkeletonDraft",
    "FamilySpecError",
    "FamilySpecRequest",
    "SealedFeasibilityCoordinates",
    "SealedSuiteCoordinates",
    "SealedTrajectoryCoordinates",
    "SpecCovariateDecision",
    "build_descriptive_skeleton",
    "build_fixed_window_trajectory_skeleton",
    "build_family_spec_request",
    "build_landmark_association_skeleton",
    "build_landmark_categorical_skeleton",
    "build_landmark_survival_skeleton",
    "build_phenotyping_skeleton",
    "build_prediction_skeleton",
    "build_source_feasibility_skeleton",
    "exposure_companion_columns",
    "family_template_id_for_context",
    "keeps_unmeasured_covariate_rows",
    "sealed_feasibility_coordinates",
    "sealed_survival_suite_coordinates",
    "sealed_trajectory_suite_coordinates",
    "validate_family_plan_spec",
]
