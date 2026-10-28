"""Metadata-only canonical family shapes for the workflow hub.

These coordinates are placeholders, never a submitted plan or execution
authority. The family owners build every node, product and dependency. A
question without a fixed family template keeps its package composition.
"""

from __future__ import annotations

import json

from easyicu.research_agent.planning.family_spec import (
    BASELINE_ASSOCIATION_FAMILY_ID,
    LANDMARK_CATEGORICAL_FAMILY_ID,
    LANDMARK_SURVIVAL_FAMILY_ID,
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    PHENOTYPING_FAMILY_ID,
    PREDICTION_FAMILY_ID,
    FamilyPlanSpec,
    FamilySpecRequest,
    build_baseline_association_skeleton,
    build_landmark_categorical_skeleton,
    build_phenotyping_skeleton,
    build_prediction_skeleton,
    build_landmark_survival_skeleton,
    build_fixed_window_trajectory_skeleton,
)
from easyicu.research_agent.planning.family_spec.contract import AdjustmentCandidate, SealedTrajectoryCoordinates
from easyicu.research_agent.planning.family_spec.request import proposed_survival_suite_coordinates
from easyicu.research_agent.planning.family_spec.trajectory_template import CANDIDATE_OUTPUTS
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences
from easyicu.research_agent.trajectory.plan_contract import STABILITY_EXECUTOR_INPUTS


# Identity bindings only. Scientific content stays in the named owners.
TEMPLATES = {
    "adjusted-exposure-outcome-study": (BASELINE_ASSOCIATION_FAMILY_ID, build_baseline_association_skeleton),
    "ordinal-dose-response-study": (LANDMARK_CATEGORICAL_FAMILY_ID, build_landmark_categorical_skeleton),
    "fixed-landmark-association-study": (LANDMARK_CATEGORICAL_FAMILY_ID, build_landmark_categorical_skeleton),
    "clinical-risk-prediction": (PREDICTION_FAMILY_ID, build_prediction_skeleton),
    "cross-sectional-phenotyping": (PHENOTYPING_FAMILY_ID, build_phenotyping_skeleton),
    "survival-time-to-event": (LANDMARK_SURVIVAL_FAMILY_ID, build_landmark_survival_skeleton),
    "trajectory-phenotyping": (FIXED_WINDOW_TRAJECTORY_FAMILY_ID, build_fixed_window_trajectory_skeleton),
}


def _survival_coordinates():
    """Ask the proposal owner for its canonical binary-exposure suite."""
    binary = {"is_binary": True, "n_unique": 2, "levels": [0, 1]}
    context = ResearchContext(
        research_question="Canonical workflow shape",
        cohort=CohortDescriptor(cohort_name="workflow_metadata", database="synthetic", n_stays=0,
                                id_columns=["stay_id"]),
        primary_exposure="exposure", target_outcome="mort_28d",
        variables=[ConceptDescriptor(name="exposure", role="other", dtype="bool", observed_domain=binary),
                   ConceptDescriptor(name="mort_28d", role="outcome", dtype="bool", observed_domain=binary),
                   ConceptDescriptor(name="followup_days_28d", role="outcome", dtype="float64")],
        user_preferences=UserPreferences(data_constraints=json.dumps({
            "materialization_window": {"anchor": "icu_admission", "hours": 24,
                                       "role": "outer_observation_window"},
        })),
    )
    result = proposed_survival_suite_coordinates(context)
    if result is None:
        raise ValueError("workflow_survival_shape_unavailable")
    return result


def canonical_workflow_shape(skill_id: str):
    """Return an owner-built canonical draft and its typed role coordinates."""
    binding = TEMPLATES.get(skill_id)
    if binding is None:
        return None
    family, build = binding
    features = family in {PHENOTYPING_FAMILY_ID, PREDICTION_FAMILY_ID, FIXED_WINDOW_TRAJECTORY_FAMILY_ID}
    landmark = family == LANDMARK_CATEGORICAL_FAMILY_ID
    ordered = skill_id == "ordinal-dose-response-study"
    candidate = AdjustmentCandidate(
        name="feature", semantic_role="baseline", host_temporal_role="baseline_static",
        allowed_codings=["continuous"], selectable=True, boundary="Canonical metadata shape only.",
    )
    candidates = [candidate, candidate.model_copy(update={"name": "feature_other"})]
    coordinates = dict(
        family_id=family,
        analysis_type=("prediction_model" if family == PREDICTION_FAMILY_ID else
                       "trajectory_clustering" if family in {PHENOTYPING_FAMILY_ID, FIXED_WINDOW_TRAJECTORY_FAMILY_ID}
                       else "survival" if family == LANDMARK_SURVIVAL_FAMILY_ID else "association_study"),
        research_question="Canonical workflow shape", cohort_name="workflow_metadata",
        cohort_selection_mode="all_input_rows", identity_column="stay_id",
        primary_exposure="" if features else "exposure", exposure_kind="none" if features else "categorical",
        exposure_levels=[] if features else ["0", "1", "2"] if ordered else ["0", "1"],
        reference_level_index=0, primary_contrast_level_index=0 if features else 1,
        exposure_is_ordered=ordered, outcome="outcome", outcome_levels=["0", "1"], event_level_index=1,
        adjustment_selection="planner_selectable", adjustment_candidates=[candidate],
        feature_candidates=candidates if features else [],
        measurement_audit_columns=[item.name for item in candidates],
        observation_window_hours=24,
        variable_roster=["stay_id", "exposure", "outcome", *(item.name for item in candidates)],
    )
    if landmark:
        coordinates.update(landmark_hours=24, landmark_spec_id="workflow_landmark",
                           event_time_column="event_time", observation_duration_column="followup",
                           observation_duration_unit="hours")
        coordinates["variable_roster"] += ["event_time", "followup"]
    if family == LANDMARK_SURVIVAL_FAMILY_ID:
        suite = _survival_coordinates()
        coordinates.update(proposed_suite=suite, outcome=suite.event_column)
        coordinates["variable_roster"] += list(suite.source_columns)
        coordinates["variable_roster"] = list(dict.fromkeys(coordinates["variable_roster"]))
    if family == FIXED_WINDOW_TRAJECTORY_FAMILY_ID:
        coordinates["sealed_trajectory"] = SealedTrajectoryCoordinates(
            representation_owner="signed_workflow_metadata", candidate_owner="workflow_metadata",
            coordinate_concepts=[candidate.name], window_hours=(0, 24), grid_width_hours=6,
            candidate_cluster_counts=[2, 3],
            representation_outputs=sorted(STABILITY_EXECUTOR_INPUTS - set(CANDIDATE_OUTPUTS)),
        )
    request = FamilySpecRequest(**coordinates)
    spec = FamilyPlanSpec(
        family_id=family, request_sha256=request.request_sha256,
        baseline_variables=[candidate.name], feature_variables=[item.name for item in candidates] if features else [],
        adjustment_set=[] if features else [dict(name=candidate.name, coding="continuous",
            clinical_rationale="Placeholder covariate for the canonical metadata shape.")],
        reader_display_labels={name: name.replace("_", " ").title() for name in request.variable_roster},
        roster_decision_note="Canonical shape for read-only discovery.",
    )
    draft = build(request, spec)
    minimal_request = request.model_copy(update={"outcome": "", "outcome_levels": [], "event_level_index": 0}) if family in {
        PHENOTYPING_FAMILY_ID, FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    } else request
    minimal = build(minimal_request, spec.model_copy(update={
        "request_sha256": minimal_request.request_sha256, "baseline_variables": [],
    }))
    required = {item.step.step_id for item in minimal.materializations}
    optional = {item.step.step_id for item in draft.materializations} - required
    return request, draft, optional
