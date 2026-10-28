"""Metadata-only canonical family shapes for the workflow hub.

These coordinates are placeholders, never a submitted plan or execution
authority. The family owners build every node, product and dependency. A
question without a fixed family template keeps its package composition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

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
from easyicu.research_agent.planning.family_spec import (
    DYNAMIC_LANDMARK_FAMILY_ID,
    TARGET_TRIAL_FAMILY_ID,
    TIME_VARYING_SURVIVAL_FAMILY_ID,
    build_dynamic_landmark_skeleton,
    build_target_trial_skeleton,
    build_time_varying_survival_skeleton,
)
from easyicu.research_agent.planning.family_spec.contract import AdjustmentCandidate, SealedTrajectoryCoordinates
from easyicu.research_agent.planning.family_spec.longitudinal_canonical import canonical_longitudinal_request
from easyicu.research_agent.planning.family_spec.request import proposed_survival_suite_coordinates
from easyicu.research_agent.planning.family_spec.trajectory_template import CANDIDATE_OUTPUTS
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences
from easyicu.research_agent.trajectory.plan_contract import STABILITY_EXECUTOR_INPUTS
from easyicu.research_agent.contracts.trajectory_design import canonical_repeated_measure_spec, REPEATED_OUTPUTS


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
# Longitudinal families publish their own read-only catalog coordinates; the
# authorities they embed are refused at runtime as workflow metadata.
LONGITUDINAL_TEMPLATES = {
    "target-trial-emulation": (TARGET_TRIAL_FAMILY_ID, build_target_trial_skeleton),
    "time-varying-exposure-survival-study": (TIME_VARYING_SURVIVAL_FAMILY_ID, build_time_varying_survival_skeleton),
    "dynamic-landmark-prediction": (DYNAMIC_LANDMARK_FAMILY_ID, build_dynamic_landmark_skeleton),
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
    if skill_id == "repeated-measure-trajectory":
        return canonical_repeated_measure_workflow()
    longitudinal = LONGITUDINAL_TEMPLATES.get(skill_id)
    if longitudinal is not None:
        family, build = longitudinal
        request, spec = canonical_longitudinal_request(family)
        draft, optional = build(request, spec), set()
        if skill_id in {"target-trial-emulation", "time-varying-exposure-survival-study"}:
            draft, optional = g_method_workflow_metadata(draft)
        return request, draft, optional
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
    if family == PREDICTION_FAMILY_ID:
        from easyicu.research_agent.planning.prediction_timing import external_validation_workflow_metadata

        draft, external_optional = external_validation_workflow_metadata(draft)
        optional |= external_optional
    return request, draft, optional


def g_method_workflow_metadata(draft):
    """Optional read-only branch; supplying verified history is an explicit prerequisite."""
    from dataclasses import replace

    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.contracts.time_varying_exposure import G_METHOD_ACTION, G_METHOD_OUTPUTS
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressiveOutlineStep, ProgressiveOutputIntent, ProgressiveProductRef,
        ProgressiveSkeletonStep, ProgressiveStepMaterialization,
    )

    intake_id, msm_id, product = "g_method_history", "longitudinal_msm", "cohort:person_period"
    history = ProgressiveSkeletonStep(
        step_id=intake_id, module_id="custom_analysis", planned_analysis_role="auxiliary",
        depends_on=["measurement_audit"], custom_method="reviewed_g_method_history_intake",
        objective="Supply source-verified person-period treatment and pretreatment confounder histories from the selected database; stop if unavailable. Baseline rows and running exposure maxima alone do not qualify.",
        outputs=[ProgressiveOutputIntent(product_id=product, semantic_role="custom")],
    )
    msm = ProgressiveSkeletonStep(
        step_id=msm_id, module_id="custom_analysis", planned_analysis_role="secondary",
        scientific_action_id=G_METHOD_ACTION, custom_method="reviewed_g_method_metadata_only",
        objective="Optionally review MSM treatment/confounder histories, lags, censoring weights and truncation. Estimate with weighted pooled logistic and report interval positivity, weights and ESS; observational assumptions remain unverified.",
        depends_on=[intake_id], product_inputs=[ProgressiveProductRef(producer_step_id=intake_id, product_id=product)],
        outputs=[ProgressiveOutputIntent(product_id=p, semantic_role="report" if p.startswith("report:") else "custom") for p in G_METHOD_OUTPUTS],
    )
    materials, outlines = list(draft.materializations), list(draft.outline.steps)
    for step in (history, msm):
        outline = ProgressiveOutlineStep(step_id=step.step_id, module_id=step.module_id,
            planned_analysis_role=step.planned_analysis_role, objective=step.objective,
            depends_on=step.depends_on, scientific_action_id=step.scientific_action_id,
            variable_names=list(draft.outline.steps[0].variable_names))
        outlines.insert(next(i for i, item in enumerate(outlines) if item.step_id == "report"), outline)
        materials.insert(next(i for i, item in enumerate(materials) if item.step.step_id == "report"),
            ProgressiveStepMaterialization(outline_step_sha256=canonical_sha256(outline.model_dump(mode="json")), foundation=None, step=step))
    for i, item in enumerate(materials):
        if item.step.step_id == "report":
            report = item.step.model_copy(update={
                "depends_on": [*item.step.depends_on, msm_id],
                "product_inputs": [*item.step.product_inputs, *(ProgressiveProductRef(producer_step_id=msm_id, product_id=p) for p in G_METHOD_OUTPUTS)],
            })
            materials[i] = item.model_copy(update={"step": report})
    return replace(draft, outline=draft.outline.model_copy(update={"steps": outlines}), materializations=tuple(materials)), {intake_id, msm_id}

@dataclass(frozen=True)
class RepeatedMeasureWorkflowCoordinates:
    """Read-only hub coordinates; no source or execution authority is issued."""

    family_id: str = "repeated_measure_trajectory"
    survival_suite: None = None
    exposure_kind: str = "none"
    exposure_is_ordered: bool = False
    observation_window_hours: float = 144
    cohort_time_zero_hours: float = 0
    observation_duration_column: None = None


@dataclass(frozen=True)
class RepeatedMeasureWorkflowDraft:
    materializations: tuple


def canonical_repeated_measure_workflow():
    """Project the same spec and products that the reviewed execution owner uses."""
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.planning.progressive_contract import ProgressiveSkeletonStep, ProgressiveStepMaterialization

    spec = canonical_repeated_measure_spec()
    cohort = {"producer_step_id": "cohort_accounting", "product_id": "artifact:analysis_cohort"}
    steps = [
        ProgressiveSkeletonStep(
            step_id="cohort_accounting", module_id="cohort_definition", planned_analysis_role="auxiliary",
            objective="Account for the source population supplying the repeated long panel.", raw_inputs=[spec.identity_column],
        ),
        ProgressiveSkeletonStep(
            step_id="measurement_audit", module_id="measurement_audit", planned_analysis_role="auxiliary",
            objective="Audit measurement density and missingness on the reviewed ICU time grid.",
            depends_on=["cohort_accounting"], raw_inputs=list(spec.required_columns), product_inputs=[cohort],
            outputs=[{"product_id": "table:missingness_profile", "semantic_role": "missingness_profile"}],
        ),
        ProgressiveSkeletonStep(
            step_id="repeated_model", module_id="custom_analysis", planned_analysis_role="primary",
            custom_method="repeated_measures",
            objective="Fit the reviewed growth model; state informative death dropout and its planned sensitivity.",
            depends_on=["cohort_accounting", "measurement_audit"], raw_inputs=list(spec.required_columns),
            product_inputs=[cohort], scientific_action_id=spec.action_id, repeated_measure_spec=spec,
            outputs=[{"product_id": product, "semantic_role": "custom"} for product in REPEATED_OUTPUTS],
        ),
        ProgressiveSkeletonStep(
            step_id="report", module_id="report", planned_analysis_role="auxiliary",
            objective="Report aggregate effects, variance/ICC or robust GEE diagnostics and the informative-death-dropout limitation.",
            depends_on=["repeated_model"],
            product_inputs=[{"producer_step_id": "repeated_model", "product_id": product} for product in REPEATED_OUTPUTS],
            outputs=[{"product_id": "report:report", "semantic_role": "report"}],
        ),
    ]
    materials = tuple(ProgressiveStepMaterialization(
        outline_step_sha256=canonical_sha256(step.model_dump(mode="json")), foundation=None, step=step,
    ) for step in steps)
    return RepeatedMeasureWorkflowCoordinates(), RepeatedMeasureWorkflowDraft(materials), set()
