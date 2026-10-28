"""Read-only workflow metadata for the three explicit longitudinal families.

These placeholder coordinates let a workflow catalog draw the existing family
graphs without importing test fixtures, reading data or selecting a study. Times,
columns and target-trial confirmation fields only satisfy the builders' schemas;
they describe no researcher approval, source binding or patient observations.
Request digests identify catalog metadata, never an approved execution plan.
"""

from __future__ import annotations

from typing import Any

from ...authority.target_trial_runtime import TargetTrialRuntimeAuthority
from ...authority.time_varying_runtime import TimeVaryingRuntimeAuthority
from ...canonical_json import canonical_json, canonical_sha256
from ...contracts.target_trial_design import target_trial_host_policy_sha256
from .contract import FamilyPlanSpec, FamilySpecError, FamilySpecRequest
from .longitudinal_coordinates import (
    DYNAMIC_LANDMARK_FAMILY_ID,
    LONGITUDINAL_FAMILY_IDS,
    TARGET_TRIAL_FAMILY_ID,
    TIME_VARYING_SURVIVAL_FAMILY_ID,
    DynamicLandmarkCoordinates,
)


def _metadata_digest(family_id: str, coordinate: str) -> str:
    return canonical_sha256({
        "scope": "read_only_workflow_metadata", "family_id": family_id, "coordinate": coordinate,
    })


def _target_trial_authority() -> TargetTrialRuntimeAuthority:
    compile_digest = _metadata_digest(TARGET_TRIAL_FAMILY_ID, "unapproved_compile")
    products = {
        "protocol_product": "table:target_trial_protocol",
        "eligibility_product": "table:target_trial_eligibility_flow",
        "table_one_product": "table:target_trial_table_one",
        "risk_curve_product": "table:target_trial_risk_curves",
        "effect_product": "table:target_trial_effect_estimates",
        "weight_model_product": "table:target_trial_weight_models",
        "weight_product": "table:target_trial_weight_diagnostics",
        "balance_product": "table:target_trial_covariate_balance",
        "positivity_product": "table:target_trial_positivity",
        "adherence_product": "table:target_trial_adherence",
        "bootstrap_product": "table:target_trial_bootstrap_replicates",
        "receipt_product": "log:target_trial_runtime_receipt",
        "figure_product": "figure:target_trial_emulation",
    }
    body = {
        "schema_version": "easyicu.target_trial_runtime_authority/1",
        "authority_kind": "target_trial_suite",
        "authority_origin": "workflow_metadata",
        "protocol_content_sha256": _metadata_digest(TARGET_TRIAL_FAMILY_ID, "protocol"),
        "target_trial_compile_sha256": compile_digest,
        "target_trial_compile_confirmation_lines": 1,
        "host_policy_sha256": target_trial_host_policy_sha256(),
        "confirmation": {
            "confirmed_by": "researcher",
            "approval_event_id": "workflow-metadata:unapproved",
            "confirmed_compile_sha256": compile_digest,
            "n_lines_confirmed": 1,
        },
        "plan_method": "signed_target_trial_suite",
        "plan_intent": "Display the target-trial workflow graph from read-only metadata.",
        "plan_outputs": list(products.values()),
        "development_execution_only_allowed": False,
        "database": "workflow_metadata",
        "analysis_unit_label": "Workflow stays",
        "eligibility_label": "Workflow eligibility",
        "treatment_label": "Workflow treatment",
        "initiate_label": "Workflow initiation",
        "defer_label": "Workflow deferral",
        "outcome_label": "Workflow outcome",
        "unit_id_column": "stay_id",
        "resampling_unit": "patient",
        "patient_group_column": "workflow_patient_id",
        "patient_group_derivation": "identity",
        "patient_group_delimiter": None,
        "treatment_onset_columns": ["workflow_treatment_onset"],
        "treatment_onset_window_hours": [0.0, 12.0],
        "time_zero_hours": 6,
        "grace_period_hours": 6,
        "event_column": "mort_28d",
        "followup_time_column": "followup_days_28d",
        "endpoint_horizon_days": 28,
        "endpoint_time_origin": "icu_admission",
        "death_status_column": "death",
        "death_time_column": "death_time",
        "icu_length_of_stay_column": "los_icu",
        "covariates": [{
            "column": "workflow_baseline", "label": "Workflow baseline",
            "coding": "continuous", "levels": [], "reference_level": None, "unmeasured_state": False,
        }],
        "estimator": "clone_censor_weight",
        "interpretation": "per_protocol_effect_under_emulation_assumptions",
        "evidence_ceiling": "analysis_only",
        **products,
    }
    return TargetTrialRuntimeAuthority.model_validate_json(canonical_json({
        **body, "execution_contract_sha256": canonical_sha256(body),
    }))


def _time_varying_authority() -> TimeVaryingRuntimeAuthority:
    body = {
        "schema_version": "easyicu.time_varying_runtime_authority/1",
        "authority_kind": "time_varying_exposure_association",
        "authority_origin": "workflow_metadata",
        "protocol_content_sha256": _metadata_digest(TIME_VARYING_SURVIVAL_FAMILY_ID, "protocol"),
        "sensitivity_spec_id": "workflow_time_varying",
        "specification": {
            "exposure_concept": "workflow_exposure",
            "exposure_summary": "running_max_of_direct_measurements",
            "exposure_window_hours": 24,
            "followup": "hospital_death_or_discharge_from_icu_admission",
            "missingness_policy": "observed_state_indicator",
            "baseline_columns": ["workflow_baseline"],
            "baseline_categorical_encodings": {},
            "interpretation": "descriptive_time_updated_association_not_causal",
        },
        "exposure_column": "workflow_exposure",
        "outcome_column": "death",
        "identity_column": "patient_stay_id",
        "primary_cohort_selection_mode": "all_input_rows",
        "development_execution_only_allowed": True,
        "plan_method": "time_varying_exposure_model",
        "plan_intent": "Display the time-varying workflow graph from read-only metadata.",
        "plan_outputs": [
            "table:time_varying_cox_estimates", "table:time_varying_input_audit", "log:time_varying_runtime_receipt",
        ],
    }
    return TimeVaryingRuntimeAuthority.model_validate_json(canonical_json({
        **body, "execution_contract_sha256": canonical_sha256(body),
    }))


def canonical_longitudinal_request(family_id: str) -> tuple[FamilySpecRequest, FamilyPlanSpec]:
    """Return fresh, deterministic catalog coordinates; refuse other families."""

    if family_id not in LONGITUDINAL_FAMILY_IDS:
        raise FamilySpecError(
            "family_spec_template_mismatch", "No canonical longitudinal workflow for this family.", path="family_id",
        )
    fields: dict[str, Any] = {
        "family_id": family_id, "research_question": "Read-only workflow metadata.",
        "cohort_name": "workflow_metadata", "cohort_selection_mode": "all_input_rows", "cluster_unit": "patient",
        "primary_exposure": "", "exposure_kind": "none", "exposure_levels": [], "reference_level_index": 0,
        "primary_contrast_level_index": 0, "exposure_is_ordered": False,
        "outcome_levels": ["0", "1"], "event_level_index": 1, "adjustment_selection": "exact",
    }
    if family_id == TARGET_TRIAL_FAMILY_ID:
        trial = _target_trial_authority()
        fields.update(
            analysis_type="causal_inference", sealed_target_trial=trial,
            identity_column=trial.unit_id_column, outcome=trial.event_column,
            variable_roster=list(trial.required_columns), exact_roster=[item.column for item in trial.covariates],
        )
    elif family_id == TIME_VARYING_SURVIVAL_FAMILY_ID:
        time_varying = _time_varying_authority()
        fields.update(
            analysis_type="association_study", sealed_time_varying=time_varying,
            identity_column=time_varying.identity_column, outcome=time_varying.outcome_column,
            primary_exposure=time_varying.exposure_column, exposure_kind="continuous",
            variable_roster=list(time_varying.required_columns), exact_roster=list(time_varying.specification.baseline_columns),
        )
    else:
        dynamic = DynamicLandmarkCoordinates(
            trajectory_bound=True, trajectory_time_unit="hours", trajectory_time_origin="icu_admission",
            patient_column="workflow_patient_id", event_time_column="workflow_event_hours",
            followup_end_column="workflow_followup_hours", feature_concepts=["workflow_feature"],
            landmark_hours=[6, 12], lookback_hours=6, horizon_hours=[24], aggregations=["last", "mean"],
        )
        fields.update(
            family_id=DYNAMIC_LANDMARK_FAMILY_ID, analysis_type="dynamic_prediction", dynamic_landmark=dynamic,
            identity_column="stay_id", outcome="workflow_outcome",
            variable_roster=["stay_id", "workflow_outcome", *dynamic.source_columns],
        )
    fields["required_reader_label_keys"] = list(fields["variable_roster"])
    fields["level_label_keys"] = [f"{fields['outcome']}=0", f"{fields['outcome']}=1"]
    request = FamilySpecRequest(**fields)
    spec = FamilyPlanSpec(
        family_id=request.family_id, request_sha256=request.request_sha256,
        reader_display_labels={
            key: "Workflow " + key.replace("_", " ").replace("=", " state ")
            for key in [*request.required_reader_label_keys, *request.level_label_keys]
        },
        roster_decision_note="Read-only workflow metadata with placeholder columns and no study-specific choices.",
    )
    return request, spec


__all__ = ["canonical_longitudinal_request"]
