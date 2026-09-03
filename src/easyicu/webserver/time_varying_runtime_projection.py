"""Web adapter for the declared time-updated, analysis-only execution owner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.time_varying_exposure import (
    TimeVaryingExposureSpecification,
)
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)

from .scientific_runtime_projection import (
    WebScientificRuntimeProjection,
    WebScientificRuntimeProjectionError,
)
from .study_contexts import primary_cohort_selection_mode


def time_varying_specification(
    specs: Sequence[PrespecifiedSensitivitySpec],
) -> tuple[str, TimeVaryingExposureSpecification] | None:
    matching = [spec for spec in specs if spec.strategy == "time_varying"]
    if not matching:
        return None
    if len(matching) != 1 or matching[0].time_varying_execution is None:
        raise WebScientificRuntimeProjectionError(
            "web_time_varying_specification_incomplete",
            "The agent must close one explicit time-varying exposure, missingness and baseline-coding specification in the plan.",
            details={"owner": "plan_agent", "human_question_required": False},
        )
    return matching[0].spec_id, matching[0].time_varying_execution


def compile_time_varying_runtime_projection(
    *,
    study: Mapping[str, Any],
    sensitivity_specs: Sequence[PrespecifiedSensitivitySpec],
    primary_exposure: str | None,
    primary_exposure_source: str | None,
    target_outcome: str | None,
    declared_covariates: Sequence[str],
    covariate_operationalizations: Mapping[str, str],
    target_is_event_status: bool,
    universe_path: Path,
    scientific_configuration_sha256: str,
    dependence: Any = None,
) -> WebScientificRuntimeProjection | None:
    selected = time_varying_specification(sensitivity_specs)
    if selected is None:
        return None
    spec_id, specification = selected
    selection_mode = primary_cohort_selection_mode(study)
    if selection_mode != "all_input_rows":
        raise WebScientificRuntimeProjectionError(
            "web_time_varying_filtered_cohort_not_bound",
            "The time-varying owner requires a source-bound all-input-row cohort; additional cohort predicates need their own verified projection.",
            details={"owner": "source_runtime", "selection_mode": selection_mode},
        )
    resolved_baseline = tuple(
        covariate_operationalizations.get(column) or column
        for column in declared_covariates
    )
    if (
        not primary_exposure
        or primary_exposure_source != specification.exposure_concept
        or target_outcome != "death"
        or not target_is_event_status
        or study.get("covariate_selection") != "exact"
        or resolved_baseline != specification.baseline_columns
        or dependence is None
        or dependence.group_source != "patient_stay_id"
    ):
        raise WebScientificRuntimeProjectionError(
            "web_time_varying_study_binding_mismatch",
            "The time-varying contract differs from the study exposure, outcome, baseline roster or patient grouping.",
            details={"owner": "plan_agent", "human_question_required": False},
        )
    authority = build_current_case_scientific_runtime_authority(
        {
            "schema_version": "easyicu.time_varying_runtime_authority/1",
            "authority_kind": "time_varying_exposure_association",
            "protocol_content_sha256": scientific_configuration_sha256,
            "specification": specification.model_dump(mode="json"),
            "sensitivity_spec_id": spec_id,
            "exposure_column": primary_exposure,
            "outcome_column": "death",
            "identity_column": "patient_stay_id",
            "primary_cohort_selection_mode": selection_mode,
            "development_execution_only_allowed": True,
            "plan_method": "time_varying_exposure_model",
            "plan_intent": (
                f"Use direct {specification.exposure_concept} measurements as a running maximum during ICU hours 0–24, "
                "retain unmeasured states and early deaths, freeze the last observed maximum after hour 24, "
                "and fit hospital death/discharge counting-process Cox with patient-clustered uncertainty. "
                f"Baseline adjustment: {', '.join(specification.baseline_columns)}. "
                "Development analysis only; no causal or publication claim."
            ),
            "plan_outputs": [
                "table:time_varying_cox_estimates",
                "table:time_varying_input_audit",
                "log:time_varying_runtime_receipt",
            ],
        }
    )
    payload = authority.model_dump(mode="json")
    return WebScientificRuntimeProjection(
        authority=payload,
        projection_sha256=canonical_sha256(payload),
        analysis_only_execution=True,
    )


def materialize_web_time_varying_input(
    acquisition: Any,
    *,
    specs: Sequence[PrespecifiedSensitivitySpec],
    export_path: Path,
    database: str,
    patient_grouping: Any,
    exposure_column: str,
):
    from easyicu.research_agent.acquisition.time_varying_materialization import (
        materialize_time_varying_acquisition,
    )
    from .raw_source_authority import resolve_raw_mimic_iv_source_binding

    selected = time_varying_specification(specs)
    if selected is None:
        return acquisition
    binding = resolve_raw_mimic_iv_source_binding(
        export_path=export_path, database=database
    )
    if binding is None or patient_grouping is None:
        raise WebScientificRuntimeProjectionError(
            "web_time_varying_source_authority_required",
            "Hospital follow-up and patient grouping require source-bound authority.",
            details={"owner": "source_runtime"},
        )
    return materialize_time_varying_acquisition(
        acquisition,
        specification=selected[1],
        hospital_followup=binding.materialize_hospital_mortality_followup(),
        raw_source_receipt=binding.public_receipt(),
        patient_grouping=patient_grouping,
        exposure_column=exposure_column,
    )


def materialize_web_hospital_followup(
    acquisition: Any,
    *,
    specs: Sequence[PrespecifiedSensitivitySpec],
    export_path: Path,
    database: str,
):
    """Attach a landmark's exact hospital endpoint through the same raw owner."""
    if not any(
        spec.strategy == "landmark"
        and spec.observation_duration_variable == "hospital_followup_time_hours"
        for spec in specs
    ):
        return acquisition
    from easyicu.research_agent.acquisition.hospital_followup_materialization import (
        materialize_hospital_followup_acquisition,
    )
    from .raw_source_authority import resolve_raw_mimic_iv_source_binding

    binding = resolve_raw_mimic_iv_source_binding(
        export_path=export_path, database=database
    )
    if binding is None:
        raise WebScientificRuntimeProjectionError(
            "web_hospital_followup_authority_required",
            "The landmark hospital endpoint requires verified raw follow-up.",
            details={"owner": "source_runtime"},
        )
    return materialize_hospital_followup_acquisition(
        acquisition,
        followup=binding.materialize_hospital_mortality_followup(),
        raw_source_receipt=binding.public_receipt(),
    )


__all__ = [
    "time_varying_specification",
    "compile_time_varying_runtime_projection",
    "materialize_web_time_varying_input",
]
