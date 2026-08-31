"""Compile one Research Agent launch into an immutable prepared state.

This module owns the order-sensitive launch boundary: scientific authority,
source/package binding, provider credentials, runtime availability, and resume
coordinates are all validated before the JobManager runner can perform side
effects.  The execution bridge supplies its established primitive operations;
this owner supplies the single orchestration seam and returned state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from easyicu.databases import normalize_database_key
from easyicu.extensions import (
    ExtensionActivationSnapshot,
    ExtensionRegistry,
    ExtensionRegistryError,
)
from easyicu.research_agent.publication_skills import (
    publication_skill_flags_from_settings,
)
from easyicu.webserver import (
    capabilities as capability_policy,
    dataio,
    provider_adapter,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

_RUNNER_IMAGE_ENV = "EASYICU_RUNNER_IMAGE"
_DEVELOPMENT_RESUME_JOB_ENV = "EASYICU_DEVELOPMENT_PROGRESSIVE_RESUME_SOURCE_JOB_ID"
_DEVELOPMENT_RESUME_SEQUENCE_ENV = (
    "EASYICU_DEVELOPMENT_PROGRESSIVE_RESUME_CHECKPOINT_SEQUENCE"
)


@dataclass(frozen=True)
class ResearchPipelinePreparationOperations:
    """Primitive owner operations used by the launch preparation boundary.

    Keeping these dependencies explicit preserves existing owner modules and
    test seams while giving their order-sensitive composition exactly one
    owner.  Callers construct this adapter at launch so monkeypatched owner
    operations cannot be bypassed by stale function references.
    """

    clean_text: Callable[[Any, int], str]
    neutral_materialization_scope: Callable[..., Dict[str, Any]]
    target_outcome: Callable[[Mapping[str, Any]], Optional[str]]
    primary_exposure: Callable[[Mapping[str, Any]], Optional[str]]
    primary_exposure_aggregation: Callable[[Mapping[str, Any]], Optional[str]]
    metadata_only_planning_coordinates: Callable[..., Dict[str, Any]]
    validate_primary_concept_selection: Callable[..., None]
    configured_covariates: Callable[[Mapping[str, Any]], tuple[str, ...]]
    configured_covariate_selection: Callable[[Mapping[str, Any]], str]
    configured_sensitivity_specs: Callable[[Mapping[str, Any]], tuple[Any, ...]]
    cohort_window: Callable[[Mapping[str, Any]], tuple[float, float]]
    validate_analysis_design: Callable[[Mapping[str, Any]], Dict[str, str]]
    patient_grouping_for_analysis_design: Callable[..., Any]
    metadata_planning_operationalized_columns: Callable[..., tuple[str, ...]]
    data_foundation_profile: Callable[..., Dict[str, Any]]
    validated_pipeline_credential_source: Callable[..., str]
    development_progressive_resume_binding: Callable[..., tuple[Path, str]]
    development_resume_acquisition_profile: Callable[..., Any]
    development_resume_literature_bundle: Callable[..., Dict[str, Any]]
    require_execution_runtime: Callable[..., None]


@dataclass(frozen=True)
class PreparedResearchPipelineRun:
    """Validated, side-effect-free inputs captured for one pipeline runner."""

    export_path: str
    study: Mapping[str, Any]
    project_root: str
    provider: Mapping[str, Any]
    literature_search_authorized: bool
    question: str
    database: str
    budget_mode: str
    materialization_study: Mapping[str, Any]
    configured_target: Optional[str]
    configured_primary_exposure: Optional[str]
    target: Optional[str]
    primary_exposure: Optional[str]
    covariates: tuple[str, ...]
    covariate_selection: str
    sensitivity_specs: tuple[Any, ...]
    cohort_window: tuple[float, float]
    validated_analysis_design: Mapping[str, str]
    patient_grouping: Any
    metadata_only_planning: bool
    metadata_planning_coordinates: Mapping[str, Any]
    execution_concepts: Mapping[str, Any]
    planning_exposure_source: Optional[str]
    metadata_operationalized_columns: tuple[str, ...]
    prepared_package_binding: Optional[Mapping[str, Any]]
    foundation_profile: Mapping[str, Any]
    credential_source: str
    development_resume_binding: Optional[tuple[Path, str]]
    development_resume_acquisition: Any
    development_resume_literature: Optional[Mapping[str, Any]]
    publication_skill_flags: Mapping[str, Any]
    user_extension_activation: Any
    provider_environment: Mapping[str, str]
    plan_revision_source_run_id: str
    execution_resume_source_run_id: str
    runner_image: str


def prepare_research_pipeline_run(
    *,
    export_path: str,
    study_context: Mapping[str, Any],
    project_root: Optional[str],
    provider: Mapping[str, Any],
    provider_environment: Optional[Mapping[str, str]],
    credential_source: str,
    literature_search_authorized: bool,
    plan_revision_source_run_id: str,
    execution_resume_source_run_id: str,
    development_resume_source_job_id: str,
    budget_mode: str,
    runner_image: Optional[str],
    operations: ResearchPipelinePreparationOperations,
) -> PreparedResearchPipelineRun:
    """Validate and compile every launch-time authority before side effects."""

    study = dict(study_context)
    question = operations.clean_text(study.get("question"), 1_200)
    if not question:
        raise ResearchPipelineRunError(
            "research_pipeline_question_required",
            "A scientific question is required before starting the pipeline.",
        )
    source = study.get("data_source")
    if not isinstance(source, Mapping):
        raise ResearchPipelineRunError(
            "research_pipeline_database_required",
            "A typed ICU database is required before pipeline launch.",
        )
    database_raw = operations.clean_text(source.get("database"), 64)
    if not database_raw:
        raise ResearchPipelineRunError(
            "research_pipeline_database_required",
            "A typed ICU database is required before pipeline launch.",
        )
    try:
        database = normalize_database_key(database_raw)
    except KeyError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_database_unknown",
            "The configured ICU database is not supported.",
            details={"database": database_raw},
        ) from exc

    selected_budget_mode = str(budget_mode or "").strip().lower()
    materialization_study = operations.neutral_materialization_scope(
        study,
        export_path=export_path,
    )
    configured_target = operations.target_outcome(study)
    configured_primary_exposure = operations.primary_exposure(study)
    planning_coordinates = operations.metadata_only_planning_coordinates(
        question=question,
        database=database,
    )
    target = configured_target or planning_coordinates.get("target_outcome")
    primary_exposure = configured_primary_exposure or planning_coordinates.get(
        "primary_exposure"
    )
    operations.validate_primary_concept_selection(
        study,
        configured_primary_exposure,
    )
    covariates = operations.configured_covariates(study)
    covariate_selection = operations.configured_covariate_selection(study)
    sensitivity_specs = operations.configured_sensitivity_specs(study)
    window = operations.cohort_window(materialization_study)
    validated_analysis_design = operations.validate_analysis_design(study)
    patient_grouping = (
        operations.patient_grouping_for_analysis_design(study)
        if validated_analysis_design.get("variance_estimator") == "cluster_robust"
        else None
    )

    metadata_only_planning = selected_budget_mode == "planner_canary"
    metadata_planning_coordinates: Dict[str, Any] = dict(planning_coordinates)
    execution_concepts_raw = study.get("execution_concepts")
    execution_concepts = (
        dict(execution_concepts_raw)
        if isinstance(execution_concepts_raw, Mapping)
        else {}
    )
    planning_exposure_source = (
        operations.clean_text(execution_concepts.get("primary_exposure"), 160)
        or metadata_planning_coordinates.get("primary_exposure")
        or None
    )
    planning_exposure_aggregation = operations.primary_exposure_aggregation(study)
    metadata_operationalized_columns = (
        operations.metadata_planning_operationalized_columns(
            primary_exposure_source=planning_exposure_source,
            primary_exposure_aggregation=planning_exposure_aggregation,
            covariates=covariates,
            covariate_selection=covariate_selection,
            covariate_operationalizations=(
                study.get("covariate_operationalizations") or {}
            ),
            sensitivity_specs=sensitivity_specs,
        )
        if metadata_only_planning
        else ()
    )

    prepared_package_binding: Optional[Dict[str, Any]] = None
    if metadata_only_planning:
        foundation_profile: Dict[str, Any] = {
            "allowed_modules": (),
            "static_concepts": (),
            "outcome_concepts": (),
            "required_feature_concepts": (),
            "require_outcome": False,
            "primary_exposure_source_concept": None,
        }
    else:
        foundation_profile = operations.data_foundation_profile(
            export_path=export_path,
            study=materialization_study,
            target=configured_target,
            primary_exposure=configured_primary_exposure,
            covariates=covariates,
            sensitivity_specs=sensitivity_specs,
        )
        try:
            package_receipt = dataio.validate_research_pipeline_source(
                export_path,
                database=database,
            )
        except dataio.ExportCohortError as exc:
            raise ResearchPipelineRunError(
                str(exc.detail.get("error") or "research_pipeline_source_invalid"),
                "The Research Agent requires a manifest-backed prepared data package.",
                details={
                    key: value for key, value in exc.detail.items() if key != "error"
                },
            ) from exc
        prepared_package_binding = dict(package_receipt["binding"])

    try:
        provider_adapter.web_research_agent_hard_stop_limits(selected_budget_mode)
    except ValueError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_budget_mode_invalid",
            "Choose an explicit reviewed Research Agent budget mode.",
        ) from exc
    selected_credential_source = operations.validated_pipeline_credential_source(
        credential_source,
        provider=provider,
    )
    if provider_environment is None:
        raise ResearchPipelineRunError(
            (
                "research_pipeline_codex_user_auth_required"
                if selected_credential_source == "codex_user_auth"
                else "research_pipeline_pi_verified_credentials_required"
            ),
            "A verified provider credential is required for this pipeline.",
        )
    if not project_root:
        raise ResearchPipelineRunError(
            "research_pipeline_project_workspace_required",
            "A server-owned Pi project workspace is required for this pipeline.",
        )

    development_resume_binding: tuple[Path, str] | None = None
    selected_resume_source = operations.clean_text(
        development_resume_source_job_id or os.environ.get(_DEVELOPMENT_RESUME_JOB_ENV),
        80,
    )
    if selected_resume_source:
        development_resume_binding = operations.development_progressive_resume_binding(
            project_root=project_root,
            study_id=str(study.get("id") or ""),
            source_job_id=selected_resume_source,
            budget_mode=selected_budget_mode,
            checkpoint_sequence=os.environ.get(_DEVELOPMENT_RESUME_SEQUENCE_ENV),
        )
    development_resume_acquisition = (
        operations.development_resume_acquisition_profile(
            checkpoint_path=development_resume_binding[0],
            database=database,
            cohort_window=window,
            outcome_concepts=foundation_profile["outcome_concepts"],
            static_concepts=foundation_profile["static_concepts"],
            required_feature_concepts=foundation_profile["required_feature_concepts"],
            planning_target_outcome=metadata_planning_coordinates.get("target_outcome"),
            planning_endpoint=metadata_planning_coordinates.get("endpoint"),
            planning_operationalized_columns=metadata_operationalized_columns,
        )
        if development_resume_binding is not None
        else None
    )
    development_resume_literature = (
        operations.development_resume_literature_bundle(
            checkpoint_path=development_resume_binding[0]
        )
        if development_resume_binding is not None
        else None
    )

    capability_settings = capability_policy.capability_settings()
    publication_skill_flags = publication_skill_flags_from_settings(capability_settings)
    try:
        extension_registry = ExtensionRegistry()
        extension_snapshot = extension_registry.snapshot()
        if not bool(capability_settings.get("mcp_tools_enabled", False)):
            extension_snapshot = ExtensionActivationSnapshot.build(
                revision=extension_snapshot.revision,
                skills=extension_snapshot.skills,
                mcp_servers=(),
            )
        user_extension_activation = extension_registry.pipeline_activation(
            extension_snapshot
        )
    except ExtensionRegistryError as exc:
        raise ResearchPipelineRunError(
            exc.code,
            exc.message,
            details=exc.details,
        ) from exc

    selected_runner_image_raw = (
        runner_image if runner_image is not None else os.environ.get(_RUNNER_IMAGE_ENV)
    )
    selected_runner_image = str(selected_runner_image_raw or "").strip()
    if selected_runner_image_raw is not None and (
        not selected_runner_image
        or "\n" in selected_runner_image
        or "\r" in selected_runner_image
        or "\0" in selected_runner_image
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_runner_image_invalid",
            "The server-owned runner image must be one non-empty reference.",
        )
    operations.require_execution_runtime(
        budget_mode=selected_budget_mode,
        runner_image=selected_runner_image,
    )

    return PreparedResearchPipelineRun(
        export_path=export_path,
        study=study,
        project_root=project_root,
        provider=dict(provider),
        literature_search_authorized=bool(literature_search_authorized),
        question=question,
        database=database,
        budget_mode=selected_budget_mode,
        materialization_study=materialization_study,
        configured_target=configured_target,
        configured_primary_exposure=configured_primary_exposure,
        target=target,
        primary_exposure=primary_exposure,
        covariates=tuple(covariates),
        covariate_selection=covariate_selection,
        sensitivity_specs=tuple(sensitivity_specs),
        cohort_window=window,
        validated_analysis_design=validated_analysis_design,
        patient_grouping=patient_grouping,
        metadata_only_planning=metadata_only_planning,
        metadata_planning_coordinates=metadata_planning_coordinates,
        execution_concepts=execution_concepts,
        planning_exposure_source=planning_exposure_source,
        metadata_operationalized_columns=metadata_operationalized_columns,
        prepared_package_binding=prepared_package_binding,
        foundation_profile=foundation_profile,
        credential_source=selected_credential_source,
        development_resume_binding=development_resume_binding,
        development_resume_acquisition=development_resume_acquisition,
        development_resume_literature=development_resume_literature,
        publication_skill_flags=publication_skill_flags,
        user_extension_activation=user_extension_activation,
        provider_environment=dict(provider_environment),
        plan_revision_source_run_id=operations.clean_text(
            plan_revision_source_run_id,
            160,
        ),
        execution_resume_source_run_id=operations.clean_text(
            execution_resume_source_run_id,
            160,
        ),
        runner_image=selected_runner_image,
    )
