"""Compile one Research Agent launch into immutable prepared state.

This module owns the order-sensitive launch seam: scientific authority,
source/package binding, provider credentials, runtime availability, and resume
coordinates are validated before the JobManager runner can perform side
effects. Callers submit one launch request and consume three cohesive prepared
states; primitive policy lookups stay private to this module.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
from easyicu.webserver.research_launch_resume import (
    _development_progressive_resume_binding,
    _development_resume_acquisition_profile,
    _development_resume_literature_bundle,
)
from easyicu.webserver.research_launch_runtime import (
    _require_execution_runtime,
    _validated_pipeline_credential_source,
)
from easyicu.webserver.research_launch_scientific import (
    _cohort_window,
    _configured_covariate_selection,
    _configured_covariates,
    _configured_sensitivity_specs,
    _data_foundation_profile,
    _metadata_only_planning_coordinates,
    _metadata_planning_operationalized_columns,
    _neutral_materialization_scope,
    _patient_grouping_for_analysis_design,
    _primary_exposure,
    _primary_exposure_aggregation,
    _target_outcome,
    _validate_analysis_design,
    _validate_primary_concept_selection,
)

_RUNNER_IMAGE_ENV = "EASYICU_RUNNER_IMAGE"
_DEVELOPMENT_RESUME_JOB_ENV = "EASYICU_DEVELOPMENT_PROGRESSIVE_RESUME_SOURCE_JOB_ID"
_DEVELOPMENT_RESUME_SEQUENCE_ENV = (
    "EASYICU_DEVELOPMENT_PROGRESSIVE_RESUME_CHECKPOINT_SEQUENCE"
)


@dataclass(frozen=True)
class ResearchPipelineLaunchRequest:
    """Server-owned request for one governed Research Agent launch."""

    export_path: str
    study_context: Mapping[str, Any]
    project_root: Optional[str]
    provider: Mapping[str, Any]
    provider_environment: Optional[Mapping[str, str]]
    credential_source: str
    literature_search_authorized: bool
    plan_revision_source_run_id: str
    execution_resume_source_run_id: str
    development_resume_source_job_id: str
    budget_mode: str
    runner_image: Optional[str]


@dataclass(frozen=True)
class PreparedScientificLaunch:
    """Validated scientific configuration and source authority."""

    study: Mapping[str, Any]
    question: str
    database: str
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


@dataclass(frozen=True)
class PreparedLaunchAuthority:
    """Validated provider, extension, and publication authority."""

    provider: Mapping[str, Any]
    provider_environment: Mapping[str, str]
    credential_source: str
    literature_search_authorized: bool
    publication_skill_flags: Mapping[str, Any]
    user_extension_activation: Any


@dataclass(frozen=True)
class PreparedLaunchExecution:
    """Validated workspace, resume, and runtime coordinates."""

    export_path: str
    project_root: str
    budget_mode: str
    development_resume_binding: Optional[tuple[Path, str]]
    development_resume_acquisition: Any
    development_resume_literature: Optional[Mapping[str, Any]]
    plan_revision_source_run_id: str
    execution_resume_source_run_id: str
    runner_image: str


@dataclass(frozen=True)
class PreparedResearchPipelineRun:
    """Side-effect-free launch state consumed by the pipeline runner."""

    scientific: PreparedScientificLaunch
    authority: PreparedLaunchAuthority
    execution: PreparedLaunchExecution


@dataclass(frozen=True)
class _ProviderAuthorization:
    provider: Mapping[str, Any]
    provider_environment: Mapping[str, str]
    credential_source: str
    literature_search_authorized: bool


def _clean_text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _prepare_scientific_launch(
    request: ResearchPipelineLaunchRequest,
) -> PreparedScientificLaunch:
    study = dict(request.study_context)
    question = _clean_text(study.get("question"), 1_200)
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
    database_raw = _clean_text(source.get("database"), 64)
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

    budget_mode = str(request.budget_mode or "").strip().lower()
    materialization_study = _neutral_materialization_scope(
        study,
        export_path=request.export_path,
    )
    configured_target = _target_outcome(study)
    configured_primary_exposure = _primary_exposure(study)
    planning_coordinates = _metadata_only_planning_coordinates(
        question=question,
        database=database,
    )
    target = configured_target or planning_coordinates.get("target_outcome")
    primary_exposure = configured_primary_exposure or planning_coordinates.get(
        "primary_exposure"
    )
    _validate_primary_concept_selection(
        study,
        configured_primary_exposure,
    )
    covariates = _configured_covariates(study)
    covariate_selection = _configured_covariate_selection(study)
    sensitivity_specs = _configured_sensitivity_specs(study)
    window = _cohort_window(materialization_study)
    validated_analysis_design = _validate_analysis_design(study)
    patient_grouping = (
        _patient_grouping_for_analysis_design(study)
        if validated_analysis_design.get("variance_estimator") == "cluster_robust"
        else None
    )

    metadata_only_planning = budget_mode == "planner_canary"
    metadata_planning_coordinates: Dict[str, Any] = dict(planning_coordinates)
    execution_concepts_raw = study.get("execution_concepts")
    execution_concepts = (
        dict(execution_concepts_raw)
        if isinstance(execution_concepts_raw, Mapping)
        else {}
    )
    planning_exposure_source = (
        _clean_text(execution_concepts.get("primary_exposure"), 160)
        or metadata_planning_coordinates.get("primary_exposure")
        or None
    )
    planning_exposure_aggregation = _primary_exposure_aggregation(study)
    metadata_operationalized_columns = (
        _metadata_planning_operationalized_columns(
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
        foundation_profile = _data_foundation_profile(
            export_path=request.export_path,
            study=materialization_study,
            target=configured_target,
            primary_exposure=configured_primary_exposure,
            covariates=covariates,
            sensitivity_specs=sensitivity_specs,
        )
        try:
            package_receipt = dataio.validate_research_pipeline_source(
                request.export_path,
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

    return PreparedScientificLaunch(
        study=study,
        question=question,
        database=database,
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
    )


def _authorize_launch_provider(
    request: ResearchPipelineLaunchRequest,
) -> _ProviderAuthorization:
    budget_mode = str(request.budget_mode or "").strip().lower()
    try:
        provider_adapter.web_research_agent_hard_stop_limits(budget_mode)
    except ValueError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_budget_mode_invalid",
            "Choose an explicit reviewed Research Agent budget mode.",
        ) from exc

    credential_source = _validated_pipeline_credential_source(
        request.credential_source,
        provider=request.provider,
    )
    if request.provider_environment is None:
        raise ResearchPipelineRunError(
            (
                "research_pipeline_codex_user_auth_required"
                if credential_source == "codex_user_auth"
                else "research_pipeline_pi_verified_credentials_required"
            ),
            "A verified provider credential is required for this pipeline.",
        )

    return _ProviderAuthorization(
        provider=dict(request.provider),
        provider_environment=dict(request.provider_environment),
        credential_source=credential_source,
        literature_search_authorized=bool(request.literature_search_authorized),
    )


def _prepare_launch_execution(
    request: ResearchPipelineLaunchRequest,
    scientific: PreparedScientificLaunch,
    provider_authorization: _ProviderAuthorization,
) -> tuple[PreparedLaunchAuthority, PreparedLaunchExecution]:
    if not request.project_root:
        raise ResearchPipelineRunError(
            "research_pipeline_project_workspace_required",
            "A server-owned Pi project workspace is required for this pipeline.",
        )
    project_root = request.project_root
    budget_mode = str(request.budget_mode or "").strip().lower()

    development_resume_binding: tuple[Path, str] | None = None
    selected_resume_source = _clean_text(
        request.development_resume_source_job_id
        or os.environ.get(_DEVELOPMENT_RESUME_JOB_ENV),
        80,
    )
    if selected_resume_source:
        development_resume_binding = _development_progressive_resume_binding(
            project_root=project_root,
            study_id=str(scientific.study.get("id") or ""),
            source_job_id=selected_resume_source,
            budget_mode=budget_mode,
            checkpoint_sequence=os.environ.get(_DEVELOPMENT_RESUME_SEQUENCE_ENV),
        )
    development_resume_acquisition = (
        _development_resume_acquisition_profile(
            checkpoint_path=development_resume_binding[0],
            database=scientific.database,
            cohort_window=scientific.cohort_window,
            outcome_concepts=scientific.foundation_profile["outcome_concepts"],
            static_concepts=scientific.foundation_profile["static_concepts"],
            required_feature_concepts=scientific.foundation_profile[
                "required_feature_concepts"
            ],
            planning_target_outcome=scientific.metadata_planning_coordinates.get(
                "target_outcome"
            ),
            planning_endpoint=scientific.metadata_planning_coordinates.get("endpoint"),
            planning_operationalized_columns=(
                scientific.metadata_operationalized_columns
            ),
        )
        if development_resume_binding is not None
        else None
    )
    development_resume_literature = (
        _development_resume_literature_bundle(
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
        request.runner_image
        if request.runner_image is not None
        else os.environ.get(_RUNNER_IMAGE_ENV)
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
    _require_execution_runtime(
        budget_mode=budget_mode,
        runner_image=selected_runner_image,
    )

    return (
        PreparedLaunchAuthority(
            provider=provider_authorization.provider,
            provider_environment=provider_authorization.provider_environment,
            credential_source=provider_authorization.credential_source,
            literature_search_authorized=(
                provider_authorization.literature_search_authorized
            ),
            publication_skill_flags=publication_skill_flags,
            user_extension_activation=user_extension_activation,
        ),
        PreparedLaunchExecution(
            export_path=request.export_path,
            project_root=project_root,
            budget_mode=budget_mode,
            development_resume_binding=development_resume_binding,
            development_resume_acquisition=development_resume_acquisition,
            development_resume_literature=development_resume_literature,
            plan_revision_source_run_id=_clean_text(
                request.plan_revision_source_run_id,
                160,
            ),
            execution_resume_source_run_id=_clean_text(
                request.execution_resume_source_run_id,
                160,
            ),
            runner_image=selected_runner_image,
        ),
    )


def prepare_research_pipeline_run(
    request: ResearchPipelineLaunchRequest,
) -> PreparedResearchPipelineRun:
    """Validate one launch request completely before runner side effects."""

    scientific = _prepare_scientific_launch(request)
    provider_authorization = _authorize_launch_provider(request)
    authority, execution = _prepare_launch_execution(
        request,
        scientific,
        provider_authorization,
    )
    return PreparedResearchPipelineRun(
        scientific=scientific,
        authority=authority,
        execution=execution,
    )
