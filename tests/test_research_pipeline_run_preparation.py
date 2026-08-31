"""Contract tests for the Research Agent launch-preparation owner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.research_pipeline_run_preparation import (
    PreparedLaunchAuthority,
    PreparedLaunchExecution,
    PreparedScientificLaunch,
    ResearchPipelineLaunchRequest,
    prepare_research_pipeline_run,
)


def _request() -> ResearchPipelineLaunchRequest:
    return ResearchPipelineLaunchRequest(
        export_path="/typed/demo",
        study_context={
            "id": "study-1",
            "question": "How common is sepsis?",
            "data_source": {"database": "miiv"},
        },
        project_root="/server/workspace",
        provider={"provider": "openai"},
        provider_environment={"OPENAI_API_KEY": "test"},
        credential_source="pi_verified",
        literature_search_authorized=False,
        plan_revision_source_run_id="",
        execution_resume_source_run_id="",
        development_resume_source_job_id="",
        budget_mode="planner_canary",
        runner_image=None,
    )


def _scientific() -> PreparedScientificLaunch:
    return PreparedScientificLaunch(
        study={"id": "study-1"},
        question="How common is sepsis?",
        database="miiv",
        materialization_study={},
        configured_target=None,
        configured_primary_exposure=None,
        target=None,
        primary_exposure=None,
        covariates=(),
        covariate_selection="none",
        sensitivity_specs=(),
        cohort_window=(0.0, 24.0),
        validated_analysis_design={},
        patient_grouping=None,
        metadata_only_planning=True,
        metadata_planning_coordinates={},
        execution_concepts={},
        planning_exposure_source=None,
        metadata_operationalized_columns=(),
        prepared_package_binding=None,
        foundation_profile={},
    )


def test_preparation_compiles_three_frozen_states_in_authority_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    calls: list[str] = []
    scientific = _scientific()
    authority = PreparedLaunchAuthority(
        provider={},
        provider_environment={},
        credential_source="pi_verified",
        literature_search_authorized=False,
        publication_skill_flags={},
        user_extension_activation=None,
    )
    execution = PreparedLaunchExecution(
        export_path="/typed/demo",
        project_root="/server/workspace",
        budget_mode="planner_canary",
        development_resume_binding=None,
        development_resume_acquisition=None,
        development_resume_literature=None,
        plan_revision_source_run_id="",
        execution_resume_source_run_id="",
        runner_image="",
    )

    def prepare_scientific(_launch: ResearchPipelineLaunchRequest):
        calls.append("scientific")
        return scientific

    def prepare_authority(_launch: ResearchPipelineLaunchRequest):
        calls.append("authority")
        return provider_token

    def prepare_execution(
        _launch: ResearchPipelineLaunchRequest,
        prepared_scientific: PreparedScientificLaunch,
        provider_authorization: object,
    ):
        calls.append("execution")
        assert prepared_scientific is scientific
        assert provider_authorization is provider_token
        return authority, execution

    provider_token = object()
    monkeypatch.setattr(preparation, "_prepare_scientific_launch", prepare_scientific)
    monkeypatch.setattr(preparation, "_authorize_launch_provider", prepare_authority)
    monkeypatch.setattr(preparation, "_prepare_launch_execution", prepare_execution)

    prepared = prepare_research_pipeline_run(_request())

    assert prepared.scientific.database == "miiv"
    assert prepared.scientific.metadata_only_planning is True
    assert calls == ["scientific", "authority", "execution"]
    with pytest.raises(FrozenInstanceError):
        prepared.scientific = scientific  # type: ignore[misc]


def test_public_preparation_interface_does_not_expose_primitive_operations() -> None:
    signature = inspect.signature(prepare_research_pipeline_run)

    assert tuple(signature.parameters) == ("request",)
    assert len(inspect.signature(PreparedScientificLaunch).parameters) < 22
    assert len(inspect.signature(PreparedLaunchAuthority).parameters) == 6
    assert len(inspect.signature(PreparedLaunchExecution).parameters) == 9


def test_pipeline_factory_delegates_launch_policy_to_preparation_owner() -> None:
    source = inspect.getsource(agent_pipeline_runs.make_research_pipeline_run_runner)
    preparation, separator, _runner = source.partition("    def runner(")

    assert separator
    assert preparation.count("prepare_research_pipeline_run(") == 1
    assert "ResearchPipelinePreparationOperations" not in preparation
    assert "_research_pipeline_preparation_operations" not in preparation
    for policy_reason in (
        "research_pipeline_question_required",
        "research_pipeline_database_unknown",
        "research_pipeline_budget_mode_invalid",
        "research_pipeline_runner_image_invalid",
    ):
        assert policy_reason not in preparation
