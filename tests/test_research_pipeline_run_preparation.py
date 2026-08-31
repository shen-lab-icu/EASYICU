"""Contract tests for the Research Agent launch-preparation owner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
from typing import Any

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.research_pipeline_run_preparation import (
    ResearchPipelinePreparationOperations,
    prepare_research_pipeline_run,
)


def _operations(calls: list[str]) -> ResearchPipelinePreparationOperations:
    def mark(name: str, value: Any):
        def operation(*_args: Any, **_kwargs: Any) -> Any:
            calls.append(name)
            return value

        return operation

    return ResearchPipelinePreparationOperations(
        clean_text=lambda value, limit: str(value or "").strip()[:limit],
        neutral_materialization_scope=mark(
            "scope",
            {"id": "study-1", "time_window": {"hours": 24}},
        ),
        target_outcome=mark("target", None),
        primary_exposure=mark("exposure", None),
        primary_exposure_aggregation=mark("aggregation", None),
        metadata_only_planning_coordinates=mark("coordinates", {}),
        validate_primary_concept_selection=mark("concept_gate", None),
        configured_covariates=mark("covariates", ()),
        configured_covariate_selection=mark("covariate_selection", "none"),
        configured_sensitivity_specs=mark("sensitivities", ()),
        cohort_window=mark("window", (0.0, 24.0)),
        validate_analysis_design=mark("analysis_design", {}),
        patient_grouping_for_analysis_design=mark("grouping", None),
        metadata_planning_operationalized_columns=mark("columns", ()),
        data_foundation_profile=mark("foundation", {}),
        validated_pipeline_credential_source=mark("credentials", "pi_verified"),
        development_progressive_resume_binding=mark("resume", None),
        development_resume_acquisition_profile=mark("acquisition", None),
        development_resume_literature_bundle=mark("literature", {}),
        require_execution_runtime=mark("runtime", None),
    )


def test_preparation_compiles_one_frozen_state_before_runner_side_effects() -> None:
    calls: list[str] = []

    prepared = prepare_research_pipeline_run(
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
        operations=_operations(calls),
    )

    assert prepared.database == "miiv"
    assert prepared.metadata_only_planning is True
    assert calls.index("concept_gate") < calls.index("analysis_design")
    assert calls.index("analysis_design") < calls.index("credentials")
    assert calls[-1] == "runtime"
    with pytest.raises(FrozenInstanceError):
        prepared.database = "eicu"  # type: ignore[misc]


def test_pipeline_factory_delegates_launch_policy_to_preparation_owner() -> None:
    source = inspect.getsource(agent_pipeline_runs.make_research_pipeline_run_runner)
    preparation, separator, _runner = source.partition("    def runner(")

    assert separator
    assert preparation.count("prepare_research_pipeline_run(") == 1
    for policy_reason in (
        "research_pipeline_question_required",
        "research_pipeline_database_unknown",
        "research_pipeline_budget_mode_invalid",
        "research_pipeline_runner_image_invalid",
    ):
        assert policy_reason not in preparation
