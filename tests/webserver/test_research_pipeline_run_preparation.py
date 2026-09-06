"""Contract tests for the Research Agent launch-preparation owner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
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


def test_full_launch_materializes_planner_proposed_exposure_and_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    request = ResearchPipelineLaunchRequest(
        **{
            **_request().__dict__,
            "budget_mode": "full_reviewed",
            "study_context": {
                "id": "study-1",
                "question": "Is lactate associated with hospital mortality?",
                "outcome": "In-hospital mortality",
                "primary_exposure": "Peak lactate in the first 24 hours",
                "execution_concepts": {"covariates": []},
                "data_source": {"database": "miiv"},
            },
        }
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(preparation, "_neutral_materialization_scope", lambda study, **_kwargs: study)
    monkeypatch.setattr(
        preparation,
        "_metadata_only_planning_coordinates",
        lambda **_kwargs: {
            "target_outcome": "death",
            "primary_exposure": "lact",
            "endpoint": None,
        },
    )
    monkeypatch.setattr(preparation, "_validate_primary_concept_selection", lambda *_args: None)
    monkeypatch.setattr(preparation, "_configured_covariates", lambda _study: ())
    monkeypatch.setattr(preparation, "_configured_covariate_selection", lambda _study: "planner_selectable")
    monkeypatch.setattr(preparation, "_configured_sensitivity_specs", lambda _study: ())
    monkeypatch.setattr(preparation, "_cohort_window", lambda _study: (0.0, 24.0))
    monkeypatch.setattr(
        preparation,
        "_validate_analysis_design",
        lambda _study: {"variance_estimator": "model_based"},
    )

    def foundation(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "allowed_modules": ("labevents", "outcome", "demographics"),
            "static_concepts": ("age",),
            "outcome_concepts": ("death",),
            "required_feature_concepts": ("lact",),
            "require_outcome": True,
            "primary_exposure_source_concept": "lact",
        }

    monkeypatch.setattr(preparation, "_data_foundation_profile", foundation)
    monkeypatch.setattr(
        preparation.dataio,
        "validate_research_pipeline_source",
        lambda *_args, **_kwargs: {"binding": {"sha256": "a" * 64}},
    )

    scientific = preparation._prepare_scientific_launch(request)

    assert captured["target"] == "death"
    assert captured["primary_exposure"] == "lact"
    assert captured["require_target"] is False
    assert captured["require_primary_exposure"] is False
    assert scientific.foundation_profile["required_feature_concepts"] == ("lact",)


def test_candidate_plan_defers_unresolved_analysis_design_to_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    request = ResearchPipelineLaunchRequest(
        **{
            **_request().__dict__,
            "study_context": {
                "id": "study-1",
                "question": "Is lactate associated with hospital mortality?",
                "primary_exposure": "24-hour maximum lactate",
                "outcome": "hospital mortality",
                "analysis_design": {},
                "data_source": {"database": "miiv"},
            },
        }
    )
    monkeypatch.setattr(
        preparation,
        "_validate_analysis_design",
        lambda _study: (_ for _ in ()).throw(
            AssertionError("candidate planning must not require execution design")
        ),
    )

    scientific = preparation._prepare_scientific_launch(request)

    assert scientific.metadata_only_planning is True
    assert scientific.validated_analysis_design == {}
    assert scientific.patient_grouping is None


def test_candidate_plan_projects_preconfirmed_patient_grouping_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    grouping = object()
    request = ResearchPipelineLaunchRequest(
        **{
            **_request().__dict__,
            "study_context": {
                "id": "study-1",
                "question": "Is lactate associated with hospital mortality?",
                "analysis_design": {
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "cluster_robust",
                    "cluster_unit": "patient",
                },
                "data_source": {"database": "miiv"},
            },
        }
    )
    monkeypatch.setattr(
        preparation,
        "_validate_analysis_design",
        lambda _study: (_ for _ in ()).throw(
            AssertionError("candidate planning must remain metadata-only")
        ),
    )
    monkeypatch.setattr(
        preparation,
        "_patient_grouping_for_analysis_design",
        lambda _study: grouping,
    )

    scientific = preparation._prepare_scientific_launch(request)

    assert scientific.metadata_only_planning is True
    assert scientific.validated_analysis_design == {}
    assert scientific.patient_grouping is grouping


def test_candidate_plan_binds_question_operation_without_mutating_study() -> None:
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    study = {
        "id": "study-1",
        "question": "研究成人 ICU 患者前24小时最高乳酸与院内死亡",
        "data_source": {"database": "miiv"},
    }
    request = replace(_request(), study_context=study)
    result = preparation._prepare_scientific_launch(request)
    assert result.metadata_only_planning is True
    assert result.metadata_planning_coordinates["primary_exposure_aggregation"] == "max"
    assert "lact_max" in result.metadata_operationalized_columns
    assert "execution_concepts" not in study


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
    assert len(inspect.signature(PreparedLaunchExecution).parameters) == 10


@pytest.mark.parametrize("overrides", [
    {"budget_mode": "full_reviewed"},
    {"development_resume_source_job_id": "job-old"},
    {"execution_resume_source_run_id": "run-old"},
    {"plan_revision_source_run_id": "run-old"},
])
def test_direct_launch_rejects_amendments_before_runtime_or_provider(
    monkeypatch: pytest.MonkeyPatch, overrides: dict,
) -> None:
    from easyicu.webserver import research_pipeline_run_preparation as preparation
    from easyicu.webserver.plan_change_request import PlanChangeRequest
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    change = PlanChangeRequest(source_run_id="run-1", user_message="Revise the plan.")
    request = replace(_request(), plan_change_request=change, **overrides)
    scientific = replace(_scientific(), metadata_only_planning=request.budget_mode == "planner_canary")
    monkeypatch.setattr(
        preparation, "_development_progressive_resume_binding",
        lambda **_kwargs: pytest.fail("No checkpoint may be opened for new amendments"),
    )
    monkeypatch.setattr(
        preparation, "_require_execution_runtime",
        lambda **_kwargs: pytest.fail("Reject before runtime work"),
    )
    with pytest.raises(ResearchPipelineRunError) as raised:
        preparation._prepare_launch_execution(request, scientific, object())
    assert raised.value.code == "plan_changes_require_fresh_candidate"


def test_preparation_has_no_reverse_private_seam_into_pipeline_caller() -> None:
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    source = inspect.getsource(preparation)
    assert "_pipeline_policy" not in source
    assert "policy._" not in source
    assert "agent_pipeline_runs" not in source


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
