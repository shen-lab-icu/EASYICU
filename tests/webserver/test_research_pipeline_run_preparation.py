"""Contract tests for the Research Agent launch-preparation owner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
import hashlib
import json

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import research_pipeline_run_preparation as owner
from easyicu.webserver.research_launch_resume import _DevelopmentResumeLaunchScope
from tests.support.provider_recovery import recovery as recovery_fixture

recovery = recovery_fixture
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


def test_explicit_fresh_plan_runs_scientific_guards_even_with_developer_seed(monkeypatch):
    monkeypatch.setenv("EASYICU_DEVELOPMENT_PROGRESSIVE_RESUME_SOURCE_JOB_ID", "prior")
    def stop(*_args, **_kwargs):
        raise RuntimeError("fresh-scientific-guard")
    monkeypatch.setattr(owner, "stop_undistinguished_named_levels", stop)
    with pytest.raises(RuntimeError, match="fresh-scientific-guard"):
        owner._prepare_scientific_launch(replace(_request(), allow_environment_resume=False))


def test_explicit_fresh_plan_does_not_open_environment_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYICU_DEVELOPMENT_PROGRESSIVE_RESUME_SOURCE_JOB_ID", "prior")
    monkeypatch.setattr(owner, "_development_progressive_resume_binding",
                        lambda **_kwargs: pytest.fail("fresh plan cannot open an inherited checkpoint"))
    monkeypatch.setattr(owner, "_require_profile_dictionaries", lambda **_kwargs: None)
    monkeypatch.setattr(owner, "_require_execution_runtime", lambda **_kwargs: "synthetic")
    from easyicu.webserver import plan_change_requirements
    monkeypatch.setattr(plan_change_requirements, "compiled_configuration_plan_change", lambda **_kwargs: None)
    request = replace(_request(), project_root=str(tmp_path), allow_environment_resume=False)
    _, execution = owner._prepare_launch_execution(request, _scientific(), owner._authorize_launch_provider(request))
    assert execution.development_resume_binding is None
    assert execution.development_resume_scope is None
    assert execution.development_resume_literature is None


def test_a_requested_summary_cannot_read_a_different_materialization_window(monkeypatch, tmp_path):
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    request = replace(_request(), export_path=str(tmp_path))
    monkeypatch.setattr(preparation, "_metadata_only_planning_coordinates", lambda **kwargs: {
        "target_outcome_window": {"anchor": "icu_admission", "start_hours": 0.0, "end_hours": 24.0},
    })
    monkeypatch.setattr(preparation, "_cohort_window", lambda study: (0.0, 12.0))
    with pytest.raises(preparation.ResearchPipelineRunError) as error:
        preparation._prepare_scientific_launch(request)
    assert error.value.code == "windowed_summary_materialization_window_mismatch"
    assert str(error.value) == (
        "The prepared data covers a different time window from the one the question states, "
        "so the summary would not match the question. No analysis was run. Prepare the data "
        "again for the stated window."
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
                "analysis_design": {
                    "analysis_family": "association_study",
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "model_based",
                },
                "data_source": {"database": "miiv"},
            },
        }
    )
    captured: dict[str, object] = {}
    grouping = object()
    monkeypatch.setattr(
        preparation, "_patient_grouping_for_analysis_design", lambda _study: grouping,
    )
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
    assert scientific.patient_grouping is grouping


def test_a_trajectory_population_is_materialized_with_its_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    adult = {
        "concept_id": "age",
        "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24},
        "aggregation": "first",
        "op": ">=",
        "value": 18,
    }
    request = ResearchPipelineLaunchRequest(
        **{
            **_request().__dict__,
            "budget_mode": "full_reviewed",
            "study_context": {
                "id": "study-1",
                "question": "Do first-24h SOFA-2 trajectories form distinct classes?",
                "analysis_design": {
                    "analysis_family": "trajectory_clustering",
                    "analysis_unit": "icu_stay",
                    "variance_estimator": "model_based",
                },
                "trajectory_design": {
                    "coordinate_concepts": ["sofa2_resp", "sofa2_cardio"],
                    "window_end_hours": 24,
                    "grid_width_hours": 4,
                    "population": {"inclusion": [adult], "exclusion": []},
                },
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
        lambda **_kwargs: {"target_outcome": None, "primary_exposure": None, "endpoint": None},
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
            "allowed_modules": ("labevents", "demographics"),
            "static_concepts": ("age",),
            "outcome_concepts": (),
            "required_feature_concepts": ("sofa2_resp", "sofa2_cardio"),
            "require_outcome": False,
            "primary_exposure_source_concept": None,
        }

    monkeypatch.setattr(preparation, "_data_foundation_profile", foundation)
    monkeypatch.setattr(
        preparation.dataio,
        "validate_research_pipeline_source",
        lambda *_args, **_kwargs: {"binding": {"sha256": "a" * 64}},
    )

    preparation._prepare_scientific_launch(request)

    # The panel is cut from the coordinates; the population's predicates read
    # stay-level columns of the same universe, so both are materialized.
    assert captured["trajectory_concepts"] == ("sofa2_resp", "sofa2_cardio", "age")


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


def test_candidate_defers_optional_identity_checks_to_the_planning_status_owner(monkeypatch):
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    request = replace(_request(), study_context={
        **_request().study_context,
        "analysis_design": {"analysis_unit": "icu_stay", "variance_estimator": "model_based"},
    })

    def forbidden(_study):
        raise AssertionError("optional authority errors belong in the planning status")

    monkeypatch.setattr(preparation, "_patient_grouping_for_analysis_design", forbidden)
    scientific = preparation._prepare_scientific_launch(request)
    assert scientific.patient_grouping is None


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


def test_candidate_preparation_keeps_a_stated_adjustment_in_its_required_roster(monkeypatch):
    from types import SimpleNamespace
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _path: SimpleNamespace(concepts=[SimpleNamespace(concept_id="age")]),
    )
    study = {
        "id": "study-1", "question": "调整年龄后估计指标与结局的关系。",
        "data_source": {"database": "miiv"},
        "stated_adjustments": [{"variable": "年龄", "words": "调整年龄", "concept": "age", "source_status": "unbound"}],
    }
    result = preparation._prepare_scientific_launch(replace(_request(), study_context=study))
    assert result.metadata_planning_coordinates["required_adjustment_concepts"] == ["age"]
    assert result.covariate_selection == "planner_selectable"


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
    assert len(inspect.signature(PreparedLaunchExecution).parameters) == 12
    assert "runner_kind" in inspect.signature(PreparedLaunchExecution).parameters
    # Restored mode and planning constraint travel as one immutable owner value,
    # not additional adapter-controlled primitive arguments.
    assert "development_resume_scope" in inspect.signature(PreparedLaunchExecution).parameters


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


@pytest.fixture
def source(tmp_path, monkeypatch):
    run = tmp_path / "study-1/run_prior/pipeline/run_synthetic"
    run.mkdir(parents=True)
    checkpoint = run / "progressive_planner_checkpoint_002.json"
    checkpoint.write_text("synthetic checkpoint; integrity checked by the binding owner")
    scope = _DevelopmentResumeLaunchScope(
        "planner_canary", "Retain the reviewed synthetic question.",
        {"required_primary_cohort_selection_mode": "all_input_rows"},
    )
    monkeypatch.setattr(owner, "_development_progressive_resume_binding", lambda **kw: (checkpoint, "a" * 64))
    monkeypatch.setattr(owner, "_development_resume_launch_scope", lambda **kw: scope)
    monkeypatch.setattr(owner, "_development_resume_literature_bundle", lambda **kw: {"synthetic": True})
    monkeypatch.setattr(owner, "_require_execution_runtime", lambda **kw: "synthetic")
    monkeypatch.setattr(owner, "_require_profile_dictionaries", lambda **kw: None)
    request = replace(_request(), project_root=str(tmp_path), development_resume_source_job_id="prior")
    scientific = replace(_scientific(), foundation_profile={
        "outcome_concepts": (), "static_concepts": (), "required_feature_concepts": (),
    })

    def prepare(mode="planner_canary", metadata=True):
        return owner._prepare_launch_execution(
            replace(request, budget_mode=mode),
            replace(scientific, metadata_only_planning=metadata),
            owner._authorize_launch_provider(request),
        )[1]

    return checkpoint, scope, prepare


def test_missing_first_pass_catalog_restarts_planning_with_the_same_scope(source):
    checkpoint, scope, prepare = source
    prepared = prepare()
    assert prepared.development_resume_binding is None
    assert prepared.development_resume_acquisition is None
    assert prepared.development_resume_scope == scope
    assert prepared.development_resume_literature == {"synthetic": True}
    assert prepared.budget_mode == "planner_canary"
    assert not (checkpoint.parents[2] / "pipeline_input").exists()
    assert not prepared.execution_resume_source_run_id


@pytest.mark.parametrize("mode,metadata", [("full_reviewed", False), ("planner_canary", False)])
def test_later_stages_cannot_restart_without_prepared_input(source, mode, metadata):
    _, _, prepare = source
    with pytest.raises(owner.ResearchPipelineRunError):
        prepare(mode, metadata)


@pytest.mark.parametrize("name", [
    "planner_catalog.parquet", "planner_catalog_receipt.json",
    "web_research_universe.parquet", "web_research_universe_provenance.json",
    "universe.parquet",
])
def test_partial_or_unverifiable_input_never_triggers_a_fresh_restart(source, name):
    checkpoint, _, prepare = source
    inputs = checkpoint.parents[2] / "pipeline_input"
    inputs.mkdir()
    (inputs / name).write_text("unverifiable synthetic artifact")
    with pytest.raises(owner.ResearchPipelineRunError):
        prepare()


def test_linked_input_directory_never_triggers_a_restart(source, tmp_path):
    checkpoint, scope, prepare = source
    other = tmp_path / "synthetic-other"
    other.mkdir()
    (checkpoint.parents[2] / "pipeline_input").symlink_to(other, target_is_directory=True)
    assert not owner.metadata_planning_restart_allowed(checkpoint_path=checkpoint, scope=scope)
    with pytest.raises(owner.ResearchPipelineRunError):
        prepare()


@pytest.mark.parametrize("scope_change", [{"budget_mode": "full_reviewed"}, {"prepared_grouping": object()}])
def test_verified_prepared_scope_is_never_a_metadata_restart(source, scope_change):
    checkpoint, scope, _ = source
    assert not owner.metadata_planning_restart_allowed(
        checkpoint_path=checkpoint, scope=replace(scope, **scope_change),
    )


def test_available_zero_row_catalog_keeps_checkpoint_replay(source):
    import pyarrow as pa
    import pyarrow.parquet as pq

    checkpoint, scope, prepare = source
    inputs = checkpoint.parents[2] / "pipeline_input"
    inputs.mkdir()
    pq.write_table(pa.table({"stay_id": pa.array([], type=pa.int64()),
                             "age": pa.array([], type=pa.float64())}), inputs / "planner_catalog.parquet")
    (inputs / "planner_catalog_receipt.json").write_text(json.dumps({
        "schema_version": "easyicu.metadata-only-planning-catalog/1",
        "database": "miiv", "selected_concepts": ["age"],
        "selected_concepts_sha256": hashlib.sha256(b'["age"]').hexdigest(),
        "patient_rows_read": False, "patient_rows_written": False,
        "observed_feasibility_claims": False, "execution_authorized": False,
        "row_identity_column": "stay_id",
    }))
    prepared = prepare()
    assert prepared.development_resume_binding == (checkpoint, "a" * 64)
    assert prepared.development_resume_acquisition.kind == "metadata_only_planning_catalog"
    assert prepared.development_resume_acquisition.selected_concepts == ("age",)
    assert prepared.development_resume_scope == scope


def test_planning_restart_spends_the_existing_automatic_resume_budget(source, recovery):
    _, scope, prepare = source
    recovery.clock.recover_at = 0
    continued = []

    def interrupted(job):
        raise TimeoutError("synthetic stalled planner")

    def select(error):
        prepared = prepare()
        assert prepared.development_resume_scope == scope
        assert prepared.development_resume_binding is None
        return recovery.owner.CheckpointContinuation(
            {"kind": "planner", "restart_stage": "metadata_planning"},
            lambda job: continued.append(job.id) or interrupted(job),
        )

    with pytest.raises(recovery.owner.ProviderRecoveryExhausted):
        recovery.state.run(interrupted, select)
    assert len(continued) == len(recovery.state.resumes) == 2
    assert all(row["checkpoint"]["restart_stage"] == "metadata_planning" for row in recovery.state.resumes)
    assert all(row["cause_code"] == "provider_timeout" for row in recovery.state.resumes)
    from easyicu.webserver.pi_copilot.projections import project_job

    events = project_job({"id": "synthetic", "status": "failed", "events": recovery.job.events})["progress"]
    resumes = [event for event in events if event.get("step") == "provider_auto_resume"]
    assert len(resumes) == 2
    assert all(event["restart_stage"] == "metadata_planning" for event in resumes)


def test_sic_los_metadata_only_discussion_keeps_execution_scope_closed():
    from easyicu.webserver.study_family_scope import study_family_database_gap

    study = {"analysis_design": {"analysis_family": "association_study"},
             "execution_concepts": {"outcome": "los_icu", "primary_exposure": "lact"}}
    gap = study_family_database_gap(study, question="Association between first-day lactate and ICU length of stay", database="sic")
    request = _request()
    assert owner._metadata_only_sic_definition_gap(request, "sic", gap)
    assert not owner._metadata_only_sic_definition_gap(replace(request, budget_mode="full_reviewed"), "sic", gap)
    assert not owner._metadata_only_sic_definition_gap(request, "eicu", gap)
    assert not owner._metadata_only_sic_definition_gap(request, "sic", {**gap, "family": "survival"})
    assert not owner._metadata_only_sic_definition_gap(request, "sic", {**gap, "missing_capabilities": [
        *gap["missing_capabilities"], {"capability": "concept:lact", "code": "required_concept_unmapped"}]})
