"""First-day exposure timing is closed before planning, from synthetic metadata."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.research_agent.contracts.exposure_group_rules import EXPOSURE_GROUP_TRANSFORM_ID
from easyicu.research_agent.planning.family_spec import family_template_id_for_context
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import (
    AnalysisPlan, AnalysisStep, CohortDescriptor, ConceptDescriptor, ResearchContext,
    UserPreferences, VariableRole,
)
from easyicu.webserver import agent_pipeline_runs, study_contexts
from easyicu.webserver import research_pipeline_run_preparation as preparation
from easyicu.webserver.pi_copilot.decision_records import project_decision_records
from easyicu.webserver.pi_copilot.plan_decisions import (
    landmark_timing_specification, post_baseline_landmark_configuration,
)
from easyicu.webserver.research_pipeline_run_preparation import ResearchPipelineLaunchRequest


@pytest.fixture
def launch(monkeypatch, tmp_path):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    monkeypatch.setattr(study_contexts, "_now", lambda: "2026-10-23T01:00:00Z")
    monkeypatch.setattr(preparation, "_require_export_holds_study_cohort", lambda *_: None)
    monkeypatch.setattr(preparation, "_validate_primary_concept_selection", lambda *_: None)
    monkeypatch.setattr(preparation, "_patient_grouping_for_analysis_design", lambda *_: None)
    coordinates = {
        "primary_exposure": "synthetic_assay",
        "primary_exposure_aggregation": "max",
        "target_outcome": "death",
        "endpoint": EndpointSpec(
            name="death", kind="binary", absence_semantics="no_absent_rows", levels=[0, 1],
        ),
    }
    monkeypatch.setattr(preparation, "_metadata_only_planning_coordinates", lambda **_: coordinates)

    def prepare(study, **overrides):
        request = ResearchPipelineLaunchRequest(
            export_path=str(tmp_path / "synthetic-source"), study_context=study,
            project_root=str(tmp_path), provider={}, provider_environment={},
            credential_source="pi_verified", literature_search_authorized=False,
            plan_revision_source_run_id="", execution_resume_source_run_id="",
            development_resume_source_job_id="", budget_mode="planner_canary", runner_image=None,
        )
        return preparation._prepare_scientific_launch(replace(request, **overrides))

    return prepare, coordinates


def _study():
    return {
        "id": "synthetic-first-day", "question": "Estimate the stated association.",
        "data_source": {"database": "eicu"}, "modules": ["labevents"],
        "time_window": {"anchor": "icu_admission", "hours": 24},
        "analysis_design": {
            "analysis_family": "association_study", "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "execution_concepts": {
            "primary_exposure": "synthetic_assay", "primary_exposure_aggregation": "max",
            "outcome": "death", "covariates": [],
        },
        "sensitivity_specs": [], "confirmations": {"feature_time_window": True},
    }


@pytest.mark.parametrize("binding", ["study_setup", "plan_card", "metadata_coordinates"])
def test_first_plan_receives_the_landmark_for_each_binding_path(launch, binding, monkeypatch):
    prepare, _coordinates = launch
    raw = _study()
    if binding != "study_setup":
        raw["execution_concepts"] = {}
    saved = study_contexts.upsert_context(raw)
    if binding == "plan_card":
        saved = study_contexts.upsert_context({
            "id": saved["id"], "execution_concepts": _study()["execution_concepts"],
        }, expected_revision=saved["revision"])
    before = deepcopy(saved)

    scientific = prepare(saved)
    preferences = agent_pipeline_runs._research_user_preferences(scientific.materialization_study)

    assert preferences.get("landmark_hours") == 24
    [timing] = [spec for spec in scientific.sensitivity_specs if spec.axis == "timing"]
    assert timing.spec_id == "agent_post_baseline_landmark_24h"
    assert timing.event_time_variable == "death_time_hours"
    assert timing.observation_duration_variable == "hospital_followup_time_hours"
    assert timing.require_alive_at_landmark is True
    assert timing.exclude_negative_event_times is True
    assert timing.observation_duration_unit == "hours"
    assert scientific.study["confirmations"]["plan_timing_landmark_24h"] is True
    assert {"death_time_hours", "hospital_followup_time_hours"} <= set(
        scientific.metadata_operationalized_columns
    )
    stored = study_contexts.get_context(saved["id"])
    assert stored["sensitivity_specs"] == scientific.study["sensitivity_specs"]
    assert prepare(stored).study["revision"] == stored["revision"]
    assert saved == before

    records = project_decision_records(
        host_action_turns=[], data_source_authorization={}, recovery_receipts=[], study=stored,
    )
    assert records == [{
        "code": "post_baseline_landmark_selected", "actor": "agent", "at": stored["updated_at"],
    }]
    monkeypatch.setattr(study_contexts, "_now", lambda: "2026-10-23T02:00:00Z")
    edited = study_contexts.upsert_context({"id": stored["id"], "title": "Renamed study"})
    assert project_decision_records(
        host_action_turns=[], data_source_authorization={}, recovery_receipts=[], study=edited,
    ) == records


def _context(*, exposure=None, window=None):
    return ResearchContext(
        research_question="Synthetic association.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0, id_columns=["stay"]),
        variables=[
            exposure or ConceptDescriptor(name="synthetic_assay", role=VariableRole.LAB, dtype="float64"),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="bool", source_concept="death"),
            ConceptDescriptor(name="death_time_hours", role=VariableRole.TIME, dtype="float64"),
            ConceptDescriptor(name="hospital_followup_time_hours", role=VariableRole.TIME, dtype="float64"),
        ],
        primary_exposure="synthetic_assay", target_outcome="death",
        endpoint=EndpointSpec(name="death", kind="binary", levels=[0, 1], absence_semantics="no_absent_rows"),
        user_preferences=UserPreferences(data_constraints=json.dumps({
            "materialization_window": window or {
                "role": "outer_observation_window", "anchor": "icu_admission", "hours": 24,
            },
        })),
    )


@pytest.mark.parametrize("hours", [0.5, 12, 24])
def test_a_window_ending_within_the_first_day_uses_the_existing_landmark_shape(hours):
    context = _context(window={"role": "outer_observation_window", "anchor": "icu_admission", "hours": hours})
    study = _study()
    study["sensitivity_specs"] = [{
        "spec_id": "missing", "axis": "missing_data", "strategy": "complete_case",
        "execution_variables": ["synthetic_assay"],
    }]
    patch = post_baseline_landmark_configuration(study=study, context=context)
    assert patch is not None
    assert patch["sensitivity_specs"][0]["spec_id"] == "missing"
    [timing] = [spec for spec in patch["sensitivity_specs"] if spec["axis"] == "timing"]
    assert {key: value for key, value in timing.items() if value is not None} == landmark_timing_specification(
        outcome="death", landmark_hours=24, spec_id="agent_post_baseline_landmark_24h",
    )
    assert patch["confirmations"] == {
        "feature_time_window": True, "plan_timing_landmark_24h": True,
        "plan_timing_descriptive_only": False, "plan_timing_time_varying": False,
    }


@pytest.mark.parametrize("family", ["descriptive_epidemiology", "prediction_model", "survival", "causal_inference", ""])
def test_only_an_association_gets_the_default(family):
    study = _study()
    study["analysis_design"]["analysis_family"] = family
    assert post_baseline_landmark_configuration(study=study, context=_context()) is None


@pytest.mark.parametrize("exposure", ["age", "sex", "adm", "age_group_synthetic"])
def test_baseline_exposures_keep_their_admission_origin(launch, exposure):
    prepare, _ = launch
    study = _study()
    study["execution_concepts"]["primary_exposure"] = exposure
    saved = study_contexts.upsert_context(study)
    assert prepare(saved).study == saved
    assert study_contexts.get_context(saved["id"])["revision"] == saved["revision"]


def test_a_typed_grouping_of_baseline_attributes_stays_baseline():
    context = _context(exposure=ConceptDescriptor(
        name="synthetic_assay", role=VariableRole.OTHER, dtype="int64",
        unit_normalization=EXPOSURE_GROUP_TRANSFORM_ID, derived_from_concepts=["baseline_attribute"],
    ))
    context = context.model_copy(update={"variables": [
        *context.variables, ConceptDescriptor(name="baseline_attribute", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
    ]})
    assert post_baseline_landmark_configuration(study=_study(), context=context) is None


@pytest.mark.parametrize("hours", [25, 48])
def test_an_outer_window_the_landmark_cannot_close_is_not_selected(hours):
    context = _context(window={"role": "outer_observation_window", "anchor": "icu_admission", "hours": hours})
    assert post_baseline_landmark_configuration(study=_study(), context=context) is None


@pytest.mark.parametrize("window", [
    {"hours": 0}, {"hours": -1}, {"hours": "24"},
    {"hours": True}, {"hours": 24, "anchor": "hospital_admission"},
    {"hours": 24, "anchor": ""}, {"hours": 24, "role": "exposure_definition"},
])
def test_an_unreadable_outer_window_requires_an_exposure_window(window):
    context = _context(window={"role": "outer_observation_window", "anchor": "icu_admission", **window})
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        post_baseline_landmark_configuration(study=_study(), context=context)
    assert stopped.value.reason_code == "progressive_exposure_window_required"


@pytest.mark.parametrize("window, selected", [
    ("icu_admission[0,12]h", True), ("icu_admission[6,24]h", True),
    ("icu_admission[0,36]h", False), ("icu_admission[-12,0]h", False),
    ("icu_admission[-12,12]h", False), ("icu_admission[24,12]h", False),
    ("hospital_admission[0,24]h", False), ("event_onset[0,24]h", False),
    ("unknown 24h", False),
])
def test_the_exposures_own_window_has_priority(window, selected):
    context = _context(exposure=ConceptDescriptor(
        name="synthetic_assay", role=VariableRole.LAB, dtype="float64", analysis_window=window,
    ))
    assert (post_baseline_landmark_configuration(study=_study(), context=context) is not None) == selected


@pytest.mark.parametrize("outcome", ["death_icu", "los_icu", "synthetic_event"])
def test_other_outcomes_keep_the_existing_route(launch, outcome):
    prepare, coordinates = launch
    coordinates["target_outcome"] = outcome
    coordinates["endpoint"] = EndpointSpec(name=outcome, kind="binary", levels=[0, 1], absence_semantics="no_absent_rows")
    study = _study()
    study["execution_concepts"]["outcome"] = outcome
    saved = study_contexts.upsert_context(study)
    assert prepare(saved).study == saved
    context = _context().model_copy(update={"target_outcome": outcome, "endpoint": coordinates["endpoint"]})
    assert post_baseline_landmark_configuration(study=study, context=context) is None


@pytest.mark.parametrize("endpoint", [None, EndpointSpec(name="death", kind="continuous", absence_semantics="no_absent_rows")])
def test_no_binary_endpoint_contract_means_no_default(endpoint):
    assert post_baseline_landmark_configuration(
        study=_study(), context=_context().model_copy(update={"endpoint": endpoint}),
    ) is None


@pytest.mark.parametrize("choice", ["landmark", "time_varying", "descriptive", "other_confirmed"])
def test_a_researchers_timing_choice_is_untouched(launch, choice):
    prepare, _ = launch
    study = _study()
    if choice == "landmark":
        study["sensitivity_specs"] = [landmark_timing_specification(outcome="death", landmark_hours=12, spec_id="researcher_timing")]
    elif choice == "time_varying":
        study["sensitivity_specs"] = [{"spec_id": "researcher_timing", "axis": "timing", "strategy": "time_varying"}]
    else:
        key = "plan_timing_descriptive_only" if choice == "descriptive" else "plan_timing_researcher_design"
        study["confirmations"][key] = True
    assert post_baseline_landmark_configuration(study=study, context=_context()) is None
    saved = study_contexts.upsert_context(study)
    assert prepare(saved).study == saved


@pytest.mark.parametrize("preference", [
    {"landmark_hours": 12}, {"timing_and_design": "A researcher-stated timing design"},
    {"sensitivity_specs": [{"spec_id": "stated", "axis": "timing", "strategy": "time_varying"}]},
])
def test_timing_already_on_the_planning_context_is_preserved(preference):
    context = _context()
    context = context.model_copy(update={"user_preferences": UserPreferences.model_validate({
        **context.user_preferences.model_dump(), **preference,
    })})
    assert post_baseline_landmark_configuration(study=_study(), context=context) is None


def test_the_new_timing_routes_directly_and_the_review_does_not_reopen_it(launch):
    prepare, _ = launch
    scientific = prepare(study_contexts.upsert_context(_study()))
    context = _context().model_copy(update={"user_preferences": UserPreferences.model_validate(
        agent_pipeline_runs._research_user_preferences(scientific.materialization_study)
    )})
    assert family_template_id_for_context(context, analysis_types=["association_study"]) == "landmark_spline_association"
    plan = AnalysisPlan(research_question=context.research_question, analysis_type="association_study", steps=[
        AnalysisStep(
            step_id="primary", planned_analysis_role="primary", intent="Estimate the association at the landmark.",
            method="signed_landmark_restricted_cubic_spline", inputs=["cohort:analysis_set", "synthetic_assay", "death"],
            expected_outputs=["table:landmark_rcs_contrasts"], sensitivity_spec_ids=[scientific.sensitivity_specs[0].spec_id],
        ),
    ])
    review = build_plan_scientific_review(context=context, plan=plan)
    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" not in {finding.code for finding in review.findings}


def test_a_concurrent_timing_edit_wins_over_launch(launch):
    prepare, _ = launch
    saved = study_contexts.upsert_context(_study())
    choice = landmark_timing_specification(outcome="death", landmark_hours=12, spec_id="researcher_timing")
    updated = study_contexts.upsert_context({"id": saved["id"], "sensitivity_specs": [choice]})
    with pytest.raises(study_contexts.StudyContextError) as caught:
        prepare(saved)
    assert caught.value.detail["error"] == "study_context_revision_conflict"
    assert study_contexts.get_context(saved["id"]) == updated


def test_an_agent_record_cannot_be_supplied_as_a_client_field(launch):
    with pytest.raises(study_contexts.StudyContextError):
        study_contexts.upsert_context({
            **_study(), "agent_timing_decision": {
                "code": "post_baseline_landmark_selected", "actor": "agent", "at": "2026-10-23T00:00:00Z",
            },
        })


@pytest.mark.parametrize("damage", [
    {"actor": "user"}, {"actor": "system"}, {"code": "unrecognized"},
    {"at": "invalid"}, {"at": "2026-10-23T01:00:00"},
])
def test_a_timing_record_requires_the_host_code_actor_and_time(damage):
    assert project_decision_records(
        host_action_turns=[], data_source_authorization={}, recovery_receipts=[],
        study={"agent_timing_decision": {
            "code": "post_baseline_landmark_selected", "actor": "agent", "at": "2026-10-23T01:00:00Z",
            **damage,
        }},
    ) == []


def test_a_client_supplied_spec_id_is_not_an_agent_decision(launch):
    saved = study_contexts.upsert_context({
        **_study(), "sensitivity_specs": [landmark_timing_specification(
            outcome="death", landmark_hours=24, spec_id="agent_post_baseline_landmark_24h",
        )],
    })
    assert "agent_timing_decision" not in saved
    assert project_decision_records(
        host_action_turns=[], data_source_authorization={}, recovery_receipts=[], study=saved,
    ) == []


@pytest.mark.parametrize("resume", ["execution_resume_source_run_id", "development_resume_source_job_id"])
def test_a_resumed_run_keeps_its_existing_timing(launch, resume):
    prepare, _ = launch
    saved = study_contexts.upsert_context(_study())
    assert prepare(saved, **{resume: "previous-run"}).study == saved


def test_an_unbound_exposure_does_not_get_a_landmark(launch):
    prepare, coordinates = launch
    coordinates["primary_exposure"] = None
    study = _study()
    study["execution_concepts"].pop("primary_exposure")
    study["execution_concepts"].pop("primary_exposure_aggregation")
    saved = study_contexts.upsert_context(study)
    assert prepare(saved).study == saved


def test_an_exposure_defined_at_another_clinical_anchor_keeps_the_review_route():
    from easyicu.research_agent.research_context.clinical_definition import ClinicalDefinitionReference

    exposure = _context().variables[0].model_copy(update={
        "clinical_definition": ClinicalDefinitionReference(
            contract_id="synthetic-definition", definition="synthetic", version="1",
            source_id="synthetic-source", definition_time_anchor="event_onset",
            status="defined", validation_status="synthetic", canonical_definition=True,
        ),
    })
    assert post_baseline_landmark_configuration(study=_study(), context=_context(exposure=exposure)) is None
