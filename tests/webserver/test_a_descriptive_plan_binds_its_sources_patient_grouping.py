"""Synthetic metadata-only descriptive plans carry grouping through execution."""

import ast
from dataclasses import make_dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.contracts.patient_grouping_need import (
    PATIENT_GROUPING_AUTHORITY_ERROR_KEY,
    PATIENT_GROUPING_STATUS_KEY,
    steps_needing_patient_groups,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.webserver import agent_pipeline_runs, research_launch_scientific, run_patient_grouping
from tests.research_agent.planning.descriptive_interval_fixtures import (
    descriptive_context,
    descriptive_plan,
)


@pytest.fixture
def grouping(monkeypatch):
    binding = SimpleNamespace(
        output_identity_column="patient_stay_id",
        mapping_sha256="a" * 64,
        authority_coordinates={
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": "synthetic/grouping/v1",
            "mapping_sha256": "a" * 64,
            "grouping_derivation": "prefix_before_:s",
            "provider_visible_values": False,
        },
    )
    monkeypatch.setattr(run_patient_grouping, "verified_patient_grouping", lambda _: binding)
    monkeypatch.setattr(run_patient_grouping, "declared_longitudinal_design", lambda _: None)
    return binding


def _study(family="descriptive_epidemiology"):
    return {"analysis_design": {
        "analysis_family": family,
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }}


def test_descriptive_planning_status_binds_verified_grouping(grouping):
    assert run_patient_grouping.planning_patient_grouping_status(
        _study(), bound=None,
    ) == ("bound", None)


def test_descriptive_runtime_states_its_bound_patient_dependence(grouping):
    dependence = run_patient_grouping.runtime_patient_dependence(
        grouping, _study()["analysis_design"],
    )
    assert dependence is not None
    assert dependence.group_source == grouping.output_identity_column
    assert dependence.group_derivation == "prefix_before_delimiter"
    assert dependence.delimiter == ":s"


def test_runtime_reads_the_study_family_beside_the_validated_variance(grouping):
    study = _study()
    validated = research_launch_scientific.validate_analysis_design_for_execution(study)
    assert validated == {"analysis_unit": "icu_stay", "variance_estimator": "model_based"}
    tree = ast.parse(Path(agent_pipeline_runs.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "run_patient_grouping.runtime_patient_dependence"]
    assert len(calls) == 1
    assert {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords} == {
        "study": "candidate_planning_study",
    }
    dependence = run_patient_grouping.runtime_patient_dependence(grouping, validated, study=study)
    assert dependence is not None
    assert dependence.group_source == grouping.output_identity_column
    assert study == _study()
    assert validated == {"analysis_unit": "icu_stay", "variance_estimator": "model_based"}


def test_the_launch_declares_the_grouping_it_bound_after_preparation(grouping, monkeypatch):
    launch_type = make_dataclass("SyntheticLaunch", [
        "study", "materialization_study", "patient_grouping", "cohort_window",
        "metadata_planning_coordinates",
    ])
    scientific = launch_type(_study(), _study(), None, (0, 24), {})
    monkeypatch.setattr(agent_pipeline_runs, "bound_export_selection_basis", lambda *_: None)
    monkeypatch.setattr(agent_pipeline_runs, "bound_export_event_time_semantics", lambda *_: {})
    monkeypatch.setattr(agent_pipeline_runs, "bound_export_selection_report", lambda *_: None)
    tree = ast.parse(Path(agent_pipeline_runs.__file__).read_text())
    launch = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "make_research_pipeline_run_runner")
    calls = [node for node in ast.walk(launch) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "research_context_declarations"]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[0]) == "replace(scientific, patient_grouping=patient_grouping)"
    declarations = agent_pipeline_runs.research_context_declarations(
        replace(scientific, patient_grouping=grouping), export_path="/synthetic-metadata",
    )
    assert declarations["id_columns"] == [grouping.output_identity_column]
    constraints = json.loads(declarations["user_preferences"]["data_constraints"])
    assert constraints["analysis_design"] == _study()["analysis_design"]
    assert constraints["verified_patient_grouping"]["coordinate"] == grouping.output_identity_column
    assert scientific.patient_grouping is None


def test_a_grouped_distribution_step_needs_patient_groups():
    plan = descriptive_plan(descriptive_context(grouped=True))
    assert steps_needing_patient_groups(plan) == (plan.steps[-1].step_id,)
    independent = descriptive_plan(descriptive_context(repeats=False))
    assert steps_needing_patient_groups(independent) == ()


@pytest.mark.parametrize("invalid", [False, True])
def test_the_counts_only_ceiling_distinguishes_absent_and_invalid_authority(invalid):
    context = descriptive_context()
    error = "patient_grouping_authority_mapping_mismatch"
    context.cohort.provenance.update({
        PATIENT_GROUPING_STATUS_KEY: "authority_invalid" if invalid else "source_has_none",
        **({PATIENT_GROUPING_AUTHORITY_ERROR_KEY: error} if invalid else {}),
    })
    plan = descriptive_plan(context)
    spec = plan.steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == "none_counts_only"
    assert spec.dependence is None
    assert spec.interval_limitation.code == "descriptive_intervals_withheld_without_patient_grouping"
    assert spec.interval_limitation.reason == (
        "repeated_stays_possible_with_invalid_patient_grouping_authority" if invalid
        else "repeated_stays_possible_without_verified_patient_grouping"
    )
    if invalid:
        assert "failed verification" in spec.interval_limitation.text
        assert context.cohort.provenance[PATIENT_GROUPING_AUTHORITY_ERROR_KEY] == error
    else:
        assert "without verified patient grouping" in spec.interval_limitation.text
    review = build_plan_scientific_review(context=context, plan=plan)
    finding = next(f for f in review.findings if f.descriptive_interval_limitation is not None)
    assert finding.descriptive_interval_limitation == spec.interval_limitation


@pytest.mark.parametrize("status", ["available_unbound", "not_carried_by_trajectory", "source_has_none"])
def test_an_unbound_ceiling_describes_the_planned_data_without_denying_source_authority(status):
    context = descriptive_context()
    context.cohort.provenance[PATIENT_GROUPING_STATUS_KEY] = status
    spec = descriptive_plan(context).steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == "none_counts_only"
    assert spec.interval_limitation.reason == "repeated_stays_possible_without_verified_patient_grouping"
    assert "the planned data may contain repeated ICU stays without verified patient grouping" in spec.interval_limitation.text


def test_a_descriptive_binding_reaches_compilation_review_and_execution(
    grouping, tmp_path, monkeypatch,
):
    from easyicu.research_agent import build_research_context
    from easyicu.research_agent.acquisition import catalog

    study = _study()
    original = json.dumps(study, sort_keys=True)
    bound, status = run_patient_grouping.planning_patient_grouping(study, bound=None)
    assert bound is grouping
    assert status == ("bound", None)
    menu = AvailableCatalog(source="canonical", concepts=[
        CatalogConcept("exposure_x"), CatalogConcept("outcome_flag"),
    ])
    monkeypatch.setattr(catalog, "build_database_capability_catalog", lambda _: menu)
    monkeypatch.setattr(catalog, "build_available_catalog", lambda _: menu)
    acquisition = agent_pipeline_runs._metadata_only_planning_acquisition(
        database="miiv", export_path="/synthetic-metadata", question="Describe a synthetic cohort.",
        output_dir=tmp_path / "catalog", patient_grouping=bound, patient_grouping_status=status,
        llm=ScriptedMockLLMClient([json.dumps({
            "selected_concepts": ["exposure_x", "outcome_flag"],
            "rationale": "Synthetic descriptive distribution.", "inclusion_exclusion": [],
        })]),
    )
    template = descriptive_context()
    projected = build_research_context(
        research_question=template.research_question, cohort=acquisition.universe_path,
        database="miiv", cohort_name="synthetic-metadata", target_outcome="outcome_flag",
        primary_exposure="exposure_x", id_columns=[bound.output_identity_column],
        user_preferences=template.user_preferences,
    )
    assert projected.cohort.n_stays == 0
    assert projected.cohort.provenance[PATIENT_GROUPING_STATUS_KEY] == "bound"
    # Compile the synthetic plan fixture against the actual projected grouping authority.
    context = template.model_copy(update={"cohort": projected.cohort})
    plan = descriptive_plan(context)
    spec = plan.steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == "wilson"
    assert spec.confidence_level == 0.95
    assert spec.interval_limitation is None
    assert spec.dependence == run_patient_grouping.runtime_patient_dependence(
        bound, study["analysis_design"],
    )
    review = build_plan_scientific_review(context=context, plan=plan)
    assert not [f for f in review.findings if f.code.startswith("DESCRIPTIVE_INTERVALS_WITHHELD")]
    context_path = tmp_path / "context.json"
    context_path.write_text(context.model_dump_json())
    needed = steps_needing_patient_groups(plan)
    assert needed == (plan.steps[-1].step_id,)
    run_patient_grouping.require_patient_groups_for_approval(plan, context_path=context_path)
    execution = run_patient_grouping.execution_patient_grouping(study, step_ids=needed)
    assert execution is bound
    assert json.dumps(study, sort_keys=True) == original


@pytest.mark.parametrize("family", ["association_study", "prediction_model"])
def test_stay_based_families_carry_identity_without_changing_inference(grouping, family):
    study = _study(family)
    bound, status = run_patient_grouping.planning_patient_grouping(study, bound=None)
    assert bound is grouping
    assert status == ("bound", None)
    assert run_patient_grouping.runtime_patient_dependence(grouping, study["analysis_design"]) is None


def test_a_verified_first_stay_restriction_needs_no_new_binding(grouping):
    bound, status = run_patient_grouping.planning_patient_grouping(
        _study(), bound=None, first_icu_stay_restricted=True,
    )
    assert bound is None
    assert status == ("available_unbound", None)
    context = descriptive_context()
    context.cohort.provenance["first_icu_stay_restriction"] = {
        "schema_version": "easyicu.first_icu_stay_restriction/1",
        "coordinate_sha256": "c" * 64,
    }
    spec = descriptive_plan(context).steps[-1].exposure_outcome_distribution_spec
    assert spec.interval_method == "wilson"
    assert spec.dependence is None
    assert spec.interval_limitation is None


def test_a_non_stay_descriptive_design_needs_no_new_binding(grouping):
    study = _study()
    study["analysis_design"]["analysis_unit"] = "patient"
    bound, status = run_patient_grouping.planning_patient_grouping(study, bound=None)
    assert bound is None
    assert status == ("available_unbound", None)
    assert run_patient_grouping.runtime_patient_dependence(grouping, study["analysis_design"]) is None


def test_descriptive_grouping_still_cannot_replace_a_long_trajectory(grouping, monkeypatch):
    monkeypatch.setattr(run_patient_grouping, "declared_longitudinal_design", lambda _: "trajectory")
    bound, status = run_patient_grouping.planning_patient_grouping(_study(), bound=None)
    assert bound is None
    assert status == ("not_carried_by_trajectory", None)


@pytest.mark.parametrize("invalid", [False, True])
def test_descriptive_binding_keeps_missing_or_invalid_authority_typed(grouping, monkeypatch, invalid):
    error = "patient_grouping_authority_mapping_mismatch"

    def resolve(_study):
        if invalid:
            raise run_patient_grouping.ResearchPipelineRunError(error, "Synthetic authority changed.")
        return None

    monkeypatch.setattr(run_patient_grouping, "verified_patient_grouping", resolve)
    bound, status = run_patient_grouping.planning_patient_grouping(_study(), bound=None)
    assert bound is None
    assert status == (("authority_invalid", error) if invalid else ("source_has_none", None))


def test_the_planning_launch_reads_the_binding_owner():
    tree = ast.parse(Path(agent_pipeline_runs.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "run_patient_grouping.planning_patient_grouping"]
    assert len(calls) == 1
    call = calls[0]
    assert ast.unparse(call.args[0]) == "candidate_planning_study"
    assert {kw.arg: ast.unparse(kw.value) for kw in call.keywords} == {
        "bound": "patient_grouping",
        "first_icu_stay_restricted": "_verified_first_icu_stay_or_none(study) is not None",
    }
