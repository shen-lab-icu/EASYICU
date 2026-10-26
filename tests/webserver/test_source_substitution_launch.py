"""The normal launch owners carry original requirements through compilation."""
from __future__ import annotations

import json

import pytest

from easyicu.databases.research_approval import require_approved_database_capabilities
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.reviewed_requirements import bind_reviewed_requirements
from easyicu.research_agent.schema import UserPreferences
from easyicu.webserver.agent_pipeline_runs import _research_user_preferences
from easyicu.webserver.research_launch_population import prepare_source_population
from tests.support.source_substitutions import _approve, _review
from tests.support.source_substitutions import context, plan


def study(database):
    return {"question": "Estimate the association of baseline weight with hospital mortality.",
        "data_source": {"database": database}, "execution_concepts": {"outcome": "death", "primary_exposure": "weight"},
        "analysis_design": {"analysis_family": "association_study", "analysis_unit": "hospital_admission",
                            "variance_estimator": "model_based"}}


@pytest.mark.parametrize("database", ["mimic", "miiv", "eicu", "hirid", "aumc", "sic"])
def test_launch_context_population_and_approval_use_the_same_source_proposal(tmp_path, database):
    original = study(database)
    candidate = prepare_source_population(original)
    assert original["execution_concepts"]["outcome"] == "death"
    assert candidate["execution_concepts"]["outcome"] == ("icu_death" if database in {"hirid", "aumc"} else "death")
    preferences = UserPreferences.model_validate(_research_user_preferences(candidate))
    source = context(database, outcome=candidate["execution_concepts"]["outcome"])
    ctx = bind_reviewed_requirements(source.model_copy(update={"user_preferences": preferences}), PipelineConfig(workdir=tmp_path))
    compiled = plan(ctx)
    requirements = json.loads(ctx.user_preferences.data_constraints)["database_study_requirements"]
    assert requirements["analysis_unit"] == "hospitalization"
    assert requirements["outcome"] == "hospital_death"
    review = _review(ctx, compiled)
    assert review.approval_allowed
    require_approved_database_capabilities(_approve(tmp_path, EvidenceStore(tmp_path), ctx, compiled, review), tmp_path)


def test_preparing_a_candidate_twice_does_not_lose_original_unit_or_outcome():
    original = study("aumc")
    original["cohort"] = {"age_min": 65}
    candidate = prepare_source_population(original)
    assert prepare_source_population(candidate) == candidate


@pytest.mark.parametrize("database,endpoint", [("eicu", "icu_death"), ("aumc", "icu_death"), ("sic", "mort_28d")])
def test_survival_preparation_retains_the_default_request_and_proposes_the_supported_endpoint(database, endpoint):
    original = study(database)
    original["analysis_design"] = {"analysis_family": "survival", "analysis_unit": "icu_stay"}
    candidate = prepare_source_population(original)
    assert original["execution_concepts"]["outcome"] == "death"
    assert candidate["execution_concepts"]["outcome"] == candidate["outcome"] == endpoint
    assert candidate["_database_study_requirements"]["outcome"] == "hospital_death"
    assert candidate["_database_study_requirements"]["censoring_unit"] == "hospitalization"
    assert candidate["_database_capability_decision"]["status"] == "substitution"
    assert prepare_source_population(candidate) == candidate
    candidate["execution_concepts"]["outcome"] = "death"
    with pytest.raises(ValueError, match="capability_substitution_binding_drift"):
        prepare_source_population(candidate)


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_launch_does_not_convert_a_refused_hospital_los_into_an_icu_endpoint(database):
    original = study(database)
    original["execution_concepts"]["outcome"] = "los_hosp"
    assert prepare_source_population(original) == original


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_normal_launch_binds_source_proposal_before_materialization(monkeypatch, tmp_path, database):
    from easyicu.webserver import research_pipeline_run_preparation as preparation

    request = preparation.ResearchPipelineLaunchRequest(
        export_path=str(tmp_path), study_context=study(database), project_root=str(tmp_path),
        provider={"provider": "openai"}, provider_environment={}, credential_source="synthetic",
        literature_search_authorized=False, plan_revision_source_run_id="",
        execution_resume_source_run_id="", development_resume_source_job_id="",
        budget_mode="planner_canary", runner_image=None)

    class ReachedMaterialization(Exception):
        pass

    def inspect_candidate(candidate, **kwargs):
        assert candidate["analysis_design"]["analysis_unit"] == "icu_stay"
        assert candidate["_database_study_requirements"]["analysis_unit"] == "hospitalization"
        assert candidate["execution_concepts"]["outcome"] == ("icu_death" if database in {"hirid", "aumc"} else "death")
        raise ReachedMaterialization

    monkeypatch.setattr(preparation, "_neutral_materialization_scope", inspect_candidate)
    with pytest.raises(ReachedMaterialization):
        preparation._prepare_scientific_launch(request)


def test_an_unregistered_launch_has_no_source_proposal():
    original = study("synthetic_export")
    assert prepare_source_population(original) == original


@pytest.mark.parametrize("coordinate", ["source", "unit", "outcome", "decision"])
def test_repeated_preparation_rejects_changed_proposal_coordinates(coordinate):
    candidate = prepare_source_population(study("aumc"))
    if coordinate == "source":
        candidate["data_source"] = {"database": "synthetic_export"}
    elif coordinate == "unit":
        candidate["analysis_design"]["analysis_unit"] = "hospitalization"
    elif coordinate == "outcome":
        candidate["execution_concepts"]["outcome"] = "death"
    else:
        candidate["_database_capability_decision"]["substitutions"][0]["reason_code"] = "synthetic_changed_reason"
    with pytest.raises(ValueError, match="capability_substitution_binding_drift"):
        prepare_source_population(candidate)


def test_launch_checks_explicit_concept_selection_before_source_preparation(monkeypatch, tmp_path):
    from easyicu.webserver import research_pipeline_run_preparation as preparation
    from easyicu.webserver import research_launch_population as owner
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    original = study("aumc")
    original["execution_concepts"]["primary_exposure"] = "sep3_sofa2"
    request = preparation.ResearchPipelineLaunchRequest(
        export_path=str(tmp_path), study_context=original, project_root=str(tmp_path),
        provider={"provider": "openai"}, provider_environment={}, credential_source="synthetic",
        literature_search_authorized=False, plan_revision_source_run_id="",
        execution_resume_source_run_id="", development_resume_source_job_id="",
        budget_mode="planner_canary", runner_image=None)
    monkeypatch.setattr(owner, "prepare_source_population", lambda *_: pytest.fail("source preparation preceded concept selection"))
    with pytest.raises(ResearchPipelineRunError) as error:
        preparation._prepare_scientific_launch(request)
    assert error.value.code == "concept_explicit_selection_required"
