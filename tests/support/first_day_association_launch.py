"""A metadata-only first-day association launch on a synthetic study."""

from __future__ import annotations

from dataclasses import replace

from easyicu.research_agent.contracts.endpoint import EndpointSpec
from easyicu.webserver import research_pipeline_run_preparation as preparation
from easyicu.webserver import study_contexts
from easyicu.webserver.research_pipeline_run_preparation import (
    ResearchPipelineLaunchRequest,
)


def prepare_first_day_launch(monkeypatch, tmp_path):
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
