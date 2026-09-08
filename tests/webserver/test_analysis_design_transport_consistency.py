"""Planning and launch cannot resolve contradictory study scope differently."""

from copy import deepcopy
import json

import pytest

from easyicu.webserver.agent_pipeline_runs import _research_user_preferences
from easyicu.webserver.research_launch_scientific import validate_analysis_design_for_execution
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError


@pytest.mark.parametrize("design", [
    {"analysis_family": "prediction_model", "analysis_unit": "icu_stay", "variance_estimator": "model_based"},
    {"analysis_family": "association_study", "analysis_unit": "icu_stay", "variance_estimator": "cluster_robust", "cluster_unit": "patient"},
    {"analysis_unit": "icu_stay", "variance_estimator": "cluster_robust", "cluster_unit": "patient"},
])
def test_descriptive_confirmation_cannot_silently_replace_an_explicit_design(design):
    study = {"analysis_design": design, "confirmations": {"plan_timing_descriptive_only": True}}
    before = deepcopy(study)
    for consumer in (_research_user_preferences, validate_analysis_design_for_execution):
        with pytest.raises(ResearchPipelineRunError) as caught:
            consumer(study)
        assert caught.value.code == "research_pipeline_descriptive_confirmation_conflict"
    assert study == before


@pytest.mark.parametrize("design", [{}, {
    "analysis_unit": "icu_stay", "variance_estimator": "none_counts_only",
}])
def test_legacy_descriptive_confirmation_has_one_transport_and_launch_meaning(design):
    study = {"analysis_design": design, "confirmations": {"plan_timing_descriptive_only": True}}
    before = deepcopy(study)
    preferences = _research_user_preferences(study)
    transported = json.loads(preferences["data_constraints"])["analysis_design"]
    launched = validate_analysis_design_for_execution(study)
    assert transported == {
        "analysis_family": "descriptive_epidemiology", "analysis_unit": "icu_stay",
        "variance_estimator": "none_counts_only",
    }
    assert launched == {key: transported[key] for key in ("analysis_unit", "variance_estimator")}
    assert study == before


def test_legacy_prediction_counts_only_conflict_stops_before_planning():
    study = {"analysis_design": {
        "analysis_family": "prediction_model", "analysis_unit": "icu_stay",
        "variance_estimator": "none_counts_only",
    }}
    before = deepcopy(study)
    for consumer in (_research_user_preferences, validate_analysis_design_for_execution):
        with pytest.raises(ResearchPipelineRunError) as caught:
            consumer(study)
        assert caught.value.code == "analysis_design_counts_only_family_conflict"
    assert study == before
