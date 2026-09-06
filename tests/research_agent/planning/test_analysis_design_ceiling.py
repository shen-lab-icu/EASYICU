"""One scientific ceiling is enforced at setup, launch and plan binding."""

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.contracts.analysis_design import (
    AnalysisDesignConflict,
    validate_analysis_family_ceiling,
)
from easyicu.research_agent.planning.dependence_authority import (
    DependenceAuthorityError,
    context_counts_only_authority,
)
from easyicu.webserver.study_contexts import (
    StudyContextError,
    normalize_analysis_design,
)
from easyicu.webserver.research_launch_scientific import (
    validate_analysis_design_for_execution,
)


@pytest.mark.parametrize(
    "family",
    ["prediction_model", "association_study", "survival_analysis", "causal_inference"],
)
def test_counts_only_cannot_enter_a_model_family(family):
    design = {
        "analysis_family": family,
        "analysis_unit": "icu_stay",
        "variance_estimator": "none_counts_only",
    }
    with pytest.raises(AnalysisDesignConflict):
        validate_analysis_family_ceiling(
            analysis_family=family, variance_estimator="none_counts_only"
        )
    with pytest.raises(StudyContextError):
        normalize_analysis_design(design)
    with pytest.raises(Exception) as caught:
        validate_analysis_design_for_execution({"analysis_design": design})
    assert caught.value.code == "analysis_design_counts_only_family_conflict"
    context = SimpleNamespace(
        user_preferences=SimpleNamespace(
            data_constraints=json.dumps({"analysis_design": design})
        )
    )
    with pytest.raises(DependenceAuthorityError):
        context_counts_only_authority(context)


def test_descriptive_counts_and_prediction_model_design_remain_distinct_valid_choices():
    for family, estimator in [
        ("descriptive_epidemiology", "none_counts_only"),
        ("prediction_model", "model_based"),
    ]:
        design = {
            "analysis_family": family,
            "analysis_unit": "icu_stay",
            "variance_estimator": estimator,
        }
        assert normalize_analysis_design(design) == design


def test_safe_failure_projection_keeps_lower_owner_without_private_text():
    from easyicu.webserver.agent_pipeline_runs import (
        _pipeline_failure_code,
        _safe_pipeline_typed_failure,
    )

    inner = DependenceAuthorityError(
        "private source /secret/path", code="counts_only_step_untyped"
    )
    outer = RuntimeError("pipeline failed")
    outer.__cause__ = inner
    assert _safe_pipeline_typed_failure(outer) == {
        "owner": "easyicu.planning.dependence_authority_v1",
        "reason_code": "counts_only_step_untyped",
    }
    assert _pipeline_failure_code(outer) == "research_pipeline_analysis_design_conflict"


def test_docker_diagnostic_keeps_probe_coordinates_but_not_host_text():
    from easyicu.research_agent.execution.runner import (
        ExecutionRuntimeUnavailableError,
        RunnerAvailability,
    )
    from easyicu.webserver.agent_pipeline_runs import _safe_pipeline_typed_failure

    failure = ExecutionRuntimeUnavailableError(
        RunnerAvailability(
            kind="docker",
            available=False,
            image="private/image",
            reason_code="docker_probe_failed",
            probe_phase="image_inspect",
            exit_code=125,
        )
    )
    assert _safe_pipeline_typed_failure(failure) == {
        "owner": "easyicu.execution.runtime_v1",
        "reason_code": "docker_probe_failed",
        "runner_kind": "docker",
        "probe_phase": "image_inspect",
        "exit_code": 125,
    }
