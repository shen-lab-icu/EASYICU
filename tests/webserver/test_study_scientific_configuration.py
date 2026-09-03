from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.webserver.study_scientific_configuration import (
    ScientificConfiguration,
    ScientificConfigurationError,
    SetupFacts,
)


def test_execution_concepts_override_display_labels() -> None:
    configuration = ScientificConfiguration.inspect(
        {
            "outcome": "院内死亡",
            "primary_exposure": "最高乳酸",
            "execution_concepts": {
                "outcome": "death",
                "primary_exposure": "lact",
                "primary_exposure_aggregation": "max",
            },
        }
    )

    assert configuration.target_outcome() == "death"
    assert configuration.primary_exposure() == "lact"
    assert configuration.executable_target_outcome() == "death"
    assert configuration.executable_primary_exposure() == "lact"
    assert configuration.primary_exposure_aggregation() == "max"


def test_display_labels_do_not_become_executable_source_concepts() -> None:
    configuration = ScientificConfiguration.inspect(
        {
            "outcome": "院内死亡",
            "primary_exposure": "入 ICU 后前 24 小时峰值乳酸",
            "execution_concepts": {"covariates": ["age", "sex"]},
        }
    )

    assert configuration.target_outcome() == "院内死亡"
    assert configuration.primary_exposure() == "入 ICU 后前 24 小时峰值乳酸"
    assert configuration.executable_target_outcome() is None
    assert configuration.executable_primary_exposure() is None


def test_exact_covariate_binding_mismatch_fails_closed() -> None:
    configuration = ScientificConfiguration.inspect(
        {
            "covariate_selection": "exact",
            "covariates": ["age", "sex"],
            "execution_concepts": {"covariates": ["age"]},
        }
    )

    with pytest.raises(ScientificConfigurationError) as raised:
        configuration.covariates()

    assert raised.value.code == (
        "research_pipeline_covariate_execution_binding_mismatch"
    )


def test_setup_assessment_preserves_order_and_planning_boundary() -> None:
    assessment = ScientificConfiguration.inspect(
        {"id": "study-1", "revision": 1}
    ).assess_setup(SetupFacts(active_export_present=False, eligibility_stated=False))

    assert assessment.missing_fields[:5] == (
        "question",
        "data_source",
        "cohort",
        "cohort_eligibility",
        "outcome",
    )
    assert assessment.planning_prerequisites_missing == (
        "question",
        "data_source",
    )


def test_decision_state_and_patch_helpers_have_one_owner() -> None:
    configuration = ScientificConfiguration.inspect(
        {
            "confirmations": {"plan_adjustment_set_confirmed": True},
            "sensitivity_specs": [
                {"spec_id": "old", "axis": "timing", "strategy": "landmark"},
                {
                    "spec_id": "keep",
                    "axis": "cohort",
                    "strategy": "alternate_eligibility",
                },
            ],
        }
    )

    assert configuration.decision_is_resolved("ADJUSTMENT_SET_NOT_USER_CONFIRMED")
    assert configuration.merge_confirmations(export_format=True) == {
        "plan_adjustment_set_confirmed": True,
        "export_format": True,
    }
    assert configuration.replace_sensitivity(
        axis="timing",
        replacement={
            "spec_id": "new",
            "axis": "timing",
            "strategy": "time_varying",
        },
    ) == [
        {"spec_id": "keep", "axis": "cohort", "strategy": "alternate_eligibility"},
        {"spec_id": "new", "axis": "timing", "strategy": "time_varying"},
    ]


def test_repeated_stay_decision_requires_matching_typed_design() -> None:
    stale = ScientificConfiguration.inspect(
        {
            "cohort": {"exclude_readmissions": False},
            "analysis_design": {
                "analysis_unit": "icu_stay",
                "variance_estimator": "model_based",
            },
            "confirmations": {"plan_repeated_stays_clustered": True},
        }
    )
    clustered = ScientificConfiguration.inspect(
        {
            "cohort": {"exclude_readmissions": False},
            "analysis_design": {
                "analysis_unit": "icu_stay",
                "variance_estimator": "cluster_robust",
                "cluster_unit": "patient",
            },
            "confirmations": {"plan_repeated_stays_clustered": True},
        }
    )

    assert not stale.decision_is_resolved("REPEATED_STAY_IDENTITY_UNAVAILABLE")
    assert clustered.decision_is_resolved("REPEATED_STAY_IDENTITY_UNAVAILABLE")


def test_timing_decision_requires_one_matching_typed_route() -> None:
    time_varying = ScientificConfiguration.inspect(
        {
            "confirmations": {"plan_timing_time_varying": True},
            "sensitivity_specs": [
                {
                    "spec_id": "time_varying_exposure",
                    "axis": "timing",
                    "strategy": "time_varying",
                }
            ],
        }
    )
    conflicting = ScientificConfiguration.inspect(
        {
            "confirmations": {
                "plan_timing_landmark_24h": True,
                "plan_timing_time_varying": True,
            },
            "sensitivity_specs": [
                {
                    "spec_id": "landmark_24h",
                    "axis": "timing",
                    "strategy": "landmark",
                }
            ],
        }
    )

    assert time_varying.decision_is_resolved(
        "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"
    )
    assert not conflicting.decision_is_resolved(
        "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"
    )


def test_owner_has_no_adapter_or_runtime_dependency() -> None:
    source = Path("src/easyicu/webserver/study_scientific_configuration.py").read_text(
        encoding="utf-8"
    )

    assert "pi_copilot" not in source
    assert "agent_pipeline_runs" not in source
    assert "study_contexts" not in source
    assert "filesystem" not in source
