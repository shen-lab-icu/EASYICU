from __future__ import annotations

import pytest

from easyicu.webserver.pi_copilot.plan_decisions import (
    PlanDecisionError,
    compile_plan_decision,
)


def _plan() -> dict:
    return {
        "design_selection": {
            "candidates": [
                {
                    "design_id": "landmark_adjusted_association",
                    "disposition": "selected",
                    "required_variables": [
                        "lact_max",
                        "death_max",
                        "death_first_time",
                        "death_last_time",
                        "age",
                        "sex",
                    ],
                }
            ]
        },
        "steps": [
            {
                "model_requirements": [
                    {
                        "analysis_role": "primary",
                        "exposure_source": "lact_max",
                        "outcome": "death_max",
                        "covariates": ["age", "sex"],
                    }
                ]
            }
        ],
    }


def _study() -> dict:
    return {
        "cohort": {"preset": "all_icu"},
        "confirmations": {"feature_time_window": True},
        "sensitivity_specs": [
            {
                "spec_id": "complete_case",
                "axis": "missing",
                "strategy": "complete_case",
            }
        ],
    }


def test_landmark_choice_compiles_one_complete_typed_update() -> None:
    compiled = compile_plan_decision(
        decision_code="POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
        option_id="landmark_24h",
        study=_study(),
        agent_plan=_plan(),
    )

    assert compiled.next_action == "replan"
    assert compiled.patch["outcome"] == "院内死亡（第 24 小时起至出院）"
    assert compiled.patch["primary_exposure"] == "入 ICU 后 0–24 小时最高乳酸水平"
    assert compiled.patch["execution_concepts"] == {
        "outcome": "death",
        "primary_exposure": "lact",
        "primary_exposure_aggregation": "max",
    }
    specs = {item["axis"]: item for item in compiled.patch["sensitivity_specs"]}
    assert specs["missing"]["spec_id"] == "complete_case"
    assert specs["timing"] == {
        "spec_id": "landmark_24h",
        "axis": "timing",
        "strategy": "landmark",
        "execution_variables": ["death_time", "los_icu"],
        "landmark_hours": 24,
        "require_alive_at_landmark": True,
        "exclude_negative_event_times": True,
        "event_time_variable": "death_time",
        "observation_duration_variable": "los_icu",
        "observation_duration_unit": "days",
    }
    assert compiled.patch["confirmations"]["plan_timing_landmark_24h"] is True


def test_landmark_choice_fails_closed_without_one_selected_design() -> None:
    plan = _plan()
    plan["design_selection"]["candidates"] = []

    with pytest.raises(PlanDecisionError) as raised:
        compile_plan_decision(
            decision_code="POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
            option_id="landmark_24h",
            study=_study(),
            agent_plan=plan,
        )

    assert raised.value.code == "plan_decision_design_ambiguous"


def test_unknown_plan_choice_never_falls_back_to_free_text() -> None:
    with pytest.raises(PlanDecisionError) as raised:
        compile_plan_decision(
            decision_code="POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
            option_id="invented_option",
            study=_study(),
            agent_plan=_plan(),
        )

    assert raised.value.code == "plan_decision_option_unknown"


def test_adjustment_choice_saves_exact_typed_roster_once() -> None:
    study = _study()
    study["execution_concepts"] = {
        "outcome": "death",
        "primary_exposure": "lact",
        "primary_exposure_aggregation": "max",
    }

    compiled = compile_plan_decision(
        decision_code="ADJUSTMENT_SET_NOT_USER_CONFIRMED",
        option_id="accept_proposed_adjustment",
        study=study,
        agent_plan=_plan(),
    )

    assert compiled.next_action == "replan"
    assert compiled.patch["covariates"] == ["age", "sex"]
    assert compiled.patch["covariate_selection"] == "exact"
    assert compiled.patch["covariate_temporal_roles"] == {
        "age": "baseline_static",
        "sex": "baseline_static",
    }
    assert compiled.patch["execution_concepts"]["covariates"] == ["age", "sex"]
    assert compiled.patch["confirmations"]["plan_adjustment_set_confirmed"] is True


def test_repeated_stay_choice_saves_clustered_analysis_design() -> None:
    compiled = compile_plan_decision(
        decision_code="REPEATED_STAY_IDENTITY_UNAVAILABLE",
        option_id="all_icu_stays_clustered",
        study=_study(),
        agent_plan=_plan(),
    )

    assert compiled.next_action == "continue_review"
    assert compiled.patch["analysis_design"] == {
        "analysis_unit": "icu_stay",
        "variance_estimator": "cluster_robust",
        "cluster_unit": "patient",
    }
    assert compiled.patch["sensitivity_specs"][-1]["strategy"] == "cluster_robust"
    assert compiled.patch["confirmations"]["plan_repeated_stays_clustered"] is True


def test_keep_sensitivities_repairs_landmark_execution_coordinates() -> None:
    study = _study()
    study["sensitivity_specs"].append(
        {
            "spec_id": "repeated_stays_cluster_robust",
            "axis": "repeated_stays",
            "strategy": "cluster_robust",
        }
    )

    compiled = compile_plan_decision(
        decision_code="REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY",
        option_id="keep_executable_sensitivities",
        study=study,
        agent_plan=_plan(),
    )

    specs = {item["axis"]: item for item in compiled.patch["sensitivity_specs"]}
    assert specs["timing"]["event_time_variable"] == "death_time"
    assert specs["timing"]["observation_duration_variable"] == "los_icu"
    assert specs["timing"]["observation_duration_unit"] == "days"
    assert specs["repeated_stays"]["strategy"] == "cluster_robust"
    assert compiled.patch["confirmations"][
        "plan_required_sensitivities_executable"
    ] is True
