"""Synthetic coordinates close through the host, without another user decision."""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
)
from easyicu.webserver.pi_copilot.plan_decisions import (
    REPEATED_STAYS_SYSTEM_DEFAULT,
    PlanDecisionError,
    compile_agent_plan_configuration,
)
from tests.support.survival_proposal import (
    AGE,
    SEX,
    proposed_survival_plan,
    survival_context,
    survival_request,
    survival_spec,
)


SURVIVAL = "SURVIVAL_LANDMARK_OWNER_NOT_SEALED"
TRAJECTORY = "TRAJECTORY_LONGITUDINAL_OWNER_NOT_SEALED"
REPEATS = "REPEATED_STAY_METHOD_NOT_DECLARED"
TIMING = "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"
CLUSTERED = {
    "analysis_unit": "icu_stay",
    "variance_estimator": "cluster_robust",
    "cluster_unit": "patient",
}


def survival_facts():
    context = survival_context()
    plan, _ = proposed_survival_plan(
        context, survival_spec(survival_request(context), [AGE, SEX])
    )
    return build_plan_scientific_review(context=context, plan=plan).facts


def compile_design(study, facts, codes):
    return compile_agent_plan_configuration(
        study=study,
        agent_plan={"steps": []},
        runtime_finding_codes=codes,
        patient_cluster_available=True,
        first_stay_coordinate_available=True,
        review_facts=facts,
    )


def test_a_survival_family_label_is_not_a_sealed_design_and_the_default_is_recorded():
    study = {
        "question": "Renal support and survival?",
        "analysis_design": {
            "analysis_family": "survival",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "time_window": {"anchor": "icu_admission", "hours": 24},
        "confirmations": {},
    }
    patch = compile_design(study, survival_facts(), (SURVIVAL, TIMING)).patch
    assert patch["analysis_design"] == {"analysis_family": "survival", **CLUSTERED}
    assert patch["confirmations"][REPEATED_STAYS_SYSTEM_DEFAULT] is True
    [timing] = patch["sensitivity_specs"]
    assert (
        timing["landmark_hours"] == 24 and timing["require_alive_at_landmark"] is True
    )
    assert "cohort" not in patch and "question" not in patch


def test_a_fixed_window_trajectory_compiles_beside_its_repeated_stay_finding():
    study = {
        "question": "Fixed-window creatinine trajectory phenotypes",
        "analysis_design": CLUSTERED,
        "confirmations": {
            REPEATED_STAYS_SYSTEM_DEFAULT: True,
            "plan_repeated_stays_clustered": True,
        },
    }
    facts = {
        "trajectory_representation": {
            "executable": True,
            "proposed_coordinates": ["crea"],
            "trajectory_window": {
                "executable": True,
                "window_start_hours": 0,
                "window_end_hours": 48,
                "grid_width_hours": 12,
            },
        }
    }
    compiled = compile_design(study, facts, (TRAJECTORY, REPEATS))
    assert set(compiled.runtime_finding_codes) == {TRAJECTORY, REPEATS}
    patch = compiled.patch
    assert patch["analysis_design"] == {
        "analysis_family": "trajectory_clustering",
        **CLUSTERED,
    }
    assert patch["trajectory_design"]["window_end_hours"] == 48
    assert patch["confirmations"][REPEATED_STAYS_SYSTEM_DEFAULT] is True
    assert "cohort" not in patch


def test_non_closing_landmark_coordinates_still_block_with_a_clear_message():
    facts = survival_facts()
    facts["landmark_survival_suite"]["executable"] = False
    with pytest.raises(
        PlanDecisionError, match="no executable landmark survival coordinates"
    ):
        compile_design({"confirmations": {}}, facts, (SURVIVAL,))


@pytest.mark.parametrize("missing", [{"time_windows": []}, {"target_outcome": None}])
def test_an_incomplete_landmark_has_one_coordinate_question(missing):
    context = survival_context()
    plan, _ = proposed_survival_plan(
        context, survival_spec(survival_request(context), [AGE, SEX])
    )
    review = build_plan_scientific_review(
        context=context.model_copy(update=missing), plan=plan
    )
    questions = [
        finding for finding in review.findings if finding.requires_user_authorization
    ]
    assert [finding.code for finding in questions] == [SURVIVAL]
    assert questions[0].authorization_question
    assert review.approval_allowed is False


def test_a_user_first_stay_rule_is_never_replaced_by_the_default():
    study = {
        "cohort": {"exclude_readmissions": True},
        "confirmations": {
            REPEATED_STAYS_SYSTEM_DEFAULT: False,
            "plan_repeated_stays_first": True,
        },
    }
    patch = compile_design(study, survival_facts(), (SURVIVAL,)).patch
    assert patch["analysis_design"]["variance_estimator"] == "model_based"
    assert patch["confirmations"][REPEATED_STAYS_SYSTEM_DEFAULT] is False
    assert patch["confirmations"]["plan_repeated_stays_first"] is True
    assert "cohort" not in patch


def test_an_explicit_user_timing_rule_is_not_replaced_by_a_proposal():
    study = {
        "confirmations": {"plan_timing_time_varying": True},
        "sensitivity_specs": [
            {
                "spec_id": "user_time_varying",
                "axis": "timing",
                "strategy": "time_varying",
            }
        ],
    }
    with pytest.raises(PlanDecisionError, match="timing"):
        compile_design(study, survival_facts(), (SURVIVAL, TIMING))


def test_a_recorded_landmark_is_completed_without_replacing_its_provenance():
    study = {
        "analysis_design": {"analysis_family": "survival"},
        "confirmations": {"plan_timing_landmark_24h": True},
        "sensitivity_specs": [
            {
                "spec_id": "recorded_landmark",
                "axis": "timing",
                "strategy": "landmark",
                "landmark_hours": 24,
                "require_alive_at_landmark": True,
                "exclude_negative_event_times": True,
                "observation_duration_variable": "followup_days_90d",
                "observation_duration_unit": "days",
            }
        ],
    }
    patch = compile_design(study, survival_facts(), (SURVIVAL, TIMING)).patch
    assert patch["sensitivity_specs"][0]["spec_id"] == "recorded_landmark"
    assert patch["analysis_design"]["variance_estimator"] == "cluster_robust"


def test_a_declared_non_stay_unit_is_refused_without_a_default_rewrite():
    study = {
        "analysis_design": {
            "analysis_family": "survival",
            "analysis_unit": "hospital_admission",
        }
    }
    with pytest.raises(PlanDecisionError, match="ICU-stay analysis"):
        compile_design(study, survival_facts(), (SURVIVAL,))
    assert study["analysis_design"]["analysis_unit"] == "hospital_admission"


def test_a_user_clustered_rule_retains_its_provenance_and_needs_source_grouping():
    study = {
        "analysis_design": {"analysis_family": "survival", **CLUSTERED},
        "confirmations": {
            REPEATED_STAYS_SYSTEM_DEFAULT: False,
            "plan_repeated_stays_clustered": True,
        },
    }
    compiled = compile_design(study, survival_facts(), (SURVIVAL, REPEATS))
    assert compiled.patch["analysis_design"] == study["analysis_design"]
    assert compiled.patch["confirmations"][REPEATED_STAYS_SYSTEM_DEFAULT] is False
    assert compiled.patch["confirmations"]["plan_repeated_stays_clustered"] is True
    with pytest.raises(PlanDecisionError, match="verified patient grouping"):
        compile_agent_plan_configuration(
            study=study,
            agent_plan={"steps": []},
            runtime_finding_codes=(SURVIVAL, REPEATS),
            patient_cluster_available=False,
            review_facts=survival_facts(),
        )
    assert study["confirmations"][REPEATED_STAYS_SYSTEM_DEFAULT] is False


def test_a_recorded_bootstrap_rule_is_refused_without_becoming_the_default():
    study = {
        "analysis_design": {
            "analysis_family": "survival",
            "variance_estimator": "bootstrap",
        },
        "confirmations": {REPEATED_STAYS_SYSTEM_DEFAULT: False},
    }
    with pytest.raises(PlanDecisionError, match="recorded repeated-stay rule"):
        compile_design(study, survival_facts(), (SURVIVAL, REPEATS))
    assert study["analysis_design"]["variance_estimator"] == "bootstrap"
    assert study["confirmations"][REPEATED_STAYS_SYSTEM_DEFAULT] is False
