"""B2 (11-19 UX walk): the host never compiles or offers a landmark it cannot launch.

A 24-hour landmark on 28-day mortality named ``mort_28d_followup_time_hours``,
a column no owner materializes; the compiled study was then refused at launch
with no way on.  Only in-hospital death has a host follow-up axis, so a
landmark on another outcome routes to the timing decision without a
landmark option (descriptive, or re-extract for a time-varying model).
"""

from __future__ import annotations

import pytest

from easyicu.webserver.pi_copilot.plan_decisions import (
    PlanDecisionError,
    agent_plan_configuration_available,
    compile_agent_plan_configuration,
    compile_plan_decision,
    landmark_followup_axis_available,
    plan_decision_context,
)

_TIMING = "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"


def _plan(outcome: str) -> dict:
    return {
        "design_selection": {"candidates": [{
            "design_id": "landmark_adjusted_association", "disposition": "selected",
            "required_variables": ["lact_max", outcome, "age", "sex"],
        }]},
        "steps": [{"model_requirements": [{
            "analysis_role": "primary", "exposure_source": "lact_max", "outcome": outcome,
            "covariates": ["age", "sex"],
            "covariate_rationales": {"age": "Baseline age precedes the window.", "sex": "Baseline sex precedes the window."},
            "covariate_temporal_roles": {"age": "baseline_static", "sex": "baseline_static"},
        }]}],
    }


def _study() -> dict:
    return {
        "id": "b2-study", "revision": 1,
        "question": "Is the first-day maximum lactate associated with 28-day mortality?",
        "data_source": {"database": "miiv", "path": "/synthetic/export"},
        "cohort": {"preset": "all_icu"}, "confirmations": {},
        "time_window": {"hours": 24, "anchor": "ICU admission"},
    }


def test_only_in_hospital_death_has_a_host_follow_up_axis() -> None:
    assert landmark_followup_axis_available("death")
    assert not landmark_followup_axis_available("mort_28d")


def test_the_agent_landmark_on_28_day_mortality_is_not_compiled() -> None:
    with pytest.raises(PlanDecisionError) as raised:
        compile_agent_plan_configuration(
            study=_study(), agent_plan=_plan("mort_28d"), runtime_finding_codes=(_TIMING,),
            patient_cluster_available=True,
        )
    assert raised.value.code == "agent_plan_landmark_not_compilable"
    assert not agent_plan_configuration_available(
        study=_study(), agent_plan=_plan("mort_28d"), runtime_finding_codes=(_TIMING,))
    # In-hospital death keeps the host landmark.
    compiled = compile_agent_plan_configuration(
        study=_study(), agent_plan=_plan("death"), runtime_finding_codes=(_TIMING,),
        patient_cluster_available=True,
    )
    assert any(spec["strategy"] == "landmark" for spec in compiled.patch["sensitivity_specs"])


@pytest.mark.parametrize(("outcome", "profile"), [("death", "fixed_24h_landmark"), ("mort_28d", "unspecified_post_baseline")])
def test_the_timing_card_offers_a_landmark_only_where_it_can_run(outcome: str, profile: str) -> None:
    assert plan_decision_context(_plan(outcome), _TIMING, _study())["timing_profile"] == profile


def test_a_forced_landmark_choice_on_28_day_mortality_is_refused() -> None:
    with pytest.raises(PlanDecisionError) as raised:
        compile_plan_decision(decision_code=_TIMING, option_id="landmark_24h",
                              study=_study(), agent_plan=_plan("mort_28d"))
    assert raised.value.code == "plan_decision_option_not_applicable"
    assert raised.value.details["outcome"] == "mort_28d"


@pytest.mark.parametrize("option", ["descriptive_only", "time_varying_reextract"])
def test_the_offered_choices_on_28_day_mortality_compile(option: str) -> None:
    compiled = compile_plan_decision(decision_code=_TIMING, option_id=option,
                                     study=_study(), agent_plan=_plan("mort_28d"))
    assert not any(spec.get("strategy") == "landmark" for spec in compiled.patch.get("sensitivity_specs") or [])


def _outcome_catalogue() -> list[str]:
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL

    return sorted({"death", *CONCEPT_GROUPS_INTERNAL.get("outcome", ())})


@pytest.mark.parametrize("outcome", _outcome_catalogue())
def test_no_outcome_compiles_a_landmark_data_readiness_would_refuse(outcome: str) -> None:
    """S1414(1): every landmark producer goes through one spec owner, for every outcome."""
    from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
    from easyicu.webserver.pi_copilot.plan_decisions import landmark_timing_specification

    if not landmark_followup_axis_available(outcome):
        with pytest.raises(PlanDecisionError):
            landmark_timing_specification(outcome=outcome, landmark_hours=24, spec_id="probe")
        return
    spec = PrespecifiedSensitivitySpec.model_validate(
        landmark_timing_specification(outcome=outcome, landmark_hours=24, spec_id="probe"))
    # Nothing left for the source catalog to resolve: the launch gate has no column to refuse.
    assert spec.source_materialization_variables == ()
