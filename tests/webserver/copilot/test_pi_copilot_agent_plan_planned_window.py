"""The Agent's landmark compiles on the window its candidate was planned on.

A candidate plan for a study with no declared window is planned on the
standing window its launch materializes.  When the Agent selects a landmark
design there, the Host compiler declares that window together with the
landmark, so the next plan and its execution read the same window.  The
researcher's own landmark choice still requires a declared window, and a
declared window is never rewritten.

A source that cannot group stays by patient leaves the repeated-stay finding
as the limitation the reviewer states (keep every stay, disclose dependence,
no paper authority); it no longer blocks the plan's other runtime findings.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import research_launch_scientific, study_contexts
from easyicu.webserver.pi_copilot.plan_decisions import (
    PlanDecisionError,
    agent_plan_configuration_available,
    compile_agent_plan_configuration,
)

_TIMING = "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"
_POPULATION = "PRIMARY_POPULATION_EXECUTION_OWNER_MISSING"
_REPEATED = "REPEATED_STAY_IDENTITY_UNAVAILABLE"


def _plan(design_id: str = "landmark_adjusted_association") -> dict:
    return {
        "design_selection": {
            "candidates": [
                {
                    "design_id": design_id,
                    "disposition": "selected",
                    "required_variables": ["lact_max", "death", "age", "sex"],
                }
            ]
        },
        "steps": [
            {
                "model_requirements": [
                    {
                        "analysis_role": "primary",
                        "exposure_source": "lact_max",
                        "outcome": "death",
                        "covariates": ["age", "sex"],
                        "covariate_rationales": {
                            "age": "Baseline age precedes the exposure window.",
                            "sex": "Baseline sex precedes the exposure window.",
                        },
                        "covariate_temporal_roles": {
                            "age": "baseline_static",
                            "sex": "baseline_static",
                        },
                    }
                ]
            }
        ],
    }


def _study(**overrides: object) -> dict:
    study = {
        "id": "planned-window-study",
        "revision": 1,
        "question": "Is the first-day maximum lactate associated with in-hospital death?",
        "data_source": {"database": "miiv", "path": "/synthetic/export"},
        "cohort": {"preset": "all_icu"},
        "confirmations": {},
    }
    study.update(overrides)
    return study


def _compile(study: dict, plan: dict, codes: tuple[str, ...], *, grouping: bool):
    return compile_agent_plan_configuration(
        study=study,
        agent_plan=plan,
        runtime_finding_codes=codes,
        patient_cluster_available=grouping,
    )


def test_the_agents_landmark_declares_the_window_its_candidate_was_planned_on() -> None:
    study = _study()

    compiled = _compile(study, _plan(), (_TIMING, _POPULATION), grouping=True)

    # The same value the launch applies to this study, not a second default.
    scoped = research_launch_scientific._neutral_materialization_scope(
        study, export_path="/synthetic/export"
    )
    assert compiled.patch["time_window"] == scoped["time_window"]
    assert compiled.patch["time_window"] == (
        research_launch_scientific.launch_materialization_window(study)
    )
    timing = next(
        spec for spec in compiled.patch["sensitivity_specs"] if spec["axis"] == "timing"
    )
    assert timing["strategy"] == "landmark"
    assert timing["landmark_hours"] == 24
    assert compiled.patch["confirmations"]["plan_timing_landmark_24h"] is True
    assert agent_plan_configuration_available(
        study=study, agent_plan=_plan(), runtime_finding_codes=(_TIMING, _POPULATION)
    )


def test_a_declared_window_is_kept_as_declared() -> None:
    study = _study(time_window={"hours": 24, "anchor": "ICU admission"})

    compiled = _compile(study, _plan(), (_TIMING,), grouping=True)

    assert "time_window" not in compiled.patch


def test_a_declared_longer_window_still_refuses_the_landmark() -> None:
    study = _study(time_window={"hours": 48, "anchor": "ICU admission"})

    with pytest.raises(PlanDecisionError) as raised:
        _compile(study, _plan(), (_TIMING,), grouping=True)

    assert raised.value.code == "agent_plan_landmark_not_compilable"
    assert raised.value.details["time_window_hours"] == 48


def test_the_window_is_declared_only_with_the_agents_landmark() -> None:
    study = _study()

    with pytest.raises(PlanDecisionError) as raised:
        _compile(
            study,
            _plan("crude_whole_stay_comparison"),
            (_TIMING,),
            grouping=True,
        )
    assert raised.value.code == "agent_plan_landmark_not_compilable"
    assert raised.value.details["time_window_hours"] == 24.0

    without_timing = _compile(study, _plan(), (_POPULATION,), grouping=True)
    assert "time_window" not in without_timing.patch


def test_without_patient_grouping_the_repeated_stay_finding_stays_a_limitation() -> None:
    study = _study()

    compiled = _compile(
        study, _plan(), (_TIMING, _POPULATION, _REPEATED), grouping=False
    )

    assert compiled.runtime_finding_codes == (_TIMING, _POPULATION)
    assert compiled.patch["analysis_design"] == {
        "analysis_family": "association_study",
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }
    # Every stay is kept without a new cohort commitment the launch would
    # have to reconcile with model-based variance.
    assert "cohort" not in compiled.patch
    assert "plan_repeated_stays_clustered" not in compiled.patch["confirmations"]
    assert study_contexts.analysis_dependence_finding({**study, **compiled.patch}) is None


def test_with_patient_grouping_the_repeated_stay_finding_still_compiles_clustering() -> None:
    compiled = _compile(
        _study(), _plan(), (_TIMING, _POPULATION, _REPEATED), grouping=True
    )

    assert compiled.runtime_finding_codes == (_TIMING, _POPULATION, _REPEATED)
    assert compiled.patch["analysis_design"]["variance_estimator"] == "cluster_robust"
    assert compiled.patch["analysis_design"]["cluster_unit"] == "patient"


def test_the_study_context_owner_accepts_the_compiled_window(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    created = study_contexts.upsert_context(
        {
            "title": "planned window",
            "question": _study()["question"],
            "data_source": {"database": "miiv", "path": "/synthetic/export"},
        }
    )
    compiled = _compile(
        created, _plan(), (_TIMING, _POPULATION, _REPEATED), grouping=False
    )

    updated = study_contexts.upsert_context(
        {"id": created["id"], **compiled.patch},
        expected_revision=created["revision"],
        require_revision=True,
        lifecycle_write=False,
    )

    assert updated["time_window"] == compiled.patch["time_window"]
    assert research_launch_scientific.launch_materialization_window(updated) == (
        research_launch_scientific.launch_materialization_window(created)
    )
