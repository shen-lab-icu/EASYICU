"""Published design examples; inputs are assumptions, never patient records."""

import math

import pytest


def test_published_power_and_riley_examples():
    from easyicu.research_agent.skill_packages import (
        prediction_binary, survival_events, two_means, two_proportions,
    )

    # Cohen d=0.5, two-sided alpha=.05 and power=.8: 64 in each arm.
    means = two_means(difference=5, standard_deviation=10)
    assert means["group_sizes"] == [64, 64]
    assert means["achieved_power"] >= .8
    # Cohen arcsine effect for probabilities .5 and .65: 170 per arm.
    assert two_proportions(p1=.5, p2=.65)["group_sizes"] == [170, 170]
    # Schoenfeld's equal-allocation logrank approximation, HR=.7: 247 events.
    assert survival_events(hazard_ratio=.7)["required_events"] == 247
    # Riley et al. (2019), binary worked example also in pmsampsize:
    # 24 candidate parameters, Cox-Snell R²=.051, prevalence=.174.
    riley = prediction_binary(parameters=24, cox_snell_r2=.051, prevalence=.174)
    assert riley["minimum_n"] == 4115
    assert riley["criteria_n"] == {
        "shrinkage": 4115, "r2_optimism": 763, "intercept_precision": 221,
    }
    assert riley["expected_events"] == pytest.approx(716.01)
    assert riley["claim_ceiling"] == "analysis_only"
    assert riley["manuscript_findings"] is False


@pytest.mark.parametrize("value", [0, 1, -1, math.nan, math.inf, True, "0.5"])
def test_invalid_probabilities_stop_with_stable_code(value):
    from easyicu.research_agent.skill_packages import PlanningInputError, two_proportions

    with pytest.raises(PlanningInputError, match="planning.invalid_probability"):
        two_proportions(p1=value, p2=.65)


@pytest.mark.parametrize("method,kwargs,code", [
    ("two_means", {"difference": 0, "standard_deviation": 10}, "planning.null_effect"),
    ("two_means", {"difference": float("inf"), "standard_deviation": 10}, "planning.invalid_number"),
    ("two_means", {"difference": 5, "standard_deviation": 0}, "planning.positive_required"),
    ("two_means", {"difference": 5, "standard_deviation": 10, "allocation_ratio": 0}, "planning.positive_required"),
    ("two_proportions", {"p1": .5, "p2": .5}, "planning.null_effect"),
    ("two_proportions", {"p1": .5, "p2": .6, "alpha": .1, "power": .05}, "planning.power_not_above_alpha"),
    ("survival_events", {"hazard_ratio": 1}, "planning.null_effect"),
    ("survival_events", {"hazard_ratio": -1}, "planning.positive_required"),
    ("survival_events", {"hazard_ratio": .7, "allocation_fraction": 1}, "planning.invalid_probability"),
    ("survival_events", {"hazard_ratio": .7, "event_fraction": 0}, "planning.invalid_probability"),
    ("prediction_binary", {"parameters": 0, "cox_snell_r2": .05, "prevalence": .1}, "planning.invalid_parameter_count"),
    ("prediction_binary", {"parameters": True, "cox_snell_r2": .05, "prevalence": .1}, "planning.invalid_parameter_count"),
    ("prediction_binary", {"parameters": 2.5, "cox_snell_r2": .05, "prevalence": .1}, "planning.invalid_parameter_count"),
    ("prediction_binary", {"parameters": 5, "cox_snell_r2": .7, "prevalence": .01}, "planning.incompatible_r2"),
    ("prediction_binary", {"parameters": 5, "cox_snell_r2": .5, "prevalence": .5, "shrinkage": .4}, "planning.incompatible_r2"),
    ("survival_events", {"hazard_ratio": .7, "basis": "patient_rows"}, "planning.assumption_source_required"),
    ("survival_events", {"hazard_ratio": .7, "basis": "governed_aggregate"}, "planning.aggregate_receipt_required"),
    ("survival_events", {"hazard_ratio": .7, "aggregate_receipt": "receipt"}, "planning.assumption_source_mismatch"),
])
def test_invalid_designs_fail_closed(method, kwargs, code):
    from easyicu.research_agent import skill_packages as sample_size

    with pytest.raises(sample_size.PlanningInputError, match=code):
        getattr(sample_size, method)(**kwargs)


def test_unequal_allocation_direction_and_event_fraction():
    from easyicu.research_agent.skill_packages import two_means, survival_events

    args = dict(standard_deviation=10, allocation_ratio=2)
    positive = two_means(difference=5, **args)
    assert positive["group_sizes"][1] == 2 * positive["group_sizes"][0]
    assert positive["achieved_power"] >= .8
    assert two_means(difference=-5, **args)["group_sizes"] == positive["group_sizes"]
    result = survival_events(hazard_ratio=.7, event_fraction=.4,
                             basis="governed_aggregate", aggregate_receipt="aggregate:demo")
    assert result["total_n"] == 618
    assert result["aggregate_receipt"] == "aggregate:demo"
    assert survival_events(hazard_ratio=.7)["total_n"] is None
    assert survival_events(hazard_ratio=.7, allocation_fraction=.25)["required_events"] == 330


def test_all_three_riley_criteria_can_drive_the_requirement():
    from easyicu.research_agent.skill_packages import prediction_binary

    precision = prediction_binary(parameters=1, cox_snell_r2=.2, prevalence=.5)
    assert precision["minimum_n"] == precision["criteria_n"]["intercept_precision"] == 385
    optimism = prediction_binary(parameters=20, cox_snell_r2=.6, prevalence=.5, prevalence_margin=.2)
    assert optimism["minimum_n"] == optimism["criteria_n"]["r2_optimism"]
    assert optimism["criteria_n"]["r2_optimism"] > optimism["criteria_n"]["shrinkage"]


def test_calculations_read_no_patient_data_or_files(monkeypatch):
    import builtins
    import io
    from pathlib import Path

    import pandas as pd
    # Load the established numerical packages before forbidding any file IO.
    from statsmodels.stats.power import NormalIndPower, TTestIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    from easyicu.research_agent import skill_packages as sample_size

    assert NormalIndPower and TTestIndPower and proportion_effectsize

    def forbidden(*args, **kwargs):
        pytest.fail("planning calculations attempted IO")

    for owner, attribute in [(builtins, "open"), (io, "open"), (Path, "open"),
                             (pd, "read_parquet"), (pd, "read_csv")]:
        monkeypatch.setattr(owner, attribute, forbidden)
    results = [sample_size.two_means(difference=5, standard_deviation=10),
               sample_size.two_proportions(p1=.5, p2=.65),
               sample_size.survival_events(hazard_ratio=.7),
               sample_size.prediction_binary(parameters=24, cox_snell_r2=.051, prevalence=.174)]
    assert all(row["patient_data_read"] is False for row in results)
    assert all(row["claim_ceiling"] == "analysis_only" and row["manuscript_findings"] is False for row in results)


def test_solver_failure_is_not_returned_as_a_planning_number(monkeypatch):
    from statsmodels.stats.power import TTestIndPower
    from easyicu.research_agent.skill_packages import PlanningInputError, two_means

    monkeypatch.setattr(TTestIndPower, "solve_power", lambda *args, **kwargs: float("nan"))
    with pytest.raises(PlanningInputError, match="planning.numerical_failure"):
        two_means(difference=5, standard_deviation=10)
    monkeypatch.setattr(TTestIndPower, "solve_power", lambda *args, **kwargs: 2.)
    with pytest.raises(PlanningInputError, match="planning.numerical_failure"):
        two_means(difference=5, standard_deviation=10)
