"""Synthetic validation of stochastic FCS and uncertainty pooling."""

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.model_terms import MissingDataPlan
from easyicu.research_agent.methods.missing_data import (
    ImputationRefusal,
    multiple_impute,
    rubin_pool,
)


def _plan(**updates):
    return MissingDataPlan.model_validate(
        {
            "strategy": "multiple_imputation",
            "m": 5,
            "iterations": 5,
            "seed": 31,
            "variables": [{"name": "x", "method": "pmm"}],
            **updates,
        }
    )


def _sample(n=240):
    rng = np.random.default_rng(42)
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = 2 * x + z + rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "z": z, "y": y})
    frame.loc[rng.random(n) < 0.35, "x"] = np.nan
    return frame


def test_rubin_pool_matches_hand_computation():
    result = rubin_pool([1.0, 2.0, 3.0], [0.25, 0.25, 0.25])
    assert result.estimate == 2
    assert result.within_variance == 0.25
    assert result.between_variance == 1
    assert result.total_variance == pytest.approx(19 / 12)
    assert result.df == pytest.approx(2 * (19 / 16) ** 2)
    r = 16 / 3
    assert result.fmi == pytest.approx((r + 2 / (result.df + 3)) / (r + 1))


def test_pool_uses_finite_sample_df_and_handles_zero_between_variance():
    large = rubin_pool([1, 2, 3], [0.25] * 3)
    finite = rubin_pool([1, 2, 3], [0.25] * 3, complete_df=40)
    assert 0 < finite.df < large.df
    same = rubin_pool([2, 2, 2], [0.25] * 3, complete_df=40)
    assert same.total_variance == 0.25
    assert 0 < same.df < 40


@pytest.mark.parametrize(
    "estimates,variances",
    [([1], [1]), ([1, 2], [1]), ([1, 2], [-1, 1]), ([1, np.nan], [1, 1])],
)
def test_pool_refuses_invalid_inference(estimates, variances):
    with pytest.raises(ImputationRefusal):
        rubin_pool(estimates, variances)


def test_seed_reproducibility_stochastic_chains_and_observed_values_preserved():
    frame = _sample()
    result = multiple_impute(frame, plan=_plan(), outcome="y")
    repeated = multiple_impute(frame, plan=_plan(), outcome="y")
    assert len(result.datasets) == 5
    for left, right in zip(result.datasets, repeated.datasets):
        pd.testing.assert_frame_equal(left, right)
        assert not left.isna().any().any()
        pd.testing.assert_series_equal(left.loc[frame.x.notna(), "x"], frame.x.dropna())
        assert set(left.x) <= set(frame.x.dropna())  # PMM uses observed donors.
    assert not result.datasets[0].equals(result.datasets[1])
    other_seed = multiple_impute(frame, plan=_plan(seed=32), outcome="y")
    assert not result.datasets[0].equals(other_seed.datasets[0])
    summary = result.diagnostics
    assert summary["seed"] == 31 and summary["m"] == 5
    assert summary["outcome_in_model"] == "y"
    assert summary["missingness_patterns"]
    assert len(summary["convergence"]) == 25
    assert summary["distributions"]["x"]["imputed"]["count"] == 5 * int(
        frame.x.isna().sum()
    )
    assert "datasets" not in json.dumps(summary)
    assert "row" not in summary and "index" not in summary


def test_under_mar_mi_recovers_known_coefficient_better_than_complete_case():
    rng = np.random.default_rng(170)
    n = 1800
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 2 * x + z + rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "z": z, "y": y})
    # Missingness depends on an observed outcome: MAR, not MCAR.
    frame.loc[rng.random(n) < 1 / (1 + np.exp(-0.7 * y)), "x"] = np.nan

    def coefficient(data):
        return np.linalg.lstsq(
            np.column_stack([np.ones(len(data)), data.x, data.z]), data.y, rcond=None
        )[0][1]

    complete = coefficient(frame.dropna())
    result = multiple_impute(
        frame, plan=_plan(m=12, iterations=8, donors=5), outcome="y"
    )
    estimate = np.mean([coefficient(data) for data in result.datasets])
    assert abs(estimate - 2) < 0.15
    assert abs(estimate - 2) < abs(complete - 2)


def test_binary_and_polytomous_imputation_preserve_closed_levels():
    rng = np.random.default_rng(73)
    n = 300
    frame = pd.DataFrame(
        {
            "a": rng.choice(["low", "high"], n),
            "b": rng.choice(["one", "two", "three"], n),
            "y": rng.normal(size=n),
        }
    )
    frame.loc[::4, "a"] = None
    frame.loc[::5, "b"] = None
    plan = _plan(
        m=3,
        iterations=3,
        variables=[
            {"name": "a", "method": "logistic", "levels": ["low", "high"]},
            {"name": "b", "method": "polytomous", "levels": ["one", "two", "three"]},
        ],
    )
    result = multiple_impute(frame, plan=plan, outcome="y")
    for data in result.datasets:
        assert set(data.a) == {"low", "high"}
        assert set(data.b) == {"one", "two", "three"}
    inferred = plan.model_copy(
        update={
            "variables": tuple(
                spec.model_copy(update={"levels": ()}) for spec in plan.variables
            )
        }
    )
    assert set(multiple_impute(frame, plan=inferred, outcome="y").datasets[0].b) == {
        "one",
        "two",
        "three",
    }


def test_polytomous_method_refuses_a_binary_training_vocabulary():
    frame = pd.DataFrame(
        {"x": ["a", "b", None] * 40, "y": np.random.default_rng(1).normal(size=120)}
    )
    with pytest.raises(ImputationRefusal) as caught:
        multiple_impute(
            frame,
            plan=_plan(variables=[{"name": "x", "method": "polytomous"}]),
            outcome="y",
        )
    assert caught.value.reason_code == "mi_variable_type_unsupported"


def test_prediction_transform_uses_training_donors_and_refuses_held_out_outcomes():
    training = _sample()
    result = multiple_impute(
        training, plan=_plan(m=3, iterations=3), outcome="y", fit_for_prediction=True
    )
    scoring = pd.DataFrame({"x": [np.nan, 1.0], "z": [0.0, 0.2]})
    filled = result.transform(scoring)
    assert len(filled) == 3
    for data in filled:
        assert data.x.iloc[0] in set(training.x.dropna())
        assert data.x.iloc[1] == 1
    with pytest.raises(ImputationRefusal) as caught:
        result.transform(scoring.assign(y=[-1e9, 1e9]))
    assert caught.value.reason_code == "mi_fold_outcome_leakage"


def test_scoring_only_missing_values_still_need_a_declared_conditional():
    result = multiple_impute(
        _sample(), plan=_plan(m=2, iterations=2), outcome="y", fit_for_prediction=True
    )
    with pytest.raises(ImputationRefusal) as caught:
        result.transform(pd.DataFrame({"x": [1.0], "z": [np.nan]}))
    assert caught.value.reason_code == "mi_method_missing"


def test_legacy_mice_alias_emits_a_typed_single_imputation_deprecation():
    from easyicu.research_agent.methods.missing_data import (
        LegacySingleImputationWarning,
        mice_impute,
    )

    with pytest.warns(LegacySingleImputationWarning):
        _, info = mice_impute(
            column="x", target=[1.0, 2.0, np.nan], predictors=[[0.0], [1.0], [2.0]]
        )
    assert info.to_json()["multiple_imputation"] is False
    assert (
        LegacySingleImputationWarning.reason_code == "legacy_single_imputation_not_mi"
    )


@pytest.mark.parametrize("method", ["pmm", "logistic", "polytomous"])
def test_each_conditional_draw_includes_parameter_uncertainty(method):
    from easyicu.research_agent.contracts.model_terms import ImputationVariable
    from easyicu.research_agent.methods.missing_data import (
        encodings_for,
        fit_conditional,
    )

    rng = np.random.default_rng(50)
    x = rng.normal(size=160)
    values = (
        x + rng.normal(size=160)
        if method == "pmm"
        else rng.choice(
            ["a", "b"] if method == "logistic" else ["a", "b", "c"], size=160
        )
    )
    frame = pd.DataFrame({"x": values, "y": rng.normal(size=160)})
    spec = ImputationVariable(
        name="x",
        method=method,
        levels=("a", "b", "c") if method == "polytomous" else (),
    )
    encoding = encodings_for(frame, {"x": spec})
    draws = [
        fit_conditional(
            frame,
            np.ones(len(frame), dtype=bool),
            spec,
            encoding,
            np.random.default_rng(seed),
        )
        for seed in (1, 2)
    ]
    assert not np.array_equal(draws[0].coefficients, draws[1].coefficients)


@pytest.mark.parametrize("method", ["logistic", "polytomous"])
def test_categorical_imputation_follows_observed_predictive_information(method):
    from scipy.special import softmax

    rng = np.random.default_rng(444)
    y = rng.normal(size=600)
    levels = ["low", "high"] if method == "logistic" else ["low", "middle", "high"]
    logits = (
        np.column_stack([-2 * y, 2 * y])
        if method == "logistic"
        else np.column_stack([-2 * y, np.zeros(len(y)), 2 * y])
    )
    probabilities = softmax(logits, axis=1)
    values = np.asarray(levels)[
        (rng.random(len(y))[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
    ]
    frame = pd.DataFrame({"x": values, "y": y})
    missing = rng.random(len(frame)) < 0.4
    frame.loc[missing, "x"] = None
    result = multiple_impute(
        frame,
        plan=_plan(m=3, iterations=3, variables=[{"name": "x", "method": method}]),
        outcome="y",
    )
    high = pd.concat([data.loc[missing & (y > 1), "x"] for data in result.datasets])
    low = pd.concat([data.loc[missing & (y < -1), "x"] for data in result.datasets])
    assert high.eq("high").mean() > 0.7
    assert low.eq("low").mean() > 0.7


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("all_missing", "mi_insufficient_observations"),
        ("missing_outcome", "mi_outcome_missing"),
        ("undeclared", "mi_method_missing"),
        ("infinite", "mi_nonfinite_value"),
    ],
)
def test_unsupported_input_refuses_without_complete_case_fallback(defect, reason):
    frame = _sample()
    if defect == "all_missing":
        frame.x = np.nan
    elif defect == "missing_outcome":
        frame.loc[0, "y"] = np.nan
    elif defect == "undeclared":
        frame.loc[0, "z"] = np.nan
    else:
        frame.loc[0, "z"] = np.inf
    with pytest.raises(ImputationRefusal) as caught:
        multiple_impute(frame, plan=_plan(), outcome="y")
    assert caught.value.reason_code == reason
