"""Patient bootstrap validation, using small, reproducible synthetic cohorts only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from easyicu.research_agent.methods import prediction_validation as metrics
from easyicu.research_agent.execution.runners import prediction_model_executor as executor


def _simulate(seed, n, p=16):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    y = rng.binomial(1, expit(0.7 * x[:, 0] - 0.5 * x[:, 1]))
    return pd.DataFrame(x, columns=[f"x{i}" for i in range(p)]), pd.Series(y)


def test_optimism_correction_tracks_independent_performance():
    frame, y = _simulate(31, 180)
    external, external_y = _simulate(52, 4000)
    features = tuple(frame.columns)
    fits = []

    def refit(rows):
        model = executor._model_pipeline(frame.iloc[rows], features)
        model.fit(frame.iloc[rows], y.iloc[rows])
        fits.append(model)
        return model.predict_proba(frame)[:, 1]

    result = metrics.bootstrap_optimism_validation(
        outcomes=y, groups=np.arange(len(y)), refit_predict=refit,
        n_replicates=100, seed=712,
    )
    auc = result["metrics"]["auroc"]
    independent = roc_auc_score(external_y, fits[0].predict_proba(external)[:, 1])
    assert len(fits) == 101
    assert result["failed_replicates"] == 0
    assert auc["corrected"] < auc["apparent"] - 0.03
    assert abs(auc["corrected"] - independent) < 0.045
    assert abs(auc["corrected"] - independent) < abs(auc["apparent"] - independent)
    for name, values in result["metrics"].items():
        assert values["corrected"] == pytest.approx(values["apparent"] - values["optimism"])
        deltas = [r["resample"][name] - r["original"][name] for r in result["replicates"]]
        assert values["optimism"] == pytest.approx(np.mean(deltas))
    assert result["metrics"]["brier_score"]["corrected"] > result["metrics"]["brier_score"]["apparent"]


def test_each_draw_keeps_all_stays_and_refits_preprocessing():
    frame, y = _simulate(44, 90, 3)
    groups = np.repeat(np.arange(30), 3)
    frame.loc[::7, "x0"] = np.nan
    frame["category"] = np.where(np.arange(len(y)) % 2, "one", "two")
    features = tuple(frame.columns)
    samples, medians = [], []

    def refit(rows):
        samples.append(rows.copy())
        counts = np.bincount(rows, minlength=len(y)).reshape(-1, 3)
        assert np.all(counts == counts[:, :1])
        assert counts[:, 0].sum() == 30
        model = executor._model_pipeline(frame.iloc[rows], features)
        model.fit(frame.iloc[rows], y.iloc[rows])
        imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        medians.append(imputer.statistics_[0])
        assert medians[-1] == pytest.approx(frame.iloc[rows]["x0"].median())
        return model.predict_proba(frame)[:, 1]

    first = metrics.bootstrap_optimism_validation(
        outcomes=y, groups=groups, refit_predict=refit, n_replicates=20, seed=29,
    )
    original_samples = [s.copy() for s in samples]
    assert len(samples) == 21
    assert len(set(medians)) > 2
    assert any(len(np.unique(s)) < len(s) for s in samples[1:])
    samples.clear()
    second = metrics.bootstrap_optimism_validation(
        outcomes=y, groups=groups, refit_predict=refit, n_replicates=20, seed=29,
    )
    assert first == second
    assert all(np.array_equal(a, b) for a, b in zip(samples, original_samples, strict=True))
    samples.clear()
    metrics.bootstrap_optimism_validation(
        outcomes=y, groups=groups, refit_predict=refit, n_replicates=20, seed=30,
    )
    assert any(not np.array_equal(a, b) for a, b in zip(samples[1:], original_samples[1:], strict=True))


@pytest.mark.parametrize("failures, stops", [(2, False), (3, True)])
def test_failed_refits_are_counted_and_excess_failures_stop(failures, stops):
    frame, y = _simulate(44, 90, 3)
    p = expit(0.7 * frame.x0.to_numpy())
    call = 0

    def refit(rows):
        nonlocal call
        call += 1
        if 1 < call <= failures + 1:
            raise RuntimeError("synthetic nonconvergence")
        return p

    kwargs = dict(outcomes=y, groups=np.arange(len(y)), refit_predict=refit, n_replicates=20, seed=29)
    if stops:
        with pytest.raises(RuntimeError, match="prediction_bootstrap_too_many_failures") as error:
            metrics.bootstrap_optimism_validation(**kwargs)
        assert error.value.failed_replicates == failures
    else:
        result = metrics.bootstrap_optimism_validation(**kwargs)
        assert result["failed_replicates"] == failures
        assert result["successful_replicates"] == 20 - failures
        assert len(result["failures"]) == failures
    assert call == 21


def test_calibration_in_the_large_uses_a_fixed_unit_slope():
    frame, y = _simulate(11, 300, 3)
    p = expit(1.0 + 1.8 * frame.x0.to_numpy())
    result = metrics.optimism_metrics(y.to_numpy(), p)
    intercept = result["calibration_in_the_large"]
    # The offset likelihood score equation: mean calibrated risk = event rate.
    assert expit(np.log(p / (1 - p)) + intercept).mean() == pytest.approx(y.mean(), abs=1e-8)
    assert abs(intercept - metrics._calibration_model(y.to_numpy(), p, epsilon=1e-6)[1]) > 0.1


def test_score_recalibration_uses_only_development_outcomes():
    frame, y = _simulate(49, 240, 3)
    score = 10 + 3 * frame.x0.to_numpy()
    probability, fit = metrics.recalibrate_score(
        outcomes=y[:180], scores=score[:180], evaluation_scores=score[180:],
    )
    assert fit["fit_n"] == 180
    assert fit["fit_scope"] == "development_partition_only"
    assert np.allclose(probability, expit(fit["intercept"] + fit["slope"] * score[180:]))
    assert np.max(np.abs(probability - score[180:])) > 1


def test_paired_delong_matches_independent_placement_reference():
    from easyicu.research_agent.methods.auc_interval import paired_auc_difference

    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    a = np.array([.9, .5, .7, .2, .4, .4, .8, .6, .3, .1, .6, .7])
    b = np.array([.8, .6, .6, .3, .6, .2, .7, .5, .2, .4, .5, .6])
    placements = []
    for score in (a, b):
        differences = score[y == 1, None] - score[y == 0]
        placements.append((differences > 0) + .5 * (differences == 0))
    case = np.array([v.mean(axis=1) for v in placements])
    control = np.array([v.mean(axis=0) for v in placements])
    covariance = np.cov(case, ddof=1) / 6 + np.cov(control, ddof=1) / 6
    variance = np.array([1, -1]) @ covariance @ np.array([1, -1])
    result = paired_auc_difference(y, a, b, np.arange(len(y)))
    assert result.difference == pytest.approx(placements[0].mean() - placements[1].mean())
    assert result.se == pytest.approx(np.sqrt(variance), abs=1e-12)
    assert result.ci_low == pytest.approx(result.difference - 1.959963984540054 * np.sqrt(variance))
