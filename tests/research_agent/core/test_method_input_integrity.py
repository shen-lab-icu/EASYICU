"""Reject unidentifiable or invalid inputs instead of emitting valid-looking results."""

import numpy as np
import pytest

from easyicu.research_agent.methods.missing_data import mice_impute
from easyicu.research_agent.methods.delong_auc import (
    delong_auc_ci, delong_auc_variance, delong_test,
)
from easyicu.research_agent.methods.conformal import (
    _conformal_quantile, conformal_calibrate, conformal_evaluate,
    conformal_predict_sets, conformal_sentence,
)


def test_entirely_missing_target_is_not_identifiable():
    with pytest.raises(ValueError, match="observed"):
        mice_impute(column="x", target=[np.nan] * 3, predictors=[[1], [2], [3]])


@pytest.mark.parametrize("target,predictors", [
    ([1, np.inf, np.nan], [[1], [2], [3]]),
    ([1, 2, np.nan], [[1], [np.nan], [3]]),
    ([1, 2, np.nan], [[1], [2], [np.inf]]),
    ([1, np.nan], [[1]]),
    ([[1], [np.nan]], [[1], [2]]),
    ([1, np.nan], [1, 2]),
])
def test_imputation_rejects_invalid_arrays(target, predictors):
    with pytest.raises(ValueError):
        mice_impute(column="x", target=target, predictors=predictors)


def test_imputation_declares_single_deterministic_method():
    target = [1, 2, np.nan]
    filled, info = mice_impute(column="x", target=target, predictors=[[1], [2], [3]])
    assert filled[:2] == target[:2]
    assert np.isfinite(filled).all()
    assert info.to_json()["method"] == "deterministic_ridge_single_imputation"
    assert info.to_json()["multiple_imputation"] is False
    assert info.n_iterations == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_delong_rejects_nonfinite_in_all_public_estimators(bad):
    labels = [1, 1, 0, 0]
    scores = [0.9, bad, 0.1, 0.2]
    for estimator in (delong_auc_variance, delong_auc_ci):
        with pytest.raises(ValueError, match="finite"):
            estimator(labels, scores)
    for a, b in ((scores, [0.9, 0.8, 0.1, 0.2]), ([0.9, 0.8, 0.1, 0.2], scores)):
        with pytest.raises(ValueError, match="finite"):
            delong_test(labels, a, b)


@pytest.mark.parametrize("labels", [[1, np.nan, 0, 0], [1, 0.5, 0, 0], [[1], [1], [0], [0]]])
def test_delong_requires_one_dimensional_binary_labels(labels):
    with pytest.raises(ValueError):
        delong_auc_variance(labels, [0.9, 0.8, 0.1, 0.2])


@pytest.mark.parametrize("mondrian", [False, True])
def test_tiny_nonempty_calibration_includes_all_labels(mondrian):
    thresholds = conformal_calibrate([0.9], [1], alpha=0.1, mondrian=mondrian)
    assert conformal_predict_sets([0, 0.1, 0.9, 1], thresholds) == [{0, 1}] * 4


def test_conformal_uses_exact_corrected_order_statistic():
    scores = np.arange(10) / 10
    # ceil(11 * .8) = 9: ninth ordered score, not interpolated tenth.
    assert _conformal_quantile(scores, 0.2) == pytest.approx(0.8)


@pytest.mark.parametrize("probs,labels", [([np.nan], [1]), ([np.inf], [0]),
    ([-0.1], [1]), ([1.1], [1]), ([0.5], [0.5]), ([0.5], [-1]),
    ([0.5], [2]), ([[0.5]], [1])])
def test_conformal_rejects_invalid_calibration(probs, labels):
    with pytest.raises(ValueError):
        conformal_calibrate(probs, labels)


def test_conformal_test_split_length_must_match():
    with pytest.raises(ValueError, match="length"):
        conformal_evaluate([0.3, 0.7], [0, 1], [0.2, 0.8], [0])


def test_conformal_sentence_does_not_claim_patient_conditional_guarantee():
    result = conformal_evaluate([0.3, 0.7], [0, 1], [0.2, 0.8], [0, 1])
    sentence = conformal_sentence(result)
    assert "exchangeab" in sentence
    assert "not" in sentence and "individual" in sentence
    assert "per-patient guarantee" not in sentence
