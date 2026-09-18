"""Deterministic PSM/IPTW kernel tests (analysis_only, known-answer synthetic data).

Covers propensity-score recovery of a known assignment mechanism, 1:1
matching balance improvement, IPTW truncation, byte-identical determinism,
and fail-closed boundaries. No effect estimation exists in the kernel and
none is asserted here.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.methods import propensity_weighting as pw
from easyicu.research_agent.methods.propensity_weighting import (
    BalanceRow,
    IPTWResult,
    MatchingResult,
    PropensityScoreResult,
    PropensityWeightingError,
    compute_iptw_weights,
    estimate_propensity_scores,
    match_nearest_neighbor,
    standardized_mean_difference,
)


def _mechanism_data(seed: int = 123, n: int = 2000):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = -1.0 + 2.0 * x1 - 1.5 * x2
    p_true = 1.0 / (1.0 + np.exp(-logit))
    treated = (rng.uniform(size=n) < p_true).astype(int)
    covariates = np.column_stack([x1, x2])
    return treated, covariates, p_true


def _imbalance_data():
    rng = np.random.default_rng(7)
    n_t, n_c = 150, 600
    treated_x = np.column_stack(
        [rng.normal(0.8, 1.0, n_t), rng.normal(-0.5, 1.0, n_t)]
    )
    control_x = np.column_stack(
        [rng.normal(0.0, 1.0, n_c), rng.normal(0.3, 1.0, n_c)]
    )
    covariates = np.vstack([treated_x, control_x])
    treated = np.array([1] * n_t + [0] * n_c)
    return treated, covariates


def test_ps_recovers_known_assignment_mechanism():
    treated, covariates, p_true = _mechanism_data()
    result = estimate_propensity_scores(treated, covariates)
    assert isinstance(result, PropensityScoreResult)
    estimated = np.asarray(result.propensity_scores)
    assert estimated.shape == p_true.shape
    assert np.all((estimated >= 0.0) & (estimated <= 1.0))
    corr = float(np.corrcoef(p_true, estimated)[0, 1])
    assert corr > 0.99
    assert result.n_treated == int(treated.sum())
    assert result.n_control == int((treated == 0).sum())
    assert result.treated_rate == pytest.approx(float(treated.mean()))
    assert len(result.coefficients) == 2
    # Strong positive x1 / negative x2 mechanism recovered with correct signs.
    assert result.coefficients[0] > 0
    assert result.coefficients[1] < 0


def test_matching_repairs_known_imbalance():
    treated, covariates = _imbalance_data()
    ps = estimate_propensity_scores(treated, covariates)
    matched = match_nearest_neighbor(treated, covariates, ps)
    assert isinstance(matched, MatchingResult)
    assert matched.n_pairs > 0
    assert matched.n_unmatched_treated + matched.n_pairs == int((treated == 1).sum())
    assert matched.n_unmatched_control + matched.n_pairs == int((treated == 0).sum())
    # Pairs are 1:1 without replacement across arms.
    assert len({t for t, _ in matched.pairs}) == matched.n_pairs
    assert len({c for _, c in matched.pairs}) == matched.n_pairs
    for t, c in matched.pairs:
        assert treated[t] == 1 and treated[c] == 0
        assert abs(ps.propensity_scores[t] - ps.propensity_scores[c]) <= 0.1 + 1e-12
    assert len(matched.balance) == 2
    for row in matched.balance:
        assert isinstance(row, BalanceRow)
        assert abs(row.smd_before) > 0.5
        assert row.smd_after is not None and abs(row.smd_after) < 0.1
        assert row.n_matched_pairs == matched.n_pairs
    payload = matched.to_dict()
    assert set(payload["balance"][0]) == {
        "covariate",
        "mean_treated_before",
        "mean_control_before",
        "smd_before",
        "mean_treated_after",
        "mean_control_after",
        "smd_after",
        "n_matched_pairs",
    }


def test_caliper_none_matches_every_treated_subject():
    treated, covariates = _imbalance_data()
    ps = estimate_propensity_scores(treated, covariates)
    matched = match_nearest_neighbor(treated, covariates, ps, caliper=None)
    assert matched.n_pairs == int((treated == 1).sum())
    assert matched.n_unmatched_treated == 0


def test_full_pipeline_is_byte_identical_across_runs():
    treated, covariates = _imbalance_data()
    frame = pd.DataFrame(covariates, columns=["age", "sbp"])

    def run_once():
        ps = estimate_propensity_scores(
            pd.Series(treated), frame, covariate_names=["age", "sbp"]
        )
        matched = match_nearest_neighbor(treated, covariates, ps)
        weights = compute_iptw_weights(treated, ps, truncate_quantiles=(0.01, 0.99))
        return (
            json.dumps(ps.to_json(), sort_keys=True).encode(),
            json.dumps(matched.to_json(), sort_keys=True).encode(),
            json.dumps(weights.to_json(), sort_keys=True).encode(),
            ps.digest,
            matched.digest,
            weights.digest,
        )

    first, second = run_once(), run_once()
    assert first == second
    assert len({first[3], first[4], first[5]}) == 3
    for digest in first[3:]:
        assert len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)


def test_iptw_stabilized_formula_and_summaries():
    treated = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.8, 0.2, 0.7, 0.3, 0.5, 0.5])
    result = compute_iptw_weights(treated, scores)
    assert isinstance(result, IPTWResult)
    p = 0.5
    expected = np.where(
        treated == 1, p / scores, (1 - p) / (1 - scores)
    )
    assert np.allclose(result.weights, expected, atol=1e-12)
    assert result.weight_min == pytest.approx(float(np.min(expected)))
    assert result.weight_p50 == pytest.approx(float(np.median(expected)))
    assert result.weight_max == pytest.approx(float(np.max(expected)))
    assert result.weight_mean == pytest.approx(float(np.mean(expected)), abs=0.15)
    total = float(np.sum(expected))
    assert result.ess == pytest.approx(total * total / float(np.sum(expected**2)))
    assert 0.0 < result.ess <= result.n
    assert 0.0 < result.ess_treated <= 3
    assert 0.0 < result.ess_control <= 3


def test_iptw_truncation_clips_extreme_weights():
    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(0, 1, n)
    scores = 1.0 / (1.0 + np.exp(-(3.0 * x)))
    treated = (rng.uniform(size=n) < scores).astype(int)
    plain = compute_iptw_weights(treated, scores)
    truncated = compute_iptw_weights(treated, scores, truncate_quantiles=(0.05, 0.95))
    weights = np.asarray(plain.weights)
    low = float(np.quantile(weights, 0.05))
    high = float(np.quantile(weights, 0.95))
    assert truncated.weight_max == pytest.approx(high)
    assert truncated.weight_min == pytest.approx(low)
    assert truncated.weight_max < plain.weight_max
    assert truncated.truncate_quantiles == [0.05, 0.95]
    assert all(np.asarray(truncated.weights) <= high + 1e-12)
    assert all(np.asarray(truncated.weights) >= low - 1e-12)


def test_unstabilized_weights_use_unit_numerators():
    treated = np.array([1, 0])
    scores = np.array([0.25, 0.75])
    result = compute_iptw_weights(treated, scores, stabilized=False)
    assert result.weights == pytest.approx([4.0, 4.0])
    assert result.stabilized is False


@pytest.mark.parametrize(
    "treated",
    [np.ones(10, dtype=int), np.zeros(10, dtype=int)],
)
def test_single_class_treatment_is_rejected(treated):
    covariates = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises((PropensityWeightingError, ValueError)):
        estimate_propensity_scores(treated, covariates)


@pytest.mark.parametrize("bad", [0.5, 2, -1, np.nan, np.inf, "1", None])
def test_nonbinary_treatment_is_rejected(bad):
    treated = [0, 1, 1, 0, bad, 1]
    covariates = np.random.default_rng(1).normal(size=(6, 2))
    with pytest.raises((PropensityWeightingError, ValueError)):
        estimate_propensity_scores(treated, covariates)


def test_missing_inputs_are_rejected_fail_closed():
    treated = np.array([0, 1, 0, 1])
    good_x = np.array([[0.1, 1.0], [0.2, 0.5], [0.3, 1.5], [0.4, 0.2]])
    with pytest.raises((PropensityWeightingError, ValueError)):
        estimate_propensity_scores([0, 1, None, 1], good_x)
    with pytest.raises((PropensityWeightingError, ValueError)):
        estimate_propensity_scores([0, 1, np.nan, 1], good_x)
    bad_x = good_x.copy()
    bad_x[0, 0] = np.nan
    with pytest.raises((PropensityWeightingError, ValueError)):
        estimate_propensity_scores(treated, bad_x)
    bad_x = good_x.copy()
    bad_x[1, 1] = np.inf
    with pytest.raises((PropensityWeightingError, ValueError)):
        estimate_propensity_scores(treated, bad_x)
    with pytest.raises((PropensityWeightingError, ValueError)):
        estimate_propensity_scores(treated, good_x[:3])
    with pytest.raises((PropensityWeightingError, ValueError)):
        compute_iptw_weights(treated, np.array([0.1, 0.2, 1.5, 0.4]))
    with pytest.raises((PropensityWeightingError, ValueError)):
        compute_iptw_weights(treated, np.array([0.1, np.nan, 0.3, 0.4]))
    with pytest.raises((PropensityWeightingError, ValueError)):
        compute_iptw_weights(treated, np.array([0.1, 0.2, 0.3, 0.4]), truncate_quantiles=(0.9, 0.1))


def test_single_covariate_end_to_end():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(0, 1, n)
    p_true = 1.0 / (1.0 + np.exp(-(0.5 + 1.5 * x)))
    treated = (rng.uniform(size=n) < p_true).astype(int)
    single = x.reshape(-1, 1)
    ps = estimate_propensity_scores(treated, single)
    assert len(ps.covariate_names) == 1
    matched = match_nearest_neighbor(treated, single, ps)
    assert matched.n_pairs > 0
    weights = compute_iptw_weights(treated, ps)
    assert len(weights.weights) == n
    assert all(math.isfinite(w) for w in weights.weights)
    # Flat-vector spelling of one covariate is accepted too.
    ps_flat = estimate_propensity_scores(treated, x)
    assert ps_flat.digest == ps.digest


def test_pandas_inputs_use_frame_columns():
    treated, covariates = _imbalance_data()
    frame = pd.DataFrame(covariates, columns=["age", "sbp"])
    series = pd.Series(treated)
    ps = estimate_propensity_scores(series, frame)
    assert ps.covariate_names == ["age", "sbp"]
    matched = match_nearest_neighbor(series, frame, ps)
    assert matched.n_pairs > 0
    assert [row.covariate for row in matched.balance] == ["age", "sbp"]
    weights = compute_iptw_weights(series, ps)
    assert weights.n == len(series)


def test_results_are_typed_jsonable_and_analysis_only():
    treated, covariates = _imbalance_data()
    ps = estimate_propensity_scores(treated, covariates)
    matched = match_nearest_neighbor(treated, covariates, ps)
    weights = compute_iptw_weights(treated, covariates.mean(axis=1) * 0 + 0.3)
    for result in (ps, matched, weights):
        assert result.analysis_only is True
        assert "analysis_only" in result.note
        payload = result.to_dict()
        assert payload == result.to_json()
        json.dumps(payload)  # must be JSON-serializable
        assert payload["analysis_only"] is True
        assert "reportable" not in payload
    assert weights.method == "iptw"
    assert matched.method == "nearest_neighbor_1to1_without_replacement"


def test_kernel_exposes_no_effect_estimation_surface():
    import re

    public = {name for name in dir(pw) if not name.startswith("_")}
    assert "estimate_propensity_scores" in public
    assert "match_nearest_neighbor" in public
    assert "compute_iptw_weights" in public
    tokens: set = set()
    for name in public:
        tokens.update(re.split(r"[^a-z]+", name.lower()))
    for forbidden in ("effect", "outcome", "att", "ate", "hazard", "odds"):
        assert forbidden not in tokens
    import inspect

    for func in (estimate_propensity_scores, match_nearest_neighbor, compute_iptw_weights):
        params = " ".join(inspect.signature(func).parameters)
        assert "outcome" not in params.lower()
        assert "effect" not in params.lower()


def test_smd_helper_matches_hand_computation():
    treated_vals = [2.0, 4.0, 6.0, 8.0]
    control_vals = [1.0, 3.0, 5.0, 7.0]
    value = standardized_mean_difference(treated_vals, control_vals)
    pooled = (np.var(treated_vals, ddof=1) + np.var(control_vals, ddof=1)) / 2.0
    assert value == pytest.approx((np.mean(treated_vals) - np.mean(control_vals)) / np.sqrt(pooled))
    assert standardized_mean_difference([1.0], [1.0, 2.0]) is None
    assert standardized_mean_difference([5.0, 5.0], [5.0, 5.0]) == 0.0
    assert standardized_mean_difference([5.0, 5.0], [6.0, 6.0]) is None


def test_matching_digest_binds_pairs_not_just_inputs() -> None:
    """Review finding: a digest over inputs alone cannot serve as an origin
    receipt. Different pairings of identical inputs must digest differently."""

    from easyicu.research_agent.methods.propensity_weighting import (
        estimate_propensity_scores,
        match_nearest_neighbor,
    )

    treated, covariates = _imbalance_data()
    ps = estimate_propensity_scores(treated, covariates)
    wide = match_nearest_neighbor(treated, covariates, ps, caliper=None)
    narrow = match_nearest_neighbor(treated, covariates, ps, caliper=0.01)
    assert wide.pairs != narrow.pairs
    assert wide.digest != narrow.digest
    assert wide.digest == match_nearest_neighbor(
        treated, covariates, ps, caliper=None
    ).digest
