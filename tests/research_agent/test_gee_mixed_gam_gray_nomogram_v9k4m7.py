"""Kernels v9k4m7: GEE / MixedLM / Gray / GAM / nomogram.

Validity anchors:

* GEE/MixedLM/GAM are thin statsmodels wrappers: every fit is crossed
  against a direct package call (params/SE agreement is the floor).
* Gray's test has no package reference (lifelines ships none), so it is
  self-evidenced by two hard properties: degeneracy to the lifelines
  multivariate log-rank without competition (p tolerance 1e-6), and the
  hand-worked 4-subject example (statistic exactly 8/13, see
  ``methods/gray_test.py`` for the step-by-step arithmetic).
* The nomogram never fits: points -> total -> probability must reproduce
  the model's own link arithmetic (tolerance 1e-9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.methods import gam as gam_kernel
from easyicu.research_agent.methods import gee as gee_kernel
from easyicu.research_agent.methods import gray_test as gray_kernel
from easyicu.research_agent.methods import mixed_effects as mixed_kernel
from easyicu.research_agent.methods import nomogram as nomo_kernel
from easyicu.research_agent.methods.gam import fit_gam
from easyicu.research_agent.methods.gee import fit_gee
from easyicu.research_agent.methods.gray_test import gray_test
from easyicu.research_agent.methods.mixed_effects import fit_mixed_effects
from easyicu.research_agent.methods.nomogram import build_nomogram, nomogram_predict

# ---------------------------------------------------------------------------
# shared fixtures (fixed seeds; no RNG at test time)
# ---------------------------------------------------------------------------

_RNG_GEE = np.random.default_rng(0)
_N_CLU, _CLU_SIZE = 20, 5
_GEE_GROUPS = np.repeat(np.arange(_N_CLU), _CLU_SIZE)
_GEE_X = _RNG_GEE.normal(size=_N_CLU * _CLU_SIZE)
_GEE_P = 1.0 / (1.0 + np.exp(0.5 - 0.9 * _GEE_X))
_GEE_Y_BIN = (_RNG_GEE.uniform(size=_N_CLU * _CLU_SIZE) < _GEE_P).astype(float)
_GEE_Y_GAU = 1.0 + 2.0 * _GEE_X + _RNG_GEE.normal(size=_N_CLU * _CLU_SIZE)
_GEE_TIME = np.tile(np.arange(_CLU_SIZE), _N_CLU)

_RNG_MIX = np.random.default_rng(7)
_MIX_N_GROUPS, _MIX_SIZE = 8, 12
_MIX_GROUPS = np.repeat(np.arange(_MIX_N_GROUPS), _MIX_SIZE)
_MIX_X = _RNG_MIX.normal(size=_MIX_N_GROUPS * _MIX_SIZE)
_MIX_RE = np.repeat(_RNG_MIX.normal(scale=1.5, size=_MIX_N_GROUPS), _MIX_SIZE)
_MIX_Y = 1.0 + 2.0 * _MIX_X + _MIX_RE + _RNG_MIX.normal(
    scale=0.5, size=_MIX_N_GROUPS * _MIX_SIZE
)

_RNG_GAM = np.random.default_rng(3)
_GAM_N = 150
_GAM_X1 = _RNG_GAM.uniform(0.0, 3.0, size=_GAM_N)
_GAM_X2 = _RNG_GAM.normal(size=_GAM_N)
_GAM_Y = (
    0.5 * np.sin(2.0 * _GAM_X1)
    + 0.7 * _GAM_X2
    + _RNG_GAM.normal(scale=0.4, size=_GAM_N)
)
_GAM_Y_BIN = (
    _RNG_GAM.uniform(size=_GAM_N)
    < 1.0 / (1.0 + np.exp(-(0.3 * np.sin(2.0 * _GAM_X1) + 0.8 * _GAM_X2)))
).astype(float)

# Hand-worked Gray example from methods/gray_test.py docstring:
# A1: T=1 cause 2 | B1: T=2 cause 1 | A2: T=3 cause 1 | B2: T=4 cause 1
_GRAY_HAND_T = [1.0, 2.0, 3.0, 4.0]
_GRAY_HAND_E = [2, 1, 1, 1]
_GRAY_HAND_G = ["A", "B", "A", "B"]

_RNG_GRAY = np.random.default_rng(11)
_GRAY_N = 80
_GRAY_T = _RNG_GRAY.uniform(0.5, 5.0, size=_GRAY_N)
_GRAY_E = (_RNG_GRAY.uniform(size=_GRAY_N) < 0.6).astype(int)
_GRAY_C = np.where(_RNG_GRAY.uniform(size=_GRAY_N) < 0.7, _GRAY_E, 0)
_GRAY_G = np.where(_RNG_GRAY.uniform(size=_GRAY_N) < 0.5, "A", "B")


def _assert_canonical_stable(result: object, sha_fn: object) -> None:
    payload = result.to_json()  # type: ignore[union-attr]
    assert canonical_json(payload) == canonical_json(payload)
    assert sha_fn(result) == canonical_sha256(payload)


# ---------------------------------------------------------------------------
# GEE
# ---------------------------------------------------------------------------


def test_gee_binomial_known_answer_and_package_crosscheck() -> None:
    import statsmodels.api as sm

    res = fit_gee(
        _GEE_Y_BIN, pd.DataFrame({"x": _GEE_X}), _GEE_GROUPS, cov_struct="exchangeable"
    )
    assert res.var_names == ("const", "x")
    assert res.converged is True
    assert res.n_obs == 100 and res.n_clusters == 20
    assert res.claim_ceiling == "analysis_only"
    slope = res.coef[1]
    assert slope > 0.0  # true slope is +0.9
    assert res.p[1] < 0.05
    assert res.ci_low[1] <= slope <= res.ci_high[1]

    ref = sm.GEE(
        _GEE_Y_BIN,
        sm.add_constant(pd.DataFrame({"x": _GEE_X}), has_constant="add"),
        groups=_GEE_GROUPS,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    np.testing.assert_allclose(list(res.coef), np.asarray(ref.params), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(list(res.se), np.asarray(ref.bse), rtol=1e-8, atol=1e-10)

    again = fit_gee(
        _GEE_Y_BIN, pd.DataFrame({"x": _GEE_X}), _GEE_GROUPS, cov_struct="exchangeable"
    )
    assert canonical_json(res.to_json()) == canonical_json(again.to_json())
    _assert_canonical_stable(res, gee_kernel.result_sha256)


def test_gee_gaussian_independence_known_answer_and_crosscheck() -> None:
    import statsmodels.api as sm

    res = fit_gee(
        _GEE_Y_GAU,
        pd.DataFrame({"x": _GEE_X}),
        _GEE_GROUPS,
        family="gaussian",
        cov_struct="independence",
    )
    assert res.coef[1] == pytest.approx(2.0, abs=0.2)
    assert res.coef[0] == pytest.approx(1.0, abs=0.3)
    ref = sm.GEE(
        _GEE_Y_GAU,
        sm.add_constant(pd.DataFrame({"x": _GEE_X}), has_constant="add"),
        groups=_GEE_GROUPS,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()
    np.testing.assert_allclose(list(res.coef), np.asarray(ref.params), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(list(res.se), np.asarray(ref.bse), rtol=1e-8, atol=1e-10)


def test_gee_ar1_needs_explicit_time() -> None:
    with pytest.raises(ValueError):
        fit_gee(
            _GEE_Y_BIN, pd.DataFrame({"x": _GEE_X}), _GEE_GROUPS, cov_struct="ar1"
        )
    res = fit_gee(
        _GEE_Y_BIN,
        pd.DataFrame({"x": _GEE_X}),
        _GEE_GROUPS,
        cov_struct="ar1",
        time=_GEE_TIME,
    )
    assert res.converged is True and res.coef[1] > 0.0
    with pytest.raises(ValueError):  # time would be silently ignored here
        fit_gee(
            _GEE_Y_BIN,
            pd.DataFrame({"x": _GEE_X}),
            _GEE_GROUPS,
            cov_struct="exchangeable",
            time=_GEE_TIME,
        )


def test_gee_fail_closed() -> None:
    frame = pd.DataFrame({"x": _GEE_X})
    with pytest.raises(ValueError):  # unknown family
        fit_gee(_GEE_Y_BIN, frame, _GEE_GROUPS, family="poisson")
    with pytest.raises(ValueError):  # unknown correlation structure
        fit_gee(_GEE_Y_BIN, frame, _GEE_GROUPS, cov_struct="unstructured")
    with pytest.raises(ValueError):  # non-binary binomial outcome
        fit_gee(_GEE_Y_GAU, frame, _GEE_GROUPS, family="binomial")
    with pytest.raises(ValueError):  # single class
        fit_gee(np.zeros(100), frame, _GEE_GROUPS, family="binomial")
    with pytest.raises(ValueError):  # single cluster
        fit_gee(_GEE_Y_BIN, frame, np.zeros(100, dtype=int))
    with pytest.raises(ValueError):  # length mismatch
        fit_gee(_GEE_Y_BIN, frame, _GEE_GROUPS[:50])
    with pytest.raises(ValueError):  # non-finite outcome
        bad = _GEE_Y_GAU.copy()
        bad[0] = np.nan
        fit_gee(bad, frame, _GEE_GROUPS, family="gaussian")


# ---------------------------------------------------------------------------
# MixedLM
# ---------------------------------------------------------------------------


def test_mixed_effects_known_answer_and_package_crosscheck() -> None:
    import statsmodels.api as sm

    res = fit_mixed_effects(
        _MIX_Y, pd.DataFrame({"x": _MIX_X}), _MIX_GROUPS, reml=True
    )
    assert res.var_names == ("const", "x")
    assert res.converged is True
    assert res.n_groups == 8 and res.n_obs == 96
    assert res.claim_ceiling == "analysis_only"
    assert res.coef[1] == pytest.approx(2.0, abs=0.15)
    assert res.group_var > 0.0  # true tau^2 is 2.25
    assert res.resid_scale > 0.0

    ref = sm.MixedLM(
        _MIX_Y,
        sm.add_constant(pd.DataFrame({"x": _MIX_X}), has_constant="add"),
        groups=_MIX_GROUPS,
    ).fit(reml=True)
    np.testing.assert_allclose(
        list(res.coef), np.asarray(ref.fe_params), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        list(res.se), np.asarray(ref.bse_fe), rtol=1e-8, atol=1e-10
    )
    assert res.group_var == pytest.approx(
        float(np.asarray(ref.cov_re).ravel()[0]), rel=1e-8
    )

    again = fit_mixed_effects(_MIX_Y, pd.DataFrame({"x": _MIX_X}), _MIX_GROUPS)
    assert canonical_json(res.to_json()) == canonical_json(again.to_json())
    _assert_canonical_stable(res, mixed_kernel.result_sha256)


def test_mixed_effects_fail_closed() -> None:
    frame = pd.DataFrame({"x": _MIX_X})
    few_groups = np.repeat(np.arange(3), 32)
    with pytest.raises(ValueError):  # <5 groups: variance not identified
        fit_mixed_effects(_MIX_Y, frame, few_groups)
    with pytest.raises(ValueError):  # single group
        fit_mixed_effects(_MIX_Y, frame, np.zeros(96, dtype=int))
    with pytest.raises(ValueError):  # non-finite outcome
        bad = _MIX_Y.copy()
        bad[0] = np.inf
        fit_mixed_effects(bad, frame, _MIX_GROUPS)
    with pytest.raises(ValueError):  # length mismatch
        fit_mixed_effects(_MIX_Y, frame, _MIX_GROUPS[:10])
    with pytest.raises(ValueError):  # constant outcome
        fit_mixed_effects(np.ones(96), frame, _MIX_GROUPS)


# ---------------------------------------------------------------------------
# Gray
# ---------------------------------------------------------------------------


def test_gray_hand_computed_competing_risk_example() -> None:
    from scipy import stats as scipy_stats

    res = gray_test(_GRAY_HAND_T, _GRAY_HAND_E, _GRAY_HAND_G)
    # Hand arithmetic in methods/gray_test.py: Z_A = -2/3, V_AA = 13/18.
    assert res.statistic == pytest.approx(8.0 / 13.0, abs=1e-12)
    assert res.p_value == pytest.approx(scipy_stats.chi2.sf(8.0 / 13.0, 1), abs=1e-12)
    assert res.df == 1
    assert tuple(res.group_scores) == pytest.approx((-2.0 / 3.0, 2.0 / 3.0), abs=1e-12)
    assert res.n_interest == 3 and res.n_competing == 1 and res.n_censored == 0
    assert res.claim_ceiling == "analysis_only"

    # The carry-over is genuinely active: censoring the competing event as a
    # plain log-rank does must give a *different* answer here.
    from lifelines.statistics import logrank_test

    durations = pd.Series(_GRAY_HAND_T)
    is_a = [g == "A" for g in _GRAY_HAND_G]
    cause1 = [1 if e == 1 else 0 for e in _GRAY_HAND_E]
    lr = logrank_test(
        durations[is_a],
        durations[[not flag for flag in is_a]],
        event_observed_A=[v for v, flag in zip(cause1, is_a) if flag],
        event_observed_B=[v for v, flag in zip(cause1, is_a) if not flag],
    )
    assert abs(float(lr.test_statistic) - res.statistic) > 0.05
    _assert_canonical_stable(res, gray_kernel.result_sha256)


def test_gray_degenerates_to_logrank_without_competition() -> None:
    from lifelines.statistics import multivariate_logrank_test

    res = gray_test(_GRAY_T, _GRAY_C, _GRAY_G)
    ref = multivariate_logrank_test(_GRAY_T, _GRAY_G, _GRAY_C)
    assert res.p_value == pytest.approx(float(ref.p_value), abs=1e-6)
    assert res.statistic == pytest.approx(float(ref.test_statistic), abs=1e-6)

    three_groups = np.where(
        np.arange(_GRAY_N) % 3 == 0,
        "A",
        np.where(np.arange(_GRAY_N) % 3 == 1, "B", "C"),
    )
    res3 = gray_test(_GRAY_T, _GRAY_C, three_groups)
    ref3 = multivariate_logrank_test(_GRAY_T, three_groups, _GRAY_C)
    assert res3.df == 2
    assert res3.p_value == pytest.approx(float(ref3.p_value), abs=1e-6)

    again = gray_test(_GRAY_T, _GRAY_C, _GRAY_G)
    assert canonical_json(res.to_json()) == canonical_json(again.to_json())


def test_gray_fail_closed() -> None:
    with pytest.raises(ValueError):  # illegal event code
        gray_test([1.0, 2.0], [1, 3], ["A", "B"])
    with pytest.raises(ValueError):  # non-integral code
        gray_test([1.0, 2.0], [1, 1.5], ["A", "B"])
    with pytest.raises(ValueError):  # single group
        gray_test([1.0, 2.0], [1, 0], ["A", "A"])
    with pytest.raises(ValueError):  # no event of interest: empty risk set
        gray_test([1.0, 2.0], [2, 0], ["A", "B"])
    with pytest.raises(ValueError):  # empty input
        gray_test([], [], [])
    with pytest.raises(ValueError):  # negative duration
        gray_test([-1.0, 2.0], [1, 0], ["A", "B"])
    with pytest.raises(ValueError):  # length mismatch
        gray_test([1.0, 2.0], [1, 0], ["A"])
    with pytest.raises(ValueError):  # negative rho
        gray_test(_GRAY_HAND_T, _GRAY_HAND_E, _GRAY_HAND_G, rho=-1.0)


# ---------------------------------------------------------------------------
# GAM
# ---------------------------------------------------------------------------


def test_gam_gaussian_known_answer_and_package_crosscheck() -> None:
    import statsmodels.api as sm
    from statsmodels.gam.api import BSplines, GLMGam

    res = fit_gam(
        _GAM_Y,
        pd.DataFrame({"x1": _GAM_X1}),
        pd.DataFrame({"x2": _GAM_X2}),
        df=5,
        alpha=1.0,
    )
    assert [term.name for term in res.linear_terms] == ["const", "x2"]
    linear = {term.name: term for term in res.linear_terms}
    assert linear["x2"].coef == pytest.approx(0.7, abs=0.1)
    assert len(res.smooth_terms) == 1
    smooth = res.smooth_terms[0]
    assert smooth.variable == "x1" and smooth.df == 5
    assert smooth.n_basis == 4 and len(smooth.basis_names) == 4
    assert smooth.edf > 1.0  # the sine shape needs more than a line
    assert smooth.edf <= 5.0
    assert res.edf_total == pytest.approx(
        2.0 + smooth.edf, abs=1e-9
    )  # 1 per linear param (unpenalised) + smooth edf
    assert res.converged is True
    assert res.claim_ceiling == "analysis_only"

    smoother = BSplines(
        np.asarray(pd.DataFrame({"x1": _GAM_X1})),
        df=[5],
        degree=[3],
        variable_names=["x1"],
    )
    ref = GLMGam(
        _GAM_Y,
        exog=np.asarray(
            sm.add_constant(pd.DataFrame({"x2": _GAM_X2}), has_constant="skip")
        ),
        smoother=smoother,
        alpha=1.0,
        family=sm.families.Gaussian(),
    ).fit()
    np.testing.assert_allclose(
        [term.coef for term in res.linear_terms],
        np.asarray(ref.params)[:2],
        rtol=1e-8,
        atol=1e-10,
    )

    again = fit_gam(
        _GAM_Y,
        pd.DataFrame({"x1": _GAM_X1}),
        pd.DataFrame({"x2": _GAM_X2}),
        df=5,
        alpha=1.0,
    )
    assert canonical_json(res.to_json()) == canonical_json(again.to_json())
    _assert_canonical_stable(res, gam_kernel.result_sha256)


def test_gam_binomial_runs_and_matches_package() -> None:
    import statsmodels.api as sm
    from statsmodels.gam.api import BSplines, GLMGam

    res = fit_gam(
        _GAM_Y_BIN,
        pd.DataFrame({"x1": _GAM_X1}),
        pd.DataFrame({"x2": _GAM_X2}),
        df=4,
        family="binomial",
    )
    assert res.family == "binomial"
    assert res.linear_terms[1].coef > 0.0  # true x2 effect is +0.8
    smoother = BSplines(
        np.asarray(pd.DataFrame({"x1": _GAM_X1})),
        df=[4],
        degree=[3],
        variable_names=["x1"],
    )
    ref = GLMGam(
        _GAM_Y_BIN,
        exog=np.asarray(
            sm.add_constant(pd.DataFrame({"x2": _GAM_X2}), has_constant="skip")
        ),
        smoother=smoother,
        alpha=0.0,
        family=sm.families.Binomial(),
    ).fit()
    np.testing.assert_allclose(
        [term.coef for term in res.linear_terms],
        np.asarray(ref.params)[:2],
        rtol=1e-8,
        atol=1e-10,
    )


def test_gam_two_smooth_terms_with_per_variable_df() -> None:
    res = fit_gam(
        _GAM_Y,
        pd.DataFrame({"x1": _GAM_X1, "x2": _GAM_X2}),
        None,
        df={"x1": 5, "x2": 4},
        alpha=2.0,
    )
    assert [term.variable for term in res.smooth_terms] == ["x1", "x2"]
    assert [term.name for term in res.linear_terms] == ["const"]
    assert res.smooth_terms[0].n_basis == 4
    assert res.smooth_terms[1].n_basis == 3
    assert all(term.edf > 0.0 for term in res.smooth_terms)


def test_gam_fail_closed() -> None:
    smooth = pd.DataFrame({"x1": _GAM_X1})
    linear = pd.DataFrame({"x2": _GAM_X2})
    with pytest.raises(ValueError):  # unknown family
        fit_gam(_GAM_Y, smooth, linear, family="poisson")
    with pytest.raises(ValueError):  # df must exceed degree
        fit_gam(_GAM_Y, smooth, linear, df=3, degree=3)
    with pytest.raises(ValueError):  # df mapping misses a variable
        fit_gam(_GAM_Y, smooth, linear, df={"x9": 5})
    with pytest.raises(ValueError):  # constant smooth variable
        fit_gam(_GAM_Y, pd.DataFrame({"x1": np.ones(_GAM_N)}), linear)
    with pytest.raises(ValueError):  # non-binary binomial outcome
        fit_gam(_GAM_Y, smooth, linear, family="binomial")
    with pytest.raises(ValueError):  # non-finite input
        bad = smooth.copy()
        bad.iloc[0, 0] = np.nan
        fit_gam(_GAM_Y, bad, linear)
    with pytest.raises(ValueError):  # more parameters than observations
        tiny = pd.DataFrame({"x1": [0.5, 1.5]})
        fit_gam([0.1, 0.2], tiny, None, df=4)


# ---------------------------------------------------------------------------
# Nomogram
# ---------------------------------------------------------------------------


def test_nomogram_logistic_self_consistent_with_model() -> None:
    from scipy.special import expit

    res = build_nomogram(
        ["age", "sbp"],
        [0.04, 0.02],
        {"age": (30.0, 80.0), "sbp": (90.0, 200.0)},
        model="logistic",
        intercept=-5.0,
        n_grid=11,
    )
    assert res.claim_ceiling == "analysis_only"
    # Largest-range variable (sbp: 0.02*110 = 2.2) spans exactly 0..100.
    sbp_table = res.variable_tables[1]
    assert sbp_table.points[0] == pytest.approx(0.0, abs=1e-12)
    assert sbp_table.points[-1] == pytest.approx(100.0, abs=1e-12)
    age_table = res.variable_tables[0]
    assert age_table.points[0] == pytest.approx(0.0, abs=1e-12)
    assert age_table.points[-1] == pytest.approx(100.0 * 2.0 / 2.2, abs=1e-9)
    assert res.total_max == pytest.approx(
        age_table.points[-1] + sbp_table.points[-1], abs=1e-9
    )
    assert all(
        later >= earlier for earlier, later in zip(res.total_prob, res.total_prob[1:])
    )

    cases = [
        {"age": 30.0, "sbp": 90.0},
        {"age": 60.0, "sbp": 140.0},
        {"age": 80.0, "sbp": 200.0},
        {"age": 45.5, "sbp": 117.0},
    ]
    for case in cases:
        total, lp, prob = nomogram_predict(res, case)
        expected_lp = -5.0 + 0.04 * case["age"] + 0.02 * case["sbp"]
        assert lp == pytest.approx(expected_lp, abs=1e-9)
        assert prob == pytest.approx(float(expit(expected_lp)), abs=1e-9)
        # The total->probability grid inverts through the same closed form.
        assert prob == pytest.approx(
            float(expit(res.base_lp + total / res.points_per_unit_lp)), abs=1e-12
        )
    _assert_canonical_stable(res, nomo_kernel.result_sha256)
    again = build_nomogram(
        ["age", "sbp"],
        [0.04, 0.02],
        {"age": (30.0, 80.0), "sbp": (90.0, 200.0)},
        model="logistic",
        intercept=-5.0,
        n_grid=11,
    )
    assert canonical_json(res.to_json()) == canonical_json(again.to_json())


def test_nomogram_cox_self_consistent_with_model() -> None:
    res = build_nomogram(
        ["age", "ldl"],
        [0.03, 0.4],
        [(40.0, 70.0), (2.0, 6.0)],
        model="cox",
        baseline_survival=0.85,
        n_grid=9,
    )
    assert res.intercept is None
    for age, ldl in [(40.0, 2.0), (55.0, 4.0), (70.0, 6.0)]:
        total, lp, prob = nomogram_predict(res, [age, ldl])
        expected_lp = 0.03 * age + 0.4 * ldl
        assert lp == pytest.approx(expected_lp, abs=1e-9)
        assert prob == pytest.approx(1.0 - 0.85 ** float(np.exp(expected_lp)), abs=1e-9)


def test_nomogram_fail_closed() -> None:
    with pytest.raises(ValueError):  # length mismatch
        build_nomogram(["a", "b"], [0.1], {"a": (0.0, 1.0), "b": (0.0, 1.0)})
    with pytest.raises(ValueError):  # non-finite coef
        build_nomogram(["a"], [np.inf], {"a": (0.0, 1.0)})
    with pytest.raises(ValueError):  # lo >= hi
        build_nomogram(["a"], [0.1], {"a": (1.0, 1.0)})
    with pytest.raises(ValueError):  # reference outside range
        build_nomogram(["a"], [0.1], {"a": (0.0, 1.0)}, reference={"a": 5.0})
    with pytest.raises(ValueError):  # unknown model
        build_nomogram(["a"], [0.1], {"a": (0.0, 1.0)}, model="poisson")
    with pytest.raises(ValueError):  # Cox needs a baseline survival
        build_nomogram(["a"], [0.1], {"a": (0.0, 1.0)}, model="cox")
    with pytest.raises(ValueError):  # baseline outside (0, 1)
        build_nomogram(
            ["a"], [0.1], {"a": (0.0, 1.0)}, model="cox", baseline_survival=1.0
        )
    with pytest.raises(ValueError):  # intercept belongs to logistic, not Cox
        build_nomogram(
            ["a"],
            [0.1],
            {"a": (0.0, 1.0)},
            model="cox",
            baseline_survival=0.8,
            intercept=0.5,
        )
    with pytest.raises(ValueError):  # baseline belongs to Cox, not logistic
        build_nomogram(
            ["a"], [0.1], {"a": (0.0, 1.0)}, model="logistic", baseline_survival=0.8
        )
    with pytest.raises(ValueError):  # all-zero effects: no points scale
        build_nomogram(["a"], [0.0], {"a": (0.0, 1.0)})
    with pytest.raises(ValueError):  # extrapolation refused
        res = build_nomogram(["a"], [0.1], {"a": (0.0, 1.0)}, intercept=0.0)
        nomogram_predict(res, {"a": 2.0})


# ---------------------------------------------------------------------------
# shared contract surface
# ---------------------------------------------------------------------------


def test_shared_kernel_contract_surface() -> None:
    for module in (gee_kernel, mixed_kernel, gray_kernel, gam_kernel, nomo_kernel):
        assert module.TOOL_VERSION == "1.0.0"
        assert isinstance(module.__all__, list) and module.__all__
        assert "result_sha256" in module.__all__
        with pytest.raises(TypeError):
            module.result_sha256(object())
