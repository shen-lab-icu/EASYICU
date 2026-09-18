"""Kernel tests: categorical/continuous NRI + IDI.

Known-answer fixture (hand-computed, see module docstring for the arithmetic)::

    y     = [1, 1, 1, 0, 0, 0]
    p_old = [0.2, 0.4, 0.6, 0.3, 0.5, 0.7]
    p_new = [0.5, 0.5, 0.8, 0.2, 0.4, 0.4]

cutoffs (0.3, 0.6) -> categorical NRI = 1/3 + 2/3 = 1.0;
continuous NRI = 1.0 + 1.0 = 2.0;
IDI = (0.6 - 1/3) - (0.4 - 0.5) = 0.2666... + 0.1 = 11/30.
"""

from __future__ import annotations

import numpy as np
import pytest

from easyicu.research_agent.methods.reclassification import (
    categorical_nri,
    continuous_nri,
    idi,
)


Y = [1, 1, 1, 0, 0, 0]
P_OLD = [0.2, 0.4, 0.6, 0.3, 0.5, 0.7]
P_NEW = [0.5, 0.5, 0.8, 0.2, 0.4, 0.4]
CUTS = (0.3, 0.6)


def test_categorical_nri_known_answer() -> None:
    res = categorical_nri(Y, P_OLD, P_NEW, CUTS, n_bootstrap=200, random_state=0)
    assert res.nri == pytest.approx(1.0)
    assert res.nri_events == pytest.approx(1.0 / 3.0)
    assert res.nri_nonevents == pytest.approx(2.0 / 3.0)
    assert res.kind == "categorical"
    assert res.cutoffs == (0.3, 0.6)
    assert res.n_events == 3 and res.n_nonevents == 3
    assert res.claim_ceiling == "analysis_only"


def test_continuous_nri_known_answer() -> None:
    res = continuous_nri(Y, P_OLD, P_NEW, n_bootstrap=200, random_state=0)
    assert res.nri == pytest.approx(2.0)
    assert res.nri_events == pytest.approx(1.0)
    assert res.nri_nonevents == pytest.approx(1.0)
    assert res.kind == "continuous"


def test_idi_known_answer() -> None:
    res = idi(Y, P_OLD, P_NEW, n_bootstrap=200, random_state=0)
    assert res.idi == pytest.approx(11.0 / 30.0)
    assert res.slope_new == pytest.approx(0.6 - 1.0 / 3.0)
    assert res.slope_old == pytest.approx(0.4 - 0.5)
    assert res.idi == pytest.approx(res.slope_new - res.slope_old)
    assert res.claim_ceiling == "analysis_only"


def test_bootstrap_ci_brackets_point_and_is_deterministic() -> None:
    first = continuous_nri(Y, P_OLD, P_NEW, n_bootstrap=200, random_state=0)
    second = continuous_nri(Y, P_OLD, P_NEW, n_bootstrap=200, random_state=0)
    assert first.to_json() == second.to_json()
    assert first.ci_low <= first.nri <= first.ci_high
    assert first.se >= 0.0
    assert 0 < first.n_successful <= first.n_bootstrap

    cat = categorical_nri(Y, P_OLD, P_NEW, CUTS, n_bootstrap=200, random_state=0)
    assert cat.to_json() == categorical_nri(
        Y, P_OLD, P_NEW, CUTS, n_bootstrap=200, random_state=0
    ).to_json()
    assert cat.ci_low <= cat.nri <= cat.ci_high

    d = idi(Y, P_OLD, P_NEW, n_bootstrap=200, random_state=0)
    assert d.to_json() == idi(Y, P_OLD, P_NEW, n_bootstrap=200, random_state=0).to_json()
    assert d.ci_low <= d.idi <= d.ci_high


def test_to_json_carries_ceiling_and_params() -> None:
    payload = categorical_nri(Y, P_OLD, P_NEW, CUTS, n_bootstrap=50).to_json()
    assert payload["claim_ceiling"] == "analysis_only"
    assert payload["cutoffs"] == [0.3, 0.6]
    assert payload["ci"][0] <= payload["nri"] <= payload["ci"][1]
    assert idi(Y, P_OLD, P_NEW, n_bootstrap=50).to_json()["claim_ceiling"] == "analysis_only"


@pytest.mark.parametrize(
    "y,po,pn",
    [
        ([1, 1, 0], [0.2, 0.4], [0.3, 0.4, 0.5]),  # length mismatch
        ([1, 0, 1], [1.2, 0.4, 0.5], [0.3, 0.4, 0.5]),  # prob > 1
        ([1, 0, 1], [-0.1, 0.4, 0.5], [0.3, 0.4, 0.5]),  # prob < 0
        ([1, 0, 1], [np.nan, 0.4, 0.5], [0.3, 0.4, 0.5]),  # NaN risk
        ([1, 0, 1], [0.2, np.inf, 0.5], [0.3, 0.4, 0.5]),  # inf risk
        ([2, 0, 1], [0.2, 0.4, 0.5], [0.3, 0.4, 0.5]),  # non-binary outcome
        ([1, 1, 1], [0.2, 0.4, 0.5], [0.3, 0.4, 0.5]),  # no nonevents
        ([0, 0, 0], [0.2, 0.4, 0.5], [0.3, 0.4, 0.5]),  # no events
    ],
)
def test_fail_closed_illegal_inputs(y: object, po: object, pn: object) -> None:
    with pytest.raises(ValueError):
        continuous_nri(y, po, pn, n_bootstrap=50)
    with pytest.raises(ValueError):
        idi(y, po, pn, n_bootstrap=50)


@pytest.mark.parametrize("cuts", [[], [0.5, 0.5], [0.6, 0.3], [0.0, 0.5], [0.5, 1.0], [np.nan]])
def test_fail_closed_illegal_cutoffs(cuts: object) -> None:
    with pytest.raises(ValueError):
        categorical_nri(Y, P_OLD, P_NEW, cuts, n_bootstrap=50)


@pytest.mark.parametrize("kwargs", [{"n_bootstrap": 0}, {"ci_level": 1.0}, {"ci_level": 0.0}])
def test_fail_closed_illegal_bootstrap_params(kwargs: object) -> None:
    with pytest.raises(ValueError):
        continuous_nri(Y, P_OLD, P_NEW, **kwargs)  # type: ignore[arg-type]
