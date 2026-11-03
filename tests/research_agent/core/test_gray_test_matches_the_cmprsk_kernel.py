"""Gray's K-sample test agrees with the original cmprsk Fortran kernel.

Reference: CRAN ``cmprsk`` 2.2-12 (DESCRIPTION Date 2024-05-14),
``src/crstm.f`` from https://github.com/cran/cmprsk, sha256
``a359f29936573aebe84a56abdf91edf26cf7e46464f473c1b8b85ff1b650b829``.

How the frozen values were produced (2026-11-03): ``crstm.f`` was compiled
unchanged with GNU Fortran 15.2.0 together with a small driver that calls
``crstm`` with one stratum, after preparing the data the way ``cuminc()``
does (sort by time; status 0 = censored, 1 = the cause being compared,
2 = any other cause; groups coded 1..K). The statistic is ``s' vs^{-1} s``
from the kernel's ``s`` and ``vs`` and the p-value is the chi-square upper
tail on ``K-1`` df. The numbers below are those kernel outputs; nothing in
this file re-derives them. For the four-subject case the kernel's score is
``-1/6`` and its variance ``55/72``, so the statistic is exactly ``2/55``.
"""

from __future__ import annotations

import numpy as np
import pytest

from easyicu.research_agent.methods.gray_test import GrayTestError, gray_test

_TOL = 1e-10

_DATA = {
    "four_subject": dict(
        t=[1.0, 2.0, 3.0, 4.0],
        e=[2, 1, 1, 1],
        g=["A", "B", "A", "B"],
    ),
    "eight_subject": dict(
        t=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        e=[2, 1, 0, 1, 2, 1, 0, 1],
        g=["A", "B", "B", "A", "B", "A", "A", "B"],
    ),
    "censored_competing": dict(
        t=[0.5, 1.2, 1.9, 2.4, 3.1, 3.3, 4.0, 4.6, 5.2, 6.8, 7.5, 8.1],
        e=[1, 2, 0, 1, 1, 0, 2, 1, 0, 2, 1, 0],
        g=["A", "B", "A", "B", "A", "A", "B", "B", "A", "A", "B", "B"],
    ),
    "tied": dict(
        t=[1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6],
        e=[1, 2, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1, 0, 1, 2, 0],
        g=["A", "B", "B", "A", "B", "A", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    ),
    "three_groups": dict(
        t=[0.7, 1.1, 1.4, 2.0, 2.0, 2.6, 3.0, 3.5, 3.9, 4.4, 4.8, 5.0, 5.5, 6.1, 6.6, 7.2, 7.9, 8.4],
        e=[1, 1, 2, 0, 1, 2, 1, 0, 1, 2, 0, 1, 2, 1, 0, 1, 2, 0],
        g=["a", "b", "c", "a", "c", "b", "a", "c", "b", "a", "b", "c", "c", "a", "a", "b", "b", "c"],
    ),
}

# (data, event_of_interest, rho, statistic, p_value, crstm scores s[0..K-2])
_KERNEL_VALUES = [
    ("four_subject", 1, 0.0, 0.03636363636363636, 0.8487665010168555, (-0.16666666666666663,)),
    ("eight_subject", 1, 0.0, 0.06565507052923142, 0.7977713501627157, (0.2333333333333334,)),
    ("censored_competing", 1, 0.0, 0.13919858438755617, 0.7090790311353017, (0.36984126984126997,)),
    ("censored_competing", 2, 0.0, 0.04590320632610714, 0.8303518451174909, (-0.1759581881533101,)),
    ("censored_competing", 1, 1.0, 0.19377454622960516, 0.6597934885022264, (0.3821575543797767,)),
    ("tied", 1, 0.0, 0.029603154818510446, 0.8633937603418947, (-0.21471764173062197,)),
    ("tied", 2, 1.0, 0.011808279307893637, 0.9134674840838398, (0.09587123039898537,)),
    ("three_groups", 1, 0.0, 0.6230766147905173, 0.7323195567133243, (0.8439958705965707, 0.12622550197563048)),
    ("three_groups", 1, 1.0, 0.5788729598413992, 0.7486853479531983, (0.6906248183479863, 0.028979247186752877)),
    ("three_groups", 2, 0.0, 0.14003613451135621, 0.932376974260589, (-0.34053207766070714, 0.06369222207787367)),
]


@pytest.mark.parametrize(
    "name, cause, rho, statistic, p_value, scores",
    _KERNEL_VALUES,
    ids=[f"{row[0]}-cause{row[1]}-rho{row[2]:g}" for row in _KERNEL_VALUES],
)
def test_statistic_p_value_and_scores_match_crstm(name, cause, rho, statistic, p_value, scores):
    data = _DATA[name]
    result = gray_test(data["t"], data["e"], data["g"], event_of_interest=cause, rho=rho)
    assert result.statistic == pytest.approx(statistic, abs=_TOL)
    assert result.p_value == pytest.approx(p_value, abs=_TOL)
    assert result.df == len(scores)
    assert result.group_scores[:-1] == pytest.approx(scores, abs=_TOL)
    # crst returns K-1 scores; the reported last score closes the sum to zero.
    assert sum(result.group_scores) == pytest.approx(0.0, abs=_TOL)
    assert result.claim_ceiling == "analysis_only"
    assert result.event_of_interest == cause


def test_four_subject_case_is_two_fifty_fifths():
    data = _DATA["four_subject"]
    result = gray_test(data["t"], data["e"], data["g"])
    assert result.statistic == pytest.approx(2.0 / 55.0, abs=1e-12)
    assert result.group_scores == pytest.approx((-1.0 / 6.0, 1.0 / 6.0), abs=1e-12)


def test_cause_two_is_the_same_test_with_the_codes_swapped():
    data = _DATA["censored_competing"]
    swapped = [{1: 2, 2: 1}.get(code, code) for code in data["e"]]
    direct = gray_test(data["t"], data["e"], data["g"], event_of_interest=2)
    recoded = gray_test(data["t"], swapped, data["g"], event_of_interest=1)
    assert direct.statistic == pytest.approx(recoded.statistic, abs=1e-12)
    assert direct.n_interest == recoded.n_interest == 3
    assert direct.n_competing == recoded.n_competing == 5
    assert direct.competing_event == 1


def test_row_order_and_group_names_do_not_change_the_statistic():
    data = _DATA["three_groups"]
    reference = gray_test(data["t"], data["e"], data["g"], rho=1.0)
    order = np.random.default_rng(3).permutation(len(data["t"]))
    renamed = {"a": "z", "b": "m", "c": "b"}
    shuffled = gray_test(
        [data["t"][i] for i in order],
        [data["e"][i] for i in order],
        [renamed[data["g"][i]] for i in order],
        rho=1.0,
    )
    assert shuffled.statistic == pytest.approx(reference.statistic, abs=_TOL)
    assert shuffled.p_value == pytest.approx(reference.p_value, abs=_TOL)


def test_a_singular_covariance_is_refused_not_reported():
    # The only target event happens after group A has left follow-up, so the
    # kernel's covariance is zero; cuminc() would print a statistic of -1.
    with pytest.raises(GrayTestError, match="singular"):
        gray_test([1.0, 2.0, 3.0], [0, 1, 0], ["A", "B", "B"])
