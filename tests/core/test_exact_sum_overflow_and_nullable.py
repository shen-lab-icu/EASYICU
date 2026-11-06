"""Exact sums: finite overflow is refused; nullable numeric columns are exact.

lead-r13 delta2 D2-EXACT. ``math.fsum`` raises ``OverflowError`` when a
partial sum of finite values overflows, which depends on the order. The former
fallback summed in floating point, so one order of a multiset could publish
``inf`` and another a finite total. Such a group is now summed as exact
rationals: the same correctly rounded total in every order. A group whose
exact total is beyond the float64 range is refused with a typed error. Pandas
nullable numeric columns (``Int64``, ``Float64``) used to skip the exact path;
they now convert to float64 (``NA`` → NaN) and take it. Synthetic values only.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from easyicu.utils.exact_sum import (
    EXACT_SUM_OVERFLOW,
    ExactSumOverflowError,
    exact_group_totals,
    exact_groupby_agg,
    exact_groupby_sum,
    exact_terms_to_values,
)

_BIG = np.finfo(np.float64).max


@pytest.mark.parametrize(
    "values,expected",
    [
        ([_BIG, _BIG, -_BIG], _BIG),
        ([_BIG, _BIG, -_BIG, -_BIG, 1.5], 1.5),
        ([_BIG, _BIG / 2, -_BIG, -_BIG / 4, 5e-324], math.fsum([_BIG / 2, -_BIG / 4, 5e-324])),
    ],
)
def test_a_finite_group_with_an_overflowing_partial_sum_is_exact_in_every_order(values, expected):
    # fsum itself raises in some of these orders and not in others.
    raised = set()
    for order in itertools.permutations(values):
        try:
            math.fsum(order)
            raised.add(False)
        except OverflowError:
            raised.add(True)
        n = len(order)
        totals, _ = exact_group_totals(np.zeros(n, dtype=np.int64), np.array(order), 1)
        assert totals[0] == expected
    assert raised == {True, False}


@pytest.mark.parametrize("values", [[_BIG, _BIG, _BIG], [_BIG, _BIG / 2**53, 0.0]])
def test_a_finite_group_whose_exact_total_is_out_of_range_is_refused(values):
    for order in itertools.permutations(values):
        with pytest.raises(ExactSumOverflowError) as caught:
            exact_group_totals(np.zeros(3, dtype=np.int64), np.array(order), 1)
    assert caught.value.reason == EXACT_SUM_OVERFLOW == "exact_sum_overflow"
    assert caught.value.n_values == 3
    assert str(caught.value).startswith("exact_sum_overflow:")
    # Not an OverflowError: a caller catching that for a float fallback must
    # not swallow the refusal.
    assert not isinstance(caught.value, OverflowError)


def test_the_refusal_reaches_the_grouped_and_sql_term_paths():
    frame = pd.DataFrame({"k": [1, 1, 1, 2], "v": [_BIG, _BIG, _BIG, 1.0]})
    with pytest.raises(ExactSumOverflowError):
        exact_groupby_sum(frame, ["k"], "v")
    with pytest.raises(ExactSumOverflowError):
        exact_groupby_agg(frame, ["k"], {"v": "mean"})
    with pytest.raises(ExactSumOverflowError):
        exact_terms_to_values(pd.Series([[_BIG, _BIG, _BIG]]), "sum")


@pytest.mark.parametrize(
    "values,expected",
    [
        ([np.inf, 1.0, 2.0], np.inf),
        ([np.inf, -np.inf, 1.0], np.nan),
        ([-np.inf, _BIG, _BIG], -np.inf),
    ],
)
def test_non_finite_groups_keep_ieee_semantics_in_every_order(values, expected):
    for order in itertools.permutations(values):
        totals, _ = exact_group_totals(np.zeros(3, dtype=np.int64), np.array(order), 1)
        if np.isnan(expected):
            assert np.isnan(totals[0])
        else:
            assert totals[0] == expected


def test_a_finite_group_near_the_limit_without_overflow_is_still_exact():
    values = [_BIG / 2, _BIG / 4, -_BIG / 8, 1e-300]
    for order in itertools.permutations(values):
        totals, _ = exact_group_totals(np.zeros(4, dtype=np.int64), np.array(order), 1)
        assert totals[0] == math.fsum(values)


@pytest.mark.parametrize("dtype", ["Int64", "Int32", "Float64", "Float32", "UInt16"])
def test_nullable_numeric_columns_take_the_exact_path(dtype):
    raw = [1, 2, None, 3, 4, None, 5]
    keys = [1, 1, 1, 1, 2, 2, 2]
    nullable = pd.DataFrame({"k": keys, "v": pd.array(raw, dtype=dtype)})
    plain = pd.DataFrame({"k": keys, "v": np.array(raw, dtype=float)})

    for how in ("sum", "mean"):
        got = exact_groupby_agg(nullable, ["k"], {"v": how})
        want = exact_groupby_agg(plain, ["k"], {"v": how})
        assert got["v"].dtype == np.float64
        pd.testing.assert_frame_equal(got, want, check_exact=True)

    got = exact_groupby_sum(nullable, ["k"], "v", min_count=1)
    want = exact_groupby_sum(plain, ["k"], "v", min_count=1)
    assert got["v"].dtype == np.float64
    pd.testing.assert_frame_equal(got, want, check_exact=True)


def test_a_nullable_float_column_is_order_independent():
    # 0.1 + 0.2 + 0.3 in floating point depends on the order; the exact sum
    # does not. A Float64 column used to be summed by pandas directly.
    values = [0.1, 0.2, 0.3, None]
    results = set()
    for order in itertools.permutations(values):
        frame = pd.DataFrame({"k": [1] * 4, "v": pd.array(list(order), dtype="Float64")})
        results.add(exact_groupby_sum(frame, ["k"], "v")["v"].iloc[0])
    assert results == {math.fsum([0.1, 0.2, 0.3])}


def test_a_nullable_boolean_column_is_not_summed_as_a_number():
    frame = pd.DataFrame({"k": [1, 1, 1], "v": pd.array([True, None, True], dtype="boolean")})
    out = exact_groupby_sum(frame, ["k"], "v")
    pd.testing.assert_frame_equal(
        out, frame.groupby(["k"], sort=False, as_index=False)["v"].sum(), check_exact=True
    )


def test_ordinary_float_input_is_byte_identical_to_the_previous_algorithm():
    # The former algorithm on inputs without overflow: bincount for groups of
    # at most two values, fsum otherwise. Ordinary float64 input must publish
    # exactly the same bits.
    rng = np.random.default_rng(20261106)
    n = 5000
    keys = rng.integers(0, 400, size=n)
    values = rng.normal(100.0, 50.0, size=n) * rng.choice([1e-6, 1.0, 1e6], size=n)
    values[rng.random(n) < 0.05] = np.nan
    frame = pd.DataFrame({"k": keys, "v": values})

    def previous(frame):
        grouped = frame.groupby("k", sort=False)
        out = []
        for _, group in grouped["v"]:
            kept = group.dropna().to_numpy()
            out.append(float(kept.sum()) if len(kept) <= 2 else math.fsum(kept.tolist()))
        return np.array(out)

    got = exact_groupby_sum(frame, ["k"], "v")["v"].to_numpy()
    want = previous(frame)
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("values", [[_BIG, _BIG], [-_BIG, -_BIG / 2**52]])
def test_a_two_value_group_beyond_the_range_is_refused_too(values):
    with pytest.raises(ExactSumOverflowError) as caught:
        exact_group_totals(np.zeros(2, dtype=np.int64), np.array(values), 1)
    assert caught.value.reason == EXACT_SUM_OVERFLOW
    with pytest.raises(ExactSumOverflowError):
        exact_groupby_sum(pd.DataFrame({"k": [1, 1], "v": values}), ["k"], "v")


@pytest.mark.parametrize(
    "values,expected",
    [([_BIG], _BIG), ([_BIG, -_BIG], 0.0), ([np.inf, _BIG], np.inf), ([np.inf, -np.inf], np.nan)],
)
def test_small_groups_keep_exact_and_ieee_results(values, expected):
    totals, _ = exact_group_totals(np.zeros(len(values), dtype=np.int64), np.array(values), 1)
    assert (np.isnan(totals[0]) and np.isnan(expected)) or totals[0] == expected


def test_no_path_falls_back_to_a_floating_point_summation():
    import inspect

    from easyicu.utils import exact_sum

    source = inspect.getsource(exact_sum)
    assert "np.sum(" not in source and ".sum()" not in source
