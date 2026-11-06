"""Exact decimal boundaries decide as the clinical text reads (lead-r13 float audit)."""

import numpy as np
import pandas as pd
import pytest

from easyicu.utils.thresholds import ge, gt, le, lt


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_an_exact_creatinine_rise_and_ratio_are_on_the_boundary(dtype):
    def load(values):
        return np.array(values, dtype=dtype).astype(float)

    assert ge(load([1.4, 1.2, 0.9, 1.39]), load([1.1, 0.9, 0.6, 1.1]) + 0.3).tolist() == [True, True, True, False]
    assert ge(load([1.2, 1.5, 0.9, 1.19]), load([0.8, 1.0, 0.6, 0.8]) * 1.5).tolist() == [True, True, True, False]
    assert ge(load([2.1, 3.3]), load([0.7, 1.1]) * 3.0).tolist() == [True, True]
    # The plain operators decide these boundaries by float noise.
    assert not (1.4 >= 1.1 + 0.3)
    assert not (1.2 >= 0.8 * 1.5)


@pytest.mark.parametrize("value, threshold, expected", [
    (0.5 - 1e-15, 0.5, {"ge": True, "gt": False, "le": True, "lt": False}),
    (0.5 + 1e-15, 0.5, {"ge": True, "gt": False, "le": True, "lt": False}),
    (0.5 - 1e-5, 0.5, {"ge": False, "gt": False, "le": True, "lt": True}),
    (0.5 + 1e-5, 0.5, {"ge": True, "gt": True, "le": False, "lt": False}),
    (400.0 - 4e-13, 400.0, {"ge": True, "gt": False, "le": True, "lt": False}),
    (float(np.float32(60.0)) / float(np.float32(0.15)), 400.0, {"ge": True, "gt": False, "le": True, "lt": False}),
    (400.01, 400.0, {"ge": True, "gt": True, "le": False, "lt": False}),
])
def test_operators_include_or_exclude_the_boundary(value, threshold, expected):
    got = {"ge": ge(value, threshold), "gt": gt(value, threshold),
           "le": le(value, threshold), "lt": lt(value, threshold)}
    assert {k: bool(v) for k, v in got.items()} == expected


def test_nullable_columns_keep_three_valued_logic():
    x = pd.Series([pd.NA, 0.3, 0.2], dtype="Float64")
    out = ge(x, 0.3)
    assert out.dtype == "boolean"
    assert out.isna().tolist() == [True, False, False]
    assert out.dropna().tolist() == [True, False]


def test_a_48h_window_limit_keeps_an_exact_48h_elapsed_time():
    # 48 h apart on minute clocks relative to an admission at hh:mm:30.
    elapsed = (50 + 1 / 120) - (2 + 1 / 120)
    assert le(elapsed, 48.0) and not gt(elapsed, 48.0)
    assert gt(48.0 + 1 / 3600, 48.0)


def test_missing_values_compare_false_and_series_keep_their_index():
    x = pd.Series([np.nan, 0.3, 0.2], index=[7, 8, 9])
    out = ge(x, 0.3)
    assert out.tolist() == [False, True, False]
    assert out.index.tolist() == [7, 8, 9]
    assert lt(x, 0.3).tolist() == [False, False, True]
