"""The parity census compares published frames byte for byte, counts only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.duckdb_raw_parity_census import compare_frames


def _frame(values, dtype="float64", times=(0.0, 1.0, 2.0)):
    return pd.DataFrame({"stay_id": [1] * len(times), "charttime": list(times),
                         "urine": np.asarray(values, dtype=dtype)})


def test_identical_frames_are_equal_including_missing_values():
    report = compare_frames(_frame([1.0, np.nan, 3.0]), _frame([1.0, np.nan, 3.0]))
    assert report["equal"] and report["value_cells_differ"] == 0


def test_one_ulp_is_a_difference():
    left = _frame([0.1 + 0.2, 1.0, 2.0])
    right = _frame([0.3, 1.0, 2.0])
    report = compare_frames(left, right)
    assert not report["equal"] and report["value_cells_differ"] == 1


def test_rows_and_dtypes_are_reported_without_identifiers():
    left = _frame([1, 2, 3], dtype="int64")
    right = _frame([1.0, 2.0], times=(0.0, 1.0))
    report = compare_frames(left, right)
    assert report["rows_only_left"] == 1 and report["rows_only_right"] == 0
    assert report["dtype_differs"] == ["urine"]
    assert not report["equal"]
    assert set(report) <= {"rows_left", "rows_right", "rows_only_left", "rows_only_right",
                           "value_cells_differ", "dtype_differs", "equal", "columns_differ"}


def test_nullable_and_text_values_compare_missing_as_equal():
    left = pd.DataFrame({"stay_id": [1, 1], "charttime": [0.0, 1.0],
                         "flag": pd.array([True, pd.NA], dtype="boolean")})
    right = left.copy()
    assert compare_frames(left, right)["equal"]
    right.loc[0, "flag"] = False
    assert compare_frames(left, right)["value_cells_differ"] == 1


def test_repeated_keys_are_matched_one_to_one():
    left = pd.DataFrame({"stay_id": [1, 1, 1], "charttime": [0.0, 0.0, 1.0],
                         "mech_vent": ["invasive", "noninvasive", "invasive"]})
    report = compare_frames(left, left.iloc[[1, 0, 2]].copy())
    assert report["equal"] and report["value_cells_differ"] == 0
