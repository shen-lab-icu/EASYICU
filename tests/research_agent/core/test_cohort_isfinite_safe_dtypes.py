"""Regression: the analysis cohort is materialised with numpy-numeric dtypes so
generated analysis code never crashes on ``np.isfinite``.

Root cause (2026-07-06, H2 vasopressor causal): the universe builder emits
per-concept aggregates (e.g. ``icu_readmission_{max,min,mean,n,first}``) as
pandas *nullable* extension dtypes (``boolean`` / ``Float64`` / ``Int64``) or as
object columns holding python bools, because the aggregate is mostly null.
Generated causal code does ``design_df[col].to_numpy()`` and feeds the result to
``np.isfinite``; on a nullable/object array numpy raises
``ufunc 'isfinite' not supported for the input types`` — the propensity balance
table dies, ``adjusted_effect`` is ``None``, and the primary estimate is lost
(the headline then mis-binds to an audit step's junk scalar).

``coerce_isfinite_safe_dtypes`` downcasts nullable numeric/logical columns with
NA to ``float64`` (NA -> NaN) at cohort-materialisation time. Complete logical
columns remain numpy ``bool`` so the physical analysis cohort does not drift
from its sealed boolean domain. Genuine string categoricals (``sex``, admission
type) remain untouched for dummy-encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.research_agent.cohort.schema import coerce_isfinite_safe_dtypes


def _mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plain_float": np.array([1.0, 2.0, np.nan], dtype="float64"),
            "plain_int": np.array([1, 2, 3], dtype="int64"),
            "nullable_int": pd.array([1, pd.NA, 3], dtype="Int64"),
            "nullable_float": pd.array([0.0, pd.NA, 1.5], dtype="Float64"),
            "nullable_bool": pd.array([True, pd.NA, False], dtype="boolean"),
            "object_bool": pd.Series([True, None, False], dtype=object),
            "sex": pd.Series(["Male", "Female", "Male"], dtype=object),
            "adm": pd.Series(["surg", None, "med"], dtype=object),
        }
    )


def test_raw_nullable_and_object_bool_crash_isfinite():
    # Documents the failure mode the fix targets.
    frame = _mixed_frame()
    for col in ("nullable_bool", "object_bool"):
        try:
            np.isfinite(frame[col].to_numpy())
        except TypeError:
            continue
        raise AssertionError(f"expected {col} to crash np.isfinite before coercion")


def test_coercion_makes_every_non_string_column_isfinite_safe():
    out = coerce_isfinite_safe_dtypes(_mixed_frame())
    for col in out.columns:
        if col in ("sex", "adm"):
            continue  # genuine string categoricals stay object for dummy-encoding
        # Must not raise; every remaining column is a numpy numeric array.
        np.isfinite(out[col].to_numpy())


def test_nullable_columns_become_float64_with_nan_preserved():
    out = coerce_isfinite_safe_dtypes(_mixed_frame())
    for col in ("nullable_int", "nullable_float", "nullable_bool", "object_bool"):
        assert out[col].dtype == np.float64, (col, out[col].dtype)
    # NA -> NaN, values preserved
    int_vals = out["nullable_int"].to_numpy()
    assert int_vals[0] == 1.0 and int_vals[2] == 3.0
    assert np.isnan(int_vals[1])
    bool_vals = out["nullable_bool"].to_numpy()
    assert bool_vals[0] == 1.0 and bool_vals[2] == 0.0
    assert np.isnan(bool_vals[1])


def test_complete_logical_columns_preserve_boolean_domain():
    frame = pd.DataFrame(
        {
            "nullable_bool": pd.array([True, False, True], dtype="boolean"),
            "object_bool": pd.Series([True, False, True], dtype=object),
        }
    )

    out = coerce_isfinite_safe_dtypes(frame)

    for col in frame.columns:
        assert out[col].dtype == np.bool_, (col, out[col].dtype)
        assert out[col].tolist() == [True, False, True]
        np.isfinite(out[col].to_numpy())


def test_string_categoricals_are_left_untouched():
    out = coerce_isfinite_safe_dtypes(_mixed_frame())
    assert out["sex"].dtype == object
    assert out["adm"].dtype == object
    assert out["sex"].tolist() == ["Male", "Female", "Male"]


def test_all_numeric_frame_returned_unchanged_identity():
    # No coercible columns -> return the same object (no needless copy).
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": np.array([3, 4], dtype="int64")})
    assert coerce_isfinite_safe_dtypes(frame) is frame


def test_non_dataframe_passthrough():
    assert coerce_isfinite_safe_dtypes(None) is None
