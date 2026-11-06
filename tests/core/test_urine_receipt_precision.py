"""Urine receipts keep raw-path clock precision on every load path.

lead-r13: the DuckDB aggregation path stored the ``_urine_*_lag_h`` receipts
as float32, while the raw path kept float64.  ``*_available_at`` then differed
by up to ~1e-6 h, and one MIIV stay's exact 12 h record gap became 12.0 on one
path and 12.000000000000007 on the other, so urine24 kept the window on one
path and dropped it on the other.
"""

import numpy as np
import pandas as pd

from easyicu.datasource import _downcast_value_columns
from easyicu.scores.urine_output import daily_volume_batch


def test_receipt_lags_stay_float64_values_are_downcast():
    df = pd.DataFrame({
        "stay_id": [1, 2], "charttime": [1.0, 2.0], "urine": [100.0, 250.5],
        "_urine_source_lag_h": [0.25833333333333336] * 2,
        "_urine_first_lag_h": [0.25833333333333336] * 2,
        "_urine_availability_lag_h": [0.4083333333333333] * 2,
    })
    _downcast_value_columns(df, {"stay_id", "charttime"})
    assert df["urine"].dtype == np.float32
    for c in ("_urine_source_lag_h", "_urine_first_lag_h", "_urine_availability_lag_h"):
        assert df[c].dtype == np.float64
    assert df["_urine_availability_lag_h"].iloc[0] == 0.4083333333333333


def _windows(gap):
    hours = np.arange(40, 59, dtype=float)
    hours = np.r_[hours, 70.0, np.arange(71, 95, dtype=float)]
    gaps = np.r_[np.nan, np.ones(len(hours) - 1)]
    gaps[hours == 70.0] = gap
    df = pd.DataFrame({
        "stay_id": 1, "charttime": hours, "urine": 50.0,
        "_urine_available_at": hours + 0.4, "_urine_gap_h": gaps,
    })
    return daily_volume_batch(df, "stay_id", "charttime", "urine", 1.0, 24, 18, 12)


def test_an_exact_12h_gap_with_float_noise_is_an_exact_12h_gap():
    exact = _windows(12.0)
    noisy_up = _windows((70 + 0.25833333333333336) - (58 + 0.25833333333333336))
    noisy_down = _windows(12.0 - 7e-15)
    assert noisy_up["urine24"].notna().sum() > 0
    pd.testing.assert_frame_equal(noisy_up, exact)
    pd.testing.assert_frame_equal(noisy_down, exact)


def test_a_gap_one_second_over_12h_still_starts_a_new_segment():
    exact = _windows(12.0)
    over = _windows(12.0 + 1 / 3600)
    assert not exact["urine24"].equals(over["urine24"])
