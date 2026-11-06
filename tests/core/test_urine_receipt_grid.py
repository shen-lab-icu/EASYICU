"""Urine receipt hours do not depend on the order the clocks were added."""

import numpy as np
import pandas as pd

from easyicu.scores.urine_output import on_receipt_grid, source_availability


def test_sql_and_pandas_lag_orders_publish_the_same_receipt_hours():
    rng = np.random.default_rng(7)
    chart_s = rng.integers(0, 30 * 86400, 5000)
    entry_s = chart_s + rng.integers(0, 7200, 5000)
    bucket_h = np.floor(chart_s / 3600.0)
    clock_lag = chart_s / 3600.0 - bucket_h
    entry_lag = (entry_s - chart_s) / 3600.0
    # One path adds the clock lag first, the other sums both lags first.
    pandas_path = (bucket_h + clock_lag) + entry_lag
    sql_path = bucket_h + (clock_lag + entry_lag)
    assert not np.array_equal(pandas_path, sql_path)
    assert np.array_equal(on_receipt_grid(pandas_path), on_receipt_grid(sql_path))
    assert np.allclose(on_receipt_grid(pandas_path), entry_s / 3600.0, rtol=0, atol=1e-12)


def test_receipts_land_on_the_millisecond_grid():
    frame = pd.DataFrame({
        "stay_id": [1, 1], "charttime": [1.0, 2.0], "urine": [50.0, 60.0],
        "_urine_availability_lag_h": [0.1 + 0.2, 7 / 3600],
    })
    receipts = source_availability(frame, ["stay_id"], "charttime", None)
    assert receipts["_urine_available_at"].tolist() == [1.3, 2.0 + 7 / 3600]
