"""R7-F014: MIMIC-III CareVue charted rates (mimic_rate_cv), synthetic data only.

The published hourly rate must not depend on the dtype or the row order the
table reader delivers. S732TW integrates each setting over active time; an
amount-only row does not truncate it, and an unknown terminal setting adds
no fabricated active time.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from easyicu.utils.callback_utils import mimic_rate_cv

INTIME = pd.Timestamp("2100-01-01 00:20")


def _run(rows, value_dtype=float):
    frame = pd.DataFrame(rows, columns=["row_id", "icustay_id", "minute", "linkorderid", "rate"])
    frame["charttime"] = INTIME + pd.to_timedelta(frame.pop("minute"), unit="min")
    frame["rate"] = frame["rate"].astype(value_dtype)
    frame = frame[["row_id", "icustay_id", "charttime", "rate", "linkorderid"]]
    out = mimic_rate_cv(
        frame, val_col="rate", grp_var="linkorderid", id_cols=["icustay_id"],
        admission_times=pd.DataFrame({"icustay_id": [1], "intime": [INTIME]}),
    )
    out = out.dropna(subset=["rate"])
    hours = (out["charttime"] - INTIME) / pd.Timedelta(hours=1)
    return dict(zip(hours.round(6), pd.to_numeric(out["rate"])))


def test_object_dtype_rates_integrate_same_hour_settings():
    rows = [(1, 1, 5, 7, "1.0"), (2, 1, 40, 7, "3.0"), (3, 1, 190, 7, "2.0")]
    as_object = _run(rows, value_dtype=object)
    as_float = _run([(*r[:4], float(r[4])) for r in rows])
    assert as_object == as_float
    assert as_object[0.0] == pytest.approx((35 + 3 * 20) / 55)
    assert as_object[1.0] == pytest.approx(3.0) and as_object[2.0] == pytest.approx(3.0)


def test_latest_chart_of_an_hour_carries_forward_whatever_the_row_order():
    rows = [(1, 1, 5, 7, 1.0), (2, 1, 40, 7, 3.0), (3, 1, 190, 7, 2.0)]
    for order in itertools.permutations(rows):
        out = _run(list(order))
        assert out[1.0] == pytest.approx(3.0)
        assert out[2.0] == pytest.approx(3.0)
        assert out[3.0] == pytest.approx(3.0)  # active until the unknown terminal point


def test_amount_only_row_at_the_rate_time_does_not_truncate_the_infusion():
    # CareVue writes the amount-only row (rate NA) first, then the rate row.
    rows = [(1, 1, 10, 7, np.nan), (2, 1, 10, 7, 1.0), (3, 1, 190, 7, 2.0)]
    results = [_run(list(order)) for order in itertools.permutations(rows)]
    assert all(r == results[0] for r in results)
    assert results[0] == {0.0: 1.0, 1.0: 1.0, 2.0: 1.0, 3.0: 1.0}


def test_stop_and_unknown_singleton_contribute_no_active_span():
    rows = [(1, 1, 10, 7, 0.0), (2, 1, 10 + 9 * 60, 7, 1.5)]
    out = _run(rows)
    assert out == {}


def test_two_runs_over_the_same_rows_are_bitwise_identical():
    # The reader may return the same rows in a different order on each run
    # (parallel DuckDB scan); the published rates must not change by a bit.
    rows = [(i, 1, m, 7 + (i % 2), float(r)) for i, (m, r) in enumerate(
        [(5, 1.0), (5, np.nan), (40, 3.0), (95, 2.5), (130, 0.0), (190, 2.0), (190, np.nan), (400, 4.25)], start=1)]
    runs = []
    for seed in (1, 2):
        order = np.random.default_rng(seed).permutation(len(rows))
        runs.append(_run([rows[i] for i in order]))
    assert list(runs[0]) == list(runs[1])
    assert np.array_equal(np.array(list(runs[0].values())), np.array(list(runs[1].values())), equal_nan=True)
