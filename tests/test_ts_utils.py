import numpy as np
import pandas as pd

from easyicu.api import _get_auto_chunk_strategy
from easyicu.io.ts_utils import locb, locf, slide


def test_locf_sorts_by_time_before_fill_when_unsorted_and_no_max_gap():
    # Rows out of chronological order, max_gap omitted. LOCF must still carry the
    # earliest observation forward in time, not in (wrong) row order.
    df = pd.DataFrame({
        "stay_id": [1, 1, 1],
        "charttime": [10, 0, 5],          # deliberately unsorted
        "value": [np.nan, 7.0, np.nan],   # only the t=0 row is observed
    })
    out = locf(df, id_cols=["stay_id"], index_col="charttime").sort_values("charttime")
    # t=0 -> 7 (observed), t=5 and t=10 -> carried forward 7
    assert out["value"].tolist() == [7.0, 7.0, 7.0]


def test_locb_sorts_by_time_before_fill_when_unsorted_and_no_max_gap():
    df = pd.DataFrame({
        "stay_id": [1, 1, 1],
        "charttime": [10, 0, 5],
        "value": [9.0, np.nan, np.nan],   # only the t=10 row is observed
    })
    out = locb(df, id_cols=["stay_id"], index_col="charttime").sort_values("charttime")
    # t=10 -> 9 (observed), t=0 and t=5 -> carried backward 9
    assert out["value"].tolist() == [9.0, 9.0, 9.0]


def test_numeric_charttime_slide_uses_hours_for_long_cohorts():
    rows = []
    for stay_id in range(1, 26):
        rows.append({"stay_id": stay_id, "charttime": 0.0, "value": 5.0})
        rows.append({"stay_id": stay_id, "charttime": 1000.0, "value": 1.0})

    # A very long stay in the same cohort used to make the bulk slide path
    # misclassify all numeric charttime values as minutes instead of hours.
    rows.append({"stay_id": 99, "charttime": 10000.0, "value": 2.0})

    result = slide(
        pd.DataFrame(rows),
        ["stay_id"],
        "charttime",
        before=pd.Timedelta(hours=24),
        agg_func={"value": "max"},
    )

    observed = result.loc[
        (result["stay_id"] == 1) & (result["charttime"] == 1000.0),
        "value",
    ].item()
    assert observed == 1.0


def test_sofa_auto_chunk_size_is_capped_for_large_cohorts(monkeypatch):
    monkeypatch.setenv("EASYICU_AUTO_CHUNK_SIZE", "8000")
    monkeypatch.setattr("easyicu.runtime.memory_manager.get_available_memory_mb", lambda: 16 * 1024)

    strategy = _get_auto_chunk_strategy(
        ["sofa"],
        50000,
        merge=True,
        chunk_size=None,
        batch_size=None,
        parallel_workers=None,
        concept_workers=None,
    )

    assert strategy is not None
    assert strategy["chunk_size"] == 2000
