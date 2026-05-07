import pandas as pd

from easyicu.api import _get_auto_chunk_strategy
from easyicu.ts_utils import slide


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
    monkeypatch.setattr("easyicu.memory_manager.get_available_memory_mb", lambda: 16 * 1024)

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
