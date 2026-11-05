"""The id_tbl DuckDB fast path reads the table's configured value column.

MIMIC chartevents declare ``val_var = valuenum``.  The fast path (taken for
directory and bucketed tables) read the string ``value`` column instead and
required it to be non-null, so a row with only ``valuenum`` was dropped; the
Python path used for a single-file table kept it.  The raw fallback ran only
when the fast path returned no row at all, so a source whose ``value`` was
null for some stays and set for others lost those stays silently.  In the CareVue view 12 of
3000 sampled stays lost their weight once chartevents was bucketed.
"""

from __future__ import annotations

import pandas as pd
import pytest

import easyicu
from tests.support.bucket_storage import write_fixture_parquet

T = pd.Timestamp("2150-01-01 08:00")


def _write(tmp_path, layout):
    pd.DataFrame(
        {
            "row_id": [1, 2], "subject_id": [10, 20], "hadm_id": [100, 200],
            "icustay_id": [1000, 2000], "dbsource": ["carevue"] * 2,
            "first_careunit": ["MICU"] * 2, "last_careunit": ["MICU"] * 2,
            "first_wardid": [1, 1], "last_wardid": [1, 1],
            "intime": [T, T], "outtime": [T + pd.Timedelta(days=2)] * 2, "los": [2.0, 2.0],
        }
    ).to_parquet(tmp_path / "icustays.parquet")
    chart = pd.DataFrame(
        {
            "row_id": [1, 2, 3], "subject_id": [10, 10, 20], "hadm_id": [100, 100, 200],
            "icustay_id": [1000, 1000, 2000], "itemid": [580, 580, 580],
            "charttime": [T, T + pd.Timedelta(hours=1), T], "storetime": [T, T, T],
            "cgid": [1, 1, 1], "value": [None, None, "70"], "valuenum": [80.0, 82.0, 70.0],
            "valueuom": ["kg", "kg", "kg"], "warning": [0, 0, 0], "error": [0, 0, 0],
            "resultstatus": [None] * 3, "stopped": [None] * 3,
        }
    )
    if layout == "file":
        chart.to_parquet(tmp_path / "chartevents.parquet")
    elif layout == "directory":
        (tmp_path / "chartevents").mkdir()
        chart.to_parquet(tmp_path / "chartevents" / "1.parquet")
    else:
        bucket = tmp_path / "chartevents_bucket" / "bucket_id=0"
        bucket.mkdir(parents=True)
        write_fixture_parquet(chart, bucket / "part.parquet")


@pytest.mark.parametrize("layout", ["file", "directory", "bucket"])
def test_weight_uses_valuenum_when_value_is_null(tmp_path, layout):
    _write(tmp_path, layout)
    out = easyicu.load_concepts(
        ["weight"], database="mimic", data_path=str(tmp_path),
        patient_ids=[1000, 2000], merge=False, verbose=False,
    )
    out = out["weight"] if isinstance(out, dict) else out
    assert dict(zip(out["icustay_id"], out["weight"])) == {1000: 81.0, 2000: 70.0}
