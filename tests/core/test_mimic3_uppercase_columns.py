"""MIMIC-III exports may spell every column in upper case.

2026-11-03 extraction audit round 2 (R2-01): the id_tbl DuckDB fast path
returned ``ICUSTAY_ID`` from an upper-case chartevents export, and weight
(then demographics, renal and sofa2_score) failed with a missing
``icustay_id``. The query now aliases the configured id spelling.
"""

from __future__ import annotations

import pandas as pd

import easyicu
from tests.support.bucket_storage import write_fixture_parquet


def test_weight_from_upper_case_chartevents_keeps_the_configured_id(tmp_path):
    t = pd.Timestamp("2150-01-01 08:00")
    pd.DataFrame(
        {
            "ROW_ID": [1, 2], "SUBJECT_ID": [10, 20], "HADM_ID": [100, 200],
            "ICUSTAY_ID": [1000, 2000], "DBSOURCE": ["metavision"] * 2,
            "FIRST_CAREUNIT": ["MICU"] * 2, "LAST_CAREUNIT": ["MICU"] * 2,
            "FIRST_WARDID": [1, 1], "LAST_WARDID": [1, 1],
            "INTIME": [t, t], "OUTTIME": [t + pd.Timedelta(days=2)] * 2, "LOS": [2.0, 2.0],
        }
    ).to_parquet(tmp_path / "icustays.parquet")
    bucket = tmp_path / "chartevents_bucket" / "bucket_id=0"
    bucket.mkdir(parents=True)
    _chart = pd.DataFrame(
        {
            "ROW_ID": [1, 2, 3], "SUBJECT_ID": [10, 10, 20], "HADM_ID": [100, 100, 200],
            "ICUSTAY_ID": [1000, 1000, 2000], "ITEMID": [226512, 226512, 762],
            "CHARTTIME": [t, t + pd.Timedelta(hours=1), t], "STORETIME": [t, t, t],
            "CGID": [1, 1, 1], "VALUE": ["80", "82", "70"], "VALUENUM": [80.0, 82.0, 70.0],
            "VALUEUOM": ["kg", "kg", "kg"], "WARNING": [0, 0, 0], "ERROR": [0, 0, 0],
            "RESULTSTATUS": [None] * 3, "STOPPED": [None] * 3,
        }
    )
    write_fixture_parquet(_chart, bucket / "part.parquet")

    out = easyicu.load_concepts(
        ["weight"], database="mimic", data_path=str(tmp_path),
        patient_ids=[1000, 2000], merge=False, verbose=False,
    )
    out = out["weight"] if isinstance(out, dict) else out

    assert "icustay_id" in out.columns and "ICUSTAY_ID" not in out.columns
    assert dict(zip(out["icustay_id"], out["weight"])) == {1000: 81.0, 2000: 70.0}
