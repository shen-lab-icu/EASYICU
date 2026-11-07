"""A table installed both flat (``table/``) and bucketed (``table_bucket/``)
is read from exactly one layout, never both.

The two directories are alternative installations of the same rows: the
bucket cache is built from the flat table.  The storage policy owner
(``storage_layout.resolve_table_files``) chooses the validated bucket
installation and otherwise the flat table, and every reader consumes that
one set, so rows are never summed across layouts (final review U01).  An
explicit error would refuse an ordinary, valid installation, so the policy
reads one layout instead.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.datasource import FilterOp, FilterSpec, ICUDataSource
from easyicu.resources import load_data_sources
from easyicu.scores.outcomes import _raw_table
from tests.support.bucket_storage import bind_fixture_source, write_contract

T = pd.Timestamp("2150-01-01 08:00")


def _stays():
    return pd.DataFrame({
        "row_id": [1, 2], "subject_id": [10, 20], "hadm_id": [100, 200],
        "icustay_id": [1000, 2000], "intime": [T, T],
        "outtime": [T + pd.Timedelta(days=2)] * 2,
    })


def _chart():
    return pd.DataFrame({
        "row_id": [1, 2, 3], "subject_id": [10, 10, 20], "hadm_id": [100, 100, 200],
        "icustay_id": [1000, 1000, 2000], "itemid": [580, 580, 580],
        "charttime": [T, T + pd.Timedelta(hours=1), T], "valuenum": [80.0, 82.0, 70.0],
    })


def _both_layouts(root, table, frame, key):
    (root / table).mkdir()
    frame.to_parquet(root / table / "1.parquet")
    bucket = root / f"{table}_bucket" / "bucket_id=0"
    bucket.mkdir(parents=True)
    frame.to_parquet(bucket / "part.parquet")
    write_contract(root / f"{table}_bucket", key=key)
    bind_fixture_source(root / f"{table}_bucket", table)


def _source(root):
    return ICUDataSource(config=load_data_sources().get("mimic"), base_path=root)


@pytest.mark.parametrize("filters", [None, "many"])
def test_load_table_reads_one_layout(tmp_path, filters):
    _both_layouts(tmp_path, "chartevents", _chart(), "itemid")
    spec = (
        [FilterSpec(column="icustay_id", op=FilterOp.IN, value=[1000, 2000, *range(5000, 5150)])]
        if filters else None
    )
    frame = _source(tmp_path).load_table("chartevents", filters=spec).data
    assert sorted(frame["valuenum"]) == [70.0, 80.0, 82.0]


def test_the_policy_resolves_only_the_bucket_installation(tmp_path):
    _both_layouts(tmp_path, "chartevents", _chart(), "itemid")
    files = _source(tmp_path).resolve_target_files("chartevents")
    assert {p.parent.parent.name for p in files} == {"chartevents_bucket"}


def test_outcome_raw_table_reads_one_layout(tmp_path):
    _both_layouts(tmp_path, "icustays", _stays(), "icustay_id")
    stays = _raw_table("mimic", tmp_path, "icustays")
    assert stays["icustay_id"].tolist() == [1000, 2000]
