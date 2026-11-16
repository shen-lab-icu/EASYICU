"""MIMIC microbiologyevents: stay assignment inside a multi-stay admission.

``microbiologyevents`` is keyed by ``hadm_id`` with a date index
(``chartdate``) and a precise ``charttime`` that is missing for date-only
specimens. When one admission holds several ICU stays the loader assigns each
row by a rolling join on ``outtime`` (ricu roll = -Inf, rollends = TRUE).

Two losses on the public MIMIC-III demo (``samp`` versus ricu 0.6.3):

* a specimen without ``charttime`` was dropped before the join, although its
  ``chartdate`` places it (ricu keeps it and indexes it at chartdate + 12 h);
* a specimen taken a few hours after one stay's discharge was handed to the
  next stay of the admission, days later, whose episode window excludes it,
  so it vanished although the earlier stay's 24 h post-discharge allowance
  (utils/icu_window.py) covers it. ricu keeps it with the earlier stay
  because ricu rolls on the date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.config import DataSourceRegistry
from easyicu.datasource import ICUDataSource

REPO = Path(__file__).resolve().parents[2]
ts = pd.Timestamp


def _source(root: Path, database: str = "mimic") -> ICUDataSource:
    config = DataSourceRegistry.from_json(REPO / "src/easyicu/data/data-sources.json").get(database)
    return ICUDataSource(config=config, base_path=root)


def _write_tables(root: Path) -> None:
    # One admission, two ICU stays: stay 1 ends 01-03 09:00, stay 2 starts
    # 01-08 10:00 (the gap is longer than both 24 h allowances).
    pd.DataFrame(
        {
            "row_id": [1, 2],
            "subject_id": [9, 9],
            "hadm_id": [100, 100],
            "icustay_id": [1, 2],
            "dbsource": ["carevue", "carevue"],
            "first_careunit": ["MICU", "MICU"],
            "last_careunit": ["MICU", "MICU"],
            "first_wardid": [1, 1],
            "last_wardid": [1, 1],
            "intime": [ts("2101-01-01 15:00"), ts("2101-01-08 10:00")],
            "outtime": [ts("2101-01-03 09:00"), ts("2101-01-10 09:00")],
            "los": [1.75, 1.96],
        }
    ).to_parquet(root / "icustays.parquet", index=False)
    rows = [
        # (row_id, chartdate, charttime, label)
        (11, "2101-01-02", "2101-01-02 08:00", "inside stay 1"),
        (12, "2101-01-02", None, "date only, during stay 1"),
        (13, "2101-01-03", "2101-01-03 14:00", "5 h after stay 1 discharge"),
        (14, "2101-01-05", "2101-01-05 12:00", "between stays, outside both windows"),
        (15, "2101-01-07", "2101-01-07 22:00", "12 h before stay 2"),
        (16, "2101-01-07", None, "date only, the day before stay 2"),
        (17, "2101-01-09", "2101-01-09 06:00", "inside stay 2"),
    ]
    pd.DataFrame(
        {
            "row_id": [r[0] for r in rows],
            "subject_id": [9] * len(rows),
            "hadm_id": [100] * len(rows),
            "chartdate": [ts(r[1]) for r in rows],
            "charttime": pd.to_datetime([r[2] for r in rows]),
            "spec_itemid": [70012] * len(rows),
            "spec_type_desc": [r[3] for r in rows],
            "org_itemid": pd.array([80002, None, None, 80002, None, None, 80002], dtype="Int64"),
            "org_name": ["E. COLI", None, None, "E. COLI", None, None, "E. COLI"],
            "isolate_num": pd.array([1, None, None, 1, None, None, 1], dtype="Int64"),
        }
    ).to_parquet(root / "microbiologyevents.parquet", index=False)


def _assigned(root: Path) -> dict:
    frame = _source(root).load_table("microbiologyevents", verbose=False).data
    assert frame["row_id"].is_unique, "each specimen keeps exactly one row"
    return dict(zip(frame["spec_type_desc"], frame["icustay_id"].astype(int)))


def test_date_only_specimen_in_a_multi_stay_admission_is_kept(tmp_path):
    _write_tables(tmp_path)
    assigned = _assigned(tmp_path)

    # Without charttime the row rolls on its chartdate, as in ricu.
    assert assigned.get("date only, during stay 1") == 1
    assert assigned.get("date only, the day before stay 2") == 2


def test_post_discharge_specimen_stays_with_the_stay_whose_allowance_covers_it(tmp_path):
    _write_tables(tmp_path)
    assigned = _assigned(tmp_path)

    assert assigned.get("5 h after stay 1 discharge") == 1


def test_forward_roll_is_unchanged_where_the_next_stay_window_covers_the_row(tmp_path):
    _write_tables(tmp_path)
    assigned = _assigned(tmp_path)

    assert assigned["inside stay 1"] == 1
    assert assigned["inside stay 2"] == 2
    # Within stay 2's 24 h pre-admission window: the ricu forward roll stands.
    assert assigned["12 h before stay 2"] == 2
    # Covered by neither window: the forward roll stands (the episode window
    # later drops it); no allowance is stretched to rescue it.
    assert assigned["between stays, outside both windows"] == 2
    assert len(assigned) == 7
