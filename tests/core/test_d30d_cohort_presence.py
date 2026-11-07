"""Stay discovery has one presence owner: absent, valid or broken (delta30d).

B-D30-COHORT-PRESENCE: an empty ``icustays.parquet/`` or ``icustays/``
directory beside a CSV let the CSV stand in. Dangling links under ``icu/``
or ``hosp/``, a dangling ``icu`` root and an upper-case name were outside
the candidates checked. ``storage_layout.table_presence`` now returns
absent, valid or broken for every path and case the owner's contract
covers. Broken refuses; only absent falls back to the CSV. A healthy
zero-row Parquet is valid. Rows are synthetic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.storage_layout import StorageIntegrityError

UNREADABLE = "table_scan_file_unreadable"
BROKEN = "table_layout_broken"


def _discover(root, cap=None):
    from easyicu.api import get_all_patient_ids

    listing: dict = {}
    ids, _column = get_all_patient_ids(root, database="mimic", max_patients=cap, listing=listing)
    return ids, listing


def _with_csv(root):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"icustay_id": [900, 901]}).to_csv(root / "icustays.csv", index=False)
    return root


def _dangling(path, root):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(root / "nowhere")


@pytest.mark.parametrize("make,reason", [
    (lambda r: (r / "icustays.parquet").mkdir(), BROKEN),  # empty dir named .parquet
    (lambda r: (r / "icustays").mkdir(), BROKEN),  # empty table directory
    (lambda r: _dangling(r / "icu" / "icustays.parquet", r), UNREADABLE),
    (lambda r: _dangling(r / "hosp" / "icustays.parquet", r), UNREADABLE),
    (lambda r: _dangling(r / "icu" / "icustays", r), UNREADABLE),
    (lambda r: _dangling(r / "icu" / "icustays_bucket", r), UNREADABLE),
    (lambda r: _dangling(r / "icu", r), UNREADABLE),  # a dangling root above every leaf
    (lambda r: _dangling(r / "ICUSTAYS.parquet", r), UNREADABLE),  # the upper-case name
], ids=["empty_parquet_dir", "empty_table_dir", "icu_flat", "hosp_flat", "icu_table",
        "icu_bucket", "icu_root", "upper_case"])
def test_a_present_but_broken_stay_table_is_refused_not_replaced_by_the_csv(tmp_path, make, reason):
    root = _with_csv(tmp_path / "mimic")
    make(root)
    with pytest.raises(StorageIntegrityError) as error:
        _discover(root)
    assert getattr(error.value, "reason", "") == reason


def test_a_healthy_zero_row_parquet_is_a_valid_empty_stay_table(tmp_path):
    root = _with_csv(tmp_path / "mimic")
    pd.DataFrame({"icustay_id": pd.Series([], dtype="int64")}).to_parquet(root / "icustays.parquet", index=False)
    ids, listing = _discover(root)
    assert ids == [] and listing["source_format"] == "parquet"
