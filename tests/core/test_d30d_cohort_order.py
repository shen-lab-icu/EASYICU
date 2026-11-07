"""Stay discovery returns IDs in source order, deduplicated, then capped (delta30d).

B-D30-COHORT-ORDER: IDs were read per physical bucket part, deduplicated
and capped without restoring source order. A capped cohort therefore
differed between a flat and a bucket installation of the same source. IDs
now follow the source ordinal the normal readers restore; they are
deduplicated in that order, and the cap comes last. Flat, sharded and
bucket installations of one source give one cohort, with or without a cap
and with duplicate rows. Rows are synthetic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from easyicu import storage_layout as sl


def _discover(root, cap=None):
    from easyicu.api import get_all_patient_ids

    listing: dict = {}
    ids, _column = get_all_patient_ids(root, database="mimic", max_patients=cap, listing=listing)
    return ids, listing


# Source order: not sorted, with duplicates (a stay listed twice).
SOURCE = [*range(530, 550), *range(500, 520), 535, 505]
EXPECTED = list(dict.fromkeys(SOURCE))


def _flat(root):
    root.mkdir(parents=True)
    pd.DataFrame({"icustay_id": SOURCE}).to_parquet(root / "icustays.parquet", index=False)


def _shards(root):
    (root / "icustays").mkdir(parents=True)
    pd.DataFrame({"icustay_id": SOURCE[:25]}).to_parquet(root / "icustays" / "part-0.parquet", index=False)
    pd.DataFrame({"icustay_id": SOURCE[25:]}).to_parquet(root / "icustays" / "part-1.parquet", index=False)


def _bucket(root):
    """A range bucket whose parts are sorted by key and carry the source ordinal."""
    frame = pd.DataFrame({"icustay_id": SOURCE, sl.ORDINAL: np.arange(len(SOURCE), dtype="int64")})
    bucket = root / "icustays_bucket"
    bucket.mkdir(parents=True)
    (bucket / "_COMPLETE").write_text("synthetic")
    (bucket / sl.BUILD_RECEIPT).write_text("{}")
    inventory = []
    for i, part in enumerate([frame[frame.icustay_id <= 520], frame[frame.icustay_id > 520]]):
        path = bucket / f"bucket_id={i}" / "part.parquet"
        path.parent.mkdir()
        part.sort_values("icustay_id").to_parquet(path, index=False)
        inventory.append({"relative_path": f"bucket_id={i}/part.parquet",
                          "size": path.stat().st_size, "row_count": len(part)})
    (bucket / sl.MANIFEST).write_text(json.dumps({
        "schema": sl.SCHEMA, "scheme": "range", "key_column": "icustay_id", "key_type": "BIGINT",
        "range_breakpoints": [520], "sort_keys": ["icustay_id", sl.ORDINAL],
        "source_order": sl.ORDINAL, "row_group_target": 65536,
        "builder_version": sl.BUILDER_VERSION, "inventory": inventory,
    }))


@pytest.mark.parametrize("cap", [None, 10, 25])
def test_flat_shards_and_bucket_of_one_source_give_one_cohort(tmp_path, cap):
    results = {}
    for name, build in (("flat", _flat), ("shards", _shards), ("bucket", _bucket)):
        root = tmp_path / name
        build(root)
        results[name] = _discover(root, cap)
    expected = EXPECTED if cap is None else EXPECTED[:cap]
    for name, (ids, listing) in results.items():
        assert ids == expected, name
        assert listing == {"order": "source_file_order", "source_format": "parquet"}, name
