"""A bound, seal-shaped bucket-only farm opens its physical source lineage.

Each case first proves the healthy reader path (fails at 01af932fe), then
checks a damaged installation through the same real resolver. Synthetic only.
"""

import hashlib
import json
import os

import pandas as pd
import pytest

from easyicu import storage_binding as sb
from easyicu import storage_layout as sl
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource


def _farm(root):
    install = root / "physical"
    install.mkdir()
    flat = install / "chartevents.parquet"
    frame = pd.DataFrame({
        "icustay_id": range(100, 120), "itemid": [220045] * 20,
        "value": ["60"] * 20,
    })
    frame.to_parquet(flat, index=False)
    bucket = install / "chartevents_bucket"
    (bucket / "bucket_id=0").mkdir(parents=True)
    part = bucket / "bucket_id=0" / "part.parquet"
    frame.assign(**{sl.ORDINAL: range(len(frame))}).to_parquet(part, index=False)
    (bucket / "_COMPLETE").write_text("synthetic")
    (bucket / sl.BUILD_RECEIPT).write_text("{}")
    stat = flat.stat()
    manifest = {
        "schema": sl.SCHEMA, "scheme": "range", "key_column": "icustay_id",
        "key_type": "BIGINT", "range_breakpoints": [],
        "sort_keys": ["icustay_id", sl.ORDINAL], "source_order": sl.ORDINAL,
        "row_group_target": 65536, "builder_version": sl.BUILDER_VERSION,
        "inventory": [{"relative_path": "bucket_id=0/part.parquet",
                       "size": part.stat().st_size, "row_count": len(frame)}],
        "derived_from": {"table": "chartevents", "files": [{
            "path": flat.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(flat.read_bytes()).hexdigest(),
        }]},
    }
    (bucket / sl.MANIFEST).write_text(json.dumps(manifest))
    farm = root / "farm"
    farm.mkdir()
    (farm / bucket.name).symlink_to(bucket, target_is_directory=True)
    source = ICUDataSource(load_src_cfg("mimic"), base_path=farm, enable_cache=False)
    return source, flat, part


@pytest.mark.parametrize("damage,reason", [
    ("source_missing", "bucket_derived_from_source_missing"),
    ("part_missing", "bucket_inventory_mismatch"),
    ("source_changed", "bucket_derived_from_source_changed"),
])
def test_bucket_only_farm_reads_bound_physical_lineage_and_refuses_damage(tmp_path, damage, reason):
    source, flat, part = _farm(tmp_path)
    assert sorted(p.name for p in source.base_path.iterdir()) == ["chartevents_bucket"]
    with sb.binding_unit():
        files = source.resolve_target_files("chartevents")
        assert tuple(p.resolve() for p in files) == (part,)
        resolved_count = sum(len(pd.read_parquet(p)) for p in files)
        read_count = len(source._load_raw_frame("chartevents", ["icustay_id", "itemid", "value"]))
        assert resolved_count == read_count == 20
        # Receipt lineage and inode pairing use the same physical anchor.
        layout = sl.read_bucket_contract(source.base_path / "chartevents_bucket")
        record = sl._bucket_record(layout)
        assert record["derived_from_recorded"]
        assert record["derived_from"]["files"][0]["realpath"] == str(flat)
        assert sl._bucket_derives_from(layout.root, flat)

    if damage == "source_missing":
        flat.unlink()
    elif damage == "part_missing":
        part.unlink()
    else:
        stat = flat.stat()
        os.utime(flat, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    # A fresh unit must refuse on contract validation, independently of a
    # previous run's payload or binding cache.
    with sb.binding_unit(), pytest.raises(sl.StorageIntegrityError) as error:
        source.resolve_target_files("chartevents")
    assert error.value.reason == reason
