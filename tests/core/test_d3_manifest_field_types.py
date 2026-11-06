"""A manifest field of the wrong JSON type is a typed layout refusal (Codex delta3 D3-MANIFEST-TYPE).

``secondary: []`` in a variable partition reached ``list.values()`` and
escaped as ``AttributeError``, which no storage caller treats as a storage
integrity failure. Every ill-typed field the contract reader touches must
end as ``BucketLayoutError`` (a ``StorageIntegrityError``) with a stable
reason; a conversion manifest's ill-typed table entry ends as its own
``TableScanError``. Rows are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.storage_layout import (
    MANIFEST,
    BucketLayoutError,
    StorageIntegrityError,
    TableScanError,
    _conversion_index,
    read_bucket_contract,
)
from scripts.build_itemid_bucket_cache import build_bucket_cache, build_variable_cache


def _events(root: Path) -> None:
    frame = pd.DataFrame({
        "stay_id": [40, 10, 30, 20, 10, 20],
        "item": ["HR", "GCS", "Temp", "GCS", "HR", "HR"],
        "detail": ["a", "eyes", "c", "motor", "a", "a"],
        "value": [float(i) for i in range(6)],
    })
    (root / "events").mkdir(parents=True)
    frame.to_parquet(root / "events" / "1.parquet", index=False)


def _variable(root: Path) -> Path:
    _events(root)
    build_variable_cache(
        root, table="events", key="item", stay_key="stay_id", bucket_count=2,
        rows_per_bucket=1, secondary_keys=["detail"], memory_limit="256MB", threads=1,
    )
    return root / "events_bucket"


def _hash(root: Path) -> Path:
    _events(root)
    build_bucket_cache(root, table="events", key="item", bucket_count=2,
                       memory_limit="256MB", threads=1)
    return root / "events_bucket"


def _edit(bucket: Path, change) -> None:
    manifest = json.loads((bucket / MANIFEST).read_text())
    change(manifest)
    (bucket / MANIFEST).write_text(json.dumps(manifest))


def _refused(bucket: Path) -> BucketLayoutError:
    with pytest.raises(BucketLayoutError) as error:
        read_bucket_contract(bucket)
    assert isinstance(error.value, StorageIntegrityError)
    return error.value


@pytest.mark.parametrize("secondary", [[], "eyes", 3])
def test_a_partition_secondary_that_is_not_an_object_is_refused(tmp_path, secondary):
    bucket = _variable(tmp_path)
    _edit(bucket, lambda m: m["partitions"][0].__setitem__("secondary", secondary))
    assert _refused(bucket).reason == "bucket_partition_invalid"


def test_a_partition_secondary_value_list_of_lists_is_refused(tmp_path):
    bucket = _variable(tmp_path)
    _edit(bucket, lambda m: m["partitions"][0].__setitem__("secondary", {"detail": [["a"]]}))
    assert _refused(bucket).reason == "bucket_partition_invalid"


def test_with_no_secondary_keys_an_empty_list_secondary_is_still_refused(tmp_path):
    # set([]) == set([]) let ``[]`` past the column check straight into .values().
    bucket = _variable(tmp_path)

    def change(m):
        m["secondary_keys"] = []
        for part in m["partitions"]:
            part["secondary"] = {}
        m["partitions"][0]["secondary"] = []

    _edit(bucket, change)
    assert _refused(bucket).reason == "bucket_partition_invalid"


def test_a_partition_entry_that_is_not_an_object_is_refused(tmp_path):
    bucket = _variable(tmp_path)
    _edit(bucket, lambda m: m["partitions"].__setitem__(0, [0, "HR"]))
    assert _refused(bucket).reason == "bucket_partition_invalid"


@pytest.mark.parametrize("build", [_variable, _hash])
def test_an_inventory_entry_that_is_not_an_object_is_refused(tmp_path, build):
    bucket = build(tmp_path)
    _edit(bucket, lambda m: m["inventory"].__setitem__(0, [m["inventory"][0]["relative_path"]]))
    assert _refused(bucket).reason == "bucket_inventory_mismatch"


def test_secondary_keys_of_the_wrong_type_keep_the_cause(tmp_path):
    bucket = _variable(tmp_path)
    _edit(bucket, lambda m: m.__setitem__("secondary_keys", {"detail": 1}))
    error = _refused(bucket)
    assert error.reason == "bucket_manifest_invalid"
    assert error.__cause__ is not None


def test_the_intact_layouts_still_open(tmp_path):
    assert read_bucket_contract(_variable(tmp_path / "v")).partitions
    assert read_bucket_contract(_hash(tmp_path / "h")).modulus == 2


@pytest.mark.parametrize("entry", ["events", ["events"], {"table": "events", "outputs": "x"},
                                   {"table": "events", "outputs": ["events/1.parquet"]}])
def test_an_ill_typed_conversion_manifest_entry_is_a_typed_scan_error(tmp_path, entry):
    (tmp_path / "conversion_manifest.json").write_text(json.dumps({"tables": [entry]}))
    with pytest.raises(TableScanError) as error:
        _conversion_index(tmp_path)
    assert error.value.reason == "conversion_manifest_unreadable"
