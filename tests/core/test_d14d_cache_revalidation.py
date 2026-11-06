"""A warm raw-frame payload is revalidated by its owner before every hit (Codex delta14d).

The ``_load_raw_frame`` cache used to check only the captured files'
(size, mtime_ns, ino). A bound flat table whose file and conversion manifest
had both vanished became a miss. The miss then fell through to the
missing-table branch and returned an empty table. A shard added beside the
unchanged files (manifest updated or absent) let the old payload be served.
Now every hit asks the owner: the resolver (layout choice, file set, empty
resolve), the bucket contract, or the CSV admission, plus the metadata
sha256. In a unit that bound the source, a mismatch raises
``storage_source_changed`` / ``storage_source_missing``. Only a new unit may
refresh. Rows are synthetic.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from easyicu.config import load_src_cfg
from easyicu.datasource import FilterOp, FilterSpec, ICUDataSource
from easyicu.storage_binding import StorageSourceChanged, binding_unit, consumed_files
from tests.support.bucket_storage import b6_source, install_layout

TABLE = "chartevents"


def _manifest(root: Path) -> None:
    """Declare every Parquet output under ``root`` (a converted installation)."""
    tables: dict[str, list] = {}
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root)
        if any(part.endswith("_bucket") for part in relative.parts):
            continue
        table = relative.parts[0].removesuffix(".parquet")
        tables.setdefault(table, []).append(
            {"relative_path": relative.as_posix(), "size_bytes": path.stat().st_size}
        )
    (root / "conversion_manifest.json").write_text(json.dumps({
        "schema_version": "easyicu.conversion_manifest/1",
        "tables": [{"table": t, "row_count": None, "outputs": o} for t, o in tables.items()],
    }))


def _cached(root: Path) -> ICUDataSource:
    return ICUDataSource(load_src_cfg("miiv"), base_path=root, enable_cache=True)


def _read(source, itemids=None):
    filters = None if itemids is None else [FilterSpec(column="itemid", op=FilterOp.IN, value=itemids)]
    return source.load_table(TABLE, columns=["stay_id", "itemid", "valuenum"], filters=filters).data


def _add_shard(root: Path) -> None:
    frame = pd.read_parquet(root / TABLE / "1.parquet")
    frame.assign(valuenum=frame["valuenum"] + 1000).to_parquet(root / TABLE / "2.parquet", index=False)


@pytest.fixture
def flat(tmp_path):
    root = tmp_path / "miiv"
    b6_source(root, "miiv")
    return root


def test_a_warm_payload_is_served_and_records_its_files(flat):
    _manifest(flat)
    source = _cached(flat)
    calls = []
    real = source._load_raw_frame_uncached
    source._load_raw_frame_uncached = lambda *a, **k: calls.append(1) or real(*a, **k)
    with binding_unit():
        first = _read(source)
    with binding_unit():
        second = _read(source)
        consumed = {e["path"] for e in consumed_files() if e["table"] == TABLE}
    assert len(calls) == 1 and second.equals(first)
    assert consumed == {str(flat / TABLE / "1.parquet")}


def test_a_vanished_bound_flat_table_and_manifest_is_missing_not_empty(flat):
    # A single-file table: on a miss, the missing-table compatibility branch
    # would return an empty table.
    _manifest(flat)
    source = _cached(flat)
    with binding_unit():
        assert len(source.load_table("icustays", columns=["stay_id"]).data)
        (flat / "icustays.parquet").unlink()
        (flat / "conversion_manifest.json").unlink()
        with pytest.raises(StorageSourceChanged) as error:
            source.load_table("icustays", columns=["stay_id"])
    assert error.value.reason == "storage_source_missing"


def test_a_shard_added_with_an_updated_manifest_is_a_changed_source(flat):
    _manifest(flat)
    source = _cached(flat)
    with binding_unit():
        _read(source)
        _add_shard(flat)
        _manifest(flat)
        with pytest.raises(StorageSourceChanged) as error:
            _read(source)
    assert error.value.reason == "storage_source_changed"


def test_a_shard_added_to_a_manifestless_flat_table_is_a_changed_source(flat):
    assert not (flat / "conversion_manifest.json").exists()
    source = _cached(flat)
    with binding_unit():
        _read(source)
        _read(source)  # a warm hit first: the memo is in place
        _add_shard(flat)
        with pytest.raises(StorageSourceChanged) as error:
            _read(source)
    assert error.value.reason == "storage_source_changed"


def test_a_new_unit_refreshes_the_payload(flat):
    source = _cached(flat)
    with binding_unit():
        before = len(_read(source))
    _add_shard(flat)
    with binding_unit():
        assert len(_read(source)) == 2 * before


@pytest.fixture
def bucket(tmp_path):
    root = tmp_path / "miiv"
    install_layout(b6_source(root, "miiv"), "hash")
    return root, root / f"{TABLE}_bucket"


def _parts(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("bucket_id=*/*.parquet"))


def test_a_bucket_part_removed_under_a_warm_entry_is_refused(bucket):
    root, layout_root = bucket
    source = _cached(root)
    with binding_unit():
        assert len(_read(source, [220045]))
        read = {e["path"] for e in consumed_files() if e["format"] == "bucket_part"}
        other = next(p for p in _parts(layout_root) if str(p) not in read)
        other.unlink()
        with pytest.raises(StorageSourceChanged):
            _read(source, [220045])


def test_a_bucket_part_added_under_a_warm_entry_is_refused(bucket):
    root, layout_root = bucket
    source = _cached(root)
    with binding_unit():
        assert len(_read(source, [220045]))
        first = _parts(layout_root)[0]
        shutil.copy(first, first.with_name("extra.parquet"))
        with pytest.raises(StorageSourceChanged):
            _read(source, [220045])
