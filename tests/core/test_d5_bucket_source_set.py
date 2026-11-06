"""A bucket layout is bound to its whole flat source set, not only the recorded files (Codex delta5 B-SOURCE-SET).

``derived_from`` records every source file the builder read. The reader
used to check only those files' size and mtime, so a shard added beside
them left the layout open and every read silently missed the shard's rows.
On every open, cold or warm, the current source set found by the
builder's own discovery rule must now be exactly the recorded set; any
other set is ``bucket_derived_from_source_set_changed``. Beside a complete
flat source the resolver then reads the flat source (recording the
fallback); the bucket is never served. Rows are synthetic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.storage_layout import (
    BucketLayoutError,
    StorageIntegrityError,
    bucket_source_files,
    read_bucket_contract,
    scan_fallbacks,
)
from scripts.build_itemid_bucket_cache import _source_files, build_bucket_cache, build_variable_cache
from tests.support.bucket_storage import b6_source, install_variable_layout


def _events(root: Path) -> None:
    frame = pd.DataFrame({
        "stay_id": [40, 10, 30, 20, 10, 20],
        "item": ["HR", "GCS", "Temp", "GCS", "HR", "HR"],
        "value": [float(i) for i in range(6)],
    })
    (root / "events").mkdir(parents=True)
    frame.iloc[:3].to_parquet(root / "events" / "1.parquet", index=False)
    frame.iloc[3:].to_parquet(root / "events" / "2.parquet", index=False)


def _shard(root: Path, name: str = "3.parquet") -> Path:
    path = root / "events" / name
    pd.DataFrame({"stay_id": [50], "item": ["HR"], "value": [9.0]}).to_parquet(path, index=False)
    return path


def _variable(root: Path) -> Path:
    build_variable_cache(root, table="events", key="item", stay_key="stay_id", bucket_count=2,
                         rows_per_bucket=1, memory_limit="256MB", threads=1)
    return root / "events_bucket"


def _hash(root: Path) -> Path:
    build_bucket_cache(root, table="events", key="item", bucket_count=2,
                       memory_limit="256MB", threads=1)
    return root / "events_bucket"


_BUILDS = [_variable, _hash]


def _reason(bucket: Path, cache: dict | None = None) -> str:
    with pytest.raises(BucketLayoutError) as error:
        read_bucket_contract(bucket, cache=cache)
    return error.value.reason


def test_the_builder_and_the_reader_share_one_discovery_rule(tmp_path):
    _events(tmp_path)
    (tmp_path / "events" / ".hidden.parquet").write_bytes(b"")
    assert tuple(_source_files(tmp_path, "events")) == bucket_source_files(tmp_path, "events")
    assert [p.name for p in bucket_source_files(tmp_path, "events")] == ["1.parquet", "2.parquet"]


@pytest.mark.parametrize("build", _BUILDS)
def test_a_shard_added_beside_the_recorded_sources_is_refused_cold(tmp_path, build):
    _events(tmp_path)
    bucket = build(tmp_path)
    _shard(tmp_path)
    assert _reason(bucket) == "bucket_derived_from_source_set_changed"


@pytest.mark.parametrize("build", _BUILDS)
def test_a_shard_added_after_a_warm_open_is_refused(tmp_path, build):
    _events(tmp_path)
    bucket = build(tmp_path)
    cache: dict = {}
    assert read_bucket_contract(bucket, cache=cache).root == bucket
    assert cache  # the footer validation is cached; the source set is not
    _shard(tmp_path)
    assert _reason(bucket, cache) == "bucket_derived_from_source_set_changed"


@pytest.mark.parametrize("build", _BUILDS)
def test_a_hidden_file_beside_the_sources_does_not_count(tmp_path, build):
    _events(tmp_path)
    bucket = build(tmp_path)
    _shard(tmp_path, ".partial.parquet")
    assert read_bucket_contract(bucket).root == bucket


def test_a_resolved_source_never_serves_the_bucket_after_a_shard_is_added(tmp_path):
    source = b6_source(tmp_path / "miiv", "miiv")
    install_variable_layout(source)
    bucket = tmp_path / "miiv" / "chartevents_bucket"
    assert source.resolve_bucket_directory("chartevents") == bucket
    warm = source.resolve_target_files("chartevents")
    assert warm and all(bucket in p.parents for p in warm)
    shard = tmp_path / "miiv" / "chartevents" / "2.parquet"
    pd.read_parquet(tmp_path / "miiv" / "chartevents" / "1.parquet").iloc[:1].to_parquet(
        shard, index=False)
    try:
        files = source.resolve_target_files("chartevents")
    except StorageIntegrityError:
        return  # refused: never a silent subset
    assert shard in files and not any(bucket in p.parents for p in files)
    assert any(
        e.get("reason") == "bucket_cache_ignored"
        and e.get("bucket_reason") == "bucket_derived_from_source_set_changed"
        for e in scan_fallbacks()
    )
