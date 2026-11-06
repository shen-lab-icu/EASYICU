"""A layout's ``derived_from`` binding is checked against the current sources on open (Codex delta3).

The builder records each source file's path, size, mtime and sha256; the
reader used to check only the record's shape. On every open the recorded
sources (relative to the installation root) must still exist with the
recorded size and mtime: a changed source is
``bucket_derived_from_source_changed``, a missing one
``bucket_derived_from_source_missing``. A layout without ``derived_from``
is read as before. Structural defects keep their own reasons. Rows are
synthetic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pytest

from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.storage_layout import (
    MANIFEST,
    BucketLayoutError,
    StorageIntegrityError,
    read_bucket_contract,
    scan_fallbacks,
)
from scripts.build_itemid_bucket_cache import build_bucket_cache, build_variable_cache
from tests.support.bucket_storage import b6_dictionary, b6_source, install_variable_layout


def _events(root: Path) -> Path:
    frame = pd.DataFrame({
        "stay_id": [40, 10, 30, 20, 10, 20],
        "item": ["HR", "GCS", "Temp", "GCS", "HR", "HR"],
        "value": [float(i) for i in range(6)],
    })
    (root / "events").mkdir(parents=True)
    frame.iloc[:3].to_parquet(root / "events" / "1.parquet", index=False)
    frame.iloc[3:].to_parquet(root / "events" / "2.parquet", index=False)
    return root / "events" / "1.parquet"


def _variable(root: Path) -> Path:
    build_variable_cache(root, table="events", key="item", stay_key="stay_id", bucket_count=2,
                         rows_per_bucket=1, memory_limit="256MB", threads=1)
    return root / "events_bucket"


def _hash(root: Path) -> Path:
    build_bucket_cache(root, table="events", key="item", bucket_count=2,
                       memory_limit="256MB", threads=1)
    return root / "events_bucket"


_BUILDS = [_variable, _hash]


def _reason(bucket: Path) -> str:
    with pytest.raises(BucketLayoutError) as error:
        read_bucket_contract(bucket)
    assert isinstance(error.value, StorageIntegrityError)
    return error.value.reason


@pytest.mark.parametrize("build", _BUILDS)
def test_an_intact_binding_opens(tmp_path, build):
    _events(tmp_path)
    assert read_bucket_contract(build(tmp_path)).root.name == "events_bucket"


@pytest.mark.parametrize("build", _BUILDS)
def test_a_touched_source_is_refused(tmp_path, build):
    source = _events(tmp_path)
    bucket = build(tmp_path)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert _reason(bucket) == "bucket_derived_from_source_changed"


@pytest.mark.parametrize("build", _BUILDS)
def test_a_rewritten_source_is_refused(tmp_path, build):
    source = _events(tmp_path)
    bucket = build(tmp_path)
    stat = source.stat()
    pd.DataFrame({"stay_id": [99] * 50, "item": ["HR"] * 50, "value": [1.0] * 50}).to_parquet(
        source, index=False)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same mtime, new size
    assert source.stat().st_size != stat.st_size
    assert _reason(bucket) == "bucket_derived_from_source_changed"


@pytest.mark.parametrize("build", _BUILDS)
def test_a_deleted_source_is_refused(tmp_path, build):
    source = _events(tmp_path)
    bucket = build(tmp_path)
    source.unlink()
    assert _reason(bucket) == "bucket_derived_from_source_missing"


def test_a_structural_defect_keeps_its_own_reason_without_the_source(tmp_path):
    source = _events(tmp_path)
    bucket = _variable(tmp_path)
    source.unlink()
    manifest = json.loads((bucket / MANIFEST).read_text())
    (bucket / manifest["inventory"][0]["relative_path"]).unlink()
    assert _reason(bucket) == "bucket_inventory_mismatch"


@pytest.mark.parametrize("path", ["../events/1.parquet", "/events/1.parquet"])
def test_a_recorded_path_outside_the_installation_is_invalid(tmp_path, path):
    _events(tmp_path)
    bucket = _variable(tmp_path)
    manifest = json.loads((bucket / MANIFEST).read_text())
    manifest["derived_from"]["files"][0]["path"] = path
    (bucket / MANIFEST).write_text(json.dumps(manifest))
    assert _reason(bucket) == "bucket_derived_from_invalid"


@pytest.mark.parametrize("build", _BUILDS)
def test_a_layout_without_a_binding_is_read_as_before(tmp_path, build):
    # Layouts built before derived_from existed carry no binding: nothing to
    # compare, so a later source change is not detected here (unchanged).
    source = _events(tmp_path)
    bucket = build(tmp_path)
    manifest = json.loads((bucket / MANIFEST).read_text())
    del manifest["derived_from"]
    (bucket / MANIFEST).write_text(json.dumps(manifest))
    source.unlink()
    assert read_bucket_contract(bucket).root == bucket


def test_beside_a_changed_source_public_loading_reads_the_source_and_marks_the_fallback(tmp_path):
    from easyicu.concept import ConceptResolver

    source = b6_source(tmp_path / "eicu", "eicu")
    install_variable_layout(source)
    flat = sorted((tmp_path / "eicu" / "nursecharting").glob("*.parquet")) or [
        tmp_path / "eicu" / "nursecharting.parquet"]
    stat = flat[0].stat()
    os.utime(flat[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    def hr(src):
        return ConceptResolver(b6_dictionary("eicu")).load_concepts(
            ["hr"], src, merge=False, interval=pd.Timedelta(hours=1),
            concept_workers=1, verbose=False,
        )

    bound = hr(source)
    assert any(
        e.get("reason") == "bucket_cache_ignored"
        and e.get("bucket_reason") == "bucket_derived_from_source_changed"
        for e in scan_fallbacks()
    )
    shutil.rmtree(tmp_path / "eicu" / "nursecharting_bucket")
    flat_only = hr(ICUDataSource(load_src_cfg("eicu"), base_path=tmp_path / "eicu", enable_cache=False))
    pd.testing.assert_frame_equal(
        bound.reset_index(drop=True), flat_only.reset_index(drop=True), check_exact=True)


def test_without_its_source_a_bound_layout_refuses_public_loading(tmp_path):
    from easyicu.concept import ConceptResolver

    source = b6_source(tmp_path / "eicu", "eicu")
    install_variable_layout(source)
    for path in (tmp_path / "eicu" / "nursecharting", tmp_path / "eicu" / "nursecharting.parquet"):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    with pytest.raises(StorageIntegrityError) as error:
        ConceptResolver(b6_dictionary("eicu")).load_concepts(
            ["hr"], source, merge=False, interval=pd.Timedelta(hours=1),
            concept_workers=1, verbose=False,
        )
    assert error.value.reason == "bucket_derived_from_source_missing"
