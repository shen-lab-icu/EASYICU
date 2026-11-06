"""Per table actually read, the receipt names the layout used (formal-run F2).

The storage policy (``storage_layout.resolve_table_files``) records which
layout served each table in the run's storage log, the same log that carries
the fallbacks, so worker processes report into one receipt. A run's
``storage_reads`` (and the native manifest) list one entry per table and
layout; a flat read of a table whose bucket cache was bypassed carries that
fallback's reason. Rows are synthetic.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import pytest

from easyicu import storage_layout
from tests.support.bucket_storage import TABLE_CONFIGS, b6_source, install_layout


@pytest.fixture
def arm_log(tmp_path, monkeypatch):
    """Start the run's storage log (after any fixture setup reads)."""

    def arm():
        path = tmp_path / "run_storage_log.jsonl"
        monkeypatch.setenv(storage_layout.SCAN_FALLBACK_LOG_ENV, str(path))
        return path

    return arm


def _receipt(log):
    return storage_layout.table_layout_receipt(
        storage_layout.read_table_read_log(log), storage_layout.read_scan_fallback_log(log)
    )


@pytest.mark.parametrize(
    ("scheme", "layout"),
    [("hash", "bucket_hash"), ("variable", "bucket_variable_stay_range")],
)
def test_a_bucket_read_names_its_scheme(tmp_path, arm_log, scheme, layout):
    source = b6_source(tmp_path / "eicu", "eicu")
    install_layout(source, scheme)
    table = TABLE_CONFIGS["eicu"][0]
    run_log = arm_log()

    assert source.resolve_target_files(table)

    entries = [e for e in _receipt(run_log) if e["table"] == table]
    bucket = source.base_path / f"{table}_bucket"
    assert entries == [{"root": str(source.base_path), "table": table, "layout": layout,
                        "layout_path": str(bucket),
                        "layout_realpath": os.path.realpath(bucket)}]
    assert storage_layout.read_scan_fallback_log(run_log) == ()


def test_a_bucket_read_is_bound_to_its_manifest_and_sources(tmp_path, arm_log):
    source = b6_source(tmp_path / "eicu", "eicu")
    install_layout(source, "hash")
    table = TABLE_CONFIGS["eicu"][0]
    run_log = arm_log()
    assert source.resolve_target_files(table)
    (entry,) = [e for e in _receipt(run_log) if e["table"] == table]

    binding = storage_layout.bucket_source_receipt(entry["layout_path"])

    bucket = source.base_path / f"{table}_bucket"
    manifest = bucket / storage_layout.MANIFEST
    assert binding["layout_realpath"] == entry["layout_realpath"]
    assert binding["manifest"] == {
        "path": str(manifest),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    recorded = json.loads(manifest.read_text())["derived_from"]
    assert binding["derived_from"]["table"] == recorded["table"]
    assert [f["sha256"] for f in binding["derived_from"]["files"]] == [
        f["sha256"] for f in recorded["files"]
    ]
    for item in binding["derived_from"]["files"]:
        real = os.path.realpath(source.base_path / item["path"])
        assert item["realpath"] == real
        assert hashlib.sha256(open(real, "rb").read()).hexdigest() == item["sha256"]


def test_a_bucket_whose_source_changed_has_no_receipt(tmp_path, arm_log):
    source = b6_source(tmp_path / "eicu", "eicu")
    install_layout(source, "hash")
    table = TABLE_CONFIGS["eicu"][0]
    bucket = source.base_path / f"{table}_bucket"
    first = json.loads((bucket / storage_layout.MANIFEST).read_text())["derived_from"]["files"][0]
    os.utime(source.base_path / first["path"], ns=(1_800_000_000_000_000_000,) * 2)

    with pytest.raises(storage_layout.BucketLayoutError) as error:
        storage_layout.bucket_source_receipt(bucket)
    assert error.value.reason == "bucket_derived_from_source_changed"


def test_a_flat_read_through_links_names_the_directory_it_really_read(tmp_path, arm_log):
    source = b6_source(tmp_path / "real", "eicu")
    table = TABLE_CONFIGS["eicu"][0]
    files = storage_layout.flat_table_files(source, table)
    linked = tmp_path / "linked"
    linked.mkdir()
    for item in source.base_path.iterdir():
        (linked / item.name).symlink_to(item, target_is_directory=item.is_dir())
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource

    through = ICUDataSource(load_src_cfg("eicu"), base_path=linked, enable_cache=False)
    run_log = arm_log()
    assert through.resolve_target_files(table)

    (entry,) = [e for e in _receipt(run_log) if e["table"] == table]
    assert entry["root"] == str(linked)
    assert entry["layout_path"].startswith(str(linked))
    assert entry["layout_realpath"] == os.path.commonpath(
        [os.path.dirname(os.path.realpath(f)) for f in files]
    )


def test_a_flat_read_is_flat_and_carries_its_fallback_reason(tmp_path, arm_log):
    # The fixture has no conversion manifest, so the flat read is also the
    # directory-only fallback; the receipt says both.
    source = b6_source(tmp_path / "eicu", "eicu")
    table = TABLE_CONFIGS["eicu"][0]
    run_log = arm_log()

    assert source.resolve_target_files(table)

    (entry,) = [e for e in _receipt(run_log) if e["table"] == table]
    assert {k: entry[k] for k in ("root", "table", "layout", "fallback_reason")} == {
        "root": str(source.base_path), "table": table, "layout": "flat",
        "fallback_reason": "table_scan_unmanifested",
    }
    assert os.path.realpath(entry["layout_path"]) == entry["layout_realpath"]
    assert entry["layout_realpath"].startswith(os.path.realpath(source.base_path))


def test_a_published_run_lists_every_table_read_with_the_fallback_reason(tmp_path):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.api.extraction import extract_database
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    broken = root / "observations_bucket" / "bucket_id=0"
    broken.mkdir(parents=True)
    shutil.copy(root / "observations" / "1.parquet", broken / "1.parquet")  # no contract

    output = tmp_path / "export"
    try:
        result = extract_database(
            "hirid", data_path=str(root), output_dir=str(output),
            modules=["outcome"], native_export_v2=True, verbose=False,
        )
    finally:
        clear_global_loader()

    published = json.loads((output / "_manifest.json").read_text())
    assert published["storage_reads"] == result["storage_reads"]
    by_table = {(e["table"], e["layout"]): e for e in result["storage_reads"]}
    observations = by_table[("observations", "flat")]
    assert {k: v for k, v in observations.items() if not k.startswith("layout_")} == {
        "root": str(root), "table": "observations", "layout": "flat",
        "fallback_reason": "bucket_cache_ignored",
        "bucket_reason": "bucket_manifest_missing",
    }
    assert observations["layout_realpath"] == os.path.realpath(root / "observations")
    # Every other table read is listed too, each with its own layout.
    assert len(by_table) > 1
    assert all(e["layout"] == "flat" for e in result["storage_reads"])
    # The fallback list keeps its own contract: no table-read events in it.
    assert all("event" not in e for e in result["storage_fallbacks"])
