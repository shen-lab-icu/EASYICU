"""Per table actually read, the receipt names the layout used (formal-run F2).

The storage policy (``storage_layout.resolve_table_files``) records which
layout served each table in the run's storage log, the same log that carries
the fallbacks, so worker processes report into one receipt. A run's
``storage_reads`` (and the native manifest) list one entry per table and
layout; a flat read of a table whose bucket cache was bypassed carries that
fallback's reason. Rows are synthetic.
"""

from __future__ import annotations

import json
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
    assert entries == [{"root": str(source.base_path), "table": table, "layout": layout}]
    assert storage_layout.read_scan_fallback_log(run_log) == ()


def test_a_flat_read_is_flat_and_carries_its_fallback_reason(tmp_path, arm_log):
    # The fixture has no conversion manifest, so the flat read is also the
    # directory-only fallback; the receipt says both.
    source = b6_source(tmp_path / "eicu", "eicu")
    table = TABLE_CONFIGS["eicu"][0]
    run_log = arm_log()

    assert source.resolve_target_files(table)

    assert [e for e in _receipt(run_log) if e["table"] == table] == [
        {"root": str(source.base_path), "table": table, "layout": "flat",
         "fallback_reason": "table_scan_unmanifested"}
    ]


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
    assert by_table[("observations", "flat")] == {
        "root": str(root), "table": "observations", "layout": "flat",
        "fallback_reason": "bucket_cache_ignored",
        "bucket_reason": "bucket_manifest_missing",
    }
    # Every other table read is listed too, each with its own layout.
    assert len(by_table) > 1
    assert all(e["layout"] == "flat" for e in result["storage_reads"])
    # The fallback list keeps its own contract: no table-read events in it.
    assert all("event" not in e for e in result["storage_fallbacks"])
