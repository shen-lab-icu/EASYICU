"""Storage fallbacks publish, and the receipt says so.

A corrupt bucket cache falls back to its flat source; an installation
without a conversion manifest falls back to the directory-only check.

Policy (supervisor ruling, r13fixS2): the bucket cache is derived from the
flat table, so a cache that fails validation is bypassed for the true source
and publication is allowed, but the fallback is a receipt fact, not only a
log warning: ``result["storage_fallbacks"]`` and the published
``_manifest.json`` name the table and the bucket's reason code.  Rows are
synthetic.
"""

from __future__ import annotations

import json
import shutil

from easyicu.storage_layout import read_scan_fallback_log


def test_a_broken_bucket_beside_its_flat_source_publishes_with_the_fallback_marked(tmp_path):
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

    expected = {"reason": "bucket_cache_ignored", "root": str(root),
                "table": "observations", "bucket_reason": "bucket_manifest_missing"}
    assert expected in result["storage_fallbacks"]
    assert result["native_export_v2"]["manifest"]
    published = json.loads((output / "_manifest.json").read_text())
    assert expected in published["storage_fallbacks"]


def test_a_parquet_install_without_a_manifest_publishes_with_the_fallback_marked(tmp_path):
    # The converter now reports conversion_manifest_missing_rebuild_required
    # and hands such an install to this recorded directory-only fallback.
    from easyicu.api.concepts import clear_global_loader
    from easyicu.api.extraction import extract_database
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    assert not (root / "conversion_manifest.json").exists()

    output = tmp_path / "export"
    try:
        result = extract_database(
            "hirid", data_path=str(root), output_dir=str(output),
            modules=["outcome"], native_export_v2=True, verbose=False,
        )
    finally:
        clear_global_loader()

    expected = {"reason": "table_scan_unmanifested", "root": str(root), "table": "observations"}
    assert expected in result["storage_fallbacks"]
    published = json.loads((output / "_manifest.json").read_text())
    assert expected in published["storage_fallbacks"]


def test_the_fallback_log_is_shared_by_worker_processes(tmp_path, monkeypatch):
    import multiprocessing

    from easyicu.storage_layout import SCAN_FALLBACK_LOG_ENV

    log = tmp_path / "fallbacks.jsonl"
    monkeypatch.setenv(SCAN_FALLBACK_LOG_ENV, str(log))
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_record_in_child, args=(str(tmp_path),))
    process.start()
    process.join(60)
    assert process.exitcode == 0
    assert read_scan_fallback_log(log) == (
        {"reason": "table_read_csv_unmanifested", "root": str(tmp_path), "table": "child"},
    )


def _record_in_child(root):
    from pathlib import Path
    from types import SimpleNamespace

    from easyicu.storage_layout import csv_table_scan

    csv_table_scan(SimpleNamespace(base_path=Path(root)), "child", Path(root) / "child.csv")
