"""A run's storage evidence reaches its result, its receipt and its failures (Codex delta10d).

``extract_database`` collects every worker's ``table_consumed`` events into
``result["storage_consumed"]``. ``result["storage_binding"]`` holds the
parent binding digest, plus ``tables``: what served each consumed table,
built from every worker's ``table_layout`` events. A consumed table without a
layout record is ``storage_layout_unrecorded``. Both are published in the
manifest. When the run raises, the same exception propagates with
``storage_evidence``. The run log variable is restored and its file removed.
Rows are synthetic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from easyicu.storage_binding import binding_unit
from easyicu.storage_layout import SCAN_FALLBACK_LOG_ENV, TableScanError
from tests.support.bucket_storage import b6_source, install_layout


def _extract(root: Path, output: Path, **kwargs):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.api.extraction import extract_database

    try:
        return extract_database(
            "hirid", data_path=str(root), output_dir=str(output),
            modules=["outcome"], verbose=False, **kwargs,
        )
    finally:
        clear_global_loader()


@pytest.fixture
def hirid(tmp_path):
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    return root


def test_a_real_extraction_delivers_consumed_files_and_table_layouts(hirid, tmp_path):
    output = tmp_path / "export"
    result = _extract(hirid, output, native_export_v2=True)

    consumed = result["storage_consumed"]
    assert consumed and all(e["event"] == "table_consumed" for e in consumed)
    assert {"root", "table", "path", "realpath", "format"} <= set(consumed[0])
    tables = result["storage_binding"]["tables"]
    assert set(tables) == {e["table"] for e in consumed}
    for record in tables.values():
        assert record["layout"] in ("flat", "bucket") and record["root"]
        assert record["reads"] and all("root" in r and "derived_from" in r for r in record["reads"])
    assert {"digest", "files", "scope"} <= set(result["storage_binding"])

    published = json.loads((output / "_manifest.json").read_text())
    assert published["storage_consumed"] == consumed
    assert published["storage_binding"]["tables"] == tables


def test_a_raising_extraction_keeps_its_evidence_and_cleans_up(hirid, tmp_path, monkeypatch):
    from easyicu.api import extraction

    outer = tmp_path / "outer.jsonl"
    monkeypatch.setenv(SCAN_FALLBACK_LOG_ENV, str(outer))
    seen = {}
    cause = ValueError("worker result unreadable")

    def fail(*_args, **_kwargs):
        seen["log"] = os.environ[SCAN_FALLBACK_LOG_ENV]
        raise RuntimeError("collect failed") from cause

    monkeypatch.setattr(extraction, "_oom_kill_delta", fail)
    with pytest.raises(RuntimeError) as error:
        _extract(hirid, tmp_path / "export")

    assert type(error.value) is RuntimeError and error.value.__cause__ is cause
    assert error.value.storage_evidence["storage_consumed"]
    assert "storage_fallbacks" in error.value.storage_evidence
    assert os.environ[SCAN_FALLBACK_LOG_ENV] == str(outer)
    assert seen["log"] != str(outer) and not Path(seen["log"]).exists()


def test_an_unreadable_run_log_never_masks_the_original_error(hirid, tmp_path, monkeypatch):
    from easyicu.api import extraction

    monkeypatch.delenv(SCAN_FALLBACK_LOG_ENV, raising=False)

    def fail(*_args, **_kwargs):
        with open(os.environ[SCAN_FALLBACK_LOG_ENV], "a", encoding="utf-8") as handle:
            handle.write("not json\n")
        raise LookupError("original")

    monkeypatch.setattr(extraction, "_oom_kill_delta", fail)
    with pytest.raises(LookupError, match="original") as error:
        _extract(hirid, tmp_path / "export")
    assert error.value.storage_evidence["error"].endswith("storage_run_log_unreadable")
    assert SCAN_FALLBACK_LOG_ENV not in os.environ


def test_an_interrupt_still_restores_the_log_variable(hirid, tmp_path, monkeypatch):
    from easyicu.api import extraction

    monkeypatch.delenv(SCAN_FALLBACK_LOG_ENV, raising=False)
    seen = {}

    def interrupt(*_args, **_kwargs):
        seen["log"] = os.environ[SCAN_FALLBACK_LOG_ENV]
        raise KeyboardInterrupt

    monkeypatch.setattr(extraction, "_oom_kill_delta", interrupt)
    with pytest.raises(KeyboardInterrupt) as error:
        _extract(hirid, tmp_path / "export")
    assert not hasattr(error.value, "storage_evidence")
    assert SCAN_FALLBACK_LOG_ENV not in os.environ and not Path(seen["log"]).exists()


def test_a_publication_failure_carries_the_results_evidence(hirid, tmp_path, monkeypatch):
    from easyicu.api import extraction

    def refuse(**_kwargs):
        raise ValueError("publication refused")

    monkeypatch.setattr(extraction, "_publish_native_export_v2", refuse)
    with pytest.raises(extraction.NativeExportPublicationError) as error:
        _extract(hirid, tmp_path / "export", native_export_v2=True)
    assert isinstance(error.value.__cause__, ValueError)
    assert error.value.storage_evidence["storage_consumed"]


def test_a_bucket_read_records_its_manifest_and_derived_from(tmp_path, monkeypatch):
    source = b6_source(tmp_path / "miiv", "miiv")
    install_layout(source, "hash")  # the build reads the flat table: before the run log
    log = tmp_path / "run.jsonl"
    monkeypatch.setenv(SCAN_FALLBACK_LOG_ENV, str(log))
    with binding_unit():
        assert source.resolve_target_files("chartevents", "itemid", [220045])
    from easyicu.storage_layout import consumed_table_layouts, read_consumed_log, read_layout_log

    tables = consumed_table_layouts(read_consumed_log(log), read_layout_log(log))
    record = tables["chartevents"]
    assert record["layout"] == "bucket" and record["derived_from_recorded"] is True
    assert record["root"] == str(tmp_path / "miiv" / "chartevents_bucket")
    assert record["manifest"]["sha256"] and record["derived_from"]["files"]
    assert record["reads"] == [{k: v for k, v in record.items() if k != "reads"}]


def test_a_consumed_table_without_a_layout_record_is_refused():
    from easyicu.storage_layout import consumed_table_layouts

    consumed = [{"event": "table_consumed", "root": "/data", "table": "orphan",
                 "path": "/data/orphan.parquet", "realpath": "/data/orphan.parquet",
                 "format": "parquet"}]
    with pytest.raises(TableScanError) as error:
        consumed_table_layouts(consumed, ())
    assert error.value.reason == "storage_layout_unrecorded"
