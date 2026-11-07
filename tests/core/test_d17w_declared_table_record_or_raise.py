"""A declared table with no physical source is recorded or refused, never silently empty (delta17w).

The conversion manifest is the authority, not an exception name.
- Case 1: a declared table the manifest lists, whose files are missing,
  raises ``table_scan_files_missing``. This holds on the eICU/generic path
  (a), the eICU optional-channel reader (b) and the MIIV demo branch (c).
- Case 2: a declared table the manifest does not list, or an installation
  with no manifest, may read as empty. It is never silent: a
  ``table_unavailable`` record (``availability`` source_unavailable,
  ``missing_tables``) is written first. It reaches the run log, the module
  receipt and the run manifest, tagged with the module that read it.
Rows are synthetic.
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pandas as pd
import pytest

from easyicu.datasource import MissingTableSourceError
from easyicu.storage_layout import (
    SCAN_FALLBACK_LOG_ENV,
    STORAGE_MODULE_ENV,
    TableScanError,
    read_unavailable_log,
    storage_module,
)
from tests.support.bucket_storage import b6_source
from tests.support.eicu_source import _eicu_source


def _manifest(root: Path, *, extra_listed: str | None = None) -> None:
    """List every Parquet output under ``root``, plus ``extra_listed`` (absent on disk)."""
    tables: dict[str, list] = {}
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root)
        tables.setdefault(relative.parts[0].removesuffix(".parquet"), []).append(
            {"relative_path": relative.as_posix(), "size_bytes": path.stat().st_size}
        )
    if extra_listed:
        tables[extra_listed] = [{"relative_path": f"{extra_listed}.parquet", "size_bytes": 1}]
    (root / "conversion_manifest.json").write_text(json.dumps({
        "schema_version": "easyicu.conversion_manifest/1",
        "tables": [{"table": t, "row_count": None, "outputs": o} for t, o in tables.items()],
    }))


@pytest.fixture
def run_log(tmp_path, monkeypatch):
    log = tmp_path / "run.jsonl"
    monkeypatch.setenv(SCAN_FALLBACK_LOG_ENV, str(log))
    monkeypatch.delenv(STORAGE_MODULE_ENV, raising=False)
    return log


def _miiv(tmp_path):
    return b6_source(tmp_path / "miiv", "miiv")  # icustays + chartevents only


def _eicu(tmp_path):
    root = tmp_path / "eicu"
    root.mkdir()
    return _eicu_source(root, "eicu")  # patient only


def _optional(source, table="microlab"):
    from easyicu.concept.eicu_source_qc import load_optional

    return load_optional(source, table, [1], columns=["patientunitstayid"])


# --- case 1: listed in the manifest, files missing: raise on every path ------


def test_c_the_miiv_demo_branch_raises_for_a_listed_table(tmp_path, run_log):
    source = _miiv(tmp_path)
    _manifest(source.base_path, extra_listed="patients")
    with pytest.raises(TableScanError) as error:
        source.load_table("patients")
    assert error.value.reason == "table_scan_files_missing"
    assert read_unavailable_log(run_log) == ()


def test_a_the_generic_path_raises_for_a_listed_table(tmp_path, run_log):
    source = _eicu(tmp_path)
    _manifest(source.base_path, extra_listed="microlab")
    with pytest.raises(TableScanError) as error:
        source.load_table("microlab")
    assert error.value.reason == "table_scan_files_missing"


def test_b_the_optional_channel_raises_for_a_listed_table(tmp_path, run_log):
    source = _eicu(tmp_path)
    _manifest(source.base_path, extra_listed="microlab")
    with pytest.raises(TableScanError):
        _optional(source)


# --- case 2: unlisted or no manifest: empty or absent, with a record first ---


@pytest.mark.parametrize("manifest", ["unlisted", "absent"])
def test_c_the_miiv_demo_branch_records_before_returning_empty(tmp_path, run_log, manifest):
    source = _miiv(tmp_path)
    if manifest == "unlisted":
        _manifest(source.base_path)
    assert source.load_table("patients").data.empty
    [record] = read_unavailable_log(run_log)
    assert record["table"] == "patients" and record["manifest"] == manifest
    assert record["availability"] == "source_unavailable"
    assert record["missing_tables"] == ["patients"] and record["module"] is None


@pytest.mark.parametrize("manifest", ["unlisted", "absent"])
def test_a_and_b_record_before_absence(tmp_path, run_log, manifest):
    source = _eicu(tmp_path)
    if manifest == "unlisted":
        _manifest(source.base_path)
    with pytest.raises(MissingTableSourceError):
        source.load_table("microlab")
    assert _optional(source).empty
    records = read_unavailable_log(run_log)
    assert {(r["table"], r["manifest"]) for r in records} == {("microlab", manifest)}


def test_a_vanished_file_is_not_an_absent_optional_channel(tmp_path, monkeypatch):
    source = _eicu(tmp_path)

    def vanished(*_args, **_kwargs):
        raise FileNotFoundError("microlab.parquet")

    monkeypatch.setattr(source, "load_table", vanished)
    with pytest.raises(FileNotFoundError):
        _optional(source)


@pytest.mark.parametrize("path,table", [("c_empty", "patients"), ("a_raise", "microlab")])
def test_the_record_survives_a_failure_after_it(tmp_path, run_log, monkeypatch, path, table):
    from easyicu import storage_layout

    source = _miiv(tmp_path) if path == "c_empty" else _eicu(tmp_path)
    real = storage_layout.record_table_unavailable

    def record_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("injected after the record")

    monkeypatch.setattr(storage_layout, "record_table_unavailable", record_then_fail)
    with pytest.raises(RuntimeError, match="injected"):
        source.load_table(table)
    assert [r["table"] for r in read_unavailable_log(run_log)] == [table]


# --- module tag: in-process, isolated child, missing ---------------------------


def _record_in_child(root: str) -> None:
    from types import SimpleNamespace

    from easyicu.storage_layout import record_table_unavailable

    record_table_unavailable(SimpleNamespace(base_path=Path(root)), "child_table")


def test_the_module_tag_reaches_in_process_and_child_records(tmp_path, run_log):
    from types import SimpleNamespace

    from easyicu.storage_layout import record_table_unavailable

    with storage_module("vitals"):
        record_table_unavailable(SimpleNamespace(base_path=tmp_path), "parent_table")
        child = multiprocessing.get_context("spawn").Process(
            target=_record_in_child, args=(str(tmp_path),)
        )
        child.start()
        child.join(60)
        assert child.exitcode == 0
    record_table_unavailable(SimpleNamespace(base_path=tmp_path), "untagged_table")
    modules = {r["table"]: r["module"] for r in read_unavailable_log(run_log)}
    assert modules == {"parent_table": "vitals", "child_table": "vitals", "untagged_table": None}


def test_without_a_run_log_a_module_reports_only_its_own_run(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from easyicu import storage_layout
    from easyicu.storage_layout import module_unavailability, record_table_unavailable

    monkeypatch.delenv(SCAN_FALLBACK_LOG_ENV, raising=False)
    monkeypatch.setattr(storage_layout, "_UNAVAILABLE", [])
    source = SimpleNamespace(base_path=tmp_path)
    with storage_module("vitals"):
        record_table_unavailable(source, "earlier_run")
        assert [r["table"] for r in module_unavailability("vitals")] == ["earlier_run"]
    with storage_module("vitals"):
        assert module_unavailability("vitals") == []
        record_table_unavailable(source, "earlier_run")
        assert [r["table"] for r in module_unavailability("vitals")] == ["earlier_run"]


@pytest.mark.parametrize("isolated", [False, True], ids=["in_process", "isolated_child"])
def test_a_module_run_tags_its_records_and_restores_the_tag(tmp_path, run_log, monkeypatch, isolated):
    import os
    import shutil

    from easyicu import storage_layout
    from easyicu.api import extraction as api
    from easyicu.api.concepts import clear_global_loader
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    shutil.rmtree(root / "pharma")
    (tmp_path / "out").mkdir()
    stays = pd.read_parquet(root / "general.parquet", columns=["patientid"])["patientid"].tolist()
    monkeypatch.setattr(storage_layout, "_UNAVAILABLE", [])
    monkeypatch.setattr(api, "_requires_isolated_stream_batch", lambda *_args: isolated)
    try:
        api._run_module_extraction(
            module_name="medications", concepts=list(CONCEPT_GROUPS_INTERNAL["medications"]),
            database="hirid", data_path=str(root), patient_ids_filter={"patientid": stays},
            batch_size=2, output_dir=str(tmp_path / "out"), stream_output_batches=True,
        )
    finally:
        clear_global_loader()
    assert STORAGE_MODULE_ENV not in os.environ
    records = [r for r in read_unavailable_log(run_log) if r["table"] == "pharma"]
    assert records and {r["module"] for r in records} == {"medications"}
    in_parent = [r for r in storage_layout._UNAVAILABLE if r["table"] == "pharma"]
    # An isolated batch reads in a fresh child; only the run log carries its record.
    assert (len(records) > len(in_parent)) is isolated
    receipt = json.loads((tmp_path / "out" / "_manifest.json").read_text())
    assert {r["module"] for r in receipt["storage_unavailable"]} == {"medications"}


# --- receipts: the module receipt and the run manifest -------------------------


def test_an_extraction_publishes_the_record_in_the_module_receipt_and_run_manifest(tmp_path):
    import shutil

    from easyicu.api.concepts import clear_global_loader
    from easyicu.api.extraction import extract_database
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    shutil.rmtree(root / "pharma")  # declared, no manifest: case 2
    output = tmp_path / "export"
    try:
        result = extract_database(
            "hirid", data_path=str(root), output_dir=str(output),
            modules=["outcome", "medications"], native_export_v2=True, verbose=False,
        )
    finally:
        clear_global_loader()

    expected = {("pharma", "absent", "medications")}
    run = {(r["table"], r["manifest"], r["module"]) for r in result["storage_unavailable"]}
    assert expected <= run
    module = {(r["table"], r["manifest"], r["module"])
              for r in result["modules"]["medications"]["storage_unavailable"]}
    assert expected <= module
    receipt = json.loads((output / "medications.manifest.json").read_text())
    assert expected <= {(r["table"], r["manifest"], r["module"]) for r in receipt["storage_unavailable"]}
    published = json.loads((output / "_manifest.json").read_text())
    assert expected <= {(r["table"], r["manifest"], r["module"]) for r in published["storage_unavailable"]}
