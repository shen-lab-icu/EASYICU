"""The formal-run receipt binds SIC cases.parquet to cases.csv.gz (formal-run F1).

The prepared SICdb conversion manifest records ``cases.parquet`` from
``cases.csv.gz`` without content digests or a converter commit, so formal
intake refuses it. ``tools/formal_run_receipt.py`` carries a ``cases_binding``
block: both sha256s, the manifest record, the converter identity, and a
reproduction with the sealed converter that must be byte- or value-identical.
A mismatch stops the receipt. Rows are synthetic.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "formal_run_receipt.py"
SPEC = importlib.util.spec_from_file_location("formal_run_receipt", TOOL)
assert SPEC and SPEC.loader
receipt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = receipt
SPEC.loader.exec_module(receipt)

CODE = {"source": "git_checkout", "commit": "c" * 40, "dirty": False, "version": "0.1.2",
        "installed_dist_version": "9.9.9", "import_path": "/x"}


def _prepared(root: Path) -> Path:
    from easyicu.io.data_converter import DataConverter

    root.mkdir(parents=True)
    frame = pd.DataFrame({
        "CaseID": [3, 1, 2],
        "PatientID": [30, 10, 20],
        "AdmissionYear": [2015, 2016, 2017],
        "TimeOfStay": [3600.5, 7200.0, 120.25],
        "HospitalUnit": ["A", None, "B"],
    })
    with gzip.open(root / "cases.csv.gz", "wt", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)
    os.utime(root / "cases.csv.gz", ns=(1_700_000_000_000_000_000,) * 2)
    DataConverter(root, database="sic", parallel_workers=1, verbose=False).convert_all()
    assert (root / "cases.parquet").is_file()
    assert (root / "conversion_manifest.json").is_file()
    return root


def test_a_reproducible_prepared_table_is_bound_by_content(tmp_path):
    prepared = _prepared(tmp_path / "prepared")

    block = receipt.sic_cases_binding(prepared, tmp_path / "scratch", CODE)

    assert block["cases_csv_gz"]["sha256"] == receipt.sha256_file(prepared / "cases.csv.gz")
    assert block["cases_parquet"]["sha256"] == receipt.sha256_file(prepared / "cases.parquet")
    assert block["conversion_manifest"]["record"]["table"] == "cases"
    assert block["conversion_manifest"]["record_key"].startswith("tables[")
    assert block["converter_identity"] == CODE
    assert block["reproduction"]["result"] in {"byte_identical", "value_identical"}
    # The reconversion scratch is removed; nothing is written beside the input.
    assert not (tmp_path / "scratch" / "sic_cases_reconvert").exists()


def test_a_prepared_table_the_converter_does_not_reproduce_stops_the_receipt(tmp_path):
    prepared = _prepared(tmp_path / "prepared")
    parquet = prepared / "cases.parquet"
    frame = pd.read_parquet(parquet)
    frame.loc[0, "TimeOfStay"] = frame.loc[0, "TimeOfStay"] + 1.0
    frame.to_parquet(parquet, index=False)
    # A manifest that describes the altered file, so only the reproduction can catch it.
    manifest_path = prepared / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["tables"]:
        if entry["table"] == "cases":
            for output in entry["outputs"]:
                output["size_bytes"] = parquet.stat().st_size
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.sic_cases_binding(prepared, tmp_path / "scratch", CODE)

    assert error.value.code == "sic_cases_reproduction_mismatch"
    assert error.value.details["differing_columns"] == ["TimeOfStay"]


def test_a_manifest_record_that_does_not_describe_the_source_stops_the_receipt(tmp_path):
    prepared = _prepared(tmp_path / "prepared")
    os.utime(prepared / "cases.csv.gz", ns=(1_800_000_000_000_000_000,) * 2)

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.sic_cases_binding(prepared, tmp_path / "scratch", CODE)

    assert error.value.code == "sic_cases_manifest_record_mismatch"


def test_code_that_is_not_the_sealed_checkout_is_refused():
    with pytest.raises(receipt.ReceiptError) as error:
        receipt.sealed_code_identity("0" * 40)
    assert error.value.code == "code_identity_not_sealed"


def _flat(root: Path, table: str, directory: Path) -> dict:
    return {"root": str(root), "table": table, "layout": "flat",
            "layout_path": str(directory), "layout_realpath": os.path.realpath(directory)}


def _run_root(tmp_path: Path, prepared: Path, *, commit: str, reads=None,
              version: str = CODE["version"]) -> Path:
    run = tmp_path / "run"
    export = run / "sic" / "export"
    export.mkdir(parents=True)
    (run / "status").write_text("tag=t sha=x\nsic exit=0\nDONE\n")
    (run / "sic" / "summary.json").write_text(json.dumps({
        "secs": 12.5, "cgroup_peak_gib": 1.5, "peak_tree_rss_gib": 1.2,
        "cgroup_memory_max": "15032385536", "batch_size": None,
        "errors": [], "worker_failures": 0,
    }))
    (export / "_manifest.json").write_text(json.dumps({
        "runtime_provenance": {
            "easyicu_git_commit": commit, "easyicu_git_dirty": False,
            "easyicu_code_identity_source": "git_checkout",
            "easyicu_version": version, "easyicu_installed_dist_version": "9.9.9",
        },
        "storage_reads": reads if reads is not None else [_flat(prepared, "cases", prepared)],
    }))
    return run


def test_the_sic_section_of_the_receipt_carries_the_binding_and_the_reads(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"])
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    built = receipt.build_run_receipt(
        run, CODE["commit"], [f"sic:{prepared}:144-145"], scratch=tmp_path / "scratch",
        started_at=0.0, parameters={"BENCH_MEMMAX": "14G"},
    )
    path = receipt.write_run_receipt(run, built)

    saved = json.loads(path.read_text())
    sic = saved["databases"]["sic"]
    assert saved["sealed_sha"] == CODE["commit"]
    assert sic["cases_binding"]["reproduction"]["result"] in {"byte_identical", "value_identical"}
    assert sic["storage_reads"] == [_flat(prepared, "cases", prepared)]
    assert sic["data_root_realpath"] == os.path.realpath(prepared)
    assert sic["native_manifest"]["easyicu_git_commit"] == CODE["commit"]
    assert saved["easyicu_version"] == CODE["version"]
    assert sic["native_manifest"]["easyicu_version"] == CODE["version"]
    assert sic["native_manifest"]["easyicu_installed_dist_version"] == "9.9.9"
    with pytest.raises(receipt.ReceiptError) as error:
        receipt.write_run_receipt(run, built)
    assert error.value.code == "run_receipt_exists"


def test_a_native_manifest_from_other_code_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit="d" * 40)
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == "native_manifest_not_sealed"


# F5: a SIC run whose data path is a layout root (bucket installations beside
# links into the prepared tree) says, per table read, where it was served from;
# a bucketed table carries its manifest and build-receipt sha256 and the
# prepared files it was derived from.

def _sic_layout_root(tmp_path: Path, prepared: Path) -> Path:
    from scripts.build_itemid_bucket_cache import build_bucket_cache

    lab = pd.DataFrame({"CaseID": [1, 2, 3, 1], "LaboratoryID": [7, 8, 9, 7],
                        "Offset": [0, 60, 120, 180], "Val": [1.0, 2.0, 3.0, 4.0]})
    (prepared / "laboratory").mkdir()
    lab.to_parquet(prepared / "laboratory" / "1.parquet", index=False)
    root = tmp_path / "b6_root" / "sic_108"
    (root / "laboratory").mkdir(parents=True)
    for name in ("cases.csv.gz", "cases.parquet", "conversion_manifest.json"):
        (root / name).symlink_to(prepared / name)
    (root / "laboratory" / "1.parquet").symlink_to(prepared / "laboratory" / "1.parquet")
    build_bucket_cache(root, table="laboratory", key="LaboratoryID", bucket_count=4,
                       memory_limit="256MB", threads=1)
    (root / "laboratory_bucket" / "_BUCKET_BUILD_RECEIPT.json").write_text("{}")
    return root


def _bucket_read(root: Path, table: str) -> dict:
    bucket = root / f"{table}_bucket"
    return {"root": str(root), "table": table, "layout": "bucket_hash",
            "layout_path": str(bucket), "layout_realpath": os.path.realpath(bucket)}


def _sic_receipt(tmp_path, root, reads, monkeypatch):
    run = _run_root(tmp_path, root, commit=CODE["commit"], reads=reads)
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))
    return receipt.build_run_receipt(run, CODE["commit"], [f"sic:{root}:144-145"],
                                     scratch=tmp_path / "scratch")


def test_a_sic_layout_root_receipt_maps_each_table_to_the_prepared_files(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    reads = [_flat(root, "cases", root), _bucket_read(root, "laboratory")]
    reads[0]["layout_realpath"] = os.path.realpath(prepared)

    sic = _sic_receipt(tmp_path, root, reads, monkeypatch)["databases"]["sic"]

    assert sic["input_path"] == str(root)
    assert sic["input_realpath"] == os.path.realpath(root)
    assert sic["data_root_realpath"] == os.path.realpath(prepared)
    cases, laboratory = sic["storage_reads"]
    assert "bucket_receipt" not in cases
    binding = laboratory["bucket_receipt"]
    bucket = root / "laboratory_bucket"
    assert binding["layout_realpath"] == os.path.realpath(bucket)
    assert binding["manifest"]["sha256"] == receipt.sha256_file(bucket / "_BUCKET_MANIFEST.json")
    assert binding["build_receipt"]["sha256"] == receipt.sha256_file(
        bucket / "_BUCKET_BUILD_RECEIPT.json")
    (source,) = binding["derived_from"]["files"]
    real = prepared / "laboratory" / "1.parquet"
    assert source["realpath"] == os.path.realpath(real)
    assert source["sha256"] == receipt.sha256_file(real)
    assert sic["cases_binding"]["cases_parquet"]["realpath"] == os.path.realpath(
        prepared / "cases.parquet")


def test_a_bucket_derived_from_files_outside_the_prepared_tree_stops_the_receipt(
        tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    # A copy of the table that is not the prepared file, with the bucket rebuilt from it.
    link = root / "laboratory" / "1.parquet"
    link.unlink()
    pd.read_parquet(prepared / "laboratory" / "1.parquet").to_parquet(link, index=False)
    shutil.rmtree(root / "laboratory_bucket")
    from scripts.build_itemid_bucket_cache import build_bucket_cache

    build_bucket_cache(root, table="laboratory", key="LaboratoryID", bucket_count=4,
                       memory_limit="256MB", threads=1)

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, [_bucket_read(root, "laboratory")], monkeypatch)
    assert error.value.code == "bucket_source_outside_input"


def test_a_bucket_whose_source_changed_since_the_build_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    os.utime(prepared / "laboratory" / "1.parquet", ns=(1_800_000_000_000_000_000,) * 2)

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, [_bucket_read(root, "laboratory")], monkeypatch)
    assert error.value.code == "bucket_receipt_unverifiable"
    assert error.value.details["reason"] == "bucket_derived_from_source_changed"


def test_a_read_without_its_layout_directory_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    reads = [{"root": str(prepared), "table": "cases", "layout": "flat"}]

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, reads, monkeypatch)
    assert error.value.code == "storage_read_layout_missing"


def test_a_flat_read_outside_the_data_root_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, [_flat(prepared, "cases", elsewhere)], monkeypatch)
    assert error.value.code == "storage_read_outside_input"


def test_a_native_manifest_naming_another_version_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"], version="9.9.9")
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == "native_manifest_version_mismatch"
