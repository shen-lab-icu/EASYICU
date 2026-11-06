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

CODE = {"source": "git_checkout", "commit": "c" * 40, "dirty": False, "import_path": "/x"}


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


def _run_root(tmp_path: Path, prepared: Path, *, commit: str) -> Path:
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
        },
        "storage_reads": [{"root": str(prepared), "table": "cases", "layout": "flat"}],
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
    assert sic["storage_reads"] == [{"root": str(prepared), "table": "cases", "layout": "flat"}]
    assert sic["native_manifest"]["easyicu_git_commit"] == CODE["commit"]
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
