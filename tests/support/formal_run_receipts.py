"""Synthetic SIC inputs and formal run logs for the receipt contracts."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from tests.support.formal_tools import load_receipt_module

receipt = load_receipt_module("formal_run_receipt")
TOOL = Path(receipt.__file__)


CODE = {"source": "git_checkout", "commit": "c" * 40, "dirty": False, "diff_sha256": None,
        "version": "0.1.2", "installed_dist_version": "9.9.9", "import_path": "/x"}


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


def _flat(root: Path, table: str, directory: Path) -> dict:
    return {"root": str(root), "table": table, "layout": "flat",
            "layout_path": str(directory), "layout_realpath": os.path.realpath(directory)}


_REAL_ADMISSION = receipt._admit_consumed


@pytest.fixture(autouse=True)
def _admission(monkeypatch):
    """Stand in for the storage owner's admission rule (tests of the rule use ``real_admission``)."""
    monkeypatch.setattr(receipt, "_admit_consumed", lambda base, table, path: {
        "target_realpath": os.path.realpath(path), "admitted_by": "data_root",
        "size": None, "sha256": None,
    })


@pytest.fixture
def real_admission(monkeypatch):
    monkeypatch.setattr(receipt, "_admit_consumed", _REAL_ADMISSION)


_REAL_SEAL_PUBLISHED = getattr(receipt, "_seal_published", None)


_TOOL_RECORDS = [{"name": "formal_run_receipt.py", "path": str(TOOL),
                  "seal_path": "tools/formal_run_receipt.py", "sha256": "d" * 64}]


@pytest.fixture(autouse=True)
def _tool_scripts(monkeypatch):
    """Stand in for the formal tool identity (test_r13_formal_tool_identity.py tests it)."""
    monkeypatch.setattr(receipt, "formal_tool_scripts",
                        lambda sha, code, run_root, approved_sha: list(_TOOL_RECORDS))
    monkeypatch.setattr(receipt, "formal_import_origins",
                        lambda run_root, sha, code, databases: {"receipt": {"live": True}})
    # Seal publication is tested on its own below (item 4); fixtures carry no seal receipts.
    monkeypatch.setattr(receipt, "_seal_published", lambda db, reads: [], raising=False)


def _consumed(path: Path, table: str = "cases", fmt: str = "parquet") -> dict:
    return {"table": table, "path": str(path), "realpath": os.path.realpath(path),
            "format": fmt}


def _snapshots(run: Path, input_root: Path, db="sic") -> None:
    """What bench7 writes: the input snapshot before the jobs and its verification after."""
    from easyicu import formal_inputs

    pre = formal_inputs.take_snapshot([f"{db}:{input_root}:144-145"])
    formal_inputs.write_snapshot(run / "INPUT_SNAPSHOT.json", pre)
    formal_inputs.write_snapshot(run / "INPUT_SNAPSHOT_POST.json",
                                 formal_inputs.verify_snapshot(pre))


def _binding_read(input_root: Path, read: dict) -> dict:
    """The storage owner's layout record of one read (``storage_binding["tables"]``)."""
    from easyicu.storage_layout import bucket_source_receipt, declared_view_chain

    if read["layout"] == "flat":
        return {"layout": "flat", "root": str(input_root),
                "derived_from": declared_view_chain(input_root)}
    held = bucket_source_receipt(read["layout_path"])
    return {"layout": "bucket", "root": read["layout_path"], "manifest": held["manifest"],
            "derived_from": held["derived_from"], "derived_from_recorded": True}


def _tables(input_root: Path, reads: list) -> dict:
    """What d5fixB's consumed_table_layouts publishes for these reads."""
    tables: dict = {}
    for read in reads:
        tables.setdefault(read["table"], []).append(_binding_read(input_root, read))
    out = {}
    for table, records in tables.items():
        primary = next((r for r in records if r["layout"] == "bucket"), records[0])
        out[table] = {**primary, "reads": records}
    return out


def _required(reads: list) -> dict:
    """REQUIRED_LAYOUT.json (schema /2) that these reads satisfy."""
    tables: dict = {}
    for read in reads:
        if read["layout"] == "flat":
            tables.setdefault(read["table"], {"layout": "flat"})
        else:
            tables[read["table"]] = {"layout": "bucket",
                                     "root_realpath": os.path.realpath(read["layout_path"])}
    return {"schema": "easyicu.required_layout/2", "complete": True,
            "databases": {"sic": tables}}


def _run_root(tmp_path: Path, prepared: Path, *, commit: str, reads=None,
              version: str = CODE["version"], consumed=None,
              installed: str | None = "9.9.9", tables=None, required=None,
              fallbacks=None, unavailable=(), db="sic") -> Path:
    run = tmp_path / "run"
    export = run / db / "export"
    export.mkdir(parents=True)
    _snapshots(run, prepared, db)
    consumed = consumed if consumed is not None else [_consumed(prepared / "cases.parquet")]
    if reads is None:
        # What the storage owner logs: the directory of the files really read.
        reads = [_flat(prepared, "cases", prepared)]
        parquet = [os.path.dirname(c["realpath"]) for c in consumed
                   if c["table"] == "cases" and c["format"] == "parquet"]
        if parquet:
            reads[0]["layout_realpath"] = os.path.commonpath(parquet)
    # The storage owner's run-log keys: every entry was consumed under this input.
    consumed = [{"event": "table_consumed", "root": str(prepared), **e} for e in consumed]
    if tables is None:
        tables = _tables(prepared, [r for r in reads if r.get("layout_path") or
                                    r["layout"] == "flat"])
    if required is None:
        required = _required(reads)
        required["databases"][db] = required["databases"].pop("sic")
    (run / "REQUIRED_LAYOUT.json").write_text(json.dumps(required))
    (run / "status").write_text(f"tag=t sha=x\n{db} exit=0\nDONE\n")
    (run / db / "summary.json").write_text(json.dumps({
        "secs": 12.5, "cgroup_peak_gib": 1.5, "peak_tree_rss_gib": 1.2,
        "cgroup_memory_max": "15032385536", "batch_size": None,
        "errors": [], "worker_failures": 0,
    }))
    (export / "_manifest.json").write_text(json.dumps({
        "runtime_provenance": {
            "easyicu_git_commit": commit, "easyicu_git_dirty": False,
            "easyicu_code_identity_source": "git_checkout",
            "easyicu_git_diff_sha256": None,
            "easyicu_version": version, "easyicu_installed_dist_version": installed,
        },
        "storage_reads": reads,
        "storage_binding": {"digest": "b" * 64, "files": 3, "scope": "parent_run_binding",
                            "tables": tables},
        "storage_consumed": consumed,
        **({"storage_fallbacks": fallbacks} if fallbacks is not None else {}),
        **({"storage_unavailable": list(unavailable)} if unavailable is not None else {}),
    }))
    return run


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
    manifest = json.loads((prepared / "conversion_manifest.json").read_text())
    manifest["tables"].append({"table": "laboratory", "status": "completed", "outputs": [
        {"relative_path": "laboratory/1.parquet",
         "size_bytes": (prepared / "laboratory" / "1.parquet").stat().st_size}]})
    (prepared / "conversion_manifest.json").write_text(json.dumps(manifest))
    build_bucket_cache(root, table="laboratory", key="LaboratoryID", bucket_count=4,
                       memory_limit="256MB", threads=1)
    (root / "laboratory_bucket" / "_BUCKET_BUILD_RECEIPT.json").write_text("{}")
    return root


def _bucket_read(root: Path, table: str) -> dict:
    bucket = root / f"{table}_bucket"
    return {"root": str(root), "table": table, "layout": "bucket_hash",
            "layout_path": str(bucket), "layout_realpath": os.path.realpath(bucket)}


def _sic_receipt(tmp_path, root, reads, monkeypatch, *, consumed=None, after_run=None,
                 **options):
    run = _run_root(tmp_path, root, commit=CODE["commit"], reads=reads, consumed=consumed,
                    **options)
    if after_run is not None:
        after_run(run)
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))
    return receipt.build_run_receipt(run, CODE["commit"], [f"sic:{root}:144-145"],
                                     scratch=tmp_path / "scratch")
