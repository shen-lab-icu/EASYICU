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
import time
from pathlib import Path

import pandas as pd
import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "formal_run_receipt.py"
SPEC = importlib.util.spec_from_file_location("formal_run_receipt", TOOL)
assert SPEC and SPEC.loader
receipt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = receipt
SPEC.loader.exec_module(receipt)

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
    assert sic["storage_binding"]["digest"] == "b" * 64
    assert sic["storage_binding"]["scope"] == "parent_run_binding"
    assert "parent extraction" in sic["storage_binding"]["note"]
    assert sic["native_manifest"]["easyicu_version"] == CODE["version"]
    assert sic["native_manifest"]["easyicu_installed_dist_version"] == "9.9.9"
    pre = json.loads((run / "INPUT_SNAPSHOT.json").read_text())["databases"]["sic"]
    assert saved["input_snapshot"]["sha256"] == receipt.sha256_file(run / "INPUT_SNAPSHOT.json")
    assert sic["input_snapshot"]["digest"] == pre["digest"]
    assert sic["input_snapshot"]["post_digest"] == pre["digest"]
    assert sic["input_snapshot"]["post_equal"] is True
    assert sic["input_snapshot"]["bytes"] == pre["bytes"]
    (consumed,) = sic["storage_consumed"]["files"]
    assert consumed["admitted_by"] == "data_root"
    assert consumed["sha256"] == receipt.sha256_file(prepared / "cases.parquet")
    assert "declared_view_chain" not in sic["storage_consumed"]
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


def test_a_bucket_whose_source_changed_after_the_snapshot_stops_the_receipt(
        tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)

    def touch(run):
        os.utime(prepared / "laboratory" / "1.parquet", ns=(1_800_000_000_000_000_000,) * 2)

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, [_bucket_read(root, "laboratory")], monkeypatch,
                     after_run=touch)
    assert error.value.code == "bucket_receipt_unverifiable"
    assert error.value.details["reason"] == "bucket_derived_from_source_changed"


def test_a_read_without_its_layout_directory_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    reads = [{"root": str(prepared), "table": "cases", "layout": "flat"}]

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, reads, monkeypatch)
    assert error.value.code == "formal_read_set_unknown"


# delta9b ruling: the run's inputs are hashed before and after; every consumed
# file is a snapshotted file.

def _rewrite_post(run, change):
    """Edit the post-run snapshot's file list and keep it self-consistent."""
    from easyicu.formal_inputs import snapshot_digest

    post = json.loads((run / "INPUT_SNAPSHOT_POST.json").read_text())
    entry = post["databases"]["sic"]
    change(entry)
    entry["digest"] = snapshot_digest(entry["files"])
    entry["file_count"] = len(entry["files"])
    entry["bytes"] = sum(f["size"] for f in entry["files"])
    (run / "INPUT_SNAPSHOT_POST.json").write_text(json.dumps(post))


def test_inputs_that_differ_before_and_after_the_run_stop_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    def tamper(run):
        _rewrite_post(run, lambda e: e["files"][0].update(sha256="0" * 64))

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, after_run=tamper)
    assert error.value.code == "input_changed_during_run"


# delta14v B-SNAPSHOT: the consumer recomputes both snapshots from their files.

def _forge_post(change):
    def forge(run):
        post = json.loads((run / "INPUT_SNAPSHOT_POST.json").read_text())
        change(post["databases"]["sic"])
        (run / "INPUT_SNAPSHOT_POST.json").write_text(json.dumps(post))
    return forge


@pytest.mark.parametrize(("forge", "code"), [
    # The reviewer's case: same self-reported digest, emptied file list.
    (_forge_post(lambda e: e.update(files=[])), "input_snapshot_inconsistent"),
    (_forge_post(lambda e: e["files"].pop()), "input_snapshot_inconsistent"),
    (_forge_post(lambda e: e.update(file_count=e["file_count"] + 1)),
     "input_snapshot_inconsistent"),
    (_forge_post(lambda e: e.update(bytes=e["bytes"] + 1)), "input_snapshot_inconsistent"),
    (_forge_post(lambda e: e["files"][0].update(sha256="0" * 64)),
     "input_snapshot_inconsistent"),
    (_forge_post(lambda e: e.update(data_path="/elsewhere", data_path_realpath="/elsewhere")),
     "input_snapshot_other_input"),
    (_forge_post(lambda e: e.update(database="eicu")), "input_snapshot_other_input"),
    # Self-consistent but a different set: a file dropped, or another file named.
    (lambda run: _rewrite_post(run, lambda e: e["files"].pop()), "input_changed_during_run"),
    (lambda run: _rewrite_post(run, lambda e: e["files"][0].update(realpath="/x/other")),
     "input_changed_during_run"),
    (lambda run: _rewrite_post(run, lambda e: e.update(buckets=[{"table": "x"}])),
     "input_changed_during_run"),
])
def test_a_forged_or_changed_post_run_snapshot_stops_the_receipt(
        tmp_path, monkeypatch, forge, code):
    prepared = _prepared(tmp_path / "prepared")

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, after_run=forge)
    assert error.value.code == code


def test_a_forged_pre_run_snapshot_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    def forge(run):
        for name in ("INPUT_SNAPSHOT.json", "INPUT_SNAPSHOT_POST.json"):
            payload = json.loads((run / name).read_text())
            payload["databases"]["sic"]["files"] = payload["databases"]["sic"]["files"][:1]
            (run / name).write_text(json.dumps(payload))

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, after_run=forge)
    assert error.value.code == "input_snapshot_inconsistent"


def test_a_run_without_its_post_run_snapshot_has_no_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch,
                     after_run=lambda run: (run / "INPUT_SNAPSHOT_POST.json").unlink())
    assert error.value.code == "receipt_input_missing"


def test_a_consumed_file_outside_the_snapshot_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    stray = tmp_path / "elsewhere" / "cases.parquet"
    stray.parent.mkdir()
    stray.write_bytes((prepared / "cases.parquet").read_bytes())

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, consumed=[_consumed(stray)])
    assert error.value.code == "storage_consumed_outside_snapshot"


def test_a_run_that_consumed_nothing_has_no_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, consumed=[])
    assert error.value.code == "storage_consumed_empty"


def test_a_consumed_link_that_now_resolves_elsewhere_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    entry = _consumed(root / "cases.parquet")

    def retarget(run):
        (root / "cases.parquet").unlink()
        (root / "cases.parquet").symlink_to(prepared / "cases.csv.gz")

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, None, monkeypatch, consumed=[entry], after_run=retarget)
    assert error.value.code == "storage_consumed_realpath_changed"


def test_a_bucket_read_must_be_an_installation_the_snapshot_covers(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    read = _bucket_read(root, "laboratory")

    def drop_bucket(run):
        for name in ("INPUT_SNAPSHOT.json", "INPUT_SNAPSHOT_POST.json"):
            payload = json.loads((run / name).read_text())
            payload["databases"]["sic"]["buckets"] = []
            (run / name).write_text(json.dumps(payload))

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, [read], monkeypatch, after_run=drop_bucket)
    assert error.value.code == "storage_read_outside_snapshot"


# B9P-CSV-BINDING with the storage owner's admission rule.

def test_csv_and_parquet_consumed_side_by_side_are_each_bound(tmp_path, monkeypatch, real_admission):
    prepared = _prepared(tmp_path / "prepared")
    consumed = [_consumed(prepared / "cases.parquet"),
                _consumed(prepared / "cases.csv.gz", fmt="csv")]

    sic = _sic_receipt(tmp_path, prepared, None, monkeypatch,
                       consumed=consumed)["databases"]["sic"]

    files = {f["format"]: f for f in sic["storage_consumed"]["files"]}
    assert set(files) == {"csv", "parquet"}
    assert files["csv"]["sha256"] == receipt.sha256_file(prepared / "cases.csv.gz")
    assert all(f["admitted_by"] == "data_root" for f in files.values())
    assert "declared_view_chain" not in sic["storage_consumed"]


def test_a_csv_linked_from_outside_the_data_root_is_refused(tmp_path, monkeypatch, real_admission):
    prepared = _prepared(tmp_path / "prepared")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "cases.csv.gz").write_bytes((prepared / "cases.csv.gz").read_bytes())
    root = tmp_path / "root"
    root.mkdir()
    for name in ("cases.parquet", "conversion_manifest.json"):
        (root / name).symlink_to(prepared / name)
    (root / "cases.csv.gz").symlink_to(outside / "cases.csv.gz")

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, None, monkeypatch,
                     consumed=[_consumed(root / "cases.csv.gz", fmt="csv")])
    assert error.value.code in {"storage_consumed_not_admitted",
                                "storage_consumed_outside_snapshot"}


def _view(tmp_path: Path, prepared: Path) -> Path:
    import hashlib

    farm = tmp_path / "farm"
    farm.mkdir()
    for name in ("cases.parquet", "cases.csv.gz"):
        (farm / name).symlink_to(prepared / name)
    manifest = json.loads((prepared / "conversion_manifest.json").read_text())
    manifest["derived_from"] = {
        "path": str(prepared / "conversion_manifest.json"),
        "sha256": hashlib.sha256((prepared / "conversion_manifest.json").read_bytes()).hexdigest(),
    }
    (farm / "conversion_manifest.json").write_text(json.dumps(manifest))
    return farm


def test_a_declared_view_read_records_its_chain(tmp_path, monkeypatch, real_admission):
    prepared = _prepared(tmp_path / "prepared")
    farm = _view(tmp_path, prepared)

    sic = _sic_receipt(tmp_path, farm, None, monkeypatch,
                       consumed=[_consumed(farm / "cases.parquet")])["databases"]["sic"]

    (entry,) = sic["storage_consumed"]["files"]
    assert entry["admitted_by"] == "declared_view"
    chain = sic["storage_consumed"]["declared_view_chain"]
    assert chain["source_manifest_path"] == str(prepared / "conversion_manifest.json")
    assert chain["farm_manifest_sha256"] == receipt.sha256_file(farm / "conversion_manifest.json")


def test_a_csv_preferring_reader_is_receipted_by_the_csv_it_consumed(tmp_path, monkeypatch, real_admission):
    # MIMIC-III services prefers CSV over a converted parquet beside it; the
    # consumed-files event, not the parquet resolver, must name the CSV.
    from easyicu.datasource import ICUDataSource
    from easyicu.resources import load_data_sources
    from easyicu.storage_layout import SCAN_FALLBACK_LOG_ENV, read_consumed_log

    root = tmp_path / "mimic"
    root.mkdir()
    frame = pd.DataFrame({"row_id": [1], "subject_id": [10], "hadm_id": [100],
                          "transfertime": ["2150-01-01 00:00:00"], "prev_service": [None],
                          "curr_service": ["MED"]})
    frame.to_csv(root / "services.csv.gz", index=False)
    frame.assign(curr_service="SURG").to_parquet(root / "services.parquet", index=False)
    # Hospital tables are mapped to ICU stays: a MIMIC-III source has icustays.
    pd.DataFrame({"subject_id": [10], "hadm_id": [100], "icustay_id": [1000],
                  "intime": pd.to_datetime(["2150-01-01"]),
                  "outtime": pd.to_datetime(["2150-01-03"])}).to_parquet(
        root / "icustays.parquet", index=False)
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": [
        {"table": "services", "status": "completed",
         "input": {"relative_path": "services.csv.gz"},
         "outputs": [{"relative_path": "services.parquet"}]},
        {"table": "icustays", "status": "completed",
         "outputs": [{"relative_path": "icustays.parquet"}]}]}))
    log = tmp_path / "run_storage_log.jsonl"
    monkeypatch.setenv(SCAN_FALLBACK_LOG_ENV, str(log))
    ICUDataSource(config=load_data_sources().get("mimic"), base_path=root).load_table("services")
    consumed = [dict(e) for e in read_consumed_log(log) if e["table"] == "services"]

    assert [c["format"] for c in consumed] == ["csv"]
    run = tmp_path / "run"
    (run / "mimic" / "export").mkdir(parents=True)
    from easyicu import formal_inputs

    pre = formal_inputs.take_snapshot([f"mimic:{root}"])
    entry = receipt._consumed("mimic", str(root), consumed, pre["databases"]["mimic"])
    (named,) = entry["files"]
    # The reader may name it SERVICES.csv.gz (case-insensitive file systems);
    # then the receipt says it matched the snapshot by (device, inode).
    assert os.path.samefile(named["realpath"], root / "services.csv.gz")
    if named["realpath"] != os.path.realpath(root / "services.csv.gz"):
        assert named["snapshot_realpath"] == os.path.realpath(root / "services.csv.gz")
        assert named["note"].startswith("matched by (device, inode)")
    else:
        assert "note" not in named
    assert named["admitted_by"] == "data_root"
    assert named["sha256"] == receipt.sha256_file(root / "services.csv.gz")


def test_a_native_manifest_naming_another_version_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"], version="9.9.9")
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == "native_manifest_version_mismatch"
    assert error.value.details == {"database": "sic", "manifest_version": "9.9.9",
                                   "sealed_version": CODE["version"]}


@pytest.mark.parametrize("installed", ["0.0.1", None])
def test_an_installed_dist_version_that_differs_never_stops_the_receipt(
        tmp_path, monkeypatch, installed):
    # Supervisor ruling: easyicu_installed_dist_version is diagnostic only; the
    # commit (and the code identity's own version) is authoritative.
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"], installed=installed)
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    built = receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                      scratch=tmp_path / "scratch")

    native = built["databases"]["sic"]["native_manifest"]
    assert native["easyicu_version"] == CODE["version"] == built["easyicu_version"]
    assert native["easyicu_installed_dist_version"] == installed
    assert built["code_identity"]["installed_dist_version"] == "9.9.9"


# delta9p NB9P-SCRATCH-CONTAINMENT: nothing the receipt writes may sit in an input.

@pytest.mark.parametrize("alias", [False, True])
def test_a_scratch_inside_the_input_is_refused_before_anything_is_written(
        tmp_path, monkeypatch, alias):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"])
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))
    scratch = prepared / "scratch"
    if alias:  # a link elsewhere that resolves into the prepared tree
        (tmp_path / "alias").symlink_to(prepared, target_is_directory=True)
        scratch = tmp_path / "alias" / "scratch"
    before = sorted(p.name for p in prepared.iterdir())

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=scratch)

    assert error.value.code == "scratch_inside_input"
    assert sorted(p.name for p in prepared.iterdir()) == before


def test_a_scratch_inside_the_real_prepared_tree_behind_a_layout_root_is_refused(
        tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    run = _run_root(tmp_path, root, commit=CODE["commit"])
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{root}:144-145"],
                                  scratch=prepared / "s")
    assert error.value.code == "scratch_inside_input"


def test_a_run_root_inside_the_input_is_refused(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(prepared / "run", CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == "run_root_inside_input"


# Malformed inputs are typed stops (delta9p NB).

@pytest.mark.parametrize(
    ("relative", "payload", "code"),
    [
        ("sic/summary.json", "{not json", "receipt_input_malformed"),
        ("sic/summary.json", None, "receipt_input_missing"),
        ("sic/export/_manifest.json", "[]", "receipt_input_malformed"),
        ("sic/export/_manifest.json", '{"runtime_provenance": "x"}', "receipt_input_malformed"),
    ],
)
def test_a_malformed_receipt_input_is_a_typed_stop(tmp_path, monkeypatch, relative, payload, code):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"])
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))
    if payload is None:
        (run / relative).unlink()
    else:
        (run / relative).write_text(payload)

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == code


def test_a_malformed_prepared_manifest_is_a_typed_stop(tmp_path):
    prepared = _prepared(tmp_path / "prepared")
    (prepared / "conversion_manifest.json").write_text("{broken")

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.sic_cases_binding(prepared, tmp_path / "scratch", CODE)
    assert error.value.code == "receipt_input_malformed"


def test_a_native_manifest_without_the_run_binding_digest_stops_the_receipt(
        tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"])
    path = run / "sic" / "export" / "_manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["storage_binding"]
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == "storage_binding_missing"


def test_a_different_file_with_the_same_contents_is_never_matched_by_identity(
        tmp_path, monkeypatch):
    # The (device, inode) fallback admits only the very same file: a copy with
    # identical bytes elsewhere is outside the snapshot.
    prepared = _prepared(tmp_path / "prepared")
    copy = tmp_path / "copy" / "CASES.parquet"
    copy.parent.mkdir()
    copy.write_bytes((prepared / "cases.parquet").read_bytes())

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, consumed=[_consumed(copy)])
    assert error.value.code == "storage_consumed_outside_snapshot"


def test_a_hard_link_to_a_snapshotted_file_is_matched_with_a_note(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    alias = prepared / "cases_alias.parquet"
    os.link(prepared / "cases.parquet", alias)

    sic = _sic_receipt(tmp_path, prepared, None, monkeypatch,
                       consumed=[_consumed(alias)])["databases"]["sic"]

    (entry,) = sic["storage_consumed"]["files"]
    assert entry["snapshot_realpath"] == os.path.realpath(prepared / "cases.parquet")
    assert "matched by (device, inode)" in entry["note"]
    assert entry["sha256"] == receipt.sha256_file(prepared / "cases.parquet")


def test_a_real_extraction_delivers_every_consumed_file_to_the_receipt(tmp_path):
    # End to end (D10D-CONSUMED-DELIVERY): extract_database's workers log
    # table_consumed into the run log; the parent reads it before deleting the
    # log, the native manifest publishes it, and the receipt verifies it
    # against a snapshot of the same input.
    from easyicu import formal_inputs
    from easyicu.api.concepts import clear_global_loader
    from easyicu.api.extraction import extract_database
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    outputs = {}
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root)
        table = relative.parts[0] if len(relative.parts) > 1 else path.stem
        outputs.setdefault(table, []).append({"relative_path": relative.as_posix()})
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": [
        {"table": t, "status": "completed", "outputs": o} for t, o in outputs.items()]}))
    snapshot = formal_inputs.take_snapshot([f"hirid:{root}"])["databases"]["hirid"]

    output = tmp_path / "export"
    try:
        result = extract_database(
            "hirid", data_path=str(root), output_dir=str(output),
            modules=["outcome", "vitals"], native_export_v2=True, verbose=False,
        )
    finally:
        clear_global_loader()

    published = json.loads((output / "_manifest.json").read_text())
    consumed = published["storage_consumed"]
    assert consumed == result["storage_consumed"] and consumed
    # Every table any worker read is among the consumed files.
    assert {e["table"] for e in published["storage_reads"]} <= {e["table"] for e in consumed}
    verified = receipt._consumed("hirid", str(root), consumed, snapshot)
    assert verified["count"] == len(consumed)
    assert all(f["admitted_by"] == "data_root" for f in verified["files"]
               if f["format"] in {"csv", "parquet"})
    assert all(set(f) >= {"table", "path", "realpath", "format"} and not set(f) & {"event", "root"}
               for f in verified["files"])
    assert published["storage_binding"]["digest"] == result["storage_binding"]["digest"]
    # G2 on the real record: every consumed and read table has a layout
    # record, every read in it is checked, and the run used no fallback.
    observed = receipt._layout_tables("hirid", str(root), published["storage_binding"],
                                      verified["files"], published["storage_reads"], snapshot)
    assert set(observed) == {e["table"] for e in consumed}
    assert not published.get("storage_fallbacks")
    # Every declared table this run read had a physical source.
    assert published["storage_unavailable"] == []
    assert receipt._storage_unavailable("hirid", published) == []


# G1: the code that ran carries no working-tree diff.

@pytest.mark.parametrize("diff", ["a" * 64, "<absent>"])
def test_a_sealed_checkout_with_a_diff_or_no_diff_field_is_refused(monkeypatch, diff):
    import easyicu.code_identity as code_identity

    identity = {"source": "git_checkout", "commit": "c" * 40, "dirty": False, "version": "0.1.2"}
    if diff != "<absent>":
        identity["diff_sha256"] = diff
    monkeypatch.setattr(code_identity, "resolve_code_identity", lambda root: dict(identity))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.sealed_code_identity("c" * 40)
    assert error.value.code == "code_identity_diff_present"


def test_a_clean_sealed_checkout_records_its_null_diff(monkeypatch):
    import easyicu.code_identity as code_identity

    monkeypatch.setattr(code_identity, "resolve_code_identity", lambda root: {
        "source": "git_checkout", "commit": "c" * 40, "dirty": False, "diff_sha256": None,
        "version": "0.1.2"})

    assert receipt.sealed_code_identity("c" * 40)["diff_sha256"] is None


@pytest.mark.parametrize("diff", ["a" * 64, "<absent>"])
def test_a_native_manifest_with_a_diff_or_no_diff_field_stops_the_receipt(
        tmp_path, monkeypatch, diff):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"])
    path = run / "sic" / "export" / "_manifest.json"
    manifest = json.loads(path.read_text())
    if diff == "<absent>":
        del manifest["runtime_provenance"]["easyicu_git_diff_sha256"]
    else:
        manifest["runtime_provenance"]["easyicu_git_diff_sha256"] = diff
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == "native_manifest_diff_present"


def test_the_receipt_copies_both_null_diffs(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    built = _sic_receipt(tmp_path, prepared, None, monkeypatch)

    assert built["code_identity"]["diff_sha256"] is None
    assert built["databases"]["sic"]["native_manifest"]["easyicu_git_diff_sha256"] is None


# G5: the run log's event/root keys are checked and dropped.

def test_consumed_entries_carry_no_run_log_keys(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    sic = _sic_receipt(tmp_path, prepared, None, monkeypatch)["databases"]["sic"]

    (entry,) = sic["storage_consumed"]["files"]
    assert not {"event", "root"} & set(entry)


@pytest.mark.parametrize(("change", "code"), [
    ({"root": "/elsewhere"}, "storage_consumed_other_root"),
    ({"event": "table_layout"}, "receipt_input_malformed"),
])
def test_a_consumed_event_of_another_root_or_kind_stops_the_receipt(
        tmp_path, monkeypatch, change, code):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"])
    path = run / "sic" / "export" / "_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["storage_consumed"][0].update(change)
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == code


# G2: storage_layout_unrecorded keyed on storage_binding["tables"]; every read checked.

def _layout_root_reads(tmp_path):
    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    reads = [_flat(root, "cases", root), _bucket_read(root, "laboratory")]
    reads[0]["layout_realpath"] = os.path.realpath(prepared)
    return prepared, root, reads


def test_a_layout_root_receipt_checks_every_layout_record(tmp_path, monkeypatch):
    _prepared_root, root, reads = _layout_root_reads(tmp_path)

    sic = _sic_receipt(tmp_path, root, reads, monkeypatch)["databases"]["sic"]

    assert sic["layout_requirement"]["observed"] == {
        "cases": [{"layout": "flat", "root_realpath": os.path.realpath(root)}],
        "laboratory": [{"layout": "bucket",
                        "root_realpath": os.path.realpath(root / "laboratory_bucket")}],
    }
    assert sic["storage_binding"]["tables"]["laboratory"]["derived_from_recorded"] is True


@pytest.mark.parametrize("drop", ["consumed_table", "read_table", "all"])
def test_a_table_without_a_layout_record_stops_the_receipt(tmp_path, monkeypatch, drop):
    _prepared_root, root, reads = _layout_root_reads(tmp_path)
    tables = _tables(root, reads)
    if drop == "consumed_table":
        del tables["cases"]
    elif drop == "read_table":
        del tables["laboratory"]
    else:
        tables = None

    run = _run_root(tmp_path, root, commit=CODE["commit"], reads=reads, tables={})
    path = run / "sic" / "export" / "_manifest.json"
    manifest = json.loads(path.read_text())
    if tables is None:
        del manifest["storage_binding"]["tables"]
    else:
        manifest["storage_binding"]["tables"] = tables
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{root}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == "storage_layout_unrecorded"


def _tampered_tables(root, reads, mutate):
    tables = _tables(root, reads)
    mutate(tables)
    return tables


def _second_flat_read_elsewhere(tables):
    # Not the primary record: every read is checked, not only the first.
    tables["laboratory"]["reads"].append({"layout": "flat", "root": "/elsewhere",
                                          "derived_from": None})


@pytest.mark.parametrize(("mutate", "code"), [
    (_second_flat_read_elsewhere, "storage_read_other_root"),
    (lambda t: t["laboratory"]["reads"][0].update(derived_from_recorded=False),
     "bucket_derived_from_unrecorded"),
    (lambda t: t["laboratory"]["reads"][0]["manifest"].update(sha256="0" * 64),
     "bucket_manifest_not_snapshotted"),
    (lambda t: t["laboratory"]["reads"][0]["derived_from"]["files"][0].update(sha256="0" * 64),
     "bucket_derived_from_sha_mismatch"),
    (lambda t: t["laboratory"]["reads"][0]["derived_from"]["files"][0].update(
        realpath="/elsewhere/1.parquet"), "bucket_derived_from_outside_snapshot"),
    (lambda t: t["cases"]["reads"][0].update(derived_from={"farm_manifest_sha256": "0" * 64}),
     "declared_view_chain_mismatch"),
    (lambda t: t["laboratory"]["reads"][0].update(layout="flat", root="x") or
     t["laboratory"].update(layout="flat"), "storage_read_other_root"),
])
def test_every_layout_record_read_is_checked_against_the_snapshot(
        tmp_path, monkeypatch, mutate, code):
    _prepared_root, root, reads = _layout_root_reads(tmp_path)
    tables = _tampered_tables(root, reads, mutate)

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, reads, monkeypatch, tables=tables)
    assert error.value.code == code


def test_a_read_whose_layout_disagrees_with_the_record_stops_the_receipt(tmp_path, monkeypatch):
    _prepared_root, root, reads = _layout_root_reads(tmp_path)
    tables = _tables(root, reads)
    tables["laboratory"] = {"layout": "flat", "root": str(root), "derived_from": None,
                            "reads": [{"layout": "flat", "root": str(root), "derived_from": None}]}

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, reads, monkeypatch, tables=tables)
    assert error.value.code == "storage_layout_disagrees"


# G3: a storage fallback in a formal run is a stop, with counts per reason.

def test_a_storage_fallback_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    fallbacks = [{"reason": "bucket_cache_invalid", "root": str(prepared), "table": "a"},
                 {"reason": "bucket_cache_invalid", "root": str(prepared), "table": "b"},
                 {"reason": "unmanifested", "root": str(prepared), "table": "c"}]

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, fallbacks=fallbacks)
    assert error.value.code == "storage_fallback_in_formal_run"
    assert error.value.details["count"] == 3
    assert error.value.details["kinds"] == {"bucket_cache_invalid": 2, "unmanifested": 1}


def test_a_run_without_fallbacks_records_none(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    sic = _sic_receipt(tmp_path, prepared, None, monkeypatch, fallbacks=[])["databases"]["sic"]

    assert sic["storage_fallbacks"] == []


# G4: REQUIRED_LAYOUT.json (schema /2), complete, with its sha256: every read
# of every table matches the required layout AND root realpath, else a stop.

def _schema2(tables: dict) -> dict:
    return {"schema": "easyicu.required_layout/2", "complete": True,
            "databases": {"sic": tables}}


def test_the_required_layout_is_recorded_with_its_sha256(tmp_path, monkeypatch):
    _prepared_root, root, reads = _layout_root_reads(tmp_path)
    bucket = os.path.realpath(root / "laboratory_bucket")
    required = _schema2({
        "cases": {"layout": "flat"},
        "laboratory": {"layout": "bucket", "root_realpath": bucket},
        "medication": {"layout": "bucket", "root_realpath": "/elsewhere/medication_bucket"},
    })

    built = _sic_receipt(tmp_path, root, reads, monkeypatch, required=required)

    run = tmp_path / "run"
    assert built["required_layout"]["sha256"] == receipt.sha256_file(run / "REQUIRED_LAYOUT.json")
    assert built["required_layout"]["complete"] is True
    check = built["databases"]["sic"]["layout_requirement"]
    assert check["not_read"] == ["medication"]
    assert check["required"]["cases"] == {"layout": "flat",
                                          "root_realpath": os.path.realpath(root)}
    assert check["observed"]["laboratory"] == [{"layout": "bucket", "root_realpath": bucket}]


@pytest.mark.parametrize("spelling", ["realpath", "link", "explicit_flat"])
def test_a_bucket_at_its_required_root_passes_however_the_root_is_spelled(
        tmp_path, monkeypatch, spelling):
    _prepared_root, root, reads = _layout_root_reads(tmp_path)
    bucket = root / "laboratory_bucket"
    cases = {"layout": "flat"}
    if spelling == "link":  # a symlink to the same installation resolves to it
        (tmp_path / "bucket_link").symlink_to(bucket, target_is_directory=True)
        bucket = tmp_path / "bucket_link"
    if spelling == "explicit_flat":
        cases["root_realpath"] = os.path.realpath(root)
    required = _schema2({"cases": cases,
                         "laboratory": {"layout": "bucket", "root_realpath": str(bucket)}})

    sic = _sic_receipt(tmp_path, root, reads, monkeypatch, required=required)["databases"]["sic"]

    assert sic["layout_requirement"]["not_read"] == []


def _deviation_cases(tmp_path, root, prepared):
    bucket = os.path.realpath(root / "laboratory_bucket")
    flat = {"layout": "flat", "root_realpath": os.path.realpath(root)}
    other = str(tmp_path / "b6_root" / "sic_other" / "laboratory_bucket")
    observed_bucket = {"layout": "bucket", "root_realpath": bucket}
    return {
        # The supervisor's rule: a bucket from another installation is a deviation.
        "wrong_bucket_root": (
            {"cases": {"layout": "flat"},
             "laboratory": {"layout": "bucket", "root_realpath": other}},
            {"table": "laboratory", "required": {"layout": "bucket",
                                                 "root_realpath": os.path.realpath(other)},
             "observed": observed_bucket}),
        "flat_explicit_other_root": (
            {"cases": {"layout": "flat", "root_realpath": str(prepared)},
             "laboratory": {"layout": "bucket", "root_realpath": bucket}},
            {"table": "cases", "required": {"layout": "flat",
                                            "root_realpath": os.path.realpath(prepared)},
             "observed": flat}),
        "bucket_required_flat": (
            {"cases": {"layout": "flat"}, "laboratory": {"layout": "flat"}},
            {"table": "laboratory", "required": flat, "observed": observed_bucket}),
        "flat_required_bucket": (
            {"cases": {"layout": "bucket", "root_realpath": bucket},
             "laboratory": {"layout": "bucket", "root_realpath": bucket}},
            {"table": "cases", "required": {"layout": "bucket", "root_realpath": bucket},
             "observed": flat}),
        "unlisted": (
            {"cases": {"layout": "flat"}},
            {"table": "laboratory", "required": None, "observed": observed_bucket}),
    }


@pytest.mark.parametrize("case", ["wrong_bucket_root", "flat_explicit_other_root",
                                  "bucket_required_flat", "flat_required_bucket", "unlisted"])
def test_a_read_from_another_layout_or_root_is_a_layout_deviation(tmp_path, monkeypatch, case):
    prepared, root, reads = _layout_root_reads(tmp_path)
    tables, deviation = _deviation_cases(tmp_path, root, prepared)[case]

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, reads, monkeypatch, required=_schema2(tables))
    assert error.value.code == "layout_deviation"
    assert error.value.details["deviations"] == [deviation]


def test_every_read_of_a_table_must_match_its_required_root(tmp_path, monkeypatch):
    # A flat read beside the required bucket is a second read of the table:
    # it does not match the bucket requirement.
    _prepared_root, root, reads = _layout_root_reads(tmp_path)
    tables = _tables(root, reads)
    tables["laboratory"]["reads"].append({"layout": "flat", "root": str(root),
                                          "derived_from": None})
    required = _schema2({"cases": {"layout": "flat"}, "laboratory": {
        "layout": "bucket", "root_realpath": os.path.realpath(root / "laboratory_bucket")}})

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, reads, monkeypatch, tables=tables, required=required)
    assert error.value.code == "layout_deviation"
    (deviation,) = error.value.details["deviations"]
    assert deviation["observed"] == {"layout": "flat", "root_realpath": os.path.realpath(root)}


_FLAT = {"cases": {"layout": "flat"}}


@pytest.mark.parametrize(("payload", "code"), [
    (None, "required_layout_missing"),
    ({**_schema2(_FLAT), "complete": False}, "required_layout_incomplete"),
    ({**_schema2(_FLAT), "databases": {"eicu": _FLAT}}, "required_layout_missing"),
    ({**_schema2(_FLAT), "schema": "easyicu.required_layout/1"}, "receipt_input_malformed"),
    ({**_schema2(_FLAT), "databases": {"sic": {"cases": "flat"}}}, "receipt_input_malformed"),
    (_schema2({"cases": {"layout": "parquet"}}), "receipt_input_malformed"),
    (_schema2({"cases": {"layout": "bucket"}}), "receipt_input_malformed"),
    (_schema2({"cases": {"layout": "bucket", "root_realpath": "rel/bucket"}}),
     "receipt_input_malformed"),
    (_schema2({"cases": {"layout": "flat", "root_realpath": None}}), "receipt_input_malformed"),
    (_schema2({"cases": {"layout": "flat", "root": "/x"}}), "receipt_input_malformed"),
    (_schema2({"cases": {"layout": "flat"}, "CASES": {"layout": "flat"}}),
     "receipt_input_malformed"),
    ({"complete": True, "databases": {"sic": _FLAT}}, "receipt_input_malformed"),
])
def test_a_missing_incomplete_or_malformed_required_layout_stops_the_receipt(
        tmp_path, monkeypatch, payload, code):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"])
    if payload is None:
        (run / "REQUIRED_LAYOUT.json").unlink()
    else:
        (run / "REQUIRED_LAYOUT.json").write_text(json.dumps(payload))
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == code


# delta14v B-CSV: a CSV the conversion manifest lists as the table's input is
# a read type of its own ("csv"), not a fallback; any other CSV read stays one.

def _mimic_services(tmp_path: Path, *, listed: bool) -> Path:
    root = tmp_path / "mimic"
    root.mkdir()
    frame = pd.DataFrame({"row_id": [1], "subject_id": [10], "hadm_id": [100],
                          "transfertime": ["2150-01-01 00:00:00"], "prev_service": [None],
                          "curr_service": ["MED"]})
    frame.to_csv(root / "services.csv.gz", index=False)
    frame.to_parquet(root / "services.parquet", index=False)
    pd.DataFrame({"subject_id": [10], "hadm_id": [100], "icustay_id": [1000],
                  "intime": pd.to_datetime(["2150-01-01"]),
                  "outtime": pd.to_datetime(["2150-01-03"])}).to_parquet(
        root / "icustays.parquet", index=False)
    services = {"table": "services", "status": "completed",
                "outputs": [{"relative_path": "services.parquet"}]}
    if listed:
        services["input"] = {"relative_path": "services.csv.gz"}
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": [
        services, {"table": "icustays", "status": "completed",
                   "outputs": [{"relative_path": "icustays.parquet"}]}]}))
    return root


def _services_run(tmp_path: Path, monkeypatch, *, listed: bool):
    """Load MIMIC-III services through the real reader; return the run's storage record."""
    from easyicu import formal_inputs
    from easyicu.datasource import ICUDataSource
    from easyicu.resources import load_data_sources
    from easyicu.storage_binding import binding_unit
    from easyicu.storage_layout import (
        SCAN_FALLBACK_LOG_ENV, consumed_table_layouts, read_consumed_log, read_layout_log,
        read_scan_fallback_log, read_table_read_log, table_layout_receipt,
    )

    root = _mimic_services(tmp_path, listed=listed)
    log = tmp_path / "run_storage_log.jsonl"
    monkeypatch.setenv(SCAN_FALLBACK_LOG_ENV, str(log))
    with binding_unit(None):
        ICUDataSource(config=load_data_sources().get("mimic"), base_path=root).load_table(
            "services")
    consumed = [dict(e) for e in read_consumed_log(log)]
    fallbacks = list(read_scan_fallback_log(log))
    reads = table_layout_receipt(read_table_read_log(log), fallbacks)
    tables = consumed_table_layouts(consumed, read_layout_log(log))
    snapshot = formal_inputs.take_snapshot([f"mimic:{root}"])["databases"]["mimic"]
    return root, consumed, fallbacks, reads, tables, snapshot


def test_a_listed_input_csv_read_is_a_csv_read_not_a_fallback(tmp_path, monkeypatch, real_admission):
    root, consumed, fallbacks, reads, tables, snapshot = _services_run(
        tmp_path, monkeypatch, listed=True)

    assert fallbacks == []
    assert [r["layout"] for r in reads if r["table"] == "services"] == ["csv"]
    assert tables["services"]["layout"] == "csv"
    verified = receipt._consumed("mimic", str(root), consumed, snapshot)
    (entry,) = [f for f in verified["files"] if f["table"] == "services"]
    assert entry["format"] == "csv" and entry["admitted_by"] == "data_root"
    observed = receipt._layout_tables("mimic", str(root), {"tables": tables},
                                      verified["files"], reads, snapshot)
    assert observed["services"] == [{"layout": "csv", "root_realpath": os.path.realpath(root)}]
    receipt._refuse_fallbacks("mimic", fallbacks)
    required = {"services": {"layout": "csv"}, "icustays": {"layout": "flat"}}
    check = receipt._layout_requirement("mimic", str(root), required, observed)
    assert check["observed"] == observed
    with pytest.raises(receipt.ReceiptError) as error:
        receipt._layout_requirement("mimic", str(root), {**required, "services": {
            "layout": "flat"}}, observed)
    assert error.value.code == "layout_deviation"


def test_an_unlisted_csv_read_beside_the_manifest_stays_a_refused_fallback(
        tmp_path, monkeypatch, real_admission):
    root, consumed, fallbacks, _reads, _tables, snapshot = _services_run(
        tmp_path, monkeypatch, listed=False)

    assert [f["reason"] for f in fallbacks] == ["table_read_csv_beside_manifest"]
    with pytest.raises(receipt.ReceiptError) as error:
        receipt._refuse_fallbacks("mimic", fallbacks)
    assert error.value.code == "storage_fallback_in_formal_run"
    # An unlisted CSV is not an input of the installation: not snapshotted.
    with pytest.raises(receipt.ReceiptError) as error:
        receipt._consumed("mimic", str(root), consumed, snapshot)
    assert error.value.code == "storage_consumed_outside_snapshot"


def test_a_consumed_csv_the_manifest_no_longer_lists_is_refused(
        tmp_path, monkeypatch, real_admission):
    root, consumed, _fallbacks, _reads, _tables, snapshot = _services_run(
        tmp_path, monkeypatch, listed=True)
    manifest = json.loads((root / "conversion_manifest.json").read_text())
    for entry in manifest["tables"]:
        entry.pop("input", None)
    (root / "conversion_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt._consumed("mimic", str(root), consumed, snapshot)
    assert error.value.code == "storage_consumed_csv_unlisted"


# delta14v B-NESTED: the snapshot enumerates buckets through the reader's own
# resolver (root, icu/, hosp/), so a nested installation is snapshotted.

def _nested_layout_root(tmp_path: Path, prepared: Path, where: str) -> Path:
    from scripts.build_itemid_bucket_cache import build_bucket_cache

    root = _sic_layout_root(tmp_path, prepared)
    shutil.rmtree(root / "laboratory_bucket")
    (root / where).mkdir()
    build_bucket_cache(root, table="laboratory", key="LaboratoryID", bucket_count=4,
                       memory_limit="256MB", threads=1,
                       output_root=root / where / "laboratory_bucket")
    return root


@pytest.mark.parametrize("where", ["icu", "hosp"])
def test_a_nested_bucket_is_snapshotted_and_receipted(tmp_path, monkeypatch, where):
    from easyicu import formal_inputs

    prepared = _prepared(tmp_path / "prepared")
    root = _nested_layout_root(tmp_path, prepared, where)
    nested = root / where / "laboratory_bucket"

    planned = formal_inputs.plan_input_files("sic", root)
    assert [b["layout_realpath"] for b in planned["buckets"]] == [os.path.realpath(nested)]
    read = {"root": str(root), "table": "laboratory", "layout": "bucket_hash",
            "layout_path": str(nested), "layout_realpath": os.path.realpath(nested)}
    cases = _flat(root, "cases", root)
    cases["layout_realpath"] = os.path.realpath(prepared)
    sic = _sic_receipt(tmp_path, root, [cases, read], monkeypatch)["databases"]["sic"]
    assert sic["layout_requirement"]["observed"]["laboratory"] == [
        {"layout": "bucket", "root_realpath": os.path.realpath(nested)}]


# delta14v B-TABLE-SOURCE: a bucket of table X must be built from X's own
# sources, exactly the outputs its conversion manifest lists for X.

def _register(prepared: Path, root: Path, table: str) -> None:
    """Give ``table`` a converted output and a conversion-manifest entry."""
    (prepared / table).mkdir()
    pd.DataFrame({"CaseID": [1], "DrugID": [5]}).to_parquet(
        prepared / table / "1.parquet", index=False)
    (root / table).mkdir()
    (root / table / "1.parquet").symlink_to(prepared / table / "1.parquet")
    manifest = json.loads((prepared / "conversion_manifest.json").read_text())
    manifest["tables"].append({"table": table, "status": "completed", "outputs": [
        {"relative_path": f"{table}/1.parquet",
         "size_bytes": (prepared / table / "1.parquet").stat().st_size}]})
    (prepared / "conversion_manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("how", ["copy", "link"])
def test_a_bucket_built_from_another_tables_sources_is_refused_by_the_snapshot(
        tmp_path, how):
    from easyicu import formal_inputs

    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    _register(prepared, root, "medication")
    if how == "copy":
        shutil.copytree(root / "laboratory_bucket", root / "medication_bucket")
    else:
        (root / "medication_bucket").symlink_to(root / "laboratory_bucket",
                                                target_is_directory=True)

    with pytest.raises(formal_inputs.InputSnapshotError) as error:
        formal_inputs.plan_input_files("sic", root)
    assert error.value.code == "bucket_derived_from_other_table"


def test_bucket_sources_that_are_not_the_owner_manifests_outputs_are_refused(tmp_path):
    from easyicu import formal_inputs

    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    extra = prepared / "laboratory_extra.parquet"
    extra.write_bytes((prepared / "laboratory" / "1.parquet").read_bytes())
    (root / "laboratory_extra.parquet").symlink_to(extra)
    manifest = json.loads((prepared / "conversion_manifest.json").read_text())
    for entry in manifest["tables"]:
        if entry["table"] == "laboratory":
            entry["outputs"].append({"relative_path": "laboratory_extra.parquet",
                                     "size_bytes": extra.stat().st_size})
    (prepared / "conversion_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(formal_inputs.InputSnapshotError) as error:
        formal_inputs.plan_input_files("sic", root)
    assert error.value.code == "bucket_derived_from_not_owner_outputs"


def _other_snapshotted_source(t, prepared):
    import hashlib

    path = prepared / "cases.parquet"
    t["laboratory"]["reads"][0]["derived_from"]["files"] = [{
        "path": "cases.parquet", "realpath": os.path.realpath(path), "size": path.stat().st_size,
        "mtime_ns": 0, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda t, p: t["laboratory"]["reads"][0]["derived_from"].update(table="cases"),
     "bucket_derived_from_other_table"),
    (lambda t, p: t["laboratory"]["reads"][0]["derived_from"].update(files=[]),
     "bucket_derived_from_empty"),
    (_other_snapshotted_source, "bucket_derived_from_not_snapshot"),
])
def test_a_bucket_read_must_name_its_own_tables_snapshotted_sources(
        tmp_path, monkeypatch, mutate, code):
    prepared, root, reads = _layout_root_reads(tmp_path)
    tables = _tables(root, reads)
    mutate(tables, prepared)

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, reads, monkeypatch, tables=tables)
    assert error.value.code == code


def test_a_snapshot_binding_the_installation_to_another_table_stops_the_receipt(
        tmp_path, monkeypatch):
    _prepared_root, root, reads = _layout_root_reads(tmp_path)

    def rebind(run):
        for name in ("INPUT_SNAPSHOT.json", "INPUT_SNAPSHOT_POST.json"):
            payload = json.loads((run / name).read_text())
            for bucket in payload["databases"]["sic"]["buckets"]:
                bucket["table"] = bucket["derived_table"] = "medication"
            (run / name).write_text(json.dumps(payload))

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, reads, monkeypatch, after_run=rebind)
    assert error.value.code == "bucket_derived_from_other_table"


# delta14v NBs.

def test_consumed_tables_without_any_storage_read_stop_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    tables = _tables(prepared, [_flat(prepared, "cases", prepared)])

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, [], monkeypatch, tables=tables,
                     required=_schema2({"cases": {"layout": "flat"}}))
    assert error.value.code == "storage_reads_missing"


def test_a_flat_read_naming_a_directory_it_did_not_read_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, [_flat(prepared, "cases", elsewhere)], monkeypatch)
    assert error.value.code == "storage_layout_disagrees"


def test_a_declared_view_read_without_its_chain_stops_the_receipt(
        tmp_path, monkeypatch, real_admission):
    prepared = _prepared(tmp_path / "prepared")
    farm = _view(tmp_path, prepared)
    tables = _tables(farm, [_flat(farm, "cases", farm)])
    tables["cases"]["derived_from"] = tables["cases"]["reads"][0]["derived_from"] = None

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, farm, None, monkeypatch, tables=tables,
                     consumed=[_consumed(farm / "cases.parquet")])
    assert error.value.code == "declared_view_chain_mismatch"


def test_extra_top_level_keys_in_the_required_layout_are_malformed(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"],
                    required={**_schema2(_FLAT), "note": "x"})
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == "receipt_input_malformed"


# delta16v B-UNUSED-NESTED: the snapshot resolves the conversion-registered
# tables only; any other on-disk installation is listed, never validated, and
# a run that reads one anyway is refused.

def _stale_copy(source: Path, target: Path) -> Path:
    """A legitimate-format installation without derived_from (an older build)."""
    target.parent.mkdir(exist_ok=True)
    shutil.copytree(source, target)
    manifest = json.loads((target / "_BUCKET_MANIFEST.json").read_text())
    manifest.pop("derived_from")
    (target / "_BUCKET_MANIFEST.json").write_text(json.dumps(manifest))
    return target


@pytest.mark.parametrize("where", ["icu", "hosp", "."])
def test_a_stale_bucket_no_registered_table_uses_is_listed_not_validated(
        tmp_path, monkeypatch, where):
    from easyicu import formal_inputs

    _prepared_root, root, reads = _layout_root_reads(tmp_path)
    stale = _stale_copy(root / "laboratory_bucket", root / where / "retired_bucket")

    planned = formal_inputs.plan_input_files("sic", root)

    assert planned["unused_installations"] == [{
        "path": str(root / where / "retired_bucket").replace("/./", "/"),
        "realpath": os.path.realpath(stale), "registered_table": False}]
    assert [b["table"] for b in planned["buckets"]] == ["laboratory"]
    sic = _sic_receipt(tmp_path, root, reads, monkeypatch)["databases"]["sic"]
    assert sic["layout_requirement"]["observed"]["laboratory"][0]["layout"] == "bucket"


def test_a_stale_nested_bucket_the_resolver_selects_is_refused(tmp_path):
    from easyicu import formal_inputs

    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    _stale_copy(root / "laboratory_bucket", root / "icu" / "laboratory_bucket")
    shutil.rmtree(root / "laboratory_bucket")  # the nested one is now selected

    with pytest.raises(formal_inputs.InputSnapshotError) as error:
        formal_inputs.plan_input_files("sic", root)
    assert error.value.code == "input_bucket_invalid"
    assert error.value.details["reason"] == "bucket_derived_from_missing"


def test_a_shadowed_installation_of_a_registered_table_is_listed_as_unused(tmp_path):
    from easyicu import formal_inputs

    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    stale = _stale_copy(root / "laboratory_bucket", root / "icu" / "laboratory_bucket")

    planned = formal_inputs.plan_input_files("sic", root)

    assert [b["layout_realpath"] for b in planned["buckets"]] == [
        os.path.realpath(root / "laboratory_bucket")]
    assert planned["unused_installations"] == [{
        "path": str(stale), "realpath": os.path.realpath(stale), "registered_table": True}]


def test_a_run_that_read_an_unused_installation_is_refused(tmp_path, monkeypatch):
    _prepared_root, root, reads = _layout_root_reads(tmp_path)
    stale = _stale_copy(root / "laboratory_bucket", root / "icu" / "retired_bucket")
    read = {"root": str(root), "table": "laboratory", "layout": "bucket_hash",
            "layout_path": str(stale), "layout_realpath": os.path.realpath(stale)}
    tables = _tables(root, reads[:1])
    tables["laboratory"] = {"layout": "bucket", "root": str(stale), "manifest": None,
                            "derived_from": None, "derived_from_recorded": False,
                            "reads": [{"layout": "bucket", "root": str(stale)}]}

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, [reads[0], read], monkeypatch, tables=tables,
                     required=_schema2({"cases": {"layout": "flat"}, "laboratory": {
                         "layout": "bucket", "root_realpath": os.path.realpath(stale)}}))
    assert error.value.code == "storage_read_unvalidated_installation"


def test_an_unused_installation_appearing_during_the_run_changes_the_inputs(
        tmp_path, monkeypatch):
    _prepared_root, root, reads = _layout_root_reads(tmp_path)

    def add_unused(run):
        _rewrite_post(run, lambda e: e.update(unused_installations=[
            {"path": "/x/retired_bucket", "realpath": "/x/retired_bucket",
             "registered_table": False}]))

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, reads, monkeypatch, after_run=add_unused)
    assert error.value.code == "input_changed_during_run"


# delta16v NBs: a flat read names exactly its files' common directory; a
# listed CSV is the listed path, not another hard link to it; a bucket read's
# manifest is its own installation's.

@pytest.mark.parametrize("field", ["layout_realpath", "layout_path"])
def test_a_flat_read_naming_an_ancestor_directory_stops_the_receipt(
        tmp_path, monkeypatch, field):
    prepared = _prepared(tmp_path / "prepared")
    read = _flat(prepared, "cases", prepared)
    read[field] = "/"

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, [read], monkeypatch)
    assert error.value.code == "storage_layout_disagrees"


def test_an_unlisted_hard_link_to_the_listed_csv_is_refused(tmp_path, monkeypatch, real_admission):
    from easyicu.storage_layout import listed_csv_input

    root, consumed, _fallbacks, _reads, _tables, snapshot = _services_run(
        tmp_path, monkeypatch, listed=True)
    alias = root / "services_copy.csv.gz"
    os.link(root / "services.csv.gz", alias)

    assert listed_csv_input(root, "services", root / "services.csv.gz")
    assert not listed_csv_input(root, "services", alias)
    entry = next(e for e in consumed if e["format"] == "csv")
    forged = [{**e, "path": str(alias), "realpath": os.path.realpath(alias)}
              if e is entry else e for e in consumed]
    with pytest.raises(receipt.ReceiptError) as error:
        receipt._consumed("mimic", str(root), forged, snapshot)
    assert error.value.code == "storage_consumed_csv_unlisted"


def test_a_bucket_read_naming_another_snapshotted_json_as_its_manifest_is_refused(
        tmp_path, monkeypatch):
    prepared, root, reads = _layout_root_reads(tmp_path)
    tables = _tables(root, reads)
    other = root / "conversion_manifest.json"
    tables["laboratory"]["reads"][0]["manifest"] = {
        "path": str(other), "sha256": receipt.sha256_file(other)}

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, reads, monkeypatch, tables=tables)
    assert error.value.code == "bucket_manifest_not_snapshotted"


# delta17v NB-CSV-CASEFOLD: letter case is folded only where the file system is
# shown (read-only) to be case-insensitive.

def test_the_case_probe_is_read_only_and_matches_the_file_system(tmp_path):
    from easyicu import storage_layout

    path = tmp_path / "services.csv.gz"
    path.write_bytes(b"x")
    before = sorted(p.name for p in tmp_path.iterdir())

    insensitive = storage_layout._case_insensitive_at(str(path))

    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert insensitive == (tmp_path / "SERVICES.CSV.GZ").exists()
    assert storage_layout._case_insensitive_at(str(tmp_path / "1234.gz")) is False


def _case_variant(root: Path) -> Path:
    """SERVICES.csv.gz: the listed file itself on a case-insensitive file
    system, an unlisted hard link to it on a case-sensitive one."""
    upper = root / "SERVICES.csv.gz"
    if not upper.exists():
        os.link(root / "services.csv.gz", upper)
    return upper


def test_a_hard_link_differing_only_in_case_on_a_case_sensitive_fs_is_refused(
        tmp_path, monkeypatch, real_admission):
    from easyicu import storage_layout

    root, consumed, _fallbacks, _reads, _tables, snapshot = _services_run(
        tmp_path, monkeypatch, listed=True)
    upper = _case_variant(root)
    monkeypatch.setattr(storage_layout, "_case_insensitive_at", lambda path: False)

    assert storage_layout.listed_csv_input(root, "services", root / "services.csv.gz")
    assert not storage_layout.listed_csv_input(root, "services", upper)
    entry = next(e for e in consumed if e["format"] == "csv")
    forged = [{**e, "path": str(upper), "realpath": os.path.realpath(upper)}
              if e is entry else e for e in consumed]
    with pytest.raises(receipt.ReceiptError) as error:
        receipt._consumed("mimic", str(root), forged, snapshot)
    assert error.value.code in {"storage_consumed_csv_unlisted",
                                "storage_consumed_outside_snapshot"}


def test_the_listed_csv_named_in_other_case_on_a_case_insensitive_fs_is_accepted(
        tmp_path, monkeypatch, real_admission):
    from easyicu import storage_layout

    root, *_rest = _services_run(tmp_path, monkeypatch, listed=True)
    upper = _case_variant(root)
    if not (root / "SERVICES.CSV.GZ").exists() or not os.path.samefile(
            root / "SERVICES.CSV.GZ", root / "services.csv.gz"):
        # A case-sensitive test file system: stand in for the probe.
        monkeypatch.setattr(storage_layout, "_case_insensitive_at", lambda path: True)

    assert storage_layout.listed_csv_input(root, "services", upper)


# delta17w follow-up: a formal run never reads a declared table as empty for
# want of a physical source unless the root's REQUIRED_LAYOUT.json
# ``declared_absent`` allowlist excuses it (fail-closed; the allowlist is empty
# until the user approves its entries).

_PHARMA = {"event": "table_unavailable", "root": "/x", "table": "pharma",
           "availability": "source_unavailable", "reason_code": "declared_table_not_installed",
           "manifest": "unlisted", "missing_tables": ["pharma"], "module": "medications"}
_EXCUSE = {"db": "sic", "table": "pharma", "reason": "fixture reason",
           "evidence": "fixture evidence", "readers": 0}


def _approve(monkeypatch, *entries):
    """Stand in for the sealed approved allowlist (tests of the real file below)."""
    monkeypatch.setattr(receipt, "approved_declared_absent", lambda: (list(entries), "a" * 64))


def test_a_declared_table_read_as_empty_stops_a_formal_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, unavailable=[_PHARMA])
    assert error.value.code == "storage_unavailable_in_formal_run"
    assert error.value.details["database"] == "sic"
    assert error.value.details["tables"] == ["pharma"]
    (cause,) = error.value.details["causes"]
    assert cause["reason_code"] == "declared_table_not_installed"
    assert cause["module"] == "medications"


@pytest.mark.parametrize("excuse", [
    {**_EXCUSE, "db": "eicu"},          # another database's entry
    {**_EXCUSE, "table": "pharmaref"},  # another table
])
def test_an_allowlist_entry_for_something_else_does_not_excuse_it(tmp_path, monkeypatch, excuse):
    _approve(monkeypatch, excuse)
    prepared = _prepared(tmp_path / "prepared")
    required = {**_required([_flat(prepared, "cases", prepared)]), "declared_absent": [excuse]}

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, unavailable=[_PHARMA],
                     required=required)
    assert error.value.code == "storage_unavailable_in_formal_run"


def test_a_declared_absent_table_is_recorded_with_its_excuse_and_passes(tmp_path, monkeypatch):
    _approve(monkeypatch, _EXCUSE)
    prepared = _prepared(tmp_path / "prepared")
    required = {**_required([_flat(prepared, "cases", prepared)]), "declared_absent": [_EXCUSE]}

    built = _sic_receipt(tmp_path, prepared, None, monkeypatch, unavailable=[_PHARMA],
                         required=required)

    (record,) = built["databases"]["sic"]["storage_unavailable"]
    assert record["table"] == "pharma" and record["excused_by"] == _EXCUSE
    assert built["required_layout"]["declared_absent"] == [_EXCUSE]
    assert built["required_layout"]["approved_declared_absent_sha256"] == "a" * 64


def test_an_empty_unavailable_list_passes_and_is_recorded(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    built = _sic_receipt(tmp_path, prepared, None, monkeypatch, unavailable=[])

    assert built["databases"]["sic"]["storage_unavailable"] == []
    assert built["required_layout"]["declared_absent"] == []


def test_a_manifest_that_does_not_record_unavailability_stops_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, unavailable=None)
    assert error.value.code == "storage_unavailable_missing"


@pytest.mark.parametrize("absent", [
    {"db": "sic", "table": "pharma"},
    {**_EXCUSE, "evidence": ""},
    {**_EXCUSE, "extra": "x"},
    {**_EXCUSE, "readers": 1},
    {**_EXCUSE, "readers": False},
    {k: v for k, v in _EXCUSE.items() if k != "readers"},
    "pharma",
])
def test_a_malformed_declared_absent_entry_stops_the_receipt(tmp_path, monkeypatch, absent):
    prepared = _prepared(tmp_path / "prepared")
    required = {**_required([_flat(prepared, "cases", prepared)]), "declared_absent": [absent]}

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, required=required)
    assert error.value.code == "receipt_input_malformed"


def test_a_duplicate_declared_absent_entry_stops_the_receipt(tmp_path, monkeypatch):
    _approve(monkeypatch, _EXCUSE, {**_EXCUSE, "table": "PHARMA"})
    prepared = _prepared(tmp_path / "prepared")
    required = {**_required([_flat(prepared, "cases", prepared)]),
                "declared_absent": [_EXCUSE, {**_EXCUSE, "table": "PHARMA"}]}

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, required=required)
    assert error.value.code == "receipt_input_malformed"


def test_a_non_formal_run_records_an_uninstalled_table_and_a_formal_receipt_refuses_it(tmp_path):
    # A real extraction (no receipt) reads the missing table as empty and
    # records it; it does not refuse. The same native manifest in a formal
    # receipt is a stop unless the root declares the table absent.
    import shutil

    from easyicu.api.concepts import clear_global_loader
    from easyicu.api.extraction import extract_database
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    shutil.rmtree(root / "pharma")
    output = tmp_path / "export"
    try:
        result = extract_database(
            "hirid", data_path=str(root), output_dir=str(output),
            modules=["outcome", "medications"], native_export_v2=True, verbose=False,
        )
    finally:
        clear_global_loader()

    published = json.loads((output / "_manifest.json").read_text())
    assert "pharma" in {r["table"] for r in published["storage_unavailable"]}
    assert result["storage_unavailable"]
    with pytest.raises(receipt.ReceiptError) as error:
        receipt._storage_unavailable("hirid", published)
    assert error.value.code == "storage_unavailable_in_formal_run"
    assert "pharma" in error.value.details["tables"]
    excuse = [{"db": "hirid", "table": t, "reason": "fixture", "evidence": "fixture", "readers": 0}
              for t in sorted({r["table"] for r in published["storage_unavailable"]})]
    excused = receipt._storage_unavailable("hirid", published, excuse)
    assert {r["excused_by"]["table"] for r in excused} == {r["table"] for r in excused}


# The approved allowlist (user-approved with the reseal; supervisor texts).

_APPROVED = {
    ("miiv", "demo"): ("ricu module-output name declared as a table; not a source table; "
                       "0 readers"),
    ("aumc", "vital"): ("ricu module-output name declared as a table; not a source table; "
                        "0 readers"),
    ("mimic", "datetimeevents"): "view omission, unread (0 readers in dictionaries/src)",
}


def test_the_approved_allowlist_is_exactly_the_three_reviewed_entries():
    entries, digest = receipt.approved_declared_absent()

    assert {(e["db"], e["table"]): e["reason"] for e in entries} == _APPROVED
    assert all(e["readers"] == 0 for e in entries) and len(digest) == 64
    by_key = {(e["db"], e["table"]): e for e in entries}
    assert by_key[("aumc", "vital")]["evidence"] == (
        "installed 7-table AUMCdb set; no release checksum list on the server")
    carevue = by_key[("mimic", "datetimeevents")]
    assert carevue["evidence"] == ("MIMIC-III 1.4 ships DATETIMEEVENTS.csv.gz; the CareVue view "
                                   "(manifest eb698096\u2026) omits it")
    assert "omission" in carevue["reason"] and "not shipped" not in json.dumps(carevue)
    # HiRID ordinal/variables are present (configured-name lookup): never allowlisted.
    assert not {("hirid", "ordinal"), ("hirid", "variables")} & set(by_key)


def test_the_three_approved_entries_excuse_exactly_their_tables(tmp_path):
    entries, _digest = receipt.approved_declared_absent()
    run = tmp_path / "run"
    run.mkdir()
    (run / "REQUIRED_LAYOUT.json").write_text(json.dumps({
        "schema": "easyicu.required_layout/2", "complete": True, "declared_absent": entries,
        "databases": {db: {"t": {"layout": "flat"}} for db in ("miiv", "aumc", "mimic")}}))
    payload, _sha = receipt.load_required_layout(run, ["miiv", "aumc", "mimic"])

    for (db, table), _reason in _APPROVED.items():
        record = {**_PHARMA, "table": table}
        (excused,) = receipt._storage_unavailable(
            db, {"storage_unavailable": [record]}, payload["declared_absent"])
        assert excused["excused_by"]["table"] == table and excused["excused_by"]["db"] == db
        with pytest.raises(receipt.ReceiptError) as error:  # a fourth, unlisted table
            receipt._storage_unavailable(
                db, {"storage_unavailable": [record, {**_PHARMA, "table": "unlisted"}]},
                payload["declared_absent"])
        assert error.value.code == "storage_unavailable_in_formal_run"
        assert error.value.details["tables"] == ["unlisted"]


@pytest.mark.parametrize("entry", [
    {"db": "hirid", "table": "ordinal", "reason": "x", "evidence": "y", "readers": 0},
    {"db": "mimic", "table": "datetimeevents", "reason": "not shipped", "evidence": "y",
     "readers": 0},
])
def test_a_declared_absent_entry_not_in_the_approved_allowlist_is_refused(tmp_path, entry):
    run = tmp_path / "run"
    run.mkdir()
    (run / "REQUIRED_LAYOUT.json").write_text(json.dumps({
        "schema": "easyicu.required_layout/2", "complete": True, "declared_absent": [entry],
        "databases": {"hirid": {"t": {"layout": "flat"}}}}))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.load_required_layout(run, ["hirid"])
    assert error.value.code == "declared_absent_not_approved"


# r13prov additions (after delta28p). (1), the formal tool identity, is
# tests/core/test_r13_formal_tool_identity.py (delta32p copy-then-exec).

# (2) Contract inputs reject a duplicate key at any level; names dedup by casefold.

_LAYOUT_HEAD = '"schema": "easyicu.required_layout/2", "complete": true'


@pytest.mark.parametrize("text", [
    '{' + _LAYOUT_HEAD + ', "complete": true, "databases": {"sic": {"t": {"layout": "flat"}}}}',
    '{' + _LAYOUT_HEAD + ', "databases": {"sic": {"t": {"layout": "flat"}}, '
    '"sic": {"t": {"layout": "flat"}}}}',
    '{' + _LAYOUT_HEAD + ', "databases": {"sic": {"t": {"layout": "flat"}, '
    '"t": {"layout": "flat"}}}}',
    '{' + _LAYOUT_HEAD + ', "databases": {"sic": {"t": {"layout": "flat", "layout": "bucket"}}}}',
    '{' + _LAYOUT_HEAD + ', "databases": {"sic": {"t": {"layout": "flat"}}}, "declared_absent": '
    '[{"db": "sic", "table": "x", "reason": "r", "evidence": "e", "readers": 0, "readers": 0}]}',
])
def test_a_duplicate_key_in_required_layout_at_any_level_is_refused(tmp_path, text):
    run = tmp_path / "run"
    run.mkdir()
    (run / "REQUIRED_LAYOUT.json").write_text(text)

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.load_required_layout(run, ["sic"])
    assert error.value.code == "receipt_input_duplicate_key"
    assert error.value.details["input"] == "REQUIRED_LAYOUT.json"


@pytest.mark.parametrize("text", [
    '{"schema": "easyicu.declared_absent/1", "schema": "easyicu.declared_absent/1", '
    '"entries": []}',
    '{"schema": "easyicu.declared_absent/1", "entries": [{"db": "mimic", "db": "mimic", '
    '"table": "t", "reason": "r", "evidence": "e", "readers": 0}]}',
])
def test_a_duplicate_key_in_the_approved_allowlist_is_refused(tmp_path, monkeypatch, text):
    approved = tmp_path / "formal_declared_absent.json"
    approved.write_text(text)
    monkeypatch.setattr(receipt, "APPROVED_DECLARED_ABSENT_FILE", approved)

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.approved_declared_absent()
    assert error.value.code == "receipt_input_duplicate_key"


def test_the_layout_digest_is_of_the_bytes_parsed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    text = '{' + _LAYOUT_HEAD + ', "databases": {"sic": {"t": {"layout": "flat"}}}}'
    (run / "REQUIRED_LAYOUT.json").write_text(text)

    _payload, digest = receipt.load_required_layout(run, ["sic"])
    assert digest == receipt.sha256_file(run / "REQUIRED_LAYOUT.json")


@pytest.mark.parametrize("pair", [
    ({"db": "sic", "table": "STRASSE"}, {"db": "sic", "table": "straße"}),  # lower() differs
    ({"db": "sic", "table": "pharma"}, {"db": "SIC", "table": "pharma"}),
])
def test_declared_absent_entries_equal_under_casefold_are_duplicates(tmp_path, monkeypatch, pair):
    entries = [{**_EXCUSE, **names} for names in pair]
    _approve(monkeypatch, *entries)
    run = tmp_path / "run"
    run.mkdir()
    (run / "REQUIRED_LAYOUT.json").write_text(json.dumps({
        "schema": "easyicu.required_layout/2", "complete": True, "declared_absent": entries,
        "databases": {"sic": {"t": {"layout": "flat"}}}}))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.load_required_layout(run, ["sic"])
    assert error.value.code == "receipt_input_malformed"


@pytest.mark.parametrize("databases", [
    {"sic": {"STRASSE": {"layout": "flat"}, "straße": {"layout": "flat"}}},
    {"sic": {"t": {"layout": "flat"}}, "SIC": {"t": {"layout": "flat"}}},
])
def test_required_names_equal_under_casefold_are_duplicates(tmp_path, databases):
    run = tmp_path / "run"
    run.mkdir()
    (run / "REQUIRED_LAYOUT.json").write_text(json.dumps({
        "schema": "easyicu.required_layout/2", "complete": True, "databases": databases}))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.load_required_layout(run, ["sic"])
    assert error.value.code == "receipt_input_malformed"


def test_an_unavailable_record_matches_its_entry_under_casefold():
    entry = {**_EXCUSE, "table": "straße"}

    (excused,) = receipt._storage_unavailable(
        "sic", {"storage_unavailable": [{**_PHARMA, "table": "STRASSE"}]}, [entry])
    assert excused["excused_by"] == entry


# (3) The label a CareVue root's reads carry is the configured database name
# "mimic" (the bench spec's db; RUN_RECEIPT, native manifest and snapshot of
# the interim CareVue run all say "mimic"), so the approved datetimeevents entry
# excuses the record the storage owner really writes.

def test_the_approved_carevue_entry_excuses_the_record_the_storage_owner_writes(
        tmp_path, monkeypatch):
    from easyicu.datasource import ICUDataSource
    from easyicu.resources import load_data_sources
    from easyicu.storage_layout import SCAN_FALLBACK_LOG_ENV, read_unavailable_log

    root = tmp_path / "carevue"
    root.mkdir()
    pd.DataFrame({"subject_id": [10], "hadm_id": [100], "icustay_id": [1000],
                  "intime": pd.to_datetime(["2150-01-01"]),
                  "outtime": pd.to_datetime(["2150-01-03"])}).to_parquet(
        root / "icustays.parquet", index=False)
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": [
        {"table": "icustays", "status": "completed",
         "outputs": [{"relative_path": "icustays.parquet"}]}]}))
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv(SCAN_FALLBACK_LOG_ENV, str(log))
    config = load_data_sources().get("mimic")
    assert config.name == "mimic" and "datetimeevents" in config.tables
    with pytest.raises(Exception):  # the reader gets no frame; the absence is logged
        ICUDataSource(config=config, base_path=root).load_table("datetimeevents")
    (record,) = read_unavailable_log(log)
    assert record["table"] == "datetimeevents"
    assert record["reason_code"] == "declared_table_not_installed"

    entries, _digest = receipt.approved_declared_absent()  # the real sealed allowlist
    run = tmp_path / "run"
    run.mkdir()
    (run / "REQUIRED_LAYOUT.json").write_text(json.dumps({
        "schema": "easyicu.required_layout/2", "complete": True,
        "declared_absent": [e for e in entries if e["db"] == "mimic"],
        "databases": {"mimic": {"icustays": {"layout": "flat"}}}}))
    payload, _sha = receipt.load_required_layout(run, ["mimic"])
    manifest = {"storage_unavailable": [record]}

    (excused,) = receipt._storage_unavailable("mimic", manifest, payload["declared_absent"])
    assert excused["excused_by"]["db"] == "mimic"
    assert excused["excused_by"]["table"] == "datetimeevents"
    with pytest.raises(receipt.ReceiptError) as error:  # any other label is not excused
        receipt._storage_unavailable("mimic_cv", manifest, payload["declared_absent"])
    assert error.value.code == "storage_unavailable_in_formal_run"


# Item 4 (formal half of r13slpub): every bucket read comes from an installation
# the seal tool published and accepted; the read set must be known first.

def _seal_spy(monkeypatch, fail=None):
    from easyicu import storage_seal_publication as slpub

    calls = []

    def spy(root, *, formal):
        calls.append((str(root), formal))
        if fail:
            raise slpub.SealPublicationError(fail, Path(root).parent)

    monkeypatch.setattr(slpub, "assert_seal_published", spy)
    monkeypatch.setattr(receipt, "_seal_published", _REAL_SEAL_PUBLISHED)
    return calls


@pytest.mark.parametrize("layout", ["flat", "csv"])
def test_a_flat_only_read_set_passes_with_no_seal_check(tmp_path, monkeypatch, layout):
    # HiRID's formal layout is all flat: a known flat-only read set is a PASS
    # with zero seal checks, never a refusal.
    calls = _seal_spy(monkeypatch)
    root = tmp_path / "hirid"
    root.mkdir()
    frame = pd.DataFrame({"value": [1.0]})
    path = root / f"observations.{ 'parquet' if layout == 'flat' else 'csv' }"
    if layout == "flat":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": [
        {"table": "observations", "status": "completed",
         "input": {"relative_path": path.name},
         "outputs": [{"relative_path": path.name}] if layout == "flat" else []}]}))
    read = {**_flat(root, "observations", root), "layout": layout}
    tables = {"observations": {"layout": layout, "root": str(root),
                               "reads": [{"layout": layout, "root": str(root)}]}}
    run = _run_root(tmp_path, root, commit=CODE["commit"], reads=[read], tables=tables,
                    db="hirid", consumed=[_consumed(path, "observations",
                                                  "parquet" if layout == "flat" else "csv")],
                    required={"schema": "easyicu.required_layout/2", "complete": True,
                              "databases": {"hirid": {"observations": {"layout": layout}}}})
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    built = receipt.build_run_receipt(run, CODE["commit"], [f"hirid:{root}:144-145"],
                                      scratch=tmp_path / "scratch")
    assert calls == [] and built["databases"]["hirid"]["seal_checks"] == []
    assert receipt.write_run_receipt(run, built).is_file()


def test_every_bucket_read_is_checked_once_as_formal(tmp_path, monkeypatch):
    calls = _seal_spy(monkeypatch)
    reads = [_flat(tmp_path, "cases", tmp_path), _bucket_read(tmp_path, "laboratory"),
             _bucket_read(tmp_path, "laboratory"), _bucket_read(tmp_path, "medication")]

    checked = receipt._seal_published("sic", reads)

    assert calls == [(str(tmp_path / "laboratory_bucket"), True),
                     (str(tmp_path / "medication_bucket"), True)]
    assert [c["table"] for c in checked] == ["laboratory", "medication"]


@pytest.mark.parametrize("reason", ["bucket_install_unpublished", "bucket_install_refused",
                                    "bucket_install_unaccepted"])
def test_a_seal_refusal_refuses_the_receipt_with_its_reason(tmp_path, monkeypatch, reason):
    _seal_spy(monkeypatch, fail=reason)

    with pytest.raises(receipt.ReceiptError) as error:
        receipt._seal_published("sic", [_bucket_read(tmp_path, "laboratory")])
    assert error.value.code == reason and error.value.details["seal_reason"] == reason
    assert error.value.details["table"] == "laboratory"


@pytest.mark.parametrize("entry", [
    {"layout": "parquet"}, {"layout": "bucket"}, {"layout": "bucket_"}, {"layout": None},
    {"layout": "bucket_Hash"}, {"layout": "bucket_hash\n"},
    {"root": "None"}, {"root": ""}, {"root": None}, {"layout_path": None},
    {"layout_path": ""}, {"layout_path": "None"}, {"table": ""}, {"table": "None"},
    {"table": None}, "not-a-dict",
])
def test_an_unknown_read_set_is_refused_before_any_seal_check(tmp_path, monkeypatch, entry):
    calls = _seal_spy(monkeypatch)
    prepared = _prepared(tmp_path / "prepared")
    def corrupt(run):
        path = run / "sic" / "export" / "_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["storage_reads"] = [_bucket_read(tmp_path, "laboratory"),
            entry if isinstance(entry, str) else {**_flat(prepared, "cases", prepared), **entry}]
        path.write_text(json.dumps(manifest))

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, unavailable=[], after_run=corrupt)
    assert error.value.code == "formal_read_set_unknown" and calls == []


@pytest.mark.parametrize("reads", [None, {}, "unknown", 0])
def test_a_read_set_that_is_not_a_list_is_unknown(tmp_path, monkeypatch, reads):
    calls = _seal_spy(monkeypatch)
    prepared = _prepared(tmp_path / "prepared")

    def corrupt(run):
        path = run / "sic" / "export" / "_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["storage_reads"] = reads
        path.write_text(json.dumps(manifest))

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, after_run=corrupt)
    assert error.value.code == "formal_read_set_unknown" and calls == []


def test_a_missing_read_set_is_still_storage_reads_missing(tmp_path, monkeypatch):
    calls = _seal_spy(monkeypatch)
    prepared = _prepared(tmp_path / "prepared")

    def drop(run):
        path = run / "sic" / "export" / "_manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["storage_reads"]
        path.write_text(json.dumps(manifest))

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, unavailable=[], after_run=drop)
    assert error.value.code == "storage_reads_missing" and calls == []
    # Also demonstrate the positive flat-only path now records an empty seal set.
    assert _sic_receipt(tmp_path / "complete", prepared, None, monkeypatch)[
        "databases"]["sic"]["seal_checks"] == []


def test_a_bucket_read_where_the_required_layout_says_flat_is_a_deviation(tmp_path, monkeypatch):
    # HiRID: REQUIRED_LAYOUT says observations is flat; a read served from
    # b6_root/hirid/observations_bucket is layout_deviation.
    from scripts.build_itemid_bucket_cache import build_bucket_cache

    data = tmp_path / "b6_root" / "hirid"
    (data / "observations").mkdir(parents=True)
    source = data / "observations" / "part.parquet"
    pd.DataFrame({"variableid": [1], "value": [1.0]}).to_parquet(source, index=False)
    (data / "conversion_manifest.json").write_text(json.dumps({"tables": [
        {"table": "observations", "status": "completed", "outputs": [
            {"relative_path": "observations/part.parquet"}]}]}))
    build_bucket_cache(data, table="observations", key="variableid", bucket_count=2,
                       memory_limit="256MB", threads=1)
    bucket = data / "observations_bucket"
    (bucket / "_BUCKET_BUILD_RECEIPT.json").write_text("{}")
    calls = _seal_spy(monkeypatch)
    run = _run_root(tmp_path, data, commit=CODE["commit"], db="hirid",
                    reads=[_bucket_entry(bucket)],
                    consumed=[_consumed(p, "observations") for p in bucket.rglob("*.parquet")],
                    required={"schema": "easyicu.required_layout/2", "complete": True,
                              "databases": {"hirid": {"observations": {"layout": "flat"}}}})
    # Ignore the fixture's read-time checks; this call is the formal receipt's.
    calls.clear()
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: dict(CODE))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.build_run_receipt(run, CODE["commit"], [f"hirid:{data}:144-145"],
                                  scratch=tmp_path / "scratch")
    assert error.value.code == "layout_deviation"
    assert calls == [(str(bucket), True)]
    (deviation,) = error.value.details["deviations"]
    assert deviation["required"] == {"layout": "flat", "root_realpath": os.path.realpath(data)}


# The real seal module on synthetic installations.

def _install(root: Path, table: str = "chartevents", *, sealed_sha: str | None = None,
             verdict: str = "PASS") -> Path:
    bucket = root / f"{table}_bucket"
    bucket.mkdir(parents=True)
    (bucket / "_BUCKET_MANIFEST.json").write_text(json.dumps({"table": table}))
    digest = sealed_sha or receipt.sha256_file(bucket / "_BUCKET_MANIFEST.json")
    (root / "_SEAL_BUILD_RECEIPT.json").write_text(json.dumps(
        {"verdict": verdict, "tables": {table: {"manifest_sha256": digest}}}))
    return bucket


def _bucket_entry(bucket: Path) -> dict:
    return {"root": str(bucket.parent), "table": bucket.name[:-len("_bucket")],
            "layout": "bucket_hash", "layout_path": str(bucket),
            "layout_realpath": os.path.realpath(bucket)}


def test_an_accepted_install_passes_the_real_seal_check(tmp_path, monkeypatch):
    monkeypatch.setattr(receipt, "_seal_published", _REAL_SEAL_PUBLISHED)
    bucket = _install(tmp_path / "inst")

    assert receipt._seal_published("miiv", [_bucket_entry(bucket)])[0]["table"] == "chartevents"


@pytest.mark.parametrize("change", ["accepted", "no_receipt", "verdict", "sha", "pending",
                                     "refused"])
def test_an_unaccepted_install_gets_no_pass_receipt(tmp_path, monkeypatch, change):
    monkeypatch.setattr(receipt, "_seal_published", _REAL_SEAL_PUBLISHED)
    prepared = _prepared(tmp_path / "prepared")
    root = _sic_layout_root(tmp_path, prepared)
    bucket = root / "laboratory_bucket"
    digest = "0" * 64 if change == "sha" else receipt.sha256_file(
        bucket / "_BUCKET_MANIFEST.json")
    if change != "no_receipt":
        (root / "_SEAL_BUILD_RECEIPT.json").write_text(json.dumps({
            "verdict": "FAIL" if change == "verdict" else "PASS",
            "tables": {"laboratory": {"manifest_sha256": digest}}}))
    # Mark it only after fixture reads and snapshots, as a publication changing
    # between extraction and the formal receipt would do.
    def mark(run):
        if change in {"pending", "refused"}:
            (root.parent / f".{root.name}.seal_{change.upper()}").write_text("")

    reads = [_flat(root, "cases", root), _bucket_read(root, "laboratory")]
    reads[0]["layout_realpath"] = str(prepared)
    if change == "accepted":
        built = _sic_receipt(tmp_path, root, reads, monkeypatch)
        assert built["databases"]["sic"]["seal_checks"] == [
            {"table": "laboratory", "layout_path": str(bucket)}]
        assert receipt.write_run_receipt(tmp_path / "run", built).is_file()
        return
    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, root, reads, monkeypatch, after_run=mark)
    expected = {"pending": "bucket_install_unpublished", "refused": "bucket_install_refused"}.get(
        change, "bucket_install_unaccepted")
    assert error.value.code == expected
    assert not (tmp_path / "run" / "RUN_RECEIPT.json").exists()


@pytest.mark.parametrize("matching", [True, False])
def test_a_carevue_farm_link_binds_through_the_realpath_install(tmp_path, monkeypatch, matching):
    # The farm's own receipt (bucket_links, no tables) is never enough: the
    # link is followed to the realpath install's receipt, whose manifest sha
    # must match.
    monkeypatch.setattr(receipt, "_seal_published", _REAL_SEAL_PUBLISHED)
    real = _install(tmp_path / "mimic_cv_s1234", sealed_sha=None if matching else "0" * 64)
    farm = tmp_path / "mimic_cv_dir_s1234"
    farm.mkdir()
    os.symlink(real, farm / "chartevents_bucket")
    (farm / "_SEAL_BUILD_RECEIPT.json").write_text(json.dumps(
        {"verdict": "PASS", "bucket_links": {"chartevents_bucket": str(real)}}))
    entry = _bucket_entry(farm / "chartevents_bucket")

    if matching:
        assert receipt._seal_published("mimic", [entry])
    else:
        with pytest.raises(receipt.ReceiptError) as error:
            receipt._seal_published("mimic", [entry])
        assert error.value.code == "bucket_install_unaccepted"


def test_a_farm_receipt_alone_is_never_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(receipt, "_seal_published", _REAL_SEAL_PUBLISHED)
    real = tmp_path / "elsewhere" / "chartevents_bucket"
    real.mkdir(parents=True)
    (real / "_BUCKET_MANIFEST.json").write_text("{}")
    farm = tmp_path / "farm"
    farm.mkdir()
    os.symlink(real, farm / "chartevents_bucket")
    (farm / "_SEAL_BUILD_RECEIPT.json").write_text(json.dumps(
        {"verdict": "PASS", "bucket_links": {"chartevents_bucket": str(real)}}))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt._seal_published("mimic", [_bucket_entry(farm / "chartevents_bucket")])
    assert error.value.code == "bucket_install_unaccepted"


# Root cause of a failed step (supervisor addition).

def test_the_receipt_shows_the_root_cause_of_a_retyped_failure(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")
    root = {"type": "easyicu.storage_seal_publication.SealPublicationError",
            "reason": "bucket_install_unaccepted",
            "chain": ["ConceptError", "SealPublicationError"]}

    def failed(run):
        path = run / "sic" / "summary.json"
        summary = json.loads(path.read_text())
        path.write_text(json.dumps({**summary, "exception": "ConceptError: ...",
                                    "exception_root": root}))

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, unavailable=[], after_run=failed)
    assert error.value.code == "database_run_reported_errors"
    assert error.value.details["root_type"].endswith("SealPublicationError")
    assert error.value.details["root_reason"] == "bucket_install_unaccepted"
    assert error.value.details["chain"] == root["chain"]
