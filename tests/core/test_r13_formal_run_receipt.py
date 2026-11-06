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


def _consumed(path: Path, table: str = "cases", fmt: str = "parquet") -> dict:
    return {"table": table, "path": str(path), "realpath": os.path.realpath(path),
            "format": fmt}


def _snapshots(run: Path, input_root: Path) -> None:
    """What bench7 writes: the input snapshot before the jobs and its verification after."""
    from easyicu import formal_inputs

    pre = formal_inputs.take_snapshot([f"sic:{input_root}:144-145"])
    formal_inputs.write_snapshot(run / "INPUT_SNAPSHOT.json", pre)
    formal_inputs.write_snapshot(run / "INPUT_SNAPSHOT_POST.json",
                                 formal_inputs.verify_snapshot(pre))


def _run_root(tmp_path: Path, prepared: Path, *, commit: str, reads=None,
              version: str = CODE["version"], consumed=None,
              installed: str | None = "9.9.9") -> Path:
    run = tmp_path / "run"
    export = run / "sic" / "export"
    export.mkdir(parents=True)
    _snapshots(run, prepared)
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
            "easyicu_version": version, "easyicu_installed_dist_version": installed,
        },
        "storage_reads": reads if reads is not None else [_flat(prepared, "cases", prepared)],
        "storage_binding": {"digest": "b" * 64, "files": 3, "scope": "parent_run_binding"},
        "storage_consumed": (consumed if consumed is not None
                             else [_consumed(prepared / "cases.parquet")]),
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


def _sic_receipt(tmp_path, root, reads, monkeypatch, *, consumed=None, after_run=None):
    run = _run_root(tmp_path, root, commit=CODE["commit"], reads=reads, consumed=consumed)
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
    assert error.value.code == "storage_read_layout_missing"


# delta9b ruling: the run's inputs are hashed before and after; every consumed
# file is a snapshotted file.

def test_inputs_that_differ_before_and_after_the_run_stop_the_receipt(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path / "prepared")

    def tamper(run):
        post = json.loads((run / "INPUT_SNAPSHOT_POST.json").read_text())
        post["databases"]["sic"]["digest"] = "0" * 64
        (run / "INPUT_SNAPSHOT_POST.json").write_text(json.dumps(post))

    with pytest.raises(receipt.ReceiptError) as error:
        _sic_receipt(tmp_path, prepared, None, monkeypatch, after_run=tamper)
    assert error.value.code == "input_changed_during_run"


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
    consumed = [{k: e[k] for k in ("table", "path", "realpath", "format")}
                for e in read_consumed_log(log) if e["table"] == "services"]

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
    assert published["storage_binding"]["digest"] == result["storage_binding"]["digest"]
