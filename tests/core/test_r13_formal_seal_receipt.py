"""Formal receipt acceptance of synthetic storage seal publications."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from tests.support.formal_run_receipts import (
    CODE,
    _admission as _admission,
    _bucket_read,
    _consumed,
    _flat,
    _prepared,
    _run_root,
    _sic_layout_root,
    _sic_receipt,
    _tool_scripts as _tool_scripts,
    receipt,
    _REAL_SEAL_PUBLISHED,
)


# Item 4 (formal half of r13slpub): every bucket read comes from an installation
# the seal tool published and accepted; the read set must be known first.

def _seal_spy(monkeypatch, fail=None):
    from easyicu import storage_seal_publication as slpub

    calls = []
    read_time_check = slpub.assert_seal_published

    def spy(root, *, formal):
        if not formal:
            # This spy counts formal acceptance checks. Keep the merged
            # read-time publication defense real during fixture construction.
            return read_time_check(root, formal=False)
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
    from easyicu.storage_seal_publication import SealPublicationError

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
    error_type = SealPublicationError if change in {"pending", "refused"} else receipt.ReceiptError
    with pytest.raises(error_type) as error:
        _sic_receipt(tmp_path, root, reads, monkeypatch, after_run=mark)
    expected = {"pending": "bucket_install_unpublished", "refused": "bucket_install_refused"}.get(
        change, "bucket_install_unaccepted")
    if change in {"pending", "refused"}:
        assert error.value.reason == expected  # entry gate preserves the typed refusal
    else:
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
