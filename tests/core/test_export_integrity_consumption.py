"""delta42e: the files consumed, verified and receipted must be the same set."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu import export_publication as publication
from easyicu.webserver import agent_outputs, dataio, export_download


def _save(root: Path, manifest: dict) -> None:
    (root / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _package(root: Path, *, legacy: bool = False) -> dict:
    root.mkdir()
    (root / "data.csv").write_bytes(b"value\n1\n")
    manifest = {"files": [{"file": "data.csv"}]}
    if not legacy:
        manifest.update(
            schema_version="easyicu_native_export_v2",
            member_integrity=dict(publication.MEMBER_INTEGRITY_DECLARATION),
        )
        manifest["files"][0].update(publication.member_digest_record(root / "data.csv"))
    _save(root, manifest)
    return manifest


def _bundle(root: Path):
    return export_download.prepare_registered_export_bundle(
        "src_0123456789ab",
        registry={"sources": [{"id": "src_0123456789ab", "ok": True, "path": str(root)}]},
        temp_dir=root.parent / "downloads",
    )


@pytest.mark.parametrize("pre_kind", [False, True])
def test_legacy_unlisted_member_accepted_is_refused(tmp_path, pre_kind):
    root = tmp_path / "export"
    manifest = _package(root, legacy=True)
    if pre_kind:
        manifest.update(schema_version="easyicu_native_export_v2", data_path="synthetic",
                        cohort_contract={}, column_metadata={})
        _save(root, manifest)
    assert publication.verify_member_integrity(root, "data.csv") == "unrecorded"
    (root / "stray.csv").write_bytes(b"value\n2\n")
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        publication.verify_member_integrity(root, "stray.csv")
    assert caught.value.reason == publication.REASON_MEMBER_UNRECORDED


def test_inventory_never_opens_an_unlisted_legacy_member(tmp_path, monkeypatch):
    root = tmp_path / "export"
    manifest = _package(root, legacy=True)
    (root / "stray.csv").write_bytes(b"value\n2\n")
    opened = []
    monkeypatch.setattr(dataio, "_read_columns", lambda path: opened.append(path.name) or ["value"])
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        dataio._export_file_inventory(root, manifest, count_rows=False)
    assert caught.value.reason == publication.REASON_MEMBER_UNRECORDED
    assert "stray.csv" not in opened


def test_schema_version_null_is_not_legacy(tmp_path):
    root = tmp_path / "export"
    manifest = _package(root, legacy=True)
    manifest["schema_version"] = None
    _save(root, manifest)
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        publication.verify_member_integrity(root, "data.csv")
    assert caught.value.reason == publication.REASON_DIGEST_UNRECORDED
    assert publication.legacy_manifest_format(manifest) is None


@pytest.mark.parametrize("marker", ["export_kind", "contract_revision", "column_metadata"])
def test_modern_marker_without_schema_is_not_legacy(tmp_path, marker):
    root = tmp_path / "export"
    manifest = _package(root, legacy=True)
    manifest[marker] = None  # Presence matters even when a marker has no value.
    _save(root, manifest)
    assert publication.legacy_manifest_format(manifest) is None
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        publication.verify_member_integrity(root, "data.csv")
    assert caught.value.reason == publication.REASON_DIGEST_UNRECORDED


@pytest.mark.parametrize("marker", ["time_window_authority", "data_provenance"])
def test_post_kind_fields_cannot_forge_the_pre_kind_signature(tmp_path, marker):
    root = tmp_path / "export"
    manifest = _package(root, legacy=True)
    manifest.update(schema_version="easyicu_native_export_v2", data_path="synthetic",
                    cohort_contract={}, column_metadata={})
    manifest[marker] = {}
    _save(root, manifest)
    assert publication.legacy_manifest_format(manifest) is None
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        publication.verify_member_integrity(root, "data.csv")
    assert caught.value.reason == publication.REASON_DIGEST_UNRECORDED


def test_agent_outputs_checks_members_omitted_by_a_stale_source(tmp_path):
    root = tmp_path / "export"
    manifest = _package(root)
    manifest["files"].append({"file": "outcome.parquet", "module": "outcome"})
    _save(root, manifest)
    # The cached source omits the now-missing outcome member entirely.
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        agent_outputs._load_context(str(root), {"files": []})
    assert caught.value.reason == publication.REASON_MEMBER_MISSING


@pytest.mark.parametrize("name,payload", [
    ("empty.csv", b""),
    ("blank.csv", b" \t\n"),
    ("broken.csv", b'"unterminated\n'),
    ("truncated.xlsx", b"PK\x03\x04truncated"),
])
def test_legacy_download_refuses_unreadable_csv_and_xlsx(tmp_path, name, payload):
    root = tmp_path / "export"
    manifest = _package(root, legacy=True)
    (root / name).write_bytes(payload)
    manifest["files"] = [{"file": name}]
    _save(root, manifest)
    with pytest.raises(export_download.ExportDownloadError) as caught:
        _bundle(root)
    assert caught.value.code == publication.UNREADABLE_CODE
    assert isinstance(caught.value.__cause__, publication.ExportMemberUnreadableError)
    assert caught.value.__cause__.__cause__ is not None
    assert not list((tmp_path / "downloads").glob("*.zip"))


def test_coverage_audit_accepts_unlisted_members_is_refused(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[2] / "tools/audit_extraction_coverage.py"
    # As in the reviewer probe, isolate the actual reader functions. The
    # standalone tool's historical loader constants no longer import; the
    # synthetic policy below tests integrity without changing clinical policy.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"module_cols", "audit"}]
    coverage = {"os": os, "pq": pq, "ALL_DBS": {"hirid"}, "SPECIAL_GROUPS": {
        "synthetic": {"module": "renal", "concepts": ["value"], "excluded_dbs": set()}
    }}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), coverage)
    root = tmp_path / "exports"
    root.mkdir()
    database = root / "hirid"
    _package(database)
    for group in coverage["SPECIAL_GROUPS"].values():
        pq.write_table(pa.table({name: [1] for name in group["concepts"]}),
                       database / (group["module"] + ".parquet"))
    opened = []
    real_parser = pq.ParquetFile

    def parse(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_parser(path, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", parse)
    assert coverage["audit"](str(root)) == 1
    assert not opened
    assert "REFUSED" in capsys.readouterr().out


@pytest.mark.parametrize("inventory", ["empty", "definition_files", "feature_definitions", "column_metadata"])
def test_zip_formal_empty_inventory_bypass_is_refused(tmp_path, inventory):
    root = tmp_path / "export"
    _package(root, legacy=True)
    manifest = {"files": []}
    if inventory == "definition_files":
        manifest[inventory] = [{"file": "data.csv"}]
    elif inventory == "feature_definitions":
        manifest[inventory] = {"files": [{"file": "data.csv"}]}
    elif inventory == "column_metadata":
        manifest[inventory] = {"file": "data.csv"}
    _save(root, manifest)
    with publication.formal_integrity_scope(propagate_to_children=False):
        with pytest.raises(export_download.ExportDownloadError) as caught:
            _bundle(root)
    assert caught.value.code == publication.INTEGRITY_FAILED_CODE
    assert not list((tmp_path / "downloads").glob("*.zip"))


def test_zip_checks_extra_packed_members_even_with_valid_data(tmp_path):
    root = tmp_path / "export"
    manifest = _package(root)
    (root / "extra.csv").write_bytes(b"value\n2\n")
    manifest["definition_files"] = [{"file": "extra.csv"}]
    _save(root, manifest)
    with publication.formal_integrity_scope(propagate_to_children=False):
        with pytest.raises(export_download.ExportDownloadError) as caught:
            _bundle(root)
    assert caught.value.code == publication.INTEGRITY_FAILED_CODE
    assert caught.value.__cause__.reason == publication.REASON_DIGEST_UNRECORDED


def test_zip_receipt_includes_verified_auxiliary_members(tmp_path):
    root = tmp_path / "export"
    manifest = _package(root)
    (root / "definitions.csv").write_bytes(b"concept,module\nvalue,synthetic\n")
    manifest["definition_files"] = [
        {"file": "definitions.csv", **publication.member_digest_record(root / "definitions.csv")}
    ]
    _save(root, manifest)
    with publication.formal_integrity_scope(propagate_to_children=False):
        bundle = _bundle(root)
    try:
        with zipfile.ZipFile(bundle.path) as archive:
            receipt = json.loads(archive.read(export_download.INTEGRITY_RECEIPT_NAME))
            assert receipt["members"] == {"data.csv": "verified", "definitions.csv": "verified"}
            assert set(receipt["snapshots"]) == set(archive.namelist()) - {
                export_download.INTEGRITY_RECEIPT_NAME}
    finally:
        bundle.path.unlink()


def test_zip_verified_receipt_covers_changed_bytes_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "export"
    manifest = _package(root)
    original_manifest = (root / "_manifest.json").read_bytes()
    original_bytes = (root / "data.csv").read_bytes()
    real_enter = zipfile.ZipFile.__enter__
    changed = []

    def change_after_verification(archive):
        if archive.mode == "w":
            (root / "data.csv").write_bytes(b"value\n2\n")
            (root / "_manifest.json").write_bytes(b"{}")
            changed.append(True)
        return real_enter(archive)

    monkeypatch.setattr(zipfile.ZipFile, "__enter__", change_after_verification)
    with publication.formal_integrity_scope(propagate_to_children=False):
        bundle = _bundle(root)
    try:
        with zipfile.ZipFile(bundle.path) as archive:
            assert changed
            assert archive.read("data.csv") == original_bytes
            assert archive.read("_manifest.json") == original_manifest
            receipt = json.loads(archive.read(export_download.INTEGRITY_RECEIPT_NAME))
            assert receipt["status"] == "verified"
            assert receipt["members"] == {"data.csv": "verified"}
            assert set(receipt["snapshots"]) == set(archive.namelist()) - {
                export_download.INTEGRITY_RECEIPT_NAME}
            for name, facts in receipt["snapshots"].items():
                payload = archive.read(name)
                assert facts == {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            assert receipt["snapshots"]["data.csv"]["sha256"] == manifest["files"][0]["sha256"]
    finally:
        bundle.path.unlink()


@pytest.mark.parametrize("scope", [False, True])
def test_formal_verification_bypasses_an_existing_stat_memo(tmp_path, monkeypatch, scope):
    root = tmp_path / "export"
    _package(root)
    monkeypatch.setattr(publication, "_RACY_WINDOW_NS", -(10**18))
    publication.clear_integrity_memo()
    try:
        publication.verify_member_integrity(root, "data.csv")
        assert publication._DIGEST_MEMO
        hashed = []
        real_hash = publication._hash_file

        def hash_file(path):
            hashed.append(path)
            return real_hash(path)

        monkeypatch.setattr(publication, "_hash_file", hash_file)
        if scope:
            with publication.formal_integrity_scope(propagate_to_children=False):
                publication.verify_member_integrity(root, "data.csv")
                publication.verify_member_integrity(root, "data.csv")
        else:
            publication.verify_member_integrity(root, "data.csv", formal=True)
            publication.verify_member_integrity(root, "data.csv", formal=True)
        assert len(hashed) == 2
    finally:
        publication.clear_integrity_memo()
