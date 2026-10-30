"""Metadata partition candidates preserve the registered-source boundary."""
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import source_partition_candidates as discovery


@pytest.fixture
def source(tmp_path, monkeypatch):
    export = tmp_path / "export"
    root = tmp_path / "raw"
    export.mkdir()
    (root / "laboratory").mkdir(parents=True)
    cases = pa.table({"CaseID": [1], "PatientID": [1], "ICUOffset": [0], "TimeOfStay": [10]})
    lab = pa.table({"CaseID": [1], "LaboratoryID": [2], "Offset": [0], "LaboratoryValue": [1.]})
    pq.write_table(cases, root / "cases.parquet")
    pq.write_table(lab, root / "laboratory" / "1.parquet")
    pq.write_table(lab, root / "laboratory" / "2.parquet")
    manifest = {"schema_version": "easyicu.conversion_manifest/1", "database": "sic", "tables": [
        {"table": "cases", "status": "completed", "outputs": [
            {"relative_path": "cases.parquet"}, {"relative_path": "cases.parquet"}]},
        {"table": "laboratory", "status": "completed", "outputs": [
            {"relative_path": "laboratory/1.parquet"}, {"relative_path": "laboratory/2.parquet"}]},
    ]}
    (root / "conversion_manifest.json").write_text(json.dumps(manifest))
    identity = SourceIdentity(database="sic", registration_id="registered")
    monkeypatch.setattr(discovery.sources, "load_registry", lambda: {"sources": [{"ok": True, "path": str(export)}]})
    monkeypatch.setattr(discovery.sources, "registered_source_identity", lambda p: identity if p == str(export) else None)
    binding = SimpleNamespace(source_root=root, export_manifest_sha256="a" * 64, table_sha256={"cases": "b" * 64})
    monkeypatch.setattr(discovery, "resolve_raw_source_identity_binding", lambda **_: binding)
    return identity, root, manifest


def test_discovers_sealed_export_then_local_declared_partitions_without_claiming_completeness(source):
    identity, _, _ = source
    result = discovery.discover_source_partition_candidates(identity=identity,
        requested_columns={"cases": ["CaseID", "PatientID"], "laboratory": ["CaseID", "Offset", "LaboratoryValue"]})
    assert [(item["table"], item["path"]) for item in result["files"]] == [
        ("cases", "cases.parquet"), ("laboratory", "laboratory/1.parquet"), ("laboratory", "laboratory/2.parquet")]
    assert all(len(item["schema_sha256"]) == 64 for item in result["files"])
    assert result["identical_duplicate_entries_coalesced"] == 1
    assert result["export_manifest_sha256"] == "a" * 64 and result["identity_table_sha256"] == {"cases": "b" * 64}
    assert not result["source_rows_decoded"] and not result["files_content_verified"]
    assert not result["full_source_roster_verified"] and not result["analysis_authorized"]


def test_unlisted_partition_and_unsafe_paths_are_rejected(source):
    identity, root, manifest = source
    (root / "laboratory" / "unlisted.parquet").write_bytes(b"extra")
    with pytest.raises(ValueError, match="directory_roster_mismatch"):
        discovery.discover_source_partition_candidates(identity=identity, requested_columns={"laboratory": ["CaseID"]})
    (root / "laboratory" / "unlisted.parquet").unlink()
    manifest["tables"][1]["outputs"][0]["relative_path"] = "../evil.parquet"
    (root / "conversion_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path_invalid"):
        discovery.discover_source_partition_candidates(identity=identity, requested_columns={"laboratory": ["CaseID"]})


def test_missing_unconfigured_and_unavailable_source_refuse(source, monkeypatch):
    identity, root, _ = source
    with pytest.raises(ValueError, match="columns_not_configured"):
        discovery.discover_source_partition_candidates(identity=identity, requested_columns={"cases": ["unexpected"]})
    (root / "laboratory" / "2.parquet").unlink()
    with pytest.raises(ValueError, match="schema_unavailable"):
        discovery.discover_source_partition_candidates(identity=identity, requested_columns={"laboratory": ["CaseID"]})
    monkeypatch.setattr(discovery, "resolve_raw_source_identity_binding", lambda **_: None)
    with pytest.raises(ValueError, match="binding_unavailable"):
        discovery.discover_source_partition_candidates(identity=identity, requested_columns={"cases": ["CaseID"]})


def test_source_or_manifest_changes_while_discovering_are_rejected(source, monkeypatch):
    identity, root, manifest = source
    calls = 0
    original = discovery.resolve_raw_source_identity_binding
    def changed(**kwargs):
        nonlocal calls
        calls += 1
        value = original(**kwargs)
        if calls == 2:
            manifest["database"] = "other"
            (root / "conversion_manifest.json").write_text(json.dumps(manifest))
            return SimpleNamespace(**{**vars(value), "export_manifest_sha256": "c" * 64})
        return value
    monkeypatch.setattr(discovery, "resolve_raw_source_identity_binding", changed)
    with pytest.raises(ValueError, match="source_changed"):
        discovery.discover_source_partition_candidates(identity=identity, requested_columns={"cases": ["CaseID"]})


def test_conflicting_duplicate_and_cross_table_file_assignment_are_rejected(source):
    identity, root, manifest = source
    manifest["tables"][0]["outputs"][1]["size_bytes"] = 123
    (root / "conversion_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate_conflict"):
        discovery.discover_source_partition_candidates(identity=identity, requested_columns={"cases": ["CaseID"]})
    manifest["tables"][0]["outputs"][1].pop("size_bytes")
    manifest["tables"][1]["outputs"] = [{"relative_path": "cases.parquet"}]
    (root / "conversion_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="file_assigned_twice"):
        discovery.discover_source_partition_candidates(identity=identity,
            requested_columns={"cases": ["CaseID"], "laboratory": ["CaseID"]})
