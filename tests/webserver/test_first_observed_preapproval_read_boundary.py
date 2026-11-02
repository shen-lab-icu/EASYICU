"""A source choice does not authorize reading raw patient bytes."""

import json
import io
import os
from types import SimpleNamespace
import pytest

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.webserver import demo_source_storage, first_observed_preparation, raw_preparation_projection
from easyicu.research_agent.research_context.export_selection import SourceIdentity


def _refuse_read(*args, **kwargs):
    raise AssertionError("pre-approval code tried to access patient data")


def test_scope_generation_asks_for_metadata_only(monkeypatch, tmp_path):
    def metadata_only(**kwargs):
        assert kwargs["verify_source_bytes"] is False
        return {"read_scope_version": 2}, tmp_path
    monkeypatch.setattr(first_observed_preparation, "_host_inputs", metadata_only)
    assert first_observed_preparation.proposed_first_observed_scope(run_dir=tmp_path, request=object()) == {"read_scope_version": 2}


def test_legacy_scope_is_not_upgraded_or_consumed(monkeypatch, tmp_path):
    def metadata_only(**kwargs):
        assert kwargs["verify_source_bytes"] is False
        return {"read_scope_version": 2}, tmp_path
    monkeypatch.setattr(first_observed_preparation, "_host_inputs", metadata_only)
    monkeypatch.setattr(first_observed_preparation, "consume_host_review", _refuse_read)
    with pytest.raises(ValueError, match="first_observed_preparation_scope_mismatch"):
        first_observed_preparation.prepare_first_observed_coordinate(
            run_dir=tmp_path, checkpoint=SimpleNamespace(request=object()), binding=None,
            scope={"patient_table_sha256": "a" * 64})


def test_registered_scope_metadata_never_opens_patient_or_archive(monkeypatch, tmp_path):
    export, raw = tmp_path / "export", tmp_path / "raw"
    export.mkdir(); raw.mkdir()
    (export / "_manifest.json").write_text(json.dumps({"database": "eicu_demo", "data_path": str(raw)}))
    (raw / "conversion_manifest.json").write_text("{}")
    for name in ("prepared", "extracted", "converted"):
        (tmp_path / name).write_text("{}")
    patient, archive = raw / "patient.parquet", tmp_path / "archive.zip"
    patient.write_bytes(b"private synthetic bytes"); archive.write_bytes(b"private synthetic bytes")
    source = SimpleNamespace(database="eicu_demo", id="demo", version="1", archive_sha256="a" * 64)
    paths = SimpleNamespace(raw=raw, prepared_marker=tmp_path / "prepared",
                            extracted_marker=tmp_path / "extracted", converted_marker=tmp_path / "converted")
    identity = SourceIdentity(database="eicu_demo", registration_id="synthetic", source_id="demo",
                              version="1", archive_sha256="a" * 64)
    monkeypatch.setattr(first_observed_preparation.sources, "stored_source_path_for_identity", lambda _: str(export))
    monkeypatch.setattr(demo_source_storage, "verified_public_demo_replanning_export", lambda _: (source, paths))
    monkeypatch.setattr(demo_source_storage, "read_marker", lambda *_: {"archive_sha256": "a" * 64})
    real_open = os.open
    def reject_patient(path, *args, **kwargs):
        assert str(path) not in {str(patient), str(archive), patient.name, archive.name}
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", reject_patient)
    selected, metadata = first_observed_preparation._metadata_source(identity)
    assert selected == export and metadata["archive_sha256"] == "a" * 64
    assert "patient_table_sha256" not in metadata
    (raw / "conversion_manifest.json").write_text('{"changed":true}')
    _, changed = first_observed_preparation._metadata_source(identity)
    assert changed["conversion_metadata_sha256"] != metadata["conversion_metadata_sha256"]
    (export / "_manifest.json").write_text(json.dumps({"database": "eicu_demo", "data_path": "elsewhere"}))
    with pytest.raises(ValueError, match="first_observed_preparation_source_mismatch"):
        first_observed_preparation._metadata_source(identity)


def test_real_demo_metadata_helpers_do_not_open_data_inodes(monkeypatch, tmp_path):
    source = next(item for item in demo_source_storage.SOURCES if item.database == "eicu_demo")
    cache = tmp_path / "cache"
    monkeypatch.setenv(demo_source_storage.CACHE_ENV, str(cache))
    monkeypatch.setenv("EASYICU_HOSTED", "0")
    paths = demo_source_storage.source_paths(source)
    raw, export = paths.raw, paths.root / "exports" / ("1" * 16)
    raw.mkdir(parents=True); export.mkdir(parents=True)
    (export / "_manifest.json").write_text(json.dumps({"database": source.database, "data_path": str(raw)}))
    (raw / "conversion_manifest.json").write_text("{}")
    for marker in (raw / ".easyicu-demo-extracted.json", raw / ".easyicu-demo-converted.json"):
        demo_source_storage.write_marker(marker, source, archive_sha256=source.archive_sha256)
    demo_source_storage.write_marker(export / ".easyicu-demo-prepared.json", source,
        archive_sha256=source.archive_sha256, producer_identity="sha256:" + "1" * 64)
    patient, archive = raw / "patient.parquet", paths.archive
    patient.write_bytes(b"synthetic patient data"); archive.write_bytes(b"synthetic archive data")
    registration = first_observed_preparation.sources._source_id(str(export))
    identity = SourceIdentity(database=source.database, registration_id=registration, source_id=source.id,
                              version=source.version, archive_sha256=source.archive_sha256)
    registry = tmp_path / "registry.json"
    row = {**identity.to_dict(), "id": registration, "path": str(export)}
    row.pop("registration_id")
    registry.write_text(json.dumps({"sources": [row]}))
    monkeypatch.setattr(first_observed_preparation.sources, "_CONFIG_PATH", registry)
    forbidden = {(p.stat().st_dev, p.stat().st_ino) for p in (patient, archive)}
    real_open, real_io_open = os.open, io.open
    def checked_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) in forbidden:
            os.close(fd)
            raise AssertionError("pre-approval opened a patient or archive inode")
        return fd
    def checked_io_open(path, *args, **kwargs):
        handle = real_io_open(path, *args, **kwargs)
        info = os.fstat(handle.fileno())
        if (info.st_dev, info.st_ino) in forbidden:
            handle.close()
            raise AssertionError("pre-approval opened a patient or archive inode")
        return handle
    monkeypatch.setattr(os, "open", checked_open)
    monkeypatch.setattr(io, "open", checked_io_open)
    selected, metadata = first_observed_preparation._metadata_source(identity)
    assert selected == export and metadata["source_id"] == source.id
    assert metadata["source_version"] == source.version and "patient_table_sha256" not in metadata


def test_old_pending_review_is_informational_without_source_preview(monkeypatch, tmp_path):
    monkeypatch.setattr(raw_preparation_projection, "_candidate_projection", _refuse_read)
    scope = {"adapter_id": "eicu-demo-first-observed-single-hospital-adult/1",
             "purpose": "prepare_only", "analysis_authorized": False}
    payload = {"first_observed_icu_preparation_scope": scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Review the read scope",
                                        payload=payload, authority_sha256=canonical_sha256(payload))
    checkpoint = RawPreparationCheckpoint.pending(plan_sha256="a" * 64, request=request)
    (tmp_path / "raw_preparation_checkpoint.json").write_text(checkpoint.model_dump_json())
    projected = raw_preparation_projection.pending_preparation_projection(
        {"run_id": "synthetic", "project_dir": str(tmp_path)}, study_context_id="synthetic-study")
    assert projected["scope"] == scope
    assert projected["approvable"] is False
    assert "candidate_preparation_preview" not in projected
