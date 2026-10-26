"""Saved registration metadata retains only the host's verified provenance."""

import json

import pytest

from easyicu.webserver import sources


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "sources.json")
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(sources, "_preferred_configured_export", lambda *a, **k: None)
    descriptor = {"ok": True, "database": "miiv", "source_kind": "synthetic", "provenance": {}}
    monkeypatch.setattr(sources.dataio, "describe_export_source", lambda path: dict(descriptor))
    return tmp_path / "fixture", descriptor


def test_verified_synthetic_flag_survives_persistence_without_a_second_scan(registry, monkeypatch):
    path, _ = registry
    sources.register_source(str(path))
    assert json.loads(sources._CONFIG_PATH.read_text())["sources"][0]["source_kind"] == "synthetic"
    monkeypatch.setattr(sources, "load_registry", lambda: pytest.fail("rescanned registry"))
    monkeypatch.setattr(sources.dataio, "describe_export_source", lambda *a: pytest.fail("read export"))
    record, = sources.stored_registration_metadata()
    assert record["source_kind"] == "synthetic"
    assert "path" not in record
    assert "provenance" not in record


def test_browser_flags_cannot_make_a_source_synthetic_and_replacement_clears_it(registry):
    path, descriptor = registry
    sources.register_source(str(path))
    descriptor.pop("source_kind")
    sources.save_registry({"sources": [{"path": str(path), "source_kind": "synthetic"}]})
    record, = sources.stored_registration_metadata()
    assert "source_kind" not in record


def test_snapshot_filters_removed_disallowed_and_malformed_records(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "sources.json")
    raw = {"sources": [None, {"path": []}, {"path": ""},
                       {"path": "/removed", "id": "gone"}, {"path": "/denied", "id": "denied"},
                       {"path": "/fixture", "id": "kept", "database": "miiv", "summary": {"rows": 99}}],
           "removed_paths": ["/removed"]}
    sources._CONFIG_PATH.write_text(json.dumps(raw))
    monkeypatch.setattr(sources, "path_allowed", lambda path, **kw: path != "/denied")
    assert sources.stored_registration_metadata() == ({"id": "kept", "database": "miiv"},)


@pytest.mark.parametrize("record", [
    {"path": "/fixture", "database": "miiv"},
    {"path": "/fixture", "database": "miiv", "id": "/invalid/id"},
])
def test_no_identity_is_invented_for_an_unidentified_record(tmp_path, monkeypatch, record):
    from easyicu.webserver.data_map import build_data_map

    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "sources.json")
    sources._CONFIG_PATH.write_text(json.dumps({"sources": [record]}))
    assert all(row.source_total == 0 for row in build_data_map(concept="hr").databases)
