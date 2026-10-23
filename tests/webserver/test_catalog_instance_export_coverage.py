"""The instance dictionary reports provided concepts from synthetic manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from easyicu.concept import catalog as cc
from easyicu.webserver import catalog, dataio, sources


@pytest.fixture
def instance(monkeypatch):
    catalog._INSTANCE_COVERAGE_CACHE.clear()
    concepts = ("signal", "derived_signal", "other_signal")
    monkeypatch.setattr(cc, "CONCEPT_DICTIONARY", {
        key: (key, "测试", "") for key in concepts
    })
    monkeypatch.setattr(catalog, "catalog_definitions", lambda: {
        "signal": {"sources": {"eicu": [{}], "miiv": [{}]}},
        "other_signal": {"sources": {"miiv": [{}]}},
    })
    registry = {"sources": [], "active_path": None}
    monkeypatch.setattr(sources, "load_registry", lambda: registry)

    def no_rows(*args, **kwargs):
        raise AssertionError("instance coverage must not read data rows or schemas")

    monkeypatch.setattr(dataio, "_read_columns", no_rows)
    monkeypatch.setattr(dataio, "_read_export_frame", no_rows)
    monkeypatch.setattr(dataio, "_fast_stay_ids", no_rows)
    monkeypatch.setattr(catalog, "_file_concept_coverage", no_rows)
    return registry


def _file(
    concepts=("signal",), *, availability="available", field="columns", name="vitals.parquet"
):
    return {
        "file": name,
        "module": "vitals",
        "rows": 3,
        field: ({key: "double" for key in concepts} if field == "physical_schema" else list(concepts)),
        "concept_status": {key: {"availability": availability} for key in concepts},
    }


def _source(root, *, database="eicu", files=None, native=True, marker="_manifest.json"):
    root.mkdir()
    files = [_file()] if files is None else files
    for entry in files:
        member = root / entry["file"]
        member.parent.mkdir(parents=True, exist_ok=True)
        member.touch()
    manifest = {"database": database, "files": files}
    if native:
        manifest["schema_version"] = "easyicu_native_export_v2"
    (root / marker).write_text(json.dumps(manifest), encoding="utf-8")
    return {"id": root.name, "path": str(root), "database": database, "ok": True}


def _coverage(concept="signal"):
    return catalog.build_catalog()["conceptCoverage"][concept]


@pytest.mark.parametrize("availability", [
    "available", "produced_all_null", "structurally_unavailable_placeholder",
])
@pytest.mark.parametrize("field", ["columns", "physical_schema", "column_metadata_columns"])
def test_instance_coverage_reads_native_columns_and_value_status(
    tmp_path, instance, availability, field,
):
    instance["sources"] = [_source(
        tmp_path / "export", files=[_file(availability=availability, field=field)],
    )]
    coverage = _coverage()
    assert coverage["instance_databases_covered"] == (["eicu"] if availability == "available" else [])
    assert coverage["instance_coverage_basis"] == "export_manifest"


def test_a_declared_or_selected_concept_without_a_column_is_not_provided(tmp_path, instance):
    entry = _file(concepts=("other_signal",))
    entry["concept_ids"] = ["signal"]
    entry["concept_status"]["signal"] = {"availability": "available"}
    instance["sources"] = [_source(tmp_path / "export", files=[entry])]
    assert _coverage()["instance_databases_covered"] == []


@pytest.mark.parametrize("empty", ["no_files", "no_columns", "empty_columns"])
def test_native_metadata_without_columns_never_falls_back(tmp_path, instance, empty):
    entry = _file()
    if empty == "no_columns":
        entry.pop("columns")
    if empty == "empty_columns":
        entry["columns"] = []
        # An explicit empty physical column list wins over older hints.
        entry["column_metadata_columns"] = ["signal"]
    instance["sources"] = [_source(
        tmp_path / "export", files=[] if empty == "no_files" else [entry],
    )]
    assert _coverage()["instance_databases_covered"] == []
    assert _coverage()["instance_coverage_basis"] == "export_manifest"


def test_an_exported_loader_output_needs_no_declared_mapping(tmp_path, instance):
    instance["sources"] = [_source(
        tmp_path / "export", files=[_file(concepts=("derived_signal",))],
    )]
    assert _coverage("derived_signal")["instance_databases_covered"] == ["eicu"]


@pytest.mark.parametrize("unavailable", ["missing_file", "empty_module", "unavailable_module"])
def test_a_module_that_provides_no_values_is_not_instance_coverage(tmp_path, instance, unavailable):
    entry = _file()
    if unavailable == "empty_module":
        entry["rows"] = 0
    if unavailable == "unavailable_module":
        entry["availability"] = "structurally_unavailable"
    source = _source(tmp_path / "export", files=[entry])
    if unavailable == "missing_file":
        (Path(source["path"]) / entry["file"]).unlink()
    instance["sources"] = [source]
    assert _coverage()["instance_databases_covered"] == []


def test_native_columns_without_an_availability_statement_remain_usable(tmp_path, instance):
    entry = _file()
    del entry["concept_status"]
    instance["sources"] = [_source(tmp_path / "export", files=[entry])]
    assert _coverage()["instance_databases_covered"] == ["eicu"]
    assert _coverage()["instance_coverage_basis"] == "export_manifest"


def test_another_module_or_source_can_provide_the_same_concept(tmp_path, instance):
    instance["sources"] = [
        _source(tmp_path / "eicu_empty", files=[_file(availability="produced_all_null")]),
        _source(tmp_path / "miiv", database="miiv", files=[
            _file(availability="produced_all_null"),
            _file(name="other.parquet"),
        ]),
        _source(tmp_path / "eicu_present"),
    ]
    assert _coverage()["instance_databases_covered"] == ["eicu", "miiv"]
    instance["sources"].pop()
    assert _coverage()["instance_databases_covered"] == ["miiv"]


def test_removing_and_restoring_a_cached_module_changes_coverage(tmp_path, instance):
    source = _source(tmp_path / "export", files=[_file(name="modules/vitals.parquet")])
    instance["sources"] = [source]
    assert _coverage()["instance_databases_covered"] == ["eicu"]
    member = Path(source["path"]) / "modules/vitals.parquet"
    member.unlink()
    assert _coverage()["instance_databases_covered"] == []
    member.touch()
    assert _coverage()["instance_databases_covered"] == ["eicu"]


@pytest.mark.parametrize("outside", ["absolute", "parent"])
def test_manifest_members_must_stay_in_the_registered_export(tmp_path, instance, outside):
    source = _source(tmp_path / "export")
    instance["sources"] = [source]
    escaped = tmp_path / "outside.parquet"
    escaped.touch()
    manifest = Path(source["path"]) / "_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["files"][0]["file"] = str(escaped) if outside == "absolute" else "../outside.parquet"
    manifest.write_text(json.dumps(payload))
    assert _coverage()["instance_databases_covered"] == []


@pytest.mark.parametrize("legacy", ["no_manifest", "older_manifest", "no_path"])
def test_only_sources_without_native_v2_use_declared_mapping(tmp_path, instance, legacy):
    source = _source(tmp_path / "export", native=False, files=[_file(concepts=("derived_signal",))])
    if legacy == "no_manifest":
        (Path(source["path"]) / "_manifest.json").unlink()
    if legacy == "no_path":
        source.pop("path")
    instance["sources"] = [source]
    assert _coverage()["instance_databases_covered"] == ["eicu"]
    assert _coverage()["instance_coverage_basis"] == "declared_mapping"
    assert _coverage("derived_signal")["instance_databases_covered"] == []


def test_mixed_sources_state_the_declared_fallback_conservatively(tmp_path, instance):
    instance["sources"] = [
        _source(tmp_path / "eicu", files=[_file(availability="produced_all_null")]),
        _source(tmp_path / "miiv", database="miiv", native=False),
    ]
    assert _coverage()["instance_databases_covered"] == ["miiv"]
    assert _coverage()["instance_coverage_basis"] == "declared_mapping"


def test_invalid_or_unidentified_registry_entries_do_not_contribute(tmp_path, instance):
    invalid = _source(tmp_path / "invalid")
    invalid["ok"] = False
    unidentified = _source(tmp_path / "unidentified")
    unidentified["database"] = None
    instance["sources"] = [invalid, unidentified]
    body = catalog.build_catalog()
    assert body["instance_databases"] == []
    assert body["conceptCoverage"]["signal"]["instance_databases_covered"] == []


def test_api_catalog_publishes_manifest_coverage_and_its_basis(tmp_path, instance):
    from fastapi.testclient import TestClient

    from easyicu.webserver.app import app

    instance["sources"] = [_source(
        tmp_path / "export", files=[_file(concepts=("derived_signal",))],
    )]
    response = TestClient(app).get("/api/catalog")
    assert response.status_code == 200
    coverage = response.json()["conceptCoverage"]
    assert coverage["derived_signal"]["instance_databases_covered"] == ["eicu"]
    assert coverage["derived_signal"]["instance_coverage_basis"] == "export_manifest"
    assert coverage["signal"]["instance_databases_covered"] == []


@pytest.mark.parametrize("marker", ["_manifest.json", "easyicu_export_manifest.json"])
def test_native_manifest_reads_are_cached_and_cannot_be_changed_by_a_caller(
    tmp_path, instance, monkeypatch, marker,
):
    instance["sources"] = [_source(tmp_path / "export", marker=marker)]
    reads = []
    real = dataio.read_prepared_export_manifest

    def counting(path):
        reads.append(path)
        return real(path)

    monkeypatch.setattr(dataio, "read_prepared_export_manifest", counting)
    for _ in range(5):
        coverage = _coverage()
        assert coverage["instance_databases_covered"] == ["eicu"]
        coverage["instance_databases_covered"].clear()
    assert len(reads) == 1


def test_both_supported_manifest_markers_invalidate_the_cache(tmp_path, instance):
    source = _source(tmp_path / "export", marker="easyicu_export_manifest.json")
    instance["sources"] = [source]
    assert _coverage()["instance_databases_covered"] == ["eicu"]
    # The reader gives _manifest.json precedence over the older marker.
    current = Path(source["path"]) / "_manifest.json"
    current.write_text(json.dumps({
        "schema_version": "easyicu_native_export_v2", "files": [],
    }))
    assert _coverage()["instance_databases_covered"] == []
    current.unlink()
    assert _coverage()["instance_databases_covered"] == ["eicu"]


def test_a_changed_registry_path_cannot_reuse_another_exports_cache(tmp_path, instance):
    first = _source(tmp_path / "first")
    second = _source(tmp_path / "second", files=[_file(concepts=("absent",))])
    first_manifest = Path(first["path"]) / "_manifest.json"
    second_manifest = Path(second["path"]) / "_manifest.json"
    stat = first_manifest.stat()
    assert second_manifest.stat().st_size == stat.st_size
    os.utime(second_manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    instance["sources"] = [first]
    assert _coverage()["instance_databases_covered"] == ["eicu"]
    # An entry may be repointed without changing its id or database.
    first["path"] = second["path"]
    assert _coverage()["instance_databases_covered"] == []


def test_native_manifest_cache_remains_bounded(tmp_path, instance, monkeypatch):
    monkeypatch.setattr(catalog, "_INSTANCE_COVERAGE_CACHE_LIMIT", 2)
    for index in range(3):
        instance["sources"] = [_source(tmp_path / f"export_{index}")]
        assert _coverage()["instance_databases_covered"] == ["eicu"]
    assert len(catalog._INSTANCE_COVERAGE_CACHE) == 2


@pytest.mark.parametrize("change", ["mtime", "size", "registry_entry", "manifest_added"])
def test_instance_coverage_refreshes_when_its_authority_changes(tmp_path, instance, change):
    source = _source(tmp_path / "export")
    instance["sources"] = [source]
    path = Path(source["path"]) / "_manifest.json"
    original = path.read_text()
    if change == "manifest_added":
        path.unlink()
    assert _coverage()["instance_databases_covered"] == ["eicu"]
    previous = path.stat() if path.exists() else None
    if change == "registry_entry":
        source["database"] = "miiv"
        assert _coverage()["instance_databases_covered"] == ["miiv"]
    else:
        # Equal-length column rename isolates the mtime part of the key.
        updated = original.replace('"signal"', '"absent"')
        if change == "size":
            updated += " "
        path.write_text(updated)
        if change == "size":
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        elif change == "mtime":
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
        assert _coverage()["instance_databases_covered"] == []
