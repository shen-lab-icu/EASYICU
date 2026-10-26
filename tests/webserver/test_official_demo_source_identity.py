"""Public-demo classification follows host receipts across metadata-only edits."""

from dataclasses import replace

import pytest

from easyicu.webserver import demo_sources, sources
from easyicu.webserver.data_map import build_data_map
from easyicu.webserver.routes import workspaces


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "sources.json")
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(sources, "_preferred_configured_export", lambda *a, **k: None)
    descriptor = {"ok": True, "database": "miiv", "files": []}
    monkeypatch.setattr(sources.dataio, "describe_export_source", lambda path: dict(descriptor))
    return tmp_path / "fixture", descriptor


def receipt():
    demo = demo_sources.get_source("mimic_iv_demo_v2_2")
    return {"source_id": demo.id, "version": demo.version, "archive_sha256": demo.archive_sha256}


@pytest.mark.parametrize("edit", ["rename", "patch"])
def test_renamed_miiv_demo_keeps_its_receipt_kind_counts_and_data_map(registry, edit):
    path, _ = registry
    sources.register_source(str(path), label="Official release", source_identity=receipt())
    body = {"path": str(path), "label": "My teaching collection", "source_id": "forged",
            "archive_sha256": "0" * 64, "source_kind": "own", "source_identity": {}}
    if edit == "rename":
        result = workspaces.post_workspaces_rename(body)
    else:
        result = workspaces.post_workspaces_registry({"sources": [body]})
    row, = result["sources"]
    assert row["label"] == "My teaching collection"
    assert sources.is_official_demo_source(row)
    assert sources.is_demo_source(row)
    assert row["source_kind"] == "public_demo"
    record, = sources.stored_registration_metadata()
    for key, value in receipt().items():
        assert row[key] == record[key] == value
    assert "path" not in record
    mapped, = build_data_map(database="miiv", concept="hr").databases[0].sources
    assert mapped.kind == "public_demo"
    assert mapped.identity.source_id == receipt()["source_id"]
    assert sources.registered_source_counts()["by_database"]["miiv"] == 0


@pytest.mark.parametrize("identity,expected", [
    ({}, True),  # A user-entered matching title retains the explicit legacy fallback.
    ({"source_id": "unlisted_release"}, False),
    ({"source_id": "mimic_iv_demo_v2_2"}, False),
    ({"source_id": "mimic_iv_demo_v2_2", "archive_sha256": "0" * 64}, False),
    ({"archive_sha256": "0" * 64}, False),
])
def test_only_rows_without_a_release_receipt_use_the_legacy_title(identity, expected):
    demo = demo_sources.get_source("mimic_iv_demo_v2_2")
    row = {"database": "miiv", "label": f"  {demo.title.upper()} v{demo.version}  ", **identity}
    assert sources.is_official_demo_source(row) is expected
    assert sources.is_demo_source(row) is expected


def test_unpinned_catalog_release_needs_only_an_allowlisted_identity(monkeypatch):
    demo = replace(demo_sources.get_source("mimic_iv_demo_v2_2"), archive_sha256=None)
    monkeypatch.setattr(demo_sources, "get_source", lambda source_id: demo)
    assert sources.is_official_demo_source({"source_id": demo.id, "label": "Renamed"})


def test_browser_registration_cannot_supply_a_demo_receipt_or_kind(registry):
    path, _ = registry
    result = workspaces.post_workspaces_register({
        "path": str(path), "label": "My export", **receipt(),
        "source_identity": receipt(), "source_kind": "public_demo",
    })
    row, = result["sources"]
    assert row["source_kind"] == "own"
    assert not sources.is_official_demo_source(row)
    assert "source_id" not in row and "archive_sha256" not in row


def test_user_registration_with_a_demo_title_retains_the_legacy_fallback(registry):
    path, _ = registry
    demo = demo_sources.get_source("mimic_iv_demo_v2_2")
    result = sources.register_source(str(path), label=f"{demo.title} v{demo.version}")
    row, = result["sources"]
    assert "source_id" not in row and "archive_sha256" not in row
    assert row["source_kind"] == "public_demo"
    assert sources.is_official_demo_source(row)
    mapped, = build_data_map(database="miiv", concept="hr").databases[0].sources
    assert mapped.kind == "public_demo"


@pytest.mark.parametrize("database", ["eicu_demo", "mimic_demo"])
def test_legacy_profiled_demo_keeps_its_kind_without_a_release_receipt(registry, database):
    path, descriptor = registry
    descriptor["database"] = database
    row, = sources.register_source(str(path), label="Older teaching export")["sources"]
    assert "source_id" not in row
    assert not sources.is_official_demo_source(row)
    assert sources.is_demo_source(row)
    assert row["source_kind"] == "public_demo"


def test_renaming_a_discovered_synthetic_export_saves_its_provenance(registry, monkeypatch):
    path, descriptor = registry
    descriptor.update(source_kind="synthetic", provenance={})
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda: [str(path)])
    sources.rename_source(str(path), "Discovered fixture")
    record, = sources.stored_registration_metadata()
    assert record["source_kind"] == "synthetic"
    mapped, = build_data_map(database="miiv", concept="hr").databases[0].sources
    assert mapped.kind == "synthetic"


@pytest.mark.parametrize("edit", ["rename", "patch", "rename_other"])
def test_synthetic_receipt_and_title_stay_synthetic_in_registry_and_map(registry, edit):
    path, descriptor = registry
    descriptor.update(source_kind="synthetic", provenance={})
    demo = demo_sources.get_source("mimic_iv_demo_v2_2")
    sources.register_source(str(path), label=f"{demo.title} v{demo.version}", source_identity=receipt())
    if edit == "rename_other":
        other = path.parent / "other"
        sources.register_source(str(other), label="Other")
        sources.rename_source(str(other), "Other renamed")
    elif edit == "rename":
        sources.rename_source(str(path), "Teaching fixture")
    else:
        sources.save_registry({"sources": [{"path": str(path), "label": "Teaching fixture"}]})
    row = next(row for row in sources.load_registry()["sources"] if row["path"] == str(path))
    record = next(saved for saved in sources.stored_registration_metadata() if saved["id"] == row["id"])
    assert row["source_kind"] == record["source_kind"] == "synthetic"
    assert sources.is_demo_source(row)
    mapped = next(row for row in build_data_map(database="miiv", concept="hr").databases[0].sources
                  if row.identity.registration_id == record["id"])
    assert mapped.kind == "synthetic"


def test_a_study_bound_to_a_renamed_demo_records_the_demo_release(registry, tmp_path, monkeypatch):
    from easyicu.webserver import study_contexts
    from easyicu.webserver.pi_copilot import study_context_update
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext
    from easyicu.webserver.pi_copilot.service import PiCopilotService

    path, _ = registry
    sources.register_source(str(path), label="Official release", source_identity=receipt())
    workspaces.post_workspaces_rename({"path": str(path), "label": "Teaching collection"})
    row, = sources.load_registry()["sources"]
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "contexts.json")
    study = study_contexts.upsert_context({"id": "study-renamed-demo", "question": "Describe ICU stays."})
    result = study_context_update.update_study_context(
        ToolExecutionContext(session=PiSessionRecord(session_id="pi-renamed-demo"), allowed_actions={"configure"}),
        {"bind_source_id": row["id"]}, load_context=lambda _binding: study,
        project_workflow=lambda _context, *, study_override: {}, source_registry={"sources": [row]},
    )
    assert result["status"] == "ok"
    demo = demo_sources.get_source(receipt()["source_id"])
    saved = study_contexts.get_context(study["id"])
    assert saved["data_source"]["label"] == f"{demo.title} v{demo.version}"
    reference = PiCopilotService._session_source_reference(saved)
    # The demo, never the full release the database key also names.
    assert (reference.database, reference.label, reference.reference_release) == (
        "miiv", f"{demo.title} v{demo.version}", None)
