"""Saved source resolution never inspects unrelated or unapproved data."""

import json

import pytest

from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import sources


@pytest.fixture
def saved_sources(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    selected = str((tmp_path / "selected").resolve())
    other = str((tmp_path / "other").resolve())
    identity = SourceIdentity(database="eicu", registration_id=sources._source_id(selected),
                              version="2.0", source_id="demo", archive_sha256="a" * 64)
    row = {**identity.to_dict(), "id": identity.registration_id, "path": selected}
    row.pop("registration_id")
    other_row = {"id": sources._source_id(other), "path": other, "database": "mimiciv"}
    monkeypatch.setattr(sources, "_CONFIG_PATH", registry)
    monkeypatch.setattr(sources, "path_allowed", lambda path, **kwargs: True)

    def forbidden(*args, **kwargs):
        pytest.fail("stored identity lookup must not inspect or discover export data")

    monkeypatch.setattr(sources, "load_registry", forbidden)
    monkeypatch.setattr(sources.dataio, "describe_export_source", forbidden)
    monkeypatch.setattr(sources, "_autodiscovered_paths", forbidden)

    def write(rows, removed=()):
        registry.write_text(json.dumps({"sources": rows, "removed_paths": list(removed)}))

    return identity, row, other_row, write


def test_resolves_only_saved_full_identity_without_scanning_sources(saved_sources):
    identity, row, other, write = saved_sources
    write([row, other])
    assert sources.stored_source_path_for_identity(identity) == row["path"]
    assert sources.stored_source_path_for_identity(SourceIdentity("eicu", identity.registration_id)) is None


@pytest.mark.parametrize("mutation", ["duplicate_id", "duplicate_path", "forged_path", "removed", "bad_digest", "unsafe"])
def test_rejects_malicious_or_ambiguous_registry(saved_sources, monkeypatch, mutation):
    identity, row, other, write = saved_sources
    removed = []
    rows = [row, other]
    if mutation == "duplicate_id":
        rows.append({**other, "id": identity.registration_id})
    elif mutation == "duplicate_path":
        rows.append({**other, "path": row["path"]})
    elif mutation == "forged_path":
        rows[0] = {**row, "path": other["path"]}
        rows.pop()
    elif mutation == "removed":
        removed = [row["path"]]
    elif mutation == "bad_digest":
        rows[0] = {**row, "archive_sha256": "invalid"}
    else:
        monkeypatch.setattr(sources, "path_allowed", lambda path, **kwargs: False)
    write(rows, removed)
    assert sources.stored_source_path_for_identity(identity) is None
