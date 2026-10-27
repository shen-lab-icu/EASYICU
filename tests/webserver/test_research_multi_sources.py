"""Source exemptions are scoped to one independently verified registration."""
from dataclasses import replace

import pytest

from easyicu.research_agent.providers.client_trust import source_requires_clearance
from easyicu.webserver.research_multi_sources import egress_grant, select_source
from easyicu.webserver.research_multi_contracts import DatabaseTarget, MultiDatabaseError


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_verified_db6b_fixture_is_exempt_only_for_its_own_database(tmp_path, database):
    from easyicu.synthetic_sources import generate_source

    path = tmp_path / database
    generate_source(path, database=database, stays=4)
    source = dict(id="fixture_" + database, database=database, path=str(path))
    grant = egress_grant(source, "a" * 64, provider_sha256="b" * 64, data_handling="unspecified")
    assert grant.clearance == "verified_public_or_synthetic"
    assert source_requires_clearance(database)
    neighbour = {**source, "id": "credentialed", "database": "miiv", "path": str(tmp_path / "credentialed")}
    assert egress_grant(neighbour, "c" * 64, provider_sha256="b" * 64, data_handling="unspecified").clearance == "provider_required"


def test_public_demo_does_not_clear_a_credentialed_target(tmp_path, monkeypatch):
    from easyicu.webserver import demo_source_storage as storage
    from easyicu.webserver.demo_source_contracts import SOURCES

    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "demo"))
    public = SOURCES[0]
    paths = storage.source_paths(public)
    paths.export.mkdir(parents=True)
    (paths.export / "_manifest.json").write_text("{}")
    storage.write_export_marker(paths, public, archive_sha256=public.archive_sha256, export={})
    source = dict(id="demo", database=public.database, path=str(paths.export))
    assert egress_grant(source, "a" * 64, provider_sha256="b" * 64, data_handling="unspecified").clearance == "verified_public_or_synthetic"
    source["path"] = str(tmp_path / "credentialed")
    assert egress_grant(source, "c" * 64, provider_sha256="b" * 64, data_handling="unspecified").clearance == "provider_required"


def test_a_credentialed_target_is_marked_provider_required(tmp_path):
    source = dict(id="credentialed", database="aumc", path=str(tmp_path / "credentialed"))
    assert egress_grant(source, "a" * 64, provider_sha256="b" * 64, data_handling="unspecified").clearance == "provider_required"
    assert egress_grant(source, "a" * 64, provider_sha256="b" * 64, data_handling="on_device").clearance == "provider_required"


def test_registry_ambiguity_never_picks_an_arbitrary_export(monkeypatch, tmp_path):
    from easyicu.webserver import sources
    from easyicu.research_agent.research_context.export_selection import SourceIdentity

    rows = [dict(id="one", database="aumc", path=str(tmp_path / "one"), ok=True),
            dict(id="two", database="aumc", path=str(tmp_path / "two"), ok=True)]
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": rows})
    identity = SourceIdentity(database="aumc", registration_id="two")
    monkeypatch.setattr(sources, "registered_source_identity", lambda _: identity)
    with pytest.raises(MultiDatabaseError, match="source_ambiguous"):
        select_source(DatabaseTarget(database="aumc"))
    selected, digest = select_source(DatabaseTarget(database="aumc", source_id="two"))
    assert selected == rows[1]
    identity = replace(identity, version="changed")
    assert select_source(DatabaseTarget(database="aumc", source_id="two"))[1] != digest
    with pytest.raises(MultiDatabaseError, match="source_unavailable"):
        select_source(DatabaseTarget(database="hirid"))


@pytest.mark.parametrize("database,registration", [("sic", "one"), ("aumc", "neighbour")])
def test_registration_identity_must_match_the_selected_row(monkeypatch, tmp_path, database, registration):
    from easyicu.webserver import sources
    from easyicu.research_agent.research_context.export_selection import SourceIdentity

    row = dict(id="one", database="aumc", path=str(tmp_path / "synthetic"), ok=True)
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [row]})
    monkeypatch.setattr(sources, "registered_source_identity", lambda _: SourceIdentity(database=database, registration_id=registration))
    with pytest.raises(MultiDatabaseError, match="multi_database_source_changed"):
        select_source(DatabaseTarget(database="aumc", source_id="one"))
