"""A run retains the exact source registration; fixtures contain no rows."""

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.webserver import agent_pipeline_runs, demo_source_prepare, demo_source_storage, sources
from easyicu.webserver import research_pipeline_run_preparation as preparation
from easyicu.webserver.demo_source_contracts import get_source
from easyicu.webserver.demo_source_storage import source_paths, write_marker
from tests.support.claim_lineage import project, source_store


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "sources.json")
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda: [])
    monkeypatch.setattr(sources, "_preferred_configured_export", lambda *a, **k: None)
    monkeypatch.setattr(demo_source_storage, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(sources.dataio, "describe_export_source", lambda path: {
        "ok": True, "database": "eicu_demo", "files": [],
    })
    monkeypatch.setattr(preparation, "_metadata_only_planning_coordinates", lambda **k: {})
    monkeypatch.setattr(preparation, "_authorize_launch_provider", lambda request: None)
    monkeypatch.setattr(preparation, "_prepare_launch_execution", lambda *a: (None, None))
    return tmp_path


def _register_demo(root, digest):
    official = get_source("eicu_demo_v2_0_1")
    paths = source_paths(official)
    write_marker(paths.prepared_marker, official, archive_sha256=digest)
    demo_source_prepare.register_export(official, paths, SimpleNamespace(emit=lambda event: None))
    return paths.export


def _launch(export):
    request = preparation.ResearchPipelineLaunchRequest(
        export_path=str(export), study_context={
            "id": "synthetic", "question": "Describe the available measurements.",
            "data_source": {"database": "eicu_demo", "path": str(export),
                            "source_id": "untrusted-study-label"},
        }, project_root=None, provider={}, provider_environment={}, credential_source="pi_verified",
        literature_search_authorized=False, plan_revision_source_run_id="", execution_resume_source_run_id="",
        development_resume_source_job_id="", budget_mode="planner_canary", runner_image=None,
    )
    return preparation.prepare_research_pipeline_run(request)


@pytest.mark.parametrize("kind,expected_match", [("official", True), ("changed", False), ("user", None)])
def test_registered_source_reaches_the_runs_claim_lineage(registry, kind, expected_match):
    official = get_source("eicu_demo_v2_0_1")
    if kind == "user":
        export = registry / official.id  # Neither this path nor label confers an official identity.
        sources.register_source(str(export), label=f"{official.title} v{official.version}")
    else:
        export = _register_demo(registry, official.archive_sha256 if kind == "official" else "c" * 64)
    row = next(row for row in sources.load_registry()["sources"] if row["path"] == str(export))
    prepared = _launch(export)
    scientific = prepared.scientific
    expected = {"database": official.database, "registration_id": row["id"]}
    if kind != "user":
        expected.update(version=official.version, source_id=official.id,
                        archive_sha256=official.archive_sha256 if kind == "official" else "c" * 64)
    assert prepared.source_identity.to_dict() == expected

    # A different active source and later registry edits cannot change the frozen launch.
    sources.register_source(str(registry / "other"), label="Other export")
    sources._CONFIG_PATH.write_text('{}', encoding="utf-8")
    declarations = agent_pipeline_runs.research_context_declarations(
        scientific, export_path=str(export), source_identity=prepared.source_identity.to_dict(),
    )
    context = build_research_context(
        research_question=scientific.question, cohort=pd.DataFrame({"measurement": pd.Series(dtype=float)}),
        database=official.database, **declarations,
    )
    assert context.cohort.provenance["source_identity"] == expected
    run_root = registry / "synthetic-run"
    run_root.mkdir()

    def change(recorded_context, fingerprint, summary):
        recorded_context["cohort"]["provenance"]["source_identity"] = context.cohort.provenance["source_identity"]

    node = project(source_store(run_root, change=change))["lineage"][-1]
    assert node["status"] == "current"
    assert node["facts"]["registration_id"] == row["id"]
    if expected_match is None:
        assert "source_id" not in node["facts"]
        assert "official_sha256_matches" not in node["facts"]
    else:
        assert node["facts"]["official_sha256_matches"] is expected_match
    if expected_match is not True:
        assert "doi" not in node["facts"] and "physionet_slug" not in node["facts"]
    assert str(registry) not in json.dumps(node)


def test_registry_edits_preserve_the_preparation_record(registry):
    official = get_source("eicu_demo_v2_0_1")
    export = _register_demo(registry, "c" * 64)
    other = registry / "other"
    sources.register_source(str(other))
    sources.rename_source(str(export), "Renamed source")
    sources.rename_source(str(other), "Renamed other")
    sources.remove_source(str(other))
    sources.save_registry({"sources": [{"path": str(export), "label": "Edited source",
                                       "source_id": "forged", "archive_sha256": "d" * 64}]})
    identity = _launch(export).source_identity.to_dict()
    assert identity["source_id"] == official.id
    assert identity["archive_sha256"] == "c" * 64
    assert identity["version"] == official.version


def test_missing_archive_receipt_does_not_borrow_the_catalog_digest(registry):
    export = _register_demo(registry, None)
    identity = _launch(export).source_identity.to_dict()
    assert "archive_sha256" not in identity
    assert identity["source_id"] == "eicu_demo_v2_0_1"


def test_unregistered_export_does_not_borrow_an_active_registration(registry):
    official = get_source("eicu_demo_v2_0_1")
    _register_demo(registry, official.archive_sha256)
    scientific = _launch(registry / "unregistered")
    assert not scientific.source_identity


@pytest.mark.parametrize("bound", [True, False])
def test_extraction_carries_only_its_bound_registration_receipt(registry, monkeypatch, bound):
    from easyicu.webserver.routes import jobs

    official = get_source("eicu_demo_v2_0_1")
    export = _register_demo(registry, "c" * 64)
    raw, derived = registry / "raw", registry / "derived"
    study = {"id": "synthetic", "revision": 1,
             "data_source": {"path": str(export), "database": official.database}}
    monkeypatch.setattr(jobs.settings_store, "load_settings", lambda: {})
    monkeypatch.setattr(jobs.context_store, "get_context", lambda _: study)
    monkeypatch.setattr(jobs.context_store, "handoff_context", lambda *a, **k: study)
    monkeypatch.setattr(jobs.context_store, "clear_active_job_if", lambda *a, **k: {"context": study})
    monkeypatch.setattr(jobs, "_study_binding_for_extraction", lambda *a: None)
    monkeypatch.setattr(jobs.dataio, "resolve_registered_export_binding",
                        lambda *a: {"source_data_path": str(raw)})
    # Metadata-only fake completion: no extraction, tables, worker or server is started.
    monkeypatch.setattr(jobs.dataio, "make_export_runner", lambda **k: lambda job: {
        "out_dir": str(derived), "manifest": "synthetic-manifest.json", "files": [],
    })
    captured = []
    job = SimpleNamespace(id="synthetic", kind="extract", status="pending", cancel_requested=False)
    monkeypatch.setattr(jobs, "submit_job", lambda kind, runner: captured.append(runner) or job)
    jobs.jobs_extract({
        "path": str(raw), "database": official.database,
        "registered_export_path": str(export), "study_context_id": "synthetic" if bound else "",
        "source_identity": {"source_id": official.id, "archive_sha256": official.archive_sha256},
    })
    # Re-registering the parent as a user export must not change the queued extraction's snapshot.
    sources.register_source(str(export))
    captured[0](job)
    identity = _launch(derived).source_identity.to_dict()
    assert identity["registration_id"] != _launch(export).source_identity.registration_id
    if bound:
        assert identity["source_id"] == official.id
        assert identity["archive_sha256"] == "c" * 64
    else:
        assert "source_id" not in identity and "archive_sha256" not in identity


def test_registering_a_user_export_at_a_demo_path_revokes_the_old_release(registry):
    export = _register_demo(registry, "c" * 64)
    sources.register_source(str(export))
    identity = _launch(export).source_identity.to_dict()
    assert set(identity) == {"database", "registration_id"}


def test_launch_snapshot_is_immutable_and_only_contains_recorded_coordinates(registry):
    export = registry / "user-export"
    sources.register_source(str(export), source_identity={
        "version": "synthetic-v1", "archive_sha256": "d" * 64,
        "path": "/private/ignored", "credential": "ignored", "registration_id": "forged",
    })
    identity = _launch(export).source_identity
    assert identity.to_dict() == {
        "database": "eicu_demo", "registration_id": sources.load_registry()["sources"][0]["id"],
        "version": "synthetic-v1", "archive_sha256": "d" * 64,
    }
    with pytest.raises(FrozenInstanceError):
        identity.version = "changed"


@pytest.mark.parametrize("constraints", [None, "broken", "[]", '{"source_identity": []}',
                                         '{"source_identity": {"database": "eicu_demo"}}'])
def test_context_does_not_invent_an_absent_registration(constraints):
    from easyicu.research_agent.research_context.export_selection import source_identity_provenance

    assert source_identity_provenance({"data_constraints": constraints}) == {}


def test_source_receipt_drops_paths_and_malformed_release_coordinates():
    from easyicu.research_agent.research_context.export_selection import SourceIdentity

    record = {"database": "synthetic_db", "registration_id": "synthetic_id",
              "source_id": "/private/source", "version": "https://private/version",
              "archive_sha256": "not-a-sha", "path": "/private/path"}
    assert SourceIdentity.from_record(record).to_dict() == {
        "database": "synthetic_db", "registration_id": "synthetic_id",
    }
    assert SourceIdentity.from_record({**record, "registration_id": "/private/id"}) is None


@pytest.mark.parametrize("naive", [False, True])
def test_both_context_builders_keep_the_typed_preferences_snapshot(naive):
    from easyicu.research_agent.research_context.builder import build_naive_research_context
    from easyicu.research_agent.schema import UserPreferences

    identity = {"database": "eicu_demo", "registration_id": "synthetic_registration"}
    builder = build_naive_research_context if naive else build_research_context
    context = builder(
        research_question="Describe the available measurements.", cohort_name="synthetic",
        cohort=pd.DataFrame({"measurement": pd.Series(dtype=float)}), database="eicu_demo",
        user_preferences=UserPreferences(data_constraints=json.dumps({"source_identity": identity})),
    )
    assert context.cohort.provenance["source_identity"] == identity


def test_trial_context_records_the_same_selected_registration(registry, monkeypatch):
    from easyicu.research_agent.research_context import builder

    export = _register_demo(registry, "c" * 64)
    prepared = _launch(export)
    monkeypatch.setattr(agent_pipeline_runs, "target_trial_run_coordinates", lambda *a: {
        "target_outcome": None, "primary_exposure": None, "outcome_columns": (),
    })
    monkeypatch.setattr(builder, "build_research_context", lambda **kwargs: kwargs)
    context_args = agent_pipeline_runs.target_trial_research_context(
        prepared.scientific, SimpleNamespace(universe_path=registry / "synthetic-cohort.parquet"),
        export_path=str(export), endpoint=None,
    )
    constraints = json.loads(context_args["user_preferences"]["data_constraints"])
    assert constraints["source_identity"] == prepared.source_identity.to_dict()

