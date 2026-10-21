"""Hosted requests must not turn indirect coordinates into unbounded reads."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from easyicu.webserver import agent_runs, crossdb_review, guided_sessions, run_refs, settings
from easyicu.webserver.app import app
from easyicu.webserver.ideas import mining
from easyicu.webserver.routes import agent as agent_routes
from easyicu.webserver.routes import jobs as job_routes
from easyicu.webserver.routes import pi_copilot as pi_routes
from easyicu.webserver.pi_copilot.session_storage import SessionStorageMaintenance
from easyicu.webserver.run_record import RunDirectory


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    root, home, outside = (tmp_path / name for name in ("data", "home", "outside"))
    for path in (root, home, outside):
        path.mkdir()
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(root))
    monkeypatch.setenv("EASYICU_HOME", str(home))
    monkeypatch.setattr(settings, "_CONFIG_DIR", home)
    monkeypatch.setattr(settings, "_CONFIG_PATH", home / "settings.json")
    monkeypatch.setattr(guided_sessions, "_CONFIG_PATH", home / "guided.json")
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", home / "projects")
    monkeypatch.setattr(mining, "_RUN_ROOT", home / "ideas")
    monkeypatch.setattr(mining, "_AGENT_PROJECTS_PATH", home / "idea-projects.json")
    return root, home, outside


def refused(response):
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["error"] == "path_outside_data_roots"


@pytest.mark.parametrize("route", ["/api/crossdb-review/raw-distribution", "/api/jobs/crossdb-raw-distribution"])
def test_hosted_raw_request_requires_an_explicit_root(boundary, monkeypatch, route):
    _, _, outside = boundary
    monkeypatch.setenv("EASYICU_DATA_PATH", str(outside))
    scans, submissions = [], []
    monkeypatch.setattr(crossdb_review, "_detect_raw_databases", lambda p: scans.append(p) or [])
    monkeypatch.setattr(job_routes, "submit_job", lambda *a: submissions.append(a))
    response = TestClient(app, raise_server_exceptions=False).post(route, json={})
    refused(response)
    assert scans == submissions == []


def test_desktop_keeps_raw_root_discovery(boundary, monkeypatch):
    _, _, outside = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "0")
    monkeypatch.setenv("EASYICU_DATA_PATH", str(outside))
    monkeypatch.setattr(crossdb_review, "_detect_raw_databases", lambda p: ["eicu", "miiv"])
    assert crossdb_review._resolve_raw_data_root({}) == outside


def test_hosted_raw_job_reports_a_missing_permitted_root_as_invalid(boundary, monkeypatch):
    root, _, _ = boundary
    submissions = []
    monkeypatch.setattr(job_routes, "submit_job", lambda *a: submissions.append(a) or SimpleNamespace(id="synthetic", kind="crossdb", status="running"))
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/jobs/crossdb-raw-distribution", json={"data_root": str(root / "missing")},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["error"] == "raw_data_root_not_found"
    assert submissions == []


@pytest.mark.parametrize("owner", [crossdb_review.crossdb_raw_root_scan, crossdb_review._resolve_raw_data_root])
@pytest.mark.parametrize("field", ["root", "data_root"])
def test_raw_owners_check_their_input_before_discovery(boundary, monkeypatch, owner, field):
    _, _, outside = boundary
    calls = []
    monkeypatch.setattr(crossdb_review, "_normalize_raw_root", lambda p: calls.append(p) or p)
    with pytest.raises(HTTPException) as error:
        owner({field: str(outside)})
    assert error.value.status_code == 403
    assert error.value.detail["error"] == "path_outside_data_roots"
    assert calls == []
    monkeypatch.setenv("EASYICU_HOSTED", "0")
    owner({field: str(outside)})
    assert calls == [outside]


@pytest.mark.parametrize("hosted", [True, False])
def test_raw_normalization_does_not_scan_an_unpermitted_parent(boundary, monkeypatch, hosted):
    root, _, _ = boundary
    selected = root / "eicu"
    selected.mkdir()
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(selected))
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    scans = []
    monkeypatch.setattr(crossdb_review, "_detect_raw_databases", lambda p: scans.append(p) or ["eicu", "miiv"])
    assert crossdb_review._normalize_raw_root(selected) == (selected if hosted else root)
    assert scans == ([] if hosted else [root])


@pytest.mark.parametrize("hosted", [True, False])
def test_resolved_database_is_checked_before_use(boundary, monkeypatch, hosted):
    root, _, outside = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    monkeypatch.setattr(crossdb_review, "find_database_path", lambda *a: str(outside))
    if hosted:
        with pytest.raises(HTTPException) as error:
            crossdb_review._raw_database_resolved_path(root, "eicu")
        assert error.value.detail["error"] == "path_outside_data_roots"
    else:
        assert crossdb_review._raw_database_resolved_path(root, "eicu") == outside


@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("hosted", [True, False])
def test_raw_routes_check_derived_databases_before_loading_or_submitting(boundary, monkeypatch, background, hosted):
    root, _, outside = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    monkeypatch.setattr(crossdb_review, "find_database_path", lambda *a: str(outside))
    reached = []

    def load(**kwargs):
        reached.append("loader")
        raise crossdb_review.CrossdbReviewError({"error": "synthetic_loader"})

    def submit(*args):
        reached.append("submit")
        return SimpleNamespace(id="synthetic", kind="crossdb", status="running")

    monkeypatch.setattr(crossdb_review, "_load_raw_feature_data", load)
    monkeypatch.setattr(job_routes, "submit_job", submit)
    route = "/api/jobs/crossdb-raw-distribution" if background else "/api/crossdb-review/raw-distribution"
    response = TestClient(app).post(route, json={"data_root": str(root), "databases": ["miiv", "eicu"]})
    if hosted:
        refused(response)
        assert reached == []
    else:
        assert response.status_code == (200 if background else 400)
        assert reached == ["submit" if background else "loader"]


@pytest.mark.parametrize("hosted", [True, False])
def test_raw_alias_link_is_refused_before_resolver_scans_target(boundary, monkeypatch, hosted):
    root, _, outside = boundary
    (root / "eicu").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    listings = []
    original = os.listdir

    def listed(path):
        if isinstance(path, (str, Path)) and Path(path).resolve() == outside:
            listings.append(path)
        return original(path)

    monkeypatch.setattr(os, "listdir", listed)
    response = TestClient(app).post("/api/crossdb-review/raw-root-scan", json={"root": str(root)})
    if hosted:
        refused(response)
        assert listings == []
    else:
        assert response.status_code == 200
        assert listings


@pytest.mark.parametrize("owner", ["_normalize_raw_root", "_raw_database_resolved_path"])
def test_raw_directory_helpers_refuse_before_scanning(boundary, monkeypatch, owner):
    _, _, outside = boundary
    calls = []
    monkeypatch.setattr(crossdb_review, "find_database_path", lambda *a: calls.append(a) or str(outside))
    monkeypatch.setattr(crossdb_review, "_detect_raw_databases", lambda *a: calls.append(a) or [])
    args = (outside, "eicu") if owner == "_raw_database_resolved_path" else (outside,)
    with pytest.raises(HTTPException) as error:
        getattr(crossdb_review, owner)(*args)
    assert error.value.detail["error"] == "path_outside_data_roots"
    assert calls == []


def test_instance_owned_run_directories_remain_available(boundary):
    root, home, _ = boundary
    for directory in (root, home):
        assert RunDirectory.resolve(directory).path == directory
        assert RunDirectory.create(directory, "synthetic", "job").path == directory / "synthetic" / "run_job"
        assert agent_runs.list_run_history(project_root=str(directory))["runs"] == []


@pytest.mark.parametrize("route", ["review", "signoff", "artifact", "download-artifact", "download-bundle"])
@pytest.mark.parametrize("hosted", [True, False])
def test_opaque_run_ref_rechecks_the_resolved_directory(boundary, monkeypatch, route, hosted):
    _, _, outside = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    monkeypatch.setattr(run_refs, "resolve", lambda ref: str(outside))
    reads = []
    monkeypatch.setattr(agent_runs, "_read_safe_artifact_bytes", lambda p, name: reads.append(p) or (None, None, "synthetic"))
    response = TestClient(app).post(f"/api/agent-runs/{route}", json={"run_ref": "opaque", "artifact": "quality_gate.json"})
    if hosted:
        refused(response)
        assert reads == []
    else:
        assert response.status_code == 400
        assert reads and set(reads) == {outside}


@pytest.mark.parametrize("level", ["root", "study", "run"])
@pytest.mark.parametrize("hosted", [True, False])
def test_run_history_rechecks_each_directory_before_listing(boundary, monkeypatch, level, hosted):
    _, home, outside = boundary
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    root = home / "history"
    study = root / "synthetic"
    study.mkdir(parents=True)
    if level == "root":
        root = outside
    elif level == "study":
        (root / "escape").symlink_to(outside, target_is_directory=True)
    else:
        (study / "run_escape").symlink_to(outside, target_is_directory=True)
    listings = []
    reads = []
    original = Path.iterdir
    monkeypatch.setattr(agent_runs, "_load_run_artifacts", lambda p: reads.append(p) or {"ok": False, "error": "synthetic"})

    def listed(path):
        if path.resolve() == outside:
            listings.append(path)
        return original(path)

    monkeypatch.setattr(Path, "iterdir", listed)
    if hosted:
        with pytest.raises(HTTPException) as error:
            agent_runs.list_run_history(project_root=str(root))
        assert error.value.detail["error"] == "path_outside_data_roots"
        assert listings == reads == []
    else:
        assert agent_runs.list_run_history(project_root=str(root))["ok"]
        assert listings or reads


@pytest.mark.parametrize("hosted", [True, False])
def test_run_creation_checks_the_final_directory(boundary, monkeypatch, hosted):
    _, home, outside = boundary
    (home / "synthetic").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    if hosted:
        with pytest.raises(HTTPException) as error:
            RunDirectory.create(home, "synthetic", "job")
        assert error.value.detail["error"] == "path_outside_data_roots"
    else:
        assert RunDirectory.create(home, "synthetic", "job").path == home / "synthetic" / "run_job"


@pytest.mark.parametrize("hosted", [True, False])
def test_agent_seed_checks_the_child_file(boundary, monkeypatch, hosted):
    _, home, outside = boundary
    seed = outside / "seed.json"
    seed.write_text("{}")
    (home / "project_seed.json").symlink_to(seed)
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    reads = []
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.resolve() == seed:
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    if hosted:
        with pytest.raises(HTTPException) as error:
            agent_routes._validate_agent_project_seed_for_run(str(home), "synthetic")
        assert error.value.detail["error"] == "path_outside_data_roots"
        assert reads == []
    else:
        assert agent_routes._validate_agent_project_seed_for_run(str(home), "synthetic")["ok"]
        assert reads


@pytest.mark.parametrize("filename", ["guided_draft.json", "guided_copilot_session.json"])
@pytest.mark.parametrize("hosted", [True, False])
def test_guided_project_checks_the_file_it_opens(boundary, monkeypatch, filename, hosted):
    _, home, outside = boundary
    project = home / "projects" / "synthetic"
    project.mkdir(parents=True)
    target = outside / filename
    target.write_text("{}")
    (project / filename).symlink_to(target)
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    reads = []
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.resolve() == target:
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    response = TestClient(app).post("/api/guided/project/open", json={"project_dir": str(project)})
    if hosted:
        refused(response)
        assert reads == []
    else:
        assert response.status_code == 200
        assert reads


@pytest.mark.parametrize("hosted", [True, False])
def test_guided_trash_rechecks_the_saved_coordinate(boundary, monkeypatch, hosted):
    _, _, outside = boundary
    (outside / "guided_draft.json").write_text(json.dumps({"id": "synthetic"}))
    guided_sessions._CONFIG_PATH.write_text(json.dumps({"drafts": [{"id": "synthetic", "project_dir": str(outside)}]}))
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    moved = []
    monkeypatch.setattr(guided_sessions, "_send2trash", moved.append)
    response = TestClient(app).post("/api/guided/drafts/remove", json={"draft_id": "synthetic", "trash_confirmation": "synthetic", "trash_project_folder": True})
    if hosted:
        refused(response)
        assert moved == []
    else:
        assert response.status_code == 200
        assert moved == [str(outside)]


@pytest.mark.parametrize("kind", ["directory", "file"])
@pytest.mark.parametrize("hosted", [True, False])
def test_idea_run_id_cannot_read_outside_state(boundary, monkeypatch, kind, hosted):
    _, home, outside = boundary
    root = home / "ideas"
    root.mkdir()
    (outside / "idea_mining_run.json").write_text("{}")
    if kind == "directory":
        (root / "synthetic").symlink_to(outside, target_is_directory=True)
    else:
        (root / "synthetic").mkdir()
        (root / "synthetic" / "idea_mining_run.json").symlink_to(outside / "idea_mining_run.json")
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    reads = []
    listings = []
    original = Path.read_text
    original_iterdir = Path.iterdir

    def listed(path):
        if path.resolve().is_relative_to(outside):
            listings.append(path)
        return original_iterdir(path)

    def read(path, *args, **kwargs):
        if path.resolve().is_relative_to(outside):
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "iterdir", listed)
    response = TestClient(app).post("/api/ideas/run", json={"run_id": "synthetic"})
    if hosted:
        refused(response)
        assert reads == listings == []
    else:
        assert response.status_code == 400  # The synthetic empty record is absent.
        assert reads


@pytest.mark.parametrize("kind", ["directory", "file"])
@pytest.mark.parametrize("hosted", [True, False])
def test_quarantine_id_checks_the_manifest_before_restoring(boundary, monkeypatch, kind, hosted):
    _, home, outside = boundary
    owner = SessionStorageMaintenance(home / "sessions")
    owner.quarantine_root.mkdir(parents=True)
    batch = "20261021T000000Z-01234567"
    manifest = outside / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "easyicu.pi-session-quarantine/1", "quarantine_id": batch, "files": []}))
    selected = owner.quarantine_root / batch
    if kind == "directory":
        selected.symlink_to(outside, target_is_directory=True)
    else:
        selected.mkdir()
        (selected / "manifest.json").symlink_to(manifest)
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    service = SimpleNamespace(maintain_session_storage=lambda **kw: owner.restore(kw["quarantine_id"], confirm=kw["confirm"]))
    monkeypatch.setattr(pi_routes, "get_pi_copilot_service", lambda: service)
    reads = []
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.resolve() == manifest:
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    response = TestClient(app).post("/api/copilot/pi/session-maintenance", json={"action": "restore", "confirm": True, "quarantine_id": batch})
    if hosted:
        refused(response)
        assert reads == []
    else:
        assert response.status_code == 200
        assert response.json()["restored_files"] == 0
        assert reads
