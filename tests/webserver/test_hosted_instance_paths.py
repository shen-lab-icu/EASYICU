"""Hosted filesystem refusals, using empty synthetic directories only."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import capabilities, dataio, guided_sessions, settings, sources, state_paths
from easyicu.webserver.app import app
from tests.support.hosted_http import stub_route


# Each public coordinate must be checked before its owner touches the disk.
PATH_ROUTES = [
    ("GET", "/api/fs/list", "path"),
    ("POST", "/api/fs/mkdir", "path"),
    ("POST", "/api/data/scan", "path"),
    ("POST", "/api/workspace/summary", "path"),
    ("POST", "/api/jobs/convert", "path"),
    ("POST", "/api/jobs/extract", "path"),
    ("POST", "/api/jobs/extract", "out_dir"),
    ("POST", "/api/jobs/extract", "registered_export_path"),
    ("POST", "/api/workspaces/register", "path"),
    ("POST", "/api/workspaces/rename", "path"),
    ("POST", "/api/workspaces/remove", "path"),
    ("POST", "/api/workspaces/registry", "active_path"),
    ("POST", "/api/workspaces/registry", "crossdb_paths"),
    ("POST", "/api/workspaces/crossdb-summary", "paths"),
    ("POST", "/api/jobs/crossdb-summary", "paths"),
    ("POST", "/api/jobs/crossdb-summary", "source_paths"),
    ("POST", "/api/jobs/crossdb-raw-distribution", "data_root"),
    ("POST", "/api/crossdb-review/raw-distribution", "root"),
    ("POST", "/api/crossdb-review/raw-root-scan", "data_root"),
    ("POST", "/api/crossdb-review/summary", "paths"),
    ("POST", "/api/patient-review/drilldown", "source_path"),
    ("POST", "/api/patient-review/entities", "source_path"),
    ("POST", "/api/patient-review/entity", "source_path"),
    ("POST", "/api/patient-review/table-preview", "source_path"),
    ("POST", "/api/patient-review/feature", "source_path"),
    ("POST", "/api/patient-review/sources", "path"),
    ("POST", "/api/cohort-review/summary", "source_path"),
    ("POST", "/api/extraction/filter-options", "path"),
    ("POST", "/api/extraction/filter-preview", "path"),
    ("POST", "/api/guided/drafts", "parent_dir"),
    ("POST", "/api/guided/project/open", "project_dir"),
    ("POST", "/api/guided/session", "project_dir"),
    ("POST", "/api/agent-runs/review", "project_dir"),
    ("POST", "/api/agent-runs/science-workbench", "project_dir"),
    ("POST", "/api/agent-runs/signoff", "project_dir"),
    ("POST", "/api/agent-runs/artifact", "project_dir"),
    ("POST", "/api/agent-runs/download-artifact", "project_dir"),
    ("POST", "/api/agent-runs/download-bundle", "project_dir"),
    ("POST", "/api/agent-runs/history", "project_root"),
    ("POST", "/api/jobs/agent-run", "project_root"),
    ("POST", "/api/jobs/agent-run", "project_seed_dir"),
    ("POST", "/api/jobs/agent-run", "project_seed_path"),
    ("POST", "/api/study-contexts", "data_source"),
]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "instance"
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("EASYICU_HOME", str(home))
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(root))
    monkeypatch.setattr(settings, "_CONFIG_DIR", home)
    monkeypatch.setattr(settings, "_CONFIG_PATH", home / "settings.json")
    monkeypatch.setattr(sources, "_CONFIG_DIR", home)
    monkeypatch.setattr(sources, "_CONFIG_PATH", home / "sources.json")
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", home)
    monkeypatch.setattr(guided_sessions, "_CONFIG_PATH", home / "guided.json")
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", state_paths.projects_root())
    return root, home


@pytest.mark.parametrize("method,route,field", PATH_ROUTES)
@pytest.mark.parametrize("hosted", [True, False])
def test_path_routes_refuse_before_io_only_when_hosted(isolated, monkeypatch, method, route, field, hosted):
    root, _ = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    calls = stub_route(monkeypatch, route)
    outside = str(root.parent / "another-user")
    value = [outside] if field.endswith("paths") else outside
    if field == "data_source":
        value = {"path": outside}
    arguments = {"params" if method == "GET" else "json": {field: value}}
    response = TestClient(app).request(method, route, **arguments)
    if hosted:
        assert response.status_code == 403, response.text
        assert response.json()["detail"]["error"] == "path_outside_data_roots"
        assert not calls
    else:
        assert response.status_code == 200, response.text
        assert len(calls) == 1


@pytest.mark.parametrize("roots", ["", "relative", ":", "relative:valid", "/"])
def test_missing_or_invalid_roots_fail_closed(isolated, monkeypatch, roots):
    root, _ = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_DATA_ROOTS", roots.replace("valid", str(root)))
    calls = stub_route(monkeypatch, "/api/data/scan")
    response = TestClient(app).post("/api/data/scan", json={"path": str(root)})
    assert response.status_code == 403
    assert not calls


@pytest.mark.parametrize("kind", ["symlink", "dotdot", "prefix", "relative"])
def test_path_resolution_cannot_escape_roots(isolated, monkeypatch, kind):
    root, _ = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    outside = root.parent / "data-other"
    outside.mkdir()
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    candidate = {"symlink": link / "new", "dotdot": root / ".." / outside.name,
                 "prefix": outside, "relative": Path("relative")}[kind]
    if kind == "relative":
        monkeypatch.chdir(root)
    response = TestClient(app).post("/api/fs/mkdir", json={"path": str(candidate)})
    assert response.status_code == 403
    assert not (outside / "new").exists()


def test_picker_starts_at_root_hides_parent_shortcuts_and_escaping_links(isolated, monkeypatch):
    root, _ = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    (root / "allowed").mkdir()
    (root / "escape").symlink_to(root.parent, target_is_directory=True)
    probed = []
    monkeypatch.setattr(dataio, "_has_glob", lambda path, pattern: probed.append(path) or False)
    client = TestClient(app)
    result = client.get("/api/fs/list").json()
    assert result["path"] == str(root)
    assert result["parent"] is None
    assert [entry["name"] for entry in result["entries"]] == ["allowed"]
    assert all(path.resolve().is_relative_to(root) for path in probed)
    assert all(Path(entry["path"]).is_relative_to(root) for entry in result["shortcuts"])
    assert client.get("/api/fs/list", params={"path": "/"}).status_code == 403
    assert client.get("/api/fs/list", params={"path": str(root.parent)}).status_code == 403
    created = client.post("/api/fs/mkdir", json={"path": str(root / "new")})
    assert created.status_code == 200 and (root / "new").is_dir()


@pytest.mark.parametrize("hosted", [True, False])
def test_default_project_is_instance_owned_and_keeps_chinese_slug(isolated, monkeypatch, hosted):
    root, home = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    response = TestClient(app).post("/api/guided/drafts", json={"title": "我的研究", "parent_dir": ""})
    assert response.status_code == 200, response.text
    assert response.json()["draft"]["data_mode"] == ("real" if hosted else "demo")
    project = Path(response.json()["draft"]["project_dir"])
    assert project.parent == state_paths.projects_root()
    assert project.is_relative_to(home) and not project.is_relative_to(root)
    assert "我的研究" in project.name
    assert (project / "guided_draft.json").is_file()


def test_instance_run_coordinate_remains_readable_without_data_roots(isolated, monkeypatch):
    _, home = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.delenv("EASYICU_DATA_ROOTS")
    calls = stub_route(monkeypatch, "/api/agent-runs/review")
    response = TestClient(app).post("/api/agent-runs/review", json={"project_dir": str(home / "run")})
    assert response.status_code == 200
    assert len(calls) == 1


def test_registry_does_not_probe_stale_outside_paths(isolated, monkeypatch):
    root, home = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    home.mkdir()
    sources._CONFIG_PATH.write_text('{"sources":[{"path":"/another-user/export"}]}')
    probed = []
    monkeypatch.setattr(dataio, "describe_export_source", lambda path: probed.append(path) or {"ok": False})
    assert sources.load_registry()["sources"] == []
    assert all(Path(path).is_relative_to(root) or Path(path).is_relative_to(home) for path in probed)


def _server_paths(value, roots):
    if isinstance(value, dict):
        return [path for item in value.values() for path in _server_paths(item, roots)]
    if isinstance(value, list):
        return [path for item in value for path in _server_paths(item, roots)]
    if isinstance(value, str) and value.startswith("/"):
        if any(Path(value).is_relative_to(root) for root in roots):
            return [value]
    return []


@pytest.mark.parametrize("route", ["/api/settings", "/api/capabilities", "/api/workspaces/registry"])
@pytest.mark.parametrize("hosted", [True, False])
def test_diagnostic_responses_hide_instance_paths_only_when_hosted(isolated, monkeypatch, route, hosted):
    from easyicu.webserver.pi_copilot.provider_config import PiProviderConfigStore

    root, home = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", home / "capability_tool_audit.jsonl")
    monkeypatch.setattr(PiProviderConfigStore, "public_status", lambda self: {"connection_verified": False})
    settings.update_settings({"export_dir": str(home / "exports")})
    response = TestClient(app).get(route)
    assert response.status_code == 200
    paths = _server_paths(response.json(), (home, root))
    assert bool(paths) is not hosted


@pytest.mark.parametrize("hosted", [True, False])
def test_audit_responses_keep_paths_on_desktop_only(isolated, monkeypatch, hosted):
    root, home = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    monkeypatch.setattr(capabilities, "_STATE_DIR", home)
    monkeypatch.setattr(capabilities, "_AUDIT_PATH", home / "audit.jsonl")
    result = capabilities.record_tool_event("synthetic", {"count": 1})
    assert result["recorded"] is True
    assert bool(_server_paths(result, (home, root))) is not hosted
    response = TestClient(app).post("/api/capabilities/audit-events", json={})
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert bool(_server_paths(response.json(), (home, root))) is not hosted
    capabilities._AUDIT_PATH.unlink()
    capabilities._AUDIT_PATH.mkdir()
    failure = capabilities.record_tool_event("synthetic", {})
    assert failure["reason"] == "tool_audit_write_failed"
    assert bool(_server_paths(failure, (home, root))) is not hosted


def test_download_rechecks_id_resolved_path_before_manifest_read(isolated, monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [
        {"id": "src_0123456789ab", "ok": True, "path": "/another-user/export"},
    ]})
    response = TestClient(app).post("/api/workspaces/download", json={"source_id": "src_0123456789ab"})
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "path_outside_data_roots"


def test_picker_without_roots_does_not_fall_back_to_home(isolated, monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.delenv("EASYICU_DATA_ROOTS")
    assert TestClient(app).get("/api/fs/list").status_code == 403


def test_second_configured_root_is_available(isolated, monkeypatch):
    import os

    root, _ = isolated
    second = root.parent / "second"
    second.mkdir()
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_DATA_ROOTS", os.pathsep.join([str(root), str(second)]))
    response = TestClient(app).get("/api/fs/list", params={"path": str(second)})
    assert response.status_code == 200 and response.json()["parent"] is None


@pytest.mark.parametrize("reader", ["summarize_export_workspace", "describe_export_source"])
def test_export_readers_reject_manifest_symlink_escape(isolated, monkeypatch, reader):
    root, _ = isolated
    outside = root.parent / "outside-manifest.json"
    outside.write_text("{}")
    (root / "_manifest.json").symlink_to(outside)
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    from fastapi import HTTPException

    original_read = Path.read_text

    def read(path, *args, **kwargs):
        assert path != root / "_manifest.json", "Escaping manifest was read before refusal"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(HTTPException) as error:
        getattr(dataio, reader)(str(root))
    assert error.value.status_code == 403
    assert error.value.detail["error"] == "path_outside_data_roots"


def test_own_generated_export_can_be_addressed_outside_shared_roots(isolated, monkeypatch):
    _, home = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    stub_route(monkeypatch, "/api/workspace/summary")
    response = TestClient(app).post("/api/workspace/summary", json={"path": str(home / ".easyicu/exports/synthetic")})
    assert response.status_code == 200


def test_saved_project_coordinate_cannot_escape_through_notes(isolated, monkeypatch):
    import json

    _, home = isolated
    home.mkdir()
    guided_sessions._CONFIG_PATH.write_text(json.dumps({"drafts": [
        {"id": "synthetic", "project_dir": str(home.parent / "another-user")},
    ]}))
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    response = TestClient(app).get("/api/guided/projects/synthetic/notes")
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "path_outside_data_roots"


def test_project_notes_symlink_cannot_read_another_user(isolated, monkeypatch):
    root, _ = isolated
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    client = TestClient(app)
    created = client.post("/api/guided/drafts", json={"title": "我的研究"}).json()["draft"]
    project = Path(created["project_dir"])
    outside = root.parent / "other-notes.md"
    outside.write_text("Synthetic private notes")
    (project / "project_notes.md").symlink_to(outside)
    assert client.get(f"/api/guided/projects/{created['id']}/notes").status_code == 403


@pytest.mark.parametrize("hosted", [True, False])
def test_new_guided_session_inherits_the_deployment_data_mode(isolated, monkeypatch, hosted):
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    response = TestClient(app).post("/api/guided/session", json={"title": "Synthetic session"})
    assert response.status_code == 200
    assert response.json()["session"]["context"]["data_mode"] == ("real" if hosted else "demo")
