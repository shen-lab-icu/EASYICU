"""First-run folder setup uses the server's isolated, initially absent root."""

from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import guided_sessions, state_paths
from easyicu.webserver.app import app


@pytest.fixture
def isolated_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "real-home"))
    root = state_paths.projects_root()
    monkeypatch.setattr(guided_sessions, "_PROJECTS_ROOT", root)
    monkeypatch.setattr(guided_sessions, "_CONFIG_DIR", tmp_path / "registry")
    monkeypatch.setattr(
        guided_sessions, "_CONFIG_PATH", tmp_path / "registry" / "guided.json"
    )
    return root


def test_guided_list_exposes_server_root_without_creating_it(isolated_projects):
    result = TestClient(app).post("/api/guided/drafts/list", json={}).json()

    assert result["ok"] is True
    assert result["default_parent_dir"] == str(isolated_projects.resolve())
    assert not isolated_projects.exists()


@pytest.mark.parametrize("explicit_parent", [True, False])
def test_guided_create_accepts_absent_server_root(isolated_projects, explicit_parent):
    body = {"title": "Synthetic first study", "parent_dir": ""}
    if explicit_parent:
        body["parent_dir"] = str(isolated_projects)

    result = guided_sessions.create_guided_draft(body)

    assert result["ok"] is True
    project = Path(result["draft"]["project_dir"])
    assert project.parent == isolated_projects
    assert (project / "guided_draft.json").is_file()


def test_guided_create_refuses_absent_real_home_parent(isolated_projects):
    real_home_parent = Path.home() / "easyicu" / "projects"
    assert real_home_parent != isolated_projects
    assert not real_home_parent.exists()

    result = guided_sessions.create_guided_draft(
        {"title": "Synthetic invalid parent", "parent_dir": "~/easyicu/projects"}
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_guided_parent_dir"
    assert not real_home_parent.exists()
    assert not isolated_projects.exists()


def test_folder_browse_does_not_create_absent_default(isolated_projects):
    result = TestClient(app).get(
        "/api/fs/list", params={"path": str(isolated_projects)}
    ).json()

    assert result["ok"] is False
    assert result["error"] == "not_a_directory"
    assert result["path"] == str(isolated_projects)
    assert result["entries"] == []
    assert not isolated_projects.exists()


def test_guided_folder_dialog_node_contract():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node, str(root / "tests/js/guided_folder_dialog.test.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
