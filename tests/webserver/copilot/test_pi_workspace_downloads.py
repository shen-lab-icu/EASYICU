"""Exact workspace attachments and explicit, isolated preview downloads."""

import hashlib

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.projections import _project_replay_resource
from easyicu.webserver.pi_copilot.tools import _workspace_resource
from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace
from easyicu.webserver.routes import pi_copilot as routes


def test_host_checked_file_digest_survives_replay_without_promoting_write_sha():
    written = _workspace_resource({"file": "demo.html", "sha256": "a" * 64})
    assert "checked_sha256" not in written
    checked = _workspace_resource({"file": "demo.html", "checked_sha256": "b" * 64})
    assert checked["kind"] == "file"
    assert _project_replay_resource(checked)["checked_sha256"] == "b" * 64
    assert "checked_sha256" not in _project_replay_resource(written)


@pytest.fixture
def workspace_client(tmp_path, monkeypatch):
    workspace = ProjectWorkspace(tmp_path / "workspace")

    class Service:
        get_workspace_download = PiCopilotService.get_workspace_download
        get_workspace_preview = PiCopilotService.get_workspace_preview

        def _assert_project_initialized(self, project_id):
            if project_id != "project-a":
                raise PiCopilotError("pi_project_not_initialized", "Not initialized", status_code=409)
            return project_id

    service = Service()
    service.workspace = workspace
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    return workspace, TestClient(app)


@pytest.mark.parametrize("filename,content", [
    ("合成演示.html", "<!doctype html><title>合成数据</title><p>仅用于演示</p>"),
    ("synthetic.csv", "synthetic_id,age\nSYN-001,42\n"),
])
def test_attachment_serves_exact_snapshot_without_executing_it(workspace_client, filename, content):
    workspace, client = workspace_client
    written = workspace.write_file("project-a", filename, content)
    response = client.get("/api/copilot/pi/projects/project-a/workspace/download", params={
        "file": filename, "expected_sha256": written["sha256"],
    })
    assert response.status_code == 200
    assert response.content == content.encode()
    assert response.headers["content-disposition"].startswith("attachment; filename*=UTF-8''")
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-security-policy"] == "sandbox; default-src 'none'"
    assert response.headers["x-easyicu-artifact-sha256"] == hashlib.sha256(response.content).hexdigest()
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_attachment_refuses_changed_bytes_and_uninitialized_projects(workspace_client):
    workspace, client = workspace_client
    written = workspace.write_file("project-a", "demo.html", "<p>first</p>")
    workspace.edit_file("project-a", "demo.html", old_text="first", new_text="second",
                        expected_sha256=written["sha256"])
    response = client.get("/api/copilot/pi/projects/project-a/workspace/download", params={
        "file": "demo.html", "expected_sha256": written["sha256"],
    })
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "pi_workspace_download_stale"
    response = client.get("/api/copilot/pi/projects/project-b/workspace/download", params={
        "file": "demo.html", "expected_sha256": written["sha256"],
    })
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "pi_project_not_initialized"


@pytest.mark.parametrize("filename", ["../outside.html", "/tmp/outside.html", "nested/../../outside.html"])
def test_attachment_reuses_workspace_path_boundary(workspace_client, filename):
    workspace, client = workspace_client
    with pytest.raises(PiCopilotError) as error:
        workspace.download_file("project-a", filename, expected_sha256="a" * 64)
    assert error.value.code == "pi_workspace_path_escape"
    assert client.get("/api/copilot/pi/projects/project-a/workspace/download", params={
        "file": "demo.html", "expected_sha256": "not-a-sha256",
    }).status_code == 422


def test_attachment_refuses_file_symlink_outside_project(workspace_client, tmp_path):
    workspace, _ = workspace_client
    outside = tmp_path / "private.html"
    outside.write_text("private")
    root = workspace.project_root("project-a")
    (root / "alias.html").symlink_to(outside)
    with pytest.raises(PiCopilotError) as error:
        workspace.download_file("project-a", "alias.html", expected_sha256=hashlib.sha256(b"private").hexdigest())
    assert error.value.code == "pi_workspace_symlink_blocked"


@pytest.mark.parametrize("allow_downloads", [False, True])
def test_preview_download_opt_in_keeps_network_and_origin_isolation(workspace_client, allow_downloads):
    workspace, client = workspace_client
    written = workspace.write_file("project-a", "demo.html", "<p>synthetic</p><script>void 0</script>")
    params = {"file": "demo.html", "checked_sha256": written["sha256"]}
    if allow_downloads:
        params["allow_downloads"] = "true"
    response = client.get("/api/copilot/pi/projects/project-a/workspace/preview", params=params)
    assert response.status_code == 200
    sandbox = "allow-scripts allow-downloads" if allow_downloads else "allow-scripts"
    assert f'sandbox="{sandbox}"' in response.text
    assert response.headers["content-security-policy"].startswith(f"sandbox {sandbox};")
    policy = response.headers["content-security-policy"]
    assert "connect-src 'none'" in policy and "default-src 'none'" in policy
    assert "form-action 'none'" in policy and "base-uri 'none'" in policy
    for forbidden in ["allow-same-origin", "allow-popups", "allow-forms", "allow-top-navigation"]:
        assert forbidden not in response.text and forbidden not in policy
    assert "<script>void 0</script>" not in response.text
    workspace.edit_file("project-a", "demo.html", old_text="synthetic", new_text="changed",
                        expected_sha256=written["sha256"])
    stale = client.get("/api/copilot/pi/projects/project-a/workspace/preview", params=params)
    assert stale.status_code == 409

