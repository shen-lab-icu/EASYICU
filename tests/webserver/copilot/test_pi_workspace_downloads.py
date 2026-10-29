"""Exact workspace attachments and explicit, isolated preview downloads."""

import asyncio
import base64
import hashlib
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from starlette.requests import Request

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.projections import _project_replay_resource
from easyicu.webserver.pi_copilot.tools import _workspace_resource
from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace
from easyicu.webserver.pi_copilot.workspace_csv_download import MAX_CSV_BYTES, MAX_FORM_BYTES, read_csv_download_form
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


def _csv_form(digest, **overrides):
    fields = {"file": "demo.html", "expected_sha256": digest,
              "filename": "synthetic.csv", "media_type": "text/csv",
              "csv_base64": base64.b64encode("id,标记\nSYN-001,演示\n".encode()).decode()}
    return {**fields, **overrides}


def test_csv_bridge_attachment_preserves_exact_utf8_and_reads_no_requested_csv_file(workspace_client):
    workspace, client = workspace_client
    written = workspace.write_file("project-a", "demo.html", "<!doctype html><p>synthetic</p>")
    before = workspace.list_files("project-a")
    response = client.post("/api/copilot/pi/projects/project-a/workspace/download-csv",
                           data=_csv_form(written["sha256"]), headers={"Origin": "http://testserver"})
    assert response.status_code == 200
    assert response.content == "id,标记\nSYN-001,演示\n".encode()
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"].endswith("synthetic.csv")
    assert response.headers["content-security-policy"] == "sandbox; default-src 'none'"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-easyicu-source-sha256"] == written["sha256"]
    assert workspace.list_files("project-a") == before
    workspace.edit_file("project-a", "demo.html", old_text="synthetic", new_text="changed", expected_sha256=written["sha256"])
    assert client.post("/api/copilot/pi/projects/project-a/workspace/download-csv",
                       data=_csv_form(written["sha256"]), headers={"Origin": "http://testserver"}).status_code == 409


@pytest.mark.parametrize("origin", [None, "null", "http://evil.example", "http://testserver:8181", "http://testserver/path"])
def test_csv_bridge_rejects_missing_null_and_wrong_origins(workspace_client, origin):
    _, client = workspace_client
    headers = {"Origin": origin} if origin is not None else {}
    assert client.post("/api/copilot/pi/projects/project-a/workspace/download-csv",
                       data=_csv_form("a" * 64), headers=headers).status_code == 403


@pytest.mark.parametrize("overrides", [
    {"filename": "../private.csv"}, {"filename": "a\\b.csv"}, {"filename": "a:b.csv"},
    {"filename": "a\u202eb.csv"}, {"filename": "x.html"}, {"media_type": "text/html"},
    {"csv_base64": "!invalid"}, {"csv_base64": base64.b64encode(b"\xff").decode()},
    {"csv_base64": base64.b64encode(b"x\0y").decode()}, {"unexpected": "value"},
    {"expected_sha256": "not-hash"},
])
def test_csv_bridge_rejects_unsafe_form_fields(workspace_client, overrides):
    _, client = workspace_client
    assert client.post("/api/copilot/pi/projects/project-a/workspace/download-csv",
                       data=_csv_form("a" * 64, **overrides), headers={"Origin": "http://testserver"}).status_code == 422


def test_csv_bridge_rejects_duplicate_fields_and_utf8_byte_overflow(workspace_client):
    _, client = workspace_client
    body = urlencode(_csv_form("a" * 64)) + "&filename=another.csv"
    headers = {"Origin": "http://testserver", "Content-Type": "application/x-www-form-urlencoded"}
    assert client.post("/api/copilot/pi/projects/project-a/workspace/download-csv", content=body, headers=headers).status_code == 422
    too_big = base64.b64encode(("中" * (MAX_CSV_BYTES // 3 + 1)).encode()).decode()
    assert client.post("/api/copilot/pi/projects/project-a/workspace/download-csv",
                       content=urlencode(_csv_form("a" * 64, csv_base64=too_big)), headers=headers).status_code == 413


def test_csv_bridge_stream_limit_stops_before_consuming_the_remainder():
    consumed = []

    async def receive():
        consumed.append(1)
        return {"type": "http.request", "body": b"x" * (MAX_FORM_BYTES + 1), "more_body": True}

    request = Request({"type": "http", "method": "POST", "scheme": "http", "path": "/", "query_string": b"",
                       "server": ("testserver", 80), "headers": [(b"host", b"testserver"), (b"origin", b"http://testserver"),
                       (b"content-type", b"application/x-www-form-urlencoded")]}, receive)
    with pytest.raises(HTTPException) as error:
        asyncio.run(read_csv_download_form(request))
    assert error.value.status_code == 413 and len(consumed) == 1


def test_csv_bridge_is_injected_only_with_opt_in_and_valid_per_frame_nonce(workspace_client):
    workspace, client = workspace_client
    written = workspace.write_file("project-a", "demo.html", "<!doctype html><script>void 0</script>")
    params = {"file": "demo.html", "checked_sha256": written["sha256"], "download_nonce": "b" * 64}
    base = "/api/copilot/pi/projects/project-a/workspace/preview"
    assert "easyicu:workspace-csv" not in client.get(base, params=params).text
    response = client.get(base, params={**params, "allow_downloads": "true"})
    assert "easyicu:workspace-csv-child" in response.text
    assert "event.source!==frame.contentWindow" in response.text
    assert "connect-src 'none'" in response.headers["content-security-policy"]
    assert client.get(base, params={**params, "download_nonce": "<script>"}).status_code == 422

