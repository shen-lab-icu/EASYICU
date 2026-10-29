"""CSV cards use exact user attachments while model data reads stay closed."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.projections import _project_replay_resource
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.tools import _workspace_resource
from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace
from easyicu.webserver.routes import pi_copilot as routes


@pytest.mark.parametrize("hosted", [False, True])
def test_delivered_csv_card_is_an_exact_user_download_without_a_model_data_read(tmp_path, monkeypatch, hosted):
    monkeypatch.setenv("EASYICU_HOSTED", "1" if hosted else "0")
    workspace = ProjectWorkspace(tmp_path / "workspace")
    content = "synthetic_id,sex,glucose\nSYN-001,F,6.2\n"
    written = workspace.write_file("synthetic-project", "synthetic.csv", content)
    card = _project_replay_resource(_workspace_resource(written))
    assert card["sha256"] == written["sha256"]
    assert "checked_sha256" not in card

    class Service:
        get_workspace_file = PiCopilotService.get_workspace_file
        get_workspace_download = PiCopilotService.get_workspace_download

        def _assert_project_initialized(self, project_id):
            assert project_id == "synthetic-project"
            return project_id

    service = Service()
    service.workspace = workspace
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    complete_read = Mock(wraps=workspace._read_complete_file)
    monkeypatch.setattr(workspace, "_read_complete_file", complete_read)
    client = TestClient(app)
    response = client.get("/api/copilot/pi/projects/synthetic-project/workspace/file", params={"file": card["file"]})
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "tabular_read_unavailable_when_hosted"
    complete_read.assert_not_called()

    response = client.get("/api/copilot/pi/projects/synthetic-project/workspace/download", params={
        "file": card["file"], "expected_sha256": card["sha256"],
    })
    assert response.status_code == 200
    assert response.content == content.encode("utf-8")
    assert response.headers["x-easyicu-artifact-sha256"] == card["sha256"]
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert response.headers["content-security-policy"] == "sandbox; default-src 'none'"
    assert complete_read.call_count == 1
    assert "scientific_evidence" not in card
    assert "checked_sha256" not in card
