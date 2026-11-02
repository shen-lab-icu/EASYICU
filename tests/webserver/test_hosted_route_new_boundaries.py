"""Hosted boundaries for previously unaudited, already-existing routes."""

import base64

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.workspace import ProjectWorkspace
from easyicu.webserver.pi_copilot.workspace_runtime_qa import RuntimeQaStore
from easyicu.webserver.routes import agent, non_executable_proposal, pi_copilot
from tests.webserver.copilot.pi_copilot_contract_fixtures import study_state as _study_state
from tests.support.pi_reply_continuation import RetryGateway, setup_service

study_state = _study_state


@pytest.fixture
def hosted_workspace(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(home))
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(tmp_path / "allowed"))
    (tmp_path / "allowed").mkdir()

    class Service:
        get_workspace_download = PiCopilotService.get_workspace_download
        get_workspace_preview = PiCopilotService.get_workspace_preview
        begin_workspace_runtime_qa = PiCopilotService.begin_workspace_runtime_qa
        get_workspace_runtime_qa = PiCopilotService.get_workspace_runtime_qa
        record_workspace_runtime_qa = PiCopilotService.record_workspace_runtime_qa

        def _assert_project_initialized(self, project_id):
            if project_id != "project":
                raise PiCopilotError("pi_project_not_initialized", "Not initialized", status_code=409)
            return project_id

    service = Service()
    service.workspace = ProjectWorkspace(home / "workspace")
    service.runtime_qa = RuntimeQaStore()
    monkeypatch.setattr(pi_copilot, "get_pi_copilot_service", lambda: service)
    return service.workspace, TestClient(app), tmp_path


@pytest.mark.parametrize("route", ["download", "qa_get", "qa_post", "qa_report", "csv"])
@pytest.mark.parametrize("path_kind", ["traversal", "symlink"])
def test_hosted_workspace_download_and_qa_reject_escape(hosted_workspace, route, path_kind):
    workspace, client, tmp_path = hosted_workspace
    outside = tmp_path / "outside.html"
    outside.write_text("private")
    if path_kind == "symlink":
        root = workspace.project_root("project")
        (root / "escape.html").symlink_to(outside)
        filename = "escape.html"
    else:
        filename = "../../outside.html"
    base = "/api/copilot/pi/projects/project/workspace"
    digest = "a" * 64
    body = {"file": filename, "checked_sha256": digest}
    if route == "download":
        response = client.get(base + "/download", params={"file": filename, "expected_sha256": digest})
    elif route == "qa_get":
        response = client.get(base + "/runtime-qa", params=body)
    elif route == "qa_post":
        response = client.post(base + "/runtime-qa", json=body)
    elif route == "qa_report":
        response = client.post(base + "/runtime-qa/report", json={**body, "nonce": digest, "report": {}})
    else:
        response = client.post(base + "/download-csv", headers={"origin": "http://testserver"}, data={
            "file": filename, "expected_sha256": digest, "filename": "safe.csv", "media_type": "text/csv",
            "csv_base64": base64.b64encode(b"x\n1\n").decode(),
        })
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == (
        "pi_workspace_path_escape" if path_kind == "traversal" else "pi_workspace_symlink_blocked"
    )
    assert b"private" not in response.content
    assert outside.read_text() == "private"


@pytest.mark.parametrize("route", ["candidate", "review", "projection-download"])
def test_hosted_preparation_rejects_path_alias_before_dispatch(tmp_path, monkeypatch, route):
    from easyicu.webserver.pi_copilot import service as service_module

    home = tmp_path / "home"
    allowed = tmp_path / "allowed"
    home.mkdir()
    allowed.mkdir()
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(home))
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(allowed))

    def forbidden(*args, **kwargs):
        pytest.fail("a forbidden data path reached the route owner")

    monkeypatch.setattr(agent, "_host_action", forbidden)
    monkeypatch.setattr(service_module, "get_pi_copilot_service", forbidden)
    response = TestClient(app).post("/api/jobs/raw-preparation-" + route,
        json={"run_id": "run", "study_context_id": "study", "session_id": "session", "project_id": "project",
              "unexpected": {"source_path": str(tmp_path / "outside")}})
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "path_outside_data_roots"


@pytest.mark.parametrize("method,suffix,form", [
    ("GET", "", None),
    ("GET", "/source-candidate", None),
    ("POST", "/candidate", {"decision_id": "question", "citation_key": "pmid:1",
                              "source_evidence_id": "evidence", "verbatim_excerpt": "a valid length excerpt"}),
    ("POST", "/candidate/save", {"decision_id": "question", "citation_key": "pmid:1",
                                   "source_evidence_id": "evidence", "verbatim_excerpt": "a valid length excerpt"}),
    ("POST", "/pubmed-abstract", {"citation_key": "pmid:1"}),
    ("POST", "/pubmed-candidate", {"pmid": "1"}),
])
@pytest.mark.parametrize("mismatch", ["session", "run", "study", "configuration"])
def test_hosted_proposal_requires_bound_session_before_evidence_or_network(
        tmp_path, monkeypatch, method, suffix, form, mismatch):
    from easyicu.webserver import non_executable_proposal_view as proposal

    home, allowed = tmp_path / "home", tmp_path / "allowed"
    home.mkdir()
    allowed.mkdir()
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(home))
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(allowed))

    class ProjectStore:
        def resolve(self, project_id):
            return "study" if project_id == "project" else None

    class Service:
        project_store = ProjectStore()

        def list_sessions(self, *, project_id, limit):
            assert project_id == "project"
            return {"sessions": [{"session_id": "other-session" if mismatch == "session" else "unbound-session"}]}

        class replay_store:
            @staticmethod
            def snapshot(**kwargs):
                return {"child_jobs": [{"run_id": "other-run" if mismatch == "run" else "run"}]}

    def forbidden(*args, **kwargs):
        pytest.fail("unbound proposal reached evidence or PubMed transport")

    monkeypatch.setattr(non_executable_proposal, "get_pi_copilot_service", lambda: Service())
    monkeypatch.setattr(proposal, "list_bound_run_history", lambda **kwargs: [{
        "run_id": "run", "study_id": "other-study" if mismatch == "study" else "study",
        "scientific_configuration_sha256": "old"}])
    monkeypatch.setattr(proposal.study_contexts, "get_context", lambda study_id: {"id": study_id})
    monkeypatch.setattr(proposal.study_contexts, "scientific_configuration_sha256", lambda study: "current")
    monkeypatch.setattr(proposal, "_candidate_evidence_root", forbidden)
    monkeypatch.setattr(proposal.PubMedLiteratureClient, "_http_get", forbidden)
    monkeypatch.setattr(proposal.EvidenceStore, "register_text", forbidden)
    response = TestClient(app).request(method, "/api/pi/proposal/project/unbound-session/run" + suffix,
        headers={"origin": "http://testserver"}, data=form)
    assert response.status_code == 409


@pytest.mark.parametrize("route,gateway", [("continue-reply", None), ("retry-unstarted", RetryGateway)])
@pytest.mark.parametrize("case", ["project", "stale_error", "path_alias", "wrong_recovery", "cancelled"])
def test_hosted_reply_recovery_refuses_unbound_or_changed_authority(
        tmp_path, monkeypatch, study_state, route, gateway, case):
    from easyicu.webserver.pi_copilot import service as service_module

    home, allowed = tmp_path / "home", tmp_path / "allowed"
    home.mkdir()
    allowed.mkdir()
    selected_gateway = (None if gateway else RetryGateway) if case == "wrong_recovery" else gateway
    service, connection, session = setup_service(home, **({"gateway_cls": selected_gateway} if selected_gateway else {}))
    if selected_gateway:
        record = service._get_record(session)
        record.last_turn_allowed_actions = ["workspace_write"]
        service._save_record(record)
    if case == "cancelled":
        record = service._get_record(session)
        record.last_turn_status = "cancelled"
        service._save_record(record)
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(home))
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(allowed))
    monkeypatch.setattr(pi_copilot, "get_pi_copilot_service", lambda: service)
    body = {"project_id": "other-project" if case == "project" else "recovery-project",
            "source_error_entry_id": "old-entry" if case == "stale_error" else "failed-entry"}
    if case == "path_alias":
        body["source_path"] = str(tmp_path / "outside")
    before = list(connection.calls)
    response = TestClient(app).post(f"/api/copilot/pi/sessions/{session}/{route}", json=body)
    assert response.status_code in {403, 404, 409}
    if case == "path_alias":
        assert response.json()["detail"]["error"] == "path_outside_data_roots"
    elif case == "stale_error":
        assert response.json()["detail"]["error"] == "pi_reply_continuation_stale"
    elif case == "wrong_recovery":
        assert response.json()["detail"]["error"] == "pi_workspace_recovery_action_mismatch"
    elif case == "cancelled":
        assert response.json()["detail"]["error"] == "pi_workspace_recovery_cancelled"
    assert not any(method in {"session.continue_reply", "session.retry_failed_turn"}
                   for method, *_ in connection.calls[len(before):])
    assert service._get_record(session).last_message_job_id == "original-failed-job"
    assert not service_module.jobs.MANAGER.get("original-failed-job")
