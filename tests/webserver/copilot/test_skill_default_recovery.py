"""Project defaults are explicit and cannot be cleared across a concurrent edit."""
import pytest
from easyicu.extensions.skill_attachments import attachment_payload
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from tests.webserver.copilot.test_turn_skill_attachments import _install, _session

pytest_plugins = ["tests.webserver.copilot.test_turn_skill_attachments"]


def test_stale_default_can_be_cleared_explicitly_without_changing_old_session(service, registry):
    installed = _install(registry)
    ref = {"source": "user", "id": installed["name"], "digest": installed["digest"]}
    _session(service)
    service.project_default_skills(project_id="synthetic-project", attached_skills=[ref])
    old = _session(service)
    newer = _install(registry, "A changed instruction.")
    with pytest.raises(PiCopilotError) as exc:
        _session(service)
    assert exc.value.code == "skill_attachment_stale"
    # The stale default is visible; nothing is silently removed by the refusal.
    assert service.project_default_skills(project_id="synthetic-project")["attached_skills"] == [ref]
    new_ref = {**ref, "digest": newer["digest"]}
    service.project_default_skills(project_id="synthetic-project", attached_skills=[new_ref])
    with pytest.raises(PiCopilotError) as exc:
        service.project_default_skills(project_id="synthetic-project", attached_skills=[], expected_attached_skills=[ref])
    assert exc.value.code == "skill_defaults_changed"
    assert service.project_default_skills(project_id="synthetic-project")["attached_skills"] == [new_ref]
    service.project_default_skills(project_id="synthetic-project", attached_skills=[], expected_attached_skills=[new_ref])
    assert _session(service)["draft_attached_skills"] == []
    assert attachment_payload(service._get_record(old["session_id"]).draft_attached_skills) == [ref]


def test_recovery_api_rejects_changed_default_without_mutation(service, registry, monkeypatch):
    from fastapi.testclient import TestClient
    from easyicu.webserver.app import app
    from easyicu.webserver.routes import skill_attachments as routes

    installed = _install(registry)
    old = {"source": "user", "id": installed["name"], "digest": installed["digest"]}
    _session(service)
    service.project_default_skills(project_id="synthetic-project", attached_skills=[old])
    newer = _install(registry, "Newer advice.")
    current = {**old, "digest": newer["digest"]}
    service.project_default_skills(project_id="synthetic-project", attached_skills=[current])
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    client = TestClient(app)
    url = "/api/copilot/pi/projects/synthetic-project/default-skills"
    refusal = client.put(url, json={"attached_skills": [], "expected_attached_skills": [old]})
    assert refusal.status_code == 409
    assert refusal.json()["detail"]["error"] == "skill_defaults_changed"
    assert client.get(url).json()["attached_skills"] == [current]
    assert client.put(url, json={"attached_skills": [], "expected_attached_skills": [current]}).status_code == 200
    assert client.get(url).json()["attached_skills"] == []
