"""HTTP contract tests for the Pi Copilot route owner."""

from __future__ import annotations

from fastapi.testclient import TestClient

from easyicu.webserver.app import app
from easyicu.webserver.pi_copilot.contracts import (
    PiCopilotError,
    ResearchProviderBinding,
)
from easyicu.webserver.routes import pi_copilot as route_module


class FakeService:
    def runtime_status(self) -> dict:
        return {"ok": True, "runtime": {"status": "ready"}}

    def verified_api_research_provider_binding(self) -> ResearchProviderBinding:
        """The API-key branch of session creation compiles a binding first."""
        return ResearchProviderBinding(model="fake-configured-model")

    def create_session(self, **kwargs) -> dict:
        return {"ok": True, "received": kwargs}

    def initialize_project(self, **kwargs) -> dict:
        return {"ok": True, "status": "ready", "received": kwargs}

    def get_project_workflow(self, **kwargs) -> dict:
        return {
            "ok": True,
            "workflow": {
                "schema_version": "easyicu.pi-research-workflow/1",
                "current_stage": "setup",
                "completed_required_stages": 1,
                "required_stage_count": 7,
                "stages": [],
            },
            "received": kwargs,
        }

    def get_workspace_file(self, **kwargs) -> dict:
        return {
            "ok": True,
            "artifact": {
                "file": kwargs["relative_file"],
                "media_type": "text/html",
                "text": "<h1>Safe code view</h1>",
            },
        }

    def get_workspace_preview(self, **kwargs) -> dict:
        return {
            "ok": True,
            "artifact": {
                "file": kwargs["relative_file"],
                "checked_sha256": kwargs["checked_sha256"],
                "media_type": "text/html",
                "text": "<h1>Sandboxed preview</h1><script>document.title='demo'</script>",
            },
        }

    def get_research_artifact(self, **kwargs) -> dict:
        return {
            "ok": True,
            "run_id": kwargs["run_id"],
            "artifact": {
                "name": kwargs["artifact_name"],
                "media_type": "application/json",
            },
            "payload": {"status": "ready"},
            "governance": {
                "authority_class": "easyicu_run_artifact",
                "gate_status": "analysis_only",
                "readiness_status": "awaiting_human_signoff",
                "human_signoff": "required",
                "reportable": False,
                "claim_ceiling": "analysis_only",
            },
        }

    def get_research_document(self, **kwargs) -> dict:
        return {
            "content": b"<!doctype html><title>System validation</title>",
            "media_type": "text/html",
            "claim_ceiling": "engineering_validation_only",
        }

    def configure_provider(self, **kwargs) -> dict:
        assert kwargs["api_key"] == "route-private-key"
        return {
            "ok": True,
            "runtime": {"status": "ready"},
            "configuration": {
                "credential_present": True,
                "connection_verified": True,
                "secrets_returned": False,
            },
        }

    def send_message(self, session_id: str, **kwargs) -> dict:
        return {"ok": True, "session_id": session_id, "received": kwargs}

    def list_sessions(self, **kwargs) -> dict:
        return {"ok": True, "sessions": [], "received": kwargs}

    def get_session(self, session_id: str, **kwargs) -> dict:
        return {"ok": True, "session_id": session_id, "received": kwargs}

    def rebind_session(self, session_id: str, **kwargs) -> dict:
        return {"ok": True, "session_id": session_id, "received": kwargs}

    def set_presentation_pin(self, session_id: str, **kwargs) -> dict:
        return {"ok": True, "session_id": session_id, "received": kwargs}

    def archive_child_job(self, session_id: str, **kwargs) -> dict:
        return {"ok": True, "session_id": session_id, "received": kwargs}

    def abort_session(self, session_id: str, **kwargs) -> dict:
        return {"ok": True, "session_id": session_id, "received": kwargs}


def test_status_and_create_routes_preserve_strict_boolean_opt_in(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    assert client.get("/api/copilot/pi/status").json()["runtime"]["status"] == "ready"
    created = client.post(
        "/api/copilot/pi/sessions",
        json={
            "project_id": "guided-project-1",
            "title": "Review session",
            "language": "zh",
            "thinking_level": "high",
            "external_llm_opt_in": True,
        },
    )
    assert created.status_code == 200
    assert created.json()["received"]["external_llm_opt_in"] is True

    string_boolean = client.post(
        "/api/copilot/pi/sessions",
        json={"project_id": "guided-project-1", "external_llm_opt_in": "true"},
    )
    assert string_boolean.status_code == 422
    unknown = client.post(
        "/api/copilot/pi/sessions",
        json={
            "project_id": "guided-project-1",
            "external_llm_opt_in": True,
            "api_key": "must-not-be-accepted",
        },
    )
    assert unknown.status_code == 422
    assert created.json()["received"]["project_id"] == "guided-project-1"


def test_session_queries_are_scoped_to_one_research_project(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    assert client.get("/api/copilot/pi/sessions").status_code == 422
    listed = client.get(
        "/api/copilot/pi/sessions",
        params={"project_id": "guided-project-2", "limit": 7},
    )
    assert listed.status_code == 200
    assert listed.json()["received"] == {
        "project_id": "guided-project-2",
        "limit": 7,
        "agent_mode": None,
    }

    assert client.get("/api/copilot/pi/sessions/pi-test").status_code == 422
    opened = client.get(
        "/api/copilot/pi/sessions/pi-test",
        params={"project_id": "guided-project-2"},
    )
    assert opened.status_code == 200
    assert opened.json()["received"]["project_id"] == "guided-project-2"


def test_project_initialization_is_an_explicit_typed_mutation(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    assert (
        client.post("/api/copilot/pi/projects/initialize", json={}).status_code == 422
    )
    initialized = client.post(
        "/api/copilot/pi/projects/initialize",
        json={
            "project_id": "guided-project-2",
            "title": "Existing study",
            "confirm_initialization": False,
        },
    )
    assert initialized.status_code == 200
    assert initialized.json()["received"] == {
        "project_id": "guided-project-2",
        "title": "Existing study",
        "confirm_initialization": False,
        "binding_receipt": None,
    }
    assert (
        client.post(
            "/api/copilot/pi/projects/initialize",
            json={
                "project_id": "guided-project-2",
                "confirm_initialization": "false",
            },
        ).status_code
        == 422
    )


def test_project_workspace_file_and_preview_routes_are_bounded(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    file_response = client.get(
        "/api/copilot/pi/projects/guided-project-2/workspace/file",
        params={"file": "prototype/index.html"},
    )
    assert file_response.status_code == 200
    assert file_response.json()["artifact"]["file"] == "prototype/index.html"

    preview = client.get(
        "/api/copilot/pi/projects/guided-project-2/workspace/preview",
        params={
            "file": "prototype/index.html",
            "checked_sha256": "a" * 64,
        },
    )
    assert preview.status_code == 200
    assert "Workspace artifact · Unvalidated" in preview.text
    assert "Not scientific evidence" in preview.text
    assert 'id="easyicu-workspace-preview-content"' in preview.text
    assert "&lt;h1&gt;Sandboxed preview&lt;/h1&gt;" in preview.text
    assert "<h1>Sandboxed preview</h1>" not in preview.text
    assert preview.headers["cache-control"] == "no-store"
    assert preview.headers["x-content-type-options"] == "nosniff"
    policy = preview.headers["content-security-policy"]
    assert policy.startswith("sandbox allow-scripts;")
    assert "default-src 'none'" in policy
    assert "connect-src 'none'" in policy
    assert "frame-ancestors 'self'" in policy
    assert preview.headers["referrer-policy"] == "no-referrer"

    assert (
        client.get(
            "/api/copilot/pi/projects/guided-project-2/workspace/file"
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/copilot/pi/projects/guided-project-2/workspace/preview",
            params={"file": "prototype/index.html"},
        ).status_code
        == 422
    )


def test_workspace_preview_surfaces_stale_checked_bytes_as_conflict(monkeypatch) -> None:
    class StalePreviewService(FakeService):
        def get_workspace_preview(self, **kwargs) -> dict:
            raise PiCopilotError(
                "pi_workspace_preview_check_stale",
                "The file changed after its static check.",
                status_code=409,
            )

    monkeypatch.setattr(
        route_module,
        "get_pi_copilot_service",
        lambda: StalePreviewService(),
    )
    response = TestClient(app).get(
        "/api/copilot/pi/projects/guided-project-2/workspace/preview",
        params={
            "file": "prototype/index.html",
            "checked_sha256": "a" * 64,
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "pi_workspace_preview_check_stale"


def test_project_workflow_route_is_project_scoped(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    response = client.get("/api/copilot/pi/projects/guided-project-2/workflow")
    assert response.status_code == 200
    assert response.json()["received"] == {"project_id": "guided-project-2"}
    assert response.json()["workflow"]["current_stage"] == "setup"


def test_project_research_artifact_route_uses_path_free_identity(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    response = client.get(
        "/api/copilot/pi/projects/guided-project-2/runs/run_20260808/artifacts/table1_summary.json"
    )
    assert response.status_code == 200
    assert response.json()["run_id"] == "run_20260808"
    assert response.json()["artifact"]["name"] == "table1_summary.json"
    assert response.json()["governance"] == {
        "authority_class": "easyicu_run_artifact",
        "gate_status": "analysis_only",
        "readiness_status": "awaiting_human_signoff",
        "human_signoff": "required",
        "reportable": False,
        "claim_ceiling": "analysis_only",
    }
    assert "project_dir" not in response.text

    invalid = client.get(
        "/api/copilot/pi/projects/guided-project-2/runs/run_20260808/artifacts/not-json.txt"
    )
    assert invalid.status_code == 422


def test_system_validation_document_route_is_fixed_and_engineering_only(
    monkeypatch,
) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    response = client.get(
        "/api/copilot/pi/projects/guided-project-2/runs/run_20260808/"
        "documents/system_validation_report.html"
    )

    assert response.status_code == 200
    assert response.headers["x-easyicu-claim-ceiling"] == (
        "engineering_validation_only"
    )
    assert "style-src 'unsafe-inline'" in response.headers["content-security-policy"]
    assert "img-src data:" in response.headers["content-security-policy"]
    assert (
        client.get(
            "/api/copilot/pi/projects/guided-project-2/runs/run_20260808/"
            "documents/system_validation_report.txt"
        ).status_code
        == 422
    )


def test_provider_setup_route_is_typed_and_never_returns_secret(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    response = client.post(
        "/api/copilot/pi/provider-config",
        json={
            "provider": "easyicu-local",
            "api_key": "route-private-key",
            "base_url": "http://127.0.0.1:8317/v1",
            "model": "gpt5.6 luna",
            "api_transport": "openai-completions",
            "enable_ai": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["runtime"]["status"] == "ready"
    assert "route-private-key" not in response.text
    assert (
        client.post(
            "/api/copilot/pi/provider-config",
            json={
                "provider": "easyicu-local",
                "api_key": "route-private-key",
                "base_url": "http://127.0.0.1:8317/v1",
                "model": "gpt5.6 luna",
                "api_transport": "openai-completions",
                "enable_ai": "true",
            },
        ).status_code
        == 422
    )
    anthropic = client.post(
        "/api/copilot/pi/provider-config",
        json={
            "provider": "anthropic",
            "api_key": "route-private-key",
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-sonnet-4-6",
            "api_transport": "anthropic-messages",
            "enable_ai": True,
        },
    )
    assert anthropic.status_code == 200
    assert (
        client.post(
            "/api/copilot/pi/provider-config",
            json={
                "provider": "easyicu-local",
                "api_key": "route-private-key",
                "base_url": "http://127.0.0.1:8317/v1",
                "model": "gpt5.6 luna",
                "api_transport": "unknown",
                "enable_ai": True,
            },
        ).status_code
        == 422
    )


def test_message_route_rejects_unknown_actions_and_fields(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    accepted = client.post(
        "/api/copilot/pi/sessions/pi-test/message",
        json={
            "project_id": "guided-project-1",
            "message": "Save setup and inspect aggregate validation",
            "allowed_actions": ["configure", "run"],
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["received"]["allowed_actions"] == ["configure", "run"]

    assert (
        client.post(
            "/api/copilot/pi/sessions/pi-test/message",
            json={
                "project_id": "guided-project-1",
                "message": "Inspect",
                "allowed_actions": ["bash"],
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/copilot/pi/sessions/pi-test/message",
            json={
                "project_id": "guided-project-1",
                "message": "Inspect",
                "raw_rows": True,
            },
        ).status_code
        == 422
    )


def test_all_session_mutations_require_project_scope(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    for suffix in ("rebind", "abort"):
        assert (
            client.post(
                f"/api/copilot/pi/sessions/pi-test/{suffix}",
                json={},
            ).status_code
            == 422
        )

    rebound = client.post(
        "/api/copilot/pi/sessions/pi-test/rebind",
        json={"project_id": "guided-project-1"},
    )
    assert rebound.status_code == 200
    assert rebound.json()["received"]["project_id"] == "guided-project-1"

    aborted = client.post(
        "/api/copilot/pi/sessions/pi-test/abort",
        json={"project_id": "guided-project-1", "message_job_id": "job-1"},
    )
    assert aborted.status_code == 200
    assert aborted.json()["received"] == {
        "project_id": "guided-project-1",
        "message_job_id": "job-1",
    }


def test_presentation_and_child_replay_routes_are_project_scoped(monkeypatch) -> None:
    fake = FakeService()
    monkeypatch.setattr(route_module, "get_pi_copilot_service", lambda: fake)
    client = TestClient(app)

    pin_path = "/api/copilot/pi/sessions/pi-test/presentation"
    child_path = "/api/copilot/pi/sessions/pi-test/child-jobs/job-child-1/archive"
    assert client.post(pin_path, json={"pinned": True}).status_code == 422
    assert client.post(child_path, json={}).status_code == 422

    pinned = client.post(
        pin_path,
        json={"project_id": "guided-project-2", "pinned": True},
    )
    archived = client.post(
        child_path,
        json={"project_id": "guided-project-2"},
    )
    assert pinned.status_code == 200
    assert pinned.json()["received"] == {
        "project_id": "guided-project-2",
        "pinned": True,
    }
    assert archived.status_code == 200
    assert archived.json()["received"] == {
        "project_id": "guided-project-2",
        "job_id": "job-child-1",
    }


def test_owner_error_keeps_stable_code_and_owner(monkeypatch) -> None:
    class BlockedService(FakeService):
        def create_session(self, **kwargs) -> dict:
            raise PiCopilotError(
                "external_llm_opt_in_required",
                "Explicit opt-in is required.",
                status_code=403,
            )

    monkeypatch.setattr(
        route_module,
        "get_pi_copilot_service",
        lambda: BlockedService(),
    )
    response = TestClient(app).post(
        "/api/copilot/pi/sessions",
        json={"project_id": "guided-project-1", "external_llm_opt_in": True},
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["error"] == "external_llm_opt_in_required"
    assert detail["owner"] == "easyicu.webserver.pi_copilot"
