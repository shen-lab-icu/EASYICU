"""A localized Demo filter failure stays visible without exporting raw errors."""

from easyicu.webserver.pi_copilot.projections import project_job
from tests.webserver.copilot.pi_copilot_contract_fixtures import (
    FakeGateway,
    study_state as study_state,
)


def test_known_demo_filter_failure_has_a_safe_code_but_no_raw_exception():
    for error, expected in [
        ("不支持的数据库: eicu_demo", "extraction_database_filter_unsupported"),
        ("不支持的数据库: mimic_demo", "extraction_database_filter_unsupported"),
        ("不支持的数据库: /private/patient-data", None),
        ("failed reading /Users/researcher/patient_12345/raw.csv", None),
    ]:
        result = project_job({
            "id": "synthetic-extract", "kind": "extract", "status": "failed",
            "created": 1, "finished": 2, "error": error,
        })
        assert result["status"] == "failed"
        assert result["error_code"] == expected
        assert error not in str(result)


def test_reloaded_session_retains_terminal_extraction_and_interrupts_a_lost_turn(
    tmp_path, study_state, monkeypatch,
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from easyicu.webserver import jobs
    from easyicu.webserver.pi_copilot.service import PiCopilotService
    from easyicu.webserver.routes import pi_copilot as routes

    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())
    path = tmp_path / "sessions.json"
    first = PiCopilotService(store_path=path, gateway=FakeGateway())
    session_id = first.create_session(project_id="terminal-project", external_llm_opt_in=True)["session"]["session_id"]
    scope = {"session_id": session_id, "project_id": "terminal-project"}
    record = first._get_record(session_id)
    record.active_message_job_id = "lost-message"
    record.last_turn_status = "running"
    first._save_record(record)
    first.replay_store.start_turn(**scope, job_id="lost-message", allowed_actions=["run"])
    first.replay_store.record_host_action(
        **scope, action_id="extraction-1", action_code="start_extraction", child_job_id="synthetic-extract",
    )
    first.replay_store.archive_child_job(**scope, job=project_job({
        "id": "synthetic-extract", "kind": "extract", "status": "failed",
        "created": 1, "finished": 2, "error": "不支持的数据库: eicu_demo",
    }))
    first.close()
    restarted = PiCopilotService(store_path=path, gateway=FakeGateway())
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: restarted)
    app = FastAPI()
    app.include_router(routes.router)
    try:
        response = TestClient(app).get(
            f"/api/copilot/pi/sessions/{session_id}", params={"project_id": "terminal-project"},
        )
        assert response.status_code == 200
        session = response.json()["session"]
        assert session["active_message_job_id"] is None
        assert session["last_turn_status"] == "interrupted"
        assert session["archived_child_jobs"][0]["status"] == "failed"
        assert session["archived_child_jobs"][0]["error_code"] == "extraction_database_filter_unsupported"
        assert "不支持的数据库" not in response.text
        assert not [call for call in restarted.gateway.calls if call[0] == "session.prompt"]
    finally:
        restarted.close()
