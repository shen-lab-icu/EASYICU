"""Explanation recovery keeps original failures, budget ownership and grants."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.reply_continuation import validation_context
from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway, study_state as _study_state

study_state = _study_state


class ContinuationGateway(FakeGateway):
    ready = False
    blocked_reason = ""

    def request(self, method, params, **kwargs):
        state = super().request(method, params, **kwargs)
        if self.ready:
            state["reply_continuation"] = {
                "available": not self.blocked_reason,
                "reason_code": self.blocked_reason or "pi_reply_continuation_available",
                "source_error_entry_id": "failed-entry", "source_user_entry_id": "original-user",
                "completed_tool_count": 2, "completed_resources": [],
            }
            state["transcript"] = [{"role": "assistant", "content": [], "stop_reason": "error",
                                    "error_code": "pi_model_provider_unavailable", "entry_id": "failed-entry"}]
        if method == "session.continue_reply":
            assert kwargs.get("tool_context") is None
            assert params["validation_context"]["status"] == "not_checked"
            for event in [
                {"type": "run_start", "at": "2026-10-29T04:00:00Z"},
                {"type": "text_delta", "delta": "已保存，当前运行时验收尚未完成。", "at": "2026-10-29T04:00:01Z"},
                {"type": "message_end", "stop_reason": "stop", "at": "2026-10-29T04:00:02Z"},
            ]:
                kwargs["event_sink"](event)
            state["transcript"].append({"role": "assistant", "content": [{"type": "text", "text": "已保存，当前运行时验收尚未完成。"}], "stop_reason": "stop", "entry_id": "continued-entry"})
            state["reply_continuation"] = {"available": False, "reason_code": "pi_reply_continuation_no_failed_reply"}
        return state


def setup_service(tmp_path):
    gateway = ContinuationGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    created = service.create_session(project_id="recovery-project", external_llm_opt_in=True, agent_mode="workspace")
    session_id = created["session"]["session_id"]
    gateway.ready = True
    record = service._get_record(session_id)
    record.last_message_job_id = "original-failed-job"
    record.last_turn_status = "failed"
    service._save_record(record)
    service.replay_store.start_turn(session_id=session_id, project_id="recovery-project", job_id="original-failed-job",
                                   user_message="保存并检查", allowed_actions=["workspace_write"])
    service.replay_store.finish_turn(session_id=session_id, project_id="recovery-project", job_id="original-failed-job", status="failed")
    return service, gateway, session_id


def test_continuation_job_has_no_tool_context_and_keeps_original_failed_replay(tmp_path, study_state):
    service, gateway, session_id = setup_service(tmp_path)
    submitted = service.continue_reply(session_id, project_id="recovery-project", source_error_entry_id="failed-entry")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job.status != "running":
            break
        time.sleep(0.01)
    assert job.status == "done", job.error
    assert submitted["source_job_id"] == "original-failed-job"
    calls = [(method, params, context) for method, params, context in gateway.calls if method == "session.continue_reply"]
    assert len(calls) == 1
    assert calls[0][2] is None
    assert set(calls[0][1]) == {"session_id", "source_error_entry_id", "validation_context"}
    assert not any(method in {"session.prompt", "session.regenerate"} for method, _, _ in gateway.calls)
    record = service._get_record(session_id)
    assert record.last_turn_allowed_actions == []
    assert record.last_turn_status == "done"
    assert record.active_message_job_id is None
    replay = service.replay_store.snapshot(session_id=session_id, project_id="recovery-project")
    assert replay["turns"][0]["status"] == "failed"
    assert replay["turns"][0]["allowed_actions"] == ["workspace_write"]
    assert replay["turns"][1]["kind"] == "reply_continuation"
    assert replay["turns"][1]["source_job_id"] == "original-failed-job"
    assert replay["turns"][1]["status"] == "done"
    assert replay["turns"][1]["allowed_actions"] == []
    assert job.result["session"]["transcript"][0]["error_code"] == "pi_model_provider_unavailable"
    assert job.result["session"]["reply_continuation"]["available"] is False


@pytest.mark.parametrize("reason", ["pi_reply_continuation_tool_outcome_unknown", "pi_shell_message_provider_call_budget_exhausted"])
def test_unknown_or_exhausted_continuation_never_submits_a_job(tmp_path, study_state, reason):
    service, gateway, session_id = setup_service(tmp_path)
    gateway.blocked_reason = reason
    with pytest.raises(PiCopilotError) as caught:
        service.continue_reply(session_id, project_id="recovery-project", source_error_entry_id="failed-entry")
    assert caught.value.code == reason
    assert session_id not in service._busy_sessions
    assert not any(method == "session.continue_reply" for method, _, _ in gateway.calls)
    assert service._get_record(session_id).last_message_job_id == "original-failed-job"


def test_continuation_rejects_stale_target_and_project_mismatch(tmp_path, study_state):
    service, gateway, session_id = setup_service(tmp_path)
    for project_id, error_id in [("different-project", "failed-entry"), ("recovery-project", "old-entry")]:
        with pytest.raises(PiCopilotError):
            service.continue_reply(session_id, project_id=project_id, source_error_entry_id=error_id)
    assert not any(method == "session.continue_reply" for method, _, _ in gateway.calls)


def test_validation_context_uses_only_current_exact_sha_qa_and_retains_failure():
    calls = []
    class Workspace:
        changed = False
        def preview_file(self, project, file, *, checked_sha256):
            calls.append((project, file, checked_sha256))
            if self.changed:
                raise ValueError("stale hash")
    workspace = Workspace()
    service = SimpleNamespace(workspace=workspace, runtime_qa=SimpleNamespace(read=lambda *args: {"status": "failed"}))
    record = SimpleNamespace(project_id="project")
    state = {"completed_resources": [{"file": "brief.html", "sha256": "a" * 64}],
             "resources_complete": True, "completed_resource_total": 1}
    assert validation_context(service, record, state)["status"] == "failed"
    assert calls == [("project", "brief.html", "a" * 64)]
    workspace.changed = True
    assert validation_context(service, record, state)["status"] == "not_checked"


@pytest.mark.parametrize("resources_complete,total", [(False, 9), (True, 9), (False, 1)])
def test_incomplete_artifact_identity_or_truncated_coverage_cannot_be_called_passed(resources_complete, total):
    service = SimpleNamespace(workspace=SimpleNamespace(preview_file=lambda *args, **kwargs: {}),
                              runtime_qa=SimpleNamespace(read=lambda *args: {"status": "complete"}))
    state = {"completed_resources": [{"file": "brief.html", "sha256": "a" * 64}],
             "resources_complete": resources_complete, "completed_resource_total": total}
    assert validation_context(service, SimpleNamespace(project_id="project"), state)["status"] == "not_checked"


@pytest.mark.parametrize("failure", ["save", "finish_replay"])
def test_persistence_failure_still_releases_continuation_admission(tmp_path, study_state, monkeypatch, failure):
    service, gateway, session_id = setup_service(tmp_path)
    if failure == "save":
        save = service._save_record
        def fail_on_terminal(record):
            if record.last_turn_status == "done":
                raise RuntimeError("injected_terminal_save_failure")
            return save(record)
        monkeypatch.setattr(service, "_save_record", fail_on_terminal)
    else:
        monkeypatch.setattr(service.replay_store, "finish_turn", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected_replay_finish_failure")))
    submitted = service.continue_reply(session_id, project_id="recovery-project", source_error_entry_id="failed-entry")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job.status != "running":
            break
        time.sleep(0.01)
    assert job.status == "failed"
    assert "injected_" in job.error
    assert session_id not in service._busy_sessions
    assert session_id not in service._active_message_jobs


def test_workspace_first_turn_is_not_forced_through_research_clarification(tmp_path, study_state, monkeypatch):
    service, gateway, session_id = setup_service(tmp_path)
    gateway.ready = False
    monkeypatch.setattr(service_module, "build_project_workflow_projection", lambda **kwargs: SimpleNamespace(workflow=SimpleNamespace(current_stage="idea")))
    submitted = service.send_message(session_id, project_id="recovery-project", message="请保存一个HTML页面", allowed_actions=["workspace_write"])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job.status != "running":
            break
        time.sleep(0.01)
    prompt = next(params for method, params, _ in gateway.calls if method == "session.prompt")
    assert "intent" not in prompt


def test_continue_route_rejects_new_prompt_grants_or_browser_supplied_qa(monkeypatch):
    from easyicu.webserver.app import app
    from easyicu.webserver.routes import pi_copilot as routes
    received = []
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: SimpleNamespace(continue_reply=lambda *args, **kwargs: received.append((args, kwargs)) or {"ok": True}))
    client = TestClient(app)
    url = "/api/copilot/pi/sessions/pi_test/continue-reply"
    base = {"project_id": "project", "source_error_entry_id": "failed"}
    for extra in [{"message": "redo original"}, {"allowed_actions": ["workspace_write"]}, {"validation_context": {"status": "passed"}}]:
        assert client.post(url, json={**base, **extra}).status_code == 422
    assert received == []
    assert client.post(url, json=base).status_code == 200
    assert received[0] == (("pi_test",), base)
