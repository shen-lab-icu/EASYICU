"""Long artifact turns share the bounded transport deadline, not a 90s cap."""
from __future__ import annotations

import time
import threading
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.gateway import DEFAULT_REQUEST_TIMEOUT_SECONDS
from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import (
    FakeGateway,
    study_state as _study_state,
)

study_state = _study_state


class ArtifactDeadlineGateway(FakeGateway):
    """Report a deterministic 127s model turn without spending provider time."""

    required_model_seconds = 127
    prompt_timeouts: list[float]

    def __init__(self):
        super().__init__()
        self.prompt_timeouts = []

    def request(self, method, params, **kwargs):
        state = super().request(method, params, **kwargs)
        if method == "session.prompt":
            deadline = kwargs["timeout"]
            self.prompt_timeouts.append(deadline)
            if deadline < self.required_model_seconds:
                raise PiCopilotError("pi_gateway_timeout", "The turn exceeded its host deadline.", status_code=504)
            state["transcript"] = [{"role": "assistant", "stop_reason": "stop", "entry_id": "completed",
                                     "content": [{"type": "text", "text": "合成数据简报已完成。"}]}]
        return state


@pytest.mark.parametrize("required_model_seconds,expected_status", [(127, "done"), (901, "failed")])
def test_artifact_turn_uses_finite_canonical_deadline_without_automatic_retries(
    tmp_path: Path, study_state, required_model_seconds: int, expected_status: str,
) -> None:
    gateway = ArtifactDeadlineGateway()
    gateway.required_model_seconds = required_model_seconds
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    created = service.create_session(project_id="deadline-project", external_llm_opt_in=True, agent_mode="workspace")
    session_id = created["session"]["session_id"]
    submitted = service.send_message(session_id, project_id="deadline-project", message="请完成一份合成数据HTML简报。")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)
    assert job.status == expected_status, job.error
    assert gateway.prompt_timeouts == [DEFAULT_REQUEST_TIMEOUT_SECONDS]
    assert service._get_record(session_id).active_message_job_id is None
    assert session_id not in service._busy_sessions
    replay = service.replay_store.snapshot(session_id=session_id, project_id="deadline-project")
    assert len(replay["turns"]) == 1
    assert replay["turns"][0]["status"] == expected_status


def test_user_abort_interrupts_the_longer_artifact_deadline(tmp_path: Path, study_state) -> None:
    started, aborted = threading.Event(), threading.Event()

    class CancellableGateway(FakeGateway):
        def request(self, method, params, **kwargs):
            state = super().request(method, params, **kwargs)
            if method == "session.prompt":
                started.set()
                assert aborted.wait(3), "Abort did not reach the active sidecar turn"
            elif method == "session.abort":
                aborted.set()
            return state

    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=CancellableGateway())
    session_id = service.create_session(project_id="abort-project", external_llm_opt_in=True,
                                        agent_mode="workspace")["session"]["session_id"]
    submitted = service.send_message(session_id, project_id="abort-project", message="请生成合成数据简报。")
    assert started.wait(3)
    stopped = service.abort_session(session_id, project_id="abort-project", message_job_id=submitted["job_id"])
    assert stopped["cancel_requested"] is True
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)
    assert job.status == "cancelled", job.error
    record = service._get_record(session_id)
    assert record.last_message_job_id == submitted["job_id"]
    assert record.last_turn_status == "cancelled"
    assert record.active_message_job_id is None
    assert session_id not in service._busy_sessions
