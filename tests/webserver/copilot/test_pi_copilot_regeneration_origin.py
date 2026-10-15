"""Regeneration cannot upgrade a model-authored option into user authority."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway, study_state as _study_state

study_state = _study_state


@pytest.mark.parametrize("origin", [None, "model_option", "typed", "invalid"])
@pytest.mark.parametrize("intent", [None, "user_edited_message"])
def test_unchanged_regeneration_uses_saved_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, study_state: dict,
    origin: str | None, intent: str | None,
) -> None:
    class Gateway(FakeGateway):
        def request(self, method: str, params: dict, **kwargs: object) -> dict:
            response = super().request(method, params, **kwargs)
            if method == "session.regenerate.inspect":
                response["message"] = "生成分析计划"
                response["message_origin"] = origin
            return response

    monkeypatch.setattr(
        service_module.study_contexts, "restore_turn_configuration_snapshot",
        lambda *_args, **_kwargs: dict(study_state),
    )
    gateway = Gateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    created = service.create_session(project_id="origin-test", external_llm_opt_in=True)
    submitted = service.send_message(
        created["session"]["session_id"], project_id="origin-test",
        message="生成分析计划", allowed_actions=[],
        regenerate_user_entry_id="entry-user-1", regeneration_intent=intent,
        message_origin="typed",  # The retry cannot upgrade the saved source.
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)
    assert job is not None and job.status == "done"
    context = gateway.tool_contexts[-1]
    assert ("provider_run" in context.allowed_actions) == (origin == "typed")
    assert context.user_turns[-1] == {
        "text": "生成分析计划", "message_origin": origin or "unknown",
    }
    params = next(params for method, params, _ in gateway.calls if method == "session.regenerate")
    assert params["message_origin"] == ("typed" if origin == "typed" else "model_option")
