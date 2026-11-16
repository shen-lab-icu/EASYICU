"""A failed model decision must retain the host's saved-question offer."""

from __future__ import annotations

import pytest

from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot.contracts import ToolExecutionContext
from tests.support.wait import wait_until
from tests.webserver.copilot.test_a_saved_numerical_question_offers_its_quick_card_after_the_source_click import (
    QUESTION,
    TRANSITION,
    confirm_source_and_continue,
    conversation as conversation,
    prompts,
)


def test_failed_first_turn_and_failed_offer_retry_keep_the_exact_question(conversation, monkeypatch):
    service, record, gateway = conversation
    service.replay_store.start_turn(
        session_id=record.session_id, project_id=record.project_id,
        job_id="first-failed", allowed_actions=[], user_message=QUESTION, message_origin="typed",
    )
    service.replay_store.finish_turn(
        session_id=record.session_id, project_id=record.project_id,
        job_id="first-failed", status="failed", assistant_entry_id="first-error",
    )
    original = gateway.request
    gateway.declines = True

    def fail_offer(method, params, **kwargs):
        state = original(method, params, **kwargs)
        if method == "session.prompt":
            state["transcript"] = [{
                "role": "assistant", "content": [], "stop_reason": "error",
                "entry_id": "offer-error", "error_code": "pi_model_provider_unavailable",
            }]
            state["quick_answer_offer_retry_available"] = True
        return state

    monkeypatch.setattr(gateway, "request", fail_offer)
    submitted = service.send_message(
        record.session_id, project_id=record.project_id, message=TRANSITION,
        message_intent="advance_after_data_source_confirmation", message_origin="host_prompt",
    )
    job_id = submitted["job_id"]
    wait_until(lambda: service_module.jobs.MANAGER.get(job_id).status == "failed", timeout=30)
    assert gateway.submitted == []
    failed = service._get_record(record.session_id)
    assert failed.last_turn_status == "failed"
    public = service._public_session(failed, gateway_state={"quick_answer_offer_retry_available": True})
    assert public["quick_answer_offer_retry_available"] is True
    assert public["quick_answers"] == []

    # The existing source-continue action sends a fresh transition, revalidates
    # source/typed wording, and grants neither a calculation nor a full run.
    monkeypatch.setattr(gateway, "request", original)
    gateway.declines = False
    confirm_source_and_continue(service, record)
    assert [prompt["quick_question"] for prompt in prompts(gateway)] == [QUESTION, QUESTION]
    assert len(gateway.submitted) == 1
    assert gateway.submitted[0]["details"]["quick_answer"]["proposal"]["status"] == "approval_required"
    latest = service._get_record(record.session_id)
    assert latest.last_turn_status == "done"
    assert service._public_session(latest)["quick_answer_offer_retry_available"] is False


@pytest.mark.parametrize("status", ["running", "done", "cancelled"])
def test_a_sidecar_retry_flag_does_not_override_the_host_turn_status(conversation, status):
    service, record, _gateway = conversation
    record.last_turn_status = status
    public = service._public_session(record, gateway_state={"quick_answer_offer_retry_available": True})
    assert public["quick_answer_offer_retry_available"] is False


def test_retry_rechecks_typed_question_eligibility(conversation):
    service, record, _gateway = conversation
    context = ToolExecutionContext(session=record)
    context.bind_question_wording_context([{"text": QUESTION, "message_origin": "typed"}], "")
    assert service._transition_quick_question(context) == QUESTION
    context.bind_question_wording_context([{"text": QUESTION, "message_origin": "model_option"}], "")
    assert service._transition_quick_question(context) == ""
