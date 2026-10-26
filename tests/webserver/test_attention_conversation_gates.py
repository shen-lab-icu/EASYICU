"""A conversation source-choice gate is published once until it is resolved."""

from easyicu.webserver.attention_service import attention_for_service
from easyicu.webserver.pi_copilot.contracts import PiSessionRecord
from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway


def test_source_choice_is_derived_at_save_without_repeating_on_replies(tmp_path):
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=FakeGateway())
    record = PiSessionRecord(
        session_id="conversation-1", project_id="project-1", last_message_job_id="message-1",
        last_turn_status="done", conversation_updated_at="2026-10-26T00:00:00Z",
        data_source_authorization={"status": "pending", "reason": "local_data_selection_required"},
    )
    service._write_records([record])
    feed = service.replay_store.attention
    item, = feed.page(client_id="browser-1")["items"]
    assert (item["kind"], item["reason_code"], item["run_id"]) == (
        "needs_review", "local_data_selection_required", "message-1",
    )
    feed.acknowledge(client_id="browser-1", sequences=[1])
    for status in ["running", "done"]:
        record.last_message_job_id = "message-2"
        record.last_turn_status = status
        service._write_records([record])
    assert attention_for_service(service).page(client_id="browser-1")["items"] == []
    restarted = PiCopilotService(store_path=service.store_path, gateway=FakeGateway())
    assert attention_for_service(restarted).page(client_id="browser-1")["items"] == []
    record.data_source_authorization.status = "confirmed"
    service._write_records([record])
    record.data_source_authorization.status = "pending"
    service._write_records([record])
    item, = feed.page(client_id="browser-1")["items"]
    assert item["sequence"] == 2


def test_new_conversation_or_finished_reply_alone_needs_no_attention(tmp_path):
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=FakeGateway())
    record = PiSessionRecord(session_id="conversation-1", project_id="project-1")
    service._write_records([record])
    record.last_message_job_id = "message-1"
    record.last_turn_status = "done"
    service._write_records([record])
    assert attention_for_service(service).page(client_id="browser-1")["items"] == []


def test_a_new_source_can_require_a_new_confirmation(tmp_path):
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=FakeGateway())
    record = PiSessionRecord(
        session_id="conversation-1", project_id="project-1", last_message_job_id="message-1",
        last_turn_status="done", conversation_updated_at="2026-10-26T00:00:00Z",
        data_source_authorization={"status": "pending", "source": {
            "identity_sha256": "1" * 64, "study_revision": 1,
        }},
    )
    service._write_records([record])
    record.data_source_authorization.source = record.data_source_authorization.source.model_copy(
        update={"identity_sha256": "2" * 64, "study_revision": 2},
    )
    service._write_records([record])
    assert len(service.replay_store.attention.page(client_id="browser-1")["items"]) == 2
