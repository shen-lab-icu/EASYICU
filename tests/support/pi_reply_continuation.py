"""Shared synthetic workspace-recovery gateway and session fixture."""

from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway


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



def setup_service(tmp_path, gateway_cls=ContinuationGateway):
    gateway = gateway_cls()
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
    service.replay_store.finish_turn(session_id=session_id, project_id="recovery-project", job_id="original-failed-job", status="failed", assistant_entry_id="failed-entry")
    return service, gateway, session_id



class RetryGateway(ContinuationGateway):
    def request(self, method, params, **kwargs):
        state = super().request(method, params, **kwargs)
        if self.ready:
            state["reply_continuation"].update(available=True, reason_code="pi_workspace_retry_available",
                recovery_kind="retry_unstarted", mutation_state="not_started", budget_state="available")
        if method == "session.retry_failed_turn":
            context = kwargs["tool_context"]
            assert context.allowed_actions == frozenset({"workspace_write"})
            assert context.quick_answer_submit is None and context.propose_multi_database is None
            assert set(params) == {"session_id", "source_error_entry_id"}
            kwargs["event_sink"]({"type": "run_start", "at": "2026-10-29T04:00:00Z"})
            kwargs["event_sink"]({"type": "message_end", "stop_reason": "stop", "at": "2026-10-29T04:00:01Z"})
            state["transcript"].append({"role": "assistant", "content": [{"type": "text", "text": "Saved the original requested page."}], "stop_reason": "stop", "entry_id": "retried-entry"})
            state["reply_continuation"] = {"available": False, "recovery_kind": "none", "reason_code": "pi_reply_continuation_no_failed_reply"}
        return state

