"""Host-owned, explanation-only continuation of an interrupted workspace reply."""
from __future__ import annotations

import threading
from typing import Any, Mapping

from easyicu.webserver import jobs

from .contracts import PiCopilotError, ToolExecutionContext, utc_now
from .gateway import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .projections import project_pi_replay_event, project_transcript, reject_sensitive_message
from .user_visible_text import sanitize_user_visible_text


def public_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        value = {}
    count = value.get("completed_tool_count")
    count = count if isinstance(count, int) and not isinstance(count, bool) else 0
    read_errors = value.get("read_error_count")
    read_errors = read_errors if isinstance(read_errors, int) and not isinstance(read_errors, bool) else 0
    reason = str(value.get("reason_code") or "pi_reply_continuation_unavailable")[:160]
    kind = value.get("recovery_kind")
    if kind not in {"retry_unstarted", "continue_explanation", "blocked", "none"}:
        # Older sidecars supported only this typed explanation capability.
        kind = "continue_explanation" if reason == "pi_reply_continuation_available" and count > 0 else "blocked"
        if reason in {"pi_reply_continuation_no_failed_reply", "pi_reply_continuation_workspace_only"}:
            kind = "none"
    mutation = value.get("mutation_state")
    if mutation not in {"not_started", "completed", "unconfirmed"}:
        mutation = "completed" if kind == "continue_explanation" else "unconfirmed"
    budget = value.get("budget_state")
    if budget not in {"available", "exhausted", "unverified"}:
        budget = "exhausted" if reason.endswith("_budget_exhausted") else "unverified"
    available = value.get("available") is True and kind in {"retry_unstarted", "continue_explanation"}
    if available and "budget_state" not in value and kind == "continue_explanation":
        budget = "available"
    if available and (budget != "available"
                      or (kind == "retry_unstarted" and mutation != "not_started")
                      or (kind == "continue_explanation" and mutation != "completed")):
        available, kind = False, "blocked"
    return {
        "available": available,
        "recovery_kind": kind,
        "mutation_state": mutation,
        "budget_state": budget,
        "reason_code": reason,
        "source_error_entry_id": str(value.get("source_error_entry_id") or "")[:160],
        "source_user_entry_id": str(value.get("source_user_entry_id") or "")[:160],
        "completed_tool_count": max(0, min(1000, count)),
        "read_error_count": max(0, min(1000, read_errors)),
    }


def _block(state: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {**state, "available": False, "recovery_kind": "blocked", "reason_code": reason}


def _connection_problem(service: Any, record: Any, gateway: Any) -> str | None:
    if not record.uses_unified_model_connection:
        return "pi_session_connection_unpinned"
    if record.research_provider.provider == "codex":
        expected = record.research_provider.account_session_sha256
        return None if expected and expected == getattr(gateway, "account_binding_sha256", None) else "pi_session_connection_changed"
    expected = record.conversation_connection_sha256
    if not expected:
        return "pi_session_connection_unpinned"
    try:
        current = service._conversation_connection_sha256(gateway)
    except Exception:
        return "pi_session_connection_unverified"
    return None if current == expected else "pi_session_connection_changed"


def _retry_actions(service: Any, record: Any, state: Mapping[str, Any]) -> frozenset[str]:
    replay = service.replay_store.snapshot(session_id=record.session_id, project_id=str(record.project_id))
    source = next((turn for turn in replay.get("turns", []) if turn.get("job_id") == record.last_message_job_id), None)
    if (not source or source.get("status") not in {"failed", "interrupted"}
            or record.last_turn_status not in {"failed", "interrupted"}
            or not state.get("source_error_entry_id")
            or source.get("assistant_entry_id") != state.get("source_error_entry_id")):
        raise PiCopilotError("pi_workspace_retry_authority_unavailable", "The original failed attempt cannot be verified.", status_code=409)
    return frozenset(source.get("allowed_actions") or ()) & frozenset(record.last_turn_allowed_actions) & {"workspace_write"}


def project_recovery_state(service: Any, record: Any, gateway_state: Mapping[str, Any]) -> dict[str, Any]:
    state = public_state(gateway_state.get("reply_continuation"))
    if record.agent_mode != "workspace":
        return {**state, "available": False, "recovery_kind": "none"}
    if record.last_turn_status in {"running", "cancelled"} or gateway_state.get("streaming"):
        return _block(state, "pi_session_busy" if record.last_turn_status == "running" or gateway_state.get("streaming") else "pi_workspace_recovery_cancelled")
    if (record.last_turn_status in {"failed", "interrupted"} and state["recovery_kind"] == "none"):
        # A late SDK stop after a host timeout does not rewrite that failed job
        # into a resumable success. The host needs a confirmed terminal state.
        return _block(state, "pi_workspace_recovery_terminal_state_unconfirmed")
    if state["available"]:
        if record.last_turn_status not in {"failed", "interrupted"}:
            return _block(state, "pi_workspace_recovery_terminal_state_unconfirmed")
        gateway = service._conversation_gateway(record)
        problem = _connection_problem(service, record, gateway)
        if problem:
            return _block(state, problem)
        if state["recovery_kind"] == "retry_unstarted":
            try:
                _retry_actions(service, record, state)
            except PiCopilotError as exc:
                return _block(state, exc.code)
    return state


def validation_context(service: Any, record: Any, state: Mapping[str, Any]) -> dict[str, str]:
    """Only current, exact-byte QA can be named in the resumed explanation."""
    resources = state.get("completed_resources")
    if (state.get("resources_complete") is not True or not isinstance(resources, list) or not resources
            or state.get("completed_resource_total") != len(resources) or len(resources) > 8):
        return {"status": "not_checked", "summary": "No exact artifact validation receipt is available."}
    receipts = []
    for resource in resources[:8]:
        if not isinstance(resource, Mapping):
            return {"status": "not_checked", "summary": "Artifact identity is incomplete."}
        file, sha256 = str(resource.get("file") or ""), str(resource.get("sha256") or "")
        try:
            service.workspace.preview_file(str(record.project_id), file, checked_sha256=sha256)
            receipt = service.runtime_qa.read(str(record.project_id), file, sha256)
        except Exception:
            return {"status": "not_checked", "summary": "The saved artifact has changed or has no current runtime acceptance receipt."}
        if not isinstance(receipt, Mapping):
            return {"status": "not_checked", "summary": "No current runtime acceptance receipt is available."}
        receipts.append((file, receipt))
    failed = [file for file, receipt in receipts if receipt.get("status") == "failed"]
    if failed:
        return {"status": "failed", "summary": "Current runtime acceptance failed for: " + ", ".join(failed)}
    if all(receipt.get("status") == "complete" for _, receipt in receipts):
        return {"status": "passed", "summary": "Current runtime acceptance receipts passed for the exact saved artifacts."}
    return {"status": "not_checked", "summary": "Current runtime acceptance is partial or incomplete."}


def continue_reply(service: Any, session_id: str, *, project_id: str, source_error_entry_id: str) -> dict[str, Any]:
    return _recover_reply(service, session_id, project_id=project_id, source_error_entry_id=source_error_entry_id,
                          recovery_kind="continue_explanation")


def retry_unstarted_reply(service: Any, session_id: str, *, project_id: str, source_error_entry_id: str) -> dict[str, Any]:
    return _recover_reply(service, session_id, project_id=project_id, source_error_entry_id=source_error_entry_id,
                          recovery_kind="retry_unstarted")


def _recover_reply(service: Any, session_id: str, *, project_id: str, source_error_entry_id: str, recovery_kind: str) -> dict[str, Any]:
    record = service._scoped_record(session_id, project_id=project_id)
    if record.agent_mode != "workspace":
        raise PiCopilotError("pi_reply_continuation_workspace_only", "Only a workspace reply explanation can be continued.", status_code=409)
    service._provider_gate(external_llm_opt_in=record.external_llm_opt_in, provider=record.research_provider.provider)
    from .tools import require_copilot_data_clearance

    require_copilot_data_clearance(ToolExecutionContext(
        session=record, data_handling=service._data_handling_for_session(record),
        note_credentialed_history=lambda: service._note_credentialed_history(record.session_id),
    ))
    with service._lock:
        if session_id in service._busy_sessions or session_id in service._dispatching_followups:
            raise PiCopilotError("pi_session_busy", "This session already has active work.", status_code=409)
        service._busy_sessions.add(session_id)
    try:
        state = service._ensure_open(record)
        continuation = project_recovery_state(service, record, state)
        if not continuation.get("available"):
            raise PiCopilotError(str(continuation.get("reason_code") or "pi_reply_continuation_unavailable"), "This reply cannot be continued safely.", status_code=409)
        if str(continuation.get("source_error_entry_id") or "") != str(source_error_entry_id):
            raise PiCopilotError("pi_reply_continuation_stale", "The failed reply is no longer current.", status_code=409)
        if continuation["recovery_kind"] != recovery_kind:
            raise PiCopilotError("pi_workspace_recovery_action_mismatch", "Use the recovery action verified for this failed attempt.", status_code=409)
        gateway = service._conversation_gateway(record, refresh_account=True)
        service._admit_new_work(gateway=gateway)
        qa = validation_context(service, record, state.get("reply_continuation") or {})
        requested_actions = _retry_actions(service, record, continuation) if recovery_kind == "retry_unstarted" else frozenset()
        tool_context = ToolExecutionContext(
            session=record.model_copy(deep=True), data_handling=service._data_handling_for_session(record),
            allowed_actions=requested_actions,
            note_credentialed_history=lambda: service._note_credentialed_history(record.session_id),
            authority_validator=lambda binding: service._binding_stale_details(binding, project_id=record.project_id),
            workspace=service.workspace, extension_registry=service.extension_registry, runtime_qa_store=service.runtime_qa,
        ) if recovery_kind == "retry_unstarted" else None
    except Exception:
        with service._lock:
            service._busy_sessions.discard(session_id)
        raise
    source_job_id = record.last_message_job_id
    gate = threading.Event()

    def runner(job: jobs.Job) -> dict[str, Any]:
        gate.wait()
        status = "failed"
        reply = ""
        assistant_entry_id = None
        try:
            started = service._get_record(session_id)
            started.active_message_job_id = job.id
            started.last_message_job_id = job.id
            started.last_turn_status = "running"
            started.last_turn_allowed_actions = sorted(requested_actions)
            started.conversation_updated_at = utc_now()
            service._save_record(started)
            service.replay_store.start_turn(session_id=session_id, project_id=project_id, job_id=job.id,
                allowed_actions=sorted(requested_actions), message_origin="workspace_retry" if recovery_kind == "retry_unstarted" else "reply_continuation", source_job_id=source_job_id,
                source_error_entry_id=source_error_entry_id,
                **({"recovery_kind": "retry_unstarted"} if recovery_kind == "retry_unstarted" else {}))

            def emit(event: dict[str, Any]) -> None:
                if event.get("type") == "thinking_delta":
                    return
                if event.get("type") == "text_delta":
                    event = {**event, "delta": sanitize_user_visible_text(str(event.get("delta") or ""), fragment=True)}
                job.emit({"type": "pi_event", "event": event})
                projected = project_pi_replay_event(event)
                if projected is not None:
                    service.replay_store.append_event(session_id=session_id, project_id=project_id, job_id=job.id, event=projected)

            if job.cancel_requested:
                raise RuntimeError("pi_message_cancelled_before_delivery")
            # Explanation recovery gets no executor. An unstarted retry gets
            # only the original routine workspace permission; all scientific
            # callbacks and one-shot grants remain absent.
            params = {"session_id": session_id, "source_error_entry_id": source_error_entry_id}
            if recovery_kind == "continue_explanation":
                params["validation_context"] = qa
            completed = gateway.request("session.retry_failed_turn" if recovery_kind == "retry_unstarted" else "session.continue_reply",
                params, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS, event_sink=emit, tool_context=tool_context)
            transcript = project_transcript(completed.get("transcript"))
            terminal = next((row for row in reversed(transcript) if row.get("role") == "assistant"), {})
            assistant_entry_id = str(terminal.get("entry_id") or "") or None
            if terminal.get("stop_reason") == "error":
                raise RuntimeError(str(terminal.get("error_code") or "pi_model_provider_error"))
            if terminal.get("stop_reason") != "stop":
                raise RuntimeError("pi_reply_continuation_incomplete")
            reply = "\n".join(str(part.get("text") or "") for part in terminal.get("content", []) if part.get("type") == "text")[:8192]
            try:
                reject_sensitive_message(reply)
            except PiCopilotError:
                reply = ""
            status = "cancelled" if job.cancel_requested else "done"
        finally:
            try:
                finished = service._get_record(session_id)
                if finished.active_message_job_id == job.id:
                    finished.active_message_job_id = None
                    finished.last_turn_status = "cancelled" if job.cancel_requested else status
                    service._save_record(finished)
                service.replay_store.finish_turn(session_id=session_id, project_id=project_id, job_id=job.id,
                                                status="cancelled" if job.cancel_requested else status, assistant_reply=reply,
                                                assistant_entry_id=assistant_entry_id)
            finally:
                # A failed persistence write must never strand the in-memory
                # admission lock or make the completed job look still active.
                with service._lock:
                    if service._active_message_jobs.get(session_id) == job.id:
                        service._active_message_jobs.pop(session_id, None)
                    service._busy_sessions.discard(session_id)
            if session_id in service._pending_retirements:
                service._flush_pending_retirement(session_id)
            if status == "done" and not job.cancel_requested:
                service._schedule_followups(session_id, job)
            else:
                service._cancel_pending_followups(session_id, error="previous_turn_failed")
        return {"session": service._public_session(service._get_record(session_id), gateway_state=completed),
                "message_status": status, "source_job_id": source_job_id, "source_error_entry_id": source_error_entry_id}

    try:
        job = jobs.MANAGER.submit("pi-copilot-workspace-retry" if recovery_kind == "retry_unstarted" else "pi-copilot-reply-continuation", runner)
        with service._lock:
            service._active_message_jobs[session_id] = job.id
    except jobs.JobCapacityError as exc:
        with service._lock:
            service._busy_sessions.discard(session_id)
        raise PiCopilotError("job_capacity_reached", "EasyICU cannot continue another reply right now.", status_code=429,
                             details={"max_running": exc.max_running, "running": exc.running}) from exc
    except Exception:
        with service._lock:
            service._busy_sessions.discard(session_id)
        raise
    gate.set()
    return {"ok": True, "job_id": job.id, "kind": job.kind, "status": job.status,
            "session_id": session_id, "source_job_id": source_job_id, "source_error_entry_id": source_error_entry_id}
