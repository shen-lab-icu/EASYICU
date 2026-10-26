"""Conversation and HTTP adapters for one combined review and durable job."""
from __future__ import annotations

import re
import uuid

from easyicu.research_agent.canonical_json import canonical_sha256
from . import jobs, state_paths
from .agent_review_recovery import _write
from .research_append_source import read_json
from .research_multi_contracts import MultiDatabaseError, MultiDatabaseRequest
from .research_multi_pipeline import PipelineDatabaseOwner
from .research_multi_run import MultiDatabaseRun

SCHEMA = "easyicu.multi_database_dispatch/1"


def batch_root(batch_id):
    if not re.fullmatch(r"[a-f0-9]{64}", batch_id):
        raise MultiDatabaseError("multi_database_request_invalid")
    return state_paths.state_root() / "multi-database" / batch_id


def open_batch(service, *, batch_id, project_id, session_id):
    root = batch_root(batch_id)
    request = MultiDatabaseRequest.model_validate(read_json(root / "admission.json"))
    if request.project_id != project_id or request.session_id != session_id:
        raise MultiDatabaseError("multi_database_study_mismatch")
    service.host_action_session(session_id, project_id=project_id, study_context_id=request.study_context_id)
    return MultiDatabaseRun(root, PipelineDatabaseOwner(service, request, root)), request


def _runner(service, descriptor):
    batch, request = open_batch(service, **{key: descriptor[key] for key in ("batch_id", "project_id", "session_id")})

    def run(job):
        if descriptor["phase"] == "prepare":
            card = batch.prepare(request, job=job)
            if any(row.egress and row.egress.clearance == "provider_required" for row in card.targets):
                service._note_credentialed_history(request.session_id)
            from .attention_projection import AttentionTransition
            from .pi_copilot.contracts import utc_now

            service.replay_store.attention.record([AttentionTransition(
                request.project_id, request.session_id, "multi-" + card.batch_id[:32],
                "needs_review", "multi_database_review_required", utc_now(), "multi_review:" + card.digest)])
            return {"card": card.model_dump(mode="json"), "card_sha256": card.digest}
        batch.execute(job=job)
        from .research_multi_results import combined_answer

        return combined_answer(batch)

    return run


def recover_multi_runner(descriptor, job):
    from .pi_copilot import get_pi_copilot_service

    if descriptor.get("schema_version") != SCHEMA or descriptor.get("phase") not in {"prepare", "execute"}:
        raise MultiDatabaseError("multi_database_request_invalid")
    return _runner(get_pi_copilot_service(), descriptor)


def _dispatch(service, request, batch_id, phase):
    from .durable_research import current_service

    descriptor = dict(schema_version=SCHEMA, batch_id=batch_id, phase=phase, project_id=request.project_id,
                      session_id=request.session_id, study_context_id=request.study_context_id)
    runner = _runner(service, descriptor)
    durable = current_service()
    if durable is None:
        job = jobs.MANAGER.submit("agent-run", runner)
    else:
        job = durable.reserve(descriptor)
        durable.activate(job.id, runner)
    return dict(job_id=job.id, kind=job.kind, status=job.status, multi_database_batch_id=batch_id)


def propose(service, request: MultiDatabaseRequest, *, _message_job_id=None):
    service.host_action_session(request.session_id, project_id=request.project_id, study_context_id=request.study_context_id)
    record = service._scoped_record(request.session_id, project_id=request.project_id)
    with service._lock:
        if _message_job_id is not None:
            # Only the host callback bound to the currently executing Pi turn
            # may prepare a proposal while that conversation is busy.
            if service._active_message_jobs.get(request.session_id) != _message_job_id:
                raise MultiDatabaseError("multi_database_turn_expired")
        elif request.session_id in service._busy_sessions or request.session_id in service._dispatching_followups:
            raise MultiDatabaseError("pi_session_busy")
    if not record.external_llm_opt_in or record.agent_mode != "research":
        raise MultiDatabaseError("multi_database_opt_in_required")
    batch_id = canonical_sha256([request.model_dump(mode="json"), uuid.uuid4().hex])
    _write(batch_root(batch_id) / "admission.json", request.model_dump(mode="json"))
    from .pi_copilot.contracts import utc_now

    service.replay_store.record_multi_database_batch(
        session_id=request.session_id, project_id=request.project_id,
        reference=dict(batch_id=batch_id, study_context_id=request.study_context_id,
                       source_run_id=request.source_run_id, message_job_id=_message_job_id, created_at=utc_now()),
    )
    return _dispatch(service, request, batch_id, "prepare")


def approve(service, *, project_id, session_id, batch_id, card_sha256):
    from .agent_pipeline_runs import server_reviewer_identity

    batch, request = open_batch(service, batch_id=batch_id, project_id=project_id, session_id=session_id)
    batch.approve(card_sha256=card_sha256, reviewer=server_reviewer_identity())
    return _dispatch(service, request, batch_id, "execute")


def status(service, *, project_id, session_id, batch_id):
    batch, _ = open_batch(service, batch_id=batch_id, project_id=project_id, session_id=session_id)
    if not (batch.root / "card.json").exists():
        return {"batch_id": batch_id, "status": "preparing"}
    _, card, _ = batch.load()
    from .research_multi_results import combined_answer

    return {"card": card.model_dump(mode="json"), "card_sha256": card.digest, "status": batch.state()["status"],
            "answer": combined_answer(batch)}


def cancel(service, *, project_id, session_id, batch_id, database):
    batch, _ = open_batch(service, batch_id=batch_id, project_id=project_id, session_id=session_id)
    batch.cancel(database)
    return status(service, project_id=project_id, session_id=session_id, batch_id=batch_id)


def dispatch_followup(service, *, project_id, session_id, run_id, suggestion_id, expected_source_sha256, targets=()):
    from .research_followups import result_followups
    from .research_multi_contracts import DatabaseTarget

    offered = result_followups(service, project_id=project_id, run_id=run_id)
    selected = next((row for row in offered["suggestions"] if row["suggestion_id"] == suggestion_id
                     and row["target_action"] == "other_database"), None)
    if offered["source_run_sha256"] != expected_source_sha256 or selected is None:
        raise MultiDatabaseError("multi_database_followup_changed")
    row = service._research_run_row(project_id, run_id)
    target = DatabaseTarget(database=selected["delta"]["capability_decision"]["database"])
    chosen = tuple(DatabaseTarget.model_validate(item) for item in targets) or (target,)
    if target.database not in {item.database for item in chosen}:
        raise MultiDatabaseError("multi_database_followup_changed")
    return propose(service, MultiDatabaseRequest(project_id=project_id, session_id=session_id,
        study_context_id=row["study_id"], source_run_id=run_id, targets=chosen))
