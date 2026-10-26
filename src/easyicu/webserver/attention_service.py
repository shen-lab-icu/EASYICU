"""Repair the delivery journal from this instance's bounded conversation replay."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from easyicu.webserver.attention_projection import AttentionTransition
from easyicu.webserver.attention_store import AttentionStore, AttentionUnavailable
from easyicu.webserver.hosted import require_instance_path

if TYPE_CHECKING:
    from easyicu.webserver.pi_copilot.service import PiCopilotService


def attention_for_service(service: PiCopilotService) -> AttentionStore:
    """Read only instance-owned metadata, without launching work or pollers.

    Reconciliation closes the crash window between the authoritative replay
    replacement and attention insertion. Each transition's unique receipt key
    preserves both its sequence and its ACK across repeated reconciliation.
    """
    from easyicu.webserver import jobs
    from easyicu.webserver.pi_copilot.projections import project_job

    require_instance_path(service.store_path)
    replay = service.replay_store
    require_instance_path(replay.root)
    require_instance_path(replay.attention.path)
    with service._lock:
        records = service._read_records()
        replay.attention.observe_sessions([record.model_dump(mode="json") for record in records])
    for record in records:
        if not record.project_id:
            continue
        _reconcile_append(service, record)
        with replay._lock:
            payload = replay._read(record.session_id, record.project_id)
            # A job can settle after its conversation closed or before its
            # replay watcher archived it. Consult only jobs the replay binds.
            referenced = {
                str(turn.get("child_job_id") or "") for turn in payload["turns"]
            } | {
                str(event.get("job_id") or "") for turn in payload["turns"]
                for event in turn.get("events", [])
            }
            archived = {row["job_id"]: row for row in payload["child_jobs"]}
            for job_id in sorted(referenced - {""}):
                if archived.get(job_id, {}).get("status") in {"done", "failed", "cancelled"}:
                    continue
                child = jobs.MANAGER.get(job_id)
                if child is not None and child.status in {"done", "failed", "cancelled"}:
                    replay.archive_child_job(
                        session_id=record.session_id, project_id=record.project_id,
                        job=project_job(child.snapshot()),
                    )
            replay.attention.observe_replay(replay._read(record.session_id, record.project_id))
    return replay.attention


def publish_append_review(service: PiCopilotService, *, project_id: str, session_id: str, checkpoint, strict: bool = False) -> None:
    """An append proposal pauses synchronously, before there is a child job.

    The append owner has already checked project/conversation ownership and
    committed the checkpoint. A retried proposal returns the same checkpoint,
    so this immutable receipt also deduplicates repeated clicks.
    """
    if checkpoint.state != "pending":
        return
    try:
        service.replay_store.attention.record([AttentionTransition(
            project_id, session_id, checkpoint.run_id, "needs_review", "human_plan_review_required",
            checkpoint.created_at.replace("+00:00", "Z"), f"append_review:{checkpoint.run_id}",
        )])
    except AttentionUnavailable:
        if strict:
            raise
        logging.getLogger(__name__).error("attention_store_unavailable")


def _reconcile_append(service: PiCopilotService, record) -> None:
    """Repair a synchronous proposal from the checkpoint its binding names."""
    from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpointError
    from easyicu.webserver.research_append_run import append_run_for_id

    if not str(record.binding.run_id or "").startswith("append-"):
        return
    try:
        child = append_run_for_id(record.binding.run_id)
    except (OSError, ValueError, HumanReviewCheckpointError):
        # An expired/unreadable checkpoint grants no new review alert.
        return
    if child is not None and child.record["study"]["id"] == record.binding.study_context_id:
        publish_append_review(service, project_id=record.project_id, session_id=record.session_id,
                              checkpoint=child.checkpoint, strict=True)
