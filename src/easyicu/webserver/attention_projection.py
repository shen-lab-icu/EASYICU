"""Metadata-only attention derived from the existing conversation/job owners."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import Any, Literal, Mapping

AttentionKind = Literal["needs_review", "completed", "failed", "cancelled"]
IDENTIFIER = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}"
CLIENT_IDENTIFIER = r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}"
MAX_SEQUENCE = 2**53 - 1


def identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(IDENTIFIER, value) else None


def code(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,159}", value) else None


def timestamp(value: Any) -> str | None:
    if type(value) not in {float, int} or not math.isfinite(value) or not 0 < value < 253402300799:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class AttentionTransition:
    project_id: str
    conversation_id: str
    run_id: str
    kind: AttentionKind
    reason_code: str
    timestamp: str
    # Internal identity of the owner receipt, never an independently writable state.
    source_key: str


def job_transition(project_id: str, conversation_id: str, job: Mapping[str, Any]) -> AttentionTransition | None:
    """Project a terminal child job, including append runs and extraction.

    ``project_job`` owns result readiness and stable failure codes. In
    particular a completed reply or a recovered lifecycle receipt proves no
    scientific result. Jobs before a scientific run exists use their job ID.
    """
    from easyicu.webserver.pi_copilot.conversation_status import research_job_state

    job_id = identifier(job.get("job_id"))
    at = timestamp(job.get("finished_at_epoch"))
    status = job.get("status")
    if not (identifier(project_id) and identifier(conversation_id) and job_id and at):
        return None
    if status not in {"done", "failed", "cancelled", "interrupted"}:
        return None
    kind: AttentionKind
    if status == "cancelled":
        kind, reason = "cancelled", code(job.get("cancel_reason_code")) or "user_cancelled"
    elif status in {"failed", "interrupted"}:
        kind, reason = "failed", code(job.get("error_code")) or code(job.get("gate_reason_code")) or "job_failed"
    elif job.get("result_omitted") is True:
        return None
    elif isinstance(trial := job.get("target_trial_compile"), Mapping):
        if trial.get("status") == "compiled":
            kind, reason = "needs_review", "target_trial_review"
        else:
            kind, reason = "failed", code(trial.get("reason_code")) or "target_trial_compile_failed"
    elif research_job_state(job) == "needs_user":
        kind, reason = "needs_review", code(job.get("gate_reason_code")) or "human_review_required"
    elif job.get("analysis_results_available") is True or job.get("report_revision_ready") is True:
        kind, reason = "completed", "result_ready"
    elif job.get("gate_status") == "blocked":
        kind, reason = "failed", code(job.get("gate_reason_code")) or "research_gate_blocked"
    elif job.get("kind") == "extract":
        kind, reason = "completed", "extraction_ready"
    else:
        return None
    return AttentionTransition(
        project_id, conversation_id, identifier(job.get("run_id")) or job_id,
        kind, reason, at, f"job:{job_id}:{kind}",
    )


def source_gate(record: Mapping[str, Any]) -> tuple[str, AttentionTransition | None] | None:
    """A source-choice edge; an in-progress reply does not reopen its gate.

    None leaves the delivery cursor alone; an empty fingerprint clears it.
    The fingerprint contains only the gate code and the source's opaque digest.
    """
    from easyicu.webserver.pi_copilot.conversation_status import conversation_status

    project, conversation = identifier(record.get("project_id")), identifier(record.get("session_id"))
    if not project or not conversation:
        return None
    choice = record.get("data_source_authorization") or {}
    if choice.get("status") not in {"pending", "selection_in_progress"}:
        return "", None
    state = conversation_status(
        message_status=record.get("last_turn_status"), message_running=bool(record.get("active_message_job_id")),
        message_at=None, research_state=None, research_at=None, data_choice_status=choice.get("status"),
    )
    job_id = identifier(record.get("last_message_job_id"))
    if state != "needs_user" or not job_id:
        return None
    try:
        at = datetime.fromisoformat(str(record.get("conversation_updated_at") or record.get("updated_at")).replace("Z", "+00:00"))
        if at.tzinfo is None:
            return None
    except ValueError:
        return None
    reason = code(choice.get("reason")) or "data_source_confirmation_required"
    digest = (choice.get("source") or {}).get("identity_sha256")
    source_id = digest if isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest) else "unbound"
    return f"{reason}:{source_id}", AttentionTransition(
        project, conversation, job_id, "needs_review", reason,
        at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), "source_gate",
    )
