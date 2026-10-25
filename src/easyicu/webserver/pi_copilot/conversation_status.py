"""A conversation's state in the project rail, from its latest job.

The rail used to read the state of the conversation's last model turn, so a
conversation whose research job failed after a finished reply read 「已完成」.
The state now comes from the latest job the conversation ran -- a model turn
or a research job it started -- and its terminal state:

==================  =====================================
latest job          rail state
==================  =====================================
running             ``running``     (进行中 / Running)
waiting for you     ``needs_user``  (待确认 / Needs you)
failed / stopped    ``stopped``     (已停止 / Stopped)
completed reply     ``replied``     (已回复 / Replied)
completed task      ``done``        (任务已结束 / Task ended)
==================  =====================================

A research job waits for the researcher when it paused for plan review; a
finished reply waits when the conversation still asks which data to use.  A
finished reply neither clears a plan review the researcher still owes nor
reopens a research job that stopped: only the next research job does.  The
rail is a metadata-only index, so the research job's state is persisted on the
session record when the job starts and settles (``latest_research_job``).
Its legacy ``done`` state contains no phase receipt: it must not be presented
as analysis completion. ``replied`` is a display projection, not a new stored
research-job state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Optional

ConversationStatus = Literal["running", "needs_user", "stopped", "done", "replied"]

_STOPPED = frozenset({"failed", "cancelled", "interrupted"})
_DATA_CHOICE_PENDING = frozenset({"pending", "selection_in_progress"})


def research_job_state(job: Mapping[str, Any]) -> ConversationStatus:
    """The rail state of one research job, from its browser projection.

    ``job`` is ``projections.project_job`` output, or a bare status for a job
    this process no longer holds.
    """

    status = str(job.get("status") or "")
    if status == "running":
        return "running"
    if status != "done":
        # failed, cancelled, interrupted -- or a state no job reports.
        return "stopped"
    if job.get("human_review_pending") is True:
        return "needs_user"
    # A run whose gate is blocked without a review to answer stopped short.
    if str(job.get("gate_status") or "") == "blocked":
        return "stopped"
    return "done"


def conversation_status(
    *,
    message_status: Optional[str],
    message_running: bool,
    message_at: Optional[str],
    research_state: Optional[str],
    research_at: Optional[str],
    data_choice_status: Optional[str] = None,
) -> Optional[ConversationStatus]:
    """The rail state of one conversation; ``None`` before its first job."""

    if message_running or message_status == "running" or research_state == "running":
        return "running"
    message_state: Optional[ConversationStatus] = None
    if message_status in _STOPPED:
        message_state = "stopped"
    elif message_status == "done":
        message_state = (
            "needs_user" if data_choice_status in _DATA_CHOICE_PENDING else "replied"
        )
    research: Optional[ConversationStatus] = (
        research_state  # type: ignore[assignment]
        if research_state in {"needs_user", "stopped", "done"}
        else None
    )
    if research is None:
        return message_state
    if message_state is None:
        return research
    # A reply (say, asking why the run failed) does not settle the study's
    # research job; the rail keeps its state until the next research job.
    if message_state == "replied":
        return research
    # ISO-8601 UTC stamps of one writer compare as text.
    return research if str(research_at or "") >= str(message_at or "") else message_state


def research_job_state_at(job: Mapping[str, Any], state: str) -> Optional[str]:
    """When ``state`` began: the job's start while it runs, else its end."""

    epoch = job.get("created_at_epoch" if state == "running" else "finished_at_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or epoch <= 0:
        return None
    return (
        datetime.fromtimestamp(float(epoch), timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def later_research_job(current: Any, noted: Any) -> Any:
    """The research job the rail follows after ``noted`` is observed.

    The latest-started job wins.  A job's own later state replaces its
    earlier one, except that a late sight of it running never reopens a job
    that already ended.
    """

    if current is None:
        return noted
    if noted.job_id == current.job_id:
        if current.state != "running" and noted.state == "running":
            return current
        return noted
    return noted if noted.started_at >= current.started_at else current


def session_rail_status(
    record: Any, *, job_is_running: Optional[Callable[[str], bool]] = None
) -> Optional[ConversationStatus]:
    """The rail state of one ``PiSessionRecord``.

    ``job_is_running`` reports whether this process still runs a job.  A
    research job recorded as running that the process no longer runs ended
    without a terminal record (the Web process restarted): it stopped.
    """

    authorization = getattr(record, "data_source_authorization", None)
    research = getattr(record, "latest_research_job", None)
    research_state = getattr(research, "state", None)
    research_at = getattr(research, "at", None)
    if (
        research_state == "running"
        and job_is_running is not None
        and not job_is_running(research.job_id)
    ):
        # It ended after anything the conversation recorded since it began.
        research_state = "stopped"
        research_at = max(str(research_at), str(record.conversation_updated_at or ""))
    return conversation_status(
        message_status=record.last_turn_status,
        message_running=bool(record.active_message_job_id),
        message_at=record.conversation_updated_at,
        research_state=research_state,
        research_at=research_at,
        data_choice_status=getattr(authorization, "status", None),
    )


__all__ = [
    "ConversationStatus",
    "conversation_status",
    "later_research_job",
    "research_job_state",
    "research_job_state_at",
    "session_rail_status",
]
