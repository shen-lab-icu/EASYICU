"""Bounded host recovery of a job through its existing checkpoint owners.

Transport retries remain the provider's policy. This owner waits for service
recovery, accounts probes against the original task, and records system actions.
It never constructs a plan, review decision, or execution grant.
"""

from __future__ import annotations

import json
import re
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopError
from easyicu.research_agent.providers.clients import safe_provider_http_status_code, transient_transport_failure
from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.reporting.writer_only_migration import WriterOnlyMigrationError
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

RECOVERY_WINDOW_SECONDS = 1800.0
PROBE_INTERVAL_SECONDS = 120.0
MAX_AUTOMATIC_RESUMES = 2
#: The system-action receipt each wrapper of a recovered job keeps.
RECEIPT_NAME = "provider_recovery.json"
RECEIPT_SCHEMA = "easyicu.provider_job_recovery/1"
_RECEIPT_MAX_BYTES = 64 * 1024
_CAUSE_CODE = re.compile(r"[a-z][a-z0-9_]{0,79}")
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")


class ProviderRecoveryExhausted(ResearchPipelineRunError):
    def __init__(self, reason: str) -> None:
        super().__init__(
            "research_pipeline_provider_recovery_exhausted",
            "Automatic provider recovery reached its bound; the verified checkpoint is preserved.",
            details={"reason_code": reason},
        )


def transient_provider_cause(exc: BaseException) -> str | None:
    """Only typed transport causes qualify; implicit exception context does not."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ProviderHardStopError):
            return None
        status = safe_provider_http_status_code(exc)
        if status is not None:
            return "provider_http_error" if 500 <= status <= 599 else None
        transport = transient_transport_failure(exc)
        if transport is not None:
            return str(transport["reason_code"])
        if getattr(exc, "easyicu_safe_diagnostic", None) is not None:
            return None
        if isinstance(exc, WriterOnlyMigrationError) and exc.code in {
            "WRITER_ONLY_REPAIR_FAILED_PRIOR_PRESERVED",
            "WRITER_ONLY_AUTHORITY_REPAIR_FAILED_PRIOR_PRESERVED",
        }:
            exc = exc.__cause__
            continue
        # Only transport-neutral host envelopes may expose a causal failure.
        if not isinstance(exc, ResearchPipelineRunError) or exc.code not in {
            "research_pipeline_execution_failed", "research_pipeline_approved_run_failed",
            "research_pipeline_review_resume_failed", "research_pipeline_provider_timeout",
            "research_pipeline_planner_provider_unavailable",
        }:
            return None
        exc = exc.__cause__
    return None


@dataclass(frozen=True)
class CheckpointContinuation:
    checkpoint: Mapping[str, Any]
    run: Callable[[Any], Any]


class RecoveryJob:
    """A continuation is still the same JobManager job and cancellation owner."""

    def __init__(self, recovery: ProviderJobRecovery, job_id: str) -> None:
        self.provider_recovery = recovery
        self.id = job_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider_recovery.job, name)


class _ProbeTask:
    """The existing transport observes the smaller job-recovery deadline."""

    def __init__(self, recovery: ProviderJobRecovery) -> None:
        self.recovery = recovery

    def assert_active(self) -> float:
        return self.recovery.remaining()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.recovery.task, name)


class ProviderJobRecovery:
    def __init__(self, job: Any) -> None:
        self.job = job
        self.task: Any = None
        self.client: Any = None
        self.wrapper_dir: Path | None = None
        self.wrappers: set[Path] = set()
        self.deadline: float | None = None
        self.resumes: list[dict[str, Any]] = []
        self.stop_reason: str | None = None
        self.execution_resume_run_id: str = ""
        self.continuation_refusal: dict[str, str] | None = None

    def record_continuation_refusal(self, code: str) -> None:
        """Keep a safe explanation even when no automatic attempt was spent."""
        self.continuation_refusal = {
            "code": code,
            "message": "The saved preparation could not be verified for continuation; prepare the same reviewed candidate again.",
        }
        self.persist()
        self.job.emit({"type": "progress", "step": "provider_continuation_refused", "actor": "system", "reason": code})

    def bind(self, *, wrapper_dir: Path, task: Any, client: Any) -> None:
        if self.task is not None and task is not self.task:
            raise ValueError("automatic recovery cannot replace its provider task")
        self.wrapper_dir, self.task, self.client = wrapper_dir, task, client
        self.wrappers.add(wrapper_dir)
        if self.deadline is not None:
            self.persist()

    def persist(self) -> None:
        record = {
            "schema_version": RECEIPT_SCHEMA,
            "job_id": self.job.id, "window_seconds": RECOVERY_WINDOW_SECONDS,
            "max_automatic_resumes": MAX_AUTOMATIC_RESUMES,
            "probe_interval_seconds": PROBE_INTERVAL_SECONDS,
            "automatic_resumes": self.resumes, "stop_reason": self.stop_reason,
            "continuation_refusal": self.continuation_refusal,
        }
        for root in self.wrappers:
            path = root / RECEIPT_NAME
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)

    def remaining(self) -> float:
        if getattr(self.job, "cancel_requested", False):
            raise ResearchPipelineRunError("research_pipeline_cancelled", "The research job was cancelled.")
        remaining = self.task.assert_active()
        window = self.deadline - time.monotonic()
        if window <= 0:
            raise ProviderRecoveryExhausted("recovery_window_exhausted")
        return min(remaining, window)

    def wait_for_provider(self, cause: str) -> None:
        # The wait names the continuation it waits for, so the conversation
        # can say which of the bounded attempts this is (n/limit).
        self.job.emit({
            "type": "progress", "step": "provider_recovery_wait", "actor": "system", "reason": cause,
            "current": len(self.resumes) + 1, "total": MAX_AUTOMATIC_RESUMES,
        })
        probe = HardStopClient(self.client, role="provider_recovery_probe", task=_ProbeTask(self))
        while True:
            wait = min(PROBE_INTERVAL_SECONDS, self.remaining())
            # Cancellation and the original active wall clock continue binding.
            while wait > 0:
                seconds = min(5.0, wait, self.remaining())
                time.sleep(seconds)
                wait -= seconds
            self.remaining()
            try:
                probe.complete([LLMMessage(role="user", content="Reply OK.")], max_tokens=16, temperature=0)
                self.remaining()
                return
            except Exception as exc:
                if transient_provider_cause(exc) is None:
                    if isinstance(exc, (ProviderHardStopError, ResearchPipelineRunError)):
                        raise
                    raise ResearchPipelineRunError(
                        "research_pipeline_provider_recovery_probe_failed",
                        "The provider recovery probe stopped on a non-transient failure.",
                    ) from exc

    def run(self, operation: Callable[[Any], Any], select: Callable[[BaseException], CheckpointContinuation | None]) -> Any:
        try:
            while True:
                original_id = str(getattr(self.job, "id", ""))
                job_id = original_id if not self.resumes else f"{original_id}-auto-{len(self.resumes)}"
                try:
                    return operation(RecoveryJob(self, job_id))
                except Exception as exc:
                    cause = transient_provider_cause(exc)
                    if cause is None or self.task is None:
                        raise
                    continuation = select(exc)
                    if continuation is None:
                        raise
                    if self.deadline is None:
                        self.deadline = time.monotonic() + RECOVERY_WINDOW_SECONDS
                    if len(self.resumes) >= MAX_AUTOMATIC_RESUMES:
                        raise ProviderRecoveryExhausted("automatic_resume_limit_exhausted") from exc
                    self.persist()
                    self.wait_for_provider(cause)
                    # Revalidate after waiting: scientific scope or checkpoint
                    # bytes may have changed while the provider was unavailable.
                    current = select(exc)
                    if current is None or current.checkpoint != continuation.checkpoint:
                        raise exc
                    self.remaining()
                    event = {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "actor": "system", "cause_code": cause,
                        "checkpoint": dict(current.checkpoint),
                        "number": len(self.resumes) + 1,
                    }
                    self.resumes.append(event)
                    self.persist()
                    self.job.emit({
                        "type": "progress", "step": "provider_auto_resume",
                        "actor": "system", "reason": cause,
                        "current": event["number"], "total": MAX_AUTOMATIC_RESUMES,
                    })
                    operation = current.run
        except BaseException as exc:
            self.stop_reason = str(getattr(exc, "code", "") or type(exc).__name__)
            if self.stop_reason.startswith("research_pipeline_development_resume_"):
                self.record_continuation_refusal(self.stop_reason)
            if self.deadline is not None or self.continuation_refusal is not None:
                self.persist()
            # Failure handlers defer closing a transient call's task until
            # this job has either recovered or reached a terminal boundary.
            if self.task is not None:
                rows = self.task.ledger.snapshot().get("tasks", [])
                if any(row.get("task_id") == self.task.task_id and row.get("status") in {"running", "paused"} for row in rows):
                    self.task.finish(error=self.stop_reason)
            raise


def read_recovery_receipt(wrapper_dir: Path) -> dict[str, Any] | None:
    """The automatic continuations one wrapper's receipt records, path-free.

    Each continuation is a system action: the receipt names the job it
    continued, when, after which transport cause, and its number within the
    bound.  An unreadable, oversized, linked or malformed receipt is ``None``
    -- it is never read as somebody else's action.
    """

    path = Path(wrapper_dir) / RECEIPT_NAME
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _RECEIPT_MAX_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != RECEIPT_SCHEMA:
        return None
    job_id = str(payload.get("job_id") or "")
    limit = payload.get("max_automatic_resumes")
    rows = payload.get("automatic_resumes")
    if (
        not _JOB_ID.fullmatch(job_id)
        or type(limit) is not int
        or not 0 < limit <= 16
        or not isinstance(rows, list)
    ):
        return None
    resumes: list[dict[str, Any]] = []
    for row in rows[:limit]:
        if not isinstance(row, dict) or row.get("actor") != "system":
            return None
        number = row.get("number")
        cause = str(row.get("cause_code") or "")
        try:
            at = datetime.fromisoformat(str(row.get("at") or ""))
        except ValueError:
            return None
        if type(number) is not int or not 0 < number <= limit or at.tzinfo is None:
            return None
        resumes.append({
            "at": at.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "number": number,
            "cause_code": cause if _CAUSE_CODE.fullmatch(cause) else "",
        })
    return {"job_id": job_id, "max_automatic_resumes": limit, "automatic_resumes": resumes}


def recovery_for_job(job: Any) -> ProviderJobRecovery | None:
    return job.provider_recovery if isinstance(job, RecoveryJob) else None


def defer_provider_failure(job: Any, exc: BaseException) -> bool:
    return recovery_for_job(job) is not None and transient_provider_cause(exc) is not None
