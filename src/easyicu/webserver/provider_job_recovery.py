"""Bounded host recovery of a job through its existing checkpoint owners.

Transport retries remain the provider's policy. This owner waits for service
recovery, accounts probes against the original task, and records system actions.
It never constructs a plan, review decision, or execution grant.
"""

from __future__ import annotations

import json
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

    def bind(self, *, wrapper_dir: Path, task: Any, client: Any) -> None:
        if self.task is not None and task is not self.task:
            raise ValueError("automatic recovery cannot replace its provider task")
        self.wrapper_dir, self.task, self.client = wrapper_dir, task, client
        self.wrappers.add(wrapper_dir)
        if self.deadline is not None:
            self.persist()

    def persist(self) -> None:
        record = {
            "schema_version": "easyicu.provider_job_recovery/1",
            "job_id": self.job.id, "window_seconds": RECOVERY_WINDOW_SECONDS,
            "max_automatic_resumes": MAX_AUTOMATIC_RESUMES,
            "probe_interval_seconds": PROBE_INTERVAL_SECONDS,
            "automatic_resumes": self.resumes, "stop_reason": self.stop_reason,
        }
        for root in self.wrappers:
            path = root / "provider_recovery.json"
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
        self.job.emit({"type": "progress", "step": "provider_recovery_wait", "actor": "system", "reason": cause})
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
            if self.deadline is not None:
                self.persist()
            # Failure handlers defer closing a transient call's task until
            # this job has either recovered or reached a terminal boundary.
            if self.task is not None:
                rows = self.task.ledger.snapshot().get("tasks", [])
                if any(row.get("task_id") == self.task.task_id and row.get("status") in {"running", "paused"} for row in rows):
                    self.task.finish(error=self.stop_reason)
            raise


def recovery_for_job(job: Any) -> ProviderJobRecovery | None:
    return job.provider_recovery if isinstance(job, RecoveryJob) else None


def defer_provider_failure(job: Any, exc: BaseException) -> bool:
    return recovery_for_job(job) is not None and transient_provider_cause(exc) is not None
