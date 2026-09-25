"""Owner for approved execution-checkpoint retry policy.

``assess_execution_retry`` answers whether retrying a failed approved
execution could change its outcome.  The execution-resume owner supplies the
failed step as the retry itself would resolve it; the Copilot workflow asks
before offering the retry, and the pipeline factory asks again, bypassing the
cache, before accepting one.  Neither keeps its own reading of the step's
records.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

from easyicu.research_agent.execution.budget_epoch import earns_fresh_budget
from easyicu.research_agent.execution.retry_basis import (
    FailedStepRetryBasis,
    current_attempt_identity,
    load_failed_step_retry_basis,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError


_REPLAYABLE_GATE_REASONS = frozenset(
    {
        "research_agent_pipeline_failed_closed",
        "research_pipeline_execution_failed",
    }
)
#: How long a projected assessment may be reused across workflow polls.
ASSESSMENT_MAX_AGE_SECONDS = 30.0


def preserves_approved_execution_checkpoint(gate_reason: Any) -> bool:
    """Return whether a gate reason may reuse an approved execution checkpoint."""

    return str(gate_reason or "").strip() in _REPLAYABLE_GATE_REASONS


@dataclass(frozen=True)
class ExecutionRetryAssessment:
    """Whether a retry of the failed step could change its outcome.

    ``futile`` is claimed only when the failed step's repair budget is spent
    after a code failure and nothing the step runs on changed since then;
    every doubt (no record, no capsule, a timeout, an infrastructure failure)
    leaves the retry ``available`` or ``unknown``.  ``fresh_repair_budget``
    says whether the retry would also earn a fresh budget epoch: only when
    its identity differs from every identity the step was granted or ran on.
    """

    state: Literal["available", "futile", "unknown"]
    reason_code: str
    failed_step_id: str = ""
    repair_attempts: Optional[int] = None
    repair_limit: Optional[int] = None
    changed_components: tuple[str, ...] = ()
    image_checked: bool = False
    fresh_repair_budget: bool = False

    def public(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "reason_code": self.reason_code,
            "failed_step_id": self.failed_step_id,
            "repair_attempts": self.repair_attempts,
            "repair_limit": self.repair_limit,
            "changed_components": list(self.changed_components),
            "image_checked": self.image_checked,
            "fresh_repair_budget": self.fresh_repair_budget,
        }


def _runner_image_id(image: Optional[str]) -> Optional[str]:
    """Inspect the runner image a launch would select; ``None`` if unreadable."""

    from easyicu.research_agent.execution.runner import probe_runner_availability

    availability = probe_runner_availability("docker", image=image or None)
    if not availability.available:
        return None
    return availability.image_id or None


def assess_failed_step(
    basis: Optional[FailedStepRetryBasis],
    *,
    read_image_id: Callable[[], Optional[str]] = lambda: _runner_image_id(None),
) -> ExecutionRetryAssessment:
    """Decide from one failed step's recorded basis; Docker is asked last."""

    if basis is None:
        return ExecutionRetryAssessment("unknown", "execution_retry_basis_unavailable")
    common: Dict[str, Any] = {
        "failed_step_id": basis.step_id,
        "repair_attempts": basis.repair_attempts,
        "repair_limit": basis.repair_limit,
    }
    if basis.failure_class != "code":
        return ExecutionRetryAssessment(
            "available", f"execution_retry_failure_{basis.failure_class}", **common
        )
    if not basis.repair_budget_exhausted:
        return ExecutionRetryAssessment(
            "available", "execution_retry_repair_budget_remaining", **common
        )
    if basis.identity is None:
        return ExecutionRetryAssessment(
            "unknown", "execution_retry_failed_identity_unknown", **common
        )
    current = current_attempt_identity(image_id=None)
    changed = current.changes_since(basis.identity)
    if changed:
        return ExecutionRetryAssessment(
            "available", "execution_retry_code_changed",
            changed_components=changed,
            fresh_repair_budget=earns_fresh_budget(current, basis.used_identities),
            **common,
        )
    if basis.identity.image_id:
        # The failed attempt ran in a recorded image: a rebuilt image is a
        # change, and an image nobody can read now is not proof of none.
        image_id = read_image_id()
        if image_id is None:
            return ExecutionRetryAssessment(
                "unknown", "execution_retry_runner_image_unreadable", **common
            )
        current = current_attempt_identity(image_id=image_id)
        changed = current.changes_since(basis.identity)
        if changed:
            return ExecutionRetryAssessment(
                "available", "execution_retry_runner_image_changed",
                changed_components=changed, image_checked=True,
                fresh_repair_budget=earns_fresh_budget(current, basis.used_identities),
                **common,
            )
    return ExecutionRetryAssessment(
        "futile", "execution_retry_repeats_failure",
        image_checked=bool(basis.identity.image_id), **common,
    )


_cache_lock = threading.Lock()
_cache: Dict[tuple[str, str, str], tuple[float, ExecutionRetryAssessment]] = {}


def assess_execution_retry(
    *,
    source_run_id: str,
    configuration_sha256: str,
    resolve_failed_step: Callable[[], tuple[Path, Optional[str]]],
    runner_image: Optional[str] = None,
    max_age_seconds: float = ASSESSMENT_MAX_AGE_SECONDS,
) -> ExecutionRetryAssessment:
    """Assess retrying one failed approved run; ``max_age_seconds=0`` rereads.

    ``resolve_failed_step`` is the retry owner's own resolution of the
    pipeline run and first failed step; its typed refusal leaves the answer
    ``unknown``.  ``runner_image`` is the reference the retry would launch
    with, or ``None`` for the server's selected runner image.
    """

    key = (str(source_run_id or ""), str(configuration_sha256 or ""), str(runner_image or ""))
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None and now - cached[0] < max_age_seconds:
        return cached[1]
    try:
        run_dir, step_id = resolve_failed_step()
        basis = load_failed_step_retry_basis(run_dir, step_id) if step_id else None
    except ResearchPipelineRunError as exc:
        assessment = ExecutionRetryAssessment("unknown", exc.code)
    except (OSError, ValueError):
        # An unreadable record is doubt, not proof that a retry is futile.
        assessment = ExecutionRetryAssessment("unknown", "execution_retry_records_unreadable")
    else:
        assessment = assess_failed_step(
            basis, read_image_id=lambda: _runner_image_id(runner_image)
        )
    with _cache_lock:
        if len(_cache) > 64:
            _cache.clear()
        _cache[key] = (now, assessment)
    return assessment


__all__ = [
    "ASSESSMENT_MAX_AGE_SECONDS",
    "ExecutionRetryAssessment",
    "assess_execution_retry",
    "assess_failed_step",
    "preserves_approved_execution_checkpoint",
]
