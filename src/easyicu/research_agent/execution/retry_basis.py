"""What a retry of a failed step would repeat, read from the run's own records.

A retry of a failed approved execution resumes the first failed step from its
sealed code capsule, and that step's LLM-repair budget is restored as the
monotonic maximum of its attempt records.  Once the budget is spent, the retry
can change the outcome only if something the step runs on changed since the
failure: the research-agent code, the prompt pack, the execution kernel, or
the runner image.  This module reads the failed step's basis without writing
anything; the Web retry policy decides what to offer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

from ..authority.run_input import engine_code_sha256
from ..authority.runtime_artifacts import (
    RunArtifactAuthorityError,
    current_step_records,
    load_run_artifact_authority,
)
from ..authority.step_capsule import read_verified_content
from ..authority.step_runtime import (
    StepAuthorityRuntimeError,
    load_checkpoint_selected_step_capsule,
)
from ..canonical_json import canonical_sha256
from ..providers.prompts import prompt_pack_files
from .provider_budget_runtime import monotonic_step_llm_repair_history

FailureClass = Literal[
    "code",
    "timeout",
    "infrastructure",
    "fail_closed",
    "host_exception",
    "ledger_invalid",
    "pending",
    "other",
]

#: A generated script ran and failed, so only an LLM repair can change it.
_CODE_FAILURE_STATUSES = frozenset({"execution_failed", "contract_failed", "repair_failed"})
#: Container/sandbox exits that say the script never ran to completion.
_INFRASTRUCTURE_RETURN_CODES = frozenset({125, 126, 127, 137, -9})
_IDENTITY_FIELDS = (
    "engine_code_sha256",
    "prompt_pack_sha256",
    "execution_kernel_identity_sha256",
    "image_id",
)


@dataclass(frozen=True)
class AttemptIdentity:
    """The code and runtime one attempt ran on; ``None`` means not recorded."""

    engine_code_sha256: Optional[str] = None
    prompt_pack_sha256: Optional[str] = None
    execution_kernel_identity_sha256: Optional[str] = None
    image_id: Optional[str] = None

    def changes_since(self, failed: "AttemptIdentity") -> tuple[str, ...]:
        """Components known on both sides that differ; unknown is unchanged."""

        return tuple(
            name
            for name in _IDENTITY_FIELDS
            if getattr(failed, name)
            and getattr(self, name)
            and getattr(failed, name) != getattr(self, name)
        )


@dataclass(frozen=True)
class FailedStepRetryBasis:
    step_id: str
    status: str
    failure_class: FailureClass
    repair_attempts: int
    repair_limit: Optional[int]
    provider_calls: Optional[int]
    provider_limit: Optional[int]
    repair_budget_exhausted: bool
    #: ``None`` when the step's capsule is missing or unreadable.
    identity: Optional[AttemptIdentity]


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _failure_class(record: Mapping[str, Any]) -> FailureClass:
    status = str(record.get("status") or "")
    if any(str(key).startswith("capsule_pending_") for key in record):
        return "pending"
    if record.get("step_llm_repair_history_invalid") is True or record.get(
        "provider_call_budget_receipt_invalid"
    ):
        return "ledger_invalid"
    failure = str(record.get("runtime_failure_class") or "")
    if record.get("timed_out") is True or failure == "execution_timeout":
        return "timeout"
    if (
        status == "execution_environment_failed"
        or failure == "isolation_backend_unavailable"
        or record.get("runner_failure_code")
        or _int(record.get("returncode")) in _INFRASTRUCTURE_RETURN_CODES
    ):
        return "infrastructure"
    if status == "execution_raised":
        return "host_exception"
    if record.get("runtime_repair_route") == "fail_closed":
        return "fail_closed"
    return "code" if status in _CODE_FAILURE_STATUSES else "other"


def _repair_budget(
    attempts: Sequence[Mapping[str, Any]], latest: Mapping[str, Any]
) -> tuple[int, Optional[int], Optional[int], Optional[int], bool]:
    limit = _int(latest.get("step_llm_repair_budget"))
    repairs, _classes, invalid = monotonic_step_llm_repair_history(
        attempts, limit=limit or 0
    )
    provider_calls = _int(latest.get("step_provider_call_attempts"))
    provider_limit = _int(latest.get("step_provider_call_budget"))
    remaining = _int(latest.get("step_provider_call_remaining"))
    exhausted = bool(
        invalid
        or latest.get("step_llm_repair_budget_exhausted") is True
        or (limit is not None and repairs >= limit)
        or latest.get("step_provider_call_budget_exhausted") is True
        or remaining == 0
    )
    return repairs, limit, provider_calls, provider_limit, exhausted


def _failed_identity(run_dir: Path, step_id: str) -> Optional[AttemptIdentity]:
    try:
        verified = load_checkpoint_selected_step_capsule(run_dir, step_id=step_id)
    except (StepAuthorityRuntimeError, RunArtifactAuthorityError):
        return None
    if verified is None:
        return None
    capsule = verified.capsule
    kernel = image = None
    if capsule.execution is not None:
        try:
            provenance = json.loads(
                read_verified_content(run_dir, capsule.execution.runtime_provenance)
            )
        except (ValueError, OSError, RuntimeError):
            provenance = None
        if isinstance(provenance, Mapping):
            kernel = str(provenance.get("execution_kernel_identity_sha256") or "") or None
            image = str(provenance.get("image_id") or "") or None
    return AttemptIdentity(
        engine_code_sha256=capsule.engine_code_sha256,
        prompt_pack_sha256=capsule.prompt_pack_sha256,
        execution_kernel_identity_sha256=kernel,
        image_id=image,
    )


def load_failed_step_retry_basis(
    run_dir: str | Path, step_id: str
) -> Optional[FailedStepRetryBasis]:
    """Read one failed step's retry basis, or ``None`` when it has no record."""

    root = Path(run_dir)
    try:
        authority = load_run_artifact_authority(root)
    except RunArtifactAuthorityError:
        return None
    records = authority.get("per_step_records") if isinstance(authority, Mapping) else None
    if not isinstance(records, list):
        return None
    # The attempt set the resume bootstrap restores the budget from: the
    # durable attempt history (or, without one, the ledger) plus the ledger.
    history = authority.get("step_attempt_history")
    attempts = [
        record
        for record in [*(history if isinstance(history, list) and history else records), *records]
        if isinstance(record, Mapping) and str(record.get("step_id") or "") == step_id
    ]
    latest = next(
        (
            record
            for record in current_step_records(records)
            if str(record.get("step_id") or "") == step_id
        ),
        None,
    )
    if latest is None:
        return None
    repairs, limit, provider_calls, provider_limit, exhausted = _repair_budget(
        attempts, latest
    )
    return FailedStepRetryBasis(
        step_id=step_id,
        status=str(latest.get("status") or ""),
        failure_class=_failure_class(latest),
        repair_attempts=repairs,
        repair_limit=limit,
        provider_calls=provider_calls,
        provider_limit=provider_limit,
        repair_budget_exhausted=exhausted,
        identity=_failed_identity(root, step_id),
    )


@lru_cache(maxsize=1)
def _current_code_identity() -> tuple[str, str, Optional[str]]:
    from .kernel_identity import ExecutionKernelIdentityError, build_execution_kernel_identity

    try:
        kernel: Optional[str] = build_execution_kernel_identity(
            Path(__file__).resolve().parents[2]
        ).identity_sha256
    except (ExecutionKernelIdentityError, OSError):
        kernel = None
    return (
        engine_code_sha256(),
        canonical_sha256(dict(prompt_pack_files())),
        kernel,
    )


def current_attempt_identity(*, image_id: Optional[str]) -> AttemptIdentity:
    """The identity a retry started by this process would run on.

    The code digests describe the code this process loaded, which is the code
    a retry it starts executes; the image identifier is the caller's fresh
    reading of the selected runner image.
    """

    engine, prompts, kernel = _current_code_identity()
    return AttemptIdentity(
        engine_code_sha256=engine,
        prompt_pack_sha256=prompts,
        execution_kernel_identity_sha256=kernel,
        image_id=image_id or None,
    )


__all__ = [
    "AttemptIdentity",
    "FailedStepRetryBasis",
    "current_attempt_identity",
    "load_failed_step_retry_basis",
]
