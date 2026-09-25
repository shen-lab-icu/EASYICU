"""What a retry of a failed step would repeat, read from the run's own records.

A retry of a failed approved execution resumes the first failed step from its
sealed code capsule, and that step's LLM-repair budget is restored as the
monotonic maximum of its attempt records.  Once the budget is spent, the retry
can change the outcome only if something the step runs on changed since the
failure: the research-agent code, the prompt pack, the execution kernel, or
the runner image.  A retry under an identity that no earlier grant or attempt
used also opens a fresh budget epoch (:mod:`.budget_epoch`), so the budget
read here is the latest epoch's.  This module reads the failed step's basis
without writing anything; the Web retry policy decides what to offer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

from ..authority.runtime_artifacts import (
    RunArtifactAuthorityError,
    current_step_records,
    load_run_artifact_authority,
)
from .budget_epoch import (
    AttemptIdentity,
    checkpoint_capsule_identity,
    current_attempt_identity,
    select_budget_epoch,
)
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
    #: Every identity an earlier grant or attempt of the step ran on; a retry
    #: whose identity differs from all of them earns a fresh budget epoch.
    used_identities: tuple[AttemptIdentity, ...] = ()
    #: The budget epoch the counters above describe.
    budget_epoch: int = 0


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
    attempts: Sequence[Mapping[str, Any]],
    latest: Mapping[str, Any],
    *,
    limit: Optional[int],
    counter_field: str,
) -> tuple[int, Optional[int], Optional[int], bool]:
    """Repairs spent, provider counters and exhaustion within one epoch.

    ``latest`` is the epoch's newest record, or empty when the epoch was
    opened and nothing has spent from it yet.
    """

    repairs, _classes, invalid = monotonic_step_llm_repair_history(
        attempts, limit=limit or 0, counter_field=counter_field
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
    return repairs, provider_calls, provider_limit, exhausted


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
    epoch = select_budget_epoch(
        run_dir=root,
        step_id=step_id,
        records=attempts,
        latest_record=latest,
        current_identity=None,
        explicit_rerun=False,
        attempt_id="",
        reserved_final_category=None,
        commit=False,
    )
    limit = _int(latest.get("step_llm_repair_budget"))
    epoch_latest = latest if epoch.prior_record is latest else {}
    repairs, provider_calls, provider_limit, exhausted = _repair_budget(
        epoch.records, epoch_latest, limit=limit, counter_field=epoch.counter_field
    )
    return FailedStepRetryBasis(
        step_id=step_id,
        status=str(latest.get("status") or ""),
        # A ledger the bootstrap cannot trust fails the retry closed there;
        # here it is doubt, which keeps the retry offered.
        failure_class="ledger_invalid" if epoch.error else _failure_class(latest),
        repair_attempts=repairs,
        repair_limit=limit,
        provider_calls=provider_calls,
        provider_limit=provider_limit,
        repair_budget_exhausted=bool(epoch.error) or exhausted,
        identity=checkpoint_capsule_identity(root, step_id),
        used_identities=epoch.used_identities,
        budget_epoch=epoch.epoch,
    )


__all__ = [
    "AttemptIdentity",
    "FailedStepRetryBasis",
    "current_attempt_identity",
    "load_failed_step_retry_basis",
]
