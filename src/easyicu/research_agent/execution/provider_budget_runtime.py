"""Crash-safe provider and logical-repair budget preparation for one step."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..authority.provider_budget import (
    PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
    ProviderCallBudgetReceiptError,
    StepProviderCallBudget,
    load_provider_call_budget_state,
    provider_call_budget_receipt_path,
)
from ..repairs.coordination import StepRepairBudget


@dataclass(frozen=True)
class StepProviderBudgetRuntime:
    """Prepared budget owners and any fail-closed receipt defect."""

    provider_budget: StepProviderCallBudget
    repair_budget: StepRepairBudget
    receipt_path: Path
    receipt_relative_path: str
    reserved_final_category: Optional[str]
    integrity_error: Optional[str]


def monotonic_step_llm_repair_history(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[int, List[str], bool]:
    """Recover the largest durable logical-repair counter for one step.

    Step records are append-only attempts.  The latest attempt may terminate
    before copying the logical counter (for example, on a damaged provider
    receipt), so latest-record-only recovery can incorrectly buy a fresh
    repair budget.  A malformed explicit counter is treated conservatively as
    exhausted instead of being ignored.
    """

    attempts = 0
    classes: List[str] = []
    invalid_snapshot = False
    for record in records:
        if "step_llm_repair_attempts" in record:
            raw_attempts = record.get("step_llm_repair_attempts")
            if (
                isinstance(raw_attempts, bool)
                or not isinstance(raw_attempts, int)
                or raw_attempts < 0
            ):
                invalid_snapshot = True
            else:
                attempts = max(attempts, raw_attempts)
        raw_classes = record.get("step_llm_repair_classes")
        if not isinstance(raw_classes, list):
            continue
        normalized = [str(item).strip() for item in raw_classes]
        if any(not item for item in normalized):
            invalid_snapshot = True
            continue
        if len(normalized) > len(classes):
            classes = normalized
    if invalid_snapshot:
        attempts = max(attempts, max(0, int(limit)))
    return attempts, classes, invalid_snapshot


def step_snapshot_requires_provider_receipt(
    record: Mapping[str, Any],
    *,
    provider_attempts: int,
    logical_repair_attempts: int,
) -> bool:
    """Whether a checkpoint proves a durable provider ledger must exist."""

    if record.get("step_provider_call_receipt_version") not in {
        1,
        2,
        3,
        4,
        5,
        6,
        PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
    }:
        return False
    return bool(
        provider_attempts > 0
        or logical_repair_attempts > 0
        or record.get("capsule_pending_initial_transport_id")
        or record.get("step_provider_call_receipt")
    )


def prepare_step_provider_budget(
    *,
    prior_attempt_records: Sequence[Mapping[str, Any]],
    prior_step_record: Mapping[str, Any] | None,
    run_dir: Path,
    step_id: str,
    step_record: Dict[str, Any],
    max_provider_calls: int,
    max_llm_repairs: int,
    reserve_concept_audit: bool,
    allow_terminal_initial_generation_restart: bool,
) -> StepProviderBudgetRuntime:
    """Restore one step's durable provider ledger without buying fresh calls."""

    (
        step_llm_repair_attempts,
        prior_repair_classes,
        repair_history_invalid,
    ) = monotonic_step_llm_repair_history(
        prior_attempt_records,
        limit=max_llm_repairs,
    )
    if step_llm_repair_attempts:
        step_record["step_llm_repair_attempts"] = step_llm_repair_attempts
        step_record["step_llm_repair_budget"] = max_llm_repairs
    if prior_repair_classes:
        step_record["step_llm_repair_classes"] = list(prior_repair_classes)
    if repair_history_invalid:
        step_record["step_llm_repair_history_invalid"] = True
        step_record["step_llm_repair_budget_exhausted"] = True
    configured_provider_limit = max_provider_calls
    effective_provider_limit = configured_provider_limit
    reserved_final_category = "concept_audit" if reserve_concept_audit else None
    provider_receipt_path = provider_call_budget_receipt_path(
        run_dir,
        step_id=step_id,
    )
    provider_receipt_relative_path = str(provider_receipt_path.relative_to(run_dir))
    prior_provider_categories: tuple[str, ...] = ()
    prior_logical_repair_entries: tuple[Dict[str, object], ...] = ()
    prior_initial_generation_entries: tuple[Dict[str, object], ...] = ()
    prior_required_reservation_token: Optional[str] = None
    prior_reservation_bound_provider_history_len: Optional[int] = None
    prior_completed_reservation_token: Optional[str] = None
    prior_reservation_released = False
    prior_reserved_category_extensions: tuple[Dict[str, object], ...] = ()
    prior_provider_attempts = 0
    provider_receipt_integrity_error: Optional[str] = None
    prior_snapshot_present = False
    if isinstance(prior_step_record, Mapping):
        snapshot_keys = {
            "step_provider_call_budget",
            "step_provider_call_attempts",
            "step_provider_call_categories",
        }
        prior_snapshot_present = any(key in prior_step_record for key in snapshot_keys)
        if prior_snapshot_present:
            prior_limit = prior_step_record.get("step_provider_call_budget")
            prior_attempts_raw = prior_step_record.get("step_provider_call_attempts")
            prior_categories_raw = prior_step_record.get(
                "step_provider_call_categories"
            )
            if (
                isinstance(prior_limit, bool)
                or not isinstance(prior_limit, int)
                or prior_limit < 0
                or isinstance(prior_attempts_raw, bool)
                or not isinstance(prior_attempts_raw, int)
                or prior_attempts_raw < 0
                or not isinstance(prior_categories_raw, list)
            ):
                provider_receipt_integrity_error = (
                    "Prior provider-call budget snapshot is incomplete or invalid."
                )
            else:
                normalized_categories = tuple(
                    str(item).strip() for item in prior_categories_raw
                )
                if any(
                    not item for item in normalized_categories
                ) or prior_attempts_raw != len(normalized_categories):
                    provider_receipt_integrity_error = (
                        "Prior provider-call attempts and category history disagree."
                    )
                else:
                    prior_provider_attempts = prior_attempts_raw
                    prior_provider_categories = normalized_categories
                    effective_provider_limit = min(
                        effective_provider_limit,
                        prior_limit,
                    )

    if provider_receipt_integrity_error is None and provider_receipt_path.exists():
        try:
            receipt_state = load_provider_call_budget_state(
                provider_receipt_path,
                step_id=step_id,
                expected_reserved_final_category=reserved_final_category,
            )
            receipt_limit = receipt_state.limit
            receipt_categories = receipt_state.categories
            prior_logical_repair_entries = receipt_state.logical_repairs
            prior_initial_generation_entries = receipt_state.initial_generations
            prior_required_reservation_token = receipt_state.required_reservation_token
            prior_reservation_bound_provider_history_len = (
                receipt_state.reservation_bound_provider_history_len
            )
            prior_completed_reservation_token = (
                receipt_state.completed_reservation_token
            )
            prior_reservation_released = receipt_state.reservation_released
            prior_reserved_category_extensions = (
                receipt_state.reserved_category_extensions
            )
            effective_provider_limit = min(
                effective_provider_limit,
                receipt_limit,
            )
            if prior_snapshot_present and (
                len(receipt_categories) < len(prior_provider_categories)
                or receipt_categories[: len(prior_provider_categories)]
                != prior_provider_categories
            ):
                raise ProviderCallBudgetReceiptError(
                    "Durable provider-call receipt conflicts with the latest "
                    "step snapshot."
                )
            prior_provider_categories = receipt_categories
            prior_provider_attempts = len(receipt_categories)
        except ProviderCallBudgetReceiptError as exc:
            provider_receipt_integrity_error = str(exc)
    elif (
        provider_receipt_integrity_error is None
        and isinstance(prior_step_record, Mapping)
        and step_snapshot_requires_provider_receipt(
            prior_step_record,
            provider_attempts=prior_provider_attempts,
            logical_repair_attempts=step_llm_repair_attempts,
        )
    ):
        provider_receipt_integrity_error = (
            "Durable provider/repair receipt is missing for a prior reservation."
        )

    provider_budget = StepProviderCallBudget(
        effective_provider_limit,
        step_id=step_id,
        consumed_categories=prior_provider_categories,
        logical_repair_entries=prior_logical_repair_entries,
        initial_generation_entries=prior_initial_generation_entries,
        allow_terminal_initial_generation_restart=(
            allow_terminal_initial_generation_restart
        ),
        receipt_path=provider_receipt_path,
        reserved_final_category=reserved_final_category,
        required_reservation_token=prior_required_reservation_token,
        reservation_bound_provider_history_len=(
            prior_reservation_bound_provider_history_len
        ),
        completed_reservation_token=prior_completed_reservation_token,
        reservation_released=prior_reservation_released,
        reserved_category_extensions=prior_reserved_category_extensions,
    )

    if provider_receipt_integrity_error is None:
        try:
            # A crash before the first provider call leaves an exact unpaid
            # reservation that can be resumed. A crash after any paid call
            # but before the result digest was sealed is unknowable and must
            # block the step before any other route can ignore or replace it.
            provider_budget.next_logical_repair_attempt_id()
            initial_resume_status = provider_budget.initial_generation_resume_status()
            if initial_resume_status == "paid_pending":
                raise ProviderCallBudgetReceiptError(
                    "Initial generation has paid provider calls but no durable "
                    "transport result."
                )
            if (
                initial_resume_status == "failed"
                and not provider_budget.terminal_initial_generation_restart_allowed
            ):
                raise ProviderCallBudgetReceiptError(
                    "Initial generation previously reached a terminal provider failure."
                )
        except ProviderCallBudgetReceiptError as exc:
            provider_receipt_integrity_error = str(exc)

    try:
        step_repair_budget = StepRepairBudget(
            provider_budget=provider_budget,
            step_record=step_record,
            max_llm_repairs=max_llm_repairs,
            initial_llm_repair_attempts=step_llm_repair_attempts,
            initial_repair_classes=(
                prior_repair_classes if provider_receipt_integrity_error is None else ()
            ),
            provider_receipt_relative_path=provider_receipt_relative_path,
        )
    except (ProviderCallBudgetReceiptError, ValueError) as exc:
        provider_receipt_integrity_error = str(exc)
        step_repair_budget = StepRepairBudget(
            provider_budget=provider_budget,
            step_record=step_record,
            max_llm_repairs=max_llm_repairs,
            initial_llm_repair_attempts=step_llm_repair_attempts,
            provider_receipt_relative_path=provider_receipt_relative_path,
        )
    return StepProviderBudgetRuntime(
        provider_budget=provider_budget,
        repair_budget=step_repair_budget,
        receipt_path=provider_receipt_path,
        receipt_relative_path=provider_receipt_relative_path,
        reserved_final_category=reserved_final_category,
        integrity_error=provider_receipt_integrity_error,
    )


__all__ = [
    "StepProviderBudgetRuntime",
    "monotonic_step_llm_repair_history",
    "prepare_step_provider_budget",
    "step_snapshot_requires_provider_receipt",
]
