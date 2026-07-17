"""Mutable authority state for one already-planned analysis-step attempt.

This module owns checkpoint/capsule mechanics only.  It does not choose a
cohort, exposure, outcome, model, estimand, repair route, or validation policy.
The run checkpoint remains the sole selector of current capsule authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

from .provider_budget import (
    ProviderCallBudgetReceiptError,
    ProviderCallBudgetReceiptState,
)
from .schema import ValidationFinding
from .step_authority_capsule import (
    ContentRef,
    StepAuthorityCapsuleError,
    StepAuthorityCapsuleRef,
    VerifiedStepAuthorityCapsule,
)
from .step_authority_runtime import (
    StepAuthorityCoordinates,
    StepAuthorityRuntimeError,
)


@dataclass(slots=True)
class StepAttemptState:
    """One step attempt's mutable authority coordinates and replay state."""

    coordinates: Optional[StepAuthorityCoordinates] = None
    current_capsule_ref: Optional[StepAuthorityCapsuleRef] = None
    selected_resume_capsule: Optional[VerifiedStepAuthorityCapsule] = None
    capsule_execution_replay_consumed: bool = False
    last_completed_repair_parent_ref: Optional[StepAuthorityCapsuleRef] = None
    last_completed_repair_code_sha256: Optional[str] = None
    capsule_audit_findings_by_digest: Dict[str, Tuple[List[ValidationFinding], str]] = (
        field(default_factory=dict)
    )


@dataclass(frozen=True)
class StepAuthorityOperations:
    """Injected storage operations, keeping integration patch points explicit."""

    load_verified_capsule: Callable[..., VerifiedStepAuthorityCapsule]
    persist_candidate_code: Callable[..., ContentRef]
    seal_deterministic_candidate: Callable[..., StepAuthorityCapsuleRef]
    seal_legacy_candidate: Callable[..., StepAuthorityCapsuleRef]
    seal_initial_candidate: Callable[..., StepAuthorityCapsuleRef]
    seal_repair_candidate: Callable[..., StepAuthorityCapsuleRef]
    load_provider_receipt: Callable[..., ProviderCallBudgetReceiptState]


class CheckpointAuthority:
    """Persist and restore one attempt's capsule selector atomically."""

    _PENDING_REPAIR_FIELDS = (
        "capsule_pending_repair_attempt_id",
        "capsule_pending_repair_binding_sha256",
        "capsule_pending_repair_failure_status",
    )
    _PENDING_INITIAL_FIELDS = (
        "capsule_pending_initial_transport_id",
        "capsule_pending_initial_binding_sha256",
    )

    def __init__(
        self,
        *,
        run_dir: Path,
        step_id: str,
        state: StepAttemptState,
        step_record: MutableMapping[str, Any],
        per_step_records: list[dict[str, Any]],
        step_attempt_history: list[dict[str, Any]],
        shared_lock: Any,
        flush_partial_manifest: Callable[[], None],
        upsert_checkpoint: Callable[[list[dict[str, Any]], dict[str, Any]], None],
        provider_receipt_path: Path,
        reserved_final_category: Optional[str],
        sync_provider_budget: Callable[[], None],
        operations: StepAuthorityOperations,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._step_id = str(step_id)
        self.state = state
        self._step_record = step_record
        self._per_step_records = per_step_records
        self._step_attempt_history = step_attempt_history
        self._shared_lock = shared_lock
        self._flush_partial_manifest = flush_partial_manifest
        self._upsert_checkpoint = upsert_checkpoint
        self._provider_receipt_path = Path(provider_receipt_path)
        self._reserved_final_category = reserved_final_category
        self._sync_provider_budget = sync_provider_budget
        self._ops = operations

    def _load_receipt(self) -> ProviderCallBudgetReceiptState:
        return self._ops.load_provider_receipt(
            self._provider_receipt_path,
            step_id=self._step_id,
            expected_reserved_final_category=self._reserved_final_category,
        )

    def checkpoint_state(
        self,
        status: str,
        *,
        extra: Optional[Mapping[str, object]] = None,
        delete_fields: tuple[str, ...] = (),
    ) -> None:
        """Write a transient checkpoint without making it a terminal attempt."""

        record_before = dict(self._step_record)
        try:
            current_ref = self.state.current_capsule_ref
            if current_ref is not None:
                verified = self._ops.load_verified_capsule(
                    self._run_dir,
                    ref=current_ref,
                    expected_step_id=self._step_id,
                )
                ref_payload = current_ref.model_dump(mode="json")
                self._step_record["step_authority_capsule_ref"] = ref_payload
                self._step_record["step_authority_capsule_stage"] = (
                    verified.capsule.stage
                )
            if extra:
                self._step_record.update(dict(extra))
            for field_name in delete_fields:
                self._step_record.pop(field_name, None)
            snapshot = dict(self._step_record)
            snapshot["status"] = str(status)
            with self._shared_lock:
                records_before = list(self._per_step_records)
                history_before = list(self._step_attempt_history)
                try:
                    self._upsert_checkpoint(self._per_step_records, snapshot)
                    self._flush_partial_manifest()
                except BaseException:
                    self._per_step_records[:] = records_before
                    self._step_attempt_history[:] = history_before
                    raise
        except BaseException:
            self._step_record.clear()
            self._step_record.update(record_before)
            raise

    def checkpoint_capsule(
        self,
        ref: StepAuthorityCapsuleRef,
        *,
        status: str,
        extra: Optional[Mapping[str, object]] = None,
        delete_fields: tuple[str, ...] = (),
    ) -> None:
        """Select ``ref`` only if its checkpoint is durably persisted."""

        previous_ref = self.state.current_capsule_ref
        self.state.current_capsule_ref = ref
        try:
            self.checkpoint_state(
                status,
                extra=extra,
                delete_fields=delete_fields,
            )
        except BaseException:
            self.state.current_capsule_ref = previous_ref
            raise

    def ensure_candidate(
        self,
        script_text: str,
        *,
        reason: str,
    ) -> Optional[StepAuthorityCapsuleRef]:
        """Seal/checkpoint exact candidate bytes unless already current."""

        coordinates = self.state.coordinates
        if coordinates is None:
            return None
        code_ref = self._ops.persist_candidate_code(coordinates, script_text)
        current_ref = self.state.current_capsule_ref
        if current_ref is not None:
            current = self._ops.load_verified_capsule(
                self._run_dir,
                ref=current_ref,
                expected_step_id=self._step_id,
            )
            if current.capsule.candidate_code == code_ref:
                return current_ref
            ref = self._ops.seal_deterministic_candidate(
                coordinates,
                parent_ref=current_ref,
                code_ref=code_ref,
                reason=reason,
            )
        else:
            ref = self._ops.seal_legacy_candidate(coordinates, code_ref=code_ref)
        self.checkpoint_capsule(ref, status="candidate_checkpointed")
        return ref

    def seal_completed_repair_candidate(
        self,
        code_ref: object,
        logical_attempt_id: int,
        *,
        failure_status: str,
    ) -> None:
        """Join a paid repair result to its exact parent and checkpoint it."""

        coordinates = self.state.coordinates
        parent_ref = self.state.current_capsule_ref
        if coordinates is None or parent_ref is None:
            return
        if not hasattr(code_ref, "sha256"):
            raise StepAuthorityRuntimeError(
                "repair persistence did not return a content reference"
            )
        sealed_ref = self._ops.seal_repair_candidate(
            coordinates,
            parent_ref=parent_ref,
            checkpoint_parent_ref=parent_ref,
            code_ref=code_ref,
            receipt_state=self._load_receipt(),
            attempt_id=logical_attempt_id,
            failure_status=failure_status,
        )
        self.checkpoint_capsule(
            sealed_ref,
            status="candidate_checkpointed",
            delete_fields=self._PENDING_REPAIR_FIELDS,
        )
        self.state.last_completed_repair_parent_ref = parent_ref
        self.state.last_completed_repair_code_sha256 = str(code_ref.sha256)

    def reject_completed_repair_candidate(
        self,
        rejected_code: str,
        *,
        reason: str,
    ) -> None:
        """Restore the exact parent when a paid candidate fails host checks."""

        rejected_digest = hashlib.sha256(rejected_code.encode("utf-8")).hexdigest()
        if (
            self.state.last_completed_repair_parent_ref is None
            or self.state.last_completed_repair_code_sha256 != rejected_digest
        ):
            return
        self.checkpoint_capsule(
            self.state.last_completed_repair_parent_ref,
            status="candidate_checkpointed",
            extra={"step_authority_rejected_repair_candidate": reason},
        )
        self.state.last_completed_repair_parent_ref = None
        self.state.last_completed_repair_code_sha256 = None

    def checkpoint_initial_reservation(
        self,
        transport_id: str,
        binding_sha256: str,
    ) -> None:
        """Persist an unpaid initial-generation reservation before transport."""

        try:
            self._sync_provider_budget()
            self.checkpoint_state(
                "initial_generation_pending",
                extra={
                    "capsule_pending_initial_transport_id": transport_id,
                    "capsule_pending_initial_binding_sha256": binding_sha256,
                },
            )
        except (
            ProviderCallBudgetReceiptError,
            StepAuthorityRuntimeError,
            StepAuthorityCapsuleError,
        ):
            raise
        except Exception as exc:
            raise StepAuthorityRuntimeError(
                "Initial-generation reservation could not be checkpointed."
            ) from exc

    def seal_initial_candidate(
        self,
        code_ref: object,
    ) -> None:
        """Seal initial code after its provider receipt is completed."""

        try:
            coordinates = self.state.coordinates
            if coordinates is None or not hasattr(code_ref, "sha256"):
                return
            candidate_ref = self._ops.seal_initial_candidate(
                coordinates,
                code_ref=code_ref,
                receipt_state=self._load_receipt(),
            )
            self._sync_provider_budget()
            self.checkpoint_capsule(
                candidate_ref,
                status="candidate_checkpointed",
                delete_fields=self._PENDING_INITIAL_FIELDS,
            )
        except (
            ProviderCallBudgetReceiptError,
            StepAuthorityRuntimeError,
            StepAuthorityCapsuleError,
        ):
            raise
        except Exception as exc:
            raise StepAuthorityRuntimeError(
                "Initial-generation candidate authority could not be checkpointed."
            ) from exc

    def clear_failed_repair_transport(self, logical_attempt_id: int) -> None:
        """Retire pending markers only after the receipt records terminal failure."""

        receipt_state = self._load_receipt()
        if not (
            1 <= logical_attempt_id <= len(receipt_state.logical_repairs)
            and dict(
                receipt_state.logical_repairs[logical_attempt_id - 1].get("transport")
                or {}
            ).get("state")
            == "failed"
        ):
            return
        if self.state.current_capsule_ref is not None:
            self.checkpoint_state(
                "candidate_checkpointed",
                delete_fields=self._PENDING_REPAIR_FIELDS,
            )
        else:
            for key in self._PENDING_REPAIR_FIELDS:
                self._step_record.pop(key, None)


__all__ = [
    "CheckpointAuthority",
    "StepAttemptState",
    "StepAuthorityOperations",
]
