"""Digest-bound reservation of one step repair attempt.

The execute coordinator supplies the current immutable scientific/control
coordinates.  This owner binds them to the repair ticket, reserves the exact
logical attempt in the durable provider ledger, and checkpoints the pending
transport.  It never decides whether a repair is scientifically appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..authority.evidence_store import sha256_of_bytes
from ..authority.run_input import canonical_sha256
from ..authority.step_attempt import CheckpointAuthority, StepAttemptState
from ..repairs.coordination import RepairAuthorityBinding, StepRepairBudget
from ..repairs.reasons import RepairPromptAuthority, repair_prompt_binding_sha256
from ..schema import AnalysisStep


@dataclass(frozen=True)
class StepRepairReservation:
    """Host coordinates required to reserve and checkpoint one repair."""

    step: AnalysisStep
    repair_budget: StepRepairBudget
    checkpoint_authority: CheckpointAuthority
    attempt_state: StepAttemptState
    coder_context: Any
    coder_authority: Any
    resolved_inputs_sha256: str
    coder_provider_identity_sha256: str
    prompt_version: str
    run_input_capsule_sha256: str | None
    deterministic_gate_stamp: Mapping[str, Any]

    def consume(
        self,
        repair_class: str,
        *,
        before_code: str,
        repair_ticket: str,
        repair_authority: RepairPromptAuthority,
        current_repair_authority: RepairPromptAuthority | None = None,
        provider_category: str,
        failure_status: str,
    ) -> bool:
        """Reserve one repair bound to its exact host-owned authority."""

        self.checkpoint_authority.ensure_candidate(
            before_code,
            reason="pre_repair_authority_binding",
        )
        coordinates = self.attempt_state.coordinates
        binding = RepairAuthorityBinding(
            step_id=self.step.step_id,
            attempt_id=self.repair_budget.next_attempt_id,
            repair_class=str(repair_class),
            provider_category=provider_category,
            before_code_sha256=sha256_of_bytes(before_code.encode("utf-8")),
            step_spec_sha256=canonical_sha256(self.step.model_dump(mode="json")),
            resolved_inputs_sha256=self.resolved_inputs_sha256,
            coder_context_sha256=(
                coordinates.scoped_coder_context.sha256
                if coordinates is not None
                else canonical_sha256(
                    {
                        "research_context": self.coder_context.model_dump(mode="json"),
                        "host_coder_authority": self.coder_authority.payload(),
                    }
                )
            ),
            repair_ticket_sha256=repair_prompt_binding_sha256(
                untrusted_diagnostic=repair_ticket,
                repair_authority=repair_authority,
                current_repair_authority=current_repair_authority,
            ),
            engine_validator_sha256=(
                coordinates.deterministic_gate_fingerprint
                if coordinates is not None
                else canonical_sha256(
                    {
                        "schema": "easyicu.step_control_plane_fingerprint/1",
                        "deterministic_gate_fingerprint": (
                            self.deterministic_gate_stamp[
                                "deterministic_gate_fingerprint"
                            ]
                        ),
                        "coder_provider_identity_sha256": (
                            self.coder_provider_identity_sha256
                        ),
                    }
                )
            ),
            prompt_pack_version=self.prompt_version,
            run_input_capsule_sha256=self.run_input_capsule_sha256,
        )
        consumed = self.repair_budget.consume(
            repair_class,
            authority_binding=binding,
        )
        if consumed:
            self.checkpoint_authority.checkpoint_state(
                "repair_transport_pending",
                extra={
                    "capsule_pending_repair_attempt_id": (
                        self.repair_budget.llm_repair_attempts
                    ),
                    "capsule_pending_repair_binding_sha256": binding.sha256,
                    "capsule_pending_repair_failure_status": failure_status,
                },
            )
        return consumed
