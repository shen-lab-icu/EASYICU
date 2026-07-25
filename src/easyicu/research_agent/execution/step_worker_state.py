"""Data-only progress state for one analysis-step worker.

This object consolidates transient counters and provenance labels that belong
to one worker attempt.  It deliberately owns no checkpoint, evidence, input,
or scientific authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StepWorkerProgress:
    """Mutable scratch progress for exactly one step worker."""

    resumed_code_reuse_used: bool = False
    critic_resume_repair_used: bool = False
    deterministic_fallback_used: bool = False
    deterministic_standard_executor_used: bool = False
    preexecution_runner_repair_name: str | None = None
    runner_repair_name: str | None = None
    concept_repair_attempts: int = 0
    concept_audit_error_count: int = 0
    deterministic_concept_repairs: int = 0
    applied_concept_repair_names: list[str] = field(default_factory=list)
    llm_repair_used: bool = False
    # Output-mutation counters only. They are not provider-call receipts or
    # the logical LLM budget, and concept repairs remain separately counted.
    repair_attempts: int = 0
    contract_repair_attempts: int = 0
    # Automatic structural repairs remain visible in
    # ``contract_repair_attempts`` but must not consume the independent paid
    # contract-repair allowance.
    llm_contract_repair_attempts: int = 0
    visual_repair_attempts: int = 0
    runtime_repair_attempts: int = 0

    def generation_mode(self, *, llm_repair_used: bool | None = None) -> str:
        """Describe the code that executed, preserving legacy label priority."""

        used_llm_repair = (
            self.llm_repair_used if llm_repair_used is None else bool(llm_repair_used)
        )
        if self.deterministic_standard_executor_used:
            return "deterministic_standard"
        if used_llm_repair:
            return "repaired"
        if self.deterministic_fallback_used:
            return "fallback"
        if self.runner_repair_name:
            return "runner_repaired"
        if self.repair_attempts > 0 or (
            self.concept_repair_attempts > 0 or self.deterministic_concept_repairs > 0
        ):
            return "repaired"
        if self.resumed_code_reuse_used:
            return "resumed_code_reuse"
        return "llm"
