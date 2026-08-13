"""Candidate recovery and repair transport for one execute-phase step.

The execute loop owns ordering.  This owner holds the mutable collaborators
required to resume rejected or previously generated code and to seal every paid
repair against its exact authority capsule.  Its public request makes formerly
implicit closure state reviewable and testable.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..authority.provider_budget import (
    ProviderCallBudgetReceiptError,
    StepProviderCallBudget,
)
from ..authority.step_attempt import CheckpointAuthority, StepAttemptState
from ..authority.step_capsule import StepAuthorityCapsuleError
from ..authority.step_runtime import (
    StepAuthorityRuntimeError,
    persist_candidate_code,
)
from ..authority.coder_authority import HostCoderAuthority
from ..contracts.runtime import ValidationFinding
from ..orchestration.resume import QuarantinedConceptDraft, ResumeController
from ..repairs.coordination import (
    StepRepairBudget,
    resume_deterministic_repair_candidate,
)
from ..repairs.reasons import RepairPromptAuthority
from ..schema import AnalysisStep, ResearchContext
from .concept_audit import ConceptQuarantineState
from .concept_reaudit import (
    deterministic_concept_reaudit_authority,
    deterministic_concept_reaudit_pending_errors,
)
from .step_worker_state import StepWorkerProgress


@dataclass(frozen=True)
class StepCandidateRecoveryRequest:
    """Explicit collaborators for candidate recovery and paid repair."""

    step: AnalysisStep
    run_dir: Path
    run_id: str
    step_current: int
    total_steps: int
    requested_resume_from_step_id: Optional[str]
    prior_step_record: Optional[Mapping[str, Any]]
    prior_attempt_records: Sequence[Mapping[str, Any]]
    provider_budget: StepProviderCallBudget
    step_repair_budget: StepRepairBudget
    step_record: Dict[str, Any]
    findings: List[ValidationFinding]
    shared_lock: threading.Lock
    worker_progress: StepWorkerProgress
    quarantine_state: ConceptQuarantineState
    resume_controller: ResumeController
    analysis_family: Optional[str]
    coder: Any
    coder_context: ResearchContext
    coder_authority: HostCoderAuthority
    step_attempt_state: StepAttemptState
    checkpoint_authority: CheckpointAuthority
    deterministic_runner_repair_enabled: bool
    emit_progress: Callable[..., None]
    remember_concept_constraints: Callable[[Sequence[ValidationFinding]], None]
    consume_llm_repair_budget: Callable[..., bool]
    sync_provider_budget: Callable[[], None]
    authorize_automatic_repair: Callable[..., Optional[Tuple[str, str]]]
    record_repair: Callable[..., None]


class StepCandidateRecovery:
    """Recover, repair, and checkpoint candidate code for one step."""

    def __init__(self, request: StepCandidateRecoveryRequest) -> None:
        self.request = request

    def use_quarantined_draft(self, draft: QuarantinedConceptDraft) -> str:
        request = self.request
        state = request.quarantine_state
        state.resumed_draft_used = True
        state.draft_active = True
        state.repair_succeeded = False
        budget_snapshot = request.provider_budget.snapshot()
        historical_repair_names = deterministic_concept_reaudit_authority(
            code_sha256=draft.sha256,
            current_repair_count=0,
            current_repair_names=(),
            prior_step_record=request.prior_step_record,
            prior_step_records=request.prior_attempt_records,
            provider_used=budget_snapshot["used"],
            provider_limit=budget_snapshot["limit"],
        )
        reaudit_errors = deterministic_concept_reaudit_pending_errors(
            draft.findings,
            provider_used=budget_snapshot["used"],
            provider_limit=budget_snapshot["limit"],
        )
        active_findings = (
            reaudit_errors
            if historical_repair_names and reaudit_errors
            else draft.findings
        )
        state.pending_errors = [
            ValidationFinding.model_validate(payload) for payload in active_findings
        ]
        request.remember_concept_constraints(
            [ValidationFinding.model_validate(payload) for payload in draft.findings]
        )
        if historical_repair_names and reaudit_errors:
            state.repair_materially_changed = True
            request.step_record["resumed_deterministic_concept_reaudit"] = {
                "code_sha256": draft.sha256,
                "repair_names": list(historical_repair_names),
                "diagnostic_code": "deterministic_repair_budget_only_quarantine_v1",
            }
        request.step_record.update(
            {
                "resumed_quarantined_draft": True,
                "quarantined_draft_sha256": draft.sha256,
                "quarantined_draft_relative_path": draft.relative_path,
                "quarantined_requires_repair": True,
                "quarantined_repair_succeeded": False,
            }
        )
        request.emit_progress(
            "coder",
            f"Resuming rejected draft for mandatory repair: {request.step.step_id}.",
            status="warning",
            run_id=request.run_id,
            step_id=request.step.step_id,
            current_step=request.step_current,
            total_steps=request.total_steps,
        )
        return draft.code

    def use_resumed_code(
        self,
        resumed_code: Tuple[str, Dict[str, Any]],
        *,
        error: Optional[BaseException] = None,
    ) -> str:
        request = self.request
        request.worker_progress.resumed_code_reuse_used = True
        prior_code, resumed_record = resumed_code
        request.step_record["generation_mode"] = "resumed_code_reuse"
        request.step_record["resumed_code_evidence_id"] = resumed_record.get(
            "evidence_id"
        )
        request.step_record["resumed_code_relative_path"] = resumed_record.get(
            "relative_path"
        )
        evidence_mode = str(resumed_record.get("generation_mode") or "")
        source_mode = evidence_mode
        if evidence_mode == "resumed_code_reuse":
            metadata = resumed_record.get("metadata")
            if isinstance(metadata, dict):
                source_mode = str(metadata.get("resumed_from_generation_mode") or "")
        request.step_record["resumed_code_evidence_generation_mode"] = evidence_mode
        request.step_record["resumed_from_generation_mode"] = source_mode
        detail = {
            "step_id": request.step.step_id,
            "resume_from_step_id": request.requested_resume_from_step_id,
            "evidence_id": resumed_record.get("evidence_id"),
            "relative_path": resumed_record.get("relative_path"),
            "resumed_from_generation_mode": source_mode,
        }
        if error is None:
            message = (
                "Explicit resume reused prior agent-generated code "
                f"(source mode: {source_mode}) for step {request.step.step_id} "
                "before requesting a new coder script."
            )
        else:
            detail["error"] = str(error)
            message = (
                f"Coder agent failed for step {request.step.step_id}; reused prior "
                "agent-generated code from resume evidence "
                f"(source mode: {source_mode})."
            )
        with request.shared_lock:
            request.findings.append(
                ValidationFinding(
                    validator="coder",
                    severity="warning",
                    message=message,
                    detail=detail,
                )
            )
        request.emit_progress(
            "coder",
            f"Reused prior generated analysis script for {request.step.step_id}.",
            status="warning",
            run_id=request.run_id,
            step_id=request.step.step_id,
            current_step=request.step_current,
            total_steps=request.total_steps,
        )
        return prior_code

    def repair_with_capsule(
        self,
        *,
        failure_status: str,
        context: ResearchContext,
        step: AnalysisStep,
        code: str,
        run_log: str,
        repair_authority: RepairPromptAuthority,
        current_repair_authority: Optional[RepairPromptAuthority] = None,
        attempt: int,
        provider_budget: StepProviderCallBudget,
        provider_category: str,
        logical_repair_attempt_id: int,
    ) -> str:
        request = self.request
        coordinates = request.step_attempt_state.coordinates
        try:
            return request.coder.repair(
                context=context,
                step=step,
                host_authority=request.coder_authority,
                code=code,
                run_log=run_log,
                repair_authority=repair_authority,
                current_repair_authority=current_repair_authority,
                attempt=attempt,
                provider_budget=provider_budget,
                provider_category=provider_category,
                logical_repair_attempt_id=logical_repair_attempt_id,
                persist_candidate=(
                    (lambda candidate: persist_candidate_code(coordinates, candidate))
                    if coordinates is not None
                    else None
                ),
                on_candidate_completed=(
                    lambda ref, _mode, logical_id: (
                        request.checkpoint_authority.seal_completed_repair_candidate(
                            ref,
                            logical_id,
                            failure_status=failure_status,
                        )
                        if coordinates is not None
                        else None
                    )
                ),
            )
        except Exception:
            request.checkpoint_authority.clear_failed_repair_transport(
                logical_repair_attempt_id
            )
            raise

    def reserve_compatibility_repair(
        self,
        before_code: str,
        repair_ticket: str,
        repair_authority: RepairPromptAuthority,
    ) -> Optional[int]:
        request = self.request
        if not request.consume_llm_repair_budget(
            "compatibility",
            before_code=before_code,
            repair_ticket=repair_ticket,
            repair_authority=repair_authority,
            provider_category="compatibility_repair",
            failure_status="concept_failed",
        ):
            return None
        return request.step_repair_budget.llm_repair_attempts

    def resume_deterministic_repair_code(self) -> Optional[str]:
        request = self.request
        if (
            request.requested_resume_from_step_id != request.step.step_id
            or not request.deterministic_runner_repair_enabled
        ):
            return None
        resumed_code = request.resume_controller.prior_code_for_step(
            request.step.step_id
        )
        if resumed_code is None:
            return None
        prior_code, _resumed_record = resumed_code
        candidate = resume_deterministic_repair_candidate(
            code=prior_code,
            step_dir=request.run_dir / "steps" / request.step.step_id,
            analysis_family=request.analysis_family,
        )
        if candidate is None:
            return None
        repair, source, trigger = candidate
        repair = request.authorize_automatic_repair(
            repair,
            step=request.step,
            source=source,
            before_code=prior_code,
        )
        if repair is None:
            return None
        repair_name, repaired_code = repair
        self.use_resumed_code(resumed_code)
        request.worker_progress.preexecution_runner_repair_name = repair_name
        request.step_record["runner_repair"] = repair_name
        request.step_record["resume_deterministic_repair"] = repair_name
        request.record_repair(
            repair_id=repair_name,
            step_id=request.step.step_id,
            trigger=trigger,
            transformation=(
                "Reused the explicitly resumed step's prior generated code after "
                "deterministic runtime/summary repair, before requesting a new "
                "coder script."
            ),
            before_code=prior_code,
            after_code=repaired_code,
            selection_rule=(
                "only when the prior step_summary triggers a case-neutral "
                "deterministic summary repair"
            ),
        )
        with request.shared_lock:
            request.findings.append(
                ValidationFinding(
                    validator="coder",
                    severity="info",
                    message=(
                        "Applied deterministic resume repair for step "
                        f"{request.step.step_id}: {repair_name}."
                    ),
                    detail={
                        "step_id": request.step.step_id,
                        "repair_id": repair_name,
                        "source": source,
                    },
                )
            )
        request.emit_progress(
            "runner_repair",
            (
                f"Applied deterministic resume repair for "
                f"{request.step.step_id}: {repair_name}."
            ),
            run_id=request.run_id,
            step_id=request.step.step_id,
            current_step=request.step_current,
            total_steps=request.total_steps,
        )
        return repaired_code

    def resume_critic_repair_code(self) -> Optional[str]:
        """Repair selected prior code from structured Critic feedback."""

        request = self.request
        report = request.resume_controller.prior_negative_critic_report_for_step(
            request.step.step_id
        )
        if report is None:
            return None
        resumed_code = request.resume_controller.prior_code_for_step(
            request.step.step_id
        )
        if resumed_code is None:
            return None
        prior_code = resumed_code[0]
        critique_log = (
            "PRIOR CRITIC REVIEW (binding repair requirements):\n"
            + json.dumps(report, indent=2, ensure_ascii=False, default=str)
        )
        authority = RepairPromptAuthority.create(
            typed_ticket=[
                {
                    "reason": "OUTPUT_CONTRACT_INVALID",
                    "validator": "critic_resume",
                    "detail": {"critic_report": report},
                }
            ]
        )
        if not request.consume_llm_repair_budget(
            "critic_resume",
            before_code=prior_code,
            repair_ticket=critique_log,
            repair_authority=authority,
            provider_category="critic_resume_repair",
            failure_status="critic_failed",
        ):
            return None
        prior_code = self.use_resumed_code(resumed_code)
        request.emit_progress(
            "coder",
            f"Repairing prior Critic findings for {request.step.step_id}.",
            status="warning",
            run_id=request.run_id,
            step_id=request.step.step_id,
            current_step=request.step_current,
            total_steps=request.total_steps,
        )
        try:
            repaired = self.repair_with_capsule(
                failure_status="critic_failed",
                context=request.coder_context,
                step=request.step,
                code=prior_code,
                run_log=critique_log,
                repair_authority=authority,
                attempt=1,
                provider_budget=request.provider_budget,
                provider_category="critic_resume_repair",
                logical_repair_attempt_id=(
                    request.step_repair_budget.llm_repair_attempts
                ),
            )
            request.sync_provider_budget()
        except (
            ProviderCallBudgetReceiptError,
            StepAuthorityRuntimeError,
            StepAuthorityCapsuleError,
        ):
            raise
        except Exception as exc:
            request.sync_provider_budget()
            with request.shared_lock:
                request.findings.append(
                    ValidationFinding(
                        validator="critic_resume_repair",
                        severity="warning",
                        message=(
                            "Prior Critic-guided repair was unavailable; falling "
                            "back to ordinary code generation."
                        ),
                        detail={
                            "step_id": request.step.step_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:300],
                        },
                    )
                )
            return None
        request.worker_progress.critic_resume_repair_used = True
        request.step_record["critic_resume_repair"] = True
        request.step_record["critic_resume_repair_status"] = report.get("status")
        return repaired


__all__ = ["StepCandidateRecovery", "StepCandidateRecoveryRequest"]
