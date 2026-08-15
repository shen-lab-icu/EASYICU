"""Step-authority coordinate preparation and crash-safe capsule resume."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from ..authority.coder_authority import HostCoderAuthority
from ..authority.evidence_store import sha256_of_file
from ..authority.provider_budget import (
    StepProviderCallBudget,
    load_provider_call_budget_state,
)
from ..authority.run_input import (
    canonical_sha256,
)
from ..authority.step_attempt import CheckpointAuthority, StepAttemptState
from ..authority.step_capsule import (
    StepAuthorityCapsuleError,
    StepAuthorityCapsuleRef,
    load_verified_step_authority_capsule,
)
from ..authority.step_runtime import (
    StepAuthorityRuntimeError,
    adopt_candidate_for_control_plane_revalidation,
    adopt_frozen_scoped_coder_context,
    capsule_matches_coordinates,
    coordinates_from_verified_capsule,
    initial_generation_code_ref,
    load_checkpoint_selected_step_capsule,
    prepare_step_authority_coordinates,
    repair_code_ref,
    seal_initial_generation_candidate,
    select_explicit_step_capsule_for_targeted_resume,
)
from ..schema import AnalysisStep, ResearchContext
from .concept_audit import (
    verified_capsule_concept_audit_replay as _verified_capsule_concept_audit_replay,
)


@dataclass(frozen=True)
class StepAuthorityResumeRequest:
    """All host authority required to select one resumable step capsule."""

    run_dir: Path
    step: AnalysisStep
    run_input_capsule_sha256: str
    deterministic_gate_stamp: Mapping[str, str]
    engine_code_sha256: str
    validator_code_sha256: str
    seal_repair_candidate: Callable[..., StepAuthorityCapsuleRef]
    coder_context: ResearchContext
    coder_authority: HostCoderAuthority
    coder_provider_identity_sha256: str
    resolved_inputs_path: Path
    resolved_input_bindings: Any
    resolved_input_evidence_ids: Sequence[str]
    cohort_path: Path
    universe_path: Path
    resume_state: Mapping[str, Any] | None
    requested_resume_from_step_id: Optional[str]
    prior_step_record: Mapping[str, Any] | None
    prior_attempt_records: Sequence[Mapping[str, Any]]
    prompt_version: str
    prompt_files: Mapping[str, Any]
    provider_budget: StepProviderCallBudget
    provider_receipt_path: Path
    reserved_final_category: Optional[str]
    llm_concept_auditor_identity_sha256: Optional[str]
    llm_concept_auditor_implementation_sha256: Optional[str]
    concept_audit_environment_sha256: str
    step_attempt_state: StepAttemptState
    checkpoint_authority: CheckpointAuthority
    step_record: Dict[str, Any]


def _prepare_current_coordinates(
    request: StepAuthorityResumeRequest,
) -> Mapping[str, str]:
    """Bind current planner, context, inputs, gates, engine, and prompt bytes."""

    run_dir = request.run_dir
    step = request.step
    coder_context = request.coder_context
    coder_authority = request.coder_authority
    coder_provider_identity_sha256 = request.coder_provider_identity_sha256
    resolved_inputs_path = request.resolved_inputs_path
    resolved_input_bindings = request.resolved_input_bindings
    resolved_input_evidence_ids = request.resolved_input_evidence_ids
    cohort_path = request.cohort_path
    universe_path = request.universe_path
    prompt_version = request.prompt_version
    prompt_files = request.prompt_files
    run_input_capsule_sha256 = request.run_input_capsule_sha256
    step_attempt_state = request.step_attempt_state

    gate_stamp = request.deterministic_gate_stamp
    step_control_plane_fingerprint = canonical_sha256(
        {
            "schema": "easyicu.step_control_plane_fingerprint/1",
            "deterministic_gate_fingerprint": gate_stamp[
                "deterministic_gate_fingerprint"
            ],
            "coder_provider_identity_sha256": (coder_provider_identity_sha256),
        }
    )
    step_attempt_state.coordinates = prepare_step_authority_coordinates(
        run_dir=run_dir,
        step_id=step.step_id,
        run_input_capsule_sha256=run_input_capsule_sha256,
        planner_scope=step.model_dump(mode="json"),
        scoped_coder_context={
            "research_context": coder_context.model_dump(mode="json"),
            "host_coder_authority": coder_authority.payload(),
        },
        resolved_inputs_path=resolved_inputs_path,
        typed_bindings=resolved_input_bindings,
        upstream_authority={
            "resolved_input_evidence_ids": list(resolved_input_evidence_ids),
            "resolved_input_bindings": resolved_input_bindings,
            "cohort_sha256": sha256_of_file(cohort_path),
            "universe_sha256": sha256_of_file(universe_path),
        },
        deterministic_gate_fingerprint=step_control_plane_fingerprint,
        engine_code_sha256=request.engine_code_sha256,
        validator_code_sha256=request.validator_code_sha256,
        prompt_pack_version=prompt_version,
        prompt_pack=prompt_files,
    )
    return gate_stamp


def _select_resume_candidate(
    request: StepAuthorityResumeRequest,
    *,
    gate_stamp: Mapping[str, str],
) -> None:
    """Select an explicit or checkpointed capsule under current gate identity."""

    run_dir = request.run_dir
    step = request.step
    resume_state = request.resume_state
    requested_resume_from_step_id = request.requested_resume_from_step_id
    prior_step_record = request.prior_step_record
    prior_attempt_records = request.prior_attempt_records
    step_attempt_state = request.step_attempt_state
    step_record = request.step_record

    step_attempt_state.selected_resume_capsule = load_checkpoint_selected_step_capsule(
        run_dir,
        step_id=step.step_id,
        checkpoint=(resume_state if isinstance(resume_state, Mapping) else None),
    )
    if (
        step_attempt_state.selected_resume_capsule is None
        and requested_resume_from_step_id == step.step_id
        and isinstance(prior_step_record, Mapping)
    ):
        explicit_selection = select_explicit_step_capsule_for_targeted_resume(
            run_dir,
            step_id=step.step_id,
            current_record=prior_step_record,
            records=prior_attempt_records,
            deterministic_gate_fingerprint=gate_stamp["deterministic_gate_fingerprint"],
        )
        if explicit_selection is not None:
            (
                step_attempt_state.selected_resume_capsule,
                explicit_metadata,
            ) = explicit_selection
            step_record.update(explicit_metadata)
    if (
        step_attempt_state.selected_resume_capsule is None
        and isinstance(prior_step_record, Mapping)
        and str(prior_step_record.get("status") or "").strip().lower()
        == "resume_validator_invalid"
        and isinstance(
            prior_step_record.get("resume_revalidation_candidate_capsule_ref"),
            Mapping,
        )
    ):
        try:
            recovery_ref = StepAuthorityCapsuleRef.model_validate(
                prior_step_record["resume_revalidation_candidate_capsule_ref"]
            )
            recovery_capsule = load_verified_step_authority_capsule(
                run_dir,
                ref=recovery_ref,
                expected_step_id=step.step_id,
            )
        except (ValueError, StepAuthorityCapsuleError) as exc:
            raise StepAuthorityRuntimeError(
                "resume revalidation candidate is invalid"
            ) from exc
        expected_recovery_sha256 = str(
            prior_step_record.get("resume_revalidation_candidate_code_sha256") or ""
        )
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_recovery_sha256)
            or recovery_capsule.capsule.candidate_code.sha256
            != expected_recovery_sha256
        ):
            raise StepAuthorityRuntimeError(
                "resume revalidation candidate digest is inconsistent"
            )
        step_attempt_state.selected_resume_capsule = recovery_capsule
        step_record["resume_validator_invalid_candidate_reused"] = True


def _recover_pending_repair(
    request: StepAuthorityResumeRequest,
) -> None:
    """Seal a paid repair transport against its exact historical parent."""

    run_dir = request.run_dir
    step = request.step
    prior_step_record = request.prior_step_record
    provider_receipt_path = request.provider_receipt_path
    reserved_final_category = request.reserved_final_category
    step_attempt_state = request.step_attempt_state
    checkpoint_authority = request.checkpoint_authority

    # A paid repair result belongs to the historical parent and
    # coordinates recorded before a crash. Recover that exact
    # candidate first; only then may current engine/validator drift
    # adopt its bytes for revalidation. Reversing this order makes
    # the receipt's before-code and authority binding impossible to
    # satisfy after a control-plane update.
    if (
        step_attempt_state.selected_resume_capsule is not None
        and isinstance(prior_step_record, Mapping)
        and prior_step_record.get("capsule_pending_repair_attempt_id") is not None
    ):
        step_attempt_state.current_capsule_ref = (
            step_attempt_state.selected_resume_capsule.ref
        )
        pending_attempt = prior_step_record.get("capsule_pending_repair_attempt_id")
        pending_binding = str(
            prior_step_record.get("capsule_pending_repair_binding_sha256") or ""
        )
        pending_failure_status = str(
            prior_step_record.get("capsule_pending_repair_failure_status") or ""
        )
        receipt_state = load_provider_call_budget_state(
            provider_receipt_path,
            step_id=step.step_id,
            expected_reserved_final_category=reserved_final_category,
        )
        if (
            isinstance(pending_attempt, bool)
            or not isinstance(pending_attempt, int)
            or not 1 <= pending_attempt <= len(receipt_state.logical_repairs)
            or str(
                receipt_state.logical_repairs[pending_attempt - 1].get("binding_sha256")
                or ""
            )
            != pending_binding
            or not pending_failure_status
        ):
            raise StepAuthorityRuntimeError(
                "completed repair lacks its exact pending checkpoint"
            )
        historical_coordinates = coordinates_from_verified_capsule(
            run_dir,
            step_attempt_state.selected_resume_capsule,
        )
        recovered_code_ref = repair_code_ref(
            receipt_state,
            attempt_id=pending_attempt,
        )
        recovered_ref = request.seal_repair_candidate(
            historical_coordinates,
            parent_ref=step_attempt_state.current_capsule_ref,
            checkpoint_parent_ref=step_attempt_state.current_capsule_ref,
            code_ref=recovered_code_ref,
            receipt_state=receipt_state,
            attempt_id=pending_attempt,
            failure_status=pending_failure_status,
        )
        checkpoint_authority.checkpoint_capsule(
            recovered_ref,
            status="candidate_checkpointed",
        )
        step_attempt_state.selected_resume_capsule = (
            load_verified_step_authority_capsule(
                run_dir,
                ref=recovered_ref,
                expected_step_id=step.step_id,
            )
        )


def _adopt_resume_context(
    request: StepAuthorityResumeRequest,
    *,
    coder_context: ResearchContext,
) -> ResearchContext:
    """Replay cached audit authority and adopt only provable control-plane drift."""

    run_dir = request.run_dir
    step = request.step
    prior_step_record = request.prior_step_record
    llm_concept_auditor_identity_sha256 = request.llm_concept_auditor_identity_sha256
    llm_concept_auditor_implementation_sha256 = (
        request.llm_concept_auditor_implementation_sha256
    )
    concept_audit_environment_sha256 = request.concept_audit_environment_sha256
    step_attempt_state = request.step_attempt_state
    step_record = request.step_record

    current_validator_identity = (
        llm_concept_auditor_implementation_sha256
        or canonical_sha256("llm_concept_auditor_unavailable")
    )
    if step_attempt_state.selected_resume_capsule is not None:
        source_audit_replay = _verified_capsule_concept_audit_replay(
            step_attempt_state.selected_resume_capsule,
            run_dir=run_dir,
            auditor_identity_sha256=(llm_concept_auditor_identity_sha256),
            environment_sha256=concept_audit_environment_sha256,
            validator_implementation_sha256=(current_validator_identity),
        )
        if source_audit_replay is not None:
            step_attempt_state.capsule_audit_findings_by_digest[
                step_attempt_state.selected_resume_capsule.capsule.candidate_code.sha256
            ] = source_audit_replay
    if step_attempt_state.selected_resume_capsule is not None and not (
        capsule_matches_coordinates(
            step_attempt_state.selected_resume_capsule,
            step_attempt_state.coordinates,
        )
    ):
        frozen_context = adopt_frozen_scoped_coder_context(
            step_attempt_state.selected_resume_capsule,
            step_attempt_state.coordinates,
        )
        if frozen_context is None:
            adopted_candidate = adopt_candidate_for_control_plane_revalidation(
                step_attempt_state.selected_resume_capsule,
                step_attempt_state.coordinates,
            )
            if adopted_candidate is None:
                step_record["step_authority_capsule_cache_miss"] = "authority_drift"
                step_attempt_state.selected_resume_capsule = None
            else:
                (
                    coder_context,
                    step_attempt_state.coordinates,
                    adopted_ref,
                ) = adopted_candidate
                step_attempt_state.current_capsule_ref = adopted_ref
                step_attempt_state.selected_resume_capsule = (
                    load_verified_step_authority_capsule(
                        run_dir,
                        ref=adopted_ref,
                        expected_step_id=step.step_id,
                    )
                )
                step_record["step_authority_capsule_cache_miss"] = (
                    "control_plane_drift_revalidation"
                )
        else:
            coder_context, step_attempt_state.coordinates = frozen_context
            step_record["step_authority_frozen_context_reused"] = True
    if (
        step_attempt_state.selected_resume_capsule is not None
        and isinstance(prior_step_record, Mapping)
        and prior_step_record.get("quarantined_requires_repair") is True
    ):
        step_record["step_authority_capsule_cache_miss"] = "quarantine_not_migrated"
        step_attempt_state.selected_resume_capsule = None
    return coder_context


def _checkpoint_resume_candidate(
    request: StepAuthorityResumeRequest,
) -> None:
    """Checkpoint a selected capsule or recover a paid initial generation."""

    run_dir = request.run_dir
    step = request.step
    prior_step_record = request.prior_step_record
    provider_budget = request.provider_budget
    provider_receipt_path = request.provider_receipt_path
    reserved_final_category = request.reserved_final_category
    step_attempt_state = request.step_attempt_state
    checkpoint_authority = request.checkpoint_authority
    step_record = request.step_record

    if step_attempt_state.selected_resume_capsule is not None:
        step_attempt_state.current_capsule_ref = (
            step_attempt_state.selected_resume_capsule.ref
        )
        step_record["step_authority_capsule_ref"] = (
            step_attempt_state.selected_resume_capsule.ref.model_dump(mode="json")
        )
        step_record["step_authority_capsule_stage"] = (
            step_attempt_state.selected_resume_capsule.capsule.stage
        )
        selected_digest = (
            step_attempt_state.selected_resume_capsule.capsule.candidate_code.sha256
        )
        if (
            step_attempt_state.selected_resume_capsule.capsule.concept_audit is not None
            and selected_digest
            not in step_attempt_state.capsule_audit_findings_by_digest
        ):
            step_record["step_authority_audit_cache_miss"] = "audit_identity_drift"
        checkpoint_authority.checkpoint_state("capsule_revalidation_pending")
    elif (
        provider_budget.initial_generation_resume_status() == "completed"
        and isinstance(prior_step_record, Mapping)
        and str(prior_step_record.get("status") or "") == "initial_generation_pending"
    ):
        initial_entry = provider_budget.initial_generation_entry
        pending_binding = str(
            prior_step_record.get("capsule_pending_initial_binding_sha256") or ""
        )
        pending_transport = str(
            prior_step_record.get("capsule_pending_initial_transport_id") or ""
        )
        if (
            initial_entry is None
            or pending_binding != str(initial_entry.get("binding_sha256") or "")
            or pending_transport
            != str(initial_entry.get("provider_transport_id") or "")
        ):
            raise StepAuthorityRuntimeError(
                "completed initial generation lacks its exact pending checkpoint"
            )
        recovered_code_ref = initial_generation_code_ref(
            step_attempt_state.coordinates,
            load_provider_call_budget_state(
                provider_receipt_path,
                step_id=step.step_id,
                expected_reserved_final_category=reserved_final_category,
            ),
        )
        recovered_ref = seal_initial_generation_candidate(
            step_attempt_state.coordinates,
            code_ref=recovered_code_ref,
            receipt_state=load_provider_call_budget_state(
                provider_receipt_path,
                step_id=step.step_id,
                expected_reserved_final_category=reserved_final_category,
            ),
        )
        checkpoint_authority.checkpoint_capsule(
            recovered_ref,
            status="candidate_checkpointed",
        )
        step_attempt_state.selected_resume_capsule = (
            load_verified_step_authority_capsule(
                run_dir,
                ref=recovered_ref,
                expected_step_id=step.step_id,
            )
        )


def prepare_step_authority_resume(
    request: StepAuthorityResumeRequest,
) -> ResearchContext:
    """Prepare current coordinates and recover only an exactly bound capsule."""

    gate_stamp = _prepare_current_coordinates(request)
    _select_resume_candidate(request, gate_stamp=gate_stamp)
    _recover_pending_repair(request)
    coder_context = _adopt_resume_context(
        request,
        coder_context=request.coder_context,
    )
    _checkpoint_resume_candidate(request)
    return coder_context


__all__ = [
    "StepAuthorityResumeRequest",
    "prepare_step_authority_resume",
]
