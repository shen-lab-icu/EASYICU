"""Digest-bound selective revalidation of resumed successful steps.

This module owns the append-only replay lifecycle: it rebuilds trusted views from
sealed evidence, reruns deterministic gates, propagates invalidation through
dependency edges, and commits alias retirement only after the checkpoint write.
The execute-phase orchestrator supplies its replaceable gate/checkpoint seams
through :class:`ResumeRevalidationServices`; this module never imports the
execute-phase god module.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..agents.core import CriticAgent
from ..audits.step_summary_integrity import StepSummaryIntegrityValidator
from ..audits.validators import (
    ClinicalConstraintValidator,
    CrossStepCohortLockValidator,
    CrossStepReconciliationTraceValidator,
    CrossStepRegisteredOutputValidator,
    CrossStepSourceStatusValidator,
    FigureContractQualityValidator,
    FigureSourceDataValidator,
    PrimaryModelContractValidator,
    StatisticalGuard,
    StatisticalValidator,
    StepSummaryFractionValidator,
)
from ..authority.plausibility import (
    compile_resumed_flag_only_plausibility_scope,
    restore_revalidated_resolved_inputs_sha256,
)
from ..authority.plan_scope import _serializable_plan_scientific_scope_signature
from ..authority.run_input import (
    RunInputIdentityError,
    _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND,
    _HOST_COHORT_MATERIALIZER_GENERATION_MODE,
    _host_cohort_materializer_authority_error,
    _host_probe_authority_error,
)
from ..authority.runtime_artifacts import (
    current_step_records,
    verified_run_evidence_path,
)
from ..authority.typed_binding import (
    _evidence_record_field,
    _resume_typed_input_bindings,
    _resume_typed_input_bindings_fingerprint,
)
from ..contracts.runtime import ValidationFinding
from ..schema import AnalysisPlan, EvidenceRef, ResearchContext
from .final_validation import (
    _FinalDeterministicGateFindings,
    _bind_findings_to_step_attempt,
)


@dataclass(frozen=True)
class ResumeDeterministicRevalidationResult:
    """Append-only resume ledger after selective deterministic replay."""

    resume_state: Dict[str, Any]
    revalidated_step_ids: Tuple[str, ...]
    invalidated_step_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ResumeRevalidationServices:
    """Replaceable execute-layer seams used by deterministic resume replay."""

    deterministic_gate_stamp: Callable[[], Mapping[str, str]]
    trusted_resume_success_records: Callable[
        ..., Tuple[List[Dict[str, Any]], Dict[str, str]]
    ]
    resume_success_dependencies: Callable[..., Dict[str, Set[str]]]
    verified_explicit_step_authority: Callable[..., Tuple[Any, Path]]
    verify_resume_step_script_lineage: Callable[..., None]
    materialize_verified_step_output_view: Callable[..., Dict[str, str]]
    project_verified_replay_output_paths: Callable[..., Dict[str, Any]]
    evaluate_final_deterministic_gates: Callable[..., _FinalDeterministicGateFindings]
    deterministic_code_gate_findings: Callable[..., Sequence[ValidationFinding]]
    actionable_validator_messages: Callable[..., List[str]]
    discard_stale_resolved_input_receipts: Callable[[Dict[str, Any]], None]
    write_run_checkpoint: Callable[[Path, Mapping[str, Any]], None]


def revalidate_resume_successes(
    *,
    resume_state: Dict[str, Any],
    plan: AnalysisPlan,
    context: ResearchContext,
    evidence: Any,
    run_dir: Path,
    cohort_path: Path,
    universe_path: Path,
    resume_from_step_id: Optional[str],
    development_sample: Optional[Any] = None,
    services: ResumeRevalidationServices,
) -> ResumeDeterministicRevalidationResult:
    """Replay changed deterministic gates against sealed evidence only.

    The function runs before :class:`ResumeController` applies skip decisions.
    It never invokes the coder, runner, analyzer, or LLM concept auditor.
    Successful replay appends a new ``ok`` authority checkpoint; any error
    appends ``resume_validator_invalid`` and makes the step executable again.
    """

    state = dict(resume_state)
    authority_history = [
        dict(record)
        for record in (resume_state.get("per_step_records") or [])
        if isinstance(record, Mapping)
    ]
    saved_attempt_history = [
        dict(record)
        for record in (resume_state.get("step_attempt_history") or [])
        if isinstance(record, Mapping)
    ]
    history = saved_attempt_history or list(authority_history)
    for authority_record in authority_history:
        if authority_record not in history:
            history.append(authority_record)
    # Resume audit history is append-only and may contain a newer invalidation
    # than the compact outer authority view (for example after an interrupted
    # execution attempt rewrote only the latter).  Resolve latest-per-step from
    # the merged monotonic history so validator-invalid checkpoints cannot
    # disappear and fall through to a second initial-generation purchase.
    current_records = [dict(record) for record in current_step_records(history)]
    current_successes = [
        record
        for record in current_records
        if str(record.get("status") or "").strip().lower() == "ok"
    ]
    steps_by_id = {step.step_id: step for step in plan.steps}
    step_order = {"00_probe": -1, **{s.step_id: i for i, s in enumerate(plan.steps)}}
    seeded_invalidated = {
        str(record.get("step_id") or "").strip(): (
            "prior checkpoint already lacks current resume authority "
            f"(status={str(record.get('status') or '').strip().lower()})"
        )
        for record in current_records
        if str(record.get("status") or "").strip().lower()
        in {"resume_evidence_invalid", "resume_validator_invalid"}
    }
    if resume_from_step_id and seeded_invalidated:
        cut = step_order.get(resume_from_step_id)
        earlier_invalid = sorted(
            step_id
            for step_id in seeded_invalidated
            if cut is not None and step_order.get(step_id, cut) < cut
        )
        if earlier_invalid:
            raise RunInputIdentityError(
                "Cannot start resume after an already-invalid upstream "
                "authority; resume at or before: " + ", ".join(earlier_invalid)
            )
    stamp = services.deterministic_gate_stamp()
    stale_successes = [
        record
        for record in current_successes
        if record.get("deterministic_gate_fingerprint")
        != stamp["deterministic_gate_fingerprint"]
    ]
    if not stale_successes and not seeded_invalidated:
        return ResumeDeterministicRevalidationResult(state, (), ())

    evidence_records = list(evidence.records())
    evidence_by_id = {
        str(_evidence_record_field(record, "evidence_id") or ""): record
        for record in evidence_records
    }
    trusted_records, trusted_summary_errors = services.trusted_resume_success_records(
        records=current_successes,
        evidence_by_id=evidence_by_id,
        run_dir=run_dir,
    )
    trusted_by_step = {
        str(record.get("step_id") or ""): record for record in trusted_records
    }
    current_by_step = {
        str(record.get("step_id") or ""): record for record in current_successes
    }
    dependencies = services.resume_success_dependencies(
        plan=plan,
        current_records=current_records,
        evidence_by_id=evidence_by_id,
    )
    invalidated: Dict[str, str] = dict(seeded_invalidated)
    revalidated: List[str] = []
    invalid_payloads: Dict[str, Dict[str, Any]] = {}
    retirement_records: Dict[str, Mapping[str, Any]] = {}

    def attempt_identity(step_id: str) -> Tuple[str, str]:
        sequence = 1 + sum(
            1
            for record in history
            if str(record.get("step_id") or "") == step_id
            and record.get("revalidated_without_execution") is True
        )
        attempt_id = f"{step_id}:resume_revalidation:{sequence}"
        return attempt_id, f"{attempt_id}:deterministic_review"

    def indexed_alias_evidence_ids(prior_record: Mapping[str, Any]) -> List[str]:
        step_id = str(prior_record.get("step_id") or "").strip()
        indexed_ids: List[str] = []
        for raw_id in prior_record.get("evidence_ids") or []:
            evidence_id = str(raw_id).strip()
            authority = evidence_by_id.get(evidence_id)
            if (
                authority is not None
                and str(_evidence_record_field(authority, "produced_by_step") or "")
                == step_id
            ):
                indexed_ids.append(evidence_id)
        return list(dict.fromkeys(indexed_ids))

    for invalid_step_id in seeded_invalidated:
        prior_success = next(
            (
                record
                for record in reversed(history)
                if str(record.get("step_id") or "").strip() == invalid_step_id
                and str(record.get("status") or "").strip().lower() == "ok"
            ),
            None,
        )
        if prior_success is not None:
            retirement_records[invalid_step_id] = prior_success
            current_invalid = next(
                (
                    record
                    for record in reversed(history)
                    if str(record.get("step_id") or "").strip() == invalid_step_id
                    and str(record.get("status") or "").strip().lower()
                    in {"resume_evidence_invalid", "resume_validator_invalid"}
                ),
                None,
            )
            raw_capsule_ref = prior_success.get("step_authority_capsule_ref")
            prior_code_sha256 = str(
                prior_success.get("executed_code_sha256")
                or prior_success.get("concept_approved_code_sha256")
                or ""
            )
            if (
                isinstance(current_invalid, Mapping)
                and "resume_revalidation_candidate_capsule_ref" not in current_invalid
                and isinstance(raw_capsule_ref, Mapping)
                and re.fullmatch(r"[0-9a-f]{64}", prior_code_sha256)
            ):
                # Append a monotonic continuation for invalid checkpoints
                # written before recovery coordinates existed. Never mutate
                # the historical checkpoint or reset its provider receipt.
                history.append(
                    {
                        **dict(current_invalid),
                        "attempt_id": (
                            f"{str(current_invalid.get('attempt_id') or invalid_step_id)}"
                            ":candidate_recovery"
                        ),
                        "resume_revalidation_candidate_capsule_ref": dict(
                            raw_capsule_ref
                        ),
                        "resume_revalidation_candidate_code_sha256": (
                            prior_code_sha256
                        ),
                        "resume_revalidation_candidate_attempt_id": str(
                            prior_success.get("attempt_id") or ""
                        ),
                    }
                )

    def append_invalid(
        *,
        prior_record: Mapping[str, Any],
        reason: str,
        code_findings: Sequence[ValidationFinding] = (),
        gate_findings: Optional[_FinalDeterministicGateFindings] = None,
    ) -> None:
        step_id = str(prior_record.get("step_id") or "").strip()
        if step_id in invalidated:
            return
        attempt_id, checkpoint_id = attempt_identity(step_id)
        if not code_findings and gate_findings is None:
            code_findings = _bind_findings_to_step_attempt(
                [
                    ValidationFinding(
                        validator="resume_deterministic_revalidation",
                        severity="error",
                        message=(
                            f"Prior success for step {step_id} failed current "
                            "deterministic replay."
                        ),
                        detail={"reason": reason},
                    )
                ],
                step_id=step_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
            )
        payload: Dict[str, Any] = {
            "step_id": step_id,
            "status": "resume_validator_invalid",
            "revalidated_without_execution": True,
            "attempt_id": attempt_id,
            "review_checkpoint_id": checkpoint_id,
            "resume_invalidation_reason": reason,
            "invalidated_evidence_ids": list(prior_record.get("evidence_ids") or []),
            "evidence_ids": [],
            "deterministic_code_findings": [
                finding.model_dump(mode="json") for finding in code_findings
            ],
            "retired_current_aliases": {},
            **stamp,
        }
        raw_capsule_ref = prior_record.get("step_authority_capsule_ref")
        prior_code_sha256 = str(
            prior_record.get("executed_code_sha256")
            or prior_record.get("concept_approved_code_sha256")
            or ""
        )
        if isinstance(raw_capsule_ref, Mapping) and re.fullmatch(
            r"[0-9a-f]{64}", prior_code_sha256
        ):
            # Invalid status retires current authority, but this explicit
            # immutable coordinate lets the next attempt revalidate the exact
            # candidate without purchasing a second initial generation.
            payload.update(
                {
                    "resume_revalidation_candidate_capsule_ref": dict(raw_capsule_ref),
                    "resume_revalidation_candidate_code_sha256": (prior_code_sha256),
                    "resume_revalidation_candidate_attempt_id": str(
                        prior_record.get("attempt_id") or ""
                    ),
                }
            )
        for key, value in prior_record.items():
            if key.startswith("step_provider_call_") or key.startswith(
                "step_llm_repair_"
            ):
                payload[key] = value
        if gate_findings is not None:
            payload.update(
                {
                    "stat_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.stat_findings
                    ],
                    "clinical_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.clinical_findings
                    ],
                    "guard_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.guard_findings
                    ],
                    "contract_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.contract_findings
                    ],
                    "figure_source_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.figure_source_findings
                    ],
                }
            )
        invalidated[step_id] = reason
        invalid_payloads[step_id] = payload
        retirement_records[step_id] = prior_record
        history.append(payload)

    stale_successes.sort(
        key=lambda record: step_order.get(
            str(record.get("step_id") or ""), len(step_order)
        )
    )
    for prior_record in stale_successes:
        prior_record = restore_revalidated_resolved_inputs_sha256(
            prior_record=prior_record,
            checkpoint_history=history,
            run_dir=run_dir,
        )
        step_id = str(prior_record.get("step_id") or "").strip()
        invalid_upstream = sorted(
            dependencies.get(step_id, set()).intersection(invalidated)
        )
        if invalid_upstream:
            append_invalid(
                prior_record=prior_record,
                reason=(
                    "current success depends on invalidated upstream step(s): "
                    + ", ".join(invalid_upstream)
                ),
            )
            continue
        if step_id == "00_probe":
            summary_error = trusted_summary_errors.get(step_id)
            if summary_error is not None or step_id not in trusted_by_step:
                append_invalid(
                    prior_record=prior_record,
                    reason=(summary_error or "probe summary authority is unavailable"),
                )
                continue
            evidence_payloads = {
                evidence_id: (
                    record.model_dump(mode="json")
                    if hasattr(record, "model_dump")
                    else dict(record)
                )
                for evidence_id, record in evidence_by_id.items()
            }
            error = _host_probe_authority_error(
                record=prior_record,
                evidence_ids=list(prior_record.get("evidence_ids") or []),
                step_id=step_id,
                run_dir=run_dir,
                records=evidence_payloads,
            )
            if error is not None:
                append_invalid(prior_record=prior_record, reason=error)
                continue
            attempt_id, checkpoint_id = attempt_identity(step_id)
            summary = trusted_by_step[step_id]["step_summary"]
            replayed = {
                **prior_record,
                "status": "ok",
                "step_summary": dict(summary),
                "revalidated_without_execution": True,
                "attempt_id": attempt_id,
                "review_checkpoint_id": checkpoint_id,
                **stamp,
            }
            services.discard_stale_resolved_input_receipts(replayed)
            history.append(replayed)
            trusted_by_step[step_id] = replayed
            revalidated.append(step_id)
            continue

        is_host_cohort_materializer = (
            str(prior_record.get("generation_mode") or "").strip().lower()
            == _HOST_COHORT_MATERIALIZER_GENERATION_MODE
            or prior_record.get("step_authority_kind")
            == _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
        )
        if is_host_cohort_materializer:
            evidence_payloads = {
                evidence_id: (
                    record.model_dump(mode="json")
                    if hasattr(record, "model_dump")
                    else dict(record)
                )
                for evidence_id, record in evidence_by_id.items()
            }
            error = _host_cohort_materializer_authority_error(
                record=prior_record,
                evidence_ids=list(prior_record.get("evidence_ids") or []),
                step_id=step_id,
                run_dir=run_dir,
                records=evidence_payloads,
            )
            if error is not None:
                append_invalid(prior_record=prior_record, reason=error)
                continue
            attempt_id, checkpoint_id = attempt_identity(step_id)
            replayed = {
                **prior_record,
                "status": "ok",
                "step_summary": dict(prior_record["step_summary"]),
                "plan_scientific_signature": (
                    _serializable_plan_scientific_scope_signature(plan)
                ),
                "revalidated_without_execution": True,
                "attempt_id": attempt_id,
                "review_checkpoint_id": checkpoint_id,
                **stamp,
            }
            services.discard_stale_resolved_input_receipts(replayed)
            history.append(replayed)
            trusted_by_step[step_id] = replayed
            revalidated.append(step_id)
            continue

        step = steps_by_id.get(step_id)
        summary_error = trusted_summary_errors.get(step_id)
        if step is None or summary_error is not None:
            append_invalid(
                prior_record=prior_record,
                reason=(summary_error or "successful step is absent from active plan"),
            )
            continue
        trusted_record = trusted_by_step[step_id]
        attempt_id, checkpoint_id = attempt_identity(step_id)
        try:
            services.verify_resume_step_script_lineage(
                record=prior_record,
                evidence_by_id=evidence_by_id,
            )
            _, script_path = services.verified_explicit_step_authority(
                record=prior_record,
                field="script_evidence_id",
                expected_kind="code",
                expected_source_name=None,
                evidence_by_id=evidence_by_id,
                run_dir=run_dir,
            )
            script_text = script_path.read_text(encoding="utf-8")
            plausibility_scope = compile_resumed_flag_only_plausibility_scope(
                prior_record=prior_record,
                run_dir=run_dir,
                context=context,
                step=step,
            )
            code_findings = _bind_findings_to_step_attempt(
                services.deterministic_code_gate_findings(
                    context=context,
                    step=step,
                    script_text=script_text,
                    plausibility_scope=plausibility_scope,
                ),
                step_id=step_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
            )
            if any(finding.severity == "error" for finding in code_findings):
                append_invalid(
                    prior_record=prior_record,
                    reason="current deterministic code preflight failed",
                    code_findings=code_findings,
                )
                continue
            trusted_current_records = [
                record
                for record in trusted_by_step.values()
                if str(record.get("status") or "").lower() == "ok"
                and str(record.get("step_id") or "") not in invalidated
            ]
            resolved_bindings, resolved_input_evidence_ids = (
                _resume_typed_input_bindings(
                    step=step,
                    plan=plan,
                    evidence_records=evidence_records,
                    trusted_step_records=trusted_current_records,
                    run_dir=run_dir,
                    cohort_path=cohort_path,
                    development_sample=development_sample,
                )
            )
            with tempfile.TemporaryDirectory(
                prefix=f".resume_gate_{step_id}_",
                dir=run_dir,
            ) as temporary_root:
                replay_out_dir = Path(temporary_root) / "outputs"
                materialized_outputs = services.materialize_verified_step_output_view(
                    record=prior_record,
                    evidence_by_id=evidence_by_id,
                    run_dir=run_dir,
                    destination=replay_out_dir,
                )
                replay_step_summary = services.project_verified_replay_output_paths(
                    trusted_record["step_summary"],
                    materialized_evidence_by_source_name=materialized_outputs,
                )
                completed_records = [
                    record
                    for record in trusted_current_records
                    if str(record.get("step_id") or "") != step_id
                    and step_order.get(str(record.get("step_id") or ""), -1)
                    < step_order.get(step_id, len(step_order))
                ]
                gate_findings = services.evaluate_final_deterministic_gates(
                    context=context,
                    plan=plan,
                    cohort_path=cohort_path,
                    universe_path=universe_path,
                    run_dir=run_dir,
                    out_dir=replay_out_dir,
                    step=step,
                    step_summary=replay_step_summary,
                    step_record=prior_record,
                    completed_step_records=completed_records,
                    resolved_input_bindings=resolved_bindings,
                    plausibility_scope=plausibility_scope,
                    script_text=script_text,
                    attempt_id=attempt_id,
                    checkpoint_id=checkpoint_id,
                    stat_validator=StatisticalValidator(),
                    clinical_validator=ClinicalConstraintValidator(),
                    statistical_guard=StatisticalGuard(),
                    cross_step_cohort_lock_validator=CrossStepCohortLockValidator(),
                    cross_step_registered_output_validator=(
                        CrossStepRegisteredOutputValidator()
                    ),
                    cross_step_reconciliation_trace_validator=(
                        CrossStepReconciliationTraceValidator()
                    ),
                    step_summary_integrity_validator=StepSummaryIntegrityValidator(),
                    step_summary_fraction_validator=StepSummaryFractionValidator(),
                    cross_step_source_status_validator=CrossStepSourceStatusValidator(),
                    primary_model_contract_validator=PrimaryModelContractValidator(),
                    figure_contract_validator=FigureContractQualityValidator(),
                    figure_source_validator=FigureSourceDataValidator(),
                )
            if any(
                finding.severity == "error" for finding in gate_findings.all_findings()
            ):
                append_invalid(
                    prior_record=prior_record,
                    reason="current deterministic artifact gates failed",
                    code_findings=code_findings,
                    gate_findings=gate_findings,
                )
                continue
            prior_critique = prior_record.get("critique_report")
            prior_critique_status = (
                str(prior_critique.get("status") or "").strip().lower()
                if isinstance(prior_critique, Mapping)
                else ""
            )
            if prior_critique_status in {"blocked", "needs_revision"}:
                append_invalid(
                    prior_record=prior_record,
                    reason=(
                        "prior deterministic Critic status remains "
                        f"{prior_critique_status}"
                    ),
                    code_findings=code_findings,
                    gate_findings=gate_findings,
                )
                continue
            evidence_refs = [
                EvidenceRef(
                    evidence_id=str(_evidence_record_field(authority, "evidence_id")),
                    kind=_evidence_record_field(authority, "kind"),
                    description=str(
                        _evidence_record_field(authority, "description") or ""
                    ),
                    relative_path=str(
                        _evidence_record_field(authority, "relative_path") or ""
                    ),
                )
                for evidence_id in (prior_record.get("evidence_ids") or [])
                if (authority := evidence_by_id.get(str(evidence_id))) is not None
                and verified_run_evidence_path(run_dir, authority) is not None
            ]
            critique = CriticAgent().review_step(
                step=step,
                step_summary=dict(trusted_record["step_summary"]),
                evidence_refs=evidence_refs,
                findings=services.actionable_validator_messages(
                    code_findings,
                    gate_findings.all_findings(),
                ),
            )
            if critique.status != "pass":
                append_invalid(
                    prior_record=prior_record,
                    reason=f"current deterministic Critic status={critique.status}",
                    code_findings=code_findings,
                    gate_findings=gate_findings,
                )
                continue
        except (OSError, TypeError, UnicodeError, ValueError) as exc:
            append_invalid(
                prior_record=prior_record,
                reason=f"{type(exc).__name__}: {exc}",
            )
            continue

        replayed = {
            **prior_record,
            "status": "ok",
            "step_summary": dict(trusted_record["step_summary"]),
            "resolved_input_evidence_ids": resolved_input_evidence_ids,
            "deterministic_code_findings": [
                finding.model_dump(mode="json") for finding in code_findings
            ],
            "stat_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.stat_findings
            ],
            "clinical_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.clinical_findings
            ],
            "guard_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.guard_findings
            ],
            "contract_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.contract_findings
            ],
            "figure_source_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.figure_source_findings
            ],
            "critique_report": critique.model_dump(mode="json"),
            "revalidated_without_execution": True,
            "attempt_id": attempt_id,
            "review_checkpoint_id": checkpoint_id,
            **stamp,
        }
        services.discard_stale_resolved_input_receipts(replayed)
        replayed["resolved_inputs_sha256"] = prior_record["resolved_inputs_sha256"]
        replayed["flag_only_plausibility_scope"] = plausibility_scope.to_dict()
        replayed["revalidated_input_bindings_fingerprint"] = (
            _resume_typed_input_bindings_fingerprint(resolved_bindings)
        )
        history.append(replayed)
        trusted_by_step[step_id] = replayed
        revalidated.append(step_id)

    # Propagate invalid authority through immutable plan/evidence edges, even
    # when a downstream success already carries the current fingerprint.
    while True:
        changed = False
        for step_id, prior_record in current_by_step.items():
            if step_id in invalidated:
                continue
            failed_dependencies = sorted(
                dependencies.get(step_id, set()).intersection(invalidated)
            )
            if not failed_dependencies:
                continue
            append_invalid(
                prior_record=prior_record,
                reason=(
                    "current success depends on invalidated upstream step(s): "
                    + ", ".join(failed_dependencies)
                ),
            )
            changed = True
        if not changed:
            break

    # An explicit cut after a newly detected invalid authority must fail
    # before any alias or manifest mutation.  The caller can restart at or
    # before the earliest invalid upstream step.
    if resume_from_step_id and invalidated:
        cut = step_order.get(resume_from_step_id)
        earlier_invalid = sorted(
            step_id
            for step_id in invalidated
            if cut is not None and step_order.get(step_id, cut) < cut
        )
        if earlier_invalid:
            raise RunInputIdentityError(
                "Cannot start resume after deterministic-validator-invalid "
                "upstream evidence; resume at or before: " + ", ".join(earlier_invalid)
            )

    if invalid_payloads:
        state_findings = list(resume_state.get("findings") or [])
        for step_id, payload in invalid_payloads.items():
            reason = str(payload.get("resume_invalidation_reason") or "")
            state_findings.append(
                ValidationFinding(
                    validator="resume_deterministic_revalidation",
                    severity="warning",
                    message=(
                        f"Prior success for step {step_id} was invalidated by "
                        "current deterministic gates and requires re-execution."
                    ),
                    detail={
                        "step_id": step_id,
                        "reason": reason,
                        "requires_reexecution": True,
                    },
                ).model_dump(mode="json")
            )
        state["findings"] = state_findings
    state["step_attempt_history"] = history
    state["per_step_records"] = [
        dict(record) for record in current_step_records(history)
    ]

    retirement_batch = {
        step_id: evidence_ids
        for step_id, prior_record in retirement_records.items()
        if (evidence_ids := indexed_alias_evidence_ids(prior_record))
    }
    current_aliases = evidence.aliases() if retirement_batch else {}
    for step_id, evidence_ids in retirement_batch.items():
        payload = invalid_payloads.get(step_id)
        if payload is not None:
            payload["retired_current_aliases"] = {
                alias: evidence_id
                for alias, evidence_id in current_aliases.items()
                if evidence_id in set(evidence_ids)
            }

    # Persist the append-only checkpoint before revoking aliases.  A failed
    # checkpoint write leaves aliases untouched; the batch retirement itself
    # is atomic across every invalid step.  If retirement fails, restore the
    # prior manifest so the two authority ledgers cannot disagree.
    checkpoint_path = run_dir / "manifest_partial.json"
    services.write_run_checkpoint(checkpoint_path, state)
    if retirement_batch:
        try:
            evidence.retire_steps_current_aliases(retirement_batch)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            try:
                services.write_run_checkpoint(checkpoint_path, resume_state)
            except (OSError, TypeError, ValueError) as rollback_exc:
                raise RuntimeError(
                    "resume revalidation alias retirement and manifest rollback "
                    "both failed"
                ) from rollback_exc
            raise RuntimeError(
                "resume revalidation alias retirement failed; manifest was rolled back"
            ) from exc

    return ResumeDeterministicRevalidationResult(
        resume_state=state,
        revalidated_step_ids=tuple(revalidated),
        invalidated_step_ids=tuple(sorted(invalidated)),
    )
