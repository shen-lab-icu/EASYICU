"""Final deterministic validation for one sealed execution attempt.

This module owns the *read-only* final-review boundary of the execute phase.
It receives a step, its sealed outputs and the host validators, then returns
attempt-bound findings.  It deliberately does not write evidence, mutate a
checkpoint, select a repair, or decide an outer step status; those lifecycle
decisions remain with :mod:`execution.phase`.

Keeping this boundary separate prevents the execute orchestrator from growing
another domain-specific validation branch whenever a validator is added.  The
public contract is ``_evaluate_final_deterministic_gates`` plus
``_FinalDeterministicGateFindings``; the leading underscores are retained for
the existing execution compatibility surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..audits.envelope_consumers import StepSummaryFractionEnvelopeDualReader
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
from ..authority.plausibility import FlagOnlyPlausibilityScope
from ..contracts.runtime import ValidationFinding
from ..contracts.survival import SURVIVAL_PRIMARY_OWNER
from ..contracts.survival_execution import SURVIVAL_PRIMARY_ANALYSIS_KIND
from ..gates.contract import _step_deterministic_contract_findings
from ..gates.plausibility_receipt import plausibility_audit_receipt_findings
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext
from .cohort_routing import (
    bound_step_execution_cohort_path as _bound_step_execution_cohort_path,
    step_execution_cohort_path as _step_execution_cohort_path,
)
from .envelope_sealing import (
    SealedStepResultEnvelopeSnapshot,
    compile_sealed_step_result_shadow,
)
from .figure_preparation import _family_has_deterministic_figure_renderer


_PRIMARY_DETERMINISTIC_RUNNERS: set[str] = {SURVIVAL_PRIMARY_ANALYSIS_KIND}


def _bind_findings_to_step_attempt(
    findings: Sequence[ValidationFinding],
    *,
    step_id: str,
    attempt_id: str,
    checkpoint_id: str,
) -> List[ValidationFinding]:
    """Attach host-owned execution identity to deterministic findings."""

    bound: List[ValidationFinding] = []
    for finding in findings:
        detail = dict(finding.detail or {})
        detail.update(
            {
                "step_id": step_id,
                "attempt_id": attempt_id,
                "checkpoint_id": checkpoint_id,
            }
        )
        bound.append(finding.model_copy(update={"detail": detail}))
    return bound


def _primary_runner_core_estimate_present(
    kind: Optional[str], step_summary: Mapping[str, Any]
) -> bool:
    """Whether a registered host owner emitted its bound core estimate."""

    if kind not in _PRIMARY_DETERMINISTIC_RUNNERS:
        return False
    if not isinstance(step_summary, Mapping):
        return False
    if str(step_summary.get("status") or "").lower() != "ok":
        return False
    if kind == SURVIVAL_PRIMARY_ANALYSIS_KIND and (
        step_summary.get("receipt_issuer") != SURVIVAL_PRIMARY_OWNER
    ):
        return False
    if kind in ("causal_primary_iptw", "ordinal_dose_response"):
        return step_summary.get("adjusted_effect") is not None
    if step_summary.get("hazard_ratio") is not None:
        return True
    primary_model = step_summary.get("primary_model")
    return isinstance(primary_model, Mapping) and primary_model.get(
        "hazard_ratio"
    ) is not None


def _demote_step_contract_for_primary_runner(
    step_record: Mapping[str, Any],
    step_summary: Mapping[str, Any],
    findings: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Keep the legacy runner compatibility demotion deliberately narrow."""

    kind = step_record.get("deterministic_standard_analysis")
    if not _primary_runner_core_estimate_present(kind, step_summary):
        return list(findings)
    demoted: List[ValidationFinding] = []
    for finding in findings:
        if (
            getattr(finding, "validator", "") == "step_contract"
            and finding.severity == "error"
        ):
            finding = finding.model_copy(
                update={
                    "severity": "warning",
                    "message": (
                        finding.message
                        + f" [advisory: step satisfied by deterministic {kind} "
                        "runner; extra planner-requested outputs are non-blocking]"
                    ),
                }
            )
        demoted.append(finding)
    return demoted


def _is_too_few_panels_figure_finding(finding: ValidationFinding) -> bool:
    """Identify only the deterministic-family single-panel advisory case."""

    if getattr(finding, "validator", "") != "figure_contract_quality":
        return False
    if getattr(finding, "severity", "") != "error":
        return False
    detail = getattr(finding, "detail", None) or {}
    panel_count = detail.get("panel_count") if isinstance(detail, Mapping) else None
    return isinstance(panel_count, int) and panel_count < 2


def _demote_result_figure_shape_for_family_renderer(
    context: Any,
    findings: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Defer only the family-owned multi-panel shape check to the write phase."""

    if not any(_is_too_few_panels_figure_finding(finding) for finding in findings):
        return list(findings)
    if not _family_has_deterministic_figure_renderer(context):
        return list(findings)
    demoted: List[ValidationFinding] = []
    for finding in findings:
        if _is_too_few_panels_figure_finding(finding):
            finding = finding.model_copy(
                update={
                    "severity": "warning",
                    "message": (
                        finding.message
                        + " [advisory: this study-design family builds its "
                        "manuscript-facing primary figure deterministically in "
                        "the write phase; the display-suite gate remains the "
                        "fail-closed backstop for panel count and role diversity]"
                    ),
                }
            )
        demoted.append(finding)
    return demoted


@dataclass(frozen=True)
class _FinalDeterministicGateFindings:
    """Attempt-bound finding groups produced by the final deterministic gate."""

    stat_findings: Tuple[ValidationFinding, ...]
    clinical_findings: Tuple[ValidationFinding, ...]
    guard_findings: Tuple[ValidationFinding, ...]
    contract_findings: Tuple[ValidationFinding, ...]
    figure_source_findings: Tuple[ValidationFinding, ...]
    result_envelope_snapshot: Optional[SealedStepResultEnvelopeSnapshot] = None

    def all_findings(self) -> Tuple[ValidationFinding, ...]:
        """Return groups in the historical manifest publication order."""

        return (
            *self.stat_findings,
            *self.clinical_findings,
            *self.guard_findings,
            *self.contract_findings,
            *self.figure_source_findings,
        )


def _evaluate_final_deterministic_gates(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    cohort_path: Path,
    universe_path: Path,
    run_dir: Path,
    out_dir: Path,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    step_record: Mapping[str, Any],
    completed_step_records: Sequence[Mapping[str, Any]],
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
    plausibility_scope: FlagOnlyPlausibilityScope,
    script_text: str,
    attempt_id: str,
    checkpoint_id: str,
    stat_validator: StatisticalValidator,
    clinical_validator: ClinicalConstraintValidator,
    statistical_guard: StatisticalGuard,
    cross_step_cohort_lock_validator: CrossStepCohortLockValidator,
    cross_step_registered_output_validator: CrossStepRegisteredOutputValidator,
    cross_step_reconciliation_trace_validator: CrossStepReconciliationTraceValidator,
    step_summary_integrity_validator: Any,
    step_summary_fraction_validator: StepSummaryFractionValidator,
    cross_step_source_status_validator: CrossStepSourceStatusValidator,
    primary_model_contract_validator: PrimaryModelContractValidator,
    figure_contract_validator: FigureContractQualityValidator,
    figure_source_validator: FigureSourceDataValidator,
) -> _FinalDeterministicGateFindings:
    """Evaluate final deterministic review without changing run lifecycle state."""

    current_step_status = (
        str(step_record.get("status")).strip()
        if step_record.get("status") is not None
        else None
    )
    execution_cohort_path = _step_execution_cohort_path(
        step=step,
        plan=plan,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
    )
    execution_cohort_path = _bound_step_execution_cohort_path(
        run_dir=run_dir,
        fallback_path=execution_cohort_path,
        resolved_input_bindings=resolved_input_bindings,
    )
    execution_cohort_sha256 = str(
        step_record.get("execution_cohort_sha256") or ""
    ).strip()
    result_envelope_snapshot = compile_sealed_step_result_shadow(
        step=step,
        step_summary=step_summary,
        output_dir=out_dir,
        run_dir=run_dir,
        resolved_input_bindings=resolved_input_bindings,
        execution_cohort_path=(
            execution_cohort_path if execution_cohort_sha256 else None
        ),
        execution_cohort_sha256=execution_cohort_sha256 or None,
        current_status=current_step_status,
    )

    stat_findings = stat_validator.audit(
        context=context,
        cohort_path=execution_cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    clinical_findings = clinical_validator.audit(
        context=context,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    guard_findings = statistical_guard.audit(
        context=context,
        cohort_path=execution_cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    contract_findings = _step_deterministic_contract_findings(
        step=step,
        plan=plan,
        context=context,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
        resolved_input_bindings=resolved_input_bindings,
        out_dir=out_dir,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
        execution_cohort_path=execution_cohort_path,
        cross_step_cohort_lock_validator=cross_step_cohort_lock_validator,
        cross_step_registered_output_validator=cross_step_registered_output_validator,
        cross_step_reconciliation_trace_validator=(
            cross_step_reconciliation_trace_validator
        ),
        step_summary_integrity_validator=step_summary_integrity_validator,
        step_summary_fraction_validator=step_summary_fraction_validator,
        cross_step_source_status_validator=cross_step_source_status_validator,
        primary_model_contract_validator=primary_model_contract_validator,
        final_fraction_envelope_validator=StepSummaryFractionEnvelopeDualReader(),
        final_fraction_envelope=result_envelope_snapshot.envelope,
        final_fraction_current_status=current_step_status,
    )
    contract_findings.extend(
        figure_contract_validator.audit(
            step=step,
            out_dir=out_dir,
            run_dir=run_dir,
            step_summary=step_summary,
        )
    )
    contract_findings = _demote_step_contract_for_primary_runner(
        step_record,
        step_summary,
        contract_findings,
    )
    contract_findings = _demote_result_figure_shape_for_family_renderer(
        context,
        contract_findings,
    )
    contract_findings.extend(
        plausibility_audit_receipt_findings(
            step_summary=step_summary,
            step=step,
            script_text=script_text,
            scope=plausibility_scope,
        )
    )
    figure_source_findings = figure_source_validator.audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
        resolved_input_bindings=resolved_input_bindings,
    )

    def _bind(
        group: Sequence[ValidationFinding],
    ) -> Tuple[ValidationFinding, ...]:
        return tuple(
            _bind_findings_to_step_attempt(
                group,
                step_id=step.step_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
            )
        )

    return _FinalDeterministicGateFindings(
        stat_findings=_bind(stat_findings),
        clinical_findings=_bind(clinical_findings),
        guard_findings=_bind(guard_findings),
        contract_findings=_bind(contract_findings),
        figure_source_findings=_bind(figure_source_findings),
        result_envelope_snapshot=result_envelope_snapshot,
    )


__all__ = [
    "_FinalDeterministicGateFindings",
    "_PRIMARY_DETERMINISTIC_RUNNERS",
    "_bind_findings_to_step_attempt",
    "_demote_result_figure_shape_for_family_renderer",
    "_demote_step_contract_for_primary_runner",
    "_evaluate_final_deterministic_gates",
    "_is_too_few_panels_figure_finding",
    "_primary_runner_core_estimate_present",
]
