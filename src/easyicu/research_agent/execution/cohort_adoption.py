"""Adopt host-materialized cohort products without scheduling the Coder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..authority.evidence_store import EvidenceStore
from ..authority.plan_scope import _serializable_plan_scientific_scope_signature
from ..authority.run_input import (
    _declares_host_cohort_products,
    _planned_host_cohort_checkpoint,
)
from ..cohort.schema import load_materialized_analysis_cohort_result
from ..schema import AnalysisPlan, ValidationFinding


def stage_candidate_cohort_plan(
    plan: AnalysisPlan,
    definition: Any,
) -> AnalysisPlan:
    """Return an isolated plan carrying a not-yet-authoritative cohort."""

    return plan.model_copy(deep=True, update={"cohort": definition})


def commit_staged_cohort_plan(
    live_plan: AnalysisPlan,
    staged_plan: AnalysisPlan,
    *,
    materialization_status: object,
    authority_state: Any,
    context: Any,
) -> bool:
    """Commit a staged cohort only after data application and authority bind."""

    if str(materialization_status or "").strip().casefold() != "applied":
        return False
    authority_state.rebind_cohort(plan=staged_plan, context=context)
    live_plan.cohort = staged_plan.cohort
    return True


def record_planned_host_cohort_checkpoint(
    *,
    plan: AnalysisPlan,
    result: Mapping[str, Any],
    cohort_path: Path,
    evidence: EvidenceStore,
    prompt_pack_version: str,
    llm_signature: str,
    run_dir: Path,
    reason: str,
    gate_stamp: Mapping[str, Any],
    per_step_records: List[Dict[str, Any]],
    preexecuted_step_ids: set[str],
    findings: List[ValidationFinding],
    budget_snapshot: Optional[Mapping[str, Any]] = None,
) -> None:
    """Seal one host-owned cohort producer and select it for this execution."""

    step_ids = [
        str(step.step_id) for step in plan.steps if _declares_host_cohort_products(step)
    ]
    if len(step_ids) != 1 or step_ids[0] in preexecuted_step_ids:
        return
    step_id, checkpoint, authority_error = _planned_host_cohort_checkpoint(
        plan=plan,
        result=result,
        cohort_path=cohort_path,
        evidence=evidence,
        prompt_pack_version=prompt_pack_version,
        llm_signature=llm_signature,
        run_dir=run_dir,
        reason=reason,
        gate_stamp=gate_stamp,
        budget_snapshot=budget_snapshot,
    )
    if checkpoint is not None and step_id is not None:
        checkpoint["plan_scientific_signature"] = (
            _serializable_plan_scientific_scope_signature(plan)
        )
        per_step_records.append(checkpoint)
        preexecuted_step_ids.add(step_id)
    elif authority_error is not None and step_id is not None:
        findings.append(
            ValidationFinding(
                validator="cohort_materializer_authority",
                severity="error",
                message=(
                    "Host cohort materializer could not seal its exact "
                    "planned-product authority."
                ),
                detail={"step_id": step_id, "reason": authority_error},
            )
        )


def adopt_existing_host_cohort_materialization(
    *,
    plan: AnalysisPlan,
    run_dir: Path,
    cohort_path: Path,
    evidence: EvidenceStore,
    prompt_pack_version: str,
    llm_signature: str,
    gate_stamp: Mapping[str, Any],
    per_step_records: List[Dict[str, Any]],
    preexecuted_step_ids: set[str],
    findings: List[ValidationFinding],
) -> None:
    """Adopt a verified plan-phase cohort without scheduling the Coder."""

    result = load_materialized_analysis_cohort_result(run_dir=run_dir, plan=plan)
    if result is None:
        return
    record_planned_host_cohort_checkpoint(
        plan=plan,
        result=result,
        cohort_path=cohort_path,
        evidence=evidence,
        prompt_pack_version=prompt_pack_version,
        llm_signature=llm_signature,
        run_dir=run_dir,
        reason="locked_plan_cohort_materialization",
        gate_stamp=gate_stamp,
        per_step_records=per_step_records,
        preexecuted_step_ids=preexecuted_step_ids,
        findings=findings,
    )
