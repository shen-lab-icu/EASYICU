"""Adopt host-materialized cohort products without scheduling the Coder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    cohort_concept_ids: Sequence[str] = (),
) -> None:
    """Adopt a verified plan-phase cohort without scheduling the Coder."""

    result = load_materialized_analysis_cohort_result(
        run_dir=run_dir,
        plan=plan,
        cohort_concept_ids=cohort_concept_ids,
    )
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


def _materialize_approved_case_duration_if_needed(pipeline, *, plan_result, cohort_path, run_dir):
    """Transition a zero-row planning catalog only after signed host review."""
    from easyicu.databases.research_substitutions import case_duration_selection
    from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority

    if (not hasattr(plan_result.context, "user_preferences")
            or case_duration_selection(plan_result.context) is None):
        return cohort_path
    sealed_input = load_verified_materialized_cohort_authority(cohort_path)
    if sealed_input is not None:
        return cohort_path
    import pyarrow.parquet as pq
    if pq.ParquetFile(cohort_path).metadata.num_rows != 0:
        raise ValueError("case_duration_planning_input_is_not_metadata_only")
    from easyicu.databases.research_approval import require_approved_case_duration_pre_materialization

    services = getattr(pipeline, "_services", None)
    approved = require_approved_case_duration_pre_materialization(
        plan_result, run_dir, time_origin_attestor=getattr(services, "time_origin_attestor", None),
        case_duration_attestor=getattr(services, "case_duration_attestor", None),
        runtime_authority=getattr(getattr(pipeline, "_scientific_runtime_authorities", None), "current_case", None),
    )
    materializer = getattr(services, "approved_case_duration_materializer", None)
    if materializer is None:
        raise ValueError("case_duration_approved_materializer_unavailable")
    return Path(materializer(plan_result=plan_result, run_dir=run_dir, approved=approved))


def prepare_approved_analysis_cohort(*, pipeline, preparation, plan_result, plan, context, universe_path, run_dir):
    """Recover or materialize the approved source cohort before binding inputs."""
    from easyicu.databases.research_approval import ApprovedCaseDurationBinding
    from ..cohort.schema import (
        materialize_approved_care_unit_landmark,
        materialize_locked_analysis_cohort,
        recover_approved_source_analysis_cohort,
    )

    approved = preparation.approved_source_time_axis
    case_duration = approved if isinstance(approved, ApprovedCaseDurationBinding) else None
    path = run_dir / "cohort_analysis.parquet"
    if case_duration is not None:
        services = getattr(pipeline, "_services", None)
        runtime = getattr(getattr(pipeline, "_scientific_runtime_authorities", None), "current_case", None)
        result = materialize_approved_care_unit_landmark(
            run_dir=run_dir, universe_path=universe_path, plan_result=plan_result,
            time_origin_attestor=getattr(services, "time_origin_attestor", None),
            case_duration_attestor=getattr(services, "case_duration_attestor", None),
            runtime_authority=runtime,
        )
        path = Path(result["path"])
    elif approved is not None:
        prior = recover_approved_source_analysis_cohort(
            run_dir=run_dir, plan=plan, universe_path=universe_path,
            cohort_concept_ids=plan_result.cohort_concept_ids,
        )
        if prior is None:
            services = getattr(pipeline, "_services", None)
            runtime = getattr(getattr(pipeline, "_scientific_runtime_authorities", None), "current_case", None)
            materialize_locked_analysis_cohort(
                run_dir=run_dir, plan=plan, universe_path=universe_path, context=context,
                cohort_concept_ids=plan_result.cohort_concept_ids,
                approved_plan_result=plan_result,
                time_origin_attestor=getattr(services, "time_origin_attestor", None),
                runtime_authority=runtime,
            )
    return path, case_duration


def approved_database_source_time_axis(pipeline, plan_result, run_dir, cohort_path, require_approval):
    """Resolve the reviewed database time axis before initializing execution."""
    services = getattr(pipeline, "_services", None)
    return require_approval(
        plan_result, run_dir, cohort_path=cohort_path,
        time_origin_attestor=getattr(services, "time_origin_attestor", None),
        case_duration_attestor=getattr(services, "case_duration_attestor", None),
        runtime_authority=getattr(getattr(pipeline, "_scientific_runtime_authorities", None), "current_case", None),
    )


def bind_approved_execution_inputs(*, pipeline, preparation, plan_result, plan, context, universe_path,
                                   run_dir, trajectory_binding, binding_owner, legacy_trajectory_verifier):
    """Bind the approved analysis and its verified trajectory to this run."""
    analysis_path, case_duration = prepare_approved_analysis_cohort(
        pipeline=pipeline, preparation=preparation, plan_result=plan_result,
        plan=plan, context=context, universe_path=universe_path, run_dir=run_dir,
    )
    return binding_owner.bind(
        universe_path=universe_path, analysis_path=analysis_path,
        trajectory_binding=trajectory_binding, run_dir=run_dir,
        legacy_trajectory_verifier=legacy_trajectory_verifier, plan=plan, context=context,
        approved_case_duration_binding=case_duration,
    )
