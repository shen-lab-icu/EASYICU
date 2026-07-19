"""Normalize replanner candidates without owning provider or run mutation.

The Planner/Replanner retains every scientific choice.  This module only
projects a returned candidate through host-owned invariants before the execute
orchestrator decides whether to register it.  Provider calls, plan revision
files, EvidenceStore promotion, cohort materialization, runner rebuilding, and
replan-budget state deliberately remain outside this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..plan_utils import (
    _augment_measurement_companion_inputs,
    _augment_report_typed_product_inputs,
    _cap_plan_preserving_figure_steps,
    _preserve_figure_steps_after_replan,
    _preserve_primary_estimand_step_after_replan,
)
from ..robustness_panel import (
    RobustnessSpec,
    robustness_specs_for_execution,
    robustness_specs_sha,
)
from ..authority.runtime_artifacts import current_successful_step_records
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext, ValidationFinding
from ..trajectory.plan_contract import augment_trajectory_plan_products
from .plan_scope import _plan_scientific_scope_signature, _plan_signature

__all__ = [
    "NormalizedPlanCandidate",
    "_preserve_completed_step_snapshots_after_replan",
    "_preserve_locked_robustness_specs_after_replan",
    "normalize_replan_candidate",
]


@dataclass(frozen=True)
class NormalizedPlanCandidate:
    """One immutable candidate projection returned to the orchestrator."""

    plan: AnalysisPlan
    findings: Tuple[ValidationFinding, ...]
    substantive: bool


def _preserve_completed_step_snapshots_after_replan(
    *,
    current_plan: AnalysisPlan,
    revised_plan: AnalysisPlan,
    completed_records: Sequence[Mapping[str, Any]],
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Keep already-executed Planner steps immutable across replans.

    A replanner may change future work, but it cannot retroactively change the
    scientific request that produced registered evidence. The host-recorded
    ``analysis_request.step`` snapshot and the current plan-level scientific
    scope are execution authority. Replacing either would launder stale evidence
    or permanently block every downstream typed consumer, so restore them before
    accepting the revised DAG.
    """

    current_ids = {str(step.step_id) for step in current_plan.steps}
    snapshots: Dict[str, AnalysisStep] = {}
    completed_current_records = [
        record
        for record in current_successful_step_records(completed_records)
        if str(record.get("step_id") or "").strip() in current_ids
    ]
    for record in completed_current_records:
        step_id = str(record.get("step_id") or "").strip()
        analysis_request = record.get("analysis_request")
        raw_step = (
            analysis_request.get("step")
            if isinstance(analysis_request, Mapping)
            else None
        )
        if step_id not in current_ids or not isinstance(raw_step, Mapping):
            continue
        try:
            snapshot = AnalysisStep.model_validate(raw_step)
        except (TypeError, ValueError):
            continue
        if str(snapshot.step_id) == step_id:
            snapshots[step_id] = snapshot
    changed_ids: List[str] = []
    revised_steps: List[AnalysisStep] = []
    revised_ids: Set[str] = set()
    for step in revised_plan.steps:
        step_id = str(step.step_id)
        snapshot = snapshots.get(step_id)
        if snapshot is not None:
            revised_ids.add(step_id)
            if step.model_dump(mode="json") != snapshot.model_dump(mode="json"):
                changed_ids.append(step_id)
            revised_steps.append(snapshot)
        else:
            revised_steps.append(step)
            revised_ids.add(step_id)

    reinserted_ids: List[str] = []
    current_positions = {
        str(step.step_id): index for index, step in enumerate(current_plan.steps)
    }
    for step_id in sorted(
        snapshots,
        key=lambda value: current_positions.get(value, len(current_positions)),
    ):
        if step_id in revised_ids:
            continue
        insert_at = min(
            current_positions.get(step_id, len(revised_steps)), len(revised_steps)
        )
        revised_steps.insert(insert_at, snapshots[step_id])
        revised_ids.add(step_id)
        reinserted_ids.append(step_id)

    current_scope = _plan_scientific_scope_signature(current_plan)
    revised_scope = _plan_scientific_scope_signature(revised_plan)
    restored_plan_scope = bool(completed_current_records) and (
        revised_scope != current_scope
    )
    restored_plan_scope_fields: List[str] = []
    if restored_plan_scope:
        for field_name in (
            "research_question",
            "analysis_type",
            "cohort",
            "robustness_specs",
            "rationale",
        ):
            if getattr(revised_plan, field_name) != getattr(current_plan, field_name):
                restored_plan_scope_fields.append(field_name)

    if not changed_ids and not reinserted_ids and not restored_plan_scope:
        return revised_plan, []
    update: Dict[str, Any] = {"steps": revised_steps}
    if restored_plan_scope:
        update.update(
            {
                "research_question": current_plan.research_question,
                "analysis_type": current_plan.analysis_type,
                "cohort": current_plan.cohort,
                "robustness_specs": current_plan.robustness_specs,
                "rationale": current_plan.rationale,
            }
        )
    preserved = revised_plan.model_copy(update=update)
    return preserved, [
        ValidationFinding(
            validator="replanner",
            severity="warning",
            message=(
                "Replanner attempted to change completed execution authority; "
                "restored the host-recorded step snapshots and plan-level "
                "scientific scope so registered evidence remains bound to "
                "immutable scientific requests."
            ),
            detail={
                "restored_changed_step_ids": sorted(set(changed_ids)),
                "reinserted_step_ids": reinserted_ids,
                "restored_plan_scope": restored_plan_scope,
                "restored_plan_scope_fields": restored_plan_scope_fields,
                "reason": "completed_step_snapshot_immutable",
            },
        )
    ]


def _project_locked_robustness_specs_after_replan(
    *,
    revised_plan: AnalysisPlan,
    locked_specs: Sequence[RobustnessSpec],
) -> tuple[AnalysisPlan, Optional[ValidationFinding]]:
    """Project an already verified plan-time robustness lock onto a candidate."""

    revised_specs = list(revised_plan.robustness_specs or [])
    if robustness_specs_sha(revised_specs) == robustness_specs_sha(locked_specs):
        return revised_plan, None
    preserved = revised_plan.model_copy(update={"robustness_specs": list(locked_specs)})
    return preserved, ValidationFinding(
        validator="replanner",
        severity="warning",
        message=(
            "Replanner attempted to change the immutable plan-time robustness "
            "specifications; preserved the verified lock and retained only the "
            "other plan revisions."
        ),
        detail={
            "reason": "preserve_locked_robustness_specs",
            "locked_spec_ids": [spec.spec_id for spec in locked_specs],
        },
    )


def _preserve_locked_robustness_specs_after_replan(
    *,
    current_plan: AnalysisPlan,
    revised_plan: AnalysisPlan,
    run_dir: Path,
) -> tuple[AnalysisPlan, Optional[ValidationFinding]]:
    """Compatibility entrypoint resolving the lock before pure projection."""

    locked_specs = robustness_specs_for_execution(
        run_dir=run_dir,
        plan=current_plan,
    )
    return _project_locked_robustness_specs_after_replan(
        revised_plan=revised_plan,
        locked_specs=locked_specs,
    )


def normalize_replan_candidate(
    *,
    current_plan: AnalysisPlan,
    candidate_plan: AnalysisPlan,
    completed_records: Sequence[Mapping[str, Any]],
    context: ResearchContext,
    max_total_steps: int,
    locked_robustness_specs: Sequence[RobustnessSpec],
) -> NormalizedPlanCandidate:
    """Apply host invariants to one provider-returned candidate, without I/O."""

    findings: List[ValidationFinding] = []
    revised, immutable_step_findings = _preserve_completed_step_snapshots_after_replan(
        current_plan=current_plan,
        revised_plan=candidate_plan,
        completed_records=completed_records,
    )
    findings.extend(immutable_step_findings)

    revised, estimand_findings = _preserve_primary_estimand_step_after_replan(
        current=current_plan,
        revised=revised,
    )
    findings.extend(estimand_findings)
    revised, figure_findings = _preserve_figure_steps_after_replan(
        current=current_plan,
        revised=revised,
    )
    findings.extend(figure_findings)
    revised, report_input_findings = _augment_report_typed_product_inputs(plan=revised)
    findings.extend(report_input_findings)

    if max_total_steps > 0:
        protected_step_ids = [
            str(record.get("step_id"))
            for record in current_successful_step_records(completed_records)
            if record.get("step_id") and record.get("status") == "ok"
        ]
        revised, cap_findings = _cap_plan_preserving_figure_steps(
            plan=revised,
            cap=max_total_steps,
            protected_step_ids=protected_step_ids,
        )
        findings.extend(
            finding.model_copy(
                update={
                    "validator": "replanner",
                    "message": (finding.message or "").replace(
                        "Initial plan had",
                        "Replanner produced",
                    ),
                }
            )
            for finding in cap_findings
        )

    revised, robustness_finding = _project_locked_robustness_specs_after_replan(
        revised_plan=revised,
        locked_specs=locked_robustness_specs,
    )
    if robustness_finding is not None:
        findings.append(robustness_finding)
    revised, trajectory_findings = augment_trajectory_plan_products(
        plan=revised,
        context=context,
    )
    findings.extend(trajectory_findings)
    revised, companion_findings = _augment_measurement_companion_inputs(
        plan=revised,
        context=context,
    )
    findings.extend(companion_findings)

    # Structural transforms may touch an already completed step. Re-apply the
    # immutable execution snapshots after every transform, not only before them.
    revised, post_transform_findings = _preserve_completed_step_snapshots_after_replan(
        current_plan=current_plan,
        revised_plan=revised,
        completed_records=completed_records,
    )
    findings.extend(post_transform_findings)
    return NormalizedPlanCandidate(
        plan=revised,
        findings=tuple(findings),
        substantive=_plan_signature(revised) != _plan_signature(current_plan),
    )
