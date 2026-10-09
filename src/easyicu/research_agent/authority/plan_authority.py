"""Normalize replanner candidates without owning provider or run mutation.

The Planner/Replanner retains every scientific choice. This owner only projects
host invariants; provider, persistence, materialization, and budgets stay outside.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ..planning.cohort_contract import (
    cohort_concept_id_scope,
    cohort_definition_concept_ids,
)
from ..planning.figure_step_contract import preserve_figure_steps_after_replan
from ..planning.plan_graph import cap_plan_preserving_figure_steps
from ..planning import figure_plan_shaping
from ..robustness.panel import (
    RobustnessSpec,
    robustness_specs_for_execution,
    robustness_specs_sha,
)
from ..authority.runtime_artifacts import current_successful_step_records
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding
from ..trajectory.plan_contract import augment_trajectory_plan_products
from .declared_levels import bind_step_declared_levels
from .executed_step_snapshots import (
    INVALID_AUTHORITY_PROJECTION_REASON,
    invalid_authority_projection_finding,
    preserve_completed_step_snapshots_after_replan as _preserve_completed_step_snapshots_after_replan,
)
from .table_one_binding import bind_table_one_execution_spec
from .plan_input_closure import close_measurement_companion_inputs
from .plan_scope import plan_signature

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


def _has_invalid_authority_projection(
    findings: Sequence[ValidationFinding],
) -> bool:
    return any(
        finding.detail.get("reason") == INVALID_AUTHORITY_PROJECTION_REASON
        for finding in findings
    )


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
    if _has_invalid_authority_projection(findings):
        return NormalizedPlanCandidate(
            plan=current_plan,
            findings=tuple(findings),
            substantive=False,
        )
    revised, figure_findings = preserve_figure_steps_after_replan(
        current=current_plan,
        revised=revised,
    )
    findings.extend(figure_findings)
    revised = figure_plan_shaping.apply_required_plan_obligations(revised, context, findings)
    revised, report_input_findings = figure_plan_shaping.augment_report_typed_product_inputs(plan=revised)
    findings.extend(report_input_findings)

    if max_total_steps > 0:
        protected_step_ids = [
            str(record.get("step_id"))
            for record in current_successful_step_records(completed_records)
            if record.get("step_id") and record.get("status") == "ok"
        ]
        revised, cap_findings = cap_plan_preserving_figure_steps(
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
    revised, companion_findings = close_measurement_companion_inputs(
        plan=revised,
        context=context,
    )
    findings.extend(companion_findings)
    revised, panel_findings = figure_plan_shaping.bind_deterministic_figure_panels(plan=revised)
    findings.extend(panel_findings)

    # Structural transforms may touch an already completed step. Re-apply the
    # immutable execution snapshots after every transform, not only before them.
    revised, post_transform_findings = _preserve_completed_step_snapshots_after_replan(
        current_plan=current_plan,
        revised_plan=revised,
        completed_records=completed_records,
    )
    findings.extend(post_transform_findings)
    if _has_invalid_authority_projection(findings):
        return NormalizedPlanCandidate(
            plan=current_plan,
            findings=tuple(findings),
            substantive=False,
        )
    try:
        with cohort_concept_id_scope(
            cohort_definition_concept_ids(revised.cohort)
        ):
            revised = AnalysisPlan.model_validate(revised.model_dump(mode="json"))
        for revised_step in revised.steps:
            bind_table_one_execution_spec(revised_step, context)
            bind_step_declared_levels(revised_step, context)
    except (TypeError, ValueError) as exc:
        findings.append(invalid_authority_projection_finding(exc))
        return NormalizedPlanCandidate(
            plan=current_plan,
            findings=tuple(findings),
            substantive=False,
        )
    return NormalizedPlanCandidate(
        plan=revised,
        findings=tuple(findings),
        substantive=plan_signature(revised) != plan_signature(current_plan),
    )
