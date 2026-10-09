"""A replan cannot rewrite a Planner step that already ran.

Owner
-----
A replanner may change future work, but a completed step's recorded request
(``analysis_request.step``) and the plan-level scientific scope it ran under
are execution authority.  ``preserve_completed_step_snapshots_after_replan``
restores them onto a candidate before the candidate is accepted.  The
rejection it returns when restoring them yields an invalid plan is the one
``authority.plan_authority`` also returns for its own projections.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from ..planning.cohort_contract import (
    cohort_concept_id_scope,
    cohort_definition_concept_ids,
)
from ..authority.runtime_artifacts import current_successful_step_records
from ..schema import AnalysisPlan, AnalysisStep, ValidationFinding
from .plan_scope import plan_scientific_scope_signature
from .planned_role import verified_planned_analysis_role


INVALID_AUTHORITY_PROJECTION_REASON = (
    "replanner_candidate_invalid_after_authority_projection"
)


def invalid_authority_projection_finding(exc: Exception) -> ValidationFinding:
    return ValidationFinding(
        validator="replanner",
        severity="warning",
        message=(
            "Rejected a replanner candidate because restoring immutable host "
            "authority produced an invalid analysis plan; kept the current plan "
            "without assigning or demoting a scientific role."
        ),
        detail={
            "reason": INVALID_AUTHORITY_PROJECTION_REASON,
            "error_type": type(exc).__name__,
        },
    )


def preserve_completed_step_snapshots_after_replan(
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
        if verified_planned_analysis_role(record) is None:
            return current_plan, [
                invalid_authority_projection_finding(
                    ValueError("completed step has inconsistent Planner role authority")
                )
            ]
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

    current_scope = plan_scientific_scope_signature(current_plan)
    revised_scope = plan_scientific_scope_signature(revised_plan)
    # A replanner owns revisions to the remaining step DAG, not a new research
    # question, cohort, analysis family, robustness lock, display semantics, or
    # rationale.  Preserve that Planner-authored scope even before the first
    # ordinary step completes (for example, immediately after the host probe).
    # Otherwise a partial JSON response with ``cohort: null`` can be registered
    # as the newest plan and make a later resume collide with the immutable
    # cohort lock.
    restored_plan_scope = revised_scope != current_scope
    restored_plan_scope_fields: List[str] = []
    if restored_plan_scope:
        for field_name in (
            "research_question",
            "analysis_type",
            "cohort",
            "robustness_specs",
            "display_labels",
            "know_how_decisions",
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
                "display_labels": current_plan.display_labels,
                "know_how_decisions": current_plan.know_how_decisions,
                "rationale": current_plan.rationale,
            }
        )
    payload = revised_plan.model_dump(mode="json")
    payload.update(update)
    # Both plans were validated when they were made.  Their own cohort ids
    # are the ones re-validating this merge needs: a column the run
    # materialized is known only inside the run's concept scope.
    cohort_ids = (
        *cohort_definition_concept_ids(current_plan.cohort),
        *cohort_definition_concept_ids(revised_plan.cohort),
    )
    try:
        with cohort_concept_id_scope(cohort_ids):
            preserved = AnalysisPlan.model_validate(payload)
    except (TypeError, ValueError) as exc:
        return current_plan, [invalid_authority_projection_finding(exc)]
    # Say which of the two things actually happened. The single sentence
    # claiming both used to be emitted whenever EITHER fired, so a run whose
    # step snapshots were left changed still reported that they had been
    # restored -- h1 and h2 each recorded exactly that, with
    # restored_changed_step_ids=[] one line below the claim, and each then lost
    # every downstream step to producer_plan_snapshot_mismatch on the step the
    # message said was protected. A guard that reports work it did not do sends
    # the next reader somewhere else entirely.
    restored_steps = sorted(set(changed_ids))
    parts = ["Replanner attempted to change completed execution authority;"]
    if restored_steps or reinserted_ids:
        parts.append(
            "restored the host-recorded step snapshots "
            f"({len(restored_steps)} changed, {len(reinserted_ids)} reinserted)"
        )
    else:
        parts.append("no completed step snapshot needed restoring")
    if restored_plan_scope:
        parts.append("and restored plan-level scientific scope")
    parts.append(
        "so registered evidence remains bound to immutable scientific requests."
    )
    return preserved, [
        ValidationFinding(
            validator="replanner",
            severity="warning",
            message=" ".join(parts),
            detail={
                "restored_changed_step_ids": restored_steps,
                "reinserted_step_ids": reinserted_ids,
                "restored_plan_scope": restored_plan_scope,
                "restored_plan_scope_fields": restored_plan_scope_fields,
                "reason": "completed_step_snapshot_immutable",
            },
        )
    ]


__all__ = [
    "INVALID_AUTHORITY_PROJECTION_REASON",
    "invalid_authority_projection_finding",
    "preserve_completed_step_snapshots_after_replan",
]
