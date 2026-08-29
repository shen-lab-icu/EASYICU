"""Reader projection for a governed ``agent_plan.json`` preview.

Owner: Copilot artifact projection.
Public contract: stamp the research-agent's own compiled plan semantics onto
the bounded browser payload, so the reader never re-derives them.

The plan reader needs to know which phase of the run each step belongs to and
which analysis family a design candidate is. Both are owned by
``research_agent.planning`` -- the step phase by ``step_phase.compile_step_phase``
and the family by ``study_design``. Before this projection existed the browser
renderer reconstructed both from the free-text ``method``/``step_id``, and where
reconstruction failed the display fell back to canned prose that asserted one
case's exposure, outcome, and time zero for every plan.

The compiled values are added here rather than to ``AnalysisStep`` because
``plan_sha256`` covers ``plan.model_dump()``: a new schema field, even one
defaulting to ``None``, would invalidate the stored digest of every run already
on disk and break digest-verified resume. The persisted evidence artifact and
its digest are therefore untouched; only the browser projection carries the
compiled reading aids, and they are marked as projections.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


PLAN_ARTIFACT_NAME = "agent_plan.json"


def _projected_steps(steps: Any) -> List[Any] | None:
    from easyicu.research_agent.planning.step_phase import compile_step_phase

    if not isinstance(steps, list):
        return None
    projected: List[Any] = []
    changed = False
    for step in steps:
        if not isinstance(step, Mapping):
            projected.append(step)
            continue
        row = dict(step)
        row["planned_phase"] = compile_step_phase(row)
        projected.append(row)
        changed = True
    return projected if changed else None


def _projected_candidates(candidates: Any) -> List[Any] | None:
    from easyicu.research_agent.planning.step_phase import design_analysis_family

    if not isinstance(candidates, list):
        return None
    projected: List[Any] = []
    changed = False
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            projected.append(candidate)
            continue
        row = dict(candidate)
        family = design_analysis_family(row.get("analysis_type"))
        if family:
            row["analysis_family"] = family
            changed = True
        projected.append(row)
    return projected if changed else None


def project_plan_reader_fields(artifact_name: str, payload: Any) -> Any:
    """Return ``payload`` with the owner-compiled reading aids stamped on.

    A no-op for every artifact other than the plan, and for any payload shape
    that does not carry steps or design candidates. Never mutates the input and
    never raises: a preview must degrade to the un-annotated plan rather than
    fail, because the reader already handles a plan that states no phase.
    """

    if str(artifact_name or "").strip().lower() != PLAN_ARTIFACT_NAME:
        return payload
    if not isinstance(payload, Mapping):
        return payload

    try:
        projected: Dict[str, Any] = dict(payload)
        steps = _projected_steps(projected.get("steps"))
        if steps is not None:
            projected["steps"] = steps
        selection = projected.get("design_selection")
        if isinstance(selection, Mapping):
            candidates = _projected_candidates(selection.get("candidates"))
            if candidates is not None:
                selection_row = dict(selection)
                selection_row["candidates"] = candidates
                projected["design_selection"] = selection_row
        if projected == dict(payload):
            return payload
        projected["reader_projection"] = {
            "owner": "easyicu.research_agent.planning.step_phase",
            "fields": ["planned_phase", "analysis_family"],
            "persisted": False,
        }
        return projected
    except Exception:  # noqa: BLE001 - a reading aid must never block a preview
        return payload


__all__ = [
    "PLAN_ARTIFACT_NAME",
    "project_plan_reader_fields",
]
