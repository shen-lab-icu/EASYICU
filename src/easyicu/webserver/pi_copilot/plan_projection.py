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

import re
from typing import Any, Dict, List, Mapping, Optional


PLAN_ARTIFACT_NAME = "agent_plan.json"

# A plan may legitimately require the database's identity columns to define
# clustering or repeated-admission handling.  Those exact column names are not
# useful to the reader and the model-visible projection correctly rejects them
# as row-identifier vocabulary.  Project the closed set to semantic roles at
# this owner boundary; never relax the global PHI guard.
_IDENTITY_VARIABLE_LABELS = {
    "stay_id": "ICU stay grouping key",
    "icustay_id": "ICU stay grouping key",
    "hadm_id": "hospital admission grouping key",
    "subject_id": "patient clustering key",
    "patient_id": "patient clustering key",
    "entity_id": "entity grouping key",
}
_IDENTITY_VARIABLE_PATTERN = re.compile(
    r"\b(?:" + "|".join(map(re.escape, _IDENTITY_VARIABLE_LABELS)) + r")\b",
    re.I,
)


def _project_plan_text(value: Any, *, limit: int) -> str:
    """Keep plan prose useful without exposing exact row-identity columns."""

    text = " ".join(str(value or "").split())

    def replace(match: re.Match[str]) -> str:
        return _IDENTITY_VARIABLE_LABELS[match.group(0).lower()]

    return _IDENTITY_VARIABLE_PATTERN.sub(replace, text)[:limit]


def _project_required_variable(value: Any) -> str:
    text = str(value or "").strip()
    return _IDENTITY_VARIABLE_LABELS.get(text.lower(), text)[:120]


def project_plan_conversation_preview(payload: Any) -> Optional[Dict[str, Any]]:
    """Project the selected design's six review items into the conversation.

    The research-agent design contract owns both the item order and wording.
    Copilot only supplies bounded, path-free display coordinates; it never
    invents a cohort, exposure, endpoint, method, or preprocessing step.
    """

    if not isinstance(payload, Mapping):
        return None
    selection = payload.get("design_selection")
    if not isinstance(selection, Mapping):
        return None
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        return None
    selected = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and str(candidate.get("disposition") or "") == "selected"
        ),
        None,
    )
    if not isinstance(selected, Mapping):
        return None
    reviewable = selected.get("reviewable_plan")
    if not isinstance(reviewable, list):
        return None

    from easyicu.research_agent.planning.design_selection import (
        REVIEWABLE_PLAN_ITEM_ORDER,
    )

    if len(reviewable) != len(REVIEWABLE_PLAN_ITEM_ORDER):
        return None
    texts = [_project_plan_text(value, limit=800) for value in reviewable]
    if any(not value for value in texts):
        return None

    steps = payload.get("steps")
    steps = steps if isinstance(steps, list) else []
    from easyicu.research_agent.planning.step_phase import compile_step_phase

    phases = [compile_step_phase(step) for step in steps if isinstance(step, Mapping)]
    analysis_step_count = sum(
        phase in {"cohort", "analysis", "robustness"} for phase in phases
    )
    output_step_count = sum(phase == "reporting" for phase in phases)
    outputs = {
        str(output or "")
        for step in steps
        if isinstance(step, Mapping)
        for output in (
            step.get("expected_outputs")
            if isinstance(step.get("expected_outputs"), list)
            else []
        )
        if str(output or "").strip()
    }
    design = {
        key: _project_plan_text(selected.get(key), limit=1_200)
        for key in (
            "estimand",
            "time_zero",
            "observation_window",
            "primary_method",
        )
        if str(selected.get(key) or "").strip()
    }
    required_variables = [
        _project_required_variable(value)
        for value in (
            selected.get("required_variables")
            if isinstance(selected.get("required_variables"), list)
            else []
        )
        if str(value or "").strip()
    ][:24]
    if required_variables:
        design["required_variables"] = required_variables
    return {
        "research_question": _project_plan_text(
            payload.get("research_question"), limit=500
        ),
        "analysis_type": str(payload.get("analysis_type") or "")[:120],
        "items": [
            {"key": key, "text": text}
            for key, text in zip(REVIEWABLE_PLAN_ITEM_ORDER, texts, strict=True)
        ],
        "step_count": len(steps),
        "analysis_step_count": analysis_step_count,
        "output_step_count": output_step_count,
        "table_count": sum(value.startswith("table:") for value in outputs),
        "figure_count": sum(value.startswith("figure:") for value in outputs),
        "design": design,
    }


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
    "project_plan_conversation_preview",
    "project_plan_reader_fields",
]
