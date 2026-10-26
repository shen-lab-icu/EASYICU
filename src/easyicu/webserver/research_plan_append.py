"""Structural admission for an append to a completed study's approved plan.

This boundary grants neither execution nor approval. The caller must resolve a
completed source and an executable, host-compiled addition before binding it to
PlanRevisionGuard. Existing repair revisions retain their original policy.
"""
from __future__ import annotations

from typing import Any, Mapping

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError


def _refuse(reason: str) -> None:
    raise ResearchPipelineRunError("plan_append_source_changed", "The append must preserve " + reason + ".")


def validate_append_structure(source: Mapping[str, Any], revised: Mapping[str, Any], *,
                              action: str, append_run: bool = False) -> None:
    """Retain the exact prefix, scientific coordinates and product identities.

    A new run can upgrade its own tier to article. That never edits the source
    run's tier. Plan-level specs remain exact: changing a previously bound
    spec would also change an old step's meaning.
    """
    if action not in {"sensitivity", "exposure", "subgroup", "article"}:
        _refuse("one supported append action")
    if append_run:
        # The study revision includes the old plan as evidence, while the new
        # run executes only its suffix. Never put completed steps in its roster.
        if {step["step_id"] for step in source["steps"]} & {step["step_id"] for step in revised["steps"]}:
            _refuse("distinct source and append step identities")
    before = AnalysisPlan.model_validate(source).model_dump(mode="json")
    after = AnalysisPlan.model_validate(revised).model_dump(mode="json")
    old_steps, new_steps = before.pop("steps"), after.pop("steps")
    if not old_steps or not new_steps or (not append_run and (
            len(new_steps) <= len(old_steps)
            or canonical_sha256(revised["steps"][:len(old_steps)]) != canonical_sha256(source["steps"]))):
        _refuse("every approved step, in order, with a nonempty addition")
    outputs = [output for step in new_steps for output in step.get("expected_outputs", ())]
    if len(outputs) != len(set(outputs)):
        _refuse("unique output identities")
    if after.pop("revision") != before.pop("revision") + 1:
        _refuse("the next plan revision")
    old_design, new_design = before.get("design_selection"), after.get("design_selection")
    if action == "article":
        if not old_design or not new_design or old_design.get("execution_tier") != "first_result":
            _refuse("a first_result source for an article extension")
        if new_design.get("execution_tier", "article") != "article":
            _refuse("the article tier on the new run")
        # These are fresh model_dump payloads, never caller-owned objects.
        old_design.pop("execution_tier", None)
        new_design.pop("execution_tier", None)
    if before != after:
        _refuse("the approved scientific scope and plan-level specifications")
    known = {output for step in old_steps for output in step.get("expected_outputs", ())}
    for step in new_steps if append_run else new_steps[len(old_steps):]:
        if not {value for value in step.get("inputs", ()) if ":" in value} <= known:
            _refuse("dependencies on the existing plan or earlier additions")
        known.update(step.get("expected_outputs", ()))
