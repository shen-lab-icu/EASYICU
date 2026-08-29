"""Owner-compiled run phase for one planned analysis step.

Owner: research-agent planning authority.
Public contract: given a planned step, say which *phase of the run* it belongs
to. The contract is case-neutral -- it names no benchmark case, variable,
score, database, or figure -- and carries no claim authority of its own.

Why this exists
---------------
``planned_analysis_role`` (schema.py) answers a governance question: which step
carries the study's claim. It deliberately collapses cohort materialisation,
data-quality auditing, and figure rendering into a single ``auxiliary`` value,
because none of them carry a claim. A reader that wants to show the *shape of
the run* therefore has no field to read, and every consumer that wanted one has
so far re-derived it from the free-text ``method``/``step_id``. That habit is
how a canned sentence keyed on ``step_id`` came to assert one case's exposure
and outcome for every plan; ``AnalysisStep.scientific_capability`` already
warns about the same failure mode ("routing a capability by keyword is how a
feasibility audit and an interaction model become indistinguishable").

So the derivation is compiled once, here, by the layer that owns plan
semantics, and consumers read the compiled value instead of rebuilding it.

Why it is not a field on ``AnalysisStep``
-----------------------------------------
``plan_sha256`` is ``canonical_sha256(plan.model_dump(mode="json"))`` over the
whole plan, so adding any field -- even one defaulting to ``None`` -- changes
the digest of every plan already on disk and invalidates digest-verified
resume and human-review binding for runs in flight. A reading convenience must
not cost that. The phase is therefore a *projection* compiled on demand from an
immutable plan, never persisted into the evidence artifact.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping, Sequence


PlannedStepPhase = Literal[
    "cohort",
    "data_check",
    "analysis",
    "robustness",
    "reporting",
    "support",
]

PLANNED_STEP_PHASES: tuple[PlannedStepPhase, ...] = (
    "cohort",
    "data_check",
    "analysis",
    "robustness",
    "reporting",
    "support",
)

# Method-family markers. These read a step's declared METHOD, never its prose
# intent, and never a variable, outcome, score, or database name.
_RENDERING = re.compile(r"visuali|render|figure|plot|chart", re.IGNORECASE)
_COHORT = re.compile(r"cohort|attrition|eligib", re.IGNORECASE)
_DATA_CHECK = re.compile(
    r"table_one|baseline|missing|measurement|quality|audit|applicab|profile|"
    r"readiness|coverage",
    re.IGNORECASE,
)
_ROBUSTNESS = re.compile(r"sensitivit|robust", re.IGNORECASE)


def _field(step: Any, name: str) -> Any:
    if isinstance(step, Mapping):
        return step.get(name)
    return getattr(step, name, None)


def _text(step: Any, name: str) -> str:
    return str(_field(step, name) or "").strip()


def _outputs(step: Any) -> tuple[str, ...]:
    raw = _field(step, "expected_outputs")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(str(value or "").strip() for value in raw if str(value or "").strip())


def compile_step_phase(step: Any) -> PlannedStepPhase:
    """Compile the run phase of one planned step.

    Accepts a typed ``AnalysisStep`` or the equivalent mapping loaded from a
    persisted ``agent_plan.json``, so a consumer never has to reconstruct the
    full typed plan just to read the phase.

    The Planner-declared role is authoritative wherever it speaks: a step the
    plan calls ``primary``/``secondary`` is a result, and a step it calls
    ``sensitivity`` is a stress test, whatever its method string looks like. A
    step the plan calls ``auxiliary`` is never promoted into a result here --
    where no supporting shape matches, it stays ``support``. Only a step with
    no declared role at all (historical plans, host fixtures) is classified by
    method alone.
    """

    role = _text(step, "planned_analysis_role").lower()
    if role in {"primary", "secondary"}:
        return "analysis"
    if role == "sensitivity":
        return "robustness"

    method = _text(step, "method")
    step_id = _text(step, "step_id")
    marker = f"{method} {step_id}"
    outputs = _outputs(step)

    if _RENDERING.search(marker):
        return "reporting"
    if outputs and all(value.lower().startswith("figure:") for value in outputs):
        return "reporting"
    if _COHORT.search(marker):
        return "cohort"
    if _DATA_CHECK.search(marker):
        return "data_check"
    if _ROBUSTNESS.search(marker):
        return "robustness"
    if role == "auxiliary":
        # The plan declared this step carries no result claim. Absent a
        # supporting shape we say exactly that, rather than promoting it.
        return "support"
    if not role:
        # No declared role at all: the method heuristic is the only signal, and
        # an unrecognised analysis method is still an analysis.
        return "analysis"
    return "support"


def compile_plan_step_phases(steps: Any) -> tuple[PlannedStepPhase, ...]:
    """Compile the phase of every step of a plan, in plan order."""

    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return ()
    return tuple(compile_step_phase(step) for step in steps)


def design_analysis_family(analysis_type: Any) -> str | None:
    """Resolve a design candidate's analysis family, or ``None`` if unknown.

    The family is already owned by ``study_design.py``; this wrapper only makes
    it safe for a projection to call on an arbitrary persisted value, so a
    consumer never re-implements the mapping or guesses from method prose.
    """

    value = str(analysis_type or "").strip()
    if not value:
        return None
    from .study_design import study_design_family_for_analysis_type

    try:
        return str(study_design_family_for_analysis_type(value))
    except ValueError:
        return None


__all__ = [
    "PLANNED_STEP_PHASES",
    "PlannedStepPhase",
    "compile_plan_step_phases",
    "compile_step_phase",
    "design_analysis_family",
]
