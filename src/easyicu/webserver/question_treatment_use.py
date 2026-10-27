"""A treatment's use the question asks that no concept records apart from its dose.

Owner of one decision: whether a question asks whether (or when) a treatment
was used -- 「使用去甲肾上腺素的患者占多少」, "patients who received
norepinephrine" -- where the data record only its dose or rate, and if so the
launch stops (:data:`TREATMENT_USE_UNRECORDED`) before any model is called.

Why
---
The question reader (``study_intent.stated_treatment_uses``) reads a
treatment named with use words as its use status, a concept its owner
declares an event status (given or not in a window): vasopressor use is
``vaso_ind``, not the norepinephrine-equivalent dose.  Some treatments the
reader names have no such status: norepinephrine is recorded as an infusion
rate (``norepi_rate``) only, whose absent record means "unmeasured", not "not
given".  A plan on the rate would answer a question about the dose, and
nothing would say so.  Until an owner records that treatment's use, the
honest answer is to stop and ask for a question the data answer: the
treatment's dose, or a class of treatment whose use is recorded.

A question the target-trial reader reads as a trial is not stopped here: the
trial owner reads its treatment from the treatment capture registry, which
states per database what an absent record means.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from easyicu.webserver.study_intent import StatedTreatmentUse, stated_treatment_uses

__all__ = [
    "TREATMENT_USE_UNRECORDED",
    "stop_unrecorded_treatment_uses",
    "unrecorded_treatment_uses",
]

#: Why a launch stops: the question asks a treatment's use the data do not record.
TREATMENT_USE_UNRECORDED = "question_treatment_use_unrecorded"


def unrecorded_treatment_uses(question: str) -> Tuple[StatedTreatmentUse, ...]:
    """The treatments ``question`` asks the use of that no concept records the use of."""

    return tuple(
        reading for reading in stated_treatment_uses(question)
        if reading.form == "use" and reading.status is None
    )


def _message(found: Tuple[StatedTreatmentUse, ...]) -> str:
    named = "; ".join(
        f"{item.evidence!r} (recorded only as {', '.join(item.doses)})" for item in found
    )
    return (
        f"The question asks whether a treatment was used: {named}. The data record "
        "how much of it was given, not whether it was given, so a plan would answer "
        "a question about its dose. Planning stopped before any model was called, and "
        "no analysis was run. Ask about its dose, or about a treatment class whose use "
        "is recorded (for example \"vasopressors\")."
    )


def stop_unrecorded_treatment_uses(
    question: str, *, study: Optional[Mapping[str, Any]] = None
) -> None:
    """Stop a launch whose question asks a treatment's use the data do not record."""

    from easyicu.research_agent.planning.analysis_types import canonical_analysis_family
    from easyicu.webserver.causal_trial_reading import causal_trial_reading
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    found = unrecorded_treatment_uses(question)
    if not found or causal_trial_reading(str(question or "").strip()) is not None:
        return
    design = (study or {}).get("analysis_design")
    if isinstance(design, Mapping) and canonical_analysis_family(
        str(design.get("analysis_family") or "")
    ) == "causal_inference":
        return
    details: Dict[str, Any] = {
        "concept": found[0].concept_id,
        "treatment_uses": [item.record() for item in found],
    }
    raise ResearchPipelineRunError(TREATMENT_USE_UNRECORDED, _message(found), details=details)
