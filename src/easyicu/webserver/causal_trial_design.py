"""The causal design a study's question states, recorded by the host.

Owner of one decision: a question the host reads as a target trial's causal
question (``causal_trial_reading``) in a study that records no analysis
design gets the causal design from the host, with the words it rests on, so
the Copilot's next step is the trial statement, never an association plan.
One rule serves two callers:

* the study setup writer (``pi_copilot.study_context_update``), on any turn
  the study has a question and no design;
* the research launch (:func:`stop_unset_causal_design`), for a study saved
  before this reading existed, or whose question only the planner's own
  keyword routing reads as causal.  The launch never plans such a study: it
  records the design when the reading allows, and stops with the trial entry.

The design follows the bound source's patient grouping.  A source that groups
stays by patient gets a bootstrap that resamples patients; one that cannot
gets a bootstrap of stays, and the trial compile decides whether its stays may
be resampled alone (no patient with repeated stays) or stops for patient
identity (``tte_patient_identity_unavailable``).

A trial v1 cannot emulate is a capability gap, stated on the first turn: the
host records no design and the launch refuses the same question with the same
code.  Two gaps: a study bound to a database v1 emulates no trial in (the
trial setup's own refusal, ``target_trial_setup``), and a treatment v1 does
not register, stated with the treatment classes it does
(``planning.treatment_capture``).  The registry decides: a treatment concept is registered
when the registry records it, derives a registered concept from it, names it
as a drug, or the concept catalog files it with a registered concept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from easyicu.webserver.causal_trial_reading import CausalTrialReading, causal_trial_reading

__all__ = [
    "CAUSAL_TRIAL_GAP_FIELD",
    "CAUSAL_TRIAL_READING_FIELD",
    "STUDY_CAUSAL_TRIAL_DESIGN_UNSET",
    "TARGET_TRIAL_STATEMENT_NEEDED",
    "TARGET_TRIAL_TREATMENT_NOT_REGISTERED",
    "CausalDesignDecision",
    "causal_design_for",
    "question_causal_design",
    "question_treatment_concepts",
    "registered_treatment_concepts",
    "stop_unset_causal_design",
    "supported_treatment_classes",
]

#: The study field, receipt field and stop detail that carry the reading.
CAUSAL_TRIAL_READING_FIELD = "causal_trial_reading"
#: The receipt field and stop detail that carry a treatment the v1 trial cannot read.
CAUSAL_TRIAL_GAP_FIELD = "causal_trial_gap"
STUDY_CAUSAL_TRIAL_DESIGN_UNSET = "study_causal_trial_design_unset"
TARGET_TRIAL_TREATMENT_NOT_REGISTERED = "target_trial_treatment_not_registered"
#: The conversation's next step for a causal study (``target_trial_card``).
TARGET_TRIAL_STATEMENT_NEEDED = "target_trial_statement_needed"

#: How the receipt names the registry's classes; a class not listed here is
#: named by its id.
_CLASS_LABELS: Mapping[str, Tuple[str, str]] = {
    "inotrope": ("inotropes", "正性肌力药"),
    "vasoactive": ("vasoactive drugs", "血管活性药"),
    "vasopressor": ("vasopressors", "升压药"),
}
_ROMAN = frozenset({"ii", "iii", "iv"})


def _registry() -> Any:
    from easyicu.research_agent.planning.treatment_capture import (
        packaged_treatment_capture_registry,
    )

    return packaged_treatment_capture_registry().registry


def _agent_label(agent: str) -> str:
    return " ".join(word.upper() if word in _ROMAN else word for word in agent.split("_"))


def supported_treatment_classes() -> List[Dict[str, Any]]:
    """The treatment classes v1 registers, each with its drugs, for the receipt."""

    classes = []
    for name, item in sorted(_registry().treatment_classes.items()):
        label_en, label_zh = _CLASS_LABELS.get(name, (name, name))
        classes.append(
            {
                "id": name,
                "label_en": label_en,
                "label_zh": label_zh,
                "agents": [{"id": agent, "label_en": _agent_label(agent)} for agent in item.agents],
            }
        )
    return classes


def registered_treatment_concepts() -> frozenset[str]:
    """Every concept a v1 trial's treatment may be read from, by the registry."""

    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL

    registry = _registry()
    entries = {entry.concept for entry in registry.entries}
    components = {
        item.component for entry in registry.entries for item in entry.definition.components
    }
    agents = {agent for item in registry.treatment_classes.values() for agent in item.agents}
    grouped = {
        str(concept)
        for members in CONCEPT_GROUPS_INTERNAL.values()
        if entries & {str(member) for member in members}
        for concept in members
    }
    return frozenset(entries | components | agents | grouped)


def question_treatment_concepts(question: str, reading: CausalTrialReading) -> Tuple[str, ...]:
    """The concepts the initiate strategy's treatment names, read in the question.

    Each catalog name inside the treatment's words is read with the question
    around it (``study_intent.substance_reading``): "开始静脉输注白蛋白" is the
    albumin given, not the albumin measured.  Empty when the words name no
    catalog concept.
    """

    from easyicu.webserver.study_intent import named_study_concepts, substance_reading

    if reading.treatment is None:
        return ()
    start, end = reading.treatment
    words = question[start:end]
    found: List[str] = []
    for concepts, phrase in named_study_concepts(words):
        at = words.lower().find(str(phrase).lower())
        if at < 0:
            continue
        for concept in substance_reading(
            question, concept_id=concepts[0], start=start + at, end=start + at + len(phrase)
        ):
            if concept not in found:
                found.append(concept)
    return tuple(found)


def _concept_labels(concepts: Tuple[str, ...]) -> List[Dict[str, str]]:
    from easyicu.concept.catalog import CONCEPT_DICTIONARY

    labels = []
    for concept in concepts:
        entry = CONCEPT_DICTIONARY.get(concept)
        label_en, label_zh = (entry[0], entry[1]) if entry else (concept, concept)
        labels.append({"id": concept, "label_en": str(label_en), "label_zh": str(label_zh)})
    return labels


def _treatment_gap(question: str, reading: CausalTrialReading) -> Optional[Dict[str, Any]]:
    concepts = question_treatment_concepts(question, reading)
    if not concepts or set(concepts) & registered_treatment_concepts():
        return None
    return {
        "code": TARGET_TRIAL_TREATMENT_NOT_REGISTERED,
        "treatment_concepts": _concept_labels(concepts),
        "supported_classes": supported_treatment_classes(),
    }


def _database_gap(study: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    from easyicu.webserver.target_trial_setup import target_trial_database_gap

    return target_trial_database_gap(study)


def _gap_reason(gap: Mapping[str, Any]) -> str:
    if gap["code"] == TARGET_TRIAL_TREATMENT_NOT_REGISTERED:
        return "of a treatment the v1 trial emulation does not register"
    return f"in {gap['database']}, where v1 emulates no trial"


def _source_groups_patients(study: Mapping[str, Any]) -> bool:
    from easyicu.webserver.research_launch_scientific import verified_patient_grouping
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    try:
        return verified_patient_grouping(study) is not None
    except ResearchPipelineRunError:
        # A grouping authority that fails its checks groups no one; the trial
        # compile then stops for patient identity if repeated stays are possible.
        return False


def causal_design_for(study: Mapping[str, Any]) -> Dict[str, str]:
    """The causal design the host records for ``study``, by its source's grouping."""

    design = {
        "analysis_family": "causal_inference",
        "analysis_unit": "icu_stay",
        "variance_estimator": "bootstrap",
    }
    if _source_groups_patients(study):
        design["cluster_unit"] = "patient"
    return design


@dataclass(frozen=True)
class CausalDesignDecision:
    """What the host records for a question it reads as causal."""

    reading: CausalTrialReading
    #: ``None`` when v1 cannot emulate the trial (``gap``).
    design: Optional[Dict[str, str]]
    gap: Optional[Dict[str, Any]]


def question_causal_design(
    question: Any, study: Mapping[str, Any]
) -> Optional[CausalDesignDecision]:
    """The causal design ``question`` states for ``study``, or ``None``."""

    text = str(question or "").strip()
    reading = causal_trial_reading(text)
    if reading is None:
        return None
    gap = _database_gap(study) or _treatment_gap(text, reading)
    if gap is not None:
        return CausalDesignDecision(reading=reading, design=None, gap=gap)
    return CausalDesignDecision(reading=reading, design=causal_design_for(study), gap=None)


def _design_unset(study: Mapping[str, Any]) -> bool:
    design = study.get("analysis_design")
    return not (isinstance(design, Mapping) and str(design.get("analysis_family") or "").strip())


def _planner_reads_causal(question: str, database: str) -> bool:
    """Whether the planner's own family routing reads ``question`` as causal."""

    from easyicu.research_agent.planning.analysis_types import infer_analysis_type
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    context = ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="study", database=database, n_stays=0),
        variables=[],
    )
    return infer_analysis_type(context).key == "causal_inference"


def _record(study: Mapping[str, Any], decision: CausalDesignDecision) -> bool:
    """Persist the host's design and its reading in the study, at its revision."""

    from easyicu.webserver import study_contexts

    try:
        study_contexts.upsert_context(
            {
                "id": str(study.get("id") or ""),
                "analysis_design": dict(decision.design or {}),
                CAUSAL_TRIAL_READING_FIELD: decision.reading.record(),
            },
            active=True,
            expected_revision=int(study.get("revision") or 0),
            require_revision=True,
            lifecycle_write=False,
            _server_causal_trial_reading_write=True,
        )
    except study_contexts.StudyContextError:
        # The study changed since the launch read it: the stop below still
        # returns the conversation to the trial, which reads the study afresh.
        return False
    return True


def stop_unset_causal_design(
    study: Mapping[str, Any], *, question: str, database: str
) -> None:
    """Stop a launch whose study records no design for a question read as causal.

    The study is never planned from the question's keywords: a reading records
    the host's design first (``study_contexts``), and either way the launch
    stops with the trial entry.  Returns when the study records a design, or
    its question reads as neither causal nor trial-shaped.
    """

    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    if not _design_unset(study):
        return
    decision = question_causal_design(question, study)
    if decision is not None and decision.gap is not None:
        raise ResearchPipelineRunError(
            decision.gap["code"],
            f"This question asks for a target trial {_gap_reason(decision.gap)}, "
            "so planning stopped before any model was called and no analysis was run.",
            details={
                CAUSAL_TRIAL_READING_FIELD: decision.reading.record(),
                CAUSAL_TRIAL_GAP_FIELD: decision.gap,
            },
        )
    if decision is not None:
        details: Dict[str, Any] = {
            CAUSAL_TRIAL_READING_FIELD: decision.reading.record(),
            "design_recorded": _record(study, decision),
        }
    elif _planner_reads_causal(question, database):
        details = {}
    else:
        return
    raise ResearchPipelineRunError(
        STUDY_CAUSAL_TRIAL_DESIGN_UNSET,
        "This question is planned as a target trial: state the trial in the "
        "conversation first. Planning stopped before any model was called and "
        "no analysis was run.",
        details={**details, "next_action": TARGET_TRIAL_STATEMENT_NEEDED},
    )
