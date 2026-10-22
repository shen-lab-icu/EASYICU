"""The design a study's question states for its analysis family, recorded by the host.

Owner of one decision beside ``causal_trial_design``: a question the host
reads as an association study or a description (``study_family_reading``), in
a study that records no analysis design, gets that family's design from the
host, with the words it rests on, so the run plans the family the question
asks for and never one the planner's keyword routing guesses.  One rule serves
two callers:

* the study setup writer (``pi_copilot.study_context_update``), on any turn
  the study has a question and no design;
* the research launch (:func:`record_unset_family_design`), for a study saved
  before this reading existed.  Unlike a causal design, a description or an
  association needs no statement before it is planned, so the launch records
  the design and goes on.

A design the study already states, or the turn proposes, is kept.  When the
question's words read another family, the reading is recorded beside it with
that conflict (:func:`family_reading_conflict`), so the study card shows the
difference; nothing stops.  A causal design is ``causal_trial_design``'s.

The design follows the bound source's patient grouping, as the causal one
does: a source that groups stays by patient gets patient-clustered variance,
one that cannot gets model-based variance, and the study's dependence rule
(``study_contexts.analysis_dependence_finding``) still asks for a patient
grouping or first stays when repeated stays are kept.

The same rule sets a descriptive design's variance whenever counts only are
on the table (:func:`descriptive_variance`): a descriptive study reports
counts without any interval only when the researcher's words ask for it
(``design_variance_basis.counts_only_request``).  Counts only the turn
proposes on no such request are replaced by the source's variance before
the study is saved; counts only the study already records stay, with the
conflict recorded for the study card.  Nothing stops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from easyicu.webserver.design_variance_basis import (
    COUNTS_ONLY,
    COUNTS_ONLY_NOT_REQUESTED,
    DESIGN_VARIANCE_BASIS_FIELD,
    USER_WORDS,
    counts_only_request,
)
from easyicu.webserver.study_family_reading import (
    StudyFamilyReading,
    planner_stated_family,
    study_family_reading,
)

__all__ = [
    "STUDY_FAMILY_READING_FIELD",
    "FamilyDesignDecision",
    "VarianceDecision",
    "descriptive_variance",
    "family_design_for",
    "family_reading_conflict",
    "checked_family_design",
    "question_family_design",
    "record_unset_family_design",
]

#: The study field and receipt field that carry the reading.
STUDY_FAMILY_READING_FIELD = "study_family_reading"
_CAUSAL_FAMILY = "causal_inference"
_DESCRIPTIVE_FAMILY = "descriptive_epidemiology"


def _design_family(design: Any) -> str:
    from easyicu.research_agent.planning.analysis_types import canonical_analysis_family

    if not isinstance(design, Mapping):
        return ""
    family = str(design.get("analysis_family") or "").strip()
    return str(canonical_analysis_family(family) or family) if family else ""


def family_design_for(study: Mapping[str, Any], family: str) -> Dict[str, str]:
    """The design the host records for ``family`` in ``study``, by its source's grouping."""

    from easyicu.webserver.causal_trial_design import source_groups_patients

    if source_groups_patients(study):
        return {
            "analysis_family": family,
            "analysis_unit": "icu_stay",
            "variance_estimator": "cluster_robust",
            "cluster_unit": "patient",
        }
    return {
        "analysis_family": family,
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }


@dataclass(frozen=True)
class FamilyDesignDecision:
    """What the host records for a question whose words state its family."""

    reading: StudyFamilyReading
    design: Dict[str, str]


@dataclass(frozen=True)
class VarianceDecision:
    """A descriptive design's variance, and the record of why."""

    design: Dict[str, str]
    basis: Dict[str, Any]


def descriptive_variance(
    study: Mapping[str, Any],
    *,
    design: Any,
    saved: Any,
    message: Any,
) -> Optional[VarianceDecision]:
    """The variance a descriptive ``design`` keeps, when counts only are on the table.

    ``design`` is the design about to be saved, ``saved`` the one the study
    records, ``message`` the researcher's turn.  ``None`` when the design is
    not descriptive, reports intervals, or its counts only are already
    explained by the researcher's words.
    """

    if not isinstance(design, Mapping) or _design_family(design) != _DESCRIPTIVE_FAMILY:
        return None
    if design.get("variance_estimator") != COUNTS_ONLY:
        return None
    current = dict(design)
    words = counts_only_request(message)
    if words is not None:
        return VarianceDecision(
            design=current,
            basis={"variance_estimator": COUNTS_ONLY, "basis": USER_WORDS, "evidence": words},
        )
    recorded = study.get(DESIGN_VARIANCE_BASIS_FIELD)
    if isinstance(saved, Mapping) and saved.get("variance_estimator") == COUNTS_ONLY:
        # The study records counts only: they stay, and a record that does
        # not rest on the researcher's words says so.
        if isinstance(recorded, Mapping) and recorded.get("basis") == USER_WORDS:
            return None
        return VarianceDecision(
            design=current,
            basis={"variance_estimator": COUNTS_ONLY, "conflict": COUNTS_ONLY_NOT_REQUESTED},
        )
    chosen = family_design_for(study, _DESCRIPTIVE_FAMILY)
    replaced = {key: value for key, value in current.items() if key != "cluster_unit"}
    replaced["variance_estimator"] = chosen["variance_estimator"]
    if "cluster_unit" in chosen:
        replaced["cluster_unit"] = chosen["cluster_unit"]
    return VarianceDecision(
        design=replaced,
        basis={
            "variance_estimator": chosen["variance_estimator"],
            "basis": (
                "source_patient_grouping"
                if "cluster_unit" in chosen
                else "source_without_patient_grouping"
            ),
            "replaced": COUNTS_ONLY,
        },
    )


def question_family_design(
    question: Any, study: Mapping[str, Any]
) -> Optional[FamilyDesignDecision]:
    """The design ``question`` states for ``study``, or ``None``."""

    reading = study_family_reading(question)
    if reading is None:
        return None
    return FamilyDesignDecision(reading=reading, design=family_design_for(study, reading.family))


def family_reading_conflict(question: Any, design: Any) -> Optional[Dict[str, Any]]:
    """The record of a reading the stated ``design``'s family differs from, or ``None``.

    A causal design is not compared: ``causal_trial_design`` owns it.
    """

    family = _design_family(design)
    if not family or family == _CAUSAL_FAMILY:
        return None
    reading = study_family_reading(question)
    if reading is None or reading.family == family:
        return None
    return reading.record(design_family=family)


#: Families whose design the host records from the conversation's checked
#: reading alone: their design is the family, unit and variance.
_CHECKED_DESIGN_FAMILIES = frozenset(
    {"association_study", "descriptive_epidemiology", "prediction_model"}
)


@dataclass(frozen=True)
class CheckedFamily:
    """What the conversation's checked reading says of the study's family."""

    #: The researcher is asked to choose: nothing is recorded or planned.
    disagreement: bool
    #: The settled family, when the host's own words do not read it.
    family: Optional[str]
    #: Its design, when the host records one for that family.
    design: Optional[Dict[str, str]]


def checked_family_design(study: Mapping[str, Any], question: Any) -> CheckedFamily:
    """The study's family as the conversation's checked reading settles it.

    ``family`` is set only where it decides beside the host's words: the
    host's words read no family, or the researcher chose another one.
    """

    from easyicu.webserver.question_family_check import (
        QUESTION_FAMILY_CHECK_FIELD,
        open_disagreement,
        settled_question_family,
    )

    check = study.get(QUESTION_FAMILY_CHECK_FIELD)
    if open_disagreement(check, question) is not None:
        return CheckedFamily(disagreement=True, family=None, design=None)
    settled = settled_question_family(check, question)
    reading = study_family_reading(question)
    if settled is None or (reading is not None and reading.family == settled):
        return CheckedFamily(disagreement=False, family=None, design=None)
    design = family_design_for(study, settled) if settled in _CHECKED_DESIGN_FAMILIES else None
    return CheckedFamily(disagreement=False, family=settled, design=design)


#: Families whose launch preparation reads the study's recorded design: a
#: prediction study's population is the stays still in the ICU after its
#: prediction time (``primary_cohort.cohort_required_concepts``).
_PREPARED_FROM_DESIGN = frozenset({"prediction_model"})


def record_unset_family_design(
    study: Mapping[str, Any], *, question: str
) -> Mapping[str, Any]:
    """Record the design a launch's study states in its question, when it has none.

    Returns the study as the launch should read it: with the recorded design
    and reading, or unchanged when the study states a design, its question
    reads as no family, or the study changed since the launch read it.
    """

    from easyicu.webserver import study_contexts

    if _design_family(study.get("analysis_design")):
        return study
    checked = checked_family_design(study, question)
    if checked.disagreement:
        # The researcher chooses first; the launch's scope check says so.
        return study
    decision = None if checked.family else question_family_design(question, study)
    patch: Dict[str, Any]
    if checked.design is not None:
        patch = {"analysis_design": dict(checked.design)}
    elif checked.family:
        return study
    elif decision is not None:
        patch = {
            "analysis_design": dict(decision.design),
            STUDY_FAMILY_READING_FIELD: decision.reading.record(),
        }
    else:
        # A family whose launch preparation reads the recorded design is
        # recorded as the planner reads it in the question (pilot1 10-22: a
        # stated prediction model stopped for the ICU stay its population
        # reads, asking the researcher to declare what the question says).
        family = planner_stated_family(question)
        if family not in _PREPARED_FROM_DESIGN:
            return study
        patch = {"analysis_design": family_design_for(study, family)}
    try:
        saved = study_contexts.upsert_context(
            {"id": str(study.get("id") or ""), **patch},
            active=True,
            expected_revision=int(study.get("revision") or 0),
            require_revision=True,
            lifecycle_write=False,
            _server_study_family_reading_write=True,
        )
    except study_contexts.StudyContextError:
        # Another write won: the planner reads the study as that write left it.
        return study
    return {
        **dict(study),
        "analysis_design": saved.get("analysis_design"),
        STUDY_FAMILY_READING_FIELD: saved.get(STUDY_FAMILY_READING_FIELD),
        "revision": saved.get("revision", study.get("revision")),
    }
