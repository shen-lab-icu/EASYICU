"""The form a question names a substance in -- measured or given -- held against the plan.

Owner
-----
Some substances are both a level measured and a drug, fluid or blood product
given: serum albumin and albumin given intravenously, a platelet count and a
platelet transfusion.  The Web reader (``webserver.study_intent``) reads which
form the question names from its words and records each decided reading in
the run's data constraints (:data:`QUESTION_SUBSTANCE_FORMS_KEY`).  A plan
whose primary exposure reads the substance in the other form answers another
question, so planning stops (:data:`SUBSTITUTED_REASON`) rather than analyse
it.  A substance the question names without saying which form is recorded by
no one here: the plan's choice of form is shown for review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..schema import AnalysisPlan, ResearchContext

#: The data-constraints key the Web host records the readings under.
QUESTION_SUBSTANCE_FORMS_KEY = "question_substance_forms"
#: Why planning stops when the plan reads the other form.  Stable.
SUBSTITUTED_REASON = "question_substance_form_substituted"
#: Why a recorded reading is refused: the host wrote it, so it is the host's defect.
MALFORMED_REASON = "question_substance_forms_malformed"

_FORM_WORDS = {"given": "as given", "measured": "as a measured level"}


class QuestionSubstanceFormError(ValueError):
    """A typed refusal of this owner's inputs; ``reason_code`` names it."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code


class QuestionSubstanceForm(BaseModel):
    """A substance the question names in one form, and the columns of its other form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concepts: list[str] = Field(min_length=1, max_length=8)
    form: Literal["given", "measured"]
    other: list[str] = Field(min_length=1, max_length=8)
    evidence: str = Field(min_length=1, max_length=120)


def question_substance_forms(
    context: ResearchContext,
) -> tuple[QuestionSubstanceForm, ...]:
    """The forms the Web reader read in the question, from the run's data constraints.

    Absent on a run the Web did not launch, or whose question decides no
    substance's form.  A malformed record is refused, never read as none.
    """

    raw = (
        getattr(context.user_preferences, "data_constraints", None)
        if context.user_preferences
        else None
    )
    if not raw:
        return ()
    try:
        constraints = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError as exc:
        raise QuestionSubstanceFormError(
            MALFORMED_REASON, "the run's data constraints are not JSON"
        ) from exc
    if not isinstance(constraints, dict):
        return ()
    entries = constraints.get(QUESTION_SUBSTANCE_FORMS_KEY)
    if entries is None:
        return ()
    if not isinstance(entries, list):
        raise QuestionSubstanceFormError(
            MALFORMED_REASON, f"{QUESTION_SUBSTANCE_FORMS_KEY} is not a list"
        )
    try:
        return tuple(QuestionSubstanceForm.model_validate(entry) for entry in entries)
    except ValidationError as exc:
        raise QuestionSubstanceFormError(
            MALFORMED_REASON, "an entry does not state one substance's form"
        ) from exc


@dataclass(frozen=True)
class SubstanceFormSubstitution:
    """The plan's primary exposure reads a substance in the form the question did not name."""

    stated: QuestionSubstanceForm
    used: str

    def message(self) -> str:
        other = "measured" if self.stated.form == "given" else "given"
        return (
            f"The question names {self.stated.evidence!r} {_FORM_WORDS[self.stated.form]} "
            f"({', '.join(self.stated.concepts)}); the plan's primary exposure "
            f"{self.used!r} reads it {_FORM_WORDS[other]}."
        )


def substance_form_substitutions(
    forms: Sequence[QuestionSubstanceForm],
    *,
    plan: AnalysisPlan,
    relatives: Callable[[str], frozenset[str]],
) -> tuple[SubstanceFormSubstitution, ...]:
    """Each primary exposure of ``plan`` that reads a named substance in its other form.

    ``relatives`` gives the run's columns related to a concept
    (``question_requirements.concept_relatives``): a column of the other form
    that no concept of the named form relates to is a substitution.
    """

    sources = dict.fromkeys(
        source
        for step in plan.steps
        for source in step.required_primary_exposure_sources()
    )
    found: list[SubstanceFormSubstitution] = []
    for stated in forms:
        named = frozenset().union(*(relatives(concept) for concept in stated.concepts))
        other = frozenset().union(*(relatives(concept) for concept in stated.other))
        found.extend(
            SubstanceFormSubstitution(stated=stated, used=source)
            for source in sources
            if source in other and source not in named
        )
    return tuple(found)


__all__ = [
    "MALFORMED_REASON",
    "QUESTION_SUBSTANCE_FORMS_KEY",
    "SUBSTITUTED_REASON",
    "QuestionSubstanceForm",
    "QuestionSubstanceFormError",
    "SubstanceFormSubstitution",
    "question_substance_forms",
    "substance_form_substitutions",
]
