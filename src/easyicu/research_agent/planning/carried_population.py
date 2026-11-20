"""Carry a reviewed candidate's unapplied population criteria into its execution plan.

A metadata-only candidate may state a population criterion the compiler cannot
apply (for example 「成人 ICU 患者」 written as a condition on age).  The
researcher reviews that omission on the candidate.  The execution run then plans
again on the prepared package, and nothing made it keep the criterion: the
executed plan, its report and the paper lost it (UX on rc_512, S1747).

The host carries every such criterion verbatim (S1750/S1751).  After the
execution plan is compiled, a carried criterion that no cohort predicate reads
and that the plan does not already list is appended to
``CohortDefinition.unapplied_population_criteria``, which the review card, the
report (``reporting.population_selection``) and the paper already read.  This
selects no row and approves nothing: the plan's own population review runs again
on the result.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .population_review import unapplied_inclusions
from .population_shadow import criterion_concepts
from .population_spec import PopulationCriterion

CARRIED_SOURCE = "carried_from_candidate"
_MAX_CRITERIA = 20
_CRITERION: TypeAdapter[PopulationCriterion] = TypeAdapter(PopulationCriterion)


class CarriedPopulationCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quote: str = Field(min_length=1, max_length=400)
    concepts: tuple[str, ...] = Field(default=(), max_length=8)


class CarriedPopulationLimitations(BaseModel):
    """The candidate's unapplied criteria a run must keep, host-issued."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["carried_from_candidate"] = CARRIED_SOURCE
    criteria: tuple[CarriedPopulationCriterion, ...] = Field(min_length=1, max_length=_MAX_CRITERIA)


def _words(text: Any) -> str:
    return " ".join(str(text or "").split())


def candidate_population_limitations(
    *, cohort_unapplied: Iterable[str], review_requests: Iterable[Any] = (),
) -> Optional[dict[str, Any]]:
    """The carried record for one candidate, or ``None`` when it omitted nothing.

    Every unapplied text of the candidate's cohort is carried.  Its concepts
    come from the candidate's sealed population review (``unapplied_inclusions``
    of a digest-bound request) when a criterion there quotes the same words;
    otherwise the criterion is carried with no concept, so only an identical
    listing in the execution plan accounts for it.
    """

    concepts: dict[str, tuple[str, ...]] = {}
    for request in review_requests:
        for row in unapplied_inclusions(request):
            criterion = row.get("criterion") if isinstance(row, Mapping) else None
            if not isinstance(criterion, Mapping):
                continue
            try:
                typed = _CRITERION.validate_python(dict(criterion))
            except ValueError:
                continue
            concepts.setdefault(_words(criterion.get("quote")), criterion_concepts(typed))
    criteria = []
    for text in dict.fromkeys(_words(value) for value in cohort_unapplied if _words(value)):
        criteria.append({"quote": text[:400], "concepts": list(concepts.get(text, ()))[:8]})
    if not criteria:
        return None
    return CarriedPopulationLimitations.model_validate(
        {"source": CARRIED_SOURCE, "criteria": criteria[:_MAX_CRITERIA]}
    ).model_dump(mode="json")


def bind_carried_population_limitations(plan: Any, carried: Optional[Mapping[str, Any]]) -> Any:
    """Append each carried criterion the plan neither applies nor lists.

    Applied means a cohort predicate reads one of the criterion's concepts (or
    a column derived from it).  Listed means the plan's unapplied criteria hold
    the same words.  A plan without a cohort cannot keep the criterion and is
    refused rather than run without it.
    """

    if not carried:
        return plan
    record = CarriedPopulationLimitations.model_validate(dict(carried))
    cohort = getattr(plan, "cohort", None)
    if cohort is None:
        raise ValueError(
            "The execution plan has no cohort, so it cannot keep the population "
            "criteria the reviewed candidate did not apply."
        )
    read = {
        str(predicate.concept_id)
        for predicate in (*cohort.inclusion, *cohort.exclusion)
    }
    listed = [_words(text) for text in cohort.unapplied_population_criteria]
    appended = []
    for criterion in record.criteria:
        applied = any(
            column == concept or column.startswith(concept + "_")
            for column in read for concept in criterion.concepts
        )
        if applied or _words(criterion.quote) in listed or _words(criterion.quote) in appended:
            continue
        appended.append(_words(criterion.quote))
    if not appended:
        return plan
    return plan.model_copy(update={"cohort": replace(
        cohort, unapplied_population_criteria=(*cohort.unapplied_population_criteria, *appended),
    )})


__all__ = [
    "CARRIED_SOURCE",
    "CarriedPopulationCriterion",
    "CarriedPopulationLimitations",
    "bind_carried_population_limitations",
    "candidate_population_limitations",
]
