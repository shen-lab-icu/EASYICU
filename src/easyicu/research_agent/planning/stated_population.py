"""The recorded conditions a question restricts its population to (S1757).

A question such as 「脓毒症 ICU 患者中，…」 restricts the study to patients
with a condition EasyICU records (sepsis, AKI, mechanical ventilation).  The
host reads that restriction from the question's own words
(``webserver.study_intent``) and passes it as a planning constraint: each
condition with its words and the concepts that record it.

The plan accounts for each such condition: a cohort predicate or a stated
population criterion reads one of its concepts (or a column derived from
one), or a stated criterion lists its words.  A Planner that accounts for none
of them is asked again with the reason.  When its attempts are spent, the host
states the condition itself: present on the first recorded concept the input
holds, quoting the question, over the window that concept was summarized in,
missing values excluded (a ``condition_present`` criterion of a typed
population, or a stated criterion and its predicate otherwise).  The criterion is marked as
host-stated on the cohort, so the plan card says who stated it; the
researcher's plan approval is the consent, and 修改人群 remains.

This module reads and states; the population compiler decides how each
criterion reaches the rows, and nothing here selects a row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..research_context.concept_population import context_data_constraints
from ..research_context.materialization_window import context_column_windows
from .cohort_eligibility import predicate_context_column

STATED_POPULATION_CONDITIONS_KEY = "stated_population_conditions"
DEFAULT_WINDOW_HOURS = (0.0, 24.0)
_MAX_CRITERIA = 99


@dataclass(frozen=True)
class StatedCondition:
    """One recorded condition the question restricts its patients to."""

    quote: str
    #: The concepts that record it which this input holds, in the host's order.
    concepts: tuple[str, ...]


def stated_conditions(context: Any, *, allowed: Iterable[str]) -> tuple[StatedCondition, ...]:
    """The host's stated conditions this input can express, quoted in the question."""

    raw = context_data_constraints(context).get(STATED_POPULATION_CONDITIONS_KEY)
    if not isinstance(raw, (list, tuple)):
        return ()
    question = str(getattr(context, "research_question", "") or "")
    allowed_set = {str(item) for item in allowed}
    found = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        quote = " ".join(str(item.get("quote") or "").split())
        concepts = tuple(
            dict.fromkeys(str(c) for c in item.get("concepts") or () if str(c) in allowed_set)
        )
        if len(quote) >= 2 and concepts and quote in question:
            found.append(StatedCondition(quote=quote, concepts=concepts))
    return tuple(found)


def _criterion_concepts(criterion: Any) -> tuple[str, ...]:
    concepts = list(getattr(criterion, "concepts_all_of", None) or ())
    single = getattr(criterion, "concept", None)
    if single:
        concepts.append(single)
    return tuple(str(item) for item in concepts)


def unaccounted_conditions(
    context: Any,
    conditions: Sequence[StatedCondition],
    *,
    named: Iterable[str],
    texts: Iterable[str],
) -> tuple[StatedCondition, ...]:
    """The conditions that no named concept or column reads and no text lists.

    ``named`` are the concepts or columns the cohort's predicates and stated
    criteria read; a column derived from a condition's concept (``x_max``, or
    any column whose source is the concept) reads it.  ``texts`` are the
    stated criteria's words: a criterion that quotes the condition lists it,
    applied or not.
    """

    variables = {str(v.name): v for v in getattr(context, "variables", ()) or ()}
    names = [str(item) for item in named]
    words = [str(item).casefold() for item in texts]

    def reads(column: str, concept: str) -> bool:
        source = getattr(variables.get(column), "source_concept", None)
        return column == concept or column.startswith(concept + "_") or source == concept

    missing = []
    for condition in conditions:
        if any(reads(name, concept) for name in names for concept in condition.concepts):
            continue
        if any(condition.quote.casefold() in text for text in words):
            continue
        missing.append(condition)
    return tuple(missing)


def cohort_reads(cohort: Any, spec: Any) -> tuple[list[str], list[str]]:
    """The concepts/columns and the words a cohort intent (or family population) states."""

    named: list[str] = []
    texts: list[str] = []
    for predicate in (*(getattr(cohort, "inclusion", None) or ()), *(getattr(cohort, "exclusion", None) or ())):
        named.append(str(predicate.concept_id))
    for item in getattr(cohort, "population_criteria", None) or getattr(cohort, "criteria", None) or ():
        named.extend(str(c) for c in getattr(item, "concept_ids", ()) or ())
        texts.append(str(getattr(item, "criterion", "") or ""))
    for criterion in getattr(spec, "criteria", None) or ():
        named.extend(_criterion_concepts(criterion))
        texts.append(str(getattr(criterion, "quote", "") or ""))
    return named, texts


DEFAULT_ANCHOR = "icu_admission"


@dataclass(frozen=True)
class HostStatement:
    """The host's own statement of one condition: present on its concept, in its window."""

    quote: str
    concept: str
    anchor: str
    start_hours: float
    end_hours: float

    def typed_criterion(self, criterion_id: str) -> dict[str, Any]:
        """As a ``population_spec`` criterion."""

        return {
            "id": criterion_id, "quote": self.quote, "source": "question",
            "role": "include", "kind": "condition_present", "concepts_all_of": [self.concept],
            "window": {"start_hours": self.start_hours, "end_hours": self.end_hours},
            "missing_policy": "exclude",
        }

    def legacy_criterion(self) -> dict[str, Any]:
        """As a stated criterion of the legacy population representation."""

        return {"criterion": self.quote, "concept_ids": [self.concept]}

    def legacy_predicate(self) -> dict[str, Any]:
        """The predicate applying it: recorded present in the window (missing excluded)."""

        return {
            "concept_id": self.concept, "anchor": self.anchor,
            "start_offset_hours": self.start_hours, "end_offset_hours": self.end_hours,
            "aggregation": "max", "op": "==",
            "value": {"mode": "number", "string_value": None, "number_value": 1.0,
                      "boolean_value": None, "string_list": [], "number_list": []},
        }

    def provenance(self) -> dict[str, Any]:
        """The record the cohort keeps for the plan card."""

        return {"quote": self.quote, "concept": self.concept,
                "start_hours": self.start_hours, "end_hours": self.end_hours}


def _window(context: Any, concept: str) -> tuple[str, float, float]:
    variables = {str(v.name): v for v in getattr(context, "variables", ()) or ()}
    window = context_column_windows(context).get(predicate_context_column(variables, concept, "max"))
    if (
        window is not None and window.anchor is not None
        and window.start_hours is not None and window.end_hours is not None
        and math.isfinite(window.start_hours) and math.isfinite(window.end_hours)
        and window.end_hours > window.start_hours
    ):
        return str(window.anchor), float(window.start_hours), float(window.end_hours)
    return (DEFAULT_ANCHOR, *DEFAULT_WINDOW_HOURS)


def host_statements(context: Any, conditions: Sequence[StatedCondition]) -> tuple[HostStatement, ...]:
    """The host's statement of each condition, on the first concept the input holds."""

    found = []
    for condition in conditions:
        concept = condition.concepts[0]
        anchor, start, end = _window(context, concept)
        found.append(HostStatement(quote=condition.quote, concept=concept, anchor=anchor,
                                   start_hours=start, end_hours=end))
    return tuple(found)


def next_criterion_ids(taken: Iterable[str], count: int) -> list[str]:
    """``count`` unused ``cN`` ids after ``taken``."""

    used = {str(item) for item in taken}
    ids = []
    number = 1
    while len(ids) < count and number <= _MAX_CRITERIA:
        if f"c{number}" not in used:
            ids.append(f"c{number}")
        number += 1
    return ids


def missing_condition_instruction(missing: Sequence[StatedCondition]) -> str:
    """The located reason a Planner is asked again with."""

    parts = "; ".join(f"{item.quote!r} (recorded by {', '.join(item.concepts)})" for item in missing)
    return (
        "The research question restricts its patients to a recorded condition the cohort "
        f"does not account for: {parts}. State it in population.population_spec as a "
        "condition_present criterion quoting the question's words, over one of those "
        "concepts and inside the host's time window, or list it as a criterion that cannot "
        "be applied; a cohort label or a covariate is not a population condition."
    )


__all__ = [
    "DEFAULT_ANCHOR",
    "DEFAULT_WINDOW_HOURS",
    "STATED_POPULATION_CONDITIONS_KEY",
    "StatedCondition",
    "cohort_reads",
    "HostStatement",
    "host_statements",
    "next_criterion_ids",
    "missing_condition_instruction",
    "stated_conditions",
    "unaccounted_conditions",
]
