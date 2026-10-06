"""A cohort predicate has to separate rows by the values its column takes.

A predicate filters one column the host materialized before planning: its
bare concept, else its ``<concept>_<aggregation>`` summary
(``cohort_eligibility.predicate_context_column``).  When that column's values
form a closed set -- the levels observed in the cohort, else the set its
owner declares (``closed_planning_levels_for``) -- each value is judged with
the comparison the cohort builder applies (``predicate_accepts_closed_level``).
The judgement covers the rows that have a value: a row without one meets no
comparison except ``!=`` and ``not_in``.

An inclusion that no value meets keeps no row that has a value, and an
exclusion that every value meets removes every such row: the cohort is empty,
or holds only rows that lack the column.  An inclusion that every value
meets, or an exclusion that none meets, cannot separate rows by value, so the
criterion it was written for is not applied -- unless the study states a
bound that every value meets.  That second pair is judged only against a
declared set: levels observed in one cohort are not every value a column can
take, and an exclusion that happens to match no one there is not an error.

A planning schema has no rows, so only a declared set can catch a threshold
copied from a concept's description, such as "SOFA >= 2" read off a 0/1
diagnosis flag.  A set describes a predicate only under a summary that keeps
the column's own values (max, min, first, last); ``any`` and ``all``
summarize an event status, so they keep a 0/1 set only.  A mean, a median, a
sum or a count takes other values.  Predicates without a value (``missing``,
``not_missing``) are not judged either, nor is a column without a closed set
(a count, a time, a numeric summary).  An observed set is described by its
size only, never by its values.
"""

from __future__ import annotations

import json
import numbers
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from ..authority.declared_levels import closed_planning_levels_for, observed_levels_for
from ..cohort.schema import ConceptPredicate, predicate_accepts_closed_level
from ..schema import ResearchContext
from .cohort_eligibility import predicate_context_column

_VALUE_FREE_OPS = frozenset({"missing", "not_missing"})
_VALUE_KEEPING_AGGREGATIONS = frozenset({"max", "min", "first", "last", "any", "all"})
_EVENT_STATUS_AGGREGATIONS = frozenset({"any", "all"})

Effect = Literal["keeps_no_row", "removes_every_row", "keeps_every_row", "removes_no_row"]

_OUTCOMES: dict[str, str] = {
    "keeps_no_row": "none of them meets it, so it keeps no row that has a value",
    "removes_every_row": "every one of them meets it, so it removes every row that has a value",
    "keeps_every_row": "every one of them meets it, so it keeps every row that has a value",
    "removes_no_row": "none of them meets it, so it removes no row that has a value",
}


@dataclass(frozen=True)
class PredicateOutsideColumnDomain:
    """One cohort predicate whose comparison its column's values cannot honour."""

    side: Literal["inclusion", "exclusion"]
    index: int
    concept_id: str
    aggregation: str
    op: str
    value: Any
    column: str
    effect: Effect
    level_count: int
    #: The owner's declared values; ``None`` when the set was observed.
    declared_levels: tuple[Any, ...] | None

    @property
    def empties_cohort(self) -> bool:
        """No row with a value is left: it keeps none, or removes every one."""

        return self.effect in {"keeps_no_row", "removes_every_row"}

    def message(self) -> str:
        predicate = (
            f"{self.concept_id} ({self.aggregation}) {self.op} {_value_text(self.value)}"
        )
        if self.declared_levels is not None:
            values = (
                f"its owner declares the values "
                f"{json.dumps(list(self.declared_levels), ensure_ascii=False)}"
            )
        else:
            values = f"it holds {self.level_count} observed values"
        return (
            f"The cohort {self.side} {predicate} filters column {self.column!r}: "
            f"{values}, and {_OUTCOMES[self.effect]}"
        )


def cohort_predicates_outside_column_domain(
    context: ResearchContext,
    *,
    inclusion: Iterable[ConceptPredicate],
    exclusion: Iterable[ConceptPredicate],
) -> tuple[PredicateOutsideColumnDomain, ...]:
    """Each predicate that empties the cohort or cannot separate rows by value."""

    variables = {variable.name: variable for variable in context.variables}
    found: list[PredicateOutsideColumnDomain] = []
    for side, predicates in (("inclusion", inclusion), ("exclusion", exclusion)):
        for index, predicate in enumerate(predicates):
            op = str(predicate.op or "").strip().casefold()
            aggregation = str(predicate.aggregation or "").strip().casefold()
            if op in _VALUE_FREE_OPS or aggregation not in _VALUE_KEEPING_AGGREGATIONS:
                continue
            column = predicate_context_column(
                variables, predicate.concept_id, predicate.aggregation
            )
            levels = closed_planning_levels_for(name=column, variables=variables)
            if len(levels) < 2:
                continue
            if aggregation in _EVENT_STATUS_AGGREGATIONS and not _event_status(levels):
                continue
            verdicts = [predicate_accepts_closed_level(predicate, level) for level in levels]
            if any(verdict is None for verdict in verdicts):
                continue
            met = sum(1 for verdict in verdicts if verdict)
            observed = bool(observed_levels_for(name=column, variables=variables))
            effect = _effect(side, met=met, total=len(levels), declared=not observed)
            if effect is None:
                continue
            found.append(
                PredicateOutsideColumnDomain(
                    side=side,
                    index=index,
                    concept_id=str(predicate.concept_id),
                    aggregation=str(predicate.aggregation),
                    op=str(predicate.op),
                    value=predicate.value,
                    column=column,
                    effect=effect,
                    level_count=len(levels),
                    declared_levels=None if observed else tuple(levels),
                )
            )
    return tuple(found)


def _effect(side: str, *, met: int, total: int, declared: bool) -> Effect | None:
    if side == "inclusion":
        if met == 0:
            return "keeps_no_row"
        if met == total and declared:
            return "keeps_every_row"
        return None
    if met == total:
        return "removes_every_row"
    if met == 0 and declared:
        return "removes_no_row"
    return None


def _event_status(levels: Sequence[Any]) -> bool:
    return all(
        isinstance(level, numbers.Real) and level in (0, 1) for level in levels
    )


def _value_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, str) else f"{value}"


__all__ = [
    "PredicateOutsideColumnDomain",
    "cohort_predicates_outside_column_domain",
]
