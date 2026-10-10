"""Cohort eligibility must be decided by the analysis time zero.

A stay reaches a typed minimum ICU stay exactly when it is still in the ICU at
that hour, so a minimum beyond time zero selects on survival after it.  A
concept-derived population admits a stay on a positive row up to its window's
end, so a window ending after time zero selects on what happens after it; a
window ending at time zero has decided membership by then.

The family-spec request refuses such a selection before the Planner runs, and
the scientific review refuses a plan built on one, from the same rule here.

A plan's own cohort predicates are held to the same time zero
(``cohort_predicates_after_time_zero``).  A predicate filters a column the
host materialized before planning, whatever window the predicate names: a
time-varying column was summarized over its own ``analysis_window``, else over
the window the Web host records it materialized every feature column over.  It
is decided by time zero only when both its window and that column's end by
then.  A context's analysis ``time_windows`` are not a materialization
record, so a column with neither record is refused as unrecorded.  A value
fixed at admission passes: a demographic or identifier variable, or a
stay-level concept the dictionary files under demographics.  An outcome the
stay records at its end does not, with one exception: an event status the
cohort builder reads by the event's own recorded time, keeping the stays whose
event is not in a window that ends by time zero (``_event_absence_decided``).
An ICU length of stay of ``x`` is known at ``x``, and any other stay-level
value (a first-day severity score) carries no time the host can compare.  A column that holds an event's time after ICU
admission is not summarized over a window: ``< x`` and ``<= x`` are known at
``x``, and any other test only once the event happens.  A last observation
time is a window summary like any other, since a later observation moves it.
Predicates are read as data, so a column only the run's roster knows is judged
by the same rule.

A cohort must not select stays by the study's outcome either
(``cohort_criteria_on_outcome``).  A criterion that tests the outcome's own
column cuts the outcome at its threshold, whenever it is decided; keeping the
stays still in the ICU at time zero, with their ICU length of stay as the
outcome, is a landmark cohort instead.  A criterion over a concept the outcome
is defined from selects on the outcome when it is decided after time zero; one
decided by then is a baseline value, such as an earlier event of the kind the
study counts as new.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..research_context.concept_population import ConceptCohortWindow
from ..research_context.materialization_window import host_materialization_window_hours
from ..research_context.stay_events import (
    ICU_LENGTH_OF_STAY_CONCEPT,
    column_kind,
    event_time_hours_per_unit,
    event_time_status_column,
    stay_outcome_columns,
)
from ..schema import ResearchContext
from .adjustment_authority import host_window_bound_roles
from .cohort_contract import event_status_reading


@dataclass(frozen=True)
class EligibilityAfterTimeZero:
    """One typed criterion that decides cohort membership after time zero."""

    criterion: Literal["minimum_icu_stay", "concept_population"]
    decided_by_hours: float
    time_zero_hours: float
    definition: str = ""

    def message(self) -> str:
        if self.criterion == "minimum_icu_stay":
            return (
                f"a minimum ICU stay of {self.decided_by_hours:g} h ends after the plan's "
                f"time zero at {self.time_zero_hours:g} h after ICU admission; eligibility "
                "would depend on survival after time zero"
            )
        return (
            f"the {self.definition} population admits a stay on a positive row up to "
            f"{self.decided_by_hours:g} h after ICU admission, after the plan's time zero "
            f"at {self.time_zero_hours:g} h; eligibility would depend on what happens "
            f"after time zero, so the study's cohort window must end by "
            f"{self.time_zero_hours:g} h"
        )


def eligibility_after_time_zero(
    *,
    time_zero_hours: float | None,
    minimum_icu_hours: float | None,
    concept_population: ConceptCohortWindow | None,
) -> tuple[EligibilityAfterTimeZero, ...]:
    """Each typed criterion decided after ``time_zero_hours``, minimum stay first.

    Without a time zero nothing is compared.  A criterion decided exactly at
    time zero passes.
    """

    if time_zero_hours is None:
        return ()
    found: list[EligibilityAfterTimeZero] = []
    if minimum_icu_hours is not None and minimum_icu_hours > time_zero_hours:
        found.append(
            EligibilityAfterTimeZero(
                criterion="minimum_icu_stay",
                decided_by_hours=minimum_icu_hours,
                time_zero_hours=time_zero_hours,
            )
        )
    if (
        concept_population is not None
        and concept_population.window_end_hours > time_zero_hours
    ):
        found.append(
            EligibilityAfterTimeZero(
                criterion="concept_population",
                decided_by_hours=concept_population.window_end_hours,
                time_zero_hours=time_zero_hours,
                definition=concept_population.definition,
            )
        )
    return tuple(found)


#: Anchors a population predicate's window counts from: ICU admission.
POPULATION_TIME_ZERO_ANCHORS = frozenset({"icu_admission", "icu_admit"})

#: Variable roles the host materializes over a window before planning.  A
#: ``meta`` column counts or flags observations of a concept over its window.
_WINDOWED_ROLES = frozenset(
    {
        "vital",
        "lab",
        "intervention",
        "ordinal_score",
        "composite_score",
        "other",
        "time",
        "index",
        "meta",
    }
)
_ANCHOR_WORDS = {"icu_admission": "ICU admission", "icu_admit": "ICU admission"}
_HOUR_UNITS = frozenset({"h", "hr", "hrs", "hour", "hours"})
#: Tests an event time decides once the threshold hour passes.
_BY_THRESHOLD_OPS = frozenset({"<", "<="})

PredicateReason = Literal[
    "anchor",
    "window",
    "column_window",
    "unrecorded",
    "icu_stay_length",
    "stay_outcome",
    "stay_level",
    "event_time",
    "event_time_unrecorded",
]


@dataclass(frozen=True)
class PredicateAfterTimeZero:
    """One plan cohort predicate the host cannot show decided by time zero."""

    kind: Literal["inclusion", "exclusion"]
    label: str
    reason: PredicateReason
    time_zero_hours: float
    decided_by_hours: float | None = None
    #: The window the filtered column was materialized over, as stated.
    column_window: str = ""

    def message(self) -> str:
        subject = f"the {self.kind} predicate {self.label}"
        zero = f"the plan's time zero at {self.time_zero_hours:g} h after ICU admission"
        if self.reason == "anchor":
            return f"{subject} is not counted from ICU admission, so it cannot be held to {zero}"
        if self.reason == "icu_stay_length":
            return (
                f"{subject} keeps a stay by its ICU length of stay at "
                f"{self.decided_by_hours:g} h, after {zero}; eligibility would depend on "
                "survival in the ICU after time zero"
            )
        if self.reason == "stay_outcome":
            return f"{subject} tests a value the stay records only at its end, after {zero}"
        if self.reason == "stay_level":
            return (
                f"{subject} tests a value the stay records once, which its concept owner "
                f"does not date to ICU admission, so the host cannot show it was decided by {zero}"
            )
        if self.reason == "column_window":
            return (
                f"{subject} filters a column the host summarized over {self.column_window}, "
                f"which it cannot show ended by {zero}"
            )
        if self.reason == "event_time":
            if self.decided_by_hours is None:
                return (
                    f"{subject} tests an event time it knows only once the event happens "
                    f"or its window ends, after {zero}"
                )
            return (
                f"{subject} tests an event time that is decided when "
                f"{self.decided_by_hours:g} h after ICU admission pass, after {zero}"
            )
        if self.reason == "event_time_unrecorded":
            return (
                f"{subject} tests a time whose origin or unit the context does not "
                f"record, so the host cannot show it was decided by {zero}"
            )
        if self.reason == "unrecorded":
            return (
                f"{subject} filters a column whose materialization window the context "
                f"does not record, so the host cannot show it was decided by {zero}"
            )
        decided = (
            "has no finite end"
            if self.decided_by_hours is None
            else f"is decided at {self.decided_by_hours:g} h after ICU admission"
        )
        return f"{subject} {decided}, after {zero}"


def cohort_predicates_after_time_zero(
    context: ResearchContext,
    *,
    inclusion: Iterable[Mapping[str, Any]],
    exclusion: Iterable[Mapping[str, Any]],
    time_zero_hours: float | None,
) -> tuple[PredicateAfterTimeZero, ...]:
    """Each cohort predicate the host cannot show decided by ``time_zero_hours``.

    The predicates are read as data (``concept_id``, ``time_window``,
    ``aggregation``, ``op``, ``value``) and judged by the context's own
    variables: the column a predicate filters is its bare concept, else its
    ``<concept>_<aggregation>`` summary, as the cohort builder names universe
    columns.  Without a time zero nothing is compared.  A predicate decided
    exactly at time zero passes.
    """

    if time_zero_hours is None:
        return ()
    variables = {str(variable.name): variable for variable in context.variables}
    outcomes = stay_outcome_columns(context)
    # Only a column's own window proves it here: the analysis time windows
    # are not what the host materialized.
    proven = host_window_bound_roles(
        context,
        reference_hours=time_zero_hours,
        dynamic_roles=_WINDOWED_ROLES,
        outer_window_fallback_roles=frozenset(),
    )
    materialized = host_materialization_window_hours(context)
    found: list[PredicateAfterTimeZero] = []
    for kind, predicates in (("inclusion", inclusion), ("exclusion", exclusion)):
        for predicate in predicates:
            concept = str(predicate.get("concept_id") or "")
            column = predicate_context_column(
                variables, concept, predicate.get("aggregation")
            )
            variable = variables.get(column)
            holds = column_kind(
                variable, column=column, concept=concept, outcomes=outcomes
            )
            item = PredicateAfterTimeZero(
                kind=kind,  # type: ignore[arg-type]
                label=_predicate_label(predicate),
                reason="window",
                time_zero_hours=time_zero_hours,
            )
            if holds == "icu_stay_length":
                hours = _icu_stay_threshold_hours(
                    predicate, unit=getattr(variable, "unit", None)
                )
                if hours is None:
                    found.append(_with(item, reason="stay_outcome"))
                elif hours > time_zero_hours:
                    found.append(
                        _with(item, reason="icu_stay_length", decided_by_hours=hours)
                    )
                continue
            if holds == "stay_outcome":
                if not _event_absence_decided(
                    kind, predicate, concept, variables, time_zero_hours
                ):
                    found.append(_with(item, reason="stay_outcome"))
                continue
            if holds == "admission":
                # Fixed at admission, whatever window the predicate names.
                continue
            if holds == "stay_level":
                # One value per stay, but not an admission attribute.
                found.append(_with(item, reason="stay_level"))
                continue
            # The column first: what the host filters was decided when its
            # window ended, so no window the predicate states can repair it.
            label = str(getattr(variable, "analysis_window", None) or "").strip()
            event_time, hours_per_unit = event_time_hours_per_unit(variable)
            by_threshold = False
            if event_time and hours_per_unit is not None and proven.get(column) is None:
                # An event's time is no window summary: its test decides it.
                decided = _event_time_decided_hours(predicate, hours_per_unit)
                if decided is None or decided > time_zero_hours:
                    found.append(
                        _with(item, reason="event_time", decided_by_hours=decided)
                    )
                    continue
                by_threshold = True
            if proven.get(column) is None and not by_threshold:
                if label:
                    found.append(
                        _with(item, reason="column_window", column_window=label)
                    )
                    continue
                if event_time:
                    # No window bounds an event time the host did not date.
                    found.append(_with(item, reason="event_time_unrecorded"))
                    continue
                if materialized is None:
                    found.append(_with(item, reason="unrecorded"))
                    continue
                if materialized > time_zero_hours:
                    found.append(
                        _with(
                            item,
                            reason="column_window",
                            decided_by_hours=materialized,
                            column_window=(
                                "the host's materialization window ending at "
                                f"{materialized:g} h"
                            ),
                        )
                    )
                    continue
            window = predicate.get("time_window") or {}
            if (
                str(window.get("anchor") or "").casefold()
                not in POPULATION_TIME_ZERO_ANCHORS
            ):
                found.append(_with(item, reason="anchor"))
                continue
            end = _number(window.get("end_offset_hours"))
            if end is None or end > time_zero_hours:
                found.append(_with(item, reason="window", decided_by_hours=end))
    return tuple(found)


def icu_stay_kept_after_hours(
    context: ResearchContext, *, inclusion: Iterable[Mapping[str, Any]]
) -> float | None:
    """The latest hour after ICU admission the inclusion keeps only stays still in the ICU after.

    A ``>`` comparison of the stay's ICU length of stay keeps the stays still
    in the ICU after its threshold, read in hours as the time-zero rule reads
    it; ``>=`` also keeps a stay that left at it.  ``None`` when no inclusion
    predicate keeps such stays.  A static prediction's risk set is this
    predicate at its prediction time.
    """

    variables = {str(variable.name): variable for variable in context.variables}
    outcomes = stay_outcome_columns(context)
    kept: list[float] = []
    for predicate in inclusion:
        if str(predicate.get("op") or "").strip() != ">":
            continue
        concept = str(predicate.get("concept_id") or "")
        column = predicate_context_column(variables, concept, predicate.get("aggregation"))
        variable = variables.get(column)
        if column_kind(variable, column=column, concept=concept, outcomes=outcomes) != (
            "icu_stay_length"
        ):
            continue
        hours = _icu_stay_threshold_hours(predicate, unit=getattr(variable, "unit", None))
        if hours is not None:
            kept.append(hours)
    return max(kept) if kept else None


def predicate_context_column(
    variables: Mapping[str, Any], concept: str, aggregation: Any
) -> str:
    """The context column a plan predicate filters: its bare concept, else its summary.

    These are the first two names ``cohort.schema`` tries for a universe
    column; a column it binds otherwise is judged by the bare concept's.
    Plan review reads a predicate's column here, so every check judges the
    column the host will filter.
    """

    summary = f"{concept}_{aggregation}"
    return summary if concept not in variables and summary in variables else concept


def event_status_read_by_its_time(variables: Mapping[str, Any], concept: str) -> bool:
    """Whether the context types ``<concept>_time`` as the time of ``concept``'s event.

    The cohort builder reads a windowed predicate on an event status by the
    ``<concept>_time`` column beside it, as hours after ICU admission like the
    window itself.  The context types that column as the event's time when its
    observation semantics name ``concept`` as their event status and it counts
    from ICU admission in hours (``event_time_hours_per_unit``): a time in days or minutes
    would be compared with the window as if it were hours.  The family-spec
    risk set asks the same question before it writes such a predicate.
    """

    companion = variables.get(f"{concept}_time")
    if companion is None or event_time_status_column(companion) != concept:
        return False
    event_time, hours_per_unit = event_time_hours_per_unit(companion)
    return event_time and hours_per_unit == 1.0


#: How a cohort criterion selects stays by the study's outcome.
OutcomeSelection = Literal[
    "outcome_itself", "landmark_stay_length", "defines_outcome_after_time_zero"
]
#: Tests that ask only whether a column holds a value, not which.
_MISSINGNESS_OPS = frozenset({"missing", "not_missing"})
#: ICU length-of-stay tests that keep the stays still in the ICU at their threshold.
_STILL_IN_ICU_TESTS = frozenset(
    {("inclusion", ">="), ("inclusion", ">"), ("exclusion", "<"), ("exclusion", "<=")}
)
#: The reasons by which the time-zero rule finds a selection made after time
#: zero, the study's to change.  The others are a predicate window the Planner
#: restates over a column decided by then (``anchor``, ``window``) and a
#: record the host lacks (``unrecorded``, ``event_time_unrecorded``).
_SELECTED_AFTER_TIME_ZERO: frozenset[PredicateReason] = frozenset(
    {"column_window", "icu_stay_length", "stay_outcome", "stay_level", "event_time"}
)


@dataclass(frozen=True)
class CriterionOnOutcome:
    """One cohort criterion that selects stays by the study's outcome.

    ``index`` is a plan predicate's position on its side of the cohort, and
    ``None`` for the study's own minimum ICU stay.
    """

    kind: Literal["inclusion", "exclusion"]
    index: int | None
    label: str
    outcome: str
    selection: OutcomeSelection
    time_zero_hours: float | None
    #: When an ICU length-of-stay criterion is decided, in hours after ICU admission.
    decided_by_hours: float | None = None
    #: The concept the outcome is defined from that the criterion tests.
    concept: str = ""

    def message(self) -> str:
        subject = f"the {self.kind} criterion {self.label}"
        zero_hours = self.time_zero_hours
        zero = (
            ""
            if zero_hours is None
            else f"the plan's time zero at {zero_hours:g} h after ICU admission"
        )
        if self.selection == "landmark_stay_length":
            return (
                f"{subject} keeps the stays still in the ICU at {zero}, and the "
                f"study's outcome {self.outcome} is their whole ICU length of stay, "
                "counted from admission among the stays that reached time zero"
            )
        if self.selection == "defines_outcome_after_time_zero":
            return (
                f"{subject} tests {self.concept}, which the study's outcome "
                f"{self.outcome} is defined from, and cannot be shown decided by "
                f"{zero}, so it may keep stays by part of their own outcome"
            )
        text = (
            f"{subject} selects stays by the study's outcome {self.outcome} itself: "
            "whatever its threshold, the outcome of the stays it keeps is cut there, "
            "so estimates describe only stays whose outcome passed it"
        )
        decided = self.decided_by_hours
        if zero_hours is None:
            return text + ", and the plan states no time zero to count the outcome from"
        if decided is not None and decided < zero_hours:
            return text + (
                f"; it is decided at {decided:g} h, before {zero}, so it keeps "
                "stays whose outcome ended before time zero"
            )
        if decided is not None and decided > zero_hours:
            return text + f"; it is decided at {decided:g} h, after {zero}"
        return text


def cohort_criteria_on_outcome(
    context: ResearchContext,
    *,
    outcomes: Iterable[str],
    inclusion: Sequence[Mapping[str, Any]],
    exclusion: Sequence[Mapping[str, Any]],
    time_zero_hours: float | None,
    minimum_icu_hours: float | None,
) -> tuple[CriterionOnOutcome, ...]:
    """Each cohort criterion that selects stays by one of ``outcomes``.

    The criteria are the plan's cohort predicates, read as data as
    ``cohort_predicates_after_time_zero`` reads them, and the study's typed
    minimum ICU stay, a test of ``los_icu``.  A criterion that tests an
    outcome's own column selects on the outcome whatever its threshold or
    time.  The stays still in the ICU at the plan's time zero are the one
    exception: with an ICU length-of-stay outcome they are a landmark cohort,
    whose outcome is the whole stay's length among the stays that reached time
    zero (``landmark_stay_length``).  A criterion over a concept an outcome is
    defined from (its ``source_concept`` and ``derived_from_concepts``, through
    the context's own variables) selects on the outcome only when the
    time-zero rule finds the selection made after time zero; decided by then,
    it is a baseline value, and a predicate window the Planner restates or a
    record the host lacks stays that rule's finding alone.  Two kinds of test
    are the time-zero rule's alone and never named here: one that asks only
    whether a value is recorded, and one that keeps the stays without an
    event in a window by the event's own time
    (``_event_free_entry``), a baseline exclusion when the window ends by time
    zero and a survival into the cohort that the time-zero rule refuses when
    it ends later.
    """

    studied = tuple(
        dict.fromkeys(str(item).strip() for item in outcomes if str(item or "").strip())
    )
    if not studied:
        return ()
    variables = {str(variable.name): variable for variable in context.variables}
    defined_from = {outcome: _outcome_definition(variables, outcome) for outcome in studied}
    found: list[CriterionOnOutcome] = []
    for kind, predicates in (("inclusion", inclusion), ("exclusion", exclusion)):
        for index, predicate in enumerate(predicates):
            op = str(predicate.get("op") or "")
            concept = str(predicate.get("concept_id") or "")
            if op in _MISSINGNESS_OPS or _event_free_entry(kind, predicate, concept, variables):
                continue
            column = predicate_context_column(variables, concept, predicate.get("aggregation"))
            variable = variables.get(column)
            hours = (
                _icu_stay_threshold_hours(predicate, unit=getattr(variable, "unit", None))
                if concept == ICU_LENGTH_OF_STAY_CONCEPT
                else None
            )
            criterion = CriterionOnOutcome(
                kind=kind,  # type: ignore[arg-type]
                index=index,
                label=_predicate_label(predicate),
                outcome=column,
                selection="outcome_itself",
                time_zero_hours=time_zero_hours,
                decided_by_hours=hours,
            )
            if column in studied:
                landmark = (
                    (kind, op) in _STILL_IN_ICU_TESTS
                    and not isinstance(predicate.get("value"), list)
                    and _at_time_zero(hours, time_zero_hours)
                )
                found.append(
                    _on_outcome(criterion, selection="landmark_stay_length")
                    if landmark
                    else criterion
                )
                continue
            tested = {concept, column, str(getattr(variable, "source_concept", None) or "")}
            for outcome in studied:
                shared = sorted(tested & defined_from[outcome])
                if shared and any(
                    item.reason in _SELECTED_AFTER_TIME_ZERO
                    for item in cohort_predicates_after_time_zero(
                        context,
                        inclusion=[predicate] if kind == "inclusion" else [],
                        exclusion=[predicate] if kind == "exclusion" else [],
                        time_zero_hours=time_zero_hours,
                    )
                ):
                    found.append(
                        _on_outcome(
                            criterion,
                            outcome=outcome,
                            selection="defines_outcome_after_time_zero",
                            concept=shared[0],
                        )
                    )
                    break
    if minimum_icu_hours is not None:
        stay = CriterionOnOutcome(
            kind="inclusion",
            index=None,
            label=f"of a minimum ICU stay of {minimum_icu_hours:g} h (the study's)",
            outcome=ICU_LENGTH_OF_STAY_CONCEPT,
            selection="outcome_itself",
            time_zero_hours=time_zero_hours,
            decided_by_hours=minimum_icu_hours,
        )
        late = eligibility_after_time_zero(
            time_zero_hours=time_zero_hours,
            minimum_icu_hours=minimum_icu_hours,
            concept_population=None,
        )
        if ICU_LENGTH_OF_STAY_CONCEPT in studied:
            found.append(
                _on_outcome(stay, selection="landmark_stay_length")
                if _at_time_zero(minimum_icu_hours, time_zero_hours)
                else stay
            )
        elif late:
            for outcome in studied:
                if ICU_LENGTH_OF_STAY_CONCEPT in defined_from[outcome]:
                    found.append(
                        _on_outcome(
                            stay,
                            outcome=outcome,
                            selection="defines_outcome_after_time_zero",
                            concept=ICU_LENGTH_OF_STAY_CONCEPT,
                        )
                    )
                    break
    return tuple(found)


def _outcome_definition(variables: Mapping[str, Any], outcome: str) -> frozenset[str]:
    """The concepts ``outcome`` is defined from, through the context's own variables."""

    found: set[str] = set()
    pending = [outcome]
    while pending:
        variable = variables.get(pending.pop())
        for name in (
            getattr(variable, "source_concept", None),
            *(getattr(variable, "derived_from_concepts", None) or ()),
        ):
            concept = str(name or "").strip()
            if concept and concept != outcome and concept not in found:
                found.add(concept)
                pending.append(concept)
    return frozenset(found)


def _at_time_zero(hours: float | None, time_zero_hours: float | None) -> bool:
    return (
        hours is not None
        and time_zero_hours is not None
        and math.isclose(hours, time_zero_hours, rel_tol=0.0, abs_tol=1e-9)
    )


def _on_outcome(item: CriterionOnOutcome, **changes: Any) -> CriterionOnOutcome:
    return CriterionOnOutcome(**{**item.__dict__, **changes})


def _event_absence_decided(
    kind: str,
    predicate: Mapping[str, Any],
    concept: str,
    variables: Mapping[str, Any],
    time_zero_hours: float,
) -> bool:
    """Whether a test of an event status keeps the stays without the event by time zero.

    An outcome's status is recorded at the stay's end.  The cohort builder
    reads an equality on an event status over a finite window by the event's
    own time instead (``<concept>_time``), and the stays it then keeps are
    those whose event is not in the window: an inclusion of the event's
    absence, or an exclusion of its occurrence (``event_status_reading``).
    That is known when the window ends.  It holds only when the context types
    that time: a ``<concept>_time`` column whose event status is the concept,
    counted from ICU admission in hours.  Without one the builder
    reads the status over the whole stay.  The window is anchored at ICU
    admission, both its ends are finite and in order, and it ends by time zero.
    """

    end = _number((predicate.get("time_window") or {}).get("end_offset_hours"))
    return (
        _event_free_entry(kind, predicate, concept, variables)
        and end is not None
        and end <= time_zero_hours
    )


def _event_free_entry(
    kind: str, predicate: Mapping[str, Any], concept: str, variables: Mapping[str, Any]
) -> bool:
    """Whether a predicate keeps the stays whose event is not in its window, by the event's time.

    The cohort builder reads such a predicate by ``<concept>_time``
    (``_event_absence_decided``): an inclusion of the event's absence, or an
    exclusion of its occurrence, over a finite window counted from ICU
    admission.  Whether it is decided by time zero is the window's end.
    """

    reading = event_status_reading(
        str(predicate.get("op") or ""), predicate.get("value")
    )
    if (kind, reading) not in {("inclusion", "absence"), ("exclusion", "occurrence")}:
        return False
    if not event_status_read_by_its_time(variables, concept):
        return False
    window = predicate.get("time_window") or {}
    if str(window.get("anchor") or "").casefold() not in POPULATION_TIME_ZERO_ANCHORS:
        return False
    start = _number(window.get("start_offset_hours"))
    end = _number(window.get("end_offset_hours"))
    return start is not None and end is not None and start < end


def _event_time_decided_hours(
    predicate: Mapping[str, Any], hours_per_unit: float
) -> float | None:
    """The hour an event-time test is known; None until the event happens."""

    value = _number(predicate.get("value"))
    if str(predicate.get("op") or "") not in _BY_THRESHOLD_OPS or value is None:
        return None
    return max(value * hours_per_unit, 0.0)


def _with(item: PredicateAfterTimeZero, **changes: Any) -> PredicateAfterTimeZero:
    return PredicateAfterTimeZero(**{**item.__dict__, **changes})


def _icu_stay_threshold_hours(
    predicate: Mapping[str, Any], *, unit: Any
) -> float | None:
    """The hour an ICU length-of-stay comparison is known; None without a number."""

    values = predicate.get("value")
    numbers = [
        _number(item) for item in (values if isinstance(values, list) else [values])
    ]
    if not numbers or any(item is None for item in numbers):
        return None
    hours_per_unit = 1.0 if str(unit or "").strip().casefold() in _HOUR_UNITS else 24.0
    return max(numbers) * hours_per_unit  # type: ignore[type-var]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _predicate_label(predicate: Mapping[str, Any]) -> str:
    window = predicate.get("time_window") or {}
    op = str(predicate.get("op") or "")
    test = f"{predicate.get('concept_id')} {op}"
    if op not in {"missing", "not_missing"}:
        test += f" {predicate.get('value')!r}"
    anchor = str(window.get("anchor") or "")
    return (
        f"{test} ({predicate.get('aggregation')} over "
        f"{_hours_text(window.get('start_offset_hours'))} to "
        f"{_hours_text(window.get('end_offset_hours'))} h from "
        f"{_ANCHOR_WORDS.get(anchor.casefold(), anchor)})"
    )


def _hours_text(value: Any) -> str:
    number = _number(value)
    return f"{number:g}" if number is not None else str(value)


__all__ = [
    "CriterionOnOutcome",
    "EligibilityAfterTimeZero",
    "ICU_LENGTH_OF_STAY_CONCEPT",
    "OutcomeSelection",
    "POPULATION_TIME_ZERO_ANCHORS",
    "PredicateAfterTimeZero",
    "cohort_criteria_on_outcome",
    "cohort_predicates_after_time_zero",
    "eligibility_after_time_zero",
    "event_status_read_by_its_time",
    "icu_stay_kept_after_hours",
    "predicate_context_column",
]
