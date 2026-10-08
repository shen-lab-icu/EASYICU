"""The events a context records for each ICU stay.

The research context types an event time that applies only when a separate
event status is positive (``conditional_event_time`` observation semantics),
and names that status; ``event_time_status_column`` is the one reading of it.
The hours after ICU admission one unit of an event time stands for are read
here too (``event_time_hours_per_unit``): the cohort builder compares an event
time with a window in those hours, and refuses a time the context types in
another unit or from another origin (``cohort.schema``).
An outcome column holds what the stay records at its end
(``stay_outcome_columns``), and the time-zero rule
(``planning.cohort_eligibility``) judges a predicate on one as a test of an
outcome.  An event status recorded over the whole stay
(``whole_stay_event_columns``) cannot say whether the event happened within a
window: the cohort builder reads a window on one by the event's
``<concept>_time``, and refuses it without one (``cohort.schema``).  What a
column holds for each stay (``column_kind``) is read here once, for the
time-zero rule and for the window a column was summarized over
(``materialization_window``).
"""

from __future__ import annotations

from typing import AbstractSet, Any, Literal

from ..concept_availability import stay_level_concept_category
from .temporal_semantics import normalise_time_anchor

#: The concept dictionary's ICU length of stay, in days unless its variable
#: states hours.
ICU_LENGTH_OF_STAY_CONCEPT = "los_icu"
#: Variable roles fixed at admission: the owner-declared admission attributes
#: (the precedent of ``owner_declared_baseline_static``) and the stay's ids.
ADMISSION_ROLES = frozenset({"demographic", "id"})
#: Dictionary categories of stay-level concepts fixed at admission and known
#: only at the stay's end.
_ADMISSION_CATEGORY = "demographics"
_STAY_END_CATEGORY = "outcome"

#: What a column holds for each stay.
ColumnKind = Literal[
    "icu_stay_length",
    "stay_outcome",
    "admission",
    "stay_level",
    "event_time",
    "window_summary",
]

#: Hours per unit of the export's event-time units; any other unit is unread.
_TIME_UNIT_HOURS = {"h": 1.0, "d": 24.0, "min": 1 / 60}
#: The representation of a window's last observation time, which a later
#: observation moves until the window ends.
_LAST_TIME_TRANSFORM = "window_last_time"


def event_time_status_column(variable: Any) -> str | None:
    """The event status a typed event time belongs to, or ``None``."""

    semantics = getattr(variable, "observation_semantics", None)
    if getattr(semantics, "kind", None) != "conditional_event_time":
        return None
    status = getattr(semantics, "event_status_column", None)
    return str(status) if status else None


def _event_time_coordinates(variable: Any) -> tuple[bool, str | None, str | None]:
    """Whether a column holds an event's time, and the origin and unit typed for it."""

    if str(getattr(variable, "unit_normalization", None) or "") == _LAST_TIME_TRANSFORM:
        return False, None, None
    semantics = getattr(variable, "observation_semantics", None)
    origin = getattr(semantics, "time_origin", None)
    unit = getattr(semantics, "time_unit", None)
    resolution = str(getattr(variable, "temporal_resolution", None) or "")
    relative = (
        resolution.removeprefix("relative to ")
        if resolution.startswith("relative to ")
        else ""
    )
    parsed_origin, separator, parsed_unit = relative.rpartition(" in ")
    if separator:
        origin, unit = origin or parsed_origin, unit or parsed_unit
    event_time = (
        getattr(semantics, "kind", None) == "conditional_event_time"
        or bool(separator)
        or _role(variable) == "time"
    )
    return (
        event_time,
        str(origin) if origin else None,
        str(unit) if unit else None,
    )


def event_time_hours_per_unit(variable: Any) -> tuple[bool, float | None]:
    """Whether a column holds an event's time, and its hours per unit after ICU admission.

    The research context types an event time as ``conditional_event_time``
    observation semantics, with the export's ``relative to <origin> in
    <unit>`` resolution read as ``observation_semantics`` reads it; a
    ``time`` role is an event time too.  Its hours per unit are known only
    when its origin is ICU admission and its unit one the export writes.  A
    last observation time is no event time here: its window decides it.
    """

    event_time, origin, unit = _event_time_coordinates(variable)
    if not event_time or not origin:
        return event_time, None
    if normalise_time_anchor(origin) != "icu_admission":
        return True, None
    return True, _TIME_UNIT_HOURS.get(str(unit or "").strip())


def event_time_typed_otherwise_than_hours(variable: Any) -> bool:
    """Whether the context types an event time in other than hours after ICU admission.

    An event time whose origin and unit the context does not type is read in
    hours after ICU admission, as the builder writes every time it derives;
    one it types otherwise (in days, or from hospital admission) would be
    compared with a window in hours as if it counted them.  So would one it
    types with only its origin or only its unit: the context states it is
    typed, and not that it counts those hours.
    """

    event_time, origin, unit = _event_time_coordinates(variable)
    if not event_time or not (origin or unit):
        return False
    return event_time_hours_per_unit(variable)[1] != 1.0


def event_times_typed_otherwise_than_hours(
    context: Any,
) -> dict[str, tuple[str | None, str | None]]:
    """The event-time columns of ``context`` typed in other than hours after ICU admission.

    Each with the origin and the unit the context types for it, ``None`` for
    one it does not state.
    """

    found: dict[str, tuple[str | None, str | None]] = {}
    for variable in getattr(context, "variables", None) or ():
        if event_time_typed_otherwise_than_hours(variable):
            _event_time, origin, unit = _event_time_coordinates(variable)
            found[str(variable.name)] = (origin, unit)
    return found


def column_kind(
    variable: Any, *, column: str, concept: str, outcomes: AbstractSet[str]
) -> ColumnKind:
    """What ``column`` holds for each stay, read through ``concept``.

    In this order: the ICU length of stay; an outcome the stay records at its
    end (``column`` or ``concept`` among ``outcomes``, or a stay-level concept
    the dictionary files under outcome); a value fixed at admission (an id or
    demographic role, or a stay-level concept filed under demographics); any
    other stay-level value (a first-day severity score); an event's time
    (``event_time_hours_per_unit``); else a summary of observations over a
    window.  ``variable`` is the context's descriptor of ``column``, or
    ``None`` when the context has none.
    """

    if concept == ICU_LENGTH_OF_STAY_CONCEPT:
        return "icu_stay_length"
    category = stay_level_concept_category(concept)
    if {concept, column} & outcomes or category == _STAY_END_CATEGORY:
        return "stay_outcome"
    if _role(variable) in ADMISSION_ROLES or category == _ADMISSION_CATEGORY:
        return "admission"
    if category is not None:
        return "stay_level"
    event_time, _hours = event_time_hours_per_unit(variable)
    return "event_time" if event_time else "window_summary"


def stay_outcome_columns(context: Any) -> frozenset[str]:
    """The columns of ``context`` that hold an outcome, which the stay records at its end.

    The cohort roster's outcome columns, the target outcome and every
    variable whose role is outcome.
    """

    cohort = getattr(context, "cohort", None)
    target = getattr(context, "target_outcome", None)
    return frozenset(
        {
            *(str(column) for column in getattr(cohort, "outcome_columns", None) or ()),
            *([str(target)] if target else []),
            *(
                str(variable.name)
                for variable in getattr(context, "variables", None) or ()
                if _role(variable) == "outcome"
            ),
        }
    )


def whole_stay_event_columns(context: Any) -> frozenset[str]:
    """The columns of ``context`` that record whether an event happened during the stay.

    An outcome column (``stay_outcome_columns``) and the event status of every
    event time the context types (``conditional_event_time``), unless the
    column has its own analysis window: those record the event over the whole
    ICU stay.  The cohort builder reads a window on such a column by the
    event's ``<concept>_time``; without that column it would read the whole
    stay, so a finite window on one is refused instead
    (``cohort.schema.predicates_read_over_the_whole_stay``).
    """

    variables = {
        str(variable.name): variable
        for variable in getattr(context, "variables", None) or ()
    }
    statuses = {
        status
        for variable in variables.values()
        if (status := event_time_status_column(variable)) is not None
    }
    return frozenset(
        column
        for column in stay_outcome_columns(context) | statuses
        if not str(
            getattr(variables.get(column), "analysis_window", None) or ""
        ).strip()
    )


def _role(variable: Any) -> str:
    role = getattr(variable, "role", None)
    return str(getattr(role, "value", role) or "").casefold()


__all__ = [
    "ADMISSION_ROLES",
    "ICU_LENGTH_OF_STAY_CONCEPT",
    "ColumnKind",
    "column_kind",
    "event_time_hours_per_unit",
    "event_time_status_column",
    "event_time_typed_otherwise_than_hours",
    "event_times_typed_otherwise_than_hours",
    "stay_outcome_columns",
    "whole_stay_event_columns",
]
