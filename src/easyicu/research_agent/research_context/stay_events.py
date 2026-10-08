"""The events a context records for each ICU stay.

The research context types an event time that applies only when a separate
event status is positive (``conditional_event_time`` observation semantics),
and names that status; ``event_time_status_column`` is the one reading of it.
An outcome column holds what the stay records at its end
(``stay_outcome_columns``), and the time-zero rule
(``planning.cohort_eligibility``) judges a predicate on one as a test of an
outcome.  An event status recorded over the whole stay
(``whole_stay_event_columns``) cannot say whether the event happened within a
window: the cohort builder reads a window on one by the event's
``<concept>_time``, and refuses it without one (``cohort.schema``).
"""

from __future__ import annotations

from typing import Any


def event_time_status_column(variable: Any) -> str | None:
    """The event status a typed event time belongs to, or ``None``."""

    semantics = getattr(variable, "observation_semantics", None)
    if getattr(semantics, "kind", None) != "conditional_event_time":
        return None
    status = getattr(semantics, "event_status_column", None)
    return str(status) if status else None


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
    "event_time_status_column",
    "stay_outcome_columns",
    "whole_stay_event_columns",
]
