"""The window the Web host materialized a study's feature columns over.

``data_constraints.materialization_window`` is the study's time window as the
launch executed it (role ``outer_observation_window``): every time-varying
feature column was summarized from ICU admission through ``hours``, or
``observation_hours`` when the study binds only that.  It bounds measurement
opportunity; it is not a phenotype's definition window or the outcome's
follow-up.  A context without it (the CLI, a benchmark) records no physical
window: its ``time_windows`` name analysis windows, which default to the
whole stay, not what was materialized.  There the windows its caller declared
bind the run; the windows the context builder synthesized bind nothing
(:func:`bound_feature_window_end_hours`).

Each column holds what it was summarized over (:func:`context_column_windows`):
its own ``analysis_window`` label, else, for a time-varying column, the host's
materialization window.  A cohort predicate filters the column as it was
summarized, whatever window the predicate states, so the cohort builder reads
a predicate only over its column's window (``cohort.schema``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..icu_rules import default_time_windows
from ..schema import ResearchContext, TimeWindow
from .concept_population import context_data_constraints
from .stay_events import (
    column_kind,
    stay_outcome_columns,
    whole_stay_event_columns,
)
from .temporal_semantics import TemporalAlignmentEngine, normalise_time_anchor

#: Spellings of the admissions a window counts from, as plans and labels write
#: them, beyond those ``normalise_time_anchor`` already collapses.
_ANCHOR_ALIASES = {"icu_admit": "icu_admission", "hospital_admit": "hospital_admission"}
#: A column's own window label: ``<anchor>[<start>,<end>]h``, the form the
#: research context writes, or ``<anchor>_<start>_<end>h``.
_WINDOW_LABEL = re.compile(
    r"(?P<anchor>[a-z][a-z0-9_]*?)\s*(?:\[\s*(?P<start>-?\d+(?:\.\d+)?)\s*,"
    r"\s*(?P<end>-?\d+(?:\.\d+)?)\s*\]|_(?P<start2>-?\d+(?:\.\d+)?)_"
    r"(?P<end2>-?\d+(?:\.\d+)?))\s*h"
)
#: The same label with its punctuation written as underscores
#: (``icu admission 0-24 h``); its bounds carry no sign.
_LOOSE_WINDOW_LABEL = re.compile(
    r"(?P<anchor>[a-z][a-z0-9_]*?)_(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)_?h"
)


def window_anchor(anchor: Any) -> str:
    """One identity for the admission a window counts from."""

    text = normalise_time_anchor(str(anchor or ""))
    return _ANCHOR_ALIASES.get(text, text)


@dataclass(frozen=True)
class ColumnWindow:
    """The window a column was summarized over, as the context records it.

    ``anchor`` is ``None`` when the column's own label names no window this
    owner reads (``entire_stay``): such a column is read over no window a
    predicate can state.  ``source`` is ``"column"`` for the column's own
    label and ``"host"`` for the window the host materialized it over.
    """

    label: str
    anchor: str | None
    start_hours: float | None = None
    end_hours: float | None = None
    source: str = "column"

    def description(self) -> str:
        if self.anchor is None:
            return (
                f"its own window label {self.label!r}, which names no window a "
                "predicate can state"
            )
        if self.source == "host":
            return f"the host's materialization window {self.label}"
        return f"its own window {self.label}"

    def is_window(self, anchor: Any, start_hours: float, end_hours: float) -> bool:
        """Whether ``[start_hours, end_hours)`` from ``anchor`` is this window."""

        return (
            self.anchor is not None
            and self.start_hours is not None
            and self.end_hours is not None
            and window_anchor(anchor) == self.anchor
            and math.isclose(float(start_hours), self.start_hours)
            and math.isclose(float(end_hours), self.end_hours)
        )

    def contains(self, anchor: Any, start_hours: float, end_hours: float) -> bool:
        """Whether ``[start_hours, end_hours)`` from ``anchor`` lies within this window."""

        return (
            self.anchor is not None
            and self.start_hours is not None
            and self.end_hours is not None
            and window_anchor(anchor) == self.anchor
            and float(start_hours) >= self.start_hours
            and float(end_hours) <= self.end_hours
        )


def column_window_from_label(label: Any) -> ColumnWindow | None:
    """The window a column's own ``analysis_window`` label names; ``None`` without one."""

    text = str(label or "").strip()
    if not text:
        return None
    folded = text.casefold()
    match = _WINDOW_LABEL.fullmatch(folded)
    if match is not None:
        start = match.group("start") or match.group("start2")
        end = match.group("end") or match.group("end2")
    else:
        loose = re.sub(r"[^a-z0-9.]+", "_", folded).strip("_")
        match = _LOOSE_WINDOW_LABEL.fullmatch(loose)
        if match is None:
            return ColumnWindow(label=text, anchor=None)
        start, end = match.group("start"), match.group("end")
    return ColumnWindow(
        label=text,
        anchor=window_anchor(match.group("anchor")),
        start_hours=float(start),
        end_hours=float(end),
    )


def host_materialization_window_hours(context: ResearchContext) -> float | None:
    """Hours after ICU admission the host summarized feature columns through.

    ``None`` when the context records no readable window counted from ICU
    admission.
    """

    window = context_data_constraints(context).get("materialization_window")
    if (
        not isinstance(window, Mapping)
        or window.get("role") != "outer_observation_window"
        or normalise_time_anchor(str(window.get("anchor") or "")) != "icu_admission"
    ):
        return None
    value = window.get("hours")
    if value is None:
        value = window.get("observation_hours")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    hours = float(value)
    return hours if math.isfinite(hours) and hours > 0 else None


def caller_declared_time_windows(context: Any) -> tuple[TimeWindow, ...]:
    """The context's windows when its caller declared them; none when synthesized.

    The context builder keeps the caller's windows, else the windows it infers
    from the question's wording, else its default roster -- one kind, never a
    mix.  So the context's windows were synthesized exactly when they equal
    what the builder synthesizes for this question.
    """

    windows = list(getattr(context, "time_windows", None) or ())
    if not windows or windows == default_time_windows():
        return ()
    preferences = getattr(context, "user_preferences", None)
    inferred, _constraints = TemporalAlignmentEngine().infer(
        research_question=str(getattr(context, "research_question", "") or ""),
        timing_and_design=getattr(preferences, "timing_and_design", None),
    )
    if inferred and windows == list(inferred):
        return ()
    return tuple(windows)


def bound_feature_window_end_hours(context: Any) -> float | None:
    """Hours after ICU admission that bind the run's feature columns, if any.

    The host's materialization record answers where it exists.  Without one
    (the CLI, a benchmark), the windows the caller declared bind the run.  A
    window the context builder synthesized binds nothing: the default roster
    names analysis windows (``full_stay`` ends at 720 h), and a window inferred
    from the question's wording may be its outcome's horizon ("death within 72
    hours").  So no landmark, prediction time, time zero or covariate timing is
    read from one.  Windows with another anchor are not on the ICU-admission
    axis and are ignored.  ``None`` when nothing binds the run.
    """

    if getattr(context, "user_preferences", None) is not None:
        materialized = host_materialization_window_hours(context)
        if materialized is not None:
            return materialized
    ends = [
        float(window.end_hours)
        for window in caller_declared_time_windows(context)
        if str(getattr(window, "anchor", "") or "") == "icu_admission"
        and getattr(window, "end_hours", None) is not None
    ]
    return max(ends) if ends else None


def context_column_windows(context: Any) -> dict[str, ColumnWindow]:
    """The window each column of ``context`` was summarized over, by column name.

    A column's own ``analysis_window`` label names it.  A column without one is
    summarized over the host's materialization window when it holds a summary
    of observations over a window (``stay_events.column_kind``, read through
    its source concept, the reading the time-zero rule shares): not the ICU
    length of stay, an outcome, a value fixed at admission, another
    stay-level value or an event's time, and not an event status recorded over
    the whole stay, which is read by the event's time
    (``whole_stay_event_columns``).  A column with neither record is absent:
    no window can be stated for it.
    """

    variables = list(getattr(context, "variables", None) or ())
    if not variables:
        return {}
    hours = (
        host_materialization_window_hours(context)
        if getattr(context, "user_preferences", None) is not None
        else None
    )
    outcomes = stay_outcome_columns(context)
    whole_stay = whole_stay_event_columns(context)
    windows: dict[str, ColumnWindow] = {}
    for variable in variables:
        name = str(getattr(variable, "name", "") or "").strip()
        if not name or name in windows:
            continue
        own = column_window_from_label(getattr(variable, "analysis_window", None))
        if own is not None:
            windows[name] = own
            continue
        source = str(getattr(variable, "source_concept", "") or "").strip()
        if (
            hours is None
            or name in whole_stay
            or column_kind(
                variable, column=name, concept=source or name, outcomes=outcomes
            )
            != "window_summary"
        ):
            continue
        windows[name] = host_column_window(0.0, hours)
    return windows


def host_column_window(start_hours: float, end_hours: float) -> ColumnWindow:
    """The window the host summarized a column over, from ICU admission."""

    return ColumnWindow(
        label=f"icu_admission[{start_hours:g},{end_hours:g}]h",
        anchor="icu_admission",
        start_hours=float(start_hours),
        end_hours=float(end_hours),
        source="host",
    )


def column_materialized_window(context: Any, column: str) -> ColumnWindow | None:
    """The window ``column`` of ``context`` was summarized over, if one is recorded."""

    return context_column_windows(context).get(str(column))


__all__ = [
    "ColumnWindow",
    "bound_feature_window_end_hours",
    "caller_declared_time_windows",
    "column_materialized_window",
    "column_window_from_label",
    "context_column_windows",
    "host_column_window",
    "host_materialization_window_hours",
    "window_anchor",
]
