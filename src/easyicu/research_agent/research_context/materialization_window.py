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
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..icu_rules import default_time_windows
from ..schema import ResearchContext, TimeWindow
from .concept_population import context_data_constraints
from .temporal_semantics import TemporalAlignmentEngine, normalise_time_anchor


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


__all__ = [
    "bound_feature_window_end_hours",
    "caller_declared_time_windows",
    "host_materialization_window_hours",
]
