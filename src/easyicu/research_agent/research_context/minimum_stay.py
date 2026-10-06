"""The typed minimum ICU stay a study's export applies.

``data_constraints.cohort.min_icu_los_hours`` is the hour after ICU admission
by which a stay must still be in the ICU to enter the export, so it is the
hour by which that criterion decides membership.  Family-spec planning and the
scientific review read it here, and both compare the same hour with the
plan's time zero.
"""

from __future__ import annotations

from typing import Mapping

from ..schema import ResearchContext
from .concept_population import context_data_constraints


def minimum_icu_stay_hours(context: ResearchContext) -> float | None:
    """The typed minimum ICU stay in hours, or ``None`` when none is positive."""

    cohort = context_data_constraints(context).get("cohort")
    value = cohort.get("min_icu_los_hours") if isinstance(cohort, Mapping) else None
    if value is None or isinstance(value, bool):
        return None
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return None
    return hours if hours > 0 else None


__all__ = ["minimum_icu_stay_hours"]
