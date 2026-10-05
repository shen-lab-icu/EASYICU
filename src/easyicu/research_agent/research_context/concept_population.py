"""The concept-derived population an export of the study was selected by.

The host states it in ``data_constraints.concept_cohort_window`` from the
record the study's export carries: Data Extraction admits a stay on a positive
concept row (Sepsis-3, AKI, ventilation and the like) at or before
``window_end_hours`` after ICU admission.  The record's absence means the
population is not concept-derived.  A record that is present but cannot be
read is refused, so no owner mistakes a concept population for every stay.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from ..schema import ResearchContext


class ConceptCohortWindowError(ValueError):
    """``data_constraints.concept_cohort_window`` is present but unreadable."""


@dataclass(frozen=True)
class ConceptCohortWindow:
    definition: str
    window_end_hours: float


def context_data_constraints(context: ResearchContext) -> Mapping[str, Any]:
    """The host's JSON ``data_constraints`` object; empty when it states none."""

    raw = getattr(context.user_preferences, "data_constraints", None)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def concept_cohort_window(context: ResearchContext) -> ConceptCohortWindow | None:
    """The window the export's concept population was selected by, if any."""

    constraints = context_data_constraints(context)
    if "concept_cohort_window" not in constraints:
        return None
    window = constraints["concept_cohort_window"]
    definition = window.get("definition") if isinstance(window, Mapping) else None
    end = window.get("window_end_hours") if isinstance(window, Mapping) else None
    if (
        not isinstance(definition, str)
        or not definition.strip()
        or len(definition.strip()) > 64
        or isinstance(end, bool)
        or not isinstance(end, (int, float))
        or not math.isfinite(end)
        or end <= 0
    ):
        raise ConceptCohortWindowError(
            "data_constraints.concept_cohort_window must name the concept population and "
            "a positive window_end_hours"
        )
    return ConceptCohortWindow(definition=definition.strip(), window_end_hours=float(end))


__all__ = [
    "ConceptCohortWindow",
    "ConceptCohortWindowError",
    "concept_cohort_window",
    "context_data_constraints",
]
