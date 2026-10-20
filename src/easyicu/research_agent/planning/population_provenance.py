"""Record host attribution without rewriting a criterion's source or quote.

Overlapping words establish no restriction. Typed settings establish only
their recorded field values; they carry no recorded user-edit provenance.
Neither turns a system interpretation into the researcher's stated intent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .population_spec import PopulationCriterion, quote_written_in


@dataclass(frozen=True)
class PopulationProvenance:
    """Host observations, kept outside the Planner's criterion input schema."""

    quote_found_in_question: bool = False
    typed_study_setting: Mapping[str, float] | None = None


def _same_bound(actual: Any, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return (
        isinstance(actual, (int, float)) and not isinstance(actual, bool)
        and math.isfinite(actual) and actual == expected
    )


def _study_cohort(raw: str | None) -> Mapping[str, Any]:
    try:
        value = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return {}
    cohort = value.get("cohort") if isinstance(value, Mapping) else None
    return cohort if isinstance(cohort, Mapping) else {}


def typed_study_criterion_setting(
    criterion: PopulationCriterion, data_constraints: str | None,
) -> dict[str, float] | None:
    """Return field names and values only for an exactly matching restriction."""

    cohort = _study_cohort(data_constraints)
    if criterion.kind == "age_years":
        if (
            _same_bound(cohort.get("age_min"), criterion.min_years)
            and _same_bound(cohort.get("age_max"), criterion.max_years)
        ):
            return {
                f"cohort.{field}": cohort[field]
                for field in ("age_min", "age_max") if cohort.get(field) is not None
            }
    elif criterion.kind == "icu_stay_hours":
        if (
            _same_bound(cohort.get("min_icu_los_hours"), criterion.min_hours)
            and criterion.max_hours is None
        ):
            return {"cohort.min_icu_los_hours": cohort["min_icu_los_hours"]}
    return None


def bind_population_provenance(
    criterion: PopulationCriterion, *, question: str,
    data_constraints: str | None,
) -> PopulationProvenance:
    """Bind observations separately; never upgrade the Planner's attribution."""

    setting = typed_study_criterion_setting(criterion, data_constraints)
    return PopulationProvenance(
        quote_found_in_question=quote_written_in(criterion.quote, "question", [question]),
        typed_study_setting=MappingProxyType(setting) if setting is not None else None,
    )
