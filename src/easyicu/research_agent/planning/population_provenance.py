"""Bind population wording to existing study facts without changing criteria.

The Planner's source label is a proposal. Verbatim study words, or an exact
match to the study's typed age/stay bounds, provide stronger attribution.
Typed settings are labeled study_wording, never claimed as the user's quote.
No condition is inferred from a label, and no bound or approval is changed.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .population_spec import PopulationCriterion, PopulationSpec, quote_written_in


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


def typed_study_criterion_quote(
    criterion: PopulationCriterion, data_constraints: str | None,
) -> str | None:
    """Return a study label only for an exactly matching typed restriction."""

    cohort = _study_cohort(data_constraints)
    label = next((text.strip() for key in ("label", "review")
                  if isinstance(text := cohort.get(key), str)
                  and 2 <= len(text.strip()) <= 160), None)
    exact = False
    if criterion.kind == "age_years":
        exact = (
            _same_bound(cohort.get("age_min"), criterion.min_years)
            and _same_bound(cohort.get("age_max"), criterion.max_years)
        )
    elif criterion.kind == "icu_stay_hours":
        exact = (
            _same_bound(cohort.get("min_icu_los_hours"), criterion.min_hours)
            and criterion.max_hours is None
        )
    return label if exact else None


def bind_population_provenance(
    spec: PopulationSpec, *, question: str, study_texts: Sequence[str],
    data_constraints: str | None,
) -> PopulationSpec:
    """Correct attribution only when the host has positive, bounded evidence."""

    criteria = []
    for criterion in spec.criteria:
        bound = criterion
        if criterion.source in {"outline", "preset"}:
            if quote_written_in(criterion.quote, "question", [question]):
                bound = criterion.model_copy(update={"source": "question"})
            elif quote_written_in(criterion.quote, "study_wording", study_texts):
                bound = criterion.model_copy(update={"source": "study_wording"})
            elif (label := typed_study_criterion_quote(criterion, data_constraints)) is not None:
                bound = criterion.model_copy(update={"source": "study_wording", "quote": label})
        criteria.append(bound)
    return spec.model_copy(update={"criteria": criteria})
