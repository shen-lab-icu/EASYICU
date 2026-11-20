"""Record host attribution without rewriting a criterion's source or quote.

Overlapping words establish no restriction. Typed settings establish only
their recorded field values; they carry no recorded user-edit provenance.
Neither turns a system interpretation into the researcher's stated intent.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

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


#: Wording that names the hospitalization as the first-stay scope.
HOSPITAL_SCOPE = (
    r"每(?:一)?(?:次|个)?(?:住院|入院|住院期间)",
    r"(?:每|各)(?:次|个)?(?:医院)?住院",
    r"\b(?:per|each|every|within each|within every)\s+"
    r"(?:hospitali[sz]ation|hospital(?:\s+admission|\s+stay))s?\b",
)
#: Wording that names the patient as the first-stay scope.
PATIENT_SCOPE = (
    r"每(?:一)?(?:位|名|个)?(?:患者|病人)",
    r"(?:患者|病人)(?:各自|分别|每人)",
    r"每人",
    r"\b(?:each|every)\s+patient(?:'s)?\b",
    r"\bper\s+patient\b",
    r"\bpatients?'\s+first\b",
)


def names_scope(text: str, patterns: Sequence[str]) -> bool:
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return any(re.search(pattern, folded) for pattern in patterns)


def names_first_stay_scope(quote: str, kind: str) -> bool:
    """Whether ``quote`` itself names the scope of first-stay ``kind``.

    "first ICU stay" names neither; "each patient's first ICU stay" names the
    patient and "first ICU stay per hospitalization" the hospitalization.
    """

    if kind == "first_icu_stay_per_hospitalization":
        return names_scope(quote, HOSPITAL_SCOPE)
    if kind == "first_icu_stay":
        return names_scope(quote, PATIENT_SCOPE) and not names_scope(quote, HOSPITAL_SCOPE)
    return False


def _typed_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def typed_first_stay_scope(data_constraints: str | None) -> str | None:
    """The first-stay criterion kind the study's typed population setting keeps.

    ``first_icu_stay`` for each patient's first ICU stay
    (``cohort.exclude_readmissions``), ``first_icu_stay_per_hospitalization``
    for the first ICU stay within each hospitalization; ``None`` when the study
    types neither, or both (a conflicting pair its own owner refuses).
    """

    cohort = _study_cohort(data_constraints)
    patient = _typed_flag(cohort.get("exclude_readmissions"))
    hospitalization = _typed_flag(cohort.get("first_icu_stay_per_hospitalization"))
    if patient == hospitalization:
        return None
    return "first_icu_stay" if patient else "first_icu_stay_per_hospitalization"


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
