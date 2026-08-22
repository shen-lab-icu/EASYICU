"""Dependency-neutral authority for database-specific outcome support."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

FOLLOWUP_OUTCOME_DATABASES = frozenset(
    {"miiv", "miiv_demo", "mimic", "mimic_demo", "sic", "sic_demo", "aumc"}
)
# Reserved compatibility sets. They remain empty until an owner can prove
# complete ICU/ventilation trajectories and endpoint-specific day-28 survival.
MIMIC_READMISSION_DATABASES = frozenset()
ICU_FREE_DAY_DATABASES = frozenset()
EICU_VENTILATOR_DAY_DATABASES = frozenset()

OUTCOME_CONCEPT_SUPPORTED_DATABASES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "mort_28d": FOLLOWUP_OUTCOME_DATABASES,
        "mort_90d": FOLLOWUP_OUTCOME_DATABASES,
        "mort_365d": FOLLOWUP_OUTCOME_DATABASES,
        "followup_days_28d": FOLLOWUP_OUTCOME_DATABASES,
        "followup_days_90d": FOLLOWUP_OUTCOME_DATABASES,
        "followup_days_365d": FOLLOWUP_OUTCOME_DATABASES,
        "icu_free_days_28": ICU_FREE_DAY_DATABASES,
        "icu_readmission": MIMIC_READMISSION_DATABASES,
        "vent_free_days_28": EICU_VENTILATOR_DAY_DATABASES,
    }
)


@dataclass(frozen=True, slots=True)
class OutcomeConceptUnavailability:
    """A known structural absence, distinct from missing observed values."""

    concept_id: str
    database: str
    reason_code: str
    supported_databases: tuple[str, ...]


def structural_outcome_unavailability(
    concept_id: str,
    database: str,
) -> OutcomeConceptUnavailability | None:
    """Return a receipt only for a known unsupported concept/database pair."""

    concept = str(concept_id).strip()
    normalized_database = str(database).strip().lower()
    supported = OUTCOME_CONCEPT_SUPPORTED_DATABASES.get(concept)
    if supported is None or normalized_database in supported:
        return None
    return OutcomeConceptUnavailability(
        concept_id=concept,
        database=normalized_database,
        reason_code="outcome_concept_structurally_unavailable",
        supported_databases=tuple(sorted(supported)),
    )


__all__ = [
    "EICU_VENTILATOR_DAY_DATABASES",
    "FOLLOWUP_OUTCOME_DATABASES",
    "ICU_FREE_DAY_DATABASES",
    "MIMIC_READMISSION_DATABASES",
    "OUTCOME_CONCEPT_SUPPORTED_DATABASES",
    "OutcomeConceptUnavailability",
    "structural_outcome_unavailability",
]
