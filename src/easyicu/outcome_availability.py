"""Dependency-neutral authority for database-specific outcome support.

It also owns the closed fixed-horizon mortality vocabulary: each ``mort_<h>d``
event concept is paired with its ``followup_days_<h>d`` event/censoring time,
measured in days from ICU admission and administratively censored at the
horizon.  The outcome materializer and the Web survival projection read the
pairing here so neither can drift from the other.
"""

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
class FixedHorizonMortalityEndpoint:
    """One ``mort_<h>d`` event concept with its paired follow-up time concept."""

    event_concept: str
    followup_concept: str
    horizon_days: int
    #: Time zero shared by the event and follow-up concepts.
    time_origin: str = "icu_admission"
    #: Physical unit of ``followup_concept``.
    followup_unit: str = "days"
    #: Piecewise-interval cutpoints (days) a time-varying hazard-ratio audit
    #: uses for this horizon; each lies strictly inside the horizon.
    time_varying_cutpoints_days: tuple[int, ...] = ()

    @property
    def censoring_rule(self) -> str:
        return (
            f"Observed death time or administrative censoring at {self.horizon_days} "
            "days; exclude rows without documented horizon support."
        )


FIXED_HORIZON_MORTALITY_ENDPOINTS: Mapping[str, FixedHorizonMortalityEndpoint] = (
    MappingProxyType(
        {
            "mort_28d": FixedHorizonMortalityEndpoint(
                event_concept="mort_28d",
                followup_concept="followup_days_28d",
                horizon_days=28,
                time_varying_cutpoints_days=(7, 14),
            ),
            "mort_90d": FixedHorizonMortalityEndpoint(
                event_concept="mort_90d",
                followup_concept="followup_days_90d",
                horizon_days=90,
                time_varying_cutpoints_days=(7, 14, 28),
            ),
            "mort_365d": FixedHorizonMortalityEndpoint(
                event_concept="mort_365d",
                followup_concept="followup_days_365d",
                horizon_days=365,
                time_varying_cutpoints_days=(7, 14, 28, 90),
            ),
        }
    )
)


def fixed_horizon_mortality_endpoint(
    concept_id: str,
) -> FixedHorizonMortalityEndpoint | None:
    """Return the closed event/follow-up pairing for one ``mort_<h>d`` concept."""

    return FIXED_HORIZON_MORTALITY_ENDPOINTS.get(str(concept_id).strip())


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
    "FIXED_HORIZON_MORTALITY_ENDPOINTS",
    "FOLLOWUP_OUTCOME_DATABASES",
    "FixedHorizonMortalityEndpoint",
    "ICU_FREE_DAY_DATABASES",
    "MIMIC_READMISSION_DATABASES",
    "OUTCOME_CONCEPT_SUPPORTED_DATABASES",
    "OutcomeConceptUnavailability",
    "fixed_horizon_mortality_endpoint",
    "structural_outcome_unavailability",
]
