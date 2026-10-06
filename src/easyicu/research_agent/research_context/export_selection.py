"""The rows a run's source export selected, before any plan applies its own.

The context's inclusion and exclusion criteria are the contracts the export
applied: the Planner reads them so, and the Web caller declares there only
the typed fields Data Extraction executes (age range, minimum ICU stay,
diagnoses) and the host's first-stay restriction.  A context written before
that rule also filed the study's own cohort wording there: its label and
review as inclusion criteria, its exclusion statement as an exclusion.
Nothing executes prose.  The same words reach the Planner verbatim in
``data_constraints.cohort``, so a criterion that is one of them is the
study's wording, not a contract.  A concept-derived population is selected
by the export too (``concept_cohort_window``).

Planning and reporting read the export's selection here, so both state the
same rows as already selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..schema import ResearchContext
from .concept_population import (
    ConceptCohortWindow,
    concept_cohort_window,
    context_data_constraints,
)


#: ``data_constraints.cohort`` fields that hold the study's own words.
_STUDY_COHORT_WORDING_FIELDS = ("label", "review", "exclusion_statement")


@dataclass(frozen=True)
class AppliedContracts:
    """Criteria the source export applied, verbatim."""

    inclusion: tuple[str, ...] = ()
    exclusion: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExportAppliedSelection:
    """The source export's contracts and its concept-derived population."""

    contracts: AppliedContracts
    concept_population: ConceptCohortWindow | None

    @property
    def selects_rows(self) -> bool:
        """Whether the export applied any criterion or a concept population."""

        return bool(
            self.contracts.inclusion
            or self.contracts.exclusion
            or self.concept_population is not None
        )


def export_applied_selection(context: ResearchContext) -> ExportAppliedSelection:
    """The selection the source export applied to the context's input rows.

    Raises :class:`~.concept_population.ConceptCohortWindowError` when the
    concept-population record cannot be read; a caller fails closed.
    """

    wording = _study_cohort_wording(context)
    return ExportAppliedSelection(
        contracts=AppliedContracts(
            inclusion=_contracts(context.cohort.inclusion_criteria, wording),
            exclusion=_contracts(context.cohort.exclusion_criteria, wording),
        ),
        concept_population=concept_cohort_window(context),
    )


def _study_cohort_wording(context: ResearchContext) -> frozenset[str]:
    cohort = context_data_constraints(context).get("cohort")
    if not isinstance(cohort, Mapping):
        return frozenset()
    return frozenset(
        value.strip()
        for value in (cohort.get(field) for field in _STUDY_COHORT_WORDING_FIELDS)
        if isinstance(value, str) and value.strip()
    )


def _contracts(values: Sequence[Any], wording: frozenset[str]) -> tuple[str, ...]:
    criteria = (str(value).strip() for value in values)
    return tuple(dict.fromkeys(item for item in criteria if item and item not in wording))


__all__ = [
    "AppliedContracts",
    "ExportAppliedSelection",
    "export_applied_selection",
]
