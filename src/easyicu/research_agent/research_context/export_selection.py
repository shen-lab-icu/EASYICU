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

Criteria the context does not state are not therefore unapplied: a preset
such as ``adult_first`` applies its age bound and first-stay restriction
without a typed field, and a prepared package may carry any selection.  The
host records how it knows the export's selection in
``data_constraints.source_selection.basis``:

* ``export_contract``: the bound export's manifest states its cohort
  contract, so the stated criteria are its whole selection, each applied;
* ``package_declaration``: a prepared package declares itself the study's
  cohort; the host accepted the declaration but knows none of its criteria;
* ``unrecorded``: nothing records what the export selected.

Only ``export_contract`` makes the selection known (``recorded``).  Otherwise
what the export selected is unknown, not empty, and a stated criterion is
declared, not known to be applied.  The exception is a criterion the host
applies itself, whatever the export did (the first-stay restriction on
verified stay coordinates): the record lists it in ``host_applied``, verbatim
as the context states it.  A record written before the basis field states
``recorded`` instead; one whose basis is unknown is read as unrecorded.  A
context with no record at all (the CLI, a benchmark) has no basis.

Planning and reporting read the export's selection here, so both state the
same rows as already selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, get_args

from ..schema import ResearchContext
from .concept_population import (
    ConceptCohortWindow,
    concept_cohort_window,
    context_data_constraints,
)


#: ``data_constraints.cohort`` fields that hold the study's own words.
_STUDY_COHORT_WORDING_FIELDS = ("label", "review", "exclusion_statement")

#: How the host knows the export's selection
#: (``data_constraints.source_selection.basis``).
SelectionBasis = Literal["export_contract", "package_declaration", "unrecorded"]


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
    #: How the host knows the export's selection; ``None`` without a record.
    basis: SelectionBasis | None = None
    #: The contracts above that the host applied itself, whatever the export
    #: did; they are applied even when the selection is not recorded.
    host_applied: AppliedContracts = AppliedContracts()

    @property
    def recorded(self) -> bool:
        """Whether the export's contract records its whole selection.

        When it does not, the criteria above are what the context declares,
        not a complete account, and an empty selection means unknown.
        """

        return self.basis == "export_contract"

    @property
    def selects_rows(self) -> bool:
        """Whether the export applied any criterion or a concept population."""

        return bool(
            self.contracts.inclusion
            or self.contracts.exclusion
            or self.concept_population is not None
        )

    @property
    def known_applied(self) -> AppliedContracts:
        """The contracts known to be applied: all when recorded, else the host's."""

        return self.contracts if self.recorded else self.host_applied

    @property
    def unverified(self) -> AppliedContracts:
        """The contracts declared but not known to be applied."""

        known = self.known_applied
        return AppliedContracts(
            inclusion=tuple(item for item in self.contracts.inclusion if item not in known.inclusion),
            exclusion=tuple(item for item in self.contracts.exclusion if item not in known.exclusion),
        )


def export_applied_selection(context: ResearchContext) -> ExportAppliedSelection:
    """The selection the source export applied to the context's input rows.

    Raises :class:`~.concept_population.ConceptCohortWindowError` when the
    concept-population record cannot be read; a caller fails closed.
    """

    wording = _study_cohort_wording(context)
    contracts = AppliedContracts(
        inclusion=_contracts(context.cohort.inclusion_criteria, wording),
        exclusion=_contracts(context.cohort.exclusion_criteria, wording),
    )
    constraints = context_data_constraints(context)
    record = constraints.get("source_selection")
    return ExportAppliedSelection(
        contracts=contracts,
        concept_population=concept_cohort_window(context),
        basis=_selection_basis(record) if "source_selection" in constraints else None,
        host_applied=_host_applied(
            record.get("host_applied") if isinstance(record, Mapping) else None, contracts
        ),
    )


def _selection_basis(record: Any) -> SelectionBasis:
    """The record's basis; a record that names none known is unrecorded."""

    if not isinstance(record, Mapping):
        return "unrecorded"
    basis = record.get("basis")
    if basis in get_args(SelectionBasis):
        return basis
    if "basis" not in record and record.get("recorded") is True:
        # Written before the basis field (460825b34).
        return "export_contract"
    return "unrecorded"


def _host_applied(listed: Any, contracts: AppliedContracts) -> AppliedContracts:
    """The contracts the record marks as the host's own; other names are ignored."""

    if not isinstance(listed, Sequence) or isinstance(listed, str):
        return AppliedContracts()
    names = {value.strip() for value in listed if isinstance(value, str)}
    return AppliedContracts(
        inclusion=tuple(item for item in contracts.inclusion if item in names),
        exclusion=tuple(item for item in contracts.exclusion if item in names),
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
    "SelectionBasis",
    "export_applied_selection",
]
