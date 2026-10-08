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

A recorded selection also says how many stays each criterion excluded
(``export_selection_counts``), from the source database to the analysis
input, as a flow diagram states it.  The launch copies the export's own count
report verbatim into ``source_selection.export_report``; this module alone
reads it.  The counts are ICU stays.  They are stated only when they chain:
each step leaves the stays the next one starts from, the export holds the
stays its last step left, the host's first-stay restriction starts from them,
and the analysis input holds the stays that remain.  Otherwise no count is
stated, and the reason is kept for audit, never shown to the Writer.  An
export that capped its stays records which ones its cap kept: the first by
identifier or in the order its source lists them, never a random sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence, get_args

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


#: The unit every selection count is in: Data Extraction counts the rows of the
#: source's demographics, one per ICU stay, and the host counts stays.
SELECTION_COUNT_UNIT = "icu_stay"

#: Why a context states no selection counts.  For audit only: the Writer is
#: never shown these codes.
SelectionCountsUnavailable = Literal[
    "selection_not_recorded",
    "export_reports_no_counts",
    "export_reports_no_source_total",
    "count_unit_unsupported",
    "counts_do_not_chain",
    "host_restriction_unreadable",
    "counts_do_not_reach_the_analysis_input",
]

#: How an export's cap chose the stays it kept (``cohort_report.cap.rule``):
#: the first by identifier compared as text, the first by identifier, the
#: first in the order the source's stay table lists them, a seeded random
#: sample, or an order the export did not record.
CapRule = Literal[
    "identifier_text_order",
    "identifier_order",
    "source_file_order",
    "seeded_random_sample",
    "unrecorded",
]


@dataclass(frozen=True)
class SelectionStep:
    """One selection criterion with the stays it excluded, in the order applied."""

    #: ``export`` for a criterion Data Extraction applied, ``host`` for the
    #: host's own first-stay restriction.
    stage: Literal["export", "host"]
    criterion: str
    parameters: Mapping[str, Any]
    n_before: int
    n_excluded: int
    n_remaining: int
    #: The excluded stays with no value for the criterion; ``None`` when the
    #: step does not count them.
    n_excluded_missing: Optional[int] = None

    def record(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "criterion": self.criterion,
            "parameters": dict(self.parameters),
            "n_before": self.n_before,
            "n_excluded": self.n_excluded,
            "n_remaining": self.n_remaining,
            "n_excluded_missing": self.n_excluded_missing,
        }


@dataclass(frozen=True)
class ExportCap:
    """How the export capped the stays that met its criteria."""

    max_patients: int
    rule: CapRule
    #: Whether the cap cut stays; ``None`` when the export could not tell.
    cut: Optional[bool]

    def record(self) -> dict[str, Any]:
        return {"max_patients": self.max_patients, "rule": self.rule, "cut": self.cut}


@dataclass(frozen=True)
class SelectionCounts:
    """The stays from the source database to the analysis input, step by step."""

    source_total: int
    steps: tuple[SelectionStep, ...]
    #: The stays the export held: what its last step left.
    exported: int
    unit: str = SELECTION_COUNT_UNIT

    @property
    def analysis_input(self) -> int:
        """The stays the analysis input holds: what the last step left."""

        return self.steps[-1].n_remaining if self.steps else self.source_total

    def record(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "source_total": self.source_total,
            "steps": [step.record() for step in self.steps],
            "exported": self.exported,
            "analysis_input": self.analysis_input,
        }


@dataclass(frozen=True)
class SelectionCountsReading:
    """A context's selection counts and export cap, as far as it records them."""

    counts: Optional[SelectionCounts]
    cap: Optional[ExportCap]
    #: Why ``counts`` is ``None``; for audit, never for the Writer.
    unavailable: Optional[SelectionCountsUnavailable]


class _Unchained(ValueError):
    """The export's counts do not chain from one step to the next."""


def export_selection_counts(context: ResearchContext) -> SelectionCountsReading:
    """The stays each criterion excluded, from the source to the analysis input.

    Only a recorded selection (``export_contract``) has counts, read from the
    export's report the launch copied.  A report written before the export
    counted each demographic criterion states them as one step.  Counts that
    do not chain, or that do not end at the stays the analysis input holds,
    give none.
    """

    record = context_data_constraints(context).get("source_selection")
    if not isinstance(record, Mapping) or _selection_basis(record) != "export_contract":
        return SelectionCountsReading(None, None, "selection_not_recorded")
    report = record.get("export_report")
    if not isinstance(report, Mapping):
        return SelectionCountsReading(None, None, "export_reports_no_counts")
    cap = _export_cap(report.get("cap"))
    if report.get("count_unit", SELECTION_COUNT_UNIT) != SELECTION_COUNT_UNIT:
        # A report without a unit was written by the same producer before it
        # named one; its counts were stays all along.
        return SelectionCountsReading(None, cap, "count_unit_unsupported")
    source_total = _count(report.get("source_total"))
    if source_total is None:
        return SelectionCountsReading(None, cap, "export_reports_no_source_total")
    try:
        steps = _export_steps(report, source_total=source_total, cap=cap)
    except _Unchained:
        return SelectionCountsReading(None, cap, "counts_do_not_chain")
    exported = steps[-1].n_remaining if steps else source_total
    restriction = context.cohort.provenance.get("first_icu_stay_restriction")
    if restriction is not None:
        host = _host_restriction_step(restriction)
        if host is None:
            return SelectionCountsReading(None, cap, "host_restriction_unreadable")
        if host.n_before != exported:
            return SelectionCountsReading(
                None, cap, "counts_do_not_reach_the_analysis_input"
            )
        # Kept when it removed no stay: the restriction still applied.
        steps.append(host)
    counts = SelectionCounts(
        source_total=source_total, steps=tuple(steps), exported=exported
    )
    if counts.analysis_input != context.cohort.n_stays:
        return SelectionCountsReading(
            None, cap, "counts_do_not_reach_the_analysis_input"
        )
    return SelectionCountsReading(counts, cap, None)


def _count(value: Any) -> Optional[int]:
    """A stay count: a non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _export_cap(raw: Any) -> Optional[ExportCap]:
    if not isinstance(raw, Mapping):
        return None
    max_patients = _count(raw.get("max_patients"))
    if not max_patients:
        return None
    rule = raw.get("rule")
    cut = raw.get("cut")
    return ExportCap(
        max_patients=max_patients,
        rule=rule if rule in get_args(CapRule) else "unrecorded",
        cut=cut if isinstance(cut, bool) else None,
    )


def _step(
    stage: Literal["export", "host"],
    criterion: str,
    parameters: Mapping[str, Any],
    *,
    before: int,
    after: Optional[int],
    missing: Optional[int] = None,
) -> SelectionStep:
    """One step from the stays before it to those after; a gain does not chain."""

    if (
        after is None
        or after > before
        or (missing is not None and missing > before - after)
    ):
        raise _Unchained(criterion)
    return SelectionStep(
        stage=stage,
        criterion=criterion,
        parameters=dict(parameters),
        n_before=before,
        n_excluded=before - after,
        n_remaining=after,
        n_excluded_missing=missing,
    )


def _export_steps(
    report: Mapping[str, Any], *, source_total: int, cap: Optional[ExportCap]
) -> list[SelectionStep]:
    """The export's steps, in the order Data Extraction applies them."""

    steps: list[SelectionStep] = []
    remaining = source_total
    demographic = report.get("demographic_steps")
    if demographic is not None:
        if not isinstance(demographic, list):
            raise _Unchained("demographic_steps")
        for raw in demographic:
            criterion = raw.get("criterion") if isinstance(raw, Mapping) else None
            if not isinstance(criterion, str):
                raise _Unchained("demographic_steps")
            parameters = raw.get("parameters")
            step = _step(
                "export",
                criterion,
                parameters if isinstance(parameters, Mapping) else {},
                before=remaining,
                after=_count(raw.get("n_remaining")),
                missing=_count(raw.get("n_excluded_missing")),
            )
            if (
                _count(raw.get("n_before")) != remaining
                or _count(raw.get("n_excluded")) != step.n_excluded
            ):
                raise _Unchained(criterion)
            steps.append(step)
            remaining = step.n_remaining
    after_demographics = _count(report.get("selected_before_concept_prefilter"))
    if after_demographics is not None:
        if demographic is None and after_demographics != remaining:
            # Written before the export counted each demographic criterion:
            # they stand as one step.
            steps.append(
                _step(
                    "export",
                    "demographics",
                    {},
                    before=remaining,
                    after=after_demographics,
                )
            )
            remaining = after_demographics
        if after_demographics != remaining:
            # The criteria count rows; the selection counts distinct stays.
            raise _Unchained("selected_before_concept_prefilter")
    if report.get("concept_matches") is not None:
        definition = report.get("mode")
        steps.append(
            _step(
                "export",
                "concept_population",
                {"definition": definition} if isinstance(definition, str) else {},
                before=remaining,
                after=_count(report.get("concept_matches")),
            )
        )
        remaining = steps[-1].n_remaining
    icd = report.get("icd")
    if isinstance(icd, Mapping) and icd.get("enabled") is True:
        if _count(report.get("selected_before_icd")) != remaining:
            raise _Unchained("icd")
        steps.append(
            _step(
                "export",
                "icd",
                {
                    "include": list(icd.get("include_tokens") or []),
                    "exclude": list(icd.get("exclude_tokens") or []),
                },
                before=remaining,
                after=_count(report.get("selected_before_cap")),
            )
        )
        remaining = steps[-1].n_remaining
    before_cap = _count(report.get("selected_before_cap"))
    if before_cap is not None and before_cap != remaining:
        raise _Unchained("selected_before_cap")
    selected = _count(report.get("selected"))
    if selected is None:
        raise _Unchained("selected")
    if selected != remaining:
        # Only a cap removes stays after the criteria.
        if cap is None and report.get("max_patients_applied") is not True:
            raise _Unchained("selected")
        steps.append(
            _step(
                "export",
                "cap",
                cap.record() if cap is not None else {"rule": "unrecorded"},
                before=remaining,
                after=selected,
            )
        )
    return steps


def _host_restriction_step(raw: Any) -> Optional[SelectionStep]:
    """The host's first-stay restriction as recorded in the cohort's provenance."""

    if not isinstance(raw, Mapping):
        return None
    before = _count(raw.get("stays_before"))
    after = _count(raw.get("stays_after"))
    removed = _count(raw.get("non_first_icu_stays_removed"))
    if before is None or after is None or removed is None or before - after != removed:
        return None
    return SelectionStep(
        stage="host",
        criterion="first_icu_stay_restriction",
        parameters={},
        n_before=before,
        n_excluded=removed,
        n_remaining=after,
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
    "CapRule",
    "ExportAppliedSelection",
    "ExportCap",
    "SELECTION_COUNT_UNIT",
    "SelectionBasis",
    "SelectionCounts",
    "SelectionCountsReading",
    "SelectionCountsUnavailable",
    "SelectionStep",
    "export_applied_selection",
    "export_selection_counts",
]
