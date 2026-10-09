"""How each criterion of a study's population spec reaches the analysis rows.

The Planner states the population as typed criteria
(:mod:`.population_spec`); this module decides, for each one, how it is
applied, and builds the cohort the plan applies from those decisions.  Each
criterion gets exactly one disposition:

* ``applied_by_source``: a typed record shows the input rows already meet it
  (the export's own count report, the study contract Data Extraction
  executed, or the host's first-stay receipt);
* ``applied_by_plan``: predicates over the input's columns apply it;
* ``requires_extraction``: the input does not hold what applies it, and an
  extraction could (a concept Data Extraction defines, diagnosis codes, the
  first stay);
* ``not_applied``: nothing can apply it as stated, with a reason code.

The judgements are the owners', not this module's: which column a predicate
filters (``cohort_eligibility.predicate_context_column``), whether its window
and its event time can be read (``cohort.schema``), whether its threshold
separates rows (``cohort_predicate_domain``), whether it is decided by time
zero (``cohort_eligibility``), which values a column takes
(``authority.declared_levels``), and which concepts an extraction defines
(``concept_availability``).  This module maps their findings to reason
codes.  A source proof is a typed record whose parameters imply the
criterion; the verbatim contract strings are never parsed.  A criterion the
source applied is not judged against time zero here: that check stays with
the export's own eligibility owner (``eligibility_after_time_zero``).

Compiling reads no patient row, calls no model and writes no file.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Sequence

from ..authority.declared_levels import closed_planning_levels_for
from ..cohort.schema import (
    context_materialized_columns,
    predicates_read_by_an_event_time_not_in_hours,
    predicates_read_over_the_whole_stay,
    predicates_read_through_another_window,
)
from ..intake.materialized_metadata import FIRST_ICU_STAY_RESTRICTION_SCHEMA
from ..research_context.concept_population import (
    ConceptCohortWindowError,
    context_data_constraints,
)
from ..research_context.export_selection import (
    SelectionStep,
    export_applied_selection,
    export_selection_counts,
)
from ..research_context.materialization_window import (
    ColumnWindow,
    context_column_windows,
)
from ..research_context.stay_events import (
    ICU_LENGTH_OF_STAY_CONCEPT,
    column_kind,
    event_times_typed_otherwise_than_hours,
    stay_outcome_columns,
    whole_stay_event_columns,
)
from ..concept_availability import explain_concept_availability
from ..schema import ResearchContext
from .cohort_contract import (
    CohortDefinition,
    CohortSchemaError,
    ConceptPredicate,
    TimeWindow,
    cohort_concept_id_scope,
    sealed_cohort_concept_ids,
    validate_cohort_definition,
)
from .cohort_eligibility import (
    cohort_predicates_after_time_zero,
    predicate_context_column,
)
from .cohort_identity import cohort_identity_columns
from .cohort_predicate_domain import cohort_predicates_outside_column_domain
from .population_spec import (
    POPULATION_SPEC_SCHEMA_VERSION,
    AgeYears,
    AliveAt,
    ConditionPresent,
    DiagnosisCodes,
    EventAbsent,
    FirstIcuStay,
    IcuStayHours,
    Measurement,
    NotTyped,
    PopulationCriterion,
    PopulationSpec,
    SpecWindow,
    diagnosis_code_token,
)

POPULATION_COMPILE_SCHEMA_VERSION = "easyicu.population_compile/1"

Disposition = Literal[
    "applied_by_source", "applied_by_plan", "requires_extraction", "not_applied"
]
ProofKind = Literal[
    "export_report_step", "recorded_study_contract", "host_first_icu_stay"
]
Side = Literal["inclusion", "exclusion"]

#: Why a criterion is not applied.  Stable: a published code never changes.
NOT_APPLIED_REASONS = (
    "population_kind_not_typed",
    "population_concept_unavailable",
    "population_identifier_column",
    "population_column_unresolved",
    "population_condition_column_not_status",
    "population_unit_mismatch",
    "population_unit_unrecorded",
    "population_window_unreadable",
    "population_whole_stay_in_finite_window",
    "population_event_time_not_hours",
    "population_threshold_outside_domain",
    "population_determined_after_time_zero",
)
#: Why a criterion waits for an extraction that would hold what applies it.
REQUIRES_EXTRACTION_REASONS = (
    "population_concept_not_in_export",
    "population_diagnosis_codes_need_extraction",
    "population_first_icu_stay_not_restricted",
)
#: Why a plan cannot be approved while an inclusion is ``blocking``, one code
#: per remedy (population spec design 3.3 and 3.5): an extraction of the
#: study's own population, or a decision the study itself must make.
#: Stable: a published code never changes.
POPULATION_APPROVAL_STOPS: Mapping[str, str] = MappingProxyType(
    {
        "requires_extraction": "population_inclusion_requires_extraction",
        "not_applied": "population_inclusion_not_applied",
    }
)

#: Where every population predicate counts from.
_ANCHOR = "icu_admission"
#: The concepts the fixed kinds read.
_AGE_CONCEPT = "age"
_DEATH_CONCEPT = "death"
_HOUR_UNITS = frozenset({"h", "hr", "hrs", "hour", "hours"})
_DAY_UNITS = frozenset({"d", "day", "days"})
_YEAR_UNITS = frozenset({"y", "yr", "yrs", "year", "years"})
#: Column kinds that hold one value per stay, whatever summary is named.
_ONE_VALUE_KINDS = frozenset(
    {"admission", "stay_level", "icu_stay_length", "stay_outcome"}
)
_ONE_VALUE_TRANSFORM = "stay_level_unique_value"
#: Typed representations that hold one summary of a concept over a window:
#: a numeric summary, and a 0/1 status's presence.
_SUMMARY_TRANSFORMS = (
    "window_numeric_{summary}",
    "window_presence_{summary}",
    "fixed_window_presence_{summary}",
)
#: First characters that begin both ICD-9 and ICD-10 codes.
_BOTH_VERSIONS = frozenset({"E", "V"})
#: The export's recorded study contract (``source_selection.executed_cohort``).
_EXECUTED_COHORT = "executed_cohort"


@dataclass(frozen=True)
class SourceProof:
    """The typed record that shows the input rows already meet a criterion."""

    kind: ProofKind
    #: Where the record is: ``source_selection.export_report`` and the like.
    record_ref: str
    #: What the record states, as it states it.
    parameters: Mapping[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "record_ref": self.record_ref,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class CompiledCriterion:
    """One criterion of the spec with its single disposition."""

    criterion: PopulationCriterion
    disposition: Disposition
    #: A code from ``NOT_APPLIED_REASONS`` or ``REQUIRES_EXTRACTION_REASONS``;
    #: ``None`` when the criterion is applied.
    reason: Optional[str]
    detail: str
    #: The predicates that apply it, all on ``side``; only ``applied_by_plan``.
    predicates: tuple[ConceptPredicate, ...] = ()
    side: Optional[Side] = None
    proof: Optional[SourceProof] = None

    def __post_init__(self) -> None:
        applied_by_plan = self.disposition == "applied_by_plan"
        if applied_by_plan != bool(self.predicates) or applied_by_plan != (
            self.side is not None
        ):
            raise ValueError(
                "only a criterion applied by the plan has predicates and a side"
            )
        if (self.disposition == "applied_by_source") != (self.proof is not None):
            raise ValueError(
                "only a criterion applied by the source has a source proof"
            )
        allowed = {
            "not_applied": NOT_APPLIED_REASONS,
            "requires_extraction": REQUIRES_EXTRACTION_REASONS,
        }.get(self.disposition, (None,))
        if self.reason not in allowed:
            raise ValueError(f"reason {self.reason!r} does not fit {self.disposition}")

    @property
    def applied(self) -> bool:
        return self.disposition in {"applied_by_source", "applied_by_plan"}

    def record(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion.model_dump(mode="json"),
            "disposition": self.disposition,
            "reason": self.reason,
            "detail": self.detail,
            "side": self.side,
            "predicates": [predicate.to_dict() for predicate in self.predicates],
            "proof": self.proof.record() if self.proof is not None else None,
        }


@dataclass(frozen=True)
class CompiledPopulation:
    """Every criterion's disposition and the cohort the plan applies."""

    criteria: tuple[CompiledCriterion, ...]
    time_zero_hours: Optional[float] = None

    @property
    def blocking(self) -> tuple[CompiledCriterion, ...]:
        """Inclusions the analysis would not apply: approval waits for the user."""

        return tuple(
            item
            for item in self.criteria
            if item.criterion.role == "include" and not item.applied
        )

    def cohort_definition(self, name: str = "primary") -> CohortDefinition:
        """The cohort the plan applies; criteria it does not apply are listed as such."""

        inclusion = tuple(
            predicate
            for item in self.criteria
            if item.side == "inclusion"
            for predicate in item.predicates
        )
        exclusion = tuple(
            predicate
            for item in self.criteria
            if item.side == "exclusion"
            for predicate in item.predicates
        )
        return CohortDefinition(
            name=name,
            inclusion=inclusion,
            exclusion=exclusion,
            selection_mode=(
                "predicate_filtered" if inclusion or exclusion else "all_input_rows"
            ),
            # Criteria may share the sentence that states them.
            unapplied_population_criteria=tuple(
                dict.fromkeys(
                    " ".join(item.criterion.quote.split())
                    for item in self.criteria
                    if not item.applied
                )
            ),
        )

    def record(self) -> dict[str, Any]:
        return {
            "schema_version": POPULATION_COMPILE_SCHEMA_VERSION,
            "spec_schema_version": POPULATION_SPEC_SCHEMA_VERSION,
            "time_zero_hours": self.time_zero_hours,
            "criteria": [item.record() for item in self.criteria],
            "cohort": self.cohort_definition().plan_dict(),
        }

    def sha256(self) -> str:
        raw = json.dumps(
            self.record(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def population_spec_concepts(spec: PopulationSpec) -> tuple[str, ...]:
    """The concepts compiling ``spec`` reads, in the order its criteria state them.

    An extraction for the study holds them, so each criterion the plan can
    apply is compiled over its data: the age, the ICU stay, a condition's or
    an event's status, a measurement and the death the kinds read.  Diagnosis
    codes and a first ICU stay are applied by the export or the host, over no
    column.
    """

    concepts: list[str] = []
    for criterion in spec.criteria:
        if isinstance(criterion, AgeYears):
            concepts.append(_AGE_CONCEPT)
        elif isinstance(criterion, IcuStayHours):
            concepts.append(ICU_LENGTH_OF_STAY_CONCEPT)
        elif isinstance(criterion, ConditionPresent):
            concepts.extend(criterion.concepts_all_of)
        elif isinstance(criterion, (Measurement, EventAbsent)):
            concepts.append(criterion.concept)
        elif isinstance(criterion, AliveAt):
            concepts.append(_DEATH_CONCEPT)
    return tuple(dict.fromkeys(concepts))


def compile_population(
    spec: PopulationSpec,
    context: ResearchContext,
    *,
    time_zero_hours: Optional[float] = None,
) -> CompiledPopulation:
    """Decide how each criterion of ``spec`` reaches the rows of ``context``.

    ``time_zero_hours`` is the plan's time zero in hours after ICU admission;
    without one no predicate is held to it.
    """

    if time_zero_hours is not None and (
        isinstance(time_zero_hours, bool) or not math.isfinite(float(time_zero_hours))
    ):
        raise ValueError("time_zero_hours must be a finite number of hours")
    reading = _read_input(
        context,
        time_zero_hours=None if time_zero_hours is None else float(time_zero_hours),
    )
    with cohort_concept_id_scope(sealed_cohort_concept_ids(context)):
        population = CompiledPopulation(
            criteria=tuple(_compile(item, reading) for item in spec.criteria),
            time_zero_hours=reading.time_zero_hours,
        )
        validate_cohort_definition(population.cohort_definition())
    return population


#: The summary that names a value recorded once per stay, read over the window
#: its column states.
ONE_VALUE_SUMMARY = "value"


@dataclass(frozen=True)
class ThresholdReading:
    """How this owner reads one threshold on one concept's summary of the input.

    ``column`` is the column a predicate would filter.  ``reason`` is the code
    of the first finding against reading the threshold, one of this module's
    population codes, with ``detail`` saying why.
    """

    column: Optional[str]
    reason: Optional[str] = None
    detail: str = ""

    @property
    def requires_extraction(self) -> bool:
        """An extraction would hold what reads the threshold."""

        return self.reason in REQUIRES_EXTRACTION_REASONS


def read_threshold(
    context: ResearchContext,
    *,
    concept: str,
    summary: str,
    window: Optional[SpecWindow],
    op: str,
    value: float,
    unit: Optional[str],
) -> ThresholdReading:
    """Whether ``summary`` of ``concept`` over ``window`` can be held to a threshold.

    The column, the summary it records, the window it summarizes and the unit
    are judged as a population measurement's are.  A value recorded once per
    stay (:data:`ONE_VALUE_SUMMARY`, no window) is read over the window its
    column states.  No time zero is judged here.
    """

    reading = _read_input(context, time_zero_hours=None)
    with cohort_concept_id_scope(sealed_cohort_concept_ids(context)):
        try:
            if summary == ONE_VALUE_SUMMARY:
                column = _resolve(reading, concept, "first")
                if not _one_value_per_stay(reading, column, concept):
                    raise _NotApplied(
                        "population_column_unresolved",
                        f"{column!r} does not record one value of {concept!r} "
                        "per stay.",
                    )
                hours = _stay_value_window(reading, column)
                predicate = _threshold_predicate(
                    reading, concept, "first", hours, op=op, value=value, unit=unit
                )
            elif window is None:
                raise ValueError("a windowed summary is read over a stated window")
            else:
                predicate = _threshold_predicate(
                    reading,
                    concept,
                    summary,
                    (window.start_hours, window.end_hours),
                    op=op,
                    value=value,
                    unit=unit,
                )
            _judge(reading, predicate, "inclusion", label=f"{concept} ({summary})")
        except _NotApplied as found:
            return ThresholdReading(
                column=None, reason=found.reason, detail=found.detail
            )
        return ThresholdReading(column=_column_of(predicate, reading))


def _one_value_per_stay(reading: _Input, column: str, concept: str) -> bool:
    variable = reading.variables[column]
    if str(getattr(variable, "unit_normalization", None) or "") == _ONE_VALUE_TRANSFORM:
        return True
    source = str(getattr(variable, "source_concept", None) or "").strip()
    kind = column_kind(
        variable, column=column, concept=source or concept, outcomes=reading.outcomes
    )
    return kind in _ONE_VALUE_KINDS


# -- what the input holds -----------------------------------------------------


@dataclass(frozen=True)
class _Sources:
    """The typed records of what selected the input rows before analysis."""

    #: The export's and the host's steps, when the counts chain.
    steps: tuple[SelectionStep, ...] = ()
    #: The study contract Data Extraction executed, when the selection is recorded.
    executed: Optional[Mapping[str, Any]] = None
    #: The host's receipt for restricting the input to first ICU stays.
    first_stay_receipt: bool = False


@dataclass(frozen=True)
class _Input:
    context: ResearchContext
    variables: Mapping[str, Any]
    columns: tuple[str, ...]
    roster: frozenset[str]
    identity: frozenset[str]
    column_windows: Mapping[str, ColumnWindow]
    whole_stay: frozenset[str]
    times_not_in_hours: Mapping[str, Any]
    outcomes: frozenset[str]
    sources: _Sources
    time_zero_hours: Optional[float]


def _read_input(
    context: ResearchContext, *, time_zero_hours: Optional[float]
) -> _Input:
    return _Input(
        context=context,
        variables={str(variable.name): variable for variable in context.variables},
        columns=tuple(context_materialized_columns(context)),
        roster=frozenset(sealed_cohort_concept_ids(context)),
        identity=cohort_identity_columns(context),
        column_windows=context_column_windows(context),
        whole_stay=whole_stay_event_columns(context),
        times_not_in_hours=event_times_typed_otherwise_than_hours(context),
        outcomes=stay_outcome_columns(context),
        sources=_read_sources(context),
        time_zero_hours=time_zero_hours,
    )


def _read_sources(context: ResearchContext) -> _Sources:
    counts = export_selection_counts(context).counts
    try:
        recorded = export_applied_selection(context).recorded
    except ConceptCohortWindowError:
        # An unreadable record proves nothing.
        recorded = False
    record = context_data_constraints(context).get("source_selection")
    executed = record.get(_EXECUTED_COHORT) if isinstance(record, Mapping) else None
    receipt = context.cohort.provenance.get("first_icu_stay_restriction")
    return _Sources(
        steps=counts.steps if counts is not None else (),
        executed=executed if recorded and isinstance(executed, Mapping) else None,
        first_stay_receipt=(
            isinstance(receipt, Mapping)
            and receipt.get("schema_version") == FIRST_ICU_STAY_RESTRICTION_SCHEMA
        ),
    )


# -- one criterion ------------------------------------------------------------


class _NotApplied(Exception):
    """A criterion stops at the first owner finding against it."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _compile(criterion: PopulationCriterion, reading: _Input) -> CompiledCriterion:
    try:
        if isinstance(criterion, NotTyped):
            raise _NotApplied(
                "population_kind_not_typed",
                f"No population kind expresses it: {criterion.why}",
            )
        proof = _source_proof(criterion, reading.sources)
        if proof is not None:
            return CompiledCriterion(
                criterion=criterion,
                disposition="applied_by_source",
                reason=None,
                detail=f"The input rows already meet it ({_proof_text(proof)}).",
                proof=proof,
            )
        if isinstance(criterion, DiagnosisCodes):
            raise _NotApplied(
                "population_diagnosis_codes_need_extraction",
                "Only an export applies diagnosis codes, and no record of this input "
                "shows an export that applied exactly these codes.",
            )
        if isinstance(criterion, FirstIcuStay):
            raise _NotApplied(
                "population_first_icu_stay_not_restricted",
                "Neither the host nor the export records restricting this input to "
                "each patient's first ICU stay.",
            )
        side, predicates, note = _plan_predicates(criterion, reading)
    except _NotApplied as found:
        return CompiledCriterion(
            criterion=criterion,
            disposition=(
                "requires_extraction"
                if found.reason in REQUIRES_EXTRACTION_REASONS
                else "not_applied"
            ),
            reason=found.reason,
            detail=found.detail,
        )
    columns = sorted({_column_of(predicate, reading) for predicate in predicates})
    return CompiledCriterion(
        criterion=criterion,
        disposition="applied_by_plan",
        reason=None,
        detail=(
            f"{len(predicates)} {side} predicate{'s' if len(predicates) > 1 else ''} "
            f"over {', '.join(repr(column) for column in columns)}." + note
        ),
        predicates=predicates,
        side=side,
    )


def _plan_predicates(
    criterion: PopulationCriterion, reading: _Input
) -> tuple[Side, tuple[ConceptPredicate, ...], str]:
    """The predicates that apply ``criterion``, each judged by the owners."""

    if isinstance(criterion, AgeYears):
        column = _resolve(reading, _AGE_CONCEPT, "first")
        _require_unit_in(reading, column, _YEAR_UNITS, missing_ok=True, of="age")
        start, end = _stay_value_window(reading, column)
        predicates = _bounded(
            _AGE_CONCEPT,
            criterion.min_years,
            criterion.max_years,
            start,
            end,
            scale=1.0,
        )
        side: Side = "inclusion"
    elif isinstance(criterion, IcuStayHours):
        column = _resolve(reading, ICU_LENGTH_OF_STAY_CONCEPT, "first")
        hours_per_unit = _stay_length_hours_per_unit(reading, column)
        start, end = _stay_value_window(reading, column)
        predicates = _bounded(
            ICU_LENGTH_OF_STAY_CONCEPT,
            criterion.min_hours,
            criterion.max_hours,
            start,
            end,
            scale=1.0 / hours_per_unit,
        )
        side = "inclusion"
    elif isinstance(criterion, ConditionPresent):
        side = "inclusion" if criterion.role == "include" else "exclusion"
        predicates = tuple(
            _status_predicate(reading, concept, _hours(criterion.window))
            for concept in criterion.concepts_all_of
        )
    elif isinstance(criterion, Measurement):
        side = "inclusion" if criterion.role == "include" else "exclusion"
        predicates = (_measurement_predicate(reading, criterion),)
    elif isinstance(criterion, AliveAt):
        # The excluded condition is a death the input records before the hour.
        side = "exclusion"
        predicates = (
            _status_predicate(reading, _DEATH_CONCEPT, (0.0, criterion.hours)),
        )
    elif isinstance(criterion, EventAbsent):
        # The excluded condition is the event: the owners read it as they
        # read any condition, by its own time or over its column's window.
        side = "exclusion"
        predicates = (
            _status_predicate(reading, criterion.concept, _hours(criterion.window)),
        )
    else:  # pragma: no cover - the union is closed
        raise TypeError(f"unknown population criterion {type(criterion).__name__}")
    note = ""
    for predicate in predicates:
        note = (
            _judge(reading, predicate, side, label=f"population.{criterion.id}") or note
        )
    if isinstance(criterion, AliveAt):
        # A death the input does not record (after discharge, where it records
        # in-hospital deaths) removes no stay.
        note += (
            f" It keeps the stays with no death this input records before "
            f"{criterion.hours:g} h after ICU admission."
        )
    return side, predicates, note


def _resolve(reading: _Input, concept: str, aggregation: str) -> str:
    """The column the cohort builder would filter for ``concept``."""

    if concept not in reading.roster:
        if _extraction_defines(concept, reading.context.cohort.database):
            raise _NotApplied(
                "population_concept_not_in_export",
                f"This input does not hold {concept!r}; an extraction that holds it "
                "would let a predicate apply the criterion.",
            )
        raise _NotApplied(
            "population_concept_unavailable",
            f"{concept!r} is neither a column of this input nor a concept Data "
            "Extraction defines.",
        )
    column = predicate_context_column(reading.variables, concept, aggregation)
    if column not in reading.variables:
        raise _NotApplied(
            "population_column_unresolved",
            f"The cohort builder binds no column of this input to {concept!r} "
            f"({aggregation}).",
        )
    if column in reading.identity:
        raise _NotApplied(
            "population_identifier_column",
            f"{column!r} identifies the input rows: every row has one, so a "
            "predicate over it states no population.",
        )
    return column


def _status_predicate(
    reading: _Input, concept: str, hours: tuple[float, float]
) -> ConceptPredicate:
    """``concept`` present within ``hours``: its 0/1 status equals 1.

    Whether that window can be read is the owners' judgement (``_judge``): a
    status the input records over the whole stay is read by its event's own
    time, else over the window it was summarized over.
    """

    column = _resolve(reading, concept, "max")
    levels = closed_planning_levels_for(name=column, variables=reading.variables)
    if len(levels) < 2 or not all(_is_status_level(level) for level in levels):
        raise _NotApplied(
            "population_condition_column_not_status",
            f"{column!r} declares no 0/1 values, so whether {concept!r} is present "
            "cannot be read from it.",
        )
    start, end = hours
    return _predicate(concept, start, end, aggregation="max", op="==", value=1)


def _hours(window: Optional[SpecWindow]) -> tuple[float, float]:
    """A stated window in hours after ICU admission; none is the whole stay."""

    if window is None:
        return 0.0, math.inf
    return window.start_hours, window.end_hours


def _measurement_predicate(reading: _Input, criterion: Measurement) -> ConceptPredicate:
    return _threshold_predicate(
        reading,
        criterion.concept,
        criterion.summary,
        (criterion.window.start_hours, criterion.window.end_hours),
        op=criterion.op,
        value=criterion.value,
        unit=criterion.unit,
    )


def _threshold_predicate(
    reading: _Input,
    concept: str,
    summary: str,
    hours: tuple[float, float],
    *,
    op: str,
    value: float,
    unit: Optional[str],
) -> ConceptPredicate:
    """``summary`` of ``concept`` over ``hours`` compared with a threshold."""

    column = _resolve(reading, concept, summary)
    variable = reading.variables[column]
    if not _holds_summary(reading, variable, column, concept, summary):
        raise _NotApplied(
            "population_column_unresolved",
            f"The cohort builder would filter {column!r}, which this input does not "
            f"record as the {summary} of {concept!r}.",
        )
    if unit is not None:
        recorded = _unit_key(getattr(variable, "unit", None))
        if not recorded:
            raise _NotApplied(
                "population_unit_unrecorded",
                f"The threshold is in {unit!r}, but this input does not "
                f"record the unit of {column!r}.",
            )
        if recorded != _unit_key(unit):
            raise _NotApplied(
                "population_unit_mismatch",
                f"The threshold is in {unit!r}, but this input records "
                f"{column!r} in {getattr(variable, 'unit', None)!r}.",
            )
    start, end = hours
    return _predicate(concept, start, end, aggregation=summary, op=op, value=value)


def _holds_summary(
    reading: _Input, variable: Any, column: str, concept: str, summary: str
) -> bool:
    """Whether ``column`` holds ``summary`` of ``concept``, as the context types it.

    Its name or its typed representation says so, or it holds one value per
    stay.  An untyped 0/1 status holds its presence over its window, which is
    its maximum, as a condition reads it.
    """

    transform = str(getattr(variable, "unit_normalization", None) or "")
    if column == f"{concept}_{summary}":
        return True
    if transform in {
        template.format(summary=summary) for template in _SUMMARY_TRANSFORMS
    }:
        return True
    if transform == _ONE_VALUE_TRANSFORM:
        return True
    source = str(getattr(variable, "source_concept", None) or "").strip()
    kind = column_kind(
        variable, column=column, concept=source or concept, outcomes=reading.outcomes
    )
    if kind in _ONE_VALUE_KINDS:
        return True
    levels = closed_planning_levels_for(name=column, variables=reading.variables)
    return (
        summary == "max"
        and not transform
        and len(levels) >= 2
        and all(_is_status_level(level) for level in levels)
    )


def _require_unit_in(
    reading: _Input, column: str, units: frozenset[str], *, missing_ok: bool, of: str
) -> None:
    unit = getattr(reading.variables[column], "unit", None)
    key = _unit_key(unit)
    if not key and missing_ok:
        return
    if key not in units:
        raise _NotApplied(
            "population_unit_mismatch" if key else "population_unit_unrecorded",
            f"This input records {of} ({column!r}) in {unit!r}.",
        )


def _stay_length_hours_per_unit(reading: _Input, column: str) -> float:
    unit = getattr(reading.variables[column], "unit", None)
    key = _unit_key(unit)
    if key in _HOUR_UNITS:
        return 1.0
    if key in _DAY_UNITS:
        return 24.0
    raise _NotApplied(
        "population_unit_mismatch" if key else "population_unit_unrecorded",
        f"This input records the ICU length of stay ({column!r}) in {unit!r}, "
        "neither hours nor days.",
    )


def _stay_value_window(reading: _Input, column: str) -> tuple[float, float]:
    """The window a one-value column states: its own, else the whole stay."""

    window = reading.column_windows.get(column)
    if (
        window is not None
        and window.anchor is not None
        and window.start_hours is not None
        and window.end_hours is not None
    ):
        return window.start_hours, window.end_hours
    return 0.0, math.inf


def _bounded(
    concept: str,
    low: Optional[float],
    high: Optional[float],
    start: float,
    end: float,
    *,
    scale: float,
) -> tuple[ConceptPredicate, ...]:
    return tuple(
        _predicate(concept, start, end, aggregation="first", op=op, value=bound * scale)
        for op, bound in ((">=", low), ("<=", high))
        if bound is not None
    )


def _predicate(
    concept: str, start: float, end: float, *, aggregation: str, op: str, value: Any
) -> ConceptPredicate:
    try:
        return ConceptPredicate(
            concept_id=concept,
            time_window=TimeWindow(
                anchor=_ANCHOR, start_offset_hours=start, end_offset_hours=end
            ),
            aggregation=aggregation,  # type: ignore[arg-type]
            op=op,  # type: ignore[arg-type]
            value=value,
        )
    except CohortSchemaError as exc:
        raise _NotApplied("population_concept_unavailable", str(exc)) from exc


def _judge(
    reading: _Input, predicate: ConceptPredicate, side: Side, *, label: str
) -> str:
    """Raise at the first owner finding against ``predicate``; else a note, if any."""

    definition = CohortDefinition(
        name="population",
        inclusion=(predicate,) if side == "inclusion" else (),
        exclusion=(predicate,) if side == "exclusion" else (),
    )
    whole = predicates_read_over_the_whole_stay(
        definition,
        columns=reading.columns,
        whole_stay_columns=reading.whole_stay,
        label=label,
    )
    if whole:
        raise _NotApplied(
            "population_whole_stay_in_finite_window", whole[0].description()
        )
    windows = predicates_read_through_another_window(
        definition,
        columns=reading.columns,
        column_windows=reading.column_windows,
        label=label,
    )
    if windows:
        raise _NotApplied("population_window_unreadable", windows[0].description())
    times = predicates_read_by_an_event_time_not_in_hours(
        definition,
        columns=reading.columns,
        event_times_not_in_hours=reading.times_not_in_hours,
        label=label,
    )
    if times:
        raise _NotApplied("population_event_time_not_hours", times[0].description())
    domain = cohort_predicates_outside_column_domain(
        reading.context, inclusion=definition.inclusion, exclusion=definition.exclusion
    )
    if domain and domain[0].empties_cohort:
        raise _NotApplied(
            "population_threshold_outside_domain", domain[0].message() + "."
        )
    after = cohort_predicates_after_time_zero(
        reading.context,
        inclusion=[item.to_dict() for item in definition.inclusion],
        exclusion=[item.to_dict() for item in definition.exclusion],
        time_zero_hours=reading.time_zero_hours,
    )
    if after:
        raise _NotApplied(
            "population_determined_after_time_zero", after[0].message() + "."
        )
    # A declared domain every value of which meets the threshold: the study
    # stated that bound, so it is applied, and it removes no stay.
    return f" {domain[0].message()}." if domain else ""


def _column_of(predicate: ConceptPredicate, reading: _Input) -> str:
    return predicate_context_column(
        reading.variables, predicate.concept_id, predicate.aggregation
    )


# -- source proofs ------------------------------------------------------------


def _source_proof(
    criterion: PopulationCriterion, sources: _Sources
) -> Optional[SourceProof]:
    """A typed record whose parameters imply ``criterion`` for every input row."""

    if isinstance(criterion, AgeYears):
        return _range_proof(
            sources,
            step="age",
            step_bounds=("age_min", "age_max"),
            executed_bounds=("age_min", "age_max"),
            low=criterion.min_years,
            high=criterion.max_years,
        )
    if isinstance(criterion, IcuStayHours):
        return _range_proof(
            sources,
            step="los",
            step_bounds=("los_min", "los_max"),
            executed_bounds=("min_icu_los_hours", None),
            low=criterion.min_hours,
            high=criterion.max_hours,
        )
    if isinstance(criterion, FirstIcuStay):
        return _first_stay_proof(sources)
    if isinstance(criterion, DiagnosisCodes):
        return _diagnosis_proof(criterion, sources)
    return None


def _range_proof(
    sources: _Sources,
    *,
    step: str,
    step_bounds: tuple[str, str],
    executed_bounds: tuple[str, Optional[str]],
    low: Optional[float],
    high: Optional[float],
) -> Optional[SourceProof]:
    for item in sources.steps:
        if item.stage == "export" and item.criterion == step:
            recorded = tuple(item.parameters.get(name) for name in step_bounds)
            if _within(*recorded, low=low, high=high):
                return SourceProof(
                    kind="export_report_step",
                    record_ref="source_selection.export_report",
                    parameters={"criterion": step, **dict(item.parameters)},
                )
    executed = sources.executed
    if executed is not None:
        recorded = tuple(
            executed.get(name) if name is not None else None for name in executed_bounds
        )
        if _within(*recorded, low=low, high=high):
            return SourceProof(
                kind="recorded_study_contract",
                record_ref=f"source_selection.{_EXECUTED_COHORT}",
                parameters={
                    name: value
                    for name, value in zip(executed_bounds, recorded)
                    if name is not None
                },
            )
    return None


def _within(
    recorded_low: Any,
    recorded_high: Any,
    *,
    low: Optional[float],
    high: Optional[float],
) -> bool:
    """Whether the recorded inclusive range lies within the stated one.

    A recorded bound that is absent bounds nothing, so it shows no stated bound.
    """

    for recorded in (recorded_low, recorded_high):
        if recorded is not None and not _is_number(recorded):
            return False
    if low is not None and (recorded_low is None or float(recorded_low) < low):
        return False
    if high is not None and (recorded_high is None or float(recorded_high) > high):
        return False
    return True


def _first_stay_proof(sources: _Sources) -> Optional[SourceProof]:
    if sources.first_stay_receipt:
        return SourceProof(
            kind="host_first_icu_stay",
            record_ref="cohort.provenance.first_icu_stay_restriction",
            parameters={"schema_version": FIRST_ICU_STAY_RESTRICTION_SCHEMA},
        )
    for item in sources.steps:
        if item.stage == "host" and item.criterion == "first_icu_stay_restriction":
            return SourceProof(
                kind="host_first_icu_stay",
                record_ref="cohort.provenance.first_icu_stay_restriction",
                parameters={"criterion": item.criterion},
            )
        if (
            item.stage == "export"
            and item.criterion == "first_icu_stay"
            and item.parameters.get("first_icu_stay") is True
        ):
            return SourceProof(
                kind="export_report_step",
                record_ref="source_selection.export_report",
                parameters={"criterion": item.criterion, **dict(item.parameters)},
            )
    if sources.executed is not None and sources.executed.get("first_icu_stay") is True:
        return SourceProof(
            kind="recorded_study_contract",
            record_ref=f"source_selection.{_EXECUTED_COHORT}",
            parameters={"first_icu_stay": True},
        )
    return None


def _diagnosis_proof(
    criterion: DiagnosisCodes, sources: _Sources
) -> Optional[SourceProof]:
    """The export matched exactly these codes on the criterion's side.

    The export's records name no coding version: it matched the codes in
    every version it reads.  ICD-9 codes begin with a digit, E or V and
    ICD-10 codes with a letter, so only a code beginning with E or V names
    codes of both: the stays an export kept for one may hold the other
    version's.  Such an inclusion is not shown; an exclusion is, since the
    stays left hold neither.
    """

    stated = {diagnosis_code_token(code) for code in criterion.codes}
    side = "include" if criterion.role == "include" else "exclude"
    if side == "include" and any(code[:1] in _BOTH_VERSIONS for code in stated):
        return None
    for item in sources.steps:
        if item.stage == "export" and item.criterion == "icd":
            if _tokens(item.parameters.get(side)) == stated:
                return SourceProof(
                    kind="export_report_step",
                    record_ref="source_selection.export_report",
                    parameters={"criterion": "icd", side: sorted(stated)},
                )
    if sources.executed is not None:
        if _tokens(sources.executed.get(f"icd_{side}")) == stated:
            return SourceProof(
                kind="recorded_study_contract",
                record_ref=f"source_selection.{_EXECUTED_COHORT}",
                parameters={f"icd_{side}": sorted(stated)},
            )
    return None


def _tokens(values: Any) -> Optional[set[str]]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        return None
    if not all(isinstance(value, str) for value in values):
        return None
    return {diagnosis_code_token(value) for value in values}


def _proof_text(proof: SourceProof) -> str:
    parameters = ", ".join(
        f"{name}={value!r}" for name, value in sorted(proof.parameters.items())
    )
    return f"{proof.kind} at {proof.record_ref}: {parameters}"


# -- small readings -----------------------------------------------------------


@lru_cache(maxsize=1024)
def _extraction_defines(concept: str, database: str) -> bool:
    """Whether Data Extraction defines ``concept``, as its availability owner says.

    A dictionary concept, a derived outcome and an output a declared loader
    emits (``circ_failure``, ``sep3_sofa1``; the dictionary lacks these) are
    defined; any other name the owner does not find.  Whether ``database`` can
    derive a defined concept is the extraction's own check.
    """

    return (
        explain_concept_availability(concept=concept, database=database).reason
        != "concept_not_found"
    )


def _is_status_level(level: Any) -> bool:
    return isinstance(level, (bool, int, float)) and level in (0, 1)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit_key(unit: Any) -> str:
    return "".join(str(unit or "").split()).casefold()


__all__ = [
    "NOT_APPLIED_REASONS",
    "ONE_VALUE_SUMMARY",
    "POPULATION_APPROVAL_STOPS",
    "POPULATION_COMPILE_SCHEMA_VERSION",
    "REQUIRES_EXTRACTION_REASONS",
    "CompiledCriterion",
    "CompiledPopulation",
    "Disposition",
    "ProofKind",
    "SourceProof",
    "ThresholdReading",
    "compile_population",
    "population_spec_concepts",
    "read_threshold",
]
