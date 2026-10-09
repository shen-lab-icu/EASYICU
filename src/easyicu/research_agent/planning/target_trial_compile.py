"""How each element of a target trial reaches its emulation.

The study setup states the trial (:mod:`.target_trial_spec`); this module
decides, for each element, whether the emulation can carry it out, and records
the protocol those decisions add up to.  Each element gets exactly one
disposition:

* ``applied``: the input holds what the stated element needs;
* ``host_added``: a rule no study states and the host adds: the death time
  read to the hour, the ICU exit, the unit a bootstrap resamples, the rule
  for unmeasured baseline covariates, the weights' adjustment for the
  confounders carried;
* ``requires_extraction``: an extraction for the study would hold what the
  element needs (a treatment onset seen through the grace period, the
  endpoint, the death time, patient identity, the confounders);
* ``not_applied``: nothing can carry it out as stated, with a reason code.

The judgements are the owners', not this module's: what an absent treatment
record means and which drugs a concept records (:mod:`.treatment_capture`),
whether a concept records an event status (``concept.export_metadata``),
which fixed-horizon endpoints a database follows up (``outcome_availability``),
whether its death time is read to the hour (``utils.death_time_semantics``),
which covariates the host proves observed by time zero
(``adjustment_authority``), which rows belong to one patient
(``dependence_authority``), and whom the study includes, compiled at the
trial's time zero (:mod:`.population_compile`).  This module maps their
findings to reason codes.

A stated or host element that is neither ``applied`` nor ``host_added``
blocks approval, as do an eligibility the population owner does not apply, a
confounder that waits for an extraction, and an adjustment set with no
confounder the emulation can carry.  A confounder nothing can carry is listed
for the researcher instead.  The record also holds what the researcher
confirms at approval -- a capture reading or a class composition the
development assumed, a treatment wider than its class, a coordinate the study
did not state, a proposed adjustment set and each element the spec could not
type -- and the limitations every emulation of this kind carries.  Its
evidence ceiling is ``analysis_only``.

Compiling reads no patient row, calls no model and writes no file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional

from ...concept.export_metadata import concept_declares_event_status
from ...outcome_availability import (
    OUTCOME_CONCEPT_SUPPORTED_DATABASES,
    fixed_horizon_mortality_endpoint,
)
from ...utils.death_time_semantics import (
    DEATH_STATUS,
    DEATH_TIME_COMPANION,
    death_time_read_to_the_hour,
    native_export_death_time_semantics,
)
from ..authority.target_trial_claim_terms import TARGET_TRIAL_ASSUMPTION_PHRASE
from ..cohort.schema import context_materialized_columns
from ..concept_availability import (
    explain_concept_availability,
    normalize_database_name,
)
from ..contracts.model_retention import (
    MISSING_CATEGORY_MIN_ROWS,
    MISSING_CATEGORY_SHARE_THRESHOLD,
)
from ..contracts.target_trial_design import (
    MAX_GRACE_PERIOD_HOURS,
    MAX_TIME_ZERO_HOURS,
    MIN_TIME_ZERO_HOURS,
)
from ..research_context.materialization_window import (
    ColumnWindow,
    column_materialized_window,
    host_column_window,
)
from ..research_context.stay_events import ICU_LENGTH_OF_STAY_CONCEPT
from ..schema import ResearchContext
from .adjustment_authority import host_proven_temporal_roles
from .dependence_authority import (
    context_patient_group_authority,
    repeat_units_possible,
)
from .population_compile import CompiledPopulation
from .target_trial_spec import (
    TARGET_TRIAL_SPEC_SCHEMA_VERSION,
    TargetTrialSpec,
    TrialConfounder,
)
from .treatment_capture import (
    CaptureEntry,
    LoadedCaptureRegistry,
    packaged_treatment_capture_registry,
)

TARGET_TRIAL_COMPILE_SCHEMA_VERSION = "easyicu.target_trial_compile/1"

#: The time zeros the host offers, in whole hours after ICU admission.  Hour 0
#: is not one: a treatment start in ``[0, T0)`` marks a prevalent user, and
#: only a time zero of at least one hour leaves an hour to see it in.  The
#: bounds, and the longest grace period, are the emulation's
#: (``contracts.target_trial_design``), so its estimator refuses what this
#: compiler does not offer.
TIME_ZERO_MENU_HOURS = tuple(range(MIN_TIME_ZERO_HOURS, MAX_TIME_ZERO_HOURS + 1))
#: v1 emulations are analyses; their causal wording stays in templates.
EVIDENCE_CEILING = "analysis_only"

Disposition = Literal["applied", "host_added", "requires_extraction", "not_applied"]
ConfounderDisposition = Literal["applied", "requires_extraction", "not_applied"]
ConfirmationKind = Literal[
    "capture_assumption",
    "treatment_class",
    "treatment_outside_class",
    "design_choice",
    "confounder_set",
    "emulation_assumptions",
    "not_typed",
]

#: The elements a study states, in protocol order.
STATED_ELEMENTS = (
    "treatment",
    "strategies",
    "time_zero",
    "grace_period",
    "outcome",
    "indication",
)
#: The elements the host adds to every trial.
HOST_ELEMENTS = (
    "death_time",
    "icu_exit",
    "resampling_unit",
    "missing_baseline",
    "adjustment",
)

#: Why an element is not applied.  Stable: a published code never changes.
NOT_APPLIED_REASONS = (
    "tte_treatment_not_offered",
    "tte_treatment_capture_undeclared",
    "tte_treatment_class_unknown",
    "tte_treatment_definition_partial",
    "tte_strategy_not_typed",
    "tte_time_zero_not_offered",
    "tte_eligibility_after_time_zero",
    "tte_grace_period_not_offered",
    "tte_endpoint_unsupported",
    "tte_horizon_within_grace",
    "tte_indication_unstated",
    "tte_indication_not_in_population",
    "tte_indication_not_inclusion",
    "tte_indication_not_clinical",
    "tte_indication_not_applied",
    "tte_death_time_not_hourly",
    "tte_icu_exit_undefined",
    "tte_no_confounder_carried",
)
#: Why an element waits for an extraction that would hold what it needs.
REQUIRES_EXTRACTION_REASONS = (
    "tte_treatment_onset_not_materialized",
    "tte_grace_beyond_capture",
    "tte_endpoint_not_materialized",
    "tte_indication_requires_extraction",
    "tte_death_time_not_materialized",
    "tte_icu_exit_unavailable",
    "tte_patient_identity_unavailable",
    "tte_confounders_require_extraction",
)
#: Why a confounder is not carried, and why it waits for an extraction.
CONFOUNDER_NOT_APPLIED_REASONS = (
    "tte_confounder_unavailable",
    "tte_confounder_is_design_concept",
    "tte_confounder_after_time_zero",
)
CONFOUNDER_REQUIRES_EXTRACTION_REASONS = (
    "tte_confounder_not_in_export",
    "tte_confounder_window_after_time_zero",
)

_ANCHOR = "icu_admission"
#: Population criteria that state a clinical state: an indication is one.
_CLINICAL_KINDS = frozenset({"condition_present", "measurement", "diagnosis_codes"})
_ONSET_SUFFIX = "_onset_time"


def _frozen(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class CompiledElement:
    """One element of the trial with its single disposition."""

    element: str
    disposition: Disposition
    #: A code from ``NOT_APPLIED_REASONS`` or ``REQUIRES_EXTRACTION_REASONS``;
    #: ``None`` when the element is applied or added by the host.
    reason: Optional[str]
    detail: str
    #: What the element fixes, as the emulation reads it.
    parameters: Mapping[str, Any] = field(default_factory=lambda: _frozen({}))
    #: The words and their source, for an element the study stated.
    quote: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self) -> None:
        stated = self.element in STATED_ELEMENTS
        if not stated and self.element not in HOST_ELEMENTS:
            raise ValueError(f"{self.element!r} is no element of a target trial")
        if self.disposition == ("host_added" if stated else "applied"):
            raise ValueError(
                f"a {'stated' if stated else 'host'} element is never "
                f"{self.disposition}"
            )
        if not stated and self.quote is not None:
            raise ValueError("an element the host adds carries no stated words")
        allowed = {
            "not_applied": NOT_APPLIED_REASONS,
            "requires_extraction": REQUIRES_EXTRACTION_REASONS,
        }.get(self.disposition, (None,))
        if self.reason not in allowed:
            raise ValueError(f"reason {self.reason!r} does not fit {self.disposition}")

    @property
    def carried(self) -> bool:
        return self.disposition in {"applied", "host_added"}

    def record(self) -> dict[str, Any]:
        return {
            "element": self.element,
            "disposition": self.disposition,
            "reason": self.reason,
            "detail": self.detail,
            "parameters": dict(self.parameters),
            "quote": self.quote,
            "source": self.source,
        }


@dataclass(frozen=True)
class CompiledConfounder:
    """One stated confounder: carried at time zero, waiting for data, or not."""

    name: str
    clinical_rationale: str
    disposition: ConfounderDisposition
    reason: Optional[str]
    detail: str
    #: How the host proves it observed by time zero, when it is applied.
    temporal_role: Optional[str] = None

    def __post_init__(self) -> None:
        allowed = {
            "not_applied": CONFOUNDER_NOT_APPLIED_REASONS,
            "requires_extraction": CONFOUNDER_REQUIRES_EXTRACTION_REASONS,
        }.get(self.disposition, (None,))
        if self.reason not in allowed:
            raise ValueError(f"reason {self.reason!r} does not fit {self.disposition}")
        if (self.disposition == "applied") != (self.temporal_role is not None):
            raise ValueError("only an applied confounder has a proven temporal role")

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "clinical_rationale": self.clinical_rationale,
            "disposition": self.disposition,
            "reason": self.reason,
            "detail": self.detail,
            "temporal_role": self.temporal_role,
        }


@dataclass(frozen=True)
class Confirmation:
    """A line the researcher confirms on the approval card."""

    kind: ConfirmationKind
    element: str
    text: str

    def record(self) -> dict[str, Any]:
        return {"kind": self.kind, "element": self.element, "text": self.text}


@dataclass(frozen=True)
class CompiledTargetTrial:
    """Every element's disposition, the protocol, and what the researcher confirms."""

    spec: TargetTrialSpec
    database: str
    elements: tuple[CompiledElement, ...]
    confounders: tuple[CompiledConfounder, ...]
    population_sha256: str
    #: Whether the population owner holds approval for an inclusion it does not apply.
    population_blocking: bool
    capture_registry_sha256: str
    capture_entries: tuple[CaptureEntry, ...]
    protocol: tuple[tuple[str, str], ...]
    materialization: Mapping[str, Any]
    confirmations: tuple[Confirmation, ...]
    limitations: tuple[tuple[str, str], ...]

    @property
    def blocking(self) -> tuple[CompiledElement, ...]:
        """Elements the emulation does not carry: approval waits for the user."""

        return tuple(item for item in self.elements if not item.carried)

    @property
    def confounders_waiting(self) -> tuple[CompiledConfounder, ...]:
        """Confounders an extraction would let the emulation adjust for."""

        return tuple(
            item
            for item in self.confounders
            if item.disposition == "requires_extraction"
        )

    @property
    def approvable(self) -> bool:
        return not (
            self.blocking or self.population_blocking or self.confounders_waiting
        )

    def element(self, name: str) -> CompiledElement:
        for item in self.elements:
            if item.element == name:
                return item
        raise KeyError(name)

    def acquisition_windows(
        self,
    ) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
        """The windows an extraction for the trial reads, in hours after ICU admission.

        ``(cohort_window, event_onset_windows)`` as the acquisition takes them:
        covariates summarized over ``[0, T0)`` and each treatment concept's
        onset read over ``[0, T0 + G)``.
        """

        covariate = self.materialization["covariate_window"]
        onset = self.materialization["treatment_onset_window"]
        onset_window = (float(onset["start_hours"]), float(onset["end_hours"]))
        return (
            (float(covariate["start_hours"]), float(covariate["end_hours"])),
            {concept: onset_window for concept in self.spec.treatment.concepts},
        )

    def record(self) -> dict[str, Any]:
        return {
            "schema_version": TARGET_TRIAL_COMPILE_SCHEMA_VERSION,
            "spec_schema_version": TARGET_TRIAL_SPEC_SCHEMA_VERSION,
            "spec": self.spec.model_dump(mode="json"),
            "database": self.database,
            "elements": [item.record() for item in self.elements],
            "confounders": [item.record() for item in self.confounders],
            "population_sha256": self.population_sha256,
            "population_blocking": self.population_blocking,
            "capture_registry_sha256": self.capture_registry_sha256,
            "capture_entries": [
                entry.model_dump(mode="json") for entry in self.capture_entries
            ],
            "protocol": [{"item": item, "text": text} for item, text in self.protocol],
            "materialization": json.loads(json.dumps(dict(self.materialization))),
            "confirmations": [item.record() for item in self.confirmations],
            "limitations": [
                {"code": code, "text": text} for code, text in self.limitations
            ],
            "evidence_ceiling": EVIDENCE_CEILING,
            "approvable": self.approvable,
        }

    def sha256(self) -> str:
        return compile_record_sha256(self.record())


def compile_record_sha256(record: Mapping[str, Any]) -> str:
    """The digest of a compile record, kept or just compiled."""

    raw = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compile_target_trial(
    spec: TargetTrialSpec,
    context: ResearchContext,
    *,
    population: CompiledPopulation,
    registry: Optional[LoadedCaptureRegistry] = None,
) -> CompiledTargetTrial:
    """Decide how each element of ``spec`` reaches an emulation over ``context``.

    ``population`` is the study's population compiled at the trial's time
    zero; ``registry`` is the packaged capture registry unless given.
    """

    time_zero = spec.time_zero.hours_after_icu_admission
    if population.time_zero_hours is None or float(population.time_zero_hours) != float(
        time_zero
    ):
        raise ValueError("compile the population at the trial's time zero")
    loaded = registry or packaged_treatment_capture_registry()
    reading = _Reading(
        spec=spec,
        context=context,
        database=normalize_database_name(str(context.cohort.database or "")),
        columns=frozenset(context_materialized_columns(context)),
        variables={str(variable.name): variable for variable in context.variables},
        population=population,
        loaded=loaded,
    )
    entries = _capture_entries(reading)
    treatment = _treatment(reading, entries)
    confounders = tuple(_confounder(reading, item) for item in spec.confounders)
    elements = (
        treatment,
        _strategies(reading),
        _time_zero(reading),
        _grace_period(reading),
        _outcome(reading),
        _indication(reading),
        _death_time(reading),
        _icu_exit(reading),
        _resampling_unit(reading),
        _missing_baseline(),
        _adjustment(confounders),
    )
    return CompiledTargetTrial(
        spec=spec,
        database=reading.database,
        elements=elements,
        confounders=confounders,
        population_sha256=population.sha256(),
        population_blocking=bool(population.blocking),
        capture_registry_sha256=loaded.sha256,
        capture_entries=tuple(entry for entry in entries.values() if entry is not None),
        protocol=_protocol(reading),
        materialization=_frozen(_materialization(reading, confounders)),
        confirmations=_confirmations(reading, entries, confounders),
        limitations=_limitations(reading, entries),
    )


# -- what the input holds -----------------------------------------------------


@dataclass(frozen=True)
class _Reading:
    spec: TargetTrialSpec
    context: ResearchContext
    database: str
    columns: frozenset[str]
    variables: Mapping[str, Any]
    population: CompiledPopulation
    loaded: LoadedCaptureRegistry

    @property
    def time_zero(self) -> int:
        return self.spec.time_zero.hours_after_icu_admission

    @property
    def grace_end(self) -> int:
        return self.time_zero + self.spec.grace_period.hours

    def has_column(self, name: str) -> bool:
        return name in self.columns


def _stated(
    reading: _Reading,
    element: str,
    disposition: Disposition,
    reason: Optional[str],
    detail: str,
    parameters: Optional[Mapping[str, Any]] = None,
) -> CompiledElement:
    stated = getattr(reading.spec, element)
    return CompiledElement(
        element=element,
        disposition=disposition,
        reason=reason,
        detail=detail,
        parameters=_frozen(parameters or {}),
        quote=stated.quote,
        source=stated.source,
    )


def _capture_entries(reading: _Reading) -> dict[str, Optional[CaptureEntry]]:
    registry = reading.loaded.registry
    return {
        concept: registry.entry(reading.database, concept)
        for concept in reading.spec.treatment.concepts
    }


def _extraction_defines(concept: str, database: str) -> bool:
    """Whether Data Extraction defines ``concept``, as its availability owner says."""

    return (
        explain_concept_availability(concept=concept, database=database).reason
        != "concept_not_found"
    )


# -- the stated elements ------------------------------------------------------


def _treatment(
    reading: _Reading, entries: Mapping[str, Optional[CaptureEntry]]
) -> CompiledElement:
    treatment = reading.spec.treatment
    registry = reading.loaded.registry
    agents_of_class = registry.class_agents(treatment.treatment_class)
    if agents_of_class is None:
        return _stated(
            reading,
            "treatment",
            "not_applied",
            "tte_treatment_class_unknown",
            f"The capture registry names no treatment class "
            f"{treatment.treatment_class!r}; it names "
            f"{', '.join(sorted(registry.treatment_classes))}.",
        )
    for concept in treatment.concepts:
        if not concept_declares_event_status(concept):
            return _stated(
                reading,
                "treatment",
                "not_applied",
                "tte_treatment_not_offered",
                f"{concept!r} does not record an event status, so no hour can be "
                "read as the treatment's start.",
            )
    for concept, entry in entries.items():
        if entry is None or entry.absent_in_capture != "absent_row_is_no_event":
            stated = "no entry" if entry is None else repr(entry.absent_in_capture)
            return _stated(
                reading,
                "treatment",
                "not_applied",
                "tte_treatment_capture_undeclared",
                f"The capture registry states {stated} for {concept!r} in "
                f"{reading.database!r}: an absent record cannot be read as the "
                "treatment not given, so the deferring strategy cannot be "
                "told apart.",
            )
    recorded = frozenset(
        agent
        for entry in entries.values()
        if entry is not None
        for agent in entry.agents
    )
    missing = sorted(agents_of_class - recorded)
    if missing:
        return _stated(
            reading,
            "treatment",
            "not_applied",
            "tte_treatment_definition_partial",
            f"The concepts record no start of {', '.join(missing)}, drugs of the "
            f"{treatment.treatment_class} class: a stay that starts only these "
            "would count as deferring.",
        )
    parameters = {
        "concepts": list(treatment.concepts),
        "treatment_class": treatment.treatment_class,
        "agents": sorted(recorded),
        "onset_columns": [f"{c}{_ONSET_SUFFIX}" for c in treatment.concepts],
    }
    for concept in treatment.concepts:
        column = f"{concept}{_ONSET_SUFFIX}"
        window = _onset_window(reading, column)
        if window is None:
            return _stated(
                reading,
                "treatment",
                "requires_extraction",
                "tte_treatment_onset_not_materialized",
                f"This input holds no {column!r} with a window it records; an "
                f"extraction of its onset over [0, {reading.grace_end}) h would.",
                parameters,
            )
        if window.start_hours is None or window.start_hours > 0:
            return _stated(
                reading,
                "treatment",
                "requires_extraction",
                "tte_treatment_onset_not_materialized",
                f"{column!r} is read over {window.label}, which starts after ICU "
                "admission: a start before time zero cannot be seen.",
                parameters,
            )
    return _stated(
        reading,
        "treatment",
        "applied",
        None,
        "Its start is the first hour any of its concepts records the treatment.",
        parameters,
    )


def _onset_window(reading: _Reading, column: str) -> Optional[ColumnWindow]:
    """The window ``column`` was read over, from ICU admission; ``None`` without one."""

    if not reading.has_column(column):
        return None
    window = column_materialized_window(reading.context, column)
    if window is None or window.anchor != _ANCHOR:
        return None
    return window


def _strategies(reading: _Reading) -> CompiledElement:
    stops = [item for item in reading.spec.not_typed if item.affects_strategy]
    strategies = reading.spec.strategies
    parameters = {
        "initiate_label": strategies.initiate_label,
        "defer_label": strategies.defer_label,
    }
    if stops:
        return _stated(
            reading,
            "strategies",
            "not_applied",
            "tte_strategy_not_typed",
            "The study states what a strategy does beyond starting or deferring "
            f"the treatment, which v1 cannot emulate: {stops[0].quote!r} "
            f"({stops[0].why}).",
            parameters,
        )
    return _stated(
        reading,
        "strategies",
        "applied",
        None,
        "One strategy starts the treatment within the grace period; the other "
        "does not start it then and leaves it unrestricted afterwards.",
        parameters,
    )


def _time_zero(reading: _Reading) -> CompiledElement:
    hours = reading.time_zero
    parameters = {"hours_after_icu_admission": hours}
    if hours not in TIME_ZERO_MENU_HOURS:
        return _stated(
            reading,
            "time_zero",
            "not_applied",
            "tte_time_zero_not_offered",
            f"The host offers time zeros of {TIME_ZERO_MENU_HOURS[0]} to "
            f"{TIME_ZERO_MENU_HOURS[-1]} h after ICU admission, not {hours} h.",
            parameters,
        )
    late = [
        item.criterion.id
        for item in reading.population.criteria
        if item.reason == "population_determined_after_time_zero"
    ]
    if late:
        return _stated(
            reading,
            "time_zero",
            "not_applied",
            "tte_eligibility_after_time_zero",
            f"Population criteria {', '.join(late)} are decided after "
            f"{hours} h, so eligibility at time zero would use the future.",
            parameters,
        )
    return _stated(
        reading,
        "time_zero",
        "applied",
        None,
        "Eligibility, assignment and follow-up start at this hour.",
        parameters,
    )


def _grace_period(reading: _Reading) -> CompiledElement:
    hours = reading.spec.grace_period.hours
    parameters = {
        "hours": hours,
        "window_hours": [reading.time_zero, reading.grace_end],
    }
    if hours > MAX_GRACE_PERIOD_HOURS:
        return _stated(
            reading,
            "grace_period",
            "not_applied",
            "tte_grace_period_not_offered",
            f"The host offers grace periods of 1 to {MAX_GRACE_PERIOD_HOURS} h, "
            f"not {hours} h.",
            parameters,
        )
    for concept in reading.spec.treatment.concepts:
        window = _onset_window(reading, f"{concept}{_ONSET_SUFFIX}")
        if window is not None and (
            window.end_hours is None or window.end_hours < reading.grace_end
        ):
            return _stated(
                reading,
                "grace_period",
                "requires_extraction",
                "tte_grace_beyond_capture",
                f"The onset of {concept!r} is read over {window.label}, which ends "
                f"before the grace period does at {reading.grace_end} h.",
                parameters,
            )
    return _stated(
        reading,
        "grace_period",
        "applied",
        None,
        f"A start within [{reading.time_zero}, {reading.grace_end}) h follows the "
        "initiating strategy.",
        parameters,
    )


def _outcome(reading: _Reading) -> CompiledElement:
    endpoint_name = reading.spec.outcome.endpoint
    endpoint = fixed_horizon_mortality_endpoint(endpoint_name)
    supported = OUTCOME_CONCEPT_SUPPORTED_DATABASES.get(endpoint_name, frozenset())
    if endpoint is None or reading.database not in supported:
        return _stated(
            reading,
            "outcome",
            "not_applied",
            "tte_endpoint_unsupported",
            f"{endpoint_name!r} is not a fixed-horizon death endpoint that "
            f"{reading.database!r} follows up.",
        )
    parameters = {
        "event_concept": endpoint.event_concept,
        "followup_concept": endpoint.followup_concept,
        "horizon_days": endpoint.horizon_days,
        "time_origin": endpoint.time_origin,
    }
    if endpoint.horizon_days * 24 <= reading.grace_end:
        return _stated(
            reading,
            "outcome",
            "not_applied",
            "tte_horizon_within_grace",
            f"The {endpoint.horizon_days}-day horizon ends within the grace period.",
            parameters,
        )
    absent = [
        name
        for name in (endpoint.event_concept, endpoint.followup_concept)
        if not reading.has_column(name)
    ]
    if absent:
        return _stated(
            reading,
            "outcome",
            "requires_extraction",
            "tte_endpoint_not_materialized",
            f"This input holds no {', '.join(repr(name) for name in absent)}.",
            parameters,
        )
    return _stated(
        reading,
        "outcome",
        "applied",
        None,
        f"Death within {endpoint.horizon_days} days of ICU admission.",
        parameters,
    )


def _indication(reading: _Reading) -> CompiledElement:
    indication = reading.spec.indication
    if indication is None:
        return CompiledElement(
            element="indication",
            disposition="not_applied",
            reason="tte_indication_unstated",
            detail="The study names no population criterion that makes both "
            "strategies plausible for every eligible stay.",
        )
    compiled = {item.criterion.id: item for item in reading.population.criteria}
    parameters = {"criterion_ids": list(indication.criterion_ids)}
    for criterion_id in indication.criterion_ids:
        item = compiled.get(criterion_id)
        if item is None:
            return _stated(
                reading,
                "indication",
                "not_applied",
                "tte_indication_not_in_population",
                f"The population states no criterion {criterion_id!r}.",
                parameters,
            )
        if item.criterion.role != "include":
            return _stated(
                reading,
                "indication",
                "not_applied",
                "tte_indication_not_inclusion",
                f"{criterion_id!r} removes stays; an indication keeps them.",
                parameters,
            )
        if item.criterion.kind not in _CLINICAL_KINDS:
            return _stated(
                reading,
                "indication",
                "not_applied",
                "tte_indication_not_clinical",
                f"{criterion_id!r} states a {item.criterion.kind}, not a clinical "
                "state that calls for the treatment.",
                parameters,
            )
    states = [compiled[criterion_id] for criterion_id in indication.criterion_ids]
    unapplied = [item for item in states if item.disposition == "not_applied"]
    if unapplied:
        return _stated(
            reading,
            "indication",
            "not_applied",
            "tte_indication_not_applied",
            f"The population owner does not apply {unapplied[0].criterion.id!r} "
            f"({unapplied[0].reason}).",
            parameters,
        )
    waiting = [item for item in states if item.disposition == "requires_extraction"]
    if waiting:
        return _stated(
            reading,
            "indication",
            "requires_extraction",
            "tte_indication_requires_extraction",
            f"{waiting[0].criterion.id!r} waits for an extraction "
            f"({waiting[0].reason}).",
            parameters,
        )
    return _stated(
        reading,
        "indication",
        "applied",
        None,
        "Every eligible stay meets the indication at time zero.",
        parameters,
    )


# -- what the host adds -------------------------------------------------------


def _death_time(reading: _Reading) -> CompiledElement:
    semantics = native_export_death_time_semantics(reading.database)
    parameters = {"semantics": semantics, "column": DEATH_TIME_COMPANION}
    if death_time_read_to_the_hour(semantics) is not True:
        return CompiledElement(
            element="death_time",
            disposition="not_applied",
            reason="tte_death_time_not_hourly",
            detail=f"{reading.database!r} issues its death time as {semantics}, "
            "which no hour can be read from: deaths in the grace period could "
            "not be placed before or after a censoring.",
            parameters=_frozen(parameters),
        )
    if not (
        reading.has_column(DEATH_STATUS) and reading.has_column(DEATH_TIME_COMPANION)
    ):
        return CompiledElement(
            element="death_time",
            disposition="requires_extraction",
            reason="tte_death_time_not_materialized",
            detail=f"This input holds no {DEATH_STATUS!r} with its "
            f"{DEATH_TIME_COMPANION!r}.",
            parameters=_frozen(parameters),
        )
    return CompiledElement(
        element="death_time",
        disposition="host_added",
        reason=None,
        detail="Deaths are placed to the hour: alive at time zero, and a death "
        "in the grace period before a start counts in both strategies.",
        parameters=_frozen(parameters),
    )


def _icu_exit(reading: _Reading) -> CompiledElement:
    parameters = {"concept": ICU_LENGTH_OF_STAY_CONCEPT}
    if reading.has_column(ICU_LENGTH_OF_STAY_CONCEPT):
        return CompiledElement(
            element="icu_exit",
            disposition="host_added",
            reason=None,
            detail="ICU exit is read from the ICU length of stay: eligible stays "
            "are in the ICU at time zero, and a clone of the initiating strategy "
            "that leaves the ICU in the grace period before starting is "
            "censored then.",
            parameters=_frozen(parameters),
        )
    if _extraction_defines(ICU_LENGTH_OF_STAY_CONCEPT, reading.database):
        return CompiledElement(
            element="icu_exit",
            disposition="requires_extraction",
            reason="tte_icu_exit_unavailable",
            detail=f"This input holds no {ICU_LENGTH_OF_STAY_CONCEPT!r}.",
            parameters=_frozen(parameters),
        )
    return CompiledElement(
        element="icu_exit",
        disposition="not_applied",
        reason="tte_icu_exit_undefined",
        detail=f"Data Extraction defines no {ICU_LENGTH_OF_STAY_CONCEPT!r} for "
        f"{reading.database!r}.",
        parameters=_frozen(parameters),
    )


def _resampling_unit(reading: _Reading) -> CompiledElement:
    if not repeat_units_possible(reading.context):
        return CompiledElement(
            element="resampling_unit",
            disposition="host_added",
            reason=None,
            detail="The dependence owner finds no patient with repeated stays in "
            "this input, so the bootstrap resamples stays, each with both of its "
            "clones.",
            parameters=_frozen({"unit": "icu_stay"}),
        )
    group = context_patient_group_authority(reading.context)
    if group is not None:
        return CompiledElement(
            element="resampling_unit",
            disposition="host_added",
            reason=None,
            detail="The bootstrap resamples patients, each with all their stays "
            "and both clones of each.",
            parameters=_frozen(
                {
                    "unit": "patient",
                    "group_source": group.group_source,
                    "group_derivation": group.group_derivation,
                    "delimiter": group.delimiter,
                }
            ),
        )
    return CompiledElement(
        element="resampling_unit",
        disposition="requires_extraction",
        reason="tte_patient_identity_unavailable",
        detail="A patient may contribute several stays and this input identifies "
        "no patient: an extraction with patient identity, or one keeping each "
        "patient's first ICU stay, would let the bootstrap resample patients.",
    )


def _missing_baseline() -> CompiledElement:
    return CompiledElement(
        element="missing_baseline",
        disposition="host_added",
        reason=None,
        detail="An unmeasured baseline covariate keeps its rows as their own "
        "state under the model-retention rule, and the weight model counts "
        "that state among its parameters.",
        parameters=_frozen(
            {
                "rule": "model_retention",
                "missing_category_min_rows": MISSING_CATEGORY_MIN_ROWS,
                "missing_category_share_threshold": MISSING_CATEGORY_SHARE_THRESHOLD,
            }
        ),
    )


def _adjustment(confounders: tuple[CompiledConfounder, ...]) -> CompiledElement:
    """Whether the weights adjust for anything: at least one confounder is carried."""

    carried = [item.name for item in confounders if item.disposition == "applied"]
    waiting = [
        item.name for item in confounders if item.disposition == "requires_extraction"
    ]
    if carried:
        return CompiledElement(
            element="adjustment",
            disposition="host_added",
            reason=None,
            detail="The weights adjust for the confounders observed by time zero.",
            parameters=_frozen({"confounders": carried}),
        )
    if waiting:
        return CompiledElement(
            element="adjustment",
            disposition="requires_extraction",
            reason="tte_confounders_require_extraction",
            detail="No confounder is observed by time zero in this input; an "
            f"extraction would carry {', '.join(waiting)}.",
            parameters=_frozen({"confounders": waiting}),
        )
    return CompiledElement(
        element="adjustment",
        disposition="not_applied",
        reason="tte_no_confounder_carried",
        detail="No stated confounder can be adjusted for at time zero, so the "
        "weights would adjust for nothing and the comparison would be crude.",
    )


# -- confounders --------------------------------------------------------------


def _design_concepts(reading: _Reading) -> frozenset[str]:
    endpoint = fixed_horizon_mortality_endpoint(reading.spec.outcome.endpoint)
    names = {
        *reading.spec.treatment.concepts,
        reading.spec.outcome.endpoint,
        DEATH_STATUS,
        DEATH_TIME_COMPANION,
        ICU_LENGTH_OF_STAY_CONCEPT,
    }
    if endpoint is not None:
        names.add(endpoint.followup_concept)
    return frozenset(names)


def _is_design_concept(reading: _Reading, name: str) -> bool:
    design = _design_concepts(reading)
    variable = reading.variables.get(name)
    source = str(getattr(variable, "source_concept", "") or "").strip()
    return any(
        candidate == concept or candidate.startswith(f"{concept}_")
        for candidate in filter(None, (name, source))
        for concept in design
    )


def _confounder(reading: _Reading, item: TrialConfounder) -> CompiledConfounder:
    def compiled(
        disposition: ConfounderDisposition,
        reason: Optional[str],
        detail: str,
        role: Optional[str] = None,
    ) -> CompiledConfounder:
        return CompiledConfounder(
            name=item.name,
            clinical_rationale=item.clinical_rationale,
            disposition=disposition,
            reason=reason,
            detail=detail,
            temporal_role=role,
        )

    name = item.name
    if _is_design_concept(reading, name):
        return compiled(
            "not_applied",
            "tte_confounder_is_design_concept",
            f"{name!r} is the treatment, the outcome, a death or the ICU stay "
            "itself: the strategies, not the weights, account for it.",
        )
    if not reading.has_column(name):
        if _extraction_defines(name, reading.database):
            return compiled(
                "requires_extraction",
                "tte_confounder_not_in_export",
                f"This input holds no {name!r}; an extraction summarizing it over "
                f"[0, {reading.time_zero}) h would.",
            )
        return compiled(
            "not_applied",
            "tte_confounder_unavailable",
            f"{name!r} is neither a column of this input nor a concept Data "
            "Extraction defines.",
        )
    roles = host_proven_temporal_roles(
        reading.context, reference_hours=float(reading.time_zero)
    )
    if name in roles:
        return compiled(
            "applied",
            None,
            "The host proves it observed by time zero.",
            roles[name],
        )
    if _proven_over_covariate_window(reading, name):
        return compiled(
            "requires_extraction",
            "tte_confounder_window_after_time_zero",
            f"{name!r} is summarized past time zero here; summarized over "
            f"[0, {reading.time_zero}) h it would be observed by then.",
        )
    return compiled(
        "not_applied",
        "tte_confounder_after_time_zero",
        f"The host cannot prove {name!r} observed by time zero, even summarized "
        f"over [0, {reading.time_zero}) h.",
    )


def _proven_over_covariate_window(reading: _Reading, name: str) -> bool:
    """Whether the timing owner proves ``name`` once summarized over ``[0, T0)``.

    The question goes to the owner with the column's window replaced by the
    covariate window, so the rule stays the owner's.
    """

    window = host_column_window(0.0, float(reading.time_zero))
    variables = [
        variable.model_copy(update={"analysis_window": window.label})
        if str(variable.name) == name
        else variable
        for variable in reading.context.variables
    ]
    copy = reading.context.model_copy(update={"variables": variables})
    return name in host_proven_temporal_roles(
        copy, reference_hours=float(reading.time_zero)
    )


# -- the protocol and what the researcher confirms ----------------------------


def _horizon_days(reading: _Reading) -> Optional[int]:
    endpoint = fixed_horizon_mortality_endpoint(reading.spec.outcome.endpoint)
    return endpoint.horizon_days if endpoint is not None else None


def _protocol(reading: _Reading) -> tuple[tuple[str, str], ...]:
    spec = reading.spec
    t0, end = reading.time_zero, reading.grace_end
    treatment = " or ".join(spec.treatment.concepts)
    horizon = _horizon_days(reading)
    horizon_text = f"{horizon} days" if horizon is not None else "the horizon"
    indication = (
        f"meeting the indication ({', '.join(spec.indication.criterion_ids)})"
        if spec.indication is not None
        else "with no stated indication"
    )
    return (
        (
            "eligibility",
            f"ICU stays in the study population at {t0} h after ICU admission, "
            f"{indication}, alive and in the ICU then, with no recorded start of "
            f"{treatment} before {t0} h, and with a known vital status "
            f"{horizon_text} after ICU admission.",
        ),
        (
            "treatment_strategies",
            f"{spec.strategies.initiate_label}: start {treatment} within "
            f"[{t0}, {end}) h after ICU admission. {spec.strategies.defer_label}: "
            "do not start it within that period; afterwards unrestricted.",
        ),
        (
            "assignment",
            "Each eligible stay is cloned into both strategies; a clone is "
            "censored when its stay deviates from its strategy, and the "
            "censoring is weighted by baseline covariates.",
        ),
        (
            "time_zero",
            f"{t0} h after ICU admission: eligibility, assignment and follow-up "
            "start together.",
        ),
        (
            "follow_up",
            f"From time zero to death or {horizon_text} after ICU admission.",
        ),
        ("outcome", f"Death within {horizon_text} of ICU admission."),
        (
            "causal_contrast",
            f"The per-protocol effect of the strategies: the risk of death by "
            f"{horizon_text} under each, their difference and their ratio.",
        ),
        (
            "analysis_plan",
            "Clone, censor and weight; weighted Kaplan-Meier risk in each "
            "strategy; percentile intervals from a bootstrap that keeps each "
            "unit's clones together.",
        ),
    )


def _materialization(
    reading: _Reading, confounders: tuple[CompiledConfounder, ...]
) -> dict[str, Any]:
    """The windows and columns an extraction for the trial holds.

    Covariates are summarized over ``[0, T0)`` and treatment onsets read over
    ``[0, T0 + G)``; a confounder nothing can carry is not asked for.
    """

    endpoint = fixed_horizon_mortality_endpoint(reading.spec.outcome.endpoint)
    onset = [f"{c}{_ONSET_SUFFIX}" for c in reading.spec.treatment.concepts]
    columns = [
        *onset,
        *(
            [endpoint.event_concept, endpoint.followup_concept]
            if endpoint is not None
            else []
        ),
        DEATH_STATUS,
        DEATH_TIME_COMPANION,
        ICU_LENGTH_OF_STAY_CONCEPT,
        *(item.name for item in confounders if item.disposition != "not_applied"),
    ]
    return {
        "anchor": _ANCHOR,
        "covariate_window": {"start_hours": 0, "end_hours": reading.time_zero},
        "treatment_onset_window": {"start_hours": 0, "end_hours": reading.grace_end},
        "treatment_onset_columns": onset,
        "columns": list(dict.fromkeys(columns)),
    }


def _confirmations(
    reading: _Reading,
    entries: Mapping[str, Optional[CaptureEntry]],
    confounders: tuple[CompiledConfounder, ...],
) -> tuple[Confirmation, ...]:
    lines: list[Confirmation] = []
    spec = reading.spec
    treatment_class = spec.treatment.treatment_class
    agents_of_class = reading.loaded.registry.class_agents(treatment_class)
    for concept, entry in entries.items():
        if entry is not None and entry.basis == "development_assumption":
            lines.append(
                Confirmation(
                    kind="capture_assumption",
                    element="treatment",
                    text=f"A stay with no record of {concept} "
                    f"({', '.join(entry.agents)}) in the ICU is read as not given "
                    "them there; a use before ICU admission is not visible. "
                    "EasyICU development assumed this reading.",
                )
            )
    if agents_of_class is not None:
        lines.append(
            Confirmation(
                kind="treatment_class",
                element="treatment",
                text=f"The {treatment_class} class is read as "
                f"{', '.join(sorted(agents_of_class))}; EasyICU development set "
                "this composition.",
            )
        )
        recorded = {
            agent
            for entry in entries.values()
            if entry is not None
            for agent in entry.agents
        }
        extra = sorted(recorded - agents_of_class)
        if extra:
            lines.append(
                Confirmation(
                    kind="treatment_outside_class",
                    element="treatment",
                    text=f"A stay that starts only {' or '.join(extra)} counts as "
                    f"starting the treatment, though outside the {treatment_class} "
                    "class.",
                )
            )
    for element in STATED_ELEMENTS:
        stated = getattr(spec, element)
        if stated is not None and stated.source == "design_choice":
            lines.append(
                Confirmation(
                    kind="design_choice",
                    element=element,
                    text=f"The study did not state its {element.replace('_', ' ')}; "
                    f"the proposal is {stated.quote!r}.",
                )
            )
    lines.append(_confounder_set(spec, confounders))
    lines.append(_emulation_assumptions())
    for item in spec.not_typed:
        lines.append(
            Confirmation(
                kind="not_typed",
                element="strategies" if item.affects_strategy else "not_typed",
                text=f"Not emulated: {item.quote!r} ({item.why}).",
            )
        )
    return tuple(lines)


#: How the approval card names why a confounder is not adjusted for; one label
#: per ``CONFOUNDER_NOT_APPLIED_REASONS`` code (a code without one shows as is).
_NOT_ADJUSTED_FOR = {
    "tte_confounder_unavailable": "not defined for this database",
    "tte_confounder_is_design_concept": "part of the trial's design",
    "tte_confounder_after_time_zero": "not observed by time zero",
}


def _confounder_set(
    spec: TargetTrialSpec, confounders: tuple[CompiledConfounder, ...]
) -> Confirmation:
    """The adjustment set and the assumption it carries, confirmed for every trial."""

    def named(disposition: str) -> list[str]:
        return [item.name for item in confounders if item.disposition == disposition]

    parts = [f"Adjusted for at time zero: {', '.join(named('applied')) or 'nothing'}."]
    if named("requires_extraction"):
        parts.append(
            f"After an extraction, also: {', '.join(named('requires_extraction'))}."
        )
    uncarried = [item for item in confounders if item.disposition == "not_applied"]
    if uncarried:
        named_reasons = (
            (item.name, _NOT_ADJUSTED_FOR.get(str(item.reason), str(item.reason)))
            for item in uncarried
        )
        parts.append(
            "Not adjusted for: "
            + "; ".join(f"{name} ({why})" for name, why in named_reasons)
            + "."
        )
    proposed = [
        item.name for item in spec.confounders if item.source == "design_choice"
    ]
    if proposed:
        parts.append(
            f"Proposed by the setup, not stated by the study: {', '.join(proposed)}."
        )
    parts.append("The comparison assumes no confounding beyond the adjusted set.")
    return Confirmation(
        kind="confounder_set", element="confounders", text=" ".join(parts)
    )


def _emulation_assumptions() -> Confirmation:
    """What every estimate rests on, in the words the manuscript states it."""

    return Confirmation(
        kind="emulation_assumptions",
        element="analysis",
        text=f"Every estimate of the emulation holds only under "
        f"{TARGET_TRIAL_ASSUMPTION_PHRASE}, none of which the data can test; the "
        "manuscript states them with each estimate and among its limitations.",
    )


def _limitations(
    reading: _Reading, entries: Mapping[str, Optional[CaptureEntry]]
) -> tuple[tuple[str, str], ...]:
    lines: list[tuple[str, str]] = []
    if any(
        entry is not None and not entry.pre_admission_visible
        for entry in entries.values()
    ):
        lines.append(
            (
                "pre_admission_use_not_visible",
                "A use of the treatment before ICU admission is not recorded, so "
                "eligibility excludes only starts seen in the ICU before time zero.",
            )
        )
    lines.extend(
        (
            (
                "baseline_only_weights",
                "Within the grace period the start of treatment is weighted by "
                "baseline covariates only. A stay that worsens and starts the "
                "treatment leaves the deferring strategy, so the deferring "
                "strategy keeps healthier stays: its risk is underestimated, and "
                "the difference leans towards early treatment appearing harmful.",
            ),
            (
                "treatment_outside_icu_not_recorded",
                "A start after ICU exit is not recorded: a clone of the "
                "initiating strategy that leaves the ICU in the grace period "
                "before starting is censored then.",
            ),
            (
                "grace_period_deaths_in_both_strategies",
                "A stay that dies in the grace period before starting follows "
                "both strategies, so its death counts in the risk of each.",
            ),
        )
    )
    for item in reading.spec.not_typed:
        if not item.affects_strategy:
            lines.append(("not_typed", f"Not emulated: {item.quote!r} ({item.why})."))
    lines.append(
        (
            "evidence_ceiling",
            f"The evidence ceiling is {EVIDENCE_CEILING}: the estimates are an "
            "analysis under the emulation's assumptions.",
        )
    )
    return tuple(lines)


__all__ = [
    "CONFOUNDER_NOT_APPLIED_REASONS",
    "CONFOUNDER_REQUIRES_EXTRACTION_REASONS",
    "EVIDENCE_CEILING",
    "HOST_ELEMENTS",
    "MAX_GRACE_PERIOD_HOURS",
    "NOT_APPLIED_REASONS",
    "REQUIRES_EXTRACTION_REASONS",
    "STATED_ELEMENTS",
    "TARGET_TRIAL_COMPILE_SCHEMA_VERSION",
    "TIME_ZERO_MENU_HOURS",
    "CompiledConfounder",
    "CompiledElement",
    "CompiledTargetTrial",
    "Confirmation",
    "compile_record_sha256",
    "compile_target_trial",
]
