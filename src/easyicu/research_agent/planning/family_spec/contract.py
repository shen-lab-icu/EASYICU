"""Typed contracts for family-spec planning.

``FamilySpecRequest`` is host authority: every selectable coordinate the
Planner may fill, sealed and hashed before any provider call.  ``FamilyPlanSpec``
is the Planner's whole output for one attempt.  ``validate_family_plan_spec``
is the fail-closed boundary between the two; it rejects any name, coding,
level index, label, or citation outside the sealed request, and a rationale
never widens the host's timing authority.  When reviewed comparator design
cards are sealed into the request, the spec resolves every design dimension
from them and cites only a card that states that dimension.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ....utils.death_time_semantics import DEATH_STATUS, death_time_read_to_the_hour
from ....utils.time_units import ICU_TIME_PRE_ADMISSION_HOURS
from ...canonical_json import canonical_sha256
from ...contracts.primary_cohort import (
    STUDY_POPULATION_PRODUCTS,
    study_population_product_for,
)
from ..adjustment_authority import HostTemporalRole
from ..design_selection import ResearchDesignCandidate
from ..literature_design_authority import (
    LITERATURE_DESIGN_DIMENSIONS,
    CandidateLiteratureDesignDecision,
    LiteratureDesignEvidenceCard,
)
from ..question_requirements import (
    MAX_NAMED_QUESTION_CONCEPTS,
    MAX_QUESTION_REQUIREMENTS,
    NamedQuestionConcept,
    QuestionRequirement,
    question_requirement_problems,
)
from ..progressive_contract import (
    ModelTermCoding,
    ProgressiveCohortPredicate,
    ProgressivePopulationCriterion,
    ProgressivePredicateValue,
)

FAMILY_SPEC_SCHEMA_VERSION = "easyicu.family_plan_spec/1"
FAMILY_SPEC_REQUEST_SCHEMA_VERSION = "easyicu.family_spec_request/1"
LANDMARK_CATEGORICAL_FAMILY_ID = "landmark_categorical_association"
LANDMARK_SPLINE_FAMILY_ID = "landmark_spline_association"
DESCRIPTIVE_FAMILY_ID = "descriptive_exposure_outcome"
PHENOTYPING_FAMILY_ID = "cross_sectional_phenotyping"
PREDICTION_FAMILY_ID = "static_prediction_model"
LANDMARK_SURVIVAL_FAMILY_ID = "landmark_survival_suite"
LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID = "landmark_continuous_survival_suite"
FIXED_WINDOW_TRAJECTORY_FAMILY_ID = "fixed_window_trajectory_suite"
SOURCE_FEASIBILITY_FAMILY_ID = "source_feasibility_fail_closed"
LANDMARK_FAMILY_IDS = frozenset({LANDMARK_CATEGORICAL_FAMILY_ID, LANDMARK_SPLINE_FAMILY_ID})
#: Families whose template applies the plan's own cohort, so a population the
#: study states can be applied there; the survival families only for a
#: proposed suite.  A sealed suite's cohort is its study design's, and the
#: feasibility family decides nothing by a time zero.
POPULATION_FAMILY_IDS = frozenset(
    {
        *LANDMARK_FAMILY_IDS,
        DESCRIPTIVE_FAMILY_ID,
        PHENOTYPING_FAMILY_ID,
        PREDICTION_FAMILY_ID,
        LANDMARK_SURVIVAL_FAMILY_ID,
        LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID,
    }
)
#: The anchor a population predicate counts from: the family's time zero is
#: stated in hours after ICU admission.
POPULATION_ANCHOR = "icu_admission"
#: Families whose every scientific coordinate is a sealed runtime authority's;
#: the Planner only labels columns and writes comparator applications.
SEALED_SUITE_FAMILY_IDS = frozenset(
    {
        LANDMARK_SURVIVAL_FAMILY_ID,
        LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID,
        FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
        SOURCE_FEASIBILITY_FAMILY_ID,
    }
)
FamilyId = Literal[
    "landmark_categorical_association",
    "landmark_spline_association",
    "descriptive_exposure_outcome",
    "cross_sectional_phenotyping",
    "static_prediction_model",
    "landmark_survival_suite",
    "landmark_continuous_survival_suite",
    "fixed_window_trajectory_suite",
    "source_feasibility_fail_closed",
]
ExposureKind = Literal["categorical", "continuous", "none"]
#: The most levels one categorical exposure may carry in a request.  The
#: router reads the same bound, so it never offers a template an exposure
#: this contract refuses.
MAX_EXPOSURE_LEVELS = 24

_RATIONALE_MIN = 16
_RATIONALE_MAX = 500
_APPLICATION_MIN = 8
_APPLICATION_MAX = 1200


class FamilySpecError(ValueError):
    """A spec, request, or template projection violated its typed contract."""

    def __init__(self, reason_code: str, message: str, *, path: str = "") -> None:
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code
        self.path = path


class AdjustmentCandidate(BaseModel):
    """One host-projected candidate covariate offered to the Planner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    semantic_role: str = Field(min_length=1, max_length=64)
    host_temporal_role: Optional[HostTemporalRole] = None
    allowed_codings: list[ModelTermCoding] = Field(min_length=1)
    closed_domain_size: Optional[int] = Field(default=None, ge=2)
    selectable: bool
    boundary: str = Field(min_length=1, max_length=300)
    #: The measured share of rows without a value, when the context measured
    #: it; a landmark categorical template keeps frequently unmeasured
    #: covariates' rows as an explicit unmeasured state.
    missing_share: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, exclude_if=lambda value: value is None
    )


class AcceptedFeatureGroup(BaseModel):
    """One accepted primary-analysis input and the fit features that represent it.

    A package-bound attempt keeps the reviewed candidate's primary inputs; the
    Planner chooses each input's value representation, not whether it stays.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    concept: str = Field(min_length=1, max_length=128)
    columns: list[str] = Field(min_length=1)


class AcceptedBaselineRow(BaseModel):
    """One row of an accepted baseline roster and the columns that describe it.

    A plan change after review keeps the reviewed candidate's Table 1 content
    (``accepted_baseline_requirements``).  The template keeps each row in its
    own Table 1; the first column is the one it adds when none is present.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    required: str = Field(min_length=1, max_length=128)
    columns: list[str] = Field(min_length=1)
    summary: Literal["mean_sd", "median_iqr", "both", "count_percent"]


class SensitivityAxisBinding(BaseModel):
    """One prespecified StudyContext sensitivity spec projected for the template."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    axis: str = Field(min_length=1, max_length=64)
    strategy: str = Field(min_length=1, max_length=64)
    execution_variables: list[str] = Field(default_factory=list)


class SealedSuiteCoordinates(BaseModel):
    """Coordinates a sealed host suite disclosed for the planner to name, not choose.

    The signing runtime authority (for example the fixed-landmark survival
    suite) owns exposure timing, endpoint, horizon, adjustment set and output
    products.  The template copies these verbatim into the single primary
    step; the host's ``bind_plan`` then replaces that step with the signed
    owner and its figure, so nothing here is a Planner decision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_owner: str = Field(pattern=r"^signed_[a-z][a-z0-9_]{0,79}$")
    exposure_status_column: str = Field(min_length=1, max_length=128)
    exposure_onset_column: str = Field(min_length=1, max_length=128)
    #: The onset is the first record of the exposure as present; ``None`` for a
    #: suite signed before that, which timed it by its first record of any value.
    #: Left out of the dump while unset, so such a request keeps its digest.
    exposure_onset_representation: Literal["first_truthy_event_time"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    #: The hours the suite's prevalence-definition sensitivity analysis
    #: re-fits at; ``None`` for a suite without it, left out of the dump.
    prevalence_sensitivity_cutoffs_hours: list[float] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    event_column: str = Field(min_length=1, max_length=128)
    followup_time_column: str = Field(min_length=1, max_length=128)
    landmark_hours: float = Field(gt=0.0)
    endpoint_horizon_days: float = Field(gt=0.0)
    adjustment_columns: list[str] = Field(default_factory=list)
    plan_outputs: list[str] = Field(min_length=1)

    @field_validator("adjustment_columns", "plan_outputs")
    @classmethod
    def _unique_nonblank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("sealed suite rosters must contain unique non-empty values")
        return cleaned

    @property
    def source_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.exposure_status_column,
                    self.exposure_onset_column,
                    self.event_column,
                    self.followup_time_column,
                    *self.adjustment_columns,
                )
            )
        )

    @property
    def analysis_outputs(self) -> tuple[str, ...]:
        """Every owned product except the composite figure the host renders."""

        return tuple(
            value for value in self.plan_outputs if not value.startswith("figure:")
        )


class SealedContinuousSuiteCoordinates(BaseModel):
    """The continuous-exposure landmark survival suite's coordinates, named, not chosen.

    The signing runtime authority models one window summary of a continuous
    exposure, recorded from ICU admission to the landmark, per unit of its
    source's scale.  Like :class:`SealedSuiteCoordinates`, the template copies
    these verbatim into the single primary step and ``bind_plan`` replaces it
    with the signed owner and its figure.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_owner: Literal["signed_landmark_continuous_survival_suite"]
    exposure_column: str = Field(min_length=1, max_length=128)
    #: How the materializer summarised the exposure over hours 0 to the landmark.
    exposure_window_summary: Literal["max", "min", "mean", "first"]
    #: The unit of the exposure source's recorded scale, when the context states it.
    exposure_unit: Optional[str] = Field(default=None, min_length=1, max_length=32)
    event_column: str = Field(min_length=1, max_length=128)
    followup_time_column: str = Field(min_length=1, max_length=128)
    landmark_hours: float = Field(gt=0.0)
    endpoint_horizon_days: float = Field(gt=0.0)
    adjustment_columns: list[str] = Field(default_factory=list)
    plan_outputs: list[str] = Field(min_length=1)

    @field_validator("adjustment_columns", "plan_outputs")
    @classmethod
    def _unique_nonblank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("sealed suite rosters must contain unique non-empty values")
        return cleaned

    @model_validator(mode="after")
    def _landmark_precedes_the_horizon(self) -> "SealedContinuousSuiteCoordinates":
        if self.landmark_hours / 24.0 >= self.endpoint_horizon_days:
            raise ValueError("the suite's landmark must precede its endpoint horizon")
        return self

    @property
    def source_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.exposure_column,
                    self.event_column,
                    self.followup_time_column,
                    *self.adjustment_columns,
                )
            )
        )

    @property
    def analysis_outputs(self) -> tuple[str, ...]:
        """Every owned product except the composite figure the host renders."""

        return tuple(
            value for value in self.plan_outputs if not value.startswith("figure:")
        )


class SealedTrajectoryCoordinates(BaseModel):
    """Coordinates of a sealed fixed-window trajectory suite, disclosed to be named.

    The signing ``TrajectoryScientificRuntimeAuthority`` owns the coordinate
    concepts, the fixed grid, the candidate cluster grid, the selection rule
    and the stability design.  The template names the two signed owners; the
    host's ``bind_plan`` then compiles the four signed steps from the
    authority, so nothing here is a Planner decision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    representation_owner: str = Field(pattern=r"^signed_[a-z][a-z0-9_]{0,79}$")
    candidate_owner: str = Field(pattern=r"^[a-z][a-z0-9_]{0,119}$")
    coordinate_concepts: list[str] = Field(min_length=2)
    descriptive_only_concepts: list[str] = Field(default_factory=list)
    window_hours: tuple[int, int]
    grid_width_hours: int = Field(gt=0)
    candidate_cluster_counts: list[int] = Field(min_length=2)
    representation_outputs: list[str] = Field(min_length=1)
    #: The population the suite seals: the reviewed plan's cohort predicates,
    #: which ``bind_plan`` gives the signed plan.  Empty when the owners keep
    #: every input row, and then left out of the request digest.
    population_inclusion: list[ProgressiveCohortPredicate] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    population_exclusion: list[ProgressiveCohortPredicate] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )

    @field_validator("coordinate_concepts", "descriptive_only_concepts", "representation_outputs")
    @classmethod
    def _unique_nonblank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("sealed trajectory rosters must contain unique non-empty values")
        return cleaned

    @model_validator(mode="after")
    def _window_is_ordered(self) -> "SealedTrajectoryCoordinates":
        if self.window_hours[0] >= self.window_hours[1]:
            raise ValueError("sealed trajectory window must have positive width")
        return self


def sealed_cohort_predicate(item: Any) -> ProgressiveCohortPredicate:
    """One cohort predicate a sealed authority discloses, as a plan predicate.

    The disclosure carries the predicate in the canonical form the plan cohort
    states it; every coordinate is kept, and the value takes its closed form.
    A predicate without that form raises, so the caller treats the disclosure
    as unreadable instead of planning on part of a population.
    """

    window = item.get("time_window") if isinstance(item, Mapping) else None
    if not isinstance(window, Mapping):
        raise TypeError("a sealed cohort predicate is an object with a time window")
    return ProgressiveCohortPredicate(
        concept_id=item.get("concept_id"),
        anchor=window.get("anchor"),
        start_offset_hours=window.get("start_offset_hours"),
        end_offset_hours=window.get("end_offset_hours"),
        aggregation=item.get("aggregation"),
        op=item.get("op"),
        value=_closed_predicate_value(item.get("value")),
    )


#: Why a prediction's risk set does not read deaths before its prediction time.
PredictionDeathTimeReason = Literal[
    "death_time_resolution",
    "death_time_semantics_unrecorded",
    "death_status_absent",
    "death_time_companion_absent",
]


class PredictionDeathTime(BaseModel):
    """How a static prediction's risk set reads deaths before its prediction time.

    ``los_icu`` keeps the stays still in the ICU after the prediction time,
    but a stay can stay in the ICU after its recorded death (an ICU discharge
    recorded after the death), so a stay that died before the prediction can
    remain.  Where the source records the death's time to the hour, the risk
    set also keeps only the stays without a death recorded before it, read by
    that time (``applied``).  Otherwise ``reason`` says why it does not:

    * ``death_time_resolution``: the export's death time is a date, a proxy
      or none (``semantics`` is its producer's label);
    * ``death_time_semantics_unrecorded``: the export labels no death time
      that the producer's vocabulary classifies;
    * ``death_status_absent``: the roster has no death status with two
      closed levels;
    * ``death_time_companion_absent``: the roster types no death time as the
      death's time after ICU admission.

    ``absent_level`` is the death status's level for no death, which the
    risk set's predicate compares with.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    applied: bool
    semantics: Optional[str] = Field(default=None, min_length=1, max_length=128)
    reason: Optional[PredictionDeathTimeReason] = None
    absent_level: Optional[Union[bool, float]] = None

    @model_validator(mode="after")
    def _applied_or_why_not(self) -> "PredictionDeathTime":
        if self.applied:
            if (
                self.reason is not None
                or self.absent_level is None
                or death_time_read_to_the_hour(self.semantics or "") is not True
            ):
                raise ValueError(
                    "a risk set reads deaths by a death time recorded to the hour, "
                    "compares the level for no death, and states no reason"
                )
        elif self.reason is None or self.absent_level is not None:
            raise ValueError(
                "a risk set that does not read deaths by their time states why"
            )
        return self


def prediction_risk_set_predicates(
    *,
    prediction_time_hours: float,
    icu_stay_unit: str,
    death_time: Optional[PredictionDeathTime] = None,
) -> list[ProgressiveCohortPredicate]:
    """The cohort predicates of a static prediction's risk set.

    A static prediction model predicts at the end of its observation window,
    its time zero, for the stays still in the ICU after it: ``los_icu`` above
    the prediction time, written in the unit the roster records ``los_icu`` in
    (days, the dictionary's unit, or hours) and decided at that time.  The
    family template and the check of a caller-bound population both read the
    risk set from here.

    Where the risk set reads deaths by their time (``death_time.applied``), a
    second predicate keeps the stays without a death recorded before the
    prediction time: the death status equals its level for no death over the
    export's pre-admission context up to the prediction time, a window the
    cohort builder reads by the death's recorded time.  A death at the
    prediction time or later, or one without a recorded time, stays.  It is a
    row of its own in the cohort's ledger.
    """

    hours_per_unit = 1.0 if icu_stay_unit == "hours" else 24.0
    predicates = [
        ProgressiveCohortPredicate(
            concept_id="los_icu",
            anchor="icu_admission",
            start_offset_hours=0.0,
            end_offset_hours=float(prediction_time_hours),
            aggregation="first",
            op=">",
            value=ProgressivePredicateValue(
                mode="number",
                number_value=float(prediction_time_hours) / hours_per_unit,
            ),
        )
    ]
    if death_time is not None and death_time.applied:
        predicates.append(
            ProgressiveCohortPredicate(
                concept_id=DEATH_STATUS,
                anchor="icu_admission",
                start_offset_hours=-float(ICU_TIME_PRE_ADMISSION_HOURS),
                end_offset_hours=float(prediction_time_hours),
                aggregation="any",
                op="==",
                value=_closed_predicate_value(death_time.absent_level),
            )
        )
    return predicates


def _closed_predicate_value(value: Any) -> ProgressivePredicateValue:
    if value is None:
        return ProgressivePredicateValue(mode="none")
    if isinstance(value, bool):
        return ProgressivePredicateValue(mode="boolean", boolean_value=value)
    if isinstance(value, (int, float)):
        return ProgressivePredicateValue(mode="number", number_value=float(value))
    if isinstance(value, str):
        return ProgressivePredicateValue(mode="string", string_value=value)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return ProgressivePredicateValue(mode="string_list", string_list=list(value))
    if isinstance(value, list) and value and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ):
        return ProgressivePredicateValue(
            mode="number_list", number_list=[float(item) for item in value]
        )
    raise ValueError("a sealed cohort predicate value has no closed form")


class SealedFeasibilityCoordinates(BaseModel):
    """Coordinates of a sealed fail-closed source-feasibility decision.

    The signing ``SourceFeasibilityRuntimeAuthority`` owns the decision: the
    reviewed protocol found the requested treatment contrast non-identifiable
    from the current source capture.  The template names the one signed owner
    and its products; the host's ``bind_plan`` compiles the single signed
    step, so the Planner neither drafts a contrast nor an effect estimate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sealed_owner: str = Field(pattern=r"^signed_[a-z][a-z0-9_]{0,79}$")
    plan_intent: str = Field(min_length=1, max_length=1200)
    plan_outputs: list[str] = Field(min_length=2, max_length=2)
    source: str = Field(min_length=1, max_length=256)
    audited_window_hours: tuple[int, int]
    decision: Literal["fail_closed"]
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,79}$")
    forbidden_plan_tokens: list[str] = Field(min_length=1)

    @field_validator("plan_outputs", "forbidden_plan_tokens")
    @classmethod
    def _unique_nonblank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("sealed feasibility rosters must contain unique non-empty values")
        return cleaned

    @model_validator(mode="after")
    def _closed_products(self) -> "SealedFeasibilityCoordinates":
        if self.audited_window_hours[0] >= self.audited_window_hours[1]:
            raise ValueError("sealed feasibility window must have positive width")
        kinds = sorted(value.split(":", 1)[0] for value in self.plan_outputs)
        if kinds != ["log", "table"]:
            raise ValueError("sealed feasibility outputs are one table and one log")
        return self


class StudyPopulationOccurrence(BaseModel):
    """How often the exposure occurs in the population the study selected.

    A landmark primary analysis keeps only the stays alive and observed at the
    landmark, so it cannot say how often the exposure occurs among all the
    stays the study selected.  When the question asks for that, the host seals
    this coordinate and the template estimates it on the run's study cohort,
    republished unchanged under ``product_id``.  The missing-value policies
    follow the sealed variables' known missingness.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    product_id: str
    requested_cues: list[str] = Field(min_length=1, max_length=32)
    denominator_policy: Literal["all_declared_rows", "observed_outcome_rows"]
    missing_exposure_policy: Literal["fail_closed", "exclude_from_denominator"]
    missing_outcome_policy: Literal["fail_closed", "exclude_from_denominator"]

    @field_validator("product_id")
    @classmethod
    def _study_population_product(cls, value: str) -> str:
        if value not in STUDY_POPULATION_PRODUCTS:
            raise ValueError("the occurrence reads a host study-population product")
        return value

    @model_validator(mode="after")
    def _denominator_follows_outcome_policy(self) -> "StudyPopulationOccurrence":
        if (self.missing_outcome_policy == "exclude_from_denominator") != (
            self.denominator_policy == "observed_outcome_rows"
        ):
            raise ValueError(
                "rows without an observed outcome leave the denominator exactly "
                "when it counts observed-outcome rows"
            )
        return self


class FamilySpecRequest(BaseModel):
    """Sealed host authority for one family-spec planning attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.family_spec_request/1"] = (
        FAMILY_SPEC_REQUEST_SCHEMA_VERSION
    )
    family_id: FamilyId
    analysis_type: Literal[
        "association_study",
        "descriptive_epidemiology",
        "trajectory_clustering",
        "prediction_model",
        "survival",
        "causal_inference",
    ] = "association_study"
    research_question: str = Field(min_length=1, max_length=6000)
    cohort_name: str = Field(min_length=1, max_length=128)
    cohort_selection_mode: Literal["all_input_rows", "predicate_filtered"]
    age_min: Optional[float] = None
    age_max: Optional[float] = None
    #: A typed minimum ICU stay in hours.  Omitted from the digest when absent,
    #: so requests sealed before it existed keep their identity.
    minimum_icu_hours: Optional[float] = Field(
        default=None, gt=0.0, exclude_if=lambda value: value is None
    )
    #: The prediction time of a static prediction model, in hours after ICU
    #: admission: the end of its observation window.  The model's rows are
    #: the stays still in the ICU after it.  Omitted from the digest when
    #: absent, like the minimum stay.
    prediction_time_hours: Optional[float] = Field(
        default=None, gt=0.0, exclude_if=lambda value: value is None
    )
    #: The unit the sealed roster records ``los_icu`` in, which the ICU-stay
    #: bounds above are written in.  Omitted from the digest in days, the
    #: concept dictionary's unit.
    icu_stay_unit: Literal["days", "hours"] = Field(
        default="days", exclude_if=lambda value: value == "days"
    )
    #: The caller bound every input row as the population, and the prediction
    #: family filters it only to its risk set: the age and stay bounds then
    #: describe that population and add no predicate.  Omitted from the
    #: digest when false, like the minimum stay.
    caller_binds_all_input_rows: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    #: How the prediction's risk set reads deaths before its prediction time,
    #: set only when the context records what the export's death time is.
    #: Omitted from the digest when absent, like the minimum stay, so a
    #: request built before the record keeps its identity.
    prediction_death_time: Optional[PredictionDeathTime] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    #: A concept-derived population (``sepsis3`` ...) and the hour after ICU
    #: admission by which a positive concept row admits a stay.  Omitted from
    #: the digest when absent, like the minimum stay.
    concept_cohort_definition: Optional[str] = Field(
        default=None, min_length=1, max_length=64, exclude_if=lambda value: value is None
    )
    concept_cohort_window_end_hours: Optional[float] = Field(
        default=None, gt=0.0, exclude_if=lambda value: value is None
    )
    identity_column: str = Field(min_length=1, max_length=128)
    cluster_unit: Optional[Literal["patient"]] = None
    primary_exposure: str = Field(max_length=128)
    exposure_kind: ExposureKind = "categorical"
    exposure_levels: list[str] = Field(default_factory=list, max_length=MAX_EXPOSURE_LEVELS)
    reference_level_index: int = Field(ge=0)
    primary_contrast_level_index: int = Field(ge=0)
    exposure_is_ordered: bool
    exposure_companion_columns: list[str] = Field(default_factory=list)
    #: Empty for the sealed feasibility family, which analyses no outcome, and
    #: for a trajectory suite whose study has none: its classes are discovered
    #: without an outcome, which is only ever described by frozen class.
    outcome: str = Field(max_length=128)
    outcome_levels: list[str] = Field(default_factory=list, max_length=2)
    event_level_index: int = Field(ge=0, le=1)
    landmark_hours: Optional[float] = Field(default=None, gt=0.0)
    landmark_spec_id: Optional[str] = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,79}$")
    event_time_column: Optional[str] = Field(default=None, min_length=1, max_length=128)
    observation_duration_column: Optional[str] = Field(default=None, min_length=1, max_length=128)
    observation_duration_unit: Optional[str] = Field(default=None, min_length=1, max_length=32)
    observation_window_hours: Optional[float] = Field(default=None, gt=0.0)
    level_label_keys: list[str] = Field(default_factory=list)
    counts_only: bool = False
    feature_candidates: list[AdjustmentCandidate] = Field(default_factory=list)
    accepted_feature_groups: list[AcceptedFeatureGroup] = Field(default_factory=list)
    accepted_baseline_rows: list[AcceptedBaselineRow] = Field(default_factory=list)
    membership_candidates: list[str] = Field(default_factory=list)
    secondary_continuous_outcome: Optional[str] = Field(default=None, max_length=128)
    sealed_suite: Optional[SealedSuiteCoordinates] = None
    #: The landmark survival suite the host could seal for a study that has no
    #: survival design yet: host-vocabulary coordinates with an empty roster the
    #: Planner selects.  Nothing in it is sealed.  Omitted from the digest when
    #: absent, so requests sealed before it existed keep their identity.
    proposed_suite: Optional[SealedSuiteCoordinates] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    #: The continuous-exposure survival suite, sealed or proposed like the
    #: binary suite above.  Omitted from the digest when absent, so every
    #: other request keeps its identity.
    sealed_continuous_suite: Optional[SealedContinuousSuiteCoordinates] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    proposed_continuous_suite: Optional[SealedContinuousSuiteCoordinates] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    sealed_trajectory: Optional[SealedTrajectoryCoordinates] = None
    sealed_feasibility: Optional[SealedFeasibilityCoordinates] = None
    adjustment_selection: Literal["planner_selectable", "exact"]
    exact_roster: list[str] = Field(default_factory=list)
    exact_rationales: dict[str, str] = Field(default_factory=dict)
    exact_temporal_roles: dict[str, str] = Field(default_factory=dict)
    adjustment_candidates: list[AdjustmentCandidate] = Field(default_factory=list)
    alternate_exposures: list[SensitivityAxisBinding] = Field(default_factory=list)
    first_stay: Optional[SensitivityAxisBinding] = None
    complete_case_spec_id: Optional[str] = Field(
        default=None, pattern=r"^[a-z][a-z0-9_]{0,79}$"
    )
    functional_form_spec_ids: dict[str, str] = Field(default_factory=dict)
    measurement_audit_columns: list[str] = Field(default_factory=list)
    required_reader_label_keys: list[str] = Field(default_factory=list)
    allowed_literature_citation_keys: list[str] = Field(default_factory=list)
    direct_comparator_literature_keys: list[str] = Field(default_factory=list)
    comparison_literature_keys: list[str] = Field(default_factory=list)
    comparator_titles: dict[str, str] = Field(default_factory=dict)
    #: Reviewed design cards of the included comparison sources.  Omitted
    #: from the digest when absent, so requests sealed before it existed keep
    #: their identity.
    literature_design_cards: list[LiteratureDesignEvidenceCard] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    variable_roster: list[str] = Field(min_length=1)
    #: Omitted from the digest when absent, so requests sealed before it
    #: existed keep their identity.
    study_population_occurrence: Optional[StudyPopulationOccurrence] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    #: The cohort concepts a population the study states may read.  Offered
    #: only to a family that applies its plan's own cohort by a known time
    #: zero, and only when the caller has not bound every input row; empty
    #: otherwise.  Omitted from the digest when empty, like the fields above.
    population_concepts: list[str] = Field(
        default_factory=list, max_length=1024, exclude_if=lambda value: not value
    )
    #: The study's own cohort wording (label, review, exclusion statement):
    #: a source of that population, applied by nothing until the Planner
    #: states it as predicates.  Omitted from the digest when empty.
    study_cohort_wording: dict[str, str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    #: The criteria the source export is known to have applied, verbatim:
    #: its whole recorded selection, or only the host's own criteria when
    #: that selection is not recorded (``export_applied_selection``'s
    #: ``known_applied``), and whether it is recorded.  Offered with the
    #: population, so the Planner neither restates a criterion already
    #: applied nor takes as applied one that nothing records.  Omitted from
    #: the digest when empty or false.
    source_applied_inclusion: list[str] = Field(
        default_factory=list, max_length=64, exclude_if=lambda value: not value
    )
    source_applied_exclusion: list[str] = Field(
        default_factory=list, max_length=64, exclude_if=lambda value: not value
    )
    source_selection_recorded: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    #: The concepts the research question names, each as the roster columns
    #: its name or the concept dictionary relates to it
    #: (``question_requirements.bind_named_question_concepts``).  The spec's
    #: question requirements must account for every one that is not the
    #: sealed exposure or outcome.  Omitted from the digest when empty.
    question_named_concepts: list[NamedQuestionConcept] = Field(
        default_factory=list,
        max_length=MAX_NAMED_QUESTION_CONCEPTS,
        exclude_if=lambda value: not value,
    )

    @field_validator(
        "exposure_levels",
        "exposure_companion_columns",
        "level_label_keys",
        "exact_roster",
        "measurement_audit_columns",
        "required_reader_label_keys",
        "allowed_literature_citation_keys",
        "direct_comparator_literature_keys",
        "comparison_literature_keys",
        "variable_roster",
        "population_concepts",
        "source_applied_inclusion",
        "source_applied_exclusion",
    )
    @classmethod
    def _unique_nonblank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("request rosters must contain unique non-empty values")
        return cleaned

    @property
    def typed_cohort_bounds(self) -> tuple[Optional[float], ...]:
        """The typed bounds a predicate-filtered cohort applies: ages, stay, prediction time."""

        return (self.age_min, self.age_max, self.minimum_icu_hours, self.prediction_time_hours)

    @property
    def cohort_time_zero_hours(self) -> Optional[float]:
        """Hours after ICU admission by which typed cohort eligibility is decided.

        The family's landmark, else its survival suite's (sealed or proposed;
        a suite always states one), else the end of its observation window.
        Only survival requests carry a suite, so every other family keeps the
        landmark-or-window value it had before the suite was read here.
        """

        suite = self.survival_suite
        return (
            self.landmark_hours
            or (suite.landmark_hours if suite is not None else None)
            or self.observation_window_hours
        )

    @property
    def survival_suite(
        self,
    ) -> SealedSuiteCoordinates | SealedContinuousSuiteCoordinates | None:
        """The request's survival suite, sealed or proposed, of either exposure kind."""

        return (
            self.sealed_suite
            or self.proposed_suite
            or self.sealed_continuous_suite
            or self.proposed_continuous_suite
        )

    @property
    def suite_proposal(
        self,
    ) -> SealedSuiteCoordinates | SealedContinuousSuiteCoordinates | None:
        """The survival suite the host proposes, whose roster the Planner selects."""

        return self.proposed_suite or self.proposed_continuous_suite

    @property
    def sealed_survival_suite(
        self,
    ) -> SealedSuiteCoordinates | SealedContinuousSuiteCoordinates | None:
        """The survival suite the host has sealed, of either exposure kind."""

        return self.sealed_suite or self.sealed_continuous_suite

    @model_validator(mode="after")
    def _exposure_shape(self) -> "FamilySpecRequest":
        """A family's exposure kind fixes which level coordinates are meaningful."""

        landmark_fields = (
            self.landmark_hours,
            self.landmark_spec_id,
            self.event_time_column,
            self.observation_duration_column,
            self.observation_duration_unit,
        )
        if self.family_id == SOURCE_FEASIBILITY_FAMILY_ID:
            if self.outcome or self.outcome_levels:
                raise ValueError("the sealed feasibility family analyses no outcome")
        elif self.family_id == FIXED_WINDOW_TRAJECTORY_FAMILY_ID and not self.outcome:
            if self.outcome_levels or self.event_level_index:
                raise ValueError("a trajectory suite without an outcome has no outcome levels")
        elif self.family_id == PHENOTYPING_FAMILY_ID and not self.outcome:
            # Phenotypes are discovered from their features; an outcome the
            # study has is only described by cluster.
            if self.outcome_levels or self.event_level_index:
                raise ValueError("a phenotyping study without an outcome has no outcome levels")
        elif not self.outcome or len(self.outcome_levels) != 2:
            raise ValueError("every result-bearing family needs one two-level outcome")
        if self.family_id == SOURCE_FEASIBILITY_FAMILY_ID:
            if self.analysis_type != "causal_inference":
                raise ValueError("the sealed feasibility family plans a causal-inference study")
            if self.sealed_feasibility is None:
                raise ValueError("the sealed feasibility family needs its sealed coordinates")
            if any(value is not None for value in landmark_fields):
                raise ValueError("the sealed feasibility decision carries no landmark coordinates")
            if self.exposure_kind != "none" or self.primary_exposure:
                raise ValueError("the sealed feasibility decision drafts no exposure contrast")
            if self.adjustment_selection == "exact" and self.exact_roster:
                raise ValueError("the sealed feasibility decision fits no adjusted model")
        elif self.family_id in LANDMARK_FAMILY_IDS:
            if self.analysis_type != "association_study":
                raise ValueError("landmark families plan an association study")
            if any(value is None for value in landmark_fields):
                raise ValueError("a landmark family needs its typed landmark coordinates")
        elif self.family_id == DESCRIPTIVE_FAMILY_ID:
            if self.analysis_type != "descriptive_epidemiology":
                raise ValueError("the descriptive family plans a descriptive study")
            if any(value is not None for value in landmark_fields):
                raise ValueError("the descriptive family carries no landmark coordinates")
            if self.exposure_kind != "categorical":
                raise ValueError("the descriptive family describes a closed-level exposure")
            if self.adjustment_selection == "exact" and self.exact_roster:
                raise ValueError("the descriptive family fits no adjusted model")
        elif self.family_id == LANDMARK_SURVIVAL_FAMILY_ID:
            if self.analysis_type != "survival":
                raise ValueError("the landmark survival family plans a survival study")
            suite = self.sealed_suite or self.proposed_suite
            if suite is None:
                raise ValueError("the landmark survival family needs its sealed suite coordinates")
            if self.sealed_suite is not None and self.proposed_suite is not None:
                raise ValueError("a landmark survival request is either sealed or proposed, not both")
            if any(value is not None for value in landmark_fields):
                raise ValueError("the sealed survival suite owns its landmark coordinates")
            if self.exposure_kind != "categorical" or len(self.exposure_levels) != 2:
                raise ValueError("the sealed survival suite contrasts one binary exposure status")
            if self.primary_exposure != suite.exposure_status_column:
                raise ValueError("the survival exposure must be the sealed exposure status column")
            if self.outcome != suite.event_column:
                raise ValueError("the survival outcome must be the sealed event column")
            if self.sealed_suite is not None and (
                self.adjustment_selection != "exact"
                or self.exact_roster != self.sealed_suite.adjustment_columns
            ):
                raise ValueError("the survival adjustment set is sealed, not selectable")
            if self.proposed_suite is not None and self.proposed_suite.adjustment_columns:
                raise ValueError("a proposed survival suite seals no adjustment roster")
        elif self.family_id == LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID:
            if self.analysis_type != "survival":
                raise ValueError("the landmark survival family plans a survival study")
            suite = self.sealed_continuous_suite or self.proposed_continuous_suite
            if suite is None:
                raise ValueError(
                    "the continuous survival family needs its sealed suite coordinates"
                )
            if (
                self.sealed_continuous_suite is not None
                and self.proposed_continuous_suite is not None
            ):
                raise ValueError("a landmark survival request is either sealed or proposed, not both")
            if any(value is not None for value in landmark_fields):
                raise ValueError("the sealed survival suite owns its landmark coordinates")
            if self.exposure_kind != "continuous":
                raise ValueError("the continuous survival suite models one continuous exposure")
            if self.primary_exposure != suite.exposure_column:
                raise ValueError("the survival exposure must be the suite's exposure column")
            if self.outcome != suite.event_column:
                raise ValueError("the survival outcome must be the sealed event column")
            if self.sealed_continuous_suite is not None and (
                self.adjustment_selection != "exact"
                or self.exact_roster != self.sealed_continuous_suite.adjustment_columns
            ):
                raise ValueError("the survival adjustment set is sealed, not selectable")
            if (
                self.proposed_continuous_suite is not None
                and self.proposed_continuous_suite.adjustment_columns
            ):
                raise ValueError("a proposed survival suite seals no adjustment roster")
        elif self.family_id == FIXED_WINDOW_TRAJECTORY_FAMILY_ID:
            if self.analysis_type != "trajectory_clustering":
                raise ValueError("the trajectory suite family plans a clustering study")
            if self.sealed_trajectory is None:
                raise ValueError("the trajectory suite family needs its sealed coordinates")
            if any(value is not None for value in landmark_fields):
                raise ValueError("the sealed trajectory suite carries no landmark coordinates")
            if self.exposure_kind != "none" or self.primary_exposure:
                raise ValueError("the sealed trajectory suite has no primary exposure")
            if self.adjustment_selection == "exact" and self.exact_roster:
                raise ValueError("the sealed trajectory suite fits no adjusted model")
        elif self.family_id == PHENOTYPING_FAMILY_ID:
            if self.analysis_type != "trajectory_clustering":
                raise ValueError("the phenotyping family plans a clustering study")
            if any(value is not None for value in landmark_fields):
                raise ValueError("the phenotyping family carries no landmark coordinates")
            if not any(item.selectable for item in self.feature_candidates):
                raise ValueError("the phenotyping family needs selectable feature candidates")
            if self.adjustment_selection == "exact" and self.exact_roster:
                raise ValueError("the phenotyping family fits no adjusted model")
        else:
            if self.analysis_type != "prediction_model":
                raise ValueError("the prediction family plans a prediction study")
            if any(value is not None for value in landmark_fields):
                raise ValueError("the prediction family carries no landmark coordinates")
            if self.exposure_kind != "none" or self.primary_exposure:
                raise ValueError("a prediction model has no primary exposure")
            if not any(item.selectable for item in self.feature_candidates):
                raise ValueError("the prediction family needs selectable predictor candidates")
            if self.adjustment_selection == "exact" and self.exact_roster:
                raise ValueError("the prediction family fits no adjusted model")
            if self.prediction_time_hours is not None and (
                self.prediction_time_hours != self.observation_window_hours
                or self.cohort_selection_mode != "predicate_filtered"
            ):
                raise ValueError(
                    "a prediction time is the end of the observation window, and the stays "
                    "still in the ICU after it filter the cohort"
                )
        if (
            self.proposed_suite is not None
            and self.family_id != LANDMARK_SURVIVAL_FAMILY_ID
        ):
            raise ValueError(
                "proposed suite coordinates belong to the landmark survival family"
            )
        if (
            self.prediction_time_hours is not None
            and self.family_id != PREDICTION_FAMILY_ID
        ):
            raise ValueError("a prediction time belongs to the prediction family")
        if (
            self.prediction_death_time is not None
            and self.prediction_time_hours is None
        ):
            raise ValueError("a death reading belongs to a prediction's risk set")
        if self.caller_binds_all_input_rows and (
            self.prediction_time_hours is None or self.population_concepts
        ):
            raise ValueError(
                "a population the caller binds to every input row is filtered only "
                "by a prediction's risk set, and offers no population to state"
            )
        if (
            self.icu_stay_unit != "days"
            and self.minimum_icu_hours is None
            and self.prediction_time_hours is None
        ):
            raise ValueError("an ICU-stay unit belongs to a typed ICU-stay bound")
        if (
            self.sealed_continuous_suite is not None or self.proposed_continuous_suite is not None
        ) and self.family_id != LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID:
            raise ValueError(
                "continuous suite coordinates belong to the continuous survival family"
            )
        if self.sealed_trajectory is not None and self.family_id != FIXED_WINDOW_TRAJECTORY_FAMILY_ID:
            raise ValueError("sealed trajectory coordinates belong to the trajectory suite family")
        if self.sealed_feasibility is not None and self.family_id != SOURCE_FEASIBILITY_FAMILY_ID:
            raise ValueError("sealed feasibility coordinates belong to the feasibility family")
        if self.exposure_kind == "none":
            if self.family_id not in {
                PREDICTION_FAMILY_ID,
                PHENOTYPING_FAMILY_ID,
                FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
                SOURCE_FEASIBILITY_FAMILY_ID,
            }:
                raise ValueError(
                    "only the prediction, phenotyping, sealed trajectory and sealed "
                    "feasibility families have no exposure"
                )
            if self.exposure_levels or self.exposure_is_ordered or self.exposure_companion_columns:
                raise ValueError("an absent exposure carries no levels or companions")
            if self.reference_level_index or self.primary_contrast_level_index:
                raise ValueError("an absent exposure has no level indices")
        elif not self.primary_exposure:
            raise ValueError("a categorical or continuous exposure needs a column name")
        if self.sealed_suite is not None and self.family_id != LANDMARK_SURVIVAL_FAMILY_ID:
            raise ValueError("sealed suite coordinates belong to the landmark survival family")
        if self.exposure_kind == "categorical":
            if self.family_id not in {
                LANDMARK_CATEGORICAL_FAMILY_ID,
                DESCRIPTIVE_FAMILY_ID,
                PHENOTYPING_FAMILY_ID,
                LANDMARK_SURVIVAL_FAMILY_ID,
            }:
                raise ValueError("a categorical exposure belongs to a closed-level family")
            if len(self.exposure_levels) < 2:
                raise ValueError("a categorical exposure needs at least two closed levels")
            if (
                self.reference_level_index >= len(self.exposure_levels)
                or self.primary_contrast_level_index >= len(self.exposure_levels)
                or self.reference_level_index == self.primary_contrast_level_index
            ):
                raise ValueError("reference and primary contrast must index distinct levels")
            if self.exposure_companion_columns:
                raise ValueError("companion columns are a continuous-exposure coordinate")
        elif self.exposure_kind == "continuous":
            if self.family_id not in {
                LANDMARK_SPLINE_FAMILY_ID,
                LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID,
            }:
                raise ValueError(
                    "a continuous exposure belongs to the spline family or the "
                    "continuous survival suite"
                )
            if self.exposure_levels or self.exposure_is_ordered:
                raise ValueError("a continuous exposure carries no closed level set")
            if self.reference_level_index or self.primary_contrast_level_index:
                raise ValueError("a continuous exposure has no level indices")
            if self.alternate_exposures:
                raise ValueError(
                    "alternate exposure definitions are not projected for a continuous exposure"
                )
        occurrence = self.study_population_occurrence
        if occurrence is not None:
            if self.family_id != LANDMARK_CATEGORICAL_FAMILY_ID:
                raise ValueError(
                    "a study-population occurrence belongs to the categorical landmark family"
                )
            if occurrence.product_id != study_population_product_for(self.cohort_name):
                raise ValueError(
                    "the study population is published under the product the locked "
                    "cohort cannot claim"
                )
        return self

    @property
    def request_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        # A request without accepted inputs or an accepted baseline keeps the
        # digest it had before those fields existed.
        for field in ("accepted_feature_groups", "accepted_baseline_rows"):
            if not payload.get(field):
                payload.pop(field, None)
        return canonical_sha256(payload)

    @property
    def selectable_candidates(self) -> tuple[AdjustmentCandidate, ...]:
        return tuple(item for item in self.adjustment_candidates if item.selectable)

    def candidate(self, name: str) -> Optional[AdjustmentCandidate]:
        for item in self.adjustment_candidates:
            if item.name == name:
                return item
        return None


class SpecCovariateDecision(BaseModel):
    """One Planner-selected adjustment covariate with its confounding rationale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    coding: Literal["continuous", "binary", "categorical"]
    reference_level_index: Optional[int] = Field(default=None, ge=0)
    clinical_rationale: str = Field(min_length=_RATIONALE_MIN, max_length=_RATIONALE_MAX)

    @field_validator("clinical_rationale")
    @classmethod
    def _collapse_whitespace(cls, value: str) -> str:
        return " ".join(str(value or "").split())


class SpecReaderLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=256)
    value: str = Field(min_length=1, max_length=256)


class SpecComparatorApplication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")
    application: str = Field(min_length=_APPLICATION_MIN, max_length=_APPLICATION_MAX)


def design_field_max_length(field: str) -> int:
    """The design schema's own bound for one field, read from its single owner."""

    for item in ResearchDesignCandidate.model_fields[field].metadata:
        bound = getattr(item, "max_length", None)
        if bound is not None:
            return int(bound)
    raise ValueError(f"design field {field!r} declares no maximum length")


#: A fit roster must fit the plan it compiles into: the selected design names
#: every variable it needs, and the host adds the row identity and the outcome.
#: A longer roster would pass the Planner and fail only after it was paid for.
MAX_FIT_FEATURES = design_field_max_length("required_variables") - 2


def landmark_design_roster(request: "FamilySpecRequest", covariates: Sequence[str]) -> list[str]:
    """Every variable a landmark design names, in the order the plan lists them.

    The host seals the row identity, exposure, outcome, landmark timing,
    alternate definitions, first-stay flag and audit columns; the adjustment
    roster is the Planner's or the exact user roster.  The spec is checked and
    the design is built from this one list, so a roster the design cannot name
    is refused before it compiles instead of being cut short after.
    """

    return list(
        dict.fromkeys(
            [
                request.identity_column,
                request.primary_exposure,
                request.outcome,
                *(
                    column
                    for column in (request.event_time_column, request.observation_duration_column)
                    if column
                ),
                *covariates,
                *([request.secondary_continuous_outcome] if request.secondary_continuous_outcome else []),
                *(item.execution_variables[0] for item in request.alternate_exposures),
                *([request.first_stay.execution_variables[0]] if request.first_stay else []),
                *request.measurement_audit_columns,
            ]
        )
    )


def planner_selects_adjustment(request: "FamilySpecRequest") -> bool:
    """Whether the Planner selects this request's adjustment roster.

    A landmark association family and a proposed survival suite estimate an
    adjusted association; unless an exact user roster is sealed, the Planner
    selects its roster from the selectable candidates.  The spec contract and
    the written response shape both read this, so they cannot disagree.
    """

    return request.adjustment_selection == "planner_selectable" and (
        request.family_id in LANDMARK_FAMILY_IDS or request.suite_proposal is not None
    )


def table_one_group_column(request: "FamilySpecRequest") -> Optional[str]:
    """The column a family's own Table 1 is grouped by, or ``None`` without one.

    The templates build Table 1 from this, and the request binding checks an
    accepted baseline roster against it before any Provider call.  A sealed
    suite family has none here: its signed owner replaces the plan's steps
    when the plan is bound, so a template table would not reach the plan.
    """

    if request.family_id in LANDMARK_FAMILY_IDS:
        # A continuous exposure has no levels to group by: the baseline table
        # is grouped by the binary outcome and describes the exposure instead.
        if request.exposure_kind == "continuous":
            return request.outcome
        return request.primary_exposure
    if request.family_id == DESCRIPTIVE_FAMILY_ID:
        return request.primary_exposure
    if request.family_id == PREDICTION_FAMILY_ID:
        return request.outcome
    return None


def accepted_baseline_additions(
    request: "FamilySpecRequest", present: Sequence[str]
) -> list[AcceptedBaselineRow]:
    """The accepted baseline rows a template's own Table 1 does not describe.

    A row is described when any of its columns is already in the table; the
    template adds the first column of every other row.
    """

    shown = set(present)
    return [
        row for row in request.accepted_baseline_rows if not shown.intersection(row.columns)
    ]


class SpecPopulation(BaseModel):
    """The population a study states beyond the host's typed cohort bounds.

    Each criterion is written in the words that state it, with the offered
    population concepts that express it, and the predicates apply it, each
    decided by the family's time zero.  A criterion that no offered concept
    expresses carries none: it is stated, and nothing applies it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    criteria: list[ProgressivePopulationCriterion] = Field(min_length=1, max_length=6)
    inclusion: list[ProgressiveCohortPredicate] = Field(default_factory=list, max_length=8)
    exclusion: list[ProgressiveCohortPredicate] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _unique_criteria(self) -> "SpecPopulation":
        stated = [item.criterion.casefold() for item in self.criteria]
        if len(stated) != len(set(stated)):
            raise ValueError("population criteria must be unique")
        return self


class FamilyPlanSpec(BaseModel):
    """The Planner's complete output for one family-spec attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.family_plan_spec/1"] = FAMILY_SPEC_SCHEMA_VERSION
    family_id: FamilyId
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adjustment_set: list[SpecCovariateDecision] = Field(default_factory=list, max_length=24)
    baseline_variables: list[str] = Field(default_factory=list, max_length=16)
    feature_variables: list[str] = Field(default_factory=list, max_length=MAX_FIT_FEATURES)
    cohort_membership_column: Optional[str] = Field(default=None, max_length=128)
    reader_display_labels: list[SpecReaderLabel] = Field(default_factory=list, max_length=64)
    comparator_applications: list[SpecComparatorApplication] = Field(
        default_factory=list, max_length=8
    )
    #: One decision per design dimension when the request carries reviewed
    #: design cards; omitted from the spec digest when absent.
    literature_design_decisions: list[CandidateLiteratureDesignDecision] = Field(
        default_factory=list,
        max_length=len(LITERATURE_DESIGN_DIMENSIONS),
        exclude_if=lambda value: not value,
    )
    roster_decision_note: str = Field(min_length=8, max_length=1200)
    #: The population the question or the study's own cohort wording states
    #: beyond the typed bounds.  Omitted from the spec digest when absent.
    population: Optional[SpecPopulation] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    #: What the question asks for beyond the sealed design, in its words
    #: (``planning.question_requirements``).  Omitted from the spec digest
    #: when empty.
    question_requirements: list[QuestionRequirement] = Field(
        default_factory=list,
        max_length=MAX_QUESTION_REQUIREMENTS,
        exclude_if=lambda value: not value,
    )

    @field_validator("reader_display_labels", mode="before")
    @classmethod
    def _keyed_label_transport(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return [{"key": key, "value": label} for key, label in value.items()]
        return value

    @field_validator("comparator_applications", mode="before")
    @classmethod
    def _keyed_application_transport(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return [
                {"citation_key": key, "application": text}
                for key, text in value.items()
            ]
        return value

    @property
    def labels(self) -> dict[str, str]:
        return {item.key: " ".join(item.value.split()) for item in self.reader_display_labels}

    @property
    def applications(self) -> dict[str, str]:
        return {
            item.citation_key: " ".join(item.application.split())
            for item in self.comparator_applications
        }


def _is_mechanical_label(key: str, value: str) -> bool:
    normalized_value = " ".join(str(value or "").split()).casefold()
    normalized_key = str(key or "").strip().casefold()
    return normalized_value in {normalized_key, normalized_key.replace("_", " ")}


def validate_family_plan_spec(spec: FamilyPlanSpec, request: FamilySpecRequest) -> None:
    """Reject any spec coordinate outside the sealed request; return nothing."""

    if spec.family_id != request.family_id:
        raise FamilySpecError(
            "family_spec_family_mismatch",
            f"spec family {spec.family_id!r} differs from the sealed request",
            path="family_id",
        )
    if spec.request_sha256 != request.request_sha256:
        raise FamilySpecError(
            "family_spec_request_digest_mismatch",
            "spec did not bind the host-sealed request digest",
            path="request_sha256",
        )
    names = [item.name for item in spec.adjustment_set]
    if len(names) != len(set(names)):
        raise FamilySpecError(
            "family_spec_covariate_duplicate",
            "adjustment covariates must be unique",
            path="adjustment_set",
        )
    reserved = {request.primary_exposure, request.outcome, request.identity_column}
    if request.family_id == DESCRIPTIVE_FAMILY_ID and spec.adjustment_set:
        raise FamilySpecError(
            "family_spec_adjustment_not_applicable",
            "the descriptive family fits no adjusted model; propose baseline_variables instead",
            path="adjustment_set",
        )
    if request.family_id in LANDMARK_FAMILY_IDS and spec.baseline_variables:
        raise FamilySpecError(
            "family_spec_baseline_variables_not_applicable",
            "landmark families describe the adjustment roster in Table 1; baseline_variables "
            "belongs to the descriptive and phenotyping families",
            path="baseline_variables",
        )
    if request.family_id in SEALED_SUITE_FAMILY_IDS and request.suite_proposal is None:
        if spec.adjustment_set:
            raise FamilySpecError(
                "family_spec_adjustment_not_applicable",
                "a sealed suite owns its adjustment set; the spec carries only reader "
                "labels and comparator applications",
                path="adjustment_set",
            )
        if spec.baseline_variables:
            raise FamilySpecError(
                "family_spec_baseline_variables_not_applicable",
                "a sealed suite owns its Table 1 roster",
                path="baseline_variables",
            )
    if request.family_id in {PHENOTYPING_FAMILY_ID, PREDICTION_FAMILY_ID}:
        if spec.adjustment_set:
            raise FamilySpecError(
                "family_spec_adjustment_not_applicable",
                f"the {request.family_id} family fits no adjusted model; propose feature_variables instead",
                path="adjustment_set",
            )
        features = [str(value).strip() for value in spec.feature_variables]
        offered = {item.name: item for item in request.feature_candidates}
        if len(features) < 2 or len(features) != len(set(features)):
            raise FamilySpecError(
                "family_spec_feature_roster_invalid",
                "the cluster solution needs at least two distinct fit features",
                path="feature_variables",
            )
        design_limit = design_field_max_length("required_variables")
        roster = (
            (2 if request.outcome else 1)  # the row identity and any outcome
            + (1 if spec.cohort_membership_column else 0)
            + len(spec.baseline_variables)
            + len(features)
        )
        if roster > design_limit:
            raise FamilySpecError(
                "family_spec_roster_exceeds_design",
                f"the design names at most {design_limit} variables including the row identity "
                f"and any outcome; this roster needs {roster}: keep the most informative features "
                "and baseline descriptors",
                path="feature_variables",
            )
        for index, name in enumerate(features):
            candidate = offered.get(name)
            if candidate is None or not candidate.selectable:
                raise FamilySpecError(
                    "family_spec_feature_unavailable",
                    f"{name!r} is not a host-offered fit feature",
                    path=f"feature_variables[{index}]",
                )
        missing_inputs = [
            group.concept
            for group in request.accepted_feature_groups
            if not set(group.columns) & set(features)
        ]
        if missing_inputs:
            raise FamilySpecError(
                "family_spec_accepted_input_missing",
                "the reviewed design keeps every accepted primary input; select at least one "
                "of its columns for: "
                + "; ".join(
                    f"{group.concept} ({', '.join(group.columns)})"
                    for group in request.accepted_feature_groups
                    if group.concept in missing_inputs
                ),
                path="feature_variables",
            )
        if request.family_id == PREDICTION_FAMILY_ID and spec.baseline_variables:
            raise FamilySpecError(
                "family_spec_baseline_variables_not_applicable",
                "the prediction family describes the predictors in Table 1; baseline_variables "
                "belongs to the descriptive and phenotyping families",
                path="baseline_variables",
            )
        if spec.cohort_membership_column is not None and (
            spec.cohort_membership_column not in request.membership_candidates
        ):
            raise FamilySpecError(
                "family_spec_membership_unavailable",
                f"{spec.cohort_membership_column!r} is not a host-offered membership flag",
                path="cohort_membership_column",
            )
    else:
        if spec.feature_variables:
            raise FamilySpecError(
                "family_spec_feature_variables_not_applicable",
                "feature_variables belongs to the phenotyping family",
                path="feature_variables",
            )
        if spec.cohort_membership_column is not None:
            raise FamilySpecError(
                "family_spec_membership_not_applicable",
                "cohort_membership_column belongs to the phenotyping family",
                path="cohort_membership_column",
            )
    baseline = [str(value).strip() for value in spec.baseline_variables]
    if len(baseline) != len(set(baseline)):
        raise FamilySpecError(
            "family_spec_baseline_variable_duplicate",
            "baseline variables must be unique",
            path="baseline_variables",
        )
    for index, name in enumerate(baseline):
        candidate = request.candidate(name)
        if name in reserved or candidate is None or not candidate.selectable:
            raise FamilySpecError(
                "family_spec_baseline_variable_unavailable",
                f"{name!r} is not a host-offered baseline variable",
                path=f"baseline_variables[{index}]",
            )
    if request.adjustment_selection == "exact":
        if names and names != list(request.exact_roster):
            raise FamilySpecError(
                "family_spec_exact_roster_mismatch",
                "an exact user-reviewed roster cannot be added to, removed from, "
                f"or reordered: expected {list(request.exact_roster)!r}",
                path="adjustment_set",
            )
    for index, item in enumerate(spec.adjustment_set):
        path = f"adjustment_set[{index}]"
        if item.name in reserved:
            raise FamilySpecError(
                "family_spec_covariate_reserved",
                f"{item.name!r} is the exposure, outcome, or identity column",
                path=path,
            )
        candidate = request.candidate(item.name)
        if candidate is None:
            raise FamilySpecError(
                "family_spec_covariate_unavailable",
                f"{item.name!r} is outside the sealed candidate roster",
                path=path,
            )
        if request.adjustment_selection != "exact" and not candidate.selectable:
            raise FamilySpecError(
                "family_spec_covariate_timing_unproven",
                f"{item.name!r} is not Planner-selectable: {candidate.boundary}",
                path=path,
            )
        if item.coding not in candidate.allowed_codings:
            raise FamilySpecError(
                "family_spec_covariate_coding_unavailable",
                f"{item.name!r} supports codings {list(candidate.allowed_codings)!r}",
                path=f"{path}.coding",
            )
        treatment = item.coding in {"binary", "categorical"}
        if treatment != (item.reference_level_index is not None):
            raise FamilySpecError(
                "family_spec_reference_index_shape",
                "binary/categorical covariates need a reference index; continuous "
                "covariates must omit it",
                path=f"{path}.reference_level_index",
            )
        if treatment and (
            candidate.closed_domain_size is None
            or int(item.reference_level_index or 0) >= candidate.closed_domain_size
        ):
            raise FamilySpecError(
                "family_spec_reference_index_out_of_domain",
                f"{item.name!r} reference index exceeds its closed domain",
                path=f"{path}.reference_level_index",
            )
    selectable = [item.name for item in request.selectable_candidates]
    if not names and selectable and planner_selects_adjustment(request):
        # An adjusted-association family whose roster is the Planner's: an
        # empty set is no adjusted estimate and leaves its Table 1 nothing
        # to describe, so the Planner repairs it rather than the build failing.
        raise FamilySpecError(
            "family_spec_adjustment_set_empty",
            "this family estimates an adjusted association and the host offers "
            f"selectable confounders ({', '.join(selectable[:8])}"
            f"{', ...' if len(selectable) > 8 else ''}): select those that can cause both "
            "the exposure and the outcome and are fixed before time zero, each with "
            "its clinical_rationale; an empty adjustment set is not an adjusted estimate",
            path="adjustment_set",
        )
    proposal = request.suite_proposal
    if proposal is not None:
        if spec.baseline_variables:
            raise FamilySpecError(
                "family_spec_baseline_variables_not_applicable",
                "the survival suite describes its adjustment roster in Table 1",
                path="baseline_variables",
            )
        design_limit = design_field_max_length("required_variables")
        roster = [request.identity_column, *proposal.source_columns, *names]
        if len(roster) > design_limit:
            raise FamilySpecError(
                "family_spec_roster_exceeds_design",
                f"the design names at most {design_limit} variables including the row identity, "
                f"exposure, endpoint and follow-up columns; this roster needs "
                f"{len(roster)}: keep the adjustment covariates that matter most",
                path="adjustment_set",
            )
    if request.family_id in LANDMARK_FAMILY_IDS:
        covariates = list(request.exact_roster) if request.adjustment_selection == "exact" else names
        design_limit = design_field_max_length("required_variables")
        roster = landmark_design_roster(request, covariates)
        if len(roster) > design_limit:
            raise FamilySpecError(
                "family_spec_roster_exceeds_design",
                f"the design names at most {design_limit} variables including the row identity, "
                f"exposure, outcome, landmark timing and audit columns; this roster needs {len(roster)}: "
                "keep the adjustment covariates that matter most",
                path="adjustment_set",
            )
    labels = spec.labels
    selected_label_keys = [
        *request.required_reader_label_keys,
        # Variables the Planner placed in the design need reader labels too;
        # the request cannot list them because it is sealed before selection.
        *(spec.feature_variables if request.family_id in {PHENOTYPING_FAMILY_ID, PREDICTION_FAMILY_ID} else []),
        *(spec.baseline_variables if request.family_id == PHENOTYPING_FAMILY_ID else []),
        # A proposed survival suite's roster is the Planner's selection.
        *(names if request.suite_proposal is not None else []),
        # The study-population distribution names each exposure level.
        *(request.level_label_keys if request.study_population_occurrence is not None else []),
    ]
    for key in dict.fromkeys(selected_label_keys):
        value = labels.get(key, "")
        if not value or _is_mechanical_label(key, value):
            raise FamilySpecError(
                "family_spec_reader_label_missing",
                f"a concise clinical reader label is required for {key!r}",
                path="reader_display_labels",
            )
    if request.study_population_occurrence is not None and len(
        {" ".join(labels[key].split()).casefold() for key in request.level_label_keys}
    ) != len(request.level_label_keys):
        raise FamilySpecError(
            "family_spec_level_labels_not_distinct",
            "each exposure level needs its own reader label",
            path="reader_display_labels",
        )
    unknown_labels = sorted(
        set(labels) - set(request.variable_roster) - set(request.level_label_keys)
    )
    if unknown_labels:
        raise FamilySpecError(
            "family_spec_reader_label_unavailable",
            f"labels name variables outside the sealed roster: {unknown_labels!r}",
            path="reader_display_labels",
        )
    applications = spec.applications
    allowed_keys = set(request.allowed_literature_citation_keys)
    unknown_keys = sorted(set(applications) - allowed_keys)
    if unknown_keys:
        raise FamilySpecError(
            "family_spec_citation_unavailable",
            f"comparator applications cite keys outside the sealed roster: {unknown_keys!r}",
            path="comparator_applications",
        )
    missing = [
        key for key in request.direct_comparator_literature_keys if key not in applications
    ]
    if missing:
        raise FamilySpecError(
            "family_spec_comparator_application_missing",
            "every screened direct comparator needs one application statement: "
            + ", ".join(missing),
            path="comparator_applications",
        )
    _validate_literature_design_decisions(spec, request)
    _validate_population(spec, request)
    _validate_question_requirements(spec, request)


def _validate_question_requirements(
    spec: FamilyPlanSpec, request: FamilySpecRequest
) -> None:
    """The question's requirements quote it, read offered concepts, and account for its names."""

    problems = question_requirement_problems(
        spec.question_requirements,
        question=request.research_question,
        roster=request.variable_roster,
        named=request.question_named_concepts,
        sealed=(request.primary_exposure, request.outcome),
    )
    if problems:
        first = problems[0]
        raise FamilySpecError(first.code, first.message, path=first.path)


def population_required(request: FamilySpecRequest) -> bool:
    """Whether only a population the Planner states can filter this cohort.

    The caller binds a predicate-filtered cohort (a reviewed candidate chose
    one), no typed bound applies, and population concepts are offered.  A
    phenotyping plan may filter by its membership flag instead.
    """

    return (
        bool(request.population_concepts)
        and request.cohort_selection_mode == "predicate_filtered"
        and all(value is None for value in request.typed_cohort_bounds)
    )


def _validate_population(spec: FamilyPlanSpec, request: FamilySpecRequest) -> None:
    """A stated population reads offered concepts only, by time zero, and is applied.

    Every predicate applies a stated criterion, and every criterion that has
    concepts is applied by at least one predicate over one of them, so no
    restriction goes unstated and no stated restriction goes unapplied.  When
    only the stated population can filter a caller-bound filtered cohort, it
    must apply a predicate.
    """

    population = spec.population
    if (
        population_required(request)
        and spec.cohort_membership_column is None
        and not (population is not None and (population.inclusion or population.exclusion))
    ):
        raise FamilySpecError(
            "family_spec_population_required",
            "the caller binds a predicate-filtered cohort and no typed age or stay bound "
            "applies, so only the population you state can filter it: state the "
            "population the question or the study's wording names, with at least one "
            "inclusion or exclusion predicate",
            path="population",
        )
    if population is None:
        return
    if not request.population_concepts:
        raise FamilySpecError(
            "family_spec_population_not_applicable",
            "this request offers no population concepts: the caller binds every input "
            "row, or the family applies no plan cohort by a time zero; population must "
            "be null",
            path="population",
        )
    offered = set(request.population_concepts)
    stated: set[str] = set()
    for index, item in enumerate(population.criteria):
        unknown = [concept for concept in item.concept_ids if concept not in offered]
        if unknown:
            raise FamilySpecError(
                "family_spec_population_concept_unavailable",
                f"criterion {item.criterion!r} names concepts that are not offered: "
                f"{unknown!r}",
                path=f"population.criteria[{index}].concept_ids",
            )
        stated.update(item.concept_ids)
    time_zero = request.cohort_time_zero_hours
    for side in ("inclusion", "exclusion"):
        for index, predicate in enumerate(getattr(population, side)):
            path = f"population.{side}[{index}]"
            if predicate.concept_id not in offered:
                raise FamilySpecError(
                    "family_spec_population_concept_unavailable",
                    f"{predicate.concept_id!r} is not an offered population concept",
                    path=f"{path}.concept_id",
                )
            if predicate.concept_id not in stated:
                raise FamilySpecError(
                    "family_spec_population_predicate_unstated",
                    f"no criterion names {predicate.concept_id!r}; state the restriction "
                    "this predicate applies",
                    path=path,
                )
            if predicate.anchor != POPULATION_ANCHOR:
                raise FamilySpecError(
                    "family_spec_population_anchor_unavailable",
                    f"a population predicate counts from {POPULATION_ANCHOR}, the anchor "
                    "of the family's time zero",
                    path=f"{path}.anchor",
                )
            if time_zero is None or predicate.end_offset_hours > time_zero:
                raise FamilySpecError(
                    "family_spec_population_after_time_zero",
                    "a population predicate must be decided by time zero "
                    f"({time_zero if time_zero is not None else 'unknown'} h after ICU "
                    f"admission); its window ends at {predicate.end_offset_hours:g} h",
                    path=f"{path}.end_offset_hours",
                )
    read = {
        predicate.concept_id for predicate in (*population.inclusion, *population.exclusion)
    }
    for index, item in enumerate(population.criteria):
        if item.concept_ids and not read.intersection(item.concept_ids):
            raise FamilySpecError(
                "family_spec_population_criterion_unapplied",
                f"criterion {item.criterion!r} names {item.concept_ids!r}, but no inclusion "
                "or exclusion predicate reads one of them; apply it, or give it no "
                "concepts only when no offered concept expresses it over the "
                "window the criterion states",
                path=f"population.criteria[{index}]",
            )


def literature_design_card_keys_by_dimension(
    request: FamilySpecRequest,
) -> dict[str, list[str]]:
    """The sealed card keys that state each design dimension, in card order."""

    return {
        dimension: [
            card.citation_key
            for card in request.literature_design_cards
            if any(item.dimension == dimension for item in card.evidence)
        ]
        for dimension in LITERATURE_DESIGN_DIMENSIONS
    }


def _validate_literature_design_decisions(
    spec: FamilyPlanSpec, request: FamilySpecRequest
) -> None:
    decisions = spec.literature_design_decisions
    path = "literature_design_decisions"
    if not request.literature_design_cards:
        if decisions:
            raise FamilySpecError(
                "family_spec_literature_decision_unrequested",
                "no reviewed design card is sealed into this request",
                path=path,
            )
        return
    dimensions = [item.dimension for item in decisions]
    missing = [item for item in LITERATURE_DESIGN_DIMENSIONS if item not in dimensions]
    if missing or len(dimensions) != len(set(dimensions)):
        raise FamilySpecError(
            "family_spec_literature_decision_missing",
            "give exactly one decision for each design dimension; missing: "
            + ", ".join(missing),
            path=path,
        )
    stated = literature_design_card_keys_by_dimension(request)
    unsupported = sorted(
        (item.dimension, key)
        for item in decisions
        for key in item.citation_keys
        if key not in stated[item.dimension]
    )
    if unsupported:
        raise FamilySpecError(
            "family_spec_literature_decision_source_unsupported",
            "each decision cites only a sealed card that states its dimension: "
            f"{unsupported!r}",
            path=path,
        )


def spec_from_mapping(payload: Mapping[str, Any]) -> FamilyPlanSpec:
    """Parse one provider payload into the typed spec (pydantic errors propagate)."""

    return FamilyPlanSpec.model_validate(dict(payload))


__all__ = [
    "AcceptedBaselineRow",
    "AcceptedFeatureGroup",
    "FAMILY_SPEC_REQUEST_SCHEMA_VERSION",
    "FAMILY_SPEC_SCHEMA_VERSION",
    "DESCRIPTIVE_FAMILY_ID",
    "LANDMARK_CATEGORICAL_FAMILY_ID",
    "LANDMARK_CONTINUOUS_SURVIVAL_FAMILY_ID",
    "LANDMARK_FAMILY_IDS",
    "LANDMARK_SPLINE_FAMILY_ID",
    "PHENOTYPING_FAMILY_ID",
    "POPULATION_ANCHOR",
    "POPULATION_FAMILY_IDS",
    "PREDICTION_FAMILY_ID",
    "SOURCE_FEASIBILITY_FAMILY_ID",
    "AdjustmentCandidate",
    "ExposureKind",
    "FamilyId",
    "FamilyPlanSpec",
    "FamilySpecError",
    "FamilySpecRequest",
    "SealedContinuousSuiteCoordinates",
    "SealedFeasibilityCoordinates",
    "SensitivityAxisBinding",
    "SpecComparatorApplication",
    "SpecCovariateDecision",
    "SpecPopulation",
    "SpecReaderLabel",
    "StudyPopulationOccurrence",
    "accepted_baseline_additions",
    "literature_design_card_keys_by_dimension",
    "population_required",
    "PredictionDeathTime",
    "PredictionDeathTimeReason",
    "prediction_risk_set_predicates",
    "sealed_cohort_predicate",
    "spec_from_mapping",
    "table_one_group_column",
    "validate_family_plan_spec",
]
