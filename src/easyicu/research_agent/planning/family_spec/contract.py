"""Typed contracts for family-spec planning.

``FamilySpecRequest`` is host authority: every selectable coordinate the
Planner may fill, sealed and hashed before any provider call.  ``FamilyPlanSpec``
is the Planner's whole output for one attempt.  ``validate_family_plan_spec``
is the fail-closed boundary between the two; it rejects any name, coding,
level index, label, or citation outside the sealed request, and a rationale
never widens the host's timing authority.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...canonical_json import canonical_sha256
from ..adjustment_authority import HostTemporalRole
from ..progressive_contract import ModelTermCoding

FAMILY_SPEC_SCHEMA_VERSION = "easyicu.family_plan_spec/1"
FAMILY_SPEC_REQUEST_SCHEMA_VERSION = "easyicu.family_spec_request/1"
LANDMARK_CATEGORICAL_FAMILY_ID = "landmark_categorical_association"
LANDMARK_SPLINE_FAMILY_ID = "landmark_spline_association"
DESCRIPTIVE_FAMILY_ID = "descriptive_exposure_outcome"
PHENOTYPING_FAMILY_ID = "cross_sectional_phenotyping"
PREDICTION_FAMILY_ID = "static_prediction_model"
LANDMARK_SURVIVAL_FAMILY_ID = "landmark_survival_suite"
FIXED_WINDOW_TRAJECTORY_FAMILY_ID = "fixed_window_trajectory_suite"
SOURCE_FEASIBILITY_FAMILY_ID = "source_feasibility_fail_closed"
LANDMARK_FAMILY_IDS = frozenset({LANDMARK_CATEGORICAL_FAMILY_ID, LANDMARK_SPLINE_FAMILY_ID})
#: Families whose every scientific coordinate is a sealed runtime authority's;
#: the Planner only labels columns and writes comparator applications.
SEALED_SUITE_FAMILY_IDS = frozenset(
    {
        LANDMARK_SURVIVAL_FAMILY_ID,
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
    "fixed_window_trajectory_suite",
    "source_feasibility_fail_closed",
]
ExposureKind = Literal["categorical", "continuous", "none"]

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
    identity_column: str = Field(min_length=1, max_length=128)
    cluster_unit: Optional[Literal["patient"]] = None
    primary_exposure: str = Field(max_length=128)
    exposure_kind: ExposureKind = "categorical"
    exposure_levels: list[str] = Field(default_factory=list, max_length=24)
    reference_level_index: int = Field(ge=0)
    primary_contrast_level_index: int = Field(ge=0)
    exposure_is_ordered: bool
    exposure_companion_columns: list[str] = Field(default_factory=list)
    #: Empty only for the sealed feasibility family, which analyses no outcome.
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
    membership_candidates: list[str] = Field(default_factory=list)
    secondary_continuous_outcome: Optional[str] = Field(default=None, max_length=128)
    sealed_suite: Optional[SealedSuiteCoordinates] = None
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
    variable_roster: list[str] = Field(min_length=1)

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
    )
    @classmethod
    def _unique_nonblank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("request rosters must contain unique non-empty values")
        return cleaned

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
            if self.sealed_suite is None:
                raise ValueError("the landmark survival family needs its sealed suite coordinates")
            if any(value is not None for value in landmark_fields):
                raise ValueError("the sealed survival suite owns its landmark coordinates")
            if self.exposure_kind != "categorical" or len(self.exposure_levels) != 2:
                raise ValueError("the sealed survival suite contrasts one binary exposure status")
            if self.primary_exposure != self.sealed_suite.exposure_status_column:
                raise ValueError("the survival exposure must be the sealed exposure status column")
            if self.outcome != self.sealed_suite.event_column:
                raise ValueError("the survival outcome must be the sealed event column")
            if (
                self.adjustment_selection != "exact"
                or self.exact_roster != self.sealed_suite.adjustment_columns
            ):
                raise ValueError("the survival adjustment set is sealed, not selectable")
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
        if self.sealed_trajectory is not None and self.family_id != FIXED_WINDOW_TRAJECTORY_FAMILY_ID:
            raise ValueError("sealed trajectory coordinates belong to the trajectory suite family")
        if self.sealed_feasibility is not None and self.family_id != SOURCE_FEASIBILITY_FAMILY_ID:
            raise ValueError("sealed feasibility coordinates belong to the feasibility family")
        if self.exposure_kind == "none":
            if self.family_id not in {
                PREDICTION_FAMILY_ID,
                FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
                SOURCE_FEASIBILITY_FAMILY_ID,
            }:
                raise ValueError(
                    "only the prediction, sealed trajectory and sealed feasibility "
                    "families have no exposure"
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
            if self.family_id != LANDMARK_SPLINE_FAMILY_ID:
                raise ValueError("a continuous exposure belongs to the spline family")
            if self.exposure_levels or self.exposure_is_ordered:
                raise ValueError("a continuous exposure carries no closed level set")
            if self.reference_level_index or self.primary_contrast_level_index:
                raise ValueError("a continuous exposure has no level indices")
            if self.alternate_exposures:
                raise ValueError(
                    "alternate exposure definitions are not projected for the spline family"
                )
        return self

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))

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


class FamilyPlanSpec(BaseModel):
    """The Planner's complete output for one family-spec attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.family_plan_spec/1"] = FAMILY_SPEC_SCHEMA_VERSION
    family_id: FamilyId
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adjustment_set: list[SpecCovariateDecision] = Field(default_factory=list, max_length=24)
    baseline_variables: list[str] = Field(default_factory=list, max_length=16)
    feature_variables: list[str] = Field(default_factory=list, max_length=64)
    cohort_membership_column: Optional[str] = Field(default=None, max_length=128)
    reader_display_labels: list[SpecReaderLabel] = Field(default_factory=list, max_length=64)
    comparator_applications: list[SpecComparatorApplication] = Field(
        default_factory=list, max_length=8
    )
    roster_decision_note: str = Field(min_length=8, max_length=1200)

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
    if request.family_id in SEALED_SUITE_FAMILY_IDS:
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
        for index, name in enumerate(features):
            candidate = offered.get(name)
            if candidate is None or not candidate.selectable:
                raise FamilySpecError(
                    "family_spec_feature_unavailable",
                    f"{name!r} is not a host-offered fit feature",
                    path=f"feature_variables[{index}]",
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
    labels = spec.labels
    selected_label_keys = [
        *request.required_reader_label_keys,
        # Variables the Planner placed in the design need reader labels too;
        # the request cannot list them because it is sealed before selection.
        *(spec.feature_variables if request.family_id in {PHENOTYPING_FAMILY_ID, PREDICTION_FAMILY_ID} else []),
        *(spec.baseline_variables if request.family_id == PHENOTYPING_FAMILY_ID else []),
    ]
    for key in dict.fromkeys(selected_label_keys):
        value = labels.get(key, "")
        if not value or _is_mechanical_label(key, value):
            raise FamilySpecError(
                "family_spec_reader_label_missing",
                f"a concise clinical reader label is required for {key!r}",
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


def spec_from_mapping(payload: Mapping[str, Any]) -> FamilyPlanSpec:
    """Parse one provider payload into the typed spec (pydantic errors propagate)."""

    return FamilyPlanSpec.model_validate(dict(payload))


__all__ = [
    "FAMILY_SPEC_REQUEST_SCHEMA_VERSION",
    "FAMILY_SPEC_SCHEMA_VERSION",
    "DESCRIPTIVE_FAMILY_ID",
    "LANDMARK_CATEGORICAL_FAMILY_ID",
    "LANDMARK_FAMILY_IDS",
    "LANDMARK_SPLINE_FAMILY_ID",
    "PHENOTYPING_FAMILY_ID",
    "PREDICTION_FAMILY_ID",
    "SOURCE_FEASIBILITY_FAMILY_ID",
    "AdjustmentCandidate",
    "ExposureKind",
    "FamilyId",
    "FamilyPlanSpec",
    "FamilySpecError",
    "FamilySpecRequest",
    "SealedFeasibilityCoordinates",
    "SensitivityAxisBinding",
    "SpecComparatorApplication",
    "SpecCovariateDecision",
    "SpecReaderLabel",
    "spec_from_mapping",
    "validate_family_plan_spec",
]
