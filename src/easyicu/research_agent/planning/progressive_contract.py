"""Typed high-level contract for Progressive Planner v2.

The model declares scientific intent and choices.  It does not spell the
complete executable :class:`AnalysisPlan` DAG: product wiring, closed observed
levels, deterministic method tokens, consumption contracts, and host-owned
specification objects are compiled later by ``progressive_compiler``.

This contract is deliberately case-neutral.  Exact variables, products,
citations, and task requirements are supplied at run time and are never baked
into the shared schema or prompt.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping, Optional, Sequence, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..canonical_json import canonical_sha256
from ..contracts.product_identity import is_canonical_typed_product_token


ProgressiveModuleId = Literal[
    "cohort_definition",
    "table_one",
    "exposure_outcome_distribution",
    "measurement_audit",
    "adjusted_association",
    "robustness_replay",
    "custom_analysis",
    "visualization",
    "report",
]


def progressive_module_ids_for_analysis_types(
    analysis_types: Sequence[str],
) -> tuple[str, ...]:
    """Return the union of modules the selected analysis families can execute.

    Descriptive epidemiology has no fitted primary effect or uncertainty
    interval.  Its outline must therefore not advertise the adjusted-model or
    locked-effect replay owners that require those quantities.
    """

    normalized = {
        str(value or "").strip().casefold()
        for value in analysis_types
        if str(value or "").strip()
    }
    modules = list(get_args(ProgressiveModuleId))
    if normalized and normalized <= {"descriptive_epidemiology"}:
        modules = [
            module
            for module in modules
            if module not in {"adjusted_association", "robustness_replay"}
        ]
    return tuple(modules)
ProgressiveOutputRole = Literal[
    "analysis_cohort",
    "cohort_flow",
    "table_one",
    "exposure_outcome_distribution",
    "measurement_missingness",
    "missingness_profile",
    "measurement_source",
    "measurement_process",
    "event_timing",
    "component_completeness",
    "analytic_denominators",
    "adjusted_association_estimates",
    "adjusted_association_model",
    "scientific_sensitivity",
    "robustness_matrix",
    "robustness_summary",
    "specification_grid",
    "membership_change",
    "outcome_label_executability",
    "missingness_strategy_notes",
    "primary_effect",
    "complete_case_n",
    "figure",
    "report",
    "custom",
]
TableOneMode = Literal["independent_inference", "descriptive_smd_only"]
OutcomeType = Literal["binary", "continuous"]
ModelTermCoding = Literal[
    "continuous",
    "binary",
    "categorical",
    "ordinal_linear",
]


class ProgressiveDisplayLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=256)
    value: str = Field(min_length=1, max_length=256)


class ProgressivePredicateValue(BaseModel):
    """Closed representation of one CTAS predicate value.

    Exactly one representation is active.  Splitting scalar and list fields
    avoids an unconstrained ``Any`` in the strict provider schema.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["none", "string", "number", "boolean", "string_list", "number_list"]
    string_value: Optional[str] = None
    number_value: Optional[float] = None
    boolean_value: Optional[bool] = None
    string_list: list[str] = Field(default_factory=list)
    number_list: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_representation(self) -> "ProgressivePredicateValue":
        active = {
            "string": self.string_value is not None,
            "number": self.number_value is not None,
            "boolean": self.boolean_value is not None,
            "string_list": bool(self.string_list),
            "number_list": bool(self.number_list),
        }
        if self.mode == "none":
            if any(active.values()):
                raise ValueError("predicate value mode='none' must carry no value")
            return self
        if not active.get(self.mode, False) or sum(active.values()) != 1:
            raise ValueError(
                "predicate value must populate exactly the field selected by mode"
            )
        return self

    def materialize(self) -> object:
        if self.mode == "none":
            return None
        if self.mode == "string":
            return self.string_value
        if self.mode == "number":
            return self.number_value
        if self.mode == "boolean":
            return self.boolean_value
        if self.mode == "string_list":
            return self.string_list
        return self.number_list


class ProgressiveCohortPredicate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: str = Field(min_length=1, max_length=128)
    anchor: str = Field(min_length=1, max_length=128)
    start_offset_hours: float
    end_offset_hours: float
    aggregation: Literal[
        "max",
        "min",
        "mean",
        "median",
        "last",
        "first",
        "any",
        "all",
        "count",
        "sum",
    ]
    op: Literal[
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "in",
        "not_in",
        "missing",
        "not_missing",
    ]
    value: ProgressivePredicateValue

    @model_validator(mode="after")
    def _closed_window_and_value(self) -> "ProgressiveCohortPredicate":
        if self.end_offset_hours <= self.start_offset_hours:
            raise ValueError("cohort predicate end offset must exceed start offset")
        if self.op in {"missing", "not_missing"} and self.value.mode != "none":
            raise ValueError(
                "missing/not_missing predicates must use value mode='none'"
            )
        if self.op not in {"missing", "not_missing"} and self.value.mode == "none":
            raise ValueError(f"cohort predicate op={self.op!r} requires a value")
        if self.op in {"in", "not_in"} and self.value.mode not in {
            "string_list",
            "number_list",
        }:
            raise ValueError("in/not_in predicates require a list value")
        return self


class ProgressiveCohortIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    selection_mode: Literal["all_input_rows", "predicate_filtered"]
    inclusion: list[ProgressiveCohortPredicate] = Field(default_factory=list)
    exclusion: list[ProgressiveCohortPredicate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _selection_is_explicit(self) -> "ProgressiveCohortIntent":
        if self.selection_mode == "all_input_rows" and (
            self.inclusion or self.exclusion
        ):
            raise ValueError(
                "all_input_rows cohort intent must have empty predicate lists"
            )
        if self.selection_mode == "predicate_filtered" and not (
            self.inclusion or self.exclusion
        ):
            raise ValueError(
                "predicate_filtered cohort intent requires at least one predicate"
            )
        return self


class ProgressiveRobustnessIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    axis: Literal["cohort", "missing", "outcome"]
    description: str = Field(min_length=8, max_length=600)
    missing_strategy: Literal["none", "complete_case"] = "none"
    complete_case_variables: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _missing_strategy_shape(self) -> "ProgressiveRobustnessIntent":
        if self.missing_strategy == "complete_case":
            if self.axis != "missing" or not self.complete_case_variables:
                raise ValueError(
                    "complete_case robustness requires axis='missing' and variables"
                )
        elif self.complete_case_variables:
            raise ValueError(
                "complete_case_variables require missing_strategy='complete_case'"
            )
        return self


class ProgressiveOutputIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    product_id: str = Field(pattern=r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$")
    semantic_role: ProgressiveOutputRole

    @field_validator("product_id")
    @classmethod
    def _canonical_product(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not is_canonical_typed_product_token(cleaned):
            raise ValueError("product_id must be one canonical kind:product token")
        return cleaned


class ProgressiveProductRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    producer_step_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{0,79}$")
    product_id: str = Field(pattern=r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$")


class ProgressiveTableOneVariable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    summary: Literal["mean_sd", "median_iqr", "both", "count_percent"]


class ProgressiveModelTermIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    role: Literal["exposure", "covariate"]
    coding: ModelTermCoding
    reference_level_index: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _reference_matches_coding(self) -> "ProgressiveModelTermIntent":
        treatment = self.coding in {"binary", "categorical"}
        if treatment != (self.reference_level_index is not None):
            raise ValueError(
                "binary/categorical terms require a reference index; continuous/"
                "ordinal_linear terms must omit it"
            )
        return self


class ProgressiveLiteratureBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")
    design_elements: list[
        Literal[
            "population",
            "time_zero",
            "exposure",
            "outcome",
            "estimand",
            "adjustment",
            "dependence",
            "missing_data",
            "robustness",
            "reporting",
        ]
    ] = Field(min_length=1)
    application: str = Field(min_length=8, max_length=1200)
    divergence: Optional[str] = Field(default=None, max_length=1200)

    @field_validator("design_elements")
    @classmethod
    def _unique_elements(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("literature design elements must be unique")
        return values


class ProgressiveKnowHowDecision(BaseModel):
    """Claim-level disposition for one host-retrieved protocol card."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    card_version: str = Field(pattern=r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
    card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    disposition: Literal["adopted", "rejected", "unresolved", "requires_confirmation"]
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    rationale: str = Field(min_length=1, max_length=500)
    citation_ids: list[str] = Field(min_length=1, max_length=8)

    @field_validator("citation_ids")
    @classmethod
    def _unique_citation_ids(cls, values: list[str]) -> list[str]:
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{2,79}", item) for item in values):
            raise ValueError("know-how citation ids must be stable lowercase ids")
        if len(values) != len(set(values)):
            raise ValueError("know-how citation ids must be unique")
        return values


class ProgressiveOutlineStep(BaseModel):
    """One concise scientific step before any executable detail is requested."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{0,79}$")
    planned_analysis_role: Literal["primary", "secondary", "sensitivity", "auxiliary"]
    module_id: ProgressiveModuleId
    objective: str = Field(min_length=8, max_length=600)
    depends_on: list[str] = Field(default_factory=list)
    variable_names: list[str] = Field(min_length=1, max_length=24)
    literature_citation_keys: list[str] = Field(default_factory=list, max_length=12)
    scientific_action_id: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$",
    )

    @field_validator("depends_on", "variable_names", "literature_citation_keys")
    @classmethod
    def _unique_roster(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("outline rosters must contain unique non-empty values")
        return cleaned


class ProgressivePlanOutline(BaseModel):
    """Small retrieval-informed plan; the host materializes one step at a time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_plan_outline/1"] = (
        "easyicu.progressive_plan_outline/1"
    )
    analysis_type: str = Field(min_length=1, max_length=128)
    cohort_objective: str = Field(min_length=8, max_length=600)
    steps: list[ProgressiveOutlineStep] = Field(min_length=1, max_length=24)
    rationale: str = Field(min_length=8, max_length=1200)

    @model_validator(mode="after")
    def _closed_dag(self) -> "ProgressivePlanOutline":
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("progressive outline step ids must be unique")
        positions = {step_id: index for index, step_id in enumerate(step_ids)}
        for index, step in enumerate(self.steps):
            for dependency in step.depends_on:
                if dependency not in positions:
                    raise ValueError(
                        f"step {step.step_id!r} depends on unknown step {dependency!r}"
                    )
                if positions[dependency] >= index:
                    raise ValueError(
                        f"step {step.step_id!r} dependency {dependency!r} must precede it"
                    )
        primary = [
            step.step_id
            for step in self.steps
            if step.planned_analysis_role == "primary"
        ]
        if len(primary) > 1:
            raise ValueError("progressive outline may declare at most one primary step")
        return self


class ProgressiveSkeletonStep(BaseModel):
    """Strict detail for only the step currently being materialized by the host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{0,79}$")
    planned_analysis_role: Literal["primary", "secondary", "sensitivity", "auxiliary"]
    module_id: ProgressiveModuleId
    objective: str = Field(min_length=8, max_length=600)
    depends_on: list[str] = Field(default_factory=list)
    raw_inputs: list[str] = Field(default_factory=list)
    product_inputs: list[ProgressiveProductRef] = Field(default_factory=list)
    outputs: list[ProgressiveOutputIntent] = Field(default_factory=list)
    scientific_action_id: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$",
    )
    custom_method: Optional[str] = Field(default=None, min_length=1, max_length=128)
    table_one_group_by: Optional[str] = Field(default=None, max_length=128)
    table_one_mode: Optional[TableOneMode] = None
    table_one_variables: list[ProgressiveTableOneVariable] = Field(default_factory=list)
    primary_exposure: Optional[str] = Field(default=None, max_length=128)
    outcome: Optional[str] = Field(default=None, max_length=128)
    outcome_type: Optional[OutcomeType] = None
    model_terms: list[ProgressiveModelTermIntent] = Field(default_factory=list)
    event_level_index: Optional[int] = Field(default=None, ge=0)
    reference_exposure_level_index: Optional[int] = Field(default=None, ge=0)
    comparison_exposure_level_index: Optional[int] = Field(default=None, ge=0)
    primary_contrast_level_index: Optional[int] = Field(default=None, ge=0)
    denominator_policy: Optional[
        Literal["all_declared_rows", "observed_outcome_rows"]
    ] = None
    missing_exposure_policy: Optional[
        Literal["fail_closed", "exclude_from_denominator"]
    ] = None
    missing_outcome_policy: Optional[
        Literal[
            "fail_closed",
            "exclude_from_denominator",
            "structural_absence_is_non_event",
        ]
    ] = None
    confidence_level: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    sensitivity_spec_ids: list[str] = Field(default_factory=list)
    literature_bindings: list[ProgressiveLiteratureBinding] = Field(
        default_factory=list
    )

    @field_validator("depends_on", "raw_inputs", "sensitivity_spec_ids")
    @classmethod
    def _unique_nonblank_strings(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("step string rosters must contain unique non-empty values")
        return cleaned

    @model_validator(mode="after")
    def _module_shape(self) -> "ProgressiveSkeletonStep":
        output_ids = [item.product_id for item in self.outputs]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("step output product ids must be unique")
        if self.module_id == "table_one":
            if not (
                self.table_one_group_by
                and self.table_one_mode
                and self.table_one_variables
            ):
                raise ValueError(
                    "table_one module requires group, mode, and variable roster"
                )
        elif self.table_one_group_by or self.table_one_mode or self.table_one_variables:
            raise ValueError("Table 1 fields belong only to module_id='table_one'")
        if self.module_id == "adjusted_association":
            if not (
                self.primary_exposure
                and self.outcome
                and self.outcome_type
                and self.model_terms
            ):
                raise ValueError(
                    "adjusted_association requires exposure, outcome, type, and terms"
                )
        if self.module_id == "exposure_outcome_distribution":
            required = (
                self.primary_exposure,
                self.outcome,
                self.event_level_index,
                self.reference_exposure_level_index,
                self.comparison_exposure_level_index,
                self.denominator_policy,
                self.missing_exposure_policy,
                self.missing_outcome_policy,
                self.confidence_level,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "exposure_outcome_distribution requires variables, level "
                    "indices, denominator/missingness policies, and confidence"
                )
        if self.module_id == "custom_analysis" and not self.custom_method:
            raise ValueError("custom_analysis requires custom_method")
        if self.module_id != "custom_analysis" and self.custom_method is not None:
            raise ValueError("custom_method belongs only to custom_analysis")
        if self.module_id == "visualization" and not (
            self.product_inputs or self.depends_on
        ):
            raise ValueError("visualization requires an upstream product source")
        if (
            self.module_id
            in {
                "measurement_audit",
                "custom_analysis",
                "visualization",
                "report",
            }
            and not self.outputs
        ):
            raise ValueError(f"module {self.module_id!r} requires explicit outputs")
        return self


class ProgressivePlanFoundation(BaseModel):
    """Plan-wide detail bound once while the first outline step is materialized."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort: ProgressiveCohortIntent
    display_labels: list[ProgressiveDisplayLabel] = Field(default_factory=list)
    robustness_intents: list[ProgressiveRobustnessIntent] = Field(default_factory=list)
    know_how_decisions: list[ProgressiveKnowHowDecision] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_rosters(self) -> "ProgressivePlanFoundation":
        label_keys = [item.key for item in self.display_labels]
        if len(label_keys) != len(set(label_keys)):
            raise ValueError("display label keys must be unique")
        robustness_ids = [item.spec_id for item in self.robustness_intents]
        if len(robustness_ids) != len(set(robustness_ids)):
            raise ValueError("robustness intent ids must be unique")
        know_how_coordinates = [
            (item.card_id, item.claim_id) for item in self.know_how_decisions
        ]
        if len(know_how_coordinates) != len(set(know_how_coordinates)):
            raise ValueError("know-how decisions must not repeat a card/claim pair")
        return self


class ProgressiveFoundationMaterialization(BaseModel):
    """Plan-wide choices returned separately from any executable step.

    The outline digest is host supplied and transport locked.  Keeping this
    response separate prevents the first step request from carrying two
    independent high-entropy contracts at once.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_plan_foundation/1"] = (
        "easyicu.progressive_plan_foundation/1"
    )
    outline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation: ProgressivePlanFoundation


class ProgressiveStepMaterialization(BaseModel):
    """One outline-bound strict step response, never a full executable DAG."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_step_materialization/1"] = (
        "easyicu.progressive_step_materialization/1"
    )
    outline_step_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation: Optional[ProgressivePlanFoundation]
    step: ProgressiveSkeletonStep


class ProgressivePlannerCheckpoint(BaseModel):
    """Append-only state after one host-validated planning boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_planner_checkpoint/1"] = (
        "easyicu.progressive_planner_checkpoint/1"
    )
    sequence: int = Field(ge=0)
    stage: Literal["outline", "foundation", "step"]
    request_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_checkpoint_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    outline: ProgressivePlanOutline
    foundation: Optional[ProgressiveFoundationMaterialization] = None
    materializations: tuple[ProgressiveStepMaterialization, ...] = ()
    prompt_metrics: dict[str, Any]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _closed_checkpoint_chain(self) -> "ProgressivePlannerCheckpoint":
        if self.stage == "outline":
            if self.sequence != 0 or self.foundation is not None or self.materializations:
                raise ValueError("outline checkpoint must be sequence 0 without suffix")
            if self.previous_checkpoint_sha256 is not None:
                raise ValueError("outline checkpoint cannot name a predecessor")
        elif self.stage == "foundation":
            if self.sequence != 1 or self.foundation is None or self.materializations:
                raise ValueError(
                    "foundation checkpoint must be sequence 1 without steps"
                )
        else:
            if self.foundation is None or not self.materializations:
                raise ValueError("step checkpoint requires foundation and prefix")
            if self.sequence != len(self.materializations) + 1:
                raise ValueError("step checkpoint sequence must follow prefix length")
        outline_sha256 = canonical_sha256(self.outline.model_dump(mode="json"))
        if self.prompt_metrics.get("outline_sha256") != outline_sha256:
            raise ValueError("checkpoint prompt metrics identify another outline")
        if self.foundation is not None:
            if self.foundation.outline_sha256 != outline_sha256:
                raise ValueError("checkpoint foundation identifies another outline")
        if any(item.foundation is not None for item in self.materializations):
            raise ValueError("checkpoint step repeats the separately sealed foundation")
        if self.sequence and self.previous_checkpoint_sha256 is None:
            raise ValueError("non-initial checkpoint requires predecessor authority")
        unsigned = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if canonical_sha256(unsigned) != self.checkpoint_sha256:
            raise ValueError("progressive checkpoint digest mismatch")
        return self


class ProgressivePlanSkeleton(BaseModel):
    """Host-assembled research skeleton compiled from step materializations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_plan_skeleton/1"] = (
        "easyicu.progressive_plan_skeleton/1"
    )
    analysis_type: str = Field(min_length=1, max_length=128)
    cohort: ProgressiveCohortIntent
    display_labels: list[ProgressiveDisplayLabel] = Field(default_factory=list)
    robustness_intents: list[ProgressiveRobustnessIntent] = Field(default_factory=list)
    know_how_decisions: list[ProgressiveKnowHowDecision] = Field(default_factory=list)
    steps: list[ProgressiveSkeletonStep] = Field(min_length=1, max_length=24)
    rationale: str = Field(min_length=8, max_length=1200)

    @model_validator(mode="after")
    def _closed_dag(self) -> "ProgressivePlanSkeleton":
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("progressive skeleton step ids must be unique")
        positions = {step_id: index for index, step_id in enumerate(step_ids)}
        for index, step in enumerate(self.steps):
            for dependency in step.depends_on:
                if dependency not in positions:
                    raise ValueError(
                        f"step {step.step_id!r} depends on unknown step {dependency!r}"
                    )
                if positions[dependency] >= index:
                    raise ValueError(
                        f"step {step.step_id!r} dependency {dependency!r} must precede it"
                    )
            for reference in step.product_inputs:
                position = positions.get(reference.producer_step_id)
                if position is None or position >= index:
                    raise ValueError(
                        f"step {step.step_id!r} product source "
                        f"{reference.producer_step_id!r} must precede it"
                    )
        primary = [
            step.step_id
            for step in self.steps
            if step.planned_analysis_role == "primary"
        ]
        if len(primary) > 1:
            raise ValueError(
                "progressive skeleton may declare at most one primary step"
            )
        label_keys = [item.key for item in self.display_labels]
        if len(label_keys) != len(set(label_keys)):
            raise ValueError("display label keys must be unique")
        robustness_ids = [item.spec_id for item in self.robustness_intents]
        if len(robustness_ids) != len(set(robustness_ids)):
            raise ValueError("robustness intent ids must be unique")
        know_how_coordinates = [
            (item.card_id, item.claim_id) for item in self.know_how_decisions
        ]
        if len(know_how_coordinates) != len(set(know_how_coordinates)):
            raise ValueError("know-how decisions must not repeat a card/claim pair")
        return self


class ProgressiveSuffixRevision(BaseModel):
    """Replacement for one unlocked suffix; compiled prefix is not repeated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_suffix_revision/1"] = (
        "easyicu.progressive_suffix_revision/1"
    )
    replace_from_step_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{0,79}$")
    replacement_steps: list[ProgressiveSkeletonStep] = Field(
        min_length=1,
        max_length=24,
    )
    rationale: str = Field(min_length=8, max_length=800)

    @model_validator(mode="after")
    def _starts_at_declared_coordinate(self) -> "ProgressiveSuffixRevision":
        if self.replacement_steps[0].step_id != self.replace_from_step_id:
            raise ValueError("replacement_steps must begin with replace_from_step_id")
        return self


class ProgressiveCompiledStepReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    skeleton_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_step_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    immutable_prefix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProgressivePlanCompileReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_plan_compile_receipt/1"] = (
        "easyicu.progressive_plan_compile_receipt/1"
    )
    owner: Literal["easyicu.planning.progressive_compiler_v1"] = (
        "easyicu.planning.progressive_compiler_v1"
    )
    compiler_version: Literal["1"] = "1"
    skeleton_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_steps: list[ProgressiveCompiledStepReceipt]


class ProgressivePlanCompileError(ValueError):
    """Stable, attributable compiler finding suitable for suffix retry."""

    owner = "easyicu.planning.progressive_compiler_v1"

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        step_id: Optional[str] = None,
        step_index: Optional[int] = None,
        path: Optional[str] = None,
        findings: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,79}", reason_code):
            raise ValueError(
                f"invalid progressive compiler reason code {reason_code!r}"
            )
        self.reason_code = reason_code
        self.code = reason_code
        self.step_id = step_id
        self.step_index = step_index
        self.path = path
        self.details = {
            "owner": self.owner,
            "reason_code": reason_code,
            "step_id": step_id,
            "step_index": step_index,
            "path": path,
            "message": str(message),
        }
        # Public/Web diagnostics may expose only these bounded host-authored
        # coordinates.  The human-readable compiler message can contain
        # candidate values, so it deliberately stays out of this projection.
        self.easyicu_safe_diagnostic = {
            "owner": self.owner,
            "reason_code": reason_code,
            "step_id": step_id,
            "step_index": step_index,
            "path": path,
        }
        if findings:
            self.details["findings"] = [dict(item) for item in findings]
        coordinate = f" step={step_id!r}" if step_id else ""
        if path:
            coordinate += f" path={path}"
        super().__init__(f"{reason_code}:{coordinate} {message}".strip())


__all__ = [
    "ProgressiveCompiledStepReceipt",
    "ProgressiveCohortIntent",
    "ProgressiveDisplayLabel",
    "ProgressiveFoundationMaterialization",
    "ProgressiveLiteratureBinding",
    "ProgressiveKnowHowDecision",
    "ProgressiveModelTermIntent",
    "ProgressiveOutputIntent",
    "ProgressiveOutlineStep",
    "ProgressivePlanCompileError",
    "ProgressivePlanCompileReceipt",
    "ProgressivePlanFoundation",
    "ProgressivePlanOutline",
    "ProgressivePlannerCheckpoint",
    "ProgressivePlanSkeleton",
    "ProgressiveProductRef",
    "ProgressiveRobustnessIntent",
    "ProgressiveSkeletonStep",
    "ProgressiveStepMaterialization",
    "ProgressiveSuffixRevision",
    "ProgressiveTableOneVariable",
    "progressive_module_ids_for_analysis_types",
]
