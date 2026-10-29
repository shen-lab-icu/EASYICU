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

import json
import re
from typing import Annotated, Any, ClassVar, Literal, Mapping, Optional, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from ..canonical_json import canonical_sha256
from ..contracts.primary_cohort import STUDY_POPULATION_PRODUCTS
from ..contracts.model_terms import OutcomeDefinition, OutcomeType
from ..contracts.product_identity import is_canonical_typed_product_token
from ..contracts.functional_form import FunctionalFormSpec
from ..contracts.prediction_validation import BootstrapSpec
from ..contracts.prediction_validation import ExternalValidationPlanFields, ExternalValidationSpec
from ..contracts.model_terms import DataPreparationPlan
from ..contracts.model_tokens import BinaryEffectMeasure
from ..contracts.exposure_outcome_distribution import PatientWeightedIntervalAssumptions
from ..contracts.phenotyping_features import PHENOTYPING_PRIMARY_ACTION, require_phenotyping_features
from ..contracts.trajectory_design import REPEATED_ACTIONS, RepeatedMeasurePlanFields
from ..contracts.prediction_subgroup import PredictionSubgroupPlanFields, PredictionSubgroupSpec
from .diagnostic_step_fields import DiagnosticPlanFields
from .design_selection import ResearchDesignSelection


from .progressive_module_ids import (
    ProgressiveModuleId,
    progressive_module_ids_for_analysis_types,
)


ProgressiveOutputRole = Literal[
    "analysis_cohort",
    "cohort_flow",
    "table_one",
    "exposure_outcome_distribution",
    "continuous_outcome_summary",
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
PROGRESSIVE_HOST_COMPILED_OUTPUTS: Mapping[
    str, tuple[tuple[str, str], ...]
] = {
    "cohort_definition": (
        ("artifact:analysis_cohort", "analysis_cohort"),
        ("table:cohort_flow", "cohort_flow"),
    ),
    "table_one": (("table:table_one", "table_one"),),
    "exposure_outcome_distribution": (
        ("table:exposure_outcome_distribution", "exposure_outcome_distribution"),
    ),
    "continuous_outcome_summary": (("table:continuous_outcome_summary", "continuous_outcome_summary"),),
    "adjusted_association": (
        ("table:adjusted_association_estimates", "adjusted_association_estimates"),
    ),
    "absolute_risk_context": (
        ("table:absolute_risk_context", "absolute_risk_context"),
    ),
    "robustness_replay": (
        ("table:robustness_matrix", "robustness_matrix"),
        ("table:robustness_summary", "robustness_summary"),
    ),
}
PROGRESSIVE_ARTICLE_ROLES: Mapping[str, frozenset[str]] = {
    "cohort_definition": frozenset({"cohort_accounting"}),
    "table_one": frozenset({"baseline_context"}),
    "exposure_outcome_distribution": frozenset({"descriptive_result"}),
    "continuous_outcome_summary": frozenset({"descriptive_result"}),
    "measurement_audit": frozenset({"data_quality"}),
    "adjusted_association": frozenset({"primary_estimand"}),
    "absolute_risk_context": frozenset({"descriptive_result"}),
    "robustness_replay": frozenset({"robustness"}),
}
# These modules compile a fixed estimator/product, not the action named by the
# model. A null action retains that estimator; it is not a custom-method escape.
PROGRESSIVE_FIXED_MODULE_ACTION_IDS: Mapping[str, tuple[str, ...]] = {
    "adjusted_association": ("association.adjusted_association",),
    "absolute_risk_context": (),
    "robustness_replay": (),
}
TableOneMode = Literal["independent_inference", "descriptive_smd_only"]
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


class ProgressivePopulationCriterion(BaseModel):
    """One restriction the question places on whom the study includes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion: str = Field(min_length=2, max_length=160)
    #: Allowed cohort concepts that express it; empty when none does.
    concept_ids: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("concept_ids")
    @classmethod
    def _unique_concepts(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("population criterion concepts must be unique and non-empty")
        return cleaned


class ProgressiveCohortIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    selection_mode: Literal["all_input_rows", "predicate_filtered"]
    inclusion: list[ProgressiveCohortPredicate] = Field(default_factory=list)
    exclusion: list[ProgressiveCohortPredicate] = Field(default_factory=list)
    #: The population the question names, stated by the Planner; each
    #: criterion with concepts is applied by one of the predicates above
    #: (``validate_progressive_foundation``).  Omitted from the digest when
    #: empty, so intents sealed before it existed keep their identity.
    population_criteria: list[ProgressivePopulationCriterion] = Field(
        default_factory=list, max_length=6, exclude_if=lambda value: not value
    )
    #: The same population stated again as a typed spec
    #: (``planning.population_spec``).  Once stated, it decides the plan's
    #: cohort: the host compiles the cohort from it
    #: (``planning.progressive_compiler``), and the predicates above no longer
    #: select rows; the audit records where they differ
    #: (``planning.population_shadow``).  It is kept as the Planner wrote it,
    #: any JSON value, so parsing never refuses it; the foundation check reads
    #: it strictly and sends a spec its owner refuses back to the Planner.
    #: Omitted from the digest when absent.  The model's own schema is a
    #: closed null, so every strict request built from these models stays
    #: closed; the foundation request alone offers the spec's schema
    #: (``agents.population_spec_transport``).
    population_spec: Annotated[Optional[Any], WithJsonSchema({"type": "null"})] = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("population_spec")
    @classmethod
    def _spec_is_plain_json(cls, value: Any) -> Any:
        if value is None:
            return value
        try:
            size = len(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"a population spec must be plain JSON: {exc}") from exc
        if size > 32_000:
            raise ValueError("a population spec is a few criteria, not 32 kB of JSON")
        return value

    @model_validator(mode="after")
    def _selection_is_explicit(self) -> "ProgressiveCohortIntent":
        criteria = [item.criterion.casefold() for item in self.population_criteria]
        if len(criteria) != len(set(criteria)):
            raise ValueError("population criteria must be unique")
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


#: The population the study selected, published by a host root for a step
#: whose question is asked of it rather than of the primary analysis's
#: narrower cohort.  Host templates set it; a Planner cannot.
STUDY_COHORT_SCOPE = "study_cohort"
_STUDY_COHORT_MODULES = frozenset({"cohort_definition", "exposure_outcome_distribution"})


def _require_study_cohort_owner(module_id: object, change_reason: object) -> None:
    if module_id not in _STUDY_COHORT_MODULES:
        raise ValueError(
            "the study-cohort population belongs to its host root and one distribution"
        )
    if change_reason is not None:
        raise ValueError("the study-cohort population is not an amendment of another population")


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

    #: Declared fields only the host writes: a family template keeps a
    #: covariate's unmeasured rows as an explicit state from measured
    #: missingness.  The Planner transport never offers them and the provider
    #: parsers strip them (``agents/progressive_payload.py``).
    HOST_OWNED_FIELDS: ClassVar[frozenset[str]] = frozenset({"missing_handling"})

    name: str = Field(min_length=1, max_length=128)
    role: Literal["exposure", "covariate"]
    coding: ModelTermCoding
    reference_level_index: Optional[int] = Field(default=None, ge=0)
    clinical_rationale: Optional[str] = Field(
        default=None,
        min_length=16,
        max_length=500,
        description=(
            "Agent-authored clinical confounding rationale; required for a "
            "covariate and omitted for the exposure."
        ),
    )
    #: Host-owned (``HOST_OWNED_FIELDS``).
    missing_handling: Optional[Literal["unmeasured_category"]] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="before")
    @classmethod
    def _references_no_choice_states(cls, data: Any) -> Any:
        """A binary term's reference is its first level; a linear term has none.

        Neither is a choice: a binary term compares its other level with its
        first unless the Planner names another, and a continuous or
        ordinal-linear term has no reference at all.  pilot1 planning (10-22)
        spent repair rounds on writing them.  A categorical term's reference
        stays the Planner's to state.
        """

        if not isinstance(data, Mapping):
            return data
        coding = data.get("coding")
        if coding == "binary" and data.get("reference_level_index") is None:
            return {**data, "reference_level_index": 0}
        if coding in {"continuous", "ordinal_linear"} and data.get("reference_level_index") is not None:
            return {**data, "reference_level_index": None}
        return data

    @model_validator(mode="after")
    def _reference_matches_coding(self) -> "ProgressiveModelTermIntent":
        treatment = self.coding in {"binary", "categorical"}
        if treatment != (self.reference_level_index is not None):
            raise ValueError(
                "binary/categorical terms require a reference index; continuous/"
                "ordinal_linear terms must omit it"
            )
        if (self.role == "covariate") != (self.clinical_rationale is not None):
            raise ValueError(
                "covariate terms require an Agent-authored clinical_rationale; "
                "exposure terms must omit it"
            )
        if self.role == "exposure" and self.missing_handling is not None:
            raise ValueError("the exposure cannot keep its unmeasured rows as a state")
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


class _OutlineContractError(ValueError):
    """Owner-authored rule code, retained in Pydantic's error context."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.easyicu_safe_diagnostic = {
            "owner": "easyicu.planning.progressive_outline_v1",
            "reason_code": reason_code,
        }
        super().__init__(message)


def _outline_step_values(step: "ProgressiveOutlineStep") -> str:
    """Echo only validated enum/pattern coordinates, never narrative or rosters."""

    return ", ".join(
        f"{field}={value!r}" if value is not None else f"{field}=null"
        for field, value in (
            ("module_id", step.module_id),
            ("planned_analysis_role", step.planned_analysis_role),
            ("scientific_action_id", step.scientific_action_id),
            ("population_scope", step.population_scope),
        )
    )


class ProgressiveOutlineStep(BaseModel):
    """One concise scientific step before any executable detail is requested."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{0,79}$")
    planned_analysis_role: Literal["primary", "secondary", "sensitivity", "auxiliary"] = Field(
        description="For continuous_outcome_summary, set planned_analysis_role='secondary'.",
    )
    module_id: ProgressiveModuleId
    objective: str = Field(min_length=8, max_length=600)
    depends_on: list[str] = Field(default_factory=list)
    variable_names: list[str] = Field(
        min_length=1, max_length=24,
        description="For continuous_outcome_summary, list the grouping and every selected continuous outcome.",
    )
    literature_citation_keys: list[str] = Field(default_factory=list, max_length=12)
    scientific_action_id: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$",
        description="For continuous_outcome_summary, set scientific_action_id=null.",
    )
    population_scope: Optional[Literal["analysis_cohort", "primary_model", "study_cohort"]] = Field(
        default=None, exclude_if=lambda value: value is None,
        description="Absolute-risk population chosen with the complete study design; executable detail must preserve it.",
    )
    population_scope_change_reason: Optional[str] = Field(
        default=None, min_length=12, max_length=1200, exclude_if=lambda value: value is None,
        description="Explain an intentional amendment of a source-bound population for complete-plan review; this is not execution approval.",
    )

    @field_validator("population_scope_change_reason", mode="before")
    @classmethod
    def _strip_population_scope_change_reason(cls, value):
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _population_owner(self):
        if self.module_id == "continuous_outcome_summary" and (
            self.planned_analysis_role != "secondary" or self.scientific_action_id is not None
        ):
            raise self._contract_error(
                "outline_continuous_summary_role_action",
                "continuous_outcome_summary is a secondary host summary: set "
                "planned_analysis_role='secondary' and scientific_action_id=null",
            )
        if self.population_scope == STUDY_COHORT_SCOPE:
            if self.module_id not in _STUDY_COHORT_MODULES:
                raise self._contract_error(
                    "outline_study_population_owner",
                    "set population_scope=null and population_scope_change_reason=null; "
                    "the study-cohort population (population_scope='study_cohort') belongs only to host "
                    "module_id='cohort_definition' or 'exposure_outcome_distribution'",
                )
            if self.population_scope_change_reason is not None:
                raise self._contract_error(
                    "outline_study_population_amendment",
                    "set population_scope_change_reason=null; population_scope='study_cohort' "
                    "is not an amendment of another population",
                )
            return self
        if self.population_scope is not None or self.population_scope_change_reason is not None:
            if self.module_id != "absolute_risk_context":
                raise self._contract_error(
                    "outline_population_scope_owner",
                    "population_scope belongs only to absolute_risk_context: set "
                    "population_scope=null and population_scope_change_reason=null for this module_id",
                )
            if self.population_scope is None:
                raise self._contract_error(
                    "outline_population_scope_required",
                    "population_scope_change_reason requires a population_scope: set "
                    "population_scope='analysis_cohort' or 'primary_model' to match the intended population",
                )
        return self

    def _contract_error(self, code: str, fix: str) -> _OutlineContractError:
        return _OutlineContractError(code, f"{fix} (received {_outline_step_values(self)})")

    @field_validator("depends_on", "variable_names", "literature_citation_keys")
    @classmethod
    def _unique_roster(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("outline rosters must contain unique non-empty values")
        return cleaned


#: The design element a capability gap concerns, and why no offered family can
#: express it.  The first three requirements are checked against the study's
#: typed context before planning stops (``planning.capability_gap``), as is a
#: design-element gap that names a stated adjustment; other estimand/design
#: claims can be unverifiable.
CapabilityGapElement = Literal[
    "population", "exposure", "outcome", "timing", "comparison", "analysis"
]
CapabilityGapRequirement = Literal[
    "levels_from_thresholds_unavailable",
    "longitudinal_representation_unavailable",
    "multiple_sources_required",
    "estimand_unsupported",
    "design_element_unsupported",
]


def _names_its_concept(requirement: Optional[str], concept: Optional[str]) -> bool:
    """A thresholds gap names the variable to group; a design-element gap may
    name the stated adjustment it concerns (``planning.capability_gap`` checks
    it); no other gap names one.  Reusing the design-element requirement keeps
    the outline schema inside its byte budget."""

    if requirement == "levels_from_thresholds_unavailable":
        return concept is not None
    return requirement == "design_element_unsupported" or concept is None


_CONCEPT_RULE = (
    "a capability gap's concept names the variable to group by thresholds, or "
    "the stated adjustment a design_element_unsupported gap concerns; no other "
    "requirement names one"
)


class ProgressiveCapabilityGap(BaseModel):
    """Something a question needs that no offered family can express.

    ``requirement`` says what is missing.  A producer that knows which part of
    the design the gap concerns, or can say why in its own words, adds
    ``element`` and ``detail``; an outline names only the requirement and
    explains it in its rationale.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement: CapabilityGapRequirement
    concept: Optional[str] = Field(default=None, min_length=1, max_length=128)
    element: Optional[CapabilityGapElement] = None
    detail: Optional[str] = Field(default=None, min_length=8, max_length=300)

    @model_validator(mode="after")
    def _variable_gaps_name_a_concept(self) -> "ProgressiveCapabilityGap":
        if not _names_its_concept(self.requirement, self.concept):
            raise ValueError(_CONCEPT_RULE)
        return self


class ProgressivePlanOutline(BaseModel):
    """Small retrieval-informed plan; the host materializes one step at a time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.progressive_plan_outline/1"] = (
        "easyicu.progressive_plan_outline/1"
    )
    analysis_type: str = Field(min_length=1, max_length=128)
    cohort_objective: str = Field(min_length=8, max_length=600)
    design_selection: Optional[ResearchDesignSelection] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    steps: list[ProgressiveOutlineStep] = Field(max_length=24)
    rationale: str = Field(min_length=8, max_length=1200)
    # Something the question needs that no offered family can express: planning
    # stops there instead of drafting steps that quietly answer another question.
    # Two flat fields keep the outline's transport small; the rationale says why.
    capability_gap: Optional[CapabilityGapRequirement] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    capability_gap_concept: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        exclude_if=lambda value: value is None,
    )

    def declared_capability_gap(self) -> Optional[ProgressiveCapabilityGap]:
        """The gap this outline declares, in the form the host checks."""

        if self.capability_gap is None:
            return None
        return ProgressiveCapabilityGap(
            requirement=self.capability_gap, concept=self.capability_gap_concept
        )

    @model_validator(mode="after")
    def _closed_dag(self) -> "ProgressivePlanOutline":
        if (self.capability_gap is None) != bool(self.steps):
            raise self._contract_error(
                "outline_steps_or_capability_gap",
                "set capability_gap=null and provide non-empty steps for an executable outline; "
                "for an unsupported design, set capability_gap to the applicable offered "
                "requirement and steps=[]",
            )
        if not _names_its_concept(self.capability_gap, self.capability_gap_concept):
            raise self._contract_error(
                "outline_capability_gap_concept",
                "set capability_gap_concept to the non-empty variable name when "
                "capability_gap='levels_from_thresholds_unavailable', or to the stated "
                "adjustment a design_element_unsupported gap concerns; otherwise set "
                "capability_gap_concept=null",
            )
        step_ids = [step.step_id for step in self.steps]
        for index, step_id in enumerate(step_ids):
            if step_id in step_ids[:index]:
                raise self._contract_error(
                    "outline_step_ids_unique",
                    f"set steps[{index}].step_id to a unique id and update depends_on "
                    "references to the intended producer",
                    indices=(index,),
                )
        positions = {step_id: index for index, step_id in enumerate(step_ids)}
        for index, step in enumerate(self.steps):
            for dependency in step.depends_on:
                if dependency not in positions:
                    raise self._contract_error(
                        "outline_dependency_unknown",
                        f"set each steps[{index}].depends_on entry to an existing preceding "
                        "step_id; add the intended producer before this step or remove the "
                        "entry if no dependency is intended",
                        indices=(index,),
                    )
                if positions[dependency] >= index:
                    raise self._contract_error(
                        "outline_dependency_order",
                        f"fix steps[{index}].depends_on: move its producer "
                        f"steps[{positions[dependency]}] before steps[{index}], or remove "
                        "the dependency if unintended; remove self-dependencies",
                        indices=(index,),
                    )
        primary = [
            index
            for index, step in enumerate(self.steps)
            if step.planned_analysis_role == "primary"
        ]
        if len(primary) > 1:
            raise self._contract_error(
                "outline_primary_step_unique",
                "keep planned_analysis_role='primary' on at most one step; set the other "
                "steps' planned_analysis_role to 'secondary', 'sensitivity', or 'auxiliary' "
                "according to their scientific purpose",
                indices=primary,
            )
        primary_adjusted = [
            step.step_id
            for step in self.steps
            if step.planned_analysis_role == "primary"
            and step.module_id == "adjusted_association"
        ]
        for index, step in enumerate(self.steps):
            if not (
                step.planned_analysis_role == "sensitivity"
                and step.module_id == "custom_analysis"
            ):
                continue
            if len(primary_adjusted) == 1 and primary_adjusted[0] in set(step.depends_on):
                continue
            if self.analysis_type == "association_study":
                raise self._contract_error(
                    "outline_association_sensitivity_parent",
                    "association custom sensitivity steps must follow and "
                    "depend directly on the primary adjusted_association step: "
                    f"set steps[{index}].depends_on to include the step_id of a preceding "
                    "step with module_id='adjusted_association' and planned_analysis_role='primary'; "
                    "declare that primary step if missing",
                    indices=(index,),
                )
            # Without an action, a custom sensitivity is materialized in the
            # scientific-sensitivity shape, which the compiler binds only to the
            # adjusted-association contract: it inherits the primary model the
            # step depends on. Any other sensitivity keeps its own action's
            # execution boundary, and step materialization cannot add the
            # action later.
            if step.scientific_action_id is None:
                raise self._contract_error(
                    "outline_custom_sensitivity_action",
                    f"set steps[{index}].scientific_action_id to the retrieved action "
                    "that owns this sensitivity in the selected family; "
                    "scientific_action_id=null is allowed only with depends_on containing "
                    "the step_id of a preceding module_id='adjusted_association' step "
                    "with planned_analysis_role='primary'",
                    indices=(index,),
                )
        return self

    def _contract_error(
        self, code: str, fix: str, *, indices: Sequence[int] | None = None,
    ) -> _OutlineContractError:
        coordinates = "; ".join(
            f"steps[{index}]: {_outline_step_values(self.steps[index])}"
            for index in (range(len(self.steps)) if indices is None else indices)
        )
        # capability_gap is a closed enum; its concept and rationale are private.
        gap = repr(self.capability_gap) if self.capability_gap is not None else "null"
        received = coordinates or f"steps=[], capability_gap={gap}"
        return _OutlineContractError(code, f"{fix} (received {received})")


#: Step fields that only one outline coordinate may carry.  The step contract
#: refuses them elsewhere, so a transport must not offer them elsewhere either.
COORDINATE_OWNED_STEP_FIELDS = (
    "covariate_balance_spec",
    "cluster_size_spec",
    "prediction_subgroup_spec",
    "repeated_measure_spec",
    "external_validation_spec",
    "data_preparation",
    "functional_form_spec",
    "bootstrap_spec",
    "phenotyping_feature_columns",
    "phenotyping_comparison_variables",
    "repeated_unit_interval_method",
    "patient_weighted_interval_assumptions",
    "effect_measure",
)


def coordinate_owned_step_fields(
    *,
    module_id: str,
    scientific_action_id: Optional[str],
    planned_analysis_role: str,
) -> frozenset[str]:
    """The coordinate-owned fields a step with this module, action and role may carry."""

    owned: set[str] = set()
    diagnostics = {
        "causal_emulation.covariate_balance": "covariate_balance_spec",
        "phenotyping.cluster_sizes": "cluster_size_spec",
        "prediction.subgroup_fairness": "prediction_subgroup_spec",
    }
    if module_id == "custom_analysis" and planned_analysis_role in {"secondary", "auxiliary"}:
        if scientific_action_id in diagnostics:
            owned.add(diagnostics[scientific_action_id])
    if module_id == "custom_analysis" and scientific_action_id in REPEATED_ACTIONS:
        owned.add("repeated_measure_spec")
    if scientific_action_id == "prediction.external_validation" and planned_analysis_role == "secondary":
        owned.add("external_validation_spec")
    if module_id == "adjusted_association" or scientific_action_id in {
        "prediction.discrimination_calibration", "prediction.external_validation", "time_to_event.cox_hr",
    }:
        owned.add("data_preparation")
    if module_id == "adjusted_association":
        owned.add("effect_measure")
    if module_id == "exposure_outcome_distribution":
        owned.update({"repeated_unit_interval_method", "patient_weighted_interval_assumptions"})
    if module_id == "custom_analysis" and planned_analysis_role == "sensitivity":
        owned.add("functional_form_spec")
    if scientific_action_id == "prediction.discrimination_calibration" and planned_analysis_role == "primary":
        owned.add("bootstrap_spec")
    if scientific_action_id == PHENOTYPING_PRIMARY_ACTION and planned_analysis_role == "primary":
        owned.add("phenotyping_feature_columns")
    if (
        scientific_action_id == "phenotyping.outcome_by_cluster"
        and planned_analysis_role == "secondary"
    ):
        owned.add("phenotyping_comparison_variables")
    return frozenset(owned)


class ProgressiveSkeletonStep(ExternalValidationPlanFields, RepeatedMeasurePlanFields,
                              DiagnosticPlanFields, PredictionSubgroupPlanFields):
    """Strict detail for only the step currently being materialized by the host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # This is a Planner wire contract. The inherited decoder restores the
    # typed object before compilation, without expanding every family schema.
    external_validation_spec: Annotated[
        Optional[ExternalValidationSpec],
        WithJsonSchema({"anyOf": [{"type": "string"}, {"type": "null"}]}),
    ] = Field(default=None, exclude_if=lambda value: value is None)
    prediction_subgroup_spec: Annotated[
        Optional[PredictionSubgroupSpec],
        WithJsonSchema({"anyOf": [{"type": "string"}, {"type": "null"}]}),
    ] = Field(default=None, exclude_if=lambda value: value is None)

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
    outcome_definition: Optional[OutcomeDefinition] = Field(default=None, exclude_if=lambda value: value is None)
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
    repeated_unit_interval_method: Optional[
        Literal["patient_cluster_robust_wald", "patient_weighted_hoeffding"]
    ] = Field(default=None, exclude_if=lambda value: value is None)
    patient_weighted_interval_assumptions: Optional[PatientWeightedIntervalAssumptions] = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    population_scope: Optional[Literal["analysis_cohort", "primary_model", "study_cohort"]] = Field(
        default=None, exclude_if=lambda value: value is None,
        description="Explicit population for absolute-risk context; primary_model reuses the preceding primary model's exact eligibility and complete cases.",
    )
    population_scope_change_reason: Optional[str] = Field(
        default=None, min_length=12, max_length=1200, exclude_if=lambda value: value is None,
        description="Required only for an intentional change from a source-bound descriptive population; disclose why the scientific scope changes for fresh complete-plan review.",
    )

    @field_validator("population_scope_change_reason", mode="before")
    @classmethod
    def _strip_population_scope_change_reason(cls, value):
        return value.strip() if isinstance(value, str) else value

    sensitivity_spec_ids: list[str] = Field(default_factory=list)
    functional_form_spec: Optional[FunctionalFormSpec] = Field(default=None, exclude_if=lambda value: value is None)
    bootstrap_spec: Optional[BootstrapSpec] = Field(default=None, exclude_if=lambda value: value is None)
    # The complete typed option is offered at its per-step materialization
    # coordinate. Broad legacy skeleton schemas retain their wire budgets.
    data_preparation: Annotated[Optional[DataPreparationPlan], WithJsonSchema({"type": "null"})] = Field(default=None, exclude_if=lambda value: value is None)
    phenotyping_feature_columns: Optional[list[str]] = Field(default=None, min_length=2, max_length=64, exclude_if=lambda value: value is None)
    phenotyping_comparison_variables: Optional[list[ProgressiveTableOneVariable]] = Field(default=None, min_length=1, max_length=64, exclude_if=lambda value: value is None)
    #: A binary adjusted association's reported measure.  ``None`` keeps the
    #: logistic odds ratio; ``risk_ratio`` selects the modified-Poisson model.
    effect_measure: Optional[BinaryEffectMeasure] = Field(default=None, exclude_if=lambda value: value is None)
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
        if self.repeated_measure_spec is not None and (
            self.module_id != "custom_analysis" or self.scientific_action_id not in REPEATED_ACTIONS
            or self.planned_analysis_role not in {"primary", "secondary"}
        ):
            raise ValueError("repeated_measure_spec belongs only to a repeated-measure analysis")
        if self.outcome_type in {"free_days", "ordinal", "count"}:
            if self.outcome_definition is None:
                raise ValueError("new outcome types require an explicit outcome_definition and estimand")
            self.outcome_definition.require_type(self.outcome_type)
        elif self.outcome_definition is not None:
            raise ValueError("outcome_definition belongs to free_days, ordinal or count outcomes")
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
        if self.module_id == "absolute_risk_context" and not (
            self.primary_exposure and self.outcome
        ):
            raise ValueError("absolute_risk_context requires exposure and outcome")
        if self.module_id == "continuous_outcome_summary":
            if (self.planned_analysis_role != "secondary" or not self.primary_exposure
                    or self.primary_exposure not in self.raw_inputs or len(self.raw_inputs) < 2
                    or self.outcome is not None or self.outcome_type is not None
                    or self.model_terms or self.scientific_action_id is not None
                    or self.sensitivity_spec_ids
                    or any(value is not None for value in (
                        self.event_level_index, self.reference_exposure_level_index,
                        self.comparison_exposure_level_index, self.primary_contrast_level_index,
                        self.denominator_policy, self.missing_exposure_policy,
                        self.missing_outcome_policy, self.confidence_level,
                    ))):
                raise ValueError("continuous_outcome_summary requires a secondary grouping/outcome roster, without binary, effect or inference controls")
        if self.population_scope == STUDY_COHORT_SCOPE:
            _require_study_cohort_owner(self.module_id, self.population_scope_change_reason)
            if self.module_id == "cohort_definition" and (
                self.planned_analysis_role != "auxiliary"
                or self.depends_on
                or self.raw_inputs
                or self.product_inputs
                or len(self.outputs) != 1
                or self.outputs[0].semantic_role != "analysis_cohort"
                or self.outputs[0].product_id not in STUDY_POPULATION_PRODUCTS
            ):
                raise ValueError(
                    "the study-population root reads nothing and publishes one "
                    "host study-population product"
                )
        elif (self.population_scope is not None or self.population_scope_change_reason is not None) and self.module_id != "absolute_risk_context":
            # A step named after absolute risk is not that module: name the
            # module the scope belongs to and this step's own, so a retry
            # cannot read the rule as being about its step_id.
            raise ValueError(
                "population_scope belongs only to module_id='absolute_risk_context', "
                f"not to module_id={self.module_id!r} (whatever its step_id): set "
                "population_scope=null and population_scope_change_reason=null"
            )
        if self.module_id == "exposure_outcome_distribution":
            # The host owns the confidence level's default and a binary
            # exposure's contrast levels; the step states the science only.
            required = {
                "primary_exposure": self.primary_exposure,
                "outcome": self.outcome,
                "event_level_index": self.event_level_index,
                "denominator_policy": self.denominator_policy,
                "missing_exposure_policy": self.missing_exposure_policy,
                "missing_outcome_policy": self.missing_outcome_policy,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "exposure_outcome_distribution requires "
                    + ", ".join(missing)
                )
        if self.repeated_unit_interval_method is not None or self.patient_weighted_interval_assumptions is not None:
            if self.module_id != "exposure_outcome_distribution":
                raise ValueError("patient interval choices belong only to distribution modules")
            if (self.repeated_unit_interval_method == "patient_weighted_hoeffding") != (
                self.patient_weighted_interval_assumptions is not None
            ):
                raise ValueError("patient_weighted_hoeffding requires explicit fixed-weight assumptions")
        if self.module_id == "custom_analysis" and not self.custom_method:
            raise ValueError("custom_analysis requires custom_method")
        if self.module_id != "custom_analysis" and self.custom_method is not None:
            raise ValueError("custom_method belongs only to custom_analysis")
        owned = coordinate_owned_step_fields(
            module_id=self.module_id,
            scientific_action_id=self.scientific_action_id,
            planned_analysis_role=self.planned_analysis_role,
        )
        for field in ("covariate_balance_spec", "cluster_size_spec", "prediction_subgroup_spec"):
            if getattr(self, field) is not None and field not in owned:
                raise ValueError(f"{field} belongs only to its exact auxiliary or secondary diagnostic action")
        diagnostic_field = {
            "causal_emulation.covariate_balance": "covariate_balance_spec",
            "phenotyping.cluster_sizes": "cluster_size_spec",
            "prediction.subgroup_fairness": "prediction_subgroup_spec",
        }.get(self.scientific_action_id)
        if diagnostic_field is not None:
            spec = getattr(self, diagnostic_field)
            if spec is None or diagnostic_field not in owned:
                raise ValueError(f"{self.scientific_action_id} requires its closed {diagnostic_field} on a supporting custom action")
            if diagnostic_field == "prediction_subgroup_spec":
                if self.raw_inputs:
                    raise ValueError("subgroup performance consumes frozen scores and membership products, without raw inputs")
            else:
                required: set[str] = set()
                if diagnostic_field == "covariate_balance_spec":
                    required.update({spec.treatment_column, *(item.column for item in spec.covariates)})
                    if spec.weight_column:
                        required.add(spec.weight_column)
                if not required <= set(self.raw_inputs):
                    raise ValueError(f"{diagnostic_field} value columns must be explicit raw_inputs; the host binds the reviewed unit ID")
        if self.functional_form_spec is not None and "functional_form_spec" not in owned:
            raise ValueError("functional_form_spec belongs only to a custom sensitivity")
        if self.bootstrap_spec is not None and "bootstrap_spec" not in owned:
            raise ValueError("bootstrap_spec belongs only to the static prediction primary")
        if self.external_validation_spec is not None and "external_validation_spec" not in owned:
            raise ValueError("external_validation_spec belongs only to a secondary external-validation action")
        if self.data_preparation is not None and "data_preparation" not in owned:
            raise ValueError("data_preparation belongs only to an association, Cox or static prediction owner")
        if self.effect_measure is not None and "effect_measure" not in owned:
            raise ValueError("effect_measure belongs only to an adjusted_association step")
        if self.phenotyping_feature_columns is not None:
            if "phenotyping_feature_columns" not in owned:
                raise ValueError("phenotyping_feature_columns belongs only to the primary cluster solution")
            require_phenotyping_features(self.phenotyping_feature_columns, inputs=self.raw_inputs)
        if self.phenotyping_comparison_variables is not None:
            if "phenotyping_comparison_variables" not in owned:
                raise ValueError("phenotyping_comparison_variables belongs only to a secondary outcome-by-cluster step")
            names = [v.name for v in self.phenotyping_comparison_variables]
            repeated = sorted({name for name in names if names.count(name) > 1})
            unlisted = sorted(set(names) - set(self.raw_inputs))
            if repeated or unlisted:
                # The Planner repairs from this message alone, so it names the
                # rule and the offending sealed variables, not only the code.
                raise ValueError(
                    "phenotype_comparison_roster_invalid: each "
                    "phenotyping_comparison_variables name must appear once and "
                    "also be listed in raw_inputs"
                    + (f"; not in raw_inputs: {', '.join(unlisted)}" if unlisted else "")
                    + (f"; repeated: {', '.join(repeated)}" if repeated else "")
                )
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
            raise ValueError(
                f"module {self.module_id!r} requires explicit outputs: replace "
                "step.outputs=[] with a non-empty array of objects containing "
                "product_id and semantic_role, using the current module's allowed "
                "output roles. Do not add a new step key to explain the outputs"
            )
        return self


class ProgressivePlanFoundation(BaseModel):
    """Plan-wide detail bound once while the first outline step is materialized."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort: ProgressiveCohortIntent
    display_labels: list[ProgressiveDisplayLabel] = Field(default_factory=list)
    robustness_intents: list[ProgressiveRobustnessIntent] = Field(default_factory=list)
    know_how_decisions: list[ProgressiveKnowHowDecision] = Field(default_factory=list)

    @field_validator("display_labels", mode="before")
    @classmethod
    def _keyed_label_transport(cls, value):
        # Run-bound structured transport owns the exact keys. The model only
        # writes their reader-facing meanings; sealed artifacts keep the same
        # canonical list representation used by the compiler and replay.
        if isinstance(value, dict):
            return [{"key": key, "value": label} for key, label in value.items()]
        return value

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

    schema_version: Literal[
        "easyicu.progressive_planner_checkpoint/1", "easyicu.progressive_planner_checkpoint/2",
    ] = (
        "easyicu.progressive_planner_checkpoint/1"
    )
    revision_offset: Optional[int] = Field(default=None, gt=0, exclude_if=lambda v: v is None)
    repair_start_index: Optional[int] = Field(default=None, ge=0, exclude_if=lambda v: v is None)
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
        if self.schema_version.endswith("/1"):
            if self.revision_offset is not None or self.repair_start_index is not None:
                raise ValueError("legacy checkpoint cannot declare a suffix revision")
        elif (
            self.stage != "step" or self.revision_offset is None
            or self.repair_start_index is None
        ):
            raise ValueError("revised checkpoint requires a typed suffix coordinate")
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
            if self.sequence != len(self.materializations) + 1 + (self.revision_offset or 0):
                raise ValueError("step checkpoint sequence must follow prefix length")
            if self.repair_start_index is not None:
                repairs = self.prompt_metrics.get("final_acceptance_repairs") or []
                if not repairs or self.repair_start_index >= len(self.materializations):
                    raise ValueError("revised checkpoint lacks its final acceptance finding")
                schemas = self.prompt_metrics.get("active_step_materialization_schema_sha256")
                if not isinstance(schemas, list) or len(schemas) != len(self.materializations):
                    raise ValueError("revised checkpoint lacks active prefix schema authorities")
                repair = repairs[-1]
                retained = self.materializations[:self.repair_start_index]
                if (
                    repair.get("retained_step_count") != self.repair_start_index
                    or repair.get("retained_materializations_sha256") != canonical_sha256(
                        [m.model_dump(mode="json") for m in retained]
                    )
                ):
                    raise ValueError("revised checkpoint changed the retained prefix")
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
    design_selection: Optional[ResearchDesignSelection] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
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


class OmittedRenderStep(BaseModel):
    """Host-recorded omission; not part of any Planner response schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    expected_outputs: list[str]
    reason: Literal["unconsumed_optional_render_without_typed_owner"] = (
        "unconsumed_optional_render_without_typed_owner"
    )
    message: str


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
    omitted_render_steps: list[OmittedRenderStep] = Field(
        default_factory=list, exclude_if=lambda value: not value,
    )


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
        metrics: Optional[Mapping[str, Any]] = None,
        cause_code: Optional[str] = None,
        concepts: Sequence[str] = (),
        element: Optional[str] = None,
        question_required: Optional[bool] = None,
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
        # A boundary finding whose own measurement is discarded cannot be acted
        # on: ``progressive_prompt_budget_exceeded`` says "too large" while
        # dropping both numbers, so every retry had to rediscover the size by
        # failing again. Host-authored non-negative integers under a canonical
        # key are safe to publish, and nothing else about the request is.
        safe_metrics = {
            str(key): int(value)
            for key, value in dict(metrics or {}).items()
            if re.fullmatch(r"[a-z][a-z0-9_]{2,63}", str(key))
            and type(value) is int
            and 0 <= value <= 10**12
        }
        if safe_metrics:
            self.easyicu_safe_diagnostic["metrics"] = safe_metrics
            self.details["metrics"] = safe_metrics
        # The stop's own typed cause beneath its reason (for example which
        # capability a gap lacks), so a reader can say why without the message.
        if cause_code is not None and re.fullmatch(r"[a-z][a-z0-9_]{2,79}", cause_code):
            self.easyicu_safe_diagnostic["cause_code"] = cause_code
            self.details["cause_code"] = cause_code
        # The concept ids the stop concerns (the variable a design element
        # needs), so a reader can name it.  Only identifier-shaped values cross.
        safe_concepts = [
            concept for concept in concepts
            if isinstance(concept, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", concept)
        ][:16]
        if safe_concepts:
            self.easyicu_safe_diagnostic["concepts"] = safe_concepts
            self.details["concepts"] = safe_concepts
        if isinstance(element, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", element):
            self.easyicu_safe_diagnostic["element"] = element
            self.details["element"] = element
        if type(question_required) is bool:
            self.easyicu_safe_diagnostic["question_required"] = question_required
            self.details["question_required"] = question_required
        if findings:
            self.details["findings"] = [dict(item) for item in findings]
        coordinate = f" step={step_id!r}" if step_id else ""
        if path:
            coordinate += f" path={path}"
        super().__init__(f"{reason_code}:{coordinate} {message}".strip())


def validate_progressive_module_action_compatibility(
    step: ProgressiveOutlineStep | ProgressiveSkeletonStep,
    *,
    available_action_ids: Sequence[str],
    step_index: int,
    phase: Literal["outline", "compile"],
) -> None:
    """Use the same fixed-estimator boundary before and after materialization."""

    compatible = PROGRESSIVE_FIXED_MODULE_ACTION_IDS.get(step.module_id)
    if compatible is None:
        return
    prefix = "progressive_outline" if phase == "outline" else "progressive"
    action = step.scientific_action_id
    if action is not None and action not in compatible:
        raise ProgressivePlanCompileError(
            f"{prefix}_action_module_mismatch",
            f"host module {step.module_id!r} cannot execute action {action!r}; "
            "select that action's executable custom_analysis contract instead "
            "of relabelling a fixed host estimator or replay",
            step_id=step.step_id,
            step_index=step_index,
            path="scientific_action_id",
            findings=({
                "module_id": step.module_id,
                "scientific_action_id": action,
                "compatible_action_ids": list(compatible),
            },),
        )
    if (
        step.planned_analysis_role == "primary"
        and compatible
        and not set(compatible).intersection(available_action_ids)
    ):
        raise ProgressivePlanCompileError(
            f"{prefix}_primary_module_family_mismatch",
            f"host module {step.module_id!r} implements a primary estimator "
            "outside the selected analysis family; scientific_action_id=null "
            "does not change that estimator. Bind an available family action "
            "to its own executable contract",
            step_id=step.step_id,
            step_index=step_index,
            path="module_id",
            findings=({
                "module_id": step.module_id,
                "compatible_action_ids": list(compatible),
                "available_family_action_ids": list(available_action_ids),
            },),
        )


def outline_step_products(
    step: ProgressiveOutlineStep,
    actions: Mapping[str, Any],
) -> tuple[str, ...]:
    """Products an outline step fixes before its executable detail exists.

    Host-compiled modules and runtime-contract actions (``actions`` maps the
    analysis type's action ids to catalog actions) own fixed products; a free
    custom step declares its products only at step materialization.
    """

    if is_study_population_root(step):
        # The root's product is fixed at materialization; advertising the
        # module's standard cohort products here would give the primary cohort
        # a second owner.
        return ()
    action = actions.get(str(step.scientific_action_id or ""))
    contract = action.runtime_contract if action is not None else None
    return tuple(
        product_id
        for product_id, _role in (
            *(contract.outputs if contract is not None else ()),
            *PROGRESSIVE_HOST_COMPILED_OUTPUTS.get(step.module_id, ()),
        )
    )


PRIMARY_POPULATION_PRODUCT = "table:adjusted_association_estimates"


def is_study_population_root(step: Any) -> bool:
    """Whether an outline or skeleton step is the host study-population root."""

    return (
        getattr(step, "module_id", None) == "cohort_definition"
        and getattr(step, "population_scope", None) == STUDY_COHORT_SCOPE
    )


def host_singleton_module_key(step: Any) -> str:
    """The key under which a host module may appear at most once in a plan.

    The study-population root shares the cohort-definition module with the
    primary cohort step; each is still singular.
    """

    module_id = str(getattr(step, "module_id", "") or "")
    return f"{module_id}:{STUDY_COHORT_SCOPE}" if is_study_population_root(step) else module_id


def duplicated_host_singletons(steps: Sequence[Any]) -> dict[str, list[str]]:
    """Host-compiled modules a plan declares more than once, by singleton key."""

    owners: dict[str, list[str]] = {}
    for step in steps:
        if getattr(step, "module_id", None) in PROGRESSIVE_HOST_COMPILED_OUTPUTS:
            owners.setdefault(host_singleton_module_key(step), []).append(step.step_id)
    return {key: step_ids for key, step_ids in owners.items() if len(step_ids) > 1}


def validate_primary_population_owner(
    outline: ProgressivePlanOutline,
    *,
    actions: Mapping[str, Any],
) -> None:
    """Reject a primary-model population that no preceding primary step owns.

    ``population_scope=primary_model`` reuses the exact rows of the primary
    result that emits ``table:adjusted_association_estimates``. The compiler
    enforces that owner step by step, where the outline-owned scope can no
    longer change; checking the outline returns the choice to its retry. Run
    after the host-module singleton check, so there is at most one owner.
    """

    owners: list[ProgressiveOutlineStep] = []
    for step_index, step in enumerate(outline.steps):
        if (
            step.module_id == "absolute_risk_context"
            and step.population_scope == "primary_model"
            and not any(owner.planned_analysis_role == "primary" for owner in owners)
        ):
            raise ProgressivePlanCompileError(
                "progressive_outline_primary_population_unsupported",
                "population_scope=primary_model means the rows of a preceding "
                f"primary step that emits {PRIMARY_POPULATION_PRODUCT} (the host "
                "adjusted_association module), and this outline has no such "
                "step. Keep the selected primary analysis. Set this step's "
                "population_scope to analysis_cohort and describe the broader "
                "eligible cohort, or remove the step; a source-bound "
                "primary_model requirement changes only with a "
                "population_scope_change_reason.",
                step_id=step.step_id,
                step_index=step_index,
                path="population_scope",
                findings=({
                    "required_product": PRIMARY_POPULATION_PRODUCT,
                    "preceding_owner_step_ids": [owner.step_id for owner in owners],
                    "primary_steps": [
                        {
                            "step_id": primary.step_id,
                            "module_id": primary.module_id,
                            "scientific_action_id": primary.scientific_action_id,
                        }
                        for primary in outline.steps
                        if primary.planned_analysis_role == "primary"
                    ],
                },),
            )
        if PRIMARY_POPULATION_PRODUCT in outline_step_products(step, actions):
            owners.append(step)


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
    "ProgressiveCapabilityGap",
    "ProgressivePlanOutline",
    "ProgressivePlannerCheckpoint",
    "ProgressivePlanSkeleton",
    "ProgressivePopulationCriterion",
    "PROGRESSIVE_ARTICLE_ROLES",
    "PROGRESSIVE_FIXED_MODULE_ACTION_IDS",
    "PROGRESSIVE_HOST_COMPILED_OUTPUTS",
    "ProgressiveProductRef",
    "ProgressiveRobustnessIntent",
    "ProgressiveSkeletonStep",
    "ProgressiveStepMaterialization",
    "ProgressiveSuffixRevision",
    "ProgressiveTableOneVariable",
    "COORDINATE_OWNED_STEP_FIELDS",
    "PRIMARY_POPULATION_PRODUCT",
    "STUDY_COHORT_SCOPE",
    "coordinate_owned_step_fields",
    "duplicated_host_singletons",
    "host_singleton_module_key",
    "is_study_population_root",
    "outline_step_products",
    "progressive_module_ids_for_analysis_types",
    "validate_primary_population_owner",
    "validate_progressive_module_action_compatibility",
]
