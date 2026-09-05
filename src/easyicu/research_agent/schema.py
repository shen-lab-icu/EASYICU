"""Schema definitions for the research agent layer.

Everything that flows between agents — context, plans, evidence,
manifests — is a typed pydantic model. This buys us:

* JSON serialisation for free (every artefact can be saved next to
  the data and re-loaded by reviewers, downstream tooling or a fresh
  pipeline run);
* validation at the boundary, so a hallucinated field from the LLM
  fails fast instead of contaminating downstream analysis;
* domain-owned contract modules with this file as a compatibility facade, so
  agents and validators consume one public type without importing owner internals.

The models are intentionally small and self-describing. They are not
the final word on what an ICU research package should contain — but
they are a stable v1 the rest of the module can lean on.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_serializer,
    model_validator,
)

from .contracts.capability_declaration import (
    validate_scientific_capability_declarations,
    validate_scientific_capability_id,
)
from .contracts.claim_ceiling import DescriptiveClaimContract
from .contracts.closed_levels import (
    validate_closed_scalar_levels,
)
from .contracts.dependence import PlannedDependenceRequirement
from .contracts.endpoint import EndpointAbsenceSemantics, EndpointKind, EndpointSpec
from .contracts.exposure_outcome_distribution import (
    ExposureOutcomeDistributionSpec,
)
from .contracts.family_primary import FamilyPrimaryResultRequirement
from .contracts.figure_plan import PlannedFigurePanelSpec
from .contracts.model_terms import ModelTermSpec, validate_model_term_roster
from .contracts.model_tokens import (
    ADJUSTED_ASSOCIATION_ANALYSIS_KIND as PLANNED_MODEL_REQUIREMENTS_OUTPUT,
    ASSOCIATION_GLM_BINOMIAL_ESTIMATOR,
    ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_OLS_ESTIMATOR,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
    PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
    canonical_association_method as _canonical_association_method,
    normalise_model_contract_token as _normalise_model_contract_token,
)
from .contracts.post_analysis import EValueConversionSpec, SubgroupAnalysisSpec
from .contracts.product_identity import is_canonical_typed_product_token
from .contracts.survival import (
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
    SurvivalAnalysisReceipt,
)
from .planning.cohort_contract import (
    CohortDefinition,
    CohortSchemaError,
    coerce_cohort_definition,
)
from .planning.design_selection import ResearchDesignSelection
from .planning.literature_contract import LiteratureDesignBinding
from .planning.robustness_contract import (
    ROBUSTNESS_REPLAY_OUTPUT_PRODUCT_KINDS,
    RobustnessPlanError,
    RobustnessSpec,
    validate_robustness_specs,
)
from .planning.sensitivity_authority import PrespecifiedSensitivitySpec
from .research_context.clinical_definition import ClinicalDefinitionReference

# Compatibility exports: these contracts have dependency-neutral owners, while
# established callers continue to import them from ``research_agent.schema``.
from .contracts.closed_levels import typed_level_key as _typed_level_key  # noqa: F401
from .contracts.exposure_outcome_distribution import (  # noqa: F401
    ExposureOutcomeRiskDifferenceContrast,
)

PlannedAnalysisRole = Literal[
    "primary",
    "secondary",
    "sensitivity",
    "auxiliary",
]

ArtifactConsumptionMode = Literal[
    "all_rows",
    "single_row",
    "one_per_role",
]

TableOneVariableKind = Literal["continuous", "categorical", "ordinal"]
TableOneSummary = Literal["mean_sd", "median_iqr", "both", "count_percent"]
TableOneTest = Literal[
    "welch_t_or_anova",
    "mann_whitney_or_kruskal",
    "chi_square_with_fisher_exact_for_sparse_2x2",
    "none_descriptive_smd_only",
]
TABLE_ONE_CLOSED_OUTPUTS = frozenset(
    {
        "table:table_one",
        "table:cohort_flow",
        "log:source_row_count_reconciliation",
    }
)

# The closed set of measurement/missingness audits the host can compute, named
# by WHAT EACH ONE ANSWERS rather than by the product id a Planner happened to
# write.  Ownership of an audit step used to be decided by looking the declared
# product id up in a literal alias table, which made a free-text namespace the
# gate: over the recorded corpus 162 audit steps -- pure counting work, nothing
# a model needs to reason about -- lost their deterministic executor purely
# because the Planner called the product ``lactate_missingness`` or
# ``data_quality_measurement_status`` instead of a spelling the table already
# knew.  Adding each new spelling only guarantees the next plan invents one
# more, so the declaration moves here and the product id becomes a label.
#
# The keys are distinguished by CONTENT, not by output filename.  The runner
# writes the source/availability table to three different filenames, so a rule
# counting distinct *files* would let one step declare all three, be claimed,
# and hand its reader the same table three times.  Distinct kinds cannot.
MEASUREMENT_AUDIT_KINDS = frozenset(
    {
        # Per concept: was it measured at all, and how much of the value column
        # is missing, splitting structural-no-source from measurement-missing.
        "measurement_missingness",
        # The same per-concept counts projected to a plain missing-n/missing-%
        # profile.
        "missingness_profile",
        # Per concept: where the value could have come from, including the
        # measured-flag/value disagreements.
        "measurement_source",
        # Per concept: how OFTEN and WHEN it was observed.
        "measurement_process",
        # Per event-timed concept: how the event column and its time behave --
        # present, absent, before the declared origin, time missing.
        "event_timing",
        # Per component of a composite score: completeness of each component.
        "component_completeness",
        # Rows surviving each analytic filter, as denominators.
        "analytic_denominators",
        # Stagewise cohort flow.
        "cohort_flow",
    }
)

# The closed set of outputs the prespecified-robustness replay produces from an
# already-locked specification grid, named by what each one answers.
#
# Same disease as MEASUREMENT_AUDIT_KINDS, different dominant organ.  There the
# product names were the gate that bled (162 steps); here it is the *method*
# string: the runner's allowlist holds exactly three heads, and over the
# recorded corpus 182 robustness steps -- with neither a figure nor a claim
# from the agent-owned validation gate -- were turned away, 62 of them for
# saying ``prespecified_sensitivity_analysis`` and 12 for the plainest possible
# ``sensitivity_analysis``, which the list simply does not contain.  Two
# runners, two allowlists, and in each one it was the *other* half that leaked.
#
# Widening the method list would be worse than the gap: this replay executes an
# already-locked spec grid, and a step named for a causal-emulation or
# weighting sensitivity may be different science the runner cannot produce.
# Claiming it and then failing for a missing product is strictly worse than
# never claiming it.  Only the Planner can say "this step IS that replay", so
# it says it here.
ROBUSTNESS_REPLAY_OUTPUTS = frozenset(ROBUSTNESS_REPLAY_OUTPUT_PRODUCT_KINDS)


def _closed_table_one_levels(values: List[Any], *, label: str) -> List[Any]:
    return validate_closed_scalar_levels(values, label=label)


_PRIMARY_RESULT_KIND_ALIASES = {
    "cohort": "dataset",
    "metric": "statistic",
    "statistics": "statistic",
}
_PRIMARY_SCIENTIFIC_RESULT_KINDS = frozenset(
    {"artifact", "dataset", "model", "statistic", "table"}
)

# ---------------------------------------------------------------------------
# Variable / concept descriptors
# ---------------------------------------------------------------------------


class VariableRole(str, Enum):
    """How the variable is used in an analysis.

    The role is *not* the same as a python dtype. Two integer columns
    can have very different roles (an ordinal SOFA component vs. an
    age in years) and we want the agent prompts to know which is
    which.
    """

    ID = "id"  # patient/stay identifier
    TIME = "time"  # absolute or relative timestamp
    DEMOGRAPHIC = "demographic"  # age, sex, weight at admission
    VITAL = "vital"  # vital sign, continuous
    LAB = "lab"  # laboratory measurement, continuous
    INTERVENTION = "intervention"  # vaso, vent, rrt, fluid bolus...
    ORDINAL_SCORE = "ordinal_score"  # SOFA component, GCS, KDIGO stage
    COMPOSITE_SCORE = "composite_score"  # total SOFA, APACHE
    OUTCOME = "outcome"  # death, los_icu, readmission
    INDEX = "index"  # row-level time index
    META = "meta"  # source dataset, cohort tag
    OTHER = "other"


class AggregationRule(str, Enum):
    """Allowed aggregation operations for a variable.

    The agent is told *which* aggregations are valid for a given
    column. Taking a mean of an ordinal SOFA component is a category
    error — the ConceptUsageAuditor flags it.
    """

    ANY = "any"  # all common ops valid
    MEAN_MEDIAN = "mean_or_median"  # continuous, well-defined mean
    MEDIAN_ONLY = "median_only"  # heavily skewed continuous
    MAX_LAST = "max_or_last"  # ordinal scores → take max or last
    SUM = "sum"  # counts / cumulative doses
    FIRST_VALUE = "first_value"  # at-admission attributes
    NONE = "none"  # do not aggregate (event variables)


class TimeWindow(BaseModel):
    """A bounded analysis window relative to an anchor event."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Short, stable identifier, e.g. 'first_24h'.")
    anchor: Literal["icu_admission", "hospital_admission", "event_onset"] = (
        "icu_admission"
    )
    start_hours: float = 0.0
    end_hours: float = 24.0
    rationale: Optional[str] = Field(
        default=None,
        description="Why this window — clinical reason or precedent paper.",
    )


class TemporalConstraint(BaseModel):
    """Deterministic representation of time semantics parsed from the request.

    Unlike ``TimeWindow``, which is a reusable cohort-level window definition,
    this model captures richer query fragments such as "worst lactate before
    vasopressor" or "AKI within 48h after ICU admission" so the runtime can
    reason about anchor events, ordering, and aggregation hints explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    raw_text: str
    relation: Literal[
        "first_window",
        "within_after",
        "before_event",
        "worst_before_event",
        "after_event",
        "relative_to_anchor",
        "unspecified",
    ] = "unspecified"
    anchor_event: str = Field(
        default="icu_admission",
        description="The anchor used to interpret the temporal phrase deterministically.",
    )
    target_concept: Optional[str] = Field(
        default=None,
        description="Concept explicitly mentioned in the phrase, e.g. lactate or AKI.",
    )
    start_hours: Optional[float] = None
    end_hours: Optional[float] = None
    aggregation_hint: Optional[str] = Field(
        default=None,
        description="Derived analytical hint, e.g. worst, first, last, max, min.",
    )
    executable_repr: str = Field(
        ...,
        description="Stable, deterministic string form suitable for provenance and replay.",
    )


class ObservationSemantics(BaseModel):
    """Typed interpretation of values absent by design rather than by loss.

    This contract is deliberately narrow.  It does not infer exposure,
    outcome, or analysis roles.  It only records a mechanically verified
    representation: a positive-only event triad, or an event time that is
    applicable only when a separate complete event-status column is positive.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["positive_only_event", "conditional_event_time"]
    event_status_column: Optional[str] = None
    event_count_column: Optional[str] = None
    measured_column: Optional[str] = None
    representative_column: Optional[str] = None
    time_origin: Optional[str] = None
    time_unit: Optional[str] = None

    @model_validator(mode="after")
    def _validate_closed_shape(self) -> "ObservationSemantics":
        if self.kind == "positive_only_event":
            required = {
                "event_count_column": self.event_count_column,
                "measured_column": self.measured_column,
                "representative_column": self.representative_column,
            }
            missing = sorted(name for name, value in required.items() if not value)
            if missing:
                raise ValueError(
                    "positive-only event semantics require " + ", ".join(missing)
                )
            if self.event_status_column is not None:
                raise ValueError(
                    "positive-only event semantics derive status from their triad"
                )
        elif not self.event_status_column:
            raise ValueError(
                "conditional event-time semantics require event_status_column"
            )
        return self


class MissingnessProfile(BaseModel):
    """Per-variable missingness summary computed at context build time."""

    model_config = ConfigDict(extra="forbid")

    fraction_missing: float = Field(ge=0.0, le=1.0)
    n_missing: int = Field(ge=0)
    n_total: int = Field(ge=0)
    raw_n_missing: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Raw null count before typed observation semantics distinguish "
            "event absence or non-applicability from true missingness."
        ),
    )
    eligible_n: Optional[int] = Field(
        default=None,
        ge=0,
        description="Rows for which this value is semantically applicable.",
    )
    not_applicable_n: int = Field(
        default=0,
        ge=0,
        description="Rows where absence is expected under the typed semantics.",
    )
    missingness_severity: Literal["low", "medium", "high", "unknown"] = "unknown"
    missingness_kind: Optional[str] = Field(
        default=None,
        description="Deprecated heuristic label retained for backward compatibility.",
    )
    missingness_test: Optional[str] = Field(
        default=None,
        description="Name of a formal missingness test if one was run, e.g. Little's MCAR test.",
    )
    missingness_test_p_value: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: Optional[str] = None


class FixedWindowTrajectoryMetadata(BaseModel):
    """Machine-readable semantics for one wide fixed-window trajectory column.

    The column name establishes only a family and relative time bin.  It does
    not establish the clinical anchor, which remains an agent/run declaration.
    ``source_scale`` preserves the underlying concept semantics while
    ``representation_kind`` distinguishes a raw discrete state from a
    fractional within-window summary that can be treated as a continuous
    representation by downstream method-compatibility checks.
    """

    model_config = ConfigDict(extra="forbid")

    family: str
    window_start_hours: float
    window_end_hours: float
    window_width_hours: float = Field(gt=0.0)
    time_axis: Literal["relative_hours"] = "relative_hours"
    anchor: Optional[str] = None
    source_scale: Literal[
        "continuous",
        "ordinal",
        "binary",
        "categorical",
        "count",
        "unknown",
    ] = "unknown"
    representation_kind: Literal[
        "fractional_window_summary",
        "continuous_window_summary",
        "discrete_window_state",
        "unknown_window_representation",
    ]
    observed_fractional_values: bool = False


class ClusterSelectionCandidate(BaseModel):
    """One agent-evaluated candidate in a clustering selection manifest."""

    model_config = ConfigDict(extra="forbid")

    n_clusters: int = Field(ge=1)
    criterion_value: float = Field(allow_inf_nan=False)


class ClusterSelectionManifest(BaseModel):
    """Agent-owned, replayable evidence for selecting a cluster count.

    The schema records the candidates and the agent's rule. Validation can
    verify a declared minimum/maximum; elbow and multi-criteria choices remain
    scientific judgments and therefore require an explicit rationale instead.
    """

    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(min_length=1)
    selection_rule: Literal["minimum", "maximum", "elbow", "multi_criteria"]
    direction: Literal["minimize", "maximize", "not_applicable"]
    selected_n_clusters: int = Field(ge=1)
    candidates: List[ClusterSelectionCandidate] = Field(min_length=2)
    rationale: Optional[str] = None

    @model_validator(mode="after")
    def _validate_selection_shape(self) -> "ClusterSelectionManifest":
        candidate_k = [item.n_clusters for item in self.candidates]
        if len(set(candidate_k)) != len(candidate_k):
            raise ValueError("candidate n_clusters values must be unique")
        if self.selected_n_clusters not in set(candidate_k):
            raise ValueError("selected_n_clusters must be one of the candidates")
        if self.selection_rule == "minimum" and self.direction != "minimize":
            raise ValueError("minimum selection requires direction=minimize")
        if self.selection_rule == "maximum" and self.direction != "maximize":
            raise ValueError("maximum selection requires direction=maximize")
        if (
            self.selection_rule in {"elbow", "multi_criteria"}
            and not str(self.rationale or "").strip()
        ):
            raise ValueError("elbow/multi_criteria selection requires rationale")
        return self


class ConceptDescriptor(BaseModel):
    """ICU-aware metadata for a single column in the analysis dataset.

    This is the artefact that lets the agent reason about *what* a
    column is rather than just its dtype. Built from the EasyICU
    concept dictionary plus an inspection of the cohort dataframe.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[str] = None
    role: VariableRole = VariableRole.OTHER
    dtype: str = Field(..., description="pandas dtype string, e.g. 'float64'.")
    unit: Optional[str] = None
    valid_range: Optional[List[float]] = Field(
        default=None,
        description="[lower, upper] physiologically plausible range. None if not applicable.",
    )
    observed_domain: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Value domain ACTUALLY OBSERVED in the provided cohort (not the "
            "dictionary's plausible range): keys may include n_unique, min, max, "
            "is_binary, is_constant. Lets the planner interpret a column by its "
            "real values instead of guessing a scale from its name — e.g. a "
            "column named '<score>_max' that is observed binary {0,1} must not be "
            "thresholded as if it ran 0-24."
        ),
    )
    allowed_aggregations: List[AggregationRule] = Field(
        default_factory=lambda: [AggregationRule.ANY]
    )
    aggregation_default: Optional[AggregationRule] = None
    is_ordinal: bool = False
    ordinal_levels: Optional[List[int]] = None
    source_concept: Optional[str] = Field(
        default=None,
        description="EasyICU concept name this column was derived from, if any.",
    )
    derived_from_concepts: List[str] = Field(
        default_factory=list,
        description="Raw EasyICU concept names used to derive this analysis variable.",
    )
    source_files: List[str] = Field(
        default_factory=list,
        description="EasyICU concept-export files that support this variable.",
    )
    source_tables: List[str] = Field(
        default_factory=list,
        description="Underlying raw ICU tables contributing to this concept when known.",
    )
    item_ids: List[str] = Field(
        default_factory=list,
        description="Item ids / concept ids associated with this variable when available.",
    )
    unit_normalization: Optional[str] = Field(
        default=None,
        description="How units were harmonized before this variable reached the cohort.",
    )
    analysis_window: Optional[str] = Field(
        default=None,
        description="Named time window used to derive this variable, e.g. 'first_24h'.",
    )
    analysis_window_role: Optional[
        Literal["exposure_definition", "outer_observation_window"]
    ] = Field(
        default=None,
        description=(
            "Whether analysis_window defines the exposure's clinical time zero "
            "or merely bounds observation of an already-derived phenotype. "
            "None means the role is not owner-verified."
        ),
    )
    clinical_definition: Optional[ClinicalDefinitionReference] = Field(
        default=None,
        description=(
            "Digest-bound clinical definition/version/source projected from the "
            "EasyICU clinical-contract registry for this concept."
        ),
    )
    temporal_resolution: Optional[str] = Field(
        default=None,
        description="Sampling granularity or windowing resolution, e.g. hourly, stay-level, event-level.",
    )
    fixed_window_trajectory: Optional[FixedWindowTrajectoryMetadata] = Field(
        default=None,
        description=(
            "Parsed wide-trajectory family/time-bin/representation metadata for "
            "columns named <family>_h<start>_<end>. The anchor is left unset "
            "unless a run-level contract declares it."
        ),
    )
    source_databases: List[str] = Field(default_factory=list)
    pitfalls: List[str] = Field(
        default_factory=list,
        description="Known traps for this variable, e.g. 'sofa==0 row may indicate missing components, not absence of dysfunction'.",
    )
    clinical_caveats: List[str] = Field(
        default_factory=list,
        description="Clinical interpretation caveats that should travel with the variable into planning and validation.",
    )
    missingness_semantics: Optional[str] = Field(
        default=None,
        description="Domain-specific interpretation of missingness for this variable.",
    )
    observation_semantics: Optional[ObservationSemantics] = Field(
        default=None,
        description=(
            "Mechanically verified representation contract for event absence or "
            "conditional event-time applicability."
        ),
    )
    forbidden_transformations: List[str] = Field(
        default_factory=list,
        description="Operations the agent should not use for this variable without explicit override.",
    )
    cross_database_notes: List[str] = Field(
        default_factory=list,
        description="Known harmonisation caveats when replicating this variable across ICU databases.",
    )
    missingness: Optional[MissingnessProfile] = None


# ---------------------------------------------------------------------------
# Cohort + context
# ---------------------------------------------------------------------------


class CohortDescriptor(BaseModel):
    """Summary of which patients are in scope and how they were chosen."""

    model_config = ConfigDict(extra="forbid")

    cohort_name: str
    database: str = Field(..., description="Source database tag, e.g. 'miiv', 'eicu'.")
    n_patients: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Distinct patients when a verified patient identifier is present; "
            "None when the cohort only exposes stay/row identity."
        ),
    )
    n_stays: int = Field(ge=0)
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    id_columns: List[str] = Field(default_factory=list)
    time_columns: List[str] = Field(default_factory=list)
    outcome_columns: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Deterministic cohort-definition provenance including source path, criteria, and resolver metadata.",
    )
    notes: Optional[str] = None


class UserPreferences(BaseModel):
    """Structured user-requested analysis preferences.

    Free-form notes remain useful, but the planner/coder layer benefits from
    seeing stable preference fields instead of a single blob of prose.
    """

    model_config = ConfigDict(extra="forbid")

    inferred_analysis_family: Optional[str] = None
    starter_template_key: Optional[str] = None
    preferred_methods: Optional[str] = None
    evaluation_focus: Optional[str] = None
    subgroup_sensitivity: Optional[str] = None
    timing_and_design: Optional[str] = None
    data_constraints: Optional[str] = None
    must_have_outputs: Optional[str] = None
    covariates: List[str] = Field(default_factory=list)
    covariate_selection: Literal["planner_selectable", "exact"] = Field(
        default="planner_selectable",
        description=(
            "Whether covariates remain a Planner-owned choice or are the exact "
            "locked adjustment roster. With 'exact', covariate_authority records "
            "whether the user fixed it or it came from an Agent Plan; an empty "
            "roster authorizes only an unadjusted model."
        ),
    )
    covariate_authority: Optional[Literal["user", "agent_plan"]] = Field(
        default=None,
        description=(
            "Owner of an exact adjustment roster. None is retained only for "
            "backward-compatible contexts; new exact rosters must identify "
            "whether they were user-authored or compiled from an Agent Plan."
        ),
    )
    covariate_rationales: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "User-reviewed clinical rationale for each exact adjustment "
            "covariate. Keys are exact covariate identifiers; values explain "
            "the pre-exposure confounding role rather than mere availability."
        ),
    )
    covariate_temporal_roles: Dict[
        str, Literal["baseline_static", "at_or_before_time_zero"]
    ] = Field(
        default_factory=dict,
        description=(
            "User-reviewed temporal eligibility for each exact adjustment "
            "covariate. Post-time-zero measurements cannot satisfy this contract."
        ),
    )
    covariate_operationalizations: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "User-reviewed mapping from each scientific covariate identity to "
            "the exact materialized analysis column. Omit identities whose "
            "source and operational column names are identical."
        ),
    )
    # Optional landmark / immortal-time origin (hours) for time-to-event
    # designs. Consumed by the deterministic survival runner; ``None`` means
    # "use the skill's case-neutral default" (24h) rather than a study-specific
    # constant baked into the skill.
    landmark_hours: Optional[float] = None
    sensitivity_specs: List[PrespecifiedSensitivitySpec] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Exact user-reviewed sensitivity analyses. These typed commitments "
            "govern materialization and plan review; prose is not authority."
        ),
    )
    extra_notes: Optional[str] = None

    @field_validator("covariate_rationales")
    @classmethod
    def _bounded_covariate_rationales(cls, value: Dict[str, str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for raw_key, raw_text in value.items():
            key = str(raw_key or "").strip()
            text = " ".join(str(raw_text or "").split())
            if not key or key in result:
                raise ValueError("covariate_rationales keys must be unique non-empty names")
            if len(text) < 8 or len(text) > 600:
                raise ValueError(
                    "covariate_rationales values must contain 8-600 characters"
                )
            result[key] = text
        return result

    @model_validator(mode="after")
    def _covariate_decisions_match_exact_roster(self) -> "UserPreferences":
        rationale_keys = set(self.covariate_rationales)
        temporal_keys = set(self.covariate_temporal_roles)
        operational_keys = set(self.covariate_operationalizations)
        roster = set(self.covariates)
        if rationale_keys - roster or temporal_keys - roster or operational_keys - roster:
            raise ValueError(
                "covariate decision keys must belong to covariates"
            )
        if self.covariate_selection != "exact" and (
            rationale_keys
            or temporal_keys
            or operational_keys
            or self.covariate_authority is not None
        ):
            raise ValueError(
                "covariate authority, rationales, temporal roles, and "
                "operationalizations require covariate_selection='exact'"
            )
        if any(
            not str(value or "").strip()
            for value in self.covariate_operationalizations.values()
        ):
            raise ValueError("covariate operationalizations must be non-empty names")
        return self


RESEARCH_CONTEXT_SCHEMA_VERSION = "easyicu.research_context/1"

# Field set that constitutes the immutable schema. A test under
# ``tests/research_agent/planning/test_research_context_schema.py`` asserts the
# actual ``ResearchContext.model_fields`` match this set so any silent
# field rename / addition / removal will fail in CI. Bump
# ``RESEARCH_CONTEXT_SCHEMA_VERSION`` when intentionally changing the
# schema and update this list at the same time.
RESEARCH_CONTEXT_FIELDS: tuple = (
    "schema_version",
    "research_question",
    "cohort",
    "variables",
    "time_windows",
    "temporal_constraints",
    "target_outcome",
    "endpoint",
    "primary_exposure",
    "cross_database_validation",
    "cohort_parquet",
    "user_preferences",
    "notes",
    "created_at",
)


class ResearchContext(BaseModel):
    """Top-level context object handed to the agent.

    The ``ResearchContext`` is the central artefact distinguishing this
    layer from a generic data-analysis agent. Every prompt the agents
    see is grounded in the fields below — variable kinds, allowed
    aggregations, time windows, known pitfalls — so that the agent
    cannot confuse an ordinal SOFA component for a continuous lab.

    Frozen as of 2026-05-24. ``model_config = ConfigDict(frozen=True)``
    blocks downstream code from silently mutating cohort/variables/
    target_outcome between construction and prompt rendering. Any
    legitimate change should go through ``model_copy(update={...})``
    so the diff is auditable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Private entropy used only to derive reversible outbound-safe tokens.
    # It is never part of research_context.json or any Provider prompt; the
    # authority layer persists it separately in a mode-0600 runtime checkpoint.
    _table_one_token_secrets: Dict[str, str] = PrivateAttr(default_factory=dict)

    schema_version: str = RESEARCH_CONTEXT_SCHEMA_VERSION
    research_question: str
    cohort: CohortDescriptor
    variables: List[ConceptDescriptor]
    time_windows: List[TimeWindow] = Field(default_factory=list)
    temporal_constraints: List[TemporalConstraint] = Field(default_factory=list)
    target_outcome: Optional[str] = Field(
        default=None,
        description="Name of the primary outcome column.",
    )
    endpoint: Optional[EndpointSpec] = Field(
        default=None,
        description=(
            "What the endpoint named by target_outcome IS -- binary flag, "
            "time-to-event pair, ordinal score -- declared once at context "
            "construction. Never reverse-inferred from a column-name suffix, a "
            "dtype, or input order. Optional so existing contexts still load; "
            "consumers that need the type must fail closed when it is absent "
            "rather than guess, which is the guessing this field exists to end."
        ),
    )
    primary_exposure: Optional[str] = Field(
        default=None,
        description=(
            "Name of the primary exposure/predictor the question requires the "
            "association model to estimate (e.g. 'sepsis3'). When set, the "
            "exposure-contract audit checks the primary model actually uses it."
        ),
    )
    cross_database_validation: List[str] = Field(
        default_factory=list,
        description="Other databases (e.g. eicu, hirid) where this analysis should be replicated.",
    )
    cohort_parquet: Optional[str] = None
    user_preferences: Optional[UserPreferences] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def variable(self, name: str) -> Optional[ConceptDescriptor]:
        for v in self.variables:
            if v.name == name:
                return v
        return None


class HypothesisBlueprint(BaseModel):
    """Pre-plan hypothesis and feasibility scaffold.

    This is the explicit "discovery" handoff: literature and ICU concept
    metadata are distilled before planning, then the planner receives a
    hypothesis, a step skeleton, self-critique, and domain gates instead of
    inventing a plan from the user question alone.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.hypothesis_blueprint/1"
    research_question: str
    hypothesis: str
    hypothesis_type: Literal[
        "confirmatory",
        "exploratory",
        "feasibility",
    ] = "exploratory"
    prior_literature_keys: List[str] = Field(default_factory=list)
    novelty_rationale: Optional[str] = None
    feasible_variables: List[str] = Field(default_factory=list)
    missing_variables: List[str] = Field(default_factory=list)
    concept_dependencies: List[str] = Field(default_factory=list)
    cross_database_feasibility: Dict[
        str,
        Literal["full", "degraded", "blocked"],
    ] = Field(default_factory=dict)
    degraded_reason: Dict[str, str] = Field(default_factory=dict)
    stepwise_plan: List[str] = Field(default_factory=list)
    self_critique: List[str] = Field(default_factory=list)
    feasibility_status: Literal["ready", "needs_data", "blocked"] = "ready"
    domain_gate_notes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Plans, evidence, manifests
# ---------------------------------------------------------------------------


# The v1 planner-owned roster is intentionally narrower than EasyICU's full
# modeling capability.  These are the adjusted-association families whose
# execution contract is currently checked by PrimaryModelContractValidator.
# Survival, prediction, mixed-effects, and clustering methods keep their own
# family-specific plans/contracts until an equally typed validator exists.
ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES = frozenset(
    {
        ASSOCIATION_LOGIT_ESTIMATOR,
        ASSOCIATION_GLM_BINOMIAL_ESTIMATOR,
    }
)
ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES = frozenset(
    {
        ASSOCIATION_OLS_ESTIMATOR,
        "statsmodels_quantreg",
        "statsmodels_quantreg_median_vcov_robust",
    }
)
# PLANNED_MODEL_REQUIREMENTS_STEP_METHOD / _OUTPUT_KIND are imported from
# contracts.model_tokens above and re-exported here unchanged, so the ownership
# predicate in contracts.association_execution reads the same two strings this
# schema validates against.


class PlannedModelRequirement(BaseModel):
    """Planner-owned obligation for a supported adjusted-association model.

    The planner chooses these scientific commitments.  Execution validators
    only reconcile the emitted model contracts against this typed roster; they
    must not infer required models from step prose or benchmark vocabulary.
    This v1 schema does not represent survival, prediction, mixed-effects, or
    clustering contracts.
    """

    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    outcome: str
    outcome_type: Literal["binary", "continuous"]
    method_family: str
    exposure_source: str
    analysis_role: Literal["primary", "secondary", "sensitivity"]
    analysis_set: Literal["source_aware", "complete_case"]
    required_for_step_success: bool = True
    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "The exact adjustment set, or null when the planner did not declare "
            "one. An empty list is a declaration of an unadjusted model, which "
            "is not the same statement as null."
        ),
    )
    covariate_rationales: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Planner-owned confounding rationale for each selected covariate. "
            "For an exact user-owned roster this is copied from sealed study "
            "authority; for a Planner-selectable roster it is proposed and "
            "reviewed as part of the complete plan."
        ),
    )
    covariate_temporal_roles: Dict[
        str, Literal["baseline_static", "at_or_before_time_zero"]
    ] = Field(
        default_factory=dict,
        description=(
            "Typed pre-time-zero eligibility for every selected covariate. "
            "Availability alone is not temporal adjustment authority."
        ),
    )
    model_terms: Optional[List[ModelTermSpec]] = Field(
        default=None,
        description=(
            "The exact exposure and covariate coding contract. Legacy plans "
            "without this field remain readable but cannot authorize a "
            "deterministic host fit."
        ),
    )
    exposure_levels: Optional[List[str]] = Field(
        default=None,
        description=(
            "The closed, ordered level set of a categorical or ordinal exposure, "
            "or null for a binary or continuous one. Declaring it commits the "
            "model to one contrast per non-reference level."
        ),
    )
    exposure_reference_level: Optional[str] = Field(
        default=None,
        description="Which declared level every contrast is taken against.",
    )
    primary_contrast_level: Optional[str] = Field(
        default=None,
        description=(
            "Which contrast is the headline estimate the manuscript reports. "
            "With more than two levels this cannot be inferred: the highest "
            "level against the reference and a per-level trend are different "
            "scientific claims, and choosing between them is the planner's."
        ),
    )
    dependence: Optional[PlannedDependenceRequirement] = Field(
        default=None,
        description=(
            "Exact repeated-unit covariance contract bound from StudyContext "
            "authority. Null means model-based covariance; execution must not "
            "infer clustering from intent text or identifier-like column names."
        ),
    )

    @field_validator(
        "requirement_id",
        "outcome",
        "exposure_source",
    )
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("planned model requirement fields must be non-empty")
        return text

    @field_validator("method_family", mode="before")
    @classmethod
    def _canonical_method_family(cls, value: object) -> str:
        method = _canonical_association_method(value)
        if not method:
            raise ValueError("planned model requirement fields must be non-empty")
        return method

    @field_validator("covariates")
    @classmethod
    def _exact_unique_covariate_names(
        cls, value: Optional[List[str]]
    ) -> Optional[List[str]]:
        """An adjustment set is a roster of exact columns or it is not declared.

        ``None`` stays ``None``: "the planner did not say" must remain
        distinguishable from "the planner said none", because a host that reads
        the first as the second would fit an unadjusted model and label it the
        pre-specified adjusted one.
        """

        if value is None:
            return None
        names = [str(item or "").strip() for item in value]
        if any(not name for name in names):
            raise ValueError("covariates must not contain blank names")
        if len(names) != len(set(names)):
            raise ValueError("covariates must not repeat a name")
        return names

    @field_validator("covariate_rationales")
    @classmethod
    def _bounded_planned_covariate_rationales(
        cls, value: Dict[str, str]
    ) -> Dict[str, str]:
        cleaned: Dict[str, str] = {}
        for raw_name, raw_rationale in value.items():
            name = str(raw_name or "").strip()
            rationale = str(raw_rationale or "").strip()
            if not name or len(rationale) < 8 or len(rationale) > 600:
                raise ValueError(
                    "planned covariate rationales require non-empty names and "
                    "8-600 character explanations"
                )
            cleaned[name] = rationale
        return cleaned

    @model_validator(mode="after")
    def _method_family_matches_supported_outcome(self) -> "PlannedModelRequirement":
        supported = (
            ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES
            if self.outcome_type == "binary"
            else ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES
        )
        method_family = _normalise_model_contract_token(self.method_family)
        if method_family not in supported:
            raise ValueError(
                "model_requirements currently support only binary logistic "
                "or continuous linear/quantile adjusted-association families; "
                f"outcome_type={self.outcome_type!r} is incompatible with "
                f"method_family={self.method_family!r}"
            )
        if (
            self.analysis_role in {"primary", "secondary"}
            and not self.required_for_step_success
        ):
            raise ValueError(
                "primary and secondary model_requirements must be required "
                "for step success; only a sensitivity requirement may be optional"
            )
        if self.covariates is not None:
            # Two adjustment sets that are wrong on their face, and wrong in a
            # way the contract can see without knowing the case.  Conditioning
            # on the outcome, or on the exposure whose effect is being
            # estimated, does not produce a weaker version of the declared
            # estimand -- it produces a different quantity that would still be
            # reported under the declared one's name.
            if self.outcome in self.covariates:
                raise ValueError(
                    "covariates must not contain the outcome "
                    f"{self.outcome!r}; conditioning on the outcome does not "
                    "estimate the declared association"
                )
            if self.exposure_source in self.covariates:
                raise ValueError(
                    "covariates must not contain the exposure "
                    f"{self.exposure_source!r}; adjusting for the exposure "
                    "removes the association the requirement declares"
                )
        rationale_keys = set(self.covariate_rationales)
        temporal_keys = set(self.covariate_temporal_roles)
        if rationale_keys or temporal_keys:
            expected_covariates = set(self.covariates or ())
            if rationale_keys != expected_covariates or temporal_keys != expected_covariates:
                raise ValueError(
                    "planned covariate rationale and temporal-role maps must "
                    "cover exactly the declared covariates"
                )
        if self.model_terms is not None:
            exposure_term, adjustment_terms = validate_model_term_roster(
                terms=self.model_terms,
                exposure=self.exposure_source,
                covariates=self.covariates,
            )
            term_covariates = [item.name for item in adjustment_terms]
            if self.covariates is None:
                self.covariates = term_covariates
            if exposure_term.transform == "treatment_contrast":
                levels = list(exposure_term.levels or ())
                reference = exposure_term.reference_level
                if (
                    self.exposure_levels is not None
                    and list(self.exposure_levels) != levels
                ):
                    raise ValueError(
                        "exposure_levels must match the exposure ModelTermSpec"
                    )
                if (
                    self.exposure_reference_level is not None
                    and self.exposure_reference_level != reference
                ):
                    raise ValueError(
                        "exposure_reference_level must match the exposure ModelTermSpec"
                    )
                self.exposure_levels = levels
                self.exposure_reference_level = reference
                if exposure_term.coding == "binary":
                    only_contrast = exposure_term.contrast_levels[0]
                    if (
                        self.primary_contrast_level is not None
                        and self.primary_contrast_level != only_contrast
                    ):
                        raise ValueError(
                            "a binary exposure's primary contrast is its one "
                            "non-reference level"
                        )
                    self.primary_contrast_level = only_contrast
                elif self.primary_contrast_level is None:
                    raise ValueError(
                        "a categorical exposure ModelTermSpec requires "
                        "primary_contrast_level on the model requirement"
                    )
            elif any(
                value is not None
                for value in (
                    self.exposure_levels,
                    self.exposure_reference_level,
                    self.primary_contrast_level,
                )
            ):
                raise ValueError(
                    "identity/ordinal-linear exposure coding cannot also declare "
                    "treatment-contrast fields"
                )
        self._check_declared_exposure_levels()
        return self

    def _check_declared_exposure_levels(self) -> None:
        """Refuse a partial categorical-exposure contrast declaration."""

        declared = {
            "exposure_levels": self.exposure_levels,
            "exposure_reference_level": self.exposure_reference_level,
            "primary_contrast_level": self.primary_contrast_level,
        }
        present = {name for name, value in declared.items() if value is not None}
        if not present:
            return
        missing = sorted(set(declared) - present)
        if missing:
            raise ValueError(
                "a categorical exposure is declared by "
                + ", ".join(sorted(declared))
                + " together; this requirement is missing "
                + ", ".join(repr(name) for name in missing)
                + ", so the host cannot tell which contrast the manuscript reports"
            )
        levels = [str(value or "").strip() for value in self.exposure_levels or []]
        if any(not level for level in levels):
            raise ValueError("exposure_levels must not contain a blank level")
        if len(levels) != len(set(levels)):
            raise ValueError("exposure_levels must not repeat a level")
        if len(levels) < 2:
            raise ValueError("exposure_levels needs at least two levels")
        reference = str(self.exposure_reference_level or "").strip()
        primary = str(self.primary_contrast_level or "").strip()
        if reference not in levels:
            raise ValueError(
                f"exposure_reference_level {reference!r} is not one of the "
                "declared exposure_levels"
            )
        if primary not in levels:
            raise ValueError(
                f"primary_contrast_level {primary!r} is not one of the "
                "declared exposure_levels"
            )
        if primary == reference:
            raise ValueError("primary_contrast_level must not be the reference level")


_DEFAULT_STABILITY_BASE_SEED = 1729
"""Recorded starting seed when the study does not pick one.

A seed makes a result reproducible; which seed it is carries no scientific
claim.  Defaulting it keeps every run's randomisation recorded and repeatable
while removing one more field the plan had to fill in to validate at all.
"""


class TrajectoryStabilitySpec(BaseModel):
    """Planner-owned design for a trajectory-cluster stability computation.

    The execution layer may compute this design, but it must not choose the
    resampling scheme, amount of resampling, randomisation policy, comparison
    metric, or label-alignment rule.  The selected clustering model, cluster
    count, representation, and missing-data fit are inherited from the typed
    upstream trajectory manifests rather than repeated here.

    Sixteen of these fields are ``Literal`` types with exactly one legal value:
    v1 of this calculator implements one resampling scheme, one comparison
    metric and one label-alignment rule, so there is nothing to choose between.
    They carry that single value as their default, which is the convention the
    other Planner-owned specs in this module already follow.  The ``Literal``
    still rejects every other spelling, the value is still serialised into the
    spec and still hashed into ``trajectory_stability_spec_sha256`` -- the
    Planner simply no longer has to retype sixteen constants correctly for the
    plan to validate at all.  If a second implementation is ever added to any
    of them, the field becomes a real choice and its default must be removed
    (``test_planner_spec_constants.py`` enforces exactly that).

    ``minimum_successful_resamples`` is likewise not a choice: the validator
    below requires it to equal ``n_resamples``.  Leave it null and it is filled
    in from ``n_resamples``; declare it and a disagreeing value is still
    rejected.
    """

    model_config = ConfigDict(extra="forbid")

    resampling_method: Literal["subsample_without_replacement"] = (
        "subsample_without_replacement"
    )
    n_resamples: int = Field(ge=2, le=500)
    sample_fraction: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    sample_size: Optional[int] = Field(default=None, ge=2)
    sample_fraction_rounding: Literal["floor"] = "floor"
    base_seed: int = Field(default=_DEFAULT_STABILITY_BASE_SEED, ge=0, le=2_147_483_647)
    seed_derivation: Literal["numpy_seedsequence_spawn_uint32_v1"] = (
        "numpy_seedsequence_spawn_uint32_v1"
    )
    cross_resample_membership: Literal["distinct_membership_required"] = (
        "distinct_membership_required"
    )
    stability_metric: Literal["adjusted_rand_index"] = "adjusted_rand_index"
    stability_aggregation: Literal["mean"] = "mean"
    metric_label_source: Literal["raw_refit_labels_label_invariant"] = (
        "raw_refit_labels_label_invariant"
    )
    evaluation_scope: Literal["sampled_overlap"] = "sampled_overlap"
    label_alignment: Literal["hungarian_maximum_overlap"] = "hungarian_maximum_overlap"
    label_alignment_reference: Literal["frozen_candidate_assignments"] = (
        "frozen_candidate_assignments"
    )
    label_alignment_tie_break: Literal[
        "minimum_rank_distance_then_lexicographic_v1"
    ] = "minimum_rank_distance_then_lexicographic_v1"
    final_assignment_policy: Literal["copy_selected_candidate_labels"] = (
        "copy_selected_candidate_labels"
    )
    minimum_successful_resamples: Optional[int] = Field(default=None, ge=2)
    failed_refit_policy: Literal["record_once_no_retry"] = "record_once_no_retry"
    refit_engine: Literal["easyicu_observed_data_diag_gmm_v1"] = (
        "easyicu_observed_data_diag_gmm_v1"
    )
    refit_initialization: Literal["random_balanced_assignments"] = (
        "random_balanced_assignments"
    )
    refit_max_iter: int = Field(default=200, ge=10, le=10_000)
    refit_tolerance: float = Field(default=1e-6, gt=0.0, le=0.1)
    refit_regularization: float = Field(default=1e-6, gt=0.0, le=1.0)
    minimum_mean_stability: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description=(
            "Optional planner-owned mean adjusted-Rand threshold. Null reports "
            "stability without making a binary accept/reject decision."
        ),
    )
    decision_mode: Optional[Literal["report_only", "minimum_mean_threshold"]] = None
    threshold_failure_action: Literal["fail_closed_require_planner_revision"] = (
        "fail_closed_require_planner_revision"
    )

    @model_validator(mode="after")
    def _closed_resampling_design(self) -> "TrajectoryStabilitySpec":
        if (self.sample_fraction is None) == (self.sample_size is None):
            raise ValueError(
                "trajectory stability requires exactly one of sample_fraction "
                "or sample_size"
            )
        if self.minimum_successful_resamples is None:
            self.minimum_successful_resamples = self.n_resamples
        elif self.minimum_successful_resamples != self.n_resamples:
            raise ValueError(
                "v1 trajectory stability requires every planned refit to succeed; "
                "minimum_successful_resamples must equal n_resamples"
            )
        # decision_mode restates minimum_mean_stability: the two are equivalent
        # by the rules below, so declaring a threshold IS choosing to gate on
        # it.  Left null it is derived; declared, a disagreement is still an
        # error rather than a silently preferred interpretation.
        implied = (
            "report_only"
            if self.minimum_mean_stability is None
            else "minimum_mean_threshold"
        )
        if self.decision_mode is None:
            self.decision_mode = implied
        elif self.decision_mode == "report_only":
            if self.minimum_mean_stability is not None:
                raise ValueError(
                    "report_only stability must not declare a binary threshold"
                )
        elif self.minimum_mean_stability is None:
            raise ValueError("minimum_mean_threshold requires minimum_mean_stability")
        return self


class ArtifactConsumptionContract(BaseModel):
    """Planner-owned rule for consuming one exact typed tabular input.

    The contract prevents a downstream consumer from silently treating a
    multi-row result as a singleton or selecting one scientific role by row
    position.  It never assigns roles itself: ``expected_roles`` must be
    declared by the Planner when role-specific consumption is intended.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.artifact_consumption/1"] = (
        "easyicu.artifact_consumption/1"
    )
    input_key: str = Field(
        ...,
        description="Exact kind:product input declared on the same step.",
    )
    mode: ArtifactConsumptionMode
    role_column: Optional[str] = None
    expected_roles: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _mode_coordinates_are_closed(self) -> "ArtifactConsumptionContract":
        if not is_canonical_typed_product_token(self.input_key):
            raise ValueError("input_key must be one canonical typed kind:product")
        roles = [str(value).strip() for value in self.expected_roles]
        if any(not value for value in roles) or len(set(roles)) != len(roles):
            raise ValueError("expected_roles must contain unique non-empty values")
        if self.mode == "one_per_role":
            if not str(self.role_column or "").strip() or not roles:
                raise ValueError("one_per_role requires role_column and expected_roles")
        elif self.role_column is not None or roles:
            raise ValueError(
                "all_rows/single_row must not declare role_column or expected_roles"
            )
        kind, _, _product = self.input_key.partition(":")
        if kind == "artifact" and self.mode != "all_rows":
            raise ValueError(
                "artifact inputs are row-level datasets and require all_rows; "
                "role-specific or singleton consumption belongs to a typed table"
            )
        self.expected_roles[:] = roles
        return self


class TableOneVariableSpec(BaseModel):
    """Planner-owned summary and comparison rule for one Table 1 variable.

    An ordinal row may declare ``levels`` even when it is summarised
    numerically. It once could not, and that refusal killed a whole task before
    a single step ran: five consecutive planning attempts were rejected on
    ``steps.2.table_one_spec.variables.2`` for declaring the level set of an
    ordinal clinical score. Every such score -- a SOFA component 0-4, a KDIGO
    stage 0-3, a GCS -- has a real closed set, the directive asks each variable
    for its "closed levels" without qualification, and the observed set is
    something the host itself now supplies in the variable catalog. Refusing
    the one declaration a Planner would naturally write, for the one variable
    kind whose set is genuinely closed, is not a safeguard.

    A declared set that nothing checks would be worse than none, because it
    reads as a guarantee. So the executor enforces it exactly as it enforces
    ``group_levels``: a value observed outside the declared set stops the step
    rather than being summarised as though it had been anticipated. Continuous
    rows keep the ban -- a measurement has no closed set to declare.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    variable_kind: TableOneVariableKind
    summary: TableOneSummary
    test: TableOneTest
    levels: List[Any] = Field(default_factory=list)

    @field_validator("levels")
    @classmethod
    def _closed_levels(cls, values: List[Any]) -> List[Any]:
        return _closed_table_one_levels(values, label="Table 1 variable levels")

    @model_validator(mode="after")
    def _compatible_summary_and_test(self) -> "TableOneVariableSpec":
        self.name = str(self.name or "").strip()
        if not self.name:
            raise ValueError("Table 1 variable name must be non-empty")
        numeric_summary = self.summary in {"mean_sd", "median_iqr", "both"}
        numeric_test = self.test in {
            "welch_t_or_anova",
            "mann_whitney_or_kruskal",
        }
        no_test = self.test == "none_descriptive_smd_only"
        if self.variable_kind == "continuous":
            if not numeric_summary or not (numeric_test or no_test) or self.levels:
                raise ValueError(
                    "continuous Table 1 variables require a numeric summary/test "
                    "and must not declare categorical levels"
                )
        elif self.summary == "count_percent":
            if self.test not in {
                "chi_square_with_fisher_exact_for_sparse_2x2",
                "none_descriptive_smd_only",
            }:
                raise ValueError(
                    "count/percent Table 1 variables require the categorical test"
                )
            if len(self.levels) < 2:
                raise ValueError(
                    "categorical/ordinal count summaries require at least two "
                    "Planner-declared closed levels"
                )
        else:
            if (
                self.variable_kind != "ordinal"
                or not numeric_summary
                or not (numeric_test or no_test)
            ):
                raise ValueError(
                    "only ordinal variables may use a numeric Table 1 summary/test "
                    "outside the continuous variable kind"
                )
            if len(self.levels) == 1:
                raise ValueError(
                    "an ordinal Table 1 variable declares either no closed levels "
                    "or at least two; one value is not a set"
                )
        return self


class TableOneSpec(BaseModel):
    """Planner-owned grouped baseline-table design.

    The host executes this declaration but never selects the grouping variable,
    variable roles, summary family, or inferential test. For exactly two
    declared groups, the host also emits comparison-minus-reference SMDs.

    ``missing_group_policy`` decides what a row whose grouping value is missing
    means -- the same question ``ExposureOutcomeDistributionSpec`` asks about an
    unobserved outcome, and it is answered the same way: by the Planner, in the
    declaration, never by the executor at run time. It once had a single legal
    value, and a single legal value is not a policy. Every recorded Table 1 that
    grouped on a fully observed column (an outcome flag, a discharge status) was
    unaffected; a Table 1 grouped on a measurement-derived clinical score could
    not be built at all, by any plan, because such a score is never observed on
    every stay. A real run measured that: KDIGO stage was absent for 696 of
    94,458 stays, 0.74%, and the step died inside its container with no legal
    declaration that could have saved it.

    ``exclude_and_report`` removes those rows from the whole table -- the
    grouped columns *and* Overall, from one filtered frame -- so that Overall
    stays equal to the sum of its groups. Splitting them is how a denominator
    and its parts come to describe two different row sets. The count removed
    travels in the emitted table itself, on every row, because that table is
    what the output gate reads and what the manuscript binds; a number kept
    anywhere else can be dropped on the way.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.table_one/1", "easyicu.table_one/2"] = (
        "easyicu.table_one/1"
    )
    group_by: str
    group_levels: List[Any] = Field(min_length=2)
    variables: List[TableOneVariableSpec] = Field(min_length=1)
    include_overall: Literal[True] = True
    missing_group_policy: Literal["fail_closed", "exclude_and_report"] = "fail_closed"
    missingness_display: Literal["n_percent_by_group"] = "n_percent_by_group"
    p_values_required: bool = True
    p_value_adjustment: Literal[
        "none_descriptive_table", "not_applicable_repeated_units"
    ] = "none_descriptive_table"
    standardized_difference_mode: Literal["auto_binary_groups"] = "auto_binary_groups"

    @field_validator("group_levels")
    @classmethod
    def _closed_group_levels(cls, values: List[Any]) -> List[Any]:
        return _closed_table_one_levels(values, label="Table 1 group_levels")

    @model_validator(mode="after")
    def _closed_design(self) -> "TableOneSpec":
        self.group_by = str(self.group_by or "").strip()
        if not self.group_by:
            raise ValueError("Table 1 group_by must be non-empty")
        names = [item.name for item in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("Table 1 variable names must be unique")
        if self.group_by in names:
            raise ValueError("Table 1 group_by must not also be a row variable")
        no_test = [
            item.name
            for item in self.variables
            if item.test == "none_descriptive_smd_only"
        ]
        if self.schema_version == "easyicu.table_one/1":
            if (
                not self.p_values_required
                or self.p_value_adjustment != "none_descriptive_table"
                or no_test
            ):
                raise ValueError(
                    "Table One /1 requires the declared independent-row tests"
                )
        elif (
            self.p_values_required
            or self.p_value_adjustment != "not_applicable_repeated_units"
            or len(no_test) != len(self.variables)
        ):
            raise ValueError(
                "Table One /2 is descriptive/SMD-only for repeated units and "
                "must not declare an inferential test"
            )
        return self


def _exact_product_naming_a_known_kind(
    product_id: str,
    kind: str,
    *,
    known_kinds: frozenset[str],
    label: str,
) -> tuple[str, str]:
    """Normalise one `<product id> IS <kind>` declaration, or refuse it."""

    product_id = str(product_id or "").strip()
    if not product_id:
        raise ValueError(f"{label} product_id must be non-empty")
    if ":" in product_id:
        raise ValueError(
            f"{label} product_id is the bare product name; drop the kind "
            f"prefix from {product_id!r}"
        )
    kind = str(kind or "").strip()
    if kind not in known_kinds:
        raise ValueError(
            f"unknown {label} {kind!r}; the host produces {sorted(known_kinds)!r}"
        )
    return product_id, kind


def _one_product_per_kind(
    products: Sequence[Any],
    *,
    kind_attribute: str,
    label: str,
) -> None:
    """Refuse a declaration that promises N products and answers fewer.

    Both hosted product families write some of their answers to more than one
    filename, so "as many distinct files as products" cannot tell a reader
    promised two tables from a reader handed one table twice.  Distinct kinds
    can, which is the whole reason these declarations are keyed on the answer.
    """

    ids = [item.product_id for item in products]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} product_id values must be unique")
    kinds = [getattr(item, kind_attribute) for item in products]
    if len(kinds) != len(set(kinds)):
        raise ValueError(
            f"each {label} may back at most one declared product; two products "
            f"naming the same one would be one answer promised twice "
            f"({sorted(kinds)!r})"
        )


def _spec_names_only_products_the_step_declares(
    expected_outputs: Sequence[Any],
    *,
    spec: Any,
    field: str,
) -> None:
    """Refuse a spec entry describing work the step never promised a reader.

    This is the one direction that is malformed on its own terms: it is a
    contradiction inside the step, true or false without reference to any
    runner, so it belongs here.  Whether a runner can *back* the declared
    products is the opposite direction, and it is a capability question --
    see :func:`spec_backs_every_declared_product`.
    """

    declared_products = {
        str(value or "").strip().split(":", 1)[-1] for value in expected_outputs
    }
    undeclared = sorted(
        item.product_id
        for item in spec.products
        if item.product_id not in declared_products
    )
    if undeclared:
        raise ValueError(
            f"{field} names products the step does not declare as outputs: "
            f"{undeclared!r}"
        )


_EXACT_TYPED_TABLE_INPUT = re.compile(r"table:[a-z][a-z0-9_]*")


def inputs_owing_a_consumption_contract(inputs: Sequence[Any]) -> set[str]:
    """The declared inputs a consumption contract has to decide about.

    Only a ``table:`` input has rows, and the contract's whole content is
    whether they are all consumed or a subset is. A ``statistic:`` input is one
    finite number in a JSON sidecar, so ``mode="all_rows"`` over it decides
    nothing; this validator has always exempted it.

    It lives here, called by both readers, because the other reader --
    ``execution/runners/figure_input_capability._contracts_match`` -- carried
    its own stricter copy and demanded a contract for *every* declared input.
    A step that satisfied this schema was then refused by the capability behind
    it, so the renderer silently declined and the figure went to the Coder.
    Measured 2026-07-30 on today's plans: 7 of 21 visualization steps lost
    their deterministic owner that way, and in 100% of them the inputs without
    a contract were exactly the statistics.
    """

    return {
        str(value)
        for value in inputs or ()
        if _EXACT_TYPED_TABLE_INPUT.fullmatch(str(value))
    }


def spec_backs_every_declared_product(
    expected_outputs: Sequence[Any],
    *,
    spec: Any,
    lookup: str,
    allowed_kinds: frozenset[str],
) -> bool:
    """Whether this declaration accounts for every product the step promises.

    A coverage shortfall makes the step *unowned*; it does not make the plan
    invalid.  That distinction is not cosmetic -- these rules first shipped as
    ``AnalysisStep`` validators, and a real fresh run then wrote a plan whose
    robustness step declared ``robustness_summary`` under two kinds.  The plan
    was sealed as evidence, and the failure surfaced only when the host
    re-parsed its own registered artifact: the parse raised, the plan-authority
    resolver swallowed it as an unreadable record, and the run died before its
    first step with "current analysis plan is not bound to immutable
    EvidenceStore authority" -- a message naming nothing about the real cause.

    A schema validator answers "is this statement well formed"; it must not
    also answer "can anything here execute it", because the second question has
    a safe answer (nobody claims the step, the Coder writes it) and the first
    one does not.
    """

    declared = [str(value or "").strip() for value in expected_outputs]
    if any(
        ":" not in value or value.partition(":")[0] not in allowed_kinds
        for value in declared
    ):
        return False
    declared_products = [value.split(":", 1)[1] for value in declared]
    if len(set(declared_products)) != len(declared_products):
        # The real recorded robustness contracts declare
        # `table:robustness_summary` AND `statistic:robustness_summary`.  The
        # legacy path normalises those to one name and satisfies both from one
        # CSV, so the reader who asked for a statistic is handed a table.  The
        # host cannot tell which one a bare product name means, so it declines.
        return False
    resolve = getattr(spec, lookup)
    return all(resolve(product) is not None for product in declared_products)


class MeasurementAuditProduct(BaseModel):
    """One declared audit product, said in terms of what it answers."""

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(
        description=(
            "The exact declared product id without its `table:` prefix, so the "
            "step keeps whatever name its reader expects."
        ),
    )
    audit: str = Field(
        description=(
            "Which audit this product IS, from the closed set the host can "
            "compute. This is the contract; `product_id` is only its label."
        ),
    )

    @model_validator(mode="after")
    def _exact_product_and_known_audit(self) -> "MeasurementAuditProduct":
        self.product_id, self.audit = _exact_product_naming_a_known_kind(
            self.product_id,
            self.audit,
            known_kinds=MEASUREMENT_AUDIT_KINDS,
            label="measurement audit",
        )
        return self


#: The two products a typed cohort-definition step promises. One spelling
#: each, because the recorded plans used four for the cohort artifact
#: (`artifact:analysis_cohort` 111, `cohort:analysis_set` 37,
#: `dataset:analysis_cohort` 4, `cohort:<bench name>` 4) and four for the flow
#: table (`table:cohort_flow` 131, `cohort_attrition` 11, `attrition_flow` 5,
#: `attrition` 2) -- and a host owner that had to recognise all of them would
#: be the string-set matching this codebase keeps having to delete.
COHORT_DEFINITION_COHORT_OUTPUT = "artifact:analysis_cohort"
COHORT_DEFINITION_FLOW_OUTPUT = "table:cohort_flow"


class CohortEligibilityCriterion(BaseModel):
    """One documented eligibility criterion of the materialised cohort."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(
        min_length=1,
        description=(
            "The identifier this criterion carries in the attrition flow "
            "table, so a reader can match the narrative to the row."
        ),
    )
    description: str = Field(
        min_length=1,
        description=(
            "What the criterion required, in the manuscript's own words. It "
            "is documentation of an eligibility rule the bound cohort already "
            "satisfies, not a filter this step will evaluate."
        ),
    )


class CohortDefinitionSpec(BaseModel):
    """Planner-owned statement that the analysis set IS the bound universe.

    The first analysis step of every recorded plan defines the analytic cohort
    and reports its attrition, and it is the single most expensive step in the
    system: measured 2026-08-02 over the recorded runs, 127 of 127 were written
    by the Coder, 21 of them failed, and each failure killed a mean of 5.1
    downstream steps -- 108 dead steps, 59% of every cascade in the corpus. The
    failure reasons do not repeat (ten distinct ones, none more than twice,
    eight with no recorded finding at all): there is no single defect to fix,
    because the step is improvised afresh every run.

    It is also, in practice, arithmetic the host can do. The cohort is already
    materialised and digest-bound before the run; the step reads it, declares
    it the analysis set, accounts for the drop from the universe and emits an
    identity receipt. Over 196 recorded attrition tables, 191 excluded ZERO
    rows -- the multi-row flows document criteria that removed nobody.

    ``analysis_set`` is a closed single-member literal on purpose. This spec
    claims host ownership, so it must be impossible to declare for a step that
    really does filter rows: a study that needs an executable exclusion simply
    omits the spec and keeps the generated-code path. Widening this literal
    later is a capability decision, not a typo fix.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.cohort_definition/1"] = (
        "easyicu.cohort_definition/1"
    )
    analysis_set: Literal["bound_universe"] = Field(
        default="bound_universe",
        description=(
            "The only analysis set this typed step can produce: every row of "
            "the bound, digest-verified cohort. It carries its one legal value "
            "as a default because a constant the Planner must retype is a "
            "constant it can misspell -- three canonical tasks once produced "
            "no plan at all that way. The ownership claim is presenting this "
            "spec, not typing this word; the Literal still rejects every "
            "other spelling."
        ),
    )
    identity_column: str = Field(
        min_length=1,
        description=(
            "The stable row key the identity receipt hashes. Declared because "
            "an executor that picked it from column order would bind the "
            "receipt to whichever column happened to come first."
        ),
    )
    eligibility_criteria: List[CohortEligibilityCriterion] = Field(
        default_factory=list,
        description=(
            "The eligibility already applied upstream, one entry per row the "
            "attrition table should carry. Each excludes no rows here -- they "
            "document how the bound cohort came to be, which is what the "
            "manuscript's flow diagram reports."
        ),
    )

    @model_validator(mode="after")
    def _criterion_ids_are_unique(self) -> "CohortDefinitionSpec":
        ids = [criterion.criterion_id for criterion in self.eligibility_criteria]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "cohort_definition_spec eligibility_criteria criterion_id "
                "values must be unique; the attrition table is keyed by them"
            )
        return self


class MeasurementAuditSpec(BaseModel):
    """Planner-owned statement of which audit each declared product is.

    An audit step is pure counting: nothing in it needs a model to reason, and
    every one of these products is something the host already computes.  What
    the host could not do was recognise the step, because recognition was a
    lookup of the declared product id in a fixed alias table.  This spec removes
    the name from that decision.

    Two products may not claim the same audit.  Several of the host's audit
    tables are written under more than one filename, so "distinct file" is not
    the same statement as "distinct answer": without this rule a step could
    promise a reader two tables and be satisfied by one table written twice.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.measurement_audit/1"] = (
        "easyicu.measurement_audit/1"
    )
    products: List[MeasurementAuditProduct] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_product_per_audit(self) -> "MeasurementAuditSpec":
        _one_product_per_kind(
            self.products,
            kind_attribute="audit",
            label="measurement audit",
        )
        return self

    def audit_for(self, product_id: str) -> Optional[str]:
        """Return the declared audit for one bare product id."""

        wanted = str(product_id or "").strip()
        for item in self.products:
            if item.product_id == wanted:
                return item.audit
        return None


class RobustnessReplayProduct(BaseModel):
    """One declared robustness product, said in terms of what it answers."""

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(
        description=(
            "The exact declared product id without its `table:`/`statistic:`/"
            "`log:` prefix, so the step keeps whatever name its reader expects."
        ),
    )
    output: str = Field(
        description=(
            "Which replay output this product IS, from the closed set the host "
            "produces. This is the contract; `product_id` is only its label."
        ),
    )

    @model_validator(mode="after")
    def _exact_product_and_known_output(self) -> "RobustnessReplayProduct":
        self.product_id, self.output = _exact_product_naming_a_known_kind(
            self.product_id,
            self.output,
            known_kinds=ROBUSTNESS_REPLAY_OUTPUTS,
            label="robustness replay output",
        )
        return self


class RobustnessReplaySpec(BaseModel):
    """Planner-owned statement that this step is the locked-spec replay.

    Declaring it is a scientific claim, not a formatting choice: it says the
    step re-estimates the ALREADY-LOCKED specification grid and changes no
    estimand.  A sensitivity analysis that introduces new science -- a
    different estimator, a causal-emulation variant, an E-value -- is not this
    step, and must not carry this spec merely to reach a deterministic runner.

    Given the claim, the method label and the product names stop deciding
    anything, which is the point: the runner's method allowlist holds three
    heads, and the corpus is full of steps that mean exactly this replay while
    spelling it `prespecified_sensitivity_analysis` or `sensitivity_analysis`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.robustness_replay/1"] = (
        "easyicu.robustness_replay/1"
    )
    products: List[RobustnessReplayProduct] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_product_per_output(self) -> "RobustnessReplaySpec":
        _one_product_per_kind(
            self.products,
            kind_attribute="output",
            label="robustness replay output",
        )
        return self

    def output_for(self, product_id: str) -> Optional[str]:
        """Return the declared replay output for one bare product id."""

        wanted = str(product_id or "").strip()
        for item in self.products:
            if item.product_id == wanted:
                return item.output
        return None


class AnalysisStep(BaseModel):
    """One step in a planner-emitted analysis plan."""

    model_config = ConfigDict(extra="forbid")

    # Local execution authority is deliberately absent from model_dump/json.
    # It may contain digest-verified categorical labels that must never return
    # to Planner, Replanner, Coder, or repair prompts.
    _table_one_execution_binding: Any = PrivateAttr(default=None)
    # Same rule, for every other level set the Planner declared in the host's
    # own opaque placeholders (see authority/declared_levels.py).
    _declared_level_binding: Any = PrivateAttr(default=None)

    step_id: str
    planned_analysis_role: PlannedAnalysisRole = Field(
        default="auxiliary",
        description=(
            "Planner-owned scientific role of this step. Exactly one step may "
            "be 'primary'; plans may also have no primary step. Host/internal "
            "construction defaults to 'auxiliary', while Planner LLM responses "
            "must explicitly declare the role for every step."
        ),
    )
    intent: str = Field(
        ..., description="One-sentence description of what this step does."
    )
    inputs: List[str] = Field(
        default_factory=list, description="Variable names or evidence ids consumed."
    )
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="Logical outputs — table_name, figure_name, statistic_name.",
    )
    method: Optional[str] = None
    scientific_action_id: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$",
        description=(
            "Exact family.method_key selected from the host-published scientific "
            "action catalog. It binds a reviewed method/resource coordinate; it "
            "does not upgrade Coder-generated work to a deterministic owner or "
            "publication-ready claim. Null preserves historical plans and steps "
            "that are not scientific analyses."
        ),
    )
    scientific_capability: Optional[str] = Field(
        default=None,
        description=(
            "Which registered scientific capability this result step executes. "
            "For an association primary step this selects the host-owned exact "
            "single-model contract or an agent-coded free-form kernel. A "
            "host-compiled binary sensitivity child may reuse the agent-coded "
            "association capability under a narrower inherited contract. "
            "Null selects the family default, which for association is the "
            "deterministic contract -- so an under-declared plan cannot obtain "
            "the looser agent-coded obligations by simply not declaring the "
            "typed one. Nothing infers this from the method string: routing a "
            "capability by keyword is how a feasibility audit and an "
            "interaction model become indistinguishable to a validator."
        ),
    )
    descriptive_claim: Optional[DescriptiveClaimContract] = Field(
        default=None,
        description=(
            "Typed descriptive-only claim ceiling and retained limitations. "
            "Scientific review accepts it only for a closed descriptive "
            "method/product shape; prose cannot create this exemption."
        ),
    )
    icu_rule_refs: List[str] = Field(default_factory=list)
    sensitivity_spec_ids: List[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Exact StudyContext sensitivity authorities this executable step "
            "implements. The host validates these ids against sealed user "
            "preferences; descriptive prose is not a binding."
        ),
    )
    literature_citation_keys: List[str] = Field(
        default_factory=list,
        description=(
            "Exact keys from this run's pre-plan LiteratureBundle that support "
            "the step's scientific design or method. Empty means no citation "
            "was bound; it must not be filled retrospectively by a renderer."
        ),
    )
    literature_design_bindings: List[LiteratureDesignBinding] = Field(
        default_factory=list,
        description=(
            "Typed article-to-design adoption records for this exact step. "
            "Each citation_key must also appear in literature_citation_keys; "
            "the host joins it to the sealed source record rather than trusting "
            "Planner-authored quotations."
        ),
    )

    @field_validator("literature_citation_keys")
    @classmethod
    def _validate_literature_citation_keys(cls, values: List[str]) -> List[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}", item) for item in cleaned):
            raise ValueError("literature_citation_keys must contain stable citation keys")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("literature_citation_keys must be unique")
        return cleaned

    @field_validator("sensitivity_spec_ids")
    @classmethod
    def _validate_sensitivity_spec_ids(cls, values: List[str]) -> List[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", item)
            for item in cleaned
        ):
            raise ValueError("sensitivity_spec_ids must contain stable ids")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("sensitivity_spec_ids must be unique")
        return cleaned
    model_requirements: List[PlannedModelRequirement] = Field(
        default_factory=list,
        description=(
            "Optional planner-owned roster for the supported binary/continuous "
            "adjusted-association contract only. Required entries must be matched "
            "by execution model contracts; an empty roster preserves the legacy "
            "step contract and is required for other analysis families."
        ),
    )
    family_primary_result_requirement: Optional[FamilyPrimaryResultRequirement] = Field(
        default=None,
        description=(
            "Planner-owned causal/survival headline-result requirement. "
            "It binds the scientific estimand to one registered result table; "
            "it is not an adjusted-association model roster."
        ),
    )
    input_consumption_contracts: List[ArtifactConsumptionContract] = Field(
        default_factory=list,
        description=(
            "Optional Planner-owned cardinality/role rules for exact typed table "
            "inputs. Rendering-only children synthesized by the host receive "
            "all_rows contracts so they cannot silently collapse a multi-row "
            "source to one row."
        ),
    )
    figure_panels: List[PlannedFigurePanelSpec] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Typed article role, chart family, figure product, and exact source "
            "products for every planned figure panel. Figure names, input table "
            "names, and intent prose are not substitutes for this declaration."
        ),
    )
    table_one_spec: Optional[TableOneSpec] = Field(
        default=None,
        description=(
            "Planner-owned grouping, variable summaries, and tests for an exact "
            "table:table_one output. Overall-only descriptive summaries must use "
            "a different product name instead of masquerading as Table 1."
        ),
    )
    trajectory_stability_spec: Optional[TrajectoryStabilitySpec] = Field(
        default=None,
        description=(
            "Optional PlannerAgent-owned resampling design for a dedicated "
            "trajectory stability/freeze step. The standard executor is "
            "eligible only when this complete typed packet is present."
        ),
    )
    exposure_outcome_distribution_spec: Optional[ExposureOutcomeDistributionSpec] = (
        Field(
            default=None,
            description=(
                "Planner-owned exposure, outcome, event value, denominator "
                "policy and interval method for an exact "
                "table:exposure_outcome_distribution output. Without it the "
                "step has no typed statement of which column is which, and "
                "the host must not infer that from names or input order."
            ),
        )
    )
    cohort_definition_spec: Optional[CohortDefinitionSpec] = Field(
        default=None,
        description=(
            "Planner-owned statement that this step's analysis set is exactly "
            "the bound universe, plus the identity column and the eligibility "
            "the attrition table should report. Present it only when no "
            "exclusion remains for the step to apply; a step that really "
            "filters rows omits it and keeps the generated-code path."
        ),
    )
    measurement_audit_spec: Optional[MeasurementAuditSpec] = Field(
        default=None,
        description=(
            "For a counting-only measurement/missingness audit step: which "
            "audit each declared table product IS. Declaring it lets the step "
            "keep whatever product names its reader expects while still being "
            "executed by the host, instead of being demoted to generated code "
            "because the name was not one the host had seen before."
        ),
    )
    robustness_replay_spec: Optional[RobustnessReplaySpec] = Field(
        default=None,
        description=(
            "For a step that re-estimates the already-locked robustness "
            "specification grid and changes no estimand: which replay output "
            "each declared product IS. Declare it only when that is genuinely "
            "the step; a sensitivity analysis introducing new science is a "
            "different step and must not claim it to reach a host runner."
        ),
    )

    @field_validator("scientific_capability")
    @classmethod
    def _validate_stable_capability_id(cls, value: Optional[str]) -> Optional[str]:
        return validate_scientific_capability_id(value)

    def required_primary_exposure_sources(self) -> tuple[str, ...]:
        """Return required PRIMARY sources from the Planner-owned model roster."""

        return tuple(
            dict.fromkeys(
                item.exposure_source
                for item in self.model_requirements
                if item.analysis_role == "primary" and item.required_for_step_success
            )
        )

    def primary_exposure_authority_sources(
        self,
        declared_primary: str,
        operational_sources: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Unify the clinical concept, prior alias, and typed model source."""

        values = [str(declared_primary or "").strip()]
        values.extend(str(value or "").strip() for value in operational_sources)
        values.extend(self.required_primary_exposure_sources())
        return tuple(dict.fromkeys(value for value in values if value))

    def without_required_primary_exposure_terms(
        self,
        terms: Sequence[str],
    ) -> List[str]:
        """Remove only exact typed PRIMARY sources from adjustment terms."""

        sources = {
            source.casefold() for source in self.required_primary_exposure_sources()
        }
        return [term for term in terms if str(term).strip().casefold() not in sources]

    @model_validator(mode="after")
    def _model_requirement_ids_are_unique(self) -> "AnalysisStep":
        binding_keys = [item.citation_key for item in self.literature_design_bindings]
        if len(binding_keys) != len(set(binding_keys)):
            raise ValueError(
                "literature_design_bindings must contain at most one record per citation_key"
            )
        unbound_design_keys = sorted(
            set(binding_keys) - set(self.literature_citation_keys)
        )
        if unbound_design_keys:
            raise ValueError(
                "literature_design_bindings citation_key values must also appear "
                "in literature_citation_keys; unbound " + ", ".join(unbound_design_keys)
            )
        if self.table_one_spec is not None:
            if "table:table_one" not in self.expected_outputs:
                raise ValueError(
                    "table_one_spec requires expected output 'table:table_one'"
                )
            declared_outputs = {
                str(value or "").strip() for value in self.expected_outputs
            }
            unsupported_outputs = sorted(declared_outputs - TABLE_ONE_CLOSED_OUTPUTS)
            if unsupported_outputs:
                raise ValueError(
                    "table_one_spec supports only the closed host-executable "
                    "outputs 'table:table_one', optional 'table:cohort_flow', "
                    "and optional 'log:source_row_count_reconciliation'; "
                    f"unsupported {unsupported_outputs!r}"
                )
            required_inputs = {
                self.table_one_spec.group_by,
                *(item.name for item in self.table_one_spec.variables),
            }
            missing_inputs = sorted(required_inputs - set(self.inputs))
            if missing_inputs:
                raise ValueError(
                    "table_one_spec variables must be explicit step inputs; "
                    f"missing {missing_inputs!r}"
                )
        if self.exposure_outcome_distribution_spec is not None:
            if "table:exposure_outcome_distribution" not in self.expected_outputs:
                raise ValueError(
                    "exposure_outcome_distribution_spec requires expected output "
                    "'table:exposure_outcome_distribution'"
                )
            spec_columns = {
                self.exposure_outcome_distribution_spec.exposure,
                self.exposure_outcome_distribution_spec.outcome,
            }
            missing_spec_inputs = sorted(spec_columns - set(self.inputs))
            if missing_spec_inputs:
                raise ValueError(
                    "exposure_outcome_distribution_spec exposure and outcome must "
                    f"be explicit step inputs; missing {missing_spec_inputs!r}"
                )
        if self.cohort_definition_spec is not None:
            # The step must promise BOTH halves the spec describes: the cohort
            # it declares and the attrition table that accounts for it. A spec
            # attached to a step promising only one of them would claim host
            # ownership of work the step never said it would deliver.
            declared = list(self.expected_outputs)
            if COHORT_DEFINITION_COHORT_OUTPUT not in declared:
                raise ValueError(
                    "cohort_definition_spec requires expected output "
                    f"{COHORT_DEFINITION_COHORT_OUTPUT!r}"
                )
            if COHORT_DEFINITION_FLOW_OUTPUT not in declared:
                raise ValueError(
                    "cohort_definition_spec requires expected output "
                    f"{COHORT_DEFINITION_FLOW_OUTPUT!r}"
                )
        if self.measurement_audit_spec is not None:
            _spec_names_only_products_the_step_declares(
                self.expected_outputs,
                spec=self.measurement_audit_spec,
                field="measurement_audit_spec",
            )
        if self.robustness_replay_spec is not None:
            _spec_names_only_products_the_step_declares(
                self.expected_outputs,
                spec=self.robustness_replay_spec,
                field="robustness_replay_spec",
            )
        consumption_keys = [
            contract.input_key for contract in self.input_consumption_contracts
        ]
        if len(consumption_keys) != len(set(consumption_keys)):
            raise ValueError(
                "input_consumption_contracts input_key values must be unique"
            )
        missing_consumption_inputs = sorted(set(consumption_keys) - set(self.inputs))
        if missing_consumption_inputs:
            raise ValueError(
                "input_consumption_contracts must target exact inputs on the same "
                f"step; missing {missing_consumption_inputs!r}"
            )
        if (
            str(self.method or "").strip().lower().split(" with ", 1)[0]
            == "visualization"
            and consumption_keys
        ):
            if set(consumption_keys) != inputs_owing_a_consumption_contract(
                self.inputs
            ):
                raise ValueError(
                    "visualization input_consumption_contracts must cover every "
                    "exact typed table input"
                )
        if self.figure_panels:
            method_head = str(self.method or "").strip().lower().split(" with ", 1)[0]
            if method_head != "visualization":
                raise ValueError(
                    "figure_panels are valid only on method='visualization' steps"
                )
            panel_ids = [panel.panel_id for panel in self.figure_panels]
            if len(panel_ids) != len(set(panel_ids)):
                raise ValueError("figure_panels panel_id values must be unique per step")
            declared_outputs = set(self.expected_outputs)
            declared_inputs = set(self.inputs)
            undeclared_outputs = sorted(
                {
                    panel.figure_output
                    for panel in self.figure_panels
                    if panel.figure_output not in declared_outputs
                }
            )
            if undeclared_outputs:
                raise ValueError(
                    "figure_panels figure_output values must be declared in "
                    f"expected_outputs; missing {undeclared_outputs!r}"
                )
            undeclared_sources = sorted(
                {
                    source
                    for panel in self.figure_panels
                    for source in panel.source_products
                    if source not in declared_inputs
                }
            )
            if undeclared_sources:
                raise ValueError(
                    "figure_panels source_products must be exact step inputs; "
                    f"missing {undeclared_sources!r}"
                )
        requirement_ids = [item.requirement_id for item in self.model_requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("model_requirements requirement_id values must be unique")
        if self.model_requirements:
            primary_requirements = [
                item
                for item in self.model_requirements
                if item.analysis_role == "primary"
            ]
            if self.planned_analysis_role == "primary" and not primary_requirements:
                raise ValueError(
                    "a primary adjusted-association step must declare at least one "
                    "primary model requirement"
                )
            method_head = str(self.method or "").lower().split(" with ", 1)[0]
            method = _normalise_model_contract_token(method_head)
            outputs = set()
            for output in self.expected_outputs:
                output_kind, separator, output_name = str(output).partition(":")
                if not separator:
                    continue
                outputs.add(
                    (
                        _normalise_model_contract_token(output_kind),
                        _normalise_model_contract_token(output_name),
                    )
                )
            model_requirement_methods = {
                PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
                # The signed landmark route preserves the same explicit
                # adjusted-association requirement and changes only the
                # cohort/executor authority. It must survive a plan
                # serialize/rehydrate boundary after host binding.
                "signed_landmark_categorical_association",
            }
            if (
                method not in model_requirement_methods
                or (
                    PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
                    PLANNED_MODEL_REQUIREMENTS_OUTPUT,
                )
                not in outputs
            ):
                raise ValueError(
                    "model_requirements are currently supported only on "
                    "the adjusted-association or signed categorical-landmark "
                    "methods when they declare "
                    "expected output 'table:adjusted_association_estimates'; "
                    "other analysis families must use their family-specific "
                    "planning and validation contracts"
                )
        family_requirement = self.family_primary_result_requirement
        if family_requirement is not None:
            if self.planned_analysis_role != "primary":
                raise ValueError(
                    "family_primary_result_requirement belongs only on the primary step"
                )
            if family_requirement.expected_result_product not in self.expected_outputs:
                raise ValueError(
                    "family_primary_result_requirement expected_result_product must "
                    "be declared in this step's expected_outputs"
                )
        return self


class KnowHowDecision(BaseModel):
    """Planner disposition for one retrieved, evidence-bound protocol claim."""

    model_config = ConfigDict(extra="forbid")

    card_id: str
    card_version: str
    card_sha256: str
    claim_id: str
    disposition: Literal["adopted", "rejected", "unresolved", "requires_confirmation"]
    reason_code: str
    rationale: str = Field(max_length=500)
    citation_ids: List[str] = Field(min_length=1, max_length=8)

    @field_validator("card_id", "claim_id", "reason_code")
    @classmethod
    def _validate_stable_id(cls, value: str) -> str:
        value = str(value or "").strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,79}", value):
            raise ValueError(
                "know-how decision ids must be stable lowercase identifiers"
            )
        return value

    @field_validator("card_version")
    @classmethod
    def _validate_card_version(cls, value: str) -> str:
        value = str(value or "").strip()
        if not re.fullmatch(r"[1-9][0-9]*\.[0-9]+\.[0-9]+", value):
            raise ValueError("know-how decision card_version must be semantic x.y.z")
        return value

    @field_validator("card_sha256")
    @classmethod
    def _validate_card_sha(cls, value: str) -> str:
        value = str(value or "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("know-how decision card_sha256 must be SHA-256")
        return value

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, value: str) -> str:
        value = " ".join(str(value or "").split())
        if not value:
            raise ValueError("know-how decision rationale must be non-empty")
        return value

    @field_validator("citation_ids")
    @classmethod
    def _validate_citations(cls, values: List[str]) -> List[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{2,79}", item) for item in cleaned):
            raise ValueError("know-how decision citation_ids must be stable ids")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("know-how decision citation_ids must be unique")
        return cleaned


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_question: str
    analysis_type: Optional[str] = Field(
        default=None,
        description=(
            "EHR analysis family declared by the agent/planner (e.g. survival, "
            "trajectory_clustering, prediction_model). When omitted, the framework "
            "may attach an inferred family suggestion for review; it must not use "
            "that suggestion to replace the agent's scientific method or estimand."
        ),
    )
    steps: List[AnalysisStep]
    cohort: Optional[CohortDefinition] = Field(
        default=None,
        description=(
            "Typed primary cohort definition. If supplied, every predicate must "
            "include concept_id, time_window, aggregation, operator and value."
        ),
    )
    endpoint: Optional[EndpointSpec] = Field(
        default=None,
        description=(
            "Backward-compatible projection of ResearchContext.endpoint. It is "
            "not independent authority: when present it must equal the sealed "
            "context declaration exactly."
        ),
    )
    evalue_conversion_spec: Optional[EValueConversionSpec] = Field(
        default=None,
        description=(
            "Planner-owned evidence and population binding for optional odds-"
            "ratio E-value conversion. No declaration means no OR conversion."
        ),
    )
    subgroup_analysis_spec: Optional[SubgroupAnalysisSpec] = Field(
        default=None,
        description=(
            "Planner-owned predictor, outcome, subgroup axes, binning, sample-"
            "size thresholds, and multiplicity family for optional O24 work."
        ),
    )
    robustness_specs: List[RobustnessSpec] = Field(
        default_factory=list,
        description=(
            "Pre-specified robustness specifications locked before execution. "
            "When present, the specs must cover cohort, missingness, and outcome axes."
        ),
    )
    display_labels: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Planner/run-owned human-facing labels keyed by exact variable, "
            "contrast, or robustness-spec id. Renderers may consume these "
            "labels but must not invent clinical endpoint semantics from a "
            "column name."
        ),
    )
    know_how_decisions: List[KnowHowDecision] = Field(
        default_factory=list,
        description=(
            "Claim-level dispositions for retrieved protocol advice. Card, claim, "
            "citation, version, and SHA coordinates are validated against this run's "
            "deterministic retrieval. Empty decisions are omitted for legacy/default-off runs."
        ),
    )
    design_selection: Optional[ResearchDesignSelection] = Field(default=None, exclude_if=lambda value: value is None)
    rationale: Optional[str] = None
    revision: int = 1

    @field_validator("know_how_decisions")
    @classmethod
    def _validate_know_how_decisions(
        cls, values: List[KnowHowDecision]
    ) -> List[KnowHowDecision]:
        coordinates = [(item.card_id, item.claim_id) for item in values]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("know_how_decisions must not repeat a card/claim pair")
        return values

    @model_serializer(mode="wrap")
    def _serialize_optional_know_how_decisions(self, handler: Any) -> Dict[str, Any]:
        payload = handler(self)
        if not self.know_how_decisions:
            payload.pop("know_how_decisions", None)
        return payload

    @field_validator("display_labels", mode="before")
    @classmethod
    def _validate_display_labels(cls, value: Any) -> Dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("display_labels must be an object mapping ids to labels")
        labels: Dict[str, str] = {}
        normalized_keys: Dict[str, str] = {}
        for raw_key, raw_label in value.items():
            key = str(raw_key or "").strip()
            label = " ".join(str(raw_label or "").split())
            if not key or not label:
                raise ValueError("display_labels keys and values must be non-empty")
            if len(key) > 256 or len(label) > 256:
                raise ValueError(
                    "display_labels keys and values must be <=256 characters"
                )
            normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
            if not normalized:
                raise ValueError(
                    "display_labels keys must contain at least one letter or digit"
                )
            prior = normalized_keys.get(normalized)
            if prior is not None and prior != label:
                raise ValueError(
                    f"display_labels contains conflicting normalized keys: {key!r}"
                )
            normalized_keys[normalized] = label
            labels[key] = label
        return labels

    @field_validator("cohort", mode="before")
    @classmethod
    def _coerce_cohort_definition(cls, value: Any) -> Optional[CohortDefinition]:
        if value is None:
            return None
        try:
            return coerce_cohort_definition(value)
        except CohortSchemaError as exc:
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _validate_robustness_specs(self) -> "AnalysisPlan":
        step_ids = [str(step.step_id) for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("analysis plan step_id values must be unique")
        primary_step_ids = [
            str(step.step_id)
            for step in self.steps
            if step.planned_analysis_role == "primary"
        ]
        if len(primary_step_ids) > 1:
            raise ValueError(
                "analysis plan may declare at most one step with "
                "planned_analysis_role='primary'; found " + ", ".join(primary_step_ids)
            )
        validate_scientific_capability_declarations(self.steps)
        for step in self.steps:
            if step.planned_analysis_role != "primary":
                continue
            typed_output_kinds = {
                _PRIMARY_RESULT_KIND_ALIASES.get(kind, kind)
                for raw in step.expected_outputs
                if isinstance(raw, str)
                and (separator := raw.strip().partition(":"))[1]
                and (kind := separator[0].strip().lower())
                and separator[2].strip()
            }
            if not (typed_output_kinds & _PRIMARY_SCIENTIFIC_RESULT_KINDS):
                raise ValueError(
                    "a planned primary step must declare at least one typed, "
                    "non-rendering scientific result product owned by the analysis"
                )
        if self.robustness_specs:
            try:
                # Structural validity only.  Constructing an ``AnalysisPlan`` is
                # not the same act as accepting one from the Planner: this path
                # also loads a recorded plan from disk, re-reads a lock on
                # resume, and builds the framework's own case-neutral
                # placeholders.  A requirement about what the *Planner must
                # declare* judged all of those too -- 190 of 409 recorded plan
                # documents stopped parsing, and a resume of any of them would
                # have failed at load.  The Planner-output rule is applied where
                # the Planner's output is accepted, in ``agents.core``, so it
                # still reaches the model that can act on it.
                validate_robustness_specs(self.robustness_specs)
            except RobustnessPlanError as exc:
                raise ValueError(str(exc)) from exc
        return self


class EvidenceRef(BaseModel):
    """Typed reference to a registered evidence artifact."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    kind: Optional[str] = None
    description: Optional[str] = None
    relative_path: Optional[str] = None


class ConceptRef(BaseModel):
    """Stable reference to an ICU concept or analysis variable."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: Optional[VariableRole] = None
    source_concept: Optional[str] = None
    analysis_window: Optional[str] = None


class ClinicalSemanticsResolution(BaseModel):
    """Structured ICU-semantic interpretation for planning/execution."""

    model_config = ConfigDict(extra="forbid")

    analysis_family: str
    target_outcome: Optional[str] = None
    temporal_constraints: List[TemporalConstraint] = Field(default_factory=list)
    recommended_time_windows: List[TimeWindow] = Field(default_factory=list)
    target_concepts: List[ConceptRef] = Field(default_factory=list)
    ambiguity_notes: List[str] = Field(default_factory=list)
    safety_guardrails: List[str] = Field(default_factory=list)
    provenance_notes: List[str] = Field(default_factory=list)


class DataExtractionRequest(BaseModel):
    """Typed request for deterministic cohort/concept retrieval."""

    model_config = ConfigDict(extra="forbid")

    cohort_name: str
    database: str
    concept_refs: List[ConceptRef] = Field(default_factory=list)
    time_windows: List[TimeWindow] = Field(default_factory=list)
    temporal_constraints: List[TemporalConstraint] = Field(default_factory=list)
    cohort_provenance: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class DataExtractionResult(BaseModel):
    """Typed result from the constrained data extraction layer."""

    model_config = ConfigDict(extra="forbid")

    cohort_path: str
    n_rows: int = Field(ge=0, default=0)
    concept_refs: List[ConceptRef] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)


class StatisticalAnalysisRequest(BaseModel):
    """Typed handoff from planner/runtime into the analysis agent."""

    model_config = ConfigDict(extra="forbid")

    step: AnalysisStep
    analysis_family: str
    target_outcome: Optional[str] = None
    covariates: List[str] = Field(default_factory=list)
    evaluation_focus: Optional[str] = None
    must_have_outputs: Optional[str] = None
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class StatisticalAnalysisResult(BaseModel):
    """Structured analysis summary returned after a step executes."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    method_family: str
    primary_estimate: Optional[float] = None
    estimate_label: Optional[str] = None
    estimate_interval: Optional[List[float]] = None
    summary_metrics: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    validator_messages: List[str] = Field(default_factory=list)


class VisualizationRequest(BaseModel):
    """Typed request for figure-generation agents and publication exporters."""

    model_config = ConfigDict(extra="forbid")

    step: AnalysisStep
    analysis_family: str
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    required_formats: List[str] = Field(
        default_factory=lambda: ["png", "svg", "pdf", "tiff"]
    )
    must_have_outputs: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class VisualizationResult(BaseModel):
    """Structured description of generated figures."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    figure_titles: List[str] = Field(default_factory=list)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    qa_messages: List[str] = Field(default_factory=list)


class ManuscriptDraftPacket(BaseModel):
    """Typed packet sent into manuscript drafting/binding."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = None
    abstract_focus: Optional[str] = None
    analysis_family: Optional[str] = None
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)


class CritiqueReport(BaseModel):
    """Structured critique used in execute→critique→revise loops."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pass", "needs_revision", "blocked"] = "pass"
    reviewer: str
    concerns: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    missing_evidence_refs: List[str] = Field(default_factory=list)
    suggested_repairs: List[str] = Field(default_factory=list)
    related_evidence_refs: List[EvidenceRef] = Field(default_factory=list)


class ReflectionMemoryEntry(BaseModel):
    """Persistable run-level reflection record without raw PHI."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["successful_workflow", "failed_pattern", "reusable_template"]
    summary: str
    analysis_family: Optional[str] = None
    trigger: Optional[str] = None
    recommendation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentRuntimeState(BaseModel):
    """Shared, typed state exchanged between supervisor and worker agents."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    analysis_family: Optional[str] = None
    semantics: Optional[ClinicalSemanticsResolution] = None
    extraction_request: Optional[DataExtractionRequest] = None
    extraction_result: Optional[DataExtractionResult] = None
    current_step: Optional[AnalysisStep] = None
    analysis_request: Optional[StatisticalAnalysisRequest] = None
    analysis_result: Optional[StatisticalAnalysisResult] = None
    visualization_request: Optional[VisualizationRequest] = None
    visualization_result: Optional[VisualizationResult] = None
    manuscript_packet: Optional[ManuscriptDraftPacket] = None
    critique: Optional[CritiqueReport] = None
    reflection_memory: List[ReflectionMemoryEntry] = Field(default_factory=list)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    """A registered, hashed analytical artefact.

    Manuscript-scaffold sentences may *only* cite ``evidence_id`` values
    that resolve to a record here. That constraint is enforced at write
    time, not at trust-the-LLM time.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    kind: Literal["table", "figure", "statistic", "log", "code"]
    description: str
    relative_path: str = Field(..., description="Path inside the evidence/ directory.")
    sha256: str
    produced_by_step: Optional[str] = None
    inputs: List[str] = Field(default_factory=list)
    script_evidence_id: Optional[str] = None
    producer: Optional[str] = Field(
        default=None,
        description="Which subsystem produced this artefact, e.g. planner/coder/writer/pipeline.",
    )
    generation_mode: Optional[str] = Field(
        default=None,
        description="Provenance mode such as llm, repaired, fallback, deterministic_skill, or system.",
    )
    prompt_pack_version: Optional[str] = None
    finding_severity: Optional[Literal["info", "warning", "error"]] = None
    finding_messages: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ValidationFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validator: str
    severity: Literal["info", "warning", "error"] = "info"
    message: str
    evidence_ids: List[str] = Field(default_factory=list)
    detail: Optional[Dict[str, Any]] = None


class CostRecord(BaseModel):
    """A single LLM call's token usage and (optionally) estimated cost.

    The pipeline accumulates these in ``AnalysisManifest.cost_records``
    when ``enable_cost_tracking=True``. Real OpenAI-compatible clients
    populate ``prompt_tokens`` / ``completion_tokens`` from the SDK's
    ``response.usage``; the mock client and any client that does not
    expose ``last_usage`` falls back to a 4-chars-per-token heuristic.

    ``estimated_cost_usd`` is populated only when the meter has a price
    table for the model. Reviewers should treat it as an order-of-
    magnitude estimate; provider invoices remain the source of truth.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role: Optional[str] = Field(
        default=None,
        description="planner / coder / analyzer / writer / literature, or None for unrouted calls.",
    )
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: Optional[float] = None
    is_heuristic: bool = Field(
        default=False,
        description="True when token counts came from the chars/4 fallback rather than the SDK.",
    )


class AnalysisManifest(BaseModel):
    """Top-level provenance manifest for a pipeline run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.research_manifest/1"
    checkpoint_sequence: Optional[int] = Field(default=None, ge=1)
    run_id: str
    research_question: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    context_path: str
    plan_path: Optional[str] = None
    current_plan_authority: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Exact immutable EvidenceStore plan selected for execution, including "
            "revision, evidence id, relative path, and SHA-256."
        ),
    )
    evidence: List[EvidenceRecord] = Field(default_factory=list)
    findings: List[ValidationFinding] = Field(default_factory=list)
    per_step_records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Execution records for planner steps, sorted in plan order. "
            "The same records are streamed to manifest_partial.json during a run."
        ),
    )
    step_attempt_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Append-only execution and deterministic-revalidation history. "
            "Live checkpoints retain the records inline. Final manifests may "
            "externalize them through step_attempt_history_ref; the authority "
            "loader restores the exact digest-bound records before resume."
        ),
    )
    step_attempt_history_ref: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Digest-bound JSONL reference used by finalized manifests to avoid "
            "embedding every full superseded attempt snapshot."
        ),
    )
    cost_records: List[CostRecord] = Field(default_factory=list)
    reproducibility: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Compact LLM reproducibility envelope (O20): per-call prompt/response "
            "sha256, requested seed, temperature, provider/model, and a "
            "PHI-safe environment snapshot. Populated when "
            "ResearchAgentPipeline(enable_reproducibility_envelope=True)."
        ),
    )
    provider_authorization: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Exact non-secret provider/model/endpoint destination and the "
            "factory authorization decision used by this run."
        ),
    )
    execution_identity: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Content-addressed profile/runner/image/network/provider/prompt/git/seed "
            "identity required for exact reuse and paper acceptance."
        ),
    )
    code_version: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Identity of the code that produced this run: git commit sha, "
            "branch, dirty-worktree flag, and installed easyicu package "
            "version. Lets a manifest be tied back to the exact source, "
            "which the reproducibility claim depends on. None only when "
            "capture failed entirely (e.g. no git and no package metadata)."
        ),
    )
    submission_profile_name: Optional[str] = Field(
        default=None,
        description="Name of the versioned paper-facing submission profile, if any.",
    )
    submission_profile_version: Optional[str] = Field(
        default=None,
        description="Version token for the paper-facing submission profile, if any.",
    )
    submission_profile_locked_at: Optional[str] = Field(
        default=None,
        description="Timestamp at which the submission profile was frozen.",
    )
    concept_dict_path: Optional[str] = Field(
        default=None,
        description="Package-relative EasyICU concept dictionary path used by the run.",
    )
    concept_dict_sha: Optional[str] = Field(
        default=None,
        description="SHA-256 of the EasyICU concept dictionary used by the run.",
    )
    sofa2_dict_path: Optional[str] = Field(
        default=None,
        description="Package-relative EasyICU SOFA-2 dictionary path used by the run.",
    )
    sofa2_dict_sha: Optional[str] = Field(
        default=None,
        description="SHA-256 of the EasyICU SOFA-2 dictionary used by the run.",
    )
    concept_dict_fingerprint: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Full concept-dictionary fingerprint including concept-dict and "
            "SOFA-2 dictionary paths, hashes, and computed_at timestamp."
        ),
    )
    readiness: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Fail-closed readiness gates for the run: execution_complete, "
            "evidence_complete, numeric_verified, analysis_validated, "
            "manuscript_ready and publication_ready."
        ),
    )
    artifact_paths: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Canonical run-level diagnostic artefacts such as run_status.json, "
            "claim_ledger.csv, evidence_audit.json, numeric_audit.json, "
            "author_review_note.md and, only when gated, manuscript_ready.md."
        ),
    )
    robustness_panel_path: Optional[str] = None
    robustness_panel_sha: Optional[str] = None
    robustness_n_variants: Optional[int] = None
    robustness_range_low: Optional[float] = None
    robustness_range_high: Optional[float] = None
    cohort_locked_path: Optional[str] = None
    cohort_locked_sha: Optional[str] = None
    side_findings_path: Optional[str] = None
    side_findings_sha: Optional[str] = None
    side_findings_count: int = 0
    writer_probe_mode: bool = False
    writer_probe_failed_steps: List[str] = Field(default_factory=list)
    report_path: Optional[str] = None
    manuscript_path: Optional[str] = None
    audit_log_path: Optional[str] = None
    workflow_graph_path: Optional[str] = None
    execution_replay_path: Optional[str] = None
    experiment_spec_path: Optional[str] = None
    llm_signature: Optional[str] = None
    used_mock_llm: bool = False
    prompt_pack_version: Optional[str] = None
    prompt_pack_files: Dict[str, str] = Field(default_factory=dict)
    notes: Optional[str] = None


class PipelineResult(BaseModel):
    """Lightweight return value from :meth:`ResearchAgentPipeline.run`."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    workdir: str
    context_path: str
    plan_path: str
    manifest_path: str
    report_path: str
    manuscript_path: str
    evidence_count: int
    findings_count: int
    paper_profile_path: Optional[str] = None
    replication_spec_path: Optional[str] = None
    replication_report_path: Optional[str] = None

    def as_paths(self) -> Dict[str, Path]:
        return {
            "context": Path(self.context_path),
            "plan": Path(self.plan_path),
            "manifest": Path(self.manifest_path),
            "report": Path(self.report_path),
            "manuscript": Path(self.manuscript_path),
        }


class PaperClaimRecord(BaseModel):
    """Typed representation of one original-paper result claim."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    section: str = "results"
    sentence: str
    metric: Optional[str] = None
    paper_value: Optional[str] = None
    numeric_value: Optional[float] = None
    direction: Optional[str] = None
    predictor: Optional[str] = None
    outcome: Optional[str] = None


class PaperProfile(BaseModel):
    """Deterministic paper-normalisation output for replication mode."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.paper_profile/1"
    paper_source: str
    paper_title: Optional[str] = None
    paper_type: Literal[
        "descriptive",
        "association",
        "prediction",
        "survival",
        "fairness",
        "causal-like",
        "unsupported_or_underspecified",
    ] = "unsupported_or_underspecified"
    research_question: str = ""
    target_outcome: Optional[str] = None
    cohort_definition: str = ""
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    primary_exposure: Optional[str] = None
    primary_predictor: Optional[str] = None
    covariates: List[str] = Field(default_factory=list)
    primary_analysis_method: str = ""
    secondary_analyses: List[str] = Field(default_factory=list)
    table_figure_inventory: List[str] = Field(default_factory=list)
    key_claims: List[PaperClaimRecord] = Field(default_factory=list)
    unsupported_reasons: List[str] = Field(default_factory=list)


class PaperReplicationSpec(BaseModel):
    """Typed execution contract derived from a paper profile."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.paper_replication_spec/1"
    paper_title: Optional[str] = None
    paper_type: str
    replication_goal: str = "design_and_conclusion_alignment"
    mapped_concepts: Dict[str, str] = Field(default_factory=dict)
    unmappable_items: List[str] = Field(default_factory=list)
    approximate_substitutions: Dict[str, str] = Field(default_factory=dict)
    time_windows: List[str] = Field(default_factory=list)
    required_outputs: List[str] = Field(default_factory=list)
    alignment_targets: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class PaperResultLedger(BaseModel):
    """Compact ledger of original-paper claims and EasyICU metrics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.paper_result_ledger/1"
    paper_claims: List[PaperClaimRecord] = Field(default_factory=list)
    easyicu_metrics: Dict[str, Any] = Field(default_factory=dict)


class ReplicationDeviationItem(BaseModel):
    """One explicit mismatch or approximation in a replication attempt."""

    model_config = ConfigDict(extra="forbid")

    item: str
    severity: Literal["info", "warning", "error"] = "warning"
    original: Optional[str] = None
    easyicu_proxy: Optional[str] = None
    reason: str


class ReplicationDeviationReport(BaseModel):
    """Fail-closed report of deviations between paper and EasyICU setup."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.replication_deviation_report/1"
    supported: bool = False
    summary: str = ""
    items: List[ReplicationDeviationItem] = Field(default_factory=list)


class ProbeSummary(BaseModel):
    """Schema for the deterministic-probe step's output dict.

    Built once per run by ``_build_probe_summary``. Represents the cohort
    snapshot the planner sees before the first analysis step runs:
    row/column counts, top-missing columns, and outcome-blind
    per-score completeness summaries when the cohort exposes component
    availability. ``extra='allow'`` so future probe metrics can be
    added without churning this schema.
    """

    model_config = ConfigDict(extra="allow")

    n_rows: int
    n_columns: int
    target_outcome: Optional[str] = None
    top_missing_columns: List[Dict[str, Any]] = Field(default_factory=list)
    score_completeness: List[Dict[str, Any]] = Field(default_factory=list)
    outcome_rate: Optional[float] = None


class StepRecord(BaseModel):
    """Schema for a per-step record accumulated into ``per_step_records``.

    Each entry in ``per_step_records`` is one (step_id, status, summary)
    snapshot of how a planner-emitted step ran. The shape grew
    organically across many pipeline code paths; this model centralises
    the field list so:

    * a reader can find the field set in one place;
    * resume / manifest-load code can validate persisted records via
      :meth:`StepRecord.model_validate`;
    * mypy can flag obvious typos on new code that constructs records.

    ``extra='allow'`` keeps backward compatibility with the existing
    dict-style construction sites (they remain valid; this schema is
    documentation-and-opt-in-validation rather than a forced rewrite).
    Field names mirror what is already written to disk in
    ``manifest.json`` and consumed by the resume path.
    """

    model_config = ConfigDict(extra="allow")

    # Always set on construction.
    step_id: str
    intent: str
    planned_analysis_role: Optional[PlannedAnalysisRole] = Field(
        default=None,
        description=(
            "Host-recorded copy of the Planner-owned step role. New execution "
            "records populate it; None remains readable for historical records."
        ),
    )

    # Lifecycle / status.
    status: Optional[str] = Field(
        default=None,
        description=(
            "Terminal state. Observed values include 'ok', "
            "'skipped_dependency_failed', 'coder_failed', "
            "'execution_failed', 'repair_failed', 'blocked_by_concept_audit'."
        ),
    )
    generation_mode: Optional[str] = Field(
        default=None,
        description="How the code was produced: 'llm', 'system', 'fallback', 'deterministic_probe'.",
    )
    diagnostic_only: Optional[bool] = None
    dependency_step_id: Optional[str] = None

    # Step result payload.
    step_summary: Optional[Dict[str, Any]] = None
    evidence_ids: List[str] = Field(default_factory=list)
    analysis_request: Optional[Dict[str, Any]] = None
    visualization_request: Optional[Dict[str, Any]] = None
    semantics_family: Optional[str] = None

    # Runner observability.
    returncode: Optional[int] = None
    timed_out: Optional[bool] = None
    requested_network_policy: Optional[str] = None
    effective_isolation: Optional[str] = None
    isolation_degraded: Optional[bool] = None
    isolation_degradation_reason: Optional[str] = None
    code_repair_attempts: Optional[int] = None
    runner_repair: Optional[str] = None
    deterministic_code_fallback: Optional[str] = None

    # Immutable StepAuthorityCapsule checkpoint/replay coordinates. These do
    # not grant evidence authority; they select the exact candidate/audit/
    # execution snapshot that resume may verify and reuse.
    step_authority_capsule_ref: Optional[Dict[str, Any]] = None
    step_authority_capsule_stage: Optional[str] = None
    step_authority_capsule_reused: Optional[bool] = None
    step_authority_frozen_context_reused: Optional[bool] = None
    step_authority_capsule_cache_miss: Optional[str] = None
    step_authority_execution_cache_miss: Optional[str] = None
    step_authority_audit_cache_miss: Optional[str] = None
    capsule_execution_replayed: Optional[bool] = None
    capsule_concept_audit_replayed: Optional[bool] = None
    capsule_pending_initial_transport_id: Optional[str] = None
    capsule_pending_initial_binding_sha256: Optional[str] = None
    capsule_pending_repair_attempt_id: Optional[int] = None
    capsule_pending_repair_binding_sha256: Optional[str] = None
    capsule_pending_repair_failure_status: Optional[str] = None

    # Concept / contract / quality findings.
    concept_audit_error_count: Optional[int] = None
    concept_repair_attempts: Optional[int] = None
    usage_findings: List[Dict[str, Any]] = Field(default_factory=list)
    stat_findings: List[Dict[str, Any]] = Field(default_factory=list)
    clinical_findings: List[Dict[str, Any]] = Field(default_factory=list)
    guard_findings: List[Dict[str, Any]] = Field(default_factory=list)
    contract_findings: List[Dict[str, Any]] = Field(default_factory=list)
    visual_findings: List[Dict[str, Any]] = Field(default_factory=list)
    visual_qa_demoted: Optional[bool] = None

    # Downstream artefacts.
    critique_report: Optional[Dict[str, Any]] = None
    interpretation_evidence_id: Optional[str] = None


__all__ = [
    "PlannedAnalysisRole",
    "EndpointAbsenceSemantics",
    "EndpointKind",
    "EndpointSpec",
    "EValueConversionSpec",
    "SubgroupAnalysisSpec",
    "VariableRole",
    "AggregationRule",
    "TimeWindow",
    "TemporalConstraint",
    "MissingnessProfile",
    "FixedWindowTrajectoryMetadata",
    "ClusterSelectionCandidate",
    "ClusterSelectionManifest",
    "ClinicalDefinitionReference",
    "ConceptDescriptor",
    "CohortDescriptor",
    "ResearchContext",
    "HypothesisBlueprint",
    "FamilyPrimaryResultRequirement",
    "PlannedFigurePanelSpec",
    "SURVIVAL_ANALYSIS_RECEIPT_PRODUCT",
    "SurvivalAnalysisReceipt",
    "AnalysisStep",
    "AnalysisPlan",
    "EvidenceRef",
    "ConceptRef",
    "ClinicalSemanticsResolution",
    "DataExtractionRequest",
    "DataExtractionResult",
    "StatisticalAnalysisRequest",
    "StatisticalAnalysisResult",
    "VisualizationRequest",
    "VisualizationResult",
    "ManuscriptDraftPacket",
    "CritiqueReport",
    "ReflectionMemoryEntry",
    "AgentRuntimeState",
    "EvidenceRecord",
    "ValidationFinding",
    "CostRecord",
    "AnalysisManifest",
    "PipelineResult",
    "PaperClaimRecord",
    "PaperProfile",
    "PaperReplicationSpec",
    "PaperResultLedger",
    "ReplicationDeviationItem",
    "ReplicationDeviationReport",
    "ProbeSummary",
    "StepRecord",
]
