"""Schema definitions for the research agent layer.

Everything that flows between agents — context, plans, evidence,
manifests — is a typed pydantic model. This buys us:

* JSON serialisation for free (every artefact can be saved next to
  the data and re-loaded by reviewers, downstream tooling or a fresh
  pipeline run);
* validation at the boundary, so a hallucinated field from the LLM
  fails fast instead of contaminating downstream analysis;
* a single place to evolve the contract — both agents and validators
  consume these types.

The models are intentionally small and self-describing. They are not
the final word on what an ICU research package should contain — but
they are a stable v1 the rest of the module can lean on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


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

    ID = "id"                        # patient/stay identifier
    TIME = "time"                    # absolute or relative timestamp
    DEMOGRAPHIC = "demographic"      # age, sex, weight at admission
    VITAL = "vital"                  # vital sign, continuous
    LAB = "lab"                      # laboratory measurement, continuous
    INTERVENTION = "intervention"    # vaso, vent, rrt, fluid bolus...
    ORDINAL_SCORE = "ordinal_score"  # SOFA component, GCS, KDIGO stage
    COMPOSITE_SCORE = "composite_score"  # total SOFA, APACHE
    OUTCOME = "outcome"              # death, los_icu, readmission
    INDEX = "index"                  # row-level time index
    META = "meta"                    # source dataset, cohort tag
    OTHER = "other"


class AggregationRule(str, Enum):
    """Allowed aggregation operations for a variable.

    The agent is told *which* aggregations are valid for a given
    column. Taking a mean of an ordinal SOFA component is a category
    error — the ConceptUsageAuditor flags it.
    """

    ANY = "any"                      # all common ops valid
    MEAN_MEDIAN = "mean_or_median"   # continuous, well-defined mean
    MEDIAN_ONLY = "median_only"      # heavily skewed continuous
    MAX_LAST = "max_or_last"         # ordinal scores → take max or last
    SUM = "sum"                      # counts / cumulative doses
    FIRST_VALUE = "first_value"      # at-admission attributes
    NONE = "none"                    # do not aggregate (event variables)


class TimeWindow(BaseModel):
    """A bounded analysis window relative to an anchor event."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Short, stable identifier, e.g. 'first_24h'.")
    anchor: Literal["icu_admission", "hospital_admission", "event_onset"] = "icu_admission"
    start_hours: float = 0.0
    end_hours: float = 24.0
    rationale: Optional[str] = Field(
        default=None,
        description="Why this window — clinical reason or precedent paper.",
    )


class MissingnessProfile(BaseModel):
    """Per-variable missingness summary computed at context build time."""

    model_config = ConfigDict(extra="forbid")

    fraction_missing: float = Field(ge=0.0, le=1.0)
    n_missing: int = Field(ge=0)
    n_total: int = Field(ge=0)
    missingness_kind: Literal["MCAR_likely", "MAR_likely", "MNAR_likely", "unknown"] = "unknown"
    notes: Optional[str] = None


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
    allowed_aggregations: List[AggregationRule] = Field(default_factory=lambda: [AggregationRule.ANY])
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
    analysis_window: Optional[str] = Field(
        default=None,
        description="Named time window used to derive this variable, e.g. 'first_24h'.",
    )
    source_databases: List[str] = Field(default_factory=list)
    pitfalls: List[str] = Field(
        default_factory=list,
        description="Known traps for this variable, e.g. 'sofa==0 row may indicate missing components, not absence of dysfunction'.",
    )
    missingness_semantics: Optional[str] = Field(
        default=None,
        description="Domain-specific interpretation of missingness for this variable.",
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
    n_patients: int = Field(ge=0)
    n_stays: int = Field(ge=0)
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    id_columns: List[str] = Field(default_factory=list)
    time_columns: List[str] = Field(default_factory=list)
    outcome_columns: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ResearchContext(BaseModel):
    """Top-level context object handed to the agent.

    The ``ResearchContext`` is the central artefact distinguishing this
    layer from a generic data-analysis agent. Every prompt the agents
    see is grounded in the fields below — variable kinds, allowed
    aggregations, time windows, known pitfalls — so that the agent
    cannot confuse an ordinal SOFA component for a continuous lab.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.research_context/1"
    research_question: str
    cohort: CohortDescriptor
    variables: List[ConceptDescriptor]
    time_windows: List[TimeWindow] = Field(default_factory=list)
    target_outcome: Optional[str] = Field(
        default=None,
        description="Name of the primary outcome column.",
    )
    cross_database_validation: List[str] = Field(
        default_factory=list,
        description="Other databases (e.g. eicu, hirid) where this analysis should be replicated.",
    )
    cohort_parquet: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def variable(self, name: str) -> Optional[ConceptDescriptor]:
        for v in self.variables:
            if v.name == name:
                return v
        return None


# ---------------------------------------------------------------------------
# Plans, evidence, manifests
# ---------------------------------------------------------------------------


class AnalysisStep(BaseModel):
    """One step in a planner-emitted analysis plan."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    intent: str = Field(..., description="One-sentence description of what this step does.")
    inputs: List[str] = Field(default_factory=list, description="Variable names or evidence ids consumed.")
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="Logical outputs — table_name, figure_name, statistic_name.",
    )
    method: Optional[str] = None
    icu_rule_refs: List[str] = Field(default_factory=list)


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_question: str
    steps: List[AnalysisStep]
    rationale: Optional[str] = None
    revision: int = 1


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
    run_id: str
    research_question: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    context_path: str
    plan_path: Optional[str] = None
    evidence: List[EvidenceRecord] = Field(default_factory=list)
    findings: List[ValidationFinding] = Field(default_factory=list)
    per_step_records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Execution records for planner steps, sorted in plan order. "
            "The same records are streamed to manifest_partial.json during a run."
        ),
    )
    cost_records: List[CostRecord] = Field(default_factory=list)
    report_path: Optional[str] = None
    manuscript_path: Optional[str] = None
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

    def as_paths(self) -> Dict[str, Path]:
        return {
            "context": Path(self.context_path),
            "plan": Path(self.plan_path),
            "manifest": Path(self.manifest_path),
            "report": Path(self.report_path),
            "manuscript": Path(self.manuscript_path),
        }


__all__ = [
    "VariableRole",
    "AggregationRule",
    "TimeWindow",
    "MissingnessProfile",
    "ConceptDescriptor",
    "CohortDescriptor",
    "ResearchContext",
    "AnalysisStep",
    "AnalysisPlan",
    "EvidenceRecord",
    "ValidationFinding",
    "CostRecord",
    "AnalysisManifest",
    "PipelineResult",
]
