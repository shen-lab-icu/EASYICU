"""Value types + schema for idea mining.

Pydantic models, ``Literal`` status aliases, schema-version constants, and
the id/hash helpers extracted from ``idea_mining.py`` (P1 split, 2026-06-10).
Zero behavior change: every name is re-exported from ``idea_mining`` for
backward compatibility. Leaf module — must not import ``idea_mining`` (a
module-boundary test enforces it).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .literature import CitationRecord

SourceAdapterLevel = Literal[
    "metadata_only",
    "open_access_fulltext",
    "licensed_fulltext_manifest_only",
    "user_supplied_excerpt",
]


OutcomeDeterminabilityStatus = Literal[
    "known_0_1",
    "non_binary_determinable",
    "event_present_na",
    "unknown",
]


FeatureDerivationStatus = Literal[
    "raw_concept_available",
    "derived_feature_available",
    "requires_derived_feature",
    "unsupported",
]


NoveltyLabel = Literal[
    "already_done",
    "crowded_but_differentiable",
    "sparse",
    "apparently_gap",
]


GoNoGoDecision = Literal[
    "recommend",
    "hold",
    "db-cannot-do",
]


IDEA_MINING_SNAPSHOT_SCHEMA_VERSION = "easyicu.idea_source_snapshot/1"


IDEA_NOVELTY_SNAPSHOT_SCHEMA_VERSION = "easyicu.idea_novelty_snapshot/1"


DISCOVERY_REPORT_SCHEMA_VERSION = "easyicu.discovery_candidate_report/1"


class IdeaMiningError(RuntimeError):
    """Base class for idea-mining failures."""


class IdeaExtractionError(IdeaMiningError):
    """Raised when source material cannot be converted to candidates."""


class NonExecutableCandidateError(IdeaMiningError):
    """Raised when a blocked candidate is asked to produce executable context."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _nonempty_text(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _stable_idea_id(payload: Mapping[str, Any]) -> str:
    digest = _sha256_text(_canonical_json(payload))[:16]
    return f"litidea_{digest}"


def _stable_executable_id(payload: Mapping[str, Any]) -> str:
    digest = _sha256_text(_canonical_json(payload))[:16]
    return f"execidea_{digest}"


class SourceMaterial(BaseModel):
    """Source material made available to S4.

    ``source_text`` may be a user-supplied excerpt for extraction or text used
    only for hashing. Freeze manifests store only hashes and counts, never the
    raw body text.
    """

    model_config = ConfigDict(extra="forbid")

    citation: CitationRecord
    source_adapter_level: SourceAdapterLevel = "metadata_only"
    locator: Optional[str] = None
    source_text: Optional[str] = None

    @model_validator(mode="after")
    def _validate_supported_text_level(self) -> "SourceMaterial":
        if self.source_adapter_level == "user_supplied_excerpt":
            if not str(self.source_text or "").strip():
                raise ValueError("user_supplied_excerpt requires source_text")
        return self


class SourceSnapshotItem(BaseModel):
    """One freeze-manifest item.

    The source text itself is intentionally absent to avoid storing licensed or
    copyrighted full text in reproducibility manifests.
    """

    model_config = ConfigDict(extra="forbid")

    citation: CitationRecord
    source_adapter_level: SourceAdapterLevel
    locator: Optional[str] = None
    source_text_sha256: Optional[str] = None
    source_text_char_count: int = Field(default=0, ge=0)
    source_text_stored: bool = False
    rights_note: str


class SourceSnapshotManifest(BaseModel):
    """Frozen source-set manifest for S2 registry provenance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = IDEA_MINING_SNAPSHOT_SCHEMA_VERSION
    source_snapshot_id: str
    created_at: str = Field(default_factory=_utc_now_iso)
    items: List[SourceSnapshotItem]


class LiteratureIdeaCandidate(BaseModel):
    """A research idea extracted from literature source material."""

    model_config = ConfigDict(extra="forbid")

    literature_idea_id: Optional[str] = None
    source_snapshot_id: str
    citation_key: str
    source_adapter_level: SourceAdapterLevel
    population: str
    exposure_or_predictor: str
    outcome: str
    rationale: str
    source_quote: str = Field(max_length=800)
    analysis_family: str = "association"
    time_window_hint: Optional[str] = None
    aggregation_hint: Optional[str] = None
    # The single core measurable concept, stripped of timing windows, dose
    # thresholds, and subgroup qualifiers. Optional + back-compatible: when the
    # model omits it, concept resolution falls back to the full phrase. These
    # let resolution bind on the main construct ("norepinephrine") instead of an
    # incidental qualifier concept ("lactate") that shares the phrase.
    exposure_core_concept: Optional[str] = None
    outcome_core_concept: Optional[str] = None

    @field_validator(
        "source_snapshot_id",
        "citation_key",
        "population",
        "exposure_or_predictor",
        "outcome",
        "rationale",
        "source_quote",
        "analysis_family",
    )
    @classmethod
    def _nonempty_fields(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "field")
        return _nonempty_text(value, field_name)

    @model_validator(mode="after")
    def _fill_stable_id(self) -> "LiteratureIdeaCandidate":
        if not self.literature_idea_id:
            self.literature_idea_id = _stable_idea_id(
                {
                    "citation_key": self.citation_key,
                    "population": self.population,
                    "predictor": self.exposure_or_predictor,
                    "outcome": self.outcome,
                    "quote": self.source_quote,
                }
            )
        return self


class OutcomeDeterminability(BaseModel):
    """Whether an outcome column can be used for outcome-blind feasibility.

    S4/S5 must not pass event-positive present/NA columns directly to S1,
    because non-missingness would encode the event rate. Such outcomes need a
    normalized known 0/1 column or must remain non-executable for joint
    feasibility ranking.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: str
    status: OutcomeDeterminabilityStatus = "unknown"
    normalized_outcome_concept: Optional[str] = None
    note: Optional[str] = None

    @field_validator("outcome")
    @classmethod
    def _nonempty_outcome(cls, value: str) -> str:
        return _nonempty_text(value, "outcome")


class ExecutableHypothesisCandidate(BaseModel):
    """A literature idea after concept mapping and safety gates."""

    model_config = ConfigDict(extra="forbid")

    executable_candidate_id: str
    literature_idea_id: str
    source_snapshot_id: str
    citation_key: str
    population: str
    predictor_label: str
    outcome_label: str
    resolved_predictor_concept: Optional[str] = None
    resolved_outcome_concept: Optional[str] = None
    feasibility_pair_key: Optional[Tuple[str, str]] = None
    outcome_determinability_status: OutcomeDeterminabilityStatus = "unknown"
    normalized_outcome_concept: Optional[str] = None
    analysis_family: str = "association"
    research_question: str
    source_quote: str
    feature_derivation_status: FeatureDerivationStatus = "raw_concept_available"
    feature_derivation_requirements: List[str] = Field(default_factory=list)
    feature_derivation_note: Optional[str] = None
    non_executable_reasons: List[str] = Field(default_factory=list)

    @property
    def executable(self) -> bool:
        return not self.non_executable_reasons

    def assert_research_context_allowed(self) -> bool:
        """Fail closed before any caller builds a frozen ResearchContext."""

        if not self.executable:
            raise NonExecutableCandidateError(
                "; ".join(self.non_executable_reasons)
                or "candidate is not executable"
            )
        return True


class IdeaMiningYieldReport(BaseModel):
    """Mapping and gate-yield diagnostics for an S5 dry run."""

    model_config = ConfigDict(extra="forbid")

    n_literature_ideas: int = Field(ge=0)
    n_resolved_predictor: int = Field(ge=0)
    n_resolved_outcome: int = Field(ge=0)
    n_executable: int = Field(ge=0)
    n_non_executable: int = Field(ge=0)
    unresolved_predictor_labels: List[str] = Field(default_factory=list)
    unresolved_outcome_labels: List[str] = Field(default_factory=list)
    top_non_executable_reasons: List[str] = Field(default_factory=list)


class IdeaMiningFeasibilityRecord(BaseModel):
    """Pair-level feasibility signal actually supplied to S3 ranking."""

    model_config = ConfigDict(extra="forbid")

    predictor: str
    outcome: str
    pair_key: Tuple[str, str]
    joint_fraction_complete: float = Field(ge=0.0, le=1.0)
    n_joint_complete: Optional[int] = Field(default=None, ge=0)
    denominator_n: Optional[int] = Field(default=None, ge=0)
    source: str = "precomputed"
    note: Optional[str] = None


class PriorArtSearchHit(BaseModel):
    """One prior-art search hit captured for novelty triage."""

    model_config = ConfigDict(extra="forbid")

    pmid: str
    title: str
    venue: Optional[str] = None
    year: Optional[str] = None
    relevance: Optional[str] = None
    direct_same_topic: bool = False
    direct_same_topic_rationale: Optional[str] = None
    same_topic_screened: bool = False

    @field_validator("pmid", "title")
    @classmethod
    def _nonempty_hit_fields(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "field")
        return _nonempty_text(value, field_name)


class PriorArtQueryRecord(BaseModel):
    """A frozen prior-art query and its top hits."""

    model_config = ConfigDict(extra="forbid")

    query_type: Literal["broad", "exact"]
    query: str
    hit_count: int = Field(ge=0)
    pmids: List[str] = Field(default_factory=list)
    top_hits: List[PriorArtSearchHit] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def _nonempty_query(cls, value: str) -> str:
        return _nonempty_text(value, "query")


class PriorArtAssessment(BaseModel):
    """Layered PubMed prior-art triage for one literature-derived idea.

    This is not a novelty claim. It freezes the query strategy and direct
    same-topic judgement used to produce a triage label as of a search date.
    """

    model_config = ConfigDict(extra="forbid")

    novelty_snapshot_id: str
    schema_version: str = IDEA_NOVELTY_SNAPSHOT_SCHEMA_VERSION
    literature_idea_id: str
    executable_candidate_id: Optional[str] = None
    source_snapshot_id: str
    searched_at: str
    predictor_literature_phrase: str
    outcome_literature_phrase: str
    differentiators: List[str] = Field(default_factory=list)
    has_specific_differentiator: bool = False
    feasibility_pair_key: Optional[Tuple[str, str]] = None
    query_records: List[PriorArtQueryRecord]
    direct_same_topic_pmids: List[str] = Field(default_factory=list)
    direct_same_topic_rationales: Dict[str, str] = Field(default_factory=dict)
    novelty_label: NoveltyLabel
    literature_saturation_signal: float = Field(ge=0.0, le=1.0)
    novelty_statement: str
    same_topic_screen_status: str = "automated-substring-only, NOT screened"
    scope_note: str = (
        "PubMed title/abstract triage only; does not cover Embase, preprints, "
        "grey literature, non-English indexing gaps, or licensed full text. "
        "Same-topic screening is asymmetric: an already_done label is stronger "
        "than an apparently_gap label, which remains only a human prior-art "
        "review trigger unless validated recall-at-depth metrics accompany it."
    )
    clinical_plausibility_requires_human: bool = True


class DiscoveryCandidateRecord(BaseModel):
    """Human-readable S6 discovery row backed by frozen structured evidence."""

    model_config = ConfigDict(extra="forbid")

    literature_idea_id: str
    executable_candidate_id: Optional[str] = None
    source_snapshot_id: str
    citation_key: str
    literature_source: str
    gap_evidence_quote: str
    candidate_topic: str
    prior_art: PriorArtAssessment
    database_feasibility: Dict[str, Any] = Field(default_factory=dict)
    go_no_go: GoNoGoDecision
    go_no_go_reason: str
    risks: List[str] = Field(default_factory=list)
    clinical_plausibility_requires_human: bool = True


class DiscoveryTriageResult(BaseModel):
    """S6 output bundle: novelty snapshots plus a rendered report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = DISCOVERY_REPORT_SCHEMA_VERSION
    discovery_records: List[DiscoveryCandidateRecord]
    novelty_assessments: List[PriorArtAssessment]
    markdown_report: str
    report_path: Optional[str] = None


class IdeaMiningCandidateTriageRecord(BaseModel):
    """One row in the S5 candidate triage report."""

    model_config = ConfigDict(extra="forbid")

    literature_idea_id: str
    executable_candidate_id: str
    registry_candidate_id: str
    hypothesis_family_id: str
    source_snapshot_id: str
    citation_key: str
    predictor_label: str
    outcome_label: str
    resolved_predictor_concept: Optional[str] = None
    resolved_outcome_concept: Optional[str] = None
    feasibility_pair_key: Optional[Tuple[str, str]] = None
    feature_derivation_status: FeatureDerivationStatus = "raw_concept_available"
    feature_derivation_requirements: List[str] = Field(default_factory=list)
    feature_derivation_note: Optional[str] = None
    executable: bool
    non_executable_reasons: List[str] = Field(default_factory=list)
    ranking_candidate_id: Optional[str] = None
    priority_score: Optional[float] = None
    coverage_source: Optional[str] = None
    feasibility_note: Optional[str] = None
    n_joint_complete: Optional[int] = Field(default=None, ge=0)
    denominator_n: Optional[int] = Field(default=None, ge=0)
    registry_selection_status: str = "proposed"
    multiple_testing_family_size: int = Field(ge=0)
    multiple_testing_executable_family_size: int = Field(ge=0)
    multiple_testing_note: str
    causal_audit_risk: str
    causal_audit_scope: str


class IdeaMiningDryRunResult(BaseModel):
    """S5 dry-run output bundle.

    The bundle is deliberately a stop-before-execution artefact. It contains
    source freeze metadata, extracted ideas, mapped candidates, S1/S3 triage
    signals, and S2 registry state, but no pipeline result.
    """

    model_config = ConfigDict(extra="forbid")

    source_snapshot_manifest: SourceSnapshotManifest
    hypothesis_family_id: str
    literature_ideas: List[LiteratureIdeaCandidate]
    executable_candidates: List[ExecutableHypothesisCandidate]
    yield_report: IdeaMiningYieldReport
    prior_art_assessments: List[PriorArtAssessment] = Field(default_factory=list)
    feasibility_signals: List[IdeaMiningFeasibilityRecord] = Field(default_factory=list)
    ranked_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_records: List[IdeaMiningCandidateTriageRecord] = Field(default_factory=list)
    discovery_records: List[DiscoveryCandidateRecord] = Field(default_factory=list)
    registry_path: str
    manifest_path: str
    triage_report_path: str
    novelty_snapshot_path: Optional[str] = None
    discovery_report_path: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
