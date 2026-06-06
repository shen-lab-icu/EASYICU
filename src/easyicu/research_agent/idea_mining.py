"""Literature-derived idea extraction for EasyICU triage workflows.

S4 is the first idea-mining stage that looks upstream at review/editorial
material. It deliberately stops at an auditable candidate list and concept
mapping: no research context is generated for blocked candidates, no pipeline
is invoked, and licensed/full-text source material is never stored in freeze
manifests.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .concept_availability import (
    normalize_concept_name,
    real_data_concept_feasibility,
)
from .concept_catalog import SYNONYM_GROUPS
from .hypothesis_generator import (
    HypothesisFeasibilitySignal,
    HypothesisGeneratorResult,
    generate_hypotheses,
)
from .idea_registry import (
    CandidateAlreadyRegisteredError,
    CandidateNotRegisteredError,
    CandidateRegistryEntry,
    IdeaCandidateRegistry,
)
from .literature import CitationRecord
from .llm import LLMClient, LLMMessage
from .schema import CohortDescriptor, ConceptDescriptor, MissingnessProfile, ResearchContext, VariableRole


SourceAdapterLevel = Literal[
    "metadata_only",
    "open_access_fulltext",
    "licensed_fulltext_manifest_only",
    "user_supplied_excerpt",
]

OutcomeDeterminabilityStatus = Literal[
    "known_0_1",
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
IDEA_EXTRACTION_SYSTEM_PROMPT = (
    "You extract candidate ICU research directions from review or editorial "
    "source material. Stay case-neutral: do not assume a specific disease, "
    "score, exposure, database, or outcome unless it appears in the supplied "
    "source material. Return only JSON."
)

_EXTRACTION_SUPPORTED_LEVELS = {"metadata_only", "user_supplied_excerpt"}

_GENERIC_DIFFERENTIATOR_PATTERNS = (
    "icu",
    "icu stay",
    "early icu stay",
    "adult",
    "adult patient",
    "adult patients",
    "critically ill",
    "critical illness",
    "patients",
    "patient",
    "outcome",
    "mortality",
    "association",
    "any exposure",
    "exposure",
    "trajectory summary",
)

_DERIVED_FEATURE_REQUIREMENTS: Dict[str, List[str]] = {
    "trajectory": [
        "requires repeated measurements >=2",
        "requires time ordering",
        "requires trajectory/slope or change computation",
    ],
    "clearance": [
        "requires repeated measurements >=2",
        "requires time ordering",
        "requires delta/clearance computation",
    ],
    "load": [
        "requires dose/time observations",
        "requires load or cumulative exposure computation",
    ],
    "balance": [
        "requires input/output component concepts",
        "requires balance computation",
    ],
    "trend": [
        "requires repeated measurements >=2",
        "requires time ordering",
        "requires trend computation",
    ],
    "slope": [
        "requires repeated measurements >=2",
        "requires time ordering",
        "requires slope computation",
    ],
    "delta": [
        "requires paired measurements",
        "requires delta computation",
    ],
}

_GENERIC_CONCEPT_WORDS = {
    "association",
    "available",
    "average",
    "binary",
    "candidate",
    "change",
    "changes",
    "clinical",
    "computed",
    "duration",
    "durations",
    "early",
    "event",
    "events",
    "exposure",
    "feature",
    "features",
    "first",
    "hour",
    "hours",
    "icu",
    "indicator",
    "intensity",
    "level",
    "levels",
    "measurement",
    "measurements",
    "measure",
    "observed",
    "patient",
    "patients",
    "raw",
    "score",
    "status",
    "study",
    "studies",
    "supporting",
    "total",
    "value",
    "values",
    "window",
    "within",
}

_PRIOR_ART_QUERY_STOPWORDS = _GENERIC_CONCEPT_WORDS | {
    "abstract",
    "admission",
    "adult",
    "adults",
    "after",
    "analysis",
    "and",
    "before",
    "care",
    "centered",
    "critical",
    "critically",
    "database",
    "during",
    "endpoint",
    "endpoints",
    "hospital",
    "ill",
    "illness",
    "intensive",
    "of",
    "or",
    "therapy",
    "the",
    "unit",
    "with",
}

_PRIOR_ART_SINGLETON_STOPWORDS = _PRIOR_ART_QUERY_STOPWORDS | {
    # Single-token facets from these words make PubMed relevance explode
    # without preserving same-topic specificity. Keep the full phrase and
    # multiword facets instead.
    "balance",
    "clearance",
    "count",
    "dose",
    "driving",
    "energy",
    "exposure",
    "failure",
    "free",
    "index",
    "injury",
    "mechanical",
    "pattern",
    "power",
    "pressure",
    "profile",
    "ratio",
    "red",
    "setting",
    "signature",
    "strategy",
    "timing",
    "trajectory",
}

_PRIOR_ART_QUERY_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "mortality": ("death",),
    "death": ("mortality",),
}


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


def build_prior_art_queries(
    idea: LiteratureIdeaCandidate,
) -> Dict[str, str]:
    """Build broad/exact PubMed-style queries from literature phrasing.

    N-6: exact novelty queries intentionally use the phrase as written in the
    source material plus differentiators. Broad queries use the LLM-separated
    core construct when available, with the literature phrase kept as an OR-ed
    recall fallback. This avoids false gaps caused by over-specific wording
    while still avoiding EasyICU canonical keys such as ``lact``.
    """

    predictor_phrase = _clean_literature_phrase(idea.exposure_or_predictor)
    outcome_phrase = _clean_literature_phrase(idea.outcome)
    population_phrase = _clean_literature_phrase(idea.population)
    differentiators = _candidate_differentiators(idea)

    broad_parts = [
        _pubmed_core_recall_clause(
            idea.exposure_core_concept,
            fallback_phrase=predictor_phrase,
        ),
        _pubmed_core_recall_clause(
            idea.outcome_core_concept,
            fallback_phrase=outcome_phrase,
        ),
    ]
    population_recall = _pubmed_population_recall_clause(population_phrase)
    if population_recall:
        broad_parts.append(population_recall)

    exact_parts = [
        _pubmed_phrase_clause(predictor_phrase),
        _pubmed_phrase_clause(outcome_phrase),
    ]
    for item in differentiators:
        exact_parts.append(_pubmed_phrase_clause(item))

    return {
        "broad": " AND ".join(part for part in broad_parts if part),
        "exact": " AND ".join(part for part in exact_parts if part),
    }


def assess_prior_art_for_idea(
    idea: LiteratureIdeaCandidate,
    *,
    search_client: Any,
    executable_candidate: Optional[ExecutableHypothesisCandidate] = None,
    searched_at: Optional[str] = None,
    top_n: int = 20,
) -> PriorArtAssessment:
    """Run layered prior-art triage for one literature-derived idea."""

    timestamp = searched_at or _utc_now_iso()
    queries = build_prior_art_queries(idea)
    broad = _run_prior_art_query(
        search_client,
        query_type="broad",
        query=queries["broad"],
        max_results=top_n,
        idea=idea,
    )
    exact = _run_prior_art_query(
        search_client,
        query_type="exact",
        query=queries["exact"],
        max_results=top_n,
        idea=idea,
    )
    query_records = [broad, exact]
    direct_hits = [
        hit
        for record in query_records
        for hit in record.top_hits
        if hit.direct_same_topic
    ]
    screened_direct_hits = [hit for hit in direct_hits if hit.same_topic_screened]
    direct_pmids = _ordered_unique([hit.pmid for hit in direct_hits])
    direct_rationales = {
        hit.pmid: hit.direct_same_topic_rationale or "direct same-topic hit"
        for hit in direct_hits
    }
    differentiators = _candidate_differentiators(idea)
    has_specific_differentiator = bool(differentiators)
    same_topic_screen_status = _same_topic_screen_status(query_records)
    novelty_label = _label_prior_art(
        broad_count=broad.hit_count,
        exact_count=exact.hit_count,
        direct_same_topic_count=len(screened_direct_hits),
        has_specific_differentiator=has_specific_differentiator,
    )
    if direct_hits and not screened_direct_hits:
        novelty_label = "crowded_but_differentiable"
    saturation = _saturation_for_novelty_label(novelty_label)
    feasibility_pair_key = (
        executable_candidate.feasibility_pair_key
        if executable_candidate is not None
        else None
    )
    executable_id = (
        executable_candidate.executable_candidate_id
        if executable_candidate is not None
        else None
    )
    statement = (
        f"As of {timestamp}, under the frozen PubMed title/abstract query "
        f"strategy, this candidate is triaged as {novelty_label}; this is not "
        "a novelty claim and requires human prior-art review. "
        f"Same-topic screen status: {same_topic_screen_status}."
    )
    payload = {
        "schema_version": IDEA_NOVELTY_SNAPSHOT_SCHEMA_VERSION,
        "literature_idea_id": idea.literature_idea_id,
        "executable_candidate_id": executable_id,
        "source_snapshot_id": idea.source_snapshot_id,
        "searched_at": timestamp,
        "predictor_literature_phrase": idea.exposure_or_predictor,
        "outcome_literature_phrase": idea.outcome,
        "differentiators": differentiators,
        "has_specific_differentiator": has_specific_differentiator,
        "feasibility_pair_key": feasibility_pair_key,
        "query_records": [record.model_dump(mode="json") for record in query_records],
        "direct_same_topic_pmids": direct_pmids,
        "direct_same_topic_rationales": direct_rationales,
        "novelty_label": novelty_label,
        "literature_saturation_signal": saturation,
        "same_topic_screen_status": same_topic_screen_status,
    }
    snapshot_id = f"novelty-snapshot/sha256:{_sha256_text(_canonical_json(payload))[:16]}"
    return PriorArtAssessment(
        novelty_snapshot_id=snapshot_id,
        literature_idea_id=str(idea.literature_idea_id),
        executable_candidate_id=executable_id,
        source_snapshot_id=idea.source_snapshot_id,
        searched_at=timestamp,
        predictor_literature_phrase=idea.exposure_or_predictor,
        outcome_literature_phrase=idea.outcome,
        differentiators=differentiators,
        has_specific_differentiator=has_specific_differentiator,
        feasibility_pair_key=feasibility_pair_key,
        query_records=query_records,
        direct_same_topic_pmids=direct_pmids,
        direct_same_topic_rationales=direct_rationales,
        novelty_label=novelty_label,
        literature_saturation_signal=saturation,
        novelty_statement=statement,
        same_topic_screen_status=same_topic_screen_status,
        clinical_plausibility_requires_human=True,
    )


def assess_prior_art_for_candidates(
    *,
    literature_ideas: Sequence[LiteratureIdeaCandidate],
    executable_candidates: Sequence[ExecutableHypothesisCandidate],
    search_client: Any,
    searched_at: Optional[str] = None,
    top_n: int = 20,
) -> List[PriorArtAssessment]:
    """Assess all literature ideas, preserving literature phrase provenance."""

    candidates_by_idea = {
        candidate.literature_idea_id: candidate
        for candidate in executable_candidates
    }
    return [
        assess_prior_art_for_idea(
            idea,
            search_client=search_client,
            executable_candidate=candidates_by_idea.get(str(idea.literature_idea_id)),
            searched_at=searched_at,
            top_n=top_n,
        )
        for idea in literature_ideas
    ]


def render_discovery_report(
    records: Sequence[DiscoveryCandidateRecord],
    *,
    title: str = "EasyICU Discovery Candidate Report",
    counts: Optional[Mapping[str, int]] = None,
) -> str:
    """Render S6 discovery records into an auditable markdown report."""

    report_counts = dict(counts or _discovery_report_counts(records))
    lines = [
        f"# {title}",
        "",
        (
            "This report is an idea-triage artifact, not a novelty claim. "
            "Prior-art labels are bounded by the frozen query strategy and "
            "require human review."
        ),
        "",
        (
            "Counts: "
            f"literature_rows={report_counts.get('literature_rows', len(records))}; "
            f"unique_executable_hypotheses={report_counts.get('unique_executable_hypotheses', 0)}; "
            f"multiple_testing_denominator={report_counts.get('multiple_testing_denominator', 0)}."
        ),
        "",
        "| # | Source | Gap evidence | Candidate topic | Prior-art triage | Database feasibility | Go/No-go | Risks |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for idx, record in enumerate(records, start=1):
        assessment = record.prior_art
        broad = _query_by_type(assessment, "broad")
        exact = _query_by_type(assessment, "exact")
        prior_art = (
            f"`{assessment.novelty_label}`; broad n={broad.hit_count if broad else 'n/a'}, "
            f"exact n={exact.hit_count if exact else 'n/a'}; "
            f"direct PMIDs={', '.join(assessment.direct_same_topic_pmids) or 'none'}; "
            f"searched {assessment.searched_at}"
        )
        feasibility = _format_feasibility(record.database_feasibility)
        risks = "<br>".join(_escape_md_cell(risk) for risk in record.risks) or "n/a"
        lines.append(
            "| {idx} | {source} | {gap} | {topic} | {prior} | {feas} | {decision} | {risks} |".format(
                idx=idx,
                source=_escape_md_cell(record.literature_source),
                gap=_escape_md_cell(record.gap_evidence_quote),
                topic=_escape_md_cell(record.candidate_topic),
                prior=_escape_md_cell(prior_art),
                feas=_escape_md_cell(feasibility),
                decision=_escape_md_cell(
                    f"{record.go_no_go}: {record.go_no_go_reason}"
                ),
                risks=risks,
            )
        )
    return "\n".join(lines) + "\n"


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


def freeze_source_snapshot(
    materials: Sequence[SourceMaterial | Mapping[str, Any]],
    *,
    created_at: Optional[str] = None,
) -> SourceSnapshotManifest:
    """Freeze a source set without storing source bodies.

    The manifest is safe for metadata-only, user excerpt, and licensed
    manifest-only paths. It stores citation metadata plus source text hash and
    length when text was supplied, but never the text itself.
    """

    items: List[SourceSnapshotItem] = []
    for raw in materials:
        material = (
            raw
            if isinstance(raw, SourceMaterial)
            else SourceMaterial.model_validate(raw)
        )
        text = str(material.source_text or "")
        sha = _sha256_text(text) if text else None
        rights = (
            "metadata only; no source body stored"
            if material.source_adapter_level == "metadata_only"
            else "source body omitted; manifest stores locator/hash only"
        )
        items.append(
            SourceSnapshotItem(
                citation=material.citation,
                source_adapter_level=material.source_adapter_level,
                locator=material.locator,
                source_text_sha256=sha,
                source_text_char_count=len(text),
                source_text_stored=False,
                rights_note=rights,
            )
        )

    digest_payload = [
        item.model_dump(mode="json", exclude={"rights_note"}) for item in items
    ]
    snapshot_id = f"source-snapshot/sha256:{_sha256_text(_canonical_json(digest_payload))[:16]}"
    return SourceSnapshotManifest(
        source_snapshot_id=snapshot_id,
        created_at=created_at or _utc_now_iso(),
        items=items,
    )


def build_idea_extraction_messages(
    materials: Sequence[SourceMaterial | Mapping[str, Any]],
    *,
    source_snapshot_id: str,
) -> List[LLMMessage]:
    """Build the case-neutral extraction prompt for S4."""

    parsed = [
        raw if isinstance(raw, SourceMaterial) else SourceMaterial.model_validate(raw)
        for raw in materials
    ]
    unsupported = [
        material.source_adapter_level
        for material in parsed
        if material.source_adapter_level not in _EXTRACTION_SUPPORTED_LEVELS
    ]
    if unsupported:
        raise IdeaExtractionError(
            "S4 extraction currently supports metadata_only and "
            f"user_supplied_excerpt only; unsupported={sorted(set(unsupported))}"
        )
    source_blocks: List[Dict[str, Any]] = []
    for material in parsed:
        citation = material.citation
        available_text = material.source_text
        if material.source_adapter_level == "metadata_only":
            available_text = " ".join(
                part
                for part in [
                    citation.title,
                    citation.venue or "",
                    citation.relevance or "",
                ]
                if str(part or "").strip()
            )
        source_blocks.append(
            {
                "citation_key": citation.key,
                "title": citation.title,
                "venue": citation.venue,
                "year": citation.year,
                "source_adapter_level": material.source_adapter_level,
                "available_source_text": available_text or "",
            }
        )

    contract = {
        "return": "JSON array",
        "fields": [
            "citation_key",
            "population",
            "exposure_or_predictor",
            "exposure_core_concept",
            "outcome",
            "outcome_core_concept",
            "rationale",
            "source_quote",
            "analysis_family",
            "time_window_hint",
            "aggregation_hint",
        ],
        "rules": [
            "source_quote must be copied from available_source_text",
            (
                "exposure_core_concept: the SINGLE core measurable construct "
                "being studied as the exposure, with timing windows, dose or "
                "value thresholds, and subgroup qualifiers REMOVED. e.g. for a "
                "phrase like 'early <agent> within <N> h in patients with <lab> "
                "<= <threshold>', exposure_core_concept names only '<agent>'; put "
                "the timing in time_window_hint and the subgroup restriction in "
                "population. Name exactly ONE construct, not a compound phrase."
            ),
            (
                "outcome_core_concept: the SINGLE core outcome construct with "
                "qualifiers removed (e.g. a setting- or mechanism-qualified "
                "endpoint reduces to its canonical construct name). Name exactly "
                "ONE construct."
            ),
            (
                "source_quote should identify an unresolved question, future "
                "direction, limitation, uncertainty, or evidence gap when "
                "such language is present"
            ),
            (
                "exposure_or_predictor and outcome must be specific named "
                "constructs grounded in the quote; do not fill generic "
                "placeholders such as marker, physiologic marker, trajectory, "
                "risk factor, endpoint, or patient-centered endpoint when the "
                "source does not name a concrete construct"
            ),
            (
                "omit a candidate rather than generalizing from a vague gap "
                "sentence into a broad association or generic trajectory"
            ),
            "do not infer results, effect sizes, p-values, or event rates",
            "do not create executable analysis instructions",
        ],
    }
    user_payload = {
        "source_snapshot_id": source_snapshot_id,
        "instruction": (
            "Extract open research directions from unresolved questions, "
            "future directions, limitations, or explicit uncertainty in the "
            "supplied review/editorial/guideline material. Do not extract "
            "well-established associations merely because they are mentioned. "
            "Use only supplied source text and metadata."
        ),
        "contract": contract,
        "sources": source_blocks,
    }
    return [
        LLMMessage(role="system", content=IDEA_EXTRACTION_SYSTEM_PROMPT),
        LLMMessage(role="user", content=_canonical_json(user_payload)),
    ]


def extract_literature_ideas(
    *,
    materials: Sequence[SourceMaterial | Mapping[str, Any]],
    source_snapshot_id: str,
    llm: LLMClient,
) -> List[LiteratureIdeaCandidate]:
    """Extract structured literature ideas with a case-neutral JSON prompt."""

    messages = build_idea_extraction_messages(
        materials,
        source_snapshot_id=source_snapshot_id,
    )
    parsed_materials = [
        raw if isinstance(raw, SourceMaterial) else SourceMaterial.model_validate(raw)
        for raw in materials
    ]
    source_text_by_key = _source_text_lookup(parsed_materials)
    adapter_level_by_key = {
        material.citation.key: material.source_adapter_level
        for material in parsed_materials
    }
    raw = llm.complete(messages, max_tokens=2048, temperature=0.0)
    payload = _parse_json_payload(raw)
    if not isinstance(payload, list):
        raise IdeaExtractionError("idea extraction response must be a JSON array")
    candidates: List[LiteratureIdeaCandidate] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise IdeaExtractionError("each idea extraction item must be an object")
        data = dict(item)
        data["source_snapshot_id"] = source_snapshot_id
        citation_key = str(data.get("citation_key") or "").strip()
        data.setdefault("source_adapter_level", adapter_level_by_key.get(citation_key))
        quote = str(data.get("source_quote") or "").strip()
        if not _quote_is_traceable(quote, source_text_by_key.get(citation_key, "")):
            raise IdeaExtractionError(
                f"source_quote for citation_key={citation_key!r} is not traceable"
            )
        candidates.append(LiteratureIdeaCandidate.model_validate(data))
    return candidates


def map_literature_idea_to_executable_candidate(
    candidate: LiteratureIdeaCandidate,
    *,
    available_concepts: Sequence[ConceptDescriptor | str],
    concept_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    outcome_determinability: Optional[
        Mapping[str, OutcomeDeterminability | Mapping[str, Any] | str]
    ] = None,
) -> ExecutableHypothesisCandidate:
    """Resolve one literature idea against EasyICU concepts and S4 guards.

    ``concept_aliases`` is caller-supplied dictionary metadata, not a shared
    clinical word list. It lets S6 use EasyICU dictionary descriptions or
    benchmark-local aliases while keeping novelty queries on literature
    phrasing and feasibility keys on canonical concept names.
    """

    lookup = _build_concept_lookup(available_concepts, concept_aliases=concept_aliases)
    reasons: List[str] = []

    predictor_term = (
        (candidate.exposure_core_concept or "").strip()
        or candidate.exposure_or_predictor
    )
    predictor_key = _resolve_concept(predictor_term, lookup)
    if predictor_key is None:
        reasons.append(
            f"predictor concept is not available: {candidate.exposure_or_predictor}"
        )
    feature_status, feature_requirements, feature_note = _feature_derivation_status(
        candidate.exposure_or_predictor,
        resolved_key=predictor_key,
    )
    if feature_status == "requires_derived_feature":
        reasons.append(
            "predictor requires derived feature engineering before execution: "
            f"{candidate.exposure_or_predictor}"
        )
    elif feature_status == "unsupported" and predictor_key is not None:
        reasons.append(
            "predictor feature derivation is unsupported: "
            f"{candidate.exposure_or_predictor}"
        )

    outcome_term = (
        (candidate.outcome_core_concept or "").strip() or candidate.outcome
    )
    outcome_key = _resolve_concept(outcome_term, lookup)
    if outcome_key is None:
        reasons.append(f"outcome concept is not available: {candidate.outcome}")

    determinability = _lookup_outcome_determinability(
        candidate.outcome,
        outcome_key,
        outcome_determinability or {},
    )
    normalized_outcome = None
    if determinability.status == "event_present_na":
        if determinability.normalized_outcome_concept:
            normalized = _resolve_concept(
                determinability.normalized_outcome_concept,
                lookup,
            )
            if normalized is None:
                reasons.append(
                    "event-present/NA outcome has normalized_outcome_concept "
                    "that is not available"
                )
            else:
                normalized_outcome = normalized
                outcome_key = normalized
        else:
            reasons.append(
                "outcome uses event-positive present/NA coding; normalize to "
                "explicit known 0/1 before feasibility probing"
            )
    elif determinability.status == "unknown":
        reasons.append("outcome determinability is unknown for feasibility probing")

    pair_key = None
    if predictor_key is not None and outcome_key is not None:
        pair_key = (predictor_key, outcome_key)

    executable_candidate_id = _stable_executable_id(
        {
            "literature_idea_id": candidate.literature_idea_id,
            "predictor": predictor_key or candidate.exposure_or_predictor,
            "outcome": outcome_key or candidate.outcome,
            "snapshot": candidate.source_snapshot_id,
        }
    )
    return ExecutableHypothesisCandidate(
        executable_candidate_id=executable_candidate_id,
        literature_idea_id=str(candidate.literature_idea_id),
        source_snapshot_id=candidate.source_snapshot_id,
        citation_key=candidate.citation_key,
        population=candidate.population,
        predictor_label=candidate.exposure_or_predictor,
        outcome_label=candidate.outcome,
        resolved_predictor_concept=predictor_key,
        resolved_outcome_concept=outcome_key,
        feasibility_pair_key=pair_key,
        outcome_determinability_status=determinability.status,
        normalized_outcome_concept=normalized_outcome,
        analysis_family=candidate.analysis_family,
        research_question=(
            f"Is {candidate.exposure_or_predictor} associated with "
            f"{candidate.outcome} in {candidate.population}?"
        ),
        source_quote=candidate.source_quote,
        feature_derivation_status=feature_status,
        feature_derivation_requirements=feature_requirements,
        feature_derivation_note=feature_note,
        non_executable_reasons=reasons,
    )


FeasibilityProbe = Callable[..., Mapping[str, Any]]


def _catalog_restrict_keys(
    available_concepts: Sequence[ConceptDescriptor | str],
) -> List[str]:
    keys: List[str] = []
    for item in available_concepts:
        if isinstance(item, ConceptDescriptor):
            keys.extend([item.source_concept or "", item.name])
        else:
            keys.append(str(item))
    return _ordered_unique(keys)


def _default_concept_catalog_for_idea_run(
    available_concepts: Sequence[ConceptDescriptor | str],
):
    from .concept_catalog import load_concept_catalog

    return load_concept_catalog(restrict_to=_catalog_restrict_keys(available_concepts))


def _merge_concept_aliases(
    derived: Mapping[str, Sequence[str]],
    supplied: Optional[Mapping[str, Sequence[str]]],
) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {
        str(key): _ordered_unique([str(value) for value in values])
        for key, values in derived.items()
    }
    if supplied:
        for key, values in supplied.items():
            merged[str(key)] = _ordered_unique(
                [*merged.get(str(key), []), *[str(value) for value in values]]
            )
    return merged


def run_idea_mining_dry_run(
    *,
    materials: Sequence[SourceMaterial | Mapping[str, Any]],
    llm: LLMClient,
    available_concepts: Sequence[ConceptDescriptor | str],
    output_dir: str | Path,
    concept_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    outcome_determinability: Optional[Mapping[
        str, OutcomeDeterminability | Mapping[str, Any] | str
    ]] = None,
    database: str = "miiv",
    data_path: Optional[str | Path] = None,
    registry_path: Optional[str | Path] = None,
    cohort: Optional[Mapping[str, Any]] = None,
    analytic_unit: Literal["stay", "patient"] = "stay",
    top_k: int = 5,
    citations: Sequence[Any] = (),
    feasibility_probe: Optional[FeasibilityProbe] = None,
    prior_art_search_client: Optional[Any] = None,
    prior_art_searched_at: Optional[str] = None,
    prior_art_top_n: int = 20,
) -> IdeaMiningDryRunResult:
    """Run the S4→S1→S3→S2 idea-triage dry run and stop at the human gate.

    The function freezes source provenance, extracts literature ideas, maps
    them to executable candidates, probes pair-level joint feasibility one pair
    at a time, ranks the executable candidates, and preregisters the resulting
    choice set as ``proposed``. It never imports or invokes the analysis
    pipeline, and it never marks a candidate ``accepted``.

    If callers do not supply dictionary metadata, EasyICU's concept catalog is
    loaded for the provided concept keys so literature phrases such as
    "vasopressin" or "intensive-care unit mortality" resolve without Web-only
    alias glue. Passing an explicit ``outcome_determinability`` mapping,
    including an empty one, keeps the caller's gate semantics.
    """

    parsed_materials = [
        raw if isinstance(raw, SourceMaterial) else SourceMaterial.model_validate(raw)
        for raw in materials
    ]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = freeze_source_snapshot(parsed_materials)
    manifest_path = out_dir / "source_snapshot_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    literature_ideas = extract_literature_ideas(
        materials=parsed_materials,
        source_snapshot_id=manifest.source_snapshot_id,
        llm=llm,
    )
    default_catalog = _default_concept_catalog_for_idea_run(available_concepts)
    effective_aliases = _merge_concept_aliases(
        default_catalog.concept_aliases,
        concept_aliases,
    )
    effective_outcome_determinability: Mapping[
        str, OutcomeDeterminability | Mapping[str, Any] | str
    ] = (
        default_catalog.outcome_determinability
        if outcome_determinability is None
        else outcome_determinability
    )
    executable_candidates = [
        map_literature_idea_to_executable_candidate(
            idea,
            available_concepts=available_concepts,
            concept_aliases=effective_aliases,
            outcome_determinability=effective_outcome_determinability,
        )
        for idea in literature_ideas
    ]
    unique_candidates = _unique_hypothesis_candidates(executable_candidates)
    yield_report = _build_yield_report(literature_ideas, executable_candidates)
    family_id = _stable_hypothesis_family_id(
        manifest.source_snapshot_id,
        unique_candidates,
    )

    warnings: List[str] = []
    if executable_candidates and yield_report.n_executable == 0:
        warnings.append(
            "No executable candidates after concept mapping and outcome "
            "determinability gates; this is a mapping/gating bottleneck, not "
            "an extraction failure."
        )

    prior_art_assessments: List[PriorArtAssessment] = []
    saturation_by_pair: Dict[Tuple[str, str], float] = {}
    if prior_art_search_client is not None:
        prior_art_assessments = assess_prior_art_for_candidates(
            literature_ideas=literature_ideas,
            executable_candidates=executable_candidates,
            search_client=prior_art_search_client,
            searched_at=prior_art_searched_at,
            top_n=prior_art_top_n,
        )
        prior_art_by_literature_id = {
            assessment.literature_idea_id: assessment
            for assessment in prior_art_assessments
        }
        for candidate in executable_candidates:
            if not candidate.feasibility_pair_key:
                continue
            assessment = prior_art_by_literature_id.get(candidate.literature_idea_id)
            if assessment is None:
                continue
            saturation_by_pair[_normalise_pair_tuple(candidate.feasibility_pair_key)] = (
                assessment.literature_saturation_signal
            )

    pair_feasibility, feasibility_records, feasibility_warnings = (
        _build_pair_feasibility_signals(
            candidates=executable_candidates,
            database=database,
            data_path=data_path,
            cohort=cohort,
            analytic_unit=analytic_unit,
            feasibility_probe=feasibility_probe,
        )
    )
    warnings.extend(feasibility_warnings)

    ranking_results = _rank_executable_candidates(
        candidates=unique_candidates,
        available_concepts=available_concepts,
        database=database,
        hypothesis_family_id=family_id,
        feasibility_by_pair=pair_feasibility,
        saturation_by_pair=saturation_by_pair,
        citations=citations or [material.citation for material in parsed_materials],
        top_k=top_k,
    )
    ranked_json = _flatten_ranking_results(ranking_results)
    warnings.extend(_feasibility_match_warnings(pair_feasibility, ranked_json))

    registry_file = Path(registry_path) if registry_path else out_dir / "idea_registry.json"
    registry = IdeaCandidateRegistry(registry_file)
    ranking_by_pair = _ranking_by_pair(ranked_json)
    registry_ids: Dict[str, str] = {}
    registry_id_by_key: Dict[Tuple[str, str, str, str], str] = {}
    for candidate in unique_candidates:
        pair_key = candidate.feasibility_pair_key
        ranked = ranking_by_pair.get(pair_key) if pair_key else None
        registry_candidate_id = (
            str(ranked.get("candidate_id"))
            if ranked is not None
            else candidate.executable_candidate_id
        )
        registry_ids[candidate.executable_candidate_id] = registry_candidate_id
        registry_id_by_key[_candidate_hypothesis_key(candidate)] = registry_candidate_id
        try:
            registry.register_candidate(
                CandidateRegistryEntry(
                    hypothesis_family_id=family_id,
                    candidate_id=registry_candidate_id,
                    source_snapshot_id=manifest.source_snapshot_id,
                )
            )
        except CandidateAlreadyRegisteredError:
            warnings.append(
                f"Candidate already present in registry; preserved append-only "
                f"ledger entry: {registry_candidate_id}"
            )
    for candidate in executable_candidates:
        key = _candidate_hypothesis_key(candidate)
        if candidate.executable_candidate_id not in registry_ids and key in registry_id_by_key:
            registry_ids[candidate.executable_candidate_id] = registry_id_by_key[key]

    candidate_records = _build_candidate_records(
        candidates=executable_candidates,
        ranking_by_pair=ranking_by_pair,
        registry_ids=registry_ids,
        registry=registry,
        hypothesis_family_id=family_id,
        source_snapshot_id=manifest.source_snapshot_id,
    )
    novelty_path: Optional[Path] = None
    discovery_path: Optional[Path] = None
    discovery_records: List[DiscoveryCandidateRecord] = []
    if prior_art_assessments:
        novelty_path = out_dir / "novelty_snapshot_manifest.json"
        novelty_payload = {
            "schema_version": IDEA_NOVELTY_SNAPSHOT_SCHEMA_VERSION,
            "assessments": [
                assessment.model_dump(mode="json")
                for assessment in prior_art_assessments
            ],
        }
        novelty_path.write_text(
            json.dumps(novelty_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        discovery_records = build_discovery_candidate_records(
            literature_ideas=literature_ideas,
            executable_candidates=executable_candidates,
            prior_art_assessments=prior_art_assessments,
            candidate_records=candidate_records,
            source_materials=parsed_materials,
        )
        discovery_counts = _discovery_report_counts(discovery_records)
        discovery_path = out_dir / "discovery_report.md"
        discovery_path.write_text(
            render_discovery_report(discovery_records, counts=discovery_counts),
            encoding="utf-8",
        )
    else:
        discovery_counts = {
            "literature_rows": len(literature_ideas),
            "unique_executable_hypotheses": len(
                {
                    _candidate_hypothesis_key(candidate)
                    for candidate in executable_candidates
                    if candidate.executable
                }
            ),
            "multiple_testing_denominator": len(
                {
                    _candidate_hypothesis_key(candidate)
                    for candidate in executable_candidates
                    if candidate.executable
                }
            ),
        }
    triage_path = out_dir / "candidate_triage_report.json"
    triage_payload = {
        "schema_version": "easyicu.idea_mining_dry_run/1",
        "source_snapshot_manifest": manifest.model_dump(mode="json"),
        "hypothesis_family_id": family_id,
        "yield_report": yield_report.model_dump(mode="json"),
        "prior_art_assessments": [
            assessment.model_dump(mode="json")
            for assessment in prior_art_assessments
        ],
        "feasibility_signals": [
            record.model_dump(mode="json") for record in feasibility_records
        ],
        "ranked_candidates": ranked_json,
        "candidate_records": [
            record.model_dump(mode="json") for record in candidate_records
        ],
        "discovery_counts": discovery_counts,
        "discovery_records": [
            record.model_dump(mode="json") for record in discovery_records
        ],
        "warnings": warnings,
    }
    triage_path.write_text(
        json.dumps(triage_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return IdeaMiningDryRunResult(
        source_snapshot_manifest=manifest,
        hypothesis_family_id=family_id,
        literature_ideas=literature_ideas,
        executable_candidates=executable_candidates,
        yield_report=yield_report,
        prior_art_assessments=prior_art_assessments,
        feasibility_signals=feasibility_records,
        ranked_candidates=ranked_json,
        candidate_records=candidate_records,
        discovery_records=discovery_records,
        registry_path=str(registry_file),
        manifest_path=str(manifest_path),
        triage_report_path=str(triage_path),
        novelty_snapshot_path=str(novelty_path) if novelty_path else None,
        discovery_report_path=str(discovery_path) if discovery_path else None,
        warnings=warnings,
    )


def _build_yield_report(
    literature_ideas: Sequence[LiteratureIdeaCandidate],
    candidates: Sequence[ExecutableHypothesisCandidate],
) -> IdeaMiningYieldReport:
    unresolved_predictors = [
        candidate.predictor_label
        for candidate in candidates
        if candidate.resolved_predictor_concept is None
    ]
    unresolved_outcomes = [
        candidate.outcome_label
        for candidate in candidates
        if candidate.resolved_outcome_concept is None
    ]
    reasons = [
        reason
        for candidate in candidates
        for reason in candidate.non_executable_reasons
    ]
    return IdeaMiningYieldReport(
        n_literature_ideas=len(literature_ideas),
        n_resolved_predictor=sum(
            1 for candidate in candidates if candidate.resolved_predictor_concept
        ),
        n_resolved_outcome=sum(
            1 for candidate in candidates if candidate.resolved_outcome_concept
        ),
        n_executable=sum(1 for candidate in candidates if candidate.executable),
        n_non_executable=sum(1 for candidate in candidates if not candidate.executable),
        unresolved_predictor_labels=_top_values(unresolved_predictors),
        unresolved_outcome_labels=_top_values(unresolved_outcomes),
        top_non_executable_reasons=_top_values(reasons),
    )


def build_discovery_candidate_records(
    *,
    literature_ideas: Sequence[LiteratureIdeaCandidate],
    executable_candidates: Sequence[ExecutableHypothesisCandidate],
    prior_art_assessments: Sequence[PriorArtAssessment],
    candidate_records: Sequence[IdeaMiningCandidateTriageRecord],
    source_materials: Sequence[SourceMaterial],
) -> List[DiscoveryCandidateRecord]:
    """Build S6 human-facing discovery records from frozen structured inputs."""

    candidates_by_idea = {
        candidate.literature_idea_id: candidate
        for candidate in executable_candidates
    }
    assessments_by_idea = {
        assessment.literature_idea_id: assessment
        for assessment in prior_art_assessments
    }
    triage_by_exec = {
        record.executable_candidate_id: record
        for record in candidate_records
    }
    source_by_key = {material.citation.key: material.citation for material in source_materials}
    records: List[DiscoveryCandidateRecord] = []
    for idea in literature_ideas:
        assessment = assessments_by_idea.get(str(idea.literature_idea_id))
        if assessment is None:
            continue
        candidate = candidates_by_idea.get(str(idea.literature_idea_id))
        triage = (
            triage_by_exec.get(candidate.executable_candidate_id)
            if candidate is not None
            else None
        )
        source = source_by_key.get(idea.citation_key)
        feasibility = _database_feasibility_payload(triage)
        decision, decision_reason = _go_no_go_decision(
            candidate=candidate,
            assessment=assessment,
            triage=triage,
        )
        risks = _discovery_risks(
            candidate=candidate,
            assessment=assessment,
            triage=triage,
        )
        records.append(
            DiscoveryCandidateRecord(
                literature_idea_id=str(idea.literature_idea_id),
                executable_candidate_id=(
                    candidate.executable_candidate_id if candidate else None
                ),
                source_snapshot_id=idea.source_snapshot_id,
                citation_key=idea.citation_key,
                literature_source=_format_citation_source(source, idea.citation_key),
                gap_evidence_quote=idea.source_quote,
                candidate_topic=(
                    f"{idea.exposure_or_predictor} -> {idea.outcome} "
                    f"in {idea.population}"
                ),
                prior_art=assessment,
                database_feasibility=feasibility,
                go_no_go=decision,
                go_no_go_reason=decision_reason,
                risks=risks,
                clinical_plausibility_requires_human=True,
            )
        )
    return records


def _top_values(values: Sequence[str], *, limit: int = 5) -> List[str]:
    counts: Dict[str, int] = {}
    for value in values:
        text = str(value or "").strip()
        if text:
            counts[text] = counts.get(text, 0) + 1
    return sorted(counts, key=lambda item: (-counts[item], item))[:limit]


def _clean_literature_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _candidate_differentiators(idea: LiteratureIdeaCandidate) -> List[str]:
    raw = [
        idea.time_window_hint,
        idea.aggregation_hint,
        idea.analysis_family if idea.analysis_family != "association" else None,
    ]
    out: List[str] = []
    for item in raw:
        text = _clean_literature_phrase(str(item or ""))
        if text and normalize_concept_name(text) not in {
            normalize_concept_name(idea.exposure_or_predictor),
            normalize_concept_name(idea.outcome),
        } and _is_specific_differentiator(text):
            out.append(text)
    return _ordered_unique(out)


def _is_specific_differentiator(value: str) -> bool:
    text = _clean_literature_phrase(value).lower()
    if not text:
        return False
    normalised = normalize_concept_name(text)
    generic = {normalize_concept_name(item) for item in _GENERIC_DIFFERENTIATOR_PATTERNS}
    if normalised in generic:
        return False
    return not any(pattern in text for pattern in _GENERIC_DIFFERENTIATOR_PATTERNS)


def _pubmed_phrase_clause(phrase: str) -> str:
    text = _clean_literature_phrase(phrase)
    if not text:
        return ""
    escaped = text.replace('"', "")
    if " " in escaped or "-" in escaped:
        return f'"{escaped}"[Title/Abstract]'
    return f"{escaped}[Title/Abstract]"


def _pubmed_or_clause(clauses: Sequence[str]) -> str:
    items = _ordered_unique([clause for clause in clauses if clause])
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "(" + " OR ".join(items) + ")"


def _pubmed_recall_clause(phrase: str) -> str:
    """Return a recall-oriented Title/Abstract clause from literature words."""

    text = _clean_literature_phrase(phrase)
    if not text:
        return ""
    clauses = [_pubmed_phrase_clause(text), _pubmed_mesh_clause(text)]
    for facet in _prior_art_phrase_facets(text):
        clauses.append(_pubmed_phrase_clause(facet))
    return _pubmed_or_clause(clauses)


def _pubmed_core_recall_clause(
    core_phrase: Optional[str],
    *,
    fallback_phrase: str,
) -> str:
    """Return a broad PubMed clause from core concept facets plus source words."""

    core = _clean_literature_phrase(str(core_phrase or ""))
    fallback = _clean_literature_phrase(fallback_phrase)
    phrases: List[str] = []
    if core:
        phrases.append(core)
        phrases.extend(_prior_art_synonym_phrases(core))
        clauses = [_pubmed_recall_clause(phrase) for phrase in phrases]
        if fallback and normalize_concept_name(fallback) != normalize_concept_name(core):
            clauses.append(_pubmed_phrase_clause(fallback))
        return _pubmed_or_clause(clauses)
    if fallback:
        phrases.append(fallback)
    return _pubmed_or_clause([_pubmed_recall_clause(phrase) for phrase in phrases])


def _pubmed_mesh_clause(phrase: str) -> str:
    text = _clean_literature_phrase(phrase)
    if not text:
        return ""
    escaped = text.replace('"', "")
    if " " in escaped or "-" in escaped:
        return f'"{escaped}"[MeSH Terms]'
    return f"{escaped}[MeSH Terms]"


def _pubmed_population_recall_clause(population: str) -> str:
    """Extract only durable population facets instead of an over-tight phrase."""

    text = _clean_literature_phrase(population).lower()
    if not text:
        return ""
    clauses: List[str] = []
    if "adult" in text:
        clauses.append(_pubmed_phrase_clause("adult"))
    if (
        "icu" in text
        or "intensive care" in text
        or "critical illness" in text
        or "critically ill" in text
    ):
        clauses.extend(
            [
                _pubmed_phrase_clause("ICU"),
                _pubmed_phrase_clause("intensive care"),
                _pubmed_phrase_clause("critically ill"),
                _pubmed_phrase_clause("critical illness"),
            ]
        )
    if (
        "mechanically ventilated" in text
        or "mechanical ventilation" in text
        or "ventilated" in text
    ):
        clauses.extend(
            [
                _pubmed_phrase_clause("mechanically ventilated"),
                _pubmed_phrase_clause("mechanical ventilation"),
            ]
        )
    if "septic shock" in text:
        clauses.append(_pubmed_phrase_clause("septic shock"))
    elif "sepsis" in text:
        clauses.append(_pubmed_phrase_clause("sepsis"))
    elif "shock" in text:
        clauses.append(_pubmed_phrase_clause("shock"))
    return _pubmed_or_clause(clauses)


def _prior_art_phrase_facets(phrase: str) -> List[str]:
    """Derive phrase-local recall facets without using EasyICU concept keys."""

    tokens = _prior_art_query_tokens(phrase)
    facets: List[str] = _prior_art_synonym_phrases(phrase)
    for size in (3, 2):
        for idx in range(0, len(tokens) - size + 1):
            facets.append(" ".join(tokens[idx : idx + size]))
    facets.extend(
        token for token in tokens if token not in _PRIOR_ART_SINGLETON_STOPWORDS
    )
    for token in list(tokens):
        facets.extend(_PRIOR_ART_QUERY_SYNONYMS.get(token, ()))
    original = _clean_literature_phrase(phrase).lower()
    return _ordered_unique([item for item in facets if item != original])[:6]


def _prior_art_synonym_phrases(phrase: str) -> List[str]:
    target = _clean_literature_phrase(phrase).lower()
    if not target:
        return []
    target_key = normalize_concept_name(target)
    out: List[str] = []
    for group in SYNONYM_GROUPS:
        group_keys = {normalize_concept_name(item) for item in group}
        if target_key in group_keys:
            out.extend(sorted(group))
    return _ordered_unique([item for item in out if normalize_concept_name(item) != target_key])


def _prior_art_query_tokens(phrase: str) -> List[str]:
    text = _clean_literature_phrase(phrase).lower().replace("-", " ")
    raw_tokens = re.findall(r"[a-z0-9]+", text)
    tokens = [
        token
        for token in raw_tokens
        if len(token) >= 3 and token not in _PRIOR_ART_QUERY_STOPWORDS
    ]
    return _ordered_unique(tokens)


def _run_prior_art_query(
    search_client: Any,
    *,
    query_type: Literal["broad", "exact"],
    query: str,
    max_results: int,
    idea: LiteratureIdeaCandidate,
) -> PriorArtQueryRecord:
    raw = _call_prior_art_search(
        search_client,
        query,
        max_results=max_results,
        idea=idea,
    )
    record = _coerce_prior_art_query_record(
        raw,
        query_type=query_type,
        query=query,
    )
    return record.model_copy(
        update={
            "top_hits": [
                _classify_direct_same_topic_hit(hit, idea)
                for hit in record.top_hits
            ]
        }
    )


def _call_prior_art_search(
    search_client: Any,
    query: str,
    *,
    max_results: int,
    idea: Optional[LiteratureIdeaCandidate] = None,
) -> Any:
    if hasattr(search_client, "search_prior_art"):
        try:
            return search_client.search_prior_art(
                query,
                max_results=max_results,
                idea=idea,
            )
        except TypeError:
            return search_client.search_prior_art(query, max_results=max_results)
    if callable(search_client) and not hasattr(search_client, "search"):
        try:
            return search_client(query, max_results=max_results, idea=idea)
        except TypeError:
            return search_client(query, max_results=max_results)
    if hasattr(search_client, "search"):
        search = search_client.search
        try:
            return search(query, max_results=max_results, idea=idea)
        except TypeError:
            try:
                return search(query, max_results=max_results)
            except TypeError:
                return search(query, retmax=max_results)
    raise IdeaMiningError("prior_art_search_client must provide search() or search_prior_art()")


def _coerce_prior_art_query_record(
    raw: Any,
    *,
    query_type: Literal["broad", "exact"],
    query: str,
) -> PriorArtQueryRecord:
    if isinstance(raw, PriorArtQueryRecord):
        return raw.model_copy(update={"query_type": query_type, "query": query})
    if isinstance(raw, Mapping):
        hits_raw = (
            raw.get("top_hits")
            or raw.get("hits")
            or raw.get("records")
            or raw.get("citations")
            or []
        )
        hits = [_coerce_prior_art_hit(item) for item in hits_raw]
        hit_count = int(
            raw.get("hit_count")
            if raw.get("hit_count") is not None
            else raw.get("count")
            if raw.get("count") is not None
            else raw.get("total")
            if raw.get("total") is not None
            else len(hits)
        )
        pmids = [str(pmid) for pmid in (raw.get("pmids") or []) if str(pmid).strip()]
        if not pmids:
            pmids = [hit.pmid for hit in hits]
        return PriorArtQueryRecord(
            query_type=query_type,
            query=query,
            hit_count=hit_count,
            pmids=_ordered_unique(pmids),
            top_hits=hits,
        )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        hits = [_coerce_prior_art_hit(item) for item in raw]
        return PriorArtQueryRecord(
            query_type=query_type,
            query=query,
            hit_count=len(hits),
            pmids=[hit.pmid for hit in hits],
            top_hits=hits,
        )
    return PriorArtQueryRecord(
        query_type=query_type,
        query=query,
        hit_count=0,
        pmids=[],
        top_hits=[],
    )


def _coerce_prior_art_hit(raw: Any) -> PriorArtSearchHit:
    if isinstance(raw, PriorArtSearchHit):
        return raw
    if isinstance(raw, CitationRecord):
        return PriorArtSearchHit(
            pmid=str(raw.pmid or raw.key),
            title=raw.title,
            venue=raw.venue,
            year=raw.year,
            relevance=raw.relevance,
        )
    if isinstance(raw, Mapping):
        pmid = raw.get("pmid") or raw.get("PMID") or raw.get("key") or raw.get("id")
        title = raw.get("title") or raw.get("Title")
        return PriorArtSearchHit(
            pmid=str(pmid or "unknown"),
            title=str(title or "Untitled prior-art hit"),
            venue=raw.get("venue") or raw.get("journal"),
            year=str(raw["year"]) if raw.get("year") is not None else None,
            relevance=raw.get("relevance") or raw.get("abstract"),
            direct_same_topic=bool(raw.get("direct_same_topic", False)),
            direct_same_topic_rationale=raw.get("direct_same_topic_rationale")
            or raw.get("rationale"),
            same_topic_screened=bool(
                raw.get("same_topic_screened")
                or raw.get("llm_screened")
                or raw.get("human_screened")
            ),
        )
    raise IdeaMiningError(f"Cannot coerce prior-art hit: {type(raw)!r}")


def _classify_direct_same_topic_hit(
    hit: PriorArtSearchHit,
    idea: LiteratureIdeaCandidate,
) -> PriorArtSearchHit:
    if hit.direct_same_topic:
        return hit.model_copy(
            update={
                "direct_same_topic_rationale": hit.direct_same_topic_rationale
                or "provided by prior-art search client",
                "same_topic_screened": True,
            }
        )
    text = normalize_concept_name(" ".join([hit.title, hit.relevance or ""]))
    predictor = normalize_concept_name(idea.exposure_or_predictor)
    outcome = normalize_concept_name(idea.outcome)
    differentiators = [
        normalize_concept_name(item)
        for item in _candidate_differentiators(idea)
    ]
    has_predictor = predictor and predictor in text
    has_outcome = outcome and outcome in text
    has_differentiator = not differentiators or any(item in text for item in differentiators)
    if has_predictor and has_outcome and has_differentiator:
        return hit.model_copy(
            update={
                "direct_same_topic": True,
                "direct_same_topic_rationale": (
                    "title/abstract text matched literature phrase, outcome, "
                    "and differentiator"
                ),
            }
        )
    return hit


def _same_topic_screen_status(
    records: Sequence[PriorArtQueryRecord],
) -> str:
    hits = [hit for record in records for hit in record.top_hits]
    if not hits:
        return "no_hits_to_screen"
    if all(hit.same_topic_screened for hit in hits):
        return "top-N same-topic screened"
    if any(hit.same_topic_screened for hit in hits):
        return "partially screened; automated-substring fallback remains"
    return "automated-substring-only, NOT screened"


def _label_prior_art(
    *,
    broad_count: int,
    exact_count: int,
    direct_same_topic_count: int,
    has_specific_differentiator: bool = True,
    sparse_threshold: int = 5,
) -> NoveltyLabel:
    if direct_same_topic_count > 0:
        return "already_done"
    if exact_count == 0 and broad_count <= sparse_threshold:
        return "apparently_gap" if has_specific_differentiator else "sparse"
    if exact_count <= sparse_threshold:
        return "sparse"
    return "crowded_but_differentiable"


def _saturation_for_novelty_label(label: NoveltyLabel) -> float:
    return {
        "already_done": 0.95,
        "crowded_but_differentiable": 0.70,
        "sparse": 0.25,
        "apparently_gap": 0.05,
    }[label]


def _ordered_unique(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _query_by_type(
    assessment: PriorArtAssessment,
    query_type: Literal["broad", "exact"],
) -> Optional[PriorArtQueryRecord]:
    for record in assessment.query_records:
        if record.query_type == query_type:
            return record
    return None


def _format_feasibility(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "not available"
    parts: List[str] = []
    pair = payload.get("feasibility_pair_key")
    if pair:
        parts.append(f"pair={tuple(pair)}")
    if payload.get("coverage_source"):
        parts.append(f"coverage_source={payload['coverage_source']}")
    if payload.get("feature_derivation_status"):
        parts.append(f"feature={payload['feature_derivation_status']}")
    if payload.get("feature_derivation_note"):
        parts.append(str(payload["feature_derivation_note"]))
    if payload.get("n_joint_complete") is not None and payload.get("denominator_n") is not None:
        parts.append(f"joint n={payload['n_joint_complete']}/{payload['denominator_n']}")
    if payload.get("feasibility_note"):
        parts.append(str(payload["feasibility_note"]))
    return "; ".join(parts) if parts else "not available"


def _discovery_report_counts(
    records: Sequence[DiscoveryCandidateRecord],
) -> Dict[str, int]:
    unique_executable = set()
    for record in records:
        if not bool(record.database_feasibility.get("executable")):
            continue
        pair = record.database_feasibility.get("feasibility_pair_key")
        feature = record.database_feasibility.get("feature_derivation_status")
        if pair:
            unique_executable.add((tuple(pair), feature))
        elif record.executable_candidate_id:
            unique_executable.add((record.executable_candidate_id, feature))
    return {
        "literature_rows": len(records),
        "unique_executable_hypotheses": len(unique_executable),
        "multiple_testing_denominator": len(unique_executable),
    }


def _escape_md_cell(value: str) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")


def _database_feasibility_payload(
    triage: Optional[IdeaMiningCandidateTriageRecord],
) -> Dict[str, Any]:
    if triage is None:
        return {}
    return {
        "resolved_predictor_concept": triage.resolved_predictor_concept,
        "resolved_outcome_concept": triage.resolved_outcome_concept,
        "feasibility_pair_key": triage.feasibility_pair_key,
        "feature_derivation_status": triage.feature_derivation_status,
        "feature_derivation_requirements": list(triage.feature_derivation_requirements),
        "feature_derivation_note": triage.feature_derivation_note,
        "coverage_source": triage.coverage_source,
        "feasibility_note": triage.feasibility_note,
        "n_joint_complete": triage.n_joint_complete,
        "denominator_n": triage.denominator_n,
        "executable": triage.executable,
        "non_executable_reasons": list(triage.non_executable_reasons),
    }


def _go_no_go_decision(
    *,
    candidate: Optional[ExecutableHypothesisCandidate],
    assessment: PriorArtAssessment,
    triage: Optional[IdeaMiningCandidateTriageRecord],
) -> Tuple[GoNoGoDecision, str]:
    if candidate is None:
        return "db-cannot-do", "concept or outcome gate prevents execution"
    if candidate.feature_derivation_status == "requires_derived_feature":
        return "hold", "requires feature engineering before execution"
    if candidate.feature_derivation_status == "unsupported":
        return "db-cannot-do", "feature derivation is unsupported"
    if not candidate.executable:
        return "db-cannot-do", "concept or outcome gate prevents execution"
    if assessment.novelty_label == "already_done":
        return "hold", "direct same-topic prior art found"
    if assessment.novelty_label == "crowded_but_differentiable":
        return "hold", "crowded prior-art field; needs human differentiation"
    if not assessment.has_specific_differentiator:
        return "hold", "no specific differentiator; needs screening"
    if "NOT screened" in assessment.same_topic_screen_status:
        return "hold", "top-N same-topic screen not completed"
    if triage is None or triage.coverage_source != "pair_joint_feasibility":
        return "hold", "pair-level database feasibility not established"
    if triage.n_joint_complete is not None and triage.n_joint_complete <= 0:
        return "hold", "zero joint-complete analytic units"
    return "recommend", "sparse prior art plus executable database feasibility"


def _discovery_risks(
    *,
    candidate: Optional[ExecutableHypothesisCandidate],
    assessment: PriorArtAssessment,
    triage: Optional[IdeaMiningCandidateTriageRecord],
) -> List[str]:
    risks = [
        "clinical_plausibility_requires_human",
        assessment.scope_note,
        "prior-art triage is not a novelty claim",
    ]
    if candidate is not None:
        risks.extend(candidate.non_executable_reasons)
        if candidate.feature_derivation_note:
            risks.append(candidate.feature_derivation_note)
        risks.extend(candidate.feature_derivation_requirements)
    if not assessment.has_specific_differentiator:
        risks.append("no specific differentiator; needs screening")
    if "NOT screened" in assessment.same_topic_screen_status:
        risks.append("automated-substring-only, NOT screened")
    if assessment.novelty_label == "sparse":
        risks.append("low prior-art count may reflect low clinical value")
    if triage is not None and triage.feasibility_note:
        risks.append(triage.feasibility_note)
    return _ordered_unique(risks)


def _format_citation_source(
    citation: Optional[CitationRecord],
    fallback_key: str,
) -> str:
    if citation is None:
        return fallback_key
    parts = [citation.title]
    meta = ", ".join(
        str(item)
        for item in [citation.venue, citation.year, f"PMID:{citation.pmid}" if citation.pmid else None]
        if item
    )
    if meta:
        parts.append(f"({meta})")
    return " ".join(parts)



def _stable_hypothesis_family_id(
    source_snapshot_id: str,
    candidates: Sequence[ExecutableHypothesisCandidate],
) -> str:
    payload = {
        "source_snapshot_id": source_snapshot_id,
        "candidate_hypotheses": sorted(
            list(_candidate_hypothesis_key(candidate))
            for candidate in _unique_hypothesis_candidates(candidates)
        ),
    }
    return f"idea-family/sha256:{_sha256_text(_canonical_json(payload))[:16]}"


def _candidate_hypothesis_key(
    candidate: ExecutableHypothesisCandidate,
) -> Tuple[str, str, str, str]:
    pair = candidate.feasibility_pair_key
    predictor = (
        pair[0]
        if pair
        else candidate.resolved_predictor_concept or candidate.predictor_label
    )
    outcome = (
        pair[1]
        if pair
        else candidate.resolved_outcome_concept or candidate.outcome_label
    )
    return (
        normalize_concept_name(predictor),
        normalize_concept_name(outcome),
        normalize_concept_name(candidate.analysis_family),
        candidate.feature_derivation_status,
    )


def _unique_hypothesis_candidates(
    candidates: Sequence[ExecutableHypothesisCandidate],
) -> List[ExecutableHypothesisCandidate]:
    seen: set[Tuple[str, str, str, str]] = set()
    out: List[ExecutableHypothesisCandidate] = []
    for candidate in candidates:
        key = _candidate_hypothesis_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _build_pair_feasibility_signals(
    *,
    candidates: Sequence[ExecutableHypothesisCandidate],
    database: str,
    data_path: Optional[str | Path],
    cohort: Optional[Mapping[str, Any]],
    analytic_unit: Literal["stay", "patient"],
    feasibility_probe: Optional[FeasibilityProbe],
) -> Tuple[
    Dict[Tuple[str, str], HypothesisFeasibilitySignal],
    List[IdeaMiningFeasibilityRecord],
    List[str],
]:
    warnings: List[str] = []
    pairs = _ordered_unique_pairs(
        candidate.feasibility_pair_key
        for candidate in candidates
        if candidate.executable and candidate.feasibility_pair_key
    )
    if not pairs:
        return {}, [], warnings
    if feasibility_probe is None and data_path is None:
        warnings.append(
            "Pair-level joint feasibility was not run because no data_path or "
            "feasibility_probe was supplied; S3 ranking is withheld to avoid "
            "silent fallback to single-variable missingness."
        )
        return {}, [], warnings

    probe = feasibility_probe or real_data_concept_feasibility
    signals: Dict[Tuple[str, str], HypothesisFeasibilitySignal] = {}
    records: List[IdeaMiningFeasibilityRecord] = []
    for pair in pairs:
        predictor, outcome = pair
        raw_result = probe(
            concepts=[predictor, outcome],
            database=database,
            data_path=data_path,
            cohort=cohort,
            analytic_unit=analytic_unit,
        )
        value = _lookup_probe_value(raw_result, predictor)
        if value is None:
            warnings.append(
                "S1 feasibility probe returned no record for predictor "
                f"{predictor!r} in pair {pair!r}; pair omitted from S3 ranking."
            )
            continue
        signal = _coerce_probe_feasibility_signal(value)
        signals[pair] = signal
        records.append(
            IdeaMiningFeasibilityRecord(
                predictor=predictor,
                outcome=outcome,
                pair_key=pair,
                joint_fraction_complete=signal.joint_fraction_complete,
                n_joint_complete=signal.n_joint_complete,
                denominator_n=signal.denominator_n,
                source=signal.source,
                note=signal.note,
            )
        )
    return signals, records, warnings


def _ordered_unique_pairs(
    pairs: Iterable[Optional[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    seen: set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair is None:
            continue
        normalised = _normalise_pair_tuple(pair)
        if normalised not in seen:
            seen.add(normalised)
            out.append(normalised)
    return out


def _normalise_pair_tuple(pair: Tuple[str, str]) -> Tuple[str, str]:
    return (normalize_concept_name(pair[0]), normalize_concept_name(pair[1]))


def _lookup_probe_value(raw_result: Mapping[str, Any], concept: str) -> Optional[Any]:
    if concept in raw_result:
        return raw_result[concept]
    normalised = normalize_concept_name(concept)
    for key, value in raw_result.items():
        if normalize_concept_name(str(key)) == normalised:
            return value
    return next(iter(raw_result.values()), None)


def _coerce_probe_feasibility_signal(value: Any) -> HypothesisFeasibilitySignal:
    if isinstance(value, HypothesisFeasibilitySignal):
        return HypothesisFeasibilitySignal(
            joint_fraction_complete=_bounded_fraction(value.joint_fraction_complete),
            n_joint_complete=value.n_joint_complete,
            denominator_n=value.denominator_n,
            source=value.source,
            note=value.note,
        )
    if isinstance(value, Mapping):
        joint = value.get("joint_fraction_complete")
        if joint is None:
            raise IdeaMiningError("S1 feasibility values require joint_fraction_complete")
        return HypothesisFeasibilitySignal(
            joint_fraction_complete=_bounded_fraction(joint),
            n_joint_complete=_optional_int(value.get("n_joint_complete")),
            denominator_n=_optional_int(value.get("denominator_n")),
            source=str(value.get("source") or "precomputed"),
            note=str(value["note"]) if value.get("note") is not None else None,
        )
    joint = getattr(value, "joint_fraction_complete", None)
    if joint is None:
        raise IdeaMiningError("S1 feasibility objects require joint_fraction_complete")
    return HypothesisFeasibilitySignal(
        joint_fraction_complete=_bounded_fraction(joint),
        n_joint_complete=_optional_int(getattr(value, "n_joint_complete", None)),
        denominator_n=_optional_int(getattr(value, "denominator_n", None)),
        source=value.__class__.__name__,
        note=getattr(value, "note", None),
    )


def _bounded_fraction(value: Any) -> float:
    fraction = float(value)
    return max(0.0, min(1.0, fraction))


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


_RANKABLE_PREDICTOR_ROLES = {
    VariableRole.COMPOSITE_SCORE,
    VariableRole.ORDINAL_SCORE,
    VariableRole.LAB,
    VariableRole.VITAL,
    VariableRole.INTERVENTION,
}


def _rank_executable_candidates(
    *,
    candidates: Sequence[ExecutableHypothesisCandidate],
    available_concepts: Sequence[ConceptDescriptor | str],
    database: str,
    hypothesis_family_id: str,
    feasibility_by_pair: Mapping[Tuple[str, str], HypothesisFeasibilitySignal],
    saturation_by_pair: Mapping[Tuple[str, str], float],
    citations: Sequence[Any],
    top_k: int,
) -> List[HypothesisGeneratorResult]:
    if not feasibility_by_pair:
        return []
    descriptor_lookup = _descriptor_lookup(available_concepts)
    results: List[HypothesisGeneratorResult] = []
    for candidate in candidates:
        if not candidate.executable or not candidate.feasibility_pair_key:
            continue
        pair_key = _normalise_pair_tuple(candidate.feasibility_pair_key)
        if pair_key not in feasibility_by_pair:
            continue
        context = _ranking_context_for_candidate(
            candidate,
            descriptor_lookup=descriptor_lookup,
            database=database,
        )
        results.append(
            generate_hypotheses(
                context=context,
                citations=citations,
                top_k=max(1, min(1, top_k)),
                feasibility_by_pair={pair_key: feasibility_by_pair[pair_key]},
                saturation_by_pair=(
                    {pair_key: saturation_by_pair[pair_key]}
                    if pair_key in saturation_by_pair
                    else None
                ),
                hypothesis_family_id=hypothesis_family_id,
            )
        )
    return results


def _descriptor_lookup(
    available_concepts: Sequence[ConceptDescriptor | str],
) -> Dict[str, ConceptDescriptor]:
    out: Dict[str, ConceptDescriptor] = {}
    for item in available_concepts:
        if isinstance(item, ConceptDescriptor):
            canonical = normalize_concept_name(item.source_concept or item.name)
            descriptor = item.model_copy(update={"name": canonical})
            out[canonical] = descriptor
            for key in [item.name, item.source_concept or "", *item.derived_from_concepts]:
                if str(key or "").strip():
                    out.setdefault(normalize_concept_name(str(key)), descriptor)
        else:
            canonical = normalize_concept_name(str(item))
            out[canonical] = ConceptDescriptor(
                name=canonical,
                role=VariableRole.LAB,
                dtype="float64",
                source_concept=canonical,
            )
    return out


def _ranking_context_for_candidate(
    candidate: ExecutableHypothesisCandidate,
    *,
    descriptor_lookup: Mapping[str, ConceptDescriptor],
    database: str,
) -> ResearchContext:
    candidate.assert_research_context_allowed()
    if not candidate.feasibility_pair_key:
        raise NonExecutableCandidateError("candidate is missing feasibility_pair_key")
    predictor, outcome = _normalise_pair_tuple(candidate.feasibility_pair_key)
    return ResearchContext(
        research_question=candidate.research_question,
        cohort=CohortDescriptor(
            cohort_name="idea-mining dry-run triage context",
            database=database,
            n_patients=0,
            n_stays=0,
            outcome_columns=[outcome],
            provenance={
                "source_snapshot_id": candidate.source_snapshot_id,
                "literature_idea_id": candidate.literature_idea_id,
                "dry_run_only": True,
            },
            notes="S5 ranking-only context; not a pipeline execution context.",
        ),
        variables=[
            _descriptor_for_ranking(
                predictor,
                descriptor_lookup=descriptor_lookup,
                role=VariableRole.LAB,
            ),
            _descriptor_for_ranking(
                outcome,
                descriptor_lookup=descriptor_lookup,
                role=VariableRole.OUTCOME,
            ),
        ],
        target_outcome=outcome,
        notes="S5 idea-mining dry run; stops before human gate execution.",
    )


def _descriptor_for_ranking(
    key: str,
    *,
    descriptor_lookup: Mapping[str, ConceptDescriptor],
    role: VariableRole,
) -> ConceptDescriptor:
    canonical = normalize_concept_name(key)
    base = descriptor_lookup.get(canonical)
    if base is None:
        return ConceptDescriptor(
            name=canonical,
            role=role,
            dtype="int64" if role == VariableRole.OUTCOME else "float64",
            source_concept=canonical,
            missingness=MissingnessProfile(
                fraction_missing=0.0,
                n_missing=0,
                n_total=0,
                missingness_severity="unknown",
            ),
        )
    ranking_role = role
    if role != VariableRole.OUTCOME:
        ranking_role = base.role if base.role in _RANKABLE_PREDICTOR_ROLES else role
    return base.model_copy(
        update={
            "name": canonical,
            "role": ranking_role,
            "source_concept": base.source_concept or canonical,
        }
    )


def _flatten_ranking_results(
    ranking_results: Sequence[HypothesisGeneratorResult],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for result in ranking_results:
        for candidate in result.candidates:
            out.append(candidate.to_json())
    out.sort(key=lambda item: float(item.get("priority_score") or 0.0), reverse=True)
    return out


def _ranking_by_pair(
    ranked_candidates: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    out: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for candidate in ranked_candidates:
        pair = (
            normalize_concept_name(str(candidate.get("predictor") or "")),
            normalize_concept_name(str(candidate.get("outcome") or "")),
        )
        out.setdefault(pair, candidate)
    return out


def _feasibility_match_warnings(
    feasibility_by_pair: Mapping[Tuple[str, str], HypothesisFeasibilitySignal],
    ranked_candidates: Sequence[Mapping[str, Any]],
) -> List[str]:
    if not feasibility_by_pair:
        return []
    matched = [
        candidate
        for candidate in ranked_candidates
        if candidate.get("coverage_source") == "pair_joint_feasibility"
    ]
    if not matched:
        return [
            "Pair-level feasibility was provided, but no ranked candidate used "
            "coverage_source='pair_joint_feasibility'; check S4/S1/S3 canonical "
            "concept key alignment before execution."
        ]
    if len(matched) < len(feasibility_by_pair):
        return [
            "Some provided pair-level feasibility signals did not match ranked "
            f"candidate pairs: matched={len(matched)} provided={len(feasibility_by_pair)}."
        ]
    return []


def _build_candidate_records(
    *,
    candidates: Sequence[ExecutableHypothesisCandidate],
    ranking_by_pair: Mapping[Tuple[str, str], Mapping[str, Any]],
    registry_ids: Mapping[str, str],
    registry: IdeaCandidateRegistry,
    hypothesis_family_id: str,
    source_snapshot_id: str,
) -> List[IdeaMiningCandidateTriageRecord]:
    family_size = registry.family_size(hypothesis_family_id)
    executable_family_size = len(
        {
            registry_ids.get(
                candidate.executable_candidate_id,
                candidate.executable_candidate_id,
            )
            for candidate in candidates
            if candidate.executable
        }
    )
    records: List[IdeaMiningCandidateTriageRecord] = []
    for candidate in candidates:
        pair_key = (
            _normalise_pair_tuple(candidate.feasibility_pair_key)
            if candidate.feasibility_pair_key
            else None
        )
        ranked = ranking_by_pair.get(pair_key) if pair_key else None
        registry_candidate_id = registry_ids.get(
            candidate.executable_candidate_id,
            candidate.executable_candidate_id,
        )
        try:
            selection_status = registry.latest_entry(registry_candidate_id).selection_status
        except CandidateNotRegisteredError:
            selection_status = "proposed"
        records.append(
            IdeaMiningCandidateTriageRecord(
                literature_idea_id=candidate.literature_idea_id,
                executable_candidate_id=candidate.executable_candidate_id,
                registry_candidate_id=registry_candidate_id,
                hypothesis_family_id=hypothesis_family_id,
                source_snapshot_id=source_snapshot_id,
                citation_key=candidate.citation_key,
                predictor_label=candidate.predictor_label,
                outcome_label=candidate.outcome_label,
                resolved_predictor_concept=candidate.resolved_predictor_concept,
                resolved_outcome_concept=candidate.resolved_outcome_concept,
                feasibility_pair_key=pair_key,
                feature_derivation_status=candidate.feature_derivation_status,
                feature_derivation_requirements=list(
                    candidate.feature_derivation_requirements
                ),
                feature_derivation_note=candidate.feature_derivation_note,
                executable=candidate.executable,
                non_executable_reasons=list(candidate.non_executable_reasons),
                ranking_candidate_id=str(ranked.get("candidate_id")) if ranked else None,
                priority_score=(
                    float(ranked["priority_score"])
                    if ranked and ranked.get("priority_score") is not None
                    else None
                ),
                coverage_source=(
                    str(ranked["coverage_source"])
                    if ranked and ranked.get("coverage_source") is not None
                    else None
                ),
                feasibility_note=(
                    str(ranked["feasibility_note"])
                    if ranked and ranked.get("feasibility_note") is not None
                    else None
                ),
                n_joint_complete=(
                    int(ranked["n_joint_complete"])
                    if ranked and ranked.get("n_joint_complete") is not None
                    else None
                ),
                denominator_n=(
                    int(ranked["denominator_n"])
                    if ranked and ranked.get("denominator_n") is not None
                    else None
                ),
                registry_selection_status=str(selection_status),
                multiple_testing_family_size=family_size,
                multiple_testing_executable_family_size=executable_family_size,
                multiple_testing_note=(
                    "Preregistered all-considered candidate denominator only; "
                    "executable candidate denominator is reported separately; "
                    "no p-values are computed or adjusted in S5 dry run."
                ),
                causal_audit_risk=(
                    "static_triage_marker_requires_post_analysis_causal_audit"
                ),
                causal_audit_scope=(
                    "static_triage_marker_no_per_candidate_causal_audit"
                ),
            )
        )
    return records


def _parse_json_payload(raw: str) -> Any:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise IdeaExtractionError("idea extraction response is not valid JSON") from exc


def _source_text_lookup(materials: Sequence[SourceMaterial]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for material in materials:
        citation = material.citation
        if material.source_adapter_level == "metadata_only":
            text = " ".join(
                part
                for part in [
                    citation.title,
                    citation.venue or "",
                    citation.relevance or "",
                ]
                if str(part or "").strip()
            )
        else:
            text = str(material.source_text or "")
        out[citation.key] = text
    return out


def _quote_is_traceable(quote: str, source_text: str) -> bool:
    q = " ".join(str(quote or "").split()).lower()
    s = " ".join(str(source_text or "").split()).lower()
    return bool(q) and q in s


def _build_concept_lookup(
    available_concepts: Sequence[ConceptDescriptor | str],
    *,
    concept_aliases: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for item in available_concepts:
        if isinstance(item, ConceptDescriptor):
            canonical = normalize_concept_name(
                item.source_concept or item.name
            )
            keys = [item.name, item.source_concept or "", item.description or ""]
            keys.extend(item.derived_from_concepts)
        else:
            canonical = normalize_concept_name(str(item))
            keys = [str(item)]
        lookup[canonical] = canonical
        for key in keys:
            _add_lookup_key_variants(lookup, key, canonical)
    if concept_aliases:
        for target, aliases in concept_aliases.items():
            canonical = _resolve_concept(str(target), lookup)
            if canonical is None:
                continue
            for alias in aliases:
                _add_lookup_key_variants(lookup, alias, canonical)
    return lookup


def _resolve_concept(term: str, lookup: Mapping[str, str]) -> Optional[str]:
    for variant in _concept_lookup_variants(term):
        if variant in lookup:
            return lookup[variant]
    term_tokens = _concept_signal_tokens(term)
    if not term_tokens:
        return None
    # Pick the MOST SPECIFIC subset match, not the first one encountered: rank by
    # token overlap, then by the most concise key. This stops an incidental
    # single-token hit (e.g. "ventilation" in "ventilation-induced acute kidney
    # injury" matching vent_ind) from beating the 3-token semantic match
    # ("acute kidney injury" -> kdigo_aki). Deterministic: ties keep the first
    # key seen in insertion order.
    best: Optional[str] = None
    best_score = (0, 0)
    for key, canonical in lookup.items():
        key_tokens = _concept_signal_tokens(key)
        if not key_tokens:
            continue
        if key_tokens <= term_tokens or term_tokens <= key_tokens:
            score = (len(key_tokens & term_tokens), -len(key_tokens))
            if score > best_score:
                best_score = score
                best = canonical
    return best


def _add_lookup_key_variants(
    lookup: Dict[str, str],
    key: object,
    canonical: str,
) -> None:
    for variant in _concept_lookup_variants(str(key or "")):
        if variant:
            lookup[variant] = canonical


def _concept_lookup_variants(value: str) -> List[str]:
    canonical = normalize_concept_name(value)
    compact = re.sub(r"[^a-z0-9]+", "_", canonical).strip("_")
    variants = [canonical, compact]
    stripped = _strip_derived_feature_markers(compact)
    if stripped:
        variants.append(stripped)
    generic_stripped = _strip_generic_concept_words(stripped or compact)
    if generic_stripped:
        variants.append(generic_stripped)
    suffix_stripped = re.sub(r"\d+$", "", generic_stripped or stripped or compact)
    if suffix_stripped:
        variants.append(suffix_stripped)
    return _ordered_unique(variants)


def _strip_derived_feature_markers(value: str) -> str:
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", normalize_concept_name(value))
        if token and token not in _DERIVED_FEATURE_REQUIREMENTS
    ]
    return "_".join(tokens)


def _strip_generic_concept_words(value: str) -> str:
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", normalize_concept_name(value))
        if token and token not in _GENERIC_CONCEPT_WORDS
    ]
    return "_".join(tokens)


def _concept_signal_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", normalize_concept_name(value))
        if len(token) > 2
        and token not in _GENERIC_CONCEPT_WORDS
        and token not in _DERIVED_FEATURE_REQUIREMENTS
    }


def _feature_derivation_status(
    term: str,
    *,
    resolved_key: Optional[str],
) -> Tuple[FeatureDerivationStatus, List[str], Optional[str]]:
    markers = _derived_feature_markers(term)
    if not markers:
        return "raw_concept_available", [], None
    requirements = _ordered_unique(
        requirement
        for marker in markers
        for requirement in _DERIVED_FEATURE_REQUIREMENTS.get(marker, [])
    )
    if resolved_key is None:
        return (
            "unsupported",
            requirements,
            "derived-feature phrase could not be resolved to a supporting concept",
        )
    resolved_norm = normalize_concept_name(resolved_key)
    if any(marker in resolved_norm for marker in markers):
        return (
            "derived_feature_available",
            requirements,
            "resolved concept appears to represent the derived feature itself",
        )
    return (
        "requires_derived_feature",
        requirements,
        "raw supporting concept is available, but derived feature pipeline is not established",
    )


def _derived_feature_markers(term: str) -> List[str]:
    normalised = normalize_concept_name(term)
    markers: List[str] = []
    for marker in _DERIVED_FEATURE_REQUIREMENTS:
        if marker in normalised:
            markers.append(marker)
    return markers


def _coerce_outcome_determinability(
    raw: OutcomeDeterminability | Mapping[str, Any] | str,
    *,
    outcome: str,
) -> OutcomeDeterminability:
    if isinstance(raw, OutcomeDeterminability):
        return raw
    if isinstance(raw, Mapping):
        data = dict(raw)
        data.setdefault("outcome", outcome)
        return OutcomeDeterminability.model_validate(data)
    return OutcomeDeterminability(outcome=outcome, status=str(raw))  # type: ignore[arg-type]


def _lookup_outcome_determinability(
    label: str,
    resolved_key: Optional[str],
    specs: Mapping[str, OutcomeDeterminability | Mapping[str, Any] | str],
) -> OutcomeDeterminability:
    keys = [label]
    if resolved_key:
        keys.append(resolved_key)
    for key in keys:
        canonical = normalize_concept_name(key)
        for candidate_key in (key, canonical):
            if candidate_key in specs:
                return _coerce_outcome_determinability(
                    specs[candidate_key],
                    outcome=resolved_key or label,
                )
    return OutcomeDeterminability(outcome=resolved_key or label, status="unknown")


__all__ = [
    "DISCOVERY_REPORT_SCHEMA_VERSION",
    "ExecutableHypothesisCandidate",
    "FeatureDerivationStatus",
    "DiscoveryCandidateRecord",
    "DiscoveryTriageResult",
    "IDEA_EXTRACTION_SYSTEM_PROMPT",
    "IDEA_MINING_SNAPSHOT_SCHEMA_VERSION",
    "IDEA_NOVELTY_SNAPSHOT_SCHEMA_VERSION",
    "IdeaMiningCandidateTriageRecord",
    "IdeaMiningDryRunResult",
    "IdeaExtractionError",
    "IdeaMiningError",
    "IdeaMiningFeasibilityRecord",
    "IdeaMiningYieldReport",
    "LiteratureIdeaCandidate",
    "GoNoGoDecision",
    "NonExecutableCandidateError",
    "NoveltyLabel",
    "OutcomeDeterminability",
    "OutcomeDeterminabilityStatus",
    "PriorArtAssessment",
    "PriorArtQueryRecord",
    "PriorArtSearchHit",
    "SourceAdapterLevel",
    "SourceMaterial",
    "SourceSnapshotItem",
    "SourceSnapshotManifest",
    "assess_prior_art_for_candidates",
    "assess_prior_art_for_idea",
    "build_idea_extraction_messages",
    "build_discovery_candidate_records",
    "build_prior_art_queries",
    "extract_literature_ideas",
    "freeze_source_snapshot",
    "map_literature_idea_to_executable_candidate",
    "render_discovery_report",
    "run_idea_mining_dry_run",
]
