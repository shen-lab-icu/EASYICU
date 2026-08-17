"""Literature-derived idea extraction for EasyICU triage workflows.

S4 is the first idea-mining stage that looks upstream at review/editorial
material. It deliberately stops at an auditable candidate list and concept
mapping: no research context is generated for blocked candidates, no pipeline
is invoked, and licensed/full-text source material is never stored in freeze
manifests.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ..concept_availability import (
    normalize_concept_name,
    real_data_concept_feasibility,
)
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
from ..planning.analysis_types import is_concept_set_family, normalize_analysis_family
from .idea_scope import LiteratureScopeSpec, build_pubmed_query_from_scope
from ..literature import CitationRecord
from .idea_mining_schema import (  # noqa: F401  (re-exported for back-compat)
    DISCOVERY_REPORT_SCHEMA_VERSION,
    DiscoveryCandidateRecord,
    DiscoveryTriageResult,
    ExecutableHypothesisCandidate,
    FeatureDerivationStatus,
    GoNoGoDecision,
    IDEA_MINING_SNAPSHOT_SCHEMA_VERSION,
    IDEA_NOVELTY_SNAPSHOT_SCHEMA_VERSION,
    IdeaExtractionError,
    IdeaMiningCandidateTriageRecord,
    IdeaMiningDryRunResult,
    IdeaMiningError,
    IdeaMiningFeasibilityRecord,
    IdeaMiningYieldReport,
    LiteratureIdeaCandidate,
    NonExecutableCandidateError,
    NoveltyLabel,
    OutcomeDeterminability,
    OutcomeDeterminabilityStatus,
    PriorArtAssessment,
    PriorArtQueryRecord,
    PriorArtSearchHit,
    SourceAdapterLevel,
    SourceMaterial,
    SourceSnapshotItem,
    SourceSnapshotManifest,
    _canonical_json,
    _nonempty_text,
    _sha256_text,
    _stable_executable_id,
    _stable_idea_id,
    _utc_now_iso,
)
from ..providers.protocol import LLMClient, LLMMessage
from ..providers.factory import authorized_complete
from ..schema import (
    CohortDescriptor,
    ConceptDescriptor,
    MissingnessProfile,
    ResearchContext,
    VariableRole,
)
from .idea_mining_pubmed import (  # noqa: F401  (re-exported for back-compat)
    _GENERIC_CONCEPT_WORDS,
    _GENERIC_DIFFERENTIATOR_PATTERNS,
    _PRIOR_ART_QUERY_STOPWORDS,
    _PRIOR_ART_QUERY_SYNONYMS,
    _PRIOR_ART_SINGLETON_STOPWORDS,
    _clean_literature_phrase,
    _is_specific_differentiator,
    _ordered_unique,
    _prior_art_phrase_facets,
    _prior_art_query_tokens,
    _prior_art_synonym_phrases,
    _pubmed_core_recall_clause,
    _pubmed_mesh_clause,
    _pubmed_or_clause,
    _pubmed_phrase_clause,
    _pubmed_population_recall_clause,
    _pubmed_recall_clause,
    _top_values,
)
from .idea_mining_priorart import (  # noqa: F401  (re-exported for back-compat)
    _call_prior_art_search,
    _candidate_differentiators,
    _classify_direct_same_topic_hit,
    _coerce_prior_art_hit,
    _coerce_prior_art_query_record,
    _database_feasibility_payload,
    _discovery_report_counts,
    _discovery_risks,
    _escape_md_cell,
    _format_citation_source,
    _format_feasibility,
    _go_no_go_decision,
    _label_prior_art,
    _query_by_type,
    _run_prior_art_query,
    _same_topic_screen_status,
    _saturation_for_novelty_label,
    assess_prior_art_for_candidates,
    assess_prior_art_for_idea,
    build_prior_art_queries,
    render_discovery_report,
)
from .idea_mining_extraction_receipts import (
    extraction_batch_request,
    load_verified_parsed_extraction_response,
    persist_extraction_batch_receipt,
)
from .idea_mining_feasibility_tier import (  # noqa: F401  (re-exported)
    SourceItemIndex,
    classify_feasibility_tier,
)
from .idea_mining_selection import select_actionable_prior_art_screen

IDEA_EXTRACTION_SYSTEM_PROMPT = (
    "You extract candidate ICU research directions from review or editorial "
    "source material. Stay case-neutral: do not assume a specific disease, "
    "score, exposure, database, or outcome unless it appears in the supplied "
    "source material. Return only JSON."
)

_EXTRACTION_SUPPORTED_LEVELS = {"metadata_only", "user_supplied_excerpt"}


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
                discovery_route=material.discovery_route,
                source_text_role=material.source_text_role,
                parent_citation_key=material.parent_citation_key,
                source_rank=material.source_rank,
                source_text_sha256=sha,
                source_text_char_count=len(text),
                source_text_stored=False,
                rights_note=rights,
            )
        )

    digest_payload = [
        item.model_dump(mode="json", exclude={"rights_note"}) for item in items
    ]
    snapshot_id = (
        f"source-snapshot/sha256:{_sha256_text(_canonical_json(digest_payload))[:16]}"
    )
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
                "discovery_route": material.discovery_route,
                "source_text_role": material.source_text_role,
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
            "analysis_concepts",
            "rationale",
            "source_quote",
            "analysis_family",
            "time_window_hint",
            "aggregation_hint",
        ],
        "rules": [
            "source_quote must be copied from available_source_text",
            (
                "choose the research SHAPE that fits the gap. For a "
                "predictor->outcome question (association, prediction, survival, "
                "causal, treatment response) fill exposure_or_predictor and "
                "outcome. For a concept-SET question that has NO single "
                "predictor->outcome pair -- subphenotype/phenotype clustering, "
                "descriptive epidemiology / cohort characterization, or a "
                "data-quality / measurement-bias / cohort-definition / "
                "score-policy audit -- LEAVE exposure_or_predictor and outcome "
                "empty and instead list the variables or rule elements in "
                "analysis_concepts (each a single named construct), and set "
                "analysis_family accordingly (e.g. subphenotype_clustering, "
                "descriptive_epidemiology, data_quality_audit, "
                "measurement_bias_audit, cohort_definition_sensitivity, "
                "score_policy_sensitivity)"
            ),
            (
                "analysis_concepts: 2+ specific named constructs for a "
                "clustering/phenotyping idea, or 1+ for a descriptive / "
                "audit / sensitivity idea; omit (empty) for predictor->outcome ideas"
            ),
            (
                "for cohort_definition_sensitivity or score_policy_sensitivity, "
                "analysis_concepts must be measurable rule elements, thresholds, "
                "components, or source variables named or clearly implied by the "
                "quote. Do NOT use abstract evaluation labels such as feasibility, "
                "reliability, prognostic validity, validity, implementation, or "
                "clinical utility as analysis_concepts. If the quote names only "
                "those evaluation goals and not measurable rule elements, omit the "
                "idea rather than inventing computable criteria."
            ),
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


IDEA_REFLECTION_SYSTEM_PROMPT = (
    "You are a critical ICU research reviewer refining a draft set of candidate "
    "research directions. Improve precision and honesty; do not invent. Keep each "
    "idea's citation_key and source_quote EXACTLY as given (they are provenance "
    "anchors). Return only JSON."
)


def build_idea_reflection_messages(
    ideas: Sequence[LiteratureIdeaCandidate],
    *,
    materials: Sequence[SourceMaterial | Mapping[str, Any]],
    source_snapshot_id: str,
    round_idx: int,
    num_rounds: int,
    prior_art_titles: Optional[Sequence[Sequence[str]]] = None,
) -> List[LLMMessage]:
    """Build the case-neutral self-critique/refine prompt for one round.

    When ``prior_art_titles`` is given (aligned to ``ideas``), each draft idea
    carries the top prior-art titles found for it so the model can drop or
    differentiate already-covered directions (Phase 2b retrieval augmentation).
    """

    parsed = [
        raw if isinstance(raw, SourceMaterial) else SourceMaterial.model_validate(raw)
        for raw in materials
    ]
    source_blocks = [
        {
            "citation_key": material.citation.key,
            "available_source_text": _available_source_text(material),
        }
        for material in parsed
    ]
    draft = []
    for position, idea in enumerate(ideas):
        entry = {
            "citation_key": idea.citation_key,
            "source_quote": idea.source_quote,
            "population": idea.population,
            "exposure_or_predictor": idea.exposure_or_predictor,
            "outcome": idea.outcome,
            "analysis_concepts": list(idea.analysis_concepts),
            "analysis_family": idea.analysis_family,
            "rationale": idea.rationale,
        }
        if prior_art_titles is not None and position < len(prior_art_titles):
            titles = [str(t) for t in prior_art_titles[position] if str(t).strip()]
            if titles:
                entry["prior_art_titles"] = titles
        draft.append(entry)
    augmented = prior_art_titles is not None
    instruction = (
        "Critically refine the draft ideas. For each idea you keep: keep "
        "citation_key and source_quote VERBATIM; sharpen vague constructs "
        "into a single concrete named construct (drop placeholder words like "
        "marker, biomarker, AI, machine learning, model, score, strategy, "
        "approach when the source names no concrete construct); pick the "
        "correct research SHAPE -- a predictor->outcome pair (fill "
        "exposure_or_predictor + outcome) OR a concept SET (leave the pair "
        "empty and fill analysis_concepts with 2+ named variables for "
        "clustering/phenotyping, 1+ for descriptive/audit/sensitivity) -- and set "
        "analysis_family accordingly. Diversify: drop near-duplicate ideas that "
        "restate the same construct/outcome, keeping only the single sharpest "
        "version. DROP an idea entirely (omit it) if it is vague, ungrounded in "
        "its quote, or not feasible as ICU cohort data. Do not invent results, "
        "effect sizes, or new quotes."
    )
    if augmented:
        instruction += (
            " Each idea may carry prior_art_titles: if those titles show the "
            "direction is already well studied, DROP it unless you can name a "
            "concrete differentiator grounded in the source quote."
        )
    user_payload = {
        "source_snapshot_id": source_snapshot_id,
        "round": f"{round_idx + 1}/{num_rounds}",
        "instruction": instruction,
        "return": "JSON array of the refined ideas (same field names as the draft)",
        "draft_ideas": draft,
        "sources": source_blocks,
    }
    return [
        LLMMessage(role="system", content=IDEA_REFLECTION_SYSTEM_PROMPT),
        LLMMessage(role="user", content=_canonical_json(user_payload)),
    ]


def _available_source_text(material: SourceMaterial) -> str:
    """The text the model is allowed to quote from, matching the extraction path."""
    if material.source_adapter_level == "metadata_only":
        citation = material.citation
        return " ".join(
            part
            for part in [citation.title, citation.venue or "", citation.relevance or ""]
            if str(part or "").strip()
        )
    return material.source_text or ""


# The schema's own bound for ``source_quote`` -- read from the model field so a
# tightened constraint stays in one place and the truncation guard cannot drift.
_LITERATURE_IDEA_SOURCE_QUOTE_MAX = (
    LiteratureIdeaCandidate.model_fields["source_quote"].metadata[0].max_length
)

# The idea contract's accepted field names -- used to strip echoed context fields
# (e.g. injected ``prior_art_titles``) before validating an LLM item.
_LITERATURE_IDEA_FIELDS = frozenset(LiteratureIdeaCandidate.model_fields)


def extract_literature_ideas(
    *,
    materials: Sequence[SourceMaterial | Mapping[str, Any]],
    source_snapshot_id: str,
    llm: LLMClient,
    untraceable_quote_policy: Literal["raise", "skip"] = "raise",
    dropped_untraceable: Optional[List[str]] = None,
    dropped_invalid: Optional[List[str]] = None,
    malformed_batch_policy: Literal["raise", "skip"] = "raise",
    dropped_malformed_batches: Optional[List[List[str]]] = None,
    batch_receipt_dir: Optional[str | Path] = None,
    batch_size: int = 6,
    max_tokens: int = 4096,
    reflection_rounds: int = 0,
    reflection_search_client: Optional[Any] = None,
) -> List[LiteratureIdeaCandidate]:
    """Extract structured literature ideas with a case-neutral JSON prompt.

    ``untraceable_quote_policy`` controls what happens when an extracted idea's
    ``source_quote`` is not a verbatim substring of its cited source text (the
    anti-hallucination provenance gate). The default ``"raise"`` preserves the
    strict single-idea contract. ``"skip"`` drops only the offending idea and
    continues — appropriate for a large multi-article batch where one
    paraphrased quote should not discard every other (correctly grounded)
    idea. Dropped citation keys are appended to ``dropped_untraceable`` when a
    list is supplied. Either way an untraceable quote is NEVER admitted, so the
    provenance guarantee is unchanged; only the blast radius differs.

    Materials are extracted in batches of ``batch_size``. A single call over an
    entire multi-dozen-article corpus shares one ``max_tokens`` output budget, so
    the JSON array is truncated and most candidates are silently lost (the yield
    caps at ~10 ideas regardless of corpus size). Batching gives each small group
    of articles its own budget and concatenates the results, so yield scales with
    the corpus and per-article extraction is more thorough.
    """

    parsed_materials = [
        raw if isinstance(raw, SourceMaterial) else SourceMaterial.model_validate(raw)
        for raw in materials
    ]
    source_text_by_key = _source_text_lookup(parsed_materials)
    adapter_level_by_key = {
        material.citation.key: material.source_adapter_level
        for material in parsed_materials
    }

    if malformed_batch_policy not in {"raise", "skip"}:
        raise ValueError("malformed_batch_policy must be 'raise' or 'skip'")

    candidates: List[LiteratureIdeaCandidate] = []
    step = max(1, int(batch_size))
    for start in range(0, len(parsed_materials), step):
        batch = parsed_materials[start : start + step]
        batch_index = start // step
        messages = build_idea_extraction_messages(
            batch,
            source_snapshot_id=source_snapshot_id,
        )
        citation_keys = [material.citation.key for material in batch]
        request = extraction_batch_request(
            source_snapshot_id=source_snapshot_id,
            batch_index=batch_index,
            citation_keys=citation_keys,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            provider_name=str(getattr(llm, "name", type(llm).__name__)),
        )
        raw = None
        if batch_receipt_dir is not None:
            raw = load_verified_parsed_extraction_response(
                batch_receipt_dir,
                request=request,
            )
        if raw is None:
            raw = authorized_complete(
                llm, messages, max_tokens=max_tokens, temperature=0.0
            )
        try:
            payload = _parse_json_payload(raw)
            if not isinstance(payload, list):
                raise IdeaExtractionError(
                    "idea extraction response must be a JSON array"
                )
        except IdeaExtractionError as exc:
            if batch_receipt_dir is not None:
                persist_extraction_batch_receipt(
                    batch_receipt_dir,
                    request=request,
                    raw_response=raw,
                    parse_status="malformed",
                    parse_error=str(exc),
                )
            if malformed_batch_policy == "skip":
                if dropped_malformed_batches is not None:
                    dropped_malformed_batches.append(citation_keys)
                continue
            raise
        if batch_receipt_dir is not None:
            persist_extraction_batch_receipt(
                batch_receipt_dir,
                request=request,
                raw_response=raw,
                parse_status="parsed",
            )
        for item in payload:
            try:
                coerced = _coerce_extracted_idea_item(
                    item,
                    source_snapshot_id=source_snapshot_id,
                    source_text_by_key=source_text_by_key,
                    adapter_level_by_key=adapter_level_by_key,
                    untraceable_quote_policy=untraceable_quote_policy,
                    dropped_untraceable=dropped_untraceable,
                    dropped_invalid=dropped_invalid,
                )
            except IdeaExtractionError as exc:
                if malformed_batch_policy == "skip":
                    if dropped_invalid is not None:
                        dropped_invalid.append(str(exc))
                    continue
                raise
            if coerced is not None:
                candidates.append(coerced)

    # Phase 2 -- agentic reflection. An optional self-critique/refine loop
    # (inspired by ai-scientist-v2's reflection rounds) that sharpens vague
    # constructs, fixes the analysis SHAPE/family, and drops ungrounded ideas.
    # Default ``reflection_rounds=0`` preserves the single-pass behavior exactly.
    # Provenance is never weakened: each refined idea is re-validated through the
    # same traceability + length gate, so a tampered/untraceable quote is dropped.
    rounds = max(0, int(reflection_rounds))
    for round_idx in range(rounds):
        if not candidates:
            break
        candidates = _reflect_and_refine_ideas(
            candidates,
            materials=parsed_materials,
            source_snapshot_id=source_snapshot_id,
            llm=llm,
            source_text_by_key=source_text_by_key,
            adapter_level_by_key=adapter_level_by_key,
            untraceable_quote_policy=untraceable_quote_policy,
            dropped_untraceable=dropped_untraceable,
            dropped_invalid=dropped_invalid,
            round_idx=round_idx,
            num_rounds=rounds,
            max_tokens=max_tokens,
            reflection_search_client=reflection_search_client,
        )
    # Phase 2b -- idea archive. Collapse near-duplicate constructs so the agentic
    # loop diversifies instead of restating the same idea. Gated to the reflection
    # path so single-pass extraction keeps its exact prior behavior.
    if rounds > 0:
        candidates = _collapse_near_duplicate_ideas(candidates)
    return candidates


def _coerce_extracted_idea_item(
    item: Any,
    *,
    source_snapshot_id: str,
    source_text_by_key: Mapping[str, str],
    adapter_level_by_key: Mapping[str, Any],
    untraceable_quote_policy: Literal["raise", "skip"],
    dropped_untraceable: Optional[List[str]],
    dropped_invalid: Optional[List[str]] = None,
) -> Optional[LiteratureIdeaCandidate]:
    """Validate one raw extraction/refinement item into a candidate.

    Returns ``None`` when the item is dropped under ``skip`` policy (untraceable
    quote). Raises ``IdeaExtractionError`` for hard contract violations under the
    ``raise`` policy. The traceability + over-long-quote guards live here so both
    the initial pass and the reflection pass enforce the same provenance rules.
    """
    if not isinstance(item, Mapping):
        raise IdeaExtractionError("each idea extraction item must be an object")
    data = dict(item)
    data["source_snapshot_id"] = source_snapshot_id
    citation_key = str(data.get("citation_key") or "").strip()
    data.setdefault("source_adapter_level", adapter_level_by_key.get(citation_key))
    quote = str(data.get("source_quote") or "").strip()
    if not _quote_is_traceable(quote, source_text_by_key.get(citation_key, "")):
        if untraceable_quote_policy == "skip":
            if dropped_untraceable is not None:
                dropped_untraceable.append(citation_key)
            return None
        raise IdeaExtractionError(
            f"source_quote for citation_key={citation_key!r} is not traceable"
        )
    quote_max = _LITERATURE_IDEA_SOURCE_QUOTE_MAX
    if len(quote) > quote_max:
        data["source_quote"] = quote[:quote_max].rstrip()
    # A reflection pass may omit the stable id so it can re-derive; drop a stale
    # one so the refined construct gets a fresh content-addressed id.
    data.pop("literature_idea_id", None)
    # Drop any echoed context fields the model may copy back (e.g. the injected
    # ``prior_art_titles``); the schema forbids extras, and these are not part of
    # the idea contract.
    data = {key: value for key, value in data.items() if key in _LITERATURE_IDEA_FIELDS}
    try:
        return LiteratureIdeaCandidate.model_validate(data)
    except ValidationError as exc:
        if untraceable_quote_policy == "skip":
            if dropped_invalid is not None:
                dropped_invalid.append(citation_key)
            return None
        raise IdeaExtractionError(
            f"invalid idea extraction item for citation_key={citation_key!r}: {exc}"
        ) from exc


def _reflect_and_refine_ideas(
    ideas: Sequence[LiteratureIdeaCandidate],
    *,
    materials: Sequence[SourceMaterial],
    source_snapshot_id: str,
    llm: LLMClient,
    source_text_by_key: Mapping[str, str],
    adapter_level_by_key: Mapping[str, Any],
    untraceable_quote_policy: Literal["raise", "skip"],
    dropped_untraceable: Optional[List[str]],
    dropped_invalid: Optional[List[str]],
    round_idx: int,
    num_rounds: int,
    max_tokens: int,
    reflection_search_client: Optional[Any] = None,
) -> List[LiteratureIdeaCandidate]:
    """One self-critique/refine round over the current idea set.

    The model keeps ``citation_key`` and ``source_quote`` verbatim and may only
    sharpen the construct, fix the analysis shape/family, or drop an idea by
    omitting it. Refined items are re-validated through the same provenance gate,
    so the round can never admit an ungrounded idea. On any malformed response
    the original ideas are returned unchanged (reflection is best-effort).

    When ``reflection_search_client`` is supplied (Phase 2b retrieval-augmented
    reflection), the top prior-art titles for each idea are fetched and shown to
    the model so it can drop or differentiate ideas the literature already covers
    BEFORE they are mapped -- complementing the post-mapping novelty veto net.
    """
    prior_art_titles: Optional[List[List[str]]] = None
    if reflection_search_client is not None:
        prior_art_titles = _fetch_reflection_prior_art(
            ideas, search_client=reflection_search_client
        )
    messages = build_idea_reflection_messages(
        ideas,
        materials=materials,
        source_snapshot_id=source_snapshot_id,
        round_idx=round_idx,
        num_rounds=num_rounds,
        prior_art_titles=prior_art_titles,
    )
    try:
        raw = authorized_complete(llm, messages, max_tokens=max_tokens, temperature=0.0)
        payload = _parse_json_payload(raw)
    except Exception:
        return list(ideas)
    if not isinstance(payload, list) or not payload:
        return list(ideas)
    refined: List[LiteratureIdeaCandidate] = []
    seen: set[str] = set()
    for item in payload:
        try:
            coerced = _coerce_extracted_idea_item(
                item,
                source_snapshot_id=source_snapshot_id,
                source_text_by_key=source_text_by_key,
                adapter_level_by_key=adapter_level_by_key,
                untraceable_quote_policy="skip",
                dropped_untraceable=dropped_untraceable,
                dropped_invalid=dropped_invalid,
            )
        except IdeaExtractionError:
            continue
        if coerced is None:
            continue
        if coerced.literature_idea_id in seen:
            continue
        seen.add(str(coerced.literature_idea_id))
        refined.append(coerced)
    # A reflection round that drops everything is treated as a no-op rather than
    # silently discarding the whole yield.
    return refined or list(ideas)


def _signature_tokens(text: str) -> str:
    """Order-insensitive token signature of a construct phrase for dedup."""
    tokens = sorted(
        {
            token
            for token in re.split(r"[^a-z0-9]+", str(text or "").lower())
            if len(token) > 2 and token not in _GENERIC_CONCEPT_WORDS
        }
    )
    return " ".join(tokens)


def _idea_signature(idea: LiteratureIdeaCandidate) -> Tuple[str, ...]:
    """A coarse content signature used to collapse near-duplicate ideas.

    Pairwise ideas key on (family, predictor-tokens, outcome-tokens); concept-set
    ideas key on (family, sorted concept-token sets). This collapses "lactate ->
    mortality" appearing twice, or the same variable set re-proposed, without
    needing concept resolution.
    """
    family = normalize_analysis_family(idea.analysis_family)
    has_pair = bool(idea.exposure_or_predictor.strip()) and bool(idea.outcome.strip())
    if not has_pair and any(str(c).strip() for c in idea.analysis_concepts):
        concepts = tuple(
            sorted(
                {
                    _signature_tokens(str(c))
                    for c in idea.analysis_concepts
                    if str(c).strip()
                }
            )
        )
        return (family, "SET", *concepts)
    return (
        family,
        _signature_tokens(idea.exposure_or_predictor),
        _signature_tokens(idea.outcome),
    )


def _collapse_near_duplicate_ideas(
    ideas: Sequence[LiteratureIdeaCandidate],
) -> List[LiteratureIdeaCandidate]:
    """Drop later ideas whose coarse signature already appeared (idea archive).

    Diversifies the final set so the agentic loop does not converge on the same
    construct restated across rounds/articles. Order-preserving: the first
    occurrence of each signature is kept.
    """
    seen: set[Tuple[str, ...]] = set()
    out: List[LiteratureIdeaCandidate] = []
    for idea in ideas:
        sig = _idea_signature(idea)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(idea)
    return out


def _fetch_reflection_prior_art(
    ideas: Sequence[LiteratureIdeaCandidate],
    *,
    search_client: Any,
    max_titles: int = 5,
) -> List[List[str]]:
    """Best-effort top prior-art titles per idea for retrieval-augmented reflect.

    Returns a list aligned to ``ideas``; each entry is up to ``max_titles`` titles
    of broad prior-art hits. Any search error yields an empty list for that idea,
    so reflection degrades gracefully to non-augmented behavior.
    """
    per_idea: List[List[str]] = []
    for idea in ideas:
        titles: List[str] = []
        try:
            query = build_prior_art_queries(idea).get("broad", "")
            if query:
                result = search_client.search_prior_art(query, max_results=max_titles)
                hits = (
                    result.get("top_hits", [])
                    if isinstance(result, Mapping)
                    else getattr(result, "top_hits", [])
                )
                for hit in hits[:max_titles]:
                    title = (
                        hit.get("title")
                        if isinstance(hit, Mapping)
                        else getattr(hit, "title", "")
                    )
                    if title:
                        titles.append(str(title))
        except Exception:
            titles = []
        per_idea.append(titles)
    return per_idea


# --- Gap A: SciMON-style novelty *optimisation* loop -------------------------
# The prior-art screen / novelty veto-net only *labels* an idea's crowdedness;
# it never pushes the idea toward an under-explored angle. This loop closes that
# gap with SciMON's compare-to-prior-work-and-revise pattern: measure an idea's
# crowdedness against PubMed, ask the model to revise it toward a gap the prior
# art does NOT cover, then re-measure and keep the revision only if novelty
# strictly improved. Provenance is never weakened -- every revision is
# re-validated through the same verbatim-quote traceability gate, and any
# search/LLM failure leaves the idea unchanged (best-effort).
NOVELTY_OPTIMIZATION_SYSTEM_PROMPT = (
    "You are a critical ICU research strategist sharpening a draft research idea "
    "toward a genuinely under-explored angle. The supplied titles are work that "
    "is ALREADY published on this direction. Revise the idea so it targets a gap "
    "those titles do NOT already cover -- for example a more specific "
    "subpopulation, an under-studied effect-modifier, a distinct timing window, "
    "or a comparison the titles do not address. Keep citation_key and "
    "source_quote EXACTLY as given (they are provenance anchors) and stay "
    "grounded in that quote: never invent a construct the source does not "
    "support. If you cannot find an honest, differentiated angle grounded in the "
    "quote, return the idea unchanged. Return only JSON: a single idea object "
    "with the same field names as the draft."
)


def build_novelty_optimization_messages(
    idea: LiteratureIdeaCandidate,
    *,
    prior_art_titles: Sequence[str],
    source_text: str,
    round_idx: int,
    num_rounds: int,
) -> List[LLMMessage]:
    """Build the case-neutral revise-toward-novelty prompt for one idea/round."""
    payload = {
        "round": f"{round_idx + 1}/{num_rounds}",
        "draft_idea": {
            "citation_key": idea.citation_key,
            "source_quote": idea.source_quote,
            "population": idea.population,
            "exposure_or_predictor": idea.exposure_or_predictor,
            "outcome": idea.outcome,
            "analysis_concepts": list(idea.analysis_concepts),
            "analysis_family": idea.analysis_family,
            "rationale": idea.rationale,
        },
        "already_published_titles": [
            str(title) for title in prior_art_titles if str(title).strip()
        ],
        "available_source_text": source_text,
        "instruction": (
            "Revise the draft idea toward an angle the already-published titles "
            "do NOT cover, keeping citation_key and source_quote verbatim and "
            "grounded in the quote. Return the single revised idea object."
        ),
        "return": "JSON object (one idea; same field names as draft_idea)",
    }
    return [
        LLMMessage(role="system", content=NOVELTY_OPTIMIZATION_SYSTEM_PROMPT),
        LLMMessage(role="user", content=_canonical_json(payload)),
    ]


def _measure_idea_novelty(
    idea: LiteratureIdeaCandidate,
    *,
    search_client: Any,
    max_results: int,
) -> Tuple[int, List[str]]:
    """Return (total prior-art hit count, top titles) as a crowdedness signal.

    The hit count is the PubMed esearch total for the idea's exact-phrase
    novelty query, so a lower count means a less-crowded direction. The search is
    run WITHOUT the idea (no per-hit same-topic screening) so the measurement is
    a cheap count, not a full assessment. A search error yields ``-1`` so the
    optimiser treats the measurement as unavailable and leaves the idea
    unchanged rather than mistaking a failure for high novelty.
    """
    if search_client is None or not hasattr(search_client, "search_prior_art"):
        return -1, []
    try:
        queries = build_prior_art_queries(idea)
        query = queries.get("exact") or queries.get("broad") or ""
        if not query:
            return -1, []
        result = search_client.search_prior_art(query, max_results=max_results)
    except Exception:
        return -1, []
    if not isinstance(result, Mapping):
        result = {
            "hit_count": getattr(result, "hit_count", 0),
            "top_hits": getattr(result, "top_hits", []),
        }
    count = int(result.get("hit_count") or 0)
    titles = [
        str(hit.get("title"))
        for hit in (result.get("top_hits") or [])
        if isinstance(hit, Mapping) and hit.get("title")
    ]
    return count, titles


def _idea_construct_label(idea: LiteratureIdeaCandidate) -> str:
    """A short construct/outcome label for the optimisation trace."""
    construct = idea.exposure_or_predictor.strip() or ", ".join(
        str(concept) for concept in idea.analysis_concepts if str(concept).strip()
    )
    outcome = idea.outcome.strip()
    return f"{construct} -> {outcome}" if outcome else construct


def optimize_ideas_for_novelty(
    ideas: Sequence[LiteratureIdeaCandidate],
    *,
    materials: Sequence[SourceMaterial | Mapping[str, Any]],
    source_snapshot_id: str,
    llm: LLMClient,
    search_client: Any,
    crowded_min_hits: int = 5,
    measure_max_results: int = 5,
    rounds: int = 1,
    max_tokens: int = 2048,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> List[LiteratureIdeaCandidate]:
    """Push crowded ideas toward novelty (measure -> revise -> re-measure).

    For each idea whose exact prior-art hit count is at or above
    ``crowded_min_hits``, the model is asked to revise toward a differentiated
    angle, the revision is re-measured, and it replaces the original ONLY if the
    hit count strictly drops. Otherwise the original is preserved. Each revision
    is re-validated through the same verbatim-quote provenance gate, so a revised
    idea whose quote was tampered is dropped back to the original. Returns one
    idea per input (revised ideas carry a freshly derived content id). When a
    ``trace`` list is supplied each idea's before/after signal is appended.
    """
    if not ideas or search_client is None:
        return list(ideas)
    if not hasattr(search_client, "search_prior_art"):
        return list(ideas)
    parsed_materials = [
        raw if isinstance(raw, SourceMaterial) else SourceMaterial.model_validate(raw)
        for raw in materials
    ]
    source_text_by_key = _source_text_lookup(parsed_materials)
    adapter_level_by_key = {
        material.citation.key: material.source_adapter_level
        for material in parsed_materials
    }
    rounds = max(1, int(rounds))
    optimized: List[LiteratureIdeaCandidate] = []
    for idea in ideas:
        current = idea
        base_count, titles = _measure_idea_novelty(
            current, search_client=search_client, max_results=measure_max_results
        )
        entry: Dict[str, Any] = {
            "citation_key": current.citation_key,
            "initial_construct": _idea_construct_label(current),
            "initial_exact_hits": base_count,
            "revised": False,
        }
        if base_count >= max(1, int(crowded_min_hits)):
            for round_idx in range(rounds):
                messages = build_novelty_optimization_messages(
                    current,
                    prior_art_titles=titles,
                    source_text=source_text_by_key.get(current.citation_key, ""),
                    round_idx=round_idx,
                    num_rounds=rounds,
                )
                try:
                    raw = authorized_complete(
                        llm, messages, max_tokens=max_tokens, temperature=0.3
                    )
                    payload = _parse_json_payload(raw)
                except Exception:
                    break
                item = payload[0] if isinstance(payload, list) and payload else payload
                if not isinstance(item, Mapping):
                    break
                try:
                    candidate = _coerce_extracted_idea_item(
                        item,
                        source_snapshot_id=source_snapshot_id,
                        source_text_by_key=source_text_by_key,
                        adapter_level_by_key=adapter_level_by_key,
                        untraceable_quote_policy="skip",
                        dropped_untraceable=None,
                        dropped_invalid=None,
                    )
                except IdeaExtractionError:
                    candidate = None
                if candidate is None:
                    break
                new_count, new_titles = _measure_idea_novelty(
                    candidate,
                    search_client=search_client,
                    max_results=measure_max_results,
                )
                # Keep the revision only if it is a measured improvement.
                if 0 <= new_count < base_count:
                    current = candidate
                    base_count = new_count
                    titles = new_titles
                    entry["revised"] = True
                else:
                    break
            entry["final_construct"] = _idea_construct_label(current)
            entry["final_exact_hits"] = base_count
        optimized.append(current)
        if trace is not None:
            trace.append(entry)
    return optimized


# --- Gap B: ResearchAgent-style multi-criteria validator panel ---------------
# ResearchAgent scores ideas across structured criteria with dedicated validator
# agents; our refinement was a single prompt. This adds an advisory per-candidate
# score (clarity / novelty / feasibility_fit / impact, 1-5) recorded in the
# triage report and ledger. It NEVER changes the go/no-go gate -- it annotates,
# it does not promote or demote -- so the fail-closed executability contract is
# untouched.
CANDIDATE_VALIDATION_SYSTEM_PROMPT = (
    "You are a multi-criteria ICU research reviewer scoring a mined candidate "
    "research idea. Score each criterion as an integer 1 (poor) to 5 (excellent) "
    "and give a one-sentence justification. Be calibrated and conservative. "
    "Criteria: clarity (is the construct and question precisely specified?), "
    "novelty (does it go beyond what is already well studied?), feasibility_fit "
    "(does it match what routinely collected ICU EHR data can actually measure?), "
    "impact (would answering it matter clinically?). Judge only the information "
    "shown; do not invent results. Return only JSON: "
    '{"clarity": n, "novelty": n, "feasibility_fit": n, "impact": n, '
    '"justification": "<one sentence>"}.'
)


def build_candidate_validation_messages(
    record: Mapping[str, Any],
) -> List[LLMMessage]:
    """Build the case-neutral multi-criteria scoring prompt for one candidate."""
    prior_art = record.get("prior_art")
    novelty_label = (
        prior_art.get("novelty_label") if isinstance(prior_art, Mapping) else None
    )
    payload = {
        "candidate": {
            "topic": record.get("candidate_topic"),
            "go_no_go": record.get("go_no_go"),
            "feasibility_route": record.get("feasibility_route"),
            "novelty_label": novelty_label,
            "resolved_predictor": record.get("resolved_predictor_concept"),
            "resolved_outcome": record.get("resolved_outcome_concept"),
            "gap_evidence_quote": record.get("gap_evidence_quote"),
            "literature_source": record.get("literature_source"),
        },
        "instruction": (
            "Score the candidate on clarity, novelty, feasibility_fit, and "
            "impact (each 1-5) and justify in one sentence."
        ),
    }
    return [
        LLMMessage(role="system", content=CANDIDATE_VALIDATION_SYSTEM_PROMPT),
        LLMMessage(role="user", content=_canonical_json(payload)),
    ]


def score_candidates_multicriteria(
    records: Sequence[Any],
    *,
    llm: LLMClient,
    max_candidates: int = 20,
) -> List[Dict[str, Any]]:
    """Advisory multi-criteria scores for the top candidate records.

    Each record is scored 1-5 on clarity / novelty / feasibility_fit / impact
    with a one-sentence justification. Any malformed/failed response yields a
    score row with ``None`` values so the candidate still appears with an honest
    'unscored' marker rather than being silently dropped. This is annotation
    only and never feeds back into the go/no-go gate.
    """
    scores: List[Dict[str, Any]] = []
    for record in list(records)[: max(0, int(max_candidates))]:
        as_dict = (
            record.model_dump(mode="json")
            if hasattr(record, "model_dump")
            else dict(record)
        )
        data: Mapping[str, Any] = {}
        try:
            raw = authorized_complete(
                llm,
                build_candidate_validation_messages(as_dict),
                max_tokens=300,
                temperature=0.0,
            )
            parsed = _parse_json_payload(raw)
            if isinstance(parsed, Mapping):
                data = parsed
        except Exception:
            data = {}

        def _clamp_score(key: str) -> Optional[int]:
            try:
                value = int(round(float(data.get(key))))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return min(5, max(1, value))

        scores.append(
            {
                "candidate_topic": as_dict.get("candidate_topic"),
                "candidate_id": (
                    as_dict.get("executable_candidate_id")
                    or as_dict.get("candidate_id")
                ),
                "go_no_go": as_dict.get("go_no_go"),
                "clarity": _clamp_score("clarity"),
                "novelty": _clamp_score("novelty"),
                "feasibility_fit": _clamp_score("feasibility_fit"),
                "impact": _clamp_score("impact"),
                "justification": str(data.get("justification") or "").strip(),
            }
        )
    return scores


# Generic outcome umbrella phrases. An LLM mining a review's "future research"
# sentence frequently extracts a non-specific outcome ("clinical outcomes",
# "ICU outcomes", "patient outcomes", "poor prognosis") rather than a measurable
# concept. These are not a concept but an umbrella that, in an ICU cohort study,
# defaults to the canonical hard endpoint (mortality). Resolving such an umbrella
# to the caller-declared mortality determinable -- while recording the
# substitution in ``normalized_outcome_concept`` so a human can see and confirm
# it -- prevents a false "outcome concept is not available" db-cannot-do verdict
# when the predictor itself is perfectly resolvable. Case-neutral by design: this
# is a linguistic category plus a caller-supplied default, never a hard-coded
# benchmark outcome mapping.
_GENERIC_OUTCOME_UMBRELLAS = frozenset(
    normalize_concept_name(phrase)
    for phrase in (
        "outcome",
        "outcomes",
        "clinical outcome",
        "clinical outcomes",
        "patient outcome",
        "patient outcomes",
        "icu outcome",
        "icu outcomes",
        "icu outcomes overall",
        "hospital outcome",
        "hospital outcomes",
        "in-hospital outcome",
        "in-hospital outcomes",
        "poor outcome",
        "poor outcomes",
        "worse outcome",
        "worse outcomes",
        "prognosis",
        "poor prognosis",
        "worse prognosis",
        "longer-term prognosis",
        "poor longer-term prognosis",
    )
)


def _is_generic_outcome_umbrella(term: str) -> bool:
    """Whether ``term`` is a non-specific outcome umbrella, not a concept."""
    return normalize_concept_name(term) in _GENERIC_OUTCOME_UMBRELLAS


def _default_mortality_outcome(
    specs: Mapping[str, OutcomeDeterminability | Mapping[str, Any] | str],
    lookup: Mapping[str, str],
) -> Optional[OutcomeDeterminability]:
    """Pick a caller-declared mortality determinable to back a generic outcome.

    Returns the first explicitly mortality-labelled ``known_0_1`` determinable
    whose concept resolves in the available catalog. A generic mortality phrase
    must never fall through to an unrelated binary endpoint merely because it is
    the first 0/1 variable in a mapping.
    """
    for key, raw in specs.items():
        det = _coerce_outcome_determinability(raw, outcome=str(key))
        if det.status != "known_0_1":
            continue
        semantic_labels = (
            str(key),
            det.outcome,
            det.normalized_outcome_concept or "",
        )
        if not any(_is_mortality_label(label) for label in semantic_labels):
            continue
        target = det.normalized_outcome_concept or str(key)
        if _resolve_concept(str(target), lookup) is not None:
            return det
    return None


def _is_mortality_label(value: str) -> bool:
    normalised = normalize_concept_name(str(value or ""))
    return bool(re.search(r"(?:^|_)(?:death|mortality)(?:_|$)", normalised))


# Minimum DISTINCT resolvable concepts a concept-set family needs to be
# executable. A concept SET is a multi-variable analysis, so the default floor is
# 2: a "set" that resolves to a single concept (or whose many named terms all
# collapse to one concept, e.g. seven fluid terms -> fluid_balance) is NOT the
# analysis the literature idea described and must not be flagged executable.
# Single-variable threshold/policy families legitimately operate on one concept
# (e.g. a UCR cutoff sensitivity), so they keep a floor of 1.
_CONCEPT_SET_DEFAULT_MIN_RESOLVED = 2
_CONCEPT_SET_MIN_RESOLVED = {
    "score_policy_sensitivity": 1,
    "cohort_definition_sensitivity": 1,
}


def _concept_set_research_question(
    family: str, concepts: Sequence[str], population: str
) -> str:
    joined = ", ".join(concepts) if concepts else "the named variables"
    if family == "trajectory_clustering":
        return f"Do distinct subphenotypes emerge from {joined} in {population}?"
    if family == "data_quality_audit":
        return f"What is the data quality / completeness of {joined} in {population}?"
    if family == "measurement_bias_audit":
        return f"Could measurement processes bias observed values of {joined} in {population}?"
    if family == "cohort_definition_sensitivity":
        return f"How sensitive is the cohort definition involving {joined} in {population}?"
    if family == "score_policy_sensitivity":
        return f"How sensitive are score or component policies involving {joined} in {population}?"
    return f"How are {joined} distributed in {population}?"


def _map_concept_set_candidate(
    candidate: LiteratureIdeaCandidate,
    *,
    lookup: Mapping[str, str],
) -> ExecutableHypothesisCandidate:
    """Map a concept-SET idea (clustering / descriptive / data-quality audit).

    These families have no predictor->outcome pair: the research shape is a set
    of variables to cluster on, characterize, or audit. Feasibility is "enough of
    the named concepts resolve to real available concepts", not pair joint
    coverage. Unresolved concepts are recorded as a note (not a hard block) as
    long as the minimum count resolves, so a partially-covered set is still
    executable on its resolvable members.
    """
    family = normalize_analysis_family(candidate.analysis_family)
    resolved: List[str] = []
    unresolved: List[str] = []
    for raw_term in candidate.analysis_concepts:
        term = str(raw_term).strip()
        if not term:
            continue
        key = _resolve_concept(term, lookup)
        if key is None:
            unresolved.append(term)
        elif key not in resolved:
            resolved.append(key)

    named_count = len([c for c in candidate.analysis_concepts if str(c).strip()])
    min_required = _CONCEPT_SET_MIN_RESOLVED.get(
        family, _CONCEPT_SET_DEFAULT_MIN_RESOLVED
    )
    reasons: List[str] = []
    if len(resolved) < min_required:
        reasons.append(
            f"{family} needs at least {min_required} resolvable concept(s); "
            f"resolved {len(resolved)} of {named_count}"
        )

    note = None
    if unresolved:
        note = "analysis concepts not available (omitted): " + ", ".join(unresolved)

    executable_candidate_id = _stable_executable_id(
        {
            "literature_idea_id": candidate.literature_idea_id,
            "analysis_family": family,
            "analysis_concepts": resolved,
            "snapshot": candidate.source_snapshot_id,
        }
    )
    return ExecutableHypothesisCandidate(
        executable_candidate_id=executable_candidate_id,
        literature_idea_id=str(candidate.literature_idea_id),
        source_snapshot_id=candidate.source_snapshot_id,
        citation_key=candidate.citation_key,
        population=candidate.population,
        predictor_label="",
        outcome_label="",
        resolved_predictor_concept=None,
        resolved_outcome_concept=None,
        resolved_analysis_concepts=resolved,
        feasibility_pair_key=None,
        analysis_family=family,
        research_question=_concept_set_research_question(
            family,
            resolved or [c for c in candidate.analysis_concepts],
            candidate.population,
        ),
        source_quote=candidate.source_quote,
        feature_derivation_note=note,
        non_executable_reasons=reasons,
    )


def map_literature_idea_to_executable_candidate(
    candidate: LiteratureIdeaCandidate,
    *,
    available_concepts: Sequence[ConceptDescriptor | str],
    concept_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    concept_categories: Optional[Mapping[str, str]] = None,
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

    # Concept-SET shape (clustering / descriptive / data-quality): no
    # predictor->outcome pair, mapped on a resolvable concept set so it is no
    # longer buried as predictor=None db-cannot-do. Routed by SHAPE, not just the
    # family label: an idea that carries a real predictor+outcome pair always
    # stays on the pairwise path (a "trajectory" predictor->outcome association is
    # pairwise, not clustering), so existing behavior is fully preserved.
    _has_pair = bool(candidate.exposure_or_predictor.strip()) and bool(
        candidate.outcome.strip()
    )
    _has_concept_set = any(str(c).strip() for c in candidate.analysis_concepts)
    if not _has_pair and (
        _has_concept_set or is_concept_set_family(candidate.analysis_family)
    ):
        return _map_concept_set_candidate(candidate, lookup=lookup)

    reasons: List[str] = []

    predictor_term = (
        candidate.exposure_core_concept or ""
    ).strip() or candidate.exposure_or_predictor
    predictor_key = _resolve_concept(predictor_term, lookup)
    predictor_category_conflict = _predictor_category_conflict(
        predictor_term,
        predictor_key,
        concept_categories or {},
    )
    if predictor_category_conflict:
        predictor_key = None
    feature_status, feature_requirements, feature_note = _feature_derivation_status(
        candidate.exposure_or_predictor,
        resolved_key=predictor_key,
        lookup=lookup,
    )
    # A requires_derived_feature predictor (e.g. a two-component ratio) supersedes
    # the "not available" reason: its component concepts ARE available, the gap is
    # derivation, not availability. Otherwise an unresolved predictor is reported
    # as unavailable, and a resolved-but-unsupported derivation is reported as such.
    if feature_status == "requires_derived_feature":
        reasons.append(
            "predictor requires derived feature engineering before execution: "
            f"{candidate.exposure_or_predictor}"
        )
    elif predictor_category_conflict:
        reasons.append(
            "predictor administration/treatment wording cannot bind to a "
            "non-intervention measurement concept"
        )
    elif predictor_key is None:
        reasons.append(
            f"predictor concept is not available: {candidate.exposure_or_predictor}"
        )
    elif feature_status == "unsupported":
        reasons.append(
            "predictor feature derivation is unsupported: "
            f"{candidate.exposure_or_predictor}"
        )

    outcome_term = (candidate.outcome_core_concept or "").strip() or candidate.outcome
    outcome_key = _resolve_concept(outcome_term, lookup)
    generic_outcome_determinability: Optional[OutcomeDeterminability] = None
    if outcome_key is None and _is_generic_outcome_umbrella(outcome_term):
        default_mortality = _default_mortality_outcome(
            outcome_determinability or {}, lookup
        )
        if default_mortality is not None:
            target = (
                default_mortality.normalized_outcome_concept
                or default_mortality.outcome
            )
            resolved = _resolve_concept(str(target), lookup)
            if resolved is not None:
                outcome_key = resolved
                generic_outcome_determinability = default_mortality
    if outcome_key is None:
        reasons.append(f"outcome concept is not available: {candidate.outcome}")

    # A generic outcome umbrella reuses the caller's mortality determinable directly
    # (its spec key may differ from the resolved concept), so the determinability is
    # the declared known_0_1 default rather than an unknown re-lookup.
    determinability = (
        generic_outcome_determinability
        or _lookup_outcome_determinability(
            candidate.outcome,
            outcome_key,
            outcome_determinability or {},
        )
    )
    normalized_outcome = None
    # A generic outcome umbrella resolved to the caller's mortality default: record
    # the substitution so the human gate sees the original label was non-specific.
    if generic_outcome_determinability is not None:
        normalized_outcome = outcome_key
    # ``known_0_1`` and ``non_binary_determinable`` both pass clean: the gate only
    # blocks the present/NA binary coding trap (``event_present_na``) and the case
    # where determinability could not be established at all (``unknown``). A
    # continuous/ordinal/survival outcome is determinable and must not be gated out.
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
    from ..concept_catalog import load_concept_catalog

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


def fetch_source_materials_from_scope(
    scope: LiteratureScopeSpec,
    search_client: Any,
    *,
    reference_year: Optional[int] = None,
    retmax: int = 20,
) -> List[SourceMaterial]:
    """Retrieve metadata-only source materials for a declarative scope.

    This is the discovery-lever-1 search front-end: it turns a
    ``LiteratureScopeSpec`` into a PubMed query (via
    :func:`~easyicu.research_agent.discovery.idea_scope.build_pubmed_query_from_scope`),
    runs the caller-injected ``search_client`` (``search_client.search(query,
    retmax=...) -> Sequence[CitationRecord]``), and wraps each hit as a
    ``metadata_only`` :class:`SourceMaterial`.

    Only metadata (titles/venues/ids) is captured — no abstract or full-text
    body is fetched or stored, keeping the snapshot manifest copyright-clean.
    Network I/O happens only through the explicitly injected ``search_client``;
    nothing here runs automatically.
    """
    query = build_pubmed_query_from_scope(scope, reference_year=reference_year)
    records = search_client.search(query, retmax=retmax)
    return [
        SourceMaterial(
            citation=record,
            source_adapter_level="metadata_only",
            discovery_route="scope_metadata",
            source_text_role="metadata_proxy",
            source_rank=idx,
        )
        for idx, record in enumerate(records, start=1)
    ]


def _validated_precomputed_ideas(
    raw_ideas: Sequence[LiteratureIdeaCandidate | Mapping[str, Any]],
    *,
    materials: Sequence[SourceMaterial],
    source_snapshot_id: str,
) -> List[LiteratureIdeaCandidate]:
    """Validate deterministic/data-first ideas against frozen source evidence.

    This is the non-LLM entry into the existing S4→S1→S3→S2 pipeline.  It does
    not relax source provenance: every idea must bind the snapshot produced in
    this run, cite one supplied material, and quote bytes that occur verbatim in
    that material.  Callers can therefore generate candidates from a frozen
    data profile without pretending that an LLM extracted them from a paper.
    """

    material_by_key = {material.citation.key: material for material in materials}
    ideas: List[LiteratureIdeaCandidate] = []
    for raw in raw_ideas:
        idea = (
            raw
            if isinstance(raw, LiteratureIdeaCandidate)
            else LiteratureIdeaCandidate.model_validate(raw)
        )
        if idea.source_snapshot_id != source_snapshot_id:
            raise IdeaMiningError(
                "precomputed idea source_snapshot_id does not match the frozen "
                f"source set: {idea.source_snapshot_id!r} != {source_snapshot_id!r}"
            )
        material = material_by_key.get(idea.citation_key)
        if material is None:
            raise IdeaMiningError(
                "precomputed idea cites material absent from the frozen source "
                f"set: {idea.citation_key!r}"
            )
        source_text = str(material.source_text or "")
        if not source_text or idea.source_quote not in source_text:
            raise IdeaMiningError(
                "precomputed idea source_quote is not verbatim in its frozen "
                f"source material: {idea.citation_key!r}"
            )
        ideas.append(idea)
    return ideas


def run_idea_mining_dry_run(
    *,
    materials: Sequence[SourceMaterial | Mapping[str, Any]] = (),
    llm: LLMClient,
    available_concepts: Sequence[ConceptDescriptor | str],
    output_dir: str | Path,
    concept_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    concept_categories: Optional[Mapping[str, str]] = None,
    outcome_determinability: Optional[
        Mapping[str, OutcomeDeterminability | Mapping[str, Any] | str]
    ] = None,
    database: str = "miiv",
    data_path: Optional[str | Path] = None,
    registry_path: Optional[str | Path] = None,
    cohort: Optional[Mapping[str, Any]] = None,
    analytic_unit: Literal["stay", "patient"] = "stay",
    analytic_population_age_group: Literal["adult", "pediatric", "mixed"] | None = None,
    top_k: int = 5,
    citations: Sequence[Any] = (),
    feasibility_probe: Optional[FeasibilityProbe] = None,
    prior_art_search_client: Optional[Any] = None,
    prior_art_searched_at: Optional[str] = None,
    prior_art_top_n: int = 20,
    prior_art_candidate_limit: Optional[int] = None,
    scope: Optional[LiteratureScopeSpec] = None,
    source_search_client: Optional[Any] = None,
    scope_reference_year: Optional[int] = None,
    scope_retmax: int = 20,
    untraceable_quote_policy: Literal["raise", "skip"] = "raise",
    malformed_extraction_batch_policy: Literal["raise", "skip"] = "raise",
    extraction_batch_receipt_dir: Optional[str | Path] = None,
    reflection_rounds: int = 0,
    reflection_search_client: Optional[Any] = None,
    novelty_judge: Optional[Callable[..., Mapping[str, Any]]] = None,
    novelty_optimize_rounds: int = 0,
    novelty_optimize_min_hits: int = 5,
    validate_candidates: bool = False,
    source_item_index: Optional["SourceItemIndex"] = None,
    extended_feasibility_index: Optional[object] = None,
    cross_db_targets: Optional[Sequence[str]] = None,
    precomputed_literature_ideas: Optional[
        Sequence[LiteratureIdeaCandidate | Mapping[str, Any]]
    ] = None,
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

    materials = list(materials)
    scope_query: Optional[str] = None
    if scope is not None:
        scope_query = build_pubmed_query_from_scope(
            scope, reference_year=scope_reference_year
        )
        if not materials:
            if source_search_client is None:
                raise IdeaMiningError(
                    "scope was supplied without source_search_client and no "
                    "explicit materials; cannot retrieve the literature corpus. "
                    "Pass source_search_client to fetch from scope, or supply "
                    "materials directly."
                )
            materials = fetch_source_materials_from_scope(
                scope,
                source_search_client,
                reference_year=scope_reference_year,
                retmax=scope_retmax,
            )

    parsed_materials = [
        raw if isinstance(raw, SourceMaterial) else SourceMaterial.model_validate(raw)
        for raw in materials
    ]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if scope_query is not None:
        (out_dir / "scope_query.json").write_text(
            json.dumps(
                {
                    "scope": scope.model_dump(mode="json"),
                    "scope_reference_year": scope_reference_year,
                    "scope_retmax": scope_retmax,
                    "pubmed_query": scope_query,
                    "n_materials_retrieved": len(parsed_materials),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = freeze_source_snapshot(parsed_materials)
    manifest_path = out_dir / "source_snapshot_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    dropped_untraceable: List[str] = []
    dropped_invalid: List[str] = []
    dropped_malformed_batches: List[List[str]] = []
    if precomputed_literature_ideas is None:
        literature_ideas = extract_literature_ideas(
            materials=parsed_materials,
            source_snapshot_id=manifest.source_snapshot_id,
            llm=llm,
            untraceable_quote_policy=untraceable_quote_policy,
            dropped_untraceable=dropped_untraceable,
            dropped_invalid=dropped_invalid,
            malformed_batch_policy=malformed_extraction_batch_policy,
            dropped_malformed_batches=dropped_malformed_batches,
            batch_receipt_dir=extraction_batch_receipt_dir,
            reflection_rounds=reflection_rounds,
            reflection_search_client=reflection_search_client,
        )
    else:
        literature_ideas = _validated_precomputed_ideas(
            precomputed_literature_ideas,
            materials=parsed_materials,
            source_snapshot_id=manifest.source_snapshot_id,
        )
    # Gap A -- SciMON-style novelty optimisation. Runs BEFORE concept mapping so a
    # crowded idea is revised toward a differentiated angle while still an idea
    # (mapping/feasibility then re-evaluate the sharper construct). Reuses the
    # prior-art search client as the crowdedness oracle; default rounds=0 is a
    # no-op that preserves the exact prior behaviour.
    novelty_optimization_trace: List[Dict[str, Any]] = []
    if (
        novelty_optimize_rounds > 0
        and prior_art_search_client is not None
        and literature_ideas
    ):
        literature_ideas = optimize_ideas_for_novelty(
            literature_ideas,
            materials=parsed_materials,
            source_snapshot_id=manifest.source_snapshot_id,
            llm=llm,
            search_client=prior_art_search_client,
            crowded_min_hits=novelty_optimize_min_hits,
            rounds=novelty_optimize_rounds,
            trace=novelty_optimization_trace,
        )
    default_catalog = _default_concept_catalog_for_idea_run(available_concepts)
    effective_aliases = _merge_concept_aliases(
        default_catalog.concept_aliases,
        concept_aliases,
    )
    effective_categories = {
        **default_catalog.concept_categories,
        **{str(key): str(value) for key, value in (concept_categories or {}).items()},
    }
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
            concept_categories=effective_categories,
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
    if dropped_untraceable:
        warnings.append(
            f"Dropped {len(dropped_untraceable)} idea(s) with untraceable "
            f"source_quote under untraceable_quote_policy='skip' "
            f"(citation_keys: {sorted(set(dropped_untraceable))}); the provenance "
            "gate still admitted no unverbatim quote."
        )
    if dropped_invalid:
        warnings.append(
            f"Dropped {len(dropped_invalid)} malformed idea(s) under "
            f"untraceable_quote_policy='skip' "
            f"(citation_keys: {sorted(set(dropped_invalid))}); one malformed "
            "LLM item did not abort the whole batch."
        )
    if dropped_malformed_batches:
        malformed_keys = sorted(
            {
                citation_key
                for batch_keys in dropped_malformed_batches
                for citation_key in batch_keys
            }
        )
        warnings.append(
            "Isolated "
            f"{len(dropped_malformed_batches)} malformed extraction batch(es) "
            f"under malformed_extraction_batch_policy='skip' "
            f"(source citation_keys: {malformed_keys}). No malformed JSON was "
            "repaired or admitted."
            + (
                " Verified parsed batches remain reusable from their "
                "content-bound receipts."
                if extraction_batch_receipt_dir is not None
                else ""
            )
        )
    if parsed_materials and all(
        material.source_adapter_level == "metadata_only"
        for material in parsed_materials
    ):
        warnings.append(
            "All source material is metadata_only (title/venue/relevance), so "
            "gap extraction has no access to discussion/limitations/future-work "
            "text. Unresolved-question mining is capped by source richness here; "
            "supply abstract-level or user_supplied_excerpt material to mine "
            "genuine gaps rather than inferring them from titles."
        )
    if executable_candidates and yield_report.n_executable == 0:
        warnings.append(
            "No executable candidates after concept mapping and outcome "
            "determinability gates; this is a mapping/gating bottleneck, not "
            "an extraction failure."
        )

    # Probe the host-owned data contract before spending network/API work on
    # prior art.  The default still screens the historical full candidate set;
    # callers may opt into a bounded screen for large literature corpora.
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

    screened_literature_ideas = list(literature_ideas)
    screened_executable_candidates = list(executable_candidates)
    if prior_art_candidate_limit is not None:
        if prior_art_candidate_limit < 1:
            raise IdeaMiningError("prior_art_candidate_limit must be >= 1")
        (
            screened_literature_ideas,
            screened_executable_candidates,
        ) = select_actionable_prior_art_screen(
            literature_ideas=literature_ideas,
            executable_candidates=executable_candidates,
            feasibility_by_pair=pair_feasibility,
            limit=prior_art_candidate_limit,
            analytic_population_age_group=analytic_population_age_group,
        )
        warnings.append(
            "Prior-art screening was feasibility-first and bounded: "
            f"screened={len(screened_literature_ideas)} of "
            f"{len(literature_ideas)} literature ideas; unscreened ideas carry "
            "no novelty verdict and cannot enter the proposed choice set."
        )

    prior_art_assessments: List[PriorArtAssessment] = []
    saturation_by_pair: Dict[Tuple[str, str], float] = {}
    if prior_art_search_client is not None:
        prior_art_assessments = assess_prior_art_for_candidates(
            literature_ideas=screened_literature_ideas,
            executable_candidates=screened_executable_candidates,
            search_client=prior_art_search_client,
            searched_at=prior_art_searched_at,
            top_n=prior_art_top_n,
            novelty_judge=novelty_judge,
            cross_db_targets=cross_db_targets,
        )
        prior_art_by_literature_id = {
            assessment.literature_idea_id: assessment
            for assessment in prior_art_assessments
        }
        for candidate in screened_executable_candidates:
            if not candidate.feasibility_pair_key:
                continue
            assessment = prior_art_by_literature_id.get(candidate.literature_idea_id)
            if assessment is None:
                continue
            saturation_by_pair[
                _normalise_pair_tuple(candidate.feasibility_pair_key)
            ] = assessment.literature_saturation_signal

    downstream_unique_candidates = (
        _unique_hypothesis_candidates(screened_executable_candidates)
        if prior_art_candidate_limit is not None
        else unique_candidates
    )
    downstream_pairs = {
        _normalise_pair_tuple(candidate.feasibility_pair_key)
        for candidate in downstream_unique_candidates
        if candidate.feasibility_pair_key
    }
    ranking_feasibility = {
        pair: signal
        for pair, signal in pair_feasibility.items()
        if pair in downstream_pairs
    }
    ranking_results = _rank_executable_candidates(
        candidates=downstream_unique_candidates,
        available_concepts=available_concepts,
        database=database,
        hypothesis_family_id=family_id,
        feasibility_by_pair=ranking_feasibility,
        saturation_by_pair=saturation_by_pair,
        citations=citations or [material.citation for material in parsed_materials],
        top_k=top_k,
    )
    ranked_json = _flatten_ranking_results(ranking_results)
    warnings.extend(_feasibility_match_warnings(ranking_feasibility, ranked_json))

    registry_file = (
        Path(registry_path) if registry_path else out_dir / "idea_registry.json"
    )
    registry = IdeaCandidateRegistry(registry_file)
    # Problem 4 -- snapshot the registry BEFORE this run registers anything, so a
    # candidate id already present is a PRIOR run/user (a homogenization
    # collision), not this run finding itself.
    prior_registry_ids = {entry.candidate_id for entry in registry.records}
    ranking_by_pair = _ranking_by_pair(ranked_json)
    registry_ids: Dict[str, str] = {}
    registry_id_by_key: Dict[Tuple[str, str, str, str], str] = {}
    for candidate in downstream_unique_candidates:
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
    downstream_candidates = (
        screened_executable_candidates
        if prior_art_candidate_limit is not None
        else executable_candidates
    )
    for candidate in downstream_candidates:
        key = _candidate_hypothesis_key(candidate)
        if (
            candidate.executable_candidate_id not in registry_ids
            and key in registry_id_by_key
        ):
            registry_ids[candidate.executable_candidate_id] = registry_id_by_key[key]

    candidate_records = _build_candidate_records(
        candidates=downstream_candidates,
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
            source_item_index=source_item_index,
            extended_feasibility_index=extended_feasibility_index,
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
    # Problem 4 -- homogenization signal. Any candidate whose registry id existed
    # BEFORE this run (a shared/persistent registry_path) is a collision: a prior
    # run/user already mined this construct. Advisory only.
    registry_collisions: List[Dict[str, Any]] = []
    if prior_registry_ids:
        for candidate in downstream_unique_candidates:
            reg_id = registry_ids.get(candidate.executable_candidate_id)
            if reg_id and reg_id in prior_registry_ids:
                registry_collisions.append(
                    {
                        "candidate_id": reg_id,
                        "executable_candidate_id": candidate.executable_candidate_id,
                        "predictor_label": candidate.predictor_label,
                        "outcome_label": candidate.outcome_label,
                        "note": (
                            "construct already registered in a prior run/user; "
                            "likely a homogenized direction -- differentiate or "
                            "coordinate before pursuing"
                        ),
                    }
                )

    # Gap B -- ResearchAgent-style multi-criteria validator panel. Advisory
    # scores over the final candidates (discovery rows when available, else the
    # triage records); never feeds back into the go/no-go gate.
    candidate_validation: List[Dict[str, Any]] = []
    if validate_candidates:
        validation_targets: Sequence[Any] = discovery_records or candidate_records
        if validation_targets:
            candidate_validation = score_candidates_multicriteria(
                validation_targets, llm=llm
            )

    triage_path = out_dir / "candidate_triage_report.json"
    triage_payload = {
        "schema_version": "easyicu.idea_mining_dry_run/1",
        "source_snapshot_manifest": manifest.model_dump(mode="json"),
        "hypothesis_family_id": family_id,
        "yield_report": yield_report.model_dump(mode="json"),
        "prior_art_assessments": [
            assessment.model_dump(mode="json") for assessment in prior_art_assessments
        ],
        "prior_art_screening": {
            "candidate_limit": prior_art_candidate_limit,
            "literature_ideas_total": len(literature_ideas),
            "literature_ideas_screened": len(screened_literature_ideas),
            "screened_literature_idea_ids": [
                str(idea.literature_idea_id) for idea in screened_literature_ideas
            ],
            "selection_basis": (
                "host_mapped_answerable_population_compatible_then_"
                "specific_differentiator_coverage_contrast"
                if prior_art_candidate_limit is not None
                else "unbounded_historical_behavior"
            ),
            "analytic_population_age_group": analytic_population_age_group,
        },
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
        "discovery_ledger": _discovery_ledger_rows(discovery_records),
        "novelty_optimization": novelty_optimization_trace,
        "candidate_validation": candidate_validation,
        "registry_collisions": registry_collisions,
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
        novelty_optimization=novelty_optimization_trace,
        candidate_validation=candidate_validation,
        registry_collisions=registry_collisions,
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


def _discovery_ledger_rows(
    records: Sequence[DiscoveryCandidateRecord],
) -> List[Dict[str, Any]]:
    """Flat, machine-readable discovery ledger for scripts and audits."""

    rows: List[Dict[str, Any]] = []
    for record in records:
        prior_art = record.prior_art
        feasibility = record.database_feasibility or {}
        broad_hits = [
            query.hit_count
            for query in prior_art.query_records
            if query.query_type == "broad"
        ]
        exact_hits = [
            query.hit_count
            for query in prior_art.query_records
            if query.query_type == "exact"
        ]
        rows.append(
            {
                "literature_idea_id": record.literature_idea_id,
                "executable_candidate_id": record.executable_candidate_id,
                "citation_key": record.citation_key,
                "literature_source": record.literature_source,
                "gap_evidence_quote": record.gap_evidence_quote,
                "candidate_topic": record.candidate_topic,
                "analysis_family": record.analysis_family,
                "resolved_analysis_concepts": list(record.resolved_analysis_concepts),
                "go_no_go": record.go_no_go,
                "go_no_go_reason": record.go_no_go_reason,
                "feasibility_route": record.feasibility_route,
                "feasibility_next_action": record.feasibility_next_action,
                "requires_human_confirmation": record.requires_human_confirmation,
                "resolved_predictor_concept": feasibility.get(
                    "resolved_predictor_concept"
                ),
                "resolved_outcome_concept": feasibility.get("resolved_outcome_concept"),
                "feature_derivation_status": feasibility.get(
                    "feature_derivation_status"
                ),
                "n_joint_complete": feasibility.get("n_joint_complete"),
                "denominator_n": feasibility.get("denominator_n"),
                "coverage_source": feasibility.get("coverage_source"),
                "novelty_label": prior_art.novelty_label,
                "same_topic_screen_status": prior_art.same_topic_screen_status,
                "broad_hit_count": max(broad_hits, default=0),
                "exact_hit_count": max(exact_hits, default=0),
                "direct_same_topic_pmids": list(prior_art.direct_same_topic_pmids),
                "differentiators": list(prior_art.differentiators),
                "evidence_map_counts": dict(prior_art.evidence_map_counts),
                "risks": list(record.risks),
            }
        )
    return rows


def build_discovery_candidate_records(
    *,
    literature_ideas: Sequence[LiteratureIdeaCandidate],
    executable_candidates: Sequence[ExecutableHypothesisCandidate],
    prior_art_assessments: Sequence[PriorArtAssessment],
    candidate_records: Sequence[IdeaMiningCandidateTriageRecord],
    source_materials: Sequence[SourceMaterial],
    source_item_index: Optional["SourceItemIndex"] = None,
    extended_feasibility_index: Optional[object] = None,
) -> List[DiscoveryCandidateRecord]:
    """Build S6 human-facing discovery records from frozen structured inputs.

    When ``source_item_index`` is supplied, each record is annotated with a
    three-tier source-feasibility verdict (executable / T1 re-extract / T2 new
    concept authorable / T3 not in this database) so the report no longer
    collapses every blocked candidate into one ``db-cannot-do`` cell.

    When ``extended_feasibility_index`` is supplied, a ``db-cannot-do`` verdict is
    reconsidered: if the cohort is ICD-derivable (Case 1) or a blocking construct
    is a dictionary concept reachable on this or another database (Case 2), the
    verdict is downgraded to ``hold`` with an actionable, human-confirm reason.
    This only downgrades; it never promotes to executable.
    """

    candidates_by_idea = {
        candidate.literature_idea_id: candidate for candidate in executable_candidates
    }
    assessments_by_idea = {
        assessment.literature_idea_id: assessment
        for assessment in prior_art_assessments
    }
    triage_by_exec = {
        record.executable_candidate_id: record for record in candidate_records
    }
    source_by_key = {
        material.citation.key: material.citation for material in source_materials
    }
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
        extended_meta: Optional[Dict[str, Any]] = None
        if decision == "db-cannot-do" and extended_feasibility_index is not None:
            try:
                verdict = extended_feasibility_index.reconsider(
                    idea=idea, candidate=candidate
                )
            except Exception:  # noqa: BLE001 - reconsideration is best-effort
                verdict = None
            if verdict is not None:
                decision = verdict.decision
                decision_reason = verdict.reason
                extended_meta = {"case": verdict.case, **verdict.metadata}
        risks = _discovery_risks(
            candidate=candidate,
            assessment=assessment,
            triage=triage,
        )
        tier = tier_note = None
        tier_items: List[Dict[str, Any]] = []
        if source_item_index is not None and candidate is not None:
            tier_result = classify_feasibility_tier(
                candidate, source_index=source_item_index
            )
            tier = tier_result.tier
            tier_note = tier_result.human_note
            tier_items = [
                {
                    "itemid": hit.itemid,
                    "label": hit.label,
                    "table": hit.table,
                    "category": hit.category,
                    "matched_tokens": list(hit.matched_tokens),
                }
                for hit in tier_result.source_item_hits
            ]
        route, next_action = _feasibility_route_and_next_action(
            decision=decision,
            candidate=candidate,
            triage=triage,
            feasibility_tier=tier,
            extended_feasibility=extended_meta,
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
                candidate_topic=_format_literature_candidate_topic(idea),
                analysis_family=(
                    candidate.analysis_family if candidate else idea.analysis_family
                ),
                resolved_analysis_concepts=(
                    list(candidate.resolved_analysis_concepts) if candidate else []
                ),
                prior_art=assessment,
                database_feasibility=feasibility,
                go_no_go=decision,
                go_no_go_reason=decision_reason,
                risks=risks,
                clinical_plausibility_requires_human=True,
                feasibility_tier=tier,
                feasibility_tier_note=tier_note,
                feasibility_source_items=tier_items,
                extended_feasibility=extended_meta,
                feasibility_route=route,
                feasibility_next_action=next_action,
                requires_human_confirmation=True,
            )
        )
    return records


def _format_literature_candidate_topic(idea: LiteratureIdeaCandidate) -> str:
    """Human-readable topic for pairwise and concept-set idea shapes."""
    predictor = str(idea.exposure_or_predictor or "").strip()
    outcome = str(idea.outcome or "").strip()
    population = str(idea.population or "").strip()
    if predictor and outcome:
        return f"{predictor} -> {outcome} in {population}"
    concepts = [str(c).strip() for c in idea.analysis_concepts if str(c).strip()]
    if concepts:
        family = normalize_analysis_family(idea.analysis_family)
        return f"{family}: {', '.join(concepts)} in {population}"
    family = normalize_analysis_family(idea.analysis_family)
    return f"{family} idea in {population}"


def _feasibility_route_and_next_action(
    *,
    decision: GoNoGoDecision,
    candidate: Optional[ExecutableHypothesisCandidate],
    triage: Optional[IdeaMiningCandidateTriageRecord],
    feasibility_tier: Optional[str],
    extended_feasibility: Optional[Mapping[str, Any]],
) -> Tuple[str, str]:
    """Convert coarse go/no-go into an actionable human-screening route."""
    if extended_feasibility:
        case = str(extended_feasibility.get("case") or "")
        if case == "icd_cohort":
            return (
                "icd_cohort_human_confirm",
                "curate and confirm the ICD code set before any analysis run",
            )
        if case == "derived_concept":
            return (
                "derived_concept_human_confirm",
                "build the construct from the listed primitives (human-confirm the "
                "derivation rule), then rerun mapping and joint-feasibility probing",
            )
        if case == "raw_extraction":
            return (
                "raw_extraction_human_confirm",
                "have a coding agent extract the construct from the named raw table "
                "under human review (never auto-execute), then rerun feasibility probing",
            )
        if case == "reextract_current_db":
            return (
                "reextract_current_database",
                "add the dictionary concept to the current export, then rerun mapping and joint-feasibility probing",
            )
        if case == "other_db":
            return (
                "other_database",
                "rerun the idea-mining feasibility probe on the listed database(s)",
            )
        return (
            "extended_feasibility_human_review",
            "review the extended-feasibility metadata before execution",
        )
    if decision == "recommend":
        return (
            "current_export_executable",
            "human prior-art and clinical review, then accept the registry candidate if still justified",
        )
    if feasibility_tier == "executable":
        return (
            "current_export_hold",
            "resolve the hold reason, then rerun prior-art and pair-feasibility checks",
        )
    if feasibility_tier == "T1_reextract":
        return (
            "reextract_or_derive",
            "derive or re-extract the matched source concept, then rerun mapping and joint-feasibility probing",
        )
    if feasibility_tier == "T2_new_concept":
        return (
            "new_concept_human_confirm",
            "author a dictionary concept and extraction protocol before execution",
        )
    if feasibility_tier == "T3_not_in_db":
        return (
            "not_measured_in_database",
            "do not execute in the current database unless new source evidence is found",
        )
    if candidate is None:
        return (
            "not_currently_actionable",
            "review concept mapping; no executable candidate was produced",
        )
    if (
        candidate.resolved_predictor_concept is not None
        and candidate.resolved_outcome_concept is not None
        and not candidate.executable
    ):
        return (
            "needs_outcome_operationalization",
            "define an explicit outcome event or normalized 0/1 endpoint, then rerun feasibility probing",
        )
    if (
        triage is not None
        and triage.coverage_source == "pair_joint_feasibility"
        and triage.executable
    ):
        return (
            "current_export_hold",
            "complete human novelty, differentiation, and clinical-plausibility review",
        )
    if candidate.executable and candidate.resolved_analysis_concepts:
        return (
            "concept_set_human_confirm",
            "confirm the resolved concept set and analysis protocol before execution",
        )
    return (
        "not_currently_actionable",
        "resolve unavailable constructs, derivation requirements, or prior-art screening gaps",
    )


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
    if pair:
        predictor = normalize_concept_name(pair[0])
        outcome = normalize_concept_name(pair[1])
    elif candidate.resolved_predictor_concept or candidate.predictor_label:
        predictor = normalize_concept_name(
            candidate.resolved_predictor_concept or candidate.predictor_label
        )
        outcome = normalize_concept_name(
            candidate.resolved_outcome_concept or candidate.outcome_label
        )
    else:
        # Concept-SET candidate (e.g. subphenotype clustering) has no
        # predictor/outcome pair. Keying it on empty strings collapses every
        # set idea of the same family into one key, so distinct variable sets
        # (e.g. clustering on {lactate, creatinine} vs {pao2, fio2}) are
        # silently deduped, the registry conflates them into one preregistered
        # entry, and the multiple-testing denominator undercounts. Key on the
        # sorted resolved concept set instead so distinct sets stay distinct.
        concepts = candidate.resolved_analysis_concepts or []
        predictor = "set:" + "|".join(
            sorted(normalize_concept_name(c) for c in concepts if c)
        )
        outcome = ""
    return (
        predictor,
        outcome,
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
        probe_kwargs: Dict[str, Any] = dict(
            concepts=[predictor, outcome],
            database=database,
            data_path=data_path,
            cohort=cohort,
            analytic_unit=analytic_unit,
        )
        # Exposure-side answerability only: never request contrast for the
        # outcome (its modal share would leak the event rate). Pass the kwarg
        # only when the (possibly caller-injected) probe accepts it, so probes
        # with a fixed legacy signature keep working — they simply do not emit
        # a contrast value.
        if _probe_accepts_contrast(probe):
            probe_kwargs["contrast_concepts"] = [predictor]
        raw_result = probe(**probe_kwargs)
        value = _lookup_probe_value(raw_result, predictor)
        if value is None:
            warnings.append(
                "S1 feasibility probe returned no record for predictor "
                f"{predictor!r} in pair {pair!r}; pair omitted from S3 ranking."
            )
            continue
        signal = _coerce_probe_feasibility_signal(value)
        signals[pair] = signal
        warnings.extend(_exposure_contrast_warnings(predictor, pair, signal))
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
                predictor_contrast_fraction=signal.predictor_contrast_fraction,
            )
        )
    return signals, records, warnings


# A predictor whose minority share falls below this rule-of-thumb floor offers
# almost no exposure contrast; flagged as a caution, not a hard reject (the
# human gate decides adequacy). A share of exactly 0 is a degenerate exposure.
_MIN_EXPOSURE_CONTRAST = 0.01


def _probe_accepts_contrast(probe: FeasibilityProbe) -> bool:
    """Whether ``probe`` accepts the ``contrast_concepts`` keyword.

    A caller-injected feasibility probe may keep the legacy signature (no
    ``contrast_concepts``); passing the kwarg unconditionally would raise. We
    pass it only for probes that declare the parameter or accept ``**kwargs``.
    """
    try:
        params = inspect.signature(probe).parameters.values()
    except (TypeError, ValueError):
        return False
    for param in params:
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "contrast_concepts":
            return True
    return False


def _exposure_contrast_warnings(
    predictor: str,
    pair: Tuple[str, str],
    signal: HypothesisFeasibilitySignal,
) -> List[str]:
    contrast = signal.predictor_contrast_fraction
    if contrast is None:
        return []
    if contrast <= 0.0:
        return [
            f"Predictor {predictor!r} is single-valued in the cohort (no "
            f"exposure contrast) for pair {pair!r}; an association cannot be "
            "estimated regardless of how complete the data is."
        ]
    if contrast < _MIN_EXPOSURE_CONTRAST:
        return [
            f"Predictor {predictor!r} has very low exposure contrast "
            f"(minority share {contrast:.4f}) for pair {pair!r}; effect "
            "estimates will be imprecise — confirm exposure adequacy at the "
            "human gate."
        ]
    return []


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
    return None


def _coerce_probe_feasibility_signal(value: Any) -> HypothesisFeasibilitySignal:
    if isinstance(value, HypothesisFeasibilitySignal):
        return HypothesisFeasibilitySignal(
            joint_fraction_complete=_bounded_fraction(value.joint_fraction_complete),
            n_joint_complete=value.n_joint_complete,
            denominator_n=value.denominator_n,
            source=value.source,
            note=value.note,
            predictor_contrast_fraction=value.predictor_contrast_fraction,
        )
    if isinstance(value, Mapping):
        joint = value.get("joint_fraction_complete")
        if joint is None:
            raise IdeaMiningError(
                "S1 feasibility values require joint_fraction_complete"
            )
        return HypothesisFeasibilitySignal(
            joint_fraction_complete=_bounded_fraction(joint),
            n_joint_complete=_optional_int(value.get("n_joint_complete")),
            denominator_n=_optional_int(value.get("denominator_n")),
            source=str(value.get("source") or "precomputed"),
            note=str(value["note"]) if value.get("note") is not None else None,
            predictor_contrast_fraction=_optional_bounded_fraction(
                value.get("predictor_contrast_fraction")
                if value.get("predictor_contrast_fraction") is not None
                else value.get("value_contrast_fraction")
            ),
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
        predictor_contrast_fraction=_optional_bounded_fraction(
            getattr(value, "predictor_contrast_fraction", None)
            if getattr(value, "predictor_contrast_fraction", None) is not None
            else getattr(value, "value_contrast_fraction", None)
        ),
    )


def _bounded_fraction(value: Any) -> float:
    fraction = float(value)
    return max(0.0, min(1.0, fraction))


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_bounded_fraction(value: Any) -> Optional[float]:
    if value is None:
        return None
    return _bounded_fraction(value)


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
            for key in [
                item.name,
                item.source_concept or "",
                *item.derived_from_concepts,
            ]:
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
            selection_status = registry.latest_entry(
                registry_candidate_id
            ).selection_status
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
                analysis_family=candidate.analysis_family,
                resolved_analysis_concepts=list(candidate.resolved_analysis_concepts),
                feasibility_pair_key=pair_key,
                feature_derivation_status=candidate.feature_derivation_status,
                feature_derivation_requirements=list(
                    candidate.feature_derivation_requirements
                ),
                feature_derivation_note=candidate.feature_derivation_note,
                executable=candidate.executable,
                non_executable_reasons=list(candidate.non_executable_reasons),
                ranking_candidate_id=(
                    str(ranked.get("candidate_id")) if ranked else None
                ),
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
            canonical = normalize_concept_name(item.source_concept or item.name)
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


def _resolve_concept_exact(term: str, lookup: Mapping[str, str]) -> Optional[str]:
    """Variant-table-only resolution (no inexact token fallback).

    A non-``None`` result means the whole phrase is itself a known concept or
    alias, which is how a genuine unified ratio concept (``"P/F ratio"`` ->
    ``pafi``) is told apart from a derived ratio of two component concepts.
    """
    for variant in _concept_lookup_variants(term):
        if variant in lookup:
            return lookup[variant]
    return None


def _token_specificity(lookup: Mapping[str, str]) -> Dict[str, float]:
    """Weight each signal token by how *discriminating* it is across the lookup.

    A token is weighted ``1 / (number of distinct concepts it appears in)``.
    Tokens that name a single concept (``"urea"``, ``"creatinine"``) keep full
    weight 1.0; tokens shared across many concepts (``"ratio"`` -> pafi, safi,
    nlr...; ``"index"``; ``"acid"``) decay toward zero and can no longer let an
    incidental shared word tie a real clinical signal. This is the general,
    enumeration-free cure for the bug class where one generic token in a query
    (``"urea-to-creatinine ratio"``) matched an unrelated concept (``pafi``).
    """
    concepts_per_token: Dict[str, set] = {}
    for key, canonical in lookup.items():
        for token in _concept_signal_tokens(key):
            concepts_per_token.setdefault(token, set()).add(canonical)
    return {
        token: 1.0 / len(concepts) for token, concepts in concepts_per_token.items()
    }


_MIN_PARTIAL_CONCEPT_MATCH_SPECIFICITY = 0.5
_INTERVENTION_PHRASE_TOKENS = frozenset(
    {
        "administer",
        "administered",
        "administration",
        "bundle",
        "dose",
        "doses",
        "dosing",
        "infusion",
        "initiation",
        "protocol",
        "regimen",
        "strategy",
        "therapy",
        "treatment",
    }
)
_INTERVENTION_CONCEPT_CATEGORIES = frozenset(
    {"intervention", "medication", "medications", "treatment"}
)


def _predictor_category_conflict(
    term: str,
    resolved_key: Optional[str],
    concept_categories: Mapping[str, str],
) -> bool:
    """Reject an administration phrase bound to a measurement concept.

    This is deliberately one-way and conservative.  It does not infer that a
    concept is a treatment; it only prevents a phrase that explicitly says
    dose/initiation/therapy from being reinterpreted as a lab or vital merely
    because both share a noun (for example protein dosing -> total protein).
    Missing category metadata leaves the historical result unchanged.
    """

    if not resolved_key:
        return False
    phrase_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", normalize_concept_name(term))
        if token
    }
    if not phrase_tokens.intersection(_INTERVENTION_PHRASE_TOKENS):
        return False
    category = str(concept_categories.get(resolved_key) or "").strip().lower()
    return bool(category) and category not in _INTERVENTION_CONCEPT_CATEGORIES


def _resolve_embedded_alias(term: str, lookup: Mapping[str, str]) -> Optional[str]:
    """Recover a first-class concept whose MULTI-WORD alias is embedded in a
    noisy phrase (e.g. "urea-to-creatinine ratio cutoff threshold" ->
    bun_creatinine_ratio via the "urea-to-creatinine ratio" alias).

    Only alias variants with >=2 signal tokens are eligible (single tokens are
    too generic to match by containment), and ALL of the alias's tokens must be
    present in the term. The most specific match wins (most alias tokens, then
    highest summed token specificity), so a 3-token "urea creatinine ratio" beats
    any incidental 2-token overlap. Deterministic; returns None when nothing
    multi-word is embedded.
    """
    term_tokens = _concept_signal_tokens(term)
    if len(term_tokens) < 2:
        return None
    specificity = _token_specificity(lookup)
    best: Optional[str] = None
    best_score = (1, 0.0)  # require strictly >1 token to qualify
    for key, canonical in lookup.items():
        key_tokens = _concept_signal_tokens(key)
        if len(key_tokens) < 2 or not key_tokens <= term_tokens:
            continue
        score = (
            len(key_tokens),
            sum(specificity.get(token, 1.0) for token in key_tokens),
        )
        if score > best_score:
            best_score = score
            best = canonical
    return best


def _resolve_concept(term: str, lookup: Mapping[str, str]) -> Optional[str]:
    exact = _resolve_concept_exact(term, lookup)
    if exact is not None:
        return exact
    resolution_term = _strip_specimen_qualifiers(term) or term
    if resolution_term != term:
        exact = _resolve_concept_exact(resolution_term, lookup)
        if exact is not None:
            return exact
    # A derived ratio of distinct components ("urea-to-creatinine ratio") has no
    # unified concept; anchor it to its first real component rather than letting a
    # generic shared token ("ratio") drag the match onto an unrelated concept
    # (pafi). The downstream feature-derivation check then flags it as needing
    # both components. (Fragment resolution recurses one level and terminates: a
    # bare component carries no "ratio" token.)
    components = _ratio_component_concepts(resolution_term, lookup)
    if len(components) >= 2:
        # A noisy concept-set phrase can EMBED a genuine first-class unified
        # concept, e.g. "urea-to-creatinine ratio cutoff threshold" contains
        # "urea-to-creatinine ratio" -> bun_creatinine_ratio. Prefer that real
        # concept over decomposing the ratio into bun+crea (or, worse, returning
        # a spurious leading token like "admission" -> adm). Only fires on terms
        # that already hit the ratio-decomposition path, so non-ratio resolution
        # is unchanged.
        embedded = _resolve_embedded_alias(resolution_term, lookup)
        if embedded is not None:
            return embedded
        return components[0]
    term_tokens = _concept_signal_tokens(resolution_term)
    if not term_tokens:
        return None
    # Pick the MOST SPECIFIC subset match, not the first one encountered: rank by
    # the summed *specificity* (inverse document frequency) of the overlapping
    # tokens, then by the most concise key. Specificity weighting stops a generic
    # shared word (e.g. "ratio", shared by pafi/safi/nlr, or "ventilation" in
    # "ventilation-induced acute kidney injury") from beating the discriminating
    # clinical signal ("acute kidney injury" -> kdigo_aki, or urea+creatinine ->
    # their components). Deterministic: ties keep the first key in insertion order.
    specificity = _token_specificity(lookup)
    best: Optional[str] = None
    best_score = (0.0, 0)
    for key, canonical in lookup.items():
        key_tokens = _concept_signal_tokens(key)
        if not key_tokens:
            continue
        if key_tokens <= term_tokens or term_tokens <= key_tokens:
            overlap = key_tokens & term_tokens
            weight = sum(specificity.get(token, 1.0) for token in overlap)
            score = (weight, -len(key_tokens))
            if score > best_score:
                best_score = score
                best = canonical
    # A one-token overlap in a two-token phrase (specificity == 0.5) is still
    # too weak: ``marker c`` must not silently become ``marker a``.  Exact and
    # embedded multi-word alias matches have already returned above.
    lexical_term_tokens = {
        token
        for token in re.split(
            r"[^a-z0-9]+",
            normalize_concept_name(resolution_term),
        )
        if token
    }
    if (
        len(lexical_term_tokens) > 1
        and best_score[0] <= _MIN_PARTIAL_CONCEPT_MATCH_SPECIFICITY
    ):
        return None
    return best


_SPECIMEN_QUALIFIER_WORDS = frozenset({"serum", "plasma"})


def _strip_specimen_qualifiers(value: str) -> str:
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", normalize_concept_name(value))
        if token
    ]
    if not tokens or not any(token in _SPECIMEN_QUALIFIER_WORDS for token in tokens):
        return ""
    stripped = [token for token in tokens if token not in _SPECIMEN_QUALIFIER_WORDS]
    return " ".join(stripped) if stripped and stripped != tokens else ""


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


# Connector words that join the numerator/denominator of a named clinical ratio.
_RATIO_CONNECTOR_WORDS = frozenset({"to", "over", "per", "vs", "versus", "and", "the"})


def _ratio_component_concepts(term: str, lookup: Mapping[str, str]) -> List[str]:
    """Resolve the components of an ``X-to-Y ratio`` phrase to distinct concepts.

    Returns the ordered, de-duplicated list of canonical concepts the ratio is
    built from, but only when the phrase actually names the word "ratio" and its
    fragments map to two or more *different* concepts. A single unified ratio
    concept (``"P/F ratio"`` -> ``pafi``, whose fragments ``p``/``f`` resolve to
    nothing) yields ``[]`` and is therefore never mistaken for a derived ratio.
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", normalize_concept_name(term)) if t]
    if "ratio" not in tokens:
        return []
    fragments = [
        token
        for token in tokens
        if token != "ratio" and token not in _RATIO_CONNECTOR_WORDS and len(token) > 2
    ]
    components: List[str] = []
    for fragment in fragments:
        resolved = _resolve_concept(fragment, lookup)
        if resolved and resolved not in components:
            components.append(resolved)
    return components if len(components) >= 2 else []


def _feature_derivation_status(
    term: str,
    *,
    resolved_key: Optional[str],
    lookup: Optional[Mapping[str, str]] = None,
) -> Tuple[FeatureDerivationStatus, List[str], Optional[str]]:
    # A two-component clinical ratio (e.g. "urea-to-creatinine ratio") is a
    # derived feature, not a raw concept. A genuine *unified* ratio concept
    # ("P/F ratio" -> pafi) resolves through the exact variant table; a derived
    # ratio does not, and only its fragments resolve. So: when there is no exact
    # concept for the whole phrase but its fragments map to >=2 distinct
    # concepts, flag it as requiring derivation from those components.
    if lookup is not None and _resolve_concept_exact(term, lookup) is None:
        components = _ratio_component_concepts(term, lookup)
        if len(components) >= 2:
            return (
                "requires_derived_feature",
                [
                    f"requires component concepts: {', '.join(components)}",
                    "requires ratio computation",
                ],
                "term is a ratio of distinct concepts; derive it from "
                f"{' / '.join(components)}",
            )
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
