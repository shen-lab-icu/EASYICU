"""Prior-art assessment and discovery-report rendering for idea mining.

Physical split from ``idea_mining.py`` (P1-3, 2026-06-10), **zero behavior
change**. This leaf module hosts the literature prior-art screening helpers
(query construction, PubMed-hit coercion, same-topic classification, novelty
labelling) and the Markdown discovery-report / go-no-go renderers.

It imports only downward leaves (schema, pubmed helpers, literature,
concept_availability) and must never import ``idea_mining`` — the parent
re-exports every name defined here for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from .concept_availability import normalize_concept_name
from .literature import CitationRecord
from .idea_mining_pubmed import (
    _clean_literature_phrase,
    _is_specific_differentiator,
    _ordered_unique,
    _pubmed_core_recall_clause,
    _pubmed_phrase_clause,
    _pubmed_population_recall_clause,
)
from .idea_mining_schema import (
    DiscoveryCandidateRecord,
    ExecutableHypothesisCandidate,
    GoNoGoDecision,
    IDEA_NOVELTY_SNAPSHOT_SCHEMA_VERSION,
    IdeaMiningCandidateTriageRecord,
    IdeaMiningError,
    LiteratureIdeaCandidate,
    NoveltyLabel,
    PriorArtAssessment,
    PriorArtQueryRecord,
    PriorArtSearchHit,
    _canonical_json,
    _sha256_text,
    _utc_now_iso,
)


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
    snapshot_id = (
        f"novelty-snapshot/sha256:{_sha256_text(_canonical_json(payload))[:16]}"
    )
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
        candidate.literature_idea_id: candidate for candidate in executable_candidates
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
        if getattr(record, "feasibility_tier", None):
            tier_line = _format_feasibility_tier(record)
            feasibility = f"{tier_line}\n{feasibility}" if feasibility else tier_line
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


def _candidate_differentiators(idea: LiteratureIdeaCandidate) -> List[str]:
    raw = [
        idea.time_window_hint,
        idea.aggregation_hint,
        idea.analysis_family if idea.analysis_family != "association" else None,
    ]
    out: List[str] = []
    for item in raw:
        text = _clean_literature_phrase(str(item or ""))
        if (
            text
            and normalize_concept_name(text)
            not in {
                normalize_concept_name(idea.exposure_or_predictor),
                normalize_concept_name(idea.outcome),
            }
            and _is_specific_differentiator(text)
        ):
            out.append(text)
    return _ordered_unique(out)


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
                _classify_direct_same_topic_hit(hit, idea) for hit in record.top_hits
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
    raise IdeaMiningError(
        "prior_art_search_client must provide search() or search_prior_art()"
    )


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
            else (
                raw.get("count")
                if raw.get("count") is not None
                else raw.get("total") if raw.get("total") is not None else len(hits)
            )
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
        normalize_concept_name(item) for item in _candidate_differentiators(idea)
    ]
    has_predictor = predictor and predictor in text
    has_outcome = outcome and outcome in text
    has_differentiator = not differentiators or any(
        item in text for item in differentiators
    )
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
    if (
        payload.get("n_joint_complete") is not None
        and payload.get("denominator_n") is not None
    ):
        parts.append(
            f"joint n={payload['n_joint_complete']}/{payload['denominator_n']}"
        )
    if payload.get("feasibility_note"):
        parts.append(str(payload["feasibility_note"]))
    return "; ".join(parts) if parts else "not available"


_TIER_LABELS = {
    "executable": "executable",
    "T1_reextract": "T1 re-extract/derive",
    "T2_new_concept": "T2 new concept authorable",
    "T3_not_in_db": "T3 not in this database",
}


def _format_feasibility_tier(record: DiscoveryCandidateRecord) -> str:
    tier = record.feasibility_tier
    label = _TIER_LABELS.get(tier, tier or "")
    note = (record.feasibility_tier_note or "").strip()
    line = f"**{label}**"
    if note:
        line += f" — {note}"
    items = list(record.feasibility_source_items or [])[:2]
    if items:
        rendered = "; ".join(
            f"itemid {it.get('itemid')} '{it.get('label')}' ({it.get('table')})"
            for it in items
        )
        line += f" [source: {rendered}]"
    return line


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
        if candidate.outcome_determinability_status == "organ_support_intervention":
            # The endpoint is an organ-support THERAPY (RRT / mechanical
            # ventilation / ECMO) used as an outcome. It is determinable, but it
            # reflects a treatment DECISION, so it must not be read as a clean
            # physiological outcome: confounding by indication and unit/clinician
            # treatment thresholds drive who receives it.
            risks.append(
                "outcome is an organ-support intervention (treatment-decision "
                "endpoint): interpret with confounding-by-indication caution, not "
                "as a physiological outcome"
            )
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
        for item in [
            citation.venue,
            citation.year,
            f"PMID:{citation.pmid}" if citation.pmid else None,
        ]
        if item
    )
    if meta:
        parts.append(f"({meta})")
    return " ".join(parts)
