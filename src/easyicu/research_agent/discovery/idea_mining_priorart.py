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

import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from ..concept_availability import normalize_concept_name
from ..literature import CitationRecord
from .idea_mining_pubmed import (
    _GENERIC_CONCEPT_WORDS,
    _clean_literature_phrase,
    _is_specific_differentiator,
    _ordered_unique,
    _pubmed_core_recall_clause,
    _pubmed_or_clause,
    _pubmed_phrase_clause,
    _pubmed_population_recall_clause,
    _prior_art_synonym_phrases,
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

# Family-specific novelty term: what would make a concept-set study "already
# done" is that someone ran THAT kind of analysis on those variables.
_CONCEPT_SET_FAMILY_NOVELTY_TERM = {
    "trajectory_clustering": "(subphenotype OR phenotype OR cluster OR clustering)",
    "descriptive_epidemiology": "(epidemiology OR incidence OR prevalence)",
    "data_quality_audit": '("data quality" OR completeness OR missingness)',
    "measurement_bias_audit": '("measurement bias" OR ascertainment OR "testing frequency" OR "informative measurement")',
    "cohort_definition_sensitivity": '("cohort definition" OR "case definition" OR eligibility OR "ICD definition")',
    "score_policy_sensitivity": '("score component" OR "component missingness" OR "imputation policy" OR "score sensitivity")',
}


def _build_concept_set_prior_art_queries(
    idea: LiteratureIdeaCandidate,
) -> Dict[str, str]:
    from ..planning.analysis_types import normalize_analysis_family

    family = normalize_analysis_family(idea.analysis_family)
    concept_phrases = [
        _clean_literature_phrase(str(c))
        for c in idea.analysis_concepts
        if str(c).strip()
    ]
    concept_clauses = [_pubmed_phrase_clause(p) for p in concept_phrases]
    concept_clauses = [c for c in concept_clauses if c]
    population_recall = _pubmed_population_recall_clause(
        _clean_literature_phrase(idea.population)
    )
    family_term = _CONCEPT_SET_FAMILY_NOVELTY_TERM.get(family, "")

    # broad: family term + a recall-friendly OR of the variables (+ population)
    broad_parts = [family_term] if family_term else []
    if concept_clauses:
        broad_parts.append("(" + " OR ".join(concept_clauses) + ")")
    if population_recall:
        broad_parts.append(population_recall)

    # exact: family term AND each variable (the conjunction is the "same study")
    exact_parts = [family_term] if family_term else []
    exact_parts.extend(concept_clauses)

    return {
        "broad": " AND ".join(part for part in broad_parts if part),
        "exact": " AND ".join(part for part in exact_parts if part),
    }


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

    # Concept-SET ideas (clustering / descriptive / data-quality) have no
    # predictor->outcome phrasing; build their novelty query from the variable
    # set plus a family term so the prior-art screen is not handed an empty query.
    if (
        not idea.exposure_or_predictor.strip()
        and not idea.outcome.strip()
        and any(str(c).strip() for c in idea.analysis_concepts)
    ):
        return _build_concept_set_prior_art_queries(idea)

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

    predictor_exact = _pubmed_or_clause(
        [
            _pubmed_phrase_clause(item)
            for item in _ordered_unique(
                [predictor_phrase, *idea.exposure_literature_aliases]
            )
        ]
    )
    outcome_exact = _pubmed_or_clause(
        [
            _pubmed_phrase_clause(item)
            for item in _ordered_unique(
                [outcome_phrase, *idea.outcome_literature_aliases]
            )
        ]
    )
    exact_parts = [predictor_exact, outcome_exact]
    for item in differentiators:
        exact_parts.append(_pubmed_phrase_clause(item))

    return {
        "broad": " AND ".join(part for part in broad_parts if part),
        "exact": " AND ".join(part for part in exact_parts if part),
    }


# Novelty labels ordered MOST -> LEAST conservative (least -> most novel-looking).
_NOVELTY_CONSERVATISM_ORDER: Tuple[NoveltyLabel, ...] = (
    "already_done",
    "crowded_but_differentiable",
    "sparse",
    "apparently_gap",
)

# A secondary-index broad recall above this count means the field is populated
# there even if PubMed under-indexes it -> the apparent gap is a coverage
# artifact. Mirrors ``_label_prior_art``'s ``sparse_threshold`` semantics.
_CORROBORATION_SPARSE_MAX = 5


def _more_conservative_novelty(a: NoveltyLabel, b: NoveltyLabel) -> NoveltyLabel:
    """Return the LESS-novel (more conservative) of two novelty labels.

    Used as the Phase-3 veto-net merge: the LLM differentiation judge can only
    move a label toward ``already_done``, never toward ``sparse``/``apparently_gap``.
    Unknown labels are treated as least-conservative so they never loosen a count
    verdict.
    """
    order = {label: i for i, label in enumerate(_NOVELTY_CONSERVATISM_ORDER)}
    last = len(_NOVELTY_CONSERVATISM_ORDER)
    return a if order.get(a, last) <= order.get(b, last) else b


# Map an LLM differentiation verdict to the most-novel label it permits. The
# merge with the count label then keeps whichever is MORE conservative, so the
# LLM can tighten a false gap but never invent novelty the counts do not support.
_NOVELTY_JUDGE_VERDICT_CEILING: Dict[str, NoveltyLabel] = {
    "duplicate": "already_done",
    "crowded": "crowded_but_differentiable",
    # "differentiated" imposes no ceiling -> keep the count label as-is.
}


def _apply_novelty_judge(
    count_label: NoveltyLabel,
    *,
    idea: LiteratureIdeaCandidate,
    executable_candidate: Optional[ExecutableHypothesisCandidate],
    query_records: Sequence[Any],
    novelty_judge: Callable[..., Mapping[str, Any]],
) -> Tuple[NoveltyLabel, Optional[str]]:
    """Phase 3: let an LLM read the prior-art hits and tighten the count label.

    Returns ``(label, rationale)``. The judge is best-effort: any error or
    malformed verdict leaves the count label untouched. The judge can only make
    the label MORE conservative (veto-net), never more novel.
    """
    hits = [
        {
            "pmid": hit.pmid,
            "title": getattr(hit, "title", "") or "",
        }
        for record in query_records
        for hit in getattr(record, "top_hits", [])
    ]
    if not hits:
        return count_label, None
    try:
        verdict = novelty_judge(
            idea=idea,
            executable_candidate=executable_candidate,
            hits=hits,
            count_label=count_label,
        )
    except Exception:
        return count_label, None
    if not isinstance(verdict, Mapping):
        return count_label, None
    rationale = str(verdict.get("rationale") or "").strip() or None
    verdict_key = str(verdict.get("verdict") or "").strip().lower()
    if verdict.get("is_duplicate"):
        verdict_key = "duplicate"
    ceiling = _NOVELTY_JUDGE_VERDICT_CEILING.get(verdict_key)
    if ceiling is None:
        return count_label, rationale
    return _more_conservative_novelty(count_label, ceiling), rationale


def assess_prior_art_for_idea(
    idea: LiteratureIdeaCandidate,
    *,
    search_client: Any,
    executable_candidate: Optional[ExecutableHypothesisCandidate] = None,
    searched_at: Optional[str] = None,
    top_n: int = 20,
    novelty_judge: Optional[Callable[..., Mapping[str, Any]]] = None,
    cross_db_targets: Optional[Sequence[str]] = None,
    corroborating_search_client: Optional[Any] = None,
) -> PriorArtAssessment:
    """Run layered prior-art triage for one literature-derived idea.

    ``corroborating_search_client`` (optional) hardens the single strongest
    novelty label. The count screen runs against one index (PubMed); a topic
    studied mainly in preprints / proceedings / non-MEDLINE venues can show a low
    PubMed count that is a *coverage* artifact, not novelty. When the label is
    ``apparently_gap`` and this second client is supplied, the broad recall is
    re-run against it: a non-sparse second-index count DEMOTES the verdict to
    ``crowded_but_differentiable``. Like the LLM judge, it can only ever tighten
    a claim of novelty, never upgrade one -- the count screen stays the veto net.

    ``cross_db_targets`` (optional) enables the cross-database transportability
    novelty axis: when the field is crowded with same-topic prior art but none of
    the retrieved hits (by title/rationale) reference these target databases, a
    "cross-database transportability" differentiator is added as a HUMAN-REVIEW
    trigger (not a novelty claim) -- a study replicated across these harmonized
    public databases is differentiated from predominantly single-database prior
    art. Title-level detection is conservative; the human confirms prior art is
    single-database. Only rescues crowded fields; never embellishes an already
    apparently-novel idea.

    ``novelty_judge`` (Phase 3, optional) is an LLM-backed callable that reads the
    candidate plus the prior-art hit titles and returns a verdict
    (``duplicate`` / ``crowded`` / ``differentiated`` with a ``rationale``). It can
    only make the count-based novelty label MORE conservative -- the substring/
    count screen remains the veto net, so the judge can catch a false gap but can
    never upgrade a crowded field into apparent novelty.
    """

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
    evidence_map_counts, evidence_map_examples = _build_prior_art_evidence_map(
        query_records,
        idea=idea,
        cross_db_targets=cross_db_targets,
    )
    differentiators = _candidate_differentiators(idea)
    cross_db_diff = _cross_db_prior_art_differentiator(
        query_records, direct_hits, cross_db_targets
    )
    if cross_db_diff is not None:
        differentiators = _ordered_unique([*differentiators, cross_db_diff])
    has_specific_differentiator = bool(differentiators)
    same_topic_screen_status = _same_topic_screen_status(query_records)
    construct_is_concrete = not (
        _construct_is_vague(idea.exposure_or_predictor)
        or _construct_is_vague(idea.outcome)
    )
    novelty_label = _label_prior_art(
        broad_count=broad.hit_count,
        exact_count=exact.hit_count,
        direct_same_topic_count=len(screened_direct_hits),
        has_specific_differentiator=has_specific_differentiator,
        construct_is_concrete=construct_is_concrete,
    )
    if direct_hits and not screened_direct_hits:
        novelty_label = "crowded_but_differentiable"
    # If the broad recall screen did not actually run (network/API failure, or a
    # None/empty response), a zero hit count is an artifact, not a gap. Never let
    # a failed screen produce an apparent-novelty verdict: degrade to the
    # conservative "crowded" label and mark the same-topic screen as not run so
    # the go/no-go gate holds for human confirmation instead of recommending.
    prior_art_screen_ran = bool(getattr(broad, "search_ok", True))
    if not prior_art_screen_ran:
        if novelty_label in {"apparently_gap", "sparse"}:
            novelty_label = "crowded_but_differentiable"
        same_topic_screen_status = (
            "prior-art search unavailable, NOT screened; " + same_topic_screen_status
        )
    corroboration_note = ""
    if (
        novelty_label == "apparently_gap"
        and corroborating_search_client is not None
        and prior_art_screen_ran
    ):
        corroboration = _run_prior_art_query(
            corroborating_search_client,
            query_type="broad",
            query=queries["broad"],
            max_results=top_n,
            idea=idea,
        )
        corroboration_ran = bool(getattr(corroboration, "search_ok", True))
        if corroboration_ran and corroboration.hit_count > _CORROBORATION_SPARSE_MAX:
            novelty_label = "crowded_but_differentiable"
            corroboration_note = (
                f" A secondary index returned {corroboration.hit_count} broad "
                "hits, so the apparent PubMed gap is a single-index coverage "
                "artifact; downgraded to crowded_but_differentiable."
            )
            same_topic_screen_status += (
                "; secondary-index corroboration DEMOTED the apparent gap"
            )
        elif corroboration_ran:
            corroboration_note = (
                " A secondary index also returned a sparse broad recall, "
                "corroborating the apparent gap across two indices."
            )
            same_topic_screen_status += (
                "; secondary-index corroboration held the apparent gap"
            )
        else:
            same_topic_screen_status += (
                "; secondary-index corroboration unavailable (single index only)"
            )
    judge_rationale: Optional[str] = None
    if novelty_judge is not None:
        novelty_label, judge_rationale = _apply_novelty_judge(
            novelty_label,
            idea=idea,
            executable_candidate=executable_candidate,
            query_records=query_records,
            novelty_judge=novelty_judge,
        )
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
    if corroboration_note:
        statement += corroboration_note
    if judge_rationale:
        statement += f" LLM differentiation note: {judge_rationale}"
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
        "evidence_map_counts": evidence_map_counts,
        "evidence_map_examples": evidence_map_examples,
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
        evidence_map_counts=evidence_map_counts,
        evidence_map_examples=evidence_map_examples,
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
    novelty_judge: Optional[Callable[..., Mapping[str, Any]]] = None,
    cross_db_targets: Optional[Sequence[str]] = None,
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
            novelty_judge=novelty_judge,
            cross_db_targets=cross_db_targets,
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
            f"evidence map={_format_evidence_map_counts(assessment.evidence_map_counts)}; "
            f"searched {assessment.searched_at}"
        )
        feasibility = _format_feasibility(record.database_feasibility)
        if getattr(record, "feasibility_tier", None):
            tier_line = _format_feasibility_tier(record)
            feasibility = f"{tier_line}\n{feasibility}" if feasibility else tier_line
        if getattr(record, "feasibility_route", None):
            route_line = f"route={record.feasibility_route}"
            if getattr(record, "feasibility_next_action", None):
                route_line += f"; next={record.feasibility_next_action}"
            feasibility = f"{route_line}\n{feasibility}" if feasibility else route_line
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
    from ..planning.analysis_types import normalize_analysis_family

    family = normalize_analysis_family(idea.analysis_family)
    family_phrase: Optional[str] = None
    # Cross-database replication is evaluated by the dedicated
    # ``cross_db_targets`` screen after same-topic studies have been retrieved.
    # Putting the internal family key (or a quoted display label) into the exact
    # query would hide otherwise relevant single-database studies and create a
    # false appearance of novelty.  Retrieve the same predictor/outcome topic
    # first; then ask whether those studies already cover multiple databases.
    if family not in {"association_study", "cross_database_replication"}:
        # Preserve a literature-facing family phrase (including concise aliases
        # such as ``trajectory``) rather than emitting an internal registry key
        # or a UI display label containing slashes.
        family_phrase = str(idea.analysis_family).replace("_", " ").strip()
    raw = [
        idea.time_window_hint,
        idea.aggregation_hint,
        family_phrase,
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
    try:
        raw = _call_prior_art_search(
            search_client,
            query,
            max_results=max_results,
            idea=idea,
        )
    except IdeaMiningError:
        # No usable client at all — re-raise; this is a wiring error, not a
        # transient search failure.
        raise
    except Exception:
        # The search client failed (network/API error). Record a screen that
        # did NOT run so novelty degrades conservatively instead of reading an
        # empty result as a gap.
        return PriorArtQueryRecord(
            query_type=query_type,
            query=query,
            hit_count=0,
            pmids=[],
            top_hits=[],
            search_ok=False,
        )
    # A None response is the swallowed-error shape some clients return; treat it
    # as a failed screen, not a genuine zero-hit result.
    record = _coerce_prior_art_query_record(
        raw,
        query_type=query_type,
        query=query,
    )
    return record.model_copy(
        update={
            "search_ok": raw is not None and record.search_ok,
            "top_hits": [
                _classify_direct_same_topic_hit(hit, idea) for hit in record.top_hits
            ],
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
        response_markers = {
            "hit_count",
            "count",
            "total",
            "top_hits",
            "hits",
            "records",
            "citations",
            "pmids",
            "search_ok",
        }
        search_ok = bool(
            raw.get(
                "search_ok",
                bool(response_markers.intersection(raw))
                and not bool(raw.get("search_error")),
            )
        )
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
            search_ok=search_ok,
        )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        hits = [_coerce_prior_art_hit(item) for item in raw]
        # A bare sequence carries a page, not the true total hit count.
        # Reporting the page size as hit_count would let a crowded field
        # (hundreds of hits) look sparse (<= max_results), so fail closed:
        # keep the hits for evidence but mark the screen unusable.
        return PriorArtQueryRecord(
            query_type=query_type,
            query=query,
            hit_count=len(hits),
            pmids=[hit.pmid for hit in hits],
            top_hits=hits,
            search_ok=False,
        )
    return PriorArtQueryRecord(
        query_type=query_type,
        query=query,
        hit_count=0,
        pmids=[],
        top_hits=[],
        search_ok=False,
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
    predictors = [
        normalize_concept_name(item)
        for item in [idea.exposure_or_predictor, *idea.exposure_literature_aliases]
        if str(item).strip()
    ]
    outcomes = [
        normalize_concept_name(item)
        for item in [idea.outcome, *idea.outcome_literature_aliases]
        if str(item).strip()
    ]
    differentiators = [
        normalize_concept_name(item) for item in _candidate_differentiators(idea)
    ]
    has_predictor = any(item and item in text for item in predictors)
    has_outcome = any(item and item in text for item in outcomes)
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


_EVIDENCE_MAP_BUCKETS: Tuple[str, ...] = (
    "direct_same_topic",
    "same_topic_cross_database_or_external",
    "same_exposure_different_outcome",
    "same_outcome_different_exposure",
    "adjacent_icu_background",
    "unclear",
)


def _format_evidence_map_counts(counts: Mapping[str, int]) -> str:
    rendered = [
        f"{bucket}={int(counts.get(bucket, 0))}"
        for bucket in _EVIDENCE_MAP_BUCKETS
        if int(counts.get(bucket, 0)) > 0
    ]
    return ", ".join(rendered) if rendered else "none"


def _build_prior_art_evidence_map(
    records: Sequence[PriorArtQueryRecord],
    *,
    idea: LiteratureIdeaCandidate,
    cross_db_targets: Optional[Sequence[str]] = None,
    max_examples_per_bucket: int = 3,
) -> Tuple[Dict[str, int], Dict[str, List[Dict[str, str]]]]:
    counts: Dict[str, int] = {bucket: 0 for bucket in _EVIDENCE_MAP_BUCKETS}
    examples: Dict[str, List[Dict[str, str]]] = {
        bucket: [] for bucket in _EVIDENCE_MAP_BUCKETS
    }
    for record in records:
        for hit in record.top_hits:
            bucket = _evidence_map_bucket(
                hit,
                idea=idea,
                cross_db_targets=cross_db_targets,
            )
            counts[bucket] = counts.get(bucket, 0) + 1
            if len(examples.setdefault(bucket, [])) >= max_examples_per_bucket:
                continue
            examples[bucket].append(
                {
                    "pmid": hit.pmid,
                    "title": hit.title,
                    "query_type": record.query_type,
                    "direct_same_topic": str(bool(hit.direct_same_topic)).lower(),
                }
            )
    return (
        {key: value for key, value in counts.items() if value > 0},
        {key: value for key, value in examples.items() if value},
    )


def _evidence_map_bucket(
    hit: PriorArtSearchHit,
    *,
    idea: LiteratureIdeaCandidate,
    cross_db_targets: Optional[Sequence[str]] = None,
) -> str:
    raw_text = " ".join(
        [
            hit.title,
            hit.relevance or "",
            hit.direct_same_topic_rationale or "",
        ]
    )
    text = normalize_concept_name(raw_text)
    raw_lower = raw_text.lower()
    predictor = idea.exposure_core_concept or idea.exposure_or_predictor
    outcome = idea.outcome_core_concept or idea.outcome
    has_predictor = _evidence_term_mentioned(
        predictor, normalized_text=text, raw_text=raw_lower
    )
    has_outcome = _evidence_term_mentioned(
        outcome, normalized_text=text, raw_text=raw_lower
    )
    if hit.direct_same_topic or (has_predictor and has_outcome):
        if _hit_mentions_cross_database_or_external(hit, cross_db_targets):
            return "same_topic_cross_database_or_external"
        return "direct_same_topic"
    if has_predictor and not has_outcome:
        return "same_exposure_different_outcome"
    if has_outcome and not has_predictor:
        return "same_outcome_different_exposure"
    _adjacent_tokens = (
        "icu",
        "critical illness",
        "critically ill",
        "intensive care",
        "intensive care unit",
    )
    if any(
        token in text or normalize_concept_name(token) in text
        for token in _adjacent_tokens
    ):
        return "adjacent_icu_background"
    return "unclear"


def _evidence_term_mentioned(
    term: str,
    *,
    normalized_text: str,
    raw_text: str,
) -> bool:
    phrases = [str(term or ""), *_prior_art_synonym_phrases(str(term or ""))]
    for phrase in phrases:
        clean = _clean_literature_phrase(phrase).lower()
        if clean and clean in raw_text:
            return True
        normalized = normalize_concept_name(phrase)
        if normalized and normalized in normalized_text:
            return True
    return False


def _hit_mentions_cross_database_or_external(
    hit: PriorArtSearchHit,
    cross_db_targets: Optional[Sequence[str]],
) -> bool:
    text = " ".join(
        [
            hit.title,
            hit.relevance or "",
            hit.direct_same_topic_rationale or "",
        ]
    ).lower()
    aliases: List[str] = list(_MULTI_DB_TERMS)
    for db in cross_db_targets or ():
        aliases.extend(_TARGET_DB_ALIASES.get(str(db).lower(), (str(db).lower(),)))
    return any(alias in text for alias in aliases) or bool(
        re.search(r"\b(?:multiple|\d+)\s+(?:[a-z-]+\s+){0,3}hospitals\b", text)
    )


# Decorator / method-shell tokens that carry no queryable clinical construct.
# A predictor like "robust multiparametric clinical scores" or an outcome like
# "marker" reduces to nothing substantive once these and the generic concept
# words are removed; a low hit count for such a phrase reflects an unqueryable
# construct, NOT genuine novelty. Case-neutral: no disease/score/database here.
_VAGUE_CONSTRUCT_TOKENS = frozenset(
    {
        "robust",
        "multiparametric",
        "multiparameter",
        "novel",
        "new",
        "advanced",
        "comprehensive",
        "integrated",
        "optimal",
        "appropriate",
        "individualized",
        "individualised",
        "personalized",
        "personalised",
        "tailored",
        "dynamic",
        "complex",
        "multimodal",
        "multidimensional",
        "improved",
        "enhanced",
        "emerging",
        "innovative",
        "promising",
        "potential",
        "various",
        "different",
        "multiple",
        "several",
        "selected",
        "specific",
        "general",
        "standardized",
        "standardised",
        "strategy",
        "strategies",
        "approach",
        "approaches",
        "tool",
        "tools",
        "algorithm",
        "algorithms",
        "technique",
        "techniques",
        "method",
        "methods",
        "framework",
        "frameworks",
        "scheme",
        "score",
        "scores",
        "marker",
        "markers",
        "biomarker",
        "biomarkers",
        "factor",
        "factors",
        "predictor",
        "predictors",
        "parameter",
        "parameters",
        "variable",
        "variables",
        "index",
        "indices",
        "pattern",
        "patterns",
        "profile",
        "profiles",
        "signature",
        "signatures",
    }
)


def _construct_is_vague(phrase: str) -> bool:
    """Whether a construct has no substantive clinical noun to query on.

    After removing generic concept words and decorator/method-shell tokens, a
    construct with nothing substantive left (e.g. "marker", "robust
    multiparametric clinical scores") cannot be reliably queried, so a low hit
    count for it is an artifact, not evidence of novelty.
    """
    text = _clean_literature_phrase(phrase).lower()
    if not text:
        return True
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]
    substantive = [
        tok
        for tok in tokens
        if len(tok) > 2
        and tok not in _GENERIC_CONCEPT_WORDS
        and tok not in _VAGUE_CONSTRUCT_TOKENS
    ]
    return not substantive


def _label_prior_art(
    *,
    broad_count: int,
    exact_count: int,
    direct_same_topic_count: int,
    has_specific_differentiator: bool = True,
    construct_is_concrete: bool = True,
    sparse_threshold: int = 5,
    crowded_broad_threshold: int = 40,
) -> NoveltyLabel:
    if direct_same_topic_count > 0:
        return "already_done"
    # A large broad-recall count means the topic area is well populated, so the
    # candidate cannot be a sparse gap even when the over-specific exact phrase
    # returns nothing. The exact query is built from the literature's literal,
    # often idiosyncratic wording (e.g. an odd outcome phrase), so exact_count is
    # ~always 0 and must NOT by itself drive a "sparse"/"gap" label; otherwise a
    # heavily studied pairing (hundreds of broad hits) is mislabelled novel. A
    # novelty screen should err toward "someone likely did this" — a novelty
    # claim always needs human confirmation, never a zero-exact-count artifact.
    if broad_count >= crowded_broad_threshold:
        return "crowded_but_differentiable"
    # A construct too vague to query reliably (a decorator/method shell with no
    # substantive clinical noun) yields a low count by artifact, not novelty; we
    # cannot assert a sparse gap about it. Same conservative direction as above.
    if not construct_is_concrete:
        return "crowded_but_differentiable"
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


# Title/rationale tokens that signal a hit already used one of the target public
# databases (so the cross-DB transportability angle is NOT a differentiator).
_TARGET_DB_ALIASES = {
    "miiv": ("mimic", "mimic-iv", "mimic iv", "mimiciv"),
    "mimic": ("mimic", "mimic-iii", "mimic iii", "mimiciii"),
    "eicu": ("eicu", "eicu-crd", "philips eicu"),
    "aumc": ("amsterdam", "amsterdamumc", "amsterdamumcdb"),
    "hirid": ("hirid",),
    "sic": ("sicdb", "salzburg intensive care"),
}
_MULTI_DB_TERMS = (
    "multi-database",
    "multidatabase",
    "multiple databases",
    "external validation",
    "externally validated",
    "validation cohort",
    "transportability",
    "cross-database",
    "multi-cohort",
    "multiple cohorts",
    "multicenter database",
    "multicenter",
    "multi-center",
    "multiple centers",
    "multiple hospitals",
    "two populations",
    "both populations",
)


def _cross_db_prior_art_differentiator(
    query_records: Sequence[PriorArtQueryRecord],
    direct_hits: Sequence[PriorArtSearchHit],
    cross_db_targets: Optional[Sequence[str]],
) -> Optional[str]:
    """Cross-database transportability differentiator (human-review trigger).

    Returns a differentiator string only when (a) the axis is enabled, (b) the
    field is crowded (there is same-topic prior art to differentiate from), and
    (c) no retrieved hit's title/abstract/rationale references the target
    databases or multicenter/multi-database/external-validation work. The
    differentiator remains a human-confirm trigger, not a novelty claim.
    """
    if not cross_db_targets or not direct_hits:
        return None
    del query_records
    if any(
        _hit_mentions_cross_database_or_external(hit, cross_db_targets)
        for hit in direct_hits
    ):
        return None  # prior art already uses these DBs / is multi-DB
    n = len(list(cross_db_targets))
    return (
        f"cross-database transportability across {n} harmonized public ICU "
        f"databases (no retrieved prior art references these databases by title; "
        f"human must confirm prior art is predominantly single-database)"
    )


# Problem 2 -- data-sufficiency floor. Co-availability is necessary but not
# sufficient: a candidate whose joint-complete analytic units are very few (or a
# tiny fraction of the cohort) is underpowered/selection-biased even though the
# columns co-exist, so it must not be promoted to a clean ``recommend``. These
# are deliberately conservative floors; a candidate that fails them is held for
# human adequacy review, never auto-rejected.
_MIN_JOINT_COMPLETE_FOR_RECOMMEND = 100
_MIN_JOINT_FRACTION_FOR_RECOMMEND = 0.02


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
        # Distinguish a genuine database limit from an analyst-side gap. When
        # BOTH the predictor and the outcome resolve to concepts that exist in
        # the available catalogue, the data is present -- the candidate is
        # non-executable only because the outcome event still has to be
        # operationalized (its determinability was never declared) or otherwise
        # defined by a human. That is a "hold" (doable, needs human definition),
        # not "the database cannot do it". ``db-cannot-do`` is reserved for a
        # genuinely absent predictor or outcome concept. This does NOT fabricate
        # feasibility: the candidate stays non-executable and unranked, and no
        # joint coverage is claimed -- it is merely surfaced honestly instead of
        # being buried as a database failure.
        if (
            candidate.resolved_predictor_concept is not None
            and candidate.resolved_outcome_concept is not None
        ):
            return (
                "hold",
                "concepts resolve to available data; outcome event needs "
                "operationalization before feasibility probing",
            )
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
    # Problem 2: data-sufficiency / power floor. Presence of the columns is not
    # the same as an analyzable, representative sample -- demote a thin candidate
    # to a human-confirm hold instead of a clean recommend.
    if (
        triage.n_joint_complete is not None
        and triage.n_joint_complete < _MIN_JOINT_COMPLETE_FOR_RECOMMEND
    ):
        return (
            "hold",
            f"joint-complete analytic units below the power floor "
            f"({triage.n_joint_complete} < {_MIN_JOINT_COMPLETE_FOR_RECOMMEND}); "
            "confirm sample adequacy and representativeness before execution",
        )
    if (
        triage.n_joint_complete is not None
        and triage.denominator_n
        and (triage.n_joint_complete / triage.denominator_n)
        < _MIN_JOINT_FRACTION_FOR_RECOMMEND
    ):
        frac = triage.n_joint_complete / triage.denominator_n
        return (
            "hold",
            f"joint-complete coverage is a small fraction of the cohort "
            f"({frac:.1%} < {_MIN_JOINT_FRACTION_FOR_RECOMMEND:.0%}); confirm "
            "the analyzable subset is not selection-biased before execution",
        )
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
