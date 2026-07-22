"""Standard Idea Mining route for data-first candidate generation.

The leaf generator in :mod:`idea_mining_data_first` finds concept pairs that
are genuinely resolvable across the harmonized ICU databases.  This module
adapts those pairs into the existing Idea Mining pipeline; it does *not* create
an alternative ledger or a shortcut around novelty, feasibility, or human
confirmation.

The route is deliberately provider-free for candidate generation.  The only
network dependency is the caller-supplied prior-art search client.  Every
candidate is backed by a frozen data-profile excerpt containing the prepared
cohort SHA, then passes through the normal concept mapping, real-data
feasibility probe, PubMed assessment, registry, discovery ledger, and handoff
contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..concept_availability import (
    default_public_databases,
    hypothesis_cross_database_feasibility,
)
from ..literature import CitationRecord
from ..providers.protocol import LLMMessage
from ..schema import ConceptDescriptor
from .idea_mining import freeze_source_snapshot, run_idea_mining_dry_run
from .idea_mining_data_first import DataFirstCandidate, generate_data_first_candidates
from .idea_mining_priorart import (
    _call_prior_art_search,
    _coerce_prior_art_query_record,
)
from .idea_mining_pubmed import (
    _ordered_unique,
    _pubmed_or_clause,
    _pubmed_phrase_clause,
)
from .idea_mining_schema import (
    IdeaMiningDryRunResult,
    LiteratureIdeaCandidate,
    OutcomeDeterminability,
    PriorArtQueryRecord,
    SourceMaterial,
)

DATA_FIRST_ROUTE_SCHEMA_VERSION = "easyicu.data_first_discovery_route/2"
DATA_FIRST_SHORTLIST_SCHEMA_VERSION = "easyicu.data_first_review_shortlist/5"

_MIN_EXTERNAL_VALIDATION_COMPLETENESS = 0.90
_MIN_MEASUREMENT_AUDIT_COMPLETENESS = 0.20
_MAX_MEASUREMENT_AUDIT_COMPLETENESS = 0.70
_MAX_EXACT_HITS_FOR_EXTERNAL_VALIDATION_REVIEW = 25
_MAX_EXTERNAL_VALIDATION_REVIEW_CANDIDATES = 3


class _ProviderCallForbidden:
    """Sentinel LLM proving the deterministic route never calls a provider."""

    name = "data-first-provider-forbidden"
    __easyicu_mock_client__ = True

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
    ) -> str:
        del messages, max_tokens, temperature, seed
        raise RuntimeError("data-first discovery must not call an LLM provider")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _review_candidate_id(*, route: str, origin_id: str, topic: str) -> str:
    """Return a stable identity for a bounded human-review question.

    A measurement/source-status audit is a different scientific question from
    the association candidate that revealed the incomplete construct.  It must
    not inherit the association candidate's identity as though only the label
    had changed.
    """

    digest = hashlib.sha256(
        f"{route}\0{origin_id}\0{topic}".encode("utf-8")
    ).hexdigest()[:16]
    return f"reviewidea_{digest}"


def _measurement_audit_query(idea: LiteratureIdeaCandidate) -> str:
    predictor = _pubmed_or_clause(
        [
            _pubmed_phrase_clause(item)
            for item in _ordered_unique(
                [idea.exposure_or_predictor, *idea.exposure_literature_aliases]
            )
        ]
    )
    population = _pubmed_or_clause(
        [
            _pubmed_phrase_clause("intensive care"),
            "ICU[Title/Abstract]",
            _pubmed_phrase_clause("critically ill"),
        ]
    )
    audit = _pubmed_or_clause(
        [
            "missingness[Title/Abstract]",
            _pubmed_phrase_clause("measurement availability"),
            _pubmed_phrase_clause("measurement bias"),
            _pubmed_phrase_clause("data quality"),
            _pubmed_phrase_clause("source status"),
        ]
    )
    return " AND ".join([predictor, population, audit])


def _has_external_validation_differentiation(prior_art: Any) -> bool:
    """Reject exact-term gaps that only substitute a near-equivalent construct."""

    adjacent = int(prior_art.evidence_map_counts.get("adjacent_icu_background", 0))
    return adjacent == 0 or bool(prior_art.has_specific_differentiator)


def _screen_measurement_audit_prior_art(
    *,
    idea: LiteratureIdeaCandidate,
    search_client: Any,
    max_results: int,
) -> PriorArtQueryRecord:
    """Run a route-specific PubMed screen without reusing association counts."""

    query = _measurement_audit_query(idea)
    try:
        raw = _call_prior_art_search(
            search_client,
            query,
            max_results=max_results,
            idea=None,
        )
    except Exception:
        raw = None
    record = _coerce_prior_art_query_record(
        raw,
        query_type="exact",
        query=query,
    )
    return record.model_copy(update={"search_ok": raw is not None and record.search_ok})


def _candidate_evidence_text(
    candidate: DataFirstCandidate,
    *,
    data_sha256: str,
) -> str:
    databases = ", ".join(candidate.harmonized_databases)
    return (
        "EasyICU data-first profile: predictor "
        f"'{candidate.predictor}' and outcome '{candidate.outcome}' are "
        f"dictionary-resolvable with full availability in "
        f"{candidate.harmonized_db_count}/{candidate.total_databases} harmonized "
        f"public ICU databases ({databases}). The prepared cohort used for "
        f"joint-feasibility probing has SHA-256 {data_sha256}. This is a "
        "measurability signal, not a novelty claim."
    )


def _source_materials_and_ideas(
    candidates: Sequence[DataFirstCandidate],
    *,
    data_path: Path,
    data_sha256: str,
    literature_terms: Mapping[str, str],
    literature_aliases: Mapping[str, Sequence[str]],
    analysis_family: str,
) -> tuple[list[SourceMaterial], list[LiteratureIdeaCandidate]]:
    materials: list[SourceMaterial] = []
    evidence_by_key: dict[str, str] = {}
    for candidate in candidates:
        key_digest = hashlib.sha256(
            (
                f"{candidate.predictor}\0{candidate.outcome}\0{data_sha256}\0"
                + "\0".join(candidate.harmonized_databases)
            ).encode("utf-8")
        ).hexdigest()[:16]
        citation_key = f"easyicu_data_profile_{key_digest}"
        evidence_text = _candidate_evidence_text(
            candidate,
            data_sha256=data_sha256,
        )
        evidence_by_key[citation_key] = evidence_text
        materials.append(
            SourceMaterial(
                citation=CitationRecord(
                    key=citation_key,
                    title=(
                        "EasyICU harmonized data profile: "
                        f"{candidate.predictor} -> {candidate.outcome}"
                    ),
                    year="2026",
                    venue="EasyICU host-owned data profile",
                    relevance=(
                        "Frozen cross-database measurability evidence; not a "
                        "literature novelty source."
                    ),
                ),
                source_adapter_level="user_supplied_excerpt",
                locator=str(data_path),
                source_text=evidence_text,
                discovery_route="data_first",
                source_text_role="data_profile_evidence",
            )
        )

    snapshot = freeze_source_snapshot(materials)
    ideas: list[LiteratureIdeaCandidate] = []
    for candidate, material in zip(candidates, materials):
        evidence_text = evidence_by_key[material.citation.key]
        predictor_phrase = str(
            literature_terms.get(candidate.predictor) or candidate.predictor
        ).strip()
        outcome_phrase = str(
            literature_terms.get(candidate.outcome) or candidate.outcome
        ).strip()
        ideas.append(
            LiteratureIdeaCandidate(
                source_snapshot_id=snapshot.source_snapshot_id,
                citation_key=material.citation.key,
                source_adapter_level=material.source_adapter_level,
                population="adult ICU stays in harmonized public ICU databases",
                exposure_or_predictor=predictor_phrase,
                outcome=outcome_phrase,
                rationale=(
                    candidate.differentiator_note
                    + " The association itself remains unproven until prior-art "
                    "and clinical review are complete."
                ),
                source_quote=evidence_text,
                analysis_family=analysis_family,
                exposure_core_concept=candidate.predictor,
                outcome_core_concept=candidate.outcome,
                exposure_literature_aliases=list(
                    literature_aliases.get(candidate.predictor, ())
                ),
                outcome_literature_aliases=list(
                    literature_aliases.get(candidate.outcome, ())
                ),
            )
        )
    return materials, ideas


def _review_shortlist(
    result: IdeaMiningDryRunResult,
    *,
    prior_art_search_client: Any,
    prior_art_top_n: int,
) -> list[dict[str, Any]]:
    """Select bounded, differentiated human-review routes from standard rows.

    The shortlist is deliberately not a second novelty or acceptance gate.  It
    only decides where scarce human review effort should go after every idea has
    passed the standard mapping, real-data, PubMed, registry, and discovery
    ledger.  Route selection is semantic and name-neutral:

    * high-completeness pairs with few exact same-topic hits are candidates for
      cross-database external validation;
    * measurable but substantially incomplete pairs are candidates for a
      cross-database measurement/missingness audit, not an association claim;
      saturation of association papers does not answer that different audit.

    Both routes remain ``requires_human_confirmation`` and explicitly require a
    temporal protocol before any outcome analysis.
    """

    external_validation: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    measurement_audit_by_predictor: dict[
        str, tuple[tuple[Any, ...], dict[str, Any]]
    ] = {}
    idea_by_id = {idea.literature_idea_id: idea for idea in result.literature_ideas}
    for record in result.discovery_records:
        if record.executable_candidate_id is None or record.go_no_go == "db-cannot-do":
            continue
        feasibility = record.database_feasibility
        numerator = feasibility.get("n_joint_complete")
        denominator = feasibility.get("denominator_n")
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            continue
        if denominator <= 0 or numerator < 0 or numerator > denominator:
            continue
        completeness = numerator / denominator
        exact_records = [
            query
            for query in record.prior_art.query_records
            if query.query_type == "exact"
        ]
        if not exact_records or not all(query.search_ok for query in exact_records):
            continue
        exact_hits = sum(query.hit_count for query in exact_records)
        base = {
            "origin_literature_idea_id": record.literature_idea_id,
            "origin_executable_candidate_id": record.executable_candidate_id,
            "origin_candidate_topic": record.candidate_topic,
            "go_no_go": record.go_no_go,
            "exact_same_topic_hit_count": exact_hits,
            "joint_complete_n": numerator,
            "denominator_n": denominator,
            "joint_completeness": round(completeness, 6),
            "requires_human_confirmation": True,
            "paper_authorized": False,
            "resolved_predictor_concept": record.prior_art.feasibility_pair_key[0],
            "resolved_outcome_concept": record.prior_art.feasibility_pair_key[1],
        }
        cross_db_prior_art = int(
            record.prior_art.evidence_map_counts.get(
                "same_topic_cross_database_or_external", 0
            )
        )
        if (
            completeness >= _MIN_EXTERNAL_VALIDATION_COMPLETENESS
            and cross_db_prior_art == 0
            and exact_hits <= _MAX_EXACT_HITS_FOR_EXTERNAL_VALIDATION_REVIEW
            and _has_external_validation_differentiation(record.prior_art)
        ):
            route = "cross_database_external_validation"
            payload = {
                **base,
                "review_candidate_id": _review_candidate_id(
                    route=route,
                    origin_id=record.executable_candidate_id,
                    topic=record.candidate_topic,
                ),
                "candidate_topic": record.candidate_topic,
                "review_route": route,
                "selection_reason": (
                    "high prepared-cohort completeness, <=25 exact same-topic "
                    "PubMed hits, and no retrieved title classified as comparable "
                    "cross-database/external-validation work; exact-term gaps with "
                    "only an undifferentiated adjacent construct are excluded"
                ),
                "required_next_action": (
                    "human-review the exact hits and licensed/non-PubMed literature; "
                    "then define a pre-outcome predictor ascertainment window and "
                    "database-specific transportability estimand"
                ),
            }
            external_validation.append(
                ((exact_hits, -completeness, record.candidate_topic), payload)
            )
        elif (
            _MIN_MEASUREMENT_AUDIT_COMPLETENESS
            <= completeness
            < _MAX_MEASUREMENT_AUDIT_COMPLETENESS
        ):
            idea = idea_by_id.get(record.literature_idea_id)
            if idea is None:
                continue
            predictor_key = record.prior_art.feasibility_pair_key[0]
            existing = measurement_audit_by_predictor.get(predictor_key)
            if existing is not None:
                payload = existing[1]
                for field, value in (
                    ("origin_literature_idea_ids", record.literature_idea_id),
                    (
                        "origin_executable_candidate_ids",
                        record.executable_candidate_id,
                    ),
                    ("origin_candidate_topics", record.candidate_topic),
                    (
                        "association_outcome_contexts",
                        record.prior_art.outcome_literature_phrase,
                    ),
                ):
                    values = payload[field]
                    if value not in values:
                        values.append(value)
                        values.sort()
                continue
            audit_prior_art = _screen_measurement_audit_prior_art(
                idea=idea,
                search_client=prior_art_search_client,
                max_results=prior_art_top_n,
            )
            if not audit_prior_art.search_ok:
                continue
            route = "cross_database_measurement_bias_audit"
            predictor = record.prior_art.predictor_literature_phrase
            audit_topic = (
                "cross-database measurement/source-status audit of " + predictor
            )
            audit_base = {
                key: value
                for key, value in base.items()
                if not key.startswith("origin_")
            }
            payload = {
                **audit_base,
                "review_candidate_id": _review_candidate_id(
                    route=route,
                    origin_id=f"predictor:{predictor_key}",
                    topic=audit_topic,
                ),
                "candidate_topic": audit_topic,
                "origin_literature_idea_ids": [record.literature_idea_id],
                "origin_executable_candidate_ids": [record.executable_candidate_id],
                "origin_candidate_topics": [record.candidate_topic],
                "association_outcome_contexts": [
                    record.prior_art.outcome_literature_phrase
                ],
                "review_route": route,
                "route_prior_art": {
                    "query": audit_prior_art.query,
                    "search_ok": audit_prior_art.search_ok,
                    "hit_count": audit_prior_art.hit_count,
                    "top_hits": [
                        {
                            "pmid": hit.pmid,
                            "title": hit.title,
                            "year": hit.year,
                        }
                        for hit in audit_prior_art.top_hits
                    ],
                    "interpretation": (
                        "ranking signal for human review only; a low PubMed count "
                        "is not a novelty claim"
                    ),
                },
                "selection_reason": (
                    "dictionary-resolvable across harmonized databases but only "
                    "20-70% jointly observed in the prepared cohort; association "
                    "literature saturation does not answer the separate "
                    "measurement/source-status audit question"
                ),
                "required_next_action": (
                    "profile source-status and missingness in the existing full6 "
                    "exports before defining any outcome association; do not treat "
                    "unmeasured values as clinical absence"
                ),
            }
            measurement_audit_by_predictor[predictor_key] = (
                (
                    audit_prior_art.hit_count,
                    exact_hits,
                    completeness,
                    record.candidate_topic,
                ),
                payload,
            )

    external_validation.sort(key=lambda item: item[0])
    measurement_audit = sorted(
        measurement_audit_by_predictor.values(), key=lambda item: item[0]
    )
    selected: list[dict[str, Any]] = []
    selected_external: list[dict[str, Any]] = []
    used_predictors: set[str] = set()
    used_outcomes: set[str] = set()

    # First maximize both predictor and outcome diversity.  A second pass fills
    # the bounded list with new predictors when the caller supplied only one or
    # two outcomes.  This avoids making a single brittle top-ranked pair the
    # entire human-review surface while still preventing near-duplicate rows
    # from consuming the review budget.
    for require_new_outcome in (True, False):
        for _, payload in external_validation:
            if payload in selected_external:
                continue
            predictor = str(payload["resolved_predictor_concept"])
            outcome = str(payload["resolved_outcome_concept"])
            if predictor in used_predictors:
                continue
            if require_new_outcome and outcome in used_outcomes:
                continue
            selected_external.append(payload)
            used_predictors.add(predictor)
            used_outcomes.add(outcome)
            if len(selected_external) >= _MAX_EXTERNAL_VALIDATION_REVIEW_CANDIDATES:
                break
        if len(selected_external) >= _MAX_EXTERNAL_VALIDATION_REVIEW_CANDIDATES:
            break

    selected.extend(selected_external)
    if measurement_audit:
        selected.append(measurement_audit[0][1])
    return selected


def run_data_first_idea_mining_dry_run(
    *,
    predictor_concepts: Sequence[str],
    outcome_concepts: Sequence[str],
    available_concepts: Sequence[ConceptDescriptor | str],
    output_dir: str | Path,
    data_path: str | Path,
    prior_art_search_client: Any,
    concept_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    outcome_determinability: Optional[
        Mapping[str, OutcomeDeterminability | Mapping[str, Any] | str]
    ] = None,
    databases: Optional[Sequence[str]] = None,
    database: str = "miiv",
    feasibility_probe: Optional[Any] = None,
    min_harmonized_dbs: int = 4,
    top_k: int = 25,
    prior_art_top_n: int = 20,
    source_item_index: Optional[Any] = None,
    cross_database_feasibility_fn: Any = hypothesis_cross_database_feasibility,
    literature_terms: Optional[Mapping[str, str]] = None,
    literature_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    analysis_family: str = "cross_database_replication",
) -> IdeaMiningDryRunResult:
    """Run deterministic data-first generation through the standard funnel.

    No pair is promoted because it is merely measurable.  Cross-database
    availability creates the candidate set; the existing pair-level real-data
    probe and prior-art gate determine whether each row is executable and
    differentiated.  All rows remain proposed and require human confirmation.
    """

    prepared_data = Path(data_path).resolve()
    if not prepared_data.is_file():
        raise FileNotFoundError(f"prepared data path not found: {prepared_data}")
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_sha256 = _sha256_file(prepared_data)

    effective_databases = (
        list(databases) if databases is not None else default_public_databases()
    )
    candidates = generate_data_first_candidates(
        predictor_concepts=predictor_concepts,
        outcome_concepts=outcome_concepts,
        databases=effective_databases,
        feasibility_fn=cross_database_feasibility_fn,
        min_harmonized_dbs=min_harmonized_dbs,
        limit=max(len(predictor_concepts) * max(len(outcome_concepts), 1), top_k),
    )
    materials, ideas = _source_materials_and_ideas(
        candidates,
        data_path=prepared_data,
        data_sha256=data_sha256,
        literature_terms=literature_terms or {},
        literature_aliases=literature_aliases or {},
        analysis_family=analysis_family,
    )

    result = run_idea_mining_dry_run(
        materials=materials,
        precomputed_literature_ideas=ideas,
        llm=_ProviderCallForbidden(),
        available_concepts=available_concepts,
        concept_aliases=concept_aliases,
        outcome_determinability=outcome_determinability,
        output_dir=out_dir,
        database=database,
        data_path=prepared_data,
        analytic_unit="stay",
        feasibility_probe=feasibility_probe,
        top_k=top_k,
        prior_art_search_client=prior_art_search_client,
        prior_art_top_n=prior_art_top_n,
        source_item_index=source_item_index,
        cross_db_targets=effective_databases,
    )

    shortlist = _review_shortlist(
        result,
        prior_art_search_client=prior_art_search_client,
        prior_art_top_n=prior_art_top_n,
    )
    shortlist_path = out_dir / "data_first_review_shortlist.json"
    shortlist_path.write_text(
        json.dumps(
            {
                "schema_version": DATA_FIRST_SHORTLIST_SCHEMA_VERSION,
                "selection_scope": (
                    "bounded human-review prioritization; not a novelty, go, or "
                    "paper-authorization decision"
                ),
                "candidates": shortlist,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    route_manifest = {
        "schema_version": DATA_FIRST_ROUTE_SCHEMA_VERSION,
        "prepared_data_path": str(prepared_data),
        "prepared_data_sha256": data_sha256,
        "min_harmonized_dbs": min_harmonized_dbs,
        "harmonized_databases_considered": effective_databases,
        "predictor_concepts_considered": list(predictor_concepts),
        "outcome_concepts_considered": list(outcome_concepts),
        "analysis_family": analysis_family,
        "literature_terms": dict(literature_terms or {}),
        "literature_aliases": {
            key: list(values) for key, values in (literature_aliases or {}).items()
        },
        "data_first_candidates": [candidate.__dict__ for candidate in candidates],
        "candidate_triage_report": result.triage_report_path,
        "review_shortlist": str(shortlist_path),
        "review_shortlist_count": len(shortlist),
    }
    (out_dir / "data_first_route_manifest.json").write_text(
        json.dumps(route_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


__all__ = [
    "DATA_FIRST_ROUTE_SCHEMA_VERSION",
    "DATA_FIRST_SHORTLIST_SCHEMA_VERSION",
    "run_data_first_idea_mining_dry_run",
]
