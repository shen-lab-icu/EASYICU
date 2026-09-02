"""Domain transaction for governed Web literature retrieval and binding.

The Pi and HTTP adapters should supply arguments and project results.  This
module owns the full retrieval choice, accepted-Idea mapping, exact receipt
persistence, and StudyContext compare-and-swap binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from easyicu.webserver import literature_authority, study_contexts
from easyicu.webserver.ideas import mining as idea_mining


class LiteratureSearchTransactionError(ValueError):
    """Owner-attributed, adapter-safe failure of the literature transaction."""

    def __init__(self, code: str, message: str, *, owner: str) -> None:
        self.code = str(code)
        self.message = str(message)
        self.owner = str(owner)
        super().__init__(self.message)


@dataclass(frozen=True)
class LiteratureSearchTransaction:
    discovered: Mapping[str, Any]
    candidates: tuple[Mapping[str, Any], ...]
    idea_receipt_binding: Optional[Mapping[str, Any]]
    study_authority_binding: Optional[Mapping[str, Any]]
    updated_study: Optional[Mapping[str, Any]]
    bound_idea_run_id: str
    searched_idea_was_accepted: bool


def execute_literature_search(
    *,
    study: Mapping[str, Any],
    topic: str,
    journal: str,
    requested_limit: int,
    bound_study_id: str,
    candidate_run_id: str = "",
    candidate_idea_id: str = "",
) -> LiteratureSearchTransaction:
    """Retrieve and bind the exact literature authority for one study state."""

    idea_handoff = (
        study.get("idea_handoff")
        if isinstance(study.get("idea_handoff"), Mapping)
        else {}
    )
    execution_concepts = (
        study.get("execution_concepts")
        if isinstance(study.get("execution_concepts"), Mapping)
        else {}
    )
    bound_idea_run = str(idea_handoff.get("run_id") or "").strip()
    bound_idea_id = str(idea_handoff.get("idea_id") or "").strip()
    requested_run = str(candidate_run_id or "").strip()
    requested_idea = str(candidate_idea_id or "").strip()
    if bool(requested_run) != bool(requested_idea):
        raise LiteratureSearchTransactionError(
            "idea_identity_incomplete",
            "Both run_id and idea_id are required to search one mined candidate.",
            owner="easyicu.webserver.ideas.mining",
        )
    if requested_run and bound_idea_run and (
        requested_run != bound_idea_run or requested_idea != bound_idea_id
    ):
        raise LiteratureSearchTransactionError(
            "idea_identity_conflicts_with_accepted_handoff",
            "The requested candidate does not match the accepted Idea Mining handoff.",
            owner="easyicu.webserver.ideas.handoff",
        )
    idea_run = requested_run or bound_idea_run
    idea_id = requested_idea or bound_idea_id
    searched_idea_was_accepted = bool(
        idea_run and bound_idea_run and idea_run == bound_idea_run and idea_id == bound_idea_id
    )
    try:
        if idea_run and idea_id:
            discovered, receipt_binding = _search_idea_candidate(
                run_id=idea_run,
                idea_id=idea_id,
                requested_limit=requested_limit,
            )
        else:
            discovered = _search_study_scope(
                study=study,
                execution_concepts=execution_concepts,
                topic=topic,
                journal=journal,
                requested_limit=requested_limit,
            )
            receipt_binding = None
    except idea_mining.IdeaMiningWebError as exc:
        detail = exc.detail
        raise LiteratureSearchTransactionError(
            str(detail.get("error") or "literature_search_blocked"),
            str(
                detail.get("reason")
                or "Idea Mining rejected the literature search."
            ),
            owner="easyicu.webserver.ideas.mining",
        ) from exc

    generic_binding: Optional[Mapping[str, Any]] = None
    updated_study: Optional[Mapping[str, Any]] = None
    if bool(discovered.get("search_performed")) and not idea_run:
        generic_binding, updated_study = _bind_study_authority(
            study=study,
            discovered=discovered,
            bound_study_id=bound_study_id,
        )
    candidates = tuple(
        row
        for row in list(discovered.get("source_candidates") or [])[:requested_limit]
        if isinstance(row, Mapping)
    )
    return LiteratureSearchTransaction(
        discovered=discovered,
        candidates=candidates,
        idea_receipt_binding=(
            receipt_binding if isinstance(receipt_binding, Mapping) else None
        ),
        study_authority_binding=generic_binding,
        updated_study=updated_study,
        bound_idea_run_id=idea_run,
        searched_idea_was_accepted=searched_idea_was_accepted,
    )


def _search_idea_candidate(
    *, run_id: str, idea_id: str, requested_limit: int
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    checked = idea_mining.check_prior_art(
        {"run_id": run_id, "idea_id": idea_id, "allow_network": True}
    )
    prior = checked.get("prior_art")
    prior = prior if isinstance(prior, Mapping) else {}
    raw_candidates = [
        row
        for row in list(prior.get("results") or [])[:requested_limit]
        if isinstance(row, Mapping)
    ]
    candidates = [
        {
            "citation_key": f"idea_pubmed_{str(row.get('pmid') or '').strip()}",
            "title": row.get("title"),
            "journal": row.get("journal") or row.get("source"),
            "year": row.get("year") or row.get("pubdate"),
            "doi": row.get("doi"),
            "pmid": row.get("pmid"),
            "url": (
                f"https://pubmed.ncbi.nlm.nih.gov/{str(row.get('pmid') or '').strip()}/"
                if str(row.get("pmid") or "").strip()
                else None
            ),
            "evidence_quote": str(
                row.get("abstract_excerpt")
                or row.get("evidence_sentence")
                or (
                    "Matched the accepted Idea Mining prior-art query: "
                    + str(row.get("query") or "")
                )
            )[:600],
            "abstract_excerpt": str(row.get("abstract_excerpt") or "")[:1200],
            "design_excerpt": str(
                row.get("design_excerpt") or row.get("abstract_excerpt") or ""
            )[:1200],
            "publication_types": [
                str(value)[:120]
                for value in list(row.get("publication_types") or [])[:12]
                if str(value).strip()
            ],
            "matched_queries": [
                str(value)[:1500]
                for value in list(row.get("matched_queries") or [])[:6]
                if str(value).strip()
            ]
            or (
                [str(row.get("query") or "")[:1500]]
                if str(row.get("query") or "").strip()
                else []
            ),
            "matched_query_strata": [
                str(value)[:80]
                for value in list(row.get("matched_query_strata") or [])[:6]
                if str(value).strip()
            ]
            or ["accepted_idea_prior_art"],
            "retrieval_screen": (
                dict(row.get("retrieval_screen") or {})
                if isinstance(row.get("retrieval_screen"), Mapping)
                else {}
            ),
            "relevance": str(
                (row.get("retrieval_screen") or {}).get("rationale")
                if isinstance(row.get("retrieval_screen"), Mapping)
                else ""
            )[:600]
            or None,
        }
        for row in raw_candidates
    ]
    return (
        {
            "status": prior.get("status"),
            "search_performed": prior.get("search_performed"),
            "queries_to_run": prior.get("queries_to_run") or [],
            "query_strata": prior.get("query_strata") or [],
            "network_calls": prior.get("network_calls") or 0,
            "retrieved_result_count": prior.get("retrieved_result_count") or 0,
            "excluded_result_count": prior.get("excluded_result_count") or 0,
            "source_candidates": candidates,
        },
        idea_mining.prior_art_receipt_binding(run_id),
    )


def _search_study_scope(
    *,
    study: Mapping[str, Any],
    execution_concepts: Mapping[str, Any],
    topic: str,
    journal: str,
    requested_limit: int,
) -> Mapping[str, Any]:
    cohort_scope = study.get("cohort")
    cohort_scope = cohort_scope if isinstance(cohort_scope, Mapping) else {}
    data_source = study.get("data_source")
    data_source = data_source if isinstance(data_source, Mapping) else {}
    analysis_design = study.get("analysis_design")
    analysis_design = analysis_design if isinstance(analysis_design, Mapping) else {}
    return idea_mining.discover_literature(
        {
            "topic": topic,
            "exposure": str(study.get("primary_exposure") or "").strip(),
            "outcome": str(study.get("outcome") or "").strip(),
            "exposure_concept": str(
                execution_concepts.get("primary_exposure") or ""
            ).strip(),
            "outcome_concept": str(
                execution_concepts.get("outcome") or ""
            ).strip(),
            "population": " ".join(
                str(cohort_scope.get(key) or "").strip()
                for key in ("label", "review", "preset")
                if str(cohort_scope.get(key) or "").strip()
            ),
            "database": str(data_source.get("database") or "").strip(),
            "analysis_family": str(
                analysis_design.get("analysis_family") or ""
            ).strip(),
            "journal": journal,
            "limit": requested_limit,
            "allow_network": True,
        }
    )


def _bind_study_authority(
    *,
    study: Mapping[str, Any],
    discovered: Mapping[str, Any],
    bound_study_id: str,
) -> tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    study_id = str(study.get("id") or "").strip()
    stored = (
        study_contexts.get_context(study_id)
        if study_id and bound_study_id == study_id
        else None
    )
    if stored is None and bound_study_id:
        raise LiteratureSearchTransactionError(
            "literature_authority_study_missing",
            (
                "The bound StudyContext disappeared before its literature receipt "
                "could be attached. Search results were not authorized."
            ),
            owner="easyicu.webserver.literature_authority",
        )
    if stored is None:
        return None, None
    if int(stored.get("revision") or 0) != int(study.get("revision") or 0):
        raise LiteratureSearchTransactionError(
            "literature_authority_study_revision_conflict",
            (
                "The study changed while PubMed was being searched. Search again "
                "against the current study revision."
            ),
            owner="easyicu.webserver.literature_authority",
        )
    try:
        binding = literature_authority.persist_literature_authority(
            study=stored,
            discovered=discovered,
        )
        updated = study_contexts.bind_literature_authority(
            study_id,
            binding,
            expected_revision=int(stored.get("revision") or 0),
        )
    except literature_authority.LiteratureAuthorityError as exc:
        raise LiteratureSearchTransactionError(
            exc.code,
            exc.message,
            owner="easyicu.webserver.literature_authority",
        ) from exc
    except study_contexts.StudyContextError as exc:
        raise LiteratureSearchTransactionError(
            str(
                exc.detail.get("error")
                or "literature_authority_binding_failed"
            ),
            (
                "The StudyContext changed before the literature receipt could be "
                "bound. Search again against the current revision."
            ),
            owner="easyicu.webserver.study_contexts",
        ) from exc
    return binding, updated


__all__ = [
    "LiteratureSearchTransaction",
    "LiteratureSearchTransactionError",
    "execute_literature_search",
]
