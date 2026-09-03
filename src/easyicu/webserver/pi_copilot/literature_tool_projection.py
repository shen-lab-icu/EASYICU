"""Bounded Pi projection for one Web literature-search receipt.

Idea Mining owns retrieval, ``literature_authority`` owns the persisted exact
receipt, and Research Agent owns scientific screening.  This module owns only
the compact model-visible projection.  Keeping that responsibility out of the
large tool dispatcher prevents exact queries and abstracts from being repeated
until the Pi transport exceeds its 32 KiB contract.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from easyicu.research_agent.planning.method_literature import (
    method_literature_citations,
    method_literature_digest,
    method_literature_pack,
)
from easyicu.webserver.literature_projection import literature_source_resource


_ARTICLE_LIMIT = 6
_RESOURCE_LIMIT = 6
_QUERY_LIMIT = 5


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _tail_text(value: Any, limit: int) -> str:
    """Keep the result/conclusion end of a bounded abstract projection."""

    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[-limit:]


def _binding_projection(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    allowed = (
        "schema_version",
        "receipt_id",
        "receipt_sha256",
        "status",
        "result_count",
        "searched_at",
        "study_configuration_sha256",
    )
    return {key: value.get(key) for key in allowed if value.get(key) is not None}


def _resource_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep clickability while avoiding a second copy of long evidence text."""

    resource = {
        "kind": _text(value.get("kind"), 80),
        "label": _text(value.get("label"), 160),
        "title": _text(value.get("title"), 260),
        "year": value.get("year"),
        "venue": _text(value.get("venue"), 140) or None,
        "relevance": _text(value.get("relevance"), 240) or None,
        "doi": _text(value.get("doi"), 240) or None,
        "pmid": _text(value.get("pmid"), 32) or None,
        "url": _text(value.get("url"), 500),
        "media_type": _text(value.get("media_type"), 80),
        "authority_class": _text(value.get("authority_class"), 120),
    }
    if value.get("retrieval_fit"):
        resource["retrieval_fit"] = _text(value.get("retrieval_fit"), 80)
        resource["retrieval_rationale"] = (
            _text(value.get("retrieval_rationale"), 360) or None
        )
    return resource


def compile_literature_tool_projection(
    *,
    discovered: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    idea_receipt_binding: Mapping[str, Any] | None,
    study_authority_binding: Mapping[str, Any] | None,
    bound_idea_run_id: str,
    searched_idea_was_accepted: bool = False,
) -> dict[str, Any]:
    """Compile a compact, claim-bounded Pi result from persisted owner data.

    The exact query strings, full bounded excerpts, and every candidate remain
    in the digest-bound host receipt.  Pi receives previews plus that receipt's
    identity; it must never treat truncation here as scientific authority.
    """

    rows = [
        row for row in list(candidates)[:_ARTICLE_LIMIT] if isinstance(row, Mapping)
    ]
    queries = [
        _text(value, 800)
        for value in list(discovered.get("queries_to_run") or [])[:_QUERY_LIMIT]
        if _text(value, 800)
    ]
    query_previews_truncated = any(
        len(str(value or "")) > 800
        for value in list(discovered.get("queries_to_run") or [])[:_QUERY_LIMIT]
    )
    query_strata = [
        {
            "id": _text(row.get("id"), 80),
            "returned_count": int(row.get("returned_count") or 0),
            "retained_count": int(row.get("retained_count") or 0),
        }
        for row in list(discovered.get("query_strata") or [])[:_QUERY_LIMIT]
        if isinstance(row, Mapping)
    ]
    strata_counts = {
        str(row["id"]): int(row["returned_count"])
        for row in query_strata
        if row.get("id")
    }

    articles = []
    for row in rows:
        retrieval_screen = row.get("retrieval_screen")
        retrieval_screen = (
            retrieval_screen if isinstance(retrieval_screen, Mapping) else {}
        )
        article = {
            "citation_key": _text(row.get("citation_key") or row.get("pmid"), 120),
            "title": _text(row.get("title"), 320),
            "journal": _text(row.get("journal"), 140),
            "year": row.get("year"),
            "pmid": _text(row.get("pmid"), 32) or None,
            "matched_query_strata": [
                _text(value, 80)
                for value in list(row.get("matched_query_strata") or [])[:4]
                if _text(value, 80)
            ],
            "publication_types": [
                _text(value, 120)
                for value in list(row.get("publication_types") or [])[:6]
                if _text(value, 120)
            ],
            "screening_status": "retrieval_candidate_unreviewed",
            "evidence_role": "retrieval_candidate",
            "evidence_excerpt": _text(
                row.get("design_excerpt") or row.get("evidence_quote"), 280
            ),
            "abstract_excerpt": _tail_text(row.get("abstract_excerpt"), 600),
        }
        if retrieval_screen:
            article["retrieval_fit"] = _text(retrieval_screen.get("fit"), 80) or None
            article["retrieval_rationale"] = (
                _text(retrieval_screen.get("rationale"), 360) or None
            )
        articles.append(article)

    method_pack = method_literature_pack()
    method_sources = [
        {
            "citation_key": _text(row.get("key"), 120),
            "title": _text(row.get("title"), 220),
            "year": row.get("year"),
            "pmid": _text(row.get("pmid"), 32) or None,
            "doi": _text(row.get("doi"), 240) or None,
            "url": _text(row.get("url"), 500) or None,
            "evidence_role": "method",
        }
        for row in method_literature_citations()
    ]
    methodology = {
        "schema_version": method_pack["schema_version"],
        "sha256": method_literature_digest(),
        "cards": [
            {
                "id": _text(row.get("id"), 120),
                "layer": _text(row.get("layer"), 80),
                "source_key": _text(row.get("source_key"), 120),
            }
            for row in method_pack["cards"]
        ],
        "sources": method_sources,
        "evidence_boundary": (
            "This frozen case-neutral method pack supplies design questions; "
            "it does not make a retrieved topic paper a direct comparator."
        ),
    }

    resources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    # Browser resources are the topic-search articles the researcher may open.
    # Method references remain available under the separately labelled
    # methodology section; mixing them into this list made a zero-hit topic
    # search look as though STROBE/RECORD were supporting studies.
    for source_row in rows:
        raw = literature_source_resource(source_row)
        if raw is None:
            continue
        url = _text(raw.get("url"), 500)
        if not url or url in seen_urls:
            continue
        resources.append(_resource_projection(raw))
        seen_urls.add(url)
        if len(resources) == _RESOURCE_LIMIT:
            break

    literature_search = {
        "status": _text(discovered.get("status") or "search_failed", 80),
        "search_performed": bool(discovered.get("search_performed")),
        "query_previews": queries,
        "query_previews_truncated": query_previews_truncated,
        "exact_queries_bound_in_host_receipt": bool(study_authority_binding),
        "query_strata": query_strata,
        "retrieval_map": {
            "clinical_landscape": strata_counts.get("clinical_landscape", 0),
            "candidate_topic": strata_counts.get("candidate_topic", 0),
            "direct_observational_candidates": strata_counts.get(
                "direct_observational_candidates", 0
            ),
            "review_or_guideline": strata_counts.get("review_or_guideline", 0),
            "critical_care_database": strata_counts.get("critical_care_database", 0),
            "interpretation": (
                "retrieval_candidates_present_requires_screening"
                if rows
                else "no_candidates_in_bounded_strata_not_novelty_evidence"
            ),
            "authority": "retrieval_signal_only",
        },
        "result_count": len(rows),
        "retrieved_result_count": int(
            discovered.get("retrieved_result_count") or len(rows)
        ),
        "excluded_result_count": int(discovered.get("excluded_result_count") or 0),
        "network_calls": int(discovered.get("network_calls") or 0),
        "articles": articles,
        "methodology": methodology,
        "evidence_boundary": (
            "Every topic-search record is an unreviewed retrieval candidate, "
            "not verified supporting evidence or a direct comparator. Research "
            "Agent must re-screen publication type and exact population, "
            "exposure role, outcome, and design against the sealed study."
        ),
        "idea_plan_refresh_required": bool(idea_receipt_binding),
        "idea_handoff_refresh_required": bool(
            idea_receipt_binding and searched_idea_was_accepted
        ),
        "bound_idea_run_id": _text(bound_idea_run_id, 160) or None,
        "study_literature_authority": _binding_projection(study_authority_binding),
    }
    return {
        "literature_search": literature_search,
        "resource": resources[0] if resources else None,
        "resources": resources,
    }


__all__ = ["compile_literature_tool_projection"]
