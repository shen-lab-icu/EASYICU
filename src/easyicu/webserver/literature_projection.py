"""Browser-safe literature evidence projections for EasyICU workflows.

The Research Agent and Idea Mining remain the owners of retrieval and citation
metadata.  This module only compiles their typed/persisted outputs into a
bounded, path-free Web/Pi contract.  It never searches the network and never
asks a model to invent or repair a citation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlparse

from pydantic import BaseModel, ValidationError

from easyicu.research_agent.planning.literature_contract import (
    ImpactProxies, RecordReview, ReviewCoverage, ReviewPolicy,
)


LITERATURE_EVIDENCE_SCHEMA_VERSION = "easyicu.web-literature-evidence/5"
_MAX_CITATIONS = 400
_MAX_STEPS = 80
_MAX_EVIDENCE_CARDS = 40
_GOVERNED_SCIENTIFIC_ROLES = {"primary", "secondary", "sensitivity"}


def load_current_plan_authority(run_dir: Path) -> Mapping[str, Any]:
    """Load only the digest-verified final plan named by the run manifest.

    ``analysis_plan.json`` is the initial Planner artifact.  Replanning may
    later register a new scientific authority, so projecting citations from
    that initial file can report a false-complete mapping for a plan that never
    executed.  The manifest owner already records the exact final path/SHA;
    this reader verifies both before exposing the plan to Web.
    """

    import hashlib
    import json

    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        authority = manifest.get("current_plan_authority")
        if not isinstance(authority, Mapping):
            return {}
        relative_text = str(authority.get("relative_path") or "").strip()
        expected_sha = str(authority.get("sha256") or "").strip().lower()
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        ):
            return {}
        root = run_dir.resolve()
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
        raw = candidate.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            return {}
        payload = json.loads(raw.decode("utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _typed_review(value: Any, model: type[BaseModel]) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return model.model_validate(value).model_dump(mode="json")
    except ValidationError:
        return None


def _source_url(row: Mapping[str, Any]) -> str | None:
    pmid = _text(row.get("pmid"), 32)
    if re.fullmatch(r"[0-9]{1,12}", pmid):
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    doi = _text(row.get("doi"), 240)
    if doi and re.fullmatch(r"10\.[0-9]{4,9}/[-._;()/:A-Za-z0-9]+", doi):
        return f"https://doi.org/{quote(doi, safe='/():;._-')}"
    candidate = _text(row.get("url"), 500)
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        return None
    return candidate


def _citation_projection(
    row: Mapping[str, Any],
    *,
    screening: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    screening = screening if isinstance(screening, Mapping) else {}
    notices = row.get("bibliographic_notices")
    notices = notices if isinstance(notices, list) else []
    relevance = _text(
        row.get("relevance") or row.get("evidence_quote") or row.get("evidence_sentence"),
        1_200,
    )
    # The literature owner prefixes retained passages; a general applicability
    # note (including curated methodology prompts) is not a source abstract.
    source_excerpt = relevance.startswith(("Study-design excerpt:", "Source excerpt:"))
    if not row.get("relevance") and (row.get("evidence_quote") or row.get("evidence_sentence")):
        source_excerpt = True
    return {
        "key": _text(row.get("key") or row.get("citation_key"), 120),
        "title": _text(row.get("title"), 500),
        "year": _text(row.get("year"), 16) or None,
        "venue": _text(row.get("venue") or row.get("journal"), 240) or None,
        "review": _typed_review(row.get("review"), RecordReview),
        "impact": _typed_review(row.get("impact"), ImpactProxies),
        "relevance": relevance or None,
        "relevance_kind": "source_excerpt" if source_excerpt else "reference_note",
        "doi": _text(row.get("doi"), 240) or None,
        "pmid": _text(row.get("pmid"), 32) or None,
        "publication_types": [
            _text(value, 120)
            for value in list(row.get("publication_types") or [])[:20]
            if _text(value, 120)
        ],
        "bibliographic_notices": [
            _text(value, 600) for value in notices[:20]
            if isinstance(value, str) and _text(value, 600)
        ],
        "source_url": _source_url(row),
        "screening": (
            {
                "disposition": _text(screening.get("disposition"), 40) or None,
                "evidence_role": _text(screening.get("evidence_role"), 80) or None,
                "source": _text(screening.get("source"), 80) or None,
                "rationale": _text(screening.get("rationale"), 1_200) or None,
                "population_match": bool(screening.get("population_match")),
                "exposure_match": bool(screening.get("exposure_match")),
                "outcome_match": bool(screening.get("outcome_match")),
                "design_excerpt_available": bool(
                    screening.get("design_excerpt_available")
                ),
                "publication_type_eligible": bool(
                    screening.get("publication_type_eligible", True)
                ),
            }
            if screening
            else None
        ),
    }


def project_literature_evidence_cards(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Project registered literature cards for review without raw bundle data.

    This is a bounded view of recorded candidate evidence, not a fresh screen
    or verification of a plan's individual claims. The caller must first bind
    and verify the run-owned source bytes and perform the disclosure scan.
    """

    raw_citations = bundle.get("citations")
    raw_decisions = bundle.get("screening_decisions")
    raw_cards = bundle.get("design_evidence_cards")
    citations = raw_citations if isinstance(raw_citations, list) else []
    decisions = raw_decisions if isinstance(raw_decisions, list) else []
    source_cards = raw_cards if isinstance(raw_cards, list) else []
    citations_by_key: dict[str, list[Mapping[str, Any]]] = {}
    decisions_by_key: dict[str, list[Mapping[str, Any]]] = {}
    cards_by_key: dict[str, list[Mapping[str, Any]]] = {}
    for rows, index, field in (
        (citations, citations_by_key, "key"),
        (decisions, decisions_by_key, "citation_key"),
        (source_cards, cards_by_key, "citation_key"),
    ):
        for row in rows:
            if isinstance(row, Mapping):
                key = _text(row.get(field), 120)
                if key:
                    index.setdefault(key, []).append(row)

    projected = []
    suppressed = 0
    for key, matches in cards_by_key.items():
        cited = citations_by_key.get(key, [])
        screened = decisions_by_key.get(key, [])
        if len(matches) != 1 or len(cited) != 1 or len(screened) != 1:
            suppressed += len(matches)
            continue
        card, citation, decision = matches[0], cited[0], screened[0]
        role = _text(card.get("evidence_role"), 80)
        if decision.get("disposition") != "include" or role != decision.get("evidence_role"):
            suppressed += 1
            continue
        evidence = card.get("evidence")
        evidence = evidence if isinstance(evidence, list) else []
        facts = [
            {
                "dimension": _text(item.get("dimension"), 80),
                "review_role": _text(item.get("review_role"), 80) or None,
                "source_backed_summary": _text(item.get("source_backed_summary"), 500),
                "locator": _text(item.get("locator"), 160) or None,
            }
            for item in evidence[:14]
            if isinstance(item, Mapping)
        ]
        projected.append({
            "citation_key": key,
            "title": _text(citation.get("title"), 500),
            "year": _text(citation.get("year"), 16) or None,
            "source_url": _source_url(citation),
            "evidence_role": role,
            "access_mode": _text(card.get("access_mode"), 80),
            "supplement_status": _text(card.get("supplement_status"), 80),
            "evidence": facts,
        })
    provenance = bundle.get("search_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    return {
        "schema_version": "easyicu.web-literature-cards/1",
        "searched_at": _text(provenance.get("searched_at"), 80) or None,
        "review": _typed_review(bundle.get("review"), ReviewCoverage),
        "cards": projected[:_MAX_EVIDENCE_CARDS],
        "card_count": len(projected),
        "cards_truncated": len(projected) > _MAX_EVIDENCE_CARDS,
        "suppressed_ambiguous_cards": suppressed,
        "mapping_status": "not_assessed",
        "boundary": "Recorded candidate evidence; source relevance and claim support require review.",
    }


def project_run_literature(
    *,
    run_id: str,
    bundle: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one persisted pre-plan literature bundle and its plan mapping."""

    raw_citations = bundle.get("citations")
    raw_citations = raw_citations if isinstance(raw_citations, Sequence) else []
    raw_screening = bundle.get("screening_decisions")
    raw_screening = raw_screening if isinstance(raw_screening, Sequence) else []
    screening_by_key = {
        _text(row.get("citation_key"), 120): row
        for row in raw_screening
        if isinstance(row, Mapping) and _text(row.get("citation_key"), 120)
    }
    citations = [
        _citation_projection(
            row,
            screening=screening_by_key.get(
                _text(row.get("key") or row.get("citation_key"), 120)
            ),
        )
        for row in list(raw_citations)[:_MAX_CITATIONS]
        if isinstance(row, Mapping) and _text(row.get("title"), 500)
    ]
    citation_keys = {row["key"] for row in citations if row.get("key")}
    citation_by_key = {row["key"]: row for row in citations if row.get("key")}
    direct_comparator_keys = sorted(
        key
        for key, row in screening_by_key.items()
        if row.get("disposition") == "include"
        and row.get("evidence_role") == "direct_comparator"
        and row.get("publication_type_eligible", True) is not False
        and key in citation_keys
    )

    raw_steps = plan.get("steps")
    raw_steps = raw_steps if isinstance(raw_steps, Sequence) else []
    step_map: list[dict[str, Any]] = []
    mapped_steps = 0
    scientific_step_count = 0
    scientific_mapped_steps = 0
    unknown_keys: set[str] = set()
    for raw in list(raw_steps)[:_MAX_STEPS]:
        if not isinstance(raw, Mapping):
            continue
        requested = raw.get("literature_citation_keys")
        requested = (
            requested
            if isinstance(requested, Sequence)
            and not isinstance(requested, (str, bytes))
            else []
        )
        keys = []
        for value in requested:
            key = _text(value, 120)
            if key and key not in keys:
                keys.append(key)
        unknown_keys.update(key for key in keys if key not in citation_keys)
        valid_keys = [key for key in keys if key in citation_keys]
        raw_design_bindings = raw.get("literature_design_bindings")
        raw_design_bindings = (
            raw_design_bindings
            if isinstance(raw_design_bindings, Sequence)
            and not isinstance(raw_design_bindings, (str, bytes))
            else []
        )
        design_bindings_by_key: dict[str, Mapping[str, Any]] = {}
        for item in raw_design_bindings:
            if not isinstance(item, Mapping):
                continue
            binding_key = _text(item.get("citation_key"), 120)
            if binding_key in valid_keys and binding_key not in design_bindings_by_key:
                design_bindings_by_key[binding_key] = item
        planned_role = _text(raw.get("planned_analysis_role"), 40)
        # Typed plans always declare one of the four known roles.  Treat an
        # absent/unknown role as governed rather than allowing an older or
        # malformed plan to evade the scientific citation gate.
        governed_scientific_step = (
            planned_role in _GOVERNED_SCIENTIFIC_ROLES or not planned_role
        )
        if valid_keys:
            mapped_steps += 1
        all_citations_design_bound = bool(valid_keys) and all(
            key in design_bindings_by_key for key in valid_keys
        )
        if governed_scientific_step:
            scientific_step_count += 1
            if all_citations_design_bound:
                scientific_mapped_steps += 1
        step_map.append(
            {
                "step_id": _text(raw.get("step_id") or raw.get("id"), 160),
                "intent": _text(raw.get("intent") or raw.get("title"), 1_200),
                "planned_analysis_role": planned_role or None,
                "governed_scientific_step": governed_scientific_step,
                "citation_keys": valid_keys,
                "citation_bindings": [
                    {
                        "key": key,
                        "title": citation_by_key[key].get("title"),
                        "year": citation_by_key[key].get("year"),
                        "source_url": citation_by_key[key].get("source_url"),
                        "evidence_role": (
                            (citation_by_key[key].get("screening") or {}).get(
                                "evidence_role"
                            )
                            if isinstance(
                                citation_by_key[key].get("screening"), Mapping
                            )
                            else None
                        )
                        or "curated_method_or_context",
                        "design_elements": [
                            _text(value, 40)
                            for value in list(
                                design_bindings_by_key.get(key, {}).get(
                                    "design_elements"
                                )
                                or []
                            )[:12]
                            if _text(value, 40)
                        ],
                        "application": _text(
                            design_bindings_by_key.get(key, {}).get("application"),
                            1_200,
                        )
                        or None,
                        "divergence": _text(
                            design_bindings_by_key.get(key, {}).get("divergence"),
                            1_200,
                        )
                        or None,
                        "design_binding_status": (
                            "typed" if key in design_bindings_by_key else "citation_only"
                        ),
                    }
                    for key in valid_keys
                ],
                "support_status": (
                    "design_bound"
                    if all_citations_design_bound
                    else "citation_only"
                    if valid_keys
                    else "not_bound"
                ),
            }
        )

    provenance = bundle.get("search_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    raw_queries = provenance.get("search_queries")
    raw_queries = raw_queries if isinstance(raw_queries, Mapping) else {}
    searched = bool(provenance.get("search_conducted"))
    status = (
        "searched" if searched else ("curated_only" if citations else "unavailable")
    )
    if not step_map:
        mapping_status = "not_applicable"
    elif mapped_steps == len(step_map):
        mapping_status = "complete"
    elif mapped_steps:
        mapping_status = "partial"
    else:
        mapping_status = "not_bound"
    if not scientific_step_count:
        scientific_mapping_status = "not_applicable"
    elif scientific_mapped_steps == scientific_step_count:
        scientific_mapping_status = "complete"
    elif scientific_mapped_steps:
        scientific_mapping_status = "partial"
    else:
        scientific_mapping_status = "not_bound"

    citation_years = sorted(
        {
            int(str(row.get("year") or "").strip())
            for row in citations
            if str(row.get("year") or "").strip().isdigit()
        }
    )

    prisma = bundle.get("prisma")
    prisma = prisma if isinstance(prisma, Mapping) else None
    return {
        "schema_version": LITERATURE_EVIDENCE_SCHEMA_VERSION,
        "scope": "research_plan",
        "run_id": _text(run_id, 160),
        "research_question": _text(
            bundle.get("research_question") or plan.get("research_question"), 2_000
        ),
        "status": status,
        "review": _typed_review(bundle.get("review"), ReviewCoverage),
        "search": {
            "search_conducted": searched,
            "searched_at": _text(provenance.get("searched_at"), 80) or None,
            "curated_seed_count": (
                int(provenance["curated_seed_count"])
                if provenance.get("curated_seed_count") is not None else None
            ),
            "sources_enabled": [
                _text(value, 80)
                for value in list(provenance.get("sources_enabled") or [])[:12]
                if _text(value, 80)
            ],
            "sources_returning": [
                _text(value, 80)
                for value in list(provenance.get("sources_returning") or [])[:12]
                if _text(value, 80)
            ],
            "note": _text(provenance.get("note"), 1_200),
            "review_policy": _typed_review(provenance.get("review_policy"), ReviewPolicy),
            "sources_failing": {
                _text(source, 80): [_text(reason, 600) for reason in list(reasons)[:12]]
                for source, reasons in (provenance.get("sources_failing") or {}).items()
                if isinstance(reasons, list)
            } if isinstance(provenance.get("sources_failing"), Mapping) else {},
            "query_roles": {
                _text(source, 80): {
                    _text(role, 80): [_text(query, 4000) for query in list(queries)[:12]]
                    for role, queries in roles.items() if isinstance(queries, list)
                } for source, roles in (provenance.get("query_roles") or {}).items()
                if isinstance(roles, Mapping)
            } if isinstance(provenance.get("query_roles"), Mapping) else {},
            "prisma": dict(prisma) if prisma else None,
            "queries": {
                _text(source, 80): [
                    _text(query, 4_000)
                    for query in list(queries or [])[:24]
                    if _text(query, 4_000)
                ]
                for source, queries in raw_queries.items()
                if _text(source, 80)
            },
        },
        "citation_count": len(citations),
        "citations": citations,
        "direct_comparator_count": len(direct_comparator_keys),
        "direct_comparator_keys": direct_comparator_keys,
        "plan_step_count": len(step_map),
        "mapped_step_count": mapped_steps,
        "mapping_status": mapping_status,
        "scientific_plan_step_count": scientific_step_count,
        "scientific_mapped_step_count": scientific_mapped_steps,
        "scientific_mapping_status": scientific_mapping_status,
        "step_citation_map": step_map,
        "citation_year_range": {
            "oldest": citation_years[0] if citation_years else None,
            "newest": citation_years[-1] if citation_years else None,
        },
        "integrity": {
            "unknown_citation_keys_removed": sorted(unknown_keys),
            "path_values_returned": False,
            "patient_rows_returned": False,
        },
        "evidence_boundary": (
            "Literature roles and screening decisions support design rationale "
            "and prior-art review. A direct-comparator label is still a candidate "
            "for independent appraisal, not proof that eligibility, time zero, or "
            "the estimand may be copied. It is "
            "separate from patient/result EvidenceStore evidence and does not "
            "make an analysis result reportable."
        ),
    }


def load_run_literature_projection(
    *,
    run_dir: Path,
    run_id: str,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the fixed literature bundle against the final plan authority."""

    path = run_dir / "preplan_literature_bundle.json"
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        payload = {}
    if not isinstance(payload, Mapping):
        payload = {}
    # Before execution/finalisation there is no manifest current-plan pointer.
    # The Web pipeline bridge may pass the exact plan extracted from the
    # digest-bound PlanReviewAuthority; use it for the pause projection rather
    # than showing an empty plan while the reviewer is deciding.
    supplied_plan = dict(plan) if isinstance(plan, Mapping) and plan else {}
    final_plan = supplied_plan or load_current_plan_authority(run_dir)
    projection = project_run_literature(
        run_id=run_id,
        bundle=payload,
        plan=final_plan,
    )
    projection["integrity"]["current_plan_authority_verified"] = bool(final_plan)
    projection["integrity"]["plan_authority_source"] = (
        "digest_bound_human_review"
        if supplied_plan
        else "manifest_current_plan"
        if final_plan
        else "unavailable"
    )
    # Never fall back to the initial plan for a scientific mapping verdict.  A
    # missing/tampered manifest is itself an unavailable authority, not an
    # invitation to show a stale green status.
    if not final_plan:
        projection["mapping_status"] = "authority_unavailable"
        projection["scientific_mapping_status"] = "authority_unavailable"
    return projection


def literature_source_resource(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build a bounded click target from an Idea Mining PubMed owner record."""

    citation = _citation_projection(row)
    if not citation.get("title") or not citation.get("source_url"):
        return None
    evidence_role = _text(row.get("evidence_role"), 80)
    authority_class = (
        "literature_method"
        if evidence_role == "method"
        else "literature_retrieval_candidate"
    )
    resource = {
        "kind": "literature_source",
        "label": citation["title"][:160],
        "title": citation["title"],
        "year": citation.get("year"),
        "venue": citation.get("venue"),
        "relevance": citation.get("relevance"),
        "doi": citation.get("doi"),
        "pmid": citation.get("pmid"),
        "url": citation.get("source_url"),
        "media_type": "text/html",
        "authority_class": authority_class,
    }
    retrieval_screen = row.get("retrieval_screen")
    if isinstance(retrieval_screen, Mapping):
        resource["retrieval_fit"] = (
            _text(retrieval_screen.get("fit"), 80) or None
        )
        resource["retrieval_rationale"] = (
            _text(retrieval_screen.get("rationale"), 600) or None
        )
    return resource


__all__ = [
    "LITERATURE_EVIDENCE_SCHEMA_VERSION",
    "literature_source_resource",
    "load_current_plan_authority",
    "load_run_literature_projection",
    "project_run_literature",
]
