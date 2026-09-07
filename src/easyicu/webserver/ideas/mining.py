"""Native Web Idea Mining adapter.

This module exposes a local-first, metadata-only discovery workflow for the
FastAPI UI.  Live literature search remains explicit network opt-in.  Local PDF
and literature-folder ingestion are allowed, but only bounded excerpts, file
metadata, and hashes are returned; full text is not persisted.  The web contract
produces source evidence, idea ledger rows, data-dictionary feasibility,
active-export feasibility summaries, and a frozen plan handoff draft.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import html
import http.client
import ipaddress
import json
import re
import socket
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib import parse, request
from xml.etree import ElementTree as ET

from easyicu.concept import catalog as concept_catalog
from easyicu.research_agent.discovery.discovery_handoff import DiscoveryHandoffPacket
from easyicu.research_agent.discovery.source_provenance import (
    idea_with_readiness_overlay as _idea_with_readiness_overlay,
)
from easyicu.research_agent.discovery.idea_mining_construct_answerability import (
    assess_idea_constructs,
)
from easyicu.research_agent.literature_excerpt import select_source_backed_excerpt
from easyicu.research_agent.literature_concepts import (
    concept_id as literature_concept_id,
    literature_concept_phrase,
)
from easyicu.research_agent.know_how.registry import KnowHowRegistry
from easyicu.webserver import state_paths
from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store
from easyicu.webserver.ideas import direct_evidence_search
from easyicu.webserver.ideas.handoff import (
    CanonicalHandoffIntegrityError,
    build_web_handoff_packet,
    is_legacy_handoff_envelope,
    load_validated_canonical_handoff,
    persist_canonical_handoff,
)
from easyicu.webserver.ideas.prior_art_receipt import (
    PriorArtReceiptError,
    build_prior_art_binding,
    load_bound_prior_art_literature as _load_bound_prior_art_literature,
)
from easyicu.webserver.input_validation import parse_bool

# Aliased: `entity_ids` is also a local variable name in this module.
from easyicu.webserver import entity_ids as entity_id_contract

_CONFIG_DIR = state_paths.state_root()
_RUN_ROOT = _CONFIG_DIR / "idea_mining_runs"
_HISTORY_PATH = _CONFIG_DIR / "webserver_idea_mining_runs.json"
_AGENT_PROJECTS_ROOT = _CONFIG_DIR / "agent_project_seeds"
_AGENT_PROJECTS_PATH = _CONFIG_DIR / "webserver_agent_project_seeds.json"

_PRIOR_ART_ADJUDICATION_FILENAME = "idea_prior_art_adjudication.json"
_BOUNDED_FEASIBILITY_FILENAME = "bounded_sample_feasibility.json"
_PRIOR_ART_DECISIONS = frozenset({"already_answered", "differentiated", "uncertain"})
_CONFIRMED_DEFINITION_FIELDS = (
    "research_question",
    "population",
    "exposure",
    "outcome",
    "time_zero",
    "time_window",
)
_CONCEPT_BINDING_ROLES = (
    "primary_exposure",
    "outcome",
    "time_zero",
)

_MAX_SOURCE_QUOTE = 420
_MAX_DESIGN_EXCERPT = 1_200
_MAX_FEATURE_STATS = 24
_MAX_FETCH_BYTES = 256_000
_MAX_PUBMED_FETCH_BYTES = 1_000_000
_MAX_PMC_FETCH_BYTES = 2_000_000
_MAX_PDF_BYTES = 20 * 1024 * 1024
_MAX_PDF_BASE64_CHARS = 4 * ((_MAX_PDF_BYTES + 2) // 3)
_MAX_PDF_EXTRACT_PAGES = 8
_MAX_PDF_EXCERPT = 1_200
_MAX_LITERATURE_PDFS = 80
_IDEA_FEATURE_ROW_SCAN_LIMIT = 1_000_000
_IDEA_SAMPLE_DEFAULT_RECORDS = 100_000
_IDEA_SAMPLE_MAX_RECORDS = 250_000
_NETWORK_TIMEOUT_SEC = 8
_TIME_COLUMNS = {
    "charttime",
    "time",
    "datetime",
    "timestamp",
    "starttime",
    "endtime",
    "hour",
    "measuredat_minutes",
    "observationoffset",
}

_VENTILATION_EPISODE_CONCEPTS = frozenset(
    {
        "mech_vent",
        "vent_ind",
        "vent_mode",
        "vent_breath_seq",
        "vent_start",
        "vent_end",
        "ett_gcs",
    }
)
_DIRECT_ID_MARKERS = {"stay_id", "subject_id", "hadm_id", "tableRows"}
_INTERVENTION_PRIORITY = (
    "vaso_ind",
    "norepi_equiv",
    "norepi_rate",
    "norepi_dur",
    "total_input_ml",
    "fluid_balance_cumulative",
    "fluid_balance",
)
_SEVERITY_PRIORITY = (
    "sep3_sofa1",
    "sep3_sofa2",
    "lact",
    "map",
    "sbp",
    "shock_index",
    "urine24",
    "uo_6h",
    "mech_vent",
)
_EVENT_TRUE_STRINGS = {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "positive",
    "present",
    "invasive",
    "noninvasive",
    "non-invasive",
}
_EVENT_FALSE_STRINGS = {"0", "false", "f", "no", "n", "negative", "absent"}
_MAX_DISCOVERY_QUERIES = 6
_DISCOVERY_QUERY_STRATA = (
    "broad_icu",
    "concept_definition_or_validation",
    "direct_observational_comparator_all_years",
    "direct_observational_comparator_recent",
    "review_or_guideline",
    "critical_care_database",
)
_PRIOR_ART_QUERY_STRATA = (
    "clinical_landscape",
    "candidate_topic",
    "direct_observational_candidates",
    "review_or_guideline",
    "critical_care_database",
)


class IdeaMiningWebError(ValueError):
    """Raised when the web idea-mining adapter must fail closed."""

    def __init__(self, detail: Dict[str, Any]):
        self.detail = detail
        super().__init__(str(detail.get("error") or "idea_mining_error"))


class UnsafeURL(ValueError):
    """Raised before a user URL can reach a local or non-routable address."""


def _request_bool(body: Dict[str, Any], key: str, default: bool = False) -> bool:
    try:
        return parse_bool(body.get(key), default=default)
    except ValueError as exc:
        raise IdeaMiningWebError(
            {
                "error": "invalid_boolean",
                "field": key,
                "reason": f"{key} must be an explicit true or false value.",
            }
        ) from exc


def ingest_pdf_source(body: Dict[str, Any]) -> Dict[str, Any]:
    """Extract bounded local metadata from a selected PDF file.

    The PDF bytes come from the browser file picker to the local FastAPI
    process.  We compute metadata and a short excerpt, then discard the bytes.
    """
    filename = _clean(body.get("filename") or body.get("name") or "source.pdf", 240)
    encoded = str(body.get("content_base64") or body.get("base64") or "").strip()
    if not encoded:
        raise IdeaMiningWebError(
            {
                "error": "pdf_file_required",
                "reason": "Choose a local PDF file before ingesting a PDF source.",
            }
        )
    if len(encoded) > _MAX_PDF_BASE64_CHARS:
        raise IdeaMiningWebError(
            {
                "error": "pdf_too_large",
                "reason": f"Selected PDF is larger than the local bounded parser limit ({_MAX_PDF_BYTES // (1024 * 1024)} MB).",
            }
        )
    try:
        pdf_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise IdeaMiningWebError(
            {
                "error": "invalid_pdf_payload",
                "reason": "The selected PDF could not be decoded by the local server.",
            }
        ) from exc
    record = _extract_pdf_bytes(pdf_bytes, filename=filename)
    suggestion = _suggestion_from_pdf_record(record)
    payload = {
        "ok": True,
        "mode": "local_pdf_ingest",
        "pdf": record,
        "suggested_payload": suggestion,
        "source_adapter": {
            "status": "local_pdf_excerpt_ready",
            "source_type": "pdf",
            "network_calls": 0,
            "external_llm_calls": 0,
            "full_text_stored": False,
            "reason": "A local PDF was parsed on this machine; only a bounded excerpt and file hash are returned.",
        },
        "privacy": {
            "source_text_stored": False,
            "full_text_stored": False,
            "patient_rows_returned": False,
            "network_calls": 0,
            "external_llm_calls": 0,
            "uploads": 0,
        },
    }
    _assert_no_row_payload(payload)
    return payload


def scan_literature_folder(body: Dict[str, Any]) -> Dict[str, Any]:
    """Scan a local literature folder for PDFs without persisting full text."""
    raw_path = str(body.get("path") or body.get("folder") or "").strip()
    if not raw_path:
        raise IdeaMiningWebError(
            {
                "error": "literature_folder_required",
                "reason": "Choose a local folder that contains downloaded papers.",
            }
        )
    folder = Path(raw_path).expanduser()
    try:
        folder = folder.resolve()
    except OSError:
        pass
    if not folder.exists() or not folder.is_dir():
        raise IdeaMiningWebError(
            {
                "error": "literature_folder_not_found",
                "reason": "The selected literature folder does not exist on this machine.",
                "path": str(folder),
            }
        )
    pdfs: List[Path] = []
    try:
        for item in folder.rglob("*.pdf"):
            if len(pdfs) >= _MAX_LITERATURE_PDFS:
                break
            if any(part.startswith(".") for part in item.parts):
                continue
            # Never follow a symlinked PDF out of the selected folder: the
            # scan hashes and excerpts file bytes, so a link could otherwise
            # make the server read any local PDF.
            if item.is_symlink():
                continue
            if item.is_file():
                pdfs.append(item)
    except PermissionError as exc:
        raise IdeaMiningWebError(
            {
                "error": "literature_folder_permission_denied",
                "reason": "EasyICU could not read this local literature folder.",
                "path": str(folder),
            }
        ) from exc
    documents: List[Dict[str, Any]] = []
    representative: Optional[Dict[str, Any]] = None
    for pdf in sorted(pdfs, key=lambda p: p.name.lower()):
        meta = _pdf_file_record(pdf, extract_excerpt=representative is None)
        documents.append(meta)
        if representative is None and meta.get("excerpt"):
            representative = meta
    suggestion = (
        _suggestion_from_pdf_record(representative or documents[0]) if documents else {}
    )
    payload = {
        "ok": True,
        "mode": "local_literature_folder",
        "folder": {
            "path": str(folder),
            "name": folder.name,
            "pdf_count": len(pdfs),
            "returned": len(documents),
            "truncated": len(pdfs) >= _MAX_LITERATURE_PDFS,
        },
        "documents": documents[:20],
        "representative": representative,
        "suggested_payload": suggestion,
        "source_adapter": {
            "status": (
                "local_literature_folder_scanned" if documents else "no_pdf_found"
            ),
            "source_type": "literature_folder",
            "network_calls": 0,
            "external_llm_calls": 0,
            "full_text_stored": False,
            "reason": "Scanned local PDF filenames and one bounded representative excerpt; no full-text library was persisted.",
        },
        "privacy": {
            "source_text_stored": False,
            "full_text_stored": False,
            "patient_rows_returned": False,
            "network_calls": 0,
            "external_llm_calls": 0,
            "uploads": 0,
        },
    }
    _assert_no_row_payload(payload)
    return payload


def mine_ideas(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a local idea-mining run from user-supplied source metadata.

    The source body is not persisted.  We store citation metadata, a bounded
    evidence quote, hashes, ledger rows, feasibility decisions, and aggregate
    active-export statistics.
    """
    _require_source_seed(body)
    source = _source_record(body)
    text = _source_text(body)
    concept_hits = _match_concepts(text)
    # Conversational Idea Mining is allowed before a project binds any data.
    # In that mode, do not inherit the legacy Web workspace's global active
    # export: it may belong to another project and would make a metadata-only
    # conversation claim source-specific feasibility it never established.
    export = None if _request_bool(body, "metadata_only") else _active_export()
    export_index = _export_index(export)
    idea = _idea_from_source(source, text, concept_hits, export_index)
    pre_experiment = _pre_experiment(idea, export, export_index)
    plan = _handoff_plan(source, idea, pre_experiment)
    payload = {
        "ok": True,
        "mode": "local_metadata_first",
        "schema_version": "easyicu.web_idea_mining/1",
        "run_id": _run_id(source, idea),
        "created_at": _now(),
        "source_evidence": [source],
        "idea_ledger": [idea],
        "selected_idea_id": idea["idea_id"],
        "pre_experiment": pre_experiment,
        "handoff_plan": plan,
        "prior_art": idea["prior_art"],
        "privacy": {
            "source_text_stored": False,
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
            "network_calls": 0,
            "external_llm_calls": 0,
            "uploads": 0,
        },
        "blocked_features": _blocked_features(body),
    }
    _assert_no_row_payload(payload)
    run_dir = _write_run(payload)
    payload["run_dir"] = str(run_dir)
    _record_history(payload)
    return payload


def resolve_source(body: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a user-supplied paper/PDF/frontier seed into bounded metadata.

    URL fetching and journal/PDF discovery are opt-in.  Without opt-in this
    endpoint still returns a parseable source record and the exact blocked
    reason so the UI button is real rather than decorative.
    """
    _require_source_seed(body)
    source = _source_record(body)
    source_type = str(source.get("source_type") or "manual")
    allow_network = _request_bool(body, "allow_network")
    adapter = {
        "status": "metadata_ready",
        "source_type": source_type,
        "network_calls": 0,
        "external_llm_calls": 0,
        "fetch_performed": False,
        "full_text_stored": False,
    }
    supplied_title = _clean(body.get("title") or "", 220)
    supplied_topic = _clean(
        body.get("topic") or body.get("research_question") or "", 600
    )
    suggestion = {
        "topic": supplied_topic,
        "excerpt": _clean(
            body.get("excerpt") or source.get("evidence_quote") or "", _MAX_SOURCE_QUOTE
        ),
        "title": supplied_title or None,
        "journal": source.get("journal"),
        "year": source.get("year"),
        "doi": source.get("doi"),
        "pmid": source.get("pmid"),
        "url": source.get("url"),
        "citation_key": source.get("citation_key"),
        "zotero_key": source.get("zotero_key"),
        "source_origin": source.get("source_origin"),
        "source_origin_label": source.get("source_origin_label"),
    }
    if source_type == "url" and source.get("url") and not allow_network:
        adapter["status"] = "blocked_network_opt_in_required"
        adapter["reason"] = (
            "URL fetch is disabled until the user explicitly enables network resolution for this source."
        )
    elif source_type == "url" and source.get("url") and allow_network:
        fetched = _fetch_url_metadata(str(source["url"]))
        adapter.update(
            {
                "status": fetched.get("status"),
                "network_calls": fetched.get("network_calls", 0),
                "fetch_performed": fetched.get("network_calls", 0) > 0,
                "metadata_source": fetched.get("metadata_source"),
                "reason": fetched.get("reason"),
            }
        )
        if fetched.get("title") and not suggestion.get("title"):
            suggestion["title"] = fetched["title"]
            source["title"] = fetched["title"]
        if fetched.get("title") and not suggestion.get("topic"):
            suggestion["topic"] = fetched["title"]
        if fetched.get("journal") and not suggestion.get("journal"):
            suggestion["journal"] = fetched["journal"]
            source["journal"] = fetched["journal"]
        if fetched.get("year") and not suggestion.get("year"):
            suggestion["year"] = fetched["year"]
            source["year"] = fetched["year"]
        if fetched.get("description") and not suggestion.get("excerpt"):
            suggestion["excerpt"] = fetched["description"]
            source["evidence_quote"] = fetched["description"]
        if fetched.get("doi") and not suggestion.get("doi"):
            suggestion["doi"] = fetched["doi"]
            source["doi"] = fetched["doi"]
    elif source_type == "pdf" and body.get("source_file_sha256"):
        adapter["status"] = "local_pdf_excerpt_ready"
        adapter["reason"] = (
            "The selected local PDF has been parsed into a bounded excerpt and hash."
        )
    elif source_type == "pdf" and not suggestion.get("excerpt"):
        adapter["status"] = "blocked_pdf_excerpt_required"
        adapter["reason"] = "Choose a local PDF file or paste a bounded excerpt first."
    elif source_type == "literature_folder":
        adapter["status"] = (
            "local_literature_folder_ready"
            if body.get("literature_pdf_count")
            else "local_literature_folder_empty"
        )
        adapter["reason"] = (
            "The selected local literature folder is represented by PDF metadata and a bounded excerpt."
        )
    elif source_type == "frontier":
        adapter["status"] = "search_plan_ready"
        adapter["reason"] = (
            "The topic is ready for an opt-in prior-art/literature search stage."
        )
    payload = {
        "ok": True,
        "resolved_source": source,
        "suggested_payload": suggestion,
        "source_adapter": adapter,
        "privacy": {
            "source_text_stored": False,
            "full_text_stored": False,
            "patient_rows_returned": False,
            "network_calls": adapter["network_calls"],
            "external_llm_calls": 0,
        },
    }
    _assert_no_row_payload(payload)
    return payload


def discover_literature(body: Dict[str, Any]) -> Dict[str, Any]:
    """Search opt-in PubMed metadata and turn articles into idea candidates.

    This is the "go find frontier papers" path.  It is deliberately not a
    model call: the adapter searches bounded public metadata/abstracts only
    after per-request network opt-in, then maps source sentences to the
    EasyICU dictionary and the active export.  Without opt-in it returns the
    exact query bundle and a blocked status, not seeded fake articles.
    """
    topic = _clean(
        body.get("topic") or body.get("research_question") or body.get("title") or "",
        220,
    )
    journal = _clean(body.get("journal") or "", 120)
    allow_network = _request_bool(body, "allow_network")
    limit = max(1, min(int(body.get("limit") or 8), 20))
    if not topic:
        raise IdeaMiningWebError(
            {
                "error": "frontier_topic_required",
                "reason": "Describe the ICU topic, journal scope, review theme, DOI, or article clue before literature discovery.",
            }
        )
    queries = _discovery_queries(topic, journal, body)
    if not allow_network:
        out = {
            "ok": True,
            "mode": "frontier_literature_discovery",
            "status": "blocked_network_opt_in_required",
            "search_performed": False,
            "queries_to_run": queries,
            "source_candidates": [],
            "idea_candidates": [],
            "suggested_payload": {},
            "reason": "Literature discovery requires explicit per-source network opt-in. No request was made.",
            "privacy": {
                "source_text_stored": False,
                "full_text_stored": False,
                "patient_rows_returned": False,
                "network_calls": 0,
                "external_llm_calls": 0,
                "uploads": 0,
            },
        }
        _assert_no_row_payload(out)
        return out

    errors: List[str] = []
    network_calls = 0
    query_hits: List[Tuple[str, str, List[str]]] = []
    for index, query in enumerate(queries[:_MAX_DISCOVERY_QUERIES]):
        stratum = (
            _DISCOVERY_QUERY_STRATA[index]
            if index < len(_DISCOVERY_QUERY_STRATA)
            else f"prespecified_{index + 1}"
        )
        try:
            found = _pubmed_esearch(query, limit=limit)
            network_calls += 1
            query_hits.append((stratum, query, found))
        except Exception as exc:
            query_hits.append((stratum, query, []))
            errors.append(str(exc)[:240])
    ids, retrieval_by_pmid, query_strata = _stratified_pubmed_ids(
        query_hits,
        limit=limit,
    )
    typed_search_scope = {**body, "topic": topic}
    focus_terms = direct_evidence_search.focus_terms(typed_search_scope)
    articles: List[Dict[str, Any]] = []
    if ids:
        try:
            articles = _pubmed_article_records(ids, focus_terms=focus_terms)
            # _pubmed_article_records issues a single efetch request; the counter
            # must reflect that (the privacy audit over-reported 2 calls).
            network_calls += 1
        except Exception as exc:
            errors.append(str(exc)[:240])
            try:
                articles = _pubmed_esummary(ids)
                network_calls += 1
            except Exception as inner:
                errors.append(str(inner)[:240])
                articles = []

    # A broad topic search can honestly return only definitions, methods, or
    # papers that use the exposure as an eligibility label.  While the same
    # user-authorized network action is still active, run one bounded exact
    # comparator stratum only when source-backed screening found no eligible
    # P/E/O record.  The Research Agent will independently re-screen the
    # retained records against its sealed ResearchContext; this retrieval-stage
    # decision can schedule the fallback but cannot grant publication authority.
    preliminary = [
        direct_evidence_search.screen_article(article, typed_search_scope)
        for article in articles
    ]
    if direct_evidence_search.scope_complete(typed_search_scope) and not any(
        row.get("disposition") == "include" for row in preliminary
    ):
        fallback_query = direct_evidence_search.build_query(typed_search_scope)
        fallback_ids: List[str] = []
        try:
            fallback_ids = _pubmed_esearch(
                fallback_query,
                limit=min(20, max(12, limit * 2)),
            )
            network_calls += 1
        except Exception as exc:
            errors.append(str(exc)[:240])
        existing_ids = {str(row.get("pmid") or "") for row in articles}
        retained_fallback_ids = [
            pmid for pmid in fallback_ids if pmid not in existing_ids
        ][:limit]
        fallback_articles: List[Dict[str, Any]] = []
        if retained_fallback_ids:
            try:
                fallback_articles = _pubmed_article_records(
                    retained_fallback_ids,
                    focus_terms=focus_terms,
                )
                network_calls += 1
            except Exception as exc:
                errors.append(str(exc)[:240])
        fallback_decisions = [
            direct_evidence_search.screen_article(article, typed_search_scope)
            for article in fallback_articles
        ]
        included_fallback = [
            article
            for article, decision in zip(fallback_articles, fallback_decisions)
            if decision.get("disposition") == "include"
        ]
        related_fallback = [
            article
            for article, decision in zip(fallback_articles, fallback_decisions)
            if decision.get("disposition") != "include"
        ]
        # Keep the public ``limit`` contract while giving a valid exact
        # comparator priority over broad-query returns.  An excluded fallback
        # is only related context and must not erase the prespecified stratum
        # sample (definition, recent, review and database evidence) merely
        # because it came from the last query.
        articles = _dedupe_articles([*included_fallback, *articles, *related_fallback])[
            :limit
        ]
        fallback_retained = {
            str(row.get("pmid") or "")
            for row in articles
            if str(row.get("pmid") or "") in set(fallback_ids)
        }
        for pmid in fallback_ids:
            receipt = retrieval_by_pmid.setdefault(
                pmid,
                {"queries": [], "strata": []},
            )
            if fallback_query not in receipt["queries"]:
                receipt["queries"].append(fallback_query)
            if (
                direct_evidence_search.DIRECT_COMPARATOR_FALLBACK_STRATUM
                not in receipt["strata"]
            ):
                receipt["strata"].append(
                    direct_evidence_search.DIRECT_COMPARATOR_FALLBACK_STRATUM
                )
        query_strata.append(
            {
                "id": direct_evidence_search.DIRECT_COMPARATOR_FALLBACK_STRATUM,
                "query": fallback_query,
                "returned_count": len(fallback_ids),
                "retained_count": sum(
                    1 for pmid in fallback_ids if pmid in fallback_retained
                ),
            }
        )
        queries.append(fallback_query)

    export = _active_export()
    export_index = _export_index(export)
    source_candidates: List[Dict[str, Any]] = []
    idea_candidates: List[Dict[str, Any]] = []
    for article in articles:
        pmid = str(article.get("pmid") or "").strip()
        retrieval = retrieval_by_pmid.get(pmid) or {
            "queries": [],
            "strata": [],
        }
        source = _source_record(
            {
                "source_type": "pubmed",
                "topic": topic,
                "title": article.get("title"),
                "journal": article.get("journal"),
                "year": article.get("year"),
                "doi": article.get("doi"),
                "pmid": article.get("pmid"),
                "url": (
                    f"https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid')}/"
                    if article.get("pmid")
                    else ""
                ),
                "excerpt": article.get("evidence_sentence")
                or article.get("abstract_excerpt")
                or article.get("title"),
                "abstract": article.get("abstract_excerpt")
                or article.get("evidence_sentence")
                or "",
            }
        )
        source["design_excerpt"] = article.get("design_excerpt") or ""
        source["publication_types"] = list(article.get("publication_types") or [])
        text = "\n".join(
            [
                topic,
                str(article.get("title") or ""),
                str(article.get("abstract_excerpt") or ""),
                str(article.get("evidence_sentence") or ""),
            ]
        )
        hits = _match_concepts(text)
        idea = _idea_from_source(source, text, hits, export_index)
        source["discovery_rank"] = len(source_candidates) + 1
        source["pubmed_metadata_only"] = True
        source["matched_queries"] = list(retrieval["queries"])
        source["matched_query_strata"] = list(retrieval["strata"])
        source["direct_comparator_screen"] = direct_evidence_search.screen_article(
            article, typed_search_scope
        )
        source_candidates.append(source)
        idea_candidates.append(
            {
                "rank": len(idea_candidates) + 1,
                "source_id": source.get("source_id"),
                "idea": idea,
                "source": source,
                "suggested_payload": {
                    "source_type": "pubmed",
                    "topic": idea.get("idea_title") or topic,
                    "title": source.get("title"),
                    "journal": source.get("journal"),
                    "year": source.get("year"),
                    "doi": source.get("doi"),
                    "pmid": source.get("pmid"),
                    "url": source.get("url"),
                    "excerpt": source.get("evidence_quote"),
                },
            }
        )
    status = (
        "searched"
        if idea_candidates
        else ("search_failed" if errors else "searched_no_hits")
    )
    out = {
        "ok": True,
        "mode": "frontier_literature_discovery",
        "status": status,
        "search_performed": True,
        "searched_at": _now(),
        "queries_to_run": queries,
        "query_strata": query_strata,
        "network_calls": network_calls,
        "source_candidates": source_candidates,
        "idea_candidates": idea_candidates,
        "suggested_payload": (
            idea_candidates[0].get("suggested_payload") if idea_candidates else {}
        ),
        "errors": errors,
        "reason": "PubMed metadata/abstract discovery only; no full text, external LLM, or patient rows were used.",
        "privacy": {
            "source_text_stored": False,
            "full_text_stored": False,
            "patient_rows_returned": False,
            "network_calls": network_calls,
            "external_llm_calls": 0,
            "uploads": 0,
        },
    }
    _assert_no_row_payload(out)
    return out


def check_prior_art(body: Dict[str, Any]) -> Dict[str, Any]:
    """Run or prepare a bounded prior-art check for an idea.

    The default path is fail-closed and returns a query bundle.  When
    ``allow_network`` is true, the adapter performs a small PubMed metadata
    search using the public E-utilities endpoint and stores only citation
    metadata/snippets, never full text.
    """
    run_id = str(body.get("run_id") or "").strip()
    idea_id = str(body.get("idea_id") or "").strip()
    payload = _load_run(run_id) if run_id else None
    legacy_prior: Optional[Dict[str, Any]] = None
    if run_id and payload is None:
        # Retained pre-ledger review directories remain usable only when the
        # persisted receipt proves the exact requested run/idea identity.
        legacy_prior = _load_prior_art(run_id)
        if legacy_prior is None:
            raise IdeaMiningWebError({"error": "idea_run_not_found", "run_id": run_id})
        if str(legacy_prior.get("run_id") or "").strip() != run_id:
            raise IdeaMiningWebError({"error": "idea_run_not_found", "run_id": run_id})
        if not idea_id or str(legacy_prior.get("idea_id") or "").strip() != idea_id:
            raise IdeaMiningWebError(
                {"error": "idea_not_found", "idea_id": idea_id, "run_id": run_id}
            )

    idea = _selected_idea(payload or {}, idea_id)
    if run_id and payload is not None and idea is None:
        raise IdeaMiningWebError(
            {
                "error": "idea_not_found",
                "idea_id": idea_id,
                "run_id": run_id,
            }
        )
    source = ((payload or {}).get("source_evidence") or [{}])[0]
    if legacy_prior is not None:
        source = _source_record(body)
        persisted_prior = legacy_prior.get("prior_art") or {}
        idea = {
            "idea_id": idea_id,
            "idea_title": _clean(
                body.get("idea_title") or body.get("topic") or idea_id,
                180,
            ),
            "mapped_concepts": [],
            "prior_art": {
                "queries_to_run": persisted_prior.get("queries_to_run") or []
            },
        }
    if not idea:
        source = _source_record(body)
        idea = {
            "idea_id": _slug(body.get("idea_id") or source.get("title") or "idea"),
            "idea_title": _clean(
                body.get("idea_title")
                or body.get("topic")
                or source.get("title")
                or "ICU idea",
                180,
            ),
            "mapped_concepts": [],
        }
    idea_prior_art = (
        idea.get("prior_art") if isinstance(idea.get("prior_art"), dict) else {}
    )
    prespecified_queries = [
        str(query).strip()
        for query in idea_prior_art.get("queries_to_run") or []
        if str(query).strip()
    ]
    # Mine once, execute exactly what was shown to the user. Re-inferring the
    # topic from an Idea title can select a secondary/descriptive concept and
    # silently search a different scientific question.
    queries = prespecified_queries[:_MAX_DISCOVERY_QUERIES] or _prior_art_queries(
        source,
        str(idea.get("idea_title") or "ICU idea"),
        exposure=idea.get("exposure_or_predictor"),
        outcome=idea.get("outcome"),
    )
    allow_network = _request_bool(body, "allow_network")
    if not allow_network:
        prior = {
            "status": "blocked_network_opt_in_required",
            "search_performed": False,
            "queries_to_run": queries,
            "results": [],
            "public_database_used_by_prior_work": "unknown_until_search",
            "reason": "Prior-art interpretation needs explicit network opt-in. No request was made.",
        }
        network_calls = 0
    else:
        prior = _pubmed_prior_art(queries, source=source)
        network_calls = int(prior.get("network_calls") or 0)
    out = {
        "ok": True,
        "run_id": run_id or None,
        "idea_id": idea.get("idea_id"),
        "prior_art": prior,
        "privacy": {
            "source_text_stored": False,
            "full_text_stored": False,
            "patient_rows_returned": False,
            "network_calls": network_calls,
            "external_llm_calls": 0,
        },
    }
    _assert_no_row_payload(out)
    if run_id:
        run_dir = _run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Never clobber a successful persisted review with a blocked
        # placeholder: the opt-in checkbox resets between visits, so a casual
        # re-check without network opt-in would otherwise re-block a seed
        # whose prior-art review already completed.
        existing_prior = (_load_prior_art(run_id) or {}).get("prior_art") or {}
        existing_reviewed = bool(existing_prior.get("search_performed")) and (
            str(existing_prior.get("status") or "") != "search_failed"
        )
        new_blocked = not bool(prior.get("search_performed"))
        if new_blocked and existing_reviewed:
            out["persisted"] = False
            out["retained_prior_art_status"] = str(existing_prior.get("status") or "")
        else:
            (run_dir / "prior_art_check.json").write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    return out


def prior_art_receipt_binding(run_id: str) -> Optional[Dict[str, Any]]:
    """Bind the fixed prior-art artifact for an Idea Mining run, if complete."""

    clean_run_id = str(run_id or "").strip()
    if not clean_run_id:
        return None
    try:
        return build_prior_art_binding(_run_dir(clean_run_id) / "prior_art_check.json")
    except PriorArtReceiptError as exc:
        raise IdeaMiningWebError({"error": exc.code, "reason": exc.message}) from exc


def load_bound_prior_art_literature(
    binding: Dict[str, Any],
    *,
    research_question: str,
) -> Optional[Dict[str, Any]]:
    """Load a verified Agent literature seed from an accepted Idea handoff."""

    if not str(binding.get("prior_art_sha256") or "").strip():
        return None
    run_id = str(binding.get("run_id") or "").strip()
    if not run_id:
        raise IdeaMiningWebError(
            {
                "error": "prior_art_binding_run_required",
                "reason": "The accepted prior-art binding has no Idea Mining run id.",
            }
        )
    try:
        return _load_bound_prior_art_literature(
            _run_dir(run_id) / "prior_art_check.json",
            binding=binding,
            research_question=research_question,
        )
    except PriorArtReceiptError as exc:
        raise IdeaMiningWebError({"error": exc.code, "reason": exc.message}) from exc


def plan_idea(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create or revise the study plan draft for an idea-mining run.

    This is the missing middle stage between an idea ledger and an Agent
    project seed.  It deliberately produces a metadata-only plan artifact: no
    Agent run, no row-level data, no manuscript claim.
    """

    run_id = str(body.get("run_id") or "").strip()
    idea_id = str(body.get("idea_id") or "").strip()
    payload = _load_run(run_id)
    if not payload:
        raise IdeaMiningWebError({"error": "idea_run_not_found", "run_id": run_id})
    idea = _selected_idea(payload, idea_id)
    if not idea:
        raise IdeaMiningWebError({"error": "idea_not_found", "idea_id": idea_id})
    edits = str(body.get("plan_edits") or "").strip()
    plan_fields = _confirmed_plan_fields(body.get("plan_fields"))
    if not plan_fields:
        existing_plan = _plan_payload(_load_plan(run_id))
        existing_confirmed = existing_plan.get("confirmed_plan_fields")
        plan_fields = _confirmed_plan_fields(existing_confirmed)
    mode = str(body.get("mode") or ("replan" if edits else "plan")).strip().lower()
    source = ((payload.get("source_evidence") or [{}])[0]) or {}
    pre = payload.get("pre_experiment") or {}
    prior = _load_prior_art(run_id)
    plan = _analysis_plan_draft(
        source,
        idea,
        pre,
        prior,
        edits=edits,
        mode=mode,
        plan_fields=plan_fields,
    )
    try:
        readiness = idea_execution_readiness_binding(run_id, idea_id)
    except IdeaMiningWebError:
        readiness = {}
    if readiness:
        selected = _idea_with_readiness_overlay(idea, readiness)
        plan["active_export_contract"] = {
            "status": "ready",
            "source_id": readiness.get("source_id"),
            "source_path_hash": readiness.get("source_path_hash"),
            "feature_count": len(readiness.get("concept_modules") or {}),
            "reportable": False,
        }
        plan["execution_gate"] = _execution_gate(
            selected,
            pre,
            prior,
            readiness,
        )
        plan["execution_readiness"] = {
            key: readiness.get(key)
            for key in (
                "prior_art_decision",
                "source_feasibility_status",
                "idea_definition_sha256",
                "prior_art_adjudication_sha256",
                "source_feasibility_sha256",
                "execution_ready_for_confirmation",
            )
            if readiness.get(key) is not None
        }
        plan["prior_art_adjudication"] = (
            readiness.get("prior_art_adjudication_summary") or {}
        )
        plan["source_feasibility_summary"] = (
            readiness.get("source_feasibility_summary") or {}
        )
    out = {
        "ok": True,
        "schema_version": "easyicu.web_idea_plan/1",
        "created_at": _now(),
        "run_id": run_id,
        "idea_id": idea.get("idea_id"),
        "mode": mode if mode in {"plan", "replan"} else "plan",
        "planner": {
            "stage": "idea_mining_plan_before_agent",
            "engine": "local_agent_style_planner",
            "uses_research_agent_contract": True,
            "agent_run_created": False,
            "draft_unlocked": False,
            "notes": (
                "This is a pre-Agent plan draft assembled from the idea ledger, "
                "active-export feasibility summary, and optional prior-art metadata."
            ),
        },
        "plan": plan,
        "privacy": {
            "source_text_stored": False,
            "full_text_stored": False,
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
            "network_calls": 0,
            "external_llm_calls": 0,
            "agent_run_created": False,
            "reportable": False,
            "draft_unlocked": False,
        },
    }
    _assert_no_row_payload(out)
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "idea_plan.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def _confirmed_plan_fields(value: Any) -> Dict[str, str]:
    """Validate bounded, researcher-confirmed fields for one Idea Plan."""

    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise IdeaMiningWebError(
            {
                "error": "idea_plan_fields_invalid",
                "reason": "Confirmed Idea Plan fields must be a JSON object.",
            }
        )
    limits = {
        "research_question": 1200,
        "population": 500,
        "exposure": 500,
        "outcome": 800,
        "time_zero": 500,
        "time_window": 500,
    }
    unexpected = sorted(set(value) - set(limits))
    if unexpected:
        raise IdeaMiningWebError(
            {
                "error": "idea_plan_fields_invalid",
                "reason": "Confirmed Idea Plan fields contain unsupported keys.",
                "fields": unexpected,
            }
        )
    return {
        key: _clean(raw, limits[key])
        for key, raw in value.items()
        if str(raw or "").strip()
    }


def _canonical_payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _confirmed_definition(run_id: str, idea_id: str) -> Dict[str, Any]:
    payload = _load_run(run_id)
    if not payload:
        raise IdeaMiningWebError({"error": "idea_run_not_found", "run_id": run_id})
    idea = _selected_idea(payload, idea_id)
    if not idea:
        raise IdeaMiningWebError(
            {"error": "idea_not_found", "run_id": run_id, "idea_id": idea_id}
        )
    plan = _plan_payload(_load_plan(run_id))
    confirmed = plan.get("confirmed_plan_fields")
    confirmed = confirmed if isinstance(confirmed, Mapping) else {}
    fields = {
        key: str(confirmed.get(key) or "").strip()
        for key in _CONFIRMED_DEFINITION_FIELDS
    }
    missing = [key for key, value in fields.items() if not value]
    if missing:
        raise IdeaMiningWebError(
            {
                "error": "idea_definition_confirmation_required",
                "reason": (
                    "Confirm the research question, population, exposure, outcome, "
                    "time zero, and observation window before adjudicating literature."
                ),
                "missing_fields": missing,
            }
        )
    snapshot = {
        "run_id": run_id,
        "idea_id": str(idea.get("idea_id") or ""),
        "fields": fields,
    }
    return {**snapshot, "definition_sha256": _canonical_payload_sha256(snapshot)}


def adjudicate_prior_art(body: Dict[str, Any]) -> Dict[str, Any]:
    """Persist one human-confirmed, digest-bound prior-art decision.

    Retrieval remains owned by ``check_prior_art`` and direct-comparator
    eligibility remains owned by ``direct_evidence_search``.  This function
    records only the researcher's top-level decision against those exact
    artifacts; it does not infer novelty from result counts.
    """

    run_id = str(body.get("run_id") or "").strip()
    idea_id = str(body.get("idea_id") or "").strip()
    decision = str(body.get("decision") or "").strip().lower()
    rationale = _clean(body.get("rationale"), 1200)
    if decision not in _PRIOR_ART_DECISIONS:
        raise IdeaMiningWebError(
            {
                "error": "idea_prior_art_decision_invalid",
                "allowed": sorted(_PRIOR_ART_DECISIONS),
            }
        )
    if not rationale:
        raise IdeaMiningWebError(
            {
                "error": "idea_prior_art_rationale_required",
                "reason": "A bounded researcher rationale is required for prior-art adjudication.",
            }
        )
    definition = _confirmed_definition(run_id, idea_id)
    prior_binding = prior_art_receipt_binding(run_id)
    if not prior_binding:
        raise IdeaMiningWebError(
            {
                "error": "idea_prior_art_receipt_required",
                "reason": "Run a completed, dated literature search before adjudication.",
            }
        )
    prior_art = _load_prior_art(run_id) or {}
    prior = prior_art.get("prior_art")
    prior = prior if isinstance(prior, Mapping) else {}
    results = [
        row for row in list(prior.get("results") or []) if isinstance(row, Mapping)
    ][:20]
    if decision == "differentiated" and not results:
        raise IdeaMiningWebError(
            {
                "error": "idea_prior_art_differentiation_unsubstantiated",
                "reason": (
                    "A zero-result metadata search cannot establish differentiation. "
                    "Refine or broaden the search and adjudicate again."
                ),
            }
        )
    screened = []
    for row in results:
        screen = row.get("direct_comparator_screen")
        screen = screen if isinstance(screen, Mapping) else {}
        screened.append(
            {
                "pmid": str(row.get("pmid") or "")[:32] or None,
                "title": _clean(row.get("title"), 500),
                "disposition": str(screen.get("disposition") or "exclude")[:24],
                "evidence_role": str(screen.get("evidence_role") or "related_context")[
                    :48
                ],
                "rationale": _clean(screen.get("rationale"), 500),
                "population_match": bool(screen.get("population_match")),
                "exposure_match": bool(screen.get("exposure_match")),
                "outcome_match": bool(screen.get("outcome_match")),
                "publication_type_eligible": bool(
                    screen.get("publication_type_eligible", True)
                ),
            }
        )
    direct = [row for row in screened if row["disposition"] == "include"]
    axes = [
        {
            "axis": "population_and_setting",
            "matched_record_count": sum(
                1 for row in screened if row["population_match"]
            ),
        },
        {
            "axis": "exposure_and_time_zero",
            "matched_record_count": sum(1 for row in screened if row["exposure_match"]),
        },
        {
            "axis": "outcome_and_estimand",
            "matched_record_count": sum(1 for row in screened if row["outcome_match"]),
        },
        {
            "axis": "analysis_and_robustness",
            "status": "requires_full_text_or_human_review",
        },
        {
            "axis": "data_source_and_transportability",
            "status": "requires_full_text_or_human_review",
        },
        {
            "axis": "clinical_contribution",
            "status": "researcher_adjudicated",
        },
    ]
    out = {
        "ok": True,
        "schema_version": "easyicu.web_idea_prior_art_adjudication/1",
        "created_at": _now(),
        "run_id": run_id,
        "idea_id": idea_id,
        "decision": decision,
        "rationale": rationale,
        "definition_sha256": definition["definition_sha256"],
        "confirmed_definition": definition["fields"],
        "prior_art_binding": prior_binding,
        "screening": {
            "retrieval_candidate_count": len(results),
            "direct_comparator_count": len(direct),
            "records": screened,
        },
        "comparison_axes": axes,
        "authority": {
            "human_confirmed": True,
            "retrieval_is_not_evidence": True,
            "full_text_screening_complete": False,
            "paper_authorized": False,
        },
    }
    _assert_no_row_payload(out)
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / _PRIOR_ART_ADJUDICATION_FILENAME
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def prior_art_adjudication_binding(run_id: str, idea_id: str) -> Dict[str, Any]:
    path = _run_dir(run_id) / _PRIOR_ART_ADJUDICATION_FILENAME
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except FileNotFoundError as exc:
        raise IdeaMiningWebError(
            {
                "error": "idea_prior_art_adjudication_required",
                "reason": "A current typed prior-art adjudication is required.",
            }
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdeaMiningWebError(
            {"error": "idea_prior_art_adjudication_invalid"}
        ) from exc
    if not isinstance(payload, Mapping):
        raise IdeaMiningWebError({"error": "idea_prior_art_adjudication_invalid"})
    if (
        str(payload.get("run_id") or "") != run_id
        or str(payload.get("idea_id") or "") != idea_id
    ):
        raise IdeaMiningWebError(
            {"error": "idea_prior_art_adjudication_identity_mismatch"}
        )
    current_definition = _confirmed_definition(run_id, idea_id)
    if payload.get("definition_sha256") != current_definition["definition_sha256"]:
        raise IdeaMiningWebError(
            {
                "error": "idea_prior_art_adjudication_stale_definition",
                "reason": "The confirmed study definition changed after adjudication.",
            }
        )
    current_prior = prior_art_receipt_binding(run_id)
    if not current_prior or payload.get("prior_art_binding") != current_prior:
        raise IdeaMiningWebError(
            {
                "error": "idea_prior_art_adjudication_stale_literature",
                "reason": "The literature receipt changed after adjudication.",
            }
        )
    decision = str(payload.get("decision") or "")
    if decision not in _PRIOR_ART_DECISIONS:
        raise IdeaMiningWebError({"error": "idea_prior_art_adjudication_invalid"})
    return {
        "prior_art_adjudication_schema_version": str(payload.get("schema_version")),
        "prior_art_adjudication_sha256": hashlib.sha256(raw).hexdigest(),
        "prior_art_decision": decision,
        "prior_art_adjudicated_at": str(payload.get("created_at") or ""),
        "idea_definition_sha256": str(payload.get("definition_sha256") or ""),
        "prior_art_adjudication_summary": {
            "decision": decision,
            "rationale": str(payload.get("rationale") or "")[:1200],
            "screening": (
                payload.get("screening")
                if isinstance(payload.get("screening"), Mapping)
                else {}
            ),
            "comparison_axes": list(payload.get("comparison_axes") or [])[:6],
            "authority": (
                payload.get("authority")
                if isinstance(payload.get("authority"), Mapping)
                else {}
            ),
        },
    }


def _validated_concept_bindings(value: Any) -> Dict[str, Any]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise IdeaMiningWebError({"error": "idea_concept_bindings_invalid"})
    allowed = {*_CONCEPT_BINDING_ROLES, "covariates"}
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise IdeaMiningWebError(
            {
                "error": "idea_concept_bindings_invalid",
                "fields": unexpected,
            }
        )

    def concept_id(raw: Any) -> str:
        text = str(raw or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", text):
            raise IdeaMiningWebError({"error": "idea_concept_binding_invalid"})
        return text

    out = {
        role: concept_id(value.get(role))
        for role in _CONCEPT_BINDING_ROLES
        if str(value.get(role) or "").strip()
    }
    raw_covariates = value.get("covariates") or []
    if not isinstance(raw_covariates, list) or len(raw_covariates) > 32:
        raise IdeaMiningWebError({"error": "idea_concept_bindings_invalid"})
    out["covariates"] = list(dict.fromkeys(concept_id(item) for item in raw_covariates))
    return out


def _temporal_answerability(
    stats: List[Dict[str, Any]], bindings: Mapping[str, Any]
) -> Dict[str, bool]:
    by_concept = {
        str(row.get("concept_id") or ""): row for row in stats if row.get("concept_id")
    }
    time_zero = str(bindings.get("time_zero") or "")
    exposure = str(bindings.get("primary_exposure") or "")
    outcome = str(bindings.get("outcome") or "")
    time_zero_ready = bool(
        time_zero
        and time_zero in by_concept
        and by_concept[time_zero].get("status") == "ready"
        and by_concept[time_zero].get("time_orderable")
    )
    ordering_ready = bool(
        exposure
        and outcome
        and exposure in by_concept
        and outcome in by_concept
        and by_concept[exposure].get("time_orderable")
        and by_concept[outcome].get("time_orderable")
    )
    return {
        "time_zero_reconstructable": time_zero_ready,
        "temporal_ordering_reconstructable": ordering_ready,
    }


def _bounded_joint_observed_entities(
    root: Path,
    *,
    concept_to_file: Mapping[str, Any],
    concept_ids: List[str],
    max_records: int,
) -> Optional[int]:
    if len(concept_ids) != 2:
        return None
    entity_sets: List[set[str]] = []
    for concept_id in concept_ids:
        item = concept_to_file.get(concept_id)
        if not isinstance(item, Mapping):
            return None
        file_name = str(item.get("file") or "")
        columns = [str(value) for value in item.get("columns") or []]
        selected = _selected_feature_columns(columns, concept_id)
        if not file_name or "stay_id" not in selected or concept_id not in selected:
            return None
        try:
            frame = _read_bounded_feature_frame(root / file_name, selected, max_records)
        except Exception:
            return None
        if frame.empty:
            entity_sets.append(set())
            continue
        if _is_event_rate_concept(concept_id):
            observed = frame["stay_id"]
        else:
            observed = frame.loc[frame[concept_id].notna(), "stay_id"]
        entity_sets.append(
            {
                entity_id_contract.normalize_entity_id(value)
                for value in observed
                if entity_id_contract.normalize_entity_id(value)
            }
        )
    return len(entity_sets[0] & entity_sets[1])


def _bounded_feasibility_blockers(
    *,
    status: str,
    missing_required: List[str],
    unavailable: List[Dict[str, Any]],
    low: List[Dict[str, Any]],
    demo_like: bool,
    temporal: Mapping[str, bool],
    joint_observed: Optional[int],
    construct_answerability: List[Dict[str, Any]],
) -> List[str]:
    blockers: List[str] = []
    if demo_like:
        blockers.append("demo_source_not_execution_authority")
    if missing_required:
        blockers.append("required_concepts_missing")
    if unavailable:
        blockers.append("bounded_sample_unavailable")
    if low:
        blockers.append("low_concept_coverage")
    if not temporal.get("time_zero_reconstructable"):
        blockers.append("time_zero_not_reconstructable")
    if not temporal.get("temporal_ordering_reconstructable"):
        blockers.append("temporal_ordering_not_reconstructable")
    if joint_observed is None:
        blockers.append("joint_coverage_not_resolved")
    elif joint_observed == 0:
        blockers.append("no_jointly_observed_entities")
    if any(row.get("verdict") == "blocked" for row in construct_answerability):
        blockers.append("research_construct_unavailable")
    if any(row.get("verdict") == "needs_review" for row in construct_answerability):
        blockers.append("research_construct_requires_definition_or_materialization")
    if status == "ready":
        return []
    return list(dict.fromkeys(blockers))


def bounded_sample_feasibility(
    body: Dict[str, Any],
    *,
    export: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run a bounded row-level feasibility pass for one mined idea.

    This upgrades metadata-only concept presence into a capped local sample
    check.  It is still pre-analysis: no raw rows, entity identifiers, or effect
    estimates leave the backend payload.
    """

    run_id = str(body.get("run_id") or "").strip()
    idea_id = str(body.get("idea_id") or "").strip()
    payload = _load_run(run_id)
    if not payload:
        raise IdeaMiningWebError({"error": "idea_run_not_found", "run_id": run_id})
    idea = _selected_idea(payload, idea_id)
    if not idea:
        raise IdeaMiningWebError({"error": "idea_not_found", "idea_id": idea_id})
    adjudication_binding: Optional[Dict[str, Any]] = None
    if _request_bool(body, "require_adjudication"):
        adjudication_binding = prior_art_adjudication_binding(run_id, idea_id)
        if adjudication_binding.get("prior_art_decision") != "differentiated":
            raise IdeaMiningWebError(
                {
                    "error": "idea_prior_art_not_differentiated",
                    "reason": (
                        "Only a current differentiated prior-art adjudication can "
                        "advance to real-data feasibility."
                    ),
                    "decision": adjudication_binding.get("prior_art_decision"),
                }
            )
    export = export or _active_export()
    if not export:
        raise IdeaMiningWebError(
            {
                "error": "active_export_required",
                "reason": "Select a real EasyICU export before running sample feasibility.",
            }
        )

    source, desc = export
    export_index = _export_index(export)
    concept_to_file = export_index.get("concept_to_file") or {}
    bindings = _validated_concept_bindings(body.get("concept_bindings"))
    bound_concepts = [
        bindings.get(role) for role in _CONCEPT_BINDING_ROLES if bindings.get(role)
    ]
    bound_concepts.extend(bindings.get("covariates") or [])
    required = list(
        dict.fromkeys(
            str(value or "").strip()
            for value in (
                bound_concepts
                or [
                    row.get("concept_id")
                    for row in idea.get("mapped_concepts") or []
                    if isinstance(row, Mapping)
                ]
            )
            if str(value or "").strip()
        )
    )
    concepts = [cid for cid in required if cid in concept_to_file]
    max_records = _sample_record_limit(body.get("max_records"))
    root = Path(str(desc.get("path") or source.get("path") or ""))
    entity_ids = export_index.get("entity_ids") or set()
    resolved_denominator, denominator_resolved = _cohort_denominator(export_index, desc)
    denominator = resolved_denominator if denominator_resolved else 1

    stats = [
        row
        for row in (
            _bounded_feature_sample_stat(
                root,
                concept_id=cid,
                item=concept_to_file.get(cid) or {},
                denominator=denominator,
                max_records=max_records,
            )
            for cid in concepts[:_MAX_FEATURE_STATS]
        )
        if row
    ]
    if not denominator_resolved:
        # Cohort denominator unknown: coverage cannot be computed, so it must not
        # read as feasible. Blank coverage and block the verdict for review.
        _mark_denominator_unresolved(stats)
    sampled = {row.get("concept_id") for row in stats}
    missing_required = [cid for cid in required if cid not in sampled]
    unavailable = [row for row in stats if row.get("status") == "sample_unavailable"]
    low = [
        row
        for row in stats
        if row.get("metric_kind") != "event_rate" and row.get("low_coverage")
    ]
    # Schema-only rows are not a row-level sample: their presence must never
    # produce an aggregate "ready" verdict.
    temporal = _temporal_answerability(stats, bindings)
    joint_observed = _bounded_joint_observed_entities(
        root,
        concept_to_file=concept_to_file,
        concept_ids=[
            value
            for value in (bindings.get("primary_exposure"), bindings.get("outcome"))
            if value
        ],
        max_records=max_records,
    )
    construct_text = " ".join(
        str(value or "")
        for value in (
            idea.get("idea_title"),
            idea.get("exposure_or_predictor"),
            idea.get("outcome"),
            idea.get("rationale"),
            idea.get("source_quote"),
        )
    )
    construct_answerability = assess_idea_constructs(
        construct_text,
        mapped_concepts=tuple(required),
        database=str(desc.get("database") or "") or None,
        available_concepts=set(concept_to_file),
    )
    construct_blocked = any(
        row.get("verdict") == "blocked" for row in construct_answerability
    )
    construct_needs_review = any(
        row.get("verdict") == "needs_review" for row in construct_answerability
    )
    if not stats or missing_required or bool(export_index.get("demo_like")):
        status = "blocked"
    elif construct_blocked:
        status = "blocked"
    elif unavailable:
        status = "needs_review"
    else:
        status = "ready"
    if status == "ready" and low:
        status = "needs_review"
    if status == "ready" and construct_needs_review:
        status = "needs_review"
    if not denominator_resolved and status == "ready":
        status = "needs_review"
    if status == "ready" and (
        not temporal["time_zero_reconstructable"]
        or not temporal["temporal_ordering_reconstructable"]
        or joint_observed in (None, 0)
    ):
        status = "needs_review"

    out = {
        "ok": True,
        "schema_version": "easyicu.web_idea_bounded_sample_feasibility/2",
        "created_at": _now(),
        "run_id": run_id,
        "idea_id": idea.get("idea_id"),
        "status": status,
        "reportable": False,
        "claim_level": "feasibility_sample_not_reportable",
        "sample": {
            "basis": "bounded_file_head_sample",
            "max_records_per_feature": max_records,
            "scope": "first available records only; use as feasibility evidence, not as a clinical result",
        },
        "source": {
            "source_id": str(source.get("id") or "")[:80] or None,
            "label": source.get("label") or desc.get("label") or "Local export",
            "path_hash": _sha256(str(source.get("path") or desc.get("path") or ""))[
                :16
            ],
            "database": desc.get("database"),
            "demo_like": bool(export_index.get("demo_like")),
        },
        "cohort": {
            "entities": (
                len(entity_ids)
                if entity_ids
                else (desc.get("summary") or {}).get("stays")
            ),
            "modules": (desc.get("summary") or {}).get("modules"),
            "total_records": (desc.get("summary") or {}).get("total_rows"),
        },
        "feature_statistics": stats,
        "concept_bindings": bindings,
        "required_concepts": required,
        "design_answerability": {
            **temporal,
            "joint_observed_entities": joint_observed,
            "repeated_measure_density": {
                str(row.get("concept_id")): (
                    round(
                        int(row.get("records") or 0)
                        / max(int(row.get("observed_entities") or 0), 1),
                        2,
                    )
                    if row.get("records") is not None
                    else None
                )
                for row in stats
            },
        },
        "construct_answerability": construct_answerability,
        "missing_required_concepts": missing_required,
        "blockers": _bounded_feasibility_blockers(
            status=status,
            missing_required=missing_required,
            unavailable=unavailable,
            low=low,
            demo_like=bool(export_index.get("demo_like")),
            temporal=temporal,
            joint_observed=joint_observed,
            construct_answerability=construct_answerability,
        ),
        "prior_art_adjudication_binding": adjudication_binding,
        "interpretation": _bounded_sample_interpretation(
            stats,
            missing_required=missing_required,
            low_coverage=low,
        ),
        "privacy": {
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
            "source_text_stored": False,
            "full_text_stored": False,
            "network_calls": 0,
            "external_llm_calls": 0,
        },
    }
    _assert_no_row_payload(out)
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / _BOUNDED_FEASIBILITY_FILENAME).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def create_handoff(body: Dict[str, Any]) -> Dict[str, Any]:
    """Freeze an idea-mining plan for the downstream Agent module."""
    run_id = str(body.get("run_id") or "").strip()
    idea_id = str(body.get("idea_id") or "").strip()
    payload = _load_run(run_id)
    if not payload:
        raise IdeaMiningWebError({"error": "idea_run_not_found", "run_id": run_id})
    ideas = payload.get("idea_ledger") or []
    idea = next(
        (row for row in ideas if row.get("idea_id") == idea_id),
        None,
    )
    if not idea:
        raise IdeaMiningWebError({"error": "idea_not_found", "idea_id": idea_id})
    readiness: Dict[str, Any] = {}
    try:
        readiness = idea_execution_readiness_binding(run_id, idea_id)
    except IdeaMiningWebError:
        # A plan preview remains useful while a gate is unresolved. Acceptance
        # revalidates and requires the exact readiness binding fail-closed.
        readiness = {}
    selected_idea = _idea_with_readiness_overlay(idea, readiness)
    edits = str(body.get("plan_edits") or "").strip()
    plan_artifact = _load_plan(run_id)
    plan = dict(_plan_payload(plan_artifact) or payload.get("handoff_plan") or {})
    if edits:
        plan["human_plan_notes"] = edits[:1200]
        plan["selection_mode"] = "human_curated_with_text_edits"
        plan["plan_status"] = "replanned_requires_final_confirmation"
    elif plan_artifact:
        plan["selection_mode"] = (
            plan.get("selection_mode") or "planned_before_agent_handoff"
        )
        plan["plan_status"] = (
            plan.get("plan_status") or "planned_requires_final_confirmation"
        )
    prior_art_check = _load_prior_art(run_id)
    pre_experiment = payload.get("pre_experiment") or {}
    plan["active_export_contract"] = plan.get(
        "active_export_contract"
    ) or _active_export_contract(pre_experiment)
    plan["prior_art_review"] = _prior_art_review(prior_art_check)
    plan["execution_gate"] = _execution_gate(
        selected_idea,
        pre_experiment,
        prior_art_check,
        readiness,
    )
    plan["execution_readiness"] = {
        key: readiness.get(key)
        for key in (
            "prior_art_decision",
            "source_feasibility_status",
            "idea_definition_sha256",
            "prior_art_adjudication_sha256",
            "source_feasibility_sha256",
            "execution_ready_for_confirmation",
        )
        if readiness.get(key) is not None
    }
    plan["prior_art_adjudication"] = (
        readiness.get("prior_art_adjudication_summary") or {}
    )
    plan["source_feasibility_summary"] = (
        readiness.get("source_feasibility_summary") or {}
    )
    handoff = {
        "ok": True,
        "schema_version": "easyicu.web_idea_handoff/1",
        "created_at": _now(),
        "run_id": run_id,
        "idea_id": idea.get("idea_id"),
        "candidate_topic": idea.get("idea_title"),
        "go_no_go": selected_idea.get("go_no_go"),
        "go_no_go_reason": selected_idea.get("go_no_go_reason"),
        "selected_ledger_row": selected_idea,
        "source_evidence": payload.get("source_evidence") or [],
        "pre_experiment": pre_experiment,
        "prior_art_check": prior_art_check,
        "handoff_plan": plan,
        "agent_seed": {
            "study_id": _slug(idea.get("idea_title") or "idea"),
            "mode": "analysis",
            "question": plan.get("research_question"),
            "requires_human_confirmation": True,
            "reportable": False,
            "draft_unlocked": False,
        },
        "privacy": {
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
            "network_calls": 0,
            "external_llm_calls": 0,
        },
    }
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    source = ((payload.get("source_evidence") or [{}])[0]) or {}
    try:
        canonical_packet = build_web_handoff_packet(
            idea=selected_idea,
            source=source,
            plan=plan,
            pre_experiment=pre_experiment,
            prior_art_check=prior_art_check,
            run_dir=run_dir,
            readiness=readiness,
        )
        handoff.update(persist_canonical_handoff(canonical_packet, run_dir=run_dir))
    except (TypeError, ValueError) as exc:
        raise IdeaMiningWebError(
            {
                "error": "canonical_handoff_build_failed",
                "run_id": run_id,
                "reason": str(exc),
            }
        ) from exc
    _assert_no_row_payload(handoff)
    (run_dir / "idea_handoff.json").write_text(
        json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return handoff


def create_agent_project(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a metadata-only Agent Projects seed from a frozen handoff."""
    run_id = str(body.get("run_id") or "").strip()
    idea_id = str(body.get("idea_id") or "").strip()
    if not run_id or not idea_id:
        raise IdeaMiningWebError(
            {
                "error": "idea_identity_required",
                "run_id": run_id or None,
                "idea_id": idea_id or None,
            }
        )
    handoff = _load_handoff(run_id)
    canonical_packet: Optional[DiscoveryHandoffPacket] = None
    if not handoff:
        handoff = create_handoff(body)
    elif (
        str(handoff.get("run_id") or "").strip() != run_id
        or str(handoff.get("idea_id") or "").strip() != idea_id
    ):
        raise IdeaMiningWebError(
            {
                "error": "idea_handoff_identity_mismatch",
                "run_id": run_id,
                "idea_id": idea_id,
            }
        )
    try:
        # A canonical envelope is immutable evidence. Validate it before any
        # legitimate plan/prior-art refresh so a simultaneous update cannot
        # mask a tampered frozen packet. True pre-canonical legacy envelopes
        # are the sole exception and are rebuilt once below.
        if not is_legacy_handoff_envelope(handoff):
            canonical_packet = load_validated_canonical_handoff(
                handoff,
                run_dir=_run_dir(run_id),
            )
        if _handoff_needs_refresh(handoff, run_id) or _handoff_plan_is_stale(
            handoff, run_id, body
        ):
            refresh_body = dict(body)
            if not refresh_body.get("plan_edits"):
                # Prefer the freshest human plan notes: the on-disk plan (the user's
                # latest edit, which may never have been re-frozen) over the frozen
                # handoff notes, so a re-plan is not silently dropped from the seed.
                plan = _plan_payload(_load_plan(run_id))
                refresh_body["plan_edits"] = plan.get("human_plan_notes") or (
                    handoff.get("handoff_plan") or {}
                ).get("human_plan_notes")
            handoff = create_handoff(refresh_body)
            canonical_packet = None
        if canonical_packet is None:
            canonical_packet = load_validated_canonical_handoff(
                handoff,
                run_dir=_run_dir(run_id),
            )
    except CanonicalHandoffIntegrityError as exc:
        raise IdeaMiningWebError(
            {
                "error": "canonical_handoff_integrity_error",
                "run_id": run_id,
                "reason": exc.reason,
            }
        ) from exc
    seed = _agent_project_seed(handoff, canonical_packet)
    project_dir = _AGENT_PROJECTS_ROOT / str(seed["study_id"])
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_seed.json").write_text(
        json.dumps(seed, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    seeds = _read_agent_projects()
    seeds = [row for row in seeds if row.get("study_id") != seed["study_id"]]
    seeds.insert(0, seed)
    _AGENT_PROJECTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _AGENT_PROJECTS_PATH.write_text(
        json.dumps(seeds[:100], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out = {
        "ok": True,
        "project": seed,
        "projects": seeds[:100],
        "privacy": {
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
            "network_calls": 0,
            "external_llm_calls": 0,
            "reportable": False,
            "draft_unlocked": False,
        },
    }
    _assert_no_row_payload(out)
    return out


def list_agent_projects(body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """List product-facing metadata-only Agent project seeds.

    Historical evaluation imports are experiment evidence, not user research
    projects.  They remain on disk for the benchmark/evidence owners to audit,
    but the product project rail must not turn them into a dedicated question
    bank or imply that ordinary users choose from a fixed benchmark.
    """
    limit = int((body or {}).get("limit") or 20)
    seeds = [
        row
        for row in _read_agent_projects()
        if row.get("seed_kind") != "canonical9_import" and not row.get("benchmark")
    ]
    return {
        "ok": True,
        "projects": seeds[: max(1, min(limit, 100))],
        "storage": "metadata_only",
        "privacy": {
            "patient_rows_returned": False,
            "network_calls": 0,
            "external_llm_calls": 0,
        },
    }


def list_runs(body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """List local metadata-only idea-mining runs."""
    limit = int((body or {}).get("limit") or 20)
    history = _read_history()
    return {
        "ok": True,
        "runs": history[: max(1, min(limit, 100))],
        "privacy": {
            "source_text_stored": False,
            "patient_rows_returned": False,
            "network_calls": 0,
        },
    }


def get_run(body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load a persisted metadata-only idea-mining run for the UI history."""
    run_id = str((body or {}).get("run_id") or "").strip()
    payload = _load_run(run_id)
    if not payload:
        raise IdeaMiningWebError({"error": "idea_run_not_found", "run_id": run_id})
    out = dict(payload)
    out["ok"] = True
    out["loaded_from_history"] = True
    handoff = _load_handoff(run_id)
    if handoff:
        out["handoff"] = handoff
    prior = _load_prior_art(run_id)
    if prior:
        out["prior_art_check"] = prior
    plan = _load_plan(run_id)
    if plan:
        out["idea_plan"] = plan
    sample = _load_bounded_sample(run_id)
    if sample:
        out["bounded_sample_feasibility"] = sample
    project = _project_for_run(run_id)
    if project:
        out["agent_project"] = project
    out["privacy"] = dict(out.get("privacy") or {})
    out["privacy"].update(
        {
            "source_text_stored": False,
            "patient_rows_returned": False,
            "direct_identifiers_returned": False,
            "external_llm_calls": 0,
        }
    )
    _assert_no_row_payload(out)
    return out


def _source_record(body: Dict[str, Any]) -> Dict[str, Any]:
    topic = _clean(body.get("topic") or body.get("research_question") or "", 180)
    title = _clean(body.get("title") or topic or "Untitled source", 220)
    journal = _clean(body.get("journal") or "", 120)
    year = _year(body.get("year"))
    url = _clean(body.get("url") or "", 500)
    doi = _clean(body.get("doi") or "", 180)
    pmid = _clean(body.get("pmid") or "", 80)
    source_type = _clean(body.get("source_type") or "manual", 40)
    source_origin = _clean(body.get("source_origin") or "", 80)
    if not source_origin:
        if source_type == "zotero" and body.get("zotero_key"):
            source_origin = "zotero_desktop"
        elif source_type == "zotero":
            source_origin = "pasted_literature"
        else:
            source_origin = source_type
    source_origin_label = _clean(body.get("source_origin_label") or "", 120)
    excerpt = _clean(
        body.get("excerpt") or body.get("source_quote") or "", _MAX_SOURCE_QUOTE
    )
    source_text = _clean(
        body.get("excerpt") or body.get("abstract") or body.get("notes") or topic, 4000
    )
    citation_key = _slug(
        "|".join([title, str(year or ""), journal, doi, pmid]) or topic or "source"
    )
    source_hash = _sha256(
        "|".join([source_origin, title, journal, str(year), doi, pmid, url])
    )[:12]
    record = {
        "source_id": f"source_{source_hash}",
        "citation_key": citation_key,
        "source_type": source_type,
        "source_origin": source_origin,
        "title": title,
        "year": year,
        "journal": journal or None,
        "url": url or None,
        "doi": doi or None,
        "pmid": pmid or None,
        "evidence_quote": excerpt or _first_sentence(source_text),
        "source_text_sha256": _sha256(source_text) if source_text else None,
        "source_text_char_count": len(source_text),
        "source_text_stored": False,
        "rights_note": "Only metadata, hash and a bounded user-supplied quote are persisted.",
    }
    if source_origin_label:
        record["source_origin_label"] = source_origin_label
    if body.get("source_file_name"):
        record["source_file_name"] = _clean(body.get("source_file_name"), 240)
    if body.get("source_file_sha256"):
        record["source_file_sha256"] = _clean(body.get("source_file_sha256"), 80)
    if body.get("literature_folder"):
        record["literature_folder"] = _norm_path(
            str(body.get("literature_folder") or "")
        )
    if body.get("citation_key"):
        record["citation_key"] = _clean(body.get("citation_key"), 180)
    if body.get("zotero_key"):
        record["zotero_key"] = _clean(body.get("zotero_key"), 80)
    if body.get("literature_pdf_count") is not None:
        try:
            record["literature_pdf_count"] = int(body.get("literature_pdf_count") or 0)
        except Exception:
            record["literature_pdf_count"] = 0
    return record


def _require_source_seed(body: Dict[str, Any]) -> None:
    fields = (
        "topic",
        "research_question",
        "excerpt",
        "source_quote",
        "title",
        "abstract",
        "notes",
        "url",
        "doi",
        "pmid",
        "source_file_sha256",
        "literature_folder",
        "zotero_key",
        "citation_key",
    )
    if any(str(body.get(field) or "").strip() for field in fields):
        return
    raise IdeaMiningWebError(
        {
            "error": "idea_source_required",
            "reason": "Add a topic, source quote, title, DOI, PMID, URL, local PDF, or literature folder before running Idea Mining.",
        }
    )


def _source_text(body: Dict[str, Any]) -> str:
    return "\n".join(
        str(body.get(k) or "")
        for k in (
            "topic",
            "title",
            "abstract",
            "excerpt",
            "notes",
            "url",
            "journal",
            "source_file_name",
        )
    ).strip()


def _extract_pdf_bytes(pdf_bytes: bytes, *, filename: str) -> Dict[str, Any]:
    if len(pdf_bytes) > _MAX_PDF_BYTES:
        raise IdeaMiningWebError(
            {
                "error": "pdf_file_too_large",
                "reason": f"Selected PDF is larger than the local bounded parser limit ({_MAX_PDF_BYTES // (1024 * 1024)} MB).",
                "filename": filename,
            }
        )
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional dependency
        raise IdeaMiningWebError(
            {
                "error": "pdf_parser_unavailable",
                "reason": "Local PDF parsing requires PyMuPDF (fitz), which is not available in this environment.",
            }
        ) from exc
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise IdeaMiningWebError(
            {
                "error": "pdf_parse_failed",
                "reason": "The selected file could not be parsed as a PDF.",
                "filename": filename,
            }
        ) from exc
    try:
        metadata = dict(doc.metadata or {})
        pages = int(getattr(doc, "page_count", 0) or 0)
        text_parts: List[str] = []
        for index in range(min(pages, _MAX_PDF_EXTRACT_PAGES)):
            try:
                text_parts.append(doc.load_page(index).get_text("text") or "")
            except Exception:
                continue
    finally:
        try:
            doc.close()
        except Exception:
            pass
    extracted = _clean(" ".join(text_parts), _MAX_PDF_EXCERPT)
    title = _clean(metadata.get("title") or Path(filename).stem, 220)
    return {
        "filename": filename,
        "title": title,
        "author": _clean(metadata.get("author") or "", 240) or None,
        "page_count": pages,
        "bytes": len(pdf_bytes),
        "sha256": digest,
        "excerpt": extracted,
        "excerpt_char_count": len(extracted),
        "pages_scanned": min(pages, _MAX_PDF_EXTRACT_PAGES),
        "full_text_stored": False,
    }


def _pdf_file_record(path: Path, *, extract_excerpt: bool) -> Dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    base = {
        "filename": path.name,
        "path": str(path),
        "bytes": int(size),
        "sha256": None,
        "excerpt": "",
        "full_text_stored": False,
    }
    if size > _MAX_PDF_BYTES:
        base["status"] = "skipped_too_large"
        return base
    try:
        data = path.read_bytes()
    except OSError:
        base["status"] = "skipped_unreadable"
        return base
    if not extract_excerpt:
        base["sha256"] = hashlib.sha256(data).hexdigest()
        base["status"] = "metadata_only"
        return base
    try:
        record = _extract_pdf_bytes(data, filename=path.name)
        record["path"] = str(path)
        record["status"] = "excerpt_ready"
        return record
    except IdeaMiningWebError as exc:
        base["sha256"] = hashlib.sha256(data).hexdigest()
        base["status"] = exc.detail.get("error") or "parse_failed"
        base["reason"] = exc.detail.get("reason")
        return base


def _suggestion_from_pdf_record(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not record:
        return {}
    title = _clean(
        record.get("title") or Path(str(record.get("filename") or "paper")).stem, 220
    )
    excerpt = _clean(record.get("excerpt") or "", _MAX_PDF_EXCERPT)
    return {
        "source_type": "pdf",
        "topic": title,
        "excerpt": excerpt,
        "title": title,
        "journal": None,
        "year": None,
        "doi": None,
        "pmid": None,
        "url": None,
        "source_file_name": record.get("filename"),
        "source_file_sha256": record.get("sha256"),
    }


def _match_concepts(text: str) -> List[Dict[str, Any]]:
    haystack = _norm_text(text)
    hits: List[Dict[str, Any]] = []
    for concept_id, entry in concept_catalog.CONCEPT_DICTIONARY.items():
        en = str(entry[0] if entry else concept_id)
        zh = str(entry[1] if len(entry) > 1 else "")
        aliases = {concept_id, en, zh}
        aliases.update(_extra_aliases(concept_id, en))
        matched = [alias for alias in aliases if alias and _alias_hit(haystack, alias)]
        if not matched:
            continue
        hits.append(
            {
                "concept_id": concept_id,
                "label": en,
                "unit": str(entry[2] if len(entry) > 2 else ""),
                "matched_alias": sorted(matched, key=len, reverse=True)[0],
                "module": _concept_module(concept_id),
            }
        )
    hits.sort(key=lambda row: (row["concept_id"] in {"death", "los_icu"}, row["label"]))
    return hits[:24]


def _requested_adjustment_concepts(
    text: str,
    hits: List[Dict[str, Any]],
) -> List[str]:
    """Return only covariates the source explicitly says to adjust for.

    A concept being mentioned in an idea is not evidence that it belongs in an
    adjusted model.  In particular, severity markers listed for descriptive
    review must not silently become confounders.  This small parser recognizes
    bounded Chinese and English adjustment clauses; ambiguous prose yields an
    empty list and leaves covariate selection to the conversation/human plan
    gate.
    """

    source = _clean(text, 2_000)
    clauses: List[str] = []
    for pattern in (
        r"(?:按|依据)\s*([^。；;]{1,120}?)\s*(?:调整|校正|控制)",
        r"(?:调整|校正|控制)(?:变量|因素)?(?:包括|为|：|:)?\s*([^。；;]{1,120})",
        r"(?:adjust(?:ed|ing)?\s+for|controlled\s+for|covariates?(?:\s+include|\s+were|\s*:))\s*([^.;]{1,160})",
    ):
        clauses.extend(match.group(1) for match in re.finditer(pattern, source, re.I))
    if not clauses:
        return []

    selected: List[str] = []
    for clause in clauses:
        normalized = _norm_text(clause)
        for row in hits:
            concept_id = str(row.get("concept_id") or "").strip()
            if not concept_id or concept_id in {"death", "los_icu"}:
                continue
            label = str(row.get("label") or "")
            aliases = {concept_id, label, str(row.get("matched_alias") or "")}
            aliases.update(_extra_aliases(concept_id, label))
            if (
                any(alias and _alias_hit(normalized, alias) for alias in aliases)
                and concept_id not in selected
            ):
                selected.append(concept_id)
    return selected


def _requested_exposure_concepts(
    text: str,
    hits: List[Dict[str, Any]],
) -> List[str]:
    """Return predictors explicitly named as the primary exposure."""

    source = _clean(text, 2_000)
    clauses: List[str] = []
    for pattern in (
        r"(?:主要|核心|首要)?(?:暴露|预测因子|自变量)(?:采用|定义为|为|：|:)?\s*([^。；;，,]{1,140})",
        r"(?:primary|main)\s+(?:exposure|predictor)(?:\s+is|\s*:)?\s*([^.;,]{1,180})",
    ):
        clauses.extend(match.group(1) for match in re.finditer(pattern, source, re.I))
    if not clauses:
        return []
    selected: List[str] = []
    for clause in clauses:
        normalized = _norm_text(clause)
        ranked: List[Tuple[int, str]] = []
        for row in hits:
            concept_id = str(row.get("concept_id") or "").strip()
            if not concept_id or concept_id in {"death", "los_icu"}:
                continue
            label = str(row.get("label") or "")
            aliases = {concept_id, label, str(row.get("matched_alias") or "")}
            aliases.update(_extra_aliases(concept_id, label))
            matched = [
                _norm_text(alias)
                for alias in aliases
                if alias and _alias_hit(normalized, alias)
            ]
            if matched:
                ranked.append((max(len(alias) for alias in matched), concept_id))
        if ranked:
            concept_id = max(ranked, key=lambda item: (item[0], item[1]))[1]
            if concept_id not in selected:
                selected.append(concept_id)
    return selected


def _idea_design_support(
    text: str,
    hits: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Project one existing reviewed know-how card into Idea Mining.

    This is advisory design context, not a second planner. The registry remains
    the sole owner of topic matching and card content; Idea Mining only exposes
    the bounded fields needed to ask the next scientific question and compile
    a literature query.
    """

    registry = KnowHowRegistry.load()
    matches = registry.retrieve(
        query=text,
        available_concepts=[
            str(row.get("concept_id") or "") for row in hits if row.get("concept_id")
        ],
        top_k=1,
    )
    if not matches:
        return None
    match = matches[0]
    registry.verify_hit_source(match)
    card = registry.get(match.card_id)
    design = card.design_candidates
    return {
        "card_id": card.card_id,
        "version": card.version,
        "file_sha256": match.file_sha256,
        "trust_level": card.trust_level,
        "review_status": card.review_status,
        "summary": card.summary,
        "study_families": list(card.study_families)[:6],
        "topic_aliases": list(card.topic_aliases)[:12],
        "population": design.population,
        "time_zero": design.time_zero,
        "observation_window": design.observation_window,
        "prediction_or_followup_window": design.prediction_or_followup_window,
        "eligibility_candidates": list(design.eligibility_candidates)[:6],
        "exposure_family": design.exposure,
        "outcome_family": design.outcome,
        "estimand_family": design.estimand,
        "recommended_methods": list(design.recommended_methods)[:6],
        "sensitivity_analyses": list(design.sensitivity_analyses)[:6],
        "literature_outcome": card.title,
        "requires_confirmation": list(card.requires_confirmation)[:6],
        "stop_conditions": list(card.stop_conditions)[:6],
        "citations": [
            {
                "citation_id": row.citation_id,
                "title": row.title,
                "year": row.year,
                "url": row.url,
                "supports": list(row.supports)[:4],
            }
            for row in card.citations[:4]
        ],
        "authority": "advisory_design_support_only",
    }


def _idea_population(text: str, design_support: Optional[Dict[str, Any]]) -> str:
    low = str(text or "").lower()
    if (
        design_support
        and design_support.get("card_id") == "mechanical_ventilation_liberation"
    ):
        if any(token in low for token in ("adult", "adults", "成人")):
            return (
                "Adult ICU patients with an identifiable invasive-ventilation episode"
            )
        if any(
            token in low
            for token in ("pediatric", "paediatric", "child", "儿童", "儿科")
        ):
            return "Pediatric ICU patients with an identifiable invasive-ventilation episode"
        return "ICU patients with an identifiable invasive-ventilation episode (age scope pending)"
    if any(token in low for token in ("adult", "adults", "成人")):
        return "Adult ICU cohort"
    if any(
        token in low for token in ("pediatric", "paediatric", "child", "儿童", "儿科")
    ):
        return "Pediatric ICU cohort"
    return "ICU cohort (population pending)"


def _idea_from_source(
    source: Dict[str, Any],
    text: str,
    hits: List[Dict[str, Any]],
    export_index: Dict[str, Any],
) -> Dict[str, Any]:
    family = _analysis_family(text)
    outcome = _pick_outcome(text, hits)
    concepts = _select_idea_concepts(text, hits, outcome)
    predictor = _primary_predictor(concepts)
    if predictor is None:
        predictor = _pick_predictor(hits, outcome)
        if predictor is not None:
            predictor_id = str(predictor.get("concept_id") or "")
            concepts = [
                (
                    {**row, "role": "predictor"}
                    if str(row.get("concept_id") or "") == predictor_id
                    else row
                )
                for row in concepts
            ]
            if not any(
                str(row.get("concept_id") or "") == predictor_id for row in concepts
            ):
                concepts.insert(0, {**predictor, "role": "predictor"})
    if not concepts and hits:
        concepts = _dedupe_concepts(hits[:3])
    design_support = _idea_design_support(text, hits)
    title = _idea_title(source, predictor, outcome, concepts)
    concept_rows = [_concept_feasibility(row, export_index) for row in concepts]
    overall = _overall_feasibility(concept_rows, export_index)
    design_blockers: List[str] = []
    if family != "trajectory":
        if predictor is None:
            design_blockers.append("predictor_or_exposure_not_mapped")
        if outcome is None:
            design_blockers.append("outcome_not_mapped")
    if design_blockers:
        missing_labels = []
        if "predictor_or_exposure_not_mapped" in design_blockers:
            missing_labels.append("a primary exposure or predictor")
        if "outcome_not_mapped" in design_blockers:
            missing_labels.append("an explicit outcome")
        overall = {
            **overall,
            "tier": "design_incomplete",
            "label": "Research design incomplete",
            "reason": (
                "The source does not identify "
                + " and ".join(missing_labels)
                + "; confirm the estimand before Agent execution."
            ),
            "design_blockers": design_blockers,
        }
    literature_outcome = (
        {"label": design_support.get("literature_outcome")}
        if design_support and design_support.get("literature_outcome")
        else None
    )
    novelty = _prior_art(
        source,
        title,
        predictor=predictor,
        outcome=outcome or literature_outcome,
        topic_aliases=(design_support or {}).get("topic_aliases") or [],
    )
    construct_answerability = assess_idea_constructs(
        text,
        mapped_concepts=tuple(
            str(row.get("concept_id") or "")
            for row in concept_rows
            if str(row.get("concept_id") or "")
        ),
        database=str(export_index.get("database") or "") or None,
        available_concepts=(
            set(export_index.get("concept_to_file") or {})
            if export_index.get("source_selected")
            else None
        ),
    )
    go_no_go = (
        "hold"
        if design_blockers
        else (
            "recommend"
            if overall["tier"] == "executable"
            else (
                "hold"
                if overall["tier"].startswith("T1") or overall["tier"] == "demo_only"
                else "db-cannot-do"
            )
        )
    )
    idea_payload = {
        "idea_id": "idea_"
        + _sha256(json.dumps([source.get("source_id"), title], ensure_ascii=False))[
            :12
        ],
        "idea_title": title,
        "population": _idea_population(text, design_support),
        "exposure_or_predictor": _concept_set_label(concepts)
        or (predictor["label"] if predictor else _clean(text, 90)),
        "outcome": outcome["label"] if outcome else None,
        "design_support": design_support,
        "unresolved_slots": design_blockers,
        "analysis_family": family,
        "source_id": source.get("source_id"),
        "source_title": source.get("title"),
        "source_year": source.get("year"),
        "source_journal": source.get("journal"),
        "source_quote": source.get("evidence_quote"),
        "rationale": _rationale(text, predictor, outcome, concepts),
        "mapped_concepts": concept_rows,
        "construct_answerability": construct_answerability,
        "requested_adjustment_concepts": _requested_adjustment_concepts(text, hits),
        "feasibility": overall,
        "prior_art": novelty,
        "go_no_go": go_no_go,
        "go_no_go_reason": _go_reason(go_no_go, overall, novelty),
        "next_action": _next_action(go_no_go, overall),
        "plan_status": "draft_requires_human_confirmation",
    }
    return idea_payload


def _pre_experiment(
    idea: Dict[str, Any],
    export: Optional[Tuple[Dict[str, Any], Dict[str, Any]]],
    export_index: Dict[str, Any],
) -> Dict[str, Any]:
    if not export:
        return {
            "status": "blocked",
            "reason": "No active registered export is selected. Extract data or register an EasyICU export first.",
            "payload_scope": "no_patient_rows",
            "feature_statistics": [],
            "reportable": False,
        }
    source, desc = export
    demo_like = bool(export_index.get("demo_like"))
    entity_ids = export_index.get("entity_ids") or set()
    concepts = [
        row.get("concept_id")
        for row in idea.get("mapped_concepts") or []
        if row.get("concept_id") in export_index.get("concept_to_file", {})
    ]
    if not concepts:
        concepts = list((export_index.get("concept_to_file") or {}).keys())[:6]
    resolved_denominator, denominator_resolved = _cohort_denominator(export_index, desc)
    stats = _feature_stats(
        Path(str(desc.get("path") or source.get("path"))),
        concepts,
        export_index,
        entity_ids,
        denominator=resolved_denominator if denominator_resolved else 1,
        denominator_resolved=denominator_resolved,
    )
    required = [
        row.get("concept_id")
        for row in idea.get("mapped_concepts") or []
        if row.get("concept_id")
    ]
    present = {row.get("concept_id") for row in stats}
    missing_required = [cid for cid in required if cid not in present]
    status = (
        "ready"
        if stats and not missing_required
        else ("partial" if stats else "blocked")
    )
    if demo_like and status in {"ready", "partial"}:
        status = "demo_only"
    elif not denominator_resolved and status == "ready":
        # Coverage is indeterminate without a cohort denominator: never emit a
        # clean "ready" feasibility verdict against a fabricated denominator.
        status = "partial"
    return {
        "status": status,
        "payload_scope": "aggregate_pre_experiment_no_row_payload",
        "reportable": False,
        "reason": (
            "Current active export is a MOCK/demo source. Use this only for UI rehearsal; switch to a real EasyICU export before reporting feasibility."
            if demo_like
            else None
        ),
        "source": {
            "label": source.get("label") or desc.get("label") or "Local export",
            "path_hash": _sha256(str(source.get("path") or desc.get("path") or ""))[
                :16
            ],
            "database": desc.get("database"),
            "demo_like": demo_like,
        },
        "cohort": {
            "entities": (
                len(entity_ids)
                if entity_ids
                else (desc.get("summary") or {}).get("stays")
            ),
            "modules": (desc.get("summary") or {}).get("modules"),
            "total_rows": (desc.get("summary") or {}).get("total_rows"),
        },
        "feature_statistics": stats,
        "missing_required_concepts": missing_required,
        "interpretation": _pre_experiment_interpretation(stats, demo_like=demo_like),
    }


def _handoff_plan(
    source: Dict[str, Any],
    idea: Dict[str, Any],
    pre_experiment: Dict[str, Any],
) -> Dict[str, Any]:
    predictor = idea.get("exposure_or_predictor") or "the candidate predictor"
    outcome = idea.get("outcome") or "the selected outcome"
    population = idea.get("population") or "the selected ICU population"
    if idea.get("analysis_family") == "trajectory" and not idea.get("outcome"):
        question = (
            f"Characterize {predictor} trajectories in {population}, with "
            "time zero, observation window, and transportability checks confirmed before execution."
        )
    else:
        question = f"Evaluate whether {predictor} is associated with {outcome} in {population}."
    mapped = idea.get("mapped_concepts") or []
    return {
        "selection_mode": "human_confirm_before_agent_run",
        "research_question": question,
        "source_provenance": {
            "source_type": source.get("source_type"),
            "source_origin": source.get("source_origin"),
            "source_origin_label": source.get("source_origin_label"),
            "title": source.get("title"),
            "year": source.get("year"),
            "journal": source.get("journal"),
            "doi": source.get("doi"),
            "pmid": source.get("pmid"),
            "citation_key": source.get("citation_key"),
            "zotero_key": source.get("zotero_key"),
            "quote": source.get("evidence_quote"),
            "source_text_hash": source.get("source_text_sha256"),
        },
        "cohort": {
            "default": population,
            "requires_user_confirmation": True,
        },
        "design_support": idea.get("design_support"),
        "variables": [
            {
                "role": row.get("role")
                or (
                    "predictor"
                    if i == 0
                    else (
                        "outcome"
                        if row.get("concept_id") == "death"
                        else "covariate_or_feature"
                    )
                ),
                "concept_id": row.get("concept_id"),
                "label": row.get("label"),
                "feasibility_tier": row.get("tier"),
            }
            for i, row in enumerate(mapped)
        ],
        "pre_experiment_summary": {
            "status": pre_experiment.get("status"),
            "entities": (pre_experiment.get("cohort") or {}).get("entities"),
            "feature_count": len(pre_experiment.get("feature_statistics") or []),
        },
        "active_export_contract": _active_export_contract(pre_experiment),
        "prior_art_review": _prior_art_review(None),
        "execution_gate": _execution_gate(idea, pre_experiment, None),
        "analysis_plan": _agent_style_steps(
            str(idea.get("analysis_family") or "association"), mapped
        )[:4],
        "blocked_until": [
            "human confirms the idea and plan",
            "active export contains required features or re-extraction is complete",
            "prior-art search is reviewed when network/LLM search is enabled",
        ],
        "reportable": False,
        "draft_unlocked": False,
    }


def _analysis_plan_draft(
    source: Dict[str, Any],
    idea: Dict[str, Any],
    pre_experiment: Dict[str, Any],
    prior_art: Optional[Dict[str, Any]],
    *,
    edits: str = "",
    mode: str = "plan",
    plan_fields: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    base = _handoff_plan(source, idea, pre_experiment)
    family = str(idea.get("analysis_family") or "association")
    concepts = idea.get("mapped_concepts") or []
    source_type = str(source.get("source_type") or "manual")
    plan = dict(base)
    plan.update(
        {
            "selection_mode": (
                "agent_style_replan_requires_human_confirmation"
                if mode == "replan" or edits
                else "agent_style_plan_requires_human_confirmation"
            ),
            "plan_status": "draft_plan_requires_user_review",
            "analysis_family": family,
            "source_input_type": source_type,
            "reference_strategy": (
                "Use high-impact ICU paper method motifs as planning checklists "
                "only; do not copy another study path or claim novelty without "
                "bounded prior-art review."
            ),
            "reference_analysis_patterns": _reference_analysis_patterns(
                family, concepts
            ),
            "clinical_icu_constraints": _clinical_icu_constraints(idea, concepts),
            "required_user_confirmations": _required_plan_confirmations(
                pre_experiment, prior_art
            ),
            "prior_art_review": _prior_art_review(prior_art),
            "execution_gate": _execution_gate(idea, pre_experiment, prior_art),
            "agent_boundary": {
                "can_create_agent_seed_after_confirmation": True,
                "agent_run_created": False,
                "draft_unlocked": False,
                "reportable": False,
                "reason": (
                    "Planning produces a handoff object only. Agent Projects must still "
                    "run planner/replanner/coder/analyzer/writer and evidence checks."
                ),
            },
        }
    )
    plan["analysis_plan"] = _agent_style_steps(family, concepts)
    confirmed = dict(plan_fields or {})
    if confirmed:
        for key in (
            "research_question",
            "population",
            "exposure",
            "outcome",
            "time_zero",
            "time_window",
        ):
            if confirmed.get(key):
                plan[key] = confirmed[key]
        if confirmed.get("population"):
            plan["cohort"] = {
                **(plan.get("cohort") or {}),
                "default": confirmed["population"],
            }
        plan["confirmed_plan_fields"] = confirmed
        if confirmed.get("outcome") and confirmed.get("time_window"):
            plan["required_user_confirmations"] = [
                row
                for row in plan.get("required_user_confirmations") or []
                if row != "outcome and time window"
            ]
    if edits:
        plan["human_plan_notes"] = _clean(edits, 1200)
        plan["replan_summary"] = (
            "Human notes were captured for Agent replanning; downstream Agent "
            "Projects should treat them as constraints, not completed analysis."
        )
    return plan


def _agent_style_steps(
    family: str, concepts: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    concept_ids = {str(row.get("concept_id") or "") for row in concepts}
    treatment_strategy = bool(
        concept_ids & {"vaso_ind", "norepi_equiv", "norepi_rate", "norepi_dur"}
    ) and bool(
        concept_ids & {"total_input_ml", "fluid_balance", "fluid_balance_cumulative"}
    )
    steps: List[Dict[str, str]] = [
        {
            "phase": "Question",
            "title": "Freeze the clinical question and estimand",
            "action": "Confirm population, exposure/index time, comparator, outcome, and follow-up window before looking at any effect estimate.",
            "output": "One locked PICOT-style question plus a non-reportable planning note.",
            "guardrail": "Idea Mining proposes the question only; it has not completed cohort selection or analysis.",
        },
        {
            "phase": "Data context",
            "title": "Confirm the real EasyICU export and modules",
            "action": "Select the real local export, cohort denominator, required modules, and concept dictionary mappings with the user.",
            "output": "Confirmed export/cohort/module contract for Agent Projects.",
            "guardrail": "MOCK/demo exports are UI rehearsal only and cannot support reportable feasibility.",
        },
        {
            "phase": "Feasibility",
            "title": "Run an outcome-blind feasibility assessment",
            "action": "Check concept availability, joint completeness, time-index support, missingness structure, and event rate before modeling.",
            "output": "Feasibility table with required concepts, denominators, and blockers.",
            "guardrail": "Do not interpret feasibility checks as clinical findings.",
        },
    ]
    if treatment_strategy:
        steps.extend(
            [
                {
                    "phase": "Design",
                    "title": "Translate the article into an ICU treatment-strategy question",
                    "action": "Define vasopressor/fluid timing anchors, dose or exposure summaries, comparator groups, and eligible shock/sepsis windows.",
                    "output": "Treatment-strategy contrast ready for descriptive review.",
                    "guardrail": "Flag confounding by indication and immortal-time risk before any adjusted model.",
                },
                {
                    "phase": "Robustness",
                    "title": "Predefine balance and sensitivity checks",
                    "action": "Compare baseline severity, missingness, exposure timing, and alternative dose/window definitions before final modeling.",
                    "output": "Sensitivity checklist for Agent replanning.",
                    "guardrail": "Keep claims exploratory unless causal assumptions are explicitly audited.",
                },
            ]
        )
    elif family == "prediction":
        steps.extend(
            [
                {
                    "phase": "Model design",
                    "title": "Define prediction windows and validation",
                    "action": "Set predictor window, target outcome horizon, split strategy, calibration display, and minimum reporting set.",
                    "output": "Prediction model protocol draft.",
                    "guardrail": "Audit leakage and class balance before fitting.",
                },
                {
                    "phase": "Generalization",
                    "title": "Audit missingness and transportability",
                    "action": "Review missingness handling and cross-source transportability when multiple ICU databases are available.",
                    "output": "Validation and missingness plan.",
                    "guardrail": "Do not use post-outcome measurements as predictors.",
                },
            ]
        )
    elif family == "trajectory":
        steps.extend(
            [
                {
                    "phase": "Trajectory design",
                    "title": "Define repeated-measure anchors and summaries",
                    "action": "Choose time zero, aggregation rules, trajectory summaries, and group comparison windows.",
                    "output": "Longitudinal feature construction plan.",
                    "guardrail": "Separate measurement frequency from physiologic change.",
                },
                {
                    "phase": "Sensitivity",
                    "title": "Check observation density and missingness",
                    "action": "Run coverage and sensitivity checks for irregular ICU measurements before comparing trajectories.",
                    "output": "Trajectory feasibility and sensitivity table.",
                    "guardrail": "Sparse monitoring can create apparent trajectories.",
                },
            ]
        )
    else:
        steps.extend(
            [
                {
                    "phase": "Analysis",
                    "title": "Start with descriptive association",
                    "action": "Summarize denominators, exposure/outcome distributions, crude contrasts, and missingness before adjusted models.",
                    "output": "Descriptive cohort comparison package.",
                    "guardrail": "Add adjusted or time-to-event models only after covariates and assumptions are confirmed.",
                },
                {
                    "phase": "Robustness",
                    "title": "Predefine subgroup and sensitivity checks",
                    "action": "Choose subgroup, missingness, and alternative-definition checks before viewing final results.",
                    "output": "Sensitivity plan for Agent execution.",
                    "guardrail": "Do not add checks post hoc to rescue a result.",
                },
            ]
        )
    steps.extend(
        [
            {
                "phase": "Prior art",
                "title": "Use existing literature as an inspiration map",
                "action": "After explicit opt-in, inspect whether prior studies answer the same question, use public ICU databases, or suggest better comparators/subgroups.",
                "output": "Prior-art interpretation: already answered, partially answered, or new exploratory angle.",
                "guardrail": "Prior work does not automatically block the idea; it shapes the new question and novelty claim.",
            },
            {
                "phase": "Agent handoff",
                "title": "Create an Agent Projects seed only after confirmation",
                "action": "Send the locked question, feasibility table, prior-art interpretation, and analysis steps to Agent Projects.",
                "output": "Metadata-only project seed for planner/replanner/coder/analyzer/writer.",
                "guardrail": "Manuscript claims remain blocked until evidence IDs and human sign-off pass.",
            },
        ]
    )
    return steps


def _reference_analysis_patterns(
    family: str, concepts: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    concept_ids = {str(row.get("concept_id") or "") for row in concepts}
    if concept_ids & {"vaso_ind", "norepi_equiv", "norepi_rate", "norepi_dur"}:
        return [
            {
                "pattern": "critical-care treatment strategy",
                "use_for": "exposure timing, comparator definition, baseline balance, and sensitivity design",
                "guardrail": "Observational ICU data can support exploratory association, not causal treatment claims without a causal audit.",
            },
            {
                "pattern": "target-trial-style translation",
                "use_for": "index time, eligibility window, follow-up window, and estimand wording",
                "guardrail": "Do not import trial eligibility wholesale; adapt to what EasyICU concepts can actually observe.",
            },
        ]
    if family == "prediction":
        return [
            {
                "pattern": "ICU prediction model reporting",
                "use_for": "predictor window, outcome horizon, validation split, discrimination, calibration, and decision boundary",
                "guardrail": "Do not let feature availability or post-outcome measurements leak into predictors.",
            }
        ]
    if family == "trajectory":
        return [
            {
                "pattern": "longitudinal ICU trajectory analysis",
                "use_for": "anchor selection, repeated-measure summaries, time-varying missingness, and subgroup display",
                "guardrail": "Measurement frequency is informative in ICU data and must not be mistaken for physiologic trajectory alone.",
            }
        ]
    return [
        {
            "pattern": "descriptive ICU cohort comparison",
            "use_for": "Table 1, denominator transparency, missingness display, crude association, and sensitivity checks",
            "guardrail": "Treat this as hypothesis-generating until adjustment strategy and prior art are reviewed.",
        }
    ]


def _clinical_icu_constraints(
    idea: Dict[str, Any], concepts: List[Dict[str, Any]]
) -> List[str]:
    constraints = [
        "Use ICU stays/entities from the confirmed EasyICU export; do not assume the extraction is complete merely because an idea was mined.",
        "Keep cohort definition, exposure/index time, outcome horizon, and feature modules user-confirmed before Agent execution.",
        "Report denominators and concept coverage for every comparison.",
        "Keep claims exploratory unless the Agent evidence checks and human sign-off pass.",
    ]
    concept_ids = {str(row.get("concept_id") or "") for row in concepts}
    if "sep3_sofa2" in concept_ids or "susp_inf" in concept_ids:
        constraints.append(
            "For Sepsis-3, preserve the suspected-infection anchor and SOFA delta settings from the extraction manifest."
        )
    if "death" in concept_ids:
        constraints.append(
            "Name the mortality endpoint and time horizon explicitly; do not conflate hospital, ICU, and 28-day mortality."
        )
    if concept_ids & {"vaso_ind", "norepi_equiv", "norepi_rate", "norepi_dur"}:
        constraints.append(
            "For vasopressor exposure, handle indication bias, dose/timing definitions, and immortal-time risk before estimating associations."
        )
    if idea.get("go_no_go") != "recommend":
        constraints.append(
            "Current feasibility is not recommend; re-extract missing modules or revise the idea before Agent execution."
        )
    return constraints


def _required_plan_confirmations(
    pre_experiment: Dict[str, Any], prior_art: Optional[Dict[str, Any]]
) -> List[str]:
    confirmations = [
        "local export / database source",
        "cohort denominator and inclusion/exclusion criteria",
        "feature modules and mapped concepts",
        "outcome and time window",
        "analysis family and reporting boundary",
    ]
    status = str(pre_experiment.get("status") or "").lower()
    if not status or "blocked" in status or "demo" in status:
        confirmations.insert(0, "prepare or register a usable EasyICU export")
    prior = (prior_art or {}).get("prior_art") or {}
    if not prior.get("search_performed"):
        confirmations.append("prior-art review opt-in or explicit decision to skip")
    return confirmations


def _active_export() -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    registry = source_store.load_registry()
    active = str(registry.get("active_path") or "")
    if not active:
        return None
    active_norm = _norm_path(active)
    source = next(
        (
            s
            for s in registry.get("sources") or []
            if isinstance(s, dict)
            and _norm_path(str(s.get("path") or "")) == active_norm
        ),
        None,
    )
    if not source:
        return None
    desc = dataio.describe_export_source(str(source.get("path") or ""))
    if not desc.get("ok"):
        return None
    return source, desc


def _export_index(
    export: Optional[Tuple[Dict[str, Any], Dict[str, Any]]],
) -> Dict[str, Any]:
    if not export:
        return {
            "concept_to_file": {},
            "entity_ids": set(),
            "demo_like": False,
            "database": None,
            "source_selected": False,
        }
    source, desc = export
    concept_to_file: Dict[str, Dict[str, Any]] = {}
    concepts = set(concept_catalog.CONCEPT_DICTIONARY)
    for item in desc.get("files") or []:
        columns = [str(c) for c in item.get("columns") or []]
        for col in columns:
            if col in concepts:
                concept_to_file[col] = item
    entity_ids = set()
    try:
        path = Path(str(desc.get("path") or ""))
        stay_ids = dataio._fast_stay_ids(path, desc.get("files") or [])
        entity_ids = set(stay_ids or [])
    except Exception:
        entity_ids = set()
    return {
        "concept_to_file": concept_to_file,
        "entity_ids": entity_ids,
        "demo_like": _export_is_demo_like(source, desc),
        "database": desc.get("database"),
        "source_selected": True,
    }


def _cohort_denominator(
    export_index: Dict[str, Any], desc: Dict[str, Any]
) -> Tuple[Optional[int], bool]:
    """Resolve the cohort denominator, distinguishing 0/1 from *unknown*.

    Returns ``(denominator, resolved)``. Prefer the resolved stay-id set; fall
    back to the export summary's stay count (the value the UI already shows).
    Returns ``(None, False)`` when neither is available so callers mark coverage
    indeterminate instead of dividing by a fabricated ``1`` — which turned an
    800-entity concept into ``coverage_pct = 80000`` and read as fully feasible.
    """
    entity_ids = export_index.get("entity_ids") or set()
    if entity_ids:
        return len(entity_ids), True
    summary_stays = (desc.get("summary") or {}).get("stays")
    try:
        n = int(summary_stays)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return n, True
    return None, False


def _mark_denominator_unresolved(stats: List[Dict[str, Any]]) -> None:
    """Blank coverage on feature stats when the cohort denominator is unknown.

    Coverage/event-rate percentages computed against a fabricated denominator
    of 1 are meaningless and read as feasible; null them and flag the rows so
    the UI shows an indeterminate denominator rather than a false-feasible signal.
    """
    for row in stats:
        row["coverage_pct"] = None
        row["missing_pct"] = None
        row["low_coverage"] = None
        if row.get("metric_kind") == "event_rate":
            row["event_rate_pct"] = None
        row["denominator_resolved"] = False


def _export_is_demo_like(source: Dict[str, Any], desc: Dict[str, Any]) -> bool:
    text = " ".join(
        str(v or "")
        for v in [
            source.get("label"),
            source.get("database"),
            source.get("path"),
            desc.get("label"),
            desc.get("database"),
            desc.get("path"),
            (desc.get("summary") or {}).get("database"),
        ]
    ).lower()
    return any(marker in text for marker in ("mock", "demo", "seeded"))


def _feature_stats(
    root: Path,
    concepts: Iterable[str],
    export_index: Dict[str, Any],
    entity_ids: set[str],
    *,
    denominator: Optional[int] = None,
    denominator_resolved: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    concept_to_file = export_index.get("concept_to_file") or {}
    if denominator is None:
        denominator = max(len(entity_ids), 1)
    for concept_id in concepts:
        item = concept_to_file.get(concept_id)
        if not item:
            continue
        file_name = str(item.get("file") or "")
        path = root / file_name
        columns = [str(c) for c in item.get("columns") or []]
        selected = [c for c in ["stay_id", *_TIME_COLUMNS, concept_id] if c in columns]
        if "stay_id" not in selected or concept_id not in selected:
            continue
        if _feature_scan_too_large(item):
            out.append(_metadata_feature_stat(concept_id, item, denominator, columns))
            if len(out) >= _MAX_FEATURE_STATS:
                break
            continue
        try:
            import pandas as pd

            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path, columns=selected)
            elif path.suffix.lower() == ".xlsx":
                frame = pd.read_excel(path, usecols=selected)
            else:
                frame = pd.read_csv(path, usecols=selected)
        except Exception:
            continue
        if frame.empty or concept_id not in frame.columns:
            continue
        entity_col = frame["stay_id"].map(entity_id_contract.normalize_entity_id)
        is_event_rate = _is_event_rate_concept(concept_id)
        non_null_mask = frame[concept_id].notna()
        non_null = frame[non_null_mask].copy()
        observed_entities = int(entity_col[non_null_mask].nunique())
        if is_event_rate:
            event_mask = _event_positive_mask(frame[concept_id])
            event_entities = int(entity_col[event_mask].nunique())
            event_records = int(event_mask.sum())
            event_rate = round(event_entities / denominator * 100, 1)
            out.append(
                {
                    "concept_id": concept_id,
                    "label": _concept_label(concept_id),
                    "module": str(item.get("module") or ""),
                    "metric_kind": "event_rate",
                    "records": event_records,
                    "observed_entities": event_entities,
                    "event_entities": event_entities,
                    "non_event_entities": max(denominator - event_entities, 0),
                    "denominator_entities": denominator,
                    "event_rate_pct": event_rate,
                    "coverage_pct": event_rate,
                    "missing_pct": None,
                    "low_coverage": False,
                    "time_indexed": any(col in frame.columns for col in _TIME_COLUMNS),
                    "numeric_summary": {
                        "available": False,
                        "kind": "event_indicator",
                    },
                    "summary_label": "Binary/event indicator; non-events are not missing.",
                    "status": "ready",
                }
            )
            if len(out) >= _MAX_FEATURE_STATS:
                break
            continue
        records = int(len(non_null))
        nums = dataio._numeric_values(non_null[concept_id])[:10000]
        coverage = round(observed_entities / denominator * 100, 1)
        out.append(
            {
                "concept_id": concept_id,
                "label": _concept_label(concept_id),
                "module": str(item.get("module") or ""),
                "metric_kind": "coverage",
                "records": records,
                "observed_entities": observed_entities,
                "denominator_entities": denominator,
                "coverage_pct": coverage,
                "missing_pct": round(100 - coverage, 1),
                "low_coverage": coverage < 50,
                "time_indexed": any(col in frame.columns for col in _TIME_COLUMNS),
                "numeric_summary": _numeric_summary(nums),
                "status": "ready" if records else "missing",
            }
        )
        if len(out) >= _MAX_FEATURE_STATS:
            break
    if not denominator_resolved:
        _mark_denominator_unresolved(out)
    return out


def _bounded_feature_sample_stat(
    root: Path,
    *,
    concept_id: str,
    item: Dict[str, Any],
    denominator: int,
    max_records: int,
) -> Optional[Dict[str, Any]]:
    file_name = str(item.get("file") or "")
    if not file_name:
        return None
    path = root / file_name
    columns = [str(c) for c in item.get("columns") or []]
    selected = _selected_feature_columns(columns, concept_id)
    if "stay_id" not in selected or concept_id not in selected:
        stat = _metadata_feature_stat(concept_id, item, denominator, columns)
        stat.update(
            {
                "status": "sample_unavailable",
                "summary_label": "Concept is present, but the table cannot be sampled without an entity key.",
            }
        )
        return stat
    try:
        frame = _read_bounded_feature_frame(path, selected, max_records)
    except Exception as exc:
        stat = _metadata_feature_stat(concept_id, item, denominator, columns)
        stat.update(
            {
                "status": "sample_unavailable",
                "sample_limit_records": max_records,
                "error_type": exc.__class__.__name__,
                "summary_label": "The bounded sample could not be read safely; only manifest/schema presence is available.",
            }
        )
        return stat
    if frame.empty or concept_id not in frame.columns:
        return {
            "concept_id": concept_id,
            "label": _concept_label(concept_id),
            "module": str(item.get("module") or ""),
            "metric_kind": "coverage",
            "records": 0,
            "sample_records": int(len(frame)),
            "records_declared": _safe_nonnegative_int(item.get("rows")),
            "observed_entities": 0,
            "sample_entities": 0,
            "denominator_entities": denominator,
            "coverage_pct": 0.0,
            "missing_pct": 100.0,
            "low_coverage": True,
            "time_indexed": any(col in frame.columns for col in _TIME_COLUMNS),
            "time_orderable": False,
            "time_value_count": 0,
            "coverage_basis": "bounded_file_head_sample",
            "sample_limit_records": max_records,
            "numeric_summary": {"available": False, "kind": "empty_sample"},
            "summary_label": "No usable values were found in the bounded sample.",
            "status": "missing",
        }
    entity_col = frame["stay_id"].map(entity_id_contract.normalize_entity_id)
    time_answerability = _bounded_time_answerability(frame)
    sample_entities = max(int(entity_col.nunique()), 1)
    is_event_rate = _is_event_rate_concept(concept_id)
    if is_event_rate:
        event_mask = _event_positive_mask(frame[concept_id])
        event_entities = int(entity_col[event_mask].nunique())
        event_records = int(event_mask.sum())
        event_rate = round(event_entities / sample_entities * 100, 1)
        return {
            "concept_id": concept_id,
            "label": _concept_label(concept_id),
            "module": str(item.get("module") or ""),
            "metric_kind": "event_rate",
            "records": event_records,
            "sample_records": int(len(frame)),
            "records_declared": _safe_nonnegative_int(item.get("rows")),
            "observed_entities": event_entities,
            "sample_entities": sample_entities,
            "event_entities": event_entities,
            "non_event_entities": max(sample_entities - event_entities, 0),
            "denominator_entities": denominator,
            "event_rate_pct": event_rate,
            "coverage_pct": event_rate,
            "missing_pct": None,
            "low_coverage": False,
            "time_indexed": any(col in frame.columns for col in _TIME_COLUMNS),
            **time_answerability,
            "coverage_basis": "bounded_file_head_sample",
            "sample_limit_records": max_records,
            "numeric_summary": {"available": False, "kind": "event_indicator"},
            "summary_label": "Sampled binary/event indicator; non-events are not missing.",
            "status": "ready",
        }
    non_null_mask = frame[concept_id].notna()
    non_null = frame[non_null_mask].copy()
    records = int(len(non_null))
    observed_entities = int(entity_col[non_null_mask].nunique())
    coverage = round(observed_entities / sample_entities * 100, 1)
    nums = dataio._numeric_values(non_null[concept_id])[:10000]
    return {
        "concept_id": concept_id,
        "label": _concept_label(concept_id),
        "module": str(item.get("module") or ""),
        "metric_kind": "coverage",
        "records": records,
        "sample_records": int(len(frame)),
        "records_declared": _safe_nonnegative_int(item.get("rows")),
        "observed_entities": observed_entities,
        "sample_entities": sample_entities,
        "denominator_entities": denominator,
        "coverage_pct": coverage,
        "missing_pct": round(100 - coverage, 1),
        "low_coverage": coverage < 50,
        "time_indexed": any(col in frame.columns for col in _TIME_COLUMNS),
        **time_answerability,
        "coverage_basis": "bounded_file_head_sample",
        "sample_limit_records": max_records,
        "numeric_summary": _numeric_summary(nums),
        "summary_label": "Coverage is computed inside the bounded sample only.",
        "status": "ready" if records else "missing",
    }


def _bounded_time_answerability(frame: Any) -> Dict[str, Any]:
    import pandas as pd

    for column in _TIME_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_count = int(numeric.notna().sum())
        if numeric_count:
            return {
                "time_orderable": True,
                "time_value_count": numeric_count,
                "time_value_kind": "numeric_offset",
            }
        datetimes = pd.to_datetime(values, errors="coerce", utc=True)
        datetime_count = int(datetimes.notna().sum())
        if datetime_count:
            return {
                "time_orderable": True,
                "time_value_count": datetime_count,
                "time_value_kind": "timestamp",
            }
    return {
        "time_orderable": False,
        "time_value_count": 0,
        "time_value_kind": "unavailable",
    }


def _selected_feature_columns(columns: List[str], concept_id: str) -> List[str]:
    selected: List[str] = []
    for col in ["stay_id", *_TIME_COLUMNS, concept_id]:
        if col in columns and col not in selected:
            selected.append(col)
    return selected


def _read_bounded_feature_frame(
    path: Path, columns: List[str], max_records: int
) -> Any:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        remaining = max(0, max_records)
        frames = []
        for batch in parquet.iter_batches(
            batch_size=max(1, min(remaining or max_records, 65_536)),
            columns=columns,
        ):
            frame = batch.to_pandas()
            piece = frame.head(remaining)
            frames.append(piece)
            remaining -= len(piece)
            if remaining <= 0:
                break
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)
    if suffix == ".xlsx":
        return pd.read_excel(path, usecols=columns, nrows=max_records)
    return pd.read_csv(path, usecols=columns, nrows=max_records)


def _sample_record_limit(value: Any) -> int:
    rows = _safe_nonnegative_int(value)
    if rows is None:
        return _IDEA_SAMPLE_DEFAULT_RECORDS
    return max(100, min(rows, _IDEA_SAMPLE_MAX_RECORDS))


def _bounded_sample_interpretation(
    stats: List[Dict[str, Any]],
    *,
    missing_required: List[str],
    low_coverage: List[Dict[str, Any]],
) -> List[str]:
    notes = [
        "This is an outcome-blind bounded sample check. It can support feasibility triage but is not manuscript source data.",
    ]
    unavailable = [row for row in stats if row.get("status") == "sample_unavailable"]
    if unavailable:
        notes.append(
            f"{len(unavailable)} feature(s) still have schema-only evidence because bounded sampling was unavailable."
        )
    if missing_required:
        notes.append(
            f"{len(missing_required)} required concept(s) were not verified in the bounded sample."
        )
    if low_coverage:
        labels = ", ".join(
            _concept_label(row.get("concept_id")) for row in low_coverage[:4]
        )
        notes.append(
            f"Low sample coverage remains for: {labels}. Confirm denominator and missingness before Agent execution."
        )
    if len(notes) == 1:
        notes.append(
            "Required concepts were sample-checked without returning raw records or direct identifiers."
        )
    return notes


def _feature_scan_too_large(item: Dict[str, Any]) -> bool:
    rows = _safe_nonnegative_int(item.get("rows"))
    return rows is not None and rows > _IDEA_FEATURE_ROW_SCAN_LIMIT


def _metadata_feature_stat(
    concept_id: str,
    item: Dict[str, Any],
    denominator: int,
    columns: List[str],
) -> Dict[str, Any]:
    return {
        "concept_id": concept_id,
        "label": _concept_label(concept_id),
        "module": str(item.get("module") or ""),
        "metric_kind": "schema_presence",
        "records": None,
        "records_declared": _safe_nonnegative_int(item.get("rows")),
        "observed_entities": None,
        "denominator_entities": denominator,
        "coverage_pct": None,
        "missing_pct": None,
        "low_coverage": None,
        "time_indexed": any(col in columns for col in _TIME_COLUMNS),
        "numeric_summary": {
            "available": False,
            "kind": "metadata_only",
        },
        "summary_label": (
            "Concept is present in the export schema; row-level feasibility was "
            "deferred because the module exceeds the Idea Mining preflight scan limit."
        ),
        "coverage_basis": "manifest_file_inventory",
        "scan_limit_rows": _IDEA_FEATURE_ROW_SCAN_LIMIT,
        "status": "metadata_only",
    }


def _safe_nonnegative_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _concept_unit(concept_id: str) -> str:
    entry = concept_catalog.CONCEPT_DICTIONARY.get(concept_id)
    if not entry or len(entry) < 3:
        return ""
    return str(entry[2] or "").strip().lower()


def _is_event_rate_concept(concept_id: str) -> bool:
    """Boolean concepts represent positive indicators, not missingness.

    Many EasyICU boolean exports are sparse event/indicator tables: negative
    patients may be absent from the concept file.  Treating absent negatives as
    missing coverage recreates the classic Sepsis-3 pitfall.
    """

    return _concept_unit(concept_id) == "boolean"


def _event_positive_mask(series: Any) -> Any:
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    lowered = series.astype("string").str.strip().str.lower()
    truthy = lowered.isin(_EVENT_TRUE_STRINGS)
    falsy = lowered.isin(_EVENT_FALSE_STRINGS)
    positive = (numeric_mask & (numeric > 0)) | truthy
    return positive & ~falsy


def _concept_feasibility(
    row: Dict[str, Any], export_index: Dict[str, Any]
) -> Dict[str, Any]:
    concept_id = row.get("concept_id")
    in_export = concept_id in (export_index.get("concept_to_file") or {})
    demo_like = bool(export_index.get("demo_like"))
    if in_export and demo_like:
        tier = "demo_only"
        note = "Concept is present only in the active MOCK/demo export; switch to a real EasyICU export before treating it as feasible."
    elif in_export:
        tier = "executable"
        note = "Concept is in the EasyICU dictionary and present in the active export."
    else:
        tier = "T1_reextract"
        note = "Concept is in the EasyICU dictionary but not present in the active export; re-extract or add the module."
    item = (export_index.get("concept_to_file") or {}).get(concept_id) or {}
    return {
        "concept_id": concept_id,
        "label": row.get("label"),
        "unit": row.get("unit"),
        "module": row.get("module") or item.get("module"),
        "role": row.get("role") or "feature",
        "matched_alias": row.get("matched_alias"),
        "dictionary_present": True,
        "active_export_present": bool(in_export),
        "tier": tier,
        "human_note": note,
    }


def _overall_feasibility(
    rows: List[Dict[str, Any]], export_index: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if not rows:
        return {
            "tier": "T3_not_in_db",
            "label": "No mapped EasyICU concept",
            "reason": "No source phrase mapped to the current EasyICU dictionary.",
        }
    if (export_index or {}).get("demo_like"):
        return {
            "tier": "demo_only",
            "label": "Demo export only",
            "reason": "The mapped concepts are available only on the active MOCK/demo export. Select a real local EasyICU export before presenting feasibility.",
        }
    if all(row.get("tier") == "executable" for row in rows):
        return {
            "tier": "executable",
            "label": "Executable on active export",
            "reason": "All required mapped concepts are present in the active export.",
        }
    if any(row.get("tier") == "T1_reextract" for row in rows):
        return {
            "tier": "T1_reextract",
            "label": "Needs re-extraction or extra modules",
            "reason": "At least one mapped concept is known to EasyICU but absent from the active export.",
        }
    return {
        "tier": "T3_not_in_db",
        "label": "Not supported",
        "reason": "No executable dictionary mapping.",
    }


def _prior_art(
    source: Dict[str, Any],
    title: str,
    *,
    predictor: Optional[Dict[str, Any]] = None,
    outcome: Optional[Dict[str, Any]] = None,
    topic_aliases: Iterable[str] = (),
) -> Dict[str, Any]:
    return {
        "status": "not_checked_external_search_required",
        "novelty_label": "unknown_until_search",
        "direct_same_topic_hits": [],
        "public_database_used_by_prior_work": "unknown_until_search",
        "reason": "Prior-art interpretation has not been run. Use the source article as inspiration, then run opt-in metadata search before claiming novelty.",
        "opportunity_frame": "Existing trials, reviews, or editorials should shape the ICU-database question: comparator, subgroup, timing window, outcome horizon, and whether the new angle is exploratory rather than already answered.",
        "next_use": "After opt-in, classify the literature as already answered, partially answered, or inspiration for a new ICU exploratory analysis.",
        "queries_to_run": _prior_art_queries(
            source,
            title,
            exposure=(predictor or {}).get("concept_id")
            or (predictor or {}).get("label"),
            outcome=(outcome or {}).get("concept_id") or (outcome or {}).get("label"),
            topic_aliases=topic_aliases,
        ),
    }


def _blocked_features(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_type = str(body.get("source_type") or "manual")
    rows = [
        {
            "id": "live_literature_search",
            "status": "blocked",
            "reason": "External search requires explicit network/provider opt-in and will be implemented behind the provider gate.",
        },
        {
            "id": "pdf_fulltext_storage",
            "status": "blocked",
            "reason": "PDF/full text is not stored in this local web artifact; paste a bounded excerpt or enable a parser stage later.",
        },
    ]
    if source_type in {"url", "pdf", "frontier"}:
        rows.append(
            {
                "id": f"{source_type}_adapter",
                "status": "planned",
                "reason": "The UI captures metadata now; the live adapter will attach to this same ledger contract.",
            }
        )
    return rows


def _record_history(payload: Dict[str, Any]) -> None:
    history = _read_history()
    source = (payload.get("source_evidence") or [{}])[0]
    idea = (payload.get("idea_ledger") or [{}])[0]
    row = {
        "run_id": payload.get("run_id"),
        "created_at": payload.get("created_at"),
        "history_key": _history_key(payload.get("run_id"), payload.get("created_at")),
        "title": idea.get("idea_title"),
        "source_title": source.get("title"),
        "source_year": source.get("year"),
        "journal": source.get("journal"),
        "go_no_go": idea.get("go_no_go"),
        "feasibility_tier": (idea.get("feasibility") or {}).get("tier"),
        "run_dir": payload.get("run_dir"),
    }
    history = [row] + [item for item in history if item.get("run_id") != row["run_id"]]
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_PATH.write_text(
        json.dumps(history[:100], indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_run(payload: Dict[str, Any]) -> Path:
    run_dir = _run_dir(str(payload["run_id"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in {
        "source_evidence.json": payload.get("source_evidence"),
        "idea_ledger.json": payload.get("idea_ledger"),
        "pre_experiment.json": payload.get("pre_experiment"),
        "handoff_plan.json": payload.get("handoff_plan"),
        "idea_mining_run.json": {k: v for k, v in payload.items() if k != "run_dir"},
    }.items():
        (run_dir / name).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return run_dir


def _load_run(run_id: str) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    path = _run_dir(run_id) / "idea_mining_run.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_handoff(run_id: str) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    path = _run_dir(run_id) / "idea_handoff.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_plan(run_id: str) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    path = _run_dir(run_id) / "idea_plan.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_payload(artifact: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the plan body from current envelopes or legacy direct plans."""
    if not isinstance(artifact, dict):
        return {}
    nested = artifact.get("plan")
    if isinstance(nested, dict):
        return nested
    return artifact


def _load_bounded_sample(run_id: str) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    path = _run_dir(run_id) / _BOUNDED_FEASIBILITY_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def idea_execution_readiness_binding(
    run_id: str,
    idea_id: str,
    *,
    source_path: Any = None,
) -> Dict[str, Any]:
    """Validate the exact literature + data receipts required for handoff."""

    adjudication = prior_art_adjudication_binding(run_id, idea_id)
    if adjudication.get("prior_art_decision") != "differentiated":
        raise IdeaMiningWebError(
            {
                "error": "idea_prior_art_not_differentiated",
                "decision": adjudication.get("prior_art_decision"),
            }
        )
    path = _run_dir(run_id) / _BOUNDED_FEASIBILITY_FILENAME
    try:
        raw = path.read_bytes()
        feasibility = json.loads(raw.decode("utf-8"))
    except FileNotFoundError as exc:
        raise IdeaMiningWebError(
            {
                "error": "idea_source_feasibility_required",
                "reason": "Run source-bound Idea feasibility before accepting the handoff.",
            }
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdeaMiningWebError({"error": "idea_source_feasibility_invalid"}) from exc
    if not isinstance(feasibility, Mapping):
        raise IdeaMiningWebError({"error": "idea_source_feasibility_invalid"})
    if str(feasibility.get("schema_version") or "") != (
        "easyicu.web_idea_bounded_sample_feasibility/2"
    ):
        raise IdeaMiningWebError(
            {
                "error": "idea_source_feasibility_schema_outdated",
                "reason": (
                    "Re-run source feasibility so the receipt includes typed "
                    "clinical-construct answerability."
                ),
            }
        )
    if (
        str(feasibility.get("run_id") or "") != run_id
        or str(feasibility.get("idea_id") or "") != idea_id
    ):
        raise IdeaMiningWebError({"error": "idea_source_feasibility_identity_mismatch"})
    if feasibility.get("prior_art_adjudication_binding") != adjudication:
        raise IdeaMiningWebError(
            {
                "error": "idea_source_feasibility_stale_adjudication",
                "reason": "The prior-art adjudication changed after data feasibility.",
            }
        )
    if str(feasibility.get("status") or "") != "ready":
        raise IdeaMiningWebError(
            {
                "error": "idea_source_feasibility_not_ready",
                "status": feasibility.get("status"),
                "blockers": list(feasibility.get("blockers") or [])[:12],
            }
        )
    construct_answerability = feasibility.get("construct_answerability")
    if not isinstance(construct_answerability, list) or not construct_answerability:
        raise IdeaMiningWebError({"error": "idea_construct_answerability_required"})
    unresolved_constructs = [
        (
            str(row.get("construct_id") or "unknown_construct")
            if isinstance(row, Mapping)
            else "invalid_construct"
        )
        for row in construct_answerability
        if not isinstance(row, Mapping) or row.get("verdict") != "ready"
    ]
    if unresolved_constructs:
        raise IdeaMiningWebError(
            {
                "error": "idea_construct_answerability_not_ready",
                "constructs": unresolved_constructs[:12],
            }
        )
    source = feasibility.get("source")
    source = source if isinstance(source, Mapping) else {}
    if bool(source.get("demo_like")):
        raise IdeaMiningWebError({"error": "idea_source_feasibility_demo_only"})
    clean_source_path = str(source_path or "").strip()
    if clean_source_path:
        expected_path_hash = _sha256(clean_source_path)[:16]
        if str(source.get("path_hash") or "") != expected_path_hash:
            raise IdeaMiningWebError(
                {
                    "error": "idea_source_feasibility_source_mismatch",
                    "reason": "The bound StudyContext source changed after feasibility.",
                }
            )
    bindings = feasibility.get("concept_bindings")
    bindings = _validated_concept_bindings(bindings)
    if not all(bindings.get(role) for role in _CONCEPT_BINDING_ROLES):
        raise IdeaMiningWebError(
            {"error": "idea_source_feasibility_concept_contract_incomplete"}
        )
    modules = {
        str(row.get("concept_id") or ""): str(row.get("module") or "")
        for row in feasibility.get("feature_statistics") or []
        if isinstance(row, Mapping)
        and str(row.get("concept_id") or "")
        and str(row.get("module") or "")
    }
    required = list(
        dict.fromkeys(
            [bindings[role] for role in _CONCEPT_BINDING_ROLES if bindings.get(role)]
            + list(bindings.get("covariates") or [])
        )
    )
    missing_modules = [value for value in required if value not in modules]
    if missing_modules:
        raise IdeaMiningWebError(
            {
                "error": "idea_source_feasibility_concept_contract_incomplete",
                "missing_concept_modules": missing_modules,
            }
        )
    return {
        **adjudication,
        "source_feasibility_schema_version": str(feasibility.get("schema_version")),
        "source_feasibility_sha256": hashlib.sha256(raw).hexdigest(),
        "source_feasibility_status": "ready",
        "source_feasibility_created_at": str(feasibility.get("created_at") or ""),
        "source_id": source.get("source_id"),
        "source_path_hash": source.get("path_hash"),
        "concept_bindings": bindings,
        "concept_modules": modules,
        "source_feasibility_summary": {
            "status": feasibility.get("status"),
            "source": {
                key: source.get(key)
                for key in ("source_id", "label", "database", "demo_like")
                if source.get(key) is not None
            },
            "cohort": (
                feasibility.get("cohort")
                if isinstance(feasibility.get("cohort"), Mapping)
                else {}
            ),
            "feature_statistics": list(feasibility.get("feature_statistics") or [])[
                :12
            ],
            "design_answerability": (
                feasibility.get("design_answerability")
                if isinstance(feasibility.get("design_answerability"), Mapping)
                else {}
            ),
            "construct_answerability": construct_answerability[:12],
            "missing_required_concepts": list(
                feasibility.get("missing_required_concepts") or []
            )[:24],
            "blockers": list(feasibility.get("blockers") or [])[:12],
            "interpretation": list(feasibility.get("interpretation") or [])[:8],
        },
        "execution_ready_for_confirmation": True,
    }


def _load_prior_art(run_id: str) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    path = _run_dir(run_id) / "prior_art_check.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _project_for_run(run_id: str) -> Optional[Dict[str, Any]]:
    for row in _read_agent_projects():
        if str(row.get("source_run_id") or "") == str(run_id):
            return row
    return None


def _selected_idea(
    payload: Dict[str, Any], idea_id: str = ""
) -> Optional[Dict[str, Any]]:
    idea_id = str(idea_id or "").strip()
    if not idea_id:
        return None
    ideas = payload.get("idea_ledger") or []
    return next(
        (row for row in ideas if str(row.get("idea_id") or "") == idea_id), None
    )


def _read_history() -> List[Dict[str, Any]]:
    try:
        rows = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    valid: List[Dict[str, Any]] = []
    for index, row in enumerate(rows if isinstance(rows, list) else []):
        if not isinstance(row, dict):
            continue
        run_id = str(row.get("run_id") or "").strip()
        if run_id and (_run_dir(run_id) / "idea_mining_run.json").exists():
            row["storage"] = "local_run_dir"
            row["history_key"] = str(
                row.get("history_key")
                or _history_key(run_id, row.get("created_at") or index)
            )
            valid.append(row)
    return valid


def _read_agent_projects() -> List[Dict[str, Any]]:
    try:
        rows = json.loads(_AGENT_PROJECTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        rows = []
    valid: List[Dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        project_dir_raw = str(row.get("project_dir") or "").strip()
        if not project_dir_raw:
            continue
        project_dir = Path(project_dir_raw)
        if project_dir.is_dir() and (project_dir / "project_seed.json").exists():
            row["storage"] = "local_project_seed"
            valid.append(row)
    return valid


def _handoff_plan_is_stale(
    handoff: Dict[str, Any], run_id: str, body: Dict[str, Any]
) -> bool:
    """True when the frozen handoff plan no longer reflects the latest edits.

    ``_handoff_needs_refresh`` only checks prior-art drift, so a user who edits
    the plan (or calls /api/ideas/plan) after clicking Handoff — without
    re-clicking Handoff — would have their revision silently dropped: the
    stale plan is seeded into Agent Projects. Detect that here so the handoff
    is re-frozen from the newer plan.
    """
    frozen = handoff.get("handoff_plan") or {}
    plan_artifact = _load_plan(run_id)

    def _comparable(value: Dict[str, Any]) -> str:
        return json.dumps(
            {
                key: value.get(key)
                for key in (
                    "human_plan_notes",
                    "selection_mode",
                    "plan_status",
                    "analysis_plan",
                    "execution_gate",
                )
            },
            sort_keys=True,
            default=str,
        )

    body_edits = str(body.get("plan_edits") or "").strip()
    if body_edits and body_edits != str(frozen.get("human_plan_notes") or "").strip():
        return True
    if plan_artifact is None:
        return False
    plan = _plan_payload(plan_artifact)
    return _comparable(plan) != _comparable(frozen)


def _handoff_needs_refresh(handoff: Dict[str, Any], run_id: str) -> bool:
    # Legacy Web envelopes predate the canonical research-agent packet. Rebuild
    # them once; envelopes that claim a canonical packet are validated later
    # and must fail closed rather than being silently refreshed after tampering.
    if is_legacy_handoff_envelope(handoff):
        return True
    prior_art_check = _load_prior_art(run_id)
    if prior_art_check and not handoff.get("prior_art_check"):
        return True
    if not prior_art_check:
        return False
    prior = prior_art_check.get("prior_art") or {}
    plan_review = (handoff.get("handoff_plan") or {}).get("prior_art_review") or {}
    return bool(prior.get("search_performed")) != bool(
        plan_review.get("search_performed")
    ) or str(prior.get("status") or "") != str(plan_review.get("status") or "")


def _agent_project_seed(
    handoff: Dict[str, Any],
    canonical_packet: DiscoveryHandoffPacket,
) -> Dict[str, Any]:
    idea = handoff.get("selected_ledger_row") or {}
    plan = handoff.get("handoff_plan") or {}
    agent_seed = handoff.get("agent_seed") or {}
    base_id = _slug(agent_seed.get("study_id") or idea.get("idea_title") or "idea")
    root_id = f"idea-{base_id}" if not str(base_id).startswith("idea-") else base_id
    run_suffix = _sha256(
        str(handoff.get("run_id") or handoff.get("created_at") or root_id)
    )[:8]
    study_id = (
        root_id if root_id.endswith(f"-{run_suffix}") else f"{root_id}-{run_suffix}"
    )
    source = (handoff.get("source_evidence") or [{}])[0]
    pre = handoff.get("pre_experiment") or {}
    prior_art_check = handoff.get("prior_art_check")
    active_export_contract = plan.get(
        "active_export_contract"
    ) or _active_export_contract(pre)
    prior_art_review = plan.get("prior_art_review") or _prior_art_review(
        prior_art_check
    )
    execution_gate = plan.get("execution_gate") or _execution_gate(
        idea,
        pre,
        prior_art_check,
    )
    concepts = [
        {
            "role": row.get("role") or "feature",
            "concept_id": row.get("concept_id"),
            "label": row.get("label"),
            "module": row.get("module"),
            "tier": row.get("tier"),
            "active_export_present": bool(row.get("active_export_present")),
        }
        for row in idea.get("mapped_concepts") or []
    ]
    return {
        "schema_version": "easyicu.agent_project_seed/1",
        "created_at": _now(),
        "study_id": study_id,
        "title": idea.get("idea_title")
        or handoff.get("candidate_topic")
        or "Idea-derived study",
        "mode": "analysis",
        "status": "seeded_from_idea",
        "stage": 0,
        "source_run_id": handoff.get("run_id"),
        "source_idea_id": handoff.get("idea_id"),
        "question": plan.get("research_question") or agent_seed.get("question"),
        "cohort": _seed_cohort_label(active_export_contract)
        or (plan.get("cohort") or {}).get("default")
        or "adult ICU cohort from active EasyICU export",
        "source": {
            "source_type": source.get("source_type"),
            "source_origin": source.get("source_origin"),
            "source_origin_label": source.get("source_origin_label"),
            "title": source.get("title"),
            "year": source.get("year"),
            "journal": source.get("journal"),
            "doi": source.get("doi"),
            "pmid": source.get("pmid"),
            "citation_key": source.get("citation_key"),
            "zotero_key": source.get("zotero_key"),
            "quote": source.get("evidence_quote"),
            "source_text_hash": source.get("source_text_sha256"),
        },
        "concepts": concepts,
        "pre_experiment_summary": plan.get("pre_experiment_summary")
        or {
            "status": pre.get("status"),
            "entities": (pre.get("cohort") or {}).get("entities"),
            "feature_count": len(pre.get("feature_statistics") or []),
        },
        "active_export_contract": active_export_contract,
        "prior_art_review": prior_art_review,
        "execution_gate": execution_gate,
        "analysis_plan": list(plan.get("analysis_plan") or []),
        "human_plan_notes": plan.get("human_plan_notes"),
        "canonical_handoff": canonical_packet.model_dump(mode="json"),
        "canonical_handoff_path": handoff.get("canonical_handoff_path"),
        "canonical_handoff_sha256": handoff.get("canonical_handoff_sha256"),
        "human_confirmed": canonical_packet.human_confirmed,
        "analysis_ready": canonical_packet.analysis_ready,
        "reportable": False,
        "draft_unlocked": False,
        "requires_human_confirmation": True,
        "project_dir": str(_AGENT_PROJECTS_ROOT / study_id),
        "runs": [
            {
                "label": "idea handoff",
                "scope": "metadata seed",
                "status": "complete",
                "created_at": handoff.get("created_at"),
            },
            *_seed_gate_runs(active_export_contract, prior_art_review),
        ],
    }


def _active_export_contract(pre_experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Build a row-safe contract for the active export used by Idea Mining."""

    source = pre_experiment.get("source") or {}
    cohort = pre_experiment.get("cohort") or {}
    stats = pre_experiment.get("feature_statistics") or []
    missing = [
        str(cid) for cid in pre_experiment.get("missing_required_concepts") or [] if cid
    ]
    return {
        "status": pre_experiment.get("status") or "blocked",
        "payload_scope": pre_experiment.get("payload_scope") or "no_patient_rows",
        "label": source.get("label"),
        "database": source.get("database"),
        "path_hash": source.get("path_hash"),
        "demo_like": bool(source.get("demo_like")),
        "entities": cohort.get("entities"),
        "modules": cohort.get("modules"),
        "total_rows": cohort.get("total_rows"),
        "feature_count": len(stats),
        "feature_concepts": [
            row.get("concept_id") for row in stats[:12] if row.get("concept_id")
        ],
        "missing_required_concepts": missing,
        "reportable": False,
        "reason": pre_experiment.get("reason"),
    }


def _prior_art_review(prior_art_check: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    prior = (prior_art_check or {}).get("prior_art") or {}
    return {
        "status": prior.get("status") or "not_checked",
        "search_performed": bool(prior.get("search_performed")),
        "network_calls": int(prior.get("network_calls") or 0),
        "result_count": int(prior.get("result_count") or 0),
        "public_database_used_by_prior_work": prior.get(
            "public_database_used_by_prior_work"
        )
        or "unknown_until_search",
        "queries_to_run": list(prior.get("queries_to_run") or [])[:4],
        "direct_same_topic_hit_count": len(prior.get("direct_same_topic_hits") or []),
        "opportunity_frame": prior.get("opportunity_frame"),
        "next_use": prior.get("next_use"),
        "reason": prior.get("reason") or "Prior-art review has not been run.",
    }


#: Stable identifiers for the four conditions this gate can block on, paired
#: with the sentence older seeds carry. The gate has always decided from typed
#: conditions, but only emitted the sentence — so the UI regex-matched English
#: prose to work out what the user should do about it. The code is the
#: contract; the sentence stays for seeds already written to disk.
EXECUTION_GATE_BLOCKERS: Dict[str, str] = {
    "export_not_real": "prepare or select a real EasyICU export",
    "required_concepts_missing": "re-extract or confirm missing required concepts",
    "prior_art_not_reviewed": "run prior-art review before Agent execution",
    "idea_not_recommended": "resolve idea feasibility before Agent execution",
    "prior_art_not_differentiated": "complete a differentiated typed prior-art adjudication",
    "source_feasibility_not_ready": "complete source-bound data feasibility",
    "study_definition_not_confirmed": "confirm the complete study definition",
}


def _execution_gate(
    idea: Dict[str, Any],
    pre_experiment: Dict[str, Any],
    prior_art_check: Optional[Dict[str, Any]],
    readiness: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    export_status = str(pre_experiment.get("status") or "blocked").lower()
    missing = [
        str(cid) for cid in pre_experiment.get("missing_required_concepts") or [] if cid
    ]
    prior = (prior_art_check or {}).get("prior_art") or {}
    codes: List[str] = []
    if export_status in {"", "blocked", "demo_only"} or "demo" in export_status:
        codes.append("export_not_real")
    elif export_status == "partial" or missing:
        codes.append("required_concepts_missing")
    # A prior-art review only satisfies the gate when a search actually
    # completed: a blocked opt-in placeholder has search_performed=False, and
    # an attempted-but-failed search (status "search_failed") returned no
    # reviewable metadata, so neither counts as reviewed.
    prior_reviewed = bool(prior.get("search_performed")) and (
        str(prior.get("status") or "") != "search_failed"
    )
    if not prior_reviewed:
        codes.append("prior_art_not_reviewed")
    if idea.get("go_no_go") != "recommend":
        codes.append("idea_not_recommended")
    readiness = readiness if isinstance(readiness, Mapping) else {}
    if readiness.get("prior_art_decision") != "differentiated":
        codes.append("prior_art_not_differentiated")
    if readiness.get("source_feasibility_status") != "ready":
        codes.append("source_feasibility_not_ready")
    if not readiness.get("idea_definition_sha256"):
        codes.append("study_definition_not_confirmed")
    codes = list(dict.fromkeys(codes))
    blockers = [EXECUTION_GATE_BLOCKERS[code] for code in codes]
    return {
        "project_seed_allowed": True,
        "agent_run_ready_after_human_confirmation": not codes,
        "reportable": False,
        "draft_unlocked": False,
        "blockers": blockers,
        "blocker_codes": codes,
        "export_status": pre_experiment.get("status") or "blocked",
        "prior_art_status": prior.get("status") or "not_checked",
        "go_no_go": idea.get("go_no_go"),
        "prior_art_decision": readiness.get("prior_art_decision"),
        "source_feasibility_status": readiness.get("source_feasibility_status"),
        "execution_ready_for_confirmation": not codes,
    }


def _seed_cohort_label(active_export_contract: Dict[str, Any]) -> Optional[str]:
    label = str(active_export_contract.get("label") or "").strip()
    entities = active_export_contract.get("entities")
    if not label and entities is None:
        return None
    prefix = "adult ICU cohort"
    if label:
        prefix = f"{prefix} from {label}"
    if entities is None:
        return prefix
    try:
        count = f"{int(entities):,}"
    except (TypeError, ValueError):
        count = str(entities)
    return f"{prefix} (n={count})"


def _seed_gate_runs(
    active_export_contract: Dict[str, Any],
    prior_art_review: Dict[str, Any],
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    export_status = str(active_export_contract.get("status") or "").strip()
    if export_status:
        runs.append(
            {
                "label": "export feasibility",
                "scope": (
                    f"{export_status} · "
                    f"{active_export_contract.get('feature_count') or 0} feature(s)"
                ),
                "status": export_status,
                "created_at": _now(),
            }
        )
    prior_status = str(prior_art_review.get("status") or "").strip()
    if prior_status and prior_status != "not_checked":
        runs.append(
            {
                "label": "prior-art review",
                "scope": (
                    f"{prior_status} · "
                    f"{prior_art_review.get('result_count') or 0} metadata hit(s)"
                ),
                "status": prior_status,
                "created_at": _now(),
            }
        )
    return runs


def _prior_art_queries(
    source: Dict[str, Any],
    title: str,
    *,
    exposure: Any = None,
    outcome: Any = None,
    topic_aliases: Iterable[str] = (),
) -> List[str]:
    title = _clean(title or source.get("title") or "ICU idea", 180)
    candidate_scope = _discovery_topic_clause(
        title,
        {"exposure": exposure, "outcome": outcome},
    )
    exposure_phrase = _literature_concept_phrase(exposure)
    alias_terms = [
        _clean(value, 120)
        for value in topic_aliases
        if _clean(value, 120) and re.search(r"[A-Za-z]", str(value))
    ][:8]
    topic_clause = " OR ".join(
        _pubmed_title_abstract_clause(value) for value in alias_terms
    )
    if topic_clause:
        topic_clause = f"({topic_clause})"
    clinical_scope = (
        " AND ".join(
            value
            for value in (
                _pubmed_title_abstract_clause(exposure_phrase),
                topic_clause,
            )
            if value
        )
        if topic_clause
        else candidate_scope
    )
    conversational_scope = _conversational_literature_scope(source)
    population_filter = ""
    review_scope = clinical_scope
    if conversational_scope:
        candidate_scope = _conversational_candidate_scope(
            source,
            base_scope=conversational_scope,
        )
        clinical_scope = _conversational_variation_scope(
            source,
            base_scope=conversational_scope,
        )
        review_scope = conversational_scope
        population_filter = _conversational_population_filter(source)
    icu_clause = (
        '(ICU[Title/Abstract] OR "critical care"[Title/Abstract] OR '
        '"intensive care"[Title/Abstract])'
    )
    queries = [
        f"({clinical_scope}) AND {icu_clause}{population_filter}",
        f"({candidate_scope}) AND {icu_clause}{population_filter}",
        f"({clinical_scope}) AND {icu_clause} AND "
        "(cohort[Title/Abstract] OR observational[Title/Abstract] OR "
        "retrospective[Title/Abstract] OR prospective[Title/Abstract] OR "
        f"multicenter[Title/Abstract]){population_filter}",
        f"({review_scope}) AND {icu_clause} AND "
        "(review[Publication Type] OR systematic review[Title/Abstract] OR "
        f"guideline[Title/Abstract]){population_filter}",
        f"({review_scope}) AND {icu_clause} AND "
        "(MIMIC[Title/Abstract] OR eICU[Title/Abstract] OR "
        f'"critical care database"[Title/Abstract]){population_filter}',
    ]
    doi = _clean(source.get("doi") or "", 120)
    if doi:
        queries.insert(0, f'"{doi}"[DOI]')
    return queries[:_MAX_DISCOVERY_QUERIES]


_CONVERSATIONAL_PAIR_PROFILES = (
    (
        "lactate_aki",
        (
            (
                r"lactate|lactic|乳酸",
                ("lactate", "lactate clearance"),
                r"\blactat(?:e|ic|emia)\b",
            ),
            (
                r"aki|acute kidney injury|急性肾损伤",
                ("acute kidney injury", "AKI"),
                r"\bAKI\b|acute kidney injury",
            ),
        ),
    ),
    (
        "fluid_liberation",
        (
            (
                r"fluid balance|fluid overload|液体平衡|液体超负荷",
                ("cumulative fluid balance", "fluid balance", "fluid overload"),
                r"fluid balance|fluid overload",
            ),
            (
                r"ventilator liberation|weaning|extubation|撤机|脱机|拔管",
                (
                    "extubation failure",
                    "ventilator liberation",
                    "weaning",
                    "extubation",
                ),
                r"ventilator liberation|\bwean(?:ing|ed)?\b|\bextubat(?:e|ion|ed)\b",
            ),
        ),
    ),
    (
        "sedation_awakening",
        (
            (
                r"sedation|sedative|镇静|镇静药",
                (
                    "sedation interruption",
                    "sedative discontinuation",
                    "sedation weaning",
                ),
                r"sedat(?:ion|ive)|sedative discontinuation|sedation interruption",
            ),
            (
                r"awakening|wakefulness|delayed awakening|coma|unconscious|清醒|昏迷",
                ("awakening", "delayed awakening", "coma"),
                r"awakening|wakefulness|delayed awakening|\bcoma\b|unconscious",
            ),
        ),
    ),
)


def _conversational_pair_profile(
    text: str,
) -> Optional[Tuple[str, Tuple[Tuple[str, Tuple[str, ...], str], ...]]]:
    for profile_name, dimensions in _CONVERSATIONAL_PAIR_PROFILES:
        if all(re.search(pattern, text, re.I) for pattern, _, _ in dimensions):
            return profile_name, dimensions
    return None


def _conversational_literature_scope(source: Dict[str, Any]) -> str:
    """Preserve a bounded bilingual clinical phenomenon in PubMed queries.

    Idea Mining deliberately accepts an informal Chinese sentence.  When that
    sentence has no complete PICO, the dictionary-derived candidate title can
    collapse to a generic token such as ``ICU``.  This helper does not translate
    arbitrary prose or invent a study design; it only projects a few explicit,
    literature-bearing clinical/action phrases that the researcher actually
    supplied.  Returning an empty string leaves the existing query compiler in
    full control.
    """

    text = _norm_text(
        " ".join(str(source.get(key) or "") for key in ("title", "evidence_quote"))
    )
    pair_profile = _conversational_pair_profile(text)
    if pair_profile:
        _, dimensions = pair_profile
        clauses = []
        for _, aliases, _ in dimensions:
            clauses.append(
                "("
                + " OR ".join(_pubmed_title_abstract_clause(value) for value in aliases)
                + ")"
            )
        return " AND ".join(clauses)

    clinical_aliases: List[str] = []
    for pattern, aliases in (
        (
            r"hypotension|低血压",
            ("hypotension", "shock"),
        ),
        (r"blood pressure|血压", ("blood pressure",)),
        (r"delirium|谵妄", ("delirium",)),
        (r"fluid balance|液体平衡", ("fluid balance",)),
        (
            r"ventilator liberation|weaning|extubation|撤机|脱机|拔管",
            ("ventilator liberation", "extubation"),
        ),
    ):
        if re.search(pattern, text, re.I):
            clinical_aliases.extend(aliases)
            break
    if not clinical_aliases:
        return ""

    action_aliases: List[str] = []
    for pattern, aliases in (
        (
            r"management|treatment|intervention|处理|治疗|干预",
            ("management", "treatment"),
        ),
        (r"strategy|strategies|策略|方案", ("strategy", "management")),
        (
            r"variation|variability|difference|差异|不一致",
            ("practice variation", "variability"),
        ),
    ):
        if re.search(pattern, text, re.I):
            action_aliases.extend(aliases)
            break
    if not action_aliases:
        return ""

    clinical = " OR ".join(
        _pubmed_title_abstract_clause(value)
        for value in _dedupe_strings(clinical_aliases)[:3]
    )
    action = " OR ".join(
        _pubmed_title_abstract_clause(value)
        for value in _dedupe_strings(action_aliases)[:3]
    )
    return f"({clinical}) AND ({action})"


def _conversational_candidate_scope(source: Dict[str, Any], *, base_scope: str) -> str:
    """Make the candidate-topic stratum narrower than the clinical landscape."""

    text = _norm_text(
        " ".join(str(source.get(key) or "") for key in ("title", "evidence_quote"))
    )
    pair_profile = _conversational_pair_profile(text)
    if pair_profile:
        _, dimensions = pair_profile
        return " AND ".join(
            "("
            + " OR ".join(_pubmed_title_abstract_clause(alias) for alias in aliases[:2])
            + ")"
            for _, aliases, _ in dimensions
        )
    if re.search(r"nighttime|nocturnal|overnight|夜间|夜班", text, re.I):
        precise_scope = base_scope
        if re.search(r"hypotension|低血压", text, re.I):
            precise_scope = (
                f"({_pubmed_title_abstract_clause('hypotension')}) AND "
                f"({_pubmed_title_abstract_clause('management')} OR "
                f"{_pubmed_title_abstract_clause('treatment')})"
            )
        temporal = " OR ".join(
            _pubmed_title_abstract_clause(value)
            for value in (
                "nighttime",
                "nocturnal",
                "overnight",
                "after-hours",
                "out-of-hours",
                "night shift",
            )
        )
        return f"({precise_scope}) AND ({temporal})"
    if _asks_about_practice_variation(text):
        variation = " OR ".join(
            _pubmed_title_abstract_clause(value)
            for value in ("practice variation", "variability")
        )
        return f"({base_scope}) AND ({variation})"
    return base_scope


def _conversational_variation_scope(source: Dict[str, Any], *, base_scope: str) -> str:
    """Retain an explicitly observed practice difference in the broad stratum."""

    text = _norm_text(
        " ".join(str(source.get(key) or "") for key in ("title", "evidence_quote"))
    )
    if not _asks_about_practice_variation(text):
        return base_scope
    variation = " OR ".join(
        _pubmed_title_abstract_clause(value)
        for value in (
            "practice variation",
            "practice pattern",
            "practice patterns",
            "treatment variation",
            "management variation",
            "physician variation",
            "clinician variation",
            "organizational structure",
        )
    )
    return f"({base_scope}) AND ({variation})"


def _asks_about_practice_variation(text: str) -> bool:
    difference = re.search(
        r"variation|variability|difference|heterogeneity|差异|不一致|不同|差别",
        text,
        re.I,
    )
    practice_context = re.search(
        r"management|treatment|intervention|physician|clinician|doctor|staffing|"
        r"organization|team|hospital|unit|医生|医师|处理|处置|治疗|干预|值班|"
        r"排班|组织|团队|医院|科室",
        text,
        re.I,
    )
    return bool(difference and practice_context)


def _conversational_population_filter(source: Dict[str, Any]) -> str:
    """Keep a plain general-ICU prompt from silently becoming NICU/PICU evidence."""

    text = _norm_text(
        " ".join(str(source.get(key) or "") for key in ("title", "evidence_quote"))
    )
    if re.search(
        r"neonatal|newborn|preterm|pediatric|paediatric|child|infant|"
        r"新生儿|早产儿|儿科|儿童|婴儿",
        text,
        re.I,
    ):
        return ""
    # Chinese prose commonly joins the ASCII acronym directly to Han text
    # (for example ``ICU患者``), so word boundaries would silently miss the
    # same general-ICU population that ``ICU patients`` correctly detects.
    if not re.search(r"ICU|critical care|intensive care|重症", text, re.I):
        return ""
    return (
        " NOT (neonat*[Title/Abstract] OR newborn*[Title/Abstract] OR "
        "preterm[Title/Abstract] OR pediatric[Title/Abstract] OR "
        "paediatric[Title/Abstract] OR child*[Title/Abstract] OR "
        '"Intensive Care, Neonatal"[MeSH Terms] OR '
        '"Intensive Care Units, Pediatric"[MeSH Terms])'
    )


def _discovery_queries(topic: str, journal: str, body: Dict[str, Any]) -> List[str]:
    topic = _clean(topic, 220)
    journal = _clean(journal or body.get("journal_scope") or "", 120)
    year_from = _year(body.get("year_from") or body.get("start_year"))
    year_to = _year(body.get("year_to") or body.get("end_year"))
    year_filter = ""
    if year_from and year_to:
        year_filter = f' AND ("{year_from}"[Date - Publication] : "{year_to}"[Date - Publication])'
    elif year_from:
        year_filter = (
            f' AND ("{year_from}"[Date - Publication] : "3000"[Date - Publication])'
        )
    journal_filter = f' AND "{journal}"[Journal]' if journal else ""
    scope = _discovery_topic_clause(topic, body)
    concept_anchor = direct_evidence_search.build_concept_anchor_query(
        {**body, "topic": topic}
    )
    if not concept_anchor:
        concept_anchor = (
            f"({scope}) AND (definition[Title/Abstract] OR "
            "criteria[Title/Abstract] OR development[Title/Abstract] OR "
            "validation[Title/Abstract] OR consensus[Title/Abstract]) AND "
            '(ICU[Title/Abstract] OR "critical care"[Title/Abstract] OR '
            '"intensive care"[Title/Abstract])'
        )
    population_filter = _discovery_population_filter(topic, body)
    observational_filter = (
        " NOT (Review[Publication Type] OR Meta-Analysis[Publication Type] OR "
        "Guideline[Publication Type] OR Practice Guideline[Publication Type] OR "
        "Randomized Controlled Trial[Publication Type] OR Clinical Trial[Publication Type])"
    )
    comparator_intent_filter = (
        " AND (prevalence[Title/Abstract] OR incidence[Title/Abstract] OR "
        "association[Title/Abstract] OR associated[Title/Abstract] OR "
        "risk[Title/Abstract] OR predict*[Title/Abstract] OR "
        "prognos*[Title/Abstract])"
    )
    # The all-year direct-comparator stratum protects recall for foundational
    # ICU cohort/validation studies; the second, recent stratum independently
    # tests whether the current five-year literature is covered. Conflating
    # those questions caused an older but directly relevant cohort to vanish
    # from the candidate set. An explicit user date range remains authoritative
    # and replaces the default recent window without deleting the all-year
    # comparator stratum.
    recent_start = datetime.now(timezone.utc).year - 5
    direct_year_filter = year_filter or (
        f' AND ("{recent_start}/01/01"[Date - Publication] : '
        '"3000/12/31"[Date - Publication])'
    )
    base = f'({scope}) AND (ICU[Title/Abstract] OR "critical care"[Title/Abstract] OR "intensive care"[Title/Abstract])'
    direct = (
        f"({scope}) AND "
        "(cohort[Title/Abstract] OR observational[Title/Abstract] OR "
        "retrospective[Title/Abstract] OR prospective[Title/Abstract] OR "
        '"cross-sectional"[Title/Abstract]) '
        'AND (ICU[Title/Abstract] OR "critical care"[Title/Abstract] OR '
        '"intensive care"[Title/Abstract])'
        + population_filter
        + comparator_intent_filter
        + observational_filter
    )
    review = f"({scope}) AND (review[Publication Type] OR systematic review[Title/Abstract] OR guideline[Title/Abstract])"
    db = (
        f"({scope}) AND (MIMIC[Title/Abstract] OR eICU[Title/Abstract] OR "
        '"critical care database"[Title/Abstract]) AND '
        "(cohort[Title/Abstract] OR observational[Title/Abstract] OR "
        "retrospective[Title/Abstract] OR prospective[Title/Abstract])"
        + population_filter
        + comparator_intent_filter
        + observational_filter
    )
    return [
        base + journal_filter + year_filter,
        concept_anchor + journal_filter + year_filter,
        direct + year_filter,
        direct + direct_year_filter,
        review + journal_filter + year_filter,
        db + year_filter,
    ]


def _dedupe_articles(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preserve retrieval order while deduplicating exact PubMed identities."""

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        pmid = str(row.get("pmid") or "").strip()
        key = f"pmid:{pmid}" if pmid else "title:" + _clean(row.get("title"), 260)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _discovery_population_filter(topic: str, body: Dict[str, Any]) -> str:
    """Compile only an explicitly declared adult/pediatric population filter.

    Population filters improve precision, but silently assuming adults would
    make a pediatric project disappear from its own search.  The compiler
    therefore reads only bounded user/study text and emits no filter when the
    age population is not explicit.
    """

    return direct_evidence_search.population_filter({**body, "topic": topic})


def _stratified_pubmed_ids(
    query_hits: Iterable[Tuple[str, str, Iterable[str]]],
    *,
    limit: int,
) -> Tuple[List[str], Dict[str, Dict[str, List[str]]], List[Dict[str, Any]]]:
    """Retain PubMed hits across prespecified search strata.

    Concatenating all results before truncation lets a broad first query occupy
    the complete candidate budget, even when review and database-specific
    searches ran successfully.  Round-robin retention makes the bounded search
    faithful to its displayed design without inventing a relevance score or
    hand-picking an E1 paper.  Every retained PMID keeps the exact query or
    queries that retrieved it for the downstream digest-bound receipt.
    """

    capped = max(1, min(int(limit), 20))
    rows: List[Tuple[str, str, List[str]]] = []
    retrieval_by_pmid: Dict[str, Dict[str, List[str]]] = {}
    for stratum, query, raw_ids in query_hits:
        hits = _dedupe_strings(raw_ids)
        clean_stratum = _clean(stratum, 80) or "prespecified"
        clean_query = _clean(query, 1_500)
        rows.append((clean_stratum, clean_query, hits))
        for pmid in hits:
            receipt = retrieval_by_pmid.setdefault(
                pmid,
                {"queries": [], "strata": []},
            )
            if clean_query and clean_query not in receipt["queries"]:
                receipt["queries"].append(clean_query)
            if clean_stratum not in receipt["strata"]:
                receipt["strata"].append(clean_stratum)

    retained: List[str] = []
    seen: set[str] = set()
    rank = 0
    while len(retained) < capped:
        advanced = False
        for _, _, hits in rows:
            if rank >= len(hits):
                continue
            advanced = True
            pmid = hits[rank]
            if pmid in seen:
                continue
            seen.add(pmid)
            retained.append(pmid)
            if len(retained) >= capped:
                break
        if not advanced:
            break
        rank += 1

    summaries = [
        {
            "id": stratum,
            "query": query,
            "returned_count": len(hits),
            "retained_count": sum(1 for pmid in retained if pmid in set(hits)),
        }
        for stratum, query, hits in rows
    ]
    return retained, retrieval_by_pmid, summaries


def _discovery_topic_clause(topic: str, body: Dict[str, Any]) -> str:
    """Compile conversational study text into a bounded PubMed topic clause."""

    # Exact execution concepts and display slots are authoritative when they
    # exist.  Let the literature-query owner project their conventional PubMed
    # names; do not quote an internal materialized column name as though it were
    # a phrase authors are expected to use.
    if any(
        _clean(body.get(key), 180)
        for key in (
            "exposure_concept",
            "outcome_concept",
            "exposure",
            "outcome",
        )
    ):
        typed_clause = direct_evidence_search.build_scope_clause(body)
        if typed_clause:
            return typed_clause

    # A study slot has both a user-facing clinical label and, once configured,
    # an owner-validated execution concept.  Prefer the latter for search
    # compilation.  Quoting an entire protocol label (for example, a long
    # Sepsis-3 definition with its anchor) as one Title/Abstract phrase destroys
    # recall and is not an honest representation of the concept being searched.
    exposure = _literature_concept_phrase(
        body.get("exposure_concept") or body.get("exposure")
    )
    outcome = _literature_concept_phrase(
        body.get("outcome_concept") or body.get("outcome")
    )
    clauses = [
        _pubmed_title_abstract_clause(value) for value in (exposure, outcome) if value
    ]
    if clauses:
        return " AND ".join(clauses[:2])

    hits = _match_concepts(topic)
    outcome_hit = _pick_outcome(topic, hits)
    predictor_hit = _pick_predictor(hits, outcome_hit)
    inferred = [
        _pubmed_title_abstract_clause(
            _literature_concept_phrase(row.get("concept_id") or row.get("label"))
        )
        for row in (predictor_hit, outcome_hit)
        if row and (row.get("concept_id") or row.get("label"))
    ]
    if inferred:
        return " AND ".join(inferred[:2])

    # No dictionary match: retain a small set of literature-bearing English
    # tokens instead of sending the entire conversational instruction to
    # PubMed.  The original topic remains in the returned audit payload.
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", topic)
        if token.lower()
        not in {
            "about",
            "analysis",
            "database",
            "find",
            "help",
            "idea",
            "research",
            "study",
            "using",
            "with",
        }
    ]
    compact = " ".join(_dedupe_strings(token.lower() for token in tokens)[:8])
    return _pubmed_title_abstract_clause(compact or topic[:80])


def _literature_concept_phrase(value: Any) -> str:
    raw = _clean(value or "", 180)
    if not raw:
        return ""
    normalized = literature_concept_id(raw)
    fallback = (
        _concept_label(normalized)
        if normalized in concept_catalog.CONCEPT_DICTIONARY
        else None
    )
    return literature_concept_phrase(raw, fallback=fallback)


def _pubmed_title_abstract_clause(value: str) -> str:
    phrase = _clean(value, 180).replace('"', "")
    if not phrase:
        return ""
    return f'"{phrase}"[Title/Abstract]'


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _pubmed_article_records(
    ids: List[str],
    *,
    focus_terms: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    """Fetch PubMed metadata plus bounded abstracts via EFetch XML."""
    if not ids:
        return []
    params = parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + params
    with request.urlopen(url, timeout=_NETWORK_TIMEOUT_SEC) as resp:
        raw = resp.read(_MAX_PUBMED_FETCH_BYTES + 1)
    if len(raw) > _MAX_PUBMED_FETCH_BYTES:
        raise ValueError("PubMed metadata response exceeded the bounded XML budget")
    root = ET.fromstring(raw)
    rows: List[Dict[str, Any]] = []
    for article_node in root.findall(".//PubmedArticle"):
        medline = article_node.find("./MedlineCitation")
        article = medline.find("./Article") if medline is not None else None
        if medline is None or article is None:
            continue
        pmid = _clean(_node_text(medline.find("./PMID")), 80)
        title = _clean(_node_text(article.find("./ArticleTitle")), 260)
        journal_node = article.find("./Journal")
        journal = _clean(
            _node_text(
                journal_node.find("./Title") if journal_node is not None else None
            )
            or _node_text(
                journal_node.find("./ISOAbbreviation")
                if journal_node is not None
                else None
            ),
            160,
        )
        year = _pubmed_year(article)
        doi = _pubmed_article_id(article_node, "doi")
        pmcid = _pubmed_article_id(article_node, "pmc")
        abstract = _clean(
            " ".join(
                _node_text(node)
                for node in article.findall("./Abstract/AbstractText")
                if _node_text(node)
            ),
            3000,
        )
        evidence = _best_evidence_sentence(abstract or title)
        publication_types = _dedupe_strings(
            _node_text(node)
            for node in article.findall("./PublicationTypeList/PublicationType")
        )
        design_excerpt = _study_design_excerpt(
            abstract or title,
            focus_terms=(
                *tuple(focus_terms),
                "ICU",
                "critical care",
                "mortality",
                "death",
            ),
        )
        rows.append(
            {
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "doi": doi,
                "pmcid": pmcid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                "abstract_excerpt": _clean(abstract, _MAX_PDF_EXCERPT),
                "evidence_sentence": evidence,
                "design_excerpt": design_excerpt,
                "publication_types": publication_types[:12],
                "full_text_stored": False,
            }
        )
    order = {str(pmid): i for i, pmid in enumerate(ids)}
    rows.sort(key=lambda row: order.get(str(row.get("pmid") or ""), 9999))
    return rows


def _literature_article_kind(
    publication_types: Iterable[str],
    *,
    title: str = "",
    abstract: str = "",
) -> str:
    labels = {str(value or "").strip().lower() for value in publication_types}
    text = f"{title} {abstract}".lower()
    if labels & {"meta-analysis", "systematic review"}:
        return "systematic_review"
    if labels & {"editorial", "comment", "letter", "news"}:
        return "editorial_commentary"
    if labels & {"guideline", "practice guideline", "consensus development conference"}:
        return "guideline_consensus"
    if "clinical trial protocol" in labels or "protocol" in text:
        return "protocol"
    if "review" in labels:
        return "narrative_review"
    if labels & {
        "clinical trial",
        "randomized controlled trial",
        "observational study",
        "comparative study",
        "evaluation study",
        "multicenter study",
    }:
        return "original_research"
    if "journal article" in labels and re.search(
        r"\b(cohort|trial|cross-sectional|case-control|retrospective|prospective|"
        r"observational|randomized|randomised|we studied|we evaluated|we enrolled)\b",
        text,
    ):
        return "original_research"
    return "other"


def _pmc_full_text_evidence(pmcid: str) -> Dict[str, Any]:
    """Return bounded section excerpts from one NCBI-hosted PMC article.

    This is an on-demand reading aid, not a full-text library.  It keeps at
    most three short section excerpts and never persists the fetched XML.
    """

    normalized = str(pmcid or "").strip().upper()
    if not re.fullmatch(r"PMC[0-9]{1,12}", normalized):
        return {"status": "unavailable", "reason": "no_pmc_full_text_link"}
    params = parse.urlencode({"db": "pmc", "id": normalized[3:], "retmode": "xml"})
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + params
    try:
        with request.urlopen(url, timeout=_NETWORK_TIMEOUT_SEC) as resp:
            raw = resp.read(_MAX_PMC_FETCH_BYTES + 1)
        if len(raw) > _MAX_PMC_FETCH_BYTES:
            return {"status": "unavailable", "reason": "pmc_response_too_large"}
        root = ET.fromstring(raw)
    except Exception:
        return {"status": "unavailable", "reason": "pmc_fetch_failed"}

    preferred = (
        ("methods", ("method", "materials", "study design", "patients")),
        ("results", ("result", "finding")),
        ("discussion", ("discussion", "conclusion")),
    )
    sections: List[Dict[str, str]] = []
    used_nodes: set[int] = set()
    section_nodes = list(root.findall(".//body//sec"))
    for role, markers in preferred:
        for node in section_nodes:
            title = _clean(_node_text(node.find("./title")), 160)
            if not title or not any(marker in title.lower() for marker in markers):
                continue
            paragraphs = [
                _node_text(paragraph)
                for paragraph in node.findall("./p")
                if _node_text(paragraph)
            ]
            excerpt = _clean(" ".join(paragraphs), 800)
            if excerpt:
                sections.append({"section": role, "label": title, "excerpt": excerpt})
                used_nodes.add(id(node))
                break

    if len(sections) < 3:
        for node in section_nodes:
            if id(node) in used_nodes:
                continue
            title = _clean(_node_text(node.find("./title")), 160) or "Article body"
            paragraphs = [
                _node_text(paragraph)
                for paragraph in node.findall("./p")
                if _node_text(paragraph)
            ]
            excerpt = _clean(" ".join(paragraphs), 800)
            if not excerpt:
                continue
            sections.append({"section": "other", "label": title, "excerpt": excerpt})
            if len(sections) == 3:
                break

    if not sections:
        return {"status": "unavailable", "reason": "pmc_sections_not_extractable"}
    return {
        "status": "reviewed",
        "pmcid": normalized,
        "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{normalized}/",
        "evidence_spans": sections,
        "full_text_stored": False,
    }


def review_literature_source(pmid: str) -> Dict[str, Any]:
    """Read one selected PubMed record and attempt bounded PMC enrichment."""

    normalized = str(pmid or "").strip()
    if not re.fullmatch(r"[0-9]{1,12}", normalized):
        raise IdeaMiningWebError({"error": "literature_source_pmid_invalid"})
    try:
        records = _pubmed_article_records([normalized])
    except Exception as exc:
        raise IdeaMiningWebError(
            {
                "error": "literature_source_unavailable",
                "reason": "PubMed metadata could not be retrieved.",
            }
        ) from exc
    if not records:
        raise IdeaMiningWebError({"error": "literature_source_not_found"})
    article = records[0]
    publication_types = list(article.get("publication_types") or [])[:12]
    full_text = _pmc_full_text_evidence(str(article.get("pmcid") or ""))
    return {
        "ok": True,
        "schema_version": "easyicu.literature_source_review/1",
        "pmid": normalized,
        "title": _clean(article.get("title"), 500),
        "journal": _clean(article.get("journal"), 240),
        "year": article.get("year"),
        "doi": _clean(article.get("doi"), 240) or None,
        "publication_types": publication_types,
        "article_kind": _literature_article_kind(
            publication_types,
            title=str(article.get("title") or ""),
            abstract=str(article.get("abstract_excerpt") or ""),
        ),
        "abstract_excerpt": _clean(article.get("abstract_excerpt"), 1_200),
        "full_text": full_text,
        "authority": {
            "retrieval_candidate_only": True,
            "scientific_adjudication_required": True,
            "full_text_stored": False,
        },
    }


def _node_text(node: Any) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _pubmed_article_id(article_node: Any, kind: str) -> Optional[str]:
    for item in article_node.findall(".//ArticleId"):
        if str(item.attrib.get("IdType") or "").lower() == kind.lower():
            value = _clean(_node_text(item), 180)
            return value or None
    return None


def _pubmed_year(article: Any) -> Optional[int]:
    for path in (
        "./Journal/JournalIssue/PubDate/Year",
        "./ArticleDate/Year",
        "./Journal/JournalIssue/PubDate/MedlineDate",
    ):
        text = _node_text(article.find(path))
        match = re.search(r"\b(19|20)\d{2}\b", text)
        if match:
            return _year(match.group(0))
    return None


def _best_evidence_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?。！？])\s+", _clean(text, 2500))
    keywords = (
        "mortality",
        "death",
        "survival",
        "sepsis",
        "septic",
        "shock",
        "vasopressor",
        "fluid",
        "lactate",
        "aki",
        "ventilation",
        "ICU",
        "critical care",
    )
    for sentence in sentences:
        if any(k.lower() in sentence.lower() for k in keywords):
            return _clean(sentence, _MAX_SOURCE_QUOTE)
    return _clean(sentences[0] if sentences else text, _MAX_SOURCE_QUOTE)


_STUDY_DESIGN_TERMS = (
    "patient",
    "participant",
    "cohort",
    "observational",
    "retrospective",
    "prospective",
    "cross-sectional",
    "multicenter",
    "multi-center",
    "inclusion",
    "exclusion",
    "eligible",
    "admission",
    "follow-up",
    "follow up",
    "adult",
)


def _study_design_excerpt(
    text: str,
    *,
    focus_terms: Iterable[str] = (),
) -> str:
    """Retain a bounded extractive P/E/O/design excerpt for later screening.

    A single keyword sentence is useful for chat, but it can discard the
    population or design sentence needed to decide whether a paper is a true
    comparator.  This helper remains extractive: it selects source sentences
    and never summarizes or invents study details.
    """

    return select_source_backed_excerpt(
        _clean(text, 4_000),
        focus_terms=focus_terms,
        design_terms=_STUDY_DESIGN_TERMS,
        max_sentences=5,
        max_chars=_MAX_DESIGN_EXCERPT,
    )


@dataclass(frozen=True)
class _ResolvedPublicTarget:
    url: str
    scheme: str
    hostname: str
    port: int
    request_target: str
    host_header: str
    addresses: tuple[tuple[int, int, int, tuple[Any, ...]], ...]


def _resolve_public_http_target(url: str) -> _ResolvedPublicTarget:
    """Resolve a URL once and retain only globally routable socket targets."""
    try:
        parsed = parse.urlsplit(str(url or "").strip())
        hostname = parsed.hostname
        scheme = parsed.scheme.lower()
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeURL("URL contains an invalid host or port.") from exc
    if scheme not in {"http", "https"} or not hostname:
        raise UnsafeURL("URL must use http:// or https:// with a host.")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURL("URLs containing embedded credentials are not allowed.")
    normalized_host = hostname.rstrip(".").lower()
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise UnsafeURL("Localhost URLs are not allowed.")
    try:
        resolved = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise UnsafeURL("URL host could not be resolved.") from exc
    if not resolved:
        raise UnsafeURL("URL host did not resolve to an address.")
    addresses: list[tuple[int, int, int, tuple[Any, ...]]] = []
    seen = set()
    for family, socktype, proto, _canonname, sockaddr in resolved:
        address = str(sockaddr[0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise UnsafeURL("URL host resolved to an invalid address.") from exc
        if not ip.is_global:
            raise UnsafeURL("URL target resolves to a non-public network address.")
        key = (family, socktype, proto, sockaddr)
        if key not in seen:
            seen.add(key)
            addresses.append((family, socktype, proto, sockaddr))
    request_target = parsed.path or "/"
    if parsed.query:
        request_target += "?" + parsed.query
    display_host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 443 if scheme == "https" else 80
    host_header = display_host if port == default_port else f"{display_host}:{port}"
    return _ResolvedPublicTarget(
        url=parsed.geturl(),
        scheme=scheme,
        hostname=hostname,
        port=port,
        request_target=request_target,
        host_header=host_header,
        addresses=tuple(addresses),
    )


def _validate_public_http_url(url: str) -> str:
    return _resolve_public_http_target(url).url


def _connect_resolved_addresses(
    addresses: tuple[tuple[int, int, int, tuple[Any, ...]], ...],
    *,
    timeout: int,
) -> socket.socket:
    """Connect only to vetted sockaddr tuples; never perform a second DNS lookup."""
    last_error: OSError | None = None
    for family, socktype, proto, sockaddr in addresses:
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    raise last_error or OSError("No vetted public address was available.")


class _PinnedHTTPResponse:
    def __init__(
        self,
        connection: http.client.HTTPConnection,
        response: http.client.HTTPResponse,
        url: str,
    ) -> None:
        self._connection = connection
        self._response = response
        self._url = url
        self.status = response.status

    def read(self, amount: int = -1) -> bytes:
        return self._response.read(amount)

    def getheader(self, name: str, default: Any = None) -> Any:
        return self._response.getheader(name, default)

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()

    def __enter__(self) -> "_PinnedHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()


def _request_pinned_target(
    target: _ResolvedPublicTarget,
    req: request.Request,
    *,
    timeout: int,
) -> _PinnedHTTPResponse:
    raw_socket = _connect_resolved_addresses(target.addresses, timeout=timeout)
    if target.scheme == "https":
        context = ssl.create_default_context()
        try:
            connected_socket = context.wrap_socket(
                raw_socket,
                server_hostname=target.hostname,
            )
        except Exception:
            raw_socket.close()
            raise
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            target.hostname,
            target.port,
            timeout=timeout,
            context=context,
        )
    else:
        connected_socket = raw_socket
        connection = http.client.HTTPConnection(
            target.hostname,
            target.port,
            timeout=timeout,
        )
    connection.sock = connected_socket
    headers = dict(req.header_items())
    headers["Host"] = target.host_header
    try:
        connection.request(
            req.get_method(),
            target.request_target,
            body=req.data,
            headers=headers,
        )
        response = connection.getresponse()
    except Exception:
        connection.close()
        raise
    return _PinnedHTTPResponse(connection, response, target.url)


def _open_public_url(
    req: request.Request,
    *,
    timeout: int,
    target: _ResolvedPublicTarget | None = None,
) -> _PinnedHTTPResponse:
    """Open a pinned public URL, resolving and re-pinning every redirect hop."""
    current = target or _resolve_public_http_target(req.full_url)
    current_request = req
    for _redirect in range(6):
        response = _request_pinned_target(current, current_request, timeout=timeout)
        if response.status in {301, 302, 303, 307, 308}:
            location = response.getheader("Location")
            response.close()
            if not location:
                raise UnsafeURL("Redirect response did not provide a target URL.")
            next_url = parse.urljoin(current.url, str(location))
            current = _resolve_public_http_target(next_url)
            current_request = request.Request(
                current.url,
                headers=dict(req.header_items()),
                method="GET",
            )
            continue
        if response.status >= 400:
            status = response.status
            response.close()
            raise OSError(f"HTTP Error {status}")
        return response
    raise UnsafeURL("Too many URL redirects.")


def _fetch_url_metadata(url: str) -> Dict[str, Any]:
    try:
        target = _resolve_public_http_target(url)
    except UnsafeURL as exc:
        return {
            "status": "unsafe_url",
            "network_calls": 0,
            "reason": str(exc),
        }
    doi_hint = _doi_from_text(parse.unquote(target.url))
    calls = 0
    fetch_error = ""
    title = ""
    description = ""
    doi = doi_hint or ""
    raw = b""
    try:
        req = request.Request(
            target.url, headers={"User-Agent": "EasyICU-local-metadata-resolver/1.0"}
        )
        with _open_public_url(
            req,
            timeout=_NETWORK_TIMEOUT_SEC,
            target=target,
        ) as resp:
            calls += 1
            raw = resp.read(_MAX_FETCH_BYTES)
        text = raw.decode("utf-8", errors="replace")
    except UnsafeURL as exc:
        return {
            "status": "unsafe_url",
            "network_calls": 1,
            "reason": str(exc),
            "doi": doi_hint or None,
        }
    except Exception as exc:
        calls += 1
        fetch_error = str(exc)[:240]
        text = ""
    if text:
        title = _html_title(text) or ""
        description = (
            _html_meta(text, "description") or _html_meta(text, "og:description") or ""
        )
        doi = _doi_from_text(text) or doi
    if not (title or description) and doi:
        doi_meta = _fetch_doi_metadata(doi)
        calls += int(doi_meta.get("network_calls") or 0)
        if doi_meta.get("title") or doi_meta.get("journal") or doi_meta.get("year"):
            reason = doi_meta.get("reason") or "Resolved bounded DOI metadata."
            if fetch_error:
                reason = f"URL HTML fetch failed ({fetch_error}); {reason}"
            return {
                "status": "metadata_fetched",
                "metadata_source": doi_meta.get("metadata_source") or "doi",
                "network_calls": calls,
                "title": doi_meta.get("title"),
                "description": _clean(
                    doi_meta.get("description") or description, _MAX_SOURCE_QUOTE
                ),
                "doi": doi_meta.get("doi") or doi,
                "journal": doi_meta.get("journal"),
                "year": doi_meta.get("year"),
                "bytes_read": min(len(raw), _MAX_FETCH_BYTES),
                "reason": reason,
            }
        if not fetch_error and doi_meta.get("reason"):
            fetch_error = str(doi_meta.get("reason") or "")[:240]
    return {
        "status": (
            "metadata_fetched"
            if title or description
            else ("fetch_failed" if fetch_error else "metadata_fetch_empty")
        ),
        "metadata_source": "html" if title or description else None,
        "network_calls": calls,
        "title": title,
        "description": _clean(description, _MAX_SOURCE_QUOTE),
        "doi": doi,
        "bytes_read": min(len(raw), _MAX_FETCH_BYTES),
        "reason": (
            fetch_error
            if fetch_error and not (title or description)
            else "Stored bounded metadata only; full HTML was not persisted."
        ),
    }


def _fetch_doi_metadata(doi: str) -> Dict[str, Any]:
    doi = _clean(doi, 180)
    if not doi:
        return {"status": "invalid_doi", "network_calls": 0}
    url = "https://api.crossref.org/works/" + parse.quote(doi, safe="")
    try:
        req = request.Request(
            url, headers={"User-Agent": "EasyICU-local-metadata-resolver/1.0"}
        )
        with request.urlopen(req, timeout=_NETWORK_TIMEOUT_SEC) as resp:
            data = json.loads(
                resp.read(_MAX_FETCH_BYTES).decode("utf-8", errors="replace")
            )
    except Exception as exc:
        return {
            "status": "doi_fetch_failed",
            "network_calls": 1,
            "reason": str(exc)[:240],
        }
    message = data.get("message") or {}
    title = _first_text(message.get("title"))
    journal = _first_text(message.get("container-title"))
    abstract = _clean(_strip_tags(message.get("abstract") or ""), _MAX_SOURCE_QUOTE)
    year = _crossref_year(message)
    return {
        "status": (
            "metadata_fetched" if title or journal or year else "metadata_fetch_empty"
        ),
        "metadata_source": "crossref",
        "network_calls": 1,
        "title": title,
        "journal": journal,
        "year": year,
        "description": abstract,
        "doi": _clean(message.get("DOI") or doi, 180),
        "reason": "Resolved bounded DOI metadata through Crossref; full text was not fetched or stored.",
    }


def _first_text(value: Any) -> Optional[str]:
    if isinstance(value, list):
        value = next((item for item in value if item), "")
    return _clean(value or "", 260) or None


def _crossref_year(message: Dict[str, Any]) -> Optional[int]:
    for key in ("published-print", "published-online", "published", "issued"):
        parts = ((message.get(key) or {}).get("date-parts") or [[]])[0]
        if parts:
            return _year(parts[0])
    return None


def _strip_tags(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))


def _html_title(text: str) -> Optional[str]:
    match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.I | re.S)
    if not match:
        return None
    return _clean(html.unescape(re.sub(r"\s+", " ", match.group(1))), 220)


def _html_meta(text: str, name: str) -> Optional[str]:
    patterns = [
        rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']',
        rf'<meta[^>]+property=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']',
        rf'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']{re.escape(name)}["\']',
        rf'<meta[^>]+content=["\'](.*?)["\'][^>]+property=["\']{re.escape(name)}["\']',
    ]
    for pat in patterns:
        match = re.search(pat, text, flags=re.I | re.S)
        if match:
            return html.unescape(match.group(1))
    return None


def _doi_from_text(text: str) -> Optional[str]:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return _clean(match.group(0), 180) if match else None


def _conversational_prior_art_screen(
    row: Dict[str, Any], source: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Rank retrieval fit without upgrading a paper to scientific evidence."""

    if not source or not _conversational_literature_scope(source):
        return {
            "disposition": "include",
            "fit": "unclassified",
            "score": 0,
            "matched_dimensions": [],
            "rationale": "Prespecified PubMed retrieval candidate; scientific screening is pending.",
        }

    source_text = _norm_text(
        " ".join(str(source.get(key) or "") for key in ("title", "evidence_quote"))
    )
    title = _norm_text(str(row.get("title") or ""))
    text = _norm_text(
        " ".join(
            str(row.get(key) or "")
            for key in (
                "title",
                "abstract_excerpt",
                "evidence_sentence",
                "design_excerpt",
            )
        )
    )
    population_mismatch = bool(
        _conversational_population_filter(source)
        and re.search(
            r"neonatal|newborn|preterm|pediatric|paediatric|\bchild(?:ren)?\b|\binfant",
            text,
            re.I,
        )
    )
    care_setting_mismatch = bool(
        re.search(
            r"neuroanesthes|neurosurg|intraoperativ|permissive hypotension|"
            r"induced hypotension|prehospital|helicopter emergency medical|"
            r"nonintensive care unit admission|veterinary|\bdogs?\b|\bcats?\b",
            text,
            re.I,
        )
    )
    case_report_mismatch = bool(
        re.search(r"\bcase report\b", title, re.I)
        or any(
            "case report" in str(value or "").lower()
            for value in row.get("publication_types") or []
        )
    )
    pair_profile = _conversational_pair_profile(source_text)
    if pair_profile:
        profile_name, dimensions = pair_profile
        dimension_matches = [
            bool(re.search(match_pattern, text, re.I))
            for _, _, match_pattern in dimensions
        ]
        title_matches = [
            bool(re.search(match_pattern, title, re.I))
            for _, _, match_pattern in dimensions
        ]
        icu = bool(re.search(r"\bICU\b|critical care|intensive care", text, re.I))
        strata = list(row.get("matched_query_strata") or [])
        stratum_score = max(
            (
                {
                    "candidate_topic": 3,
                    "direct_observational_candidates": 2,
                    "clinical_landscape": 1,
                    "critical_care_database": 1,
                }.get(str(value), 0)
                for value in strata
            ),
            default=0,
        )
        score = (
            4 * sum(dimension_matches)
            + 3 * sum(title_matches)
            + int(icu)
            + stratum_score
        )
        matched_dimensions = [
            f"{profile_name}_{index + 1}"
            for index, matched in enumerate(dimension_matches)
            if matched
        ]
        if population_mismatch or care_setting_mismatch or case_report_mismatch:
            return {
                "disposition": "exclude",
                "fit": (
                    "population_mismatch"
                    if population_mismatch
                    else (
                        "care_setting_mismatch"
                        if care_setting_mismatch
                        else "design_mismatch"
                    )
                ),
                "score": -100,
                "matched_dimensions": matched_dimensions,
                "rationale": "Excluded because the population, care setting, or publication design does not match the selected ICU candidate.",
            }
        if not (all(dimension_matches) and icu):
            return {
                "disposition": "exclude",
                "fit": "topic_mismatch",
                "score": score,
                "matched_dimensions": matched_dimensions,
                "rationale": "Excluded because the metadata did not jointly match both selected scientific concepts in an ICU setting.",
            }
        direct = all(title_matches)
        return {
            "disposition": "include",
            "fit": "direct_retrieval_fit" if direct else "adjacent_retrieval_fit",
            "score": score,
            "matched_dimensions": matched_dimensions,
            "rationale": (
                "Direct retrieval fit for both selected scientific concepts in an ICU setting; full scientific screening is still required."
                if direct
                else "Adjacent retrieval fit for both selected scientific concepts in an ICU setting; it does not establish novelty or causal support."
            ),
        }
    core = bool(re.search(r"hypotension|\bshock\b|hemodynamic", text, re.I))
    title_core = bool(re.search(r"hypotension|\bshock\b|hemodynamic", title, re.I))
    icu = bool(re.search(r"\bICU\b|critical care|intensive care", text, re.I))
    practice = bool(
        re.search(
            r"practice (?:variation|pattern)|variation in [^.]{0,80}"
            r"(?:treatment|management|use|administration)|"
            r"(?:physician|clinician|hospital|ICU)[^.]{0,50}"
            r"(?:variation|variability|practice pattern)|"
            r"organizational structure|staffing level",
            text,
            re.I,
        )
    )
    temporal = bool(
        re.search(
            r"nighttime|nocturnal|overnight|after-hours|out-of-hours|"
            r"night shift|weekend|weekday nighttime",
            text,
            re.I,
        )
    )
    strategy = bool(
        re.search(
            r"vasopressor|vasoactive|norepinephrine|noradrenaline|vasopressin|"
            r"fluid resuscitation|fluid administration|fluid bolus|"
            r"resuscitation strateg",
            text,
            re.I,
        )
    )
    explicit_treatment_modality = bool(
        strategy
        or re.search(
            r"shock team|mechanical circulatory support|vasoactive|vasopressor|"
            r"fluid resuscitation|fluid administration",
            title,
            re.I,
        )
    )
    asks_variation = bool(
        re.search(
            r"variation|variability|difference|heterogeneity|差异|不一致|不同",
            source_text,
            re.I,
        )
    )
    dimensions = [
        label
        for label, matched in (
            ("hypotension_or_shock", core),
            ("general_icu", icu),
            ("practice_difference", practice),
            ("night_or_staffing", temporal),
            ("fluid_or_vasopressor_strategy", strategy),
        )
        if matched
    ]
    strata = list(row.get("matched_query_strata") or [])
    stratum_score = max(
        (
            {
                "candidate_topic": 3,
                "direct_observational_candidates": 2,
                "clinical_landscape": 1,
                "critical_care_database": 1,
            }.get(str(value), 0)
            for value in strata
        ),
        default=0,
    )
    score = (
        (6 if title_core else 0)
        + (2 if core else 0)
        + (1 if icu else 0)
        + (6 if practice else 0)
        + (4 if temporal else 0)
        + (3 if strategy else 0)
        + stratum_score
    )
    if population_mismatch or care_setting_mismatch or case_report_mismatch:
        return {
            "disposition": "exclude",
            "fit": (
                "population_mismatch"
                if population_mismatch
                else (
                    "care_setting_mismatch"
                    if care_setting_mismatch
                    else "design_mismatch"
                )
            ),
            "score": -100,
            "matched_dimensions": dimensions,
            "rationale": (
                "Excluded from the main retrieval set because it is neonatal or pediatric while the prompt described a general ICU population."
                if population_mismatch
                else (
                    "Excluded from the main retrieval set because it concerns perioperative, prehospital, veterinary, or non-ICU care rather than treatment after hypotension in the ICU."
                    if care_setting_mismatch
                    else "Excluded from the main retrieval set because a single case report cannot establish the observed ICU practice variation."
                )
            ),
        }
    if (
        not core
        or not icu
        or (
            asks_variation
            and not (practice and (temporal or explicit_treatment_modality))
        )
    ):
        return {
            "disposition": "exclude",
            "fit": "topic_mismatch",
            "score": score,
            "matched_dimensions": dimensions,
            "rationale": "Excluded from the main retrieval set because the metadata did not jointly match the ICU hypotension/shock phenomenon and its observed practice difference.",
        }
    direct = bool(
        practice
        and (
            temporal
            or re.search(
                r"organizational structure|staffing level|nighttime|weekend",
                title,
                re.I,
            )
        )
    )
    return {
        "disposition": "include",
        "fit": "direct_retrieval_fit" if direct else "adjacent_retrieval_fit",
        "score": score,
        "matched_dimensions": dimensions,
        "rationale": (
            "Direct retrieval fit for the observed ICU hypotension treatment difference; full scientific screening is still required."
            if direct
            else "Adjacent retrieval fit for ICU hypotension/shock management; it informs a candidate direction but does not establish novelty or causal support."
        ),
    }


def _pubmed_prior_art(
    queries: List[str], *, source: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    results_by_pmid: Dict[str, Dict[str, Any]] = {}
    retrieval_by_pmid: Dict[str, Dict[str, List[str]]] = {}
    query_strata: List[Dict[str, Any]] = []
    calls = 0
    errors: List[str] = []
    scientific_index = 0
    for index, query in enumerate(queries[:_MAX_DISCOVERY_QUERIES]):
        if "[DOI]" in query:
            stratum = "source_doi"
        else:
            stratum = (
                _PRIOR_ART_QUERY_STRATA[scientific_index]
                if scientific_index < len(_PRIOR_ART_QUERY_STRATA)
                else f"prespecified_{index + 1}"
            )
            scientific_index += 1
        ids: List[str] = []
        rows: List[Dict[str, Any]] = []
        try:
            ids = _pubmed_esearch(query, limit=10)
            calls += 1
            if ids:
                rows = _pubmed_article_records(ids)
                calls += 1
                for row in rows:
                    pmid = str(row.get("pmid") or "").strip()
                    if not pmid:
                        continue
                    results_by_pmid.setdefault(pmid, row)
                    receipt = retrieval_by_pmid.setdefault(
                        pmid, {"queries": [], "strata": []}
                    )
                    if query not in receipt["queries"]:
                        receipt["queries"].append(query)
                    if stratum not in receipt["strata"]:
                        receipt["strata"].append(stratum)
        except Exception as exc:
            errors.append(str(exc)[:240])
        query_strata.append(
            {
                "id": stratum,
                "query": query,
                "returned_count": len(ids),
                "retained_count": sum(
                    1 for row in rows if str(row.get("pmid") or "").strip()
                ),
            }
        )
    deduped: List[Dict[str, Any]] = []
    for pmid, row in results_by_pmid.items():
        retrieval = retrieval_by_pmid.get(pmid) or {"queries": [], "strata": []}
        deduped.append(
            {
                **row,
                "query": (retrieval["queries"] or [""])[0],
                "matched_queries": list(retrieval["queries"]),
                "matched_query_strata": list(retrieval["strata"]),
            }
        )
    screened = [
        {
            **row,
            "retrieval_screen": _conversational_prior_art_screen(row, source),
            "_retrieval_order": index,
        }
        for index, row in enumerate(deduped)
    ]
    included = sorted(
        (
            row
            for row in screened
            if (row.get("retrieval_screen") or {}).get("disposition") != "exclude"
        ),
        key=lambda row: (
            -int((row.get("retrieval_screen") or {}).get("score") or 0),
            int(row.get("_retrieval_order") or 0),
        ),
    )
    excluded = [
        row
        for row in screened
        if (row.get("retrieval_screen") or {}).get("disposition") == "exclude"
    ]
    for row in [*included, *excluded]:
        row.pop("_retrieval_order", None)
    public_hits = [
        row
        for row in included
        if re.search(
            r"\b(MIMIC|eICU|HiRID|AUMC|SICdb|public database)\b",
            json.dumps(row, ensure_ascii=False),
            re.I,
        )
    ]
    status = (
        "searched" if deduped else ("search_failed" if errors else "searched_no_hits")
    )
    if included:
        interpretation = (
            f"Found {len(included)} retrieval-fit metadata hit(s) after excluding {len(excluded)} population/topic mismatch(es). Treat them as a map of what has been tried: "
            "which comparator, cohort, database, endpoint, and time window they used. This does not automatically kill the idea; it tells the planner how to refine the new ICU exploration."
        )
    elif deduped:
        interpretation = (
            f"PubMed returned {len(deduped)} metadata hit(s), but all failed the bounded population/topic retrieval screen. "
            "This is not evidence of novelty; refine or broaden the candidate and search again."
        )
    elif errors:
        interpretation = "The metadata search failed or was incomplete. Do not claim novelty; keep the idea as a planning draft until prior art can be reviewed."
    else:
        interpretation = "No metadata hit was found with the bounded queries. This is only weak evidence of novelty; broaden queries or review manually before presenting the idea."
    return {
        "status": status,
        "search_performed": True,
        "searched_at": _now(),
        "network_calls": calls,
        "queries_to_run": queries,
        "query_strata": query_strata,
        "result_count": len(included),
        "retrieved_result_count": len(deduped),
        "excluded_result_count": len(excluded),
        "results": included[:12],
        "excluded_results": excluded[:12],
        "public_database_used_by_prior_work": (
            "possible" if public_hits else "not_detected_in_metadata"
        ),
        "direct_same_topic_hits": [
            row
            for row in included
            if (row.get("retrieval_screen") or {}).get("fit") == "direct_retrieval_fit"
        ][:5],
        "errors": errors,
        "reason": "PubMed metadata search only; full text and external LLM review were not used.",
        "opportunity_frame": interpretation,
        "next_use": "Use these hits to decide whether the source article suggests a new subgroup, timing window, exposure definition, or outcome that EasyICU can assess.",
    }


def _pubmed_esearch(query: str, limit: int = 5) -> List[str]:
    params = parse.urlencode(
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max(1, min(limit, 20)),
            "sort": "relevance",
        }
    )
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + params
    with request.urlopen(url, timeout=_NETWORK_TIMEOUT_SEC) as resp:
        data = json.loads(resp.read(_MAX_FETCH_BYTES).decode("utf-8", errors="replace"))
    return [str(x) for x in ((data.get("esearchresult") or {}).get("idlist") or [])]


def _pubmed_esummary(ids: List[str]) -> List[Dict[str, Any]]:
    params = parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + params
    with request.urlopen(url, timeout=_NETWORK_TIMEOUT_SEC) as resp:
        data = json.loads(resp.read(_MAX_FETCH_BYTES).decode("utf-8", errors="replace"))
    result = data.get("result") or {}
    rows: List[Dict[str, Any]] = []
    for pmid in ids:
        item = result.get(str(pmid)) or {}
        if not item:
            continue
        journal = item.get("fulljournalname") or item.get("source")
        year = _year(str(item.get("pubdate") or "")[:4])
        rows.append(
            {
                "pmid": str(pmid),
                "title": _clean(item.get("title") or "", 260),
                "journal": _clean(journal or "", 160),
                "year": year,
                "pubdate": _clean(item.get("pubdate") or "", 80),
                "articleids": [
                    {
                        "type": _clean(x.get("idtype") or "", 20),
                        "value": _clean(x.get("value") or "", 120),
                    }
                    for x in item.get("articleids") or []
                    if isinstance(x, dict) and x.get("value")
                ][:4],
            }
        )
    return rows


def _run_dir(run_id: str) -> Path:
    return _RUN_ROOT / _slug(run_id, fallback="idea_run")


def _assert_no_row_payload(payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    markers = [marker for marker in _DIRECT_ID_MARKERS if marker in text]
    if markers:
        raise IdeaMiningWebError(
            {"error": "row_level_payload_marker", "markers": sorted(markers)}
        )


def _outcome_evidence_text(text: str) -> str:
    """Remove bounded negative instructions before outcome detection.

    Idea-mining prompts often say "do not default to mortality". The word
    mortality is then a rejected option, not source evidence for an outcome.
    """

    value = str(text or "")
    competing_event_patterns = (
        r"(?:死亡|病死率)[^。；;\n]{0,32}(?:如何)?(?:处理|竞争事件|纳入|排除)",
        r"\b(?:death|mortality)\b[^.;\n]{0,64}\b(?:handling|handled|competing event|included|excluded)\b",
        r"\b(?:handling|treat|treated|consider)\b[^.;\n]{0,48}\b(?:death|mortality)\b",
    )
    for pattern in competing_event_patterns:
        value = re.sub(pattern, " ", value, flags=re.I)
    mortality_rejection_patterns = (
        r"(?:不|不要|不得|不能|并非|不是)[^。；;\n]{0,32}(?:院内)?(?:死亡|病死率)",
        r"\b(?:do not|don't|must not|should not|not)\b[^.;\n]{0,48}\b(?:death|mortality|survival)\b",
    )
    mortality_rejected = any(
        re.search(pattern, value, flags=re.I)
        for pattern in mortality_rejection_patterns
    )
    for pattern in mortality_rejection_patterns:
        value = re.sub(pattern, " ", value, flags=re.I)
    if mortality_rejected:
        # A model may synthesize a positive-looking title such as
        # ``fluid balance and mortality`` even when another source field
        # contains the user's explicit "do not default to mortality"
        # instruction.  The rejection must dominate across the bounded source
        # bundle; otherwise a generated title silently reverses user intent.
        value = re.sub(
            r"\b(?:in[- ]hospital\s+)?(?:death|mortality|survival)\b|(?:院内)?(?:死亡|病死率)",
            " ",
            value,
            flags=re.I,
        )
    return value


def _pick_outcome(text: str, hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    low = _outcome_evidence_text(text).lower()
    if any(tok in low for tok in ["death", "mortality", "survival", "死亡", "病死"]):
        return _concept_hit("death")
    for row in hits:
        if row.get("concept_id") == "death":
            continue
        if row.get("concept_id") in {"los_icu", "aki"}:
            return row
    # No outcome evidence in the source text: leave the outcome unassigned
    # instead of fabricating "death" for every idea.
    return None


def _pick_predictor(
    hits: List[Dict[str, Any]], outcome: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    outcome_id = outcome.get("concept_id") if outcome else None
    for row in hits:
        if row.get("concept_id") != outcome_id and row.get("concept_id") not in {
            "death",
            "los_icu",
            *_VENTILATION_EPISODE_CONCEPTS,
        }:
            return row
    return None


def _select_idea_concepts(
    text: str,
    hits: List[Dict[str, Any]],
    outcome: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep intervention-trial concepts together instead of picking one marker.

    Literature-derived ICU ideas often describe a strategy (for example,
    vasopressor-first plus fluid-sparing) rather than a single laboratory
    predictor.  The web adapter is still local/deterministic, but it should
    preserve that concept set so the feasibility assessment can honestly say which
    required modules are missing from the active export.
    """
    by_id = {str(row.get("concept_id")): row for row in hits}
    selected: List[Dict[str, Any]] = []
    low = text.lower()

    def add(concept_id: str, role: str) -> None:
        row = by_id.get(concept_id)
        if not row:
            return
        selected.append({**row, "role": role})

    for concept_id in _requested_exposure_concepts(text, hits)[:1]:
        add(concept_id, "exposure")

    if any(
        tok in low
        for tok in ("vasopressor", "norepinephrine", "noradrenaline", "pressor")
    ):
        for concept_id in _INTERVENTION_PRIORITY:
            if concept_id in by_id:
                add(concept_id, "exposure")
                break
    if any(tok in low for tok in ("fluid balance", "液体平衡")):
        for concept_id in (
            "fluid_balance_cumulative",
            "fluid_balance",
            "total_input_ml",
        ):
            if concept_id in by_id:
                add(concept_id, "exposure")
                break
    elif any(
        tok in low
        for tok in (
            "fluid",
            "fluids",
            "intravenous volume",
            "fluid-sparing",
            "resuscitation",
            "液体",
            "补液",
        )
    ):
        for concept_id in (
            "total_input_ml",
            "fluid_balance",
            "fluid_balance_cumulative",
        ):
            if concept_id in by_id:
                add(concept_id, "exposure")
                break
    for concept_id in _SEVERITY_PRIORITY:
        if concept_id in by_id and concept_id not in _VENTILATION_EPISODE_CONCEPTS:
            add(concept_id, "covariate_or_subgroup")
    if outcome:
        selected.append({**outcome, "role": "outcome"})
    for row in hits:
        if len(selected) >= 10:
            break
        concept_id = str(row.get("concept_id"))
        if concept_id in {"death", "los_icu"}:
            continue
        role = (
            "eligibility_or_episode"
            if concept_id in _VENTILATION_EPISODE_CONCEPTS
            else "feature"
        )
        selected.append({**row, "role": role})
    return _dedupe_concepts(selected)[:10]


def _dedupe_concepts(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        concept_id = str(row.get("concept_id") or "")
        if not concept_id or concept_id in seen:
            continue
        seen.add(concept_id)
        out.append(row)
    return out


def _primary_predictor(concepts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for role in ("exposure", "predictor"):
        for row in concepts:
            if row.get("role") == role and row.get("concept_id") not in {
                "death",
                "los_icu",
            }:
                return row
    return None


def _concept_hit(concept_id: str) -> Optional[Dict[str, Any]]:
    entry = concept_catalog.CONCEPT_DICTIONARY.get(concept_id)
    if not entry:
        return None
    return {
        "concept_id": concept_id,
        "label": str(entry[0]),
        "unit": str(entry[2] if len(entry) > 2 else ""),
        "matched_alias": concept_id,
        "module": _concept_module(concept_id),
    }


def _concept_module(concept_id: str) -> Optional[str]:
    for module, concepts in concept_catalog.CONCEPT_GROUPS_INTERNAL.items():
        if concept_id in concepts:
            return module
    return None


def _concept_label(concept_id: str) -> str:
    entry = concept_catalog.CONCEPT_DICTIONARY.get(concept_id)
    return str(entry[0]) if entry else concept_id


def _idea_title(
    source: Dict[str, Any],
    predictor: Optional[Dict[str, Any]],
    outcome: Optional[Dict[str, Any]],
    concepts: Optional[List[Dict[str, Any]]] = None,
) -> str:
    concept_ids = {row.get("concept_id") for row in concepts or []}
    if concept_ids & {
        "vaso_ind",
        "norepi_equiv",
        "norepi_rate",
        "norepi_dur",
    } and concept_ids & {
        "total_input_ml",
        "fluid_balance",
        "fluid_balance_cumulative",
    }:
        return "Vasopressor-fluid resuscitation strategy and outcomes in septic ICU patients"
    if predictor and outcome:
        return f"{predictor['label']} and {outcome['label']} in adult ICU patients"
    if predictor:
        return f"{predictor['label']} signal in adult ICU patients"
    return _clean(source.get("title") or "Literature-derived ICU idea", 120)


def _concept_set_label(concepts: List[Dict[str, Any]]) -> str:
    exposures = [
        str(row.get("label"))
        for row in concepts
        if row.get("role") == "exposure" and row.get("label")
    ]
    if exposures:
        return " + ".join(exposures[:3])
    predictors = [
        str(row.get("label"))
        for row in concepts
        if row.get("role") == "predictor" and row.get("label")
    ]
    return " + ".join(predictors[:3])


def _rationale(
    text: str,
    predictor: Optional[Dict[str, Any]],
    outcome: Optional[Dict[str, Any]],
    concepts: Optional[List[Dict[str, Any]]] = None,
) -> str:
    parts = []
    if predictor:
        parts.append(
            f"Source text maps the candidate predictor to `{predictor['concept_id']}`."
        )
    exposure_ids = [
        str(row.get("concept_id"))
        for row in concepts or []
        if row.get("role") == "exposure"
    ]
    if len(exposure_ids) > 1:
        parts.append(
            "The source describes a strategy, so the ledger preserves multiple exposure concepts: "
            + ", ".join(exposure_ids)
            + "."
        )
    if outcome:
        parts.append(f"Outcome can be represented by `{outcome['concept_id']}`.")
    parts.append("This is a feasibility triage result, not a manuscript finding.")
    return " ".join(parts)


def _analysis_family(text: str) -> str:
    low = text.lower()
    if any(tok in low for tok in ["predict", "prediction", "model", "risk score"]):
        return "prediction"
    if any(tok in low for tok in ["trajectory", "trend", "clearance", "slope"]):
        return "trajectory"
    return "association"


def _go_reason(go: str, feasibility: Dict[str, Any], prior_art: Dict[str, Any]) -> str:
    if go == "recommend":
        return "Mapped concepts are present in the active export; prior-art search still needs explicit review before reporting novelty."
    if go == "hold":
        if feasibility.get("tier") == "design_incomplete":
            return "Confirm the primary exposure or predictor and explicit outcome, then repeat the feasibility assessment."
        if feasibility.get("tier") == "demo_only":
            return (
                feasibility.get("reason")
                or "The current active export is demo-only and cannot support reportable feasibility."
            )
        return (
            feasibility.get("reason")
            or "Some concepts need re-extraction before analysis."
        )
    return (
        feasibility.get("reason")
        or "The idea is not executable on the current database."
    )


def _next_action(go: str, feasibility: Dict[str, Any]) -> str:
    if go == "recommend":
        return "Review feasibility statistics, interpret prior art, edit the plan, then hand off to Agent Projects."
    if go == "hold":
        if feasibility.get("tier") == "design_incomplete":
            return "Confirm the primary exposure or predictor and explicit outcome, then repeat the feasibility assessment."
        if feasibility.get("tier") == "demo_only":
            return "Select a real EasyICU export, rerun feasibility assessment, then generate the plan."
        # A held idea has the concept in the dictionary but needs more data:
        # re-extraction can help. (Previously this return sat OUTSIDE the hold
        # branch, so the db-cannot-do / T3 case below was dead code and T3 ideas
        # were told to re-extract a concept that does not exist.)
        return "Re-run extraction with missing modules/features, then repeat the feasibility assessment."
    # go == "db-cannot-do" (T3): concept is absent from the EasyICU dictionary,
    # so re-extraction can never produce it.
    return "Choose another database or revise the idea."


def _pre_experiment_interpretation(
    stats: List[Dict[str, Any]], *, demo_like: bool = False
) -> List[str]:
    if demo_like:
        return [
            "The active export is MOCK/demo. These checks are useful for UI rehearsal only and must not be used as reportable feasibility evidence.",
            "Select or register a real EasyICU export before creating an Agent project for this idea.",
        ]
    if not stats:
        return [
            "No mapped feature is present in the active export; run extraction with the needed modules first."
        ]
    low = [
        row
        for row in stats
        if row.get("metric_kind") != "event_rate"
        and row.get("coverage_pct") is not None
        and float(row["coverage_pct"]) < 50
        and row.get("status") != "metadata_only"
    ]
    event_rows = [row for row in stats if row.get("metric_kind") == "event_rate"]
    metadata_rows = [row for row in stats if row.get("status") == "metadata_only"]
    notes = [f"{len(stats)} mapped feature(s) were summarized from the active export."]
    if metadata_rows:
        notes.append(
            f"{len(metadata_rows)} feature(s) were verified by manifest/schema only because their module exceeded the preflight scan limit; run a bounded sample or Agent pipeline stage before interpreting coverage."
        )
    if event_rows:
        notes.append(
            f"{len(event_rows)} boolean/event indicator(s) are reported as positive rates; negative patients are not treated as missing."
        )
    if low:
        notes.append(
            f"{len(low)} measured feature(s) have <50% entity coverage and should be treated as feasibility risks."
        )
    elif not event_rows:
        notes.append(
            "Mapped features have at least 50% entity coverage in this feasibility summary."
        )
    return notes


def _numeric_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"available": False}
    vals = sorted(float(v) for v in values)
    n = len(vals)
    return {
        "available": True,
        "n": n,
        "min": round(vals[0], 3),
        "median": round(vals[n // 2], 3),
        "max": round(vals[-1], 3),
        "mean": round(sum(vals) / n, 3),
    }


def _extra_aliases(concept_id: str, label: str) -> set[str]:
    aliases = {concept_id.replace("_", " "), label.lower()}
    if concept_id in {"vaso_ind", "norepi_equiv", "norepi_rate", "norepi_dur"}:
        aliases.update(
            {
                "vasopressor",
                "vasopressors",
                "vasopressor use",
                "early vasopressors",
                "pressor",
                "pressors",
                "norepinephrine",
                "noradrenaline",
            }
        )
    if concept_id in {"total_input_ml", "fluid_balance", "fluid_balance_cumulative"}:
        aliases.update(
            {
                "fluid",
                "fluids",
                "iv fluid",
                "intravenous fluid",
                "intravenous fluids",
                "fluid volume",
                "fluid exposure",
                "fluid-sparing",
                "restricted fluid",
                "fluid resuscitation",
                "液体",
                "液体平衡",
                "累计液体平衡",
                "液体复苏",
            }
        )
    if concept_id in {
        "map",
        "sbp",
        "shock_index",
        "modified_shock_index",
        "diastolic_shock_index",
    }:
        aliases.update({"blood pressure", "hypotension", "shock"})
    if concept_id == "lact":
        aliases.update({"lactate", "乳酸"})
    if concept_id == "mech_vent":
        aliases.update(
            {
                "mechanical ventilation",
                "mechanically ventilated",
                "invasive ventilation",
                "noninvasive ventilation",
                "non-invasive ventilation",
                "机械通气",
                "撤机",
                "脱机",
                "呼吸机脱离",
            }
        )
    if concept_id == "death":
        aliases.update({"mortality", "death", "survival", "死亡", "病死率"})
    if concept_id == "sep3_sofa1":
        aliases.update(
            {"sepsis", "sepsis-3", "septic shock", "suspected infection", "脓毒症"}
        )
    if concept_id == "sep3_sofa2":
        aliases.update(
            {
                "sofa-2",
                "sofa 2",
                "experimental sepsis sensitivity",
                "sepsis sensitivity sofa-2",
                "sepsis-3 sofa-2",
                "sofa-2 based sepsis",
                "实验性脓毒症敏感性",
                "基于sofa-2的脓毒症",
            }
        )
    if concept_id == "aki":
        aliases.update({"aki", "acute kidney injury", "急性肾损伤"})
    return aliases


def _alias_hit(haystack: str, alias: str) -> bool:
    needle = _norm_text(alias)
    if not needle or len(needle) < 2:
        return False
    if re.search(r"[\u4e00-\u9fff]", needle):
        return needle in haystack
    return (
        re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack)
        is not None
    )


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("_", " ")).strip()


def _first_sentence(text: str) -> str:
    text = _clean(text, _MAX_SOURCE_QUOTE)
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return _clean(parts[0] if parts else text, _MAX_SOURCE_QUOTE)


def _clean(value: Any, max_len: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:max_len]


def _year(value: Any) -> Optional[int]:
    try:
        y = int(str(value).strip())
    except Exception:
        return None
    return y if 1900 <= y <= 2100 else None


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _run_id(source: Dict[str, Any], idea: Dict[str, Any]) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    nonce = f"{time.time_ns() % 1_000_000_000:09d}"
    digest = _sha256(
        json.dumps(
            [source.get("source_id"), idea.get("idea_title")], ensure_ascii=False
        )
    )[:8]
    return f"idea_{stamp}_{nonce}_{digest}"


def _history_key(run_id: Any, created_at: Any) -> str:
    return f"{str(run_id or '').strip()}::{str(created_at or '').strip()}"


def _slug(value: Any, fallback: str = "item") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip().lower()).strip(
        "-._"
    )
    return slug[:96] or fallback


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _norm_path(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return str(Path(path).expanduser())
