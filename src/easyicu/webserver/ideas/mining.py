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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import parse, request
from xml.etree import ElementTree as ET

from easyicu.concept import catalog as concept_catalog
from easyicu.research_agent.discovery.discovery_handoff import DiscoveryHandoffPacket
from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store
from easyicu.webserver.ideas.handoff import (
    CanonicalHandoffIntegrityError,
    build_web_handoff_packet,
    is_legacy_handoff_envelope,
    load_validated_canonical_handoff,
    persist_canonical_handoff,
)
from easyicu.webserver.input_validation import parse_bool

_CONFIG_DIR = Path.home() / ".easyicu"
_RUN_ROOT = _CONFIG_DIR / "idea_mining_runs"
_HISTORY_PATH = _CONFIG_DIR / "webserver_idea_mining_runs.json"
_AGENT_PROJECTS_ROOT = _CONFIG_DIR / "agent_project_seeds"
_AGENT_PROJECTS_PATH = _CONFIG_DIR / "webserver_agent_project_seeds.json"

_MAX_SOURCE_QUOTE = 420
_MAX_FEATURE_STATS = 24
_MAX_FETCH_BYTES = 256_000
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
    export = _active_export()
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
    ids: List[str] = []
    for query in queries[:3]:
        try:
            found = _pubmed_esearch(query, limit=limit)
            network_calls += 1
            ids.extend(found)
        except Exception as exc:
            errors.append(str(exc)[:240])
    ids = _dedupe_strings(ids)[:limit]
    articles: List[Dict[str, Any]] = []
    if ids:
        try:
            articles = _pubmed_article_records(ids)
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

    export = _active_export()
    export_index = _export_index(export)
    source_candidates: List[Dict[str, Any]] = []
    idea_candidates: List[Dict[str, Any]] = []
    for article in articles:
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
        "queries_to_run": queries,
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
    payload = _load_run(run_id) if run_id else None
    idea = _selected_idea(payload or {}, str(body.get("idea_id") or ""))
    source = ((payload or {}).get("source_evidence") or [{}])[0]
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
    queries = _prior_art_queries(source, str(idea.get("idea_title") or "ICU idea"))
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
        prior = _pubmed_prior_art(queries)
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
    mode = str(body.get("mode") or ("replan" if edits else "plan")).strip().lower()
    source = ((payload.get("source_evidence") or [{}])[0]) or {}
    pre = payload.get("pre_experiment") or {}
    prior = _load_prior_art(run_id)
    plan = _analysis_plan_draft(source, idea, pre, prior, edits=edits, mode=mode)
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


def bounded_sample_feasibility(body: Dict[str, Any]) -> Dict[str, Any]:
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
    export = _active_export()
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
    required = [
        str(row.get("concept_id") or "")
        for row in idea.get("mapped_concepts") or []
        if row.get("concept_id")
    ]
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
    low = [
        row
        for row in stats
        if row.get("metric_kind") != "event_rate" and row.get("low_coverage")
    ]
    status = "ready" if stats and not missing_required else "blocked"
    if status == "ready" and low:
        status = "needs_review"
    if not denominator_resolved and status == "ready":
        status = "needs_review"

    out = {
        "ok": True,
        "schema_version": "easyicu.web_idea_bounded_sample_feasibility/1",
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
        "missing_required_concepts": missing_required,
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
    (run_dir / "bounded_sample_feasibility.json").write_text(
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
        ideas[0] if ideas else None,
    )
    if not idea:
        raise IdeaMiningWebError({"error": "idea_not_found", "idea_id": idea_id})
    edits = str(body.get("plan_edits") or "").strip()
    plan_artifact = _load_plan(run_id)
    plan = dict((plan_artifact or {}).get("plan") or payload.get("handoff_plan") or {})
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
        idea,
        pre_experiment,
        prior_art_check,
    )
    handoff = {
        "ok": True,
        "schema_version": "easyicu.web_idea_handoff/1",
        "created_at": _now(),
        "run_id": run_id,
        "idea_id": idea.get("idea_id"),
        "candidate_topic": idea.get("idea_title"),
        "go_no_go": idea.get("go_no_go"),
        "go_no_go_reason": idea.get("go_no_go_reason"),
        "selected_ledger_row": idea,
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
            idea=idea,
            source=source,
            plan=plan,
            pre_experiment=pre_experiment,
            prior_art_check=prior_art_check,
            run_dir=run_dir,
        )
        handoff.update(
            persist_canonical_handoff(canonical_packet, run_dir=run_dir)
        )
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
    handoff = _load_handoff(run_id)
    canonical_packet: Optional[DiscoveryHandoffPacket] = None
    if not handoff:
        handoff = create_handoff(body)
    if idea_id and str(handoff.get("idea_id") or "") != idea_id:
        handoff = create_handoff(body)
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
                plan = _load_plan(run_id) or {}
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
    """List metadata-only Agent project seeds created from Idea Mining."""
    limit = int((body or {}).get("limit") or 20)
    seeds = _read_agent_projects()
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


def _idea_from_source(
    source: Dict[str, Any],
    text: str,
    hits: List[Dict[str, Any]],
    export_index: Dict[str, Any],
) -> Dict[str, Any]:
    outcome = _pick_outcome(text, hits)
    concepts = _select_idea_concepts(text, hits, outcome)
    predictor = _primary_predictor(concepts)
    if predictor is None:
        predictor = _pick_predictor(hits, outcome)
        if predictor is None and hits:
            predictor = hits[0]
        concepts = [row for row in [predictor, outcome] if row]
    if not concepts and hits:
        concepts = _dedupe_concepts(hits[:3])
    title = _idea_title(source, predictor, outcome, concepts)
    concept_rows = [_concept_feasibility(row, export_index) for row in concepts]
    overall = _overall_feasibility(concept_rows, export_index)
    novelty = _prior_art(source, title)
    go_no_go = (
        "recommend"
        if overall["tier"] == "executable"
        else (
            "hold"
            if overall["tier"].startswith("T1") or overall["tier"] == "demo_only"
            else "db-cannot-do"
        )
    )
    idea_payload = {
        "idea_id": "idea_"
        + _sha256(json.dumps([source.get("source_id"), title], ensure_ascii=False))[
            :12
        ],
        "idea_title": title,
        "population": "adult ICU cohort",
        "exposure_or_predictor": _concept_set_label(concepts)
        or (predictor["label"] if predictor else _clean(text, 90)),
        "outcome": outcome["label"] if outcome else "In-hospital mortality",
        "analysis_family": _analysis_family(text),
        "source_id": source.get("source_id"),
        "source_title": source.get("title"),
        "source_year": source.get("year"),
        "source_journal": source.get("journal"),
        "source_quote": source.get("evidence_quote"),
        "rationale": _rationale(text, predictor, outcome, concepts),
        "mapped_concepts": concept_rows,
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
    question = f"Evaluate whether {predictor} is associated with {outcome} in an adult ICU cohort."
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
            "default": "adult ICU cohort from active EasyICU export",
            "requires_user_confirmation": True,
        },
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
        return {"concept_to_file": {}, "entity_ids": set(), "demo_like": False}
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
        entity_col = frame["stay_id"].map(dataio._norm_id)
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
            "coverage_basis": "bounded_file_head_sample",
            "sample_limit_records": max_records,
            "numeric_summary": {"available": False, "kind": "empty_sample"},
            "summary_label": "No usable values were found in the bounded sample.",
            "status": "missing",
        }
    entity_col = frame["stay_id"].map(dataio._norm_id)
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
        "coverage_basis": "bounded_file_head_sample",
        "sample_limit_records": max_records,
        "numeric_summary": _numeric_summary(nums),
        "summary_label": "Coverage is computed inside the bounded sample only.",
        "status": "ready" if records else "missing",
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


def _prior_art(source: Dict[str, Any], title: str) -> Dict[str, Any]:
    return {
        "status": "not_checked_external_search_required",
        "novelty_label": "unknown_until_search",
        "direct_same_topic_hits": [],
        "public_database_used_by_prior_work": "unknown_until_search",
        "reason": "Prior-art interpretation has not been run. Use the source article as inspiration, then run opt-in metadata search before claiming novelty.",
        "opportunity_frame": "Existing trials, reviews, or editorials should shape the ICU-database question: comparator, subgroup, timing window, outcome horizon, and whether the new angle is exploratory rather than already answered.",
        "next_use": "After opt-in, classify the literature as already answered, partially answered, or inspiration for a new ICU exploratory analysis.",
        "queries_to_run": _prior_art_queries(source, title),
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


def _load_bounded_sample(run_id: str) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    path = _run_dir(run_id) / "bounded_sample_feasibility.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    ideas = payload.get("idea_ledger") or []
    if idea_id:
        match = next(
            (row for row in ideas if str(row.get("idea_id") or "") == idea_id), None
        )
        if match:
            return match
    return ideas[0] if ideas else None


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
    frozen_notes = str(
        (handoff.get("handoff_plan") or {}).get("human_plan_notes") or ""
    ).strip()
    body_edits = str(body.get("plan_edits") or "").strip()
    if body_edits and body_edits != frozen_notes:
        return True
    plan = _load_plan(run_id) or {}
    plan_notes = str(plan.get("human_plan_notes") or "").strip()
    if plan_notes and plan_notes != frozen_notes:
        return True
    return False


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
}


def _execution_gate(
    idea: Dict[str, Any],
    pre_experiment: Dict[str, Any],
    prior_art_check: Optional[Dict[str, Any]],
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


def _prior_art_queries(source: Dict[str, Any], title: str) -> List[str]:
    title = _clean(title or source.get("title") or "ICU idea", 180)
    source_title = _clean(source.get("title") or "", 180)
    source_quote = _clean(source.get("evidence_quote") or "", 220)
    exploratory_terms = _clean(
        " ".join(re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", source_quote))[:160], 180
    )
    queries = [
        f'("{title}") AND (ICU OR critical care)',
        f'("{title}") AND (MIMIC OR eICU OR "public database")',
    ]
    if source_title and source_title != title:
        queries.append(f'("{source_title}") AND (MIMIC OR eICU OR ICU)')
    if exploratory_terms:
        queries.append(
            f'({exploratory_terms}) AND (subgroup OR timing OR trajectory OR "public database" OR MIMIC OR eICU)'
        )
    doi = _clean(source.get("doi") or "", 120)
    if doi:
        queries.append(f'"{doi}"')
    return queries[:4]


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
    base = f'({topic}) AND (ICU OR "critical care" OR "intensive care")'
    review = f"({topic}) AND (review[Publication Type] OR editorial[Publication Type] OR perspective OR commentary)"
    db = f'({topic}) AND (MIMIC OR eICU OR "public database" OR "critical care database")'
    return [
        base + journal_filter + year_filter,
        review + journal_filter + year_filter,
        db + year_filter,
    ]


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


def _pubmed_article_records(ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch PubMed metadata plus bounded abstracts via EFetch XML."""
    if not ids:
        return []
    params = parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + params
    with request.urlopen(url, timeout=_NETWORK_TIMEOUT_SEC) as resp:
        raw = resp.read(_MAX_FETCH_BYTES)
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
        abstract = _clean(
            " ".join(
                _node_text(node)
                for node in article.findall("./Abstract/AbstractText")
                if _node_text(node)
            ),
            3000,
        )
        evidence = _best_evidence_sentence(abstract or title)
        rows.append(
            {
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "doi": doi,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                "abstract_excerpt": _clean(abstract, _MAX_PDF_EXCERPT),
                "evidence_sentence": evidence,
                "full_text_stored": False,
            }
        )
    order = {str(pmid): i for i, pmid in enumerate(ids)}
    rows.sort(key=lambda row: order.get(str(row.get("pmid") or ""), 9999))
    return rows


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
            if title or description or doi
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
            if fetch_error and not (title or description or doi)
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


def _pubmed_prior_art(queries: List[str]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    calls = 0
    errors: List[str] = []
    for query in queries[:3]:
        try:
            ids = _pubmed_esearch(query, limit=5)
            calls += 1
            if ids:
                rows = _pubmed_esummary(ids)
                calls += 1
                for row in rows:
                    row["query"] = query
                results.extend(rows)
        except Exception as exc:
            errors.append(str(exc)[:240])
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in results:
        pmid = str(row.get("pmid") or "")
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        deduped.append(row)
    public_hits = [
        row
        for row in deduped
        if re.search(
            r"\b(MIMIC|eICU|HiRID|AUMC|SICdb|public database)\b",
            json.dumps(row, ensure_ascii=False),
            re.I,
        )
    ]
    status = (
        "searched" if deduped else ("search_failed" if errors else "searched_no_hits")
    )
    if deduped:
        interpretation = (
            f"Found {len(deduped)} metadata hit(s). Treat them as a map of what has been tried: "
            "which comparator, cohort, database, endpoint, and time window they used. This does not automatically kill the idea; it tells the planner how to refine the new ICU exploration."
        )
    elif errors:
        interpretation = "The metadata search failed or was incomplete. Do not claim novelty; keep the idea as a planning draft until prior art can be reviewed."
    else:
        interpretation = "No metadata hit was found with the bounded queries. This is only weak evidence of novelty; broaden queries or review manually before presenting the idea."
    return {
        "status": status,
        "search_performed": True,
        "network_calls": calls,
        "queries_to_run": queries,
        "result_count": len(deduped),
        "results": deduped[:12],
        "public_database_used_by_prior_work": (
            "possible" if public_hits else "not_detected_in_metadata"
        ),
        "direct_same_topic_hits": deduped[:5],
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


def _pick_outcome(text: str, hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    low = text.lower()
    if any(tok in low for tok in ["death", "mortality", "survival", "死亡", "病死"]):
        return _concept_hit("death")
    for row in hits:
        if row.get("concept_id") in {"death", "los_icu", "aki"}:
            return row
    return _concept_hit("death")


def _pick_predictor(
    hits: List[Dict[str, Any]], outcome: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    outcome_id = outcome.get("concept_id") if outcome else None
    for row in hits:
        if row.get("concept_id") != outcome_id and row.get("concept_id") not in {
            "death",
            "los_icu",
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

    if any(
        tok in low
        for tok in ("vasopressor", "norepinephrine", "noradrenaline", "pressor")
    ):
        for concept_id in _INTERVENTION_PRIORITY:
            if concept_id in by_id:
                add(concept_id, "exposure")
                break
    if any(
        tok in low
        for tok in (
            "fluid",
            "fluids",
            "intravenous volume",
            "fluid-sparing",
            "resuscitation",
        )
    ):
        for concept_id in (
            "total_input_ml",
            "fluid_balance_cumulative",
            "fluid_balance",
        ):
            if concept_id in by_id:
                add(concept_id, "exposure")
                break
    for concept_id in _SEVERITY_PRIORITY:
        if concept_id in by_id:
            add(concept_id, "covariate_or_subgroup")
    if outcome:
        selected.append({**outcome, "role": "outcome"})
    for row in hits:
        if len(selected) >= 10:
            break
        concept_id = str(row.get("concept_id"))
        if concept_id in {"death", "los_icu"}:
            continue
        selected.append({**row, "role": "feature"})
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
    for role in ("exposure", "predictor", "covariate_or_subgroup", "feature"):
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
    features = [
        str(row.get("label"))
        for row in concepts
        if row.get("role") not in {"outcome"} and row.get("label")
    ]
    return " + ".join(features[:3])


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
        and float(row.get("coverage_pct") or 0) < 50
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
            }
        )
    if concept_id == "death":
        aliases.update({"mortality", "death", "survival", "死亡", "病死率"})
    if concept_id == "sep3_sofa2":
        aliases.update(
            {"sepsis", "sepsis-3", "septic shock", "suspected infection", "脓毒症"}
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
