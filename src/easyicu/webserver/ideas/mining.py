"""Native Web Idea Mining adapter.

This module exposes a local-first, metadata-only discovery workflow for the
FastAPI UI.  Live literature search remains explicit network opt-in.  Local PDF
and literature-folder ingestion are allowed, but only bounded excerpts, file
metadata, and hashes are returned; full text is not persisted.  The web contract
produces source evidence, idea ledger rows, data-dictionary feasibility,
active-export pre-experiment summaries, and a frozen plan handoff draft.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import html
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import parse, request
from xml.etree import ElementTree as ET

from easyicu.concept import catalog as concept_catalog
from easyicu.webserver import dataio
from easyicu.webserver import sources as source_store

_CONFIG_DIR = Path.home() / ".easyicu"
_RUN_ROOT = _CONFIG_DIR / "idea_mining_runs"
_HISTORY_PATH = _CONFIG_DIR / "webserver_idea_mining_runs.json"
_AGENT_PROJECTS_ROOT = _CONFIG_DIR / "agent_project_seeds"
_AGENT_PROJECTS_PATH = _CONFIG_DIR / "webserver_agent_project_seeds.json"

_MAX_SOURCE_QUOTE = 420
_MAX_FEATURE_STATS = 24
_MAX_FETCH_BYTES = 256_000
_MAX_PDF_BYTES = 20 * 1024 * 1024
_MAX_PDF_EXTRACT_PAGES = 8
_MAX_PDF_EXCERPT = 1_200
_MAX_LITERATURE_PDFS = 80
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


class IdeaMiningWebError(ValueError):
    """Raised when the web idea-mining adapter must fail closed."""

    def __init__(self, detail: Dict[str, Any]):
        self.detail = detail
        super().__init__(str(detail.get("error") or "idea_mining_error"))


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
    source = _source_record(body)
    source_type = str(source.get("source_type") or "manual")
    allow_network = bool(body.get("allow_network"))
    adapter = {
        "status": "metadata_ready",
        "source_type": source_type,
        "network_calls": 0,
        "external_llm_calls": 0,
        "fetch_performed": False,
        "full_text_stored": False,
    }
    suggestion = {
        "topic": _clean(body.get("topic") or source.get("title") or "", 600),
        "excerpt": _clean(
            body.get("excerpt") or source.get("evidence_quote") or "", _MAX_SOURCE_QUOTE
        ),
        "title": source.get("title"),
        "journal": source.get("journal"),
        "year": source.get("year"),
        "doi": source.get("doi"),
        "pmid": source.get("pmid"),
        "url": source.get("url"),
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
                "reason": fetched.get("reason"),
            }
        )
        if fetched.get("title") and not suggestion.get("title"):
            suggestion["title"] = fetched["title"]
        if fetched.get("description") and not suggestion.get("excerpt"):
            suggestion["excerpt"] = fetched["description"]
        if fetched.get("doi") and not suggestion.get("doi"):
            suggestion["doi"] = fetched["doi"]
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
    allow_network = bool(body.get("allow_network"))
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
            network_calls += 2
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
    allow_network = bool(body.get("allow_network"))
    if not allow_network:
        prior = {
            "status": "blocked_network_opt_in_required",
            "search_performed": False,
            "queries_to_run": queries,
            "results": [],
            "public_database_used_by_prior_work": "unknown_until_search",
            "reason": "Prior-art checking needs explicit network opt-in. No request was made.",
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
        (run_dir / "prior_art_check.json").write_text(
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
    plan = dict(payload.get("handoff_plan") or {})
    if edits:
        plan["human_plan_notes"] = edits[:1200]
        plan["selection_mode"] = "human_curated_with_text_edits"
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
        "pre_experiment": payload.get("pre_experiment") or {},
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
    _assert_no_row_payload(handoff)
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "idea_handoff.json").write_text(
        json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return handoff


def create_agent_project(body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a metadata-only Agent Projects seed from a frozen handoff."""
    run_id = str(body.get("run_id") or "").strip()
    idea_id = str(body.get("idea_id") or "").strip()
    handoff = _load_handoff(run_id)
    if not handoff:
        handoff = create_handoff(body)
    if idea_id and str(handoff.get("idea_id") or "") != idea_id:
        handoff = create_handoff(body)
    seed = _agent_project_seed(handoff)
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
    excerpt = _clean(
        body.get("excerpt") or body.get("source_quote") or "", _MAX_SOURCE_QUOTE
    )
    source_text = _clean(
        body.get("excerpt") or body.get("abstract") or body.get("notes") or topic, 4000
    )
    citation_key = _slug(
        "|".join([title, str(year or ""), journal, doi, pmid]) or topic or "source"
    )
    record = {
        "source_id": "source_"
        + _sha256("|".join([title, journal, str(year), doi, pmid, url]))[:12],
        "citation_key": citation_key,
        "source_type": source_type,
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
    if body.get("source_file_name"):
        record["source_file_name"] = _clean(body.get("source_file_name"), 240)
    if body.get("source_file_sha256"):
        record["source_file_sha256"] = _clean(body.get("source_file_sha256"), 80)
    if body.get("literature_folder"):
        record["literature_folder"] = _norm_path(
            str(body.get("literature_folder") or "")
        )
    if body.get("literature_pdf_count") is not None:
        try:
            record["literature_pdf_count"] = int(body.get("literature_pdf_count") or 0)
        except Exception:
            record["literature_pdf_count"] = 0
    return record


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
    overall = _overall_feasibility(concept_rows)
    novelty = _prior_art(source, title)
    go_no_go = (
        "recommend"
        if overall["tier"] == "executable"
        else ("hold" if overall["tier"].startswith("T1") else "db-cannot-do")
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
        }
    source, desc = export
    entity_ids = export_index.get("entity_ids") or set()
    concepts = [
        row.get("concept_id")
        for row in idea.get("mapped_concepts") or []
        if row.get("concept_id") in export_index.get("concept_to_file", {})
    ]
    if not concepts:
        concepts = list((export_index.get("concept_to_file") or {}).keys())[:6]
    stats = _feature_stats(
        Path(str(desc.get("path") or source.get("path"))),
        concepts,
        export_index,
        entity_ids,
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
    return {
        "status": status,
        "payload_scope": "aggregate_pre_experiment_no_row_payload",
        "source": {
            "label": source.get("label") or desc.get("label") or "Local export",
            "path_hash": _sha256(str(source.get("path") or desc.get("path") or ""))[
                :16
            ],
            "database": desc.get("database"),
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
        "interpretation": _pre_experiment_interpretation(stats),
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
            "title": source.get("title"),
            "year": source.get("year"),
            "journal": source.get("journal"),
            "quote": source.get("evidence_quote"),
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
        "analysis_plan": [
            "Confirm cohort denominator and inclusion/exclusion criteria.",
            "Run feature availability and missingness audit before modeling.",
            "Estimate descriptive association only after evidence gate passes.",
            "Block manuscript claims unless every numeric sentence binds to evidence.",
        ],
        "blocked_until": [
            "human confirms the idea and plan",
            "active export contains required features or re-extraction is complete",
            "prior-art search is reviewed when network/LLM search is enabled",
        ],
        "reportable": False,
        "draft_unlocked": False,
    }


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
        return {"concept_to_file": {}, "entity_ids": set()}
    _source, desc = export
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
    return {"concept_to_file": concept_to_file, "entity_ids": entity_ids}


def _feature_stats(
    root: Path,
    concepts: Iterable[str],
    export_index: Dict[str, Any],
    entity_ids: set[str],
) -> List[Dict[str, Any]]:
    import pandas as pd

    out: List[Dict[str, Any]] = []
    concept_to_file = export_index.get("concept_to_file") or {}
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
        try:
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
        non_null = frame[frame[concept_id].notna()].copy()
        observed_entities = int(entity_col[frame[concept_id].notna()].nunique())
        records = int(len(non_null))
        nums = dataio._numeric_values(non_null[concept_id])[:10000]
        coverage = round(observed_entities / denominator * 100, 1)
        out.append(
            {
                "concept_id": concept_id,
                "label": _concept_label(concept_id),
                "module": str(item.get("module") or ""),
                "records": records,
                "observed_entities": observed_entities,
                "coverage_pct": coverage,
                "missing_pct": round(100 - coverage, 1),
                "time_indexed": any(col in frame.columns for col in _TIME_COLUMNS),
                "numeric_summary": _numeric_summary(nums),
                "status": "ready" if records else "missing",
            }
        )
        if len(out) >= _MAX_FEATURE_STATS:
            break
    return out


def _concept_feasibility(
    row: Dict[str, Any], export_index: Dict[str, Any]
) -> Dict[str, Any]:
    concept_id = row.get("concept_id")
    in_export = concept_id in (export_index.get("concept_to_file") or {})
    if in_export:
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


def _overall_feasibility(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "tier": "T3_not_in_db",
            "label": "No mapped EasyICU concept",
            "reason": "No source phrase mapped to the current EasyICU dictionary.",
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
        "reason": "This local run did not call PubMed, journal sites, or an external LLM. Run the opt-in literature search stage before claiming novelty.",
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


def _agent_project_seed(handoff: Dict[str, Any]) -> Dict[str, Any]:
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
        "cohort": (plan.get("cohort") or {}).get("default")
        or "adult ICU cohort from active EasyICU export",
        "source": {
            "title": source.get("title"),
            "year": source.get("year"),
            "journal": source.get("journal"),
            "doi": source.get("doi"),
            "pmid": source.get("pmid"),
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
        "analysis_plan": list(plan.get("analysis_plan") or []),
        "human_plan_notes": plan.get("human_plan_notes"),
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
            }
        ],
    }


def _prior_art_queries(source: Dict[str, Any], title: str) -> List[str]:
    title = _clean(title or source.get("title") or "ICU idea", 180)
    source_title = _clean(source.get("title") or "", 180)
    queries = [
        f'("{title}") AND (ICU OR critical care)',
        f'("{title}") AND (MIMIC OR eICU OR "public database")',
    ]
    if source_title and source_title != title:
        queries.append(f'("{source_title}") AND (MIMIC OR eICU OR ICU)')
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


def _fetch_url_metadata(url: str) -> Dict[str, Any]:
    if not url.lower().startswith(("http://", "https://")):
        return {
            "status": "invalid_url",
            "network_calls": 0,
            "reason": "URL must start with http:// or https://.",
        }
    try:
        req = request.Request(
            url, headers={"User-Agent": "EasyICU-local-metadata-resolver/1.0"}
        )
        with request.urlopen(req, timeout=_NETWORK_TIMEOUT_SEC) as resp:
            raw = resp.read(_MAX_FETCH_BYTES)
        text = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return {"status": "fetch_failed", "network_calls": 1, "reason": str(exc)[:240]}
    title = _html_title(text)
    description = _html_meta(text, "description") or _html_meta(text, "og:description")
    doi = _doi_from_text(text)
    return {
        "status": (
            "metadata_fetched"
            if title or description or doi
            else "metadata_fetch_empty"
        ),
        "network_calls": 1,
        "title": title,
        "description": _clean(description, _MAX_SOURCE_QUOTE),
        "doi": doi,
        "bytes_read": min(len(raw), _MAX_FETCH_BYTES),
        "reason": "Stored bounded metadata only; full HTML was not persisted.",
    }


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
    return {
        "status": (
            "searched"
            if deduped
            else ("search_failed" if errors else "searched_no_hits")
        ),
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
        if row.get("concept_id") in {"death", "los_icu", "aki", "sep3_sofa2"}:
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
    preserve that concept set so the pre-experiment can honestly say which
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
    parts.append("This is a pre-experiment triage result, not a manuscript finding.")
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
        return "Review pre-experiment statistics, edit the plan, then hand off to Agent Projects."
    if go == "hold":
        return "Re-run extraction with missing modules/features, then repeat the pre-experiment check."
    return "Choose another database or revise the idea."


def _pre_experiment_interpretation(stats: List[Dict[str, Any]]) -> List[str]:
    if not stats:
        return [
            "No mapped feature is present in the active export; run extraction with the needed modules first."
        ]
    low = [row for row in stats if float(row.get("coverage_pct") or 0) < 50]
    notes = [f"{len(stats)} mapped feature(s) were summarized from the active export."]
    if low:
        notes.append(
            f"{len(low)} feature(s) have <50% entity coverage and should be treated as feasibility risks."
        )
    else:
        notes.append(
            "Mapped features have at least 50% entity coverage in this pre-experiment summary."
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
