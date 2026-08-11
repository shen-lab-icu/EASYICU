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


LITERATURE_EVIDENCE_SCHEMA_VERSION = "easyicu.web-literature-evidence/1"
_MAX_CITATIONS = 80
_MAX_STEPS = 80


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


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
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        return None
    return candidate


def _citation_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "key": _text(row.get("key") or row.get("citation_key"), 120),
        "title": _text(row.get("title"), 500),
        "year": _text(row.get("year"), 16) or None,
        "venue": _text(row.get("venue") or row.get("journal"), 240) or None,
        "relevance": _text(
            row.get("relevance")
            or row.get("evidence_quote")
            or row.get("evidence_sentence"),
            1_200,
        )
        or None,
        "doi": _text(row.get("doi"), 240) or None,
        "pmid": _text(row.get("pmid"), 32) or None,
        "source_url": _source_url(row),
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
    citations = [
        _citation_projection(row)
        for row in list(raw_citations)[:_MAX_CITATIONS]
        if isinstance(row, Mapping) and _text(row.get("title"), 500)
    ]
    citation_keys = {row["key"] for row in citations if row.get("key")}

    raw_steps = plan.get("steps")
    raw_steps = raw_steps if isinstance(raw_steps, Sequence) else []
    step_map: list[dict[str, Any]] = []
    mapped_steps = 0
    unknown_keys: set[str] = set()
    for raw in list(raw_steps)[:_MAX_STEPS]:
        if not isinstance(raw, Mapping):
            continue
        requested = raw.get("literature_citation_keys")
        requested = requested if isinstance(requested, Sequence) and not isinstance(requested, (str, bytes)) else []
        keys = []
        for value in requested:
            key = _text(value, 120)
            if key and key not in keys:
                keys.append(key)
        unknown_keys.update(key for key in keys if key not in citation_keys)
        valid_keys = [key for key in keys if key in citation_keys]
        if valid_keys:
            mapped_steps += 1
        step_map.append(
            {
                "step_id": _text(raw.get("step_id") or raw.get("id"), 160),
                "intent": _text(raw.get("intent") or raw.get("title"), 1_200),
                "planned_analysis_role": _text(
                    raw.get("planned_analysis_role"), 40
                )
                or None,
                "citation_keys": valid_keys,
                "support_status": "bound" if valid_keys else "not_bound",
            }
        )

    provenance = bundle.get("search_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    searched = bool(provenance.get("search_conducted"))
    status = "searched" if searched else ("curated_only" if citations else "unavailable")
    if not step_map:
        mapping_status = "not_applicable"
    elif mapped_steps == len(step_map):
        mapping_status = "complete"
    elif mapped_steps:
        mapping_status = "partial"
    else:
        mapping_status = "not_bound"

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
        "search": {
            "search_conducted": searched,
            "curated_seed_count": int(provenance.get("curated_seed_count") or 0),
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
            "prisma": dict(prisma) if prisma else None,
        },
        "citation_count": len(citations),
        "citations": citations,
        "plan_step_count": len(step_map),
        "mapped_step_count": mapped_steps,
        "mapping_status": mapping_status,
        "step_citation_map": step_map,
        "integrity": {
            "unknown_citation_keys_removed": sorted(unknown_keys),
            "path_values_returned": False,
            "patient_rows_returned": False,
        },
        "evidence_boundary": (
            "Literature supports design rationale and prior-art review. It is "
            "separate from patient/result EvidenceStore evidence and does not "
            "make an analysis result reportable."
        ),
    }


def load_run_literature_projection(
    *, run_dir: Path, run_id: str, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Load only the fixed owner-issued bundle path inside one pipeline run."""

    path = run_dir / "preplan_literature_bundle.json"
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        payload = {}
    if not isinstance(payload, Mapping):
        payload = {}
    return project_run_literature(run_id=run_id, bundle=payload, plan=plan)


def literature_source_resource(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build a bounded click target from an Idea Mining PubMed owner record."""

    citation = _citation_projection(row)
    if not citation.get("title") or not citation.get("source_url"):
        return None
    return {
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
        "authority_class": "literature_metadata",
    }


__all__ = [
    "LITERATURE_EVIDENCE_SCHEMA_VERSION",
    "literature_source_resource",
    "load_run_literature_projection",
    "project_run_literature",
]
