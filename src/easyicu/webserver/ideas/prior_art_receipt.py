"""Digest-bound bridge from Web Idea Mining prior art to Agent planning.

Owner: ``easyicu.webserver.ideas``.  The public contract accepts one fixed
``prior_art_check.json`` artifact and returns either a compact binding or a
Research-Agent ``LiteratureBundle`` payload.  It has no network, provider,
patient-data, UI, or pipeline dependencies; callers choose the fixed artifact
path and the Research Agent validates the returned payload again.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

PRIOR_ART_BINDING_SCHEMA_VERSION = "easyicu.idea-prior-art-binding/1"
_MAX_RECEIPT_BYTES = 512_000
_MAX_CITATIONS = 20


class PriorArtReceiptError(ValueError):
    """Owner-attributable integrity failure for a bound prior-art receipt."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        self.message = str(message)
        super().__init__(self.message)


def build_prior_art_binding(path: Path) -> Optional[dict[str, Any]]:
    """Return a compact binding only for a completed, non-failed search."""

    raw, payload = _read_receipt(path)
    if payload is None:
        return None
    prior = payload.get("prior_art")
    prior = prior if isinstance(prior, Mapping) else {}
    searched = bool(prior.get("search_performed"))
    status = _text(prior.get("status"), 80)
    searched_at = _aware_iso_timestamp(prior.get("searched_at"))
    if not searched or status in {"", "search_failed"} or not searched_at:
        return None
    results = _results(prior)
    return {
        "prior_art_binding_schema_version": PRIOR_ART_BINDING_SCHEMA_VERSION,
        "prior_art_sha256": hashlib.sha256(raw).hexdigest(),
        "prior_art_status": status,
        "prior_art_result_count": len(results),
        "prior_art_searched_at": searched_at,
    }


def load_bound_prior_art_literature(
    path: Path,
    *,
    binding: Mapping[str, Any],
    research_question: str,
) -> dict[str, Any]:
    """Verify a binding and project its exact PubMed records for planning."""

    schema = _text(binding.get("prior_art_binding_schema_version"), 120)
    if schema != PRIOR_ART_BINDING_SCHEMA_VERSION:
        raise PriorArtReceiptError(
            "prior_art_binding_schema_invalid",
            "The accepted Idea handoff has an unsupported prior-art binding schema.",
        )
    expected_digest = _text(binding.get("prior_art_sha256"), 80).lower()
    if not re.fullmatch(r"[a-f0-9]{64}", expected_digest):
        raise PriorArtReceiptError(
            "prior_art_binding_digest_invalid",
            "The accepted Idea handoff has no valid prior-art receipt digest.",
        )
    raw, payload = _read_receipt(path, required=True)
    assert payload is not None
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise PriorArtReceiptError(
            "prior_art_binding_digest_mismatch",
            "The Idea Mining prior-art receipt changed after acceptance.",
        )
    prior = payload.get("prior_art")
    prior = prior if isinstance(prior, Mapping) else {}
    status = _text(prior.get("status"), 80)
    expected_status = _text(binding.get("prior_art_status"), 80)
    if not bool(prior.get("search_performed")) or status in {"", "search_failed"}:
        raise PriorArtReceiptError(
            "prior_art_binding_search_incomplete",
            "The bound prior-art receipt does not contain a completed PubMed search.",
        )
    if status != expected_status:
        raise PriorArtReceiptError(
            "prior_art_binding_status_mismatch",
            "The prior-art search status no longer matches the accepted handoff.",
        )
    searched_at = _aware_iso_timestamp(prior.get("searched_at"))
    expected_searched_at = _aware_iso_timestamp(binding.get("prior_art_searched_at"))
    if not searched_at or not expected_searched_at:
        raise PriorArtReceiptError(
            "prior_art_binding_search_timestamp_invalid",
            "The prior-art binding has no valid timezone-aware search timestamp.",
        )
    if searched_at != expected_searched_at:
        raise PriorArtReceiptError(
            "prior_art_binding_search_timestamp_mismatch",
            "The prior-art search timestamp no longer matches the accepted handoff.",
        )
    results = _results(prior)
    expected_count = binding.get("prior_art_result_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise PriorArtReceiptError(
            "prior_art_binding_count_invalid",
            "The accepted handoff has no valid prior-art result count.",
        )
    if expected_count != len(results):
        raise PriorArtReceiptError(
            "prior_art_binding_count_mismatch",
            "The prior-art result count no longer matches the accepted handoff.",
        )

    citations = [_citation(row) for row in results[:_MAX_CITATIONS]]
    citations = [row for row in citations if row is not None]
    result_count = len(results)
    return {
        "research_question": _text(research_question, 2_000),
        "citations": citations,
        "prisma": {
            "identified": result_count,
            "duplicates_removed": 0,
            "screened": result_count,
            "eligible": len(citations),
            "included": len(citations),
        },
        "search_provenance": {
            "schema_version": "easyicu.literature_search_provenance/1",
            "curated_seed_count": 0,
            "sources_enabled": ["idea_mining_pubmed"],
            "sources_returning": ["idea_mining_pubmed"] if citations else [],
            "search_conducted": True,
            "searched_at": searched_at,
            "note": (
                "PubMed metadata was searched by Web Idea Mining and the exact "
                "receipt was digest-bound when the Idea handoff was accepted."
            ),
        },
    }


def _aware_iso_timestamp(value: Any) -> str:
    text = _text(value, 80)
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return ""
    return text


def _read_receipt(
    path: Path,
    *,
    required: bool = False,
) -> tuple[bytes, Optional[dict[str, Any]]]:
    try:
        size = path.stat().st_size
        if size > _MAX_RECEIPT_BYTES:
            raise PriorArtReceiptError(
                "prior_art_receipt_too_large",
                "The Idea Mining prior-art receipt exceeds its metadata budget.",
            )
        raw = path.read_bytes()
    except FileNotFoundError:
        if not required:
            return b"", None
        raise PriorArtReceiptError(
            "prior_art_receipt_missing",
            "The accepted Idea handoff's prior-art receipt is missing.",
        )
    except OSError as exc:
        raise PriorArtReceiptError(
            "prior_art_receipt_unreadable",
            "The Idea Mining prior-art receipt could not be read.",
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PriorArtReceiptError(
            "prior_art_receipt_invalid",
            "The Idea Mining prior-art receipt is not valid UTF-8 JSON.",
        ) from exc
    if not isinstance(payload, dict):
        raise PriorArtReceiptError(
            "prior_art_receipt_invalid",
            "The Idea Mining prior-art receipt root must be an object.",
        )
    return raw, payload


def _results(prior: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = prior.get("results")
    if not isinstance(values, list):
        return []
    return [row for row in values if isinstance(row, Mapping)][:_MAX_CITATIONS]


def _citation(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    pmid = _text(row.get("pmid"), 32)
    title = _text(row.get("title"), 500)
    if not pmid or not re.fullmatch(r"[0-9]{1,12}", pmid) or not title:
        return None
    year_match = re.search(
        r"\b(?:19|20)\d{2}\b",
        " ".join(_text(row.get(key), 120) for key in ("year", "pubdate", "epubdate")),
    )
    year = year_match.group(0) if year_match else "n/a"
    doi = _text(row.get("doi"), 240) or None
    query = _text(row.get("query"), 800)
    return {
        "key": f"idea_pubmed_{pmid}",
        "title": title,
        "year": year,
        "venue": _text(row.get("journal") or row.get("source"), 240) or None,
        "relevance": (
            f"Matched the accepted Idea Mining PubMed query: {query}"
            if query
            else "Returned by the accepted Idea Mining PubMed metadata search."
        ),
        "doi": doi,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "pmid": pmid,
    }


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


__all__ = [
    "PRIOR_ART_BINDING_SCHEMA_VERSION",
    "PriorArtReceiptError",
    "build_prior_art_binding",
    "load_bound_prior_art_literature",
]
