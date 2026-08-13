"""Digest-bound authority for literature searched inside a Copilot study.

Owner: ``easyicu.webserver.literature_authority``.  The public contract stores
only bibliographic metadata and bounded source excerpts returned by the
existing PubMed adapter.  It binds that exact retrieval receipt to the
scientific StudyContext scope and later projects a Research-Agent
``LiteratureBundle`` seed.  This module performs no search, model call, patient
data access, planning, or scientific screening.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from easyicu.webserver import study_contexts

LITERATURE_AUTHORITY_SCHEMA_VERSION = "easyicu.web-literature-authority/2"
_AUTHORITY_ROOT = Path.home() / ".easyicu" / "literature_authorities"
_MAX_RECEIPT_BYTES = 512_000
_MAX_CITATIONS = 20
_MAX_QUERIES = 8
_RECEIPT_ID = re.compile(r"^lit_[a-f0-9]{24}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")


class LiteratureAuthorityError(ValueError):
    """Owner-attributable integrity failure for one literature receipt."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        self.message = str(message)
        super().__init__(self.message)


def persist_literature_authority(
    *,
    study: Mapping[str, Any],
    discovered: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist one completed Web search and return its compact binding."""

    status = _text(discovered.get("status"), 80)
    if not bool(discovered.get("search_performed")) or status not in {
        "searched",
        "searched_no_hits",
    }:
        raise LiteratureAuthorityError(
            "literature_authority_search_incomplete",
            "Only a completed PubMed metadata search can become Plan authority.",
        )
    searched_at = _aware_iso_timestamp(discovered.get("searched_at")) or _now()
    queries = list(
        dict.fromkeys(
            _text(value, 1_200)
            for value in list(discovered.get("queries_to_run") or [])[:_MAX_QUERIES]
            if _text(value, 1_200)
        )
    )
    if not queries:
        raise LiteratureAuthorityError(
            "literature_authority_queries_required",
            "A completed literature search must retain its exact query strings.",
        )
    citations = [
        citation
        for citation in (
            _citation(row)
            for row in list(discovered.get("source_candidates") or [])[
                :_MAX_CITATIONS
            ]
            if isinstance(row, Mapping)
        )
        if citation is not None
    ]
    if status == "searched" and not citations:
        raise LiteratureAuthorityError(
            "literature_authority_results_required",
            "A search marked searched must retain at least one valid PubMed record.",
        )
    if status == "searched_no_hits" and citations:
        raise LiteratureAuthorityError(
            "literature_authority_status_mismatch",
            "A no-hits receipt cannot contain PubMed records.",
        )

    scope_sha256 = study_contexts.literature_search_scope_sha256(dict(study))
    payload = {
        "schema_version": LITERATURE_AUTHORITY_SCHEMA_VERSION,
        "searched_at": searched_at,
        "status": status,
        "study_configuration_sha256": scope_sha256,
        "search": {
            "source": "web_pubmed",
            "queries": queries,
            "query_strata": _query_strata(discovered.get("query_strata")),
            "network_calls": _nonnegative_int(discovered.get("network_calls")),
            "result_count": len(citations),
        },
        "citations": citations,
        "privacy": {
            "patient_rows_recorded": False,
            "full_text_recorded": False,
            "host_paths_recorded": False,
            "external_llm_calls": 0,
        },
    }
    # Content-address the receipt so a failed StudyContext CAS cannot leave an
    # unbounded trail of unreachable authority files. Repeating the exact same
    # search receipt for the exact same scientific scope is idempotent.
    semantic_digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    receipt_id = f"lit_{semantic_digest[:24]}"
    payload["receipt_id"] = receipt_id
    raw = _canonical_bytes(payload)
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise LiteratureAuthorityError(
            "literature_authority_receipt_too_large",
            "The bounded literature receipt exceeds its metadata budget.",
        )
    target = _receipt_path(receipt_id)
    _atomic_write(target, raw)
    return {
        "schema_version": LITERATURE_AUTHORITY_SCHEMA_VERSION,
        "receipt_id": receipt_id,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "status": status,
        "result_count": len(citations),
        "searched_at": searched_at,
        "study_configuration_sha256": scope_sha256,
    }


def load_bound_literature(
    *,
    study: Mapping[str, Any],
    research_question: str,
) -> Optional[dict[str, Any]]:
    """Verify the current StudyContext binding and return its exact seed."""

    binding = study.get("literature_authority")
    binding = binding if isinstance(binding, Mapping) else {}
    if not binding:
        return None
    if binding.get("schema_version") != LITERATURE_AUTHORITY_SCHEMA_VERSION:
        raise LiteratureAuthorityError(
            "literature_authority_schema_invalid",
            "The StudyContext literature binding has an unsupported schema.",
        )
    receipt_id = _text(binding.get("receipt_id"), 80)
    if not _RECEIPT_ID.fullmatch(receipt_id):
        raise LiteratureAuthorityError(
            "literature_authority_receipt_id_invalid",
            "The StudyContext literature binding has an invalid receipt id.",
        )
    expected_digest = _text(binding.get("receipt_sha256"), 80).lower()
    if not _DIGEST.fullmatch(expected_digest):
        raise LiteratureAuthorityError(
            "literature_authority_digest_invalid",
            "The StudyContext literature binding has no valid receipt digest.",
        )
    raw, payload = _read_receipt(receipt_id)
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise LiteratureAuthorityError(
            "literature_authority_digest_mismatch",
            "The Web literature receipt changed after it was bound to the study.",
        )
    if payload.get("schema_version") != LITERATURE_AUTHORITY_SCHEMA_VERSION:
        raise LiteratureAuthorityError(
            "literature_authority_receipt_schema_invalid",
            "The bound Web literature receipt has an unsupported schema.",
        )
    if payload.get("receipt_id") != receipt_id:
        raise LiteratureAuthorityError(
            "literature_authority_receipt_id_mismatch",
            "The Web literature receipt identity no longer matches its binding.",
        )

    current_scope = study_contexts.literature_search_scope_sha256(dict(study))
    expected_scope = _text(binding.get("study_configuration_sha256"), 80).lower()
    receipt_scope = _text(payload.get("study_configuration_sha256"), 80).lower()
    if not _DIGEST.fullmatch(expected_scope) or not _DIGEST.fullmatch(receipt_scope):
        raise LiteratureAuthorityError(
            "literature_authority_scope_digest_invalid",
            "The Web literature receipt has no valid scientific-scope digest.",
        )
    if current_scope != expected_scope or receipt_scope != expected_scope:
        raise LiteratureAuthorityError(
            "literature_authority_scope_mismatch",
            "The study changed after its Web literature search; search again before planning.",
        )

    status = _text(payload.get("status"), 80)
    if status not in {"searched", "searched_no_hits"} or status != _text(
        binding.get("status"), 80
    ):
        raise LiteratureAuthorityError(
            "literature_authority_status_mismatch",
            "The bound literature search status no longer matches StudyContext.",
        )
    searched_at = _aware_iso_timestamp(payload.get("searched_at"))
    if not searched_at or searched_at != _aware_iso_timestamp(
        binding.get("searched_at")
    ):
        raise LiteratureAuthorityError(
            "literature_authority_timestamp_mismatch",
            "The bound literature search timestamp is missing or changed.",
        )
    raw_citations = payload.get("citations")
    if not isinstance(raw_citations, list):
        raise LiteratureAuthorityError(
            "literature_authority_citations_invalid",
            "The bound literature receipt has no valid citation list.",
        )
    citations = [
        citation
        for citation in (
            _validated_citation(row)
            for row in raw_citations[:_MAX_CITATIONS]
            if isinstance(row, Mapping)
        )
        if citation is not None
    ]
    if len(citations) != len(raw_citations):
        raise LiteratureAuthorityError(
            "literature_authority_citations_invalid",
            "One or more bound PubMed citation records are invalid.",
        )
    expected_count = binding.get("result_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(citations)
    ):
        raise LiteratureAuthorityError(
            "literature_authority_count_mismatch",
            "The bound literature result count no longer matches its receipt.",
        )
    if (status == "searched") != bool(citations):
        raise LiteratureAuthorityError(
            "literature_authority_status_mismatch",
            "The literature status and citation count are inconsistent.",
        )
    search = payload.get("search")
    search = search if isinstance(search, Mapping) else {}
    queries = [
        _text(value, 1_200)
        for value in list(search.get("queries") or [])[:_MAX_QUERIES]
        if _text(value, 1_200)
    ]
    if not queries or int(search.get("result_count") or 0) != len(citations):
        raise LiteratureAuthorityError(
            "literature_authority_search_receipt_invalid",
            "The bound literature search receipt is incomplete.",
        )
    record_queries: dict[str, list[str]] = {}
    for raw_row, citation in zip(raw_citations, citations):
        matched = _matched_queries(raw_row)
        if any(query not in queries for query in matched):
            raise LiteratureAuthorityError(
                "literature_authority_record_query_mismatch",
                "A citation claims a retrieval query outside the bound search receipt.",
            )
        if matched:
            record_queries[citation["key"]] = matched
    count = len(citations)
    return {
        "research_question": _text(research_question, 2_000),
        "citations": citations,
        "prisma": {
            "identified": count,
            "duplicates_removed": 0,
            "screened": count,
            "eligible": 0,
            "included": 0,
        },
        "search_provenance": {
            "schema_version": "easyicu.literature_search_provenance/1",
            "curated_seed_count": 0,
            "sources_enabled": ["web_pubmed"],
            "sources_returning": ["web_pubmed"] if citations else [],
            "search_queries": {"web_pubmed": queries},
            "record_queries": record_queries,
            "search_conducted": True,
            "searched_at": searched_at,
            "note": (
                "The Web Copilot performed this PubMed metadata search and "
                "the exact receipt is digest-bound to the current StudyContext. "
                "Research Agent screening is performed again against its sealed context."
            ),
        },
        "screening_decisions": [],
    }


def _citation(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    pmid = _text(row.get("pmid"), 32)
    title = _text(row.get("title"), 500)
    if not re.fullmatch(r"[0-9]{1,12}", pmid) or not title:
        return None
    year_match = re.search(r"\b(?:19|20)\d{2}\b", _text(row.get("year"), 80))
    design_excerpt = _text(row.get("design_excerpt"), 1_200)
    excerpt = design_excerpt or _text(row.get("evidence_quote"), 1_200)
    publication_types = list(
        dict.fromkeys(
            _text(value, 120)
            for value in list(row.get("publication_types") or [])[:20]
            if _text(value, 120)
        )
    )
    doi = _text(row.get("doi"), 240) or None
    return {
        "key": f"web_pubmed_{pmid}",
        "title": title,
        "year": year_match.group(0) if year_match else "n/a",
        "venue": _text(row.get("journal"), 240) or None,
        "relevance": (
            f"Study-design excerpt: {excerpt}"
            if excerpt
            else "PubMed title metadata only; no abstract excerpt retained."
        ),
        "doi": doi,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "pmid": pmid,
        "publication_types": publication_types,
        "matched_queries": [
            _text(value, 1_200)
            for value in list(row.get("matched_queries") or [])[:_MAX_QUERIES]
            if _text(value, 1_200)
        ],
        "matched_query_strata": [
            _text(value, 80)
            for value in list(row.get("matched_query_strata") or [])[:_MAX_QUERIES]
            if _text(value, 80)
        ],
    }


def _validated_citation(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    required = {
        "key",
        "title",
        "year",
        "venue",
        "relevance",
        "doi",
        "url",
        "pmid",
    }
    allowed = required | {
        "publication_types",
        "matched_queries",
        "matched_query_strata",
    }
    if not required.issubset(row) or not set(row).issubset(allowed):
        return None
    pmid = _text(row.get("pmid"), 32)
    key = _text(row.get("key"), 120)
    url = _text(row.get("url"), 500)
    if (
        not re.fullmatch(r"[0-9]{1,12}", pmid)
        or key != f"web_pubmed_{pmid}"
        or url != f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        or not _text(row.get("title"), 500)
    ):
        return None
    publication_types = row.get("publication_types")
    if not isinstance(publication_types, list) or any(
        not isinstance(value, str) or not _text(value, 120)
        for value in publication_types[:20]
    ):
        return None
    return {
        "key": key,
        "title": _text(row.get("title"), 500),
        "year": _text(row.get("year"), 16) or "n/a",
        "venue": _text(row.get("venue"), 240) or None,
        "relevance": _text(row.get("relevance"), 1_500) or None,
        "doi": _text(row.get("doi"), 240) or None,
        "url": url,
        "pmid": pmid,
        "publication_types": list(
            dict.fromkeys(_text(value, 120) for value in publication_types[:20])
        ),
    }


def _matched_queries(row: Mapping[str, Any]) -> list[str]:
    values = row.get("matched_queries")
    if not isinstance(values, list):
        return []
    return list(
        dict.fromkeys(
            _text(value, 1_200)
            for value in values[:_MAX_QUERIES]
            if _text(value, 1_200)
        )
    )


def _query_strata(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    for row in rows[:_MAX_QUERIES]:
        if not isinstance(row, Mapping):
            continue
        stratum = _text(row.get("id"), 80)
        query = _text(row.get("query"), 1_200)
        if not stratum or not query:
            continue
        out.append(
            {
                "id": stratum,
                "query": query,
                "returned_count": _nonnegative_int(row.get("returned_count")),
                "retained_count": _nonnegative_int(row.get("retained_count")),
            }
        )
    return out


def _receipt_path(receipt_id: str) -> Path:
    if not _RECEIPT_ID.fullmatch(receipt_id):
        raise LiteratureAuthorityError(
            "literature_authority_receipt_id_invalid",
            "The literature receipt id is invalid.",
        )
    return _AUTHORITY_ROOT / f"{receipt_id}.json"


def _read_receipt(receipt_id: str) -> tuple[bytes, dict[str, Any]]:
    path = _receipt_path(receipt_id)
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise LiteratureAuthorityError(
                "literature_authority_receipt_unsafe",
                "The literature receipt is not a regular file.",
            )
        if metadata.st_size > _MAX_RECEIPT_BYTES:
            raise LiteratureAuthorityError(
                "literature_authority_receipt_too_large",
                "The literature receipt exceeds its metadata budget.",
            )
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise LiteratureAuthorityError(
            "literature_authority_receipt_missing",
            "The bound Web literature receipt is missing.",
        ) from exc
    except OSError as exc:
        raise LiteratureAuthorityError(
            "literature_authority_receipt_unreadable",
            "The bound Web literature receipt could not be read.",
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiteratureAuthorityError(
            "literature_authority_receipt_invalid",
            "The bound Web literature receipt is not valid UTF-8 JSON.",
        ) from exc
    if not isinstance(payload, dict):
        raise LiteratureAuthorityError(
            "literature_authority_receipt_invalid",
            "The bound Web literature receipt root must be an object.",
        )
    return raw, payload


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        temporary.replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


__all__ = [
    "LITERATURE_AUTHORITY_SCHEMA_VERSION",
    "LiteratureAuthorityError",
    "load_bound_literature",
    "persist_literature_authority",
]
