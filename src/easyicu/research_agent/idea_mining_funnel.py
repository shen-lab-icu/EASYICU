"""Literature-funnel helpers for broader idea mining.

This module is the "find better and more completely" layer before S4
extraction. It stays deterministic and side-effect free except for caller
injected clients:

* build multiple PubMed-style routes from one scope;
* retrieve citation metadata through an injected search client;
* optionally convert caller-accessible text into short gap excerpts.

It never fetches full text by itself and never stores full text in snapshots.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .idea_mining_schema import SourceMaterial
from .idea_scope import LiteratureScopeSpec, build_pubmed_query_from_scope
from .literature import CitationRecord

_GAP_ROUTE_TERMS = (
    '"future research" OR "future work" OR "future studies" OR '
    'limitation OR limitations OR uncertainty OR "evidence gap"'
)
_PRIMARY_LIMITATION_TERMS = (
    '"future work" OR limitation OR limitations OR "open question" OR unresolved'
)
_PLATFORM_GAP_TERMS = (
    '"external validation" OR transportability OR harmonization OR harmonisation '
    'OR missingness OR "data quality" OR "measurement bias"'
)
_GAP_KEYWORDS = (
    "future research",
    "future work",
    "future studies",
    "future study",
    "limitation",
    "limitations",
    "uncertainty",
    "uncertain",
    "unresolved",
    "knowledge gap",
    "evidence gap",
    "open question",
    "warrant",
    "needed",
    "needs to be",
    "remain",
    "remains",
)


class LiteratureFunnelRoute(BaseModel):
    """One auditable retrieval route in a funnel run."""

    model_config = ConfigDict(extra="forbid")

    route_name: str
    purpose: str
    scope: LiteratureScopeSpec
    pubmed_query: str


class LiteratureFunnelSpec(BaseModel):
    """Multi-route literature discovery spec built around a base scope."""

    model_config = ConfigDict(extra="forbid")

    base_scope: LiteratureScopeSpec
    include_review_gap_route: bool = True
    include_primary_limitation_route: bool = True
    include_platform_gap_route: bool = True
    platform_gap_terms: List[str] = Field(default_factory=list)
    max_gap_excerpt_chars: int = Field(default=1200, ge=200, le=4000)
    max_gap_sections: int = Field(default=4, ge=1, le=12)

    @model_validator(mode="after")
    def _at_least_one_route(self) -> "LiteratureFunnelSpec":
        if not (
            self.include_review_gap_route
            or self.include_primary_limitation_route
            or self.include_platform_gap_route
        ):
            raise ValueError("at least one literature funnel route must be enabled")
        return self


class LiteratureFunnelResult(BaseModel):
    """Retrieved source materials plus frozen route metadata."""

    model_config = ConfigDict(extra="forbid")

    query_routes: List[LiteratureFunnelRoute]
    materials: List[SourceMaterial]


def build_literature_funnel_queries(
    spec: LiteratureFunnelSpec | LiteratureScopeSpec,
    *,
    reference_year: Optional[int] = None,
) -> List[LiteratureFunnelRoute]:
    """Build deterministic route queries from one base literature scope."""

    funnel = (
        spec
        if isinstance(spec, LiteratureFunnelSpec)
        else LiteratureFunnelSpec(base_scope=spec)
    )
    routes: List[LiteratureFunnelRoute] = []
    base = funnel.base_scope

    if funnel.include_review_gap_route:
        scope = _route_scope(
            base,
            route_terms=_GAP_ROUTE_TERMS,
            pub_types=base.pub_types
            or [
                "review",
                "editorial",
                "guideline",
                "practice_guideline",
                "systematic_review",
                "letter",
            ],
        )
        routes.append(
            _route(
                "review_gap",
                "reviews/editorials/guidelines/letters that explicitly discuss gaps",
                scope,
                reference_year=reference_year,
            )
        )

    if funnel.include_primary_limitation_route:
        scope = _route_scope(
            base,
            route_terms=f"({_PRIMARY_LIMITATION_TERMS}) NOT (review[pt] OR editorial[pt])",
            pub_types=[],
        )
        routes.append(
            _route(
                "primary_limitation",
                "primary studies whose title/abstract flags limitations or open questions",
                scope,
                reference_year=reference_year,
            )
        )

    if funnel.include_platform_gap_route:
        extra_terms = _PLATFORM_GAP_TERMS
        if funnel.platform_gap_terms:
            supplied = " OR ".join(
                _quote_if_needed(term)
                for term in funnel.platform_gap_terms
                if str(term or "").strip()
            )
            if supplied:
                extra_terms = f"({extra_terms}) OR ({supplied})"
        scope = _route_scope(base, route_terms=extra_terms, pub_types=[])
        routes.append(
            _route(
                "platform_gap",
                "transportability, harmonization, missingness, and measurement gaps",
                scope,
                reference_year=reference_year,
            )
        )
    return routes


def fetch_literature_funnel_corpus(
    spec: LiteratureFunnelSpec | LiteratureScopeSpec,
    search_client: Any,
    *,
    text_client: Optional[Any] = None,
    reference_year: Optional[int] = None,
    retmax_per_route: int = 20,
) -> LiteratureFunnelResult:
    """Retrieve source materials for every funnel route.

    ``search_client`` must provide ``search(query, retmax=...)`` or be callable.
    ``text_client`` is optional and may provide short caller-authorized text via
    ``fetch_gap_text`` / ``fetch_text`` / ``get_text`` or be callable. Returned
    full text is immediately reduced to gap excerpts before S4 sees it.
    """

    funnel = (
        spec
        if isinstance(spec, LiteratureFunnelSpec)
        else LiteratureFunnelSpec(base_scope=spec)
    )
    routes = build_literature_funnel_queries(funnel, reference_year=reference_year)
    by_key: Dict[str, SourceMaterial] = {}
    for route in routes:
        records = _call_search(search_client, route.pubmed_query, retmax_per_route)
        for rank, raw_record in enumerate(records, start=1):
            citation = _coerce_citation(raw_record)
            material = _material_for_route(
                citation,
                route=route,
                route_rank=rank,
                text_client=text_client,
                max_gap_excerpt_chars=funnel.max_gap_excerpt_chars,
                max_gap_sections=funnel.max_gap_sections,
            )
            existing = by_key.get(citation.key)
            if existing is None or _material_is_richer(material, existing):
                by_key[citation.key] = material
    return LiteratureFunnelResult(query_routes=routes, materials=list(by_key.values()))


def fetch_literature_funnel_source_materials(
    spec: LiteratureFunnelSpec | LiteratureScopeSpec,
    search_client: Any,
    *,
    text_client: Optional[Any] = None,
    reference_year: Optional[int] = None,
    retmax_per_route: int = 20,
) -> List[SourceMaterial]:
    """Convenience wrapper returning only S4 source materials."""

    return fetch_literature_funnel_corpus(
        spec,
        search_client,
        text_client=text_client,
        reference_year=reference_year,
        retmax_per_route=retmax_per_route,
    ).materials


def extract_gap_sections_from_text(
    text: str,
    *,
    max_chars: int = 1200,
    max_sections: int = 4,
) -> List[str]:
    """Return short verbatim sections that look like gap/limitation text."""

    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []
    units = _candidate_text_units(clean)
    out: List[str] = []
    total = 0
    for unit in units:
        if not _contains_gap_keyword(unit):
            continue
        snippet = _truncate_text(unit, min(500, max_chars))
        if snippet in out:
            continue
        if total + len(snippet) > max_chars and out:
            break
        out.append(snippet)
        total += len(snippet)
        if len(out) >= max_sections:
            break
    return out


def _route_scope(
    base: LiteratureScopeSpec,
    *,
    route_terms: str,
    pub_types: Sequence[str],
) -> LiteratureScopeSpec:
    return base.model_copy(
        update={
            "pub_types": list(pub_types),
            "extra_terms": _merge_extra_terms(base.extra_terms, route_terms),
        }
    )


def _route(
    name: str,
    purpose: str,
    scope: LiteratureScopeSpec,
    *,
    reference_year: Optional[int],
) -> LiteratureFunnelRoute:
    return LiteratureFunnelRoute(
        route_name=name,
        purpose=purpose,
        scope=scope,
        pubmed_query=build_pubmed_query_from_scope(
            scope, reference_year=reference_year
        ),
    )


def _merge_extra_terms(existing: Optional[str], route_terms: str) -> str:
    existing = str(existing or "").strip()
    if existing:
        return f"({existing}) AND ({route_terms})"
    return str(route_terms).strip()


def _quote_if_needed(term: str) -> str:
    text = str(term or "").strip()
    if not text:
        return ""
    return f'"{text}"' if re.search(r"\s|-", text) else text


def _call_search(search_client: Any, query: str, retmax: int) -> Sequence[Any]:
    if hasattr(search_client, "search"):
        try:
            return search_client.search(query, retmax=retmax)
        except TypeError:
            return search_client.search(query, max_results=retmax)
    if callable(search_client):
        try:
            return search_client(query, retmax=retmax)
        except TypeError:
            return search_client(query, max_results=retmax)
    raise TypeError("search_client must provide search() or be callable")


def _coerce_citation(raw: Any) -> CitationRecord:
    if isinstance(raw, CitationRecord):
        return raw
    if isinstance(raw, Mapping):
        title = str(raw.get("title") or raw.get("Title") or "Untitled")
        year = str(raw.get("year") or raw.get("Year") or "")
        raw_key = raw.get("key") or raw.get("pmid") or raw.get("id")
        if raw_key is None or str(raw_key).strip() in {"", "None"}:
            # A keyless record (a legitimate shape for a lightweight client)
            # must NOT stringify to the literal "None": the funnel dedups by
            # citation.key, so every keyless paper from every route would
            # collapse into one and the rest are silently dropped. Synthesize a
            # stable per-record key from its content instead.
            venue = str(raw.get("venue") or raw.get("journal") or "")
            digest = hashlib.sha256(
                f"{title}|{year}|{venue}".encode("utf-8")
            ).hexdigest()[:16]
            key = f"synthetic:{digest}"
        else:
            key = str(raw_key)
        return CitationRecord(
            key=key,
            title=title,
            year=year,
            venue=raw.get("venue") or raw.get("journal"),
            relevance=raw.get("relevance") or raw.get("abstract"),
            doi=raw.get("doi"),
            url=raw.get("url"),
            pmid=str(raw["pmid"]) if raw.get("pmid") is not None else None,
        )
    raise TypeError(f"cannot coerce citation record: {type(raw)!r}")


def _material_for_route(
    citation: CitationRecord,
    *,
    route: LiteratureFunnelRoute,
    route_rank: int,
    text_client: Optional[Any],
    max_gap_excerpt_chars: int,
    max_gap_sections: int,
) -> SourceMaterial:
    excerpt = ""
    if text_client is not None:
        text = _call_text_client(text_client, citation, route)
        snippets = extract_gap_sections_from_text(
            text,
            max_chars=max_gap_excerpt_chars,
            max_sections=max_gap_sections,
        )
        excerpt = "\n\n".join(snippets)
    if excerpt:
        return SourceMaterial(
            citation=citation,
            source_adapter_level="user_supplied_excerpt",
            locator=f"{route.route_name}:gap_excerpt",
            source_text=excerpt,
            discovery_route=route.route_name,
            source_text_role="gap_excerpt",
            source_rank=route_rank,
        )
    return SourceMaterial(
        citation=citation,
        source_adapter_level="metadata_only",
        locator=route.route_name,
        discovery_route=route.route_name,
        source_text_role="metadata_proxy",
        source_rank=route_rank,
    )


def _call_text_client(
    text_client: Any, citation: CitationRecord, route: LiteratureFunnelRoute
) -> str:
    for method_name in ("fetch_gap_text", "fetch_text", "get_text"):
        if not hasattr(text_client, method_name):
            continue
        method = getattr(text_client, method_name)
        try:
            return str(method(citation, route=route) or "")
        except TypeError:
            return str(method(citation) or "")
    if callable(text_client):
        try:
            return str(text_client(citation, route=route) or "")
        except TypeError:
            return str(text_client(citation) or "")
    return ""


def _material_is_richer(new: SourceMaterial, old: SourceMaterial) -> bool:
    return bool(new.source_text) and not bool(old.source_text)


def _candidate_text_units(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _contains_gap_keyword(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in _GAP_KEYWORDS)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


__all__ = [
    "LiteratureFunnelResult",
    "LiteratureFunnelRoute",
    "LiteratureFunnelSpec",
    "build_literature_funnel_queries",
    "extract_gap_sections_from_text",
    "fetch_literature_funnel_corpus",
    "fetch_literature_funnel_source_materials",
]
