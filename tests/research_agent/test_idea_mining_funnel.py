from __future__ import annotations

from typing import Sequence

import easyicu.research_agent as ra

from easyicu.research_agent.discovery.idea_mining import freeze_source_snapshot
from easyicu.research_agent.discovery.idea_mining_funnel import (
    LiteratureFunnelSpec,
    build_literature_funnel_queries,
    extract_gap_sections_from_text,
    fetch_literature_funnel_corpus,
)
from easyicu.research_agent.discovery.idea_scope import LiteratureScopeSpec
from easyicu.research_agent.literature import CitationRecord


class FakeFunnelSearchClient:
    def __init__(self, records: Sequence[CitationRecord]):
        self.records = list(records)
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, *, retmax: int = 20) -> list[CitationRecord]:
        self.queries.append((query, retmax))
        return self.records[:retmax]


class FakeTextClient:
    def fetch_gap_text(self, citation: CitationRecord, *, route):
        return (
            f"Background sentence for {citation.key}. "
            "Limitations include incomplete measurement frequency data. "
            "Methods should not be mined as a gap. "
            "Future research should examine transportability across cohorts."
        )


def _scope() -> LiteratureScopeSpec:
    return LiteratureScopeSpec(
        journal_preset="critical_care_top3",
        last_n_years=3,
        topic_terms=["acute respiratory failure"],
    )


def test_build_literature_funnel_queries_adds_route_specific_terms() -> None:
    routes = build_literature_funnel_queries(
        LiteratureFunnelSpec(base_scope=_scope(), platform_gap_terms=["unit policy"]),
        reference_year=2026,
    )

    by_name = {route.route_name: route.pubmed_query for route in routes}
    assert set(by_name) == {"review_gap", "primary_limitation", "platform_gap"}
    assert "2024:2026[dp]" in by_name["review_gap"]
    assert '"future research"' in by_name["review_gap"]
    assert "NOT (review[pt] OR editorial[pt])" in by_name["primary_limitation"]
    assert "transportability" in by_name["platform_gap"]
    assert '"unit policy"' in by_name["platform_gap"]


def test_review_gap_route_can_sweep_commentary_and_letters() -> None:
    scope = LiteratureScopeSpec(
        journal_preset="critical_care_specialty_broad",
        pub_types=[],
        last_n_years=2,
        topic_terms=["critical illness"],
    )

    routes = build_literature_funnel_queries(
        LiteratureFunnelSpec(base_scope=scope),
        reference_year=2026,
    )

    review_query = {route.route_name: route.pubmed_query for route in routes}[
        "review_gap"
    ]
    assert '"J Crit Care"[Journal]' in review_query
    assert "guideline[pt]" in review_query
    assert "systematic review[pt]" in review_query
    assert "letter[pt]" in review_query


def test_extract_gap_sections_from_text_returns_short_verbatim_gap_snippets() -> None:
    text = (
        "This review summarizes established evidence. "
        "Limitations include inconsistent timing definitions across studies. "
        "The methods section describes the search strategy. "
        "Future work should test measurement bias in broader ICU cohorts."
    )

    snippets = extract_gap_sections_from_text(text, max_chars=180, max_sections=2)

    assert len(snippets) == 2
    assert snippets[0].startswith("Limitations include")
    assert "Future work should test" in snippets[1]
    assert all("methods section" not in snippet.lower() for snippet in snippets)


def test_fetch_literature_funnel_corpus_keeps_only_gap_excerpt_in_snapshot() -> None:
    secret_full_text = "Methods should not be mined as a gap."
    records = [
        CitationRecord(
            key="gap_review_2026",
            title="Gaps in critical care measurement",
            year="2026",
            venue="Crit Care",
            pmid="123",
        )
    ]
    search = FakeFunnelSearchClient(records)

    result = fetch_literature_funnel_corpus(
        LiteratureFunnelSpec(base_scope=_scope()),
        search,
        text_client=FakeTextClient(),
        reference_year=2026,
        retmax_per_route=2,
    )

    assert len(result.query_routes) == 3
    assert search.queries and all(retmax == 2 for _query, retmax in search.queries)
    assert len(result.materials) == 1
    material = result.materials[0]
    assert material.source_adapter_level == "user_supplied_excerpt"
    assert material.discovery_route == "review_gap"
    assert material.source_text_role == "gap_excerpt"
    assert material.source_rank == 1
    assert "Limitations include" in (material.source_text or "")
    assert secret_full_text not in (material.source_text or "")

    manifest = freeze_source_snapshot(result.materials)
    item = manifest.items[0]
    assert item.discovery_route == "review_gap"
    assert item.source_text_role == "gap_excerpt"
    assert item.source_text_sha256
    assert item.source_text_stored is False
    assert secret_full_text not in manifest.model_dump_json()


def test_package_lazy_exports_literature_funnel_helpers() -> None:
    assert callable(ra.build_literature_funnel_queries)
    assert callable(ra.fetch_literature_funnel_source_materials)
    assert callable(ra.extract_gap_sections_from_text)
