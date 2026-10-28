"""Synthetic anonymous transport receipts, venue proxies and source degradation."""

import json
import urllib.error
from urllib.parse import parse_qs, urlsplit

import pytest

from easyicu.research_agent import literature as owner
from easyicu.research_agent import bibliographic_metadata as openalex
from easyicu.research_agent.planning.literature_contract import ImpactProxies, ReviewPolicy
from tests.support.plan_literature import FakePubMed, context, record


def work(index=1):
    text = record().relevance.removeprefix("Study-design excerpt: ")
    inverted = {}
    for offset, token in enumerate(text.split()):
        inverted.setdefault(token, []).append(offset)
    return {
        "id": f"https://openalex.org/W{index}", "display_name": record().title,
        "publication_year": int(record().year), "cited_by_count": 24,
        "ids": {"pmid": f"https://pubmed.ncbi.nlm.nih.gov/{10000 + index}/"},
        "doi": f"https://doi.org/10.1234/synthetic{index}", "type": "article",
        "primary_location": {"source": {"id": "https://openalex.org/S1", "display_name": "Synthetic Journal"}},
        "abstract_inverted_index": inverted,
        "authorships": [{"author": {"display_name": "Synthetic Author"}}],
    }


def test_openalex_anonymous_transport_has_pacing_metrics_abstracts_and_lineage(monkeypatch):
    requests, pace = [], []

    class Response:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(self.data).encode()

    def urlopen(url, *, timeout):
        requests.append(url)
        assert len(pace) == len(requests)
        assert timeout == 15.0
        query = parse_qs(urlsplit(url).query)
        assert "api_key" not in query and "api-key" not in query
        if "/sources/" in url:
            return Response({"summary_stats": {"2yr_mean_citedness": 3.5, "h_index": 55}})
        assert "[Title/Abstract]" not in query["search"][0]
        return Response({"results": [work(1), work(2)]})

    monkeypatch.setattr(openalex.urllib.request, "urlopen", urlopen)
    client = openalex.OpenAlexLiteratureClient(pace=lambda: pace.append(True))
    bundle = owner.LiteratureAgent(enable_openalex=True, openalex_client=client).run(context())
    rows = [row for row in bundle.citations if row.key.startswith("openalex_")]
    assert len(rows) == 2 and bundle.prisma["included"] == 2
    assert rows[0].authors == ["Synthetic Author"]
    assert "We studied adult patients" in rows[0].relevance
    assert rows[0].impact.citations_per_year == 12
    assert rows[0].impact.venue_two_year_mean_citedness == 3.5
    assert rows[0].impact.venue_h_index == 55
    assert set(rows[0].impact.proxies_used) == {"citations_per_year", "venue_two_year_mean_citedness", "venue_h_index"}
    assert sum("/sources/" in url for url in requests) == 1
    assert bundle.search_provenance.sources_returning == ["openalex"]
    assert bundle.search_provenance.record_queries[rows[0].key]
    assert bundle.search_provenance.query_roles["openalex"]["data_source"]
    assert not bundle.search_provenance.sources_failing


@pytest.mark.parametrize("error", [TimeoutError("synthetic timeout"), urllib.error.HTTPError("synthetic", 429, "slow", None, None), ValueError("invalid JSON")])
def test_a_failing_openalex_source_preserves_pubmed_and_records_the_shortfall(monkeypatch, error):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(openalex.urllib.request, "urlopen", fail)
    client = openalex.OpenAlexLiteratureClient(pace=lambda: None)
    bundle = owner.LiteratureAgent(
        enable_pubmed=True, pubmed_client=FakePubMed([record()]),
        enable_openalex=True, openalex_client=client,
    ).run(context())
    assert bundle.prisma["included"] == 1
    assert bundle.search_provenance.sources_failing["openalex"]
    assert bundle.search_provenance.sources_returning == ["pubmed"]
    assert "openalex" in bundle.search_provenance.sources_enabled
    assert bundle.review.shortfalls and not bundle.review.targets_met
    assert bundle.search_provenance.search_queries["openalex"]


def test_cross_source_doi_dedup_keeps_impact_and_both_query_receipts():
    class Client:
        transport_failures = []

        def reset_transport_failures(self):
            pass

        def search_review(self, **_):
            return openalex.OpenAlexSearchResult(
                records=[record(2, key="openalex_W1", pmid=None, doi="https://doi.org/10.1234/DUP",
                                impact=ImpactProxies(source="openalex", citation_count=14, citations_per_year=7,
                                                     proxies_used=["citations_per_year"]))],
                search_queries=["second source query"], record_queries={"openalex_W1": ["second source query"]},
            )

    original = record(doi="10.1234/dup")
    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed([original]),
                                   enable_openalex=True, openalex_client=Client()).run(context())
    assert bundle.prisma == {"identified": 2, "screened": 1, "eligible": 1, "included": 1, "duplicates_removed": 1}
    assert len(bundle.screening_decisions) == 1 and len(bundle.design_evidence_cards) == 1
    merged = next(row for row in bundle.citations if row.key == original.key)
    assert merged.impact.citation_count == 14
    assert len(bundle.search_provenance.record_queries[original.key]) == 2


def test_missing_source_metrics_and_malformed_work_are_disclosed(monkeypatch):
    client = openalex.OpenAlexLiteratureClient(pace=lambda: None)

    def get(path, params):
        if path.startswith("sources/"):
            return {}
        return {"results": [work(), {"id": "invalid"}]}

    monkeypatch.setattr(client, "_get_json", get)
    result = client.search_review(families={"topic_gap": ["synthetic"]}, policy=ReviewPolicy())
    assert len(result.records) == 1
    assert result.records[0].impact.venue_h_index is None
    assert "venue_h_index" not in result.records[0].impact.proxies_used
    assert any("summary_stats" in reason for reason in client.transport_failures)
    assert any("malformed record" in reason for reason in client.transport_failures)


def test_openalex_client_pacing_is_shared_across_instances(monkeypatch):
    now, starts = [0.0], []
    monkeypatch.setattr(openalex, "_next_request", 0.0)
    monkeypatch.setattr(openalex, "monotonic", lambda: now[0])
    monkeypatch.setattr(openalex, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))

    def fail(*_, **__):
        starts.append(now[0])
        raise TimeoutError("synthetic")

    monkeypatch.setattr(openalex.urllib.request, "urlopen", fail)
    for client in [openalex.OpenAlexLiteratureClient(), openalex.OpenAlexLiteratureClient()]:
        assert client._get_json("works", {}) is None
    assert starts[1] - starts[0] >= 0.35


def test_preplan_enables_two_sources_when_live_retrieval_is_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(owner.PubMedLiteratureClient, "search_review_context", lambda *_, **__: owner.PubMedContextSearchResult())

    def search(self, **kwargs):
        calls.append(kwargs["families"])
        return openalex.OpenAlexSearchResult()

    monkeypatch.setattr(owner.OpenAlexLiteratureClient, "search_review", search)
    bundle = owner.build_preplan_literature_bundle(context(), enable_pubmed=True)
    assert bundle.search_provenance.sources_enabled == ["pubmed", "openalex"]
    assert len(calls) == 1
    assert bundle.prisma["screened"] == 0 and bundle.review.shortfalls
