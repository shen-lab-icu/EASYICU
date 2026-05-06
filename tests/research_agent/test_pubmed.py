"""PubMed-side tests (T2.2) — offline only.

These cases exercise the *deterministic* parts of the
:class:`PubMedLiteratureClient`: query construction from a
:class:`ResearchContext` and JSON parsing of an ``esummary`` payload.
The HTTP round-trip itself is mocked so the suite never depends on
NCBI being reachable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def test_pubmed_query_includes_question_and_variables(ra):
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is admission SOFA-2 score associated with ICU mortality?",
        cohort=schema.CohortDescriptor(
            cohort_name="t", database="miiv", n_patients=10, n_stays=10),
        variables=[
            schema.ConceptDescriptor(name="age", role="demographic", dtype="float64"),
            schema.ConceptDescriptor(name="sofa2", role="composite_score", dtype="int64"),
            schema.ConceptDescriptor(name="lact", role="lab", dtype="float64"),
            schema.ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        ],
        target_outcome="death",
    )
    from easyicu.research_agent.literature import build_pubmed_query_for_context
    q = build_pubmed_query_for_context(ctx)
    # Question must appear without punctuation, surrounded by parens.
    assert "Is admission SOFA-2 score associated with ICU mortality" in q
    # Variables in scope (composite_score, lab, outcome) appear, demographic does not.
    assert "sofa2" in q and "lact" in q and "death" in q
    assert " age " not in f" {q} ", "demographic vars should be excluded"
    # ICU filter is always appended.
    assert "intensive care" in q.lower() or "icu" in q.lower()


def test_pubmed_query_skips_id_and_time_variables(ra):
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Question here",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="miiv", n_patients=1, n_stays=1),
        variables=[
            schema.ConceptDescriptor(name="stay_id", role="id", dtype="int64"),
            schema.ConceptDescriptor(name="intime", role="time", dtype="datetime64[ns]"),
            schema.ConceptDescriptor(name="vaso", role="intervention", dtype="int64"),
        ],
    )
    from easyicu.research_agent.literature import build_pubmed_query_for_context
    q = build_pubmed_query_for_context(ctx)
    assert "stay_id" not in q
    assert "intime" not in q
    assert "vaso" in q


def test_pubmed_query_caps_variable_count(ra):
    schema = ra.schema
    # Five ordinal-score variables — only the first four should make the query.
    ctx = schema.ResearchContext(
        research_question="Big question",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="miiv", n_patients=1, n_stays=1),
        variables=[
            schema.ConceptDescriptor(name=f"score{i}", role="composite_score", dtype="int64")
            for i in range(5)
        ],
    )
    from easyicu.research_agent.literature import build_pubmed_query_for_context
    q = build_pubmed_query_for_context(ctx)
    n_present = sum(f"score{i}" in q for i in range(5))
    assert n_present == 4, q


# ---------------------------------------------------------------------------
# esummary parser
# ---------------------------------------------------------------------------


def _fixture_payload() -> Dict[str, Any]:
    """A minimal but realistic esummary JSON payload (two records).

    Captured from the real NCBI shape and trimmed to the fields the
    parser actually consults — keeps the fixture readable and the
    test from drifting if NCBI adds more fields later.
    """
    return {
        "header": {"type": "esummary", "version": "0.3"},
        "result": {
            "uids": ["8844239", "26903338"],
            "8844239": {
                "uid": "8844239",
                "title": "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure.",
                "pubdate": "1996 Jul",
                "fulljournalname": "Intensive Care Medicine",
                "source": "Intensive Care Med",
                "authors": [
                    {"name": "Vincent JL", "authtype": "Author"},
                    {"name": "Moreno R", "authtype": "Author"},
                ],
                "articleids": [
                    {"idtype": "pubmed", "value": "8844239"},
                    {"idtype": "doi", "value": "10.1007/BF01709751"},
                ],
            },
            "26903338": {
                "uid": "26903338",
                "title": "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).",
                "pubdate": "2016 Feb 23",
                "fulljournalname": "JAMA",
                "authors": [
                    {"name": "Singer M", "authtype": "Author"},
                ],
                "articleids": [
                    {"idtype": "pubmed", "value": "26903338"},
                    {"idtype": "doi", "value": "10.1001/jama.2016.0287"},
                ],
            },
        },
    }


def test_parse_esummary_extracts_core_fields(ra):
    from easyicu.research_agent.literature import parse_pubmed_esummary
    records = parse_pubmed_esummary(_fixture_payload())
    assert len(records) == 2

    by_pmid = {r.pmid: r for r in records}
    sofa = by_pmid["8844239"]
    assert sofa.year == "1996"
    assert sofa.venue == "Intensive Care Medicine"
    assert sofa.doi == "10.1007/BF01709751"
    assert sofa.url and "8844239" in sofa.url
    assert "vincent" in sofa.key

    sep3 = by_pmid["26903338"]
    assert sep3.year == "2016"
    assert sep3.venue == "JAMA"
    assert "singer" in sep3.key


def test_parse_esummary_tolerates_missing_fields(ra):
    """Records with no authors / no DOI / weird pubdate must still parse."""
    from easyicu.research_agent.literature import parse_pubmed_esummary
    payload = {
        "result": {
            "uids": ["1", "2"],
            "1": {"uid": "1", "title": "Just a title.", "pubdate": "(unknown)"},
            "2": {"uid": "2"},  # virtually empty
        }
    }
    records = parse_pubmed_esummary(payload)
    assert len(records) == 2
    titles = {r.title for r in records}
    assert "Just a title" in titles
    # Year falls back to "n/a" when no 4-digit year is present.
    years = {r.year for r in records}
    assert "n/a" in years


def test_parse_esummary_empty_payload(ra):
    from easyicu.research_agent.literature import parse_pubmed_esummary
    assert parse_pubmed_esummary({}) == []
    assert parse_pubmed_esummary({"result": {}}) == []
    assert parse_pubmed_esummary({"result": {"uids": []}}) == []


# ---------------------------------------------------------------------------
# Client integration with a stub HTTP transport
# ---------------------------------------------------------------------------


class _StubClient:
    """A PubMed client whose HTTP layer is controlled by the test."""

    def __init__(self, ra, esearch_ids: List[str], esummary_payload: Dict[str, Any]):
        from easyicu.research_agent.literature import PubMedLiteratureClient
        self._client = PubMedLiteratureClient(timeout=1.0)
        self._esearch_ids = esearch_ids
        self._esummary_payload = esummary_payload
        self.calls: List[Dict[str, Any]] = []

        def _http_get(path: str, params: Dict[str, str]) -> Optional[bytes]:
            self.calls.append({"path": path, "params": dict(params)})
            if path == "esearch.fcgi":
                return json.dumps({"esearchresult": {"idlist": esearch_ids}}).encode()
            if path == "esummary.fcgi":
                return json.dumps(esummary_payload).encode()
            return None

        # monkey-patch the bound method
        self._client._http_get = _http_get  # type: ignore[attr-defined]

    @property
    def client(self):
        return self._client


def test_client_search_round_trip_with_stub(ra):
    """search() should chain esearch → esummary and return CitationRecords."""
    stub = _StubClient(ra, esearch_ids=["8844239"], esummary_payload=_fixture_payload())
    records = stub.client.search("sofa AND ICU", retmax=5)
    assert len(records) == 2  # the fixture has two uids
    paths = [c["path"] for c in stub.calls]
    assert paths[:2] == ["esearch.fcgi", "esummary.fcgi"]
    # NCBI etiquette parameters
    assert stub.calls[0]["params"].get("tool")
    assert stub.calls[0]["params"].get("term") == "sofa AND ICU"


def test_client_search_returns_empty_on_no_hits(ra):
    stub = _StubClient(ra, esearch_ids=[], esummary_payload={})
    out = stub.client.search("query", retmax=5)
    assert out == []
    # esummary is not called when esearch returned no ids.
    assert [c["path"] for c in stub.calls] == ["esearch.fcgi"]


def test_client_search_swallows_network_failure(ra):
    """A None return from _http_get must yield an empty list, not an exception."""
    from easyicu.research_agent.literature import PubMedLiteratureClient
    client = PubMedLiteratureClient(timeout=1.0)
    client._http_get = lambda path, params: None  # type: ignore[attr-defined]
    assert client.search("anything") == []
    assert client.search_for_context.__doc__ is not None  # sanity


# ---------------------------------------------------------------------------
# LiteratureAgent integration
# ---------------------------------------------------------------------------


def test_literature_agent_merges_pubmed_with_curated(ra):
    """When enable_pubmed=True with a stub client, the bundle should
    contain both curated entries and the stub's hits."""
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="miiv", n_patients=10, n_stays=10),
        variables=[
            schema.ConceptDescriptor(name="sofa2", role="composite_score", dtype="int64"),
            schema.ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        ],
        target_outcome="death",
    )
    stub = _StubClient(ra, esearch_ids=["8844239", "26903338"],
                       esummary_payload=_fixture_payload())

    from easyicu.research_agent.literature import LiteratureAgent
    agent = LiteratureAgent(
        llm=None, enable_pubmed=True, pubmed_client=stub.client, pubmed_retmax=5,
    )
    bundle = agent.run(ctx)
    keys = {c.key for c in bundle.citations}

    # Curated entries that match the context should still be there.
    assert "vincent_sofa_1996" in keys, keys
    # PubMed hit for SOFA shares the same PMID (8844239) — dedup must
    # NOT inject a duplicate (curated already has pmid=8844239).
    pmids = [c.pmid for c in bundle.citations if c.pmid == "8844239"]
    assert len(pmids) == 1, f"duplicate SOFA PMID: {pmids}"
    # Sepsis-3 PMID may or may not already be curated (it is — pmid 26903338);
    # the dedup keeps exactly one regardless.
    pmids2 = [c.pmid for c in bundle.citations if c.pmid == "26903338"]
    assert len(pmids2) <= 1


def test_literature_agent_pubmed_failure_is_silent(ra):
    """A PubMed client that raises must not break the agent run —
    the curated baseline still comes back."""
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="x",
        cohort=schema.CohortDescriptor(cohort_name="c", database="miiv",
                                       n_patients=1, n_stays=1),
        variables=[
            schema.ConceptDescriptor(name="sofa2", role="composite_score", dtype="int64"),
        ],
    )

    class _Boom:
        def search_for_context(self, *a, **kw):
            raise RuntimeError("network down")

    from easyicu.research_agent.literature import LiteratureAgent
    bundle = LiteratureAgent(
        llm=None, enable_pubmed=True, pubmed_client=_Boom(),
    ).run(ctx)
    # Curated baseline is intact.
    assert any(c.key == "vincent_sofa_1996" for c in bundle.citations)


# ---------------------------------------------------------------------------
# Tavily live client (O5) — offline transport stubs
# ---------------------------------------------------------------------------


def test_parse_tavily_search_response_extracts_web_records(ra):
    from easyicu.research_agent.literature import parse_tavily_search_response

    payload = {
        "results": [
            {
                "title": "Surviving Sepsis Campaign Guidelines 2021",
                "url": "https://www.sccm.org/guidelines/sepsis-guidelines",
                "content": "International guideline for sepsis and septic shock.",
                "score": 0.9,
            }
        ]
    }
    records = parse_tavily_search_response(payload)
    assert len(records) == 1
    rec = records[0]
    assert rec.year == "2021"
    assert rec.venue == "sccm.org"
    assert rec.url == "https://www.sccm.org/guidelines/sepsis-guidelines"
    assert rec.key.startswith("tavily_surviving_2021_")


def test_tavily_client_posts_required_search_knobs(ra):
    from easyicu.research_agent.literature import TavilyLiteratureClient

    calls = []
    client = TavilyLiteratureClient(api_key="tvly-test", timeout=1.0)

    def _http_post(path, payload):
        calls.append({"path": path, "payload": dict(payload)})
        return json.dumps({
            "results": [
                {
                    "title": "Trial registry entry for ICU vasopressors",
                    "url": "https://clinicaltrials.gov/study/NCT00000000",
                    "content": "A registry record.",
                }
            ]
        }).encode()

    client._http_post = _http_post  # type: ignore[attr-defined]
    out = client.search("vasopressor ICU trial", max_results=3)

    assert len(out) == 1
    assert calls[0]["path"] == "search"
    payload = calls[0]["payload"]
    assert payload["include_answer"] is False
    assert payload["include_raw_content"] is False
    assert payload["max_results"] == 3
    assert payload["search_depth"] == "basic"


def test_literature_agent_merges_tavily_with_curated(ra):
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="miiv", n_patients=10, n_stays=10),
        variables=[
            schema.ConceptDescriptor(name="sofa2", role="composite_score", dtype="int64"),
            schema.ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        ],
        target_outcome="death",
    )

    class _Tavily:
        def search_for_context(self, context, *, max_results=5):
            from easyicu.research_agent.literature import CitationRecord
            return [
                CitationRecord(
                    key="tavily_guideline_2021_deadbeef",
                    title="ICU guideline outside PubMed",
                    year="2021",
                    venue="guidelines.example",
                    url="https://guidelines.example/icu",
                )
            ]

    from easyicu.research_agent.literature import LiteratureAgent

    bundle = LiteratureAgent(
        llm=None,
        enable_tavily=True,
        tavily_client=_Tavily(),
        tavily_retmax=2,
    ).run(ctx)
    keys = {c.key for c in bundle.citations}
    assert "vincent_sofa_1996" in keys
    assert "tavily_guideline_2021_deadbeef" in keys
