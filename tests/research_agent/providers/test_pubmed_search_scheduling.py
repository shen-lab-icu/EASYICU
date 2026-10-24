"""Independent synthetic searches overlap without changing retrieval lineage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event, Lock
from types import SimpleNamespace
import urllib.parse

import pytest

from easyicu.research_agent import literature


def test_strata_overlap_but_hydrate_once_in_query_order(monkeypatch):
    queries = [
        "synthetic alpha",
        "synthetic beta",
        "synthetic gamma",
        "synthetic delta",
    ]
    ids = [["11", "44"], ["22", "44"], ["33"], []]
    client = literature.PubMedLiteratureClient()
    context = SimpleNamespace(
        primary_exposure=None, target_outcome=None, research_question="Synthetic counts"
    )
    monkeypatch.setattr(
        literature, "build_pubmed_protocol_queries_for_context", lambda _: queries
    )
    monkeypatch.setattr(
        literature, "_rank_protocol_search_results", lambda _, records: records
    )
    monkeypatch.setattr(literature, "_variable_focus_terms", lambda *_: ())
    second_started = Event()
    first_finished = Event()
    overlap = []
    hydrated = []
    pool_limits = []

    def pool(*args, **kwargs):
        pool_limits.append(kwargs["max_workers"])
        return ThreadPoolExecutor(*args, **kwargs)

    # Pin the resource ceiling as well as observing actual overlapping work.
    monkeypatch.setattr(literature, "ThreadPoolExecutor", pool, raising=False)

    def search(query, *, retmax):
        assert retmax == 20
        index = queries.index(query)
        if index == 0:
            overlap.append(second_started.wait(1))
            first_finished.set()
        if index == 1:
            second_started.set()
        return ids[index]

    def hydrate(pmids, **_kwargs):
        assert first_finished.is_set()
        hydrated.append(list(pmids))
        return [
            literature.CitationRecord(
                key=f"paper_{pmid}", pmid=pmid, title="Synthetic study", year="2020"
            )
            for pmid in pmids
        ]

    monkeypatch.setattr(client, "_esearch", search)
    monkeypatch.setattr(client, "_hydrate_ids", hydrate)
    result = client.search_context_strata(context, retmax=5)

    assert overlap == [True], "independent ESearch requests still run serially"
    assert pool_limits == [2]
    assert hydrated == [["11", "22", "33", "44"]]
    assert result.search_queries == queries
    assert [record.pmid for record in result.records] == ["11", "22", "33", "44"]
    assert result.record_queries == {
        "paper_11": queries[:1],
        "paper_22": queries[1:2],
        "paper_33": queries[2:3],
        "paper_44": queries[:2],
    }


def test_request_pacing_is_shared_across_clients_and_keeps_failures(monkeypatch):
    now = [0.0]
    starts = []
    lock = Lock()

    def clock():
        return now[0]

    def sleep(seconds):
        assert seconds >= 0
        now[0] += seconds

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b"{}"

    def urlopen(url, *, timeout):
        with lock:
            starts.append(now[0])
        assert timeout == 15
        if "fail.fcgi" in url:
            raise TimeoutError("synthetic timeout")
        return Response()

    # No real transport, clock delay, external data or credentials.
    monkeypatch.setattr(literature, "_pubmed_next_request_at", 0.0, raising=False)
    monkeypatch.setattr(literature, "monotonic", clock, raising=False)
    monkeypatch.setattr(literature, "sleep", sleep, raising=False)
    monkeypatch.setattr(literature.urllib.request, "urlopen", urlopen)
    clients = [literature.PubMedLiteratureClient(), literature.PubMedLiteratureClient()]
    assert clients[0]._http_get("esearch.fcgi", {}) == b"{}"
    assert clients[1]._http_get("fail.fcgi", {}) is None
    assert clients[0]._http_get("efetch.fcgi", {}) == b"{}"
    assert len(starts) == 3
    assert all(right - left >= 0.34 for left, right in zip(starts, starts[1:]))
    assert clients[0].transport_failures == []
    assert clients[1].transport_failures == ["fail.fcgi: the request timed out"]


@pytest.mark.parametrize(
    "queries", [[], ["synthetic empty"], ["synthetic timeout", "synthetic success"]]
)
def test_parallel_search_keeps_empty_results_and_failed_source_provenance(
    monkeypatch, queries
):
    client = literature.PubMedLiteratureClient()
    context = SimpleNamespace(
        primary_exposure=None, target_outcome=None, research_question="Synthetic counts"
    )
    monkeypatch.setattr(
        literature, "build_pubmed_protocol_queries_for_context", lambda _: queries
    )
    monkeypatch.setattr(
        literature, "_rank_protocol_search_results", lambda _, records: records
    )
    monkeypatch.setattr(literature, "_variable_focus_terms", lambda *_: ())
    monkeypatch.setattr(literature, "_pace_pubmed_request", lambda: None)
    hydrated = []

    class Response:
        def __init__(self, ids):
            self.ids = ids

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"esearchresult": {"idlist": self.ids}}).encode()

    def urlopen(url, **_kwargs):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["term"][0]
        if query == "synthetic timeout":
            raise TimeoutError("synthetic unavailable source")
        return Response(["77"] if query == "synthetic success" else [])

    def hydrate(pmids, **_kwargs):
        hydrated.append(list(pmids))
        return [
            literature.CitationRecord(
                key="synthetic_77", pmid="77", title="Synthetic study", year="2020"
            )
        ]

    monkeypatch.setattr(literature.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(client, "_hydrate_ids", hydrate)
    result = client.search_context_strata(context)
    assert result.search_queries == queries
    if "synthetic success" in queries:
        assert hydrated == [["77"]]
        assert result.record_queries == {"synthetic_77": ["synthetic success"]}
        assert client.transport_failures == ["esearch.fcgi: the request timed out"]
    else:
        assert result.records == []
        assert hydrated == []
        assert client.transport_failures == []
