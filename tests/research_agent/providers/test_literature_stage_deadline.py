"""The pre-plan literature stage has one wall-clock budget (pilot1 hang, 2026-11-16)."""

import time

from easyicu.research_agent import bibliographic_metadata as openalex
from easyicu.research_agent import literature as owner
from easyicu.research_agent.literature_deadline import LITERATURE_STAGE_SECONDS, request_timeout
from tests.support.plan_literature import FakePubMed, context, record


def test_a_spent_budget_makes_no_request_and_names_the_reason(monkeypatch):
    calls = []
    monkeypatch.setattr(owner.urllib.request, "urlopen", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(owner, "_pace_pubmed_request", lambda: None)
    client = owner.PubMedLiteratureClient()
    client.deadline = time.monotonic() - 1
    assert client._http_get("esearch.fcgi", {}) is None
    assert calls == []
    assert client.transport_failures == [
        "esearch.fcgi: LiteratureDeadlineExceeded: literature stage deadline of "
        f"{LITERATURE_STAGE_SECONDS:g} s reached before this request"
    ]


def test_the_request_timeout_never_outlasts_the_stage():
    assert request_timeout(None, 15.0) == 15.0
    assert 0 < request_timeout(time.monotonic() + 2.0, 15.0) <= 2.0
    assert request_timeout(time.monotonic() + 60.0, 15.0) == 15.0


def test_the_agent_bounds_a_stalling_source_and_reports_it(monkeypatch):
    def stall(url, *, timeout):
        time.sleep(timeout)
        raise TimeoutError("synthetic stall")

    monkeypatch.setattr(openalex.urllib.request, "urlopen", stall)
    client = openalex.OpenAlexLiteratureClient(pace=lambda: None)
    started = time.monotonic()
    bundle = owner.LiteratureAgent(
        enable_pubmed=True, pubmed_client=FakePubMed([record()]),
        enable_openalex=True, openalex_client=client, stage_seconds=0.3,
    ).run(context())
    assert time.monotonic() - started < 5.0  # Without the budget: 15 s per request.
    failures = bundle.search_provenance.sources_failing["openalex"]
    assert any("LiteratureDeadlineExceeded" in reason or "TimeoutError" in reason for reason in failures)
    assert bundle.search_provenance.sources_returning == ["pubmed"]
    assert bundle.review.shortfalls and not bundle.review.targets_met
