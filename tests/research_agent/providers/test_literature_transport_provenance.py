"""Offline transport provenance distinguishes failed searches from empty results."""

from __future__ import annotations

import urllib.error

import pytest


# ---------------------------------------------------------------------------
# Transport failures must stay distinguishable from a genuinely empty search
# ---------------------------------------------------------------------------
#
# Both clients degrade to `[]` on purpose, so a literature lookup can never
# break an otherwise valid analysis. But the bare `except Exception: return
# None` made a network outage, a rejected key, a 429 and a query that
# genuinely matched nothing all indistinguishable — including to the PRISMA
# note, which then read "we searched and found nothing" when nothing was ever
# searched. Only the last of the four is a statement about the literature.


def _pubmed_client(ra):
    return ra.literature.PubMedLiteratureClient(email="a@b.c")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (urllib.error.HTTPError("u", 500, "boom", None, None), "HTTP 500"),
        (urllib.error.HTTPError("u", 429, "slow down", None, None), "rate limited"),
        (urllib.error.HTTPError("u", 401, "nope", None, None), "credentials"),
        (urllib.error.HTTPError("u", 403, "nope", None, None), "credentials"),
        (urllib.error.URLError("no route to host"), "unreachable"),
        (TimeoutError("too slow"), "timed out"),
        (ValueError("malformed"), "ValueError: malformed"),
    ],
)
def test_transport_failure_names_what_went_wrong(ra, exc, expected):
    client = _pubmed_client(ra)
    client.record_transport_failure("esearch.fcgi", exc)
    assert len(client.transport_failures) == 1
    assert expected in client.transport_failures[0]
    assert "esearch.fcgi" in client.transport_failures[0]


def test_transport_failures_start_and_reset_empty(ra):
    client = _pubmed_client(ra)
    assert client.transport_failures == []
    client.record_transport_failure("esearch.fcgi", ValueError("x"))
    client.reset_transport_failures()
    assert client.transport_failures == []


def test_tavily_without_a_key_says_no_search_was_issued(ra):
    client = ra.literature.TavilyLiteratureClient(api_key=None)
    client.api_key = None  # defeat the TAVILY_API_KEY environment fallback
    assert client.search("sepsis") == []
    assert "no Tavily API key" in client.transport_failures[0]


def test_provenance_separates_a_broken_source_from_an_empty_one(ra):
    """Absence from ``sources_returning`` cannot carry this on its own."""

    provenance = ra.literature.LiteratureSearchProvenance(
        curated_seed_count=0,
        sources_enabled=["pubmed", "tavily"],
        sources_returning=["tavily"],
        sources_failing={"pubmed": ["esearch.fcgi: rate limited by the source (HTTP 429)"]},
        search_conducted=True,
    )
    assert provenance.sources_failing["pubmed"]
    assert "tavily" not in provenance.sources_failing


def test_provenance_defaults_to_no_recorded_failures(ra):
    provenance = ra.literature.LiteratureSearchProvenance(
        curated_seed_count=4,
        search_conducted=False,
    )
    assert provenance.sources_failing == {}
