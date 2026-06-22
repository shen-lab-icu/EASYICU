from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
from types import ModuleType

from easyicu.research_agent.idea_mining_funnel import (
    LiteratureFunnelSpec,
    build_literature_funnel_queries,
)


def _load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "mine_discovery_gpt54.py"
    spec = importlib.util.spec_from_file_location("mine_discovery_gpt54", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(**kwargs) -> Namespace:
    defaults = {
        "scope": "criticalcare",
        "funnel_adult_icu_filter": None,
        "funnel_ehr_actionable_filter": None,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_criticalcare_defaults_to_adult_icu_and_ehr_actionable_filters() -> None:
    runner = _load_runner()

    terms, report = runner._combined_filter_terms(_args())

    assert report["adult_icu"] is True
    assert report["ehr_actionable"] is True
    assert terms is not None
    assert "adult[Title/Abstract]" in terms
    assert "pediatric[Title/Abstract]" in terms
    assert "mechanical ventilation" in terms
    assert "modified Rankin" in terms


def test_funnel_filters_can_be_disabled_explicitly() -> None:
    runner = _load_runner()

    terms, report = runner._combined_filter_terms(
        _args(funnel_adult_icu_filter=False, funnel_ehr_actionable_filter=False)
    )

    assert terms is None
    assert report["adult_icu"] is False
    assert report["ehr_actionable"] is False


def test_filter_terms_are_inherited_by_all_literature_routes() -> None:
    runner = _load_runner()
    terms, _report = runner._combined_filter_terms(_args())
    scope = runner._funnel_scope_for_preset(
        "criticalcare",
        last_n_years=2,
        journal_preset="critical_care_specialty_wide",
        extra_terms=terms,
    )

    routes = build_literature_funnel_queries(
        LiteratureFunnelSpec(base_scope=scope),
        reference_year=2026,
    )

    assert {route.route_name for route in routes} == {
        "review_gap",
        "primary_limitation",
        "platform_gap",
    }
    for route in routes:
        assert "adult[Title/Abstract]" in route.pubmed_query
        assert "pediatric[Title/Abstract]" in route.pubmed_query
        assert "mechanical ventilation" in route.pubmed_query
        assert "modified Rankin" in route.pubmed_query


def test_long_pubmed_query_retries_with_post(monkeypatch) -> None:
    runner = _load_runner()
    calls: list[tuple[str, int]] = []

    def fake_get(query: str, *, retmax: int):
        raise AssertionError("long queries should use POST without trying GET first")

    def fake_post(query: str, *, retmax: int):
        calls.append((query, retmax))
        return {"count": 1, "pmids": ["123"]}

    monkeypatch.setattr(runner.H, "pubmed_search", fake_get)
    monkeypatch.setattr(runner, "_pubmed_search_post", fake_post)

    query = "x" * 2000
    assert runner._pubmed_search_robust(query, retmax=7) == {
        "count": 1,
        "pmids": ["123"],
    }
    assert calls == [(query, 7)]
