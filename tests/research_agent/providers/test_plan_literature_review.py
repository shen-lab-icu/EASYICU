"""Plan review depth and scientific scope, using only synthetic bibliography."""

from datetime import datetime, timezone

import pytest

from easyicu.research_agent import literature as owner
from tests.support.plan_literature import FakePubMed, context, record
from easyicu.research_agent.planning.literature_contract import ImpactProxies, ReviewPolicy, REVIEW_ROLES
from easyicu.research_agent.planning.literature_design_authority import (
    LiteratureDesignAuthorityError, render_literature_design_cards_for_prompt,
    validate_preplan_literature_design_authority,
)


@pytest.mark.parametrize("kind,term", [("lab", "creatinine"), ("drug", "norepinephrine")])
def test_question_and_concept_names_win_over_unrelated_description(kind, term):
    ctx = context(kind)
    assert owner._protocol_search_term(ctx, ctx.primary_exposure) == term
    queries = owner.build_pubmed_protocol_queries_for_context(ctx)
    assert all("unrelated storage duration" not in query for query in queries)
    assert any(term in query for query in queries)


def test_stated_endpoint_scope_precedes_a_stale_variable_description():
    ctx = context()
    ctx = ctx.model_copy(update={
        "research_question": "Does creatinine predict hospital mortality in adult ICU patients?",
        "variables": [ctx.variables[0], ctx.variables[1].model_copy(update={"description": "ICU mortality"})],
    })
    assert owner._protocol_search_term(ctx, ctx.target_outcome) == "hospital mortality"
    assert '"hospital mortality"[Title/Abstract]' in owner.build_pubmed_protocol_query_for_context(ctx)


@pytest.mark.parametrize("kind", ["lab", "drug"])
def test_default_review_screens_enough_candidates_before_eligibility(kind):
    bundle = owner.LiteratureAgent(
        enable_pubmed=True,
        pubmed_client=FakePubMed([record(i, kind) for i in range(90)]),
    ).run(context(kind))
    assert bundle.prisma["screened"] >= 75
    assert bundle.prisma["included"] >= 8
    assert len(bundle.screening_decisions) == bundle.prisma["screened"]
    assert bundle.review.role_counts["topic_gap"] == 90
    assert bundle.review.role_counts["analysis_method"] == 90
    assert bundle.search_provenance.review_policy.screening_target == 75
    assert bundle.review.shortfalls
    assert not bundle.review.targets_met


@pytest.mark.parametrize("kind", ["lab", "drug"])
def test_real_pubmed_adapter_searches_every_role_and_screens_the_hydrated_pool(monkeypatch, kind):
    client = owner.PubMedLiteratureClient()
    calls, hydrated = [], []

    def esearch(query, *, retmax):
        calls.append(query)
        assert retmax == 25
        return [str(10000 + len(calls) * 100 + index) for index in range(retmax)]

    def hydrate(ids, *, excerpt_terms):
        hydrated.append(ids)
        return [record(int(pmid), kind, pmid=pmid) for pmid in ids]

    monkeypatch.setattr(client, "_esearch", esearch)
    monkeypatch.setattr(client, "_hydrate_ids", hydrate)
    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=client).run(context(kind))
    assert len(hydrated) == 1 and len(hydrated[0]) == 100
    assert bundle.prisma == {"identified": 100, "screened": 100, "eligible": 100,
                             "included": 100, "duplicates_removed": 0}
    provenance = bundle.search_provenance
    assert set(provenance.query_roles["pubmed"]) == set(REVIEW_ROLES)
    assert provenance.search_queries["pubmed"] == calls
    assert all(provenance.record_queries[d.citation_key] for d in bundle.screening_decisions)
    assert all("unrelated storage duration" not in query for query in calls)
    assert any("[MeSH Terms]" in query for query in calls)
    assert any(('"Creatinine"[MeSH Terms]' if kind == "lab" else '"Norepinephrine"[MeSH Terms]') in query for query in calls)
    if kind == "drug":
        assert any("noradrenaline" in query for query in calls)


def test_coverage_excludes_seeds_old_topic_evidence_and_unknown_dates():
    rows = [record(1), record(2, year="1995"), record(3, year=""), record(4, year="2999")]
    policy = ReviewPolicy(screening_target=6, role_targets={"topic_gap": 3, "analysis_method": 5})
    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed(rows), review_policy=policy).run(context())
    assert bundle.review.role_counts["topic_gap"] == 1
    assert bundle.review.role_counts["data_source"] == 1
    assert bundle.review.role_counts["analysis_method"] == 3
    assert {(s.role, s.target, s.actual, s.missing) for s in bundle.review.shortfalls} == {
        ("screening", 6, 4, 2), ("topic_gap", 3, 1, 2), ("analysis_method", 5, 3, 2),
    }
    reviewed = {row.key: row.review for row in bundle.citations if row.review}
    assert reviewed[rows[1].key].recency == "method_classic"
    assert reviewed[rows[2].key].recency == "unknown"
    assert reviewed[rows[3].key].recency == "future"


def test_irrelevant_or_retracted_high_impact_records_stay_excluded():
    unrelated = record(1, title="Orchard crop yield", venue="Nature Medicine",
                       relevance="Study-design excerpt: We measured rainfall in apple orchards.",
                       impact=ImpactProxies(source="openalex", citation_count=99999, citations_per_year=99999,
                                            proxies_used=["citations_per_year"]))
    retracted = record(2, bibliographic_notices=["Retracted"])
    valid = record(3, venue="Critical Care")
    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed([unrelated, retracted, valid])).run(context())
    assert {d.citation_key for d in bundle.screening_decisions if d.disposition == "include"} == {valid.key}
    assert bundle.prisma["included"] == 1
    assert {card.citation_key for card in bundle.design_evidence_cards} == {valid.key}
    ranked = owner._rank_protocol_search_results(context(), [unrelated, valid])
    assert ranked[0].key == valid.key
    included = next(row for row in bundle.citations if row.key == valid.key)
    assert included.impact.curated_top_venue
    assert included.impact.proxies_used == ["curated_top_venue"]


def test_recency_then_impact_break_relevance_ties():
    year = datetime.now(timezone.utc).year
    recent = record(1, title="Creatinine and mortality in adult ICU patients", year=str(year))
    old = record(2, title=recent.title, year="1995", impact=ImpactProxies(citations_per_year=999))
    influential = recent.model_copy(update={"key": "influential", "impact": ImpactProxies(citations_per_year=20)})
    assert [row.key for row in owner._rank_protocol_search_results(context(), [old, recent, influential])] == [
        influential.key, recent.key, old.key,
    ]


def test_abstract_cards_quote_sources_group_roles_and_cannot_claim_fulltext():
    from pydantic import ValidationError
    from easyicu.research_agent.planning.literature_design_authority import LiteratureDesignEvidenceCard

    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed([record(i) for i in range(90)])).run(context())
    assert len(bundle.design_evidence_cards) == 90
    records = {row.key: row for row in bundle.citations}
    for card in bundle.design_evidence_cards:
        assert card.access_mode == "abstract" and card.full_text_sha256 is None
        assert card.abstract_sha256 and card.abstract_locator
        assert all(item.source_backed_summary in records[card.citation_key].relevance for item in card.evidence)
    prompt = render_literature_design_cards_for_prompt(bundle.design_evidence_cards)
    assert "analysis_method" in prompt and "definition" in prompt and "adjustment" in prompt
    assert "no full-text review" in prompt
    assert len(prompt) < 5000
    with pytest.raises(LiteratureDesignAuthorityError) as failure:
        validate_preplan_literature_design_authority(bundle)
    assert failure.value.reason_code == "literature_fulltext_design_card_missing"
    payload = bundle.design_evidence_cards[0].model_dump(mode="json")
    with pytest.raises(ValidationError):
        LiteratureDesignEvidenceCard.model_validate({**payload, "full_text_sha256": "f" * 64})


def test_guideline_and_general_method_have_specific_roles_without_comparator_authority():
    guideline = record(1, title="Reporting guidelines for prediction models", publication_types=["Guideline"],
                       relevance="Study-design excerpt: These reporting guidelines describe prediction model validation and reporting standards.")
    method = record(2, title="Statistical methods for prediction validation", year="1995",
                    relevance="Study-design excerpt: We propose validation methods for prediction and calibration.")
    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed([guideline, method])).run(context())
    assert {d.evidence_role for d in bundle.screening_decisions} == {"method"}
    assert bundle.review.role_counts["reporting_guideline"] == 1
    assert bundle.review.role_counts["analysis_method"] == 1
    assert bundle.review.role_counts["topic_gap"] == 0
    assert bundle.prisma["included"] == 2


@pytest.mark.parametrize("relevance", [None, record().relevance.removeprefix("Study-design excerpt: ")])
def test_missing_abstract_does_not_get_a_card_or_count_toward_a_role(relevance):
    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed([record(relevance=relevance)])).run(context())
    assert bundle.prisma["included"] == 0
    assert not bundle.design_evidence_cards
    assert not any(bundle.review.role_counts.values())


@pytest.mark.parametrize("body,path", [(b"{", "esearch.fcgi"), (b"{}", "esummary.fcgi"), (b"broken xml", "efetch.fcgi")])
def test_malformed_pubmed_responses_are_recorded(monkeypatch, body, path):
    client = owner.PubMedLiteratureClient()
    monkeypatch.setattr(client, "_http_get", lambda *_: body)
    if path == "esearch.fcgi":
        assert client._esearch("synthetic", retmax=1) == []
    elif path == "esummary.fcgi":
        assert client._esummary(["1"]) == []
    else:
        assert client._protocol_article_metadata(["1"]) == {}
    assert any(path in failure for failure in client.transport_failures)


def test_projection_retains_typed_coverage_policy_and_record_quality():
    from easyicu.webserver.literature_projection import project_run_literature

    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed([record(venue="JAMA")])).run(context())
    view = project_run_literature(run_id="synthetic", bundle=bundle.model_dump(mode="json"), plan={})
    assert view["review"] == bundle.review.model_dump(mode="json")
    assert view["search"]["review_policy"] == bundle.search_provenance.review_policy.model_dump(mode="json")
    row = next(row for row in view["citations"] if row["key"] == record().key)
    assert row["impact"]["proxies_used"] == ["curated_top_venue"]
    assert row["review"]["recency"] == "recent"


def test_abstracts_reach_the_planner_prompt_without_triggering_fulltext_requirements():
    from easyicu.research_agent.planning.literature_design_authority import progressive_literature_design_kwargs

    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed([record(i) for i in range(90)])).run(context())
    kwargs = progressive_literature_design_kwargs(bundle)
    assert kwargs["literature_design_evidence_cards"] == []
    assert len(kwargs["comparison_literature_keys"]) == 90
    blueprint = owner.HypothesisBlueprintAgent().run(context=context(), literature=bundle)
    prompt = owner.render_hypothesis_blueprint_for_prompt(blueprint, literature=bundle)
    for role in ("topic_gap", "analysis_method", "definition", "adjustment", "data_source"):
        assert role in prompt
    assert "source excerpts" in prompt and "no full-text review" in prompt
    assert "literature_shortfalls: reporting_guideline 0/1" in prompt
    assert len(prompt) < 10000


def test_bound_abstracts_are_rescreened_and_keep_upstream_failure_receipts():
    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed([record()])).run(context())
    bundle.search_provenance.sources_failing = {"openalex": ["HTTP 429"]}
    # A stale abstract card is not human-reviewed design authority.
    bundle.citations = [row.model_copy(update={"title": "Orchard yields", "relevance": "Study-design excerpt: Rainfall in orchards."})
                        for row in bundle.citations if row.key == record().key]
    rebuilt = owner.LiteratureAgent(bound_seed=bundle).run(context())
    assert rebuilt.prisma["included"] == 0
    assert not rebuilt.design_evidence_cards
    assert rebuilt.search_provenance.sources_failing == {"openalex": ["HTTP 429"]}


def test_explicit_depth_is_validated_and_recorded_as_the_effective_policy():
    from pydantic import ValidationError

    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed([record(i) for i in range(90)]),
                                   pubmed_retmax=12).run(context())
    assert bundle.search_provenance.review_policy.candidates_per_source == 12
    assert bundle.prisma["screened"] == 12
    with pytest.raises(ValidationError):
        owner.LiteratureAgent(pubmed_retmax=0)
    with pytest.raises(ValidationError):
        ReviewPolicy(role_targets={"topic_gap": -1})


def test_recency_boundary_and_default_role_targets_are_explicit_shortfalls():
    year = datetime.now(timezone.utc).year
    rows = [record(1, year=str(year - 10)), record(2, year=str(year - 11))]
    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed(rows)).run(context())
    shortfalls = {s.role: s for s in bundle.review.shortfalls}
    assert shortfalls["topic_gap"].target == 8 and shortfalls["topic_gap"].actual == 1
    assert shortfalls["analysis_method"].target == 5 and shortfalls["analysis_method"].actual == 2
    assert shortfalls["data_source"].target == 3 and shortfalls["data_source"].actual == 1
    assert shortfalls["reporting_guideline"].target == 1


def test_guidance_for_a_different_design_does_not_satisfy_a_role():
    unrelated = record(1, title="Reporting guidelines for observational studies", publication_types=["Guideline"],
                       relevance="Study-design excerpt: Reporting guidelines for cohort studies specify confounding adjustment and regression reporting.")
    bundle = owner.LiteratureAgent(enable_pubmed=True, pubmed_client=FakePubMed([unrelated])).run(context("lab"))
    assert bundle.prisma["included"] == 0
    assert bundle.review.role_counts["reporting_guideline"] == 0
