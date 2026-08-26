"""The pre-plan literature pack must carry methodology, and must not overclaim.

Two defects motivated these tests, both found by reading a real run's
``preplan_literature_bundle.json``:

1. The bundle reported a PRISMA flow (``identified 4 ... included 4``) on a run
   where no retrieval source was enabled.  That reads as a systematic search
   that found four papers; it was four preset references passing through.
2. Every curated reference was topic or data-source -- what Sepsis-3 is, what
   MIMIC-IV is.  Nothing told the Planner how an observational study is
   designed, yet it was asked to choose follow-up start, repeated-stay
   handling, functional form and missing-data policy.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.research_agent.literature import (
    LiteratureBundle,
    build_preplan_literature_bundle,
)
from easyicu.research_agent.planning.method_literature import (
    METHOD_CARDS,
    MethodCard,
    method_cards_for_layers,
    method_literature_citations,
    method_literature_digest,
    method_literature_pack,
    method_binding_support,
    reporting_method_source_keys_for_guidelines,
)
from easyicu.research_agent.planning.scientific_review import (
    required_method_layers_for_plan,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ResearchContext


def _context(**overrides: Any) -> ResearchContext:
    payload: dict[str, Any] = {
        "research_question": "Estimate an association in an ICU cohort.",
        "cohort": {
            "cohort_name": "icu_stays",
            "database": "miiv",
            "n_stays": 1000,
        },
        "variables": [],
    }
    payload.update(overrides)
    return ResearchContext.model_validate(payload)


def test_a_curated_only_bundle_reports_no_search_instead_of_a_prisma_flow() -> None:
    bundle = build_preplan_literature_bundle(_context())

    assert bundle.prisma is None
    provenance = bundle.search_provenance
    assert provenance is not None
    assert provenance.search_conducted is False
    assert provenance.sources_enabled == []
    assert provenance.curated_seed_count == len(bundle.citations)
    assert "no search was performed" in provenance.note


def test_an_enabled_source_still_gets_a_real_prisma_flow() -> None:
    """The honesty fix must not disable reporting for runs that did search."""

    class _Client:
        def search_for_context(self, context: Any, retmax: int = 5) -> list[Any]:
            return []

    bundle = build_preplan_literature_bundle(
        _context(),
        enable_pubmed=True,
        pubmed_email=None,
    )

    assert bundle.search_provenance is not None
    assert bundle.search_provenance.search_conducted is True
    assert "pubmed" in bundle.search_provenance.sources_enabled
    assert bundle.search_provenance.search_queries["pubmed"]
    assert isinstance(bundle.prisma, dict)
    assert set(bundle.prisma) >= {"identified", "screened", "included"}


def test_a_source_that_returns_nothing_is_not_recorded_as_returning() -> None:
    """Enabled and productive are different claims about the same run."""

    class _EmptyClient:
        def search_for_context(self, context: Any, **kwargs: Any) -> list[Any]:
            return []

    bundle = build_preplan_literature_bundle(
        _context(),
        enable_tavily=True,
        tavily_api_key=None,
    )

    provenance = bundle.search_provenance
    assert provenance is not None
    assert "tavily" in provenance.sources_enabled
    assert "tavily" not in provenance.sources_returning


def test_digest_bound_web_search_seed_is_available_to_preplan_planning() -> None:
    context = _context(
        research_question=(
            "Is peak lactate during the first 24 ICU hours associated with "
            "hospital mortality?"
        ),
        variables=[
            {
                "name": "lact_max",
                "description": "lactate",
                "role": "lab",
                "dtype": "float64",
                "source_concept": "lact",
                "analysis_window": "icu_admission[0,24]h",
            },
            {
                "name": "death",
                "description": "hospital mortality",
                "role": "outcome",
                "dtype": "int64",
                "source_concept": "hospital_mortality",
            },
        ],
        primary_exposure="lact_max",
        target_outcome="death",
    )
    seed = LiteratureBundle.model_validate(
        {
            "research_question": "Question",
            "citations": [
                {
                    "key": "idea_pubmed_lactate_mortality_2024",
                    "title": (
                        "Early lactate and hospital mortality in intensive care"
                    ),
                    "year": "2024",
                    "venue": "Critical Care",
                    "pmid": "12345678",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    "relevance": (
                        "Source excerpt: In adult ICU patients, peak lactate "
                        "during the first 24 hours was evaluated for association "
                        "with in-hospital mortality."
                    ),
                }
            ],
            "prisma": {
                "identified": 1,
                "duplicates_removed": 0,
                "screened": 1,
                "eligible": 1,
                "included": 1,
            },
            "search_provenance": {
                "curated_seed_count": 0,
                "sources_enabled": ["idea_mining_pubmed"],
                "sources_returning": ["idea_mining_pubmed"],
                "search_queries": {
                    "idea_mining_pubmed": [
                        '"lactate"[Title/Abstract] AND "hospital mortality"[Title/Abstract]'
                    ]
                },
                "record_queries": {
                    "idea_pubmed_lactate_mortality_2024": [
                        '"lactate"[Title/Abstract] AND "hospital mortality"[Title/Abstract]'
                    ]
                },
                "search_conducted": True,
                "searched_at": "2026-08-12T12:00:00+00:00",
                "note": "Digest-bound Web search.",
            },
            "authority_trace": {
                "schema_version": "easyicu.web-literature-authority/3",
                "receipt_id": "lit_" + "a" * 24,
                "receipt_sha256": "b" * 64,
                "study_context_id": "study-e1",
                "study_context_revision": 7,
                "retrieval_scope_sha256": "c" * 64,
            },
            "screening_decisions": [
                {
                    "citation_key": "idea_pubmed_lactate_mortality_2024",
                    "source": "idea_mining_pubmed",
                    "disposition": "exclude",
                    "evidence_role": "related_context",
                    "rationale": (
                        "Upstream Idea-level screen is provisional; the exact "
                        "ResearchContext must decide comparator authority."
                    ),
                }
            ],
        }
    )

    bundle = build_preplan_literature_bundle(context, bound_seed=seed)

    assert "idea_pubmed_lactate_mortality_2024" in {
        row.key for row in bundle.citations
    }
    decision = next(
        row
        for row in bundle.screening_decisions
        if row.citation_key == "idea_pubmed_lactate_mortality_2024"
    )
    assert decision.disposition == "include"
    assert decision.evidence_role == "direct_comparator"
    assert bundle.search_provenance is not None
    assert bundle.search_provenance.search_conducted is True
    assert "idea_mining_pubmed" in bundle.search_provenance.sources_enabled
    assert bundle.search_provenance.search_queries["idea_mining_pubmed"]
    assert (
        decision.query
        == '"lactate"[Title/Abstract] AND "hospital mortality"[Title/Abstract]'
    )
    assert bundle.prisma is not None
    assert bundle.prisma["identified"] == 1
    assert bundle.prisma["included"] == 1
    assert bundle.authority_trace == seed.authority_trace


def test_bound_seed_upstream_include_cannot_promote_an_irrelevant_record() -> None:
    seed = LiteratureBundle.model_validate(
        {
            "research_question": "Broad idea",
            "citations": [
                {
                    "key": "irrelevant_but_premarked",
                    "title": "Nutrition in postoperative wards",
                    "year": "2025",
                    "relevance": (
                        "Source excerpt: Adults were enrolled after elective surgery."
                    ),
                }
            ],
            "search_provenance": {
                "curated_seed_count": 0,
                "sources_enabled": ["idea_mining_pubmed"],
                "sources_returning": ["idea_mining_pubmed"],
                "search_queries": {"idea_mining_pubmed": ["broad query"]},
                "search_conducted": True,
                "searched_at": "2026-08-12T12:00:00+00:00",
            },
            "screening_decisions": [
                {
                    "citation_key": "irrelevant_but_premarked",
                    "source": "idea_mining_pubmed",
                    "disposition": "include",
                    "evidence_role": "direct_comparator",
                    "rationale": "Provisional upstream label.",
                }
            ],
        }
    )

    bundle = build_preplan_literature_bundle(_context(), bound_seed=seed)

    assert "irrelevant_but_premarked" in {
        row.key for row in bundle.citations
    }
    decision = next(
        row
        for row in bundle.screening_decisions
        if row.citation_key == "irrelevant_but_premarked"
    )
    assert decision.disposition == "exclude"


def test_exact_question_reviewed_design_analogue_keeps_fulltext_decision() -> None:
    context = _context()
    citation_key = "reviewed_design_analogue"
    seed = LiteratureBundle.model_validate(
        {
            "research_question": context.research_question,
            "citations": [
                {
                    "key": citation_key,
                    "title": "A reviewed methodological analogue",
                    "year": "2025",
                    "pmid": "12345678",
                }
            ],
            "search_provenance": {
                "curated_seed_count": 0,
                "sources_enabled": ["pubmed"],
                "sources_returning": ["pubmed"],
                "search_queries": {"pubmed": ["verified query"]},
                "record_queries": {citation_key: ["12345678[PMID]"]},
                "search_conducted": True,
                "searched_at": "2026-08-25T12:00:00+00:00",
            },
            "screening_decisions": [
                {
                    "citation_key": citation_key,
                    "source": "pubmed",
                    "disposition": "include",
                    "evidence_role": "design_analogue",
                    "rationale": (
                        "Full text provides a reviewed design pattern for the exact question."
                    ),
                    "design_excerpt_available": True,
                }
            ],
            "design_evidence_cards": [
                {
                    "citation_key": citation_key,
                    "evidence_role": "design_analogue",
                    "access_mode": "open_access_fulltext",
                    "full_text_locator": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
                    "full_text_sha256": "a" * 64,
                    "supplement_status": "not_published",
                    "reviewed_at": "2026-08-25T12:00:00+00:00",
                    "evidence": [
                        {
                            "dimension": "primary_model_and_sensitivities",
                            "source_backed_summary": (
                                "The source describes a relevant methodological pattern."
                            ),
                        }
                    ],
                }
            ],
        }
    )

    bundle = build_preplan_literature_bundle(context, bound_seed=seed)

    decision = next(
        row for row in bundle.screening_decisions if row.citation_key == citation_key
    )
    assert decision.disposition == "include"
    assert decision.evidence_role == "design_analogue"


def test_the_methodology_layer_reaches_every_study() -> None:
    """Design guidance is not conditional on which concepts are in scope.

    The context here declares no variables at all, so every topic-triggered
    citation is absent.  The methodology sources must still be there: how to
    align time, what to do about repeated units and missing data are questions
    this study faces regardless of what it is measuring.
    """

    bundle = build_preplan_literature_bundle(_context())
    keys = {citation.key for citation in bundle.citations}

    assert {source["key"] for source in method_literature_citations()} <= keys


@pytest.mark.parametrize(
    "layer",
    [
        "reporting_standard",
        "time_alignment",
        "dependence",
        "functional_form",
        "missing_data",
        "interpretation",
        "survival_assumption",
        "survival_estimand",
    ],
)
def test_every_design_question_the_planner_faces_has_a_card(layer: str) -> None:
    cards = method_cards_for_layers([layer])

    assert cards, f"no method card covers the {layer} layer"
    assert all(card.layer == layer for card in cards)


def test_cards_are_case_neutral() -> None:
    """Shared guidance must not privilege one benchmark, score, or database.

    This pack is read by every case.  A card that named a specific score or
    database would quietly turn a general design rule into an instruction that
    only makes sense for one study -- the prompt-hygiene failure the project
    keeps having to undo.
    """

    forbidden = (
        "sepsis",
        "sep3",
        "sofa",
        "kdigo",
        "mimic",
        "miiv",
        "eicu",
        "hirid",
        "amsterdam",
        "sicdb",
    )
    for card in METHOD_CARDS:
        blob = f"{card.id} {card.question} {card.requirement}".lower()
        found = [term for term in forbidden if term in blob]
        assert not found, f"{card.id} names case-specific content: {found}"


def test_a_card_names_both_the_decision_and_what_to_report() -> None:
    """A card that only states a hazard cannot be acted on at plan time."""

    for card in METHOD_CARDS:
        assert card.question.endswith("?"), f"{card.id} question is not a question"
        assert len(card.requirement) > 80, (
            f"{card.id} requirement is too thin to act on"
        )
        assert card.source_key and card.source_title and card.source_year


def test_sources_expose_only_the_frozen_verified_identifiers() -> None:
    """Clickable method sources are exact PubMed records, never guessed URLs."""

    sources = {source["key"]: source for source in method_literature_citations()}
    assert {
        key: (row.get("pmid"), row.get("doi"), row.get("url"))
        for key, row in sources.items()
        if row.get("pmid") or row.get("doi") or row.get("url")
    } == {
        "strobe_2007": (
            "17938396",
            "10.7326/0003-4819-147-8-200710160-00010",
            "https://pubmed.ncbi.nlm.nih.gov/17938396/",
        ),
        "record_2015": (
            "26440803",
            "10.1371/journal.pmed.1001885",
            "https://pubmed.ncbi.nlm.nih.gov/26440803/",
        ),
        "suissa_immortal_time_2008": (
            "18056625",
            "10.1093/aje/kwm324",
            "https://pubmed.ncbi.nlm.nih.gov/18056625/",
        ),
        "anderson_landmark_1983": (
            "6668489",
            "10.1200/JCO.1983.1.11.710",
            "https://pubmed.ncbi.nlm.nih.gov/6668489/",
        ),
        "durrleman_splines_1989": (
            "2657958",
            "10.1002/sim.4780080504",
            "https://pubmed.ncbi.nlm.nih.gov/2657958/",
        ),
        "sterne_missing_data_2009": (
            "19564179",
            "10.1136/bmj.b2393",
            "https://pubmed.ncbi.nlm.nih.gov/19564179/",
        ),
        "grambsch_therneau_ph_1994": (
            None,
            "10.1093/biomet/81.3.515",
            "https://doi.org/10.1093/biomet/81.3.515",
        ),
        "royston_parmar_rmst_2011": (
            "21611958",
            "10.1002/sim.4274",
            "https://pubmed.ncbi.nlm.nih.gov/21611958/",
        ),
    }


def test_survival_outputs_require_assumption_and_estimand_sources() -> None:
    context = _context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="survival",
        steps=[
            AnalysisStep(
                step_id="primary_survival",
                planned_analysis_role="primary",
                intent="Estimate survival with prespecified non-PH handling.",
                method="cox survival analysis",
                expected_outputs=[
                    "table:ph_diagnostics",
                    "table:rmst_summary",
                ],
            )
        ],
    )

    layers = required_method_layers_for_plan(plan, context)

    assert "survival_assumption" in layers
    assert "survival_estimand" in layers


def test_the_pack_digest_is_stable_and_sensitive() -> None:
    """A run records which methodology it planned against; that must be exact."""

    assert method_literature_digest() == method_literature_digest()
    # A different layer selection is a different pack.
    assert method_literature_digest(["time_alignment"]) != method_literature_digest()

    pack = method_literature_pack()
    assert pack["case_neutral"] is True
    assert len(pack["cards"]) == len(METHOD_CARDS)


def test_one_source_backing_several_cards_is_cited_once() -> None:
    """The reading list is per source; the guidance is per decision."""

    sources = method_literature_citations()
    keys = [source["key"] for source in sources]

    assert len(keys) == len(set(keys))
    shared = [card for card in METHOD_CARDS if card.source_key == "strobe_2007"]
    assert len(shared) > 1, "this test needs a source that backs multiple cards"
    strobe = next(source for source in sources if source["key"] == "strobe_2007")
    # Its relevance line names every decision it backs, not just the first.
    assert strobe["relevance"].count("?") == len(shared)


def test_method_card_support_is_exact_per_design_element() -> None:
    reporting = method_binding_support("strobe_2007", ["reporting"])
    dependence = method_binding_support("strobe_2007", ["dependence"])
    unsupported = method_binding_support("strobe_2007", ["adjustment"])

    assert reporting["matched_layers"] == ["reporting_standard"]
    assert dependence["matched_layers"] == ["dependence"]
    assert unsupported["matched_layers"] == []
    assert unsupported["unsupported_design_elements"] == ["adjustment"]


def test_reporting_guidelines_resolve_only_explicit_named_standards() -> None:
    assert reporting_method_source_keys_for_guidelines(
        ["STROBE-style descriptive observational reporting"]
    ) == ("strobe_2007",)
    assert reporting_method_source_keys_for_guidelines(
        ["STROBE/RECORD-style observational reporting"]
    ) == ("strobe_2007", "record_2015")
    assert reporting_method_source_keys_for_guidelines(
        ["Transparent prediction-model reporting"]
    ) == ()


def test_method_card_is_immutable() -> None:
    card = METHOD_CARDS[0]

    with pytest.raises(Exception):
        card.requirement = "rewritten"  # type: ignore[misc]

    assert isinstance(card, MethodCard)
