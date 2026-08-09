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

from easyicu.research_agent.literature import build_preplan_literature_bundle
from easyicu.research_agent.planning.method_literature import (
    METHOD_CARDS,
    MethodCard,
    method_cards_for_layers,
    method_literature_citations,
    method_literature_digest,
    method_literature_pack,
)
from easyicu.research_agent.schema import ResearchContext


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
        assert (
            len(card.requirement) > 80
        ), f"{card.id} requirement is too thin to act on"
        assert card.source_key and card.source_title and card.source_year


def test_sources_carry_no_guessed_identifiers() -> None:
    """An invented DOI or PMID is worse than an absent one: it looks checked."""

    for source in method_literature_citations():
        assert "doi" not in source or source.get("doi") is None
        assert "pmid" not in source or source.get("pmid") is None


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


def test_method_card_is_immutable() -> None:
    card = METHOD_CARDS[0]

    with pytest.raises(Exception):
        card.requirement = "rewritten"  # type: ignore[misc]

    assert isinstance(card, MethodCard)
