"""Gap A (SciMON-style novelty optimisation) and Gap B (multi-criteria
validator panel) tests for the idea-mining layer.

Both features are advisory: Gap A may only replace an idea with a *measured*
more-novel revision while preserving the verbatim-quote provenance gate, and
Gap B annotates candidates without ever touching the go/no-go gate.
"""

from __future__ import annotations

from typing import Sequence

from easyicu.research_agent.discovery.idea_mining import (
    LiteratureIdeaCandidate,
    SourceMaterial,
    build_candidate_validation_messages,
    build_novelty_optimization_messages,
    optimize_ideas_for_novelty,
    score_candidates_multicriteria,
)
from easyicu.research_agent.literature import CitationRecord
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

SNAPSHOT = "source-snapshot/sha256:novopt"
QUOTE = "future work should study this predictor of mortality in a subgroup"
SOURCE_TEXT = (
    "The review notes that future work should study this predictor of mortality "
    "in a subgroup of mechanically ventilated patients."
)


def ScriptedLLM(responses: Sequence[object]) -> ScriptedMockLLMClient:
    import json

    return ScriptedMockLLMClient(
        [
            response if isinstance(response, str) else json.dumps(response)
            for response in responses
        ]
    )


class CountingSearch:
    """Returns a hit count + titles keyed by a needle found in the query."""

    def __init__(self, by_needle: dict[str, tuple[int, list[str]]]):
        self.by_needle = by_needle
        self.queries: list[str] = []

    def search_prior_art(self, query: str, *, max_results: int = 20) -> dict:
        self.queries.append(query)
        for needle, (count, titles) in self.by_needle.items():
            if needle in query:
                return {
                    "hit_count": count,
                    "top_hits": [{"title": title} for title in titles],
                }
        return {"hit_count": 0, "top_hits": []}


def _material() -> SourceMaterial:
    return SourceMaterial(
        citation=CitationRecord(
            key="neutral_review_2026",
            title="Critical illness review",
            year="2026",
            venue="Critical Care Review",
        ),
        source_adapter_level="user_supplied_excerpt",
        source_text=SOURCE_TEXT,
    )


def _idea(exposure: str) -> LiteratureIdeaCandidate:
    return LiteratureIdeaCandidate(
        source_snapshot_id=SNAPSHOT,
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        exposure_or_predictor=exposure,
        outcome="mortality",
        rationale="The source flags this as an open direction.",
        source_quote=QUOTE,
        analysis_family="association",
    )


def _revision(exposure: str) -> dict:
    return {
        "citation_key": "neutral_review_2026",
        "population": "adult ICU patients",
        "exposure_or_predictor": exposure,
        "outcome": "mortality",
        "rationale": "Sharpened toward an under-studied subgroup.",
        "source_quote": QUOTE,
        "analysis_family": "association",
    }


# --- Gap A: novelty optimisation ---------------------------------------------


def test_novelty_optimization_keeps_measured_improvement() -> None:
    search = CountingSearch(
        {
            "crowdedconstruct": (40, ["a", "b"]),
            "nicheconstruct": (3, ["c"]),
        }
    )
    llm = ScriptedLLM([_revision("nicheconstruct subgroup")])
    trace: list = []

    out = optimize_ideas_for_novelty(
        [_idea("crowdedconstruct trajectory")],
        materials=[_material()],
        source_snapshot_id=SNAPSHOT,
        llm=llm,
        search_client=search,
        crowded_min_hits=5,
        rounds=1,
        trace=trace,
    )

    assert len(out) == 1
    assert "nicheconstruct" in out[0].exposure_or_predictor
    # Provenance anchors preserved verbatim through the revision.
    assert out[0].source_quote == QUOTE
    assert out[0].citation_key == "neutral_review_2026"
    assert trace[0]["revised"] is True
    assert trace[0]["initial_exact_hits"] == 40
    assert trace[0]["final_exact_hits"] == 3


def test_novelty_optimization_rejects_non_improvement() -> None:
    # Revision is just as crowded -> original is preserved.
    search = CountingSearch(
        {
            "crowdedconstruct": (40, ["a"]),
            "stillcrowded": (40, ["a"]),
        }
    )
    llm = ScriptedLLM([_revision("stillcrowded variant")])
    trace: list = []

    out = optimize_ideas_for_novelty(
        [_idea("crowdedconstruct trajectory")],
        materials=[_material()],
        source_snapshot_id=SNAPSHOT,
        llm=llm,
        search_client=search,
        crowded_min_hits=5,
        rounds=1,
        trace=trace,
    )

    assert "crowdedconstruct" in out[0].exposure_or_predictor
    assert trace[0]["revised"] is False


def test_novelty_optimization_skips_below_threshold() -> None:
    # Idea is already niche (3 < min_hits 5): the LLM is never called.
    search = CountingSearch({"nicheconstruct": (3, [])})
    llm = ScriptedLLM([_revision("should not be used")])
    trace: list = []

    out = optimize_ideas_for_novelty(
        [_idea("nicheconstruct trajectory")],
        materials=[_material()],
        source_snapshot_id=SNAPSHOT,
        llm=llm,
        search_client=search,
        crowded_min_hits=5,
        rounds=1,
        trace=trace,
    )

    assert len(llm.calls) == 0
    assert "nicheconstruct" in out[0].exposure_or_predictor
    assert trace[0]["revised"] is False


def test_novelty_optimization_drops_untraceable_revision() -> None:
    # A revision whose quote is NOT in the source is rejected by the provenance
    # gate; the original (crowded) idea is preserved rather than admitted.
    search = CountingSearch(
        {
            "crowdedconstruct": (40, ["a"]),
            "nicheconstruct": (2, []),
        }
    )
    tampered = _revision("nicheconstruct subgroup")
    tampered["source_quote"] = "a fabricated quote that is not in the source text"
    llm = ScriptedLLM([tampered])
    trace: list = []

    out = optimize_ideas_for_novelty(
        [_idea("crowdedconstruct trajectory")],
        materials=[_material()],
        source_snapshot_id=SNAPSHOT,
        llm=llm,
        search_client=search,
        crowded_min_hits=5,
        rounds=1,
        trace=trace,
    )

    assert "crowdedconstruct" in out[0].exposure_or_predictor
    assert out[0].source_quote == QUOTE
    assert trace[0]["revised"] is False


def test_novelty_optimization_prompt_is_case_neutral() -> None:
    messages = build_novelty_optimization_messages(
        _idea("crowdedconstruct trajectory"),
        prior_art_titles=["existing study one", "existing study two"],
        source_text=SOURCE_TEXT,
        round_idx=0,
        num_rounds=1,
    )
    text = "\n".join(message.content for message in messages).lower()

    for forbidden in ["lactate", "sofa", "mimic", "sepsis"]:
        assert forbidden not in text
    assert "citation_key" in text
    assert "source_quote" in text
    assert "return only json" in text


# --- Gap B: multi-criteria validator -----------------------------------------


def test_multicriteria_validator_clamps_scores() -> None:
    llm = ScriptedLLM(
        [
            {
                "clarity": 7,  # over-range -> clamps to 5
                "novelty": 0,  # under-range -> clamps to 1
                "feasibility_fit": 4,
                "impact": 3,
                "justification": "Clear and feasible but incremental.",
            }
        ]
    )
    records = [
        {
            "candidate_topic": "predictor -> mortality",
            "executable_candidate_id": "cand-1",
            "go_no_go": "recommend",
            "prior_art": {"novelty_label": "sparse"},
        }
    ]

    scores = score_candidates_multicriteria(records, llm=llm)

    assert len(scores) == 1
    row = scores[0]
    assert row["clarity"] == 5
    assert row["novelty"] == 1
    assert row["feasibility_fit"] == 4
    assert row["impact"] == 3
    assert row["candidate_id"] == "cand-1"
    assert row["go_no_go"] == "recommend"
    assert row["justification"].startswith("Clear")


def test_multicriteria_validator_tolerates_malformed_response() -> None:
    llm = ScriptedLLM(["not json at all"])
    records = [{"candidate_topic": "x -> y", "go_no_go": "hold"}]

    scores = score_candidates_multicriteria(records, llm=llm)

    assert len(scores) == 1
    row = scores[0]
    # Candidate still present with honest "unscored" markers.
    assert row["clarity"] is None
    assert row["novelty"] is None
    assert row["go_no_go"] == "hold"


def test_multicriteria_validator_prompt_is_case_neutral() -> None:
    messages = build_candidate_validation_messages(
        {
            "candidate_topic": "predictor -> mortality",
            "go_no_go": "recommend",
            "prior_art": {"novelty_label": "sparse"},
        }
    )
    text = "\n".join(message.content for message in messages).lower()

    for forbidden in ["lactate", "sofa", "mimic"]:
        assert forbidden not in text
    assert "clarity" in text
    assert "feasibility_fit" in text
    assert "return only json" in text
