from __future__ import annotations

from easyicu.research_agent.discovery.idea_mining import (
    ExecutableHypothesisCandidate,
    LiteratureIdeaCandidate,
    PriorArtAssessment,
    SourceMaterial,
    assess_prior_art_for_idea,
    build_discovery_candidate_records,
)
from easyicu.research_agent.literature import CitationRecord


class OneShotPriorArtSearch:
    def __init__(self):
        self.calls = 0

    def search_prior_art(self, query: str, *, max_results: int = 20, idea=None):
        self.calls += 1
        if self.calls > 1:
            return {"hit_count": 0, "top_hits": []}
        return {
            "hit_count": 4,
            "top_hits": [
                {
                    "pmid": "1",
                    "title": "External validation of lactate and mortality in eICU",
                    "abstract": "Same topic external validation.",
                    "direct_same_topic": True,
                    "same_topic_screened": True,
                },
                {
                    "pmid": "2",
                    "title": "Lactate and acute kidney injury in ICU patients",
                    "abstract": "Same exposure, different outcome.",
                },
                {
                    "pmid": "3",
                    "title": "Albumin and mortality in critically ill patients",
                    "abstract": "Same outcome, different exposure.",
                },
                {
                    "pmid": "4",
                    "title": "ICU monitoring quality in critical illness",
                    "abstract": "Adjacent background evidence.",
                },
            ],
        }


def _idea() -> LiteratureIdeaCandidate:
    return LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        exposure_or_predictor="lactate",
        outcome="mortality",
        rationale="The source identifies this as an unresolved question.",
        source_quote="future work should study lactate and mortality",
        analysis_family="association",
    )


def test_prior_art_assessment_records_evidence_map_counts() -> None:
    assessment = assess_prior_art_for_idea(
        _idea(),
        search_client=OneShotPriorArtSearch(),
        searched_at="2026-06-16T00:00:00+00:00",
        cross_db_targets=["eicu"],
    )

    assert assessment.novelty_label == "already_done"
    assert assessment.evidence_map_counts == {
        "same_topic_cross_database_or_external": 1,
        "same_exposure_different_outcome": 1,
        "same_outcome_different_exposure": 1,
        "adjacent_icu_background": 1,
    }
    assert (
        assessment.evidence_map_examples["same_topic_cross_database_or_external"][0][
            "pmid"
        ]
        == "1"
    )


def test_discovery_record_routes_resolved_unknown_outcome_as_operationalization_hold() -> (
    None
):
    idea = _idea()
    candidate = ExecutableHypothesisCandidate(
        executable_candidate_id="execidea_test",
        literature_idea_id=str(idea.literature_idea_id),
        source_snapshot_id=idea.source_snapshot_id,
        citation_key=idea.citation_key,
        population=idea.population,
        predictor_label="lactate",
        outcome_label="mortality",
        resolved_predictor_concept="lact",
        resolved_outcome_concept="death",
        feasibility_pair_key=("lact", "death"),
        research_question="Does lactate associate with mortality?",
        source_quote=idea.source_quote,
        non_executable_reasons=[
            "outcome determinability is unknown for feasibility probing"
        ],
    )
    assessment = PriorArtAssessment(
        novelty_snapshot_id="novelty-snapshot/sha256:abc123",
        literature_idea_id=str(idea.literature_idea_id),
        source_snapshot_id=idea.source_snapshot_id,
        searched_at="2026-06-16T00:00:00+00:00",
        predictor_literature_phrase="lactate",
        outcome_literature_phrase="mortality",
        query_records=[],
        novelty_label="sparse",
        literature_saturation_signal=0.25,
        novelty_statement="not a novelty claim",
    )
    source = SourceMaterial(
        citation=CitationRecord(
            key="review_2026",
            title="Review with gap",
            year="2026",
            venue="Crit Care",
            pmid="123",
        )
    )

    records = build_discovery_candidate_records(
        literature_ideas=[idea],
        executable_candidates=[candidate],
        prior_art_assessments=[assessment],
        candidate_records=[],
        source_materials=[source],
    )

    assert len(records) == 1
    record = records[0]
    assert record.go_no_go == "hold"
    assert record.feasibility_route == "needs_outcome_operationalization"
    assert "outcome event" in (record.feasibility_next_action or "")
    assert record.requires_human_confirmation is True


def test_discovery_record_formats_concept_set_topic_without_empty_arrow() -> None:
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:set123",
        citation_key="review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        analysis_family="measurement_bias_audit",
        analysis_concepts=["lactate testing frequency", "severity score"],
        rationale="The source highlights informative measurement.",
        source_quote="future work should study measurement bias",
    )
    candidate = ExecutableHypothesisCandidate(
        executable_candidate_id="execidea_set",
        literature_idea_id=str(idea.literature_idea_id),
        source_snapshot_id=idea.source_snapshot_id,
        citation_key=idea.citation_key,
        population=idea.population,
        predictor_label="",
        outcome_label="",
        resolved_analysis_concepts=["lact", "sofa2"],
        research_question="Could measurement processes bias lactate and severity scores?",
        source_quote=idea.source_quote,
        analysis_family="measurement_bias_audit",
    )
    assessment = PriorArtAssessment(
        novelty_snapshot_id="novelty-snapshot/sha256:set123",
        literature_idea_id=str(idea.literature_idea_id),
        source_snapshot_id=idea.source_snapshot_id,
        searched_at="2026-06-16T00:00:00+00:00",
        predictor_literature_phrase="",
        outcome_literature_phrase="",
        query_records=[],
        novelty_label="sparse",
        literature_saturation_signal=0.25,
        novelty_statement="not a novelty claim",
        same_topic_screen_status="top-N same-topic screened",
    )
    source = SourceMaterial(
        citation=CitationRecord(
            key="review_2026",
            title="Review with concept-set gap",
            year="2026",
            venue="Crit Care",
            pmid="999",
        )
    )

    records = build_discovery_candidate_records(
        literature_ideas=[idea],
        executable_candidates=[candidate],
        prior_art_assessments=[assessment],
        candidate_records=[],
        source_materials=[source],
    )

    topic = records[0].candidate_topic
    assert "measurement_bias_audit" in topic
    assert "lactate testing frequency" in topic
    assert "->" not in topic
    assert records[0].feasibility_route == "concept_set_human_confirm"
    assert "concept set" in (records[0].feasibility_next_action or "")
