from __future__ import annotations

import inspect
import json
from typing import Sequence

import pytest

import easyicu.research_agent.idea_mining as idea_mining_mod
from easyicu.research_agent.idea_mining import (
    ExecutableHypothesisCandidate,
    IdeaExtractionError,
    LiteratureIdeaCandidate,
    NonExecutableCandidateError,
    OutcomeDeterminability,
    SourceMaterial,
    assess_prior_art_for_idea,
    build_idea_extraction_messages,
    build_prior_art_queries,
    extract_literature_ideas,
    freeze_source_snapshot,
    map_literature_idea_to_executable_candidate,
    run_idea_mining_dry_run,
)
from easyicu.research_agent.idea_registry import (
    CandidateNotExecutableError as RegistryCandidateNotExecutableError,
    IdeaCandidateRegistry,
)
from easyicu.research_agent.literature import CitationRecord
from easyicu.research_agent.llm import LLMMessage
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole


class CapturingIdeaLLM:
    name = "capturing-idea-llm"

    def __init__(self, response: object):
        self.response = response
        self.messages: Sequence[LLMMessage] = ()

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        self.messages = messages
        return json.dumps(self.response)


class FakePriorArtSearchClient:
    def __init__(self, responses: dict[str, object] | None = None):
        self.responses = responses or {}
        self.queries: list[str] = []

    def search_prior_art(self, query: str, *, max_results: int = 20) -> object:
        self.queries.append(query)
        for needle, response in self.responses.items():
            if needle in query:
                return response
        return {"hit_count": 0, "top_hits": []}


def _citation() -> CitationRecord:
    return CitationRecord(
        key="neutral_review_2026",
        title="Review of physiologic trajectories in critical illness",
        year="2026",
        venue="Critical Care Review",
        relevance=(
            "The review calls for studies of physiologic markers and "
            "patient-centered endpoints in broad ICU cohorts."
        ),
    )


def test_idea_extraction_prompt_is_case_neutral() -> None:
    material = SourceMaterial(citation=_citation(), source_adapter_level="metadata_only")

    messages = build_idea_extraction_messages(
        [material],
        source_snapshot_id="source-snapshot/sha256:abc123",
    )
    prompt_text = "\n".join(message.content for message in messages).lower()

    for forbidden in ["lactate", "sofa", "mimic", "sepsis", "mortality"]:
        assert forbidden not in prompt_text
    assert "return only json" in prompt_text
    assert "source_quote must be copied" in prompt_text
    assert "unresolved" in prompt_text
    assert "future directions" in prompt_text
    assert "specific named constructs" in prompt_text
    assert "generic placeholders" in prompt_text


def test_extract_literature_ideas_from_user_excerpt_with_traceable_quote() -> None:
    quote = "physiologic markers may identify modifiable risk trajectories"
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text=(
            "The authors note that physiologic markers may identify "
            "modifiable risk trajectories before deterioration."
        ),
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "physiologic marker",
                "outcome": "patient-centered endpoint",
                "rationale": "The review describes this as an open direction.",
                "source_quote": quote,
                "analysis_family": "association",
                "time_window_hint": "early ICU stay",
                "aggregation_hint": "first available value",
            }
        ]
    )

    candidates = extract_literature_ideas(
        materials=[material],
        source_snapshot_id="source-snapshot/sha256:abc123",
        llm=llm,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.literature_idea_id
    assert candidate.source_quote == quote
    assert candidate.source_adapter_level == "user_supplied_excerpt"
    assert llm.messages[0].role == "system"


def test_prior_art_queries_use_literature_phrase_not_canonical_concept_key() -> None:
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        exposure_or_predictor="lactate clearance trajectory",
        outcome="refractory circulatory support",
        rationale="The source identifies this as an unresolved trajectory question.",
        source_quote="future work should study lactate clearance trajectory",
        analysis_family="trajectory",
        time_window_hint="first six hours after admission",
    )

    queries = build_prior_art_queries(idea)

    assert '"lactate clearance trajectory"[Title/Abstract]' in queries["exact"]
    assert "trajectory[Title/Abstract]" in queries["exact"]
    assert "first six hours after admission" in queries["exact"]
    assert "lact[Title/Abstract]" not in queries["exact"]

    concepts = [
        ConceptDescriptor(
            name="lactate",
            source_concept="lact",
            derived_from_concepts=["lactate clearance trajectory"],
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="refractory_support",
            derived_from_concepts=["refractory circulatory support"],
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]
    executable = map_literature_idea_to_executable_candidate(
        idea,
        available_concepts=concepts,
        outcome_determinability={"refractory_support": "known_0_1"},
    )
    assert executable.feasibility_pair_key == ("lact", "refractory_support")

    search = FakePriorArtSearchClient()
    assessment = assess_prior_art_for_idea(
        idea,
        executable_candidate=executable,
        search_client=search,
        searched_at="2026-06-04T00:00:00+00:00",
    )
    assert any("lactate clearance trajectory" in query for query in search.queries)
    assert assessment.feasibility_pair_key == ("lact", "refractory_support")
    assert assessment.novelty_label == "apparently_gap"
    assert assessment.clinical_plausibility_requires_human is True


def test_prior_art_broad_query_uses_phrase_facets_to_avoid_false_gaps() -> None:
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients with shock",
        exposure_or_predictor="vasopressin exposure",
        outcome="intensive-care unit mortality",
        rationale="The source identifies this as an unresolved exposure question.",
        source_quote="future work should study vasopressin exposure",
        analysis_family="association",
        time_window_hint="postoperative vasoplegia",
    )

    queries = build_prior_art_queries(idea)
    broad = queries["broad"]
    exact = queries["exact"]

    assert '"vasopressin exposure"[Title/Abstract]' in broad
    assert "vasopressin[Title/Abstract]" in broad
    assert '"intensive-care unit mortality"[Title/Abstract]' in broad
    assert "mortality[Title/Abstract]" in broad
    assert "death[Title/Abstract]" in broad
    assert "ICU[Title/Abstract]" in broad
    assert '"adult ICU patients with shock"[Title/Abstract]' not in broad
    assert '"postoperative vasoplegia"[Title/Abstract]' in exact
    assert "vaso[Title/Abstract]" not in broad

    search = FakePriorArtSearchClient(
        {
            "vasopressin[Title/Abstract]": {
                "hit_count": 17,
                "top_hits": [
                    {
                        "pmid": "777",
                        "title": "Vasopressin and mortality in critically ill adults",
                        "abstract": "Adjacent prior-art hit without exact exposure wording.",
                    }
                ],
            }
        }
    )
    assessment = assess_prior_art_for_idea(
        idea,
        search_client=search,
        searched_at="2026-06-05T00:00:00+00:00",
    )

    query_by_type = {record.query_type: record for record in assessment.query_records}
    assert query_by_type["broad"].hit_count == 17
    assert query_by_type["exact"].hit_count == 0
    assert assessment.novelty_label != "apparently_gap"
    assert assessment.novelty_label == "sparse"


def test_prior_art_freeze_records_direct_same_topic_pmids_and_rationale() -> None:
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        exposure_or_predictor="marker trajectory",
        outcome="patient-centered endpoint",
        rationale="The source identifies this as an unresolved trajectory question.",
        source_quote="future work should study marker trajectory",
        analysis_family="trajectory",
    )
    search = FakePriorArtSearchClient(
        {
            '"marker trajectory"[Title/Abstract]': {
                "hit_count": 2,
                "top_hits": [
                    {
                        "pmid": "123",
                        "title": "Marker trajectory and patient-centered endpoint",
                        "direct_same_topic": True,
                        "direct_same_topic_rationale": "same population and endpoint",
                    }
                ],
            }
        }
    )

    first = assess_prior_art_for_idea(
        idea,
        search_client=search,
        searched_at="2026-06-04T00:00:00+00:00",
    )
    second = assess_prior_art_for_idea(
        idea,
        search_client=search,
        searched_at="2026-06-04T00:00:00+00:00",
    )

    assert first.novelty_label == "already_done"
    assert first.direct_same_topic_pmids == ["123"]
    assert first.direct_same_topic_rationales == {
        "123": "same population and endpoint"
    }
    assert first.novelty_snapshot_id == second.novelty_snapshot_id


def test_prior_art_substring_fallback_does_not_mark_already_done() -> None:
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        exposure_or_predictor="marker",
        outcome="endpoint",
        rationale="The source identifies this as an unresolved marker question.",
        source_quote="future work should study marker",
        analysis_family="association",
    )
    search = FakePriorArtSearchClient(
        {
            "marker[Title/Abstract]": {
                "hit_count": 12,
                "top_hits": [
                    {
                        "pmid": "999",
                        "title": "Marker and endpoint in adult ICU cohorts",
                        "abstract": "A title/abstract substring match only.",
                    }
                ],
            }
        }
    )

    assessment = assess_prior_art_for_idea(
        idea,
        search_client=search,
        searched_at="2026-06-04T00:00:00+00:00",
    )

    assert assessment.same_topic_screen_status == "automated-substring-only, NOT screened"
    assert assessment.novelty_label == "crowded_but_differentiable"
    assert assessment.literature_saturation_signal == pytest.approx(0.70)
    assert assessment.direct_same_topic_pmids == ["999"]
    exact = [record for record in assessment.query_records if record.query_type == "exact"][0]
    assert exact.top_hits[0].direct_same_topic is True
    assert exact.top_hits[0].same_topic_screened is False


def test_candidate_record_build_only_swallows_unregistered_candidates() -> None:
    class BrokenRegistry:
        def family_size(self, _hypothesis_family_id: str) -> int:
            return 1

        def latest_entry(self, _candidate_id: str):
            raise RuntimeError("registry storage unavailable")

    candidate = ExecutableHypothesisCandidate(
        executable_candidate_id="exec-a",
        literature_idea_id="idea-a",
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        population="adult ICU patients",
        predictor_label="marker",
        outcome_label="endpoint",
        resolved_predictor_concept="marker",
        resolved_outcome_concept="endpoint",
        feasibility_pair_key=("marker", "endpoint"),
        research_question="Does marker associate with endpoint?",
        source_quote="future work should study marker",
    )

    with pytest.raises(RuntimeError, match="registry storage unavailable"):
        idea_mining_mod._build_candidate_records(
            candidates=[candidate],
            ranking_by_pair={},
            registry_ids={},
            registry=BrokenRegistry(),
            hypothesis_family_id="family-a",
            source_snapshot_id="source-snapshot/sha256:abc123",
        )


def test_extract_literature_ideas_rejects_untraceable_quote() -> None:
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text="A supplied excerpt about broad clinical uncertainty.",
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "marker",
                "outcome": "endpoint",
                "rationale": "Not anchored.",
                "source_quote": "a sentence that is not in the source",
                "analysis_family": "association",
            }
        ]
    )

    with pytest.raises(IdeaExtractionError, match="not traceable"):
        extract_literature_ideas(
            materials=[material],
            source_snapshot_id="source-snapshot/sha256:abc123",
            llm=llm,
        )


def test_licensed_source_snapshot_does_not_store_full_text() -> None:
    secret_body = "copyrighted licensed full text body that must not be stored"
    manifest = freeze_source_snapshot(
        [
            SourceMaterial(
                citation=_citation(),
                source_adapter_level="licensed_fulltext_manifest_only",
                locator="publisher-page:section-3",
                source_text=secret_body,
            )
        ],
        created_at="2026-06-04T00:00:00+00:00",
    )

    payload = manifest.model_dump(mode="json")
    payload_text = json.dumps(payload, ensure_ascii=False)
    item = manifest.items[0]
    assert manifest.source_snapshot_id.startswith("source-snapshot/sha256:")
    assert item.source_text_sha256
    assert item.source_text_char_count == len(secret_body)
    assert item.source_text_stored is False
    assert secret_body not in payload_text
    assert "must not be stored" not in payload_text


def test_map_idea_to_executable_candidate_aligns_canonical_concept_keys() -> None:
    candidate = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="metadata_only",
        population="adult ICU patients",
        exposure_or_predictor="creatinine",
        outcome="patient-centered endpoint",
        rationale="The source suggests a candidate association.",
        source_quote="Reviewers call for studies of physiologic markers.",
    )
    concepts = [
        ConceptDescriptor(
            name="creatinine_first24h",
            source_concept="crea",
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="endpoint_known",
            source_concept="endpoint_known",
            derived_from_concepts=["patient-centered endpoint"],
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]

    executable = map_literature_idea_to_executable_candidate(
        candidate,
        available_concepts=concepts,
        outcome_determinability={
            "endpoint_known": OutcomeDeterminability(
                outcome="endpoint_known",
                status="known_0_1",
            )
        },
    )

    assert executable.executable
    assert executable.assert_research_context_allowed() is True
    assert executable.resolved_predictor_concept == "crea"
    assert executable.resolved_outcome_concept == "endpoint_known"
    assert executable.feasibility_pair_key == ("crea", "endpoint_known")


def test_derived_feature_requires_feature_engineering_before_execution() -> None:
    candidate = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="metadata_only",
        population="adult ICU patients",
        exposure_or_predictor="lactate clearance trajectory",
        outcome="patient-centered endpoint",
        rationale="The source suggests a trajectory feature.",
        source_quote="future work should study lactate clearance trajectory",
        analysis_family="trajectory",
    )
    concepts = [
        ConceptDescriptor(
            name="lactate_first24h",
            source_concept="lact",
            derived_from_concepts=["lactate", "lactate clearance trajectory"],
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="endpoint_known",
            source_concept="endpoint_known",
            derived_from_concepts=["patient-centered endpoint"],
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]

    blocked = map_literature_idea_to_executable_candidate(
        candidate,
        available_concepts=concepts,
        outcome_determinability={"endpoint_known": "known_0_1"},
    )

    assert blocked.resolved_predictor_concept == "lact"
    assert blocked.feasibility_pair_key == ("lact", "endpoint_known")
    assert blocked.feature_derivation_status == "requires_derived_feature"
    assert "requires repeated measurements >=2" in blocked.feature_derivation_requirements
    assert not blocked.executable
    assert any("derived feature engineering" in reason for reason in blocked.non_executable_reasons)


def test_alias_map_resolves_raw_support_but_preserves_derived_feature_gate() -> None:
    candidate = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="metadata_only",
        population="adult ICU patients",
        exposure_or_predictor="sentinel clearance trajectory",
        outcome="patient-centered endpoint",
        rationale="The source suggests a derived-feature direction.",
        source_quote="future work should study sentinel clearance trajectory",
        analysis_family="trajectory",
    )
    concepts = [
        ConceptDescriptor(
            name="marker_first24h",
            source_concept="marker",
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="endpoint_known",
            source_concept="endpoint_known",
            derived_from_concepts=["patient-centered endpoint"],
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]

    without_alias = map_literature_idea_to_executable_candidate(
        candidate,
        available_concepts=concepts,
        outcome_determinability={"endpoint_known": "known_0_1"},
    )
    with_alias = map_literature_idea_to_executable_candidate(
        candidate,
        available_concepts=concepts,
        concept_aliases={"marker": ["sentinel marker"]},
        outcome_determinability={"endpoint_known": "known_0_1"},
    )

    assert without_alias.resolved_predictor_concept is None
    assert with_alias.resolved_predictor_concept == "marker"
    assert with_alias.feature_derivation_status == "requires_derived_feature"
    assert not with_alias.executable


def test_available_derived_feature_can_be_executable() -> None:
    candidate = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="metadata_only",
        population="adult ICU patients",
        exposure_or_predictor="lactate clearance trajectory",
        outcome="patient-centered endpoint",
        rationale="The source suggests a trajectory feature.",
        source_quote="future work should study lactate clearance trajectory",
        analysis_family="trajectory",
    )
    concepts = [
        ConceptDescriptor(
            name="lactate_clearance_trajectory",
            source_concept="lactate_clearance_trajectory",
            derived_from_concepts=["lactate clearance trajectory"],
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="endpoint_known",
            source_concept="endpoint_known",
            derived_from_concepts=["patient-centered endpoint"],
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]

    executable = map_literature_idea_to_executable_candidate(
        candidate,
        available_concepts=concepts,
        outcome_determinability={"endpoint_known": "known_0_1"},
    )

    assert executable.feature_derivation_status == "derived_feature_available"
    assert executable.executable
    assert executable.feasibility_pair_key == (
        "lactate_clearance_trajectory",
        "endpoint_known",
    )


def test_generic_differentiators_do_not_create_apparently_gap() -> None:
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        exposure_or_predictor="marker",
        outcome="mortality",
        rationale="The source suggests this direction.",
        source_quote="future work should study marker patterns",
        analysis_family="association",
        time_window_hint="ICU stay",
        aggregation_hint="any exposure",
    )

    queries = build_prior_art_queries(idea)
    assert "ICU stay" not in queries["exact"]
    assert "any exposure" not in queries["exact"]

    assessment = assess_prior_art_for_idea(
        idea,
        search_client=FakePriorArtSearchClient(),
        searched_at="2026-06-04T00:00:00+00:00",
    )

    assert assessment.differentiators == []
    assert assessment.has_specific_differentiator is False
    assert assessment.novelty_label == "sparse"
    assert "Same-topic screen status" in assessment.novelty_statement


def test_llm_screened_prior_art_hits_are_frozen() -> None:
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        exposure_or_predictor="marker trajectory",
        outcome="patient-centered endpoint",
        rationale="The source suggests this direction.",
        source_quote="future work should study marker trajectory",
        analysis_family="trajectory",
    )
    search = FakePriorArtSearchClient(
        {
            '"marker trajectory"[Title/Abstract]': {
                "hit_count": 1,
                "top_hits": [
                    {
                        "pmid": "222",
                        "title": "A different marker trajectory study",
                        "abstract": "Different endpoint and design.",
                        "same_topic_screened": True,
                        "direct_same_topic": False,
                        "direct_same_topic_rationale": "LLM screen: adjacent but not direct same-topic.",
                    }
                ],
            }
        }
    )

    assessment = assess_prior_art_for_idea(
        idea,
        search_client=search,
        searched_at="2026-06-04T00:00:00+00:00",
    )

    assert assessment.same_topic_screen_status == "top-N same-topic screened"
    assert "Same-topic screening is asymmetric" in assessment.scope_note
    exact = [record for record in assessment.query_records if record.query_type == "exact"][0]
    assert exact.top_hits[0].same_topic_screened is True
    assert "LLM screen" in str(exact.top_hits[0].direct_same_topic_rationale)


def test_event_present_na_outcome_blocks_joint_feasibility_mapping() -> None:
    candidate = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="metadata_only",
        population="adult ICU patients",
        exposure_or_predictor="marker",
        outcome="endpoint event",
        rationale="The source suggests a candidate association.",
        source_quote="Reviewers call for studies of physiologic markers.",
    )
    concepts = [
        ConceptDescriptor(
            name="marker",
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="endpoint_event",
            derived_from_concepts=["endpoint event"],
            role=VariableRole.OUTCOME,
            dtype="float64",
        ),
    ]

    blocked = map_literature_idea_to_executable_candidate(
        candidate,
        available_concepts=concepts,
        outcome_determinability={
            "endpoint_event": {
                "status": "event_present_na",
                "note": "Raw column stores only event-positive rows.",
            }
        },
    )

    assert not blocked.executable
    assert blocked.feasibility_pair_key == ("marker", "endpoint_event")
    assert any(
        "event-positive present/NA" in reason
        for reason in blocked.non_executable_reasons
    )
    with pytest.raises(NonExecutableCandidateError):
        blocked.assert_research_context_allowed()


def test_unsupported_source_level_is_not_extracted_before_dg1() -> None:
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="open_access_fulltext",
        source_text="Full text is not wired in S4 v1.",
    )

    with pytest.raises(IdeaExtractionError, match="metadata_only"):
        build_idea_extraction_messages(
            [material],
            source_snapshot_id="source-snapshot/sha256:abc123",
        )


def test_package_lazy_exports_idea_mining_api() -> None:
    import easyicu.research_agent as ra

    assert ra.LiteratureIdeaCandidate is LiteratureIdeaCandidate
    assert ra.SourceMaterial is SourceMaterial
    assert callable(ra.extract_literature_ideas)
    assert callable(ra.run_idea_mining_dry_run)
    assert callable(ra.assess_prior_art_for_idea)
    assert callable(ra.build_prior_art_queries)
    assert callable(ra.render_discovery_report)


def test_dry_run_wires_pairwise_feasibility_registry_and_stops_at_gate(
    tmp_path,
) -> None:
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text=(
            "The review highlights creatinine trajectory as a candidate "
            "direction. It also highlights physiologic marker patterns for "
            "patient-centered endpoint studies."
        ),
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "creatinine",
                "outcome": "patient-centered endpoint",
                "rationale": "The source describes this direction.",
                "source_quote": "creatinine trajectory",
                "analysis_family": "association",
            },
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "physiologic marker",
                "outcome": "patient-centered endpoint",
                "rationale": "The source describes this direction.",
                "source_quote": "physiologic marker patterns",
                "analysis_family": "association",
            },
        ]
    )
    concepts = [
        ConceptDescriptor(
            name="creatinine_first24h",
            source_concept="crea",
            derived_from_concepts=["creatinine"],
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="marker_first24h",
            source_concept="marker",
            derived_from_concepts=["physiologic marker"],
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="endpoint_known",
            source_concept="endpoint_known",
            derived_from_concepts=["patient-centered endpoint"],
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]
    calls: list[tuple[str, ...]] = []

    def fake_probe(**kwargs):
        pair = tuple(kwargs["concepts"])
        calls.append(pair)
        assert len(pair) == 2
        joint = 0.82 if pair[0] == "crea" else 0.64
        return {
            pair[0]: {
                "joint_fraction_complete": joint,
                "n_joint_complete": int(joint * 100),
                "denominator_n": 100,
                "source": "synthetic_s1_fixture",
                "note": "pair-level synthetic fixture",
            },
            pair[1]: {
                "joint_fraction_complete": joint,
                "n_joint_complete": int(joint * 100),
                "denominator_n": 100,
                "source": "synthetic_s1_fixture",
            },
        }

    result = run_idea_mining_dry_run(
        materials=[material],
        llm=llm,
        available_concepts=concepts,
        outcome_determinability={
            "endpoint_known": OutcomeDeterminability(
                outcome="endpoint_known",
                status="known_0_1",
            )
        },
        output_dir=tmp_path / "dry_run",
        database="miiv",
        data_path=tmp_path / "wide_cohort.parquet",
        feasibility_probe=fake_probe,
        prior_art_search_client=FakePriorArtSearchClient(
            {
                "creatinine[Title/Abstract]": {
                    "hit_count": 1,
                    "top_hits": [
                        {
                            "pmid": "111",
                            "title": (
                                "Creatinine and patient-centered endpoint in "
                                "adult ICU patients"
                            ),
                            "direct_same_topic": True,
                            "direct_same_topic_rationale": (
                                "same marker and endpoint"
                            ),
                        }
                    ],
                }
            }
        ),
        prior_art_searched_at="2026-06-04T00:00:00+00:00",
    )

    assert calls == [("crea", "endpoint_known"), ("marker", "endpoint_known")]
    assert result.yield_report.n_literature_ideas == 2
    assert result.yield_report.n_executable == 2
    assert len(result.feasibility_signals) == 2
    assert {record.pair_key for record in result.feasibility_signals} == {
        ("crea", "endpoint_known"),
        ("marker", "endpoint_known"),
    }
    assert result.ranked_candidates
    assert {
        candidate["coverage_source"] for candidate in result.ranked_candidates
    } == {"pair_joint_feasibility"}
    assert all(
        record.multiple_testing_family_size == 2
        for record in result.candidate_records
    )
    assert all(
        record.multiple_testing_executable_family_size == 2
        for record in result.candidate_records
    )
    assert all(
        record.causal_audit_scope
        == "static_triage_marker_no_per_candidate_causal_audit"
        for record in result.candidate_records
    )
    assert all(
        record.registry_selection_status == "proposed"
        for record in result.candidate_records
    )
    assert len(result.prior_art_assessments) == 2
    assessment_by_phrase = {
        assessment.predictor_literature_phrase: assessment
        for assessment in result.prior_art_assessments
    }
    assert assessment_by_phrase["creatinine"].novelty_label == "already_done"
    assert (
        assessment_by_phrase["creatinine"].direct_same_topic_rationales["111"]
        == "same marker and endpoint"
    )
    assert (
        assessment_by_phrase["physiologic marker"].novelty_label
        == "sparse"
    )
    saturation_by_predictor = {
        candidate["predictor"]: candidate["literature_saturation_signal"]
        for candidate in result.ranked_candidates
    }
    assert saturation_by_predictor["crea"] == pytest.approx(0.95)
    assert saturation_by_predictor["marker"] == pytest.approx(0.25)
    assert len(result.discovery_records) == 2
    assert all(
        record.clinical_plausibility_requires_human
        for record in result.discovery_records
    )
    assert result.novelty_snapshot_path
    assert result.discovery_report_path
    assert "not a novelty claim" in (
        tmp_path / "dry_run" / "discovery_report.md"
    ).read_text()
    registry = IdeaCandidateRegistry(result.registry_path)
    assert len(registry.records) == 2
    for record in result.candidate_records:
        with pytest.raises(RegistryCandidateNotExecutableError):
            registry.assert_executable(record.registry_candidate_id)
    assert (tmp_path / "dry_run" / "source_snapshot_manifest.json").exists()
    assert (tmp_path / "dry_run" / "candidate_triage_report.json").exists()

    import easyicu.research_agent.idea_mining as idea_mining_module

    assert ".pipeline" not in inspect.getsource(idea_mining_module)


def test_dry_run_uses_dictionary_catalog_by_default(tmp_path) -> None:
    quote = "early vasopressin exposure and intensive-care unit mortality"
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text=(
            "The review highlights early vasopressin exposure and "
            "intensive-care unit mortality as an unresolved ICU research direction."
        ),
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "early vasopressin exposure",
                "outcome": "intensive-care unit mortality",
                "rationale": "The source describes this direction.",
                "source_quote": quote,
                "analysis_family": "association",
            }
        ]
    )
    calls: list[tuple[str, ...]] = []

    def fake_probe(**kwargs):
        pair = tuple(kwargs["concepts"])
        calls.append(pair)
        return {
            concept: {
                "joint_fraction_complete": 0.75,
                "n_joint_complete": 75,
                "denominator_n": 100,
                "source": "synthetic_s1_fixture",
            }
            for concept in pair
        }

    result = run_idea_mining_dry_run(
        materials=[material],
        llm=llm,
        available_concepts=["adh_rate", "death"],
        output_dir=tmp_path / "dry_run",
        feasibility_probe=fake_probe,
    )

    assert calls == [("adh_rate", "death")]
    assert result.yield_report.n_executable == 1
    candidate = result.executable_candidates[0]
    assert candidate.resolved_predictor_concept == "adh_rate"
    assert candidate.resolved_outcome_concept == "death"
    assert candidate.outcome_determinability_status == "known_0_1"
    assert candidate.feasibility_pair_key == ("adh_rate", "death")
    assert candidate.non_executable_reasons == []


def test_dry_run_surfaces_structural_unavailable_feasibility_note(tmp_path) -> None:
    import easyicu.research_agent.concept_availability as ca

    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text=(
            "The review highlights early vasopressin exposure and "
            "intensive-care unit mortality as an unresolved ICU research direction."
        ),
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "early vasopressin exposure",
                "outcome": "intensive-care unit mortality",
                "rationale": "The source describes this direction.",
                "source_quote": (
                    "early vasopressin exposure and intensive-care unit mortality"
                ),
                "analysis_family": "association",
            }
        ]
    )

    def fake_probe(**kwargs):
        pair = tuple(kwargs["concepts"])
        return {
            pair[0]: ca.RealDataConceptFeasibility(
                concept=pair[0],
                database="miiv",
                denominator_n=100,
                n_present=75,
                fraction_missing=0.25,
                n_joint_complete=75,
                joint_fraction_complete=0.75,
                missingness_severity="medium",
                structural_unavailable_concepts=[pair[1]],
                joint_denominator_concepts=[pair[0]],
                note=(
                    "structural unavailable concept(s) excluded from "
                    f"missingness denominator: {pair[1]}"
                ),
            )
        }

    result = run_idea_mining_dry_run(
        materials=[material],
        llm=llm,
        available_concepts=["adh_rate", "death"],
        output_dir=tmp_path / "dry_run",
        feasibility_probe=fake_probe,
        prior_art_search_client=FakePriorArtSearchClient(),
        prior_art_searched_at="2026-06-04T00:00:00+00:00",
    )

    assert result.feasibility_signals
    assert "structural unavailable" in (result.feasibility_signals[0].note or "")
    assert result.ranked_candidates
    assert "structural unavailable" in (
        result.ranked_candidates[0]["feasibility_note"] or ""
    )
    report = (tmp_path / "dry_run" / "discovery_report.md").read_text()
    assert "structural unavailable concept(s) excluded" in report


def test_dry_run_unknown_outcome_determinability_blocks_s1_without_extraction_failure(
    tmp_path,
) -> None:
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text=(
            "The review highlights biomarker pattern studies for "
            "patient-centered endpoint uncertainty."
        ),
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "biomarker pattern",
                "outcome": "patient-centered endpoint",
                "rationale": "The source describes this direction.",
                "source_quote": "biomarker pattern studies",
                "analysis_family": "association",
            }
        ]
    )
    concepts = [
        ConceptDescriptor(
            name="biomarker_pattern",
            source_concept="biomarker_pattern",
            derived_from_concepts=["biomarker pattern"],
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="endpoint_known",
            source_concept="endpoint_known",
            derived_from_concepts=["patient-centered endpoint"],
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]

    def should_not_probe(**kwargs):
        raise AssertionError("S1 must not run for unknown outcome determinability")

    result = run_idea_mining_dry_run(
        materials=[material],
        llm=llm,
        available_concepts=concepts,
        outcome_determinability={},
        output_dir=tmp_path / "dry_run",
        database="miiv",
        feasibility_probe=should_not_probe,
    )

    assert result.yield_report.n_literature_ideas == 1
    assert result.yield_report.n_executable == 0
    assert result.yield_report.n_non_executable == 1
    assert result.feasibility_signals == []
    assert result.ranked_candidates == []
    assert result.candidate_records[0].multiple_testing_family_size == 1
    assert result.candidate_records[0].multiple_testing_executable_family_size == 0
    assert "all-considered" in result.candidate_records[0].multiple_testing_note
    assert result.candidate_records[0].causal_audit_risk.startswith(
        "static_triage_marker"
    )
    assert any("outcome determinability" in warning for warning in result.warnings)
    assert any(
        "outcome determinability is unknown" in reason
        for record in result.candidate_records
        for reason in record.non_executable_reasons
    )


def test_dry_run_deduplicates_registry_denominator_before_reporting(tmp_path) -> None:
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text=(
            "The review highlights marker exposure for endpoint studies. "
            "A second passage also highlights marker exposure for endpoints."
        ),
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "marker exposure",
                "outcome": "patient-centered endpoint",
                "rationale": "First source quote.",
                "source_quote": "marker exposure for endpoint studies",
                "analysis_family": "association",
            },
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "marker exposure",
                "outcome": "patient-centered endpoint",
                "rationale": "Second source quote.",
                "source_quote": "marker exposure for endpoints",
                "analysis_family": "association",
            },
        ]
    )
    concepts = [
        ConceptDescriptor(
            name="marker",
            source_concept="marker",
            derived_from_concepts=["marker exposure"],
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="endpoint_known",
            source_concept="endpoint_known",
            derived_from_concepts=["patient-centered endpoint"],
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]

    def fake_probe(**kwargs):
        pair = tuple(kwargs["concepts"])
        return {
            pair[0]: {
                "joint_fraction_complete": 0.9,
                "n_joint_complete": 90,
                "denominator_n": 100,
                "source": "synthetic_s1_fixture",
            }
        }

    result = run_idea_mining_dry_run(
        materials=[material],
        llm=llm,
        available_concepts=concepts,
        outcome_determinability={"endpoint_known": "known_0_1"},
        output_dir=tmp_path / "dry_run",
        feasibility_probe=fake_probe,
        prior_art_search_client=FakePriorArtSearchClient(),
        prior_art_searched_at="2026-06-04T00:00:00+00:00",
    )

    registry = IdeaCandidateRegistry(result.registry_path)
    assert len(result.literature_ideas) == 2
    assert len(result.candidate_records) == 2
    assert len(registry.records) == 1
    assert {
        record.registry_candidate_id for record in result.candidate_records
    } == {registry.records[0].candidate_id}
    triage = json.loads((tmp_path / "dry_run" / "candidate_triage_report.json").read_text())
    assert triage["discovery_counts"]["literature_rows"] == 2
    assert triage["discovery_counts"]["unique_executable_hypotheses"] == 1
    assert triage["discovery_counts"]["multiple_testing_denominator"] == 1
    report = (tmp_path / "dry_run" / "discovery_report.md").read_text()
    assert "literature_rows=2" in report
    assert "unique_executable_hypotheses=1" in report


def test_dry_run_surfaces_unresolved_mapping_yield(tmp_path) -> None:
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text="The review highlights unlisted marker signals.",
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "unlisted marker",
                "outcome": "patient-centered endpoint",
                "rationale": "The source describes this direction.",
                "source_quote": "unlisted marker signals",
                "analysis_family": "association",
            }
        ]
    )
    concepts = [
        ConceptDescriptor(
            name="endpoint_known",
            source_concept="endpoint_known",
            derived_from_concepts=["patient-centered endpoint"],
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]

    result = run_idea_mining_dry_run(
        materials=[material],
        llm=llm,
        available_concepts=concepts,
        outcome_determinability={
            "endpoint_known": {"status": "known_0_1"},
        },
        output_dir=tmp_path / "dry_run",
        database="miiv",
    )

    assert result.yield_report.n_executable == 0
    assert result.candidate_records[0].multiple_testing_family_size == 1
    assert result.candidate_records[0].multiple_testing_executable_family_size == 0
    assert result.yield_report.unresolved_predictor_labels == ["unlisted marker"]
    assert any(
        "predictor concept is not available" in reason
        for reason in result.yield_report.top_non_executable_reasons
    )
