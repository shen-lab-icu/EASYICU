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
    material = SourceMaterial(
        citation=_citation(), source_adapter_level="metadata_only"
    )

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
    assert "measurable rule elements" in prompt_text
    assert "do not use abstract evaluation labels" in prompt_text


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


def test_prior_art_broad_query_uses_core_concept_facets_for_known_seed() -> None:
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients with shock",
        exposure_or_predictor="early vasopressor timing strategy after shock recognition",
        outcome="short-term patient-centered endpoint",
        rationale="The source points to early vasopressor timing.",
        source_quote="future work should study early vasopressor timing",
        analysis_family="association",
        time_window_hint="first six hours after ICU admission",
        exposure_core_concept="norepinephrine",
        outcome_core_concept="ICU mortality",
    )

    queries = build_prior_art_queries(idea)
    broad = queries["broad"]
    exact = queries["exact"]

    assert "norepinephrine[Title/Abstract]" in broad
    assert "noradrenaline[Title/Abstract]" in broad
    assert "mortality[Title/Abstract]" in broad
    assert "death[Title/Abstract]" in broad
    assert "norepinephrine[Title/Abstract]" not in exact
    assert "noradrenaline[Title/Abstract]" not in exact
    assert "short[Title/Abstract]" not in broad
    assert "term[Title/Abstract]" not in broad
    assert '"timing strategy"[Title/Abstract]' not in broad

    search = FakePriorArtSearchClient(
        {
            "norepinephrine[Title/Abstract]": {
                "hit_count": 1,
                "top_hits": [
                    {
                        "pmid": "40329359",
                        "title": "Early norepinephrine initiation and mortality in shock",
                        "abstract": "Meta-analysis of early norepinephrine and ICU mortality.",
                        "same_topic_screened": True,
                        "direct_same_topic": True,
                        "direct_same_topic_rationale": "Known same-topic seed recovered by core facets.",
                    }
                ],
            }
        }
    )
    assessment = assess_prior_art_for_idea(
        idea,
        search_client=search,
        searched_at="2026-06-06T00:00:00+00:00",
    )

    assert assessment.novelty_label == "already_done"
    assert assessment.direct_same_topic_pmids == ["40329359"]


def test_prior_art_broad_query_expands_peep_aki_core_facets() -> None:
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:def456",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult mechanically ventilated ICU patients",
        exposure_or_predictor="ventilatory pressure strategy",
        outcome="renal complication",
        rationale="The source suggests ventilation-kidney crosstalk.",
        source_quote="future work should study ventilation kidney interactions",
        analysis_family="association",
        exposure_core_concept="positive end-expiratory pressure",
        outcome_core_concept="acute kidney injury",
    )

    queries = build_prior_art_queries(idea)
    broad = queries["broad"]
    exact = queries["exact"]

    assert '"positive end-expiratory pressure"[Title/Abstract]' in broad
    assert "peep[Title/Abstract]" in broad
    assert '"acute kidney injury"[Title/Abstract]' in broad
    assert '"acute renal failure"[Title/Abstract]' in broad
    assert '"positive end-expiratory pressure"[Title/Abstract]' not in exact
    assert '"acute kidney injury"[Title/Abstract]' not in exact

    search = FakePriorArtSearchClient(
        {
            "peep[Title/Abstract]": {
                "hit_count": 2,
                "top_hits": [
                    {
                        "pmid": "111",
                        "title": "Positive end-expiratory pressure and acute kidney injury",
                        "abstract": "Review of PEEP and acute kidney injury.",
                        "same_topic_screened": True,
                        "direct_same_topic": True,
                        "direct_same_topic_rationale": "Known PEEP-AKI same-topic hit recovered.",
                    }
                ],
            }
        }
    )
    assessment = assess_prior_art_for_idea(
        idea,
        search_client=search,
        searched_at="2026-06-06T00:00:00+00:00",
    )

    assert assessment.novelty_label == "already_done"
    assert assessment.direct_same_topic_pmids == ["111"]


def test_prior_art_broad_query_does_not_overexpand_generic_singleton_facets() -> None:
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:pressure123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="mechanically ventilated patients",
        exposure_or_predictor="driving pressure",
        outcome="mortality",
        rationale="The source points to ventilator mechanics.",
        source_quote="future work should study driving pressure",
        analysis_family="association",
        time_window_hint="mechanical ventilation",
    )

    broad = build_prior_art_queries(idea)["broad"]

    assert '"driving pressure"[Title/Abstract]' in broad
    assert "driving[Title/Abstract]" not in broad
    assert "pressure[Title/Abstract]" not in broad


def test_prior_art_broad_query_expands_common_biomarker_abbreviations() -> None:
    rdw = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:rdw123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="critically ill patients",
        exposure_or_predictor="red cell distribution width",
        outcome="mortality",
        rationale="The source points to biomarker prognostication.",
        source_quote="future work should study red cell distribution width",
        analysis_family="association",
    )
    nlr = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:nlr123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="critically ill patients",
        exposure_or_predictor="neutrophil lymphocyte ratio",
        outcome="mortality",
        rationale="The source points to inflammatory biomarker prognostication.",
        source_quote="future work should study neutrophil lymphocyte ratio",
        analysis_family="association",
    )

    rdw_broad = build_prior_art_queries(rdw)["broad"]
    nlr_broad = build_prior_art_queries(nlr)["broad"]

    assert '"red cell distribution width"[Title/Abstract]' in rdw_broad
    assert '"red blood cell distribution width"[Title/Abstract]' in rdw_broad
    assert "rdw[Title/Abstract]" in rdw_broad
    assert "red[Title/Abstract]" not in rdw_broad
    assert "red[MeSH Terms]" not in rdw_broad

    assert '"neutrophil lymphocyte ratio"[Title/Abstract]' in nlr_broad
    assert '"neutrophil-to-lymphocyte ratio"[Title/Abstract]' in nlr_broad
    assert '"neutrophil to lymphocyte ratio"[Title/Abstract]' in nlr_broad
    assert "nlr[Title/Abstract]" in nlr_broad


def test_prior_art_broad_query_preserves_specific_icu_population_facets() -> None:
    ventilated = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:vent123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="mechanically ventilated patients",
        exposure_or_predictor="mechanical power",
        outcome="mortality",
        rationale="The source points to ventilator mechanics.",
        source_quote="future work should study mechanical power",
        analysis_family="association",
    )
    shock = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:shock123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="septic shock patients",
        exposure_or_predictor="vasopressor dose",
        outcome="mortality",
        rationale="The source points to shock resuscitation.",
        source_quote="future work should study vasopressor dose",
        analysis_family="association",
    )

    assert (
        '"mechanical ventilation"[Title/Abstract]'
        in build_prior_art_queries(ventilated)["broad"]
    )
    assert '"septic shock"[Title/Abstract]' in build_prior_art_queries(shock)["broad"]


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
    assert first.direct_same_topic_rationales == {"123": "same population and endpoint"}
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

    assert (
        assessment.same_topic_screen_status == "automated-substring-only, NOT screened"
    )
    assert assessment.novelty_label == "crowded_but_differentiable"
    assert assessment.literature_saturation_signal == pytest.approx(0.70)
    assert assessment.direct_same_topic_pmids == ["999"]
    exact = [
        record for record in assessment.query_records if record.query_type == "exact"
    ][0]
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


def test_extract_literature_ideas_truncates_overlong_traceable_quote() -> None:
    # A verbatim-but-over-long quote must not abort the run: it passes the
    # traceability check, then is truncated to the schema bound. The tool is
    # reviewer-facing -- one noisy extraction cannot crash the whole mining pass.
    from easyicu.research_agent.idea_mining import (
        _LITERATURE_IDEA_SOURCE_QUOTE_MAX as QUOTE_MAX,
    )

    long_quote = "lactate clearance predicts survival " * 40  # > 800 chars
    assert len(long_quote) > QUOTE_MAX
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text="Background. " + long_quote + " Future work is needed.",
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "lactate clearance",
                "outcome": "survival",
                "rationale": "Grounded but verbose.",
                "source_quote": long_quote,
                "analysis_family": "association",
            }
        ]
    )

    candidates = extract_literature_ideas(
        materials=[material],
        source_snapshot_id="source-snapshot/sha256:abc123",
        llm=llm,
    )

    assert len(candidates) == 1
    assert len(candidates[0].source_quote) <= QUOTE_MAX
    # The retained slice is still a verbatim prefix of the LLM's quote.
    assert long_quote.startswith(candidates[0].source_quote)


def test_extract_literature_ideas_skip_policy_drops_only_untraceable() -> None:
    """untraceable_quote_policy='skip' must drop ONLY the idea whose quote is
    not verbatim and keep the grounded ones, so one paraphrased quote does not
    discard an entire multi-article batch. The provenance gate still admits no
    unverbatim quote."""
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text="Lactate clearance was associated with survival in shock.",
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "lactate clearance",
                "outcome": "survival",
                "rationale": "Grounded.",
                "source_quote": "lactate clearance was associated with survival",
                "analysis_family": "association",
            },
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "marker",
                "outcome": "endpoint",
                "rationale": "Not anchored.",
                "source_quote": "a sentence that is not in the source",
                "analysis_family": "association",
            },
        ]
    )

    dropped: list[str] = []
    ideas = extract_literature_ideas(
        materials=[material],
        source_snapshot_id="source-snapshot/sha256:abc123",
        llm=llm,
        untraceable_quote_policy="skip",
        dropped_untraceable=dropped,
    )

    assert len(ideas) == 1
    assert ideas[0].exposure_or_predictor == "lactate clearance"
    assert dropped == ["neutral_review_2026"]


def test_extract_literature_ideas_skip_policy_drops_schema_invalid_item() -> None:
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text=(
            "Lactate clearance was associated with survival in shock. "
            "Future work should evaluate monitoring definitions."
        ),
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "adult ICU patients",
                "exposure_or_predictor": "lactate clearance",
                "outcome": "survival",
                "rationale": "Grounded.",
                "source_quote": "Lactate clearance was associated with survival",
                "analysis_family": "association",
            },
            {
                "citation_key": "neutral_review_2026",
                "population": "",
                "exposure_or_predictor": "",
                "outcome": "",
                "analysis_concepts": ["monitoring definitions"],
                "rationale": "Missing population should not abort the batch.",
                "source_quote": "Future work should evaluate monitoring definitions",
                "analysis_family": "data_quality_audit",
            },
        ]
    )

    dropped_invalid: list[str] = []
    ideas = extract_literature_ideas(
        materials=[material],
        source_snapshot_id="source-snapshot/sha256:abc123",
        llm=llm,
        untraceable_quote_policy="skip",
        dropped_invalid=dropped_invalid,
    )

    assert len(ideas) == 1
    assert ideas[0].exposure_or_predictor == "lactate clearance"
    assert dropped_invalid == ["neutral_review_2026"]


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


def test_non_binary_determinable_outcome_is_not_gated_as_unknown() -> None:
    # A continuous/ordinal outcome is determinable (the present/NA binary trap
    # does not apply). It must leave the dry run executable, not be blocked with
    # an "outcome determinability is unknown" reason.
    candidate = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="metadata_only",
        population="adult ICU patients",
        exposure_or_predictor="creatinine",
        outcome="organ dysfunction severity",
        rationale="The source suggests a severity-graded endpoint.",
        source_quote="future work should study severity scores",
    )
    concepts = [
        ConceptDescriptor(
            name="creatinine_first24h",
            source_concept="crea",
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="sofa_total",
            source_concept="sofa",
            derived_from_concepts=["organ dysfunction severity"],
            role=VariableRole.ORDINAL_SCORE,
            dtype="int64",
        ),
    ]

    executable = map_literature_idea_to_executable_candidate(
        candidate,
        available_concepts=concepts,
        outcome_determinability={
            "sofa": OutcomeDeterminability(
                outcome="sofa", status="non_binary_determinable"
            )
        },
    )

    assert executable.outcome_determinability_status == "non_binary_determinable"
    assert not any(
        "determinability is unknown" in reason
        for reason in executable.non_executable_reasons
    )
    assert executable.resolved_outcome_concept == "sofa"
    assert executable.executable


def test_ordinal_and_continuous_outcomes_get_non_binary_determinability_from_catalog() -> (
    None
):
    # End-to-end with the real catalog: ordinal scores and continuous labs are
    # auto-declared determinable, so the default idea-mining path stops gating
    # them out as "unknown".
    from easyicu.research_agent.concept_catalog import load_concept_catalog

    od = load_concept_catalog().outcome_determinability
    for key in ("sofa", "qsofa", "los_icu", "lact", "crea"):
        assert od[key]["status"] == "non_binary_determinable"
    # binary outcomes are unchanged; treatment/exposure concepts stay unscored.
    assert od["death"]["status"] == "known_0_1"
    assert "norepi60" not in od


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
    assert (
        "requires repeated measurements >=2" in blocked.feature_derivation_requirements
    )
    assert not blocked.executable
    assert any(
        "derived feature engineering" in reason
        for reason in blocked.non_executable_reasons
    )


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
    # The predictor "marker" is a vague construct with no substantive clinical
    # noun to query on, so a 0 hit count is an artifact, not novelty: the screen
    # must NOT call it a sparse gap. It is conservatively crowded/needs-human.
    assert assessment.novelty_label == "crowded_but_differentiable"
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
    exact = [
        record for record in assessment.query_records if record.query_type == "exact"
    ][0]
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
                            "direct_same_topic_rationale": ("same marker and endpoint"),
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
    assert {candidate["coverage_source"] for candidate in result.ranked_candidates} == {
        "pair_joint_feasibility"
    }
    assert all(
        record.multiple_testing_family_size == 2 for record in result.candidate_records
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
    assert assessment_by_phrase["physiologic marker"].novelty_label == "sparse"
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
    assert (
        "not a novelty claim"
        in (tmp_path / "dry_run" / "discovery_report.md").read_text()
    )
    registry = IdeaCandidateRegistry(result.registry_path)
    assert len(registry.records) == 2
    for record in result.candidate_records:
        with pytest.raises(RegistryCandidateNotExecutableError):
            registry.assert_executable(record.registry_candidate_id)
    assert (tmp_path / "dry_run" / "source_snapshot_manifest.json").exists()
    triage_report_path = tmp_path / "dry_run" / "candidate_triage_report.json"
    assert triage_report_path.exists()
    triage_payload = json.loads(triage_report_path.read_text())
    ledger = triage_payload["discovery_ledger"]
    assert len(ledger) == 2
    ledger_by_predictor = {
        row["resolved_predictor_concept"]: row for row in ledger
    }
    assert ledger_by_predictor["crea"]["candidate_topic"].startswith(
        "creatinine ->"
    )
    assert ledger_by_predictor["crea"]["go_no_go"] in {
        "recommend",
        "hold",
        "db-cannot-do",
    }
    assert ledger_by_predictor["crea"]["novelty_label"] == "already_done"
    assert ledger_by_predictor["crea"]["evidence_map_counts"][
        "direct_same_topic"
    ] >= 1
    assert ledger_by_predictor["crea"]["direct_same_topic_pmids"] == ["111"]

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


def test_dry_run_warns_when_all_sources_are_metadata_only(tmp_path) -> None:
    # metadata_only sources expose only title/venue/relevance, so gap mining has
    # no discussion/limitations text to work from. The dry run must surface that
    # the discovery yield is capped by source richness, not silently proceed.
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="metadata_only",
    )
    llm = CapturingIdeaLLM([])

    result = run_idea_mining_dry_run(
        materials=[material],
        llm=llm,
        available_concepts=["adh_rate", "death"],
        output_dir=tmp_path / "dry_run",
    )

    assert any(
        "metadata_only" in warning and "source richness" in warning
        for warning in result.warnings
    )


def test_dry_run_does_not_warn_metadata_only_for_excerpt_sources(tmp_path) -> None:
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text="The review names an unresolved direction in its limitations.",
    )
    llm = CapturingIdeaLLM([])

    result = run_idea_mining_dry_run(
        materials=[material],
        llm=llm,
        available_concepts=["adh_rate", "death"],
        output_dir=tmp_path / "dry_run",
    )

    assert not any("metadata_only" in warning for warning in result.warnings)


def test_dry_run_warns_on_degenerate_exposure_contrast(tmp_path) -> None:
    # Data may be fully present yet unanswerable: a single-valued predictor has
    # no exposure contrast. The dry run must request contrast for the PREDICTOR
    # only (never the outcome) and surface the degeneracy as a warning.
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
    contrast_requests: list[object] = []

    def fake_probe(**kwargs):
        pair = tuple(kwargs["concepts"])
        contrast_requests.append(kwargs.get("contrast_concepts"))
        return {
            concept: {
                "joint_fraction_complete": 0.9,
                "n_joint_complete": 90,
                "denominator_n": 100,
                "source": "synthetic_s1_fixture",
                # degenerate exposure contrast on the predictor only
                **({"predictor_contrast_fraction": 0.0} if concept == pair[0] else {}),
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

    # contrast requested for the predictor only, never the outcome
    assert contrast_requests == [["adh_rate"]]
    assert any(
        "no exposure contrast" in warning or "single-valued" in warning
        for warning in result.warnings
    )
    record = result.feasibility_signals[0]
    assert record.predictor_contrast_fraction == 0.0


def test_dry_run_tolerates_legacy_probe_without_contrast_kwarg(tmp_path) -> None:
    # A caller-injected probe may keep the legacy fixed signature (no
    # contrast_concepts, no **kwargs). The dry run must not pass the new kwarg
    # to it and must still complete; contrast is simply absent.
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

    def legacy_probe(
        *, concepts, database, data_path, cohort=None, analytic_unit="stay"
    ):
        return {
            concept: {
                "joint_fraction_complete": 0.8,
                "n_joint_complete": 80,
                "denominator_n": 100,
                "source": "legacy_fixture",
            }
            for concept in concepts
        }

    result = run_idea_mining_dry_run(
        materials=[material],
        llm=llm,
        available_concepts=["adh_rate", "death"],
        output_dir=tmp_path / "dry_run",
        feasibility_probe=legacy_probe,
    )

    assert result.yield_report.n_executable == 1
    record = result.feasibility_signals[0]
    assert record.predictor_contrast_fraction is None


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
    assert {record.registry_candidate_id for record in result.candidate_records} == {
        registry.records[0].candidate_id
    }
    triage = json.loads(
        (tmp_path / "dry_run" / "candidate_triage_report.json").read_text()
    )
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


class FakeScopeSearchClient:
    """Minimal source-search client: returns canned CitationRecords per query."""

    def __init__(self, records: Sequence[CitationRecord]):
        self.records = list(records)
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, *, retmax: int = 8) -> list[CitationRecord]:
        self.queries.append((query, retmax))
        return self.records[:retmax]


def test_fetch_source_materials_from_scope_builds_query_and_wraps_metadata() -> None:
    from easyicu.research_agent.idea_scope import LiteratureScopeSpec
    from easyicu.research_agent.idea_mining import fetch_source_materials_from_scope

    client = FakeScopeSearchClient(
        [
            CitationRecord(
                key="a_2025", title="A", year="2025", venue="Crit Care", pmid="1"
            ),
            CitationRecord(
                key="b_2025",
                title="B",
                year="2025",
                venue="Intensive Care Med",
                pmid="2",
            ),
        ]
    )
    scope = LiteratureScopeSpec(
        journal_preset="critical_care_top3",
        last_n_years=2,
        topic_terms=["septic shock"],
    )

    materials = fetch_source_materials_from_scope(
        scope, client, reference_year=2025, retmax=5
    )

    query, retmax = client.queries[0]
    assert retmax == 5
    assert "2024:2025[dp]" in query
    assert '"Crit Care"[Journal]' in query
    assert '"septic shock"' in query
    assert len(materials) == 2
    assert all(m.source_adapter_level == "metadata_only" for m in materials)
    assert all(m.source_text is None for m in materials)


def test_dry_run_scope_retrieves_corpus_and_freezes_query(tmp_path) -> None:
    from easyicu.research_agent.idea_scope import LiteratureScopeSpec

    quote = "early vasopressin exposure and intensive-care unit mortality"
    client = FakeScopeSearchClient(
        [
            CitationRecord(
                key="rev_2025",
                title="Vasopressin in critical illness: an unresolved review",
                year="2025",
                venue="Crit Care",
                relevance=f"The review calls for studies of {quote}.",
                pmid="1",
            )
        ]
    )
    scope = LiteratureScopeSpec(
        journal_preset="critical_care_top3",
        last_n_years=2,
        topic_terms=["vasopressin"],
    )
    llm = CapturingIdeaLLM(
        [
            {
                "citation_key": "rev_2025",
                "population": "adult ICU patients",
                "exposure_or_predictor": "early vasopressin exposure",
                "outcome": "intensive-care unit mortality",
                "rationale": "The source describes this direction.",
                "source_quote": quote,
                "analysis_family": "association",
            }
        ]
    )

    def fake_probe(**kwargs):
        pair = tuple(kwargs["concepts"])
        return {
            concept: {
                "joint_fraction_complete": 0.75,
                "n_joint_complete": 75,
                "denominator_n": 100,
                "source": "synthetic_s1_fixture",
            }
            for concept in pair
        }

    out_dir = tmp_path / "dry_run"
    result = run_idea_mining_dry_run(
        llm=llm,
        available_concepts=["adh_rate", "death"],
        output_dir=out_dir,
        feasibility_probe=fake_probe,
        scope=scope,
        source_search_client=client,
        scope_reference_year=2025,
        scope_retmax=10,
    )

    # The scope drove retrieval (no explicit materials passed).
    assert client.queries and client.queries[0][1] == 10
    assert result.yield_report.n_literature_ideas == 1
    assert result.yield_report.n_executable == 1

    scope_query_path = out_dir / "scope_query.json"
    assert scope_query_path.exists()
    frozen = json.loads(scope_query_path.read_text(encoding="utf-8"))
    assert "2024:2025[dp]" in frozen["pubmed_query"]
    assert frozen["n_materials_retrieved"] == 1
    assert frozen["scope_reference_year"] == 2025


def test_dry_run_scope_without_client_or_materials_fails_closed(tmp_path) -> None:
    from easyicu.research_agent.idea_scope import LiteratureScopeSpec
    from easyicu.research_agent.idea_mining import IdeaMiningError

    scope = LiteratureScopeSpec(journal_preset="critical_care_top3", last_n_years=2)
    with pytest.raises(IdeaMiningError, match="source_search_client"):
        run_idea_mining_dry_run(
            llm=CapturingIdeaLLM([]),
            available_concepts=["adh_rate", "death"],
            output_dir=tmp_path / "dry_run",
            scope=scope,
        )


def _generic_outcome_idea(outcome: str) -> LiteratureIdeaCandidate:
    return LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="critically ill ICU patients",
        exposure_or_predictor="sex",
        outcome=outcome,
        rationale="The source flags sex disparities in critical care as understudied.",
        source_quote="future research should examine sex differences in outcomes",
        analysis_family="association",
    )


def _sex_mortality_concepts() -> list:
    return [
        ConceptDescriptor(
            name="sex",
            source_concept="sex",
            role=VariableRole.DEMOGRAPHIC,
            dtype="object",
        ),
        ConceptDescriptor(
            name="death",
            source_concept="death",
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]


def test_generic_outcome_umbrella_normalizes_to_caller_mortality_default() -> None:
    # A resolvable predictor (sex) with a non-specific outcome umbrella
    # ("ICU outcomes") must NOT be a false db-cannot-do: it normalizes to the
    # caller-declared mortality determinable and records the substitution.
    executable = map_literature_idea_to_executable_candidate(
        _generic_outcome_idea("ICU outcomes"),
        available_concepts=_sex_mortality_concepts(),
        outcome_determinability={
            "mortality": OutcomeDeterminability(
                outcome="mortality",
                status="known_0_1",
                normalized_outcome_concept="death",
            ),
        },
    )
    assert executable.resolved_predictor_concept == "sex"
    assert executable.resolved_outcome_concept == "death"
    assert executable.normalized_outcome_concept == "death"
    assert executable.executable
    assert not any(
        "outcome concept is not available" in r
        for r in executable.non_executable_reasons
    )


def test_generic_outcome_keeps_original_label_for_human_confirmation() -> None:
    executable = map_literature_idea_to_executable_candidate(
        _generic_outcome_idea("poor prognosis"),
        available_concepts=_sex_mortality_concepts(),
        outcome_determinability={
            "mortality": OutcomeDeterminability(
                outcome="mortality",
                status="known_0_1",
                normalized_outcome_concept="death",
            ),
        },
    )
    # The original umbrella label is preserved; normalized_outcome_concept is the
    # visible signal that a human must confirm the mortality substitution.
    assert executable.outcome_label == "poor prognosis"
    assert executable.normalized_outcome_concept == "death"
    assert executable.executable


def test_non_generic_unavailable_outcome_still_blocks() -> None:
    # A specific outcome with no mapping ("neurological outcome") is NOT a generic
    # umbrella and must remain a genuine db-cannot-do -- the fix must not turn
    # every unresolved outcome into mortality.
    idea = _generic_outcome_idea("neurological outcome")
    executable = map_literature_idea_to_executable_candidate(
        idea,
        available_concepts=_sex_mortality_concepts(),
        outcome_determinability={
            "mortality": OutcomeDeterminability(
                outcome="mortality",
                status="known_0_1",
                normalized_outcome_concept="death",
            ),
        },
    )
    assert executable.resolved_outcome_concept is None
    assert not executable.executable
    assert any(
        "outcome concept is not available" in r
        for r in executable.non_executable_reasons
    )


def test_generic_outcome_without_mortality_default_still_blocks() -> None:
    # Case-neutral: with no caller-declared mortality default, a generic umbrella
    # cannot be silently invented into an outcome.
    executable = map_literature_idea_to_executable_candidate(
        _generic_outcome_idea("clinical outcomes"),
        available_concepts=_sex_mortality_concepts(),
        outcome_determinability={},
    )
    assert executable.resolved_outcome_concept is None
    assert not executable.executable


class _BatchAwareIdeaLLM:
    """Returns one idea per material present in each batch; records call count."""

    name = "batch-aware-idea-llm"

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def complete(
        self, messages, *, max_tokens: int = 2048, temperature: float = 0.0
    ) -> str:
        self.calls += 1
        payload = json.loads(messages[-1].content)
        sources = payload.get("sources", [])
        self.batch_sizes.append(len(sources))
        out = []
        for src in sources:
            text = src.get("available_source_text", "")
            quote = " ".join(text.split()[:6])  # traceable verbatim prefix
            out.append(
                {
                    "citation_key": src["citation_key"],
                    "population": "adult ICU patients",
                    "exposure_or_predictor": "serum lactate",
                    "outcome": "in-hospital mortality",
                    "rationale": "Open direction from the source.",
                    "source_quote": quote,
                    "analysis_family": "association",
                }
            )
        return json.dumps(out)


def _excerpt_material(idx: int) -> SourceMaterial:
    return SourceMaterial(
        citation=CitationRecord(
            key=f"review_{idx:02d}",
            title=f"ICU review {idx}",
            year="2026",
            venue="Critical Care",
        ),
        source_adapter_level="user_supplied_excerpt",
        source_text=(
            f"The authors of review {idx} note that an unresolved question "
            "remains for future study in critically ill patients."
        ),
    )


def test_extract_literature_ideas_batches_so_yield_scales_with_corpus() -> None:
    # 7 articles with batch_size=3 must produce 3 calls (3+3+1) and 7 ideas,
    # proving the corpus is fully processed rather than capped by one call.
    materials = [_excerpt_material(i) for i in range(7)]
    llm = _BatchAwareIdeaLLM()

    candidates = extract_literature_ideas(
        materials=materials,
        source_snapshot_id="source-snapshot/sha256:batch",
        llm=llm,
        batch_size=3,
    )

    assert llm.calls == 3
    assert llm.batch_sizes == [3, 3, 1]
    assert len(candidates) == 7
    assert {c.citation_key for c in candidates} == {f"review_{i:02d}" for i in range(7)}


def test_extract_literature_ideas_single_batch_when_corpus_small() -> None:
    materials = [_excerpt_material(i) for i in range(2)]
    llm = _BatchAwareIdeaLLM()

    candidates = extract_literature_ideas(
        materials=materials,
        source_snapshot_id="source-snapshot/sha256:batch",
        llm=llm,
        batch_size=6,
    )

    assert llm.calls == 1
    assert len(candidates) == 2


def test_label_prior_art_high_broad_count_blocks_false_sparse() -> None:
    # The bug: a heavily-studied pairing returns 0 on the over-specific exact
    # phrase but hundreds on broad recall; it must NOT be called sparse/gap.
    from easyicu.research_agent.idea_mining_priorart import _label_prior_art

    label = _label_prior_art(
        broad_count=300,
        exact_count=0,
        direct_same_topic_count=0,
        has_specific_differentiator=True,
    )
    assert label == "crowded_but_differentiable"
    assert label not in ("sparse", "apparently_gap")


def test_label_prior_art_genuinely_sparse_still_sparse() -> None:
    from easyicu.research_agent.idea_mining_priorart import _label_prior_art

    # Few broad hits and no exact hits remains a genuine gap/sparse signal.
    gap = _label_prior_art(
        broad_count=3,
        exact_count=0,
        direct_same_topic_count=0,
        has_specific_differentiator=True,
    )
    assert gap == "apparently_gap"
    sparse = _label_prior_art(
        broad_count=3,
        exact_count=0,
        direct_same_topic_count=0,
        has_specific_differentiator=False,
    )
    assert sparse == "sparse"


def test_label_prior_art_direct_hit_still_already_done() -> None:
    from easyicu.research_agent.idea_mining_priorart import _label_prior_art

    assert (
        _label_prior_art(broad_count=300, exact_count=10, direct_same_topic_count=2)
        == "already_done"
    )


def test_construct_is_vague_detection() -> None:
    from easyicu.research_agent.idea_mining_priorart import _construct_is_vague

    # vague: decorator/method shells with no substantive clinical noun
    assert _construct_is_vague("robust multiparametric clinical scores") is True
    assert _construct_is_vague("marker") is True
    assert _construct_is_vague("novel biomarkers") is True
    assert _construct_is_vague("") is True
    # concrete: a real measurable construct survives
    assert _construct_is_vague("serum lactate") is False
    assert _construct_is_vague("urea-to-creatinine ratio") is False
    assert _construct_is_vague("physiologic marker") is False  # "physiologic" survives


def test_label_prior_art_vague_construct_blocks_false_sparse() -> None:
    from easyicu.research_agent.idea_mining_priorart import _label_prior_art

    # Even with 0 broad/exact hits, a vague construct cannot be a sparse gap.
    assert (
        _label_prior_art(
            broad_count=0,
            exact_count=0,
            direct_same_topic_count=0,
            has_specific_differentiator=True,
            construct_is_concrete=False,
        )
        == "crowded_but_differentiable"
    )
    # A concrete construct with genuinely low counts is still a real gap.
    assert (
        _label_prior_art(
            broad_count=2,
            exact_count=0,
            direct_same_topic_count=0,
            has_specific_differentiator=True,
            construct_is_concrete=True,
        )
        == "apparently_gap"
    )


def _minimal_prior_art_assessment() -> object:
    """A neutral assessment; the go/no-go branch under test returns before it is read."""
    from easyicu.research_agent.idea_mining_priorart import PriorArtAssessment

    return PriorArtAssessment(
        novelty_snapshot_id="novelty-snapshot/sha256:deadbeef",
        literature_idea_id="litidea_test",
        source_snapshot_id="source-snapshot/sha256:abc123",
        searched_at="2026-06-15T00:00:00+00:00",
        predictor_literature_phrase="urine output",
        outcome_literature_phrase="successful CRRT liberation",
        query_records=[],
        novelty_label="sparse",
        literature_saturation_signal=0.0,
        novelty_statement="prior-art triage is not a novelty claim",
    )


def _candidate_with(
    *,
    predictor_concept,
    outcome_concept,
    non_executable_reasons,
    feature_derivation_status="raw_concept_available",
) -> ExecutableHypothesisCandidate:
    return ExecutableHypothesisCandidate(
        executable_candidate_id="execidea_test",
        literature_idea_id="litidea_test",
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        population="adult ICU patients",
        predictor_label="urine output",
        outcome_label="successful CRRT liberation",
        research_question="Does urine output predict CRRT liberation?",
        source_quote="future work should study CRRT liberation",
        resolved_predictor_concept=predictor_concept,
        resolved_outcome_concept=outcome_concept,
        feature_derivation_status=feature_derivation_status,
        non_executable_reasons=list(non_executable_reasons),
    )


def test_both_concepts_resolved_but_unknown_determinability_is_hold_not_db_cannot_do() -> (
    None
):
    # The defect this guards: a candidate whose predictor AND outcome both resolve
    # to real, present concepts was buried as "db-cannot-do" purely because the
    # outcome's determinability was never declared. The data IS present; the gap
    # is human operationalization of the outcome event -> a "hold", not a
    # database limitation. (This does not fabricate feasibility: the candidate
    # stays non-executable and unranked.)
    from easyicu.research_agent.idea_mining_priorart import _go_no_go_decision

    candidate = _candidate_with(
        predictor_concept="urine24",
        outcome_concept="rrt",
        non_executable_reasons=[
            "outcome determinability is unknown for feasibility probing"
        ],
    )
    assert not candidate.executable
    decision, reason = _go_no_go_decision(
        candidate=candidate,
        assessment=_minimal_prior_art_assessment(),
        triage=None,
    )
    assert decision == "hold"
    assert "operationaliz" in reason


def test_genuinely_absent_concept_stays_db_cannot_do() -> None:
    # The complement: when a concept is genuinely absent (predictor unresolved),
    # the verdict must remain "db-cannot-do" -- the fix must not reclassify a true
    # database limitation as a doable hold.
    from easyicu.research_agent.idea_mining_priorart import _go_no_go_decision

    candidate = _candidate_with(
        predictor_concept=None,
        outcome_concept="death",
        non_executable_reasons=[
            "predictor concept is not available: circulating renin level"
        ],
    )
    assert not candidate.executable
    decision, reason = _go_no_go_decision(
        candidate=candidate,
        assessment=_minimal_prior_art_assessment(),
        triage=None,
    )
    assert decision == "db-cannot-do"


def test_concept_set_clustering_idea_is_executable_without_predictor_outcome() -> None:
    # A subphenotype-clustering idea has no predictor->outcome pair: it names a
    # SET of variables to cluster on. Before shape-polymorphism this resolved to
    # predictor=None and was buried as db-cannot-do. It must now map to an
    # executable concept-SET candidate on its resolvable members.
    concepts = [
        ConceptDescriptor(
            name=name, source_concept=name, role=VariableRole.LAB, dtype="float64"
        )
        for name in ("lact", "crea", "sofa2", "map")
    ]
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult sepsis ICU patients",
        analysis_family="subphenotype_clustering",
        rationale="The review calls for data-driven sepsis subphenotypes.",
        source_quote="future work should identify sepsis subphenotypes",
        analysis_concepts=["lact", "crea", "sofa2", "frailty index"],
    )

    candidate = map_literature_idea_to_executable_candidate(
        idea, available_concepts=concepts
    )

    assert candidate.analysis_family == "trajectory_clustering"
    assert candidate.executable
    assert set(candidate.resolved_analysis_concepts) == {"lact", "crea", "sofa2"}
    assert candidate.resolved_predictor_concept is None
    assert candidate.resolved_outcome_concept is None
    # The unresolved member is recorded as a note, not a hard block.
    assert candidate.feature_derivation_note is not None
    assert "frailty index" in candidate.feature_derivation_note


def test_concept_set_clustering_idea_too_few_concepts_is_not_executable() -> None:
    # Clustering needs >=2 resolvable variables; one resolvable concept is not a
    # cluster space, so it stays non-executable (and the gate keeps it honest).
    concepts = [
        ConceptDescriptor(
            name="crea", source_concept="crea", role=VariableRole.LAB, dtype="float64"
        )
    ]
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        analysis_family="clustering",
        rationale="Calls for phenotypes.",
        source_quote="future work should identify phenotypes",
        analysis_concepts=["crea", "unmappable biomarker"],
    )

    candidate = map_literature_idea_to_executable_candidate(
        idea, available_concepts=concepts
    )

    assert candidate.analysis_family == "trajectory_clustering"
    assert not candidate.executable
    assert any("at least 2" in r for r in candidate.non_executable_reasons)


def test_pairwise_trajectory_idea_stays_on_predictor_outcome_path() -> None:
    # A "trajectory" predictor->outcome association is pairwise, NOT clustering:
    # the presence of a real pair must keep it on the predictor->outcome path so
    # existing behavior is preserved despite the trajectory family alias.
    concepts = [
        ConceptDescriptor(
            name="lact", source_concept="lact", role=VariableRole.LAB, dtype="float64"
        ),
        ConceptDescriptor(
            name="death",
            source_concept="death",
            role=VariableRole.OUTCOME,
            dtype="int64",
        ),
    ]
    idea = LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        exposure_or_predictor="lact",
        outcome="death",
        analysis_family="trajectory",
        rationale="Trajectory predictor of mortality.",
        source_quote="future work should study lactate trajectory and mortality",
    )

    candidate = map_literature_idea_to_executable_candidate(
        idea,
        available_concepts=concepts,
        outcome_determinability={
            "death": OutcomeDeterminability(outcome="death", status="known_0_1")
        },
    )

    assert candidate.resolved_predictor_concept == "lact"
    assert candidate.resolved_outcome_concept == "death"
    assert candidate.resolved_analysis_concepts == []


class SequenceIdeaLLM:
    """Mock LLM returning a scripted response per call (extract, then refine...)."""

    name = "sequence-idea-llm"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, **kwargs):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


def _reflection_material() -> SourceMaterial:
    return SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text=(
            "Future work should identify sepsis subphenotypes from lactate and "
            "creatinine, and study biomarkers broadly."
        ),
    )


def test_reflection_round_drops_vague_idea_and_keeps_refined() -> None:
    # Phase 2: a reflection round sharpens/keeps grounded ideas and drops a vague
    # placeholder ("biomarkers") that the source does not name concretely.
    extract = json.dumps(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "sepsis ICU",
                "exposure_or_predictor": "biomarkers",
                "outcome": "outcomes",
                "rationale": "vague",
                "source_quote": "study biomarkers broadly",
                "analysis_family": "association",
            },
            {
                "citation_key": "neutral_review_2026",
                "population": "sepsis ICU",
                "analysis_concepts": ["lactate", "creatinine"],
                "analysis_family": "subphenotype_clustering",
                "rationale": "cluster",
                "source_quote": "identify sepsis subphenotypes from lactate and creatinine",
            },
        ]
    )
    refine = json.dumps(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "sepsis ICU",
                "analysis_concepts": ["lactate", "creatinine"],
                "analysis_family": "subphenotype_clustering",
                "rationale": "cluster on physiology",
                "source_quote": "identify sepsis subphenotypes from lactate and creatinine",
            }
        ]
    )

    ideas = extract_literature_ideas(
        materials=[_reflection_material()],
        source_snapshot_id="source-snapshot/sha256:abc123",
        llm=SequenceIdeaLLM([extract, refine]),
        untraceable_quote_policy="skip",
        reflection_rounds=1,
    )

    assert len(ideas) == 1
    assert ideas[0].analysis_family == "subphenotype_clustering"
    assert ideas[0].analysis_concepts == ["lactate", "creatinine"]


def test_reflection_zero_rounds_is_single_pass() -> None:
    # Back-compat: reflection_rounds=0 must not call the refine step at all.
    extract = json.dumps(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "sepsis ICU",
                "exposure_or_predictor": "lactate",
                "outcome": "mortality",
                "rationale": "grounded",
                "source_quote": "lactate and creatinine",
                "analysis_family": "association",
            }
        ]
    )
    llm = SequenceIdeaLLM([extract])
    ideas = extract_literature_ideas(
        materials=[_reflection_material()],
        source_snapshot_id="source-snapshot/sha256:abc123",
        llm=llm,
        untraceable_quote_policy="skip",
        reflection_rounds=0,
    )
    assert len(ideas) == 1
    assert llm.calls == 1  # only the extraction call, no reflection call


def test_reflection_drops_idea_with_tampered_untraceable_quote() -> None:
    # Provenance is never weakened by reflection: if the refine step returns a
    # quote that is not verbatim in the source, that idea is dropped.
    extract = json.dumps(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "sepsis ICU",
                "exposure_or_predictor": "lactate",
                "outcome": "mortality",
                "rationale": "grounded",
                "source_quote": "lactate and creatinine",
                "analysis_family": "association",
            }
        ]
    )
    refine = json.dumps(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "sepsis ICU",
                "exposure_or_predictor": "lactate clearance",
                "outcome": "mortality",
                "rationale": "tampered",
                "source_quote": "a fabricated quote not in the source",
                "analysis_family": "association",
            }
        ]
    )
    # The refine round drops the only idea (untraceable) -> no-op fallback returns
    # the original grounded idea unchanged.
    ideas = extract_literature_ideas(
        materials=[_reflection_material()],
        source_snapshot_id="source-snapshot/sha256:abc123",
        llm=SequenceIdeaLLM([extract, refine]),
        untraceable_quote_policy="skip",
        reflection_rounds=1,
    )
    assert len(ideas) == 1
    assert ideas[0].source_quote == "lactate and creatinine"


def _sparse_idea_and_search():
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
    )
    search = FakePriorArtSearchClient(
        {
            "vasopressin[Title/Abstract]": {
                "hit_count": 17,
                "top_hits": [
                    {
                        "pmid": "777",
                        "title": "Vasopressin and mortality in critically ill adults",
                        "abstract": "Adjacent prior-art hit.",
                    }
                ],
            }
        }
    )
    return idea, search


def test_novelty_judge_duplicate_verdict_tightens_label() -> None:
    # Phase 3: the count screen alone labels this "sparse"; an LLM judge that
    # reads the hits and calls it a duplicate must tighten it to already_done.
    from easyicu.research_agent.idea_mining_priorart import assess_prior_art_for_idea

    idea, search = _sparse_idea_and_search()
    baseline = assess_prior_art_for_idea(
        idea, search_client=search, searched_at="2026-06-15T00:00:00+00:00"
    )
    assert baseline.novelty_label == "sparse"

    def judge(*, idea, executable_candidate, hits, count_label):
        assert hits  # the judge is handed the prior-art titles
        return {"verdict": "duplicate", "rationale": "PMID 777 is the same study."}

    idea2, search2 = _sparse_idea_and_search()
    judged = assess_prior_art_for_idea(
        idea2,
        search_client=search2,
        searched_at="2026-06-15T00:00:00+00:00",
        novelty_judge=judge,
    )
    assert judged.novelty_label == "already_done"
    assert "same study" in judged.novelty_statement


def test_novelty_judge_cannot_upgrade_label() -> None:
    # Veto-net: a "differentiated" verdict must NOT make a sparse label look more
    # novel; the count screen remains the ceiling on novelty.
    from easyicu.research_agent.idea_mining_priorart import assess_prior_art_for_idea

    idea, search = _sparse_idea_and_search()

    def generous_judge(*, idea, executable_candidate, hits, count_label):
        return {"verdict": "differentiated", "rationale": "Looks novel to me."}

    judged = assess_prior_art_for_idea(
        idea,
        search_client=search,
        searched_at="2026-06-15T00:00:00+00:00",
        novelty_judge=generous_judge,
    )
    # stays sparse (cannot be pushed to apparently_gap)
    assert judged.novelty_label == "sparse"


def test_more_conservative_novelty_helper() -> None:
    from easyicu.research_agent.idea_mining_priorart import _more_conservative_novelty

    assert (
        _more_conservative_novelty("apparently_gap", "already_done") == "already_done"
    )
    assert (
        _more_conservative_novelty("sparse", "crowded_but_differentiable")
        == "crowded_but_differentiable"
    )
    assert _more_conservative_novelty("already_done", "sparse") == "already_done"


def test_novelty_judge_error_is_best_effort_noop() -> None:
    # A judge that raises must leave the count label untouched (best-effort).
    from easyicu.research_agent.idea_mining_priorart import assess_prior_art_for_idea

    idea, search = _sparse_idea_and_search()

    def boom(*, idea, executable_candidate, hits, count_label):
        raise RuntimeError("judge unavailable")

    judged = assess_prior_art_for_idea(
        idea,
        search_client=search,
        searched_at="2026-06-15T00:00:00+00:00",
        novelty_judge=boom,
    )
    assert judged.novelty_label == "sparse"


def _idea(
    predictor="", outcome="", concepts=None, family="association", quote="q in src"
):
    return LiteratureIdeaCandidate(
        source_snapshot_id="source-snapshot/sha256:abc123",
        citation_key="neutral_review_2026",
        source_adapter_level="user_supplied_excerpt",
        population="adult ICU patients",
        exposure_or_predictor=predictor,
        outcome=outcome,
        analysis_concepts=list(concepts or []),
        analysis_family=family,
        rationale="r",
        source_quote=quote,
    )


def test_collapse_near_duplicate_ideas_archive() -> None:
    # Phase 2b idea archive: the same construct restated (token-order/case/synonym
    # noise) collapses to one; genuinely distinct ideas survive.
    from easyicu.research_agent.idea_mining import _collapse_near_duplicate_ideas

    ideas = [
        _idea(predictor="lactate clearance", outcome="mortality"),
        _idea(predictor="Lactate  clearance", outcome="Mortality"),  # dup
        _idea(predictor="creatinine", outcome="mortality"),  # distinct
        _idea(concepts=["compliance", "driving pressure"], family="clustering"),
        _idea(concepts=["driving pressure", "compliance"], family="clustering"),  # dup
    ]
    collapsed = _collapse_near_duplicate_ideas(ideas)
    assert len(collapsed) == 3


def test_reflection_prompt_includes_prior_art_titles_when_supplied() -> None:
    # Phase 2b retrieval augmentation: supplied prior-art titles surface in the
    # reflection prompt with the drop-or-differentiate instruction.
    from easyicu.research_agent.idea_mining import build_idea_reflection_messages

    ideas = [_idea(predictor="lactate", outcome="mortality")]
    material = SourceMaterial(
        citation=_citation(),
        source_adapter_level="user_supplied_excerpt",
        source_text="lactate and mortality discussion",
    )
    messages = build_idea_reflection_messages(
        ideas,
        materials=[material],
        source_snapshot_id="source-snapshot/sha256:abc123",
        round_idx=0,
        num_rounds=1,
        prior_art_titles=[["Lactate predicts ICU mortality: a meta-analysis"]],
    )
    blob = "\n".join(m.content for m in messages)
    assert "Lactate predicts ICU mortality" in blob
    assert "prior_art_titles" in blob
    assert "already well studied" in blob


def test_retrieval_augmented_reflection_uses_search_client() -> None:
    # End-to-end: a search client supplied to extraction is queried during the
    # reflection round (retrieval-augmented), and near-dups are collapsed after.
    extract = json.dumps(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "sepsis ICU",
                "exposure_or_predictor": "lactate",
                "outcome": "mortality",
                "rationale": "grounded",
                "source_quote": "lactate and creatinine",
                "analysis_family": "association",
            }
        ]
    )
    refine = extract  # model keeps the (grounded) idea
    material = _reflection_material()
    search = FakePriorArtSearchClient(
        {"lact": {"hit_count": 9, "top_hits": [{"title": "Lactate and ICU mortality"}]}}
    )
    ideas = extract_literature_ideas(
        materials=[material],
        source_snapshot_id="source-snapshot/sha256:abc123",
        llm=SequenceIdeaLLM([extract, refine]),
        untraceable_quote_policy="skip",
        reflection_rounds=1,
        reflection_search_client=search,
    )
    assert len(ideas) == 1
    # the reflection round issued at least one prior-art search
    assert search.queries


def test_reflection_strips_echoed_context_fields() -> None:
    # The model may echo the injected prior_art_titles context back into a refined
    # idea; the schema forbids extras, so it must be stripped, not crash the run.
    extract = json.dumps(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "sepsis ICU",
                "exposure_or_predictor": "lactate",
                "outcome": "mortality",
                "rationale": "grounded",
                "source_quote": "lactate and creatinine",
                "analysis_family": "association",
            }
        ]
    )
    refine = json.dumps(
        [
            {
                "citation_key": "neutral_review_2026",
                "population": "sepsis ICU",
                "exposure_or_predictor": "lactate",
                "outcome": "mortality",
                "rationale": "grounded",
                "source_quote": "lactate and creatinine",
                "analysis_family": "association",
                "prior_art_titles": ["Lactate and ICU mortality: a meta-analysis"],
            }
        ]
    )
    ideas = extract_literature_ideas(
        materials=[_reflection_material()],
        source_snapshot_id="source-snapshot/sha256:abc123",
        llm=SequenceIdeaLLM([extract, refine]),
        untraceable_quote_policy="skip",
        reflection_rounds=1,
    )
    assert len(ideas) == 1
    assert ideas[0].exposure_or_predictor == "lactate"
