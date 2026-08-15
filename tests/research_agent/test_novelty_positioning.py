import pytest

from easyicu.research_agent.literature import (
    CitationRecord,
    LiteratureBundle,
    LiteratureScreeningDecision,
)
from easyicu.research_agent.reporting.novelty_positioning import (
    NoveltyPositioningPacket,
    build_unsigned_novelty_positioning_packet,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Is a 0-24h ICU exposure associated with hospital mortality?",
        cohort=CohortDescriptor(
            cohort_name="all stays",
            database="miiv",
            n_stays=100,
            inclusion_criteria=["all eligible ICU stays"],
            id_columns=["stay_id"],
        ),
        variables=[
            ConceptDescriptor(
                name="exposure",
                role="composite_score",
                dtype="int64",
                description="0-24h exposure",
                source_concept="registered_exposure",
                analysis_window="0-24h",
            ),
            ConceptDescriptor(
                name="death",
                role="outcome",
                dtype="int64",
                description="hospital mortality",
                source_concept="hospital_mortality",
            ),
        ],
        primary_exposure="exposure",
        target_outcome="death",
    )


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Question",
        analysis_type="association",
        steps=[
            AnalysisStep(
                step_id="primary",
                planned_analysis_role="primary",
                intent="Estimate the unadjusted association.",
                method="logistic regression",
                expected_outputs=["table:association_estimate"],
                literature_citation_keys=["direct_2025", "strobe_2007"],
            )
        ],
    )


def test_unsigned_packet_exposes_comparator_but_never_claims_novelty() -> None:
    literature = LiteratureBundle(
        research_question="Question",
        citations=[
            CitationRecord(
                key="direct_2025",
                title="Comparable ICU cohort",
                year="2025",
                relevance=(
                    "Study-design excerpt: Adult ICU patients were classified by "
                    "the exposure and followed for hospital mortality."
                ),
            )
        ],
        screening_decisions=[
            LiteratureScreeningDecision(
                citation_key="direct_2025",
                source="pubmed",
                disposition="include",
                evidence_role="direct_comparator",
                rationale="P/E/O matched from the retained abstract.",
                population_match=True,
                exposure_match=True,
                outcome_match=True,
                design_excerpt_available=True,
            )
        ],
    )

    packet = build_unsigned_novelty_positioning_packet(
        context=_context(),
        plan=_plan(),
        literature=literature,
    )

    assert packet.status == "review_required"
    assert packet.review_disposition == "not_available"
    assert packet.direct_comparator_keys == ["direct_2025"]
    assert packet.comparators[0].source_excerpt.startswith("Study-design excerpt:")
    assert packet.comparison_dimensions["population_and_setting"].study
    assert packet.comparison_dimensions["analysis_and_robustness_route"].study
    assert packet.comparison_dimensions[
        "data_source_and_transportability"
    ].study
    assert packet.comparison_dimensions[
        "clinical_decision_or_methodological_contribution"
    ].study
    assert packet.comparison_dimensions[
        "analysis_and_robustness_route"
    ].comparator is None
    assert packet.comparison_dimensions[
        "analysis_and_robustness_route"
    ].difference is None
    assert len(packet.context_sha256) == 64
    assert len(packet.plan_sha256) == 64
    assert len(packet.literature_sha256) == 64


def test_no_direct_comparator_remains_not_established() -> None:
    packet = build_unsigned_novelty_positioning_packet(
        context=_context(),
        plan=_plan(),
        literature=LiteratureBundle(research_question="Question", citations=[]),
    )

    assert packet.status == "not_established"
    assert packet.direct_comparator_keys == []
    assert packet.comparators == []


def test_supported_packet_requires_external_owner_and_completed_dimensions() -> None:
    unsigned = build_unsigned_novelty_positioning_packet(
        context=_context(),
        plan=_plan(),
        literature=LiteratureBundle(
            research_question="Question",
            citations=[
                CitationRecord(
                    key="direct_2025",
                    title="Comparable ICU cohort",
                    year="2025",
                    relevance="Study-design excerpt: Direct comparator.",
                )
            ],
            screening_decisions=[
                LiteratureScreeningDecision(
                    citation_key="direct_2025",
                    source="pubmed",
                    disposition="include",
                    evidence_role="direct_comparator",
                    rationale="P/E/O matched.",
                    population_match=True,
                    exposure_match=True,
                    outcome_match=True,
                    design_excerpt_available=True,
                )
            ],
        ),
    ).model_dump(mode="json")
    unsigned.update(
        status="supported",
        review_disposition="independent_pre_review_pass",
    )

    with pytest.raises(ValueError, match="reviewer_owner"):
        NoveltyPositioningPacket.model_validate(unsigned)

    unsigned["reviewer_owner"] = "External reviewer"
    with pytest.raises(ValueError, match="every dimension"):
        NoveltyPositioningPacket.model_validate(unsigned)
