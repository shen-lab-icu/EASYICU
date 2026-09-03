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
        design_selection={
            "schema_version": "easyicu.research_design_selection/1",
            "claim_ceiling": "analysis_only",
            "candidates": [
                {
                    "design_id": "adjusted_primary",
                    "analysis_type": "association",
                    "estimand": "Adjusted exposure contrast for hospital mortality.",
                    "time_zero": "Start of the eligible ICU episode.",
                    "observation_window": "Exposure in 0-24h and hospital mortality follow-up.",
                    "primary_method": "Adjusted logistic regression",
                    "required_variables": ["exposure", "death"],
                    "assumptions": ["The declared adjustment set is adequate."],
                    "literature_citation_keys": ["direct_2025"],
                    "novelty_positioning": "Compare the ICU exposure definition with prior cohorts.",
                    "figure_role": "Show the adjusted estimate and uncertainty.",
                    "supports": "A prespecified adjusted association estimate.",
                    "cannot_prove": "A causal effect without stronger identification.",
                    "disposition": "selected",
                    "decision_reason": "The exposure and mortality anchors require adjusted estimation.",
                },
                {
                    "design_id": "crude_alternative",
                    "analysis_type": "association",
                    "estimand": "Unadjusted exposure contrast for hospital mortality.",
                    "time_zero": "Start of the eligible ICU episode.",
                    "observation_window": "Exposure in 0-24h and hospital mortality follow-up.",
                    "primary_method": "Unadjusted risk comparison",
                    "required_variables": ["exposure", "death"],
                    "assumptions": ["Crude group differences are interpretable."],
                    "literature_citation_keys": ["direct_2025"],
                    "novelty_positioning": "Provide a crude comparator to prior cohorts.",
                    "figure_role": "Show the crude outcome difference.",
                    "supports": "A descriptive group contrast.",
                    "cannot_prove": "An adjusted or causal exposure effect.",
                    "disposition": "rejected",
                    "decision_reason": "Reject because confounding is central to the exposure question.",
                },
            ],
        },
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
    assert packet.comparison_dimensions["data_source_and_transportability"].study
    assert packet.comparison_dimensions[
        "clinical_decision_or_methodological_contribution"
    ].study
    assert (
        "selected_estimand=Adjusted exposure contrast"
        in packet.comparison_dimensions["outcome_and_estimand"].study
    )
    assert (
        "selected_method=Adjusted logistic regression"
        in packet.comparison_dimensions["analysis_and_robustness_route"].study
    )
    assert (
        "cannot_prove=A causal effect"
        in packet.comparison_dimensions[
            "clinical_decision_or_methodological_contribution"
        ].study
    )
    assert (
        packet.comparison_dimensions["analysis_and_robustness_route"].comparator is None
    )
    assert (
        packet.comparison_dimensions["analysis_and_robustness_route"].difference is None
    )
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


def test_design_analogue_opens_review_without_claiming_direct_comparison() -> None:
    context = _context().model_copy(
        update={
            "research_question": "Identify ICU subphenotypes by clustering.",
            "primary_exposure": None,
        }
    )
    plan = _plan().model_copy(
        update={
            "steps": [
                AnalysisStep(
                    step_id="primary",
                    planned_analysis_role="primary",
                    intent="Identify stable ICU subphenotypes.",
                    method="unsupervised clustering",
                    expected_outputs=["table:cluster_profiles"],
                    literature_citation_keys=["analogue_2025"],
                )
            ]
        }
    )
    literature = LiteratureBundle(
        research_question=context.research_question,
        citations=[
            CitationRecord(
                key="analogue_2025",
                title="ICU subphenotype clustering cohort",
                year="2025",
                relevance="Study-design excerpt: Adult ICU clustering cohort.",
            )
        ],
        screening_decisions=[
            LiteratureScreeningDecision(
                citation_key="analogue_2025",
                source="pubmed",
                disposition="include",
                evidence_role="design_analogue",
                rationale="Topic and analysis-design intent matched.",
                population_match=True,
                exposure_match=False,
                outcome_match=False,
                design_excerpt_available=True,
            )
        ],
    )

    packet = build_unsigned_novelty_positioning_packet(
        context=context,
        plan=plan,
        literature=literature,
    )

    assert packet.schema_version == "easyicu.novelty_positioning/3"
    assert packet.status == "review_required"
    assert packet.direct_comparator_keys == []
    assert packet.design_analogue_keys == ["analogue_2025"]
    assert packet.comparators[0].evidence_role == "design_analogue"

    reviewed = packet.model_dump(mode="json")
    reviewed.update(
        status="supported",
        review_disposition="independent_pre_review_pass",
        reviewer_owner="Independent methods reviewer",
    )
    for dimension in reviewed["comparison_dimensions"].values():
        dimension.update(
            comparator="Source-reviewed comparator design.",
            difference="The reviewed study differs on this prespecified axis.",
            source_status="independent_reviewed",
        )
    accepted = NoveltyPositioningPacket.model_validate(reviewed)
    assert accepted.status == "supported"
    assert accepted.direct_comparator_keys == []
    assert accepted.design_analogue_keys == ["analogue_2025"]


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
