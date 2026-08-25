from __future__ import annotations

from datetime import datetime, timezone

import pytest

from easyicu.research_agent.literature import (
    CitationRecord,
    LiteratureBundle,
    LiteratureScreeningDecision,
    render_hypothesis_blueprint_for_prompt,
)
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.profiles import (
    DEV9_AI_REVIEWED_DEMO_2026_08_24,
    QUALIFICATION12_LITERATURE_DESIGN_2026_08_25,
)
from easyicu.research_agent.planning.design_selection import (
    ResearchDesignCandidate,
    ResearchDesignSelection,
)
from easyicu.research_agent.planning.literature_design_authority import (
    CandidateLiteratureDesignDecision,
    LITERATURE_DESIGN_DIMENSIONS,
    LiteratureDesignAuthorityError,
    LiteratureDesignEvidence,
    LiteratureDesignEvidenceCard,
    validate_preplan_literature_design_authority,
    validate_selected_design_against_literature,
)
from easyicu.research_agent.schema import HypothesisBlueprint


def _card(
    *,
    supplement_status: str = "not_published",
    dimensions: tuple[str, ...] = LITERATURE_DESIGN_DIMENSIONS,
) -> LiteratureDesignEvidenceCard:
    return LiteratureDesignEvidenceCard(
        citation_key="comparator_2025",
        evidence_role="direct_comparator",
        access_mode="open_access_fulltext",
        full_text_locator="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
        full_text_sha256="a" * 64,
        supplement_status=supplement_status,
        supplement_sha256=("b" * 64 if supplement_status == "reviewed" else None),
        reviewed_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        evidence=[
            LiteratureDesignEvidence(
                dimension=dimension,
                source_backed_summary=f"Reviewed source fact for {dimension} in this study.",
                locator=f"section:{dimension}",
            )
            for dimension in dimensions
        ],
    )


def _bundle(*, card: LiteratureDesignEvidenceCard | None = None) -> LiteratureBundle:
    return LiteratureBundle(
        research_question="Does exposure predict outcome in ICU patients?",
        citations=[
            CitationRecord(
                key="comparator_2025",
                title="A recent comparable ICU cohort study",
                year="2025",
                venue="Example Journal",
                pmid="12345678",
            )
        ],
        screening_decisions=[
            LiteratureScreeningDecision(
                citation_key="comparator_2025",
                source="pubmed",
                disposition="include",
                evidence_role="direct_comparator",
                rationale="Population, exposure, and outcome match.",
                population_match=True,
                exposure_match=True,
                outcome_match=True,
                design_excerpt_available=True,
            )
        ],
        design_evidence_cards=[] if card is None else [card],
    )


def _selection(*, include_all_dimensions: bool = True) -> ResearchDesignSelection:
    decisions = [
        CandidateLiteratureDesignDecision(
            dimension=dimension,
            citation_keys=["comparator_2025"],
            disposition="adapt",
            rationale=f"Adapt the reviewed {dimension} pattern to this exact cohort.",
        )
        for dimension in (
            LITERATURE_DESIGN_DIMENSIONS if include_all_dimensions else LITERATURE_DESIGN_DIMENSIONS[:-1]
        )
    ]
    shared = {
        "analysis_type": "logistic_regression",
        "required_variables": ["exposure", "outcome"],
        "assumptions": ["No unmeasured confounding is claimed."],
        "literature_citation_keys": ["comparator_2025"],
        "novelty_positioning": "Tests the question in a distinct governed ICU cohort.",
        "figure_role": "Shows the adjusted association and its uncertainty.",
        "supports": "Supports a bounded association estimate in this cohort.",
        "cannot_prove": "Cannot prove causality or transportability.",
    }
    return ResearchDesignSelection(
        candidates=[
            ResearchDesignCandidate(
                design_id="selected_primary",
                estimand="Adjusted odds ratio for the prespecified outcome.",
                time_zero="ICU admission defines time zero for every patient.",
                observation_window="Exposure uses the first 24 hours after ICU admission.",
                primary_method="Multivariable logistic regression",
                disposition="selected",
                decision_reason="Best matches the prespecified estimand and available timing.",
                literature_design_decisions=decisions,
                **shared,
            ),
            ResearchDesignCandidate(
                design_id="rejected_landmark",
                estimand="Risk difference at a later fixed landmark among survivors.",
                time_zero="A 24-hour landmark defines eligibility for this alternative.",
                observation_window="Exposure is summarized before the 24-hour landmark.",
                primary_method="Landmark risk regression",
                disposition="rejected",
                decision_reason="Changes the target population by conditioning on survival.",
                literature_design_decisions=[],
                **shared,
            ),
        ]
    )


def test_preplan_authority_accepts_reviewed_fulltext_and_complete_dimensions() -> None:
    validate_preplan_literature_design_authority(_bundle(card=_card()))


def test_preplan_authority_rejects_abstract_only_comparator() -> None:
    with pytest.raises(LiteratureDesignAuthorityError) as exc_info:
        validate_preplan_literature_design_authority(_bundle())
    assert exc_info.value.reason_code == "literature_fulltext_design_card_missing"


def test_preplan_authority_rejects_unreviewed_supplement() -> None:
    with pytest.raises(LiteratureDesignAuthorityError) as exc_info:
        validate_preplan_literature_design_authority(
            _bundle(card=_card(supplement_status="published_unreviewed"))
        )
    assert exc_info.value.reason_code == "literature_supplement_review_incomplete"


def test_preplan_authority_rejects_missing_design_dimension() -> None:
    with pytest.raises(LiteratureDesignAuthorityError) as exc_info:
        validate_preplan_literature_design_authority(
            _bundle(card=_card(dimensions=LITERATURE_DESIGN_DIMENSIONS[:-1]))
        )
    assert exc_info.value.reason_code == "literature_design_dimensions_incomplete"


def test_preplan_authority_requires_a_recent_comparison_source() -> None:
    bundle = _bundle(card=_card()).model_copy(
        update={
            "citations": [
                CitationRecord(
                    key="comparator_2025",
                    title="An older comparable ICU cohort study",
                    year="2019",
                    venue="Example Journal",
                )
            ]
        }
    )
    with pytest.raises(LiteratureDesignAuthorityError) as exc_info:
        validate_preplan_literature_design_authority(bundle)
    assert exc_info.value.reason_code == "recent_literature_comparator_missing"


def test_selected_design_must_resolve_all_dimensions() -> None:
    with pytest.raises(LiteratureDesignAuthorityError) as exc_info:
        validate_selected_design_against_literature(
            _selection(include_all_dimensions=False),
            design_evidence_cards=[_card()],
            comparison_keys=["comparator_2025"],
        )
    assert exc_info.value.reason_code == "selected_design_dimensions_incomplete"


def test_prompt_contains_bounded_design_cards_not_article_body() -> None:
    blueprint = HypothesisBlueprint(
        research_question="Does exposure predict outcome in ICU patients?",
        hypothesis="Exposure is associated with outcome.",
        hypothesis_type="confirmatory",
        prior_literature_keys=["comparator_2025"],
        novelty_rationale="Different governed cohort.",
        feasible_variables=["exposure", "outcome"],
        missing_variables=[],
        concept_dependencies=[],
        cross_database_feasibility={},
        degraded_reason={},
        stepwise_plan=["Estimate the association."],
        self_critique=["Residual confounding remains possible."],
        feasibility_status="ready",
        domain_gate_notes=[],
    )
    rendered = render_hypothesis_blueprint_for_prompt(
        blueprint,
        literature=_bundle(card=_card()),
    )
    assert "Reviewed comparator design cards" in rendered
    assert "table_and_figure_completeness" in rendered
    assert "full_text_sha256=" in rendered
    assert "article body" not in rendered.casefold()


def test_qualification_profile_enables_strict_gate_without_changing_dev9() -> None:
    config = PipelineConfig(
        workdir="/tmp/easyicu-literature-design-test",
        **QUALIFICATION12_LITERATURE_DESIGN_2026_08_25.pipeline_options(),
    )
    assert config.require_literature_design_authority is True
    assert config.require_human_plan_review is True
    assert config.planner_strategy == "progressive_v2"

    dev9_options = DEV9_AI_REVIEWED_DEMO_2026_08_24.pipeline_options()
    dev9_options.update(
        require_human_plan_review=True,
        require_literature_design_authority=True,
    )
    with pytest.raises(ValueError, match="must match the submission profile"):
        PipelineConfig(
            workdir="/tmp/easyicu-literature-design-test",
            **dev9_options,
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"enable_literature": False}, "enable_literature"),
        ({"require_human_plan_review": False}, "require_human_plan_review"),
        ({"planner_strategy": "monolithic_v1"}, "progressive_v2"),
    ],
)
def test_strict_gate_configuration_fails_closed(overrides: dict, message: str) -> None:
    base = {
        "workdir": "/tmp/easyicu-literature-design-test",
        "require_literature_design_authority": True,
        "require_human_plan_review": True,
        "planner_strategy": "progressive_v2",
    }
    base.update(overrides)
    with pytest.raises(ValueError, match=message):
        PipelineConfig(**base)
