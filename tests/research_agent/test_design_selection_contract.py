from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.design_selection import (
    ResearchDesignSelection,
    ResearchDesignSelectionError,
    validate_research_design_selection,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveOutlineStep,
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.planning.literature_design_authority import (
    LITERATURE_DESIGN_DIMENSIONS,
    LiteratureDesignEvidence,
    LiteratureDesignEvidenceCard,
)
from easyicu.research_agent.schema import AnalysisPlan


def _selection_payload() -> dict:
    common = {
        "analysis_type": "association_study",
        "time_zero": "Start of the sealed ICU episode.",
        "observation_window": "The prespecified baseline-to-outcome window.",
        "required_variables": ["exposure", "outcome"],
        "assumptions": ["The declared variables preserve their intended timing."],
        "literature_citation_keys": ["direct_comparator"],
    }
    return {
        "schema_version": "easyicu.research_design_selection/1",
        "claim_ceiling": "analysis_only",
        "candidates": [
            {
                **common,
                "design_id": "adjusted_primary",
                "estimand": "Adjusted exposure contrast for the declared outcome.",
                "primary_method": "Adjusted generalized linear model",
                "novelty_positioning": "Tests the question in the sealed ICU population.",
                "figure_role": "Display the adjusted estimate and uncertainty.",
                "supports": "A prespecified adjusted association estimate.",
                "cannot_prove": "A causal effect without stronger identification.",
                "disposition": "selected",
                "decision_reason": (
                    "The exposure and outcome anchors require an adjusted primary contrast."
                ),
            },
            {
                **common,
                "design_id": "crude_alternative",
                "estimand": "Unadjusted exposure contrast for the declared outcome.",
                "primary_method": "Unadjusted descriptive contrast",
                "novelty_positioning": "Provides a crude comparison with prior reports.",
                "figure_role": "Display the unadjusted group difference.",
                "supports": "A descriptive difference between exposure groups.",
                "cannot_prove": "An adjusted or causal exposure effect.",
                "disposition": "rejected",
                "decision_reason": (
                    "Reject because the exposure and outcome question requires confounding control."
                ),
            },
        ],
    }


def test_design_selection_binds_run_authority_and_claim_ceiling() -> None:
    selection = ResearchDesignSelection.model_validate(_selection_payload())

    validate_research_design_selection(
        selection,
        selected_analysis_type="association_study",
        allowed_analysis_types=["association_study", "survival_analysis"],
        allowed_variables=["exposure", "outcome", "age"],
        allowed_literature_citation_keys=["direct_comparator"],
        question_anchors=["exposure", "outcome"],
        required=True,
    )

    assert selection.claim_ceiling == "analysis_only"
    assert selection.selected.design_id == "adjusted_primary"


def test_design_selection_rejects_post_result_choice() -> None:
    payload = _selection_payload()
    payload["candidates"][0]["decision_reason"] = (
        "Selected because the results showed a statistically significant association."
    )

    with pytest.raises(ValidationError, match="before results"):
        ResearchDesignSelection.model_validate(payload)


@pytest.mark.parametrize("failure", ["one_candidate", "two_selected"])
def test_design_selection_requires_two_to_four_and_exactly_one_selected(
    failure: str,
) -> None:
    payload = _selection_payload()
    if failure == "one_candidate":
        payload["candidates"] = payload["candidates"][:1]
    else:
        payload["candidates"][1]["disposition"] = "selected"

    with pytest.raises(ValidationError):
        ResearchDesignSelection.model_validate(payload)


def test_design_selection_rejects_unsealed_literature_key() -> None:
    selection = ResearchDesignSelection.model_validate(_selection_payload())

    with pytest.raises(ResearchDesignSelectionError) as caught:
        validate_research_design_selection(
            selection,
            selected_analysis_type="association_study",
            allowed_analysis_types=["association_study"],
            allowed_variables=["exposure", "outcome"],
            allowed_literature_citation_keys=["another_source"],
            question_anchors=["exposure", "outcome"],
            required=True,
        )

    assert caught.value.reason_code == "design_selection_literature_key_unavailable"


def test_fresh_outline_requires_design_selection_but_legacy_models_omit_none() -> None:
    outline = ProgressivePlanOutline(
        analysis_type="association_study",
        cohort_objective="Use the sealed analysis cohort.",
        steps=[
            ProgressiveOutlineStep(
                step_id="primary",
                planned_analysis_role="primary",
                module_id="adjusted_association",
                objective="Estimate the prespecified adjusted association.",
                variable_names=["exposure", "outcome"],
                scientific_action_id="association.adjusted_association",
            )
        ],
        rationale="Use the typed primary association owner.",
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=["association_study"],
            variable_names=["exposure", "outcome"],
            allowed_literature_citation_keys=[],
            primary_exposure="exposure",
            target_outcome="outcome",
            require_design_selection=True,
        )

    assert caught.value.reason_code == "progressive_design_selection_missing"
    assert "design_selection" not in outline.model_dump(mode="json")
    assert "design_selection" not in AnalysisPlan(
        research_question="Question?",
        analysis_type="association_study",
        steps=[],
    ).model_dump(mode="json")
    assert (
        "design_selection"
        not in ProgressivePlanSkeleton.model_json_schema()["required"]
    )


def test_progressive_outline_rejects_generic_sources_before_materialization() -> None:
    payload = _selection_payload()
    payload["candidates"][0]["literature_design_decisions"] = [
        {
            "dimension": dimension,
            "citation_keys": ["direct_comparator"],
            "disposition": "adapt",
            "rationale": f"Adapt the reviewed {dimension} design to this cohort.",
        }
        for dimension in LITERATURE_DESIGN_DIMENSIONS
    ]
    outline = ProgressivePlanOutline(
        analysis_type="association_study",
        cohort_objective="Use the sealed analysis cohort.",
        design_selection=ResearchDesignSelection.model_validate(payload),
        steps=[
            ProgressiveOutlineStep(
                step_id="primary",
                planned_analysis_role="primary",
                module_id="adjusted_association",
                objective="Estimate the prespecified adjusted association.",
                variable_names=["exposure", "outcome"],
                literature_citation_keys=["direct_comparator"],
                scientific_action_id="association.adjusted_association",
            )
        ],
        rationale="Use the typed primary association owner.",
    )
    reviewed_card = LiteratureDesignEvidenceCard(
        citation_key="reviewed_comparator",
        evidence_role="design_analogue",
        access_mode="open_access_fulltext",
        full_text_locator="https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
        full_text_sha256="a" * 64,
        supplement_status="not_published",
        reviewed_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        evidence=[
            LiteratureDesignEvidence(
                dimension=dimension,
                source_backed_summary=f"Reviewed fact for {dimension} in the analogue.",
                locator=f"section:{dimension}",
            )
            for dimension in LITERATURE_DESIGN_DIMENSIONS
        ],
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=["association_study"],
            variable_names=["exposure", "outcome"],
            allowed_literature_citation_keys=[
                "direct_comparator",
                "reviewed_comparator",
            ],
            primary_exposure="exposure",
            target_outcome="outcome",
            require_design_selection=True,
            literature_design_evidence_cards=[reviewed_card],
            comparison_literature_keys=["reviewed_comparator"],
        )

    assert caught.value.reason_code == (
        "progressive_selected_design_comparator_not_bound"
    )
