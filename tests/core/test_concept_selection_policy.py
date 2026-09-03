from __future__ import annotations

from easyicu.concept.selection_policy import evaluate_concept_selection


def test_generic_sepsis_does_not_authorize_experimental_sofa2_variant() -> None:
    decision = evaluate_concept_selection(
        "sep3_sofa2",
        user_intent="What is Sepsis-3 prevalence and its mortality association?",
    )

    assert decision.allowed is False
    assert decision.reason_code == "concept_explicit_selection_required"
    assert decision.canonical_alternative == "sep3_sofa1"


def test_positive_explicit_sofa2_request_authorizes_variant() -> None:
    decision = evaluate_concept_selection(
        "sep3_sofa2",
        user_intent="Use SOFA-2 for an experimental Sepsis sensitivity analysis.",
    )

    assert decision.allowed is True
    assert decision.reason_code == "concept_selection_explicit"


def test_negated_sofa2_request_does_not_authorize_variant() -> None:
    decision = evaluate_concept_selection(
        "sep3_sofa2",
        user_intent="不要用 SOFA-2，使用标准 Sepsis-3。",
    )

    assert decision.allowed is False


def test_ordinary_concept_does_not_require_special_authorization() -> None:
    decision = evaluate_concept_selection(
        "sep3_sofa1",
        user_intent="What is Sepsis-3 prevalence?",
    )

    assert decision.allowed is True
    assert decision.selection_mode == "ordinary"
