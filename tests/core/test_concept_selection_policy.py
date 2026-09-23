from __future__ import annotations

from easyicu.concept.selection_policy import (
    EXPLICIT_ONLY,
    OBSERVABILITY_COLLAPSING_KDIGO_CONCEPTS,
    PREFER_ALTERNATIVE,
    concept_selection_policy,
    evaluate_concept_selection,
)


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


def test_numeric_sofa_references_do_not_authorize_sofa2_variant() -> None:
    for message in (
        "SOFA 28-day mortality in sepsis",
        "评估乳酸清除率与 SOFA 2.5 分界的关系",
        "SOFA 2016 定义下的脓毒症诊断准确性",
        "SOFA 2.05 threshold",
    ):
        decision = evaluate_concept_selection(
            "sep3_sofa2",
            user_intent=message,
        )
        assert decision.allowed is False, message


def test_version_spelled_sofa2_alias_still_authorizes_variant() -> None:
    for message in (
        "please use the SOFA-2.0 definition of sepsis",
        "基于 SOFA 2.0 的脓毒症与院内死亡",
    ):
        decision = evaluate_concept_selection(
            "sep3_sofa2",
            user_intent=message,
        )
        assert decision.allowed is True, message


def test_ordinary_concept_does_not_require_special_authorization() -> None:
    decision = evaluate_concept_selection(
        "sep3_sofa1",
        user_intent="What is Sepsis-3 prevalence?",
    )

    assert decision.allowed is True
    assert decision.selection_mode == "ordinary"


def test_a_cautioned_definition_is_advised_never_withheld() -> None:
    """The reference KDIGO stage is a real phenotype; refusing it would remove
    the published cross-database definition from every legitimate
    reproduction.  What it needs is the caution, delivered while the design is
    being written rather than as a refusal after the plan exists.
    """

    decision = evaluate_concept_selection(
        "aki_stage_reference",
        user_intent="KDIGO stage in the first 24 hours and hospital mortality",
    )

    assert decision.allowed is True
    assert decision.reason_code == "concept_selection_advisory"
    assert decision.selection_mode == PREFER_ALTERNATIVE
    assert decision.canonical_alternative == "aki_stage_strict"


def test_every_collapsing_kdigo_binding_carries_the_same_caution() -> None:
    assert "aki_stage_reference" in OBSERVABILITY_COLLAPSING_KDIGO_CONCEPTS
    # The observability-preserving reading and the receipts that make it
    # possible are deliberately absent: they are the remedy, not the defect.
    for remedy in (
        "aki_stage_strict",
        "aki_ascertainment",
        "creatinine_evidence_status",
        "urine_evidence_status",
        "rrt_evidence_status",
    ):
        assert remedy not in OBSERVABILITY_COLLAPSING_KDIGO_CONCEPTS
        assert concept_selection_policy(remedy) is None
    for concept in OBSERVABILITY_COLLAPSING_KDIGO_CONCEPTS:
        policy = concept_selection_policy(concept)
        assert policy is not None, concept
        assert policy.selection_mode == PREFER_ALTERNATIVE
        assert policy.canonical_alternative == "aki_stage_strict"
        assert "aki_stage_strict" in policy.rationale


def test_the_experimental_variant_is_still_withheld() -> None:
    """An advisory mode must not weaken the explicit-only contract."""

    decision = evaluate_concept_selection(
        "sep3_sofa2", user_intent="Sepsis-3 prevalence and mortality"
    )

    assert decision.allowed is False
    assert decision.selection_mode == EXPLICIT_ONLY
