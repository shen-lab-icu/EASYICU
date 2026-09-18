"""Track 1: sandbox code cannot masquerade as ``verified_tool``.

Minimal promotion chain: one passing run is at most ``sandbox_code`` /
``candidate``; ``verified_tool`` additionally requires an independent
reproduction matching the origin digest plus an affirmative applicability
attestation; ``composed_workflow`` requires every member verified.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.tool_promotion import (
    ApplicabilityAttestation,
    ComposedWorkflowDecision,
    PromotionDecision,
    ReproductionAttempt,
    ToolPromotionError,
    classify_single_run,
    decide_composed_workflow,
    decide_tool_promotion,
    require_composed_workflow,
    require_verified_tool,
)
from easyicu.research_agent.methods.tool_card import (
    ToolCard,
    tool_card_completeness_issues,
    tool_card_sha256,
)

ORIGIN_DIGEST = "ab" * 32
OTHER_DIGEST = "cd" * 32


def _card(**overrides) -> ToolCard:
    payload: dict = {
        "schema_version": "easyicu.tool_card/1",
        "tool_name": "sofa_trend_slope",
        "tool_version": "0.1.0",
        "method_meaning": "slope of daily SOFA over the first 72h of ICU stay",
        "inputs": [
            {"name": "sofa_daily", "value_kind": "float_series", "required": True}
        ],
        "outputs": [{"name": "slope", "value_kind": "float"}],
        "population_assumption": "adult ICU stays with >=3 SOFA days",
        "timing_assumption": "first 72h after ICU admission, daily grid",
        "origin_output_sha256": ORIGIN_DIGEST,
        "validation_evidence": [],
    }
    payload.update(overrides)
    return ToolCard.model_validate(payload, strict=True)


def _recorded_card() -> ToolCard:
    return _card(
        validation_evidence=[
            {
                "evidence_id": "sandbox-run-1",
                "kind": "sandbox_run",
                "passed": True,
                "output_sha256": ORIGIN_DIGEST,
            }
        ]
    )


def _attestation(**overrides) -> ApplicabilityAttestation:
    payload: dict = {
        "intended_population": "adult ICU stays with >=3 SOFA days",
        "intended_timing": "first 72h after ICU admission, daily grid",
        "covered": True,
        "reason": "intended reuse matches the card assumptions exactly",
    }
    payload.update(overrides)
    return ApplicabilityAttestation.model_validate(payload, strict=True)


def _reproduction(**overrides) -> ReproductionAttempt:
    payload: dict = {
        "attempt_id": "repro-1",
        "output_sha256": ORIGIN_DIGEST,
        "passed": True,
        "independent": True,
    }
    payload.update(overrides)
    return ReproductionAttempt.model_validate(payload, strict=True)


def test_single_run_without_receipt_is_sandbox_code() -> None:
    assert classify_single_run(_card()) == "sandbox_code"


def test_single_recorded_run_is_candidate_never_verified() -> None:
    identity = classify_single_run(_recorded_card())
    assert identity == "candidate"
    assert identity not in {"verified_tool", "composed_workflow"}


def test_tool_card_rejects_unpinned_version() -> None:
    for bad in ("latest", "main", "  "):
        with pytest.raises(ValueError):
            _card(tool_version=bad)


def test_tool_card_rejects_blank_assumptions() -> None:
    with pytest.raises(ValueError):
        _card(population_assumption="   ")
    with pytest.raises(ValueError):
        _card(timing_assumption="")


def test_tool_card_digest_stable_and_complete() -> None:
    card = _recorded_card()
    assert tool_card_sha256(card) == tool_card_sha256(_recorded_card())
    assert tool_card_completeness_issues(card) == []


def test_promotion_denied_without_reproduction() -> None:
    decision = decide_tool_promotion(
        card=_recorded_card(), applicability=_attestation()
    )
    assert isinstance(decision, PromotionDecision)
    assert decision.allowed_identity == "candidate"
    assert decision.granted_verified_tool is False
    assert any("reproduction" in reason for reason in decision.reasons)
    with pytest.raises(ToolPromotionError):
        require_verified_tool(decision)


def test_promotion_denied_on_digest_mismatch() -> None:
    decision = decide_tool_promotion(
        card=_recorded_card(),
        reproductions=[_reproduction(output_sha256=OTHER_DIGEST)],
        applicability=_attestation(),
    )
    assert decision.allowed_identity == "candidate"
    with pytest.raises(ToolPromotionError):
        require_verified_tool(decision)


def test_promotion_denied_on_dependent_reproduction() -> None:
    decision = decide_tool_promotion(
        card=_recorded_card(),
        reproductions=[_reproduction(independent=False)],
        applicability=_attestation(),
    )
    assert decision.allowed_identity == "candidate"
    with pytest.raises(ToolPromotionError):
        require_verified_tool(decision)


def test_promotion_denied_without_boundary() -> None:
    denied_none = decide_tool_promotion(
        card=_recorded_card(), reproductions=[_reproduction()]
    )
    assert denied_none.allowed_identity == "candidate"
    denied_uncovered = decide_tool_promotion(
        card=_recorded_card(),
        reproductions=[_reproduction()],
        applicability=_attestation(covered=False),
    )
    assert denied_uncovered.allowed_identity == "candidate"
    denied_reason = decide_tool_promotion(
        card=_recorded_card(),
        reproductions=[_reproduction()],
        applicability=_attestation(reason="  "),
    )
    assert denied_reason.allowed_identity == "candidate"
    with pytest.raises(ToolPromotionError):
        require_verified_tool(denied_uncovered)


def test_promotion_grants_verified_tool() -> None:
    card = _recorded_card()
    decision = decide_tool_promotion(
        card=card,
        reproductions=[_reproduction()],
        applicability=_attestation(),
    )
    assert decision.allowed_identity == "verified_tool"
    assert decision.granted_verified_tool is True
    assert decision.card_sha256 == tool_card_sha256(card)
    require_verified_tool(decision)


def test_promotion_threshold_floor_cannot_be_lowered() -> None:
    with pytest.raises(ValueError):
        decide_tool_promotion(
            card=_recorded_card(),
            reproductions=[_reproduction()],
            applicability=_attestation(),
            min_independent_reproductions=0,
        )


def test_composed_workflow_requires_all_verified() -> None:
    denied = decide_composed_workflow(["verified_tool", "candidate"])
    assert denied.allowed_identity == "candidate"
    with pytest.raises(ToolPromotionError):
        require_composed_workflow(denied)
    granted = decide_composed_workflow(["verified_tool", "verified_tool"])
    assert isinstance(granted, ComposedWorkflowDecision)
    assert granted.allowed_identity == "composed_workflow"
    require_composed_workflow(granted)


"""Track 1 wiring: grants consulted at capability resolution."""

from easyicu.research_agent.planning import capability_registry as registry_module


def _granted_pair(
    executor_module: str = "execution.runners.survival_primary_executor",
):
    card = _recorded_card()
    decision = decide_tool_promotion(
        card=card,
        reproductions=[_reproduction()],
        applicability=_attestation(),
        executor_module=executor_module,
    )
    assert decision.granted_verified_tool is True
    return card, decision


def _clear_grants():
    registry_module._TOOL_CARD_GRANTS.clear()


def test_registry_files_grant_and_rejects_mismatch():
    _clear_grants()
    try:
        card, decision = _granted_pair()
        record = registry_module.register_tool_card_grant(
            executor_module="execution.runners.survival_primary_executor",
            card=card,
            decision=decision,
        )
        assert record["tool_name"] == "sofa_trend_slope"
        assert len(record["card_sha256"]) == 64
        assert (
            registry_module.granted_tool_identity(
                "execution.runners.survival_primary_executor"
            )
            == "verified_tool"
        )
        assert (
            registry_module.granted_tool_identity("execution.runners.unknown")
            is None
        )
        other_card = _recorded_card().model_copy(update={"tool_version": "0.2.0"})
        assert (
            tool_card_sha256(other_card) != tool_card_sha256(card)
        )
        with pytest.raises(ValueError, match="unrelated executor"):
            # Same decision, different module: the grant cannot be re-pointed
            # by editing the envelope's executor field alone.
            registry_module.register_tool_card_grant(
                executor_module="execution.runners.other",
                card=card,
                decision=decision,
            )
        with pytest.raises(ValueError, match="digest"):
            registry_module.register_tool_card_grant(
                executor_module="execution.runners.other",
                card=other_card,
                decision=decision,
            )
    finally:
        _clear_grants()


def test_registry_rejects_incomplete_card_and_ungranted_decision():
    _clear_grants()
    try:
        card, decision = _granted_pair()
        with pytest.raises(ValueError, match="incomplete"):
            registry_module.register_tool_card_grant(
                executor_module="execution.runners.x",
                card=_card(),
                decision=decision,
            )
        denied = decide_tool_promotion(
            card=_recorded_card(), applicability=_attestation()
        )
        assert denied.granted_verified_tool is False
        with pytest.raises(ValueError, match="did not grant"):
            registry_module.register_tool_card_grant(
                executor_module="execution.runners.x",
                card=_recorded_card(),
                decision=denied,
            )
        with pytest.raises(ValueError, match="non-empty"):
            registry_module.register_tool_card_grant(
                executor_module="  ", card=card, decision=decision
            )
    finally:
        _clear_grants()


def test_resolve_annotates_granted_executor():
    _clear_grants()
    try:
        card, decision = _granted_pair()
        registry_module.register_tool_card_grant(
            executor_module="execution.runners.survival_primary_executor",
            card=card,
            decision=decision,
        )
        verdict = registry_module.resolve_primary_capability(
            analysis_type="survival", plan=None
        )
        assert verdict.failure_reason is None
        assert (
            "tool_identity=verified_tool("
            "execution.runners.survival_primary_executor" in verdict.detail
        )
    finally:
        _clear_grants()


def test_resolve_without_grant_annotates_nothing():
    _clear_grants()
    verdict = registry_module.resolve_primary_capability(
        analysis_type="survival", plan=None
    )
    assert verdict.failure_reason is None
    assert "tool_identity=" not in verdict.detail


def test_require_capability_tool_identity():
    _clear_grants()
    try:
        assert (
            registry_module.require_capability_tool_identity(
                executor_module="execution.runners.survival_primary_executor",
                asserted_identity=None,
            )
            is None
        )
        with pytest.raises(ToolPromotionError, match="without a registered"):
            registry_module.require_capability_tool_identity(
                executor_module="execution.runners.survival_primary_executor",
                asserted_identity="verified_tool",
            )
        with pytest.raises(ValueError, match="unknown tool identity"):
            registry_module.require_capability_tool_identity(
                executor_module="execution.runners.survival_primary_executor",
                asserted_identity="oracle",
            )
        card, decision = _granted_pair()
        registry_module.register_tool_card_grant(
            executor_module="execution.runners.survival_primary_executor",
            card=card,
            decision=decision,
        )
        assert (
            registry_module.require_capability_tool_identity(
                executor_module="execution.runners.survival_primary_executor",
                asserted_identity="verified_tool",
            )
            == "verified_tool"
        )
    finally:
        _clear_grants()
