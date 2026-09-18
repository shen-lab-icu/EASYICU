"""Promotion gate from ``sandbox_code`` to ``verified_tool`` (R4).

A single passing sandbox run may only ever be ``sandbox_code`` (run recorded)
or ``candidate`` (run recorded and filed for promotion).  The ``verified_tool``
identity additionally requires every one of these, fail-closed:

* a complete Tool Card (pinned version, inputs, method meaning, outputs,
  population/timing assumptions, filed origin run);
* at least one independent, passed reproduction whose output digest matches
  the card's origin digest;
* an affirmative applicability attestation covering the intended reuse
  population and timing with a recorded reason.

``composed_workflow`` additionally requires every member to already hold
``verified_tool``; one unverified member denies the whole composition.  The
semantic subset judgement (is the intended use truly inside the card's
assumptions) stays reviewer-owned; this gate enforces that the attestation
exists, is affirmative, and carries a reason, and binds it to the decision.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..methods.tool_card import (
    ToolCapabilityIdentity,
    ToolCard,
    tool_card_completeness_issues,
    tool_card_sha256,
)

TOOL_PROMOTION_DECISION_SCHEMA = "easyicu.tool_promotion_decision/1"
COMPOSED_WORKFLOW_DECISION_SCHEMA = "easyicu.composed_workflow_decision/1"

#: Independent reproductions required before ``verified_tool`` may be granted.
MIN_INDEPENDENT_REPRODUCTIONS = 1


class ToolPromotionError(ValueError):
    """A claimed tool identity the promotion gate did not grant."""


class ReproductionAttempt(BaseModel):
    """One rerun offered as reproduction evidence for a Tool Card."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempt_id: str = Field(min_length=1)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool
    independent: bool = False


class ApplicabilityAttestation(BaseModel):
    """Reviewer-owned boundary check for one intended reuse context."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    intended_population: str = Field(min_length=1)
    intended_timing: str = Field(min_length=1)
    covered: bool
    reason: str = ""


class PromotionDecision(BaseModel):
    """The exact identity one promotion review granted, digest-bound."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.tool_promotion_decision/1"]
    tool_name: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    executor_module: Optional[str] = Field(default=None)
    allowed_identity: ToolCapabilityIdentity
    granted_verified_tool: bool
    reasons: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _verify_grant_consistency(self) -> "PromotionDecision":
        if self.granted_verified_tool != (self.allowed_identity == "verified_tool"):
            raise ValueError("promotion grant flag disagrees with allowed identity")
        if self.executor_module is not None and not self.executor_module.strip():
            raise ValueError("executor_module must be non-blank when provided")
        return self


class ComposedWorkflowDecision(BaseModel):
    """The exact identity one workflow-composition review granted."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.composed_workflow_decision/1"]
    member_count: int = Field(ge=1)
    allowed_identity: ToolCapabilityIdentity
    reasons: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _verify_composition_grant(self) -> "ComposedWorkflowDecision":
        granted = self.allowed_identity == "composed_workflow"
        flagged = any(reason.startswith("granted_") for reason in self.reasons)
        if granted == flagged:
            return self
        raise ValueError("composed-workflow grant flag disagrees with reasons")


def classify_single_run(card: ToolCard) -> Literal["sandbox_code", "candidate"]:
    """Return the highest identity one sandbox success may claim.

    Never returns ``verified_tool`` or ``composed_workflow``: a single run
    with no filed receipt stays ``sandbox_code``; a run recorded on the card
    becomes ``candidate`` and must still pass :func:`decide_tool_promotion`.
    """

    if not isinstance(card, ToolCard):
        raise TypeError("classify_single_run requires a ToolCard")
    for item in card.validation_evidence:
        if (
            item.kind == "sandbox_run"
            and item.passed
            and item.output_sha256 == card.origin_output_sha256
        ):
            return "candidate"
    return "sandbox_code"


def _qualifying_reproductions(
    card: ToolCard, reproductions: Sequence[ReproductionAttempt]
) -> list[str]:
    seen: dict[str, str] = {}
    for attempt in reproductions:
        if not attempt.passed or not attempt.independent:
            continue
        if attempt.output_sha256 != card.origin_output_sha256:
            continue
        previous = seen.get(attempt.attempt_id)
        if previous is None:
            seen[attempt.attempt_id] = attempt.output_sha256
        elif previous != attempt.output_sha256:
            return []
    return sorted(seen)


def decide_tool_promotion(
    *,
    card: ToolCard,
    reproductions: Sequence[ReproductionAttempt] = (),
    applicability: ApplicabilityAttestation | None = None,
    min_independent_reproductions: int = MIN_INDEPENDENT_REPRODUCTIONS,
    executor_module: Optional[str] = None,
) -> PromotionDecision:
    """Grant ``verified_tool`` only if reproduction and boundary checks pass.

    ``executor_module`` optionally binds the grant to one executor at
    issuance time. Registration refuses decisions whose bound module is
    missing or differs from the registration target, so a grant cannot be
    re-pointed at an unrelated executor by editing the envelope alone.
    """

    if not isinstance(card, ToolCard):
        raise TypeError("decide_tool_promotion requires a ToolCard")
    if min_independent_reproductions < MIN_INDEPENDENT_REPRODUCTIONS:
        raise ValueError("reproduction threshold cannot be lowered below the floor")
    reasons: list[str] = []
    reasons.extend(tool_card_completeness_issues(card))
    qualifying = _qualifying_reproductions(card, reproductions)
    if len(qualifying) < min_independent_reproductions:
        reasons.append(
            "insufficient_independent_reproduction: "
            f"found={len(qualifying)} "
            f"required={min_independent_reproductions} "
            "matching_passed_independent"
        )
    if applicability is None:
        reasons.append("missing_applicability_attestation")
    else:
        if not applicability.intended_population.strip():
            reasons.append("applicability_attestation: intended_population is blank")
        if not applicability.intended_timing.strip():
            reasons.append("applicability_attestation: intended_timing is blank")
        if not applicability.covered:
            reasons.append("applicability_not_covered")
        if not applicability.reason.strip():
            reasons.append("applicability_attestation: reason is blank")
    card_digest = tool_card_sha256(card)
    bound_module = (
        str(executor_module).strip() if executor_module is not None else None
    )
    if executor_module is not None and not bound_module:
        raise ValueError("executor_module must be non-blank when provided")
    if reasons:
        if any(reason.startswith("incomplete_card") for reason in reasons):
            allowed: ToolCapabilityIdentity = "sandbox_code"
        else:
            allowed = "candidate"
        return PromotionDecision.model_validate(
            {
                "schema_version": TOOL_PROMOTION_DECISION_SCHEMA,
                "tool_name": card.tool_name,
                "tool_version": card.tool_version,
                "card_sha256": card_digest,
                "executor_module": bound_module,
                "allowed_identity": allowed,
                "granted_verified_tool": False,
                "reasons": reasons,
            },
            strict=True,
        )
    return PromotionDecision.model_validate(
        {
            "schema_version": TOOL_PROMOTION_DECISION_SCHEMA,
            "tool_name": card.tool_name,
            "tool_version": card.tool_version,
            "card_sha256": card_digest,
            "executor_module": bound_module,
            "allowed_identity": "verified_tool",
            "granted_verified_tool": True,
            "reasons": [
                f"granted_verified_tool: "
                f"reproductions={len(qualifying)} "
                f"boundary_covered={applicability.covered if applicability else False}"
            ],
        },
        strict=True,
    )


def require_verified_tool(decision: PromotionDecision) -> None:
    """Raise unless ``decision`` grants the ``verified_tool`` identity.

    Every caller that would run or advertise a tool as ``verified_tool``
    must pass through here; a ``candidate``/``sandbox_code`` decision fails
    closed instead of silently inheriting the higher identity.
    """

    if not isinstance(decision, PromotionDecision):
        raise TypeError("require_verified_tool requires a PromotionDecision")
    if not decision.granted_verified_tool or decision.allowed_identity != (
        "verified_tool"
    ):
        raise ToolPromotionError(
            f"tool {decision.tool_name}/{decision.tool_version} holds "
            f"{decision.allowed_identity!r}, not 'verified_tool': "
            + "; ".join(decision.reasons)
        )


def decide_composed_workflow(
    member_identities: Sequence[ToolCapabilityIdentity],
) -> ComposedWorkflowDecision:
    """Grant ``composed_workflow`` only if every member is ``verified_tool``."""

    members = list(member_identities)
    if not members:
        return ComposedWorkflowDecision.model_validate(
            {
                "schema_version": COMPOSED_WORKFLOW_DECISION_SCHEMA,
                "member_count": 1,
                "allowed_identity": "candidate",
                "reasons": ["empty_composition_cannot_claim_composed_workflow"],
            },
            strict=True,
        )
    unverified = [name for name in members if name != "verified_tool"]
    if unverified:
        return ComposedWorkflowDecision.model_validate(
            {
                "schema_version": COMPOSED_WORKFLOW_DECISION_SCHEMA,
                "member_count": len(members),
                "allowed_identity": "candidate",
                "reasons": [
                    f"member_not_verified_tool: count={len(unverified)} "
                    f"of={len(members)}"
                ],
            },
            strict=True,
        )
    return ComposedWorkflowDecision.model_validate(
        {
            "schema_version": COMPOSED_WORKFLOW_DECISION_SCHEMA,
            "member_count": len(members),
            "allowed_identity": "composed_workflow",
            "reasons": [f"granted_composed_workflow: members={len(members)}_verified"],
        },
        strict=True,
    )


def require_composed_workflow(decision: ComposedWorkflowDecision) -> None:
    """Raise unless ``decision`` grants the ``composed_workflow`` identity."""

    if not isinstance(decision, ComposedWorkflowDecision):
        raise TypeError("require_composed_workflow requires a ComposedWorkflowDecision")
    if decision.allowed_identity != "composed_workflow":
        raise ToolPromotionError(
            "workflow holds "
            f"{decision.allowed_identity!r}, not 'composed_workflow': "
            + "; ".join(decision.reasons)
        )


__all__ = [
    "COMPOSED_WORKFLOW_DECISION_SCHEMA",
    "MIN_INDEPENDENT_REPRODUCTIONS",
    "TOOL_PROMOTION_DECISION_SCHEMA",
    "ApplicabilityAttestation",
    "ComposedWorkflowDecision",
    "PromotionDecision",
    "ReproductionAttempt",
    "ToolPromotionError",
    "classify_single_run",
    "decide_composed_workflow",
    "decide_tool_promotion",
    "require_composed_workflow",
    "require_verified_tool",
]
