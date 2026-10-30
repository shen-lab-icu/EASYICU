"""Shared verification primitives for preparation-only, source-bound reviews.

A database adapter must independently validate that its reviewed scope fully
lists the actual files, fields, windows and semantics. These checks do not
establish an immutable snapshot or a complete producer read set.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision, HumanReviewRequest
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority


class RawPreparationCheckpoint(BaseModel):
    """One separately reviewed, expiring preparation capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    error_prefix: ClassVar[str] = "raw_preparation"
    schema_version: Literal["easyicu.raw-preparation-checkpoint/1"] = "easyicu.raw-preparation-checkpoint/1"
    state: Literal["pending", "approved", "rejected"]
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str
    expires_at: str
    request: HumanReviewRequest
    decision: HumanReviewDecision | None = None
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _binding(self):
        try:
            created, expires = datetime.fromisoformat(self.created_at), datetime.fromisoformat(self.expires_at)
        except ValueError as exc:
            raise ValueError(f"{self.error_prefix}_checkpoint_time_invalid") from exc
        if (created.tzinfo is None or expires.tzinfo is None
                or not timedelta(0) < expires - created <= timedelta(days=1)):
            raise ValueError(f"{self.error_prefix}_checkpoint_time_invalid")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"checkpoint_sha256"}))
        if expected != self.checkpoint_sha256:
            raise ValueError(f"{self.error_prefix}_checkpoint_digest_mismatch")
        if (self.state == "pending") != (self.decision is None):
            raise ValueError(f"{self.error_prefix}_checkpoint_decision_mismatch")
        if self.decision is not None and (
            self.decision.review_id != self.request.review_id
            or self.decision.authority_sha256 != self.request.authority_sha256
            or self.decision.decision != self.state
        ):
            raise ValueError(f"{self.error_prefix}_checkpoint_decision_mismatch")
        if self.decision is not None:
            try:
                decided = datetime.fromisoformat(self.decision.decided_at)
            except ValueError as exc:
                raise ValueError(f"{self.error_prefix}_checkpoint_time_invalid") from exc
            if decided.tzinfo is None or not created <= decided < expires:
                raise ValueError(f"{self.error_prefix}_checkpoint_time_invalid")
        return self

    @classmethod
    def pending(cls, *, plan_sha256: str, request: HumanReviewRequest,
                now: datetime | None = None):
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError(f"{cls.error_prefix}_checkpoint_timezone_required")
        body = {"schema_version": cls.model_fields["schema_version"].default, "state": "pending",
                "plan_sha256": plan_sha256, "created_at": now.isoformat(),
                "expires_at": (now + timedelta(days=1)).isoformat(),
                "request": request.model_dump(mode="json"), "decision": None}
        return cls.model_validate({**body, "checkpoint_sha256": canonical_sha256(body)})

    def decide(self, decision: HumanReviewDecision, *, now: datetime | None = None):
        if self.state != "pending":
            raise ValueError(f"{self.error_prefix}_checkpoint_already_decided")
        instant = now or datetime.now(timezone.utc)
        if instant.tzinfo is None or instant >= datetime.fromisoformat(self.expires_at):
            raise ValueError(f"{self.error_prefix}_checkpoint_expired")
        body = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        body.update(state=decision.decision, decision=decision.model_dump(mode="json"))
        return type(self).model_validate({**body, "checkpoint_sha256": canonical_sha256(body)})


def verify_raw_preparation_review(*, checkpoint: RawPreparationCheckpoint,
        requested_scope: dict, scope_key: str, expected_summary: str,
        registered_database: str, registered_identity_sha256: str,
        plan_sha256: str, context_sha256: str, scientific_review_sha256: str,
        now: datetime | None = None) -> None:
    """Recheck an exact adapter-validated scope against current host identity.

    ``registered_*`` must come from the host registry, not from the request.
    The adapter verifies source-specific fields, windows and source metadata.
    This contract grants preparation only and is not a producer receipt.
    """
    prefix = checkpoint.error_prefix
    checkpoint = type(checkpoint).model_validate(checkpoint.model_dump(mode="json"))
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None or instant < datetime.fromisoformat(checkpoint.created_at) or instant >= datetime.fromisoformat(checkpoint.expires_at):
        raise ValueError(f"{prefix}_checkpoint_expired")
    if checkpoint.state != "approved" or checkpoint.plan_sha256 != plan_sha256:
        raise ValueError(f"{prefix}_approval_missing")
    if (requested_scope.get("purpose") != "prepare_only"
            or requested_scope.get("analysis_authorized") is not False
            or requested_scope.get("database") != registered_database
            or requested_scope.get("source_identity_sha256") != registered_identity_sha256
            or requested_scope.get("plan_sha256") != plan_sha256
            or requested_scope.get("context_sha256") != context_sha256):
        raise ValueError(f"{prefix}_scope_binding_mismatch")
    request, payload = checkpoint.request, checkpoint.request.payload
    if (set(payload) != {"plan_review_authority", scope_key}
            or request.authority_sha256 != canonical_sha256(payload)
            or request.summary != expected_summary
            or payload[scope_key] != requested_scope):
        raise ValueError(f"{prefix}_scope_binding_mismatch")
    plan_review = PlanReviewAuthority.model_validate(payload["plan_review_authority"])
    if (plan_review.plan_sha256 != plan_sha256 or plan_review.execution is not None
            or plan_review.evidence_sha256.get("scientific_plan_review") != scientific_review_sha256):
        raise ValueError(f"{prefix}_review_mismatch")
