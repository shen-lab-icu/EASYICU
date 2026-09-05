"""Typed authority packet for human approval of an analysis plan.

Human approval is meaningful only when it binds the complete scientific plan
and the execution identity that will carry it out.  This module owns that
packet so the workflow cannot accidentally fall back to signing a short list
of step ids or a finding message.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from ..canonical_json import canonical_sha256
from ..contracts.frozen_payload import freeze_payload, thaw_payload
from ..schema import AnalysisPlan


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReviewExecutionAuthority(BaseModel):
    """Execution identity attached by the production pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.review_execution_authority/1"] = (
        "easyicu.review_execution_authority/1"
    )
    pipeline_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submission_profile_ref: str | None = Field(default=None, min_length=1)
    capability_activation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_input_capsule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PlanReviewAuthority(BaseModel):
    """Canonical, self-verifying snapshot of what a reviewer approves."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.plan_review_authority/1"] = (
        "easyicu.plan_review_authority/1"
    )
    analysis_plan_schema_version: Literal["easyicu.analysis_plan/1"] = (
        "easyicu.analysis_plan/1"
    )
    plan_payload: Mapping[str, Any]
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: Mapping[str, str] = Field(default_factory=dict, validate_default=True)
    execution: ReviewExecutionAuthority | None = None

    @field_validator("evidence_sha256")
    @classmethod
    def _validate_evidence_digests(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        cleaned: dict[str, str] = {}
        for raw_id, raw_digest in value.items():
            evidence_id = str(raw_id).strip()
            digest = str(raw_digest).strip()
            if not evidence_id:
                raise ValueError("review evidence ids must be non-empty")
            if not _SHA256_RE.fullmatch(digest):
                raise ValueError(
                    f"review evidence {evidence_id!r} must have a SHA-256 digest"
                )
            cleaned[evidence_id] = digest
        return freeze_payload(dict(sorted(cleaned.items())))

    @field_validator("plan_payload")
    @classmethod
    def _freeze_plan(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_payload(value)

    @field_serializer("plan_payload", "evidence_sha256")
    def _wire_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_payload(value)

    @model_validator(mode="after")
    def _plan_digest_matches_payload(self) -> "PlanReviewAuthority":
        if canonical_sha256(thaw_payload(self.plan_payload)) != self.plan_sha256:
            raise ValueError("plan_sha256 does not bind plan_payload")
        return self

    @classmethod
    def create(
        cls,
        *,
        plan: AnalysisPlan | Mapping[str, Any],
        evidence_sha256: Mapping[str, str] | None = None,
        execution: ReviewExecutionAuthority | Mapping[str, Any] | None = None,
    ) -> "PlanReviewAuthority":
        """Validate and freeze one complete typed plan for review.

        A loose object with only ``revision`` or ``step_id`` fields is refused:
        accepting it would silently recreate the partial-signature bug this
        authority packet exists to close.
        """

        if isinstance(plan, AnalysisPlan):
            typed_plan = plan
        elif isinstance(plan, Mapping):
            typed_plan = AnalysisPlan.model_validate(dict(plan))
        else:
            raise TypeError(
                "human review requires a complete AnalysisPlan or its mapping"
            )
        plan_payload = typed_plan.model_dump(mode="json")
        return cls(
            plan_payload=plan_payload,
            plan_sha256=canonical_sha256(plan_payload),
            evidence_sha256=dict(evidence_sha256 or {}),
            execution=execution,
        )


__all__ = [
    "PlanReviewAuthority",
    "ReviewExecutionAuthority",
]
