"""Digest-bound fail-closed pause for substantive runtime replans."""

from __future__ import annotations

from dataclasses import dataclass

from ..canonical_json import canonical_sha256
from ..schema import AnalysisPlan, ValidationFinding


@dataclass(frozen=True, slots=True)
class RuntimeReplanReviewPause:
    """A revised scientific plan that cannot execute without a new decision."""

    current_plan_sha256: str
    candidate_plan_sha256: str
    review_authority_sha256: str
    current_revision: int
    candidate_revision: int
    trigger: str

    @classmethod
    def create(
        cls,
        *,
        current_plan: AnalysisPlan,
        candidate_plan: AnalysisPlan,
        trigger: str,
    ) -> "RuntimeReplanReviewPause":
        current_payload = current_plan.model_dump(mode="json")
        candidate_payload = candidate_plan.model_dump(mode="json")
        current_sha256 = canonical_sha256(current_payload)
        candidate_sha256 = canonical_sha256(candidate_payload)
        if current_sha256 == candidate_sha256:
            raise ValueError("runtime replan review requires a substantive revision")
        authority_payload = {
            "schema_version": "easyicu.runtime_replan_review_request/1",
            "current_plan_sha256": current_sha256,
            "candidate_plan_sha256": candidate_sha256,
            "current_revision": int(current_plan.revision),
            "candidate_revision": int(candidate_plan.revision),
            "trigger": str(trigger),
        }
        return cls(
            current_plan_sha256=current_sha256,
            candidate_plan_sha256=candidate_sha256,
            review_authority_sha256=canonical_sha256(authority_payload),
            current_revision=int(current_plan.revision),
            candidate_revision=int(candidate_plan.revision),
            trigger=str(trigger),
        )

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema_version": "easyicu.runtime_replan_review_request/1",
            "current_plan_sha256": self.current_plan_sha256,
            "candidate_plan_sha256": self.candidate_plan_sha256,
            "review_authority_sha256": self.review_authority_sha256,
            "current_revision": self.current_revision,
            "candidate_revision": self.candidate_revision,
            "trigger": self.trigger,
            "human_review_required": True,
            "execution_paused": True,
        }

    def finding(self) -> ValidationFinding:
        return ValidationFinding(
            validator="runtime_replan_human_review",
            severity="error",
            message=(
                "A substantive runtime plan revision requires a new exact human "
                "decision; the candidate was not registered, applied, or executed."
            ),
            detail={
                "reason": "runtime_replan_human_review_required",
                **self.manifest_payload(),
            },
        )


def runtime_replan_review_pause(
    *,
    require_human_plan_review: bool,
    current_plan: AnalysisPlan,
    candidate_plan: AnalysisPlan,
    trigger: str,
) -> RuntimeReplanReviewPause | None:
    """Create the exact pause authority only for review-gated runs."""

    if not require_human_plan_review:
        return None
    return RuntimeReplanReviewPause.create(
        current_plan=current_plan,
        candidate_plan=candidate_plan,
        trigger=trigger,
    )


__all__ = ["RuntimeReplanReviewPause", "runtime_replan_review_pause"]
