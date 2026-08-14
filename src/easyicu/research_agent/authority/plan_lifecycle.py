"""Typed scientific-plan lineage from proposal through executable approval.

``AnalysisPlan`` remains the one execution schema.  These wrappers do not
create a second plan language; they identify who authored each stage and bind
every host transformation to an auditable before/after digest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical_json import canonical_sha256
from ..schema import AnalysisPlan
from .evidence_store import EvidenceStore
from .plan_review import PlanReviewAuthority
from .runtime_artifacts import verified_run_evidence_path


class PlanLifecycleAuthorityError(RuntimeError):
    """A plan-stage or lineage artifact is missing, stale, or inconsistent."""

    reason_code = "plan_lifecycle_authority_invalid"


def _plan_payload(plan: AnalysisPlan | Mapping[str, Any]) -> dict[str, Any]:
    typed = (
        plan
        if isinstance(plan, AnalysisPlan)
        else AnalysisPlan.model_validate(dict(plan))
    )
    return typed.model_dump(mode="json")


def _changed_fields(left: Any, right: Any, *, path: str = "") -> tuple[str, ...]:
    """Return stable JSON-pointer-like leaves changed by one transformation."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        changes: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                changes.append(child)
            else:
                changes.extend(_changed_fields(left[key], right[key], path=child))
        return tuple(changes)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return (path or "/",)
        changes = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            changes.extend(
                _changed_fields(
                    left_item,
                    right_item,
                    path=f"{path}/{index}",
                )
            )
        return tuple(changes)
    return () if left == right else (path or "/",)


class ProposedPlan(BaseModel):
    """The exact Planner/skill/resume plan before current host shaping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.proposed_plan/1"] = "easyicu.proposed_plan/1"
    source: str = Field(min_length=1, max_length=120)
    plan_payload: dict[str, Any]
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _payload_is_typed_and_bound(self) -> "ProposedPlan":
        payload = _plan_payload(self.plan_payload)
        if payload != self.plan_payload:
            raise ValueError("proposed plan payload is not canonical AnalysisPlan JSON")
        if canonical_sha256(payload) != self.plan_sha256:
            raise ValueError("proposed plan digest does not bind its payload")
        return self

    @classmethod
    def create(
        cls,
        *,
        plan: AnalysisPlan | Mapping[str, Any],
        source: str,
    ) -> "ProposedPlan":
        payload = _plan_payload(plan)
        return cls(
            source=str(source),
            plan_payload=payload,
            plan_sha256=canonical_sha256(payload),
        )


class PlanTransformationReceipt(BaseModel):
    """One deterministic or provider-owned plan rewrite in a contiguous chain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.plan_transformation_receipt/1"] = (
        "easyicu.plan_transformation_receipt/1"
    )
    transformer: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=1_000)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_fields: tuple[str, ...]
    scientific_semantics_changed: bool
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _receipt_is_self_bound(self) -> "PlanTransformationReceipt":
        if tuple(sorted(set(self.changed_fields))) != self.changed_fields:
            raise ValueError("changed_fields must be unique and sorted")
        if self.input_sha256 == self.output_sha256 and self.changed_fields:
            raise ValueError("unchanged plan digests cannot report changed fields")
        if self.input_sha256 != self.output_sha256 and not self.changed_fields:
            raise ValueError("changed plan digests require changed fields")
        if not self.changed_fields and self.scientific_semantics_changed:
            raise ValueError("a no-op transform cannot change scientific semantics")
        unsigned = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if canonical_sha256(unsigned) != self.receipt_sha256:
            raise ValueError("plan transformation receipt digest mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        transformer: str,
        reason: str,
        input_plan: AnalysisPlan | Mapping[str, Any],
        output_plan: AnalysisPlan | Mapping[str, Any],
        scientific_semantics_changed: bool,
    ) -> "PlanTransformationReceipt":
        before = _plan_payload(input_plan)
        after = _plan_payload(output_plan)
        changed = tuple(sorted(set(_changed_fields(before, after))))
        body: dict[str, Any] = {
            "schema_version": "easyicu.plan_transformation_receipt/1",
            "transformer": str(transformer),
            "reason": str(reason),
            "input_sha256": canonical_sha256(before),
            "output_sha256": canonical_sha256(after),
            "changed_fields": list(changed),
            "scientific_semantics_changed": bool(
                scientific_semantics_changed and changed
            ),
        }
        body["receipt_sha256"] = canonical_sha256(body)
        return cls.model_validate(body)


class NormalizedPlan(BaseModel):
    """Host-normalized plan and its complete contiguous transformation chain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.normalized_plan/1"] = "easyicu.normalized_plan/1"
    proposed: ProposedPlan
    transformation_receipts: tuple[PlanTransformationReceipt, ...]
    plan_payload: dict[str, Any]
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_semantics_changed: bool
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _lineage_is_contiguous_and_bound(self) -> "NormalizedPlan":
        payload = _plan_payload(self.plan_payload)
        if payload != self.plan_payload or canonical_sha256(payload) != self.plan_sha256:
            raise ValueError("normalized plan payload/digest mismatch")
        cursor = self.proposed.plan_sha256
        for receipt in self.transformation_receipts:
            if receipt.input_sha256 != cursor:
                raise ValueError("plan transformation receipt chain is not contiguous")
            cursor = receipt.output_sha256
        if cursor != self.plan_sha256:
            raise ValueError("transformation chain does not end at normalized plan")
        expected_semantics = any(
            item.scientific_semantics_changed
            for item in self.transformation_receipts
        )
        if self.scientific_semantics_changed != expected_semantics:
            raise ValueError("normalized plan semantic-change projection mismatch")
        unsigned = self.model_dump(mode="json", exclude={"authority_sha256"})
        if canonical_sha256(unsigned) != self.authority_sha256:
            raise ValueError("normalized plan authority digest mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        proposed: ProposedPlan,
        transformation_receipts: Sequence[PlanTransformationReceipt],
        plan: AnalysisPlan | Mapping[str, Any],
    ) -> "NormalizedPlan":
        payload = _plan_payload(plan)
        receipts = tuple(transformation_receipts)
        body: dict[str, Any] = {
            "schema_version": "easyicu.normalized_plan/1",
            "proposed": proposed.model_dump(mode="json"),
            "transformation_receipts": [
                item.model_dump(mode="json") for item in receipts
            ],
            "plan_payload": payload,
            "plan_sha256": canonical_sha256(payload),
            "scientific_semantics_changed": any(
                item.scientific_semantics_changed for item in receipts
            ),
        }
        body["authority_sha256"] = canonical_sha256(body)
        return cls.model_validate(body)


class ApprovedExecutablePlan(BaseModel):
    """The normalized plan released to Execute by an exact human decision set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.approved_executable_plan/1"] = (
        "easyicu.approved_executable_plan/1"
    )
    plan_payload: dict[str, Any]
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_plan_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_review_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _approval_is_self_bound(self) -> "ApprovedExecutablePlan":
        payload = _plan_payload(self.plan_payload)
        if payload != self.plan_payload or canonical_sha256(payload) != self.plan_sha256:
            raise ValueError("approved executable plan payload/digest mismatch")
        unsigned = self.model_dump(mode="json", exclude={"authority_sha256"})
        if canonical_sha256(unsigned) != self.authority_sha256:
            raise ValueError("approved executable plan authority digest mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        normalized: NormalizedPlan,
        plan_review_authority: PlanReviewAuthority | Mapping[str, Any],
        decision_set_sha256: str,
    ) -> "ApprovedExecutablePlan":
        review = (
            plan_review_authority
            if isinstance(plan_review_authority, PlanReviewAuthority)
            else PlanReviewAuthority.model_validate(plan_review_authority)
        )
        if review.plan_sha256 != normalized.plan_sha256:
            raise PlanLifecycleAuthorityError(
                "human-review authority does not bind the normalized plan"
            )
        body: dict[str, Any] = {
            "schema_version": "easyicu.approved_executable_plan/1",
            "plan_payload": normalized.plan_payload,
            "plan_sha256": normalized.plan_sha256,
            "normalized_plan_authority_sha256": normalized.authority_sha256,
            "plan_review_authority_sha256": canonical_sha256(
                review.model_dump(mode="json")
            ),
            "decision_set_sha256": str(decision_set_sha256),
        }
        body["authority_sha256"] = canonical_sha256(body)
        return cls.model_validate(body)


def build_normalized_plan_lineage(
    *,
    proposed_plan: AnalysisPlan,
    proposed_source: str,
    pre_normalization_plan: AnalysisPlan,
    normalized_plan: AnalysisPlan,
    resume_scientific_semantics_changed: bool,
    host_scientific_semantics_changed: bool,
) -> NormalizedPlan:
    """Compile the two owner boundaries without bloating the pipeline host."""

    proposed = ProposedPlan.create(plan=proposed_plan, source=proposed_source)
    receipts: list[PlanTransformationReceipt] = []
    if proposed.plan_sha256 != canonical_sha256(_plan_payload(pre_normalization_plan)):
        receipts.append(
            PlanTransformationReceipt.create(
                transformer="resume.plan_authority_migration",
                reason=(
                    "Apply typed legacy resume migrations and restore prior "
                    "digest-verified plan authorities before validation."
                ),
                input_plan=proposed_plan,
                output_plan=pre_normalization_plan,
                scientific_semantics_changed=resume_scientific_semantics_changed,
            )
        )
    if canonical_sha256(_plan_payload(pre_normalization_plan)) != canonical_sha256(
        _plan_payload(normalized_plan)
    ):
        receipts.append(
            PlanTransformationReceipt.create(
                transformer="host.plan_normalization_pipeline",
                reason=(
                    "Close typed scientific contracts, products, figures, cohort, "
                    "robustness, repeated-unit and execution bindings."
                ),
                input_plan=pre_normalization_plan,
                output_plan=normalized_plan,
                scientific_semantics_changed=host_scientific_semantics_changed,
            )
        )
    return NormalizedPlan.create(
        proposed=proposed,
        transformation_receipts=receipts,
        plan=normalized_plan,
    )


def plan_lifecycle_evidence_id(revision: int) -> str:
    return f"plan_lifecycle_revision_{int(revision)}"


def persist_normalized_plan(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    normalized: NormalizedPlan,
) -> Path:
    """Register or exact-validate one immutable normalized-plan lineage."""

    revision = AnalysisPlan.model_validate(normalized.plan_payload).revision
    evidence_id = plan_lifecycle_evidence_id(revision)
    path = Path(run_dir) / f"{evidence_id}.json"
    existing = evidence.get(evidence_id)
    if existing is not None:
        verified = verified_run_evidence_path(Path(run_dir), existing)
        if verified is None:
            raise PlanLifecycleAuthorityError(
                f"registered plan lifecycle {evidence_id!r} is unavailable"
            )
        try:
            observed = NormalizedPlan.model_validate_json(verified.read_bytes())
        except Exception as exc:
            raise PlanLifecycleAuthorityError(
                f"registered plan lifecycle {evidence_id!r} is invalid"
            ) from exc
        # A resumed run may reconstruct a different *proposal narrative* for
        # the same immutable public plan (for example ``generated`` versus
        # ``resumed``).  The first registered lineage remains authoritative;
        # it may be reused only when its exact normalized plan is unchanged.
        if observed.plan_sha256 != normalized.plan_sha256:
            raise PlanLifecycleAuthorityError(
                f"plan lifecycle revision {revision} cannot be overwritten"
            )
        return verified
    if path.exists():
        raise PlanLifecycleAuthorityError(
            f"unregistered plan lifecycle path already exists: {path.name}"
        )
    path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Planner proposal, deterministic host transformations, and the exact "
            "normalized plan offered for approval."
        ),
        source_path=path,
        evidence_id=evidence_id,
        producer="plan_lifecycle_authority",
        generation_mode="deterministic_skill",
        metadata={
            "plan_revision": revision,
            "proposed_plan_sha256": normalized.proposed.plan_sha256,
            "normalized_plan_sha256": normalized.plan_sha256,
            "scientific_semantics_changed": normalized.scientific_semantics_changed,
            "transformation_count": len(normalized.transformation_receipts),
        },
    )
    return path


def load_normalized_plan(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    revision: int,
) -> NormalizedPlan:
    evidence_id = plan_lifecycle_evidence_id(revision)
    record = evidence.get(evidence_id)
    if record is None:
        raise PlanLifecycleAuthorityError(
            f"normalized plan authority {evidence_id!r} is absent"
        )
    verified = verified_run_evidence_path(Path(run_dir), record)
    if verified is None:
        raise PlanLifecycleAuthorityError(
            f"normalized plan authority {evidence_id!r} is unavailable"
        )
    try:
        return NormalizedPlan.model_validate_json(verified.read_bytes())
    except Exception as exc:
        raise PlanLifecycleAuthorityError(
            f"normalized plan authority {evidence_id!r} is invalid"
        ) from exc


def persist_approved_executable_plan(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    approved: ApprovedExecutablePlan,
) -> Path:
    plan = AnalysisPlan.model_validate(approved.plan_payload)
    evidence_id = f"approved_executable_plan_revision_{plan.revision}"
    path = Path(run_dir) / f"{evidence_id}.json"
    existing = evidence.get(evidence_id)
    if existing is not None:
        verified = verified_run_evidence_path(Path(run_dir), existing)
        if verified is None:
            raise PlanLifecycleAuthorityError(
                "registered approved executable plan is unavailable"
            )
        observed = ApprovedExecutablePlan.model_validate_json(verified.read_bytes())
        if observed != approved:
            raise PlanLifecycleAuthorityError(
                "approved executable plan revision cannot be overwritten"
            )
        return verified
    if path.exists():
        raise PlanLifecycleAuthorityError(
            f"unregistered approved plan path already exists: {path.name}"
        )
    path.write_text(approved.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Exact normalized analysis plan released to Execute by the "
            "digest-bound human decision set."
        ),
        source_path=path,
        evidence_id=evidence_id,
        producer="plan_lifecycle_authority",
        generation_mode="human_confirmed",
        metadata={
            "plan_revision": plan.revision,
            "plan_sha256": approved.plan_sha256,
            "decision_set_sha256": approved.decision_set_sha256,
        },
    )
    return path


def approve_normalized_plan_for_execution(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    revision: int,
    review_requests: Sequence[Any],
    decision_set_sha256: str,
) -> ApprovedExecutablePlan:
    """Bind one reviewed normalized plan to the decision released to Execute.

    Both the live in-process pause and durable restart recovery call this owner.
    Keeping request parsing here prevents those control-plane paths from
    independently deciding which scientific authority a human approved.
    """

    authorities: dict[str, PlanReviewAuthority] = {}
    for request in review_requests:
        if isinstance(request, Mapping):
            request_payload = request.get("payload")
        else:
            request_payload = getattr(request, "payload", None)
        if not isinstance(request_payload, Mapping):
            continue
        raw_authority = request_payload.get("plan_review_authority")
        if not isinstance(raw_authority, Mapping):
            continue
        try:
            authority = PlanReviewAuthority.model_validate(raw_authority)
        except Exception as exc:
            raise PlanLifecycleAuthorityError(
                "human-review request contains an invalid plan authority"
            ) from exc
        authorities[canonical_sha256(authority.model_dump(mode="json"))] = authority
    if len(authorities) != 1:
        raise PlanLifecycleAuthorityError(
            "approved execution requires one shared plan review authority"
        )

    normalized = load_normalized_plan(
        run_dir=run_dir,
        evidence=evidence,
        revision=revision,
    )
    approved = ApprovedExecutablePlan.create(
        normalized=normalized,
        plan_review_authority=next(iter(authorities.values())),
        decision_set_sha256=decision_set_sha256,
    )
    persist_approved_executable_plan(
        run_dir=run_dir,
        evidence=evidence,
        approved=approved,
    )
    return approved


__all__ = [
    "ApprovedExecutablePlan",
    "NormalizedPlan",
    "PlanLifecycleAuthorityError",
    "PlanTransformationReceipt",
    "ProposedPlan",
    "approve_normalized_plan_for_execution",
    "build_normalized_plan_lineage",
    "load_normalized_plan",
    "persist_approved_executable_plan",
    "persist_normalized_plan",
    "plan_lifecycle_evidence_id",
]
