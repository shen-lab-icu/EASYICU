"""Explicit state-machine orchestration for the research-agent pipeline.

This module owns the one production phase dispatcher:
``plan → human review → execute → write → finalise``. EasyICU receipts,
capsules, evidence and checkpoints remain the sole scientific and replay
authority.

Why this design:

* The workflow receives **invoker callables** rather than raw kwargs. The
  pipeline closes over its prelude locals (audit logger, progress
  emitter, run dir, etc.) when constructing these callables, so the
  workflow itself stays free of pipeline-specific argument plumbing.
* Phase handoffs retain their real ``_PlanPhaseResult`` /
  ``_ExecutePhaseResult`` / ``_WritePhaseResult`` contracts. There is no
  second shadow state or process-local lookup table pretending to be a
  durable checkpoint.
* Aborts during planning return directly without running
  execute/write/finalise. The pipeline's own ``_finalise_aborted`` has
  already been called inside ``_run_plan_phase`` in that branch.

This deliberately chooses an explicit state machine over a half-adopted graph
framework. Human-review pauses remain honestly ``same_process`` until phase
handoffs have a complete artifact-rehydration contract.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical_json import canonical_sha256
from ..contracts.runtime import (
    _ExecutePhaseResult,
    _PlanPhaseResult,
    _WritePhaseResult,
)
from ..schema import PipelineResult


__all__ = [
    "HUMAN_REVIEW_FINDING_REASONS",
    "HUMAN_REVIEW_RESUME_SCOPE",
    "HumanReviewAuthorityError",
    "HumanReviewDecision",
    "HumanReviewPending",
    "HumanReviewRejected",
    "HumanReviewRequest",
    "OrchestrationRuntimeReceipt",
    "PipelineWorkflow",
    "WorkflowEngine",
    "WorkflowCompleted",
    "WorkflowPaused",
    "build_pipeline_workflow",
    "human_review_requests_for_plan",
    "orchestration_runtime_receipt",
]


#: What a caller may assume about resuming a paused run.
#:
#: ``same_process`` is the honest label for what exists today. The plan handoff
#: contains a live ``EvidenceStore``, provider resolver and run-scoped services.
#: Those objects have no complete artifact-rehydration contract, so a new
#: process cannot reconstruct the paused workflow from a run id alone.
#:
#: This is a declared property of the pause rather than a docstring so an
#: operator UI can read it and decline to present the run as durably
#: resumable. Making it durable requires reconstructible phase handoffs.
HUMAN_REVIEW_RESUME_SCOPE = "same_process"


class HumanReviewAuthorityError(RuntimeError):
    """Raised when a review request cannot be bound to the state it approves.

    Distinct from a validation error: nothing about the plan is wrong. The run
    simply cannot prove *what* a reviewer would be signing, and an approval
    that binds nothing would cover anything.
    """


class HumanReviewRejected(RuntimeError):
    """Terminal operator rejection of one or more paused review requests."""

    def __init__(self, review_ids: Sequence[str]) -> None:
        self.review_ids = tuple(str(item) for item in review_ids)
        super().__init__(
            "human review rejected request(s): " + ", ".join(self.review_ids)
        )


def _review_digest(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible review payload with the shared canonical owner."""

    return canonical_sha256(dict(payload))


class HumanReviewRequest(BaseModel):
    """Digest-bound pause request; it carries no authority by itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str = Field(pattern=r"^review-[0-9a-f]{16}$")
    kind: Literal["protocol_claim", "capability_request", "scientific_stop"]
    summary: str = Field(min_length=1, max_length=1_000)
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict[str, Any]

    @model_validator(mode="after")
    def _id_binds_payload(self) -> "HumanReviewRequest":
        expected = (
            "review-" + _review_digest(self.model_dump(exclude={"review_id"}))[:16]
        )
        if self.review_id != expected:
            raise ValueError("human-review id does not bind request contents")
        return self

    @classmethod
    def create(
        cls,
        *,
        kind: Literal["protocol_claim", "capability_request", "scientific_stop"],
        summary: str,
        authority_sha256: str,
        payload: Mapping[str, Any],
    ) -> "HumanReviewRequest":
        body = {
            "kind": kind,
            "summary": summary,
            "authority_sha256": authority_sha256,
            "payload": dict(payload),
        }
        return cls(review_id="review-" + _review_digest(body)[:16], **body)


class HumanReviewDecision(BaseModel):
    """Explicit operator decision bound to the paused request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=200)
    decided_at: str = Field(min_length=1, max_length=80)
    note: str = Field(default="", max_length=1_000)


class HumanReviewPending(BaseModel):
    """What :meth:`ResearchAgentPipeline.run` returns when the run paused.

    A paused run has *no* ``PipelineResult`` — the workflow stopped before
    execution and nothing downstream has run. Returning a typed object makes
    the pause a supported outcome instead of an absent-result error.

    ``thread_id`` is the resume coordinate: pass it (or the whole object) back
    to :meth:`ResearchAgentPipeline.resume_human_review` together with one
    decision per request.

    ``resume_scope`` states the boundary explicitly rather than leaving a
    caller to discover it: see :data:`HUMAN_REVIEW_RESUME_SCOPE`. A UI that
    hands the pause to a different worker, or persists it across a restart,
    must read this field and refuse rather than find out at resume time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.human_review_pending/2"] = (
        "easyicu.human_review_pending/2"
    )
    run_id: str
    thread_id: str
    run_dir: str
    requests: tuple[HumanReviewRequest, ...]
    #: Literal, not a free string: widening it is a deliberate edit here, not
    #: something a caller can assert by passing a different value.
    resume_scope: Literal["same_process"] = HUMAN_REVIEW_RESUME_SCOPE
    #: The process that can answer this pause. A caller comparing it to its own
    #: pid knows before prompting a human whether the answer can be delivered.
    resume_pid: int = Field(default_factory=os.getpid)

    @property
    def review_ids(self) -> tuple[str, ...]:
        return tuple(item.review_id for item in self.requests)

    @property
    def resumable_here(self) -> bool:
        """True when this process is the one that can accept the decisions."""

        return self.resume_pid == os.getpid()


#: Plan-phase finding reasons that must not be walked past unattended, mapped
#: to the review kind they raise. Keyed on the typed ``detail["reason"]`` rather
#: than on message text so a reworded finding cannot silently drop out of the
#: gate. A finding may also opt in directly with
#: ``detail["human_review_required"] = True``.
HUMAN_REVIEW_FINDING_REASONS: Mapping[str, str] = {
    "capability_review_required": "capability_request",
    "raw_ehr_provenance_unavailable": "protocol_claim",
    "scientific_stop": "scientific_stop",
}


def _plan_authority_payload(plan: Any, evidence: Any) -> dict[str, Any]:
    """The plan state a reviewer's signature is a signature *of*.

    Binding only the finding text meant one approval covered any plan that
    raised the same finding: the reviewer's authority digest did not move when
    the plan was revised or when the evidence underneath it changed. Including
    the revision and the source-artefact digests makes the approval specific to
    what was actually shown, so an edited plan needs a new signature.
    """

    payload: dict[str, Any] = {
        "plan_revision": getattr(plan, "revision", None),
        "plan_step_ids": [
            str(getattr(step, "step_id", "")) for step in getattr(plan, "steps", ())
        ],
    }
    digests: dict[str, str] = {}
    if evidence is not None:
        try:
            for record in evidence.records():
                digests[str(record.evidence_id)] = str(record.sha256)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed blocker
            # Swallowing this produced an approval request whose authority
            # digest bound *no evidence at all*, which is exactly the state a
            # signature is supposed to be a signature of. A reviewer would then
            # be approving a plan the run cannot show them. Fail closed: no
            # readable evidence, no review request.
            raise HumanReviewAuthorityError(
                "cannot build a human-review authority digest: the evidence "
                f"store is unreadable ({exc}). An approval that binds no "
                "evidence would cover any plan, so the review is not offered."
            ) from exc
    payload["plan_evidence_sha256"] = dict(sorted(digests.items()))
    return payload


def human_review_requests_for_plan(
    *,
    findings: Sequence[Any],
    plan: Any = None,
    evidence: Any = None,
) -> tuple[HumanReviewRequest, ...]:
    """Derive the review requests a completed plan phase implies.

    Deliberately derived from *typed finding state* rather than from a caller
    flag: the point of the gate is that the run itself decides when a human is
    required, so an operator cannot disable it by not asking for it.

    ``evidence`` binds the approval to the artefacts the plan rests on. Passing
    it is what makes the authority digest change when the underlying evidence
    changes, so a stale approval cannot be replayed onto revised work.
    """

    requests: list[HumanReviewRequest] = []
    seen: set[str] = set()
    plan_authority = _plan_authority_payload(plan, evidence)
    for finding in findings or ():
        # Severity is the run's own statement about whether the state blocks.
        # The same reason code can be raised as a warning by a development
        # profile that makes no claim about it (raw-EHR provenance is the live
        # example), and a warning must not halt a run waiting for a signature.
        if str(getattr(finding, "severity", "") or "") != "error":
            continue
        detail = getattr(finding, "detail", None) or {}
        reason = str(detail.get("reason") or "")
        kind = HUMAN_REVIEW_FINDING_REASONS.get(reason)
        if kind is None and detail.get("human_review_required"):
            kind = "scientific_stop"
        if kind is None:
            continue
        payload = {
            "validator": getattr(finding, "validator", None),
            "severity": getattr(finding, "severity", None),
            "reason": reason or "human_review_required",
            "evidence_ids": list(getattr(finding, "evidence_ids", ()) or ()),
            # A capability request is the thing being approved in the
            # ``capability_request`` case, so its own digest is part of what
            # the signature covers.
            "capability_request_sha256": detail.get("capability_request_sha256"),
            **plan_authority,
        }
        request = HumanReviewRequest.create(
            kind=kind,  # type: ignore[arg-type]
            summary=str(getattr(finding, "message", "") or reason)[:1_000],
            authority_sha256=_review_digest(payload),
            payload=payload,
        )
        if request.review_id in seen:
            continue
        seen.add(request.review_id)
        requests.append(request)
    return tuple(requests)


def _human_review_decision_record(
    *,
    request: HumanReviewRequest,
    decision: HumanReviewDecision,
    reviewer_identity: Optional[str],
) -> dict[str, Any]:
    """Build the auditable record for one human-review decision.

    ``reviewer`` and ``decided_at`` arrive in the resume payload, so they are
    whatever the client typed. They are kept for context but are not the
    authority: ``reviewer_identity`` comes from the caller's authentication
    layer (``None`` when the deployment has none, which is itself recorded),
    and ``server_decided_at`` is stamped here so a decision cannot be
    backdated by editing the payload.
    """

    request_payload = request.model_dump(mode="json")
    decision_payload = decision.model_dump(mode="json")
    record: dict[str, Any] = {
        "schema": "easyicu.human_review_decision/1",
        "review_id": decision.review_id,
        "authority_sha256": decision.authority_sha256,
        "decision": decision.decision,
        "claimed_reviewer": decision.reviewer,
        "claimed_decided_at": decision.decided_at,
        "reviewer_identity": reviewer_identity,
        "reviewer_identity_source": (
            "authenticated" if reviewer_identity else "unauthenticated_client_claim"
        ),
        "server_decided_at": datetime.now(timezone.utc).isoformat(),
        "note": decision.note,
        "request_sha256": canonical_sha256(request_payload),
        "decision_sha256": canonical_sha256(decision_payload),
    }
    return record


class OrchestrationRuntimeReceipt(BaseModel):
    """Non-scientific receipt identifying the sole phase dispatcher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.orchestration_runtime/2"] = (
        "easyicu.orchestration_runtime/2"
    )
    backend: Literal["explicit_state_machine"] = "explicit_state_machine"
    backend_version: Literal["1"] = "1"
    phase_order: tuple[str, ...] = (
        "plan",
        "human_review",
        "execute",
        "write",
        "finalise",
    )
    checkpoint_authority: Literal["easyicu_receipt_capsule_checkpoint"] = (
        "easyicu_receipt_capsule_checkpoint"
    )
    scientific_authority: Literal["easyicu_host_control_plane"] = (
        "easyicu_host_control_plane"
    )


def orchestration_runtime_receipt() -> OrchestrationRuntimeReceipt:
    """Return the exact dispatcher identity without changing scientific state."""

    return OrchestrationRuntimeReceipt()


@dataclass(frozen=True)
class WorkflowCompleted:
    """Terminal workflow outcome."""

    final_result: PipelineResult
    human_review_decisions: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class WorkflowPaused:
    """Supported non-terminal outcome awaiting operator decisions."""

    requests: tuple[HumanReviewRequest, ...]


@runtime_checkable
class WorkflowEngine(Protocol):
    """Stable orchestration seam independent of a framework implementation.

    A future durable engine may implement this interface after it can
    rehydrate every phase handoff from artifact references. Callers should
    depend on this contract, not on :class:`PipelineWorkflow` internals.
    """

    @property
    def state(self) -> str:
        """Return the engine's current lifecycle state."""

    def start(self) -> WorkflowCompleted | WorkflowPaused:
        """Run until completion or an explicit human-review pause."""

    def resume(
        self,
        decisions: Sequence[HumanReviewDecision | Mapping[str, Any]],
    ) -> WorkflowCompleted:
        """Validate the active pause, continuing only when every item is approved.

        A valid rejection raises :class:`HumanReviewRejected` after its
        evidence record is persisted and permanently terminalizes the engine.
        """


class PipelineWorkflow:
    """Single-owner state machine for one pipeline run.

    The object intentionally retains the live plan handoff only while a review
    is paused. It does not claim that a generic checkpointer can serialize an
    ``EvidenceStore`` or provider resolver.
    """

    def __init__(
        self,
        *,
        plan_invoker: Callable[[], _PlanPhaseResult],
        execute_invoker: Callable[[_PlanPhaseResult], _ExecutePhaseResult],
        write_invoker: Callable[
            [_PlanPhaseResult, _ExecutePhaseResult], _WritePhaseResult
        ],
        finalise_invoker: Callable[
            [_PlanPhaseResult, _ExecutePhaseResult, _WritePhaseResult],
            PipelineResult,
        ],
        provenance_hook: Optional[Callable[[_PlanPhaseResult], None]] = None,
        human_review_invoker: Optional[
            Callable[[_PlanPhaseResult], Sequence[HumanReviewRequest]]
        ] = None,
        human_review_recorder: Optional[
            Callable[[Sequence[Mapping[str, Any]]], None]
        ] = None,
        reviewer_identity_resolver: Optional[Callable[[], str]] = None,
    ) -> None:
        self._plan_invoker = plan_invoker
        self._execute_invoker = execute_invoker
        self._write_invoker = write_invoker
        self._finalise_invoker = finalise_invoker
        self._provenance_hook = provenance_hook
        self._human_review_invoker = human_review_invoker
        self._human_review_recorder = human_review_recorder
        self._reviewer_identity_resolver = reviewer_identity_resolver
        self._state = "created"
        self._plan_result: Optional[_PlanPhaseResult] = None
        self._requests: tuple[HumanReviewRequest, ...] = ()

    @property
    def state(self) -> str:
        return self._state

    def start(self) -> WorkflowCompleted | WorkflowPaused:
        """Run from planning until completion or an explicit review pause."""

        if self._state != "created":
            raise RuntimeError(
                f"workflow start requires state 'created', found {self._state!r}"
            )
        try:
            plan_result = self._plan_invoker()
            aborted_result = (
                plan_result.get("aborted_result")
                if isinstance(plan_result, Mapping)
                else plan_result.aborted_result
            )
            if aborted_result is not None:
                self._state = "completed"
                return WorkflowCompleted(final_result=aborted_result)
            if self._provenance_hook is not None:
                self._provenance_hook(plan_result)
            self._plan_result = plan_result
            requests = (
                tuple(self._human_review_invoker(plan_result))
                if self._human_review_invoker is not None
                else ()
            )
            request_ids = [item.review_id for item in requests]
            if len(request_ids) != len(set(request_ids)):
                raise ValueError(
                    "human review requests must have unique review_id values"
                )
            if requests:
                self._requests = requests
                self._state = "paused"
                return WorkflowPaused(requests=requests)
            return self._finish(())
        except Exception:
            if self._state != "paused":
                self._state = "failed"
            raise

    def resume(
        self,
        decisions: Sequence[HumanReviewDecision | Mapping[str, Any]],
    ) -> WorkflowCompleted:
        """Validate decisions for the exact pause and continue the run."""

        if self._state != "paused" or self._plan_result is None:
            raise RuntimeError(
                f"workflow resume requires state 'paused', found {self._state!r}"
            )
        parsed = tuple(HumanReviewDecision.model_validate(item) for item in decisions)
        decision_ids = [item.review_id for item in parsed]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("human review decisions must have unique review_id values")
        expected = {item.review_id: item for item in self._requests}
        observed = {item.review_id: item for item in parsed}
        if set(observed) != set(expected):
            raise ValueError("human review decisions must cover exact paused requests")

        records: list[dict[str, Any]] = []
        rejected_ids: list[str] = []
        # Request order is the host-owned order. Client map/list ordering must
        # not change evidence record order or downstream manifest digests.
        for request in self._requests:
            review_id = request.review_id
            decision = observed[review_id]
            if decision.authority_sha256 != request.authority_sha256:
                raise ValueError("human review decision authority digest mismatch")
            records.append(
                _human_review_decision_record(
                    request=request,
                    decision=decision,
                    reviewer_identity=(
                        self._reviewer_identity_resolver()
                        if self._reviewer_identity_resolver is not None
                        else None
                    ),
                )
            )
            if decision.decision == "rejected":
                rejected_ids.append(review_id)

        if rejected_ids:
            try:
                if self._human_review_recorder is not None:
                    self._human_review_recorder(tuple(records))
            except Exception:
                self._discard_live_pause(state="failed")
                raise
            self._discard_live_pause(state="rejected")
            raise HumanReviewRejected(rejected_ids)

        try:
            if self._human_review_recorder is not None:
                self._human_review_recorder(tuple(records))
            return self._finish(tuple(records))
        except Exception:
            self._discard_live_pause(state="failed")
            raise

    def _discard_live_pause(self, *, state: str) -> None:
        """Terminalize the engine and release non-rehydratable live handoffs."""

        self._state = state
        self._requests = ()
        self._plan_result = None

    def _finish(
        self,
        decisions: tuple[dict[str, Any], ...],
    ) -> WorkflowCompleted:
        if self._plan_result is None:
            raise RuntimeError("workflow cannot execute without a plan result")
        execute_result = self._execute_invoker(self._plan_result)
        write_result = self._write_invoker(self._plan_result, execute_result)
        final_result = self._finalise_invoker(
            self._plan_result,
            execute_result,
            write_result,
        )
        completed = WorkflowCompleted(
            final_result=final_result,
            human_review_decisions=decisions,
        )
        self._discard_live_pause(state="completed")
        return completed


def build_pipeline_workflow(
    *,
    plan_invoker: Callable[[], _PlanPhaseResult],
    execute_invoker: Callable[[_PlanPhaseResult], _ExecutePhaseResult],
    write_invoker: Callable[[_PlanPhaseResult, _ExecutePhaseResult], _WritePhaseResult],
    finalise_invoker: Callable[
        [_PlanPhaseResult, _ExecutePhaseResult, _WritePhaseResult],
        PipelineResult,
    ],
    provenance_hook: Optional[Callable[[_PlanPhaseResult], None]] = None,
    human_review_invoker: Optional[
        Callable[[_PlanPhaseResult], Sequence[HumanReviewRequest]]
    ] = None,
    human_review_recorder: Optional[
        Callable[[Sequence[Mapping[str, Any]]], None]
    ] = None,
    reviewer_identity_resolver: Optional[Callable[[], str]] = None,
) -> WorkflowEngine:
    """Build the explicit phase dispatcher for one pipeline run."""

    return PipelineWorkflow(
        plan_invoker=plan_invoker,
        execute_invoker=execute_invoker,
        write_invoker=write_invoker,
        finalise_invoker=finalise_invoker,
        provenance_hook=provenance_hook,
        human_review_invoker=human_review_invoker,
        human_review_recorder=human_review_recorder,
        reviewer_identity_resolver=reviewer_identity_resolver,
    )
