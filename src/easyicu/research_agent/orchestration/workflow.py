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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..authority.plan_review import PlanReviewAuthority, ReviewExecutionAuthority
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
    "HumanReviewStateDrift",
    "OrchestrationRuntimeReceipt",
    "PipelineRunOutcome",
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


class HumanReviewStateDrift(RuntimeError):
    """Raised when what was approved is no longer what would execute.

    A decision's ``authority_sha256`` proves it answers *the request that was
    made*. It cannot prove the run still holds the plan that request described:
    the plan handoff is a live mutable object held across the pause, so
    anything that touched it while the operator was deciding would execute
    under a signature that never covered it. Resume therefore re-derives the
    authority from the live handoff and refuses when it no longer matches.

    Distinct from :class:`HumanReviewAuthorityError`, which means the run
    cannot *tell* what is bound (transient, still resumable). Drift is a
    definitive integrity violation and terminalizes the run.
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


#: What :meth:`ResearchAgentPipeline.run` and its wrappers actually return.
#:
#: Annotating those entry points as ``PipelineResult`` alone was a promise the
#: pipeline does not keep: a run that stops for review returns
#: :class:`HumanReviewPending` instead, and nothing downstream of the pause has
#: produced a result to return. A caller typed against the narrower annotation
#: reads ``.manuscript`` off the pause and fails at runtime with an attribute
#: error rather than being told at the type level to handle the pause.
PipelineRunOutcome = Union[PipelineResult, HumanReviewPending]


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


def _plan_authority_payload(
    plan: Any,
    evidence: Any,
    execution_authority: ReviewExecutionAuthority | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The plan state a reviewer's signature is a signature *of*.

    The nested :class:`PlanReviewAuthority` owns the complete typed plan rather
    than a hand-maintained subset.  Compatibility fields remain at the top
    level for existing clients, but they are derived from the same validated
    packet and are not the security boundary.
    """

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
    try:
        authority = PlanReviewAuthority.create(
            plan=plan,
            evidence_sha256=digests,
            execution=execution_authority,
        )
    except Exception as exc:  # noqa: BLE001 - translated into a control-plane block
        raise HumanReviewAuthorityError(
            "cannot build a human-review authority digest from a complete "
            f"typed analysis plan ({exc}). A partial plan must not be approved."
        ) from exc

    plan_payload = authority.plan_payload
    return {
        "plan_revision": plan_payload["revision"],
        "plan_step_ids": [
            str(step["step_id"]) for step in plan_payload.get("steps", ())
        ],
        "plan_evidence_sha256": authority.evidence_sha256,
        "plan_review_authority": authority.model_dump(mode="json"),
    }


def _shared_plan_authority(
    requests: Sequence[HumanReviewRequest],
) -> Optional[dict[str, Any]]:
    """Return the one plan-authority packet every request in a pause shares.

    ``_plan_authority_payload`` is computed once per pause and spread into
    every request's payload, so any single request carries the whole approved
    plan. Reading it back is how resume compares what was approved with what
    the run currently holds.
    """

    return _shared_plan_authority_payload(
        {"payload": request.payload} for request in requests
    )


def _shared_plan_authority_payload(
    records: Iterable[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """Same lookup over serialized request records rather than live models.

    Resume reads the *approved* authority out of its private snapshot, which
    holds plain dicts, and the *live* one out of the current requests. Both go
    through this so the two sides cannot drift apart in how they are read.
    """

    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        authority = payload.get("plan_review_authority")
        if isinstance(authority, Mapping):
            return dict(authority)
    return None


def human_review_requests_for_plan(
    *,
    findings: Sequence[Any],
    plan: Any = None,
    evidence: Any = None,
    execution_authority: ReviewExecutionAuthority | Mapping[str, Any] | None = None,
    require_plan_review: bool = False,
) -> tuple[HumanReviewRequest, ...]:
    """Derive the review requests a completed plan phase implies.

    Deliberately derived from *typed finding state* rather than from a caller
    flag: the point of the gate is that the run itself decides when a human is
    required, so an operator cannot disable it by not asking for it.

    ``evidence`` binds the approval to the artefacts the plan rests on. Passing
    it is what makes the authority digest change when the underlying evidence
    changes, so a stale approval cannot be replayed onto revised work.
    """

    reviewable: list[tuple[Any, str, str, Mapping[str, Any]]] = []
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
        reviewable.append((finding, reason, kind, detail))

    if not reviewable and not require_plan_review:
        return ()

    requests: list[HumanReviewRequest] = []
    seen: set[str] = set()
    plan_authority = _plan_authority_payload(
        plan,
        evidence,
        execution_authority,
    )
    if require_plan_review:
        payload = {
            "validator": "operator_plan_review_policy",
            "severity": "error",
            "reason": "operator_plan_approval_required",
            "evidence_ids": [],
            "capability_request_sha256": None,
            **plan_authority,
        }
        request = HumanReviewRequest.create(
            kind="scientific_stop",
            summary=(
                "Review and explicitly approve the complete digest-bound "
                "analysis plan before any analysis step executes."
            ),
            authority_sha256=_review_digest(payload),
            payload=payload,
        )
        seen.add(request.review_id)
        requests.append(request)
    for finding, reason, kind, detail in reviewable:
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
        #: The pause exactly as it was offered, as JSON-serialized deep copies.
        #:
        #: ``HumanReviewRequest`` is ``frozen=True``, which freezes its
        #: *attributes* -- it does not stop anyone holding the request from
        #: mutating the ``payload`` dict in place. Reading the approved plan
        #: back out of the live request therefore compares the run against a
        #: value the run itself no longer controls: rewrite that dict to
        #: describe the new plan and the comparison is mutated-against-live,
        #: which passes. These copies are private, are never handed out, and
        #: are what resume compares against.
        self._pause_snapshot: tuple[dict[str, Any], ...] = ()
        self._pause_digest: str = ""
        #: Built decision records, keyed by the digest of the decision set that
        #: produced them. See :meth:`_decision_records_for` for why a resubmitted
        #: decision must not be re-stamped.
        self._decision_record_cache: dict[str, tuple[dict[str, Any], ...]] = {}

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
                # Take the snapshot before the pause is visible to anyone, so
                # what resume compares against is what the operator was asked.
                self._pause_snapshot = tuple(
                    item.model_dump(mode="json") for item in requests
                )
                self._pause_digest = canonical_sha256(list(self._pause_snapshot))
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
        # Before any comparison that reads the requests: the requests must
        # still be the ones that were offered. Everything below compares a
        # decision against `self._requests`, so a request rewritten during the
        # pause would be checked against itself and agree with itself.
        try:
            self._verify_requests_match_the_pause_offered()
        except HumanReviewStateDrift:
            self._discard_live_pause(state="failed")
            raise
        parsed = tuple(HumanReviewDecision.model_validate(item) for item in decisions)
        decision_ids = [item.review_id for item in parsed]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("human review decisions must have unique review_id values")
        expected = {item.review_id: item for item in self._requests}
        observed = {item.review_id: item for item in parsed}
        if set(observed) != set(expected):
            raise ValueError("human review decisions must cover exact paused requests")

        # Request order is the host-owned order. Client map/list ordering must
        # not change evidence record order or downstream manifest digests.
        ordered = tuple(observed[request.review_id] for request in self._requests)
        for request, decision in zip(self._requests, ordered):
            if decision.authority_sha256 != request.authority_sha256:
                raise ValueError("human review decision authority digest mismatch")

        # A matching digest proves the decision answers the request that was
        # made. It says nothing about whether the run still holds the plan that
        # request described, and the plan handoff is live and mutable across
        # the pause. Re-derive the authority from the current handoff before
        # anything is recorded or executed.
        try:
            self._verify_pause_still_binds_live_state()
        except HumanReviewStateDrift:
            # Definitive: the approved state and the executable state differ.
            # Terminalize rather than leave a pause that a caller could retry
            # into executing unapproved work.
            self._discard_live_pause(state="failed")
            raise

        records = self._decision_records_for(ordered)
        rejected_ids = [
            decision.review_id
            for decision in ordered
            if decision.decision == "rejected"
        ]

        # Recording the decision and acting on it are separate acts, and only
        # the second one is irreversible. The recorder writes two files and
        # registers two evidence entries, so a full disk, a permission change
        # or an evidence-store hiccup can fail it -- and discarding the live
        # handoff there would throw away Planner work the operator already
        # paid for, to recover from a transient write. Stay paused instead:
        # `_requests` and `_plan_result` are intact, so the same decision can
        # be resubmitted. Only a failure past this point is terminal.
        if rejected_ids:
            self._record_human_review(records)
            self._discard_live_pause(state="rejected")
            raise HumanReviewRejected(rejected_ids)

        self._record_human_review(records)
        try:
            return self._finish(tuple(records))
        except Exception:
            self._discard_live_pause(state="failed")
            raise

    def _verify_requests_match_the_pause_offered(self) -> None:
        """Refuse when the paused requests are not the ones that were offered.

        ``authority_sha256`` and ``review_id`` are frozen strings, but the
        ``payload`` they bind is a plain dict that anything holding the pause
        can rewrite in place -- including the embedded plan authority that
        :meth:`_verify_pause_still_binds_live_state` reads back as "what was
        approved". Rewriting it to describe the new plan makes that check
        compare the new plan with itself, so the drift guard passes and
        unapproved work executes under the old signature.

        Comparing the live requests with the private snapshot closes both that
        rewrite and a wholesale swap of the request tuple, because it checks
        the bytes rather than any single field. It is a definitive integrity
        violation, so it terminalizes like other drift.
        """

        if not self._pause_digest:
            return
        live = [item.model_dump(mode="json") for item in self._requests]
        if canonical_sha256(live) == self._pause_digest:
            return
        offered = {item["review_id"] for item in self._pause_snapshot}
        present = {str(item.get("review_id") or "") for item in live}
        if offered != present:
            detail = (
                f"the pause offered {sorted(offered)} but the run now holds "
                f"{sorted(present)}"
            )
        else:
            changed = sorted(
                item["review_id"]
                for item, was in zip(live, self._pause_snapshot)
                if item != was
            )
            detail = f"request(s) {changed} were modified after the pause"
        raise HumanReviewStateDrift(
            "the paused review request is not the one that was offered for "
            f"review ({detail}). A decision can only authorize the request the "
            "operator was actually shown."
        )

    def _verify_pause_still_binds_live_state(self) -> None:
        """Re-derive the approved authority and refuse if the run has moved.

        The three parts are compared differently, on purpose:

        * ``plan_sha256`` must be **identical** -- it is the scientific plan
          the reviewer read.
        * The execution identity must be **identical** -- pipeline config,
          capability activation, submission profile and run input capsule are
          the environment the reviewer signed off.
        * Evidence is checked as a **subset**: every artefact bound at the
          pause must still be present with the same digest, but additions are
          allowed. Resubmitting a decision after a failed recorder write
          legitimately adds the decision log itself, and treating that growth
          as tampering would deadlock the retry this engine promises.
        """

        if self._human_review_invoker is None or self._plan_result is None:
            return
        # Read the approved side from the private snapshot, never from the live
        # request: the request's payload is mutable in place, so deriving both
        # sides from it would compare the run against a value the run does not
        # own. `_verify_requests_match_the_pause_offered` already refuses a
        # rewritten request; this keeps the comparison correct on its own.
        approved = _shared_plan_authority_payload(self._pause_snapshot)
        if approved is None:
            return
        try:
            current_requests = tuple(self._human_review_invoker(self._plan_result))
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed blocker
            # The run cannot *tell* what is bound. That is the transient,
            # still-resumable case: same class the recorder uses, so the
            # operator can retry rather than lose the Planner work.
            raise HumanReviewAuthorityError(
                "cannot re-derive the human-review authority from the live run "
                f"state ({exc}), so an approval cannot be shown to still bind "
                "what would execute. The pause is left resumable."
            ) from exc
        current = _shared_plan_authority(current_requests)
        if current is None:
            raise HumanReviewStateDrift(
                "the human-review condition that paused this run no longer "
                "derives from its own plan state, so the decision on record "
                "approves a run state that no longer exists."
            )
        approved_plan = str(approved.get("plan_sha256") or "")
        current_plan = str(current.get("plan_sha256") or "")
        if current_plan != approved_plan:
            raise HumanReviewStateDrift(
                "the analysis plan changed after it was sent for review "
                f"(approved plan_sha256={approved_plan[:8]}, live "
                f"plan_sha256={current_plan[:8]}). The decision on record "
                "approves the earlier plan, so the current one must not "
                "execute under it."
            )
        if current.get("execution") != approved.get("execution"):
            raise HumanReviewStateDrift(
                "the execution identity changed after the plan was sent for "
                "review (pipeline config, capability activation, submission "
                "profile or run input capsule). The approval covers the "
                "identity the reviewer saw, not this one."
            )
        approved_evidence = dict(approved.get("evidence_sha256") or {})
        current_evidence = dict(current.get("evidence_sha256") or {})
        for evidence_id, digest in sorted(approved_evidence.items()):
            live = current_evidence.get(evidence_id)
            if live is None:
                raise HumanReviewStateDrift(
                    f"evidence {evidence_id!r} was bound into the approved "
                    "authority but is no longer in the run's evidence store, "
                    "so the approval rests on an artefact the run cannot show."
                )
            if str(live) != str(digest):
                raise HumanReviewStateDrift(
                    f"evidence {evidence_id!r} changed after review (approved "
                    f"sha256={str(digest)[:8]}, live sha256={str(live)[:8]})."
                )

    def _decision_records_for(
        self,
        ordered: Sequence[HumanReviewDecision],
    ) -> list[dict[str, Any]]:
        """Build the auditable records once per decision set, then reuse them.

        Building is *not* idempotent on its own: every build stamps a fresh
        ``server_decided_at``, which changes the decision file's bytes and so
        its SHA-256. The recorder registers that file under a fixed evidence
        id, and the evidence store refuses a fixed id whose digest changed.
        A resubmission after a partially failed write would therefore have been
        rejected forever -- turning the resumable pause this engine promises
        into a permanent deadlock. Caching by decision-set digest makes a retry
        present byte-identical content, which the store accepts as the same
        artefact.

        A genuinely *different* decision set gets its own fresh record, so the
        cache cannot backdate a decision the operator has not made yet.
        """

        key = canonical_sha256([item.model_dump(mode="json") for item in ordered])
        cached = self._decision_record_cache.get(key)
        if cached is not None:
            return [dict(record) for record in cached]
        reviewer_identity = (
            self._reviewer_identity_resolver()
            if self._reviewer_identity_resolver is not None
            else None
        )
        records = [
            _human_review_decision_record(
                request=request,
                decision=decision,
                reviewer_identity=reviewer_identity,
            )
            for request, decision in zip(self._requests, ordered)
        ]
        self._decision_record_cache[key] = tuple(dict(item) for item in records)
        return records

    def _record_human_review(self, records: list[dict[str, Any]]) -> None:
        """Persist the decision, leaving the pause resumable if that fails."""

        if self._human_review_recorder is not None:
            self._human_review_recorder(tuple(records))

    def _discard_live_pause(self, *, state: str) -> None:
        """Terminalize the engine and release non-rehydratable live handoffs."""

        self._state = state
        self._requests = ()
        self._pause_snapshot = ()
        self._pause_digest = ""
        self._plan_result = None
        # The run is over; nothing can be resubmitted, so the retry cache has
        # no remaining purpose and should not outlive the decisions it holds.
        self._decision_record_cache.clear()

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
