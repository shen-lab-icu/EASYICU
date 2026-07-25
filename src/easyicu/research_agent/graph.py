"""LangGraph orchestration for the research-agent pipeline.

This module is the default wrapper that orchestrates the existing
``plan → execute → write → finalise`` phases of
:class:`~easyicu.research_agent.pipeline.ResearchAgentPipeline` as a
``langgraph.graph.StateGraph``. LangGraph is the default phase dispatcher; the
existing EasyICU receipts, capsules, evidence and checkpoint remain the sole
scientific and replay authority.

Why this design:

* The graph receives **invoker callables** rather than raw kwargs. The
  pipeline closes over its prelude locals (audit logger, progress
  emitter, run dir, etc.) when constructing these callables, so the
  graph itself stays free of pipeline-specific argument plumbing.
* Phase outputs flow through a ``TypedDict`` graph state, typed with the
  real ``_PlanPhaseResult`` / ``_ExecutePhaseResult`` / ``_WritePhaseResult``
  contracts so a type checker can see phase-contract drift. Those live in
  ``contracts.runtime``, which does not import this module, so the top-level
  import is not a cycle — and it has to be top-level, because langgraph
  resolves the state annotations when the ``StateGraph`` is constructed.
* Aborts during planning route directly to ``END`` without running
  execute/write/finalise. The pipeline's own ``_finalise_aborted`` has
  already been called inside ``_run_plan_phase`` in that branch, so the
  ``final_result`` is populated by the plan node.

``langgraph`` is a core dependency of the research-agent runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Imported at module scope, not under TYPE_CHECKING: langgraph resolves the
# state TypedDict's annotations at ``StateGraph(...)`` construction time, so a
# forward reference here raises NameError. Neither module imports this one, so
# there is no cycle.
from .contracts.runtime import (
    _ExecutePhaseResult,
    _PlanPhaseResult,
    _WritePhaseResult,
)
from .schema import PipelineResult

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - py<3.8 not supported anyway
    from typing_extensions import TypedDict  # type: ignore[no-redef]


__all__ = [
    "HUMAN_REVIEW_FINDING_REASONS",
    "HUMAN_REVIEW_RESUME_SCOPE",
    "HumanReviewAuthorityError",
    "HumanReviewDecision",
    "HumanReviewPending",
    "HumanReviewRequest",
    "OrchestrationRuntimeReceipt",
    "PipelineGraphState",
    "build_pipeline_graph",
    "human_review_requests_for_plan",
    "orchestration_runtime_receipt",
]


#: What a caller may assume about resuming a paused run.
#:
#: ``same_process`` is the honest label for what exists today. The phase
#: handoffs a resumed run needs (the plan phase's live ``EvidenceStore``, its
#: lock, the resolved context) are held in the compiled graph's registry, not
#: in the checkpoint — a checkpointer cannot serialise them, which is why they
#: were moved out of the state in the first place. A new process therefore
#: cannot reconstruct the run from the checkpoint alone.
#:
#: This is a declared property of the pause rather than a docstring so an
#: operator UI can read it and decline to present the run as durably
#: resumable. Making it durable requires reconstructible phase handoffs, which
#: is a pipeline change, not a graph change.
HUMAN_REVIEW_RESUME_SCOPE = "same_process"


class HumanReviewAuthorityError(RuntimeError):
    """Raised when a review request cannot be bound to the state it approves.

    Distinct from a validation error: nothing about the plan is wrong. The run
    simply cannot prove *what* a reviewer would be signing, and an approval
    that binds nothing would cover anything.
    """


def _review_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

    A paused run has *no* ``PipelineResult`` — the graph stopped inside the
    review node and nothing downstream of it has executed. Returning a typed
    object rather than reaching into the interrupted state for a
    ``final_result`` that is not there is what makes the pause a supported
    outcome instead of a ``KeyError``.

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
    """Build the auditable record for one approved human review.

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
        "request_sha256": _canonical_sha256(request_payload),
        "decision_sha256": _canonical_sha256(decision_payload),
    }
    return record


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


class OrchestrationRuntimeReceipt(BaseModel):
    """Non-scientific receipt identifying the phase dispatcher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.orchestration_runtime/1"] = (
        "easyicu.orchestration_runtime/1"
    )
    backend: Literal["langgraph"]
    backend_version: str
    phase_order: tuple[str, ...] = ("plan", "execute", "write", "finalise")
    checkpoint_authority: Literal["easyicu_receipt_capsule_checkpoint"] = (
        "easyicu_receipt_capsule_checkpoint"
    )
    scientific_authority: Literal["easyicu_host_control_plane"] = (
        "easyicu_host_control_plane"
    )


def orchestration_runtime_receipt() -> OrchestrationRuntimeReceipt:
    """Return the exact dispatcher identity without changing scientific state."""

    try:
        version = metadata.version("langgraph")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "LangGraph is required by the default research-agent runtime; "
            "reinstall EasyICU so its core dependencies are complete"
        ) from exc
    return OrchestrationRuntimeReceipt(backend="langgraph", backend_version=version)


class PipelineGraphState(TypedDict, total=False):
    """Mutable state passed between graph nodes.

    Fields are populated incrementally as each node runs. ``aborted``
    is set by the plan node when ``_PlanPhaseResult.aborted_result`` is
    not ``None``; the conditional edge after ``plan`` then routes
    directly to ``END`` and the value of ``final_result`` is already
    correct.

    The phase-result fields carry their real types rather than ``Any``, so a
    type checker can flag phase-contract drift. They must be imported at
    module scope: langgraph resolves these annotations when the
    ``StateGraph`` is constructed, and a ``TYPE_CHECKING``-only import would
    raise ``NameError`` there.

    The three phase handoffs are ``_phase_ref`` handles rather than the objects
    themselves whenever a checkpointer is configured. A ``_PlanPhaseResult``
    holds an open ``EvidenceStore`` (and therefore a ``threading.RLock``), and
    a checkpointer serialises every node output — so putting it in the state
    made *any* checkpointed run die at the first write, which in turn made the
    human-review interrupt unusable in production. The objects live in a
    per-graph registry instead; the state carries only their keys.
    """

    plan_result: _PlanPhaseResult
    execute_result: _ExecutePhaseResult
    write_result: _WritePhaseResult
    final_result: PipelineResult
    aborted: bool
    human_review_decisions: tuple[dict[str, Any], ...]
    phase_refs: dict[str, str]


def build_pipeline_graph(
    *,
    plan_invoker: Callable[[], Any],
    execute_invoker: Callable[[Any], Any],
    write_invoker: Callable[[Any, Any], Any],
    finalise_invoker: Callable[[Any, Any, Any], Any],
    provenance_hook: Optional[Callable[[Any], None]] = None,
    human_review_invoker: Optional[
        Callable[[Any], Sequence[HumanReviewRequest]]
    ] = None,
    human_review_recorder: Optional[
        Callable[[Sequence[Mapping[str, Any]]], None]
    ] = None,
    reviewer_identity_resolver: Optional[Callable[[], str]] = None,
    checkpointer: Any = None,
):
    """Build the compiled langgraph StateGraph for the pipeline.

    Parameters
    ----------
    plan_invoker:
        Zero-arg callable that runs the plan phase and returns a
        ``_PlanPhaseResult``. If its ``aborted_result`` field is not
        ``None`` the graph routes to ``END``.
    execute_invoker:
        Receives the plan result, returns an ``_ExecutePhaseResult``.
    write_invoker:
        Receives ``(plan_result, execute_result)``, returns a
        ``_WritePhaseResult``.
    finalise_invoker:
        Receives ``(plan_result, execute_result, write_result)``,
        returns the final ``PipelineResult``.
    provenance_hook:
        Optional callable invoked after a successful plan with the
        plan result, mirroring the O27 raw-EHR provenance step in
        :meth:`ResearchAgentPipeline.run`.

    Returns
    -------
    A compiled langgraph runnable. Invoke it with an empty initial
    state dict; the final state's ``final_result`` field is the
    ``PipelineResult``.
    """

    from langgraph.graph import StateGraph, END

    # Per-graph registry for the unserialisable phase handoffs. One compiled
    # graph is one run, and ``resume_human_review`` drives the *same* compiled
    # graph, so the registry is alive for the whole pause-and-resume cycle.
    phase_results: dict[str, Any] = {}

    def _put(name: str, value: Any) -> dict[str, Any]:
        """Store a phase result and return what belongs in the graph state."""

        phase_results[name] = value
        if checkpointer is None:
            return {name: value}
        return {"phase_refs": {name: name}}

    def _get(state: PipelineGraphState, name: str) -> Any:
        if name in state:
            return state[name]
        if name in phase_results:
            return phase_results[name]
        raise RuntimeError(
            f"phase result {name!r} is not available in this process; a run "
            "paused for human review must be resumed through the same "
            "ResearchAgentPipeline instance that started it"
        )

    def plan_node(state: PipelineGraphState) -> dict[str, Any]:
        plan_result = plan_invoker()
        aborted_result = (
            plan_result.get("aborted_result")
            if isinstance(plan_result, Mapping)
            else plan_result.aborted_result
        )
        if aborted_result is not None:
            return {
                **_put("plan_result", plan_result),
                "final_result": aborted_result,
                "aborted": True,
            }
        if provenance_hook is not None:
            provenance_hook(plan_result)
        return {**_put("plan_result", plan_result), "aborted": False}

    def execute_node(state: PipelineGraphState) -> dict[str, Any]:
        execute_result = execute_invoker(_get(state, "plan_result"))
        return _put("execute_result", execute_result)

    def human_review_node(state: PipelineGraphState) -> dict[str, Any]:
        if human_review_invoker is None:
            return {"human_review_decisions": ()}
        requests = tuple(human_review_invoker(_get(state, "plan_result")))
        if not requests:
            return {"human_review_decisions": ()}
        request_ids = [item.review_id for item in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("human review requests must have unique review_id values")
        from langgraph.types import interrupt

        raw = interrupt(
            {
                "schema_version": "easyicu.human_review_interrupt/1",
                "requests": [item.model_dump(mode="json") for item in requests],
            }
        )
        if not isinstance(raw, Mapping) or not isinstance(raw.get("decisions"), list):
            raise ValueError("human review resume payload must contain decisions")
        decisions = tuple(
            HumanReviewDecision.model_validate(item) for item in raw["decisions"]
        )
        decision_ids = [item.review_id for item in decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("human review decisions must have unique review_id values")
        expected = {item.review_id: item for item in requests}
        observed = {item.review_id: item for item in decisions}
        if set(observed) != set(expected):
            raise ValueError("human review decisions must cover exact paused requests")
        records: list[dict[str, Any]] = []
        for review_id, decision in observed.items():
            request = expected[review_id]
            if decision.authority_sha256 != request.authority_sha256:
                raise ValueError("human review decision authority digest mismatch")
            if decision.decision != "approved":
                raise RuntimeError(f"human review rejected request {review_id}")
            records.append(
                _human_review_decision_record(
                    request=request,
                    decision=decision,
                    reviewer_identity=(
                        reviewer_identity_resolver()
                        if reviewer_identity_resolver is not None
                        else None
                    ),
                )
            )
        if human_review_recorder is not None:
            # Binding the decision into the run's own evidence store is what
            # lets the finished run answer "who approved what, against which
            # authority digest, and when" — the graph state alone is discarded
            # when the process exits.
            human_review_recorder(tuple(records))
        return {"human_review_decisions": tuple(records)}

    def write_node(state: PipelineGraphState) -> dict[str, Any]:
        write_result = write_invoker(
            _get(state, "plan_result"), _get(state, "execute_result")
        )
        return _put("write_result", write_result)

    def finalise_node(state: PipelineGraphState) -> dict[str, Any]:
        final_result = finalise_invoker(
            _get(state, "plan_result"),
            _get(state, "execute_result"),
            _get(state, "write_result"),
        )
        return {"final_result": final_result}

    def route_after_plan(state: PipelineGraphState) -> str:
        return "abort" if state.get("aborted") else "continue"

    graph: StateGraph = StateGraph(PipelineGraphState)
    graph.add_node("plan", plan_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("execute", execute_node)
    graph.add_node("write", write_node)
    graph.add_node("finalise", finalise_node)
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"abort": END, "continue": "human_review"},
    )
    graph.add_edge("human_review", "execute")
    graph.add_edge("execute", "write")
    graph.add_edge("write", "finalise")
    graph.add_edge("finalise", END)
    return graph.compile(checkpointer=checkpointer)
