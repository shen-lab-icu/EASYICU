"""Real LangGraph interrupt + checkpoint HITL driver for retrofit review.

A retrofit paper-readiness review must PROVE the operator decision flowed through
a genuine LangGraph interrupt + checkpointer + resume, not a bare constructed
``HumanReviewDecision``. A raw ``HumanReviewDecision`` is just a data record: any
code can instantiate one, so on its own it does not show the decision ever passed
the framework's human-in-the-loop pause.

This module runs a minimal single-node review graph with a real checkpointer. It
emits the digest-bound :class:`HumanReviewRequest`, hits ``langgraph.types.interrupt``
(which persists the paused state through the checkpointer), and only resumes when
the operator supplies a matching :class:`HumanReviewDecision` — validated with the
exact same rules as the pipeline's ``human_review_node`` (unique ids, cover the
paused requests, authority digest match, approved). It returns the decisions plus
a :class:`HumanReviewCheckpointReceipt` binding the interrupt payload digest, the
resume digest, and the post-resume checkpoint id.

Honest scope: the interrupt + checkpoint prove the MECHANISM — a real graph
pause/resume validated the decision against the paused request. The operator
identity remains a trusted local claim; nothing in an automated context can prove
a biological human typed it, so ``operator_identity`` is labelled exactly that.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field

from easyicu.research_agent.graph import HumanReviewDecision, HumanReviewRequest

__all__ = [
    "HUMAN_REVIEW_CHECKPOINT_SCHEMA",
    "HumanReviewCheckpointReceipt",
    "checkpoint_receipt_sha256",
    "run_human_review_interrupt",
    "verify_checkpoint_receipt_binds_request",
]

HUMAN_REVIEW_CHECKPOINT_SCHEMA = "easyicu.human_review_checkpoint/1"
_INTERRUPT_SCHEMA = "easyicu.human_review_interrupt/1"


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _interrupt_payload(requests: Sequence[HumanReviewRequest]) -> dict[str, Any]:
    return {
        "schema_version": _INTERRUPT_SCHEMA,
        "requests": [item.model_dump(mode="json") for item in requests],
    }


def _resume_payload(decisions: Sequence[HumanReviewDecision]) -> dict[str, Any]:
    return {"decisions": [item.model_dump(mode="json") for item in decisions]}


class HumanReviewCheckpointReceipt(BaseModel):
    """Proof a HITL decision flowed through a real interrupt + checkpoint resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.human_review_checkpoint/1"] = (
        HUMAN_REVIEW_CHECKPOINT_SCHEMA
    )
    backend: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=200)
    checkpoint_id: str = Field(min_length=1, max_length=200)
    review_ids: Tuple[str, ...] = Field(min_length=1)
    interrupt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resume_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # The interrupt proves the pause/resume mechanism, never a biological human.
    operator_identity: Literal["trusted_local_claim"] = "trusted_local_claim"


DecideFn = Callable[[Tuple[HumanReviewRequest, ...]], Sequence[HumanReviewDecision]]


def run_human_review_interrupt(
    requests: Sequence[HumanReviewRequest],
    *,
    decide: DecideFn,
    thread_id: str,
) -> Tuple[Tuple[HumanReviewDecision, ...], HumanReviewCheckpointReceipt]:
    """Run a real interrupt-backed HITL review and return (decisions, receipt).

    Builds a one-node LangGraph with a real checkpointer, interrupts on the paused
    ``requests``, obtains decisions from ``decide`` (the operator), resumes through
    ``Command(resume=...)``, and captures a checkpoint receipt. Raises if the graph
    does not actually interrupt, if the decisions do not cover the paused requests,
    if an authority digest mismatches, or if any decision is not ``approved``.
    """

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command, interrupt

    try:  # py>=3.8
        from typing import TypedDict
    except ImportError:  # pragma: no cover
        from typing_extensions import TypedDict  # type: ignore[no-redef]

    reqs = tuple(requests)
    if not reqs:
        raise ValueError("no human review requests to run")
    request_ids = [item.review_id for item in reqs]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("human review requests must have unique review_id values")
    if not str(thread_id).strip():
        raise ValueError("thread_id is required for a checkpointed review")

    interrupt_payload = _interrupt_payload(reqs)
    expected = {item.review_id: item for item in reqs}

    class _ReviewState(TypedDict, total=False):
        decisions: tuple[dict[str, Any], ...]

    def review_node(state: "_ReviewState") -> dict[str, Any]:
        raw = interrupt(interrupt_payload)
        if not isinstance(raw, Mapping) or not isinstance(raw.get("decisions"), list):
            raise ValueError("human review resume payload must contain decisions")
        decisions = tuple(
            HumanReviewDecision.model_validate(item) for item in raw["decisions"]
        )
        decision_ids = [item.review_id for item in decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("human review decisions must have unique review_id values")
        observed = {item.review_id: item for item in decisions}
        if set(observed) != set(expected):
            raise ValueError("human review decisions must cover exact paused requests")
        for review_id, decision in observed.items():
            if decision.authority_sha256 != expected[review_id].authority_sha256:
                raise ValueError("human review decision authority digest mismatch")
            if decision.decision != "approved":
                raise RuntimeError(f"human review rejected request {review_id}")
        return {"decisions": tuple(item.model_dump(mode="json") for item in decisions)}

    graph: StateGraph = StateGraph(_ReviewState)
    graph.add_node("human_review", review_node)
    graph.set_entry_point("human_review")
    graph.add_edge("human_review", END)
    saver = MemorySaver()
    app = graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": str(thread_id)}}

    first = app.invoke({}, config)
    if not isinstance(first, Mapping) or "__interrupt__" not in first:
        raise RuntimeError("review graph did not interrupt for human review")

    decisions = tuple(decide(reqs))
    if not decisions:
        raise ValueError("operator returned no human review decisions")
    resume_payload = _resume_payload(decisions)
    app.invoke(Command(resume=resume_payload), config)

    snapshot = app.get_state(config)
    if snapshot.next:
        raise RuntimeError("review graph did not resume to completion after decision")
    checkpoint_id = str(
        (snapshot.config or {}).get("configurable", {}).get("checkpoint_id") or ""
    )
    if not checkpoint_id:
        raise RuntimeError("review checkpoint id is unavailable after resume")

    receipt = HumanReviewCheckpointReceipt(
        backend=type(saver).__name__,
        thread_id=str(thread_id),
        checkpoint_id=checkpoint_id,
        review_ids=tuple(request_ids),
        interrupt_sha256=_digest(interrupt_payload),
        resume_sha256=_digest(resume_payload),
    )
    return decisions, receipt


def checkpoint_receipt_sha256(receipt: HumanReviewCheckpointReceipt | Mapping) -> str:
    """Canonical digest of a checkpoint receipt (dict or model)."""

    if isinstance(receipt, HumanReviewCheckpointReceipt):
        payload = receipt.model_dump(mode="json")
    else:
        payload = dict(receipt)
    return _digest(payload)


def verify_checkpoint_receipt_binds_request(
    receipt: Mapping[str, Any],
    *,
    request: HumanReviewRequest,
    decision: HumanReviewDecision,
) -> HumanReviewCheckpointReceipt:
    """Re-verify a stored checkpoint receipt binds ``request`` and ``decision``.

    Offline (no live checkpointer needed): re-derives the interrupt and resume
    digests from the rebuilt request + embedded decision and requires the stored
    receipt to match. A tampered or unrelated receipt fails closed. Returns the
    validated model.
    """

    try:
        parsed = HumanReviewCheckpointReceipt.model_validate(dict(receipt))
    except Exception as exc:  # noqa: BLE001 - surfaced by caller as a fail-close
        raise ValueError(f"not a valid human review checkpoint receipt: {exc}") from exc
    if parsed.review_ids != (request.review_id,):
        raise ValueError("checkpoint receipt does not bind this review request")
    if parsed.interrupt_sha256 != _digest(_interrupt_payload((request,))):
        raise ValueError("checkpoint interrupt digest does not match the request")
    if parsed.resume_sha256 != _digest(_resume_payload((decision,))):
        raise ValueError("checkpoint resume digest does not match the decision")
    return parsed
