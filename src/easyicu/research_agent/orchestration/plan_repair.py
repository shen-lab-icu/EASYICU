"""Bounded repair of a fresh plan before it can be offered for review."""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, NotRequired, Sequence, TypeVar, TypedDict

from ..authority.plan_review import HumanReviewRequest
from ..planning.progressive_contract import ProgressivePlanCompileError
from ..planning.question_quote import normalize_question_quote
from ..planning.question_requirements import (
    QUESTION_REQUIREMENTS_REVIEW_FILENAME,
    QUESTION_REQUIREMENT_APPROVAL_STOPS,
)
from ..providers.structured_retry import automatic_retry_limit

PLAN_REPAIR_LIMIT = 2
_T = TypeVar("_T")


class PlanRepairReceipt(TypedDict):
    repairs: int
    limit: int
    outcome: Literal["repeated_failure", "budget_exhausted"]


class _RequirementRepair(TypedDict):
    requirement_id: str
    quote: str
    gap: Mapping[str, Any] | None
    host_verdict: str
    host_fact: str | None
    reason_code: NotRequired[str]
    effect_measure: NotRequired[str]
    uncovered_clauses: NotRequired[tuple[str, ...]]


@dataclass(frozen=True)
class _Feedback:
    code: str
    requirements: tuple[_RequirementRepair, ...] = ()
    check: Mapping[str, Any] | None = None

    @property
    def signature(self) -> tuple[str, tuple[tuple[str, str], ...]]:
        # Requirement IDs can be reassigned by the Planner. The frozen
        # question's normalized quotes identify the same unmet requirements.
        uncovered = {
            (normalize_question_quote(row["quote"]), normalize_question_quote(quote))
            for row in self.requirements
            for quote in (row.get("uncovered_clauses") or (row["quote"] or row.get("reason_code", ""),))
        }
        return self.code, tuple(sorted(uncovered))

    def payload(self) -> dict[str, Any]:
        if self.check is not None:
            return {"check": dict(self.check), "requirements": self.requirements}
        return {"check_code": self.code, "requirements": self.requirements}


class _RetryPlan(Exception):
    """Internal handoff before a draft becomes a resumable human review."""


@dataclass
class _Repair:
    limit: int
    history: list[_Feedback] = field(default_factory=list)
    required: dict[str, _RequirementRepair] = field(default_factory=dict)
    receipt: PlanRepairReceipt | None = None
    planner_started: bool = False

    def retry(self, feedback: _Feedback) -> bool:
        repeated = any(previous.signature == feedback.signature for previous in self.history)
        if repeated or len(self.history) >= self.limit:
            self.receipt = {
                "repairs": len(self.history), "limit": self.limit,
                "outcome": "repeated_failure" if repeated else "budget_exhausted",
            }
            return False
        self.history.append(feedback)
        return True


_ACTIVE_REPAIR: ContextVar[_Repair | None] = ContextVar("active_plan_repair", default=None)


def planning_repair_prompt() -> str:
    """Mark Planner entry and return only the host's typed repair feedback."""
    state = _ACTIVE_REPAIR.get()
    if state is None:
        return ""
    state.planner_started = True
    if not state.history:
        return ""
    return (
        "Host plan repair feedback (JSON data). Fix the failed checks and keep each quoted "
        "requirement verbatim in the requirement record. Preserve the original study question, "
        "source, outcome, estimand and human decisions. A quote is evidence of what the user "
        "asked. Do not remove, reclassify, or weaken it to bypass a check.\n"
        + json.dumps([feedback.payload() for feedback in state.history], ensure_ascii=False, separators=(",", ":"))
    )


def check_plan_requirements_for_repair(run_dir: Path, requests: Sequence[HumanReviewRequest]) -> None:
    """Read the host's judgment of the final shaped plan before it can pause."""
    state = _ACTIVE_REPAIR.get()
    if state is None or not state.planner_started:
        return
    reasons = {request.payload.get("reason") for request in requests}
    if not state.required and not reasons.intersection(QUESTION_REQUIREMENT_APPROVAL_STOPS.values()):
        return
    path = run_dir / QUESTION_REQUIREMENTS_REVIEW_FILENAME
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    rows = {normalize_question_quote(row["quote"]): row for row in record.get("judged", ())}
    unresolved: dict[str, _RequirementRepair] = {}
    for quote, row in rows.items():
        if row.get("exposure_choice"):
            continue  # This changes the estimand and requires a human choice.
        verdict = row.get("gap_verification")
        uncovered = row["disposition"] == "not_covered" and row.get("verified_by_host") is True
        unverified_gap = row["disposition"] == "capability_gap" and verdict != "verified"
        if uncovered or unverified_gap:
            unresolved[quote] = {
                "requirement_id": row["id"], "quote": row["quote"], "gap": row.get("gap"),
                "host_verdict": verdict or "not_covered", "host_fact": row.get("gap_fact"),
                **({"reason_code": row["reason_code"]} if row.get("reason_code") else {}),
                **({"effect_measure": row["effect_measure"]} if row.get("effect_measure") else {}),
                "uncovered_clauses": tuple(
                    clause["quote"] for clause in row.get("clauses", ())
                    if clause["disposition"] == "not_covered"
                ),
            }
    omitted = {
        quote: prior for quote, prior in state.required.items()
        if quote not in unresolved and rows.get(quote, {}).get("verified_by_host") is not True
    }
    # Some host checks report an extra effect measure that the user never
    # requested. Feed it back but let the next plan remove it.
    state.required.update({quote: row for quote, row in unresolved.items() if quote})
    pending = tuple({**unresolved, **omitted}.values())
    if not pending:
        return
    feedback = _Feedback(QUESTION_REQUIREMENT_APPROVAL_STOPS["not_covered"], pending)
    if state.retry(feedback):
        raise _RetryPlan()
    if omitted:
        error = ProgressivePlanCompileError(
            "progressive_question_requirement_repair_omitted",
            "The revised plan no longer provides a host judgment for a previously unanswered requirement.",
            path="question_requirements",
        )
        error.easyicu_plan_repair = state.receipt
        raise error


def _check_feedback(check: Mapping[str, Any], required: dict[str, _RequirementRepair]) -> _Feedback:
    receipt = dict(check)
    requirement = receipt.pop("requirement", None)
    rows = ()
    if requirement is not None:
        required[normalize_question_quote(requirement["quote"])] = requirement
        rows = (requirement,)
    return _Feedback(str(check["reason_code"]), requirements=rows, check=receipt)


def run_plan_with_repairs(
    run: Callable[[], _T],
    *,
    check_failure: Callable[[Exception], Mapping[str, Any]],
    notify: Callable[[int], None],
    enabled: bool = True,
) -> tuple[_T, PlanRepairReceipt | None]:
    """Keep the child job and provider task while trying fresh internal drafts."""
    limit = automatic_retry_limit(PLAN_REPAIR_LIMIT) if enabled else 0
    if not limit:
        return run(), None
    state = _Repair(limit)
    token = _ACTIVE_REPAIR.set(state)
    try:
        while True:
            state.planner_started = False
            try:
                return run(), state.receipt
            except _RetryPlan:
                pass
            except Exception as error:
                check = check_failure(error) if state.planner_started else {}
                if not check or getattr(error, "easyicu_plan_repair", None):
                    raise
                if not state.retry(_check_feedback(check, state.required)):
                    error.easyicu_plan_repair = state.receipt
                    raise
            # Report the repair before sending another Planner request. A
            # cancellation raised by the host here never starts a new draft.
            notify(len(state.history))
    finally:
        _ACTIVE_REPAIR.reset(token)
