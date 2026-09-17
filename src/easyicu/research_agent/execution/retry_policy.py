"""Execution retry-policy contract: failure_class x budget mapping + denominator.

C-F13: centralizes the retry/repair budget rules that were previously inline
in the execute phase (fail-closed classes never spend an LLM code-repair
attempt; other failures consume the bounded repair budget). This module is
the single source of truth the classifier, the candidate loop, and the
validation report all consult: ``repair_route_for`` governs the loop branch,
unknown classes fail closed everywhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class FailureClassBudget:
    """Retry budget for one closed runtime-failure class."""

    failure_class: str
    consumes_llm_repair_attempt: bool
    max_llm_repairs: int
    route: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


#: failure_class value -> retry budget. Fail-closed classes never spend an LLM
#: code-repair attempt (repair is the wrong instrument); ordinary code defects
#: consume the bounded repair budget owned by the execute phase.
FAILURE_CLASS_RETRY_BUDGET: Tuple[FailureClassBudget, ...] = (
    FailureClassBudget(
        failure_class="plan_data_contract",
        consumes_llm_repair_attempt=False,
        max_llm_repairs=0,
        route="fail_closed",
    ),
    FailureClassBudget(
        failure_class="execution_timeout",
        consumes_llm_repair_attempt=False,
        max_llm_repairs=0,
        route="fail_closed",
    ),
    FailureClassBudget(
        failure_class="isolation_backend_unavailable",
        consumes_llm_repair_attempt=False,
        max_llm_repairs=0,
        route="fail_closed",
    ),
    FailureClassBudget(
        failure_class="deterministic_model_not_estimable",
        consumes_llm_repair_attempt=False,
        max_llm_repairs=0,
        route="fail_closed",
    ),
)

_BY_FAILURE_CLASS: Dict[str, FailureClassBudget] = {
    item.failure_class: item for item in FAILURE_CLASS_RETRY_BUDGET
}


def budget_for_failure_class(failure_class: str) -> FailureClassBudget | None:
    """Return the budget row for one failure_class, or None without guessing."""

    return _BY_FAILURE_CLASS.get(str(failure_class or "").strip())


def repair_route_for(failure_class: str) -> str:
    """Return the retry route governing one failure class.

    The candidate loop consults this (not a local copy of the rules) before
    deciding between fail-closed return and LLM repair. An unknown class is
    a classifier/contract drift and fails closed loudly instead of silently
    taking either path.
    """

    row = budget_for_failure_class(failure_class)
    if row is None:
        raise ValueError(
            f"unknown runtime failure class: {failure_class!r}; register it in "
            "execution/retry_policy.py before routing retries for it"
        )
    return row.route


#: Denominator definition for retry/repair accounting. Every step attempt
#: keeps its slot in the denominator, including fail-closed attempts that
#: never spent an LLM repair (they still consumed wall-clock / isolation /
#: deterministic-estimability outcomes). The numerator is LLM code-repair
#: attempts actually spent.
RETRY_DENOMINATOR_DEFINITION: str = (
    "all step attempts including fail-closed attempts without LLM repair; "
    "numerator counts LLM code-repair attempts actually spent"
)


def retry_denominator(
    attempts: int,
    *,
    include_fail_closed: bool = True,
) -> int:
    """Return the retry-accounting denominator for a step-attempt count.

    Pure helper preserving existing behavior: the denominator always includes
    fail-closed attempts (``include_fail_closed=True``). The flag exists only
    to document the rule; passing False is rejected to prevent silent
    denominator narrowing.
    """

    if not include_fail_closed:
        raise ValueError("retry denominator must include fail-closed attempts")
    return max(0, int(attempts))


def retry_policy_receipt() -> dict[str, object]:
    """Return a stable receipt for the failure_class x budget table."""

    return {
        "schema_version": "easyicu.retry_policy/1",
        "denominator": RETRY_DENOMINATOR_DEFINITION,
        "budgets": [item.to_dict() for item in FAILURE_CLASS_RETRY_BUDGET],
    }


def retry_accounting_receipt(
    attempt_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bind the retry policy to the attempts and repairs observed at runtime.

    ``step_attempt_history`` contains multiple checkpoint snapshots for a
    single attempt, so accounting is keyed by ``attempt_id`` and retains the
    largest cumulative repair counter observed for that attempt. Records from
    older manifests without an attempt id remain distinct denominator rows.
    """

    attempts: dict[str, dict[str, object]] = {}
    for index, record in enumerate(attempt_records):
        raw_attempt_id = record.get("attempt_id")
        attempt_id = (
            str(raw_attempt_id).strip()
            if isinstance(raw_attempt_id, str) and raw_attempt_id.strip()
            else f"legacy_record:{index + 1}"
        )
        raw_repairs = record.get("code_repair_attempts", 0)
        if isinstance(raw_repairs, bool) or not isinstance(raw_repairs, int):
            raise ValueError(
                f"invalid code_repair_attempts for {attempt_id!r}: {raw_repairs!r}"
            )
        if raw_repairs < 0:
            raise ValueError(
                f"negative code_repair_attempts for {attempt_id!r}: {raw_repairs}"
            )

        raw_failure_class = record.get("runtime_failure_class") or record.get(
            "failure_class"
        )
        failure_class = (
            str(raw_failure_class).strip() if raw_failure_class is not None else ""
        )
        if failure_class and budget_for_failure_class(failure_class) is None:
            raise ValueError(
                f"unknown runtime failure class: {failure_class!r}; cannot seal "
                "retry accounting"
            )

        row = attempts.setdefault(
            attempt_id,
            {
                "attempt_id": attempt_id,
                "step_id": str(record.get("step_id") or ""),
                "code_repair_attempts": 0,
                "runtime_failure_class": "",
            },
        )
        row["code_repair_attempts"] = max(
            int(row["code_repair_attempts"]), raw_repairs
        )
        prior_failure_class = str(row["runtime_failure_class"] or "")
        if failure_class and prior_failure_class not in {"", failure_class}:
            raise ValueError(
                f"inconsistent runtime failure classes for {attempt_id!r}: "
                f"{prior_failure_class!r} and {failure_class!r}"
            )
        if failure_class:
            row["runtime_failure_class"] = failure_class

    attempt_rows = list(attempts.values())
    failure_class_counts: dict[str, int] = {}
    for row in attempt_rows:
        failure_class = str(row["runtime_failure_class"] or "")
        if failure_class:
            failure_class_counts[failure_class] = (
                failure_class_counts.get(failure_class, 0) + 1
            )

    policy = retry_policy_receipt()
    canonical_policy = json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    repair_count = sum(int(row["code_repair_attempts"]) for row in attempt_rows)
    return {
        "schema_version": "easyicu.retry_accounting/1",
        "policy_sha256": hashlib.sha256(canonical_policy).hexdigest(),
        "denominator_definition": RETRY_DENOMINATOR_DEFINITION,
        "attempt_denominator": retry_denominator(len(attempt_rows)),
        "llm_code_repair_attempts": repair_count,
        "attempts_with_llm_code_repair": sum(
            int(row["code_repair_attempts"] > 0) for row in attempt_rows
        ),
        "fail_closed_attempts": sum(failure_class_counts.values()),
        "failure_class_counts": dict(sorted(failure_class_counts.items())),
        "attempts": attempt_rows,
    }


__all__ = [
    "FAILURE_CLASS_RETRY_BUDGET",
    "RETRY_DENOMINATOR_DEFINITION",
    "FailureClassBudget",
    "budget_for_failure_class",
    "repair_route_for",
    "retry_accounting_receipt",
    "retry_denominator",
    "retry_policy_receipt",
]


def _check_contract() -> None:
    seen = [item.failure_class for item in FAILURE_CLASS_RETRY_BUDGET]
    if len(seen) != len(set(seen)):
        raise RuntimeError("retry-policy failure classes must be unique")
    for item in FAILURE_CLASS_RETRY_BUDGET:
        if item.consumes_llm_repair_attempt:
            raise RuntimeError("fail-closed classes must not consume LLM repair")
        if item.max_llm_repairs != 0 or item.route != "fail_closed":
            raise RuntimeError("fail-closed budget rows must be zero/fail_closed")


_check_contract()
