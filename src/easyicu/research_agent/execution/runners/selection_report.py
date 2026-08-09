"""Render why a step did or did not reach a deterministic owner.

A step no deterministic owner claims falls to the stochastic Coder silently.
This module turns that silence into a typed record on the step.

It is a *renderer*, not a second selector.  The owner verdicts come from the
trace :func:`select_standard_executor` emits while deciding, because a
diagnostic that re-runs the ownership predicates itself cannot see the gates the
selector applies *after* a contract matches -- it would eventually report an
owner the selector had declined, which is worse than reporting nothing.  The
only predicates evaluated here are the missingness family's closed sub-contracts,
which classify a declared contract's shape and claim nothing.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ...schema import AnalysisPlan, AnalysisStep
from .selection import StandardExecutorCandidate

__all__ = [
    "STANDARD_EXECUTOR_CANDIDATE_SCHEMA_VERSION",
    "standard_executor_candidate_report",
]

STANDARD_EXECUTOR_CANDIDATE_SCHEMA_VERSION = "easyicu.standard_executor_candidates/2"


def _contract_details() -> tuple[tuple[str, Callable[[AnalysisStep], bool]], ...]:
    """Bind the missingness family's closed sub-contract classifiers.

    Ownership in that family is decided by several closed contracts over one
    input scope, so a step that satisfies the scope but no contract is the
    enrichment case: the science grew a product and lost its deterministic
    owner.  These entries locate which half said no.  None of them is an
    ownership claim.
    """

    # Imported here so this observability surface never becomes a second import
    # path that the selector's own module graph has to route around.
    from .deterministic_missingness import (
        is_compact_missingness_measurement_contract,
        is_measurement_bias_audit_contract,
        is_missingness_complete_case_contract,
        is_missingness_measurement_availability_contract,
        missingness_audit_input_scope_supported,
    )

    return (
        ("missingness_audit:input_scope", missingness_audit_input_scope_supported),
        (
            "missingness_audit:availability_contract",
            lambda step: is_missingness_measurement_availability_contract(
                step.method, step.expected_outputs
            ),
        ),
        (
            "missingness_audit:complete_case_contract",
            lambda step: is_missingness_complete_case_contract(
                step.method, step.expected_outputs
            ),
        ),
        (
            "missingness_audit:compact_contract",
            lambda step: is_compact_missingness_measurement_contract(
                step.method, step.expected_outputs
            ),
        ),
        (
            "missingness_audit:measurement_bias_contract",
            lambda step: is_measurement_bias_audit_contract(
                step.method, step.expected_outputs
            ),
        ),
    )


def standard_executor_candidate_report(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    trace: Sequence[StandardExecutorCandidate] | None = None,
    resolved_bindings: Mapping[str, Any] | None = None,
    claimed_by: str | None = None,
) -> dict[str, Any]:
    """Return the typed record of which owners considered this step.

    ``trace`` is what :func:`select_standard_executor` recorded for this exact
    call; ``claimed_by`` is the ``analysis_kind`` it chose, or ``None`` when the
    step fell through to the Coder.  The coordinates the contracts key on --
    declared method and typed products -- are recorded alongside, because those
    are what a reader needs in order to tell an unsupported analysis apart from
    a supported one wearing a new name.

    Passing no ``trace`` records the owner list as unavailable rather than
    guessing it: an absent diagnostic is recoverable, a confident wrong one is
    not.
    """

    candidates: list[dict[str, Any]] = []
    for candidate in trace or ():
        candidates.append(
            {
                "kind": "owner",
                "analysis_kind": candidate.analysis_kind,
                "contract_matches": bool(candidate.contract_matches),
                "outcome": candidate.outcome,
            }
        )
    for analysis_kind, predicate in _contract_details():
        entry: dict[str, Any] = {"kind": "detail", "analysis_kind": analysis_kind}
        try:
            entry["matches"] = bool(predicate(step))
        except Exception as exc:  # observability must never fail a step
            entry["matches"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
        candidates.append(entry)
    return {
        "schema_version": STANDARD_EXECUTOR_CANDIDATE_SCHEMA_VERSION,
        "step_id": str(step.step_id),
        "claimed_by": claimed_by,
        "trace_available": trace is not None,
        # Owners whose declared contract matched.  An owner can appear here and
        # still not be the one that ran: read ``outcome`` for that.
        "owning_candidates": [
            entry["analysis_kind"]
            for entry in candidates
            if entry["kind"] == "owner" and entry["contract_matches"]
        ],
        "declined_after_match": [
            entry["analysis_kind"]
            for entry in candidates
            if entry["kind"] == "owner"
            and entry["contract_matches"]
            and entry["outcome"] != "selected"
        ],
        "declared_method": str(step.method or ""),
        "declared_outputs": [str(value) for value in (step.expected_outputs or ())],
        "declared_typed_inputs": [
            str(value) for value in (step.inputs or ()) if ":" in str(value)
        ],
        "declared_raw_input_count": sum(
            1 for value in (step.inputs or ()) if ":" not in str(value)
        ),
        "candidates": candidates,
    }
