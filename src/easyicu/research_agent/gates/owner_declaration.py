"""Plan-time gate: a deterministic owner exists and is one declaration away.

A step can declare a product the host already computes deterministically and
still be executed by the stochastic coder, because the owner declined over a
field the Planner simply never filled in.  Nothing said so.  ``owns_step``
returned ``bool``, the selector recorded "contract declined", and the run fell
through to code generation for a result it did not need to generate.

Measured over 553 recorded real steps: 26 of them declare the paper's primary
adjusted-association model and no adjustment set.  Every one went to the coder,
whose accumulated repair guidance records a script that dropped the whole
cohort by numeric-coercing ``sex``, object-dtype design matrices handed to
statsmodels, and a contract "satisfied" with a null estimate -- for a model the
host can fit exactly as declared, once it is declared.

Ambiguity at a declaration boundary must fail closed.  This gate asks the
question at **plan time**, where the existing preflight turns a finding into
one focused replan directive and the Planner can still act, rather than after
execution where the only remaining move is to repair generated code.

It deliberately does **not** re-implement any ownership predicate.  It calls
``select_standard_executor`` -- the code that actually decides -- and reports
what the owners said through its trace.  A gate that reasons about the rule
instead of invoking it is how one call chain ends up treating the same fact as
harmless in one place and fatal in another.

Two boundaries it keeps:

* Only ``incomplete_declaration`` verdicts are reported.  A wrong-shape decline
  means no owner exists for that step and the coder path is correct; reporting
  it would send the Planner to fix something that is not broken.
* A step whose selection *raises* is reported as unevaluated rather than
  passed.  A gate that answers in the permissive direction for a fact it lacks
  is worse than no gate: it reads as "checked, fine".
"""

from __future__ import annotations

import json
from typing import Any, List, Sequence

from ..execution.runners.selection import (
    StandardExecutorCandidate,
    select_standard_executor,
)
from ..schema import ValidationFinding

__all__ = [
    "owner_declaration_plan_findings",
    "owner_declaration_replan_directive",
]

_VALIDATOR = "plan_owner_declaration"


def _declaration_gaps(step: Any, plan: Any) -> tuple[StandardExecutorCandidate, ...]:
    """Ask the real decider, and return only the owners waiting on a field.

    ``plausibility_scope`` and ``resolved_bindings`` are run-time facts a plan
    does not carry.  Neither can turn a declaration gap into a claim: an owner
    computes its verdict from the step's declaration alone, and the receipt and
    binding gates apply *after* a contract matches.  Passing ``None`` therefore
    reads the gap exactly, rather than reading an optimistic bound of it.
    """

    trace: List[StandardExecutorCandidate] = []
    select_standard_executor(step, plan=plan, trace=trace)
    return tuple(candidate for candidate in trace if candidate.missing_declarations)


def owner_declaration_plan_findings(*, plan: Any) -> List[ValidationFinding]:
    """One repairable finding per step an existing owner could claim if declared."""

    findings: List[ValidationFinding] = []
    for step in getattr(plan, "steps", None) or []:
        step_id = str(getattr(step, "step_id", "") or "unknown")
        try:
            gaps = _declaration_gaps(step, plan)
        except Exception as error:  # noqa: BLE001 - report, never assume
            findings.append(
                ValidationFinding(
                    validator=_VALIDATOR,
                    severity="error",
                    message=(
                        f"Step {step_id} could not be checked for deterministic "
                        f"ownership ({type(error).__name__}: {error}). The host "
                        "cannot say whether a declared field is missing, so this "
                        "step is unevaluated rather than accepted."
                    ),
                    detail={
                        "reason": "owner_declaration_gate_unevaluated",
                        "step_id": step_id,
                        "error_type": type(error).__name__,
                    },
                )
            )
            continue
        for gap in gaps:
            missing = ", ".join(repr(name) for name in gap.missing_declarations)
            # Cause first: only ``message`` reaches a prompt, and the prompt
            # projection clips it from the tail.
            findings.append(
                ValidationFinding(
                    validator=_VALIDATOR,
                    severity="error",
                    message=(
                        f"Step {step_id} does not declare {missing}, so the host's "
                        f"deterministic {gap.analysis_kind} owner cannot claim it "
                        f"({gap.decline_reason}). Declare the missing field(s) on "
                        "the step that already exists. Do not change the exposure, "
                        "outcome, cohort, estimand, or method to satisfy this, and "
                        "do not split or add a step."
                    ),
                    detail={
                        "reason": "owner_declaration_incomplete",
                        "step_id": step_id,
                        "analysis_kind": gap.analysis_kind,
                        "missing_declarations": list(gap.missing_declarations),
                    },
                )
            )
    return findings


def owner_declaration_replan_directive(
    findings: Sequence[ValidationFinding],
) -> str | None:
    """The focused replan instruction for this gate's findings, or None.

    It lives beside the gate rather than inline at the call site: the wording
    is part of what this finding *means*, and its four sibling directives sit
    inside an 8,000-line function where the next reader cannot tell which gate
    each belongs to.

    Every prohibition here is load-bearing.  The gap is a missing *declaration*
    of something the plan already chose, so a replanner that "fixes" it by
    picking a covariate set, splitting the step, or deleting it has changed the
    science to satisfy a bookkeeping check -- which is worse than the fall-back
    this gate exists to prevent.
    """

    if not findings:
        return None
    return (
        "Complete the declaration on steps the host can already compute "
        "deterministically, without changing any scientific choice. Each "
        "finding names a step and the exact field(s) it left undeclared; add "
        "those to the step that already exists. Do not choose an exposure, "
        "outcome, cohort, covariate, estimand, or method to satisfy this, do "
        "not split or merge steps, and do not delete a step you cannot "
        "complete. Contract findings: "
        + json.dumps(
            [
                {"message": finding.message, "detail": finding.detail}
                for finding in findings
            ],
            ensure_ascii=False,
            default=str,
        )
    )
