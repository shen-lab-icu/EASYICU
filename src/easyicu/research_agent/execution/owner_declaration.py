"""Execution-owned plan-time preflight for incomplete owner declarations.

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
``resolve_standard_executor`` -- the code that actually decides -- and reports
what the owners said without generating code.  A gate that reasons about the rule
instead of invoking it is how one call chain ends up treating the same fact as
harmless in one place and fatal in another.

Two boundaries it keeps:

* Only ``incomplete_declaration`` verdicts are reported.  A wrong-shape decline
  means no owner exists for that step and the coder path is correct; reporting
  it would send the Planner to fix something that is not broken.
* A step whose selection *raises* is reported as unevaluated rather than
  passed.  A gate that answers in the permissive direction for a fact it lacks
  is worse than no gate: it reads as "checked, fine".
* What the replan is forbidden to change is **computed from the findings**, not
  written down.  The literal list here used to contain ``covariate`` while the
  only gap this gate has ever reported is ``model_requirements[0].covariates``,
  so the finding and the directive travelled to the Planner together saying
  "declare the adjustment set" and "do not choose a covariate".  A demand that
  forbids itself is not strict, it is unsatisfiable, and doing nothing is a
  defensible reading of it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from .runners.selection import (
    StandardExecutorCandidate,
    resolve_standard_executor,
)
from .step_executor_registry import StepExecutorDecision
from ..schema import ValidationFinding
from ..contracts.claim_ceiling import first_result

__all__ = [
    "execution_declaration_refusal",
    "owner_declaration_plan_findings",
    "owner_declaration_replan_directive",
    "require_host_rendered_plan",
]

_VALIDATOR = "plan_owner_declaration"


class HostPlanStepNeedsSandbox(ValueError):
    """The desktop plan contains a step without a typed executor owner."""

    reason_code = "research_pipeline_plan_step_needs_sandbox"

    def __init__(self, step_id: str) -> None:
        super().__init__(
            f"{self.reason_code}: This step needs code the desktop app cannot "
            "run safely. Remove or change this step before approving the plan."
        )
        self.easyicu_safe_diagnostic = {
            "owner": "easyicu.execution.host_plan_v1",
            "reason_code": self.reason_code,
        }
        if isinstance(step_id, str) and re.fullmatch(r"[a-z0-9][a-z0-9_]{0,79}", step_id):
            self.easyicu_safe_diagnostic["step_id"] = step_id


def host_plan_review_root(plan_result: Any) -> Path | None:
    """Resolve the materialization root for a new or evidence-backed plan."""
    evidence = getattr(plan_result, "evidence", None)
    run_root = getattr(evidence, "root", None)
    plan_path = getattr(plan_result, "plan_path", None)
    return (Path(run_root) if run_root is not None
            else Path(plan_path).parent if plan_path is not None else None)


def require_host_plan_before_review(
    pipeline: Any, plan: Any, *, run_dir: Path | None = None
) -> None:
    """Apply the desktop ownership boundary before creating any approval."""
    if first_result(plan):
        require_first_result_steps(plan)
    if first_result(plan) or getattr(pipeline, "_runner_kind", None) == "host":
        require_host_rendered_plan(
            plan=plan,
            run_dir=run_dir,
            trajectory_scientific_runtime_authority=pipeline._scientific_runtime_authorities.trajectory,
            current_case_scientific_runtime_authority=pipeline._scientific_runtime_authorities.current_case,
            scientific_runtime_projection_sha256=pipeline._scientific_runtime_projection_sha256,
        )


def _verified_host_cohort_step(plan: Any, run_dir: Path | None) -> str | None:
    """Recognize only the typed cohort Execute can adopt without any script."""

    if run_dir is None:
        return None
    from ..authority.run_input import _declares_host_cohort_products
    from ..cohort.schema import (
        _load_locked_cohort_definition,
        cohort_definition_sha,
        load_materialized_analysis_cohort_result,
    )

    # Use the same unique product owner and verifier as cohort_adoption. A
    # second cohort declaration cannot be adopted, even if only one is typed.
    steps = [step for step in plan.steps if _declares_host_cohort_products(step)]
    if len(steps) != 1:
        return None
    step = steps[0]
    if (
        step.method != "cohort_definition_and_attrition"
        or step.planned_analysis_role != "auxiliary"
        or step.cohort_definition_spec is None
        or step.model_requirements
    ):
        return None
    try:
        result = load_materialized_analysis_cohort_result(run_dir=run_dir, plan=plan)
        if (
            result is None
            or result.get("authority_ref") is None
            or cohort_definition_sha(_load_locked_cohort_definition(run_dir))
            != result.get("cohort_definition_sha256")
        ):
            return None
    except Exception:  # malformed authority is never an ownership grant
        return None
    return str(step.step_id)


def require_host_rendered_plan(
    *, plan: Any, run_dir: Path | None = None, **runtime_authority: Any
) -> None:
    """Refuse before approval unless every step has an actual typed owner.

    In particular, naming ordinal_trend without its ordered/stratified spec
    does not grant host execution. Resolve using the same authority as Execute;
    this query never renders or accepts a model-supplied script.
    """

    # The plan-phase host materializer is script-free; Execute adopts these
    # exact verified products before scheduling steps. Product names alone are
    # not proof, and every other step must still have its typed executor.
    if first_result(plan):
        require_first_result_steps(plan)
    host_cohort_step = _verified_host_cohort_step(plan, run_dir)
    for step in plan.steps:
        if step.step_id == host_cohort_step:
            continue
        try:
            decision = resolve_standard_executor(step, plan=plan, **runtime_authority)
            owned = decision.claimed_by is not None or _declared_host_figure(
                step, plan=plan, host_cohort_step=host_cohort_step, **runtime_authority
            )
        except Exception as exc:
            raise HostPlanStepNeedsSandbox(step.step_id) from exc
        if not owned:
            raise HostPlanStepNeedsSandbox(step.step_id)


def require_first_result_steps(plan: Any) -> None:
    """A report/coder step cannot sneak into an analysis-only approval."""
    for step in plan.steps:
        if any(str(output).startswith(("report:", "manuscript:", "latex:")) for output in step.expected_outputs):
            raise ValueError(f"first_result forbids article-only step {step.step_id!r}; start an article run")
    if not any(step.measurement_audit_spec is not None for step in plan.steps):
        raise ValueError("first_result requires measurement_audit before approval")


def _declared_host_figure(
    step: Any, *, plan: Any, host_cohort_step: str | None, **runtime_authority: Any
) -> bool:
    """Check future table renderers without inventing runtime table evidence.

    The executor owns each shape predicate. This pre-review query additionally
    requires unique host-owned producers. Execute must still resolve the real
    table bindings and validate their columns and digests; no selection or
    executable code is returned here.
    """

    if any(panel.presentation is not None for panel in step.figure_panels):
        return False
    from .runners.cohort_flow_figure_executor import (
        cohort_flow_figure_executor_declares_step,
    )
    from .runners.missingness_measurement_figure_executor import (
        missingness_measurement_figure_executor_declares_step,
    )

    producers = []
    for key in step.inputs:
        matches = [parent for parent in plan.steps if key in parent.expected_outputs]
        if len(matches) != 1 or plan.steps.index(matches[0]) >= plan.steps.index(step):
            return False
        producers.append(matches[0])
    if cohort_flow_figure_executor_declares_step(step):
        return bool(
            host_cohort_step is not None
            and len(producers) == 1
            and producers[0].step_id == host_cohort_step
        )
    if not missingness_measurement_figure_executor_declares_step(step, plan=plan):
        return False
    # Only the typed audit producer guarantees these two schemas. Legacy or
    # model-generated tables need their actual bindings before being claimable.
    return bool(producers) and all(
        parent.measurement_audit_spec is not None
        and resolve_standard_executor(parent, plan=plan, **runtime_authority).claimed_by
        == "declared_missingness_audit_products"
        for parent in producers
    )


def execution_declaration_refusal(
    decision: StepExecutorDecision,
) -> Tuple[StandardExecutorCandidate, ...]:
    """The declaration gaps that must refuse a step at execution time.

    The plan-time gate above asks for the field first and spends a forced
    replan on the answer.  A step that reaches execution still under-declared
    means that replan did not supply it, and handing it to the Coder anyway is
    the fail-open this whole module exists to close.

    Two clauses, and both are what the answer *means* rather than checks bolted
    onto it:

    * ``claimed_by`` is None -- nobody is going to compute the step.  A gap
      reported by one owner while a different owner claimed it is moot, and
      refusing there would be a wrong block of work that was about to be done
      correctly.  Measured over the recorded plans (694 distinct step shapes):
      8 reach "no owner + a gap", **0** reach "an owner claimed + a gap".  So
      this clause has never fired on real data; it is kept because the failure
      it prevents is severe and nothing in the selector forbids that answer.
    * only ``incomplete_declaration`` verdicts count.  A wrong-shape decline
      means no owner exists and the Coder path is correct.

    It lives here rather than inline at the call site so the plan-time gate and
    the execution-time refusal share one definition of "under-declared".  Two
    copies of that rule drifting apart is the defect shape this package keeps
    paying for.
    """

    if decision.claimed_by is not None:
        return ()
    return tuple(candidate for candidate in decision.candidates if candidate.missing_declarations)

#: The scientific choices a replan must not *make* in order to satisfy this
#: gate.  The gap is a missing declaration of something the plan already chose,
#: so a replanner that closes it by picking a different exposure or a new cohort
#: has changed the science to satisfy a bookkeeping check -- worse than the
#: fall-through this gate exists to prevent.
#:
#: It is a starting set, not the emitted list: see :func:`_prohibited_choices`.
_SCIENTIFIC_CHOICES: Tuple[str, ...] = (
    "exposure",
    "outcome",
    "cohort",
    "covariate",
    "estimand",
    "method",
)


def _declared_choice(name: str) -> str:
    """``model_requirements[0].covariates`` -> ``covariate``.

    The containing path and the index say *where* the field lives; only the
    leaf says *which* choice it declares.  The trailing ``s`` comes off so a
    field named for a set (``covariates``) matches the singular choice it is
    the declaration of.

    Matching is exact on that normalised leaf, never a substring: an owner
    reporting ``outcome_levels`` has not been handed permission to choose the
    ``outcome``, and a substring test would silently grant it.
    """

    leaf = str(name or "").split(".")[-1].split("[")[0].strip().casefold()
    return leaf[:-1] if leaf.endswith("s") else leaf


def _prohibited_choices(missing: Iterable[str]) -> Tuple[str, ...]:
    """The choices to forbid, minus the ones this gate is asking to be declared.

    A fixed prohibition list is the defect this function exists to remove.  It
    contained ``covariate`` while the only field the gate ever reports missing
    is ``model_requirements[0].covariates`` -- measured 2026-07-30, 74 of the 81
    recorded declarations of the primary product decline for exactly that field
    -- so one replan message demanded the adjustment set and forbade choosing
    one in the same breath.  A directive that contradicts itself is not a strict
    directive but an unsatisfiable one, and the safest reading of it is to do
    nothing.

    Deriving the prohibition from the findings makes that shape impossible for
    every future owner rather than for this one field, which is the point:
    editing the literal would fix today's contradiction and leave the next
    owner's missing field to collide with it again.
    """

    demanded = {_declared_choice(name) for name in missing}
    return tuple(choice for choice in _SCIENTIFIC_CHOICES if choice not in demanded)


def _english_list(items: Sequence[str]) -> str:
    """``(a, b, c)`` -> ``"a, b, or c"``; the empty case names itself."""

    values = [str(item) for item in items if str(item).strip()]
    if not values:
        # Reached only if a finding demands every choice in the starting set,
        # which would mean the plan declared nothing at all.  Saying so beats
        # emitting "Do not change the  to satisfy this".
        return "any scientific choice already declared elsewhere in the plan"
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f", or {values[-1]}"


def _declaration_gaps(step: Any, plan: Any) -> tuple[StandardExecutorCandidate, ...]:
    """Ask the real decider, and return only the owners waiting on a field.

    ``plausibility_scope`` and ``resolved_bindings`` are run-time facts a plan
    does not carry.  Neither can turn a declaration gap into a claim: an owner
    computes its verdict from the step's declaration alone, and the receipt and
    binding gates apply *after* a contract matches.  Passing ``None`` therefore
    reads the gap exactly, rather than reading an optimistic bound of it.
    """

    decision = resolve_standard_executor(step, plan=plan)
    return tuple(candidate for candidate in decision.candidates if candidate.missing_declarations)


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
            # One owner for the prohibition, shared with the directive: when
            # the two were written separately they drifted, and the pair that
            # reached the Planner together said "declare the covariates" and
            # "do not choose a covariate".
            prohibited = _english_list(_prohibited_choices(gap.missing_declarations))
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
                        f"the step that already exists. Do not change the "
                        f"{prohibited} to satisfy this, and do not split or add "
                        "a step."
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

    Every prohibition here is load-bearing, and for that reason it is computed
    rather than written down.  The gap is a missing *declaration* of something
    the plan already chose, so a replanner that "fixes" it by splitting the
    step or deleting it has changed the science to satisfy a bookkeeping check.
    But the fields the findings name are the ones the Planner is being told to
    fill in, and a literal list that happened to contain one of them -- as this
    one contained ``covariate`` while the only reported gap was
    ``model_requirements[0].covariates`` -- demands and forbids the same act.
    ``_prohibited_choices`` subtracts what is demanded, so no future owner's
    missing field can collide with the prohibition either.
    """

    if not findings:
        return None
    demanded: List[str] = []
    for finding in findings:
        detail = getattr(finding, "detail", None)
        if isinstance(detail, dict):
            demanded.extend(
                str(name) for name in (detail.get("missing_declarations") or [])
            )
    prohibited = _english_list(_prohibited_choices(demanded))
    return (
        "Complete the declaration on steps the host can already compute "
        "deterministically, without changing any scientific choice. Each "
        "finding names a step and the exact field(s) it left undeclared; add "
        f"those to the step that already exists. Do not choose a different "
        f"{prohibited} to satisfy this, do "
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
