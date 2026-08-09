"""Plan-time gate for Planner-declared raw inputs.

A step's typed ``kind:product`` inputs are validated before execution by
``_typed_plan_dag_findings``; its **raw column** inputs were not validated at
all.  They were first read deep inside ``_execute_one_step``, where an
unresolvable name raises ``ValueError`` -- and nothing wraps ``execute_step``,
so that exception leaves ``run_sequential``, ``run_execute_phase`` and
``pipeline.run`` and kills the whole run with no sealed artifacts and no
diagnosis.  A sweep of 1,114 historical plan steps found 8 that would hard
crash this way, every one of them a Planner declaring a column the sealed
context does not carry.

This gate asks the same question at plan time, where the existing preflight
already turns a finding into one focused replan directive.

It deliberately does **not** re-implement the resolvability test.  It calls the
real consumer and reports what the consumer refused, so the gate and the
executor cannot drift apart -- a gate that reasons about the rule instead of
invoking it is how one call chain ends up treating the same fact as harmless in
one place and fatal in another.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from ..research_context.typed import resolved_raw_input_contracts
from ..schema import ValidationFinding

_VALIDATOR = "plan_declared_raw_inputs"


def _declared_raw_names(inputs: Sequence[Any] | None) -> List[str]:
    """The raw column names a step declares, in declared order.

    Mirrors the consumer's own filter: typed ``kind:name`` products carry a
    colon and stay under the manifest's separate ``inputs`` authority.
    """

    names: List[str] = []
    for value in inputs or []:
        if not isinstance(value, str):
            continue
        name = value.strip()
        if name and ":" not in name and name not in names:
            names.append(name)
    return names


def _unresolvable_names(context: Any, names: Sequence[str]) -> List[str]:
    """Ask the consumer, one name at a time, which names it refuses."""

    refused: List[str] = []
    for name in names:
        try:
            resolved_raw_input_contracts(context, (name,))
        except ValueError:
            refused.append(name)
    return refused


def declared_raw_input_plan_findings(
    *,
    plan: Any,
    context: Any,
) -> List[ValidationFinding]:
    """Return one repairable finding per step whose raw inputs cannot resolve."""

    findings: List[ValidationFinding] = []
    for step in getattr(plan, "steps", None) or []:
        declared = getattr(step, "inputs", None)
        try:
            resolved_raw_input_contracts(context, declared or [])
        except ValueError as error:
            names = _declared_raw_names(declared)
            refused = _unresolvable_names(context, names)
            step_id = str(getattr(step, "step_id", "") or "unknown")
            # Cause first: only ``message`` reaches a prompt, and the prompt
            # projection clips it from the tail.
            named = ", ".join(repr(name) for name in refused) or "unknown"
            findings.append(
                ValidationFinding(
                    validator=_VALIDATOR,
                    severity="error",
                    message=(
                        f"Step {step_id} declares raw input(s) {named} that the "
                        f"sealed research context cannot resolve ({error}). "
                        "Declare only columns the context carries, or declare "
                        "the typed product whose producer creates them."
                    ),
                    detail={
                        "reason": "declared_raw_input_unresolvable",
                        "step_id": step_id,
                        "unresolvable_inputs": refused,
                        "declared_raw_input_count": len(names),
                    },
                )
            )
    return findings


__all__ = ["declared_raw_input_plan_findings"]
