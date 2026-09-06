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

from ..concept_availability import ConceptSourceUnavailableError
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


def _input_errors(context: Any, names: Sequence[str]) -> dict[str, ValueError]:
    """Keep the consumer's source-prohibition cause distinct from absent names."""

    refused: dict[str, ValueError] = {}
    for name in names:
        try:
            resolved_raw_input_contracts(context, (name,))
        except ValueError as error:
            refused[name] = error
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
            errors = _input_errors(context, names)
            unavailable = {
                name: cause for name, cause in errors.items()
                if isinstance(cause, ConceptSourceUnavailableError)
            }
            refused = [name for name in errors if name not in unavailable]
            step_id = str(getattr(step, "step_id", "") or "unknown")
            if unavailable:
                findings.append(ValidationFinding(
                    validator=_VALIDATOR, severity="error",
                    message=f"Step {step_id}: " + " ".join(str(cause) for cause in unavailable.values()),
                    detail={
                        "reason": "declared_raw_input_structurally_unavailable",
                        "step_id": step_id,
                        "unavailable_inputs": list(unavailable),
                        "source_concepts": sorted({
                            receipt.concept_id for cause in unavailable.values()
                            for receipt in cause.receipts
                        }),
                    },
                ))
            if not refused and unavailable:
                continue
            # Cause first: only ``message`` reaches a prompt, and the prompt
            # projection clips it from the tail.
            named = ", ".join(repr(name) for name in refused) or "unknown"
            unresolved_cause = errors[refused[0]] if refused else error
            findings.append(
                ValidationFinding(
                    validator=_VALIDATOR,
                    severity="error",
                    message=(
                        f"Step {step_id} declares raw input(s) {named} that the "
                        f"sealed research context cannot resolve ({unresolved_cause}). "
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
