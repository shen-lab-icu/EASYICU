"""Plan-time gate: a product promised as a statistic and as something else.

Measured 2026-07-30 over every recorded plan (6,571 steps carrying typed
``expected_outputs``): 369 steps promise one bare product name under two kinds.
**359 of them are the same name** -- ``robustness_summary``, promised as both
``table:robustness_summary`` and ``statistic:robustness_summary`` by
``robustness_sensitivity`` steps. The remaining 10 pair ``artifact:`` with
``table:`` and are untouched by this gate.

Why that one shape cannot be executed by anyone
-----------------------------------------------
A ``statistic:`` product is defined in this system as a JSON sidecar carrying
that product's identity and **at least one finite numeric value** -- the Coder
prompt states it as a hard contract. A "robustness summary" is a grid of
specifications, not a number. So the two promises are two different artifacts
that happen to share a name.

Every typed declaration the host owns identifies its products by the **bare**
name, without the kind: ``RobustnessReplaySpec.products[].product_id``,
``MeasurementAuditProduct``, and the shared reader
``schema.spec_backs_every_declared_product``. When one bare name is promised
twice, no declaration can say which promise it backs, and the schema forbids
the only declaration that would try (``product_id`` values must be unique).
Constructed and verified against the real recorded step
``07_missingness_robustness_replay``: the Planner's own two-product spec is not
emittable, a hand-completed five-product spec is still not emittable, and the
six-product spec that would cover both kinds is rejected by the schema.

So the deterministic robustness owner -- which can produce five of that step's
six promised products -- claims none of them, and the whole step goes to the
Coder.

Why this is a gate finding and not a schema validator
-----------------------------------------------------
``schema.py`` carries the scar: these coverage rules first shipped as
``AnalysisStep`` validators, and a real fresh run then wrote a plan the host
could not re-parse. The plan was already sealed as evidence, the authority
resolver swallowed the parse error as an unreadable record, and the run died
before its first step reporting "current analysis plan is not bound to
immutable EvidenceStore authority" -- a message naming nothing about the cause.
A malformed promise must leave the plan readable and become a repairable
finding the Planner can act on.

Why it does not live in ``execution.owner_declaration``
------------------------------------------------
That gate reports *missing fields* -- its finding says "does not declare X"
and its directive says "declare the missing field(s)". Here nothing is
missing; one promise too many is present, and the repair is to remove a kind.
Routing this through ``incomplete_declaration`` would send the Planner a
sentence that is false about the step and wrong about the fix, which is the
exact failure recorded in that module's own docstring.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from ..planning.robustness_contract import ROBUSTNESS_REPLAY_OUTPUT_PRODUCT_KINDS
from ..schema import ValidationFinding

__all__ = [
    "product_promise_plan_findings",
    "product_promise_replan_directive",
]

_VALIDATOR = "plan_product_promise"
_STATISTIC = "statistic"


def _typed_promises(step: Any) -> Dict[str, List[str]]:
    """Bare product name -> the kinds it is promised under, in declared order."""

    by_name: Dict[str, List[str]] = defaultdict(list)
    for value in getattr(step, "expected_outputs", None) or []:
        token = str(value or "").strip()
        kind, separator, name = token.partition(":")
        if not separator or not kind.strip() or not name.strip():
            continue
        kinds = by_name[name.strip()]
        if kind.strip() not in kinds:
            kinds.append(kind.strip())
    return by_name


def _collisions(step: Any) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Products promised as a statistic and under at least one other kind."""

    found: List[Tuple[str, Tuple[str, ...]]] = []
    for name, kinds in sorted(_typed_promises(step).items()):
        if _STATISTIC in kinds and len(kinds) > 1:
            found.append((name, tuple(kinds)))
    return tuple(found)


def _required_robustness_kind(step: Any, product: str) -> str | None:
    """Resolve a promised label through the replay owner's public contract."""

    spec = getattr(step, "robustness_replay_spec", None)
    resolve = getattr(spec, "output_for", None)
    if not callable(resolve):
        return None
    output = resolve(product)
    if output is None:
        return None
    return ROBUSTNESS_REPLAY_OUTPUT_PRODUCT_KINDS.get(str(output))


def product_promise_plan_findings(*, plan: Any) -> List[ValidationFinding]:
    """One repairable finding per product promised as a statistic and as more.

    Deliberately narrow. Only a collision **involving ``statistic:``** is
    reported, because that is the pair the host can prove is two different
    artifacts: one of them must be a single finite number. The 10 recorded
    ``artifact:``/``table:`` collisions may well be one dataset offered in two
    forms, and blocking them would cost a healthy step a replan for a promise
    nobody has shown to be wrong.
    """

    findings: List[ValidationFinding] = []
    for step in getattr(plan, "steps", None) or []:
        step_id = str(getattr(step, "step_id", "") or "unknown")
        typed = _typed_promises(step)
        collisions = {name: kinds for name, kinds in _collisions(step)}
        for name, kinds_list in sorted(typed.items()):
            kinds = tuple(kinds_list)
            required_kind = _required_robustness_kind(step, name)
            collision = collisions.get(name)
            if required_kind is None and collision is None:
                continue
            if required_kind is not None and kinds == (required_kind,):
                continue
            if required_kind is not None:
                reason = (
                    "product_promised_as_statistic_and_more"
                    if collision is not None
                    else "product_kind_conflicts_with_owner"
                )
                findings.append(
                    ValidationFinding(
                        validator=_VALIDATOR,
                        severity="error",
                        message=(
                            f"Step {step_id} maps {name!r} through its typed "
                            "robustness replay declaration to a deterministic "
                            f"output whose only valid product kind is "
                            f"'{required_kind}:{name}', but the step promises "
                            + ", ".join(f"'{kind}:{name}'" for kind in kinds)
                            + f". Keep exactly '{required_kind}:{name}' and "
                            "delete every incompatible promise for that bare "
                            "product. Do not rename the product, remap the "
                            "replay output, add or split a step, or change any "
                            "scientific choice."
                        ),
                        detail={
                            "reason": reason,
                            "step_id": step_id,
                            "product": name,
                            "kinds": list(kinds),
                            "required_kind": required_kind,
                            "required_promise": f"{required_kind}:{name}",
                        },
                    )
                )
                continue
            # A collision outside a typed owner is still provably wrong only
            # when one of its promises is a scalar statistic.
            assert collision is not None
            others = [kind for kind in kinds if kind != _STATISTIC]
            findings.append(
                ValidationFinding(
                    validator=_VALIDATOR,
                    severity="error",
                    message=(
                        f"Step {step_id} promises {name!r} both as "
                        f"'statistic:{name}' and as "
                        + ", ".join(f"'{kind}:{name}'" for kind in others)
                        + ". A 'statistic:' product is one finite number in a "
                        "JSON sidecar, so these are two different artifacts "
                        "sharing one name, and every typed declaration in this "
                        "system names products without their kind -- no "
                        "declaration can say which of the two it backs, and the "
                        "host's deterministic owner therefore claims neither. "
                        f"Keep {name!r} under exactly one kind and delete the "
                        "other promise. Do not rename the product, do not add "
                        "or split a step, and do not change any scientific "
                        "choice."
                    ),
                    detail={
                        "reason": "product_promised_as_statistic_and_more",
                        "step_id": step_id,
                        "product": name,
                        "kinds": list(kinds),
                    },
                )
            )
    return findings


def product_promise_replan_directive(
    findings: Sequence[ValidationFinding],
) -> str | None:
    """The focused replan instruction for this gate's findings, or None.

    It lives beside the gate because the wording is part of what the finding
    means; its siblings sit inline in an 8,000-line function where the next
    reader cannot tell which gate a directive belongs to.
    """

    relevant = [
        finding
        for finding in findings
        if finding.validator == _VALIDATOR
        and (finding.detail or {}).get("reason")
        in {
            "product_promised_as_statistic_and_more",
            "product_kind_conflicts_with_owner",
        }
    ]
    if not relevant:
        return None
    return (
        "Repair the plan's product promises without changing any scientific "
        "choice. A product promised as 'statistic:<name>' must not also be "
        "promised under another kind: a statistic is a single finite number, "
        "the other artifact is not, and the host's typed declarations name "
        "products without their kind, so it cannot tell the two apart. For each "
        "finding below, use `required_promise` when it is present; that value "
        "comes from the deterministic owner's typed contract, not a choice to "
        "infer. Otherwise keep the product under the one kind that matches what "
        "the step actually produces and delete the other promise; when several are incompatible, "
        "delete every incompatible promise for the same bare product. Do not rename "
        "the product, do not add, split, or remove a step, and do not change "
        "the exposure, outcome, cohort, estimator, or analysis method. Contract "
        "findings: "
        + json.dumps(
            [
                {"message": finding.message, "detail": finding.detail}
                for finding in relevant
            ],
            ensure_ascii=False,
            default=str,
        )
    )
