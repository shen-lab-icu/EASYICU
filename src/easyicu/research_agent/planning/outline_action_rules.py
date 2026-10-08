"""Which composed outline steps the host can execute, checked at the outline.

Owner
-----
This module owns the outline-stage rule that a host action replaying another
action's sealed result follows a step that selects that producer, and the one
entry point through which the Planner's outline check runs every outline-stage
action rule (the phenotype comparison rules stay with
:mod:`.phenotype_outline_rules`).

A replaying action reads its producer's own output under the same host
executor: the cross-sectional cluster-number selection replays the cluster
solution's sealed standardized matrix, and the prediction validation reuses
the fitted model's scores.  A product of the same name from another step --
a model-coded trajectory fit, say -- is not that output, and a step can only
name its action's fixed products, so the outline is the last place the
Planner can still choose differently.  The pairs are read from the actions'
runtime contracts, never listed here: an action whose required product
inputs another action of the same executor produces replays that action.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .phenotype_outline_rules import validate_outline_phenotype_comparison
from .progressive_contract import ProgressivePlanCompileError, ProgressivePlanOutline
from .scientific_action_catalog import (
    ScientificAction,
    scientific_actions_for_analysis_type,
)

__all__ = [
    "replay_producer_rule_text",
    "replayed_producer_action_ids",
    "validate_outline_action_rules",
    "validate_outline_replay_producers",
]


def replayed_producer_action_ids(analysis_type: str, action_id: str) -> tuple[str, ...]:
    """The actions whose sealed result ``action_id`` replays; none for any other action.

    The runtime contract must require product inputs that another action run
    by the same host executor produces.  An action that also accepts another
    input set reads other owners' products, so it replays nothing.
    """

    return _replayed_producers(_actions(analysis_type), action_id)


def _actions(analysis_type: str) -> tuple[ScientificAction, ...]:
    try:
        return tuple(scientific_actions_for_analysis_type(analysis_type).actions)
    except ValueError:
        return ()


def _replayed_producers(
    actions: Sequence[ScientificAction], action_id: str
) -> tuple[str, ...]:
    contract = next(
        (
            action.runtime_contract
            for action in actions
            if action.action_id == action_id
        ),
        None,
    )
    if (
        contract is None
        or not contract.required_product_inputs
        or contract.alternative_product_inputs
        or not contract.standard_executor
    ):
        return ()
    required = set(contract.required_product_inputs)
    return tuple(
        action.action_id
        for action in actions
        if action.action_id != action_id
        and action.runtime_contract is not None
        and action.runtime_contract.standard_executor == contract.standard_executor
        and required & {product for product, _ in action.runtime_contract.outputs}
    )


def validate_outline_replay_producers(outline: ProgressivePlanOutline) -> None:
    """Refuse a replaying step that no earlier step selecting its producer precedes."""

    actions = _actions(outline.analysis_type)
    selected: set[str] = set()
    for index, step in enumerate(outline.steps):
        action_id = str(step.scientific_action_id or "")
        if not action_id:
            continue
        producers = _replayed_producers(actions, action_id)
        if producers and selected.isdisjoint(producers):
            named = " or ".join(producers)
            raise ProgressivePlanCompileError(
                "progressive_outline_replay_producer_absent",
                f"Step {step.step_id!r} selects {action_id}, which replays the sealed "
                f"result of {named}: the host runs it on that action's own output, so "
                "a product of the same name from another step is not its input. No "
                f"earlier step of this outline selects {named}. Select it in an "
                "earlier step, or remove this step: any other primary selects and "
                "checks its own result in its own step or with the action "
                "registered for that primary.",
                step_id=step.step_id,
                step_index=index,
                path="steps",
                findings=(
                    {
                        "step_id": step.step_id,
                        "action_id": action_id,
                        "producer_action_ids": list(producers),
                    },
                ),
            )
        selected.add(action_id)


def validate_outline_action_rules(
    outline: ProgressivePlanOutline, *, requested_outcomes: Iterable[str]
) -> None:
    """Every outline-stage action rule, checked before any step call is spent."""

    validate_outline_phenotype_comparison(
        outline, requested_outcomes=requested_outcomes
    )
    validate_outline_replay_producers(outline)


def replay_producer_rule_text(analysis_types: Sequence[str]) -> str:
    """The rule as the outline prompt states it; empty when no listed type has a pair."""

    pairs: dict[tuple[str, ...], list[str]] = {}
    for analysis_type in dict.fromkeys(analysis_types):
        actions = _actions(analysis_type)
        for action in actions:
            if action.execution_mode == "not_available":
                continue
            producers = _replayed_producers(actions, action.action_id)
            if producers and action.action_id not in pairs.setdefault(producers, []):
                pairs[producers].append(action.action_id)
    if not pairs:
        return ""
    clauses = "; ".join(
        f"{', '.join(replays)} after {' or '.join(producers)}"
        for producers, replays in pairs.items()
    )
    return (
        " A host action that replays another action's sealed result runs on that "
        "action's own output, so select it only after a step that selects its "
        f"producer: {clauses}. Any other primary selects and checks its own result "
        "in its own step or with the action registered for that primary."
    )
