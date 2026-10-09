"""The scientific actions a progressive Planner may select, as it reads them.

Owner
-----
This module owns the action rows the progressive Planner reads: every
available action of each candidate analysis family, with its runtime contract
and, when an outline-stage rule places the action, where an outline may
select it.  That place is stated by the owner of those rules
(:func:`.outline_action_rules.outline_action_position`), which also runs them
as the outline check, so the row an action is selected from and the check
that would refuse it say the same thing.
"""

from __future__ import annotations

from typing import Any, Sequence

from .outline_action_rules import outline_action_position
from .scientific_action_catalog import scientific_actions_for_analysis_type

__all__ = ["outline_action_catalog"]


def outline_action_catalog(
    analysis_types: Sequence[str],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Each available action id once, and one row per family that offers it."""

    action_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    for analysis_type in analysis_types:
        catalog = scientific_actions_for_analysis_type(analysis_type)
        for action in catalog.actions:
            if action.execution_mode == "not_available":
                continue
            if action.action_id not in action_ids:
                action_ids.append(action.action_id)
            contract = action.runtime_contract
            row: dict[str, Any] = {
                "analysis_type": analysis_type,
                "action_id": action.action_id,
                "name": action.name,
                "purpose": action.purpose,
                "notes": action.notes,
                "execution_mode": action.execution_mode,
                "produces": action.produces,
                "required_inputs": list(action.required_inputs),
                "runtime_contract": (
                    {
                        "outputs": [
                            {"product_id": product_id, "semantic_role": semantic_role}
                            for product_id, semantic_role in contract.outputs
                        ],
                        "required_product_inputs": list(
                            contract.required_product_inputs
                        ),
                        "article_roles": list(contract.article_roles),
                        "standard_executor": contract.standard_executor,
                        "execution_parameters": dict(contract.execution_parameters),
                    }
                    if contract is not None
                    else None
                ),
            }
            position = outline_action_position(analysis_type, action.action_id)
            if position:
                row["outline_position"] = position
            rows.append(row)
    return tuple(action_ids), rows
