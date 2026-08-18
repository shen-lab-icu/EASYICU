"""Host-rendered output-shape contracts for generated analysis code."""

from __future__ import annotations

import json
from typing import Sequence

from ..contracts.association_execution import association_binary_sensitivity_contract
from ..contracts.result_envelope import STATISTIC_PAYLOAD_KEY_ALIASES
from ..schema import AnalysisStep


def statistic_payload_shape_directive(products: Sequence[str]) -> str:
    """Publish the exact object shape read for declared statistic products."""

    aliases = "; ".join(
        f"{field} from {'/'.join(keys)}"
        for field, keys in STATISTIC_PAYLOAD_KEY_ALIASES.items()
    )
    return (
        "- Write each declared `statistic:<name>` product as a single JSON "
        "OBJECT, never a list and never a bare number: "
        '`{"name": "<name>", "value": <number>}`. A one-element list of that '
        "same object is refused as invalid_statistic_shape and kills the step "
        "after every other output is already correct. An included `name` (or "
        "`statistic`) must equal the declared product name. The host reads "
        f"{aliases}; it also reads effect_scale/effect_measure/scale and "
        "unit/units. Any other key is kept but is not read as one of those "
        "fields. Declared here: " + ", ".join(products) + "."
    )


def association_binary_sensitivity_output_contract(step: AnalysisStep) -> str:
    """Render the full host boundary for an agent-coded sensitivity grid."""

    contract = association_binary_sensitivity_contract(step)
    if contract is None:
        return ""
    return (
        "ASSOCIATION BINARY SENSITIVITY CONTRACT (binding):\n"
        "- scientific_capability=association_freeform_v1 under the closed "
        "binary-sensitivity contract; inherit "
        f"the estimand from {contract.parent_product}; do not replace its "
        "exposure, outcome, effect measure, or adjustment set.\n"
        "- Emit exactly one result row for each sensitivity_spec_id and no "
        "undeclared variants: "
        f"{json.dumps(list(contract.sensitivity_ids), ensure_ascii=False)}.\n"
        "- Register the one planned output table and mirror its rows in "
        "step_summary.analysis_rows. Every row requires analysis_id, n_stays, "
        "n_deaths, odds_ratio, ci_low, and ci_high; counts must be coherent and "
        "the positive odds ratio must lie inside its finite interval.\n"
        "- n_stays and n_deaths describe the row-specific eligibility set before "
        "complete-case model filtering. Do not relabel fitted model N as the "
        "eligibility denominator; report model N separately if useful.\n"
        "- Numerically condition every non-linear continuous adjustment before "
        "basis expansion (for example, center before squaring or use a stable "
        "spline basis). Treat overflow, divide-by-zero, invalid Hessian/standard "
        "errors, or optimizer pseudo-convergence as a failed model; never publish "
        "a fallback null estimate such as OR=1.\n"
    )


def association_binary_sensitivity_repair_lines(step: AnalysisStep) -> list[str]:
    """Return the compact immutable form used in repair transports."""

    contract = association_binary_sensitivity_contract(step)
    if contract is None:
        return []
    return [
        "ASSOCIATION BINARY SENSITIVITY CONTRACT (binding): minimal patch",
        "- Preserve scientific_capability=association_freeform_v1 under the "
        "closed binary-sensitivity contract and exactly one result row per "
        "sensitivity_spec_id: "
        + json.dumps(list(contract.sensitivity_ids), ensure_ascii=False)
        + ".",
        "- n_stays/n_deaths are row eligibility denominators before "
        "complete-case model filtering, not fitted model N; preserve "
        "positive coherent odds_ratio/ci_low/ci_high values.",
        "- Numerically condition non-linear continuous terms (center "
        "before squaring or use a stable spline basis). Treat overflow, "
        "invalid Hessian/standard errors, and optimizer pseudo-convergence "
        "as model failure; never publish a fallback null estimate such as OR=1.",
    ]


__all__ = [
    "association_binary_sensitivity_output_contract",
    "association_binary_sensitivity_repair_lines",
    "statistic_payload_shape_directive",
]
