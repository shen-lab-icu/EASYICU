"""Recognize a runtime-owned step without importing an execution authority."""

from __future__ import annotations

import re

from ..schema import AnalysisStep

_RUNTIME_CONTRACT_REF = re.compile(r"^scientific_runtime_contract:[0-9a-f]{64}$")


def has_scientific_runtime_owner(step: AnalysisStep) -> bool:
    """Identify the declared owner, not verify or grant its signed authority."""

    return any(
        _RUNTIME_CONTRACT_REF.fullmatch(str(ref or ""))
        for ref in (step.icu_rule_refs or ())
    )


def declared_runtime_outcomes(step: AnalysisStep) -> tuple[str, ...]:
    """Project only explicit, input-bound endpoints; never guess from a method."""
    contract = step.runtime_outcome_contract
    if (
        contract is None
        or contract.owner_ref not in step.icu_rule_refs
        or not set(contract.outcomes).issubset(step.inputs)
    ):
        return ()
    return contract.outcomes
