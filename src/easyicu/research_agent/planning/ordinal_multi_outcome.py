"""Typed owner for ordinal-exposure studies with two outcome kinds.

Planner-only runs intentionally know column metadata but not patient rows.  The
same metadata contract must therefore drive both outline obligations and the
deterministic compiler; otherwise the Planner can be required to emit a step
that the compiler later rejects solely because ``observed_domain`` is absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..authority.declared_levels import observed_levels_for
from ..research_context.typed import declared_domain_for_variable
from ..schema import ConceptDescriptor, ResearchContext


_NUMERIC_METADATA_DTYPE = re.compile(
    r"^(?:u?int\d*|float\d*|double|number|decimal(?:\d+)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OrdinalMultiOutcomeContract:
    """Closed planning coordinates for one supported ordinal gradient study."""

    exposure: str
    binary_outcome: str
    continuous_outcome: str
    exposure_levels: tuple[Any, ...]
    binary_levels: tuple[Any, ...]

    @property
    def variables(self) -> tuple[str, str, str]:
        return self.exposure, self.binary_outcome, self.continuous_outcome


def is_numeric_metadata_dtype(value: object) -> bool:
    """Whether a row-free descriptor declares an ordinary numeric dtype."""

    return _NUMERIC_METADATA_DTYPE.fullmatch(str(value or "").strip()) is not None


def _closed_levels(variable: ConceptDescriptor) -> tuple[Any, ...]:
    observed = observed_levels_for(name=variable.name, variables={variable.name: variable})
    if observed:
        return tuple(observed)
    declared, _basis = declared_domain_for_variable(variable)
    return tuple(declared or ())


def _is_continuous_outcome_candidate(variable: ConceptDescriptor) -> bool:
    if str(variable.role.value) != "outcome" or variable.is_ordinal:
        return False
    if _closed_levels(variable):
        return False
    domain = variable.observed_domain or {}
    if domain.get("is_binary") is True:
        return False
    return domain.get("is_binary") is False or is_numeric_metadata_dtype(
        variable.dtype
    )


def resolve_ordinal_multi_outcome_contract(
    context: ResearchContext,
) -> OrdinalMultiOutcomeContract | None:
    """Resolve an unambiguous metadata-only ordinal multi-outcome contract.

    The v1 deterministic ordered-trend owner requires a >=3-level ordinal
    exposure and a 0/1 primary endpoint.  Exactly one other cohort outcome may
    be admitted as the continuous outcome.  Ambiguity returns ``None`` rather
    than choosing a column by name or position.
    """

    exposure = str(context.primary_exposure or "").strip()
    binary_outcome = str(context.target_outcome or "").strip()
    exposure_descriptor = context.variable(exposure)
    outcome_descriptor = context.variable(binary_outcome)
    if exposure_descriptor is None or outcome_descriptor is None:
        return None
    if not (
        exposure_descriptor.is_ordinal
        or str(exposure_descriptor.role.value) == "ordinal_score"
    ):
        return None
    exposure_levels = _closed_levels(exposure_descriptor)
    if len(exposure_levels) < 3:
        return None

    observed_binary_levels = _closed_levels(outcome_descriptor)
    endpoint = context.endpoint
    endpoint_levels: tuple[Any, ...] = ()
    if (
        endpoint is not None
        and endpoint.name == binary_outcome
        and endpoint.kind == "binary"
    ):
        endpoint_levels = tuple(endpoint.levels or ())
    binary_levels = endpoint_levels or observed_binary_levels
    if list(binary_levels) != [0, 1]:
        return None
    if observed_binary_levels and list(observed_binary_levels) != [0, 1]:
        return None

    time_columns = set(context.cohort.time_columns)
    continuous_outcomes: list[str] = []
    for name in context.cohort.outcome_columns:
        if name == binary_outcome or name in time_columns:
            continue
        descriptor = context.variable(name)
        if descriptor is not None and _is_continuous_outcome_candidate(descriptor):
            continuous_outcomes.append(name)
    if len(continuous_outcomes) != 1:
        return None
    return OrdinalMultiOutcomeContract(
        exposure=exposure,
        binary_outcome=binary_outcome,
        continuous_outcome=continuous_outcomes[0],
        exposure_levels=exposure_levels,
        binary_levels=binary_levels,
    )


__all__ = [
    "OrdinalMultiOutcomeContract",
    "is_numeric_metadata_dtype",
    "resolve_ordinal_multi_outcome_contract",
]
