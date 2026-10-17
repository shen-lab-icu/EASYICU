"""Validate explicit continuous-outcome selections against typed study metadata."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..authority.declared_levels import closed_planning_levels_for
from .table_one_semantics import table_one_variable_kind


def continuous_summary_columns(
    *, grouping: str, raw_inputs: Sequence[str], variables: Mapping[str, Any],
) -> tuple[str, ...]:
    """Order a declared roster; never add, infer, or drop an outcome."""
    if (not grouping or grouping not in raw_inputs or len(raw_inputs) < 2
            or len(raw_inputs) != len(set(raw_inputs))
            or not set(raw_inputs).issubset(variables)):
        raise ValueError("continuous summary needs an explicit grouping and unique available outcomes")
    if (str(variables[grouping].role.value) in {"id", "index", "meta"}
            or len(closed_planning_levels_for(name=grouping, variables=dict(variables))) < 2):
        raise ValueError("continuous summary grouping needs a closed domain with at least two levels")
    outcomes = tuple(name for name in raw_inputs if name != grouping)
    for name in outcomes:
        variable = variables[name]
        levels = closed_planning_levels_for(name=name, variables=dict(variables))
        dtype = str(variable.dtype or "").casefold()
        if (str(variable.role.value) != "outcome"
                or not dtype.startswith(("float", "double", "int", "uint", "decimal"))
                or table_one_variable_kind(variable, levels) != "continuous"):
            raise ValueError(f"continuous summary requires a typed numeric continuous outcome: {name}")
    return (grouping, *outcomes)
