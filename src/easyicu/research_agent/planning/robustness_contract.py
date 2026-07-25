"""Pure planning contract for pre-specified robustness analyses.

This module owns only the serializable planning types and validation rules. It
must not read or write locks/evidence, execute models, inspect run artifacts, or
promote any result. Those lifecycle responsibilities remain in
``robustness_panel`` and the execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

from .cohort_contract import CohortDefinition, coerce_cohort_definition

RobustnessAxis = Literal["cohort", "missing", "outcome"]

# Robustness requirements are study- and data-dependent. A universal 3/2/2
# quota forced planners (and the historical fallback) to invent unsupported
# cohort and outcome definitions just to satisfy schema cardinality. The
# article contract decides whether robustness is required; this typed contract
# requires at least one explicit, executable candidate and validates whichever
# scientific axes the current task can actually support.
MIN_AXIS_COUNTS: Dict[str, int] = {"cohort": 0, "missing": 0, "outcome": 0}
MIN_TOTAL_SPEC_COUNT = 1


class RobustnessPlanError(ValueError):
    """Raised when robustness specifications are missing or invalid."""


@dataclass(frozen=True)
class RobustnessSpec:
    spec_id: str
    axis: RobustnessAxis
    description: str
    cohort_override: Optional[CohortDefinition] = None
    missing_override: Optional[Dict[str, Any]] = None
    outcome_override: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "axis": self.axis,
            "description": self.description,
            "cohort_override": (
                self.cohort_override.to_dict()
                if self.cohort_override is not None
                else None
            ),
            "missing_override": self.missing_override,
            "outcome_override": self.outcome_override,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobustnessSpec":
        return cls(
            spec_id=str(data.get("spec_id") or "").strip(),
            axis=str(data.get("axis") or "").strip(),  # type: ignore[arg-type]
            description=str(data.get("description") or "").strip(),
            cohort_override=coerce_cohort_definition(data.get("cohort_override")),
            missing_override=_dict_or_none(data.get("missing_override")),
            outcome_override=_dict_or_none(data.get("outcome_override")),
        )


def validate_robustness_specs(specs: Sequence[RobustnessSpec]) -> None:
    counts = {axis: 0 for axis in MIN_AXIS_COUNTS}
    seen_ids: set[str] = set()
    problems: List[str] = []
    if len(specs) < MIN_TOTAL_SPEC_COUNT:
        problems.append(
            "robustness_specs require at least one task-supported alternative"
        )
    for spec in specs:
        if not spec.spec_id:
            problems.append("spec_id must be non-empty")
        if spec.spec_id in seen_ids:
            problems.append(f"duplicate spec_id: {spec.spec_id}")
        seen_ids.add(spec.spec_id)
        if spec.axis not in counts:
            problems.append(f"unknown robustness axis for {spec.spec_id}: {spec.axis}")
            continue
        counts[spec.axis] += 1
    for axis, minimum in MIN_AXIS_COUNTS.items():
        if counts[axis] < minimum:
            problems.append(
                f"robustness_specs require at least {minimum} {axis} axis spec(s); "
                f"got {counts[axis]}"
            )
    if problems:
        raise RobustnessPlanError("; ".join(problems))


def _dict_or_none(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


__all__ = [
    "MIN_AXIS_COUNTS",
    "MIN_TOTAL_SPEC_COUNT",
    "RobustnessAxis",
    "RobustnessPlanError",
    "RobustnessSpec",
    "validate_robustness_specs",
]
