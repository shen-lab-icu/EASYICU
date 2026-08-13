"""Pure planning contract for pre-specified robustness analyses.

This module owns only the serializable planning types and validation rules. It
must not read or write locks/evidence, execute models, inspect run artifacts, or
promote any result. Those lifecycle responsibilities remain in
``robustness_panel`` and the execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

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

#: The one typed-product kind for each output of the deterministic locked-grid
#: replay.  This belongs to the dependency-neutral robustness planning contract
#: because neither the Planner nor a plan-time gate may guess whether an output
#: implemented as a row-bearing CSV is a table or a scalar JSON statistic.
#: The execution owner, schema facade, and product-promise gate all consume the
#: same immutable value.
ROBUSTNESS_REPLAY_OUTPUT_PRODUCT_KINDS: Mapping[str, str] = MappingProxyType(
    {
        "robustness_matrix": "table",
        "robustness_summary": "table",
        "specification_grid": "table",
        "membership_change": "table",
        "outcome_label_executability": "table",
        "missingness_strategy_notes": "log",
        "primary_effect": "statistic",
        "complete_case_n": "statistic",
    }
)

#: The missing-data strategy that re-fits the primary model on complete cases.
COMPLETE_CASE_STRATEGY = "complete_case"

#: Where such a spec names the variables whose completeness defines the set.
#:
#: ``missing_override`` is a free-form dict, so this key was enforced by the
#: execution-time equivalence proof and by the concept validator, and published
#: by neither -- the planner directive's own worked example showed
#: ``{"strategy": "complete_case"}`` with no variable list at all. Measured over
#: 94 locked complete-case specs from real runs: 44 wrote ``variables``, 50 wrote
#: no list (the example, copied), and 3 invented ``columns`` or
#: ``required_variables``. Every one of the 53 was refused at execution, after
#: the whole analysis had already run, with "complete-case equivalence requires
#: explicit locked variables".
#:
#: Which variables define the complete-case set is a scientific choice -- the
#: host must not infer it from the model, because a model fitted on a wider set
#: and a complete-case restriction over a narrower one are different analyses.
#: So it is required here, where the Planner can still supply it.
COMPLETE_CASE_VARIABLES_KEY = "variables"

#: Keys a real plan has used for the same list without being told the name.
#: They are reported back, not accepted: a second spelling of a scientific
#: declaration is how two consumers end up disagreeing about which variables
#: were held complete.
_COMPLETE_CASE_VARIABLES_NEAR_MISSES = ("columns", "required_variables", "vars")


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


def complete_case_variables(spec: RobustnessSpec) -> Optional[List[str]]:
    """Return the declared complete-case variables, or ``None`` if not declared.

    One reader, so the plan-time requirement and the execution-time equivalence
    proof cannot ask for different keys.
    """

    override = spec.missing_override or {}
    if not isinstance(override, dict):
        return None
    if str(override.get("strategy") or "").strip().lower() != COMPLETE_CASE_STRATEGY:
        return None
    raw = override.get(COMPLETE_CASE_VARIABLES_KEY)
    if not isinstance(raw, list) or not raw:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        return None
    return [value.strip() for value in raw]


def validate_planner_robustness_specs(specs: Sequence[RobustnessSpec]) -> None:
    """Structural validity **plus** what only a Planner-authored spec must satisfy.

    Deliberately separate from :func:`validate_robustness_specs`. That function
    is asked "are these structurally valid?" by four callers with four
    populations -- a plan under review, a lock being re-read on resume, a
    re-derived list, and the framework's own case-neutral placeholders. A
    requirement that a complete-case spec name its variables is a question
    about *Planner output*: a case-neutral placeholder cannot answer it, and a
    lock that already passed should not be re-judged by a rule added later.
    Attaching it to the shared validator judged a population the callers were
    not asking about, and 26 tests said so.
    """

    validate_robustness_specs(specs)
    problems: List[str] = []
    for spec in specs:
        problems.extend(_complete_case_problems(spec))
    if problems:
        raise RobustnessPlanError("; ".join(problems))


def _complete_case_problems(spec: RobustnessSpec) -> List[str]:
    """Refuse a complete-case spec that names no variables, while it can be fixed."""

    override = spec.missing_override
    if not isinstance(override, dict):
        return []
    if str(override.get("strategy") or "").strip().lower() != COMPLETE_CASE_STRATEGY:
        return []
    variables = complete_case_variables(spec)
    if variables is not None:
        if len(variables) != len(set(variables)):
            return [
                f"robustness_specs[{spec.spec_id}].missing_override."
                f"{COMPLETE_CASE_VARIABLES_KEY} repeats a variable"
            ]
        return []
    near_misses = [
        key for key in _COMPLETE_CASE_VARIABLES_NEAR_MISSES if key in override
    ]
    detail = (
        f" (the spec declares {', '.join(sorted(near_misses))} instead)"
        if near_misses
        else ""
    )
    return [
        f"robustness_specs[{spec.spec_id}] uses strategy "
        f"'{COMPLETE_CASE_STRATEGY}' and must declare "
        f"missing_override.{COMPLETE_CASE_VARIABLES_KEY} as a non-empty list of "
        f"column names{detail}; the host will not infer which variables define "
        "the complete-case set, because restricting on a narrower or wider set "
        "than the model uses is a different analysis"
    ]


def _dict_or_none(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


__all__ = [
    "COMPLETE_CASE_STRATEGY",
    "COMPLETE_CASE_VARIABLES_KEY",
    "MIN_AXIS_COUNTS",
    "MIN_TOTAL_SPEC_COUNT",
    "ROBUSTNESS_REPLAY_OUTPUT_PRODUCT_KINDS",
    "RobustnessAxis",
    "RobustnessPlanError",
    "RobustnessSpec",
    "complete_case_variables",
    "validate_planner_robustness_specs",
    "validate_robustness_specs",
]
