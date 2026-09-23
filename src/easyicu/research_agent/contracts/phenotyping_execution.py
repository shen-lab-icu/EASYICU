"""Exact claim boundary for the host-owned cross-sectional phenotyping executor.

The executor fixes imputation, standardisation, candidate-k scoring, clustering
and resampling mechanics under the published
``easyicu.cross_sectional_phenotyping_policy``.  Which plan steps it claims is
a separate question that planning and review also need to ask -- for example,
whether a plan's cluster-number and stability steps really run on the host's
fixed candidate grid and resampling design.  Keeping that rule here lets the
executor, the plan reviewer and the capability layer ask one
dependency-neutral owner instead of each re-deriving the step shape.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from .cohort_product_keys import sole_typed_cohort_input
from .phenotyping_features import PHENOTYPING_PRIMARY_ACTION

__all__ = [
    "CLUSTER_SELECTION_PRODUCT",
    "CLUSTER_STABILITY_PRODUCT",
    "CROSS_SECTIONAL_PHENOTYPING_ACTION_OUTPUTS",
    "PHENOTYPE_ASSIGNMENTS_PRODUCT",
    "PHENOTYPE_PROFILES_PRODUCT",
    "cross_sectional_phenotyping_owns_step",
    "phenotyping_raw_input_columns",
]

PHENOTYPE_PROFILES_PRODUCT = "table:phenotype_profiles"
PHENOTYPE_ASSIGNMENTS_PRODUCT = "table:phenotype_assignments"
CLUSTER_SELECTION_PRODUCT = "table:cluster_selection"
CLUSTER_STABILITY_PRODUCT = "table:cluster_stability"

#: Action -> the exact ordered products the owner writes for it.
CROSS_SECTIONAL_PHENOTYPING_ACTION_OUTPUTS: Mapping[str, tuple[str, ...]] = (
    MappingProxyType(
        {
            PHENOTYPING_PRIMARY_ACTION: (
                PHENOTYPE_PROFILES_PRODUCT,
                PHENOTYPE_ASSIGNMENTS_PRODUCT,
            ),
            "phenotyping.k_selection": (CLUSTER_SELECTION_PRODUCT,),
            "phenotyping.cluster_stability": (CLUSTER_STABILITY_PRODUCT,),
        }
    )
)

_DOWNSTREAM_ROLES = frozenset({"secondary", "sensitivity", "auxiliary"})


def phenotyping_raw_input_columns(step: Any) -> tuple[str, ...]:
    """Declared raw (non-typed) input columns, in declaration order."""

    return tuple(
        value
        for item in getattr(step, "inputs", None) or ()
        if (value := str(item or "").strip()) and ":" not in value
    )


def cross_sectional_phenotyping_owns_step(step: Any) -> bool:
    """Whether the host phenotyping executor claims this exact step shape.

    The primary cluster solution must be the primary analysis, read one typed
    cohort input and declare at least two raw columns.  Candidate-k selection
    and stability are downstream steps that read only the primary assignment
    product, so they replay the sealed primary matrix rather than reopening
    cohort bytes.  No other typed specification may ride on the step.
    """

    action = str(getattr(step, "scientific_action_id", "") or "")
    expected = CROSS_SECTIONAL_PHENOTYPING_ACTION_OUTPUTS.get(action)
    if expected is None or tuple(getattr(step, "expected_outputs", None) or ()) != expected:
        return False
    typed = tuple(
        value for value in getattr(step, "inputs", None) or () if ":" in value
    )
    role = getattr(step, "planned_analysis_role", None)
    if action == PHENOTYPING_PRIMARY_ACTION:
        if (
            role != "primary"
            or not sole_typed_cohort_input(step)
            or len(phenotyping_raw_input_columns(step)) < 2
        ):
            return False
    elif role not in _DOWNSTREAM_ROLES or typed != (PHENOTYPE_ASSIGNMENTS_PRODUCT,):
        return False
    return bool(
        getattr(step, "table_one_spec", None) is None
        and getattr(step, "cohort_definition_spec", None) is None
        and getattr(step, "measurement_audit_spec", None) is None
        and getattr(step, "robustness_replay_spec", None) is None
        and getattr(step, "trajectory_stability_spec", None) is None
        and not getattr(step, "model_requirements", None)
    )
