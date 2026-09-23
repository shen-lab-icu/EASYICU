"""Exact claim boundary for the host-owned static prediction executor.

The primary step may carry supporting raw columns after its typed cohort input
(for example, measurement-process columns consumed by adjacent audits).  Those
columns are not model features.  The ordered input contract is therefore:

``model columns -> one typed cohort input -> supporting inputs``.

Keeping that rule here lets planning, capability assessment, and execution ask
one dependency-neutral owner instead of independently guessing a predictor
roster from every raw input on the step.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

from .cohort_product_keys import sole_typed_cohort_input
from .ownership_verdict import OwnershipVerdict

PREDICTION_MODEL_ANALYSIS_KIND = "static_prediction_model"
PREDICTION_PRIMARY_ACTION = "prediction.discrimination_calibration"
PREDICTION_SCORES_PRODUCT = "table:prediction_scores"
PREDICTION_PERFORMANCE_PRODUCT = "table:model_performance"
PREDICTION_INTERNAL_VALIDATION_PRODUCT = "table:validation"
PREDICTION_CALIBRATION_PRODUCT = "table:calibration"
PREDICTION_CLINICAL_UTILITY_PRODUCT = "table:clinical_utility"

#: Action -> the exact ordered products the static prediction owner writes.
STATIC_PREDICTION_ACTION_OUTPUTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        PREDICTION_PRIMARY_ACTION: (
            PREDICTION_SCORES_PRODUCT,
            PREDICTION_PERFORMANCE_PRODUCT,
        ),
        "prediction.internal_validation": (PREDICTION_INTERNAL_VALIDATION_PRODUCT,),
        "prediction.calibration_metrics": (PREDICTION_CALIBRATION_PRODUCT,),
        "prediction.decision_curve": (PREDICTION_CLINICAL_UTILITY_PRODUCT,),
    }
)
_SECONDARY_ROLES = frozenset({"secondary", "auxiliary"})


def static_prediction_model_columns(step: object) -> tuple[str, ...]:
    """Return the exact ordered model roster declared before the cohort input."""

    cohort_input = sole_typed_cohort_input(step)
    inputs = tuple(
        str(value or "").strip()
        for value in (getattr(step, "inputs", None) or ())
    )
    if cohort_input is None or inputs.count(cohort_input) != 1:
        return ()
    boundary = inputs.index(cohort_input)
    columns = tuple(value for value in inputs[:boundary] if value and ":" not in value)
    if len(columns) != boundary or len(columns) != len(set(columns)):
        return ()
    return columns


def static_prediction_execution_verdict(step: object) -> OwnershipVerdict:
    """Own only a fully declared, single static binary-prediction primary."""

    if str(getattr(step, "scientific_action_id", "") or "") != PREDICTION_PRIMARY_ACTION:
        return OwnershipVerdict.wrong_shape(
            PREDICTION_MODEL_ANALYSIS_KIND,
            reason="the primary step does not declare the static prediction action",
        )
    outputs = tuple(
        str(value or "").strip()
        for value in (getattr(step, "expected_outputs", None) or ())
    )
    expected = (PREDICTION_SCORES_PRODUCT, PREDICTION_PERFORMANCE_PRODUCT)
    if outputs != expected:
        return OwnershipVerdict.wrong_shape(
            PREDICTION_MODEL_ANALYSIS_KIND,
            reason=f"the static prediction action requires exact outputs {expected!r}",
        )
    if getattr(step, "planned_analysis_role", None) != "primary":
        return OwnershipVerdict.wrong_shape(
            PREDICTION_MODEL_ANALYSIS_KIND,
            reason="the static prediction action must be the primary analysis",
        )
    if sole_typed_cohort_input(step) is None:
        return OwnershipVerdict.incomplete_declaration(
            PREDICTION_MODEL_ANALYSIS_KIND,
            missing=("inputs[typed_cohort]",),
            reason="the static prediction primary has no unique typed cohort input",
        )
    columns = static_prediction_model_columns(step)
    if len(columns) < 2:
        return OwnershipVerdict.incomplete_declaration(
            PREDICTION_MODEL_ANALYSIS_KIND,
            missing=("inputs[model_columns_before_typed_cohort]",),
            reason=(
                "the static prediction primary must declare a unique predictor/outcome "
                "roster before its typed cohort input"
            ),
        )
    forbidden = (
        "table_one_spec",
        "cohort_definition_spec",
        "measurement_audit_spec",
        "robustness_replay_spec",
        "trajectory_stability_spec",
    )
    populated = tuple(
        name for name in forbidden if getattr(step, name, None) is not None
    )
    has_model_requirements = bool(getattr(step, "model_requirements", None))
    if populated or has_model_requirements:
        mixed = populated + (("model_requirements",) if has_model_requirements else ())
        return OwnershipVerdict.wrong_shape(
            PREDICTION_MODEL_ANALYSIS_KIND,
            reason=(
                "the static prediction primary mixes another typed owner contract: "
                + ", ".join(mixed)
            ),
        )
    return OwnershipVerdict.claim(
        PREDICTION_MODEL_ANALYSIS_KIND,
        reason=(
            "the step declares one typed cohort and an exact model-column prefix "
            "for the deterministic static prediction owner"
        ),
    )


def static_prediction_owns_step(step: object) -> bool:
    """Whether the static prediction owner claims this exact action/product/input shape.

    The primary is judged by :func:`static_prediction_execution_verdict`.  A
    secondary action reads only the primary's per-row scores, so it evaluates
    the sealed predictions rather than refitting on another frame.
    """

    action = str(getattr(step, "scientific_action_id", "") or "")
    expected = STATIC_PREDICTION_ACTION_OUTPUTS.get(action)
    if expected is None or tuple(getattr(step, "expected_outputs", None) or ()) != expected:
        return False
    if action == PREDICTION_PRIMARY_ACTION:
        return static_prediction_execution_verdict(step).claimed
    typed_inputs = tuple(
        value for value in getattr(step, "inputs", None) or () if ":" in value
    )
    if (
        getattr(step, "planned_analysis_role", None) not in _SECONDARY_ROLES
        or typed_inputs != (PREDICTION_SCORES_PRODUCT,)
    ):
        return False
    return bool(
        getattr(step, "table_one_spec", None) is None
        and getattr(step, "cohort_definition_spec", None) is None
        and getattr(step, "measurement_audit_spec", None) is None
        and getattr(step, "robustness_replay_spec", None) is None
        and getattr(step, "trajectory_stability_spec", None) is None
        and not getattr(step, "model_requirements", None)
    )


def static_prediction_features(
    declared_columns: Iterable[str], *, outcome: str, group_source: Optional[str]
) -> tuple[str, ...]:
    """The predictor roster: declared model columns minus outcome and patient group."""

    excluded = {outcome, group_source}
    return tuple(column for column in declared_columns if column not in excluded)


def static_prediction_executes_robustness_spec(
    spec: Any, *, features: Iterable[str], outcome: str
) -> bool:
    """Whether the owner executes this plan-locked robustness specification.

    The owner understands exactly one variant: a complete-case refit of the
    unchanged model, outcome, predictor roster and patient split, compared
    with the primary's training-split imputation.  The complete-case set must
    be the model roster itself -- a narrower or wider set is another analysis
    -- and nothing else about the cohort or outcome may change.  Runtime facts
    (columns present, both partitions still two-class) stay with the executor.
    """

    missing = getattr(spec, "missing_override", None)
    if (
        getattr(spec, "axis", None) != "missing"
        or getattr(spec, "cohort_override", None) is not None
        or getattr(spec, "outcome_override", None) is not None
        or not isinstance(missing, Mapping)
        or str(missing.get("strategy") or "") != "complete_case"
    ):
        return False
    variables = tuple(str(value or "").strip() for value in missing.get("variables", ()) or ())
    return bool(
        variables
        and len(variables) == len(set(variables))
        and set(variables) == {*features, outcome}
    )


__all__ = [
    "PREDICTION_CALIBRATION_PRODUCT",
    "PREDICTION_CLINICAL_UTILITY_PRODUCT",
    "PREDICTION_INTERNAL_VALIDATION_PRODUCT",
    "PREDICTION_MODEL_ANALYSIS_KIND",
    "PREDICTION_PERFORMANCE_PRODUCT",
    "PREDICTION_PRIMARY_ACTION",
    "PREDICTION_SCORES_PRODUCT",
    "STATIC_PREDICTION_ACTION_OUTPUTS",
    "static_prediction_execution_verdict",
    "static_prediction_executes_robustness_spec",
    "static_prediction_features",
    "static_prediction_model_columns",
    "static_prediction_owns_step",
]
