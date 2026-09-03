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

from .cohort_product_keys import sole_typed_cohort_input
from .ownership_verdict import OwnershipVerdict

PREDICTION_MODEL_ANALYSIS_KIND = "static_prediction_model"
PREDICTION_PRIMARY_ACTION = "prediction.discrimination_calibration"
PREDICTION_SCORES_PRODUCT = "table:prediction_scores"
PREDICTION_PERFORMANCE_PRODUCT = "table:model_performance"


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


__all__ = [
    "PREDICTION_MODEL_ANALYSIS_KIND",
    "PREDICTION_PERFORMANCE_PRODUCT",
    "PREDICTION_PRIMARY_ACTION",
    "PREDICTION_SCORES_PRODUCT",
    "static_prediction_execution_verdict",
    "static_prediction_model_columns",
]
