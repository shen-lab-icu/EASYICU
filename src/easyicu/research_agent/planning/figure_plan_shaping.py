"""Typed plan-shaping rules for deterministic article figures.

This owner bridges article figure policy to an executable plan.  It may bind a
renderer only when exact typed source products have unique planner-declared
owners.  It never searches run files, chooses values, or invents a scientific
analysis.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..contracts.declared_product import typed_product
from ..schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    ResearchContext,
    ValidationFinding,
)
from .figure_strategy import (
    DATA_QUALITY_FIGURE_PRODUCT,
    DATA_QUALITY_FIGURE_REQUIRED_INPUTS,
)

_AUDIT_PANEL_TOKENS = (
    "audit",
    "completeness",
    "sensitivity",
    "leakage",
    "calibration",
)


def _method_head(method: str) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(method or "").strip().lower()
    ).strip("_")
    return normalized.split("_with_", 1)[0]


def dedicated_renderer_consumes_typed_source(
    steps: Sequence[AnalysisStep],
    *,
    source: str,
) -> bool:
    """Return whether one explicit renderer already owns a typed source."""

    source_product = typed_product(source)
    if source_product is None:
        return False
    for step in steps or []:
        if _method_head(str(step.method or "")) != "visualization":
            continue
        input_products = {
            product
            for raw_input in step.inputs or []
            if (product := typed_product(raw_input)) is not None
        }
        figure_products = [
            product
            for output in step.expected_outputs or []
            if (product := typed_product(output)) is not None
            and product[0] == "figure"
        ]
        if source_product in input_products and len(figure_products) == 1:
            return True
    return False


def step_declares_audit_panel(step: AnalysisStep) -> bool:
    """Whether a step declares an audit/sensitivity/robustness display item."""

    for text in [step.intent or "", *(step.expected_outputs or [])]:
        lowered = str(text or "").lower()
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered)
            for token in _AUDIT_PANEL_TOKENS
        ):
            return True
    return False


def ensure_data_quality_figure_step(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Bind the article data-quality figure to exact typed audit tables.

    ``context`` remains explicit because this is a plan-shaping boundary and
    callers bind it alongside every other article-level transformation.  The
    current typed rule is case-neutral and needs no context fields.
    """

    del context
    steps = list(plan.steps or [])
    required_inputs = tuple(DATA_QUALITY_FIGURE_REQUIRED_INPUTS)
    for step in steps:
        outputs = {str(value) for value in step.expected_outputs or []}
        inputs = {str(value) for value in step.inputs or []}
        if any(value.startswith("figure:") for value in outputs) and (
            DATA_QUALITY_FIGURE_PRODUCT in outputs
            or bool(inputs.intersection(required_inputs))
        ):
            return plan, []

    producers = {
        input_key: [
            str(step.step_id)
            for step in steps
            if input_key in {str(value) for value in step.expected_outputs or []}
        ]
        for input_key in required_inputs
    }
    missing = [key for key, owners in producers.items() if not owners]
    ambiguous = {key: owners for key, owners in producers.items() if len(owners) > 1}
    if missing or ambiguous:
        return plan, [
            ValidationFinding(
                validator="data_quality_figure_contract",
                severity="warning",
                message=(
                    "A source-bound data-quality figure was not appended because "
                    "its audit-table ownership is incomplete or ambiguous."
                ),
                detail={
                    "reason_code": "data_quality_figure_source_not_closed",
                    "required_inputs": list(required_inputs),
                    "missing_inputs": missing,
                    "ambiguous_inputs": ambiguous,
                },
            )
        ]

    audit_step = AnalysisStep(
        step_id=f"{len(steps) + 1:02d}_data_quality_figure",
        planned_analysis_role="auxiliary",
        intent=(
            "Render the exact missingness and measurement-process audit tables "
            "as a source-data-bound data-quality figure. Do not scan run files, "
            "redefine denominators, impute values, or re-run an analysis."
        ),
        method="visualization",
        inputs=list(required_inputs),
        expected_outputs=[DATA_QUALITY_FIGURE_PRODUCT],
        icu_rule_refs=["visualization_rule", "missingness_rule"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=input_key, mode="all_rows")
            for input_key in required_inputs
        ],
    )
    return plan.model_copy(update={"steps": [*steps, audit_step]}), [
        ValidationFinding(
            validator="data_quality_figure_contract",
            severity="warning",
            message=(
                "Plan declared both typed audit sources but no data-quality "
                f"renderer; appended '{audit_step.step_id}' with exact inputs."
            ),
            detail={
                "reason_code": "data_quality_figure_bound_to_typed_sources",
                "appended_step_id": audit_step.step_id,
                "inputs": list(required_inputs),
                "producer_step_ids": producers,
            },
        )
    ]


__all__ = [
    "dedicated_renderer_consumes_typed_source",
    "ensure_data_quality_figure_step",
    "step_declares_audit_panel",
]
