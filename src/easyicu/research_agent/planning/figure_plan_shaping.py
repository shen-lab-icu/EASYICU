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
from ..contracts.figure_plan import (
    COHORT_FLOW_FIGURE_PANELS,
    COHORT_FLOW_INPUT,
    DATA_QUALITY_FIGURE_PANELS,
    EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_PANELS,
    EXPOSURE_OUTCOME_DISTRIBUTION_INPUT,
    GROUPED_DESCRIPTIVE_DISTRIBUTION_FIGURE_PANELS,
    GROUPED_DESCRIPTIVE_DISTRIBUTION_INPUT,
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
    measurement_availability_figure_panels,
)
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

_PRIMARY_RESULT_FIGURE_TEMPLATES = {
    EXPOSURE_OUTCOME_DISTRIBUTION_INPUT: (EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_PANELS),
    GROUPED_DESCRIPTIVE_DISTRIBUTION_INPUT: (
        GROUPED_DESCRIPTIVE_DISTRIBUTION_FIGURE_PANELS
    ),
}


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


def ensure_primary_result_figure_step(
    *,
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Append one known deterministic renderer for a unique primary table.

    Existing secondary figures do not satisfy the article hero role.  When the
    Planner's single primary step already owns exactly one table supported by a
    deterministic figure contract, the host can safely add its rendering-only
    descendant without choosing an estimand or scanning run files.
    """

    primary_steps = [
        step for step in plan.steps if step.planned_analysis_role == "primary"
    ]
    if len(primary_steps) != 1:
        return plan, []
    primary_sources = [
        str(output)
        for output in primary_steps[0].expected_outputs
        if str(output) in _PRIMARY_RESULT_FIGURE_TEMPLATES
    ]
    if len(primary_sources) != 1:
        return plan, []
    source = primary_sources[0]
    if dedicated_renderer_consumes_typed_source(plan.steps, source=source):
        return plan, []

    occupied_step_ids = {str(step.step_id) for step in plan.steps}
    next_index = len(plan.steps) + 1
    while (step_id := f"{next_index:02d}_primary_result_figure") in occupied_step_ids:
        next_index += 1
    occupied_outputs = {
        str(output) for step in plan.steps for output in step.expected_outputs
    }
    figure_output = "figure:primary_result"
    suffix = 2
    while figure_output in occupied_outputs:
        figure_output = f"figure:primary_result_{suffix}"
        suffix += 1
    figure_step = AnalysisStep(
        step_id=step_id,
        planned_analysis_role="auxiliary",
        intent=(
            "Render the exact primary descriptive result table using its "
            "registered deterministic article-figure contract. Do not choose "
            "another result, recalculate an estimand, or scan run files."
        ),
        method="visualization",
        inputs=[source],
        expected_outputs=[figure_output],
        icu_rule_refs=["visualization_rule"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=source, mode="all_rows")
        ],
    )
    return plan.model_copy(update={"steps": [*plan.steps, figure_step]}), [
        ValidationFinding(
            validator="primary_result_figure_contract",
            severity="warning",
            message=(
                "Bound a rendering-only primary-result figure to the unique "
                f"typed primary source {source!r}."
            ),
            detail={
                "reason": "primary_result_figure_bound_to_typed_primary_source",
                "step_id": step_id,
                "source_product": source,
                "figure_output": figure_output,
            },
        )
    ]


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
        figure_panels=[
            panel.bind(figure_output=DATA_QUALITY_FIGURE_PRODUCT)
            for panel in DATA_QUALITY_FIGURE_PANELS
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


def bind_deterministic_figure_panels(
    *,
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Bind exact panels for a renderer already selected by typed inputs.

    This owner never chooses a source product. It only recognizes exact typed
    inputs already present in the Planner step and projects the selected
    deterministic renderer's shared contract before plan digest and review.
    The Planner response is still a draft at this boundary.  Once its typed
    inputs and all-row consumption contracts select a deterministic renderer,
    compile that renderer's exact panels into the final plan shown for human
    review.  This prevents the reviewed plan from promising a chart that the
    selected host renderer cannot produce.
    """

    templates_by_inputs = {
        frozenset({COHORT_FLOW_INPUT}): COHORT_FLOW_FIGURE_PANELS,
        frozenset({EXPOSURE_OUTCOME_DISTRIBUTION_INPUT}): (
            EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_PANELS
        ),
        frozenset({GROUPED_DESCRIPTIVE_DISTRIBUTION_INPUT}): (
            GROUPED_DESCRIPTIVE_DISTRIBUTION_FIGURE_PANELS
        ),
        frozenset({MISSINGNESS_MEASUREMENT_AUDIT_INPUT}): (
            measurement_availability_figure_panels(
                MISSINGNESS_MEASUREMENT_AUDIT_INPUT
            )
        ),
        frozenset(DATA_QUALITY_FIGURE_REQUIRED_INPUTS): DATA_QUALITY_FIGURE_PANELS,
    }
    changed = False
    findings: list[ValidationFinding] = []
    steps: list[AnalysisStep] = []
    for step in plan.steps:
        figure_outputs = [
            str(output)
            for output in step.expected_outputs
            if str(output).startswith("figure:")
        ]
        input_set = frozenset(str(value) for value in step.inputs)
        templates = templates_by_inputs.get(input_set)
        if templates is None and len(input_set) == 1:
            input_key = next(iter(input_set))
            kind, separator, product = input_key.partition(":")
            if kind == "table" and separator:
                producers = [
                    candidate
                    for candidate in plan.steps
                    if input_key in {str(value) for value in candidate.expected_outputs}
                    and candidate.measurement_audit_spec is not None
                    and candidate.measurement_audit_spec.audit_for(product)
                    == "measurement_missingness"
                ]
                if len(producers) == 1:
                    templates = measurement_availability_figure_panels(input_key)
        if (
            _method_head(str(step.method or "")) != "visualization"
            or step.planned_analysis_role != "auxiliary"
            or len(figure_outputs) != 1
            or templates is None
        ):
            steps.append(step)
            continue
        all_row_inputs = {
            str(contract.input_key)
            for contract in step.input_consumption_contracts
            if contract.mode == "all_rows"
        }
        if all_row_inputs != {str(value) for value in step.inputs}:
            steps.append(step)
            continue
        figure_output = figure_outputs[0]
        bound = [panel.bind(figure_output=figure_output) for panel in templates]
        scientific_signatures = {
            (
                panel.article_role,
                panel.chart_type,
                tuple(sorted(panel.source_products)),
            )
            for panel in step.figure_panels
        }
        bound_signatures = {
            (
                panel.article_role,
                panel.chart_type,
                tuple(sorted(panel.source_products)),
            )
            for panel in bound
        }
        if step.figure_panels != bound:
            changed = True
            reason = (
                "deterministic_figure_panels_normalized"
                if step.figure_panels and scientific_signatures != bound_signatures
                else "deterministic_figure_panels_bound"
            )
            step = step.model_copy(update={"figure_panels": bound})
            findings.append(
                ValidationFinding(
                    validator="deterministic_figure_plan_binding",
                    severity="warning",
                    message=(
                        f"Bound exact deterministic panel contracts for figure "
                        f"step {step.step_id!r}."
                    ),
                    detail={
                        "reason": reason,
                        "step_id": step.step_id,
                        "figure_output": figure_output,
                    },
                )
            )
        steps.append(step)
    return (plan.model_copy(update={"steps": steps}) if changed else plan), findings


__all__ = [
    "bind_deterministic_figure_panels",
    "dedicated_renderer_consumes_typed_source",
    "ensure_data_quality_figure_step",
    "ensure_primary_result_figure_step",
    "step_declares_audit_panel",
]
