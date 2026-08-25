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
    ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS,
    ASSOCIATION_SENSITIVITY_COMPOSITE_FIXED_INPUTS,
    COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS,
    COHORT_FLOW_FIGURE_PANELS,
    COHORT_FLOW_INPUT,
    DATA_QUALITY_AUDIT_ROLES,
    DATA_QUALITY_FIGURE_PANELS,
    EXPOSURE_OUTCOME_DISTRIBUTION_COUNTS_ONLY_FIGURE_PANELS,
    EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_PANELS,
    EXPOSURE_OUTCOME_DISTRIBUTION_INPUT,
    GROUPED_DESCRIPTIVE_DISTRIBUTION_FIGURE_PANELS,
    GROUPED_DESCRIPTIVE_DISTRIBUTION_INPUT,
    LANDMARK_ASSOCIATION_COMPOSITE_INPUTS,
    MEASUREMENT_PROCESS_AUDIT_INPUT,
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
    ROBUSTNESS_FIGURE_INPUT,
    ROBUSTNESS_FIGURE_KNOWN_INPUTS,
    association_sensitivity_composite_panels,
    absolute_risk_association_composite_panels,
    cohort_balance_association_composite_panels,
    data_quality_audit_source_candidates,
    landmark_association_composite_panels,
    measurement_availability_figure_panels,
    robustness_figure_panels,
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

_PREDICTION_FIGURE_CORE_INPUTS = frozenset(
    {
        "table:prediction_scores",
        "table:model_performance",
        "table:calibration",
        "table:validation",
    }
)
_PREDICTION_CLINICAL_UTILITY_INPUT = "table:clinical_utility"
_ASSOCIATION_FIGURE_CORE_INPUTS = frozenset(
    {
        "table:exposure_outcome_distribution",
        "table:adjusted_association_estimates",
    }
)


def _method_head(method: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(method or "").strip().lower()).strip(
        "_"
    )
    return normalized.split("_with_", 1)[0]


def dedicated_renderer_consumes_typed_source(
    steps: Sequence[AnalysisStep],
    *,
    source: str,
) -> bool:
    """Return whether one explicit renderer already owns a typed source."""

    if typed_product(source) is None:
        return False
    for step in steps or []:
        if _method_head(str(step.method or "")) != "visualization":
            continue
        inputs = {str(raw_input) for raw_input in step.inputs or []}
        figure_products = [
            product
            for output in step.expected_outputs or []
            if (product := typed_product(output)) is not None and product[0] == "figure"
        ]
        all_row_inputs = {
            str(contract.input_key)
            for contract in step.input_consumption_contracts or []
            if contract.mode == "all_rows"
        }
        if (
            inputs == {source}
            and all_row_inputs == {source}
            and len(figure_products) == 1
        ):
            return True
    return False


def _next_step_id(steps: Sequence[AnalysisStep], suffix: str) -> str:
    occupied = {str(step.step_id) for step in steps}
    next_index = len(steps) + 1
    while (step_id := f"{next_index:02d}_{suffix}") in occupied:
        next_index += 1
    return step_id


def _next_figure_output(steps: Sequence[AnalysisStep], base: str) -> str:
    occupied = {str(output) for step in steps for output in step.expected_outputs or []}
    output = base
    suffix = 2
    while output in occupied:
        output = f"{base}_{suffix}"
        suffix += 1
    return output


def _closed_data_quality_sources(
    steps: Sequence[AnalysisStep],
) -> tuple[
    tuple[str, str] | None,
    dict[str, list[tuple[str, str]]],
    list[str],
    dict[str, list[tuple[str, str]]],
]:
    candidates = data_quality_audit_source_candidates(steps)
    missing = [role for role, values in candidates.items() if not values]
    ambiguous = {role: values for role, values in candidates.items() if len(values) > 1}
    if missing or ambiguous:
        return None, candidates, missing, ambiguous
    return (
        (
            candidates["measurement_missingness"][0][0],
            candidates["measurement_process"][0][0],
        ),
        candidates,
        [],
        {},
    )


def _data_quality_panel_templates(sources: tuple[str, str]):
    replacements = {
        MISSINGNESS_MEASUREMENT_AUDIT_INPUT: sources[0],
        MEASUREMENT_PROCESS_AUDIT_INPUT: sources[1],
    }
    return tuple(
        panel.model_copy(
            update={
                "source_products": tuple(
                    replacements.get(source, source) for source in panel.source_products
                )
            }
        )
        for panel in DATA_QUALITY_FIGURE_PANELS
    )


def _dedicated_renderer_consumes_exact_sources(
    steps: Sequence[AnalysisStep],
    *,
    sources: Sequence[str],
) -> bool:
    required = {str(source) for source in sources}
    for step in steps:
        if _method_head(str(step.method or "")) != "visualization":
            continue
        figure_outputs = [
            str(output)
            for output in step.expected_outputs or []
            if str(output).startswith("figure:")
        ]
        inputs = {str(value) for value in step.inputs or []}
        all_row_inputs = {
            str(contract.input_key)
            for contract in step.input_consumption_contracts or []
            if contract.mode == "all_rows"
        }
        if (
            inputs == required
            and all_row_inputs == required
            and len(figure_outputs) == 1
        ):
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


def ensure_cohort_accounting_figure_step(
    *,
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Append the deterministic cohort-flow renderer when its source is closed."""

    steps = list(plan.steps or [])
    owners = [
        str(step.step_id)
        for step in steps
        if COHORT_FLOW_INPUT in {str(value) for value in step.expected_outputs or []}
    ]
    if len(owners) != 1 or dedicated_renderer_consumes_typed_source(
        steps,
        source=COHORT_FLOW_INPUT,
    ):
        return plan, []
    step_id = _next_step_id(steps, "cohort_accounting_figure")
    figure_output = _next_figure_output(steps, "figure:cohort_flow")
    figure_step = AnalysisStep(
        step_id=step_id,
        planned_analysis_role="auxiliary",
        intent=(
            "Render the exact cohort-accounting table using its registered "
            "deterministic article-figure contract. Do not redefine eligibility, "
            "recalculate attrition, or scan run files."
        ),
        method="visualization",
        inputs=[COHORT_FLOW_INPUT],
        expected_outputs=[figure_output],
        icu_rule_refs=["visualization_rule"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(
                input_key=COHORT_FLOW_INPUT,
                mode="all_rows",
            )
        ],
    )
    return plan.model_copy(update={"steps": [*steps, figure_step]}), [
        ValidationFinding(
            validator="cohort_accounting_figure_contract",
            severity="warning",
            message=(
                "Bound a rendering-only cohort-accounting figure to the unique "
                f"typed source {COHORT_FLOW_INPUT!r}."
            ),
            detail={
                "reason_code": "cohort_accounting_figure_bound_to_typed_source",
                "appended_step_id": step_id,
                "source_product": COHORT_FLOW_INPUT,
                "producer_step_id": owners[0],
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
    required_inputs, candidates, missing, ambiguous = _closed_data_quality_sources(
        steps
    )
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
                    "required_audit_roles": list(DATA_QUALITY_AUDIT_ROLES),
                    "missing_inputs": missing,
                    "ambiguous_inputs": ambiguous,
                },
            )
        ]
    assert required_inputs is not None
    if _dedicated_renderer_consumes_exact_sources(steps, sources=required_inputs):
        return plan, []

    figure_output = _next_figure_output(steps, DATA_QUALITY_FIGURE_PRODUCT)

    audit_step = AnalysisStep(
        step_id=_next_step_id(steps, "data_quality_figure"),
        planned_analysis_role="auxiliary",
        intent=(
            "Render the exact missingness and measurement-process audit tables "
            "as a source-data-bound data-quality figure. Do not scan run files, "
            "redefine denominators, impute values, or re-run an analysis."
        ),
        method="visualization",
        inputs=list(required_inputs),
        expected_outputs=[figure_output],
        icu_rule_refs=["visualization_rule", "missingness_rule"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=input_key, mode="all_rows")
            for input_key in required_inputs
        ],
        figure_panels=[
            panel.bind(figure_output=figure_output)
            for panel in _data_quality_panel_templates(required_inputs)
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
                "producer_step_ids": {
                    role: [step_id for _source, step_id in values]
                    for role, values in candidates.items()
                },
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
            measurement_availability_figure_panels(MISSINGNESS_MEASUREMENT_AUDIT_INPUT)
        ),
        frozenset(DATA_QUALITY_FIGURE_REQUIRED_INPUTS): DATA_QUALITY_FIGURE_PANELS,
    }
    data_quality_sources, _candidates, _missing, _ambiguous = (
        _closed_data_quality_sources(plan.steps)
    )
    changed = False
    findings: list[ValidationFinding] = []
    steps: list[AnalysisStep] = []
    clinical_utility_owners = [
        candidate.step_id
        for candidate in plan.steps
        if _PREDICTION_CLINICAL_UTILITY_INPUT
        in {str(value) for value in candidate.expected_outputs}
    ]
    sensitivity_outputs = [
        (str(output), candidate.step_id)
        for candidate in plan.steps
        if candidate.method == "verified_association_model_grid"
        for output in candidate.expected_outputs
        if str(output).startswith("table:")
    ]
    completeness_owners = [
        candidate.step_id
        for candidate in plan.steps
        if "table:exposure_component_completeness_audit"
        in {str(value) for value in candidate.expected_outputs}
    ]
    for step in plan.steps:
        figure_outputs = [
            str(output)
            for output in step.expected_outputs
            if str(output).startswith("figure:")
        ]
        input_set = frozenset(str(value) for value in step.inputs)
        if (
            _method_head(str(step.method or "")) == "visualization"
            and step.planned_analysis_role == "auxiliary"
            and input_set == _PREDICTION_FIGURE_CORE_INPUTS
            and len(figure_outputs) == 1
            and len(clinical_utility_owners) == 1
        ):
            changed = True
            step = step.model_copy(
                update={
                    "inputs": [
                        *step.inputs,
                        _PREDICTION_CLINICAL_UTILITY_INPUT,
                    ],
                    "input_consumption_contracts": [
                        *step.input_consumption_contracts,
                        ArtifactConsumptionContract(
                            input_key=_PREDICTION_CLINICAL_UTILITY_INPUT,
                            mode="all_rows",
                        ),
                    ],
                }
            )
            input_set = frozenset(str(value) for value in step.inputs)
            findings.append(
                ValidationFinding(
                    validator="deterministic_figure_plan_binding",
                    severity="warning",
                    message=(
                        "Bound the registered clinical-utility table to the "
                        f"prediction figure step {step.step_id!r}."
                    ),
                    detail={
                        "reason": "prediction_figure_clinical_utility_bound",
                        "step_id": step.step_id,
                        "source_step_id": clinical_utility_owners[0],
                        "input": _PREDICTION_CLINICAL_UTILITY_INPUT,
                    },
                )
            )
        if (
            _method_head(str(step.method or "")) == "visualization"
            and step.planned_analysis_role == "auxiliary"
            and input_set == _ASSOCIATION_FIGURE_CORE_INPUTS
            and len(figure_outputs) == 1
            and len(sensitivity_outputs) == 1
            and len(completeness_owners) == 1
        ):
            sensitivity_input, sensitivity_owner = sensitivity_outputs[0]
            additions = (
                sensitivity_input,
                "table:exposure_component_completeness_audit",
            )
            step = step.model_copy(
                update={
                    "inputs": [*step.inputs, *additions],
                    "input_consumption_contracts": [
                        *step.input_consumption_contracts,
                        *(
                            ArtifactConsumptionContract(input_key=value, mode="all_rows")
                            for value in additions
                        ),
                    ],
                }
            )
            input_set = frozenset(str(value) for value in step.inputs)
            changed = True
            findings.append(
                ValidationFinding(
                    validator="deterministic_figure_plan_binding",
                    severity="warning",
                    message=(
                        "Bound the registered scientific-sensitivity and component-"
                        f"completeness tables to association figure {step.step_id!r}."
                    ),
                    detail={
                        "reason": "association_scientific_sensitivity_bound",
                        "step_id": step.step_id,
                        "sensitivity_source_step_id": sensitivity_owner,
                        "completeness_source_step_id": completeness_owners[0],
                        "inputs": list(additions),
                    },
                )
            )
        templates = templates_by_inputs.get(input_set)
        if (
            ROBUSTNESS_FIGURE_INPUT in input_set
            and input_set <= ROBUSTNESS_FIGURE_KNOWN_INPUTS
        ):
            templates = robustness_figure_panels(step.inputs)
        if data_quality_sources is not None and input_set == frozenset(
            data_quality_sources
        ):
            templates = _data_quality_panel_templates(data_quality_sources)
        if (
            ASSOCIATION_SENSITIVITY_COMPOSITE_FIXED_INPUTS <= input_set
            and len(input_set) == 4
        ):
            templates = association_sensitivity_composite_panels(step.inputs)
        if input_set == frozenset(COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS):
            templates = cohort_balance_association_composite_panels(step.inputs)
        if input_set == frozenset(ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS):
            templates = absolute_risk_association_composite_panels(step.inputs)
        if input_set == frozenset({EXPOSURE_OUTCOME_DISTRIBUTION_INPUT}):
            producers = [
                candidate
                for candidate in plan.steps
                if EXPOSURE_OUTCOME_DISTRIBUTION_INPUT
                in {str(value) for value in candidate.expected_outputs}
            ]
            if len(producers) == 1:
                distribution = producers[0].exposure_outcome_distribution_spec
                if (
                    distribution is not None
                    and distribution.schema_version
                    == "easyicu.exposure_outcome_distribution/3"
                ):
                    templates = EXPOSURE_OUTCOME_DISTRIBUTION_COUNTS_ONLY_FIGURE_PANELS
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
        tabular_inputs = {
            str(value)
            for value in step.inputs
            if (product := typed_product(value)) is not None
            and product[0] in {"artifact", "table"}
        }
        if all_row_inputs != tabular_inputs:
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


def close_empty_deterministic_figure_contracts(
    *,
    plan: AnalysisPlan,
    eligible_step_ids: Sequence[str] | None = None,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Close output and all-row contracts when typed inputs fix a renderer.

    This is a schema migration for legacy visualization steps, not a scientific
    choice: it applies only to exact input profiles already owned by a sealed
    deterministic renderer and preserves step ids, order, intent, and inputs.
    """

    eligible = None if eligible_step_ids is None else set(eligible_step_ids)
    occupied_outputs = {
        str(output) for step in plan.steps for output in step.expected_outputs
    }
    data_quality_sources, _candidates, _missing, _ambiguous = (
        _closed_data_quality_sources(plan.steps)
    )
    changed = False
    findings: list[ValidationFinding] = []
    steps: list[AnalysisStep] = []
    for step in plan.steps:
        step_id = str(step.step_id)
        inputs = tuple(str(value) for value in step.inputs)
        input_set = frozenset(inputs)
        templates = None
        if input_set == frozenset({COHORT_FLOW_INPUT}):
            templates = COHORT_FLOW_FIGURE_PANELS
        elif (
            ROBUSTNESS_FIGURE_INPUT in input_set
            and input_set <= ROBUSTNESS_FIGURE_KNOWN_INPUTS
        ):
            templates = robustness_figure_panels(inputs)
        elif data_quality_sources is not None and input_set == frozenset(
            data_quality_sources
        ):
            templates = _data_quality_panel_templates(data_quality_sources)
        elif (
            LANDMARK_ASSOCIATION_COMPOSITE_INPUTS <= input_set
            and len(input_set) == 4
            and any(
                value.startswith("table:")
                and value.partition(":")[2].endswith("landmark_rcs_curve")
                for value in input_set
            )
            and any(
                value.partition(":")[2]
                in {"measurement_process", "measurement_process_audit"}
                for value in input_set
            )
        ):
            templates = landmark_association_composite_panels(inputs)
        elif input_set == frozenset(COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS):
            templates = cohort_balance_association_composite_panels(inputs)
        elif input_set == frozenset(ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS):
            templates = absolute_risk_association_composite_panels(inputs)
        if (
            templates is None
            or (eligible is not None and step_id not in eligible)
            or _method_head(str(step.method or "")) != "visualization"
            or step.planned_analysis_role != "auxiliary"
            or step.expected_outputs
            or step.model_requirements
            or step.trajectory_stability_spec is not None
        ):
            steps.append(step)
            continue
        base = re.sub(r"[^a-z0-9]+", "_", step_id.lower()).strip("_")
        base = re.sub(r"^[0-9]+_", "", base) or "deterministic_figure"
        figure_output = f"figure:{base}"
        suffix = 2
        while figure_output in occupied_outputs:
            figure_output = f"figure:{base}_{suffix}"
            suffix += 1
        occupied_outputs.add(figure_output)
        contracts = [
            ArtifactConsumptionContract(input_key=value, mode="all_rows")
            for value in inputs
            if typed_product(value) is not None
            and typed_product(value)[0] in {"artifact", "table"}
        ]
        revised = step.model_copy(
            update={
                "expected_outputs": [figure_output],
                "input_consumption_contracts": contracts,
                "figure_panels": [
                    panel.bind(figure_output=figure_output) for panel in templates
                ],
                "icu_rule_refs": list(
                    dict.fromkeys([*step.icu_rule_refs, "visualization_rule"])
                ),
            }
        )
        steps.append(revised)
        changed = True
        findings.append(
            ValidationFinding(
                validator="deterministic_figure_plan_binding",
                severity="warning",
                message=(
                    f"Closed empty visualization contract for step {step_id!r} "
                    "from its exact typed renderer inputs."
                ),
                detail={
                    "reason": "empty_visualization_contract_closed",
                    "step_id": step_id,
                    "figure_output": figure_output,
                },
            )
        )
    return (plan.model_copy(update={"steps": steps}) if changed else plan), findings


__all__ = [
    "bind_deterministic_figure_panels",
    "close_empty_deterministic_figure_contracts",
    "dedicated_renderer_consumes_typed_source",
    "ensure_cohort_accounting_figure_step",
    "ensure_data_quality_figure_step",
    "ensure_primary_result_figure_step",
    "step_declares_audit_panel",
]
