"""Typed plan-shaping rules for deterministic article figures.

This owner bridges article figure policy to an executable plan.  It may bind a
renderer only when exact typed source products have unique planner-declared
owners.  It never searches run files, chooses values, or invents a scientific
analysis.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts.declared_product import typed_product
from ..contracts.figure_plan import (
    ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS,
    ASSOCIATION_SUMMARY_COMPOSITE_INPUTS,
    BALANCE_ASSOCIATION_COMPOSITE_INPUTS,
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
    association_summary_composite_panels,
    absolute_risk_association_composite_panels,
    balance_association_composite_panels,
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
    MeasurementAuditProduct,
    ResearchContext,
    ValidationFinding,
)
from .figure_strategy import (
    DATA_QUALITY_FIGURE_PRODUCT,
    DATA_QUALITY_FIGURE_REQUIRED_INPUTS,
)
from .sensitivity_plan_shaping import ensure_prespecified_sensitivity_steps

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

_REPORT_INPUT_PRODUCT_KINDS = frozenset({"manifest", "statistic", "table"})


def migrate_render_step_contract(
    child: AnalysisStep,
    source_tokens: Sequence[str],
    *,
    intent: Optional[str] = None,
    method: Optional[str] = None,
) -> AnalysisStep:
    """Rebind one render step and its cardinality contracts atomically."""

    existing = {
        str(contract.input_key): contract
        for contract in child.input_consumption_contracts
    }
    contracts = [
        existing.get(token)
        or ArtifactConsumptionContract(input_key=token, mode="all_rows")
        for token in source_tokens
        if (parsed := typed_product(token)) is not None
        and parsed[0] in {"table", "statistic"}
    ]
    update: Dict[str, Any] = {
        "inputs": list(source_tokens),
        "input_consumption_contracts": contracts,
    }
    if intent is not None:
        update["intent"] = intent
    if method is not None:
        update["method"] = method
    return child.model_copy(update=update)


def augment_report_typed_product_inputs(
    *,
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Bind report consumers to unique prior structured result products."""

    producer_counts: Dict[Tuple[str, str], int] = {}
    for step in plan.steps or []:
        for output in step.expected_outputs or []:
            product = typed_product(output)
            if product is not None:
                producer_counts[product] = producer_counts.get(product, 0) + 1

    prior_outputs: List[str] = []
    revised_steps: List[AnalysisStep] = []
    additions_by_step: Dict[str, List[str]] = {}
    for step in plan.steps or []:
        is_report = any(
            (product := typed_product(output)) is not None and product[0] == "report"
            for output in step.expected_outputs or []
        )
        inputs = list(step.inputs or [])
        seen = set(inputs)
        additions: List[str] = []
        if is_report:
            for output in prior_outputs:
                product = typed_product(output)
                if (
                    product is None
                    or product[0] not in _REPORT_INPUT_PRODUCT_KINDS
                    or producer_counts.get(product) != 1
                    or output in seen
                ):
                    continue
                inputs.append(output)
                additions.append(output)
                seen.add(output)
        if additions:
            additions_by_step[str(step.step_id)] = additions
            revised_steps.append(step.model_copy(update={"inputs": inputs}))
        else:
            revised_steps.append(step)
        prior_outputs.extend(str(output) for output in step.expected_outputs or [])

    if not additions_by_step:
        return plan, []
    return plan.model_copy(update={"steps": revised_steps}), [
        ValidationFinding(
            validator="planner_input_closure",
            severity="info",
            message=(
                "Bound report consumers to unique prior typed result products "
                "so failed producers cannot be silently recomputed from raw data."
            ),
            detail={
                "reason": "report_typed_product_input_closure",
                "added_inputs_by_step": additions_by_step,
            },
        )
    ]


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


def ensure_landmark_association_composite_figure_step(
    *,
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Append the registered claim-led landmark association renderer.

    The signed landmark runtime publishes complementary tables rather than one
    overloaded result table. Once the plan has unique owners for the exact
    ratio curve, model-standardised risk curve, robustness, and
    measurement-process products,
    selecting their registered renderer is presentation plumbing rather than
    a new scientific decision.
    """

    produced = {
        str(output) for step in plan.steps for output in step.expected_outputs or []
    }
    curve_candidates = sorted(
        output
        for output in produced
        if output.startswith("table:")
        and output.partition(":")[2].endswith("landmark_rcs_curve")
    )
    adjusted_risk_candidates = sorted(
        output
        for output in produced
        if output.startswith("table:")
        and any(
            token in output.partition(":")[2]
            for token in (
                "adjusted_absolute_risk",
                "standardized_absolute_risk",
                "standardised_absolute_risk",
                "absolute_risk_curve",
            )
        )
    )
    measurement_candidates = sorted(
        output
        for output in produced
        if output.partition(":")[2]
        in {"measurement_process", "measurement_process_audit"}
    )
    if (
        len(curve_candidates) != 1
        or len(adjusted_risk_candidates) != 1
        or len(measurement_candidates) != 1
    ):
        return plan, []
    sources = (
        curve_candidates[0],
        adjusted_risk_candidates[0],
        "table:robustness_summary",
        measurement_candidates[0],
    )
    owners = {
        source: [
            str(step.step_id)
            for step in plan.steps
            if source in {str(output) for output in step.expected_outputs}
        ]
        for source in sources
    }
    if any(len(step_ids) != 1 for step_ids in owners.values()):
        return plan, []
    curve_owner = next(
        step
        for step in plan.steps
        if str(step.step_id) == owners[curve_candidates[0]][0]
    )
    if (
        curve_owner.planned_analysis_role != "primary"
        or _method_head(str(curve_owner.method or ""))
        != "signed_landmark_restricted_cubic_spline"
        or _dedicated_renderer_consumes_exact_sources(plan.steps, sources=sources)
    ):
        return plan, []

    steps = list(plan.steps)
    reusable_index = next(
        (
            index
            for index, step in enumerate(steps)
            if step.planned_analysis_role == "auxiliary"
            and _method_head(str(step.method or "")) == "visualization"
            and not step.figure_panels
            and len(step.expected_outputs) == 1
            and str(step.expected_outputs[0]).startswith("figure:")
            and "article" in (f"{step.step_id} {step.expected_outputs[0]}".lower())
            and "table:robustness_summary"
            in {str(value) for value in step.inputs}
            and (
                adjusted_risk_candidates[0]
                in {str(value) for value in step.inputs}
                or "table:absolute_risk_context"
                in {str(value) for value in step.inputs}
            )
        ),
        None,
    )
    figure_output = (
        str(steps[reusable_index].expected_outputs[0])
        if reusable_index is not None
        else _next_figure_output(steps, "figure:landmark_association")
    )
    figure_step = AnalysisStep(
        step_id=(
            str(steps[reusable_index].step_id)
            if reusable_index is not None
            else _next_step_id(steps, "landmark_association_figure")
        ),
        planned_analysis_role="auxiliary",
        intent=(
            "Render the exact signed landmark association curve, aligned "
            "model-standardised absolute-risk curve, robustness summary, and "
            "measurement-process audit using "
            "their registered deterministic composite contract. Do not refit "
            "a model, change denominators, or scan run files."
        ),
        method="visualization",
        inputs=list(sources),
        expected_outputs=[figure_output],
        icu_rule_refs=["visualization_rule", "missingness_rule"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=source, mode="all_rows")
            for source in sources
        ],
        figure_panels=[
            panel.bind(figure_output=figure_output)
            for panel in landmark_association_composite_panels(sources)
        ],
    )
    if reusable_index is None:
        steps.append(figure_step)
        reason_code = "landmark_association_composite_figure_bound"
    else:
        steps[reusable_index] = figure_step
        reason_code = "landmark_association_composite_figure_rebound"
    return plan.model_copy(update={"steps": steps}), [
        ValidationFinding(
            validator="landmark_association_figure_contract",
            severity="warning",
            message=(
                "Bound the signed landmark analysis to its exact deterministic "
                "four-panel article figure."
            ),
            detail={
                "reason_code": reason_code,
                "appended_step_id": figure_step.step_id,
                "inputs": list(sources),
                "producer_step_ids": owners,
                "figure_output": figure_output,
            },
        )
    ]


def ensure_absolute_risk_association_composite_figure_step(
    *,
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Append the registered association summary when all sources are closed.

    The four source tables already fix the descriptive context, headline
    adjusted estimate, and robustness display. Selecting their registered
    renderer is presentation plumbing; it does not choose or refit an
    estimand.
    """

    sources = ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS
    owners = {
        source: [
            str(step.step_id)
            for step in plan.steps
            if source in {str(output) for output in step.expected_outputs}
        ]
        for source in sources
    }
    if any(len(step_ids) != 1 for step_ids in owners.values()):
        return plan, []
    primary_owner = next(
        step
        for step in plan.steps
        if str(step.step_id) == owners["table:adjusted_association_estimates"][0]
    )
    if (
        primary_owner.planned_analysis_role != "primary"
        or _dedicated_renderer_consumes_exact_sources(plan.steps, sources=sources)
    ):
        return plan, []

    steps = list(plan.steps)
    step_id = _next_step_id(steps, "absolute_risk_association_figure")
    figure_output = _next_figure_output(steps, "figure:absolute_risk_association")
    figure_step = AnalysisStep(
        step_id=step_id,
        planned_analysis_role="auxiliary",
        intent=(
            "Render the exact observed absolute-risk context, adjusted primary "
            "association, and prespecified robustness products using their "
            "registered deterministic composite contract. Do not refit a model, "
            "change denominators, or scan run files."
        ),
        method="visualization",
        inputs=list(sources),
        expected_outputs=[figure_output],
        icu_rule_refs=["visualization_rule"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=source, mode="all_rows")
            for source in sources
        ],
        figure_panels=[
            panel.bind(figure_output=figure_output)
            for panel in absolute_risk_association_composite_panels(sources)
        ],
    )
    return plan.model_copy(update={"steps": [*steps, figure_step]}), [
        ValidationFinding(
            validator="absolute_risk_association_figure_contract",
            severity="warning",
            message=(
                "Bound the adjusted association plan to its exact deterministic "
                "absolute-risk and robustness article figure."
            ),
            detail={
                "reason_code": "absolute_risk_association_composite_figure_bound",
                "appended_step_id": step_id,
                "inputs": list(sources),
                "producer_step_ids": owners,
                "figure_output": figure_output,
            },
        )
    ]


def select_deterministic_result_renderers(
    *,
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Run the ordered result-renderer selection passes in one call.

    Which deterministic renderers a result plan receives, and in what order, is
    figure-shaping policy owned here -- not pipeline orchestration.  Order
    matters: the single-result pass claims a plan that has one primary product,
    so the four-panel landmark composite is offered afterwards to the plans the
    first pass declined.  The pipeline asks for the selection once instead of
    re-stating the sequence at its call site.
    """

    findings: list[ValidationFinding] = []
    for select in (
        ensure_primary_result_figure_step,
        ensure_absolute_risk_association_composite_figure_step,
        ensure_landmark_association_composite_figure_step,
    ):
        plan, pass_findings = select(plan=plan)
        findings.extend(pass_findings)
    return plan, findings


def ensure_descriptive_context_figure_step(
    *,
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Append the deterministic exposure/outcome context renderer if unique.

    In an adjusted association study the primary model owns the estimand, while
    the separately declared exposure/outcome table owns the absolute observed
    context.  A single-source rendering child is presentation plumbing only;
    it does not promote that descriptive table to the primary analysis.
    """

    source = EXPOSURE_OUTCOME_DISTRIBUTION_INPUT
    owners = [
        str(step.step_id)
        for step in plan.steps
        if source in {str(output) for output in step.expected_outputs}
    ]
    if len(owners) != 1 or dedicated_renderer_consumes_typed_source(
        plan.steps,
        source=source,
    ):
        return plan, []
    steps = list(plan.steps)
    step_id = _next_step_id(steps, "descriptive_context_figure")
    figure_output = _next_figure_output(steps, "figure:descriptive_context")
    figure_step = AnalysisStep(
        step_id=step_id,
        planned_analysis_role="auxiliary",
        intent=(
            "Render the exact exposure/outcome distribution table using its "
            "registered deterministic descriptive-result contract. Do not "
            "refit a model, change denominators, or scan run files."
        ),
        method="visualization",
        inputs=[source],
        expected_outputs=[figure_output],
        icu_rule_refs=["visualization_rule"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=source, mode="all_rows")
        ],
    )
    return plan.model_copy(update={"steps": [*steps, figure_step]}), [
        ValidationFinding(
            validator="descriptive_context_figure_contract",
            severity="warning",
            message=(
                "Bound a rendering-only descriptive-result figure to the "
                f"unique typed source {source!r}."
            ),
            detail={
                "reason": "descriptive_context_figure_bound_to_typed_source",
                "step_id": step_id,
                "source_step_id": owners[0],
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
    primary_population_sources = [
        (str(output), str(step.step_id))
        for step in steps
        if step.planned_analysis_role == "primary"
        for output in step.expected_outputs or []
        if str(output).startswith("table:")
        and str(output).partition(":")[2].endswith("population_flow")
    ]
    generic_sources = [
        (COHORT_FLOW_INPUT, str(step.step_id))
        for step in steps
        if COHORT_FLOW_INPUT in {str(value) for value in step.expected_outputs or []}
    ]
    candidates = (
        primary_population_sources
        if len(primary_population_sources) == 1
        else generic_sources
    )
    if len(candidates) != 1:
        return plan, []
    source, owner = candidates[0]
    if dedicated_renderer_consumes_typed_source(
        steps,
        source=source,
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
        inputs=[source],
        expected_outputs=[figure_output],
        icu_rule_refs=["visualization_rule"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(
                input_key=source,
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
                f"typed source {source!r}."
            ),
            detail={
                "reason_code": "cohort_accounting_figure_bound_to_typed_source",
                "appended_step_id": step_id,
                "source_product": source,
                "producer_step_id": owner,
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
    plan, audit_pair_findings = _complete_typed_data_quality_audit_pair(plan)
    steps = list(plan.steps or [])
    required_inputs, candidates, missing, ambiguous = _closed_data_quality_sources(
        steps
    )
    if missing or ambiguous:
        return plan, [
            *audit_pair_findings,
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
            ),
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
        *audit_pair_findings,
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
        ),
    ]


def _complete_typed_data_quality_audit_pair(
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, list[ValidationFinding]]:
    """Complete the mechanical missingness/process pair on one typed audit.

    The two tables are complementary views over the same already-authorized
    inputs: whether each value is present, and how measurement opportunity is
    distributed.  Adding the missing count table to a uniquely owned typed
    audit does not choose a population, estimand, variable, or missing-data
    strategy.  It only closes the deterministic source contract needed by the
    data-quality renderer.  Ambiguous or untyped plans remain untouched.
    """

    candidates = data_quality_audit_source_candidates(plan.steps)
    missing_roles = [role for role, values in candidates.items() if not values]
    present_roles = [role for role, values in candidates.items() if len(values) == 1]
    if len(missing_roles) != 1 or len(present_roles) != 1:
        return plan, []
    if any(len(values) > 1 for values in candidates.values()):
        return plan, []

    missing_role = missing_roles[0]
    present_role = present_roles[0]
    owner_step_id = candidates[present_role][0][1]
    owner_indexes = [
        index
        for index, step in enumerate(plan.steps)
        if str(step.step_id) == owner_step_id
        and step.measurement_audit_spec is not None
    ]
    if len(owner_indexes) != 1:
        return plan, []

    output_by_role = {
        "measurement_missingness": MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
        "measurement_process": MEASUREMENT_PROCESS_AUDIT_INPUT,
    }
    output = output_by_role[missing_role]
    if any(
        output in {str(value) for value in step.expected_outputs} for step in plan.steps
    ):
        return plan, []

    index = owner_indexes[0]
    owner = plan.steps[index]
    assert owner.measurement_audit_spec is not None
    product_id = output.split(":", 1)[1]
    completed_spec = owner.measurement_audit_spec.model_copy(
        update={
            "products": [
                *owner.measurement_audit_spec.products,
                MeasurementAuditProduct(product_id=product_id, audit=missing_role),
            ]
        }
    )
    completed_owner = owner.model_copy(
        update={
            "expected_outputs": [*owner.expected_outputs, output],
            "measurement_audit_spec": completed_spec,
        }
    )
    steps = list(plan.steps)
    steps[index] = completed_owner
    return plan.model_copy(update={"steps": steps}), [
        ValidationFinding(
            validator="data_quality_figure_contract",
            severity="warning",
            message=(
                "Completed the uniquely owned typed measurement audit with "
                f"its complementary {missing_role!r} count table."
            ),
            detail={
                "reason_code": "data_quality_audit_pair_completed",
                "step_id": owner_step_id,
                "preserved_audit_role": present_role,
                "appended_audit_role": missing_role,
                "appended_output": output,
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
        frozenset(BALANCE_ASSOCIATION_COMPOSITE_INPUTS): (
            balance_association_composite_panels(BALANCE_ASSOCIATION_COMPOSITE_INPUTS)
        ),
        frozenset(ASSOCIATION_SUMMARY_COMPOSITE_INPUTS): (
            association_summary_composite_panels(ASSOCIATION_SUMMARY_COMPOSITE_INPUTS)
        ),
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
                            ArtifactConsumptionContract(
                                input_key=value, mode="all_rows"
                            )
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
        if len(input_set) == 1:
            cohort_source = next(iter(input_set))
            if cohort_source.startswith("table:") and cohort_source.partition(":")[
                2
            ].endswith("population_flow"):
                templates = tuple(
                    panel.model_copy(update={"source_products": (cohort_source,)})
                    for panel in COHORT_FLOW_FIGURE_PANELS
                )
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


def apply_deterministic_figure_panels(
    plan: AnalysisPlan,
    findings: list[ValidationFinding],
) -> AnalysisPlan:
    """Bind deterministic panels and retain owner-attributable findings."""

    shaped, panel_findings = bind_deterministic_figure_panels(plan=plan)
    findings.extend(panel_findings)
    return shaped


def apply_article_figure_strategy_placements(
    *, plan: AnalysisPlan, strategy: Any
) -> AnalysisPlan:
    """Project the final article strategy onto exact planned panels.

    Panel geometry belongs to the deterministic renderer, while main versus
    supplementary placement belongs to the Planner-final article strategy.
    Compile the latter once before the plan digest is sealed so renderers do
    not infer publication hierarchy from variable names or benchmark cases.
    """

    placements = {
        str(role.role): str(role.placement)
        for role in getattr(strategy, "role_strategies", ())
    }
    # Coverage/status matrices report whether registered checks ran; they do
    # not show the direction or uncertainty of a scientific effect.  Keep
    # these audit-only grammars available in the figure suite, but never
    # promote them into the main result merely because their broad article
    # role is named ``robustness``.  A real sensitivity forest or small-
    # multiple result remains eligible for the main article through the role
    # strategy above.
    audit_only_chart_types = {
        "sensitivity_coverage_matrix",
        "status_matrix",
    }
    changed = False
    steps: list[AnalysisStep] = []
    for step in plan.steps:
        panels = []
        for panel in step.figure_panels:
            placement = placements.get(panel.article_role, panel.placement)
            if str(panel.chart_type) in audit_only_chart_types:
                placement = "supplementary"
            panels.append(panel.model_copy(update={"placement": placement}))
        if panels != step.figure_panels:
            changed = True
            step = step.model_copy(update={"figure_panels": panels})
        steps.append(step)
    return plan.model_copy(update={"steps": steps}) if changed else plan


def apply_required_plan_obligations(
    plan: AnalysisPlan,
    context: ResearchContext,
    findings: list[ValidationFinding],
) -> AnalysisPlan:
    """Close paired typed sensitivity and descriptive-context obligations."""

    shaped, sensitivity_findings = ensure_prespecified_sensitivity_steps(
        plan=plan,
        context=context,
    )
    shaped, figure_findings = ensure_descriptive_context_figure_step(plan=shaped)
    findings.extend([*sensitivity_findings, *figure_findings])
    return shaped


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
        if len(input_set) == 1 and (
            next(iter(input_set)) == COHORT_FLOW_INPUT
            or next(iter(input_set)).partition(":")[2].endswith("population_flow")
        ):
            cohort_source = next(iter(input_set))
            templates = tuple(
                panel.model_copy(update={"source_products": (cohort_source,)})
                for panel in COHORT_FLOW_FIGURE_PANELS
            )
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
                value.startswith("table:")
                and any(
                    token in value.partition(":")[2]
                    for token in (
                        "adjusted_absolute_risk",
                        "standardized_absolute_risk",
                        "standardised_absolute_risk",
                        "absolute_risk_curve",
                    )
                )
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
    "apply_article_figure_strategy_placements",
    "apply_deterministic_figure_panels",
    "apply_required_plan_obligations",
    "bind_deterministic_figure_panels",
    "close_empty_deterministic_figure_contracts",
    "dedicated_renderer_consumes_typed_source",
    "ensure_descriptive_context_figure_step",
    "ensure_cohort_accounting_figure_step",
    "ensure_data_quality_figure_step",
    "ensure_absolute_risk_association_composite_figure_step",
    "ensure_landmark_association_composite_figure_step",
    "ensure_primary_result_figure_step",
    "select_deterministic_result_renderers",
    "step_declares_audit_panel",
]
