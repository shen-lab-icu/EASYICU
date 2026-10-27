"""Pre-execution publication-readiness projections for one analysis plan.

This owner projects the context-required plan shape for planning and review:
article roles, requested results, stated adjustments/subgroups and estimators.
It also binds exact structural remediations to review-owned findings. The review
retains approval/scoring policy; these projections grant no execution authority.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..reporting.article_contract import (
    build_article_analysis_contract,
    declared_primary_lineage_step_ids,
    roles_covered_by_plan,
)
from ..schema import AnalysisPlan, ResearchContext
from ..contracts.claim_ceiling import first_result
from .analysis_types import canonical_analysis_family, infer_analysis_type, longitudinal_trajectory_requested
from .figure_strategy import (
    ArticleFigureStrategy,
    figure_panel_covers_role,
    figure_step_covers_role,
)
from .study_design import build_study_design_brief
from .progressive_module_ids import progressive_module_ids_for_analysis_types

from ..contracts.cohort_summary import DESCRIPTIVE_SUMMARY_ACTION
from ..contracts.continuous_outcome_summary import CONTINUOUS_OUTCOME_SUMMARY
from ..contracts.phenotype_comparison import (
    ASSIGNMENTS_PRODUCT, COMPARISON_ACTION, TRAJECTORY_ASSIGNMENTS_PRODUCT, TRAJECTORY_FREEZE_PRODUCT,
)
from .adjustment_authority import context_stated_adjustments, context_stated_variables, stated_adjustment_columns
from .requirement_coverage import (
    PlanShapeRemediation, RequiredPlanShape, concept_relatives, subgroup_actions_for_family,
)
from .question_requirements import EFFECT_MEASURE_NOT_REPORTED_REASON, question_effect_measures

from ..contracts.cohort_product_keys import is_closed_cohort_product_key, sole_typed_cohort_input
from ..contracts.phenotyping_features import PHENOTYPING_PRIMARY_ACTION
from ..trajectory.runtime_validation import signed_trajectory_plan_claimed


def _context_sensitivity_spec_ids(context: ResearchContext) -> set[str]:
    preferences = context.user_preferences
    return {
        str(spec.spec_id)
        for spec in (getattr(preferences, "sensitivity_specs", ()) or ())
        if str(getattr(spec, "spec_id", "") or "").strip()
    }


def _robustness_readiness(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    family: str,
    family_requirements: list[str],
    sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    required_axis_count = 0 if getattr(plan.design_selection, "execution_tier", "article") == "first_result" else min(2, len(family_requirements))
    executable_axes = sorted(set(sensitivity.get("typed_executable") or ()))
    declared_authority_ids = sorted(
        _context_sensitivity_spec_ids(context)
        | {spec.spec_id for spec in plan.robustness_specs}
    )
    available_modules = set(
        progressive_module_ids_for_analysis_types(
            (canonical_analysis_family(plan.analysis_type),)
        )
    )
    # Descriptive custom steps are reserved for registered summaries.
    planner_revision_supported = family != "descriptive" and bool(
        available_modules & {"robustness_replay", "custom_analysis"}
    )
    if required_axis_count == 0:
        status = "not_applicable"
        reason = "study_family_declares_no_robustness_requirement"
    elif len(executable_axes) >= required_axis_count:
        status = "satisfied"
        reason = "family_requirement_has_typed_executable_axes"
    elif not declared_authority_ids:
        status = "blocked"
        reason = "no_typed_sensitivity_authority"
    elif sensitivity.get("missing_spec_ids") or sensitivity.get("protocol_only"):
        status = "blocked"
        reason = "typed_sensitivity_authority_not_executable"
    else:
        status = "too_narrow"
        reason = "fewer_distinct_executable_axes_than_family_requirement"
    return {
        "status": status,
        "reason": reason,
        "family": family,
        "family_requirements": family_requirements,
        "required_axis_count": required_axis_count,
        "executable_axes": executable_axes,
        "declared_authority_ids": declared_authority_ids,
        "effect_style_grid_required": family != "descriptive" and getattr(plan.design_selection, "execution_tier", "article") == "article",
        "planner_revision_supported": planner_revision_supported,
    }


def _figure_role_facts(
    plan: AnalysisPlan,
    strategy: Optional[ArticleFigureStrategy],
) -> dict[str, Any]:
    if strategy is None:
        return {
            "required_roles": [],
            "covered_roles": [],
            "missing_roles": [],
            "figure_step_count": 0,
            "typed_panel_count": 0,
            "chart_types": [],
            "minimum_distinct_chart_types": 0,
            "distinct_chart_types_complete": True,
            "assessment_scope": "no_figure_strategy_supplied",
            "rendered_visual_qa_required_post_execution": True,
        }
    figure_steps = [
        step
        for step in plan.steps
        if any(str(value).startswith("figure:") for value in step.expected_outputs)
    ]
    required = [item for item in strategy.role_strategies if item.required]
    primary_lineage_ids = declared_primary_lineage_step_ids(plan)
    hero_requires_primary_lineage = strategy.analysis_family == "descriptive"

    def role_step_candidates(role: Any) -> list[Any]:
        if not hero_requires_primary_lineage or role.role != strategy.hero_role:
            return figure_steps
        return [step for step in figure_steps if step.step_id in primary_lineage_ids]

    covered_roles = sorted(
        {
            role.role
            for role in required
            if any(
                figure_step_covers_role(step, role)
                for step in role_step_candidates(role)
            )
        }
    )
    valid_panels = [
        panel
        for step in figure_steps
        for panel in step.figure_panels
        if any(
            figure_panel_covers_role(panel, role)
            and (
                not hero_requires_primary_lineage
                or role.role != strategy.hero_role
                or step.step_id in primary_lineage_ids
            )
            for role in required
        )
    ]
    chart_types = sorted({panel.chart_type for panel in valid_panels})
    required_roles = sorted(item.role for item in required)
    return {
        "required_roles": required_roles,
        "hero_role": strategy.hero_role,
        "hero_requires_primary_lineage": hero_requires_primary_lineage,
        "primary_lineage_step_ids": sorted(primary_lineage_ids),
        "covered_roles": covered_roles,
        "missing_roles": sorted(set(required_roles) - set(covered_roles)),
        "figure_step_count": len(figure_steps),
        "typed_panel_count": sum(len(step.figure_panels) for step in figure_steps),
        "chart_types": chart_types,
        "minimum_distinct_chart_types": strategy.minimum_distinct_chart_types,
        "chart_diversity_is_advisory": True,
        "distinct_chart_types_complete": (
            len(chart_types) >= strategy.minimum_distinct_chart_types
        ),
        "assessment_scope": "planned_roles_only_not_rendered_visual_quality",
        "rendered_visual_qa_required_post_execution": True,
    }


def build_publication_readiness_facts(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    figure_strategy: Optional[ArticleFigureStrategy],
    sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile all publication-readiness facts from their typed owners."""

    brief = build_study_design_brief(context, analysis_type=plan.analysis_type, execution_tier=getattr(plan.design_selection, "execution_tier", "article"))
    article_contract = build_article_analysis_contract(
        context,
        brief=brief,
        analysis_type=plan.analysis_type,
    )
    covered = roles_covered_by_plan(plan, article_contract)
    required = {item.article_role for item in required_plan_shape(
        context, analysis_type=plan.analysis_type, execution_tier=getattr(plan.design_selection, "execution_tier", "article"),
    ) if item.kind == "article_role"}
    return {
        "robustness": _robustness_readiness(
            context=context,
            plan=plan,
            family=brief.analysis_family,
            family_requirements=[str(value) for value in brief.sensitivity_requirements],
            sensitivity=sensitivity,
        ),
        "figure_roles": _figure_role_facts(plan, figure_strategy if getattr(plan.design_selection, "execution_tier", "article") == "article" else None),
        "content_roles": {
            "analysis_family": article_contract.analysis_family,
            "source_analysis_type": article_contract.source_analysis_type,
            "required_roles": sorted(required),
            "covered_roles": sorted(covered),
            "missing_roles": sorted(required - covered),
        },
    }


__all__ = [
    "DESCRIPTIVE_PRIMARY_SUMMARY_MISSING", "build_publication_readiness_facts", "required_plan_shape",
    "render_required_plan_shape",
]


_ARTICLE_ONLY_FINDINGS = frozenset({
    "TOP_JOURNAL_LITERATURE_SEARCH_NOT_ESTABLISHED",
    "NOVELTY_NOT_ESTABLISHED", "NOVELTY_POSITIONING_REVIEW_REQUIRED",
})


def tier_review_findings(plan, findings):
    if not first_result(plan):
        return findings
    return [finding for finding in findings if finding.code not in _ARTICLE_ONLY_FINDINGS]


def requested_outcomes(context: ResearchContext) -> tuple[str, ...]:
    """Only explicitly requested endpoints, including the primary study target."""

    return tuple(dict.fromkeys(
        str(value).strip() for value in (
            *(context.cohort.requested_outcome_columns or ()), context.target_outcome,
        ) if value and str(value).strip()
    ))


def required_plan_shape(
    context: ResearchContext, *, analysis_type: str | None = None, execution_tier: str = "article",
) -> tuple[RequiredPlanShape, ...]:
    """Only requirements of this context and selected family, without data reads."""

    family = canonical_analysis_family(analysis_type) if analysis_type else infer_analysis_type(context).key
    contract = build_article_analysis_contract(context, analysis_type=family)
    items: list[RequiredPlanShape] = []
    if execution_tier == "article":
        for role in contract.required_roles:
            modules = [item for item in contract.requirements if item.required and item.role == role]
            items.append(RequiredPlanShape(
                key=f"article:{role}", kind="article_role", article_role=role,
                module=modules[0].module_id, reason_code="ARTICLE_CONTENT_ROLES_INCOMPLETE",
            ))
    outcomes = requested_outcomes(context)
    grouped_continuous = bool(
        family == "descriptive_epidemiology" and context.primary_exposure and context.target_outcome
        and context.endpoint is not None and context.endpoint.kind == "continuous"
    )
    measured = [v.name for v in context.variables if v.role.value not in {"id", "index", "meta", "time"}]
    ungrouped = family == "descriptive_epidemiology" and not context.primary_exposure
    if grouped_continuous or (ungrouped and len(measured) == 1):
        # The scientific review owns this primary's scope
        # (PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED): one numeric
        # measurement, overall or within the declared grouping.
        items.append(RequiredPlanShape(
            key="primary_summary", kind="primary_summary", planned_analysis_role="primary",
            action=DESCRIPTIVE_SUMMARY_ACTION, module="custom_analysis",
            inputs=(context.target_outcome, context.primary_exposure) if grouped_continuous else tuple(measured),
            reason_code=DESCRIPTIVE_PRIMARY_SUMMARY_MISSING,
        ))
    elif ungrouped and len(measured) > 1:
        # One primary summary takes one measurement; a question about whether,
        # how often or when several were recorded is a primary measurement audit.
        items.append(RequiredPlanShape(
            key="primary_audit", kind="measurement_audit", planned_analysis_role="primary",
            module="measurement_audit", inputs=tuple(measured), reason_code=PRIMARY_SUMMARY_SCOPE_UNRESOLVED,
        ))
    cluster = family == "trajectory_clustering"
    if outcomes:
        action = COMPARISON_ACTION if cluster else None
        module = (CONTINUOUS_OUTCOME_SUMMARY if grouped_continuous else
                  "exposure_outcome_distribution" if family == "descriptive_epidemiology" else
                  "adjusted_association" if family in {"association_study", "ordinal_dose_response"} else "custom_analysis")
        role = "secondary" if cluster or grouped_continuous else "primary"
        inputs = tuple(dict.fromkeys((
            *((context.primary_exposure,) if grouped_continuous else ()), *outcomes,
        )))
        if not (cluster and longitudinal_trajectory_requested(context)):
            items.append(RequiredPlanShape(
                key="requested_outcomes", kind="outcome", action=action, module=module,
                planned_analysis_role=role, inputs=inputs, reason_code="REQUESTED_OUTCOME_COVERAGE_INCOMPLETE",
            ))
        if cluster:
            items.append(RequiredPlanShape(
                key="cluster_outcomes", kind="comparison", action=COMPARISON_ACTION,
                module="custom_analysis", planned_analysis_role="secondary", inputs=outcomes,
                reason_code="PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE",
            ))
    for adjustment in context_stated_adjustments(context):
        if adjustment.source_status == "available" and adjustment.concept:
            items.append(RequiredPlanShape(
                key=f"adjustment:{adjustment.concept}", kind="adjustment", planned_analysis_role="primary",
                module="adjusted_association",
                inputs=stated_adjustment_columns(context, adjustment), reason_code="STATED_ADJUSTMENT_MISSING",
            ))
    for subgroup in context_stated_variables(context, "stated_subgroups"):
        if subgroup.source_status != "available" or not subgroup.concept:
            continue
        actions = sorted(subgroup_actions_for_family(family))
        columns = tuple(v.name for v in context.variables if v.name in concept_relatives(context)(subgroup.concept)
                        and v.role.value not in {"id", "index", "meta", "time"})
        items.append(RequiredPlanShape(
            key=f"subgroup:{subgroup.concept}", kind="subgroup", planned_analysis_role="secondary",
            action=actions[0] if actions else None,
            module="custom_analysis" if actions else (CONTINUOUS_OUTCOME_SUMMARY if context.endpoint and context.endpoint.kind == "continuous" else "exposure_outcome_distribution"),
            inputs=columns,
            reason_code="STATED_SUBGROUP_MISSING" if actions or family == "descriptive_epidemiology" else "STATED_SUBGROUP_UNSUPPORTED",
        ))
    if family in {"association_study", "ordinal_dose_response"}:
        for measure in question_effect_measures(context.research_question):
            items.append(RequiredPlanShape(
                key=f"effect_measure:{measure.measure}", kind="effect_measure", planned_analysis_role="primary",
                module="adjusted_association",
                effect_measure=measure.measure,
                inputs=tuple(value for value in (context.primary_exposure, context.target_outcome) if value),
                reason_code=EFFECT_MEASURE_NOT_REPORTED_REASON,
            ))
    return tuple(items)


#: Raised by the scientific review when the context requires a primary
#: descriptive summary and no step is one; its scope, when a step is one, is
#: the review's PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED.
DESCRIPTIVE_PRIMARY_SUMMARY_MISSING = "DESCRIPTIVE_PRIMARY_SUMMARY_MISSING"
PRIMARY_SUMMARY_SCOPE_UNRESOLVED = "PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED"


def _repairs_finding(item: RequiredPlanShape, code: str) -> bool:
    """A primary-summary item repairs both its absence and the review's scope blocker."""

    return code == item.reason_code or (
        item.kind == "primary_summary" and code == PRIMARY_SUMMARY_SCOPE_UNRESOLVED
    )


def render_required_plan_shape(context: ResearchContext, **kwargs) -> str:
    """Short checklist; no truncation of context-fixed input identities."""

    items = required_plan_shape(context, **kwargs)
    lines = ["Required plan shape (host):"]
    roles = [item.article_role for item in items if item.kind == "article_role"]
    if roles:
        lines.append("- article roles: " + ", ".join(roles))
    for item in items:
        if item.kind == "article_role":
            continue
        owner = item.action or item.module or item.kind
        detail = item.effect_measure or ",".join(item.inputs)
        if item.kind == "primary_summary":
            detail = f"{item.inputs[0]} only, overall or within {item.inputs[1]}" if len(item.inputs) > 1 else f"{detail} only"
        elif item.kind == "measurement_audit":
            detail += " if asked whether/how often/when recorded; one summary takes one measurement"
        lines.append(f"- {item.planned_analysis_role or 'analysis'} {owner}: {detail} [{item.reason_code}]")
    if any(item.kind == "comparison" for item in items) and longitudinal_trajectory_requested(context):
        lines.append("- compare only frozen trajectory classes from the stability owner.")
    return "\n".join(lines)


def _remediations(
    context: ResearchContext, plan: AnalysisPlan, items: tuple[RequiredPlanShape, ...],
    code: str, missing_outcomes: tuple[str, ...], missing_roles: list[str],
) -> tuple[PlanShapeRemediation, ...]:
    repairs = []
    cohorts = tuple(dict.fromkeys(
        key for step in plan.steps for key in (*step.inputs, *step.expected_outputs)
        if is_closed_cohort_product_key(key)
    ))
    primary = next((step for step in plan.steps if step.planned_analysis_role == "primary"), None)
    for item in items:
        if not _repairs_finding(item, code) or (item.kind == "article_role" and item.article_role not in missing_roles):
            continue
        action, module, role, inputs, source = item.action, item.module, item.planned_analysis_role, item.inputs, None
        if item.kind == "primary_summary":
            module, source = None, primary
            inputs = (*inputs, *cohorts)
        elif item.kind == "measurement_audit":
            # Replaces the several-measurement primary summary the review blocked.
            source = next((step for step in plan.steps if step.planned_analysis_role == "primary"
                           and step.scientific_action_id == DESCRIPTIVE_SUMMARY_ACTION), None)
            if source is None:
                continue
            inputs = (*inputs, *cohorts)
        elif item.kind == "comparison" or (item.kind == "outcome" and action == COMPARISON_ACTION):
            source = next((step for step in plan.steps if step.scientific_action_id == COMPARISON_ACTION), None)
            clustering = next((step for step in plan.steps if step.scientific_action_id == PHENOTYPING_PRIMARY_ACTION), None)
            cohort = sole_typed_cohort_input(clustering) if clustering else None
            labels = (TRAJECTORY_ASSIGNMENTS_PRODUCT, TRAJECTORY_FREEZE_PRODUCT) if signed_trajectory_plan_claimed(plan) else (ASSIGNMENTS_PRODUCT,)
            features = tuple(clustering.phenotyping_feature_columns or ()) if clustering is not None else ()
            inputs = (*missing_outcomes, *features, *context.cohort.id_columns, *((cohort,) if cohort else cohorts), *labels)
            module = None
        elif item.kind == "outcome" and module == "continuous_outcome_summary":
            inputs = (context.primary_exposure, *missing_outcomes, *cohorts)
        elif item.kind in {"outcome", "adjustment"} and primary is not None and primary.model_requirements:
            action, module = primary.scientific_action_id, primary.method
            source = primary if item.kind == "adjustment" or context.target_outcome in missing_outcomes else None
            role = "primary" if source else "secondary"
            fixed = tuple(value for requirement in items if requirement.kind == "adjustment" for value in requirement.inputs)
            inputs = (*primary.inputs, *inputs, *fixed)
        elif item.kind == "subgroup" and action is not None and primary is not None:
            inputs = (*primary.inputs, *inputs)
        elif item.kind == "article_role" and item.article_role in {"cohort_accounting", "data_quality"}:
            module, role = "measurement_audit", "auxiliary"
            inputs = tuple(dict.fromkeys(value for step in plan.steps for value in step.inputs))
        else:
            # Article role ids and unresolved capabilities are evidence
            # obligations, never permission to invent an execution module.
            continue
        if role is None or not inputs:
            continue
        repairs.append(PlanShapeRemediation(
            operation="replace_module" if source else "add_step",
            step_id=source.step_id if source else None, action=action, module=module,
            planned_analysis_role=role, inputs=tuple(dict.fromkeys(inputs)),
        ))
    return tuple({repair.model_dump_json(): repair for repair in repairs}.values())


def bind_required_plan_shape(*, context: ResearchContext, plan: AnalysisPlan, findings, missing_outcomes, missing_roles):
    """Read the same requirements as both prompt routes; attach host scope only.

    The scientific review owns every finding; this projection only binds the
    exact structural remediation each finding admits.
    """

    items = required_plan_shape(
        context, analysis_type=plan.analysis_type,
        execution_tier=getattr(plan.design_selection, "execution_tier", "article"),
    )
    bound = []
    for finding in findings:
        if finding.requires_user_authorization or finding.remediation_route not in {"unclassified", "agent_plan_revision"}:
            bound.append(finding)
            continue
        repairs = _remediations(context, plan, items, finding.code, missing_outcomes, missing_roles)
        bound.append(finding.model_copy(update={"structural_remediations": repairs}) if repairs else finding)
    return bound, items
