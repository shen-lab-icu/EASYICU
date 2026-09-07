"""Pre-execution publication-readiness projections for one analysis plan.

This owner joins three already-typed planning contracts: article content roles,
figure panel roles/chart grammar, and family-appropriate sensitivity authority.
It does not assign findings or scores; ``scientific_review`` owns that policy.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..reporting.article_contract import (
    build_article_analysis_contract,
    declared_primary_lineage_step_ids,
    roles_covered_by_plan,
)
from ..schema import AnalysisPlan, ResearchContext
from .figure_strategy import (
    ArticleFigureStrategy,
    figure_panel_covers_role,
    figure_step_covers_role,
)
from .study_design import build_study_design_brief


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
    required_axis_count = min(2, len(family_requirements))
    executable_axes = sorted(set(sensitivity.get("typed_executable") or ()))
    declared_authority_ids = sorted(
        _context_sensitivity_spec_ids(context)
        | {spec.spec_id for spec in plan.robustness_specs}
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
        "effect_style_grid_required": family != "descriptive",
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

    brief = build_study_design_brief(context, analysis_type=plan.analysis_type)
    article_contract = build_article_analysis_contract(
        context,
        brief=brief,
        analysis_type=plan.analysis_type,
    )
    covered = roles_covered_by_plan(plan, article_contract)
    required = set(article_contract.required_roles)
    return {
        "robustness": _robustness_readiness(
            context=context,
            plan=plan,
            family=brief.analysis_family,
            family_requirements=[str(value) for value in brief.sensitivity_requirements],
            sensitivity=sensitivity,
        ),
        "figure_roles": _figure_role_facts(plan, figure_strategy),
        "content_roles": {
            "analysis_family": article_contract.analysis_family,
            "source_analysis_type": article_contract.source_analysis_type,
            "required_roles": sorted(required),
            "covered_roles": sorted(covered),
            "missing_roles": sorted(required - covered),
        },
    }


__all__ = ["build_publication_readiness_facts"]
