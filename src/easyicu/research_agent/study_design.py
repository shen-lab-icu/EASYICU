"""Study-design brief generation and plan coverage checks.

This module sits above the planner. It translates a free-form scientific
question into an article-design contract: expected analysis family, methods,
main-text displays, and supplementary materials. It is intentionally
case-neutral; concrete benchmark variables belong in the ResearchContext or
case protocol, not in these templates.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .schema import AnalysisPlan, ResearchContext, ValidationFinding
from .study_design_playbook import (
    DisplayModuleSpec,
    DisplayTier,
    StudyDesignFamily,
    anti_patterns_for_brief,
    brief_check_terms,
    design_principles_for_family,
    display_modules_for_family,
    family_template,
    role_check_terms,
    triggered_generic_modules,
)


STUDY_DESIGN_BRIEF_SCHEMA_VERSION = "easyicu.study_design_brief/2"

# The analysis-type scorer (``infer_analysis_type``) is richer than the keyword
# cascade in ``infer_study_design_family`` and is the AUTHORITATIVE signal: it
# stamps ``plan.analysis_type`` and therefore drives the plan contract. When it
# confidently detects one of these strong, result-bearing families, the design
# family must agree, otherwise the plan builds (say) a survival step while the
# figure renderer and the methodological-rigor auditor -- both keyed on
# ``infer_study_design_family`` -- route to "association" and never fire the
# survival figure / method-match check. This map is deliberately UPGRADE-ONLY:
# Include the ordinary association/descriptive paths too so disclaimer prose
# ("do not make a causal claim") cannot override the analysis-type result in a
# second, independent keyword cascade.
_ANALYSIS_TYPE_TO_DESIGN_FAMILY: dict[str, StudyDesignFamily] = {
    "survival": "time_to_event",
    "prediction_model": "prediction",
    "dynamic_prediction": "prediction",
    "validation": "prediction",
    "trajectory_clustering": "phenotyping",
    "causal_inference": "causal_emulation",
    "treatment_response": "causal_emulation",
    "association_study": "association",
    "descriptive_epidemiology": "descriptive",
    "data_quality_audit": "descriptive",
    "measurement_bias_audit": "descriptive",
    "cohort_definition_sensitivity": "descriptive",
    "score_policy_sensitivity": "descriptive",
}


class StudyDesignBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = STUDY_DESIGN_BRIEF_SCHEMA_VERSION
    analysis_family: StudyDesignFamily
    rationale: str
    reporting_guidelines: List[str] = Field(default_factory=list)
    design_principles: List[str] = Field(default_factory=list)
    required_methods: List[str] = Field(default_factory=list)
    main_text_displays: List[str] = Field(default_factory=list)
    supplementary_displays: List[str] = Field(default_factory=list)
    display_modules: List[DisplayModuleSpec] = Field(default_factory=list)
    sensitivity_requirements: List[str] = Field(default_factory=list)
    covariate_strategy: str = ""
    variable_role_requirements: List[str] = Field(default_factory=list)
    adaptive_triggers: List[str] = Field(default_factory=list)
    anti_patterns: List[str] = Field(default_factory=list)
    exemplar_search_queries: List[str] = Field(default_factory=list)
    planner_instructions: List[str] = Field(default_factory=list)


def infer_study_design_family(context: ResearchContext) -> StudyDesignFamily:
    preference = ""
    if context.user_preferences and context.user_preferences.inferred_analysis_family:
        preference = context.user_preferences.inferred_analysis_family.lower()
    preferred_methods = ""
    if context.user_preferences and context.user_preferences.preferred_methods:
        preferred_methods = str(context.user_preferences.preferred_methods)
    text = " ".join(
        [
            preference,
            context.research_question or "",
            context.target_outcome or "",
            context.primary_exposure or "",
            preferred_methods,
        ]
    ).lower()
    # Defer to the authoritative analysis-type scorer for the strong,
    # result-bearing families (see _ANALYSIS_TYPE_TO_DESIGN_FAMILY) so the design
    # family stays consistent with the plan contract. Best-effort: any failure
    # falls through to the keyword cascade -- this classifier must never raise
    # into figure rendering or the rigor audit.
    try:
        from .analysis_types import (
            infer_analysis_type,
            strong_trajectory_clustering_framing,
        )

        strong_family = _ANALYSIS_TYPE_TO_DESIGN_FAMILY.get(
            infer_analysis_type(context).key
        )
        clustering_framed = strong_trajectory_clustering_framing(text)
    except Exception:
        strong_family = None
        clustering_framed = False
    if strong_family:
        return strong_family
    if any(
        token in text
        for token in (
            "target trial",
            "causal",
            "treatment effect",
            "iptw",
            "propensity",
            "indication bias",
            "confounding",
            "因果",
            "治疗效应",
            "倾向",
            "指征偏倚",
            "混杂",
        )
    ):
        return "causal_emulation"
    if any(
        token in text
        for token in (
            "survival",
            "time-to-event",
            "time to event",
            "hazard",
            "cox",
            "生存",
            "时间到事件",
            "风险比",
        )
    ):
        return "time_to_event"
    if any(
        token in text
        for token in (
            "predict",
            "prediction",
            "auroc",
            "calibration",
            "risk score",
            "external validation",
            "预测",
            "校准",
            "判别",
            "泛化",
            "迁移",
        )
    ):
        return "prediction"
    if clustering_framed:
        return "phenotyping"
    if any(
        token in text
        for token in (
            "association",
            "associated",
            "dose-response",
            "dose response",
            "odds ratio",
            "risk ratio",
            "adjust",
            "adjusted",
            "关联",
            "相关",
            "比值比",
            "风险比",
            "校正",
            "调整",
        )
    ):
        return "association"
    if any(
        token in text
        for token in (
            "describe",
            "descriptive",
            "distribution",
            "prevalence",
            "coverage",
            "denominator",
            "描述",
            "分布",
            "患病率",
            "覆盖率",
            "denominator",
        )
    ):
        return "descriptive"
    return "descriptive"


def _context_text(context: ResearchContext) -> str:
    parts = [
        context.research_question or "",
        context.target_outcome or "",
        context.primary_exposure or "",
        context.notes or "",
    ]
    if context.user_preferences is not None:
        parts.extend(
            [
                context.user_preferences.inferred_analysis_family or "",
                context.user_preferences.preferred_methods or "",
                context.user_preferences.evaluation_focus or "",
                context.user_preferences.must_have_outputs or "",
                context.user_preferences.extra_notes or "",
            ]
        )
    return " ".join(parts).lower()


def _adaptive_triggers_for_context(
    context: ResearchContext,
    family: StudyDesignFamily,
) -> List[str]:
    text = _context_text(context)
    triggers: List[str] = []
    if context.cross_database_validation or any(
        token in text
        for token in (
            "cross database",
            "cross-database",
            "multiple database",
            "multi-database",
            "external validation",
            "transportability",
            "generalization",
            "generalisation",
            "跨数据库",
            "多数据库",
            "多库",
            "六库",
            "六个 icu 数据库",
            "外部验证",
            "泛化",
            "迁移",
        )
    ):
        triggers.append(
            "cross_database_or_site_heterogeneity: add source-level coverage, "
            "site/database-specific estimates, or transportability displays."
        )
    if "missing" in text or "missingness" in text or any(
        v.missingness
        and v.missingness.missingness_severity in {"medium", "high", "unknown"}
        for v in context.variables
    ):
        triggers.append(
            "nontrivial_missingness_or_measurement_process: show availability, "
            "measurement frequency, and missing-data sensitivity explicitly."
        )
    if any(
        token in text
        for token in (
            "prevalence",
            "incidence",
            "event rate",
            "mortality rate",
            "患病率",
            "发生率",
            "事件率",
            "死亡率",
        )
    ):
        triggers.append(
            "prevalence_or_event_rate_question: show exposure prevalence, "
            "outcome incidence, or event rates before adjusted modelling."
        )
    if family == "prediction" and any(
        token in text
        for token in (
            "threshold",
            "triage",
            "deployment",
            "net benefit",
            "decision curve",
            "阈值",
            "分诊",
            "部署",
            "净获益",
            "决策曲线",
        )
    ):
        triggers.append(
            "clinically_actionable_prediction_thresholds: include decision-curve, "
            "net-benefit, or threshold-utility displays."
        )
    if family == "time_to_event" or any(
        token in text
        for token in (
            "follow-up",
            "time zero",
            "time-zero",
            "censor",
            "随访",
            "时间零点",
            "删失",
        )
    ):
        triggers.append(
            "longitudinal_time_origin: make time zero, censoring, follow-up, and "
            "risk-set accounting explicit."
        )
    if family == "causal_emulation":
        triggers.append(
            "causal_claim_or_treatment_comparison: require target-trial protocol, "
            "pre-time-zero confounders, balance, positivity, and sensitivity."
        )
    return triggers


def _display_modules_for_context(
    context: ResearchContext,
    family: StudyDesignFamily,
    triggers: Sequence[str],
) -> List[DisplayModuleSpec]:
    modules = display_modules_for_family(family)
    existing_ids = {module.module_id for module in modules}
    for module in triggered_generic_modules(triggers):
        if module.module_id not in existing_ids:
            modules.append(module)
            existing_ids.add(module.module_id)
    return modules


def build_study_design_brief(context: ResearchContext) -> StudyDesignBrief:
    family = infer_study_design_family(context)
    template = family_template(family)
    question = (context.research_question or "").strip()
    exemplar_query = question if question else family.replace("_", " ")
    adaptive_triggers = _adaptive_triggers_for_context(context, family)
    display_modules = _display_modules_for_context(context, family, adaptive_triggers)
    planner_instructions = [
        "Select methods and displays from the study-design brief before writing executable steps.",
        "If the plan omits a required main-text display or sensitivity analysis, state why in the plan rationale.",
        "Keep case-specific variables in step inputs/expected outputs; keep global prompts case-neutral.",
        "Declare article-facing outputs separately from diagnostic-only artifacts.",
        "Treat the display playbook as flexible modules, not a rigid checklist; justify substitutions by evidence role.",
        "Before defaulting to a forest plot, ask whether the question needs cohort, missingness, calibration, survival, phenotype, causal, or transportability displays.",
    ]
    return StudyDesignBrief(
        analysis_family=family,
        rationale=f"Detected {family} design family from the research question and context.",
        reporting_guidelines=list(template["reporting_guidelines"]),
        design_principles=design_principles_for_family(family),
        required_methods=list(template["required_methods"]),
        main_text_displays=list(template["main_text_displays"]),
        supplementary_displays=list(template["supplementary_displays"]),
        display_modules=display_modules,
        sensitivity_requirements=list(template["sensitivity_requirements"]),
        covariate_strategy=str(template["covariate_strategy"]),
        variable_role_requirements=[
            "Classify each analysis variable by role and timing before modelling.",
            "Separate main estimand variables from adjustment covariates, diagnostics, and supplementary descriptors.",
        ],
        adaptive_triggers=adaptive_triggers,
        anti_patterns=anti_patterns_for_brief(),
        exemplar_search_queries=[
            f"{exemplar_query} study design supplementary material",
            f"{exemplar_query} observational ICU analysis figures tables",
            f"{exemplar_query} reporting guideline analysis plan",
        ],
        planner_instructions=planner_instructions,
    )


def render_study_design_brief_for_prompt(brief: StudyDesignBrief) -> str:
    module_lines = [
        (
            f"  - [{module.tier}] {module.module_id} "
            f"(role={module.role}; acceptable={', '.join(module.acceptable_outputs[:4])})"
        )
        for module in brief.display_modules
    ]
    lines = [
        "STUDY DESIGN BRIEF:",
        f"- analysis_family: {brief.analysis_family}",
        "- reporting_guidelines: " + "; ".join(brief.reporting_guidelines),
        "- design_principles: " + "; ".join(brief.design_principles),
        "- required_methods: " + "; ".join(brief.required_methods),
        "- main_text_displays: " + "; ".join(brief.main_text_displays),
        "- supplementary_displays: " + "; ".join(brief.supplementary_displays),
        "- display_playbook:",
        *module_lines,
        "- sensitivity_requirements: " + "; ".join(brief.sensitivity_requirements),
        "- adaptive_triggers: " + ("; ".join(brief.adaptive_triggers) or "none detected"),
        "- anti_patterns: " + "; ".join(brief.anti_patterns),
        f"- covariate_strategy: {brief.covariate_strategy}",
        "- planner_instructions: " + "; ".join(brief.planner_instructions),
    ]
    return "\n".join(lines)


def _plan_blob(plan: AnalysisPlan) -> str:
    """Return only structured declarations used for coverage validation.

    Research-question, rationale, step-id, and intent prose are excluded so a
    plan cannot satisfy a method/display requirement merely by repeating it.
    """

    parts: List[str] = [plan.analysis_type or ""]
    for step in plan.steps or []:
        parts.extend(
            [
                step.method or "",
                " ".join(step.expected_outputs or []),
            ]
        )
    return "\n".join(parts).lower()


def _brief_item_covered(item: str, blob: str) -> bool:
    terms = brief_check_terms(item)
    if terms is None:
        terms = tuple(re.split(r"[/, ]+", item.lower()))
    return any(term and term.lower() in blob for term in terms)


def _module_covered(module: DisplayModuleSpec, blob: str) -> bool:
    terms: List[str] = [
        module.module_id,
        module.module_id.replace("_", " "),
        module.role,
        module.role.replace("_", " "),
        *module.acceptable_outputs,
        *role_check_terms(module.role),
    ]
    return any(term and term.lower() in blob for term in terms)


def _covered_modules(
    plan: AnalysisPlan,
    brief: StudyDesignBrief,
) -> List[DisplayModuleSpec]:
    blob = _plan_blob(plan)
    return [
        module
        for module in brief.display_modules
        if module.tier != "supplementary" and _module_covered(module, blob)
    ]


def validate_plan_against_study_design_brief(
    *,
    plan: AnalysisPlan,
    brief: StudyDesignBrief,
) -> List[ValidationFinding]:
    blob = _plan_blob(plan)
    core_modules = [module for module in brief.display_modules if module.tier == "core"]
    conditional_modules = [
        module for module in brief.display_modules if module.tier == "conditional"
    ]
    missing_core_modules = [
        module
        for module in core_modules
        if not _module_covered(module, blob)
    ]
    missing_conditional_modules = [
        module
        for module in conditional_modules
        if not _module_covered(module, blob)
    ]
    covered_modules = _covered_modules(plan, brief)
    covered_roles = sorted({module.role for module in covered_modules})
    required_core_roles = sorted({module.role for module in core_modules})
    missing_main = [
        item for item in brief.main_text_displays if not _brief_item_covered(item, blob)
    ]
    missing_methods = [
        item for item in brief.required_methods if not _brief_item_covered(item, blob)
    ]
    missing_sensitivity = [
        item
        for item in brief.sensitivity_requirements
        if not _brief_item_covered(item, blob)
    ]
    findings: List[ValidationFinding] = []
    if missing_core_modules:
        findings.append(
            ValidationFinding(
                validator="study_design_brief",
                severity="warning",
                message=(
                    "Analysis plan does not cover all core article-display modules "
                    f"for {brief.analysis_family} studies."
                ),
                detail={
                    "missing_core_display_modules": [
                        {
                            "module_id": module.module_id,
                            "role": module.role,
                            "acceptable_outputs": module.acceptable_outputs,
                        }
                        for module in missing_core_modules
                    ],
                    "covered_display_roles": covered_roles,
                },
            )
        )
    if missing_conditional_modules:
        findings.append(
            ValidationFinding(
                validator="study_design_brief",
                severity="warning",
                message=(
                    "Analysis plan may be missing conditionally triggered "
                    f"article-display modules for {brief.analysis_family} studies."
                ),
                detail={
                    "missing_conditional_display_modules": [
                        {
                            "module_id": module.module_id,
                            "role": module.role,
                            "acceptable_outputs": module.acceptable_outputs,
                            "triggers": module.triggers,
                        }
                        for module in missing_conditional_modules
                    ],
                    "adaptive_triggers": brief.adaptive_triggers,
                    "covered_display_roles": covered_roles,
                },
            )
        )
    min_roles = min(3, len(required_core_roles))
    if len(covered_roles) < min_roles:
        findings.append(
            ValidationFinding(
                validator="study_design_brief",
                severity="warning",
                message=(
                    "Analysis plan is too narrow for an article-level study design; "
                    "it should combine design/accounting, data-quality, primary-result, "
                    "and robustness or family-specific displays."
                ),
                detail={
                    "covered_display_roles": covered_roles,
                    "expected_core_roles": required_core_roles,
                },
            )
        )
    if missing_main:
        findings.append(
            ValidationFinding(
                validator="study_design_brief",
                severity="warning",
                message=(
                    "Analysis plan does not declare all main-text displays expected "
                    f"for {brief.analysis_family} studies."
                ),
                detail={"missing_main_text_displays": missing_main},
            )
        )
    if missing_methods:
        findings.append(
            ValidationFinding(
                validator="study_design_brief",
                severity="warning",
                message=(
                    "Analysis plan may be missing expected method components for "
                    f"{brief.analysis_family} studies."
                ),
                detail={"missing_methods": missing_methods},
            )
        )
    if missing_sensitivity:
        findings.append(
            ValidationFinding(
                validator="study_design_brief",
                severity="warning",
                message=(
                    "Analysis plan may be missing expected sensitivity requirements "
                    f"for {brief.analysis_family} studies."
                ),
                detail={"missing_sensitivity_requirements": missing_sensitivity},
            )
        )
    return findings


__all__ = [
    "DisplayModuleSpec",
    "DisplayTier",
    "StudyDesignBrief",
    "build_study_design_brief",
    "infer_study_design_family",
    "render_study_design_brief_for_prompt",
    "validate_plan_against_study_design_brief",
]
