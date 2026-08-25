"""Pre-plan article analysis blueprint.

The study-design brief, article-analysis contract, and figure strategy are
useful on their own, but planners need one coherent upstream object: what kind
of scientific question is this, what article structure should it emulate, what
main and supplementary displays are expected, and which visual anti-patterns
must be avoided.

This module builds that bridge without hard-coding a benchmark case. The
``prior_art_design_brief`` is a design scaffold: it states what to learn from
top-journal articles and supplements, plus search queries for an optional
literature/scouting layer. It does not claim that a live search was performed
and it never imports case-specific variables, scores, or databases into global
rules.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..reporting.article_contract import (
    ArticleAnalysisContract,
    ArticleDisplayRequirement,
    build_article_analysis_contract,
)
from .figure_strategy import (
    ArticleFigureStrategy,
    FigureRoleStrategy,
    build_article_figure_strategy,
    figure_step_covers_role,
)
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding
from .study_design import StudyDesignBrief, build_study_design_brief
from .study_design_playbook import DisplayTier, StudyDesignFamily

ANALYSIS_BLUEPRINT_SCHEMA_VERSION = "easyicu.analysis_blueprint/1"
PRIOR_ART_DESIGN_BRIEF_SCHEMA_VERSION = "easyicu.prior_art_design_brief/1"


class PriorArtDesignBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PRIOR_ART_DESIGN_BRIEF_SCHEMA_VERSION
    analysis_family: StudyDesignFamily
    source_mode: str = "deterministic_family_playbook"
    purpose: str
    design_reference_queries: List[str] = Field(default_factory=list)
    design_questions: List[str] = Field(default_factory=list)
    main_text_pattern: List[str] = Field(default_factory=list)
    supplement_pattern: List[str] = Field(default_factory=list)
    extraction_guardrails: List[str] = Field(default_factory=list)


class BlueprintArticleRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    module_id: str
    tier: DisplayTier
    required: bool = True
    rationale: str = ""
    acceptable_outputs: List[str] = Field(default_factory=list)
    search_terms: List[str] = Field(default_factory=list)


class BlueprintVisualRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    required: bool = True
    placement: str = "main"
    rationale: str = ""
    acceptable_chart_types: List[str] = Field(default_factory=list)
    required_text_terms: List[str] = Field(default_factory=list)
    search_terms: List[str] = Field(default_factory=list)


class AnalysisBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ANALYSIS_BLUEPRINT_SCHEMA_VERSION
    analysis_family: StudyDesignFamily
    question_classification: str
    prior_art_design_brief: PriorArtDesignBrief
    reporting_guidelines: List[str] = Field(default_factory=list)
    design_principles: List[str] = Field(default_factory=list)
    required_methods: List[str] = Field(default_factory=list)
    required_article_roles: List[str] = Field(default_factory=list)
    recommended_article_roles: List[str] = Field(default_factory=list)
    article_roles: List[BlueprintArticleRole] = Field(default_factory=list)
    main_text_display_roles: List[str] = Field(default_factory=list)
    supplementary_display_roles: List[str] = Field(default_factory=list)
    sensitivity_requirements: List[str] = Field(default_factory=list)
    figure_archetype: str
    figure_hero_role: str
    minimum_distinct_chart_types: int = 2
    visual_roles: List[BlueprintVisualRole] = Field(default_factory=list)
    anti_patterns: List[str] = Field(default_factory=list)
    planner_sequence: List[str] = Field(default_factory=list)
    validation_gates: List[str] = Field(default_factory=list)
    source_brief_schema_version: str = ""
    source_contract_schema_version: str = ""
    source_figure_strategy_schema_version: str = ""


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _dedupe(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _prior_art_design_questions(family: StudyDesignFamily) -> List[str]:
    base = [
        "Which tables and figures appear in the main text before the primary result?",
        "Which diagnostics, definitions, sensitivity analyses, and code/data provenance move to supplementary material?",
        "Which display carries the article's reader-facing hero role, and which panels provide audit context?",
        "Which uncertainty, denominator, missingness, and measurement-process details are visible to readers?",
    ]
    family_questions = {
        "association": [
            "How do high-quality observational papers show absolute risk before adjusted relative effects?",
            "How are covariate timing, confounding rationale, and missing-data sensitivity reported?",
        ],
        "prediction": [
            "How are discrimination, calibration, validation split, and threshold utility distributed across main and supplementary figures?",
            "How do supplements report preprocessing, feature leakage checks, and hyperparameters?",
        ],
        "time_to_event": [
            "How are time zero, censoring, risk tables, and adjusted survival contrasts shown together?",
            "Which assumption checks and competing-risk or censoring sensitivities appear in supplements?",
        ],
        "phenotyping": [
            "How are feature provenance, embedding/cluster structure, phenotype profiles, and stability separated?",
            "How are downstream outcomes labelled without implying causal phenotype discovery?",
        ],
        "causal_emulation": [
            "How is the target-trial protocol shown before the causal contrast?",
            "How are balance, positivity, trimming, and unmeasured-confounding sensitivity reported?",
        ],
        "descriptive": [
            "How are denominators, distributions, stratified summaries, and data coverage shown without forcing a model?",
            "Which descriptive tables are main-text anchors versus extended tables?",
        ],
    }
    return [*base, *family_questions.get(family, [])]


def _build_prior_art_design_brief(
    *,
    brief: StudyDesignBrief,
    contract: ArticleAnalysisContract,
    figure_strategy: ArticleFigureStrategy,
) -> PriorArtDesignBrief:
    queries = [
        *brief.exemplar_search_queries,
        *contract.design_reference_queries,
    ]
    for role in figure_strategy.role_strategies:
        if role.required:
            query = " ".join(
                [brief.analysis_family, role.role, "main figure supplement"]
            )
            queries.append(query)
    return PriorArtDesignBrief(
        analysis_family=brief.analysis_family,
        purpose=(
            "Extract article-structure patterns from high-impact papers and "
            "supplementary materials before planning executable steps."
        ),
        design_reference_queries=_dedupe(queries),
        design_questions=_prior_art_design_questions(brief.analysis_family),
        main_text_pattern=list(brief.main_text_displays),
        supplement_pattern=list(brief.supplementary_displays),
        extraction_guardrails=[
            "Extract study-design and display structure only; do not copy claims, numerical results, or causal language.",
            "Keep case-specific variables in the run context or benchmark item, not in global prompts.",
            "Treat examples as non-binding precedents; adapt by evidence role and available data.",
        ],
    )


def _article_role(requirement: ArticleDisplayRequirement) -> BlueprintArticleRole:
    return BlueprintArticleRole(
        role=requirement.role,
        module_id=requirement.module_id,
        tier=requirement.tier,
        required=requirement.required,
        rationale=requirement.rationale,
        acceptable_outputs=list(requirement.acceptable_outputs),
        search_terms=list(requirement.search_terms),
    )


def _visual_role(role: FigureRoleStrategy) -> BlueprintVisualRole:
    return BlueprintVisualRole(
        role=role.role,
        required=role.required,
        placement=role.placement,
        rationale=role.rationale,
        acceptable_chart_types=list(role.acceptable_chart_types),
        required_text_terms=list(role.required_text_terms),
        search_terms=list(role.search_terms),
    )


def build_analysis_blueprint(
    context: ResearchContext,
    *,
    brief: Optional[StudyDesignBrief] = None,
    contract: Optional[ArticleAnalysisContract] = None,
    figure_strategy: Optional[ArticleFigureStrategy] = None,
) -> AnalysisBlueprint:
    resolved_brief = brief or build_study_design_brief(context)
    resolved_contract = contract or build_article_analysis_contract(
        context,
        brief=resolved_brief,
    )
    resolved_strategy = figure_strategy or build_article_figure_strategy(context)
    prior_art = _build_prior_art_design_brief(
        brief=resolved_brief,
        contract=resolved_contract,
        figure_strategy=resolved_strategy,
    )
    article_roles = [_article_role(req) for req in resolved_contract.requirements]
    visual_roles = [_visual_role(role) for role in resolved_strategy.role_strategies]
    anti_patterns = _dedupe(
        [
            *resolved_brief.anti_patterns,
            *resolved_contract.anti_patterns,
            *resolved_strategy.anti_patterns,
        ]
    )
    planner_sequence = [
        "Classify the scientific question and estimand before choosing methods.",
        "Review the prior-art design brief for article and supplement structure.",
        "Plan cohort/denominator accounting before primary modelling.",
        "Plan required data-quality and missingness displays before interpreting results, placing routine audit detail in supplementary material.",
        "Plan the primary analysis and uncertainty definition.",
        "Plan robustness or sensitivity modules required by the study family.",
        "Plan a coherent article-level display package, usually 2-4 complementary main figures plus 2-3 main tables, with detailed diagnostics and routine missingness in supplementary material; treat these counts as planning targets rather than fixed gates.",
    ]
    validation_gates = [
        "study_design_brief",
        "article_analysis_contract",
        "article_figure_strategy",
        "display_suite",
        "review_artifacts",
    ]
    return AnalysisBlueprint(
        analysis_family=resolved_brief.analysis_family,
        question_classification=resolved_brief.rationale,
        prior_art_design_brief=prior_art,
        reporting_guidelines=list(resolved_brief.reporting_guidelines),
        design_principles=list(resolved_brief.design_principles),
        required_methods=list(resolved_brief.required_methods),
        required_article_roles=list(resolved_contract.required_roles),
        recommended_article_roles=list(resolved_contract.recommended_roles),
        article_roles=article_roles,
        main_text_display_roles=[
            role.role
            for role in article_roles
            if role.required and role.tier != "supplementary"
        ],
        supplementary_display_roles=list(resolved_brief.supplementary_displays),
        sensitivity_requirements=list(resolved_brief.sensitivity_requirements),
        figure_archetype=resolved_strategy.archetype,
        figure_hero_role=resolved_strategy.hero_role,
        minimum_distinct_chart_types=resolved_strategy.minimum_distinct_chart_types,
        visual_roles=visual_roles,
        anti_patterns=anti_patterns,
        planner_sequence=planner_sequence,
        validation_gates=validation_gates,
        source_brief_schema_version=resolved_brief.schema_version,
        source_contract_schema_version=resolved_contract.schema_version,
        source_figure_strategy_schema_version=resolved_strategy.schema_version,
    )


def render_analysis_blueprint_for_prompt(blueprint: AnalysisBlueprint) -> str:
    required_article = [
        role
        for role in blueprint.article_roles
        if role.required and role.tier != "supplementary"
    ]
    required_visual = [role for role in blueprint.visual_roles if role.required]
    lines = [
        "ANALYSIS BLUEPRINT:",
        f"- analysis_family: {blueprint.analysis_family}",
        f"- question_classification: {blueprint.question_classification}",
        "- reporting_guidelines: " + "; ".join(blueprint.reporting_guidelines),
        "- planner_sequence: " + "; ".join(blueprint.planner_sequence),
        "",
        "PRIOR-ART DESIGN BRIEF:",
        f"- source_mode: {blueprint.prior_art_design_brief.source_mode}",
        "- purpose: " + blueprint.prior_art_design_brief.purpose,
        "- design_reference_queries: "
        + "; ".join(blueprint.prior_art_design_brief.design_reference_queries[:5]),
        "- design_questions: "
        + "; ".join(blueprint.prior_art_design_brief.design_questions),
        "- main_text_pattern: "
        + "; ".join(blueprint.prior_art_design_brief.main_text_pattern),
        "- supplement_pattern: "
        + "; ".join(blueprint.prior_art_design_brief.supplement_pattern[:6]),
        "- extraction_guardrails: "
        + "; ".join(blueprint.prior_art_design_brief.extraction_guardrails),
        "",
        "ARTICLE ANALYSIS CONTRACT:",
        "- required_article_roles: " + ", ".join(blueprint.required_article_roles),
        "- required_modules:",
    ]
    for role in required_article:
        lines.append(
            "  - "
            f"{role.module_id} (role={role.role}; tier={role.tier}; "
            f"typed_example=table:{role.module_id}; "
            f"acceptable={', '.join(role.acceptable_outputs[:4])}; "
            f"rationale={role.rationale})"
        )
    lines.extend(
        [
            "",
            "ARTICLE FIGURE STRATEGY:",
            f"- archetype: {blueprint.figure_archetype}",
            f"- hero_role: {blueprint.figure_hero_role}",
            f"- minimum_distinct_chart_types: {blueprint.minimum_distinct_chart_types}",
            "- required_visual_roles:",
        ]
    )
    for role in required_visual:
        lines.append(
            "  - "
            f"{role.role} (placement={role.placement}; acceptable_chart_types={', '.join(role.acceptable_chart_types[:5])}; "
            f"rationale={role.rationale})"
        )
    lines.extend(
        [
            "- anti_patterns: " + "; ".join(blueprint.anti_patterns),
            "- validation_gates: " + "; ".join(blueprint.validation_gates),
            "- rule: every required article role must be owned by an explicit analysis step whose expected_outputs include its typed_example (or an equally explicit typed product using an acceptable term); Intent-only prose does not count.",
            "- rule: executable steps must map outputs to article roles; a single technically valid composite is one figure, not an article package. Use separate numbered figures when evidence roles answer different reader questions.",
            "- rule: routine missingness and measurement-process audits belong in supplementary material unless missingness is the research question or materially changes the interpretation of the primary result.",
        ]
    )
    return "\n".join(lines)


def _step_text(plan: Optional[AnalysisPlan]) -> str:
    if plan is None:
        return ""
    parts: List[str] = [
        plan.analysis_type or "",
        plan.rationale or "",
    ]
    for step in plan.steps or []:
        parts.extend(
            [
                step.step_id or "",
                step.intent or "",
                step.method or "",
                " ".join(step.inputs or []),
                " ".join(step.expected_outputs or []),
                " ".join(step.icu_rule_refs or []),
            ]
        )
    return _normalise_space("\n".join(parts))


def _covered_article_roles(
    plan_text: str,
    roles: Sequence[BlueprintArticleRole],
) -> List[str]:
    covered: set[str] = set()
    for role in roles:
        if any(_normalise_space(term) in plan_text for term in role.search_terms):
            covered.add(role.role)
    return sorted(covered)


def _covered_visual_roles(
    plan: Optional[AnalysisPlan],
    roles: Sequence[BlueprintVisualRole],
) -> List[str]:
    """Return roles owned by explicit figure-producing steps only.

    The old implementation searched the complete plan text.  A Table 1 or
    missingness *table* elsewhere in the plan could therefore satisfy a
    required visual role even when no figure consumed that evidence.  That
    made the plan gate credit article figures that the executor would never
    produce.
    """

    covered: set[str] = set()
    figure_steps = [
        step
        for step in (plan.steps if plan is not None else ())
        if any(str(value).startswith("figure:") for value in step.expected_outputs)
    ]
    for role in roles:
        strategy_role = FigureRoleStrategy(
            role=role.role,
            required=role.required,
            rationale=role.rationale,
            acceptable_chart_types=list(role.acceptable_chart_types),
            required_text_terms=list(role.required_text_terms),
            search_terms=list(role.search_terms),
        )
        if any(figure_step_covers_role(step, strategy_role) for step in figure_steps):
            covered.add(role.role)
    return sorted(covered)


def validate_plan_against_analysis_blueprint(
    *,
    plan: Optional[AnalysisPlan],
    blueprint: AnalysisBlueprint,
) -> List[ValidationFinding]:
    plan_text = _step_text(plan)
    required_article_roles = {
        role.role for role in blueprint.article_roles if role.required
    }
    required_visual_roles = {
        role.role for role in blueprint.visual_roles if role.required
    }
    covered_article_roles = set(
        _covered_article_roles(plan_text, blueprint.article_roles)
    )
    covered_visual_roles = set(_covered_visual_roles(plan, blueprint.visual_roles))
    missing_article_roles = sorted(required_article_roles - covered_article_roles)
    missing_visual_roles = sorted(required_visual_roles - covered_visual_roles)
    findings: List[ValidationFinding] = []
    if missing_article_roles:
        findings.append(
            ValidationFinding(
                validator="analysis_blueprint",
                severity="warning",
                message=(
                    "Analysis plan does not cover all article-level roles in "
                    f"the {blueprint.analysis_family} blueprint."
                ),
                detail={
                    "analysis_family": blueprint.analysis_family,
                    "covered_article_roles": sorted(covered_article_roles),
                    "missing_article_roles": missing_article_roles,
                },
            )
        )
    if missing_visual_roles:
        findings.append(
            ValidationFinding(
                validator="analysis_blueprint",
                severity="warning",
                message=(
                    "Analysis plan does not cover all required visual roles in "
                    f"the {blueprint.analysis_family} blueprint."
                ),
                detail={
                    "analysis_family": blueprint.analysis_family,
                    "covered_visual_roles": sorted(covered_visual_roles),
                    "missing_visual_roles": missing_visual_roles,
                    "figure_hero_role": blueprint.figure_hero_role,
                },
            )
        )
    min_roles = min(3, len(required_article_roles))
    if len(covered_article_roles) < min_roles:
        findings.append(
            ValidationFinding(
                validator="analysis_blueprint",
                severity="warning",
                message=(
                    "Analysis plan is too narrow for the article blueprint; "
                    "it should combine design/accounting, data-quality, "
                    "primary-result, and robustness or family-specific roles."
                ),
                detail={
                    "covered_article_roles": sorted(covered_article_roles),
                    "expected_article_roles": sorted(required_article_roles),
                },
            )
        )
    return findings


__all__ = [
    "ANALYSIS_BLUEPRINT_SCHEMA_VERSION",
    "PRIOR_ART_DESIGN_BRIEF_SCHEMA_VERSION",
    "AnalysisBlueprint",
    "BlueprintArticleRole",
    "BlueprintVisualRole",
    "PriorArtDesignBrief",
    "build_analysis_blueprint",
    "render_analysis_blueprint_for_prompt",
    "validate_plan_against_analysis_blueprint",
]
