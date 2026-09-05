"""Article-level figure strategy contracts.

The article-analysis contract answers "which evidence roles must exist?".
This module answers the adjacent figure question: "which visual roles and
chart families must be visible before a manuscript-facing figure suite can be
called article-grade?"  It is deliberately case-neutral: requirements are
keyed by study-design family rather than a benchmark variable or database.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Set

from pydantic import BaseModel, ConfigDict, Field

from ..contracts.figure_plan import (
    MEASUREMENT_PROCESS_AUDIT_INPUT,
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
)
from ..figures.contracts import figure_contract_paths, panel_chart_type, panel_text
from ..schema import ResearchContext, ValidationFinding
from .study_design import infer_study_design_family
from .study_design_playbook import StudyDesignFamily

ARTICLE_FIGURE_STRATEGY_SCHEMA_VERSION = "easyicu.article_figure_strategy/1"
ARTICLE_FIGURE_STRATEGY_AUDIT_SCHEMA_VERSION = "easyicu.article_figure_strategy_audit/1"

# One owner vocabulary for the typed data-quality display. Plan shaping and
# the deterministic renderer both import these identities; a generic
# sensitivity plot can therefore never masquerade as missingness/measurement
# evidence simply because it contains the word "audit".
DATA_QUALITY_FIGURE_REQUIRED_INPUTS = (
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
    MEASUREMENT_PROCESS_AUDIT_INPUT,
)
DATA_QUALITY_FIGURE_PRODUCT = "figure:data_quality"

_GENERIC_CHART_TYPES = {"bar", "forest", "heatmap", "unspecified"}
_GENERIC_PANEL_ROLES = {
    "",
    "audit",
    "display",
    "figure",
    "panel",
    "relationship",
    "result",
    "results",
    "visual",
}
_PRIMARY_PUBLICATION_MIN_ROLES = {
    "association": 3,
    "prediction": 3,
    "time_to_event": 2,
    "phenotyping": 3,
    "causal_emulation": 3,
    "descriptive": 2,
}


class FigureRoleStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    required: bool = True
    placement: Literal["main", "supplementary"] = "main"
    rationale: str
    acceptable_chart_types: List[str] = Field(default_factory=list)
    required_text_terms: List[str] = Field(default_factory=list)
    search_terms: List[str] = Field(default_factory=list)


class ArticleFigureStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ARTICLE_FIGURE_STRATEGY_SCHEMA_VERSION
    analysis_family: StudyDesignFamily
    archetype: str
    hero_role: str
    minimum_distinct_chart_types: int = Field(
        default=3,
        description="Advisory design target; never a scientific or publication gate.",
    )
    role_strategies: List[FigureRoleStrategy] = Field(default_factory=list)
    anti_patterns: List[str] = Field(default_factory=list)
    prompt_rules: List[str] = Field(default_factory=list)


def _normalise_role_match_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def figure_panel_covers_role(panel: Any, role: FigureRoleStrategy) -> bool:
    """Whether one typed panel exactly satisfies a strategy role."""

    expected_role = _normalise_role_match_text(role.role).replace(" ", "_")
    declared_role = _normalise_role_match_text(
        getattr(panel, "article_role", "")
    ).replace(" ", "_")
    if declared_role != expected_role:
        return False
    chart_type = _normalise_role_match_text(getattr(panel, "chart_type", "")).replace(
        " ", "_"
    )
    return _acceptable_chart_match(role, chart_type)


def figure_step_covers_role(step: Any, role: FigureRoleStrategy) -> bool:
    """Whether a figure step explicitly declares this role and chart grammar.

    Input and output product names are lineage coordinates, not visual
    semantics.  A renderer that consumes a cohort-flow, missingness, and result
    table is not thereby three article panels.  Only the Planner's typed panel
    declarations can satisfy plan-time figure coverage; execution later checks
    the resulting FigureContract and source data independently.
    """

    return any(
        figure_panel_covers_role(panel, role)
        for panel in getattr(step, "figure_panels", ())
    )


def _role(
    role: str,
    rationale: str,
    acceptable_chart_types: Sequence[str],
    *,
    required_text_terms: Sequence[str] = (),
    search_terms: Sequence[str] = (),
    required: bool = True,
    placement: Literal["main", "supplementary"] = "main",
) -> FigureRoleStrategy:
    return FigureRoleStrategy(
        role=role,
        required=required,
        placement=placement,
        rationale=rationale,
        acceptable_chart_types=list(acceptable_chart_types),
        required_text_terms=list(required_text_terms),
        search_terms=list(search_terms),
    )


_FAMILY_STRATEGIES: Dict[StudyDesignFamily, Dict[str, Any]] = {
    "association": {
        "archetype": "asymmetric_mixed_modality",
        "hero_role": "descriptive_result",
        "minimum_distinct_chart_types": 3,
        "roles": [
            _role(
                "descriptive_result",
                "Show exposure prevalence and absolute outcome risk before relative adjusted estimates.",
                (
                    "dot_interval_absolute_risk",
                    "event_rate_panel",
                    "prevalence_panel",
                    "absolute_risk_curve",
                    "marginal_probability_plot",
                    "dot_interval",
                ),
                required_text_terms=(
                    "absolute risk",
                    "absolute outcome risk",
                    "outcome risk",
                    "event rate",
                    "event-rate",
                    "exposure prevalence",
                    "prevalence",
                ),
                search_terms=(
                    "prevalence",
                    "event rate",
                    "absolute risk",
                    "outcome risk",
                    "outcome by exposure",
                ),
            ),
            _role(
                "primary_estimand",
                "Expose the primary adjusted estimand, scale, uncertainty, and adjustment context.",
                (
                    "forest",
                    "coefficient_plot",
                    "dot_interval",
                    "marginal_effect_panel",
                    "table",
                ),
                search_terms=(
                    "adjusted",
                    "odds ratio",
                    "risk ratio",
                    "effect estimate",
                ),
            ),
            _role(
                "robustness",
                "Show sensitivity across definitions, missing-data choices, or model specifications.",
                (
                    "specification_grid",
                    "sensitivity_forest",
                    "small_multiples",
                    "dot_interval",
                ),
                search_terms=(
                    "sensitivity",
                    "robustness",
                    "specification",
                    "alternative",
                ),
            ),
            _role(
                "data_quality",
                "Make missingness and measurement-process context visible.",
                ("missingness_matrix", "availability_panel", "coverage_heatmap", "bar"),
                search_terms=("missingness", "measurement", "availability", "coverage"),
                placement="supplementary",
            ),
        ],
        "anti_patterns": [
            "A lone adjusted-estimate forest plot.",
            "A bar/forest/heatmap-only suite with no absolute-risk context.",
            "A risk-difference sensitivity panel used as a substitute for exposure prevalence or absolute outcome risk.",
            "Equal-sized audit panels with no reader-facing hero panel.",
        ],
        "prompt_rules": [
            "Start the figure plan with the absolute-risk/prevalence context, then the adjusted estimand, then robustness and audit panels.",
            "Risk-difference sensitivity is useful, but it must not replace absolute outcome risk by exposure.",
        ],
    },
    "prediction": {
        "archetype": "asymmetric_mixed_modality",
        "hero_role": "calibration",
        "minimum_distinct_chart_types": 3,
        "roles": [
            _role(
                "model_performance",
                "Discrimination must be visible but is not sufficient by itself.",
                (
                    "roc_curve",
                    "precision_recall_curve",
                    "metric_dot_interval",
                    "performance_table",
                ),
                search_terms=("roc", "auroc", "precision-recall", "discrimination"),
            ),
            _role(
                "calibration",
                "Clinical prediction figures need calibration to show whether risks are usable.",
                ("calibration_curve", "calibration_belt", "risk_decile_plot"),
                search_terms=("calibration", "brier", "slope"),
            ),
            _role(
                "validation",
                "The figure suite must distinguish development, temporal, site, or external validation.",
                (
                    "split_diagram",
                    "validation_panel",
                    "database_small_multiples",
                    "metric_dot_interval",
                ),
                search_terms=("validation", "external", "temporal", "test set"),
            ),
            _role(
                "data_quality",
                "Feature availability, imputation, and leakage checks affect reported model performance.",
                (
                    "feature_availability_panel",
                    "availability_panel",
                    "missingness_matrix",
                    "coverage_heatmap",
                    "leakage_audit",
                ),
                search_terms=(
                    "missingness",
                    "feature availability",
                    "leakage",
                    "preprocessing",
                ),
                placement="supplementary",
            ),
        ],
        "anti_patterns": [
            "AUROC-only reporting.",
            "Feature-importance panels replacing validation and calibration.",
        ],
        "prompt_rules": [
            "Pair discrimination with calibration and validation design in the main figure strategy.",
        ],
    },
    "time_to_event": {
        "archetype": "asymmetric_mixed_modality",
        "hero_role": "temporal_absolute_risk",
        "minimum_distinct_chart_types": 3,
        "roles": [
            _role(
                "temporal_absolute_risk",
                "Absolute-risk curves and risk sets orient readers before hazard contrasts.",
                ("kaplan_meier_curve", "cumulative_incidence_curve", "risk_table"),
                search_terms=(
                    "survival curve",
                    "kaplan",
                    "cumulative incidence",
                    "risk table",
                ),
            ),
            _role(
                "survival_effect",
                "Adjusted hazard or risk contrasts quantify the primary survival estimand.",
                (
                    "hazard_ratio_forest",
                    "risk_difference_panel",
                    "survival_contrast_table",
                ),
                search_terms=("hazard", "cox", "survival contrast"),
            ),
            _role(
                "diagnostics",
                "Censoring and proportional-hazards assumptions need visible checks.",
                ("diagnostic_panel", "followup_distribution", "schoenfeld_plot"),
                search_terms=("censoring", "proportional hazards", "diagnostic"),
            ),
        ],
        "anti_patterns": [
            "Hazard-ratio-only display without absolute-risk curves or risk sets.",
        ],
        "prompt_rules": [
            "Do not collapse absolute risk over time and adjusted hazard contrasts into one forest plot.",
        ],
    },
    "phenotyping": {
        "archetype": "asymmetric_mixed_modality",
        "hero_role": "phenotype_structure",
        "minimum_distinct_chart_types": 3,
        "roles": [
            _role(
                "phenotype_structure",
                "Show whether discovered groups are separable, continuous, or weakly structured.",
                ("embedding_plot", "cluster_heatmap", "dendrogram", "pca_umap"),
                search_terms=("embedding", "umap", "pca", "cluster heatmap"),
            ),
            _role(
                "phenotype_profile",
                "Clinical profiles make unsupervised groups interpretable.",
                (
                    "profile_heatmap",
                    "radar",
                    "parallel_coordinates",
                    "characteristics_table",
                ),
                search_terms=("profile", "characteristics", "radar"),
            ),
            _role(
                "stability",
                "Stability evidence protects against arbitrary cluster cuts.",
                ("stability_grid", "consensus_matrix", "bootstrap_panel"),
                search_terms=("stability", "bootstrap", "consensus"),
            ),
            _role(
                "data_quality",
                "Feature availability and scaling affect cluster geometry.",
                ("feature_missingness_matrix", "scaling_summary", "availability_panel"),
                search_terms=("feature missingness", "scaling", "availability"),
                placement="supplementary",
            ),
        ],
        "anti_patterns": [
            "A cluster heatmap without stability or clinical profile panels.",
            "Outcome association used as proof that clusters are causal entities.",
        ],
        "prompt_rules": [
            "Separate structure, profile, and stability panels before adding downstream outcomes.",
        ],
    },
    "causal_emulation": {
        "archetype": "schematic_led_composite",
        "hero_role": "causal_protocol",
        "minimum_distinct_chart_types": 3,
        "roles": [
            _role(
                "causal_protocol",
                "The figure strategy must make eligibility, time zero, strategies, and estimand explicit.",
                ("target_trial_schematic", "protocol_table", "timeline_diagram"),
                search_terms=("target trial", "time zero", "estimand", "strategy"),
            ),
            _role(
                "balance_positivity",
                "Balance and positivity must be inspected before interpreting a causal contrast.",
                ("love_plot", "weight_distribution", "positivity_panel"),
                search_terms=(
                    "balance",
                    "standardized mean difference",
                    "positivity",
                    "weight",
                ),
            ),
            _role(
                "causal_contrast",
                "The main causal contrast must state estimator, effect scale, and uncertainty.",
                ("causal_contrast_panel", "effect_curve", "estimate_table"),
                search_terms=("causal contrast", "iptw", "g-computation", "matching"),
            ),
            _role(
                "robustness",
                "Unmeasured-confounding, trimming, and estimator choices need sensitivity evidence.",
                ("sensitivity_grid", "trimming_panel", "alternative_estimator_panel"),
                search_terms=("sensitivity", "trimming", "unmeasured confounding"),
            ),
        ],
        "anti_patterns": [
            "An effect-estimate forest plot without target-trial protocol or balance panels.",
        ],
        "prompt_rules": [
            "Lead causal figures with protocol/timing, then balance, then effect contrast and sensitivity.",
        ],
    },
    "descriptive": {
        "archetype": "quantitative_grid",
        "hero_role": "distribution",
        "minimum_distinct_chart_types": 2,
        "roles": [
            _role(
                "distribution",
                "The main figure must answer the distributional or prevalence question.",
                (
                    "distribution_plot",
                    "density",
                    "histogram",
                    "ridge",
                    "point_range",
                    "prevalence_panel",
                    "point_absolute_risk",
                    "dot_interval_absolute_risk",
                ),
                search_terms=("distribution", "prevalence", "density", "histogram"),
            ),
            _role(
                "cohort_accounting",
                "Descriptive results still need explicit denominators.",
                ("cohort_flow", "denominator_panel", "attrition_table"),
                search_terms=("cohort", "denominator", "attrition"),
            ),
            _role(
                "data_quality",
                "Coverage and missingness determine how interpretable descriptive summaries are.",
                ("coverage_heatmap", "missingness_matrix", "availability_panel"),
                search_terms=("missingness", "coverage", "availability"),
                placement="supplementary",
            ),
        ],
        "anti_patterns": [
            "A pooled bar chart without denominator or data-quality context.",
        ],
        "prompt_rules": [
            "Use distributional displays for descriptive questions; do not force adjusted modelling into the main figure strategy.",
        ],
    },
}


def _data_quality_is_the_scientific_question(context: ResearchContext) -> bool:
    """Return true only when measurement quality is explicitly a headline aim.

    High missingness alone is not enough: that would push routine ICU data audits
    into every main figure.  The question or author notes must make measurement,
    availability, or missingness itself part of the scientific problem.
    """

    text = " ".join((context.research_question or "", context.notes or "")).casefold()
    return any(
        token in text
        for token in (
            "missingness",
            "missing data",
            "measurement process",
            "measurement frequency",
            "measurement availability",
            "data completeness",
            "source absence",
            "not measured",
            "缺失",
            "测量过程",
            "测量频率",
            "数据完整",
            "未测量",
        )
    )


def build_article_figure_strategy(
    context: ResearchContext,
    *,
    analysis_family: StudyDesignFamily | None = None,
) -> ArticleFigureStrategy:
    family = analysis_family or infer_study_design_family(context)
    template = _FAMILY_STRATEGIES[family]
    roles = [role.model_copy(deep=True) for role in template["roles"]]
    if _data_quality_is_the_scientific_question(context):
        roles = [
            role.model_copy(update={"placement": "main"})
            if role.role == "data_quality"
            else role
            for role in roles
        ]
    return ArticleFigureStrategy(
        analysis_family=family,
        archetype=str(template["archetype"]),
        hero_role=str(template["hero_role"]),
        minimum_distinct_chart_types=int(template["minimum_distinct_chart_types"]),
        role_strategies=roles,
        anti_patterns=[
            *list(template["anti_patterns"]),
            "A single composite treated as the entire article display package.",
            "Routine missingness or measurement audits occupying a main-result panel when they are not central to the scientific question.",
        ],
        prompt_rules=[
            *list(template["prompt_rules"]),
            "Plan an article-level suite, usually 2-4 complementary main figures plus main tables; this is a planning target, not a fixed acceptance count.",
            "Put routine missingness and measurement-process detail in supplementary displays; promote it to the main text only when it is central to the research question or changes interpretation of the primary result.",
        ],
    )


def render_article_figure_strategy_for_prompt(strategy: ArticleFigureStrategy) -> str:
    lines = [
        "ARTICLE FIGURE STRATEGY:",
        f"- analysis_family: {strategy.analysis_family}",
        f"- archetype: {strategy.archetype}",
        f"- hero_role: {strategy.hero_role}",
        f"- advisory_chart_diversity_target: {strategy.minimum_distinct_chart_types} (choose only forms that help the question; this is not a publication requirement)",
        "- required_visual_roles:",
    ]
    for role in strategy.role_strategies:
        if not role.required:
            continue
        lines.append(
            "  - "
            f"{role.role} (placement={role.placement}; acceptable_chart_types={', '.join(role.acceptable_chart_types[:5])}; "
            f"rationale={role.rationale})"
        )
    lines.append("- anti_patterns: " + "; ".join(strategy.anti_patterns))
    lines.append("- rules: " + "; ".join(strategy.prompt_rules))
    return "\n".join(lines)


def _normalise(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


# Shared with display_suite / review_artifacts via figures.contracts so the
# audits cannot disagree about which contracts exist.
_contract_paths = figure_contract_paths


def _is_primary_publication_contract(path: Path, run_dir: Path) -> bool:
    try:
        path.resolve().relative_to((run_dir / "publication_figures").resolve())
    except ValueError:
        return False
    return path.name.endswith(".figure_contract.json")


_panel_text = panel_text


def _panel_role(panel: Mapping[str, Any]) -> str:
    metadata = (
        panel.get("metadata") if isinstance(panel.get("metadata"), Mapping) else {}
    )
    return _normalise(
        panel.get("article_role")
        or metadata.get("article_role")
        or panel.get("role")
        or metadata.get("role")
        or ""
    )


_panel_chart_type = panel_chart_type


def _role_matches_panel(role: FigureRoleStrategy, panel: Mapping[str, Any]) -> bool:
    panel_role = _panel_role(panel)
    if panel_role == role.role:
        return True
    text = _panel_text(panel)
    if panel_role and panel_role not in _GENERIC_PANEL_ROLES:
        return False
    return any(term and term in text for term in role.search_terms)


def _role_has_required_text(
    role: FigureRoleStrategy, panels: Sequence[Mapping[str, Any]]
) -> bool:
    if not role.required_text_terms:
        return True
    return any(
        any(term and term in _panel_text(panel) for term in role.required_text_terms)
        for panel in panels
    )


def _acceptable_chart_match(role: FigureRoleStrategy, chart_type: str) -> bool:
    if not role.acceptable_chart_types:
        return True
    if chart_type == "unspecified":
        # Missing chart coordinates are a contract defect, not low diversity.
        return False
    accepted = {item.replace(" ", "_") for item in role.acceptable_chart_types}
    # Deterministic renderers predate the article-strategy vocabulary and a
    # few of their precise chart names carry harmless geometry suffixes.  Keep
    # this translation in the strategy owner so a renderer does not have to
    # lie about its actual geometry and the audit does not fail a valid panel
    # merely because one side says ``forest`` and the other ``forest_plot``.
    # This is intentionally a small, one-way map: a selection-criterion curve,
    # for example, must not masquerade as a phenotype profile or stability
    # analysis.
    renderer_chart_aliases = {
        "grouped_absolute_risk": "dot_interval_absolute_risk",
        "forest_plot": "forest",
        "sensitivity_forest_plot": "sensitivity_forest",
        "availability_heatmap": "coverage_heatmap",
        "rmst_difference_forest": "risk_difference_panel",
    }
    chart_type = renderer_chart_aliases.get(chart_type, chart_type)
    if chart_type in accepted:
        return True
    # Inferred chart types are coarse families; map each family onto every
    # concrete chart type the role strategies declare, otherwise a correctly
    # roled panel without explicit chart_type metadata fails closed (e.g. a
    # calibration panel infers "curve" and was rejected against
    # "calibration_curve").
    family_aliases = {
        "dot_interval": {
            "dot_interval_absolute_risk",
            "event_rate_panel",
            "prevalence_panel",
            "metric_dot_interval",
            "risk_difference_panel",
        },
        "curve": {
            "absolute_risk_curve",
            "marginal_probability_plot",
            "kaplan_meier_curve",
            "cumulative_incidence_curve",
            "calibration_curve",
            "calibration_belt",
            "roc_curve",
            "precision_recall_curve",
            "effect_curve",
        },
        "bar": {
            "availability_panel",
            "feature_availability_panel",
            "denominator_panel",
            "prevalence_panel",
        },
        "heatmap": {
            "coverage_heatmap",
            "missingness_matrix",
            "cluster_heatmap",
            "profile_heatmap",
            "feature_missingness_matrix",
            "consensus_matrix",
        },
        "forest": {"sensitivity_forest", "hazard_ratio_forest", "coefficient_plot"},
        "flow": {
            "cohort_flow",
            "target_trial_schematic",
            "protocol_table",
            "timeline_diagram",
        },
        "distribution": {
            "distribution_plot",
            "density",
            "histogram",
            "ridge",
            "prevalence_panel",
            "followup_distribution",
            "weight_distribution",
        },
    }
    return bool(family_aliases.get(chart_type, set()) & accepted)


def _read_panels(
    run_dir: Path,
    *,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    panels: List[Dict[str, Any]] = []
    for path in _contract_paths(
        run_dir,
        per_step_records=per_step_records,
    ):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        raw_panels = raw.get("panels")
        if not isinstance(raw_panels, list):
            continue
        for panel in raw_panels:
            if not isinstance(panel, dict):
                continue
            item = dict(panel)
            item["_contract_path"] = str(path)
            item["_figure_id"] = str(raw.get("figure_id") or path.stem)
            item["_chart_type"] = _panel_chart_type(item)
            item["_role"] = _panel_role(item)
            item["_primary_publication_contract"] = _is_primary_publication_contract(
                path,
                run_dir,
            )
            panels.append(item)
    return panels


def summarize_article_figure_strategy_coverage(
    *,
    context: ResearchContext,
    run_dir: Path,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
    analysis_family: StudyDesignFamily | None = None,
) -> Dict[str, Any]:
    strategy = build_article_figure_strategy(
        context,
        analysis_family=analysis_family,
    )
    panels = _read_panels(run_dir, per_step_records=per_step_records)
    primary_panels = [
        panel for panel in panels if panel.get("_primary_publication_contract")
    ]
    chart_types = sorted({_panel_chart_type(panel) for panel in panels})
    nonempty_chart_types = {chart for chart in chart_types if chart != "unspecified"}
    primary_chart_types = sorted(
        {
            _panel_chart_type(panel)
            for panel in primary_panels
            if _panel_chart_type(panel) != "unspecified"
        }
    )
    covered_roles: Set[str] = set()
    primary_publication_roles: Set[str] = set()
    role_errors: List[str] = []
    role_panel_ids: Dict[str, List[str]] = {}
    primary_role_panel_ids: Dict[str, List[str]] = {}
    for role in strategy.role_strategies:
        matching = [panel for panel in panels if _role_matches_panel(role, panel)]
        role_panel_ids[role.role] = [
            f"{panel.get('_figure_id')}:{panel.get('panel_id')}" for panel in matching
        ]
        primary_matching = [
            panel for panel in primary_panels if _role_matches_panel(role, panel)
        ]
        primary_role_panel_ids[role.role] = [
            f"{panel.get('_figure_id')}:{panel.get('panel_id')}"
            for panel in primary_matching
        ]
        if not matching:
            if role.required:
                role_errors.append(f"Missing required figure role: {role.role}.")
            continue
        if not _role_has_required_text(role, matching):
            role_errors.append(
                f"Figure role {role.role} lacks required reader-facing term(s): "
                + ", ".join(role.required_text_terms)
            )
            continue
        acceptable = [
            panel
            for panel in matching
            if _acceptable_chart_match(role, _panel_chart_type(panel))
        ]
        if not acceptable:
            role_errors.append(
                f"Figure role {role.role} is present but uses unsupported chart type(s): "
                + ", ".join(sorted({_panel_chart_type(panel) for panel in matching}))
            )
            continue
        covered_roles.add(role.role)
        primary_acceptable = [
            panel
            for panel in primary_matching
            if _acceptable_chart_match(role, _panel_chart_type(panel))
        ]
        if (
            primary_matching
            and _role_has_required_text(role, primary_matching)
            and primary_acceptable
        ):
            primary_publication_roles.add(role.role)
            if role.placement == "supplementary":
                role_errors.append(
                    "Figure role "
                    f"{role.role} is designated supplementary but appears in a "
                    "primary publication figure. Promote it only when the "
                    "research question explicitly makes that evidence central."
                )

    required_roles = {role.role for role in strategy.role_strategies if role.required}
    required_main_roles = {
        role.role
        for role in strategy.role_strategies
        if role.required and role.placement == "main"
    }
    errors = list(role_errors)
    primary_minimum_required_role_count = min(
        len(required_main_roles),
        _PRIMARY_PUBLICATION_MIN_ROLES.get(
            str(strategy.analysis_family),
            min(3, len(required_main_roles)),
        ),
    )
    if not primary_panels:
        errors.append("No primary publication figure contract was found.")
    elif strategy.hero_role not in primary_publication_roles:
        errors.append(
            "Primary publication figure lacks the required hero role: "
            f"{strategy.hero_role}."
        )
    primary_required_role_count = len(primary_publication_roles & required_main_roles)
    if (
        primary_panels
        and primary_required_role_count < primary_minimum_required_role_count
    ):
        errors.append(
            "Primary publication figure covers fewer required visual roles than "
            f"expected for {strategy.analysis_family}: "
            f"{primary_required_role_count} < {primary_minimum_required_role_count}. "
            "Supporting or scratch figures can supplement the article package, "
            "but they cannot make a sparse main figure article-grade."
        )
    design_advice = []
    if len(nonempty_chart_types) < strategy.minimum_distinct_chart_types:
        design_advice.append(
            "Figure strategy uses fewer distinct chart families than expected "
            f"for {strategy.analysis_family}: {len(nonempty_chart_types)} < "
            f"{strategy.minimum_distinct_chart_types}."
        )
    if chart_types and set(chart_types) <= _GENERIC_CHART_TYPES:
        design_advice.append(
            "Figure strategy is limited to generic bar/forest/heatmap panels; "
            "consider other forms only when they communicate the planned scientific information more clearly."
        )
    if strategy.hero_role not in covered_roles:
        errors.append(
            f"Figure strategy lacks the required hero role: {strategy.hero_role}."
        )
    return {
        "article_figure_strategy_audit_schema_version": ARTICLE_FIGURE_STRATEGY_AUDIT_SCHEMA_VERSION,
        "article_figure_strategy_complete": not errors,
        "article_figure_strategy_family": strategy.analysis_family,
        "article_figure_strategy_archetype": strategy.archetype,
        "article_figure_strategy_hero_role": strategy.hero_role,
        "article_figure_strategy_required_roles": sorted(required_roles),
        "article_figure_strategy_required_main_roles": sorted(required_main_roles),
        "article_figure_strategy_covered_roles": sorted(covered_roles),
        "article_figure_strategy_missing_roles": sorted(required_roles - covered_roles),
        "article_figure_strategy_chart_types": chart_types,
        "article_figure_strategy_primary_publication_roles": sorted(
            primary_publication_roles
        ),
        "article_figure_strategy_primary_publication_chart_types": primary_chart_types,
        "article_figure_strategy_primary_publication_panel_count": len(primary_panels),
        "article_figure_strategy_primary_publication_minimum_required_role_count": (
            primary_minimum_required_role_count
        ),
        "article_figure_strategy_primary_publication_role_panels": (
            primary_role_panel_ids
        ),
        "article_figure_strategy_minimum_distinct_chart_types": strategy.minimum_distinct_chart_types,
        "article_figure_strategy_role_panels": role_panel_ids,
        "article_figure_strategy_errors": errors,
        "article_figure_strategy_design_advice": design_advice,
        "article_figure_strategy": strategy.model_dump(mode="json"),
    }


def validate_run_against_article_figure_strategy(
    *,
    context: ResearchContext,
    run_dir: Path,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
    analysis_family: StudyDesignFamily | None = None,
) -> List[ValidationFinding]:
    """Emit the finding for the same coverage readiness already gates on.

    ``analysis_family`` must be the family the readiness projection resolves
    from the final plan.  Letting this validator re-derive it from context
    alone would let the emitted finding disagree with the gate it reports on.
    """

    status = summarize_article_figure_strategy_coverage(
        context=context,
        run_dir=run_dir,
        per_step_records=per_step_records,
        analysis_family=analysis_family,
    )
    if status["article_figure_strategy_complete"]:
        return []
    return [
        ValidationFinding(
            validator="article_figure_strategy",
            severity="warning",
            message=(
                "Run figures do not yet satisfy the article-level figure strategy "
                f"for {status['article_figure_strategy_family']} studies."
            ),
            detail={
                "missing_roles": status["article_figure_strategy_missing_roles"],
                "chart_types": status["article_figure_strategy_chart_types"],
                "errors": status["article_figure_strategy_errors"],
            },
        )
    ]


__all__ = [
    "ARTICLE_FIGURE_STRATEGY_AUDIT_SCHEMA_VERSION",
    "ARTICLE_FIGURE_STRATEGY_SCHEMA_VERSION",
    "ArticleFigureStrategy",
    "DATA_QUALITY_FIGURE_PRODUCT",
    "DATA_QUALITY_FIGURE_REQUIRED_INPUTS",
    "FigureRoleStrategy",
    "build_article_figure_strategy",
    "figure_panel_covers_role",
    "figure_step_covers_role",
    "render_article_figure_strategy_for_prompt",
    "summarize_article_figure_strategy_coverage",
    "validate_run_against_article_figure_strategy",
]
