"""ClinicalSkill registry — composable ICU analysis families.

The research-agent layer exposes reusable "skills" that agents can
invoke. Here they are specialised to the EasyICU shape:

A :class:`ClinicalSkill` is a small object that, given a cohort
DataFrame, declares:

* the analysis family (association, prediction, data-quality audit, ...);
* optional target outcome / primary predictor hints when a user registers
  a local skill;
* optional time-window and expected-variable hints;
* a deterministic plan assembled by the shared skill template when the
  user explicitly picks the skill instead of writing a free-form
  research question.

Built-in skills must stay at the analysis-family level. A key such as
``association_analysis`` or ``prediction_model`` is acceptable because it
describes a reusable workflow shape; a key such as ``sofa_mortality`` is
too narrow and bakes one benchmark question into the package. Users may
still register their own local skills with concrete variables.

``plan_factory`` remains available for explicit extension points, but
the built-in registry must stay case-neutral and must not bundle
bespoke paper-specific analysis plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from .planning.analysis_types import get_analysis_type, infer_analysis_type
from .schema import (
    AnalysisPlan,
    AnalysisStep,
    ConceptDescriptor,
    ResearchContext,
    TimeWindow,
    VariableRole,
)


@dataclass
class ClinicalSkill:
    """A registered, named ICU analysis recipe."""

    key: str
    name: str
    description: str
    research_question_template: str
    target_outcome: Optional[str] = None
    primary_predictor: Optional[str] = None
    expected_variables: List[str] = field(default_factory=list)
    analysis_type_key: Optional[str] = None
    time_windows: List[TimeWindow] = field(default_factory=list)
    plan_factory: Optional[Callable[[ResearchContext], AnalysisPlan]] = None

    # ------------------------------------------------------------------

    def question_for(self, *, database: str) -> str:
        # ``database`` is required (no default) so no specific source key leaks
        # into a reusable skill's question text; the caller supplies it from the
        # active ResearchContext.database.
        return self.research_question_template.format(
            database=database, **self.__dict__
        )

    def validate_against(self, df: pd.DataFrame) -> List[str]:
        """Return a list of human-readable issues; empty list means OK."""
        issues: List[str] = []
        for v in self.expected_variables:
            if v not in df.columns:
                issues.append(
                    f"skill '{self.key}' expects variable '{v}' but the cohort has no such column"
                )
        return issues

    def plan(self, context: ResearchContext) -> AnalysisPlan:
        if self.plan_factory is not None:
            return self.plan_factory(context)
        return _default_skill_plan(self, context)


# ---------------------------------------------------------------------------
# Shared dynamic step-selection helpers
# ---------------------------------------------------------------------------


_COHORT_SUMMARY_HINTS = (
    "table 1",
    "table1",
    "baseline",
    "characteristics",
    "describe",
    "description",
    "descriptive",
    "summary",
    "summarise",
    "summarize",
    "demographic",
    "phenotype",
    "profile",
)

_OUTCOME_RATE_HINTS = (
    "incidence",
    "rate",
    "risk",
    "mortality",
    "death rate",
    "event rate",
    "frequency",
    "prevalence",
)

_ASSOCIATION_HINTS = (
    "associate",
    "association",
    "associated",
    "effect",
    "effects",
    "predict",
    "predicts",
    "prediction",
    "odds",
    "hazard",
    "risk factor",
    "linked",
    "relationship",
    "correlat",
)

_QUALITY_HINTS = (
    "missing",
    "missingness",
    "completeness",
    "data quality",
    "quality audit",
    "audit",
    "availability",
    "coverage",
    "provenance",
    "harmon",
    "mapping",
    "measurement",
)

_BINARY_OUTCOME_NAME_HINTS = (
    "death",
    "mortality",
    "readmission",
    "event",
    "failure",
)


def _question_text(context: ResearchContext) -> str:
    return (context.research_question or "").lower()


def _preference_rationale_note(context: ResearchContext) -> Optional[str]:
    prefs = context.user_preferences
    if prefs is None:
        return None
    fragments: List[str] = []
    if prefs.preferred_methods:
        fragments.append(f"Respect stated methods: {prefs.preferred_methods}.")
    if prefs.evaluation_focus:
        fragments.append(f"Prioritize evaluation focus: {prefs.evaluation_focus}.")
    if prefs.subgroup_sensitivity:
        fragments.append(
            f"Include subgroup/sensitivity requests: {prefs.subgroup_sensitivity}."
        )
    if prefs.timing_and_design:
        fragments.append(f"Honor timing/design constraints: {prefs.timing_and_design}.")
    if prefs.data_constraints:
        fragments.append(f"Honor data constraints: {prefs.data_constraints}.")
    if prefs.must_have_outputs:
        fragments.append(f"Must-have outputs: {prefs.must_have_outputs}.")
    if prefs.covariates:
        fragments.append(
            "User-specified covariates: " + ", ".join(prefs.covariates) + "."
        )
    if prefs.extra_notes:
        fragments.append(f"Additional user notes: {prefs.extra_notes}.")
    return " ".join(fragments) or None


def _contains_hint(text: str, hints: Sequence[str]) -> bool:
    def _present(hint: str) -> bool:
        token = (hint or "").strip().lower()
        if not token:
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in token):
            return token in text
        flexible = re.escape(token).replace(r"\ ", r"[\s_-]+")
        pattern = rf"(?<![a-z0-9]){flexible}(?![a-z0-9])"
        return re.search(pattern, text) is not None

    return any(_present(hint) for hint in hints)


def _candidate_descriptors(
    context: ResearchContext,
    candidate_variables: Optional[Sequence[str]] = None,
) -> List[ConceptDescriptor]:
    if candidate_variables:
        names = set(candidate_variables)
        return [v for v in context.variables if v.name in names]
    return list(context.variables)


def _looks_like_quality_only_question(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str],
    target_outcome: Optional[str],
) -> bool:
    text = _question_text(context)
    if not _contains_hint(text, _QUALITY_HINTS):
        return False
    if _contains_hint(text, _ASSOCIATION_HINTS) or _contains_hint(
        text, _OUTCOME_RATE_HINTS
    ):
        return False
    if primary_predictor and target_outcome:
        explicit_no_effect = (
            "do not estimate outcome effects" in text
            or "without estimating effects" in text
            or "without modelling outcomes" in text
        )
        if explicit_no_effect:
            return True
    return True


def _has_covariate_context(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str],
    target_outcome: Optional[str],
    candidate_variables: Optional[Sequence[str]] = None,
) -> bool:
    descriptive_roles = {
        VariableRole.DEMOGRAPHIC,
        VariableRole.VITAL,
        VariableRole.LAB,
        VariableRole.INTERVENTION,
        VariableRole.ORDINAL_SCORE,
        VariableRole.COMPOSITE_SCORE,
        VariableRole.OUTCOME,
    }
    descriptors = _candidate_descriptors(context, candidate_variables)
    others = [
        v
        for v in descriptors
        if v.name not in {primary_predictor, target_outcome}
        and v.role in descriptive_roles
    ]
    return len(others) >= 2


def should_include_table_one(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str],
    target_outcome: Optional[str],
    candidate_variables: Optional[Sequence[str]] = None,
) -> bool:
    text = _question_text(context)
    if _looks_like_quality_only_question(
        context,
        primary_predictor=primary_predictor,
        target_outcome=target_outcome,
    ):
        return False
    if _contains_hint(text, _COHORT_SUMMARY_HINTS):
        return True
    return bool(
        primary_predictor
        and target_outcome
        and _has_covariate_context(
            context,
            primary_predictor=primary_predictor,
            target_outcome=target_outcome,
            candidate_variables=candidate_variables,
        )
    )


def should_include_outcome_incidence(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str],
    target_outcome: Optional[str],
) -> bool:
    if not target_outcome:
        return False
    if _looks_like_quality_only_question(
        context,
        primary_predictor=primary_predictor,
        target_outcome=target_outcome,
    ):
        return False
    text = _question_text(context)
    if _contains_hint(text, _OUTCOME_RATE_HINTS):
        return True
    outcome_name = target_outcome.lower()
    return bool(
        primary_predictor
        and any(hint in outcome_name for hint in _BINARY_OUTCOME_NAME_HINTS)
    )


def should_include_missingness_audit(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str],
    target_outcome: Optional[str],
    candidate_variables: Optional[Sequence[str]] = None,
) -> bool:
    text = _question_text(context)
    if _contains_hint(text, _QUALITY_HINTS):
        return True
    for descriptor in _candidate_descriptors(context, candidate_variables):
        if descriptor.name in {None, ""}:
            continue
        miss = descriptor.missingness
        if miss and (
            miss.fraction_missing >= 0.20
            or miss.missingness_severity in {"medium", "high"}
        ):
            return True
        if (
            descriptor.missingness_semantics
            and "missing" in descriptor.missingness_semantics.lower()
        ):
            return True
        joined_pitfalls = " ".join(descriptor.pitfalls).lower()
        if any(
            token in joined_pitfalls
            for token in ("missing", "unmeasured", "measured", "mnar")
        ):
            return True
    return False


def should_include_primary_association(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str],
    target_outcome: Optional[str],
) -> bool:
    if not primary_predictor or not target_outcome:
        return False
    return not _looks_like_quality_only_question(
        context,
        primary_predictor=primary_predictor,
        target_outcome=target_outcome,
    )


def build_dynamic_core_plan_steps(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str],
    target_outcome: Optional[str],
    candidate_variables: Optional[Sequence[str]] = None,
    scope_label: str = "analysis",
    rationale_note: Optional[str] = None,
    analysis_type_key: Optional[str] = None,
) -> List[AnalysisStep]:
    variables = list(
        candidate_variables
        or [v.name for v in context.variables if v.role != VariableRole.ID]
    )
    steps: List[AnalysisStep] = []
    preference_note = _preference_rationale_note(context)
    combined_rationale = " ".join(
        part for part in (rationale_note, preference_note) if part
    )
    analysis_type = (
        get_analysis_type(analysis_type_key)
        if analysis_type_key
        else infer_analysis_type(
            context,
            primary_predictor=primary_predictor,
            target_outcome=target_outcome,
        )
    )

    if analysis_type.key == "data_quality_audit":
        return [
            AnalysisStep(
                step_id="03_missingness_audit",
                intent=(
                    f"Data-quality and completeness audit for variables relevant to the {scope_label}. "
                    "Keep this as an audit task; do not escalate it into an outcome-effect model."
                ),
                inputs=variables,
                expected_outputs=["table:missingness", "figure:missingness_heatmap"],
                method="missingness",
                icu_rule_refs=["missingness_kind"],
            )
        ]

    if analysis_type.key == "descriptive_epidemiology":
        steps.append(
            AnalysisStep(
                step_id="01_table_one",
                intent=(
                    f"Cohort summary (Table 1) for the {scope_label}, restricted to variables "
                    "that are relevant to the descriptive question."
                ),
                inputs=variables,
                expected_outputs=["table:table_one"],
                method="descriptive",
                icu_rule_refs=["aggregation_rule_for"],
            )
        )
        if target_outcome:
            steps.append(
                AnalysisStep(
                    step_id="02_outcome_incidence",
                    intent=f"Incidence of {target_outcome} in the {scope_label}.",
                    inputs=[target_outcome],
                    expected_outputs=[
                        "table:outcome_incidence",
                        "statistic:outcome_rate",
                    ],
                    method="incidence",
                )
            )
        if should_include_missingness_audit(
            context,
            primary_predictor=primary_predictor,
            target_outcome=target_outcome,
            candidate_variables=variables,
        ):
            steps.append(
                AnalysisStep(
                    step_id="03_missingness_audit",
                    intent=(
                        f"Missingness audit for the descriptive dataset used in the {scope_label}."
                    ),
                    inputs=variables,
                    expected_outputs=[
                        "table:missingness",
                        "figure:missingness_heatmap",
                    ],
                    method="missingness",
                    icu_rule_refs=["missingness_kind"],
                )
            )
        return steps

    if analysis_type.key in {"prediction_model", "trajectory_clustering"}:
        if should_include_table_one(
            context,
            primary_predictor=primary_predictor,
            target_outcome=target_outcome,
            candidate_variables=variables,
        ):
            steps.append(
                AnalysisStep(
                    step_id="01_table_one",
                    intent=(
                        f"Cohort summary (Table 1) for the {scope_label} to document the analytic population."
                    ),
                    inputs=variables,
                    expected_outputs=["table:table_one"],
                    method="descriptive",
                    icu_rule_refs=["aggregation_rule_for"],
                )
            )
        if target_outcome and should_include_outcome_incidence(
            context,
            primary_predictor=primary_predictor,
            target_outcome=target_outcome,
        ):
            steps.append(
                AnalysisStep(
                    step_id="02_outcome_incidence",
                    intent=f"Outcome incidence for {target_outcome} before the advanced {analysis_type.name.lower()} workflow.",
                    inputs=[target_outcome],
                    expected_outputs=[
                        "table:outcome_incidence",
                        "statistic:outcome_rate",
                    ],
                    method="incidence",
                )
            )
        if should_include_missingness_audit(
            context,
            primary_predictor=primary_predictor,
            target_outcome=target_outcome,
            candidate_variables=variables,
        ):
            steps.append(
                AnalysisStep(
                    step_id="03_missingness_audit",
                    intent=(
                        f"Missingness audit before the advanced {analysis_type.name.lower()} workflow."
                    ),
                    inputs=variables,
                    expected_outputs=[
                        "table:missingness",
                        "figure:missingness_heatmap",
                    ],
                    method="missingness",
                    icu_rule_refs=["missingness_kind"],
                )
            )
        if analysis_type.key == "prediction_model":
            steps.append(
                AnalysisStep(
                    step_id="04_prediction_model_analysis",
                    intent=(
                        "Develop and evaluate a deterministic ICU prediction model for "
                        f"{target_outcome or 'the configured target outcome'} with "
                        "explicit train/test separation, leakage safeguards, discrimination, and calibration."
                        + (f" {combined_rationale}" if combined_rationale else "")
                    ),
                    inputs=variables,
                    expected_outputs=[
                        "table:model_performance_train_test",
                        "table:model_coefficients",
                        "table:risk_predictions_test",
                        "figure:roc_curve",
                        "figure:calibration_curve",
                        "statistic:auc",
                    ],
                    method="prediction_model_analysis",
                    icu_rule_refs=["aggregation_rule_for", "missingness_kind"],
                )
            )
            steps.append(
                AnalysisStep(
                    step_id="05_publication_figure_generation",
                    intent=(
                        "Generate a claim-first, manuscript-ready publication figure for the "
                        "prediction-model analysis using easyicu.research_agent.publication_figures."
                    ),
                    inputs=variables,
                    expected_outputs=[
                        "figure:publication_figure",
                        "log:figure_contract",
                    ],
                    method="publication_figure_generation",
                )
            )
        else:
            steps.append(
                AnalysisStep(
                    step_id="04_trajectory_clustering_analysis",
                    intent=(
                        "Cluster patients by longitudinal physiology trajectories, summarise cluster stability "
                        "and compare mortality across clusters without collapsing the task into a simple association model."
                        + (f" {combined_rationale}" if combined_rationale else "")
                    ),
                    inputs=variables,
                    expected_outputs=[
                        "table:cluster_assignments",
                        "table:cluster_summary",
                        "table:cluster_outcomes",
                        "figure:trajectory_clusters",
                        "statistic:n_clusters",
                    ],
                    method="trajectory_clustering_analysis",
                    icu_rule_refs=["aggregation_rule_for", "missingness_kind"],
                )
            )
            steps.append(
                AnalysisStep(
                    step_id="05_publication_figure_generation",
                    intent=(
                        "Generate a claim-first, manuscript-ready publication figure for the "
                        "trajectory clustering analysis using easyicu.research_agent.publication_figures."
                    ),
                    inputs=variables,
                    expected_outputs=[
                        "figure:publication_figure",
                        "log:figure_contract",
                    ],
                    method="publication_figure_generation",
                )
            )
        return steps

    if analysis_type.key in {
        "survival",
        "dynamic_prediction",
        "treatment_response",
        "causal_inference",
        "reinforcement_learning",
        "multimodal",
        "validation",
    }:
        if should_include_table_one(
            context,
            primary_predictor=primary_predictor,
            target_outcome=target_outcome,
            candidate_variables=variables,
        ):
            steps.append(
                AnalysisStep(
                    step_id="01_table_one",
                    intent=(
                        f"Cohort summary (Table 1) for the {scope_label} to document the analytic population."
                    ),
                    inputs=variables,
                    expected_outputs=["table:table_one"],
                    method="descriptive",
                    icu_rule_refs=["aggregation_rule_for"],
                )
            )
        if target_outcome and should_include_outcome_incidence(
            context,
            primary_predictor=primary_predictor,
            target_outcome=target_outcome,
        ):
            steps.append(
                AnalysisStep(
                    step_id="02_outcome_incidence",
                    intent=f"Outcome incidence for {target_outcome} before the advanced {analysis_type.name.lower()} workflow.",
                    inputs=[target_outcome],
                    expected_outputs=[
                        "table:outcome_incidence",
                        "statistic:outcome_rate",
                    ],
                    method="incidence",
                )
            )
        if should_include_missingness_audit(
            context,
            primary_predictor=primary_predictor,
            target_outcome=target_outcome,
            candidate_variables=variables,
        ):
            steps.append(
                AnalysisStep(
                    step_id="03_missingness_audit",
                    intent=(
                        f"Missingness audit before the advanced {analysis_type.name.lower()} workflow."
                    ),
                    inputs=variables,
                    expected_outputs=[
                        "table:missingness",
                        "figure:missingness_heatmap",
                    ],
                    method="missingness",
                    icu_rule_refs=["missingness_kind"],
                )
            )
        steps.append(
            AnalysisStep(
                step_id=f"04_{analysis_type.key}_protocol",
                intent=(
                    f"Define the executable protocol for the {analysis_type.name.lower()} task, "
                    f"including only the candidate modules justified by the research question. "
                    + (f"{combined_rationale} " if combined_rationale else "")
                    + "Do not substitute a simple association model for this analysis family."
                ),
                inputs=variables,
                expected_outputs=[f"log:{analysis_type.key}_protocol"],
                method=f"{analysis_type.key}_protocol",
            )
        )
        return steps

    if should_include_table_one(
        context,
        primary_predictor=primary_predictor,
        target_outcome=target_outcome,
        candidate_variables=variables,
    ):
        steps.append(
            AnalysisStep(
                step_id="01_table_one",
                intent=(
                    f"Cohort summary (Table 1) for the {scope_label}, restricted to variables "
                    "that are relevant to the stated question and ICU-aware aggregation rules."
                ),
                inputs=variables,
                expected_outputs=["table:table_one"],
                method="descriptive",
                icu_rule_refs=["aggregation_rule_for"],
            )
        )

    if should_include_outcome_incidence(
        context,
        primary_predictor=primary_predictor,
        target_outcome=target_outcome,
    ):
        steps.append(
            AnalysisStep(
                step_id="02_outcome_incidence",
                intent=f"Incidence of {target_outcome} in the {scope_label}.",
                inputs=[target_outcome] if target_outcome else [],
                expected_outputs=["table:outcome_incidence", "statistic:outcome_rate"],
                method="incidence",
            )
        )

    if should_include_missingness_audit(
        context,
        primary_predictor=primary_predictor,
        target_outcome=target_outcome,
        candidate_variables=variables,
    ):
        steps.append(
            AnalysisStep(
                step_id="03_missingness_audit",
                intent=(
                    f"Missingness and data-quality audit for variables relevant to the {scope_label}. "
                    "Only run this audit when the question or the context suggests it matters."
                ),
                inputs=variables,
                expected_outputs=["table:missingness", "figure:missingness_heatmap"],
                method="missingness",
            )
        )

    if should_include_primary_association(
        context,
        primary_predictor=primary_predictor,
        target_outcome=target_outcome,
    ):
        steps.append(
            AnalysisStep(
                step_id="04_primary_association",
                intent=(
                    f"Estimate the association between {primary_predictor} and "
                    f"{target_outcome} using ICU-aware aggregation and time-window defaults."
                    + (f" {combined_rationale}" if combined_rationale else "")
                ),
                inputs=[primary_predictor, target_outcome],
                expected_outputs=[
                    "table:primary_association",
                    "figure:primary_association_curve",
                    "statistic:primary_or",
                ],
                # This deterministic fallback declares only the scientific
                # method family. The Planner/Coder still owns the concrete
                # estimator; an ``A_or_B`` placeholder is not a closed method
                # contract and must not be used to authorize effect outputs.
                method="association_analysis",
                icu_rule_refs=["aggregation_rule_for"],
            )
        )

    return steps


# ---------------------------------------------------------------------------
# Default plan factory — same heuristics as MockLLMClient's fallback planner
# ---------------------------------------------------------------------------


def _default_skill_plan(skill: ClinicalSkill, context: ResearchContext) -> AnalysisPlan:
    target_outcome = skill.target_outcome or _infer_context_target_outcome(context)
    primary_predictor = skill.primary_predictor or _infer_context_primary_predictor(
        context,
        target_outcome=target_outcome,
    )
    candidate_variables = skill.expected_variables or [
        v.name for v in context.variables if v.role != VariableRole.ID
    ]
    steps = build_dynamic_core_plan_steps(
        context,
        primary_predictor=primary_predictor,
        target_outcome=target_outcome,
        candidate_variables=candidate_variables,
        scope_label=f"skill '{skill.key}'",
        rationale_note="Respect the skill's declared time window(s) and expected variable set.",
        analysis_type_key=skill.analysis_type_key,
    )
    return AnalysisPlan(
        research_question=context.research_question,
        steps=steps,
        rationale=(
            f"Dynamically assembled plan for ClinicalSkill '{skill.key}'. "
            "The outer research loop stays fixed, but descriptive and audit "
            "steps are included only when the question or context justifies them."
        ),
    )


# ---------------------------------------------------------------------------
# Built-in skill registry
# ---------------------------------------------------------------------------


_REGISTRY: Dict[str, ClinicalSkill] = {}


def register_skill(skill: ClinicalSkill) -> None:
    if skill.key in _REGISTRY:
        raise ValueError(f"skill '{skill.key}' is already registered")
    _REGISTRY[skill.key] = skill


def get_skill(key: str) -> ClinicalSkill:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown ClinicalSkill '{key}'. Available: {sorted(_REGISTRY)}"
        ) from None


def list_skills() -> List[ClinicalSkill]:
    return list(_REGISTRY.values())


def _infer_context_target_outcome(context: ResearchContext) -> Optional[str]:
    if context.target_outcome:
        return context.target_outcome
    for descriptor in context.variables:
        if descriptor.role == VariableRole.OUTCOME:
            return descriptor.name
    return None


def _infer_context_primary_predictor(
    context: ResearchContext,
    *,
    target_outcome: Optional[str],
) -> Optional[str]:
    preferred_roles = (
        VariableRole.ORDINAL_SCORE,
        VariableRole.COMPOSITE_SCORE,
        VariableRole.INTERVENTION,
        VariableRole.LAB,
        VariableRole.VITAL,
    )
    for role in preferred_roles:
        for descriptor in context.variables:
            if descriptor.name == target_outcome:
                continue
            if descriptor.role == role:
                return descriptor.name
    for descriptor in context.variables:
        if descriptor.name == target_outcome or descriptor.role == VariableRole.ID:
            continue
        return descriptor.name
    return None


def _seed_builtin_skills() -> None:
    """Register analysis-family skills only.

    These keys describe reusable workflow shapes. They intentionally do not
    encode a specific clinical score, exposure, database, benchmark item, or
    endpoint. Concrete variable binding comes from the user's question and the
    ``ResearchContext`` for the current cohort.
    """
    register_skill(
        ClinicalSkill(
            key="association_analysis",
            name="Association analysis",
            description=(
                "Estimate an ICU-aware association between a context-selected "
                "predictor or exposure and outcome."
            ),
            research_question_template=(
                "Run an ICU-aware association analysis for the {database} cohort."
            ),
            analysis_type_key="association_study",
        )
    )
    register_skill(
        ClinicalSkill(
            key="prediction_model",
            name="Prediction model",
            description=(
                "Build and evaluate a clinical prediction model using the "
                "current cohort context."
            ),
            research_question_template=(
                "Build and evaluate an ICU prediction model for the {database} cohort."
            ),
            analysis_type_key="prediction_model",
        )
    )
    register_skill(
        ClinicalSkill(
            key="data_quality_audit",
            name="Data-quality audit",
            description=(
                "Audit missingness, completeness, and cohort data-quality issues "
                "before outcome analysis."
            ),
            research_question_template=(
                "Audit data quality and variable completeness for the {database} cohort."
            ),
            analysis_type_key="data_quality_audit",
        )
    )


_seed_builtin_skills()


__all__ = [
    "ClinicalSkill",
    "register_skill",
    "get_skill",
    "list_skills",
]
