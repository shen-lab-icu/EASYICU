"""ClinicalSkill registry — composable ICU recipes (M4-inspired).

M4 [1] frames clinical-research workflows as composable "skills" that
agents can invoke. We adopt the same idea but specialise it to the
EasyICU shape:

A :class:`ClinicalSkill` is a small object that, given a cohort
DataFrame, declares:

* the *target outcome* and primary predictor;
* the *time window* and *aggregation rules* the skill cares about;
* a stable list of expected variables (with light validation against
  the cohort);
* a pre-canned ``AnalysisPlan`` that the planner can fall through to
  when the user picks the skill instead of writing a free-form
  research question.

Skills make the user-facing API substantially cheaper — a researcher
who wants the canonical "admission-SOFA → ICU mortality" analysis
shouldn't need to spend tokens prompting a planner agent for it.
The skill emits a deterministic, reviewable plan and the rest of
the pipeline runs unchanged. The outer research loop is stable, but
the inner analysis steps are now assembled dynamically from the
research question and context instead of forcing the same descriptive
checks for every task.

References
----------
[1] M4: Infrastructure for AI-Assisted Clinical Research (MCP +
    clinical-skills tooling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from .analysis_types import get_analysis_type, infer_analysis_type
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
    target_outcome: str
    primary_predictor: str
    expected_variables: List[str]
    time_windows: List[TimeWindow] = field(default_factory=list)
    plan_factory: Optional[Callable[[ResearchContext], AnalysisPlan]] = None

    # ------------------------------------------------------------------

    def question_for(self, *, database: str = "miiv") -> str:
        return self.research_question_template.format(database=database, **self.__dict__)

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


def _contains_hint(text: str, hints: Sequence[str]) -> bool:
    return any(hint in text for hint in hints)


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
    if _contains_hint(text, _ASSOCIATION_HINTS) or _contains_hint(text, _OUTCOME_RATE_HINTS):
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
        v for v in descriptors
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
        if descriptor.missingness_semantics and "missing" in descriptor.missingness_semantics.lower():
            return True
        joined_pitfalls = " ".join(descriptor.pitfalls).lower()
        if any(token in joined_pitfalls for token in ("missing", "unmeasured", "measured", "mnar")):
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


def sofa_audit_variable(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str],
) -> Optional[str]:
    if primary_predictor and primary_predictor.lower() in {"sofa", "sofa2"}:
        return primary_predictor
    sofa_names = [v.name for v in context.variables if v.name.lower() in {"sofa", "sofa2"}]
    if len(sofa_names) == 1:
        return sofa_names[0]
    text = _question_text(context)
    if "sofa-2" in text or "sofa2" in text:
        for name in sofa_names:
            if name.lower() == "sofa2":
                return name
    if "sofa" in text:
        for name in sofa_names:
            if name.lower() == "sofa":
                return name
    return None


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
    variables = list(candidate_variables or [v.name for v in context.variables if v.role != VariableRole.ID])
    steps: List[AnalysisStep] = []
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
                        f"Missingness audit for the descriptive dataset used in the {scope_label}."
                    ),
                    inputs=variables,
                    expected_outputs=["table:missingness", "figure:missingness_heatmap"],
                    method="missingness",
                    icu_rule_refs=["missingness_kind"],
                )
            )
        return steps

    if analysis_type.key in {
        "prediction_model",
        "trajectory_clustering",
        "treatment_response",
        "causal_inference",
        "reinforcement_learning",
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
                        f"Missingness audit before the advanced {analysis_type.name.lower()} workflow."
                    ),
                    inputs=variables,
                    expected_outputs=["table:missingness", "figure:missingness_heatmap"],
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
                    + (f"{rationale_note} " if rationale_note else "")
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
                    + (f" {rationale_note}" if rationale_note else "")
                ),
                inputs=[primary_predictor, target_outcome],
                expected_outputs=[
                    "table:primary_association",
                    "figure:primary_association_curve",
                    "statistic:primary_or",
                ],
                method="logistic_regression_or_kaplan_meier",
                icu_rule_refs=["aggregation_rule_for"],
            )
        )

    sofa_var = sofa_audit_variable(context, primary_predictor=primary_predictor)
    if sofa_var and target_outcome:
        steps.append(
            AnalysisStep(
                step_id="05_sofa_zero_audit",
                intent=(
                    f"Stratum-level audit of {sofa_var}; flag the score==0 stratum "
                    "if its outcome rate exceeds the score==1 stratum (component-missingness signature)."
                ),
                inputs=[sofa_var, target_outcome],
                expected_outputs=["table:sofa_strata", "figure:sofa_strata_curve",
                                  "statistic:sofa_zero_anomaly"],
                method="stratified_incidence",
                icu_rule_refs=["sofa_pitfalls"],
            )
        )

    return steps


# ---------------------------------------------------------------------------
# Default plan factory — same heuristics as MockLLMClient's fallback planner
# ---------------------------------------------------------------------------


def _default_skill_plan(skill: ClinicalSkill, context: ResearchContext) -> AnalysisPlan:
    steps = build_dynamic_core_plan_steps(
        context,
        primary_predictor=skill.primary_predictor,
        target_outcome=skill.target_outcome,
        candidate_variables=skill.expected_variables,
        scope_label=f"skill '{skill.key}'",
        rationale_note="Respect the skill's declared time window(s) and expected variable set.",
        analysis_type_key="association_study",
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


_SHOCK_EXPECTED_VARIABLES = [
    "stay_id",
    "age",
    "sex",
    "death",
    "los_icu",
    "lactate_max_24h",
    "lactate_median_24h",
    "lactate_first_24h",
    "lactate_measured_24h",
    "hyperlactatemia_24h",
    "lactate_gt4_24h",
    "map_min_24h",
    "map_median_24h",
    "map_low_any_24h",
    "map_ge65_all_24h",
    "vaso_any_24h",
    "vaso_hours_24h",
    "norepi_equiv_max_24h",
]


def _lactate_map_vaso_plan(context: ResearchContext) -> AnalysisPlan:
    """Shock-physiology plan: lactate signal beyond MAP and vasopressors."""
    target_outcome = context.target_outcome or "death"
    steps = [
        AnalysisStep(
            step_id="01_table_one",
            intent=(
                "Describe the EasyICU shock physiology cohort with age, sex, "
                "mortality, lactate measurement, MAP summaries, vasopressor "
                "exposure and circulatory-failure markers."
            ),
            inputs=_SHOCK_EXPECTED_VARIABLES,
            expected_outputs=["table:table_one"],
            method="descriptive",
            icu_rule_refs=["aggregation_rule_for", "lactate_right_skew"],
        ),
        AnalysisStep(
            step_id="02_outcome_incidence",
            intent=(
                "Estimate hospital mortality overall and by early lactate "
                "measurement status because lactate missingness is clinically "
                "informative in ICU data."
            ),
            inputs=[target_outcome, "lactate_measured_24h"],
            expected_outputs=["table:outcome_incidence", "statistic:outcome_rate"],
            method="incidence_with_missingness_strata",
        ),
        AnalysisStep(
            step_id="03_missingness_audit",
            intent=(
                "Audit missingness for lactate, MAP, vasopressor and outcome "
                "variables, emphasizing that unmeasured lactate is not MCAR."
            ),
            inputs=_SHOCK_EXPECTED_VARIABLES,
            expected_outputs=["table:missingness", "figure:missingness_heatmap"],
            method="missingness",
        ),
        AnalysisStep(
            step_id="04_lactate_map_vaso_discordance",
            intent=(
                "Quantify clinically important discordance strata: elevated "
                "lactate despite minimum MAP >=65 mmHg, with and without "
                "vasopressor exposure, and report mortality in each stratum."
            ),
            inputs=[
                "lactate_max_24h",
                "map_min_24h",
                "vaso_any_24h",
                target_outcome,
            ],
            expected_outputs=[
                "table:shock_strata",
                "figure:lactate_map_vaso_heatmap",
                "statistic:discordance_mortality_gradient",
            ],
            method="stratified_incidence",
            icu_rule_refs=["time_window_first_24h", "vasopressor_confounding_by_indication"],
        ),
        AnalysisStep(
            step_id="05_primary_association",
            intent=(
                "Fit an adjusted association model for mortality using early "
                "lactate as the primary predictor with age, sex, MAP and "
                "vasopressor exposure as covariates; avoid causal language."
            ),
            inputs=[
                "lactate_max_24h",
                target_outcome,
                "age",
                "sex",
                "map_min_24h",
                "vaso_any_24h",
            ],
            expected_outputs=[
                "table:primary_association",
                "figure:primary_association_curve",
                "statistic:primary_or",
            ],
            method="logistic_regression_complete_case_or_missing_indicator",
            icu_rule_refs=["lactate_right_skew", "outcome_definition", "confounding_by_indication"],
        ),
        AnalysisStep(
            step_id="06_cross_database_protocol",
            intent=(
                "Emit a cross-database replication protocol for eICU/HiRID: "
                "concept mapping, time-window harmonisation, lactate/MAP/vaso "
                "aggregation rules and required audit checks."
            ),
            inputs=[
                "lactate_max_24h",
                "map_min_24h",
                "vaso_any_24h",
                target_outcome,
            ],
            expected_outputs=["table:cross_database_protocol", "log:replication_checklist"],
            method="replication_protocol",
            icu_rule_refs=["cross_database_validation"],
        ),
    ]
    return AnalysisPlan(
        research_question=context.research_question,
        steps=steps,
        rationale=(
            "Pre-canned EasyICU shock physiology skill. It uses the agent for "
            "code, interpretation and manuscript scaffolding, while pinning the "
            "clinical plan to ICU concept metadata: lactate is skewed and "
            "measurement-informed, MAP is windowed, and vasopressor exposure is "
            "an intervention marker rather than a causal treatment effect."
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


def _seed_builtin_skills() -> None:
    register_skill(ClinicalSkill(
        key="sofa_mortality",
        name="Admission SOFA → ICU mortality",
        description=(
            "Canonical association between admission-window SOFA / SOFA-2 score and "
            "ICU mortality, with the SOFA==0 stratum audit baked in."
        ),
        research_question_template=(
            "Is admission SOFA-2 score associated with ICU mortality in {database}?"
        ),
        target_outcome="death",
        primary_predictor="sofa2",
        expected_variables=["age", "sex", "sofa2", "death", "los_icu"],
        time_windows=[
            TimeWindow(name="first_24h", start_hours=0, end_hours=24,
                       rationale="Admission illness severity window."),
        ],
    ))
    register_skill(ClinicalSkill(
        key="aki_kdigo_mortality",
        name="KDIGO AKI stage → ICU mortality",
        description="KDIGO-defined AKI stage as a predictor of ICU mortality.",
        research_question_template=(
            "Is peak first-24h KDIGO AKI stage associated with ICU mortality in {database}?"
        ),
        target_outcome="death",
        primary_predictor="kdigo_stage",
        expected_variables=["age", "sex", "creat", "kdigo_stage", "death"],
        time_windows=[TimeWindow(name="first_24h", start_hours=0, end_hours=24)],
    ))
    register_skill(ClinicalSkill(
        key="vaso_exposure_mortality",
        name="Vasopressor exposure → ICU mortality",
        description=(
            "Any-vasopressor exposure within the first ICU window as a predictor of mortality."
        ),
        research_question_template=(
            "Is any-vasopressor exposure within the first 24 h associated with ICU mortality in {database}?"
        ),
        target_outcome="death",
        primary_predictor="vaso",
        expected_variables=["age", "sex", "vaso", "death", "map"],
        time_windows=[TimeWindow(name="first_24h", start_hours=0, end_hours=24)],
    ))
    register_skill(ClinicalSkill(
        key="lactate_trajectory_mortality",
        name="6-hour lactate trajectory → ICU mortality",
        description=(
            "Median lactate change within the first 6 hours as a predictor of mortality."
        ),
        research_question_template=(
            "Does the first-6-hour lactate trajectory predict ICU mortality in {database}?"
        ),
        target_outcome="death",
        primary_predictor="lact",
        expected_variables=["age", "sex", "lact", "death"],
        time_windows=[TimeWindow(name="first_6h", start_hours=0, end_hours=6)],
    ))
    register_skill(ClinicalSkill(
        key="lactate_map_vaso_shock_mortality",
        name="Lactate-MAP-vasopressor discordance → mortality",
        description=(
            "ICU shock physiology case: early lactate risk beyond apparent MAP "
            "adequacy and vasopressor exposure, with lactate missingness audited."
        ),
        research_question_template=(
            "In adult first ICU stays from {database}, does early lactate identify "
            "hospital mortality risk beyond MAP and vasopressor exposure?"
        ),
        target_outcome="death",
        primary_predictor="lactate_max_24h",
        expected_variables=[
            "stay_id",
            "age",
            "sex",
            "death",
            "los_icu",
            "lactate_max_24h",
            "lactate_median_24h",
            "lactate_first_24h",
            "lactate_measured_24h",
            "hyperlactatemia_24h",
            "lactate_gt4_24h",
            "map_min_24h",
            "map_median_24h",
            "map_low_any_24h",
            "map_ge65_all_24h",
            "vaso_any_24h",
            "vaso_hours_24h",
            "norepi_equiv_max_24h",
        ],
        time_windows=[
            TimeWindow(
                name="first_24h",
                start_hours=0,
                end_hours=24,
                rationale=(
                    "Admission resuscitation window for lactate, MAP and "
                    "vasopressor exposure."
                ),
            ),
        ],
        plan_factory=_lactate_map_vaso_plan,
    ))


_seed_builtin_skills()


__all__ = [
    "ClinicalSkill",
    "register_skill",
    "get_skill",
    "list_skills",
]
