"""Analysis-type registry for EHR/ICU research planning.

The research-agent layer needs a task-family abstraction that is more
stable than free-form prompts but less rigid than a single fixed
step list.

This module turns that need into an explicit registry the planner can
inspect. Each analysis type exposes:

* trigger terms for lightweight inference from a research question;
* a human-readable description;
* candidate steps that are common for that family but not mandatory;
* guardrails that help prompts stay honest about what the family does.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema import ResearchContext, VariableRole
from ..contracts.trajectory_design import REPEATED_OUTPUTS


@dataclass(frozen=True)
class AnalysisTypeSpec:
    key: str
    name: str
    description: str
    trigger_terms: Sequence[str]
    candidate_steps: Sequence[str]
    guardrails: Sequence[str]
    # Which ``EndpointSpec.kind`` a plan in this family must declare, or None
    # when the family imposes no requirement yet.
    #
    # MEASURED over 291 recorded runs: ``research_context.endpoint`` was null in
    # every single one. ``EndpointSpec`` has existed, with ``time_column``,
    # ``time_origin`` and ``censoring_rule`` fields and a validator that refuses
    # to infer any of them, and ``ResearchContextV2`` already verifies its
    # declared columns against the sealed cohort -- but no caller ever passed
    # one, so the type and its receipt were both dead.
    #
    # What filled the vacuum: the planner wrote the follow-up rule as prose in
    # ``icu_rule_refs``, differently each run (3 of 13 h1 plans wrote one at
    # all), and the generated code reached for whatever time column it could
    # find. Across 11 h1 runs with recovered source that produced SEVEN distinct
    # combinations of {los_icu, los_hosp, death_time, discharge_time, END_HOURS}.
    # The concept auditor then judged the code against its own reading of the
    # question and blocked the step for contradicting a "planner-required" or
    # "contract-required" rule that appears in no host artifact: `los_icu` is
    # absent from every h1 analysis plan, while the plan's own prose said
    # hospital discharge. 12 of the 29 scientific blocks on the five
    # never-passing tasks are this one missing declaration.
    #
    # Keyed on the family rather than on a question phrase on purpose. The
    # trigger-term scan below is what ROUTES a question to a family; once the
    # family is stamped on the plan it is a declaration, and a second keyword
    # pass over the same prose to re-derive what the family already implies is
    # how the routing layer grew to 1,395 lines. The requirement is compiled
    # once, here.
    required_endpoint_kind: Optional[str] = None
    # The scientific capability is an execution contract, not a display-family
    # inference.  It is declared here so readiness never has to reconstruct a
    # capability by passing through a second analysis-type -> display-family
    # mapping. ``None`` is an explicit unsupported boundary.
    capability_id: Optional[str] = None


#: Words that name a competing event or its model.  Each exists only over
#: follow-up time, so a question naming one asks a time-to-event analysis
#: (pilot1 10-26: 「…把活着转出 ICU 作为竞争事件」 was read as an association).
#: A bare cumulative incidence (「入 ICU 7 天内 AKI 的累积发生率」, "the
#: cumulative incidence by risk decile") is a proportion by a time, so it names
#: none of them.
COMPETING_EVENT_TERMS: Tuple[str, ...] = (
    "competing event",
    "competing events",
    "竞争事件",
    "fine-gray",
    "fine and gray",
    "subdistribution hazard",
)
#: A competing event the question declines to model ("not accounting for
#: competing events", 「不考虑竞争风险」): those words then name no design.
_DECLINED_COMPETING_EVENT = re.compile(
    r"(?:\b(?:not|never|without|ignor(?:e|es|ed|ing)|no)\b(?:[\s_-]+[a-z]+){0,3}?[\s_-]+"
    r"|(?:不考虑|未考虑|不计|忽略|无需考虑|不纳入|不处理)\s*)"
    r"(?:competing[\s_-]+(?:events?|risks?)|fine[\s_-]+(?:and[\s_-]+)?gray|subdistribution[\s_-]+hazards?"
    r"|竞争事件|竞争风险)",
    re.IGNORECASE,
)


_REGISTRY: Dict[str, AnalysisTypeSpec] = {
    "repeated_measures": AnalysisTypeSpec(
        key="repeated_measures", name="Repeated-measure outcome trajectories",
        description="Estimate change over ICU time in a repeated continuous or binary outcome using a reviewed mixed model or GEE.",
        trigger_terms=("repeated measures", "repeated measurements", "重复测量", "随时间变化", "纵向变化"),
        candidate_steps=("audit long-panel measurement density", "declare ICU time axis, window and aggregation",
                         "association.mixed_effects or association.gee", "death-dropout sensitivity and aggregate reporting"),
        guardrails=("Declare hours/days from ICU admission, window, bin width and measurement aggregation.",
                    "Use RepeatedMeasureSpec; select GEE working correlation in the reviewed plan.",
                    "Death may cause informative dropout: state the limitation and plan survivor restriction or a joint-model note.",
                    "Growth curves do not identify latent classes; LCGA/GBTM remains planned."),
        capability_id="repeated_measures_v1",
    ),
    "descriptive_epidemiology": AnalysisTypeSpec(
        key="descriptive_epidemiology",
        name="Descriptive epidemiology / cohort characterization",
        description=(
            "Summarize who is in the cohort, how frequent outcomes are, and "
            "how variables are distributed without making primary effect claims."
        ),
        trigger_terms=(
            "describe",
            "description",
            "characteristics",
            "baseline",
            "cohort",
            "incidence",
            "prevalence",
            "frequency",
            "burden",
            "描述",
            "患病率",
            "发生率",
            "分布",
        ),
        candidate_steps=(
            "cohort summary (Table 1)",
            "outcome incidence",
            "distribution summaries",
            "missing-data overview when relevant",
        ),
        guardrails=(
            "Do not imply causal or predictive effects from descriptive summaries.",
            "Use the study time anchor consistently when reporting incidence.",
        ),
        capability_id="descriptive_measurement_v1",
    ),
    "association_study": AnalysisTypeSpec(
        key="association_study",
        name="Association / prognostic modeling",
        description=(
            "Estimate associations between exposures, scores, or predictors and "
            "clinical outcomes with appropriate covariate adjustment."
        ),
        trigger_terms=(
            "associated",
            "association",
            "dose-response",
            "dose response",
            "predictor",
            "prognostic",
            "risk factor",
            "odds ratio",
            "hazard ratio",
            "linked",
            "relationship",
            "关联",
            "相关性",
            "剂量反应",
            "危险因素",
            "比值比",
        ),
        candidate_steps=(
            "cohort summary when confounding context matters",
            "outcome incidence for binary outcomes",
            "missing-data audit when incompleteness may bias inference",
            "primary adjusted association model",
            "score-specific or exposure-specific QC checks when relevant",
        ),
        guardrails=(
            "Avoid causal language unless the design is explicitly causal.",
            "Report effect estimates with uncertainty and analytic population.",
        ),
        capability_id="association_adjusted_v1",
    ),
    "ordinal_dose_response": AnalysisTypeSpec(
        key="ordinal_dose_response",
        name="Association / ordered exposure trend",
        description=(
            "Estimate an adjusted association for a declared ordered exposure, "
            "without inferring an ordinal scale from the column name or values."
        ),
        trigger_terms=("ordinal dose response", "ordered trend"),
        candidate_steps=(
            "declare ordered exposure levels and reference",
            "ordered-exposure quality audit",
            "primary adjusted trend model",
        ),
        guardrails=(
            "Require a declared ordinal exposure with at least three ordered levels.",
            "Do not coerce binary or continuous exposures into a graded trend.",
            "Avoid causal language unless the design is explicitly causal.",
        ),
        capability_id="association_ordinal_trend_v1",
    ),
    "prediction_model": AnalysisTypeSpec(
        key="prediction_model",
        name="Prediction model development / validation",
        description=(
            "Build or validate a clinical prediction model with explicit task "
            "definition, evaluation, and calibration."
        ),
        trigger_terms=(
            "predict",
            "prediction",
            "predictive",
            "预测",
            "判别",
            "校准",
            "early warning",
            "classifier",
            "model performance",
            "auroc",
            "auc",
            "calibration",
            "brier",
        ),
        candidate_steps=(
            "index-time and leakage audit",
            "feature audit and missing-data strategy",
            "train/validation/test or temporal split definition",
            "discrimination and calibration evaluation",
            "subgroup or external validation",
        ),
        guardrails=(
            "Do not substitute a simple association model for a full prediction workflow.",
            "Explicitly define the prediction horizon and anti-leakage rules.",
        ),
        capability_id="prediction_risk_model_v1",
    ),
    "survival": AnalysisTypeSpec(
        key="survival",
        name="Survival / time-to-event analysis",
        description=(
            "Model time-to-event outcomes with explicit time zero, censoring, "
            "and event definitions instead of collapsing them into a binary endpoint."
        ),
        trigger_terms=(
            "survival",
            "生存",
            "time-to-event",
            "time to event",
            "时间到事件",
            "cox",
            "kaplan",
            "kaplan-meier",
            "hazard",
            "competing risk",
            # Exact-token matching, so the singular never matched the ordinary
            # plural spelling ("a competing risks analysis").
            "competing risks",
            "竞争风险",
            # A competing event, a Fine-Gray model or a cumulative incidence
            # exists only over follow-up time: time-to-event by definition.
            *COMPETING_EVENT_TERMS,
            "censoring",
            "删失",
            "follow-up",
        ),
        candidate_steps=(
            "define time zero and follow-up window",
            "event / censoring audit",
            "Kaplan-Meier or cumulative-incidence summaries",
            "Cox or other time-to-event model",
            "sensitivity checks for censoring and competing risks",
        ),
        # This family's own description already says "explicit time zero,
        # censoring, and event definitions"; requiring the typed endpoint is
        # what makes that sentence checkable instead of aspirational.
        required_endpoint_kind="time_to_event",
        capability_id="survival_time_to_event_v1",
        guardrails=(
            "Do not reduce a time-to-event question to a fixed binary outcome unless the user explicitly asks for that simplification.",
            "Make the event definition, censoring mechanism, and follow-up horizon explicit.",
        ),
    ),
    "dynamic_prediction": AnalysisTypeSpec(
        key="dynamic_prediction",
        name="Dynamic prediction / early warning",
        description=(
            "Update risk estimates over time using longitudinal ICU data, with "
            "explicit prediction windows and refresh frequency."
        ),
        trigger_terms=(
            "dynamic prediction",
            "time-updated",
            "time updated",
            "early warning",
            "deterioration",
            "rolling risk",
            "horizon",
            "update frequency",
        ),
        candidate_steps=(
            "define prediction horizon and update cadence",
            "construct longitudinal feature slices",
            "anti-leakage audit for time-updated features",
            "dynamic discrimination and calibration evaluation",
            "temporal subgroup and drift checks",
        ),
        capability_id="dynamic_prediction_landmark_v1",
        guardrails=(
            "Do not treat longitudinal forecasting as a static prediction problem.",
            "Keep prediction time, observation window, and target horizon distinct.",
        ),
    ),
    "trajectory_clustering": AnalysisTypeSpec(
        key="trajectory_clustering",
        name="Trajectory clustering / phenotype discovery",
        description=(
            "Group patients using longitudinal patterns or multivariate states "
            "to discover clinically meaningful subphenotypes."
        ),
        trigger_terms=(
            "cluster",
            "clustering",
            "trajectory",
            "trajectories",
            "phenotype",
            "subphenotype",
            "longitudinal",
            "state sequence",
        ),
        candidate_steps=(
            "define time axis and panel structure",
            "longitudinal missingness audit",
            "feature representation or smoothing",
            "clustering and stability checks",
            "cluster characterization and outcome comparison",
        ),
        guardrails=(
            "Do not treat clusters as validated biology without robustness checks.",
            "Make time alignment explicit before comparing trajectories.",
        ),
        capability_id="phenotyping_cluster_v1",
    ),
    "multimodal": AnalysisTypeSpec(
        key="multimodal",
        name="Multimodal clinical modeling",
        description=(
            "Combine structured ICU data with notes, waveforms, or imaging while "
            "making modality alignment and missingness explicit."
        ),
        trigger_terms=(
            "multimodal",
            "notes",
            "waveform",
            "imaging",
            "image",
            "text modality",
            "clinical notes",
            "fusion",
            "modality",
        ),
        candidate_steps=(
            "define available modalities and alignment unit",
            "modality-specific preprocessing audit",
            "fusion strategy and ablation plan",
            "missingness and modality-dropout audit",
            "internal and external evaluation plan",
        ),
        guardrails=(
            "Do not claim multimodal support if only one modality is actually available in the cohort.",
            "Separate true multimodal modeling from structured-EHR-only baselines.",
        ),
    ),
    "validation": AnalysisTypeSpec(
        key="validation",
        name="External validation / score benchmarking",
        description=(
            "Validate an existing score or model, compare performance across cohorts, "
            "and quantify transportability."
        ),
        trigger_terms=(
            "external validation",
            "externally validate",
            "validate score",
            "benchmark",
            "compare score",
            "score comparison",
            "transportability",
            "reclassification",
            "net benefit",
        ),
        candidate_steps=(
            "define candidate scores / models and target cohort",
            "harmonize predictors and outcome definitions",
            "discrimination / calibration / reclassification evaluation",
            "subgroup transportability checks",
            "cross-database or temporal validation summary",
        ),
        guardrails=(
            "Do not present internal training performance as external validation.",
            "State clearly whether validation is temporal, geographic, or cross-database.",
        ),
    ),
    "treatment_response": AnalysisTypeSpec(
        key="treatment_response",
        name="Treatment-response / heterogeneity analysis",
        description=(
            "Characterize response patterns or heterogeneity around treatments "
            "without necessarily claiming a causal estimand."
        ),
        trigger_terms=(
            "treatment response",
            "response",
            "heterogeneity",
            "drug response",
            "therapy response",
            "responder",
            "nonresponder",
        ),
        candidate_steps=(
            "treatment definition and timing alignment",
            "confounding and selection-bias audit",
            "response summary",
            "heterogeneity analysis across clinically relevant strata",
        ),
        guardrails=(
            "Be explicit when the analysis is descriptive rather than causal.",
            "Intervention variables are often confounded by indication.",
        ),
    ),
    "causal_inference": AnalysisTypeSpec(
        key="causal_inference",
        name="Causal inference / target-trial emulation",
        description=("Estimate a treatment effect under an explicit causal design."),
        trigger_terms=(
            "causal",
            "因果",
            "treatment effect",
            "target trial",
            "propensity",
            "倾向评分",
            "ipw",
            "inverse probability",
            "逆概率加权",
            "g-formula",
            "instrumental variable",
            "do-calculus",
            "confounding by indication",
            "indication bias",
            "适应证混杂",
        ),
        candidate_steps=(
            "define target estimand and time zero",
            "eligibility and treatment strategy definition",
            "confounder set and positivity diagnostics",
            "balance diagnostics",
            "causal effect estimation and sensitivity analysis",
        ),
        guardrails=(
            "Do not present causal estimates without an explicit design.",
            "Check alignment of eligibility, treatment assignment, and follow-up.",
        ),
        capability_id="causal_target_trial_v1",
    ),
    "reinforcement_learning": AnalysisTypeSpec(
        key="reinforcement_learning",
        name="Sequential decision-making / reinforcement learning",
        description=(
            "Learn or evaluate dynamic treatment strategies over ICU trajectories."
        ),
        trigger_terms=(
            "reinforcement learning",
            "policy learning",
            "off-policy",
            "q-learning",
            "actor-critic",
            "dynamic treatment regime",
            "decision policy",
        ),
        candidate_steps=(
            "state / action / reward definition",
            "trajectory assembly and censoring audit",
            "behaviour-policy characterization",
            "off-policy evaluation",
            "safety-constrained policy learning",
        ),
        guardrails=(
            "Do not collapse sequential treatment problems into single-shot regression.",
            "Safety checks are not optional for clinical policies.",
        ),
    ),
    "data_quality_audit": AnalysisTypeSpec(
        key="data_quality_audit",
        name="Data quality / missingness audit",
        description=(
            "Audit measurement availability, units, ranges, and missingness "
            "without expanding into a full effect-estimation workflow."
        ),
        trigger_terms=(
            "missingness",
            "missing",
            "completeness",
            "coverage",
            "data quality",
            "quality audit",
            "availability",
            "schema",
            "unit check",
            "range check",
        ),
        candidate_steps=(
            "missingness profile",
            "concept coverage review",
            "unit / range sanity checks",
            "temporal or cross-source consistency checks",
        ),
        guardrails=(
            "Do not silently escalate a quality audit into an outcome model.",
            "Report what was audited and what was not available to audit.",
        ),
        capability_id="descriptive_measurement_v1",
    ),
    "measurement_bias_audit": AnalysisTypeSpec(
        key="measurement_bias_audit",
        name="Measurement bias / ascertainment audit",
        description=(
            "Audit whether measurement frequency, clinical indication, or care "
            "processes could bias an apparent association or phenotype."
        ),
        trigger_terms=(
            "measurement bias",
            "ascertainment",
            "testing frequency",
            "measurement frequency",
            "informative measurement",
            "selective measurement",
            "sampling bias",
            "monitoring bias",
            "indication bias",
        ),
        candidate_steps=(
            "define measured concept and observation opportunity",
            "measurement-frequency summary",
            "missingness-by-risk or missingness-by-care-process audit",
            "sensitivity analysis for informative measurement",
        ),
        guardrails=(
            "Do not interpret availability of a lab or score as a neutral random sample.",
            "Separate true physiology from who was selected to be measured.",
        ),
        capability_id="descriptive_measurement_v1",
    ),
    "cohort_definition_sensitivity": AnalysisTypeSpec(
        key="cohort_definition_sensitivity",
        name="Cohort definition sensitivity",
        description=(
            "Stress-test eligibility, timing, and diagnosis definitions to see "
            "whether a finding depends on a brittle cohort definition."
        ),
        trigger_terms=(
            "cohort definition",
            "eligibility criteria",
            "definition sensitivity",
            "case definition",
            "phenotype definition",
            "icd definition",
            "inclusion criteria",
            "exclusion criteria",
        ),
        candidate_steps=(
            "primary cohort definition",
            "alternative eligibility definitions",
            "overlap / attrition table",
            "sensitivity comparison across cohort definitions",
        ),
        guardrails=(
            "Do not hide cohort-definition changes inside post-hoc exclusions.",
            "Report how many patients move in or out under each definition.",
        ),
    ),
    "score_policy_sensitivity": AnalysisTypeSpec(
        key="score_policy_sensitivity",
        name="Score component / policy sensitivity",
        description=(
            "Evaluate how score component choices, missing-component handling, "
            "or imputation policies change score values or downstream claims."
        ),
        trigger_terms=(
            "score policy",
            "component policy",
            "missing component",
            "component missingness",
            "imputation policy",
            "worst value",
            "score sensitivity",
            "component sensitivity",
        ),
        candidate_steps=(
            "score component availability audit",
            "alternative component-policy definitions",
            "score distribution comparison",
            "downstream sensitivity check if an outcome is declared",
        ),
        guardrails=(
            "Do not treat a composite score as invariant to component handling.",
            "Name the exact missing-component and aggregation policy used.",
        ),
    ),
    "cross_database_replication": AnalysisTypeSpec(
        key="cross_database_replication",
        name="Cross-database replication / transportability",
        description=(
            "Replicate a cohort or analysis across ICU datasets with explicit "
            "concept mapping and harmonization notes."
        ),
        trigger_terms=(
            "replicate",
            "replication",
            "cross-database",
            "external validation",
            "transportability",
            "across mimic",
            "across eicu",
        ),
        candidate_steps=(
            "concept mapping",
            "cohort-harmonization checklist",
            "database-specific missingness comparison",
            "effect-size or performance comparison",
        ),
        guardrails=(
            "Do not assume columns with similar names are harmonized concepts.",
            "Separate feasibility/protocol notes from completed external results.",
        ),
    ),
}


_FAMILY_ALIASES: Dict[str, str] = {
    "prediction": "prediction_model",
    "prediction_model": "prediction_model",
    "clustering": "trajectory_clustering",
    "trajectory_clustering": "trajectory_clustering",
    "subphenotype_clustering": "trajectory_clustering",
    "subphenotype": "trajectory_clustering",
    "subphenotyping": "trajectory_clustering",
    "trajectory": "trajectory_clustering",
    "phenotyping": "trajectory_clustering",
    "phenotype": "trajectory_clustering",
    "association": "association_study",
    "association_study": "association_study",
    "descriptive": "descriptive_epidemiology",
    "descriptive_study": "descriptive_epidemiology",
    "descriptive_epidemiology": "descriptive_epidemiology",
    "ordinal_dose_response": "ordinal_dose_response",
    "association_ordinal": "ordinal_dose_response",
    "ordered_association": "ordinal_dose_response",
    "survival": "survival",
    "dynamic_prediction": "dynamic_prediction",
    "treatment_response": "treatment_response",
    "causal": "causal_inference",
    "causal_inference": "causal_inference",
    "reinforcement_learning": "reinforcement_learning",
    "rl": "reinforcement_learning",
    "multimodal": "multimodal",
    "validation": "validation",
    "external_validation": "validation",
    "data_quality": "data_quality_audit",
    "data_quality_audit": "data_quality_audit",
    "measurement_bias": "measurement_bias_audit",
    "measurement_bias_audit": "measurement_bias_audit",
    "ascertainment_bias": "measurement_bias_audit",
    "measurement_drift": "measurement_bias_audit",
    "cohort_sensitivity": "cohort_definition_sensitivity",
    "cohort_definition": "cohort_definition_sensitivity",
    "cohort_definition_sensitivity": "cohort_definition_sensitivity",
    "definition_sensitivity": "cohort_definition_sensitivity",
    "score_policy": "score_policy_sensitivity",
    "score_policy_sensitivity": "score_policy_sensitivity",
    "component_policy": "score_policy_sensitivity",
    "component_sensitivity": "score_policy_sensitivity",
    "imputation_policy": "score_policy_sensitivity",
    "cross_database_replication": "cross_database_replication",
}


# Families whose research shape is a CONCEPT SET rather than a predictor->outcome
# pair. Clustering/phenotyping discovers structure over a set of variables (no
# single outcome); descriptive epidemiology characterizes a set of concepts; a
# data-quality audit inspects a set of concepts. For these, idea-mining must NOT
# force a (predictor, outcome) tuple -- doing so is exactly what made clustering
# ideas resolve to predictor=None and get buried as db-cannot-do.
CONCEPT_SET_FAMILIES: frozenset[str] = frozenset(
    {
        "trajectory_clustering",
        "repeated_measures",
        "descriptive_epidemiology",
        "data_quality_audit",
        "measurement_bias_audit",
        "cohort_definition_sensitivity",
        "score_policy_sensitivity",
    }
)


def normalize_analysis_family(value: Optional[str]) -> str:
    """Map a free-text / benchmark family label to a canonical registry key.

    Unknown labels fall back to ``association_study`` (the predictor->outcome
    default), so callers always get a key that exists in the registry.
    """
    key = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if key in _REGISTRY:
        return key
    return _FAMILY_ALIASES.get(key, "association_study")


def canonical_analysis_family(value: Optional[str]) -> Optional[str]:
    """Resolve a declared family without the legacy association fallback.

    Display and discovery callers intentionally use
    :func:`normalize_analysis_family`, whose historical default is
    ``association_study``.  A planner declaration is an execution contract,
    though, so an unknown label must remain unknown and trigger structured
    retry instead of silently changing the scientific family.
    """

    key = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not key:
        return None
    if key in _REGISTRY:
        return key
    canonical = _FAMILY_ALIASES.get(key)
    return canonical if canonical in _REGISTRY else None


def is_concept_set_family(value: Optional[str]) -> bool:
    """Whether a family is shaped as a concept SET, not a predictor->outcome pair."""
    return normalize_analysis_family(value) in CONCEPT_SET_FAMILIES


def list_analysis_types() -> List[AnalysisTypeSpec]:
    return list(_REGISTRY.values())


def get_analysis_type(key: str) -> AnalysisTypeSpec:
    return _REGISTRY[key]


def optional_analysis_type_for_capability(
    capability_id: str,
) -> Optional[AnalysisTypeSpec]:
    """Return a unique subtype binding, or ``None`` for family-wide capabilities."""

    matches = [
        spec
        for spec in _REGISTRY.values()
        if spec.capability_id == str(capability_id).strip()
    ]
    if len(matches) > 1:
        raise ValueError(
            "scientific capability must map to exactly one analysis type: "
            f"{capability_id!r}"
        )
    return matches[0] if matches else None


def analysis_type_for_capability(capability_id: str) -> AnalysisTypeSpec:
    """Return the analysis subtype owned by one registered capability."""

    match = optional_analysis_type_for_capability(capability_id)
    if match is None:
        raise ValueError(
            "scientific capability must map to exactly one analysis type: "
            f"{capability_id!r}"
        )
    return match


def required_endpoint_kind_for_family(value: Optional[str]) -> Optional[str]:
    """The ``EndpointSpec.kind`` a plan in this family must declare.

    Reads the family's own registry entry. An unknown or unstamped family
    carries no requirement: a plan whose family could not be resolved is a
    different defect, and answering this question for it would be a guess.
    """

    key = canonical_analysis_family(value)
    if key is None:
        return None
    spec = _REGISTRY.get(key)
    return None if spec is None else spec.required_endpoint_kind


def _question_text(context: ResearchContext) -> str:
    parts = [(context.research_question or "").lower()]
    prefs = context.user_preferences
    if prefs is not None:
        parts.extend(
            [
                prefs.preferred_methods or "",
                prefs.evaluation_focus or "",
                prefs.subgroup_sensitivity or "",
                prefs.timing_and_design or "",
                prefs.data_constraints or "",
                prefs.must_have_outputs or "",
                prefs.extra_notes or "",
                " ".join(prefs.covariates or []),
            ]
        )
    return " ".join(part.lower() for part in parts if part).strip()


def _keyword_present(text: str, keyword: str) -> bool:
    kw = (keyword or "").strip().lower()
    if not kw:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in kw):
        return kw in text
    flexible = re.escape(kw).replace(r"\ ", r"[\s_-]+")
    pattern = rf"(?<![a-z0-9]){flexible}(?![a-z0-9])"
    return re.search(pattern, text) is not None


#: Words by which a question asks how often its exposure occurs in the study
#: population.  A question can ask that beside an association ("the proportion
#: of stays with X, and outcome Y by X"); the association may run on a narrower
#: population than the one the proportion is asked of, so the review checks
#: that this part of the question has its own step.  Latin cues match whole
#: words, so "proportional" never reads as "proportion".
_OCCURRENCE_CUES = (
    "proportion",
    "proportions",
    "percentage",
    "percentages",
    "prevalence",
    "incidence",
    "occurrence",
    "how common",
    "how frequent",
    "how frequently",
    "share of",
    "比例",
    "占比",
    "百分比",
    "构成比",
    "患病率",
    "发生率",
    "发病率",
    "检出率",
    "流行率",
)
#: Phrases that contain a cue but ask something else -- a model assumption
#: (proportional hazards or odds), a share of variance or of a mediated effect,
#: missingness, a matching, sampling or splitting ratio, a person-time rate, or
#: a time-to-event estimand.  They are removed before the cues are read.
_OCCURRENCE_GUARD_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"proportions?\s+(?:of\s+)?(?:the\s+)?(?:variance|variation|variability|effect|missing(?:ness)?)",
        r"proportions?\s+(?:mediated|explained|missing)",
        r"missing(?:ness)?\s+(?:proportion|percentage)s?",
        r"percentages?\s+of\s+missing",
        r"percentage\s+points?",
        r"cumulative\s+incidence",
        r"incidence\s+(?:rates?|density|densities|ratios?)",
        r"prevalence\s+ratios?",
    )
)
_OCCURRENCE_GUARD_PHRASES = (
    "比例风险",
    "等比例",
    "比例优势",
    "非比例",
    "比例假设",
    "缺失值比例",
    "缺失比例",
    "方差比例",
    "解释比例",
    "中介比例",
    "匹配比例",
    "比例匹配",
    "抽样比例",
    "划分比例",
    "比例划分",
    "累积发生率",
    "累积发病率",
)


def requested_exposure_occurrence_cues(context: ResearchContext) -> Tuple[str, ...]:
    """Return the words by which the question asks how often its exposure occurs.

    Only the research question and the requested outputs are read: covariate
    names, timing notes and data constraints describe the design, not what the
    question asks.  An empty result means the question does not ask for an
    occurrence estimate.  Whether one can be made -- a closed-level exposure, a
    family that reports it -- is the consumer's typed check, not this one.
    """

    prefs = context.user_preferences
    parts = [context.research_question or ""]
    if prefs is not None:
        parts.append(prefs.must_have_outputs or "")
    text = " ".join(part.lower() for part in parts if part)
    for pattern in _OCCURRENCE_GUARD_PATTERNS:
        text = pattern.sub(" ", text)
    for phrase in _OCCURRENCE_GUARD_PHRASES:
        text = text.replace(phrase, " ")
    return tuple(cue for cue in _OCCURRENCE_CUES if _keyword_present(text, cue))


#: Words by which a question asks how its outcome changes across graded levels
#: of its exposure.  A cue names the relationship itself: a bare "gradient"
#: also names physiology (the alveolar-arterial gradient), and a bare "trend" a
#: secular trend.  Latin cues match across a space, hyphen or underscore.
_DOSE_RESPONSE_CUES = (
    "dose response",
    "exposure response",
    "dose dependent",
    "dose dependence",
    "剂量反应",
    "剂量-反应",
    "剂量效应",
    "剂量依赖",
)


def requested_dose_response_cues(context: ResearchContext) -> Tuple[str, ...]:
    """Return the words by which the question asks for a dose-response relationship.

    Only the research question and the requested outputs are read, as for an
    occurrence.  Whether the exposure can carry a gradient -- three or more
    ordered levels, or a continuous scale -- is the consumer's typed check.
    """

    prefs = context.user_preferences
    parts = [context.research_question or ""]
    if prefs is not None:
        parts.append(prefs.must_have_outputs or "")
    text = " ".join(part.lower() for part in parts if part)
    # An en or em dash joins the two words as often as a hyphen does.
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return tuple(cue for cue in _DOSE_RESPONSE_CUES if _keyword_present(text, cue))



#: Words by which a question asks how its variables change over ICU time.
#: Trajectory clustering and cross-sectional phenotype discovery share one
#: analysis family; before a fixed-window design is compiled, only these words
#: tell a longitudinal question apart.  Latin cues match whole words, so
#: "haemodynamic" never reads as "dynamic".
_LONGITUDINAL_TRAJECTORY_CUES = (
    "trajectory",
    "trajectories",
    "longitudinal",
    "repeated measures",
    "repeated measurements",
    "time series",
    "time varying",
    "over time",
    "dynamic",
    "growth mixture",
    "latent class growth",
    "轨迹",
    "纵向",
    "重复测量",
    "时间序列",
    "随时间",
    "动态",
)


def longitudinal_trajectory_requested(context: ResearchContext) -> bool:
    """Whether the research question asks how its variables change over time.

    Only the question is read.  A cue does not make a design longitudinal; it
    keeps a template that clusters one feature vector per stay from claiming
    the question.  Reading a cue where none was meant only yields to the
    general planner, while missing one would answer a trajectory question
    with stay-level classes, so the cues are broad.
    """

    text = str(context.research_question or "").lower()
    return any(_keyword_present(text, cue) for cue in _LONGITUDINAL_TRAJECTORY_CUES)


_CLUSTERING_NUISANCE_PATTERNS = (
    re.compile(r"\bcluster[-\s]+robust\b", flags=re.IGNORECASE),
    re.compile(
        r"\bclustered\s+(?:standard\s+errors?|s\.?e\.?s?)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:hospital|site|centre|center|patient)[-\s]+level\s+clustering\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:account|adjust|control)(?:s|ed|ing)?\s+for\b.{0,64}"
        r"\bcluster(?:ed|ing)?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bclustering\s+(?:of|among)\s+patients?\s+within\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:gee|generalized\s+estimating\s+equations?|"
        r"generalised\s+estimating\s+equations?|mixed[-\s]+effects?)\b"
        r".{0,96}\b(?:cluster(?:ed|ing)?|within[-\s]+(?:hospital|site|"
        r"centre|center)|correlation)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"聚类\s*稳健\s*(?:标准误|方差|s\.?e\.?)?", flags=re.IGNORECASE),
    re.compile(
        r"(?:医院|中心|站点|机构|病区|患者)\s*层面(?:的)?\s*聚类",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:混合效应|广义估计方程|gee).{0,32}(?:聚类|组内相关)",
        flags=re.IGNORECASE,
    ),
)


_PREDICTION_REQUEST = re.compile(r"预测|预警|\bpredict\w*|\bforecast\w*", re.IGNORECASE)
_LANDMARK_UNIT = r"(?:小时|h|hours?|hrs?|天|日|d|days?)"
_LANDMARK_SEPARATOR = r"(?:\s*(?:、|,|，|/|和|及|与|或|以及)\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)"
_LANDMARK_NUMBER = r"\d{1,3}(?:\.\d)?"
#: Several prediction times measured on one clock: 「第 24、48、72 小时」,
#: 「在 24 和 48 小时时」, "at 24, 48 and 72 hours", "at hours 24, 48 and 72",
#: "on days 1, 2 and 3".
_PREDICTION_TIMES = re.compile(
    rf"(?:第|在|于|\b(?:at|on|by)\s+(?P<unit_first>(?:hours?|days?)\s+)?)\s*"
    rf"(?P<times>{_LANDMARK_NUMBER}(?:\s*{_LANDMARK_UNIT})?"
    rf"(?:{_LANDMARK_SEPARATOR}(?:第\s*)?{_LANDMARK_NUMBER}(?:\s*{_LANDMARK_UNIT})?){{1,7}})",
    re.IGNORECASE,
)
#: A prediction refreshed on a schedule: 「每 24 小时」「每天」, "every 24 hours".
_PREDICTION_SCHEDULE = re.compile(
    r"每\s*(?:\d{1,3}\s*(?:个)?\s*)?(?:小时|天|日)|\bevery\s+(?:\d{1,3}\s*)?(?:hours?|h|days?)\b",
    re.IGNORECASE,
)
#: The horizon that follows each prediction time: 「之后 7 天内」「接下来 24
#: 小时」, "over the next 7 days", "within the following 24 hours".
_HORIZON_AFTER = re.compile(
    r"(?:之后|此后|随后|接下来|其后|往后|未来|后续)\s*的?\s*"
    r"(?:\d{1,3}|[一二两三四五六七八九十]{1,3})\s*(?:个)?\s*(?:小时|h|天|日|周)"
    r"|\b(?:next|following|subsequent|ensuing|coming)\s+"
    r"(?:\d{1,3}|one|two|three|four|five|six|seven|ten|fourteen)[\s-]*"
    r"(?:hours?|hrs?|h|days?|d|weeks?)\b",
    re.IGNORECASE,
)


#: A time list or schedule that dates a measurement, not a prediction: "lactate
#: measured at 6 and 24 hours", "glucose measured every 4 hours",
#: 「第 6、24 小时测得的乳酸」「每 4 小时测量一次血糖」.
_MEASURED_BEFORE = re.compile(
    r"\b(?:measured|recorded|obtained|collected|sampled|drawn|charted|taken|assessed|"
    r"checked|tested|monitored|documented)\s+(?:\w+\s+)?$",
    re.IGNORECASE,
)
_MEASURED_AFTER = re.compile(
    r"^\s*(?:时|时点|时刻)?\s*(?:所|的)?\s*(?:测得|测量|检测|采集|记录|抽取|监测|采样)",
)
_MEASUREMENT_WORD = re.compile(
    r"\b(?:measured|measurements?|recorded|obtained|collected|sampled|drawn|charted)\b"
    r"|测得|测量|检测|采集|记录|监测|采样",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(r"[;；。！？!?]|\.(?=\s|$)")


def _dates_a_measurement(source: str, start: int, end: int) -> bool:
    """Whether the times at ``source[start:end]`` date a measurement, not the prediction.

    They do when a measurement word governs them ("measured at 6 and 24
    hours", 「第 6、24 小时测得的」), or when their own sentence states a
    measurement and no prediction ("Every 4 hours glucose was measured;
    predict death in the next 7 days").
    """

    if _MEASURED_BEFORE.search(source[max(0, start - 40):start]):
        return True
    if _MEASURED_AFTER.search(source[end:end + 12]):
        return True
    breaks = [match.end() for match in _CLAUSE_BREAK.finditer(source, 0, start)]
    clause_start = breaks[-1] if breaks else 0
    following = _CLAUSE_BREAK.search(source, end)
    clause = source[clause_start:following.start() if following else len(source)]
    return bool(_MEASUREMENT_WORD.search(clause)) and _PREDICTION_REQUEST.search(clause) is None


def landmark_prediction_framing(text: str) -> bool:
    """Whether ``text`` asks for one prediction made at several times.

    A dynamic (landmark) prediction states its prediction times on one clock
    -- two or more of them (「在入 ICU 第 24、48、72 小时」, "at 24 and 48 hours")
    or a schedule (「每 24 小时」, "every 24 hours") -- and a horizon that
    follows each (「预测之后 7 天内 ICU 死亡」, "over the next 24 hours").  A
    prediction from one time or one window ("use the first 24 hours to
    predict hospital death", 「在入 ICU 第 24 小时预测之后 7 天内死亡」) is a
    static prediction model, and listed horizons of one endpoint ("mortality
    at 24 and 48 hours") are not prediction times.
    """

    from easyicu.outcome_availability import mortality_horizon_spans

    source = str(text or "")
    if _PREDICTION_REQUEST.search(source) is None or _HORIZON_AFTER.search(source) is None:
        return False
    if any(
        not _dates_a_measurement(source, *match.span())
        for match in _PREDICTION_SCHEDULE.finditer(source)
    ):
        return True
    endpoint_horizons = mortality_horizon_spans(source)
    for match in _PREDICTION_TIMES.finditer(source):
        start, end = match.span("times")
        if any(begin < end and start < stop for begin, stop in endpoint_horizons):
            continue
        if _dates_a_measurement(source, match.start(), end):
            continue
        times = match.group("times")
        if not (match.group("unit_first") or re.search(_LANDMARK_UNIT, times, re.IGNORECASE)):
            continue
        if len(set(re.findall(_LANDMARK_NUMBER, times))) >= 2:
            return True
    return False


def strong_trajectory_clustering_framing(text: str) -> bool:
    """Return whether text asks to *discover groups*, not adjust SEs by group.

    A bare ``cluster`` token is not a task-family signal: it also appears in
    cluster-robust variance, clustered standard errors, and hospital/site-level
    dependence.  This predicate requires explicit phenotype-discovery language
    or a clustering algorithm/output contract.  It is shared by analysis-type,
    study-design, and plan-contract routing so those three layers cannot disagree.
    """

    normalised = str(text or "").strip().lower()
    if not normalised:
        return False
    # Existing group membership can be an exposure in an association model.  A
    # noun such as "subphenotype", "latent class", or "cluster assignment" is
    # therefore not sufficient: require an action that discovers/fits groups or
    # an unambiguous unsupervised procedure.  This keeps context inference aligned
    # with execution routing.
    discovery_action = (
        re.search(
            r"\b(?:discover|identify|derive|learn|uncover|fit|perform|run|select|"
            r"partition|group|cluster)\w*\b",
            normalised,
        )
        is not None
    )
    phenotype_target = (
        re.search(
            r"\b(?:sub[-\s]?phenotypes?|phenotypes?|phenotyping)\b",
            normalised,
        )
        is not None
    )
    generic_unsupervised_method = (
        re.search(
            r"\b(?:clustering|k[-_\s]?means|unsupervised|gmm|"
            r"gaussian[-_\s]+mixture(?:[-_\s]+models?)?|"
            r"latent[-_\s]+class(?:es)?[-_\s]+(?:analysis|models?|modeling))\b",
            normalised,
        )
        is not None
    )
    explicit_procedure = (
        re.search(
            r"\b(?:trajectory\s+clustering|unsupervised\s+clustering|"
            r"k[-_\s]?means(?:[-_\s]+clustering)?|gmm|"
            r"gaussian[-_\s]+mixture(?:[-_\s]+models?)?|"
            r"latent[-_\s]+class(?:es)?[-_\s]+(?:analysis|models?|modeling))\b",
            normalised,
        )
        is not None
    )
    cluster_action_target = (
        re.search(
            r"\b(?:cluster|partition|group)\s+(?:the\s+)?"
            r"(?:[a-z0-9_-]+\s+){0,3}(?:patients?|"
            r"trajectories?|longitudinal\s+(?:records?|profiles?|features?))\b"
            r".{0,64}\b(?:using|with|based\s+on|according\s+to|by)\b",
            normalised,
        )
        is not None
    )
    cluster_into_groups = (
        re.search(
            r"\bcluster\b.{0,96}\b(?:patients?|trajectories?|longitudinal\s+"
            r"(?:records?|profiles?|features?))\b.{0,96}\binto\s+(?:latent\s+)?"
            r"(?:classes|groups|clusters|phenotypes)\b",
            normalised,
        )
        is not None
    )
    imperative_trajectory_clustering = (
        re.search(
            r"(?:^|[.!?]\s+)(?:please\s+)?(?:cluster|partition|group)\s+"
            r"(?:[a-z0-9_-]+\s+){0,6}"
            r"(?:trajectories?|longitudinal\s+(?:records?|profiles?|features?))\b",
            normalised,
        )
        is not None
    )
    # A question also asks for discovery when the groups are the plural object
    # of a grouping verb: its profiles "cluster into distinct subgroups" or
    # "form reproducible sub-phenotypes", or it asks to "identify latent
    # classes" or "derive sub-phenotypes".  A singular or previously defined
    # group is still membership, which an association model may use.
    plural_groups = (
        r"sub[-\s]?phenotypes|phenotypes|sub[-\s]?types|endotypes|"
        r"latent\s+classes|clusters"
    )
    clusters_into_groups = (
        re.search(
            r"\bclust(?:er|ers|ered|ering)\s+into\s+(?:[a-z0-9_-]+\s+){0,3}"
            r"(?:" + plural_groups + r"|sub[-\s]?groups|groups|classes)\b",
            normalised,
        )
        is not None
    )
    profiles_form_groups = (
        re.search(
            r"\b(?:trajector(?:y|ies)|profiles?|patterns?|courses?|curves?|"
            r"time[-\s]+series)\b.{0,64}"
            r"\b(?:form|define|reveal|yield|constitute|separate)s?\s+"
            r"(?:into\s+)?(?:[a-z0-9_-]+\s+){0,3}(?:" + plural_groups + r")\b",
            normalised,
        )
        is not None
    )
    discovered_groups = (
        r"(?:latent[-\s]+(?:trajectory\s+)?class(?:es)?|sub[-\s]?phenotypes|"
        r"endotypes)"
    )
    groups_discovered = (
        re.search(
            r"\b(?:discover|identify|derive|learn|uncover|detect|find)\w*\b.{0,64}"
            r"\b" + discovered_groups + r"\b",
            normalised,
        )
        is not None
        or re.search(
            r"\b" + discovered_groups + r"\b.{0,64}\b(?:be|been|are|were|is)\s+"
            r"(?:derived|identified|discovered|learned|uncovered|detected|found)\b",
            normalised,
        )
        is not None
    )
    existing_groups = (
        re.search(
            r"\b(?:existing|previously|pre[-\s]?assigned|assigned|published|"
            r"established)\b.{0,48}"
            r"\b(?:cluster|class|phenotype|sub[-\s]?phenotype|subtype|endotype|"
            r"membership)",
            normalised,
        )
        is not None
    )
    # Trajectories divided into an unnamed number of classes ask for those
    # classes to be found ("into how many classes can the MAP trajectories be
    # divided"); trajectories split into named, rule-defined groups do not.
    classes_asked_for = (
        re.search(
            r"\btrajector(?:y|ies)\b[^.;?!]{0,80}?\b(?:divided|classified|grouped|"
            r"categori[sz]ed|separated|partitioned|split)\s+into\s+"
            r"(?:how\s+many|which|what|several|distinct|different|latent)\b"
            r"[^.;?!]{0,24}\b(?:classes|groups|types|subgroups|clusters|subtypes)\b"
            r"|\binto\s+(?:how\s+many|which|what)\s+(?:distinct\s+|latent\s+)?"
            r"(?:classes|groups|types|subgroups|clusters|subtypes)\b[^.;?!]{0,80}?"
            r"\btrajector(?:y|ies)\b[^.;?!]{0,80}?\b(?:divided|classified|grouped|"
            r"categori[sz]ed|separated|partitioned|split)\b"
            r"|\b(?:how\s+many|which|what)\s+(?:distinct\s+|latent\s+)?"
            r"(?:classes|types|subgroups|subtypes)\s+of\s+(?:[a-z0-9_-]+\s+){0,4}"
            r"trajector(?:y|ies)\b"
            # Phenotypes asked for by number or kind: "how many phenotypes
            # emerge", "which phenotypes can first-day values separate".
            r"|\b(?:how\s+many|which|what)\s+(?:distinct\s+|latent\s+)?"
            r"(?:sub[-\s]?phenotypes|phenotypes|subtypes|endotypes)\s+"
            r"(?:emerge|exist|can\s+be\s+(?:identified|derived|distinguished|separated|found))\b"
            r"|\b(?:separate|distinguish)\w*\s+(?:into\s+)?(?:how\s+many|which|what)\s+"
            r"(?:distinct\s+)?(?:sub[-\s]?phenotypes|phenotypes|subtypes|endotypes)\b",
            normalised,
        )
        is not None
    )
    plural_discovery_target = bool(
        (
            clusters_into_groups
            or profiles_form_groups
            or groups_discovered
            or classes_asked_for
        )
        and not existing_groups
    )
    chinese_discovery_disclaimer = (
        re.search(
            # Negate an action, not every word beginning with 不. In particular,
            # 不同 (different) and 不稳定 (unstable) describe discovery targets.
            r"(?:不(?:再|重新|继续)?(?:进行|作|做|开展|采用|使用|识别|发现|构建|"
            r"拟合|学习|划分)|不要|无需|不必|避免)"
            r"[^，。；;]{0,12}(?:患者)?(?:表型|亚型|轨迹|患者群)"
            r"[^，。；;]{0,8}(?:聚类|分群|识别|发现)?"
            r"|(?:不(?:再|重新|继续)?(?:进行|作|做|开展|采用|使用)|不要|无需|不必|避免)"
            r"[^，。；;]{0,8}(?:聚类|分群)"
            r"[^，。；;]{0,8}(?:患者)?(?:表型|亚型|轨迹|患者群)",
            normalised,
        )
        is not None
    )
    chinese_action_target = (
        re.search(
            r"(?:识别|发现|构建|拟合|学习|划分)"
            r"[^，。；;]{0,12}(?:患者)?(?:表型|亚型|轨迹|患者群)"
            r"[^，。；;]{0,6}(?:聚类|分群)?"
            r"|(?:患者)?(?:表型|亚型|轨迹|患者群)"
            r"[^，。；;]{0,8}(?:聚类|分群|识别|发现|构建)",
            normalised,
        )
        is not None
    )
    chinese_named_grouping = (
        re.search(
            r"(?:患者)?(?:表型|亚型|轨迹)[^，。；;]{0,4}(?:聚类|分群)",
            normalised,
        )
        is not None
    )
    # 聚成/聚为 name clustering itself, so profiles -- or the patients
    # themselves ("把脓毒症患者聚类成不同亚型", pilot1 10-22) -- that 聚成
    # groups ask for discovery even without an action verb such as 识别.
    # Patients split 分为 subtypes are discovered too; split into groups by a
    # measured value ("分为高乳酸组和低乳酸组") they are an exposure.
    chinese_clustered_into_groups = (
        re.search(
            r"(?:轨迹|特征|指标|模式|曲线|患者|病人)[^，。；;]{0,12}"
            r"(?:聚成|聚为|聚类为|聚类成|聚合为|聚合成)[^，。；;]{0,8}"
            r"(?:亚表型|亚型|表型|内型|类别|类|群|组)",
            normalised,
        )
        is not None
        or re.search(
            r"(?:患者|病人)[^，。；;]{0,12}(?:分成|分为|划分为|划分成)[^，。；;]{0,8}"
            r"(?:亚表型|亚型|表型|内型)",
            normalised,
        )
        is not None
    )
    # A question that asks how many, or which, phenotypes its features
    # separate into ("首日生命体征和化验能分出哪几种表型") asks for them to be
    # found; so do trajectories divided into an unnamed number of classes
    # ("轨迹可分为哪几类", "有哪几类轨迹") and a trajectory subtyping
    # ("轨迹分型").  Trajectories split into named groups ("分为上升组和下降组")
    # are defined by a rule, and an existing subtype ("已知的亚型") is
    # membership, neither discovered.
    chinese_existing_groups = (
        re.search(
            r"(?:已有|已知|既往|现有|已发表|已定义|预先定义|预设)[^，。；;？?]{0,8}"
            r"(?:表型|亚型|亚表型|内型|分型|类别)",
            normalised,
        )
        is not None
    )
    chinese_classes_found = (
        re.search(
            r"(?:分为|分成|分出|划分为|划分成|划分出|区分出|区分为|归为|聚为|聚成)\s*"
            r"(?:哪几|哪些|几|多少|若干)\s*(?:个|种|类)?\s*(?:亚表型|表型|亚型|内型)"
            r"|(?:哪几|哪些|几|多少)\s*(?:个|种|类)\s*(?:亚表型|表型|亚型|内型)"
            r"|轨迹[^，。；;？?]{0,8}(?:分为|分成|分出|划分为|划分成|归为|聚为|聚成)\s*"
            r"(?:哪几|哪些|几|多少|若干|不同的?)\s*个?\s*"
            r"(?:类别|类|种|型|组|亚型|模式)"
            r"|轨迹\s*(?:分型|亚型|类别|潜在类别)"
            r"|(?:哪几|哪些|几|多少)\s*(?:类|种|型)[^，。；;？?]{0,10}轨迹",
            normalised,
        )
        is not None
        and not chinese_existing_groups
    )
    chinese_discovery = bool(
        not chinese_discovery_disclaimer
        and (
            chinese_action_target
            or chinese_named_grouping
            or chinese_clustered_into_groups
            or chinese_classes_found
        )
    )
    chinese_explicit_discovery = bool(
        chinese_discovery
        and re.search(
            r"(?:识别|发现|构建|拟合|学习|划分)",
            normalised,
        )
    )
    has_nuisance = any(
        pattern.search(normalised) is not None
        for pattern in _CLUSTERING_NUISANCE_PATTERNS
    )
    explicit_discovery_target = (
        re.search(
            r"\b(?:discover|identify|derive|learn|uncover)\w*\b.{0,64}"
            r"\b(?:sub[-\s]?phenotypes?|phenotypes?|trajectory\s+clusters?|"
            r"latent[-\s]+classes?)\b",
            normalised,
        )
        is not None
    )
    english_discovery = (
        cluster_action_target
        or cluster_into_groups
        or imperative_trajectory_clustering
        or plural_discovery_target
        or (discovery_action and phenotype_target and generic_unsupervised_method)
        or (discovery_action and explicit_procedure)
        or (
            explicit_procedure
            and not has_nuisance
            and not re.search(
                r"\b(?:existing|previously|pre[-\s]?assigned|assigned)\b.{0,48}"
                r"\b(?:cluster|class|phenotype|membership)\b",
                normalised,
            )
        )
    )
    if has_nuisance:
        # Phrases such as "site-level clustering for patients" and
        # "cluster-robust longitudinal patient records" contain both halves of
        # otherwise useful discovery regexes.  In a nuisance-variance context,
        # require an unambiguous phenotype-discovery action plus an explicit
        # unsupervised procedure before allowing the family switch.
        return bool(
            (explicit_procedure and explicit_discovery_target)
            or chinese_explicit_discovery
        )
    return bool(english_discovery or chinese_discovery)


_SURVIVAL_ELIGIBILITY_PATTERNS = (
    # "the landmark row must require survival to 24 hours": the word names who
    # is *in the cohort*, not what is estimated. Landmark / immortal-time
    # guards are ordinary study setup and are written most often in questions
    # whose endpoint is a fixed binary outcome -- which is exactly when routing
    # to the survival family imposes an unsatisfiable contract.
    #
    # An eligibility marker is required in the same clause. A bare "survival to
    # 28 days" is a legitimate estimand and must keep its vote.
    re.compile(
        r"\b(?:requires?|required|requiring|restricts?|restricted|limits?|"
        r"limited|includes?|included|excludes?|excluded|eligibl\w*|"
        r"eligibility|must|only|landmark|conditional\s+on)\b"
        r"[^.;]{0,48}?\bsurviv(?:al|e|es|ed|ing)\b\s*"
        r"(?:to|until|through|beyond|past|at)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bsurviv(?:al|e|es|ed|ing)\b\s*(?:to|until|through|beyond|past|at)\b"
        r"[^.;]{0,48}?\b(?:required|requirement|landmark|"
        r"for\s+(?:inclusion|eligibility)|to\s+be\s+(?:included|eligible))\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:要求|限制|仅|只|纳入|排除|需)[^，。；;]{0,12}"
        r"(?:存活|生存)\s*(?:至|到|满|超过)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:存活|生存)\s*(?:至|到|满)[^，。；;]{0,12}"
        r"(?:者|的?患者|方可|才|纳入|入组)",
        flags=re.IGNORECASE,
    ),
)


def mask_survival_eligibility_phrases(text: str) -> str:
    """Blank spans where survival vocabulary names cohort membership.

    Returns ``text`` with each eligibility span replaced by spaces of equal
    length, so offsets are preserved and a *separate* survival term elsewhere
    in the same sentence still matches. This is deliberately a mask rather than
    a veto: a question that both restricts to 24-hour survivors *and* asks for
    a hazard ratio is still a survival question, and only the restriction
    clause loses its vote.
    """

    masked = str(text or "")
    for pattern in _SURVIVAL_ELIGIBILITY_PATTERNS:
        masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)
    return masked


def _preferred_family_key(context: ResearchContext) -> Optional[str]:
    prefs = context.user_preferences
    if prefs is None:
        return None
    candidates = [
        prefs.inferred_analysis_family,
        prefs.starter_template_key,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        # StudyContext persists the canonical registry key, while older entry
        # points may still supply one of the supported aliases.  Resolve both
        # through the same fail-closed owner used for explicit plan families.
        # Looking only in ``_FAMILY_ALIASES`` silently discarded canonical keys
        # that do not need an alias entry (notably ``descriptive_epidemiology``),
        # then re-inferred an association family from predictor/outcome prose.
        key = canonical_analysis_family(candidate)
        if key is not None:
            return key
    return None


_DEFINITION_CUE = (
    r"(?:cohort\s+definitions?|eligibility\s+criteri(?:a|on)|case\s+definitions?|"
    r"phenotype\s+definitions?|icd\s+definitions?|inclusion\s+criteri(?:a|on)|"
    r"exclusion\s+criteri(?:a|on)|eligibility\s+windows?|definitions?)"
)

_VARIATION_CUE = (
    r"(?:alternatives?|compare[sd]?|comparisons?|vary|varying|variants?|"
    r"different|sensitivity|robustness|across)"
)


def _cohort_definition_sensitivity_framing(text: str) -> bool:
    """Require an actual alternative-definition comparison, not workflow boilerplate.

    Benchmark questions routinely require one primary cohort to state inclusion
    and exclusion criteria. Those words describe normal study setup and must not
    override the scientific task family. Cohort-definition *sensitivity* needs
    either an explicit sensitivity phrase or both a definition/eligibility cue
    and a comparison/variation cue.
    """

    if any(
        _keyword_present(text, phrase)
        for phrase in (
            "cohort definition sensitivity",
            "definition sensitivity",
            "sensitivity across cohort definitions",
        )
    ):
        return True
    # The two cues must be *bound to each other*, not merely both present. As
    # two independent membership tests this fired on almost every task: every
    # analysis plan in this system is required to carry a robustness/sensitivity
    # step, so the variation half was effectively always true, and any question
    # that describes its own cohort supplies the definition half. A real run
    # routed "estimate prevalence and its association with mortality, with a
    # transparent, reproducible cohort definition" to this family because the
    # word "definition" came from the question and the word "sensitivity" came
    # from an unrelated line of the required-outputs list.
    #
    # Varying the cohort definition is the object of study only when the text
    # says so in one breath; otherwise it is a robustness component of some
    # other estimand.
    bound_variation = (
        re.search(
            rf"\b{_VARIATION_CUE}\b[^.;\n]{{0,48}}?\b{_DEFINITION_CUE}\b"
            rf"|\b{_DEFINITION_CUE}\b[^.;\n]{{0,48}}?\b{_VARIATION_CUE}\b",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )
    return bound_variation


def _treatment_response_framing(text: str) -> bool:
    """Keep severity/exposure dose-response language out of treatment routing."""

    if any(
        _keyword_present(text, phrase)
        for phrase in ("treatment response", "drug response", "therapy response")
    ):
        return True
    has_response = any(
        _keyword_present(text, phrase)
        for phrase in (
            "response",
            "heterogeneity",
            "responder",
            "responders",
            "nonresponder",
            "nonresponders",
        )
    )
    has_treatment = any(
        _keyword_present(text, phrase)
        for phrase in (
            "treatment",
            "therapy",
            "drug",
            "medication",
            "intervention",
            "administered",
        )
    )
    return has_response and has_treatment


# Shared question-wording policy. The host imports these cues rather than
# maintaining a second interpretation that can silently write a different design.
_CLAUSE = r"[^，,。；;.!?？！\n]"
_ASSOCIATION_WORD = r"(?:关系|关联|相关(?:性)?|有关)"
_DECLINED_ASSOCIATION = re.compile(
    r"\b(?:do\s+not|don't|not|without|no|avoid)\b" + _CLAUSE + r"{0,64}?"
    r"\b(?:associat\w*|relationships?|linked|related)\b"
    r"|(?:不(?:再)?(?:研究|分析|估计|评估|检验|探讨)|无需|不要|不必)"
    + _CLAUSE + r"{0,64}?" + _ASSOCIATION_WORD,
    re.IGNORECASE,
)
_OUT_OF_SCOPE_ASSOCIATION = re.compile(
    _CLAUSE + r"{0,96}?" + _ASSOCIATION_WORD + _CLAUSE + r"{0,48}?"
    r"(?:与本次任务无关|不属于本次(?:研究|任务)|不在本次(?:研究|任务)范围)"
    r"|" + _CLAUSE + r"{0,96}?\b(?:associat\w*|relationships?|linked|related)\b"
    + _CLAUSE + r"{0,64}?\b(?:outside\s+(?:the\s+)?scope|out\s+of\s+scope|"
    r"not\s+(?:in\s+scope|part\s+of\s+(?:this|the|our)\s+study))\b",
    re.IGNORECASE,
)
_DESCRIBED_ASSOCIATION = re.compile(
    r"\b(?:describe|summari[sz]e|show|compare)\b"
    r"(?:(?!\b(?:estimate|analy[sz]e|test|assess|evaluate|investigate)\b)"
    + _CLAUSE + r"){0,48}?"
    r"\b(?:associat\w*|relationships?|linked|related)\b"
    r"|(?:描述|展示|呈现|比较)(?:(?!(?:估计|分析|评估|检验|探讨|研究))"
    + _CLAUSE + r"){0,48}?" + _ASSOCIATION_WORD,
    re.IGNORECASE,
)
_INCIDENTAL_ASSOCIATION = re.compile(
    r"(?:关联|相关性)(?=的|于)"
    r"|\b(?:associated\s+with|(?:linked|related)\s+to)\s+"
    r"(?:(?:the|this|our|current)\s+)?(?:study|research|analysis|project)\b"
    r"|\b(?:records?|information|documents?|notes?)\s+(?:about|on|regarding|concerning)\b"
    + _CLAUSE + r"{0,96}",
    re.IGNORECASE,
)

RELATIONSHIP_QUESTION = re.compile(
    r"\b(?:is|are|was|were)\b" + _CLAUSE + r"{1,80}?\bassociated\s+with\b"
    r"|\b(?:is|are|was|were)\b" + _CLAUSE + r"{1,80}?\b(?:linked|related)\s+to\b"
    r"|\bassociations?\s+(?:between|of)\b" + _CLAUSE + r"{1,80}?\b(?:and|with)\b"
    r"|\brelationships?\s+between\b" + _CLAUSE + r"{1,80}?\band\b"
    r"|\brisk\s+factors?\s+for\b"
    r"|是否(?:与|和|同|跟)" + _CLAUSE + r"{1,64}?(?:相关|有关|关联)(?!的|于)"
    r"|(?:与|和|同|跟)" + _CLAUSE + r"{1,64}?是否(?:相关|有关|关联|存在关联)(?!的|于)"
    r"|(?:与|和|同|跟)" + _CLAUSE + r"{1,64}?(?:的关系|的关联|有何关系|有关系)(?!的|于)"
    r"|的危险因素",
    re.IGNORECASE,
)
PROPORTION_QUESTION = re.compile(
    r"\bwhat\s+(?:proportion|percentage|fraction|share)\b"
    r"|\b(?:prevalence|incidence|proportion)\s+of\b|\bhow\s+common\b"
    r"|多大比例|比例(?:是|为)?多少|占比|患病率|发生率|多常见",
    re.IGNORECASE,
)
INFERENTIAL_ESTIMATE = re.compile(
    r"\b(?:adjust(?:ed|ing|ment)?|regression|hazard|odds|causal(?:ly|ity)?|"
    r"effects?|predict(?:s|ed|ing|ion|ive)?)\b"
    r"|调整|校正|回归|风险比|优势比|比值比|因果|效应|预测",
    re.IGNORECASE,
)

# Read audit requests after masking exclusion, handling and population qualifiers.
_DATA_QUALITY_OBJECT = (
    r"(?:\b(?:missingness|completeness|data[\s_-]+quality|quality[\s_-]+audit|"
    r"availability|coverage|schema|unit[\s_-]+checks?|range[\s_-]+checks?|"
    r"missing(?:[\s_-]+values?)?[\s_-]+(?:rate|proportion))\b"
    r"|数据缺失|缺失(?:率|比例|模式|情况)|完整性|数据质量|可用性|覆盖率)"
)
_DATA_QUALITY_REQUEST = re.compile(
    r"\b(?:audit|assess|evaluate|report|quantify|estimate|describe|summari[sz]e|"
    r"check|measure|examine|profile|compare|calculate|list|tabulate|show|what|how)\b"
    + _CLAUSE + r"{0,96}?" + _DATA_QUALITY_OBJECT
    + r"|(?:评估|检查|审计|报告|统计|计算|描述|量化|比较|列出|汇总|展示|呈现|查看)" + _CLAUSE
    + r"{0,64}?" + _DATA_QUALITY_OBJECT
    + r"|" + _DATA_QUALITY_OBJECT + _CLAUSE + r"{0,48}?(?:多少|如何|怎样|哪些)",
    re.IGNORECASE,
)
_DATA_QUALITY_HANDLING = re.compile(
    r"(?:缺失|缺少)" + _CLAUSE + r"{0,32}?(?:插补|填补|处理)"
    r"|\bmissing(?:ness)?\b" + _CLAUSE
    + r"{0,48}?\b(?:imput(?:e|ed|ing|ation)|exclud(?:e|ed|ing)|handl(?:e|ed|ing))\b"
    r"|(?:排除|剔除|删除|去除|插补|填补|处理)" + _CLAUSE + r"{0,96}"
    r"|(?:不(?:再)?(?:研究|分析|评估|检查|审计|报告|统计|计算|列出|汇总|展示|呈现|查看)|无需|不要|不必)"
    + _CLAUSE + r"{0,96}"
    r"|\b(?:exclud(?:e|ed|ing)|drop(?:ped|ping)?|omit(?:ted|ting)?|"
    r"imput(?:e|ed|ing|ation)|handl(?:e|ed|ing))\b" + _CLAUSE + r"{0,96}"
    r"|\b(?:do\s+not|don't|not|without|avoid)\s+(?:audit|assess|evaluate|"
    r"report|quantify|estimate|check|measure|examine|profile|compare|calculate|list|tabulate|show)\b"
    + _CLAUSE + r"{0,96}",
    re.IGNORECASE,
)
# Coverage/completeness can qualify the population of a clinical summary.
_DATA_QUALITY_COHORT_QUALIFIER = re.compile(
    r"\b(?:stays?|patients?)\s+with\b" + _CLAUSE + r"{0,48}?" + _DATA_QUALITY_OBJECT
    + r"|" + _DATA_QUALITY_OBJECT + _CLAUSE + r"{0,16}?的(?:患者|住院|人群|队列)",
    re.IGNORECASE,
)
_MEASUREMENT_CONTEXT = (
    r"\b(?:across|among)\b"
    r"|\b(?:in|during|within|after|before|for)\s+(?:the\s+)?"
    r"(?:(?:first|last|past|initial|next|previous)\s+)?"
    r"(?:\d+(?:\.\d+)?\s*(?:hours?|hrs?|h|days?|d|weeks?|months?|minutes?|mins?)"
    r"|day|night|admission|stay|ICU|hospitalization)\b"
)
_MEASUREMENT_PROPORTION = re.compile(
    r"(?:\b(?:what\s+(?:proportion|percentage|fraction|share)|how\s+many)\b"
    + _CLAUSE + r"{0,64}?\b(?:are|is|were|was)\s+(?:missing|available|recorded|measured|documented)\b"
    r"|\bwhat\s+(?:proportion|percentage|fraction|share)\b"
    + _CLAUSE + r"{0,64}?\b(?:have|has|had)\s+missing\b"
    + _CLAUSE + r"{0,32}?\b(?:values?|measurements?|records?)\b"
    r"|\bwhat\s+(?:proportion|percentage|fraction|share)\b"
    + _CLAUSE + r"{0,64}?\b(?:have|has|had)\b" + _CLAUSE
    + r"{0,48}?\b(?:values?|measurements?|records?)\b"
    r"(?=\s*(?:[?.;!]|$|(?:available|recorded|measured|documented|in|during|within)\b)))"
    r"(?:(?!" + _MEASUREMENT_CONTEXT + r")" + _CLAUSE + r"){0,64}",
    re.IGNORECASE,
)
# Qualifiers and thresholds ask about a clinical state, not whether data exist.
# The measurement match stops before the time/population context, so a window
# such as "in the first 24 hours" is not mistaken for a clinical threshold.
_MEASUREMENT_CLINICAL_QUALIFIER = re.compile(
    r"\b(?:abnormal|elevated|raised|high|low|reduced|normal|hypo[\w-]*|hyper[\w-]*|"
    r"above|below|over|under|greater|less)\b|[<>=≤≥≠%]|\d"
    r"|\b(?:[munpµμ]?(?:mol|g|eq)(?:\s*/\s*(?:[md]?l|kg))?|mmhg|kpa|bpm)\b",
    re.IGNORECASE,
)
# Explicit predicates ask about the data themselves, even when the measured
# concept has a clinical qualifier. Modifiers such as "documented delirium"
# and "complete response" do not make those clinical states data-quality tasks.
_MEASUREMENT_DATA_OBJECT = re.compile(
    r"\b(?:data|values?|measurements?|records?|recordings?|documentation)\b",
    re.IGNORECASE,
)
_DATA_QUALITY_PREDICATE = re.compile(
    r"\b(?:have|has|had)\s+missing\b"
    + _CLAUSE + r"{0,32}?" + _MEASUREMENT_DATA_OBJECT.pattern
    + r"|\bhow\s+(?:complete|available|well\s+recorded)\s+(?:is|are|was|were)\b"
    + _CLAUSE + r"{0,64}?" + _MEASUREMENT_DATA_OBJECT.pattern
    + r"|\bhow\s+often\s+(?:is|are|was|were)\b" + _CLAUSE
    + r"{0,64}?\b(?:recorded|documented)\b"
    r"|\b(?:is|are|was|were)\b" + _CLAUSE
    + r"{0,64}?(?:\b(?:recorded|documented)\b|"
    + _MEASUREMENT_DATA_OBJECT.pattern + r"\s+available)\s+for\s+(?:all|every)\s+"
    r"(?:stays?|patients?|admissions?)\b"
    r"|(?:数据|记录|记载)(?:是否完整|完整吗|完整么)"
    r"|(?:有)?多少(?:住院|患者|入住|入院)" + _CLAUSE
    + r"{0,24}?(?:缺少|缺失)" + _CLAUSE + r"{0,24}?(?:记录|数据|记载|测量)"
    r"|多久(?:记录|记载)一次"
    r"|是否(?:在)?(?:所有|每(?:次|个)?)(?:住院|患者|入住)" + _CLAUSE
    + r"{0,24}?有" + _CLAUSE + r"{0,24}?(?:记录|数据|记载)",
    re.IGNORECASE,
)

# Missingness is a data question by default, including an unnamed measurement.
# Only an explicit anatomical subject makes missingness a clinical state.
_MISSINGNESS = re.compile(
    r"(?:\b(?:missing(?:ness)?|lack(?:s|ed|ing)?|absent|(?:not|no)\s+recorded)\b"
    r"|缺失|缺少|未记录)",
    re.IGNORECASE,
)
_MISSING_CLINICAL_SUBJECT = (
    r"(?:\b(?:teeth|tooth|limbs?|digits?|fingers?|toes?)\b|牙齿?|肢体|手指|脚趾)"
)
_MISSING_CLINICAL_STATE = re.compile(
    r"(?:" + _MISSINGNESS.pattern + r"\s*(?:(?:a|an|the)\s+)?"
    + _MISSING_CLINICAL_SUBJECT
    + r"|(?:" + _MISSING_CLINICAL_SUBJECT + r"|指|趾)"
    r"(?:\s+(?:is|are|was|were))?\s*(?:的)?" + _MISSINGNESS.pattern + r")"
    r"(?!\s*" + _MEASUREMENT_DATA_OBJECT.pattern + r")",
    re.IGNORECASE,
)
# A missing-data population is incidental when another predicate or a following
# clause supplies the clinical question. A bare count of that population is
# still a missingness audit (e.g. "乳酸缺失的患者占比是多少").
_MISSINGNESS_COHORT_QUALIFIER = re.compile(
    r"\b(?:stays?|patients?|admissions?)\s+(?:with|without|who)\b"
    + _CLAUSE + r"{0,48}?" + _MISSINGNESS.pattern + _CLAUSE + r"{0,48}?"
    r"(?=\s+(?:(?:have|has|had|develop(?:ed)?|died)\b|"
    r"(?:are|is|were|was)\s+(?!there\b))|[,，;；])"
    r"|" + _MISSINGNESS.pattern + _CLAUSE
    + r"{0,32}?的(?:患者|住院|人群|队列)(?:中|内)",
    re.IGNORECASE,
)


_DECLINED_ANALYSIS_START = re.compile(
    r"\b(?:do\s+not|don't|without|avoid|no|not)\b(?!\s+only\b)"
    r"|(?:不(?:再)?(?:做|作|进行|开展|采用|使用|需要|要求|估计|评估|研究|分析|生成)|"
    r"无需|不要|不必|避免|禁止|不得)",
    re.IGNORECASE,
)
_DECLINABLE_ANALYSIS_METHOD = re.compile(
    r"\b(?:predict\w*|forecast\w*|causal\w*|regression|adjust(?:ed|ing|ment)?|"
    r"statistical\s+inference|confidence\s+intervals?|significance\s+tests?|"
    r"hypothesis\s+tests?|survival|cox|kaplan(?:[- ]meier)?|propensity|"
    r"target\s+trial|iptw|ipw)\b"
    r"|统计推断|置信区间|显著性检验|假设检验|回归|调整|校正|预测|因果|"
    r"倾向评分|目标试验|逆概率加权|生存分析|时间到事件",
    re.IGNORECASE,
)
# The negation must directly govern a method, possibly through a narrow
# method-action phrase. Population verbs such as exclude/include never qualify.
_DECLINED_METHOD_PREFIX = re.compile(
    r"\s*(?:(?:build|develop|fit|use|perform|run|conduct|draw|make|estimate|"
    r"report|calculate)\s+)?(?:(?:a|an|the|any)\s+)?"
    r"(?:(?:进行|开展|采用|使用|建立|构建|估计|生成|报告)(?:任何)?)?",
    re.IGNORECASE,
)
_ANALYSIS_SCOPE_BOUNDARY = re.compile(
    r"[,，。；;.!?？！\n]|\b(?:but|however|instead|then|rather)\b"
    r"|但是|但|然而|不过|而是|而要|而需|只需要|只需|只做|只描述",
    re.IGNORECASE,
)
_ANALYSIS_LIST_JOIN = re.compile(r"\s*(?:(?:and|or)\b\s*)?", re.IGNORECASE)
# A comma may continue a list of declined methods, but a verb starts a new
# request ("without adjustment, predict mortality"). Chinese method lists use
# enumeration commas (、), already inside the original clause.
_ANALYSIS_LIST_METHOD = re.compile(
    r"(?:prediction|forecasting|predictive\s+model(?:ing|ling|s)?|"
    r"causal\s+(?:inference|analysis|estimation)|regression|adjustment|"
    r"statistical\s+inference|confidence\s+intervals?|significance\s+tests?|"
    r"hypothesis\s+tests?|survival\s+analysis|propensity\s+(?:scores?|weighting)|"
    r"target\s+trial(?:s|\s+emulation)?)\b",
    re.IGNORECASE,
)


_ANALYSIS_NOUN_LIST = re.compile(
    _ANALYSIS_LIST_METHOD.pattern
    + r"(?:\s+(?:and|or)\s+" + _ANALYSIS_LIST_METHOD.pattern + r")*\s*",
    re.IGNORECASE,
)


def mask_declined_analysis_phrases(text: str) -> str:
    """Blank explicitly declined analyses, preserving original evidence offsets.

    A negation governs its clause and comma-separated method list, including
    Chinese enumeration commas. Sentence boundaries and contrastive requests
    end its scope; a later positive request must still select its own family.
    The method must follow the negation or a listed method-action verb directly;
    an unrelated population exclusion never governs a later analysis request.
    """

    masked = list(text)
    for negation in _DECLINED_ANALYSIS_START.finditer(text):
        start = negation.start()
        boundary = _ANALYSIS_SCOPE_BOUNDARY.search(text, negation.end())
        end = boundary.start() if boundary else len(text)
        method_start = _DECLINED_METHOD_PREFIX.match(text, negation.end()).end()
        if not _DECLINABLE_ANALYSIS_METHOD.match(text, method_start, end):
            continue
        while boundary and boundary.group(0) == ",":
            item_start = _ANALYSIS_LIST_JOIN.match(text, boundary.end()).end()
            next_boundary = _ANALYSIS_SCOPE_BOUNDARY.search(text, item_start)
            item_end = next_boundary.start() if next_boundary else len(text)
            # "prediction should use ..." is a positive clause, not a noun
            # enumeration under the earlier negation.
            if not _ANALYSIS_NOUN_LIST.fullmatch(text, item_start, item_end):
                break
            boundary = next_boundary
            end = item_end
        masked[start:end] = " " * (end - start)
    return "".join(masked)


def mask_non_association_phrases(text: str) -> str:
    """Blank declined, described and incidental relationships, preserving offsets.

    Masks stop at clause boundaries so a separate requested association still
    counts. Both the host's evidence and the kernel's catalog scoring use this.
    """

    for pattern in (
        _OUT_OF_SCOPE_ASSOCIATION, _DECLINED_ASSOCIATION,
        _DESCRIBED_ASSOCIATION, _INCIDENTAL_ASSOCIATION,
    ):
        text = pattern.sub(lambda match: " " * len(match.group(0)), text)
    return text


def data_quality_question(question: str) -> bool:
    """Whether the question asks about data quality as an object of study.

    Callers pass only research_question, never preferences or column metadata.
    """

    question = _DATA_QUALITY_HANDLING.sub("", question)
    question = _MISSINGNESS_COHORT_QUALIFIER.sub("", question)
    question = _MISSING_CLINICAL_STATE.sub("", question)
    if _MISSINGNESS.search(question):
        return True
    question = _DATA_QUALITY_COHORT_QUALIFIER.sub("", question)
    if _DATA_QUALITY_REQUEST.search(question) or _DATA_QUALITY_PREDICATE.search(question):
        return True
    return any(
        _MEASUREMENT_DATA_OBJECT.search(match.group(0))
        and not _MEASUREMENT_CLINICAL_QUALIFIER.search(match.group(0))
        for match in _MEASUREMENT_PROPORTION.finditer(question)
    )


_EXTRA_TRIGGER_CUES = {"association_study": ("risk factors",)}


#: How :func:`infer_analysis_type_with_basis` chose a family: the study's
#: typed design, the question's words, supplementary context, or fallback.
FAMILY_BASIS_STUDY_DESIGN = "study_design"
FAMILY_BASIS_KEYWORD = "keyword"
FAMILY_BASIS_CONTEXT = "context"
FAMILY_BASIS_FALLBACK = "fallback"


def _proportion_question(text: str) -> bool:
    """Whether a question asks how common something is, and nothing inferential."""

    return (
        PROPORTION_QUESTION.search(text) is not None
        and INFERENTIAL_ESTIMATE.search(text) is None
    )


def infer_analysis_type(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str] = None,
    target_outcome: Optional[str] = None,
) -> AnalysisTypeSpec:
    return infer_analysis_type_with_basis(
        context, primary_predictor=primary_predictor, target_outcome=target_outcome
    )[0]


def infer_analysis_type_with_basis(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str] = None,
    target_outcome: Optional[str] = None,
) -> Tuple[AnalysisTypeSpec, str]:
    """The family a study is planned on, and how it was chosen.

    The study's typed design decides when it states one
    (:data:`FAMILY_BASIS_STUDY_DESIGN`); otherwise the question's words
    (:data:`FAMILY_BASIS_KEYWORD`) or supplementary fields
    (:data:`FAMILY_BASIS_CONTEXT`), and for a question no family's words read,
    the fallback rule (:data:`FAMILY_BASIS_FALLBACK`): a pair of outcome and
    exposure is an association, any other question is described, and only an
    empty question is a data-quality audit.
    """

    spec, basis = _infer_analysis_type_with_basis(
        context, primary_predictor=primary_predictor, target_outcome=target_outcome
    )
    if basis != FAMILY_BASIS_KEYWORD:
        return spec, basis
    # Ask the same routing policy with only the question. Metadata/preferences
    # get a context basis when they changed the winning family or supplied all
    # of its cues; a keyword in a losing family's score is not its provenance.
    question_context = context.model_copy(update={
        "user_preferences": None, "variables": [], "primary_exposure": None,
        "target_outcome": None, "cross_database_validation": [],
    })
    question_spec, question_basis = _infer_analysis_type_with_basis(question_context)
    return spec, (
        FAMILY_BASIS_KEYWORD
        if question_spec.key == spec.key and question_basis == FAMILY_BASIS_KEYWORD
        else FAMILY_BASIS_CONTEXT
    )


def _infer_analysis_type_with_basis(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str] = None,
    target_outcome: Optional[str] = None,
) -> Tuple[AnalysisTypeSpec, str]:
    preferred = _preferred_family_key(context)
    if preferred is not None:
        return _REGISTRY[preferred], FAMILY_BASIS_STUDY_DESIGN
    text = mask_declined_analysis_phrases(_question_text(context))
    primary_question_text = mask_declined_analysis_phrases(
        (context.research_question or "").lower().strip()
    )
    association_text = mask_non_association_phrases(text)
    relationship_question = RELATIONSHIP_QUESTION.search(
        mask_non_association_phrases(primary_question_text)
    ) is not None
    quality_question = data_quality_question(primary_question_text)
    primary_predictor = primary_predictor or context.primary_exposure
    target_outcome = target_outcome or context.target_outcome
    # Free-text preferences are supplementary design constraints, not a second
    # primary research question.  In particular, an ordinary association study
    # may request alternate cohort definitions as a sensitivity analysis.  If
    # those preference words are allowed to select the execution family, the
    # robustness analysis replaces the estimand it was meant to challenge.
    #
    # A cohort-definition comparison therefore owns the primary family only
    # when it is stated in ``research_question`` itself (or when the caller has
    # supplied the typed ``inferred_analysis_family`` authority handled above).
    # The merged preference text remains available below for method and output
    # routing, but it cannot silently promote an adjunct into the study family.
    cohort_sensitivity_framed = _cohort_definition_sensitivity_framing(
        primary_question_text
    )
    treatment_response_framed = _treatment_response_framing(text)
    # A bare "causal" anywhere in the text asserts the causal family, so the
    # disclaimer that cancels it must be at least as easy to say as the
    # assertion. It was not: the negation had to be a "not/avoid/without" word
    # within 40 characters, and it was matched case-sensitively while the
    # assertion beside it is case-insensitive. A guardrail reading "label the
    # estimand as observational rather than causal" therefore *selected* the
    # causal family and imposed a seven-role target-trial contract on a
    # prevalence question -- the instruction not to be causal was the only
    # reason it became causal.
    #
    # Contrastive negation ("rather than", "instead of", "as opposed to") and
    # the "non-causal" compound are ordinary English ways to disclaim, so they
    # belong here rather than in another family-specific gate. This only
    # cancels the bare-keyword branch: a question that names a causal method
    # (target trial, propensity, IPTW, ...) still routes to the causal family
    # through explicit_causal_method_framing below, which is what keeps this
    # from silently disarming genuine causal work.
    causal_disclaimer = (
        re.search(
            r"\b(?:do\s+not|don't|not|avoid|without)\b.{0,40}\bcausal(?:ity|ly)?\b"
            r"|\b(?:rather\s+than|instead\s+of|as\s+opposed\s+to)\b"
            r".{0,30}\bcausal(?:ity|ly)?\b"
            r"|\bnon[-_]?causal\b"
            r"|\bcausal\s+(?:claim|conclusion|interpretation)\b.{0,24}\b"
            r"(?:not|unsupported|avoid)\b"
            r"|(?:不(?:作|做|进行|用于|支持|解释为?)|避免|无意).{0,24}因果"
            r"|因果.{0,16}(?:不成立|不支持|不解释)"
            r"|(?:而非|不是|并非).{0,12}因果"
            r"|非因果"
            r"|(?:不得|禁止|拒绝).{0,24}因果",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )
    explicit_causal_method_framing = any(
        _keyword_present(text, term)
        for term in (
            "treatment effect",
            "target trial",
            "propensity",
            "ipw",
            "iptw",
            "inverse probability",
            "g-formula",
            "instrumental variable",
            "covariate balance",
            "positivity",
            "weighted estimate",
            "confounding by indication",
            "indication bias",
            "适应证混杂",
            "倾向评分",
            "逆概率加权",
        )
    )
    strong_causal_framing = explicit_causal_method_framing or (
        not causal_disclaimer
        and any(_keyword_present(text, term) for term in ("causal", "因果"))
    )
    survival_disclaimer = (
        re.search(
            r"\b(?:do\s+not|don't|not|avoid|without)\b.{0,40}"
            r"\b(?:survival|time[-\s]+to[-\s]+event|cox|kaplan)\b"
            r"|(?:不(?:作|做|进行|采用|使用)|避免|无需).{0,20}"
            r"(?:生存分析|时间到事件|cox|kaplan)",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )
    # Survival vocabulary spent on cohort eligibility does not get a vote. A
    # real run routed a prevalence-and-association question to the survival
    # family on a single occurrence of the word, inside the clause "the
    # landmark row must require survival to 24 hours" -- a required-outputs
    # guardrail describing who is in a cohort variant. The research question
    # itself never mentioned survival, the target outcome was a binary death
    # flag, and the Planner was then required to produce a survival curve for
    # it; five attempts failed in five different ways and nothing executed.
    survival_estimand_text = _DECLINED_COMPETING_EVENT.sub(
        lambda match: " " * len(match.group(0)), mask_survival_eligibility_phrases(text)
    )
    strong_survival_framing = (not survival_disclaimer) and any(
        _keyword_present(survival_estimand_text, term)
        for term in (
            "survival",
            "生存",
            "time-to-event",
            "time to event",
            "时间到事件",
            "cox",
            "kaplan",
            "kaplan-meier",
            "hazard",
            "competing risk",
            # Exact-token matching, so the singular never matched the ordinary
            # plural spelling ("a competing risks analysis").
            "competing risks",
            "竞争风险",
            *COMPETING_EVENT_TERMS,
            "censoring",
            "删失",
        )
    )
    # A mean survival time (「28 天内平均存活多少天」, "RMST") is a
    # time-to-event estimand, whatever else the question says.
    from easyicu.outcome_availability import stated_mean_survival_mentions

    strong_survival_framing = strong_survival_framing or (
        not survival_disclaimer and bool(stated_mean_survival_mentions(survival_estimand_text))
    )

    def _has_any(key: str, extras: Iterable[str] = ()) -> bool:
        if key == "data_quality_audit":
            return quality_question
        terms = [
            *_REGISTRY[key].trigger_terms,
            *_EXTRA_TRIGGER_CUES.get(key, ()),
            *extras,
        ]
        selected_text = association_text if key == "association_study" else text
        return (
            key == "association_study" and relationship_question
        ) or any(_keyword_present(selected_text, term) for term in terms)

    # Strong, explicit task-family cues should win before softer scoring.
    if _has_any("reinforcement_learning"):
        return _REGISTRY["reinforcement_learning"], FAMILY_BASIS_KEYWORD
    # A question explicitly framed as latent-class / trajectory clustering is a
    # descriptive discovery task even when it mentions "causal" in a disclaimer.
    # The bare "causal" cue must not hijack it into the causal-emulation family
    # (which then imposes an unsatisfiable causal contract). Gate the causal
    # family behind the ABSENCE of strong clustering framing. The cues below are
    # ones a genuine causal-emulation task never uses as its primary framing.
    # Bare "cluster" is excluded because it collides with cluster-robust and
    # clustered-standard-error language.
    _strong_clustering_framing = strong_trajectory_clustering_framing(text)
    if strong_causal_framing and not _strong_clustering_framing:
        return _REGISTRY["causal_inference"], FAMILY_BASIS_KEYWORD
    if _strong_clustering_framing:
        return _REGISTRY["trajectory_clustering"], FAMILY_BASIS_KEYWORD
    if repeated_measure_question(primary_question_text) and not landmark_prediction_framing(primary_question_text):
        return _REGISTRY["repeated_measures"], FAMILY_BASIS_KEYWORD
    if strong_survival_framing:
        return _REGISTRY["survival"], FAMILY_BASIS_KEYWORD
    # A prediction made at several times, each over the horizon that follows
    # it, is a dynamic prediction in any wording (pilot1 10-27: 「在入 ICU 第
    # 24、48、72 小时…预测之后 7 天内 ICU 死亡」 was planned as one static model).
    if _has_any("dynamic_prediction") or landmark_prediction_framing(primary_question_text):
        return _REGISTRY["dynamic_prediction"], FAMILY_BASIS_KEYWORD
    if _has_any("multimodal"):
        return _REGISTRY["multimodal"], FAMILY_BASIS_KEYWORD
    if _has_any("validation"):
        return _REGISTRY["validation"], FAMILY_BASIS_KEYWORD
    if _has_any(
        # NOTE: do NOT add the bare word "model" here. As a strong pre-scoring
        # cue it false-fires on association/descriptive questions that merely say
        # "model X continuously" or "regression model". The real prediction cues
        # (predict/AUROC/calibration/Brier/...) are already in trigger_terms.
        "prediction_model",
        extras=("evaluation metric", "evaluation metrics"),
    ):
        return _REGISTRY["prediction_model"], FAMILY_BASIS_KEYWORD
    if _has_any("measurement_bias_audit"):
        return _REGISTRY["measurement_bias_audit"], FAMILY_BASIS_KEYWORD
    if cohort_sensitivity_framed and _has_any("cohort_definition_sensitivity"):
        return _REGISTRY["cohort_definition_sensitivity"], FAMILY_BASIS_KEYWORD
    if _has_any("score_policy_sensitivity"):
        return _REGISTRY["score_policy_sensitivity"], FAMILY_BASIS_KEYWORD
    if (
        _has_any("data_quality_audit")
        and not cohort_sensitivity_framed
        and not any(
            _has_any(key)
            for key in (
                "association_study",
                "prediction_model",
                "causal_inference",
                "trajectory_clustering",
                "reinforcement_learning",
                "measurement_bias_audit",
                "score_policy_sensitivity",
            )
        )
    ):
        return _REGISTRY["data_quality_audit"], FAMILY_BASIS_KEYWORD
    if treatment_response_framed and _has_any("treatment_response"):
        return _REGISTRY["treatment_response"], FAMILY_BASIS_KEYWORD

    scores: Dict[str, int] = {key: 0 for key in _REGISTRY}
    for key, spec in _REGISTRY.items():
        if key == "data_quality_audit":
            scores[key] = int(quality_question)
            continue
        selected_text = association_text if key == "association_study" else text
        for term in (*spec.trigger_terms, *_EXTRA_TRIGGER_CUES.get(key, ())):
            if _keyword_present(selected_text, term):
                scores[key] += 1
    if relationship_question and not scores["association_study"]:
        # A paired-question pattern fills a catalog vocabulary gap (相关/有关,
        # related to); it must not double-score the same English association.
        scores["association_study"] += 1
    if _proportion_question(primary_question_text):
        scores["descriptive_epidemiology"] += 1
    if not cohort_sensitivity_framed:
        scores["cohort_definition_sensitivity"] = 0
    if not treatment_response_framed:
        scores["treatment_response"] = 0
    if not strong_causal_framing:
        scores["causal_inference"] = 0
    if not strong_survival_framing:
        # ``follow-up`` alone describes ascertainment for many fixed binary
        # endpoints and must not switch the estimand to time-to-event.
        scores["survival"] = 0

    if target_outcome and any(v.role == VariableRole.TIME for v in context.variables):
        scores["prediction_model"] += (
            1
            if any(_keyword_present(text, term) for term in ("predict", "forecast"))
            else 0
        )
        scores["trajectory_clustering"] += (
            1 if _keyword_present(text, "trajectory") else 0
        )
        scores["dynamic_prediction"] += (
            1
            if any(
                _keyword_present(text, term)
                for term in ("dynamic", "rolling", "updated", "over time")
            )
            else 0
        )
    if not _strong_clustering_framing:
        # The registry still contains broad catalog/search terms such as
        # ``cluster`` and ``trajectory``. They may aid documentation lookup, but
        # cannot score an execution family without the strong task framing above.
        scores["trajectory_clustering"] = 0

    if any(v.role == VariableRole.TIME for v in context.variables):
        scores["survival"] += (
            1
            if any(
                _keyword_present(text, term)
                for term in ("survival", "kaplan", "cox", "censor", "time-to-event")
            )
            else 0
        )
    # Word-boundary match, not substring: bare ``in`` made short modality tokens
    # like "ct" (CT scan) fire inside ordinary lab names — "ct" is a substring of
    # "la(ct)ate" — wrongly scoring a plain association cohort as multimodal.
    has_multimodal_variable = any(
        _keyword_present(((v.name or "") + " " + (v.description or "")).lower(), token)
        for v in context.variables
        for token in (
            "note",
            "notes",
            "text",
            "waveform",
            "ecg",
            "image",
            "imaging",
            "cxr",
            "ct",
            "mri",
        )
    )
    if has_multimodal_variable:
        scores["multimodal"] += 2
    if context.cross_database_validation:
        scores["validation"] += 1
        scores["cross_database_replication"] += 2

    if (
        (target_outcome or relationship_question)
        and _has_any("association_study") and not has_multimodal_variable
    ):
        # An explicit association/effect question is stronger evidence than the
        # mere presence of an outcome column.  More specialised families have
        # already returned above, so this only prevents the generic descriptive
        # fallback from winning a tie.
        scores["association_study"] += 3
    # Exposure/outcome columns also define grouped counts and rates. Their
    # presence must not outrank an explicit descriptive question. Keep the
    # legacy pair-based fallback below only when no task-family cue exists.
    if target_outcome and not primary_predictor:
        scores["descriptive_epidemiology"] += 1

    best_key = max(scores, key=scores.get)
    if scores[best_key] == 0:
        if primary_predictor and target_outcome:
            best_key = "association_study"
        elif target_outcome or primary_question_text:
            best_key = "descriptive_epidemiology"
        else:
            best_key = "data_quality_audit"
        return _REGISTRY[best_key], FAMILY_BASIS_FALLBACK
    return _REGISTRY[best_key], FAMILY_BASIS_KEYWORD


#: Detail levels for :func:`planner_analysis_type_guide`, most complete first.
#:
#: The catalog is a menu: its job is to let the Planner *choose* a family.  The
#: chosen family's modules and guardrails are restated in full by
#: :func:`locked_analysis_type_guide`, so a shortened catalog still leaves the
#: inferred family fully specified -- what it costs is detail on the families
#: the Planner might switch TO.  That is why the ladder shortens rather than
#: drops entries: every family stays selectable at every level.
#:
#: Measured 2026-07-30 on the real catalog (16 entries, 8,046 bytes):
#: ``full`` 8,046, ``without_guardrails`` 5,631 (-2,415), ``names_only`` 2,504
#: (-5,542).
CATALOG_DETAIL_LADDER: Tuple[str, ...] = (
    "full",
    "without_guardrails",
    "names_only",
)


def planner_analysis_type_guide(*, detail: str = "full") -> str:
    """Short prompt block for the planner.

    The guide intentionally presents task families and candidate steps,
    not mandatory recipes.

    ``detail`` selects a rung of :data:`CATALOG_DETAIL_LADDER`.  Callers under
    transport pressure descend it instead of failing the whole task; nothing
    else may vary it, because a catalog that changes with anything other than
    the byte budget would make the Planner's menu depend on a hidden decision.
    """
    if detail not in CATALOG_DETAIL_LADDER:
        raise ValueError(
            f"unknown analysis-type catalog detail {detail!r}; "
            f"expected one of {CATALOG_DETAIL_LADDER}"
        )
    lines = [
        "ANALYSIS-TYPE CATALOG:",
        "First infer the task family from the research question. Treat the entries below as candidate modules, not mandatory fixed recipes.",
    ]
    for spec in list_analysis_types():
        entry = f"- {spec.key}: {spec.description}"
        if detail != "names_only":
            entry += f" Common modules: {', '.join(spec.candidate_steps)}."
        if detail == "full":
            entry += f" Guardrails: {' '.join(spec.guardrails)}"
        lines.append(entry)
    lines.append(
        "Choose only the steps justified by the task family and available context. "
        "Do not force Table 1, outcome incidence, missingness, or score-specific QC unless they serve the question."
    )
    if detail != "full":
        lines.append(
            "This catalog was shortened to fit the request budget; ask for a "
            "family's guardrails through the inferred-family block above rather "
            "than assuming it has none."
        )
    return "\n".join(lines)


def planner_analysis_type_switch_guide(*, detail: str = "full") -> str:
    """Budget-aware compact menu for switching away from the inferred family.

    ``locked_analysis_type_guide`` already publishes the inferred family's
    modules and guardrails. Repeating that level of detail for every alternate
    family cost every Planner request more than five kilobytes. This menu keeps
    every family and its scientific description selectable while the separate
    action catalog carries method-level detail for the active family. Every
    rung retains every family; only alternate-family prose is shortened.
    """

    if detail not in CATALOG_DETAIL_LADDER:
        raise ValueError(
            f"unknown analysis-type switch-menu detail {detail!r}; "
            f"expected one of {CATALOG_DETAIL_LADDER}"
        )

    if detail == "names_only":
        return (
            "ANALYSIS TYPES: "
            + ",".join(spec.key for spec in list_analysis_types())
            + ". Switch only for a better estimand; explain."
        )
    lines = ["ANALYSIS-TYPE SWITCH MENU (all families remain selectable):"]
    for spec in list_analysis_types():
        if detail == "full":
            lines.append(f"- {spec.key}: {spec.description}")
        elif detail == "without_guardrails":
            lines.append(f"- {spec.key}: {spec.name}")
        else:  # pragma: no cover - returned above; keeps the ladder exhaustive.
            lines.append(f"- {spec.key}")
    lines.append(
        "Use the inferred-family block unless another family better matches the "
        "estimand; explain any switch in rationale."
    )
    if detail != "full":
        lines.append(
            "Alternate-family descriptions were shortened only to fit the request "
            "budget; no family was removed."
        )
    return "\n".join(lines)


def host_authorized_analysis_family(context: ResearchContext) -> Optional[str]:
    """Return the canonical caller-authorized family, if one was supplied."""

    raw = str(
        getattr(context.user_preferences, "inferred_analysis_family", "") or ""
    ).strip()
    if not raw:
        return None
    canonical = canonical_analysis_family(raw)
    if canonical is None:
        raise ValueError(f"Host-authorized inferred_analysis_family is unknown: {raw!r}")
    return canonical


def planner_analysis_family_authority_guide(
    context: ResearchContext,
    inferred: AnalysisTypeSpec,
    *,
    detail: str,
) -> str:
    """Publish either the ordinary switch menu or the caller's closed family."""

    authorized = host_authorized_analysis_family(context)
    if authorized is None:
        return planner_analysis_type_switch_guide(detail=detail)
    return (
        "HOST-AUTHORIZED ANALYSIS FAMILY: `analysis_type` MUST remain "
        f"{inferred.key!r}. The caller has already approved this typed family; "
        "do not switch families. If it cannot answer the question, expose the "
        "incompatibility for host review rather than substituting another family."
    )


def validate_host_authorized_analysis_family(
    context: ResearchContext,
    observed: str,
) -> None:
    """Reject a Planner family that replaces explicit caller authority."""

    authorized = host_authorized_analysis_family(context)
    if authorized is not None and observed != authorized:
        raise ValueError(
            "Planner analysis_type conflicts with the host-authorized analysis "
            f"family: expected {authorized!r}, observed {observed!r}"
        )


def locked_analysis_type_guide(spec: AnalysisTypeSpec) -> str:
    """Focused, advisory prompt block naming the inferred family.

    The historical function name remains for compatibility. The inference helps
    prevent generic-plan collapse but is not an execution lock; the planner may
    select a better-supported family with an explicit rationale.
    """
    return (
        f"INFERRED ANALYSIS FAMILY SUGGESTION: `{spec.key}` — {spec.name}.\n"
        f"{spec.description}\n"
        f"Default module backbone (omit inapplicable steps): {', '.join(spec.candidate_steps)}.\n"
        f"Mandatory guardrails for this family: {' '.join(spec.guardrails)}\n"
        "Use this family when it matches the estimand. Do not default to a generic "
        "association/logistic plan for survival, trajectory_clustering, dynamic_prediction "
        "or causal_inference. Without a caller-bound typed family, justify a better "
        "family in `rationale`; otherwise preserve caller authority.\n"
    )


def analysis_type_catalog_markdown() -> str:
    """Human-readable markdown summary for docs or reports."""
    parts: List[str] = []
    for spec in list_analysis_types():
        parts.append(f"### {spec.name} (`{spec.key}`)")
        parts.append(spec.description)
        parts.append("")
        parts.append("Candidate steps:")
        for step in spec.candidate_steps:
            parts.append(f"- {step}")
        parts.append("Guardrails:")
        for note in spec.guardrails:
            parts.append(f"- {note}")
        parts.append("")
    return "\n".join(parts).strip()


__all__ = [
    "AnalysisTypeSpec",
    "COMPETING_EVENT_TERMS",
    "canonical_analysis_family",
    "normalize_analysis_family",
    "is_concept_set_family",
    "list_analysis_types",
    "get_analysis_type",
    "optional_analysis_type_for_capability",
    "analysis_type_for_capability",
    "required_endpoint_kind_for_family",
    "requested_exposure_occurrence_cues",
    "requested_dose_response_cues",
    "infer_analysis_type",
    "landmark_prediction_framing",
    "longitudinal_trajectory_requested",
    "strong_trajectory_clustering_framing",
    "planner_analysis_type_guide",
    "planner_analysis_type_switch_guide",
    "host_authorized_analysis_family",
    "planner_analysis_family_authority_guide",
    "validate_host_authorized_analysis_family",
    "locked_analysis_type_guide",
    "analysis_type_catalog_markdown",
]


def repeated_measure_question(question: str) -> bool:
    return bool(re.search(
        r"\brepeated[ -]measur(?:es|ements)\b"
        r"|\b(?:mixed[ -]effects?|gee|general[is]zed estimating equations?)\b.{0,100}\b(?:time|days?|hours?|longitudinal)\b"
        r"|(?:change|changes|changing|growth|trend)\s+over\s+time"
        r"|重复测量|随时间(?:变化|改变)|纵向变化|增长曲线"
        r"|(?:混合效应|广义估计方程).{0,80}(?:时间|每日|纵向|每天)",
        question, re.IGNORECASE,
    ))


def repeated_measure_guardrails(context) -> str:
    if not repeated_measure_question(context.research_question or ""):
        return ""
    return (
        "Repeated-measure outcome change uses analysis_type='repeated_measures', custom_analysis and "
        "association.mixed_effects or association.gee, custom_method='repeated_measures'. Supply repeated_measure_spec as JSON with "
        "identity_column, time_column, outcome_column, time_origin='icu_admission', time_unit='hours'/'days', "
        "window_start, window_end, bin_width, aggregation, model='mixed_intercept'/'mixed_slope'/'gee', "
        "family='gaussian'/'binomial', time_basis='linear'/'natural_cubic_spline', spline_knots if needed, "
        "working_correlation for GEE, optional static covariates, dropout_limitation="
        "'death_may_cause_informative_dropout', dropout_sensitivity='survivors' with death_column or "
        "'joint_model_note'. Reuse accepted data_preparation aggregation choices when present. "
        "Do not invent measurements after death or fill missing bins. Bind artifact:analysis_cohort and "
        "all declared raw columns. Declare exactly " + ", ".join(REPEATED_OUTPUTS)
        + ". Publish aggregate results only. LCGA/GBTM is planned: no validated package in the runner lock/image."
    )
