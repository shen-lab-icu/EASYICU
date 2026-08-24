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


_REGISTRY: Dict[str, AnalysisTypeSpec] = {
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
    chinese_discovery_disclaimer = (
        re.search(
            r"(?:不|无需|避免)(?:进行|作|做|开展|采用|使用)?"
            r"[^，。；;]{0,12}(?:患者)?(?:表型|亚型|轨迹|患者群)"
            r"[^，。；;]{0,8}(?:聚类|分群|识别|发现)?"
            r"|(?:不|无需|避免)(?:进行|作|做|开展)?"
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
    chinese_discovery = bool(
        not chinese_discovery_disclaimer
        and (chinese_action_target or chinese_named_grouping)
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


def infer_analysis_type(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str] = None,
    target_outcome: Optional[str] = None,
) -> AnalysisTypeSpec:
    preferred = _preferred_family_key(context)
    if preferred is not None:
        return _REGISTRY[preferred]
    text = _question_text(context)
    primary_question_text = (context.research_question or "").lower().strip()
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
    survival_estimand_text = mask_survival_eligibility_phrases(text)
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
            "censoring",
            "删失",
        )
    )

    def _has_any(key: str, extras: Iterable[str] = ()) -> bool:
        terms = list(_REGISTRY[key].trigger_terms) + list(extras)
        return any(_keyword_present(text, term) for term in terms)

    # Strong, explicit task-family cues should win before softer scoring.
    if _has_any("reinforcement_learning"):
        return _REGISTRY["reinforcement_learning"]
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
        return _REGISTRY["causal_inference"]
    if _strong_clustering_framing:
        return _REGISTRY["trajectory_clustering"]
    if strong_survival_framing:
        return _REGISTRY["survival"]
    if _has_any("dynamic_prediction"):
        return _REGISTRY["dynamic_prediction"]
    if _has_any("multimodal"):
        return _REGISTRY["multimodal"]
    if _has_any("validation"):
        return _REGISTRY["validation"]
    if _has_any(
        # NOTE: do NOT add the bare word "model" here. As a strong pre-scoring
        # cue it false-fires on association/descriptive questions that merely say
        # "model X continuously" or "regression model". The real prediction cues
        # (predict/AUROC/calibration/Brier/...) are already in trigger_terms.
        "prediction_model",
        extras=("evaluation metric", "evaluation metrics"),
    ):
        return _REGISTRY["prediction_model"]
    if _has_any("measurement_bias_audit"):
        return _REGISTRY["measurement_bias_audit"]
    if cohort_sensitivity_framed and _has_any("cohort_definition_sensitivity"):
        return _REGISTRY["cohort_definition_sensitivity"]
    if _has_any("score_policy_sensitivity"):
        return _REGISTRY["score_policy_sensitivity"]
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
        return _REGISTRY["data_quality_audit"]
    if treatment_response_framed and _has_any("treatment_response"):
        return _REGISTRY["treatment_response"]

    scores: Dict[str, int] = {key: 0 for key in _REGISTRY}
    for key, spec in _REGISTRY.items():
        for term in spec.trigger_terms:
            if _keyword_present(text, term):
                scores[key] += 1
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

    if target_outcome and _has_any("association_study") and not has_multimodal_variable:
        # An explicit association/effect question is stronger evidence than the
        # mere presence of an outcome column.  More specialised families have
        # already returned above, so this only prevents the generic descriptive
        # fallback from winning a tie.
        scores["association_study"] += 3
    if primary_predictor and target_outcome:
        scores["association_study"] += 2
    elif target_outcome:
        scores["descriptive_epidemiology"] += 1

    best_key = max(scores, key=scores.get)
    if scores[best_key] == 0:
        if primary_predictor and target_outcome:
            best_key = "association_study"
        elif target_outcome:
            best_key = "descriptive_epidemiology"
        else:
            best_key = "data_quality_audit"
    return _REGISTRY[best_key]


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
        f"Default module backbone (use these as your step set unless the "
        f"context clearly rules one out): {', '.join(spec.candidate_steps)}.\n"
        f"Mandatory guardrails for this family: {' '.join(spec.guardrails)}\n"
        "Use this family when it matches the estimand and pick methods/figures it "
        "calls for. Do not default to a generic association/logistic plan when the "
        "suggested family is survival, trajectory_clustering, dynamic_prediction, "
        "causal_inference, etc. If the research question clearly belongs to a "
        "different family than the suggestion above and the caller did not bind "
        "a typed family, you may switch with an explicit one-line justification "
        "in `rationale`; caller-bound family authority may not be replaced.\n"
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
    "canonical_analysis_family",
    "normalize_analysis_family",
    "is_concept_set_family",
    "list_analysis_types",
    "get_analysis_type",
    "optional_analysis_type_for_capability",
    "analysis_type_for_capability",
    "required_endpoint_kind_for_family",
    "infer_analysis_type",
    "strong_trajectory_clustering_framing",
    "planner_analysis_type_guide",
    "planner_analysis_type_switch_guide",
    "host_authorized_analysis_family",
    "planner_analysis_family_authority_guide",
    "validate_host_authorized_analysis_family",
    "locked_analysis_type_guide",
    "analysis_type_catalog_markdown",
]
