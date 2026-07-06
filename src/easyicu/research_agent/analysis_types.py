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
from typing import Dict, Iterable, List, Optional, Sequence

from .schema import ResearchContext, VariableRole


@dataclass(frozen=True)
class AnalysisTypeSpec:
    key: str
    name: str
    description: str
    trigger_terms: Sequence[str]
    candidate_steps: Sequence[str]
    guardrails: Sequence[str]


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
            "time-to-event",
            "time to event",
            "cox",
            "kaplan",
            "kaplan-meier",
            "hazard",
            "competing risk",
            "censoring",
            "follow-up",
        ),
        candidate_steps=(
            "define time zero and follow-up window",
            "event / censoring audit",
            "Kaplan-Meier or cumulative-incidence summaries",
            "Cox or other time-to-event model",
            "sensitivity checks for censoring and competing risks",
        ),
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
            "treatment effect",
            "target trial",
            "propensity",
            "ipw",
            "inverse probability",
            "g-formula",
            "instrumental variable",
            "do-calculus",
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


def is_concept_set_family(value: Optional[str]) -> bool:
    """Whether a family is shaped as a concept SET, not a predictor->outcome pair."""
    return normalize_analysis_family(value) in CONCEPT_SET_FAMILIES


def list_analysis_types() -> List[AnalysisTypeSpec]:
    return list(_REGISTRY.values())


def get_analysis_type(key: str) -> AnalysisTypeSpec:
    return _REGISTRY[key]


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
        key = _FAMILY_ALIASES.get(candidate.strip().lower())
        if key and key in _REGISTRY:
            return key
    return None


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

    def _has_any(key: str, extras: Iterable[str] = ()) -> bool:
        terms = list(_REGISTRY[key].trigger_terms) + list(extras)
        return any(_keyword_present(text, term) for term in terms)

    # Strong, explicit task-family cues should win before softer scoring.
    if _has_any("reinforcement_learning"):
        return _REGISTRY["reinforcement_learning"]
    if _has_any("causal_inference"):
        return _REGISTRY["causal_inference"]
    if _has_any("trajectory_clustering"):
        return _REGISTRY["trajectory_clustering"]
    if _has_any("survival"):
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
        # "model X continuously" or "regression model" (e.g. the E2 lactate item
        # was mis-stamped prediction_model by "you may model lactate
        # continuously"). The real prediction cues (predict/auroc/calibration/
        # brier/...) are already in prediction_model's trigger_terms.
        "prediction_model",
        extras=("evaluation metric", "evaluation metrics"),
    ):
        return _REGISTRY["prediction_model"]
    if _has_any("measurement_bias_audit"):
        return _REGISTRY["measurement_bias_audit"]
    if _has_any("cohort_definition_sensitivity"):
        return _REGISTRY["cohort_definition_sensitivity"]
    if _has_any("score_policy_sensitivity"):
        return _REGISTRY["score_policy_sensitivity"]
    if _has_any("data_quality_audit") and not any(
        _has_any(key)
        for key in (
            "association_study",
            "prediction_model",
            "causal_inference",
            "trajectory_clustering",
            "reinforcement_learning",
            "measurement_bias_audit",
            "cohort_definition_sensitivity",
            "score_policy_sensitivity",
        )
    ):
        return _REGISTRY["data_quality_audit"]
    if _has_any("treatment_response"):
        return _REGISTRY["treatment_response"]

    scores: Dict[str, int] = {key: 0 for key in _REGISTRY}
    for key, spec in _REGISTRY.items():
        for term in spec.trigger_terms:
            if _keyword_present(text, term):
                scores[key] += 1

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
    if any(
        _keyword_present(((v.name or "") + " " + (v.description or "")).lower(), token)
        for v in context.variables
        for token in ("note", "notes", "text", "waveform", "ecg", "image", "imaging", "cxr", "ct", "mri")
    ):
        scores["multimodal"] += 2
    if context.cross_database_validation:
        scores["validation"] += 1
        scores["cross_database_replication"] += 2

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


def planner_analysis_type_guide() -> str:
    """Short prompt block for the planner.

    The guide intentionally presents task families and candidate steps,
    not mandatory recipes.
    """
    lines = [
        "ANALYSIS-TYPE CATALOG:",
        "First infer the task family from the research question. Treat the entries below as candidate modules, not mandatory fixed recipes.",
    ]
    for spec in list_analysis_types():
        lines.append(
            f"- {spec.key}: {spec.description} "
            f"Common modules: {', '.join(spec.candidate_steps)}. "
            f"Guardrails: {' '.join(spec.guardrails)}"
        )
    lines.append(
        "Choose only the steps justified by the task family and available context. "
        "Do not force Table 1, outcome incidence, missingness, or score-specific QC unless they serve the question."
    )
    return "\n".join(lines)


def locked_analysis_type_guide(spec: AnalysisTypeSpec) -> str:
    """Focused prompt block naming the single inferred family for THIS study.

    Injected ahead of the full catalog so the planner builds the plan around
    the locked family's candidate modules instead of silently collapsing every
    question to a generic association/logistic plan. The full catalog still
    follows as reference, and the planner may override with explicit
    justification when the question clearly belongs to another family.
    """
    return (
        f"LOCKED ANALYSIS FAMILY FOR THIS STUDY: `{spec.key}` — {spec.name}.\n"
        f"{spec.description}\n"
        f"Default module backbone (use these as your step set unless the "
        f"context clearly rules one out): {', '.join(spec.candidate_steps)}.\n"
        f"Mandatory guardrails for this family: {' '.join(spec.guardrails)}\n"
        "Build the plan around THIS family and pick methods/figures it calls "
        "for. Do not default to a generic association/logistic plan when the "
        "locked family is survival, trajectory_clustering, dynamic_prediction, "
        "causal_inference, etc. If the research question clearly belongs to a "
        "different family than the one locked above, you may switch — but only "
        "with an explicit one-line justification in `rationale`.\n"
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
    "list_analysis_types",
    "get_analysis_type",
    "infer_analysis_type",
    "planner_analysis_type_guide",
    "locked_analysis_type_guide",
    "analysis_type_catalog_markdown",
]
