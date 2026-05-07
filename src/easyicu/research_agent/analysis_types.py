"""Analysis-type registry for EHR/ICU research planning.

The research-agent layer needs a task-family abstraction that is more
stable than free-form prompts but less rigid than a single fixed
step list. Adjacent projects point in the same direction:

* OpenLens AI organises work by cooperating research modules rather
  than a universal statistical checklist.
* HealthFlow benchmarks task families derived from literature instead
  of one static analysis recipe.
* M4 frames clinical research as hypothesis screening, cohort
  characterisation, survival analysis, and related work modes.

This module turns that idea into an explicit registry the planner can
inspect. Each analysis type exposes:

* trigger terms for lightweight inference from a research question;
* a human-readable description;
* candidate steps that are common for that family but not mandatory;
* guardrails that help prompts stay honest about what the family does.
"""

from __future__ import annotations

from dataclasses import dataclass
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
            "describe", "description", "characteristics", "baseline", "cohort",
            "incidence", "prevalence", "frequency", "burden",
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
            "associated", "association", "predictor", "prognostic", "risk factor",
            "odds ratio", "hazard ratio", "linked", "relationship",
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
            "predict", "prediction", "predictive", "early warning", "classifier",
            "model performance", "auroc", "auc", "calibration", "brier",
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
    "trajectory_clustering": AnalysisTypeSpec(
        key="trajectory_clustering",
        name="Trajectory clustering / phenotype discovery",
        description=(
            "Group patients using longitudinal patterns or multivariate states "
            "to discover clinically meaningful subphenotypes."
        ),
        trigger_terms=(
            "cluster", "clustering", "trajectory", "trajectories", "phenotype",
            "subphenotype", "longitudinal", "state sequence",
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
    "treatment_response": AnalysisTypeSpec(
        key="treatment_response",
        name="Treatment-response / heterogeneity analysis",
        description=(
            "Characterize response patterns or heterogeneity around treatments "
            "without necessarily claiming a causal estimand."
        ),
        trigger_terms=(
            "treatment response", "response", "heterogeneity", "drug response",
            "therapy response", "responder", "nonresponder",
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
        description=(
            "Estimate a treatment effect under an explicit causal design."
        ),
        trigger_terms=(
            "causal", "treatment effect", "target trial", "propensity", "ipw",
            "inverse probability", "g-formula", "instrumental variable", "do-calculus",
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
            "reinforcement learning", "policy learning", "off-policy", "q-learning",
            "actor-critic", "dynamic treatment regime", "decision policy",
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
            "missingness", "missing", "completeness", "coverage", "data quality",
            "quality audit", "availability", "schema", "unit check", "range check",
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
    "cross_database_replication": AnalysisTypeSpec(
        key="cross_database_replication",
        name="Cross-database replication / transportability",
        description=(
            "Replicate a cohort or analysis across ICU datasets with explicit "
            "concept mapping and harmonization notes."
        ),
        trigger_terms=(
            "replicate", "replication", "cross-database", "external validation",
            "transportability", "across mimic", "across eicu",
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


def list_analysis_types() -> List[AnalysisTypeSpec]:
    return list(_REGISTRY.values())


def get_analysis_type(key: str) -> AnalysisTypeSpec:
    return _REGISTRY[key]


def _question_text(context: ResearchContext) -> str:
    return (context.research_question or "").lower()


def infer_analysis_type(
    context: ResearchContext,
    *,
    primary_predictor: Optional[str] = None,
    target_outcome: Optional[str] = None,
) -> AnalysisTypeSpec:
    text = _question_text(context)

    def _has_any(key: str, extras: Iterable[str] = ()) -> bool:
        terms = list(_REGISTRY[key].trigger_terms) + list(extras)
        return any(term in text for term in terms)

    # Strong, explicit task-family cues should win before softer scoring.
    if _has_any("reinforcement_learning"):
        return _REGISTRY["reinforcement_learning"]
    if _has_any("causal_inference"):
        return _REGISTRY["causal_inference"]
    if _has_any("trajectory_clustering"):
        return _REGISTRY["trajectory_clustering"]
    if _has_any("prediction_model", extras=("model", "evaluation metric", "evaluation metrics", "validate")):
        return _REGISTRY["prediction_model"]
    if _has_any("data_quality_audit") and not any(
        _has_any(key)
        for key in ("association_study", "prediction_model", "causal_inference", "trajectory_clustering", "reinforcement_learning")
    ):
        return _REGISTRY["data_quality_audit"]
    if _has_any("treatment_response"):
        return _REGISTRY["treatment_response"]

    scores: Dict[str, int] = {key: 0 for key in _REGISTRY}
    for key, spec in _REGISTRY.items():
        for term in spec.trigger_terms:
            if term in text:
                scores[key] += 1

    if target_outcome and any(v.role == VariableRole.TIME for v in context.variables):
        scores["prediction_model"] += 1 if "predict" in text or "forecast" in text else 0
        scores["trajectory_clustering"] += 1 if "trajectory" in text else 0

    if primary_predictor and target_outcome:
        scores["association_study"] += 2
    elif target_outcome:
        scores["descriptive_epidemiology"] += 1

    if context.cross_database_validation:
        scores["cross_database_replication"] += 2

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
    "analysis_type_catalog_markdown",
]
