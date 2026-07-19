"""Study-design playbooks for article-level research planning.

This module owns static, case-neutral guidance derived from reporting
standards and common top-journal article structure. It deliberately avoids
benchmark-specific variables, databases, scores, or figure numbers.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

StudyDesignFamily = Literal[
    "association",
    "prediction",
    "time_to_event",
    "phenotyping",
    "causal_emulation",
    "descriptive",
]

DisplayTier = Literal["core", "recommended", "conditional", "supplementary"]


class DisplayModuleSpec(BaseModel):
    """A flexible article-display module.

    A module describes the evidence role a table/figure should play, not a
    fixed title. This keeps the planner aligned with journal-style article
    structure while allowing the concrete display to adapt to the scientific
    question and available data.
    """

    model_config = ConfigDict(extra="forbid")

    module_id: str
    role: str
    tier: DisplayTier
    rationale: str
    acceptable_outputs: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)


_FAMILY_TEMPLATES: Dict[StudyDesignFamily, Dict[str, object]] = {
    "association": {
        "reporting_guidelines": ["STROBE/RECORD-style observational reporting"],
        "required_methods": [
            "explicit cohort eligibility and attrition",
            "primary adjusted association model matched to outcome type",
            "pre-specified covariate rationale",
            "missingness and measurement-process audit",
            "sensitivity analyses over cohort, exposure/outcome definition, and missing data",
        ],
        "main_text_displays": [
            "cohort flow / attrition",
            "Table 1 baseline characteristics",
            "exposure prevalence and absolute outcome risk",
            "primary adjusted effect estimate",
            "sensitivity / robustness summary",
            "missingness or data-quality summary",
        ],
        "supplementary_displays": [
            "full variable definitions and aggregation rules",
            "complete model coefficients",
            "missingness table",
            "sensitivity-specification matrix",
            "analysis provenance and code manifest",
        ],
        "sensitivity_requirements": [
            "alternative exposure or definition",
            "alternative missing-data handling",
            "cohort restriction or denominator check",
            "effect-scale comparison when clinically meaningful",
        ],
        "covariate_strategy": (
            "Classify covariates by role: confounder, baseline severity proxy, "
            "measurement-process indicator, precision variable, or excluded "
            "collider/post-baseline proxy. Do not include every available column."
        ),
    },
    "prediction": {
        "reporting_guidelines": ["TRIPOD-style prediction-model reporting"],
        "required_methods": [
            "development/validation split or external validation",
            "discrimination and calibration evaluation",
            "missing-data strategy fitted inside the modelling workflow",
            "feature-selection or regularisation description",
            "clinical utility or threshold analysis when relevant",
        ],
        "main_text_displays": [
            "cohort flow / modelling denominator",
            "Table 1 baseline characteristics",
            "discrimination curve or metric panel",
            "calibration plot",
            "feature importance or coefficient summary",
        ],
        "supplementary_displays": [
            "full feature dictionary",
            "hyperparameter and preprocessing specification",
            "internal/external validation metrics",
            "calibration table",
            "model-card style limitations",
        ],
        "sensitivity_requirements": [
            "temporal or site validation split",
            "complete-case versus imputed workflow",
            "threshold and decision-curve analysis when action thresholds exist",
        ],
        "covariate_strategy": (
            "Treat predictors as pre-outcome features with leakage checks; separate "
            "baseline features, time-window summaries, and post-outcome variables."
        ),
    },
    "time_to_event": {
        "reporting_guidelines": ["STROBE/RECORD with survival-analysis extensions"],
        "required_methods": [
            "time zero and follow-up definition",
            "censoring and competing-risk policy",
            "Kaplan-Meier/cumulative-incidence description",
            "Cox or other survival model with assumption checks",
            "sensitivity to censoring/follow-up definitions",
        ],
        "main_text_displays": [
            "cohort flow / risk-set denominator",
            "Table 1 baseline characteristics",
            "survival or cumulative-incidence curve with risk table",
            "adjusted hazard/risk estimate",
            "assumption or censoring diagnostic",
        ],
        "supplementary_displays": [
            "risk-set counts",
            "proportional-hazards diagnostics",
            "competing-risk sensitivity",
            "follow-up distribution",
        ],
        "sensitivity_requirements": [
            "alternative censoring rule",
            "competing-risk analysis if applicable",
            "time-window or landmark sensitivity",
        ],
        "covariate_strategy": (
            "Separate baseline adjustment covariates from time-varying quantities; "
            "do not adjust for mediators measured after time zero unless explicitly designed."
        ),
    },
    "phenotyping": {
        "reporting_guidelines": ["Transparent unsupervised-learning reporting"],
        "required_methods": [
            "feature set and scaling rationale",
            "cluster-number or latent-dimension selection",
            "cluster stability assessment",
            "clinical characterization of phenotypes",
            "external or outcome characterization without over-claiming causality",
        ],
        "main_text_displays": [
            "cohort flow / feature availability",
            "feature missingness and scaling summary",
            "embedding or heatmap of phenotype structure",
            "cluster characteristics table",
            "stability or validation diagnostic",
        ],
        "supplementary_displays": [
            "full feature list",
            "cluster stability grid",
            "alternative cluster-number results",
            "cluster assignment provenance",
        ],
        "sensitivity_requirements": [
            "alternative feature subset",
            "alternative cluster number",
            "bootstrap or resampling stability",
        ],
        "covariate_strategy": (
            "Define features as phenotype descriptors, not adjustment covariates; "
            "separate downstream outcome characterization from cluster discovery."
        ),
    },
    "causal_emulation": {
        "reporting_guidelines": ["Target-trial emulation / causal-inference reporting"],
        "required_methods": [
            "target trial specification with eligibility, time zero, strategies, and estimand",
            "confounder set justified by timing and causal role",
            "positivity and balance diagnostics",
            "primary estimator such as IPTW, g-computation, matching, or doubly robust model",
            "sensitivity to unmeasured confounding and analytic choices",
        ],
        "main_text_displays": [
            "target-trial schematic or cohort flow",
            "Table 1 before/after adjustment or weighting",
            "balance plot",
            "primary causal contrast",
            "sensitivity analysis summary",
        ],
        "supplementary_displays": [
            "causal variable/timing table",
            "propensity/balance diagnostics",
            "positivity assessment",
            "alternative estimators",
            "unmeasured-confounding sensitivity",
        ],
        "sensitivity_requirements": [
            "alternative confounder set",
            "trimming/positivity sensitivity",
            "negative or falsification outcome when available",
        ],
        "covariate_strategy": (
            "Use only pre-time-zero confounders for adjustment; label mediators, colliders, "
            "and post-treatment variables as excluded unless a formal longitudinal design is specified."
        ),
    },
    "descriptive": {
        "reporting_guidelines": ["STROBE-style descriptive observational reporting"],
        "required_methods": [
            "explicit cohort denominator",
            "descriptive summaries with uncertainty where appropriate",
            "missingness and data-quality audit",
            "clinically meaningful strata when pre-specified",
        ],
        "main_text_displays": [
            "cohort flow / denominator",
            "Table 1 or descriptive summary table",
            "distribution or prevalence figure",
            "missingness or data-quality summary",
        ],
        "supplementary_displays": [
            "variable definitions",
            "complete descriptive table",
            "stratified summaries",
            "analysis provenance and code manifest",
        ],
        "sensitivity_requirements": [
            "denominator definition sensitivity",
            "missingness/measurement availability sensitivity",
        ],
        "covariate_strategy": (
            "Do not force adjustment variables into a descriptive question; classify variables "
            "as descriptors, strata, or denominators."
        ),
    },
}


_BASE_DESIGN_PRINCIPLES = [
    "Start from the article question, not from a favourite plot type.",
    "Separate design/accounting displays, data-quality displays, primary-result displays, and robustness displays.",
    "Use enough complementary tables/figures for a reader to audit denominator, variables, uncertainty, and sensitivity.",
    "Put diagnostics and large specification grids in supplements when they support but would overload the main narrative.",
    "Prefer a display family that matches the estimand: association, prediction, survival, phenotyping, causal, or descriptive.",
]

_FAMILY_DESIGN_PRINCIPLES: Dict[StudyDesignFamily, List[str]] = {
    "association": [
        "A credible observational association article needs design context before the effect estimate.",
        "The main result should be paired with missingness, adjustment rationale, and robustness, not shown as a lone forest plot.",
    ],
    "prediction": [
        "Prediction articles are judged by validation, calibration, and clinical usefulness, not AUROC alone.",
        "Feature importance is explanatory support; it cannot replace discrimination, calibration, and leakage checks.",
    ],
    "time_to_event": [
        "Survival analyses need explicit time zero, follow-up, censoring, risk sets, and model assumptions.",
        "Absolute risk curves and adjusted contrasts answer different reader questions and should not be collapsed into one display.",
    ],
    "phenotyping": [
        "Unsupervised phenotyping needs feature provenance, structure, clinical characterization, and stability evidence.",
        "Outcome associations are downstream characterization, not proof that discovered phenotypes are causal entities.",
    ],
    "causal_emulation": [
        "Causal emulation must define the target trial before choosing an estimator.",
        "Balance, positivity, timing, and sensitivity are part of the result, not optional diagnostics.",
    ],
    "descriptive": [
        "Descriptive work still needs denominator, missingness, and stratification logic.",
        "Do not force adjusted modelling into a question whose purpose is distribution, coverage, or prevalence.",
    ],
}

_BASE_ANTI_PATTERNS = [
    "Do not make a single primary-effect forest plot the whole article display suite.",
    "Do not report model performance without calibration or validation context.",
    "Do not treat missing variables as ordinary missing-at-random cells when source structure may be absent.",
    "Do not move case-specific variables, scores, databases, or benchmark examples into global prompts.",
    "Do not claim causal effects from an observational association unless a causal design is specified.",
]


def _module(
    module_id: str,
    role: str,
    tier: DisplayTier,
    rationale: str,
    acceptable_outputs: Sequence[str],
    *,
    triggers: Sequence[str] = (),
) -> DisplayModuleSpec:
    return DisplayModuleSpec(
        module_id=module_id,
        role=role,
        tier=tier,
        rationale=rationale,
        acceptable_outputs=list(acceptable_outputs),
        triggers=list(triggers),
    )


_FAMILY_DISPLAY_MODULES: Dict[StudyDesignFamily, List[DisplayModuleSpec]] = {
    "association": [
        _module(
            "cohort_flow",
            "cohort_accounting",
            "core",
            "Readers need eligibility, exclusions, and denominators before interpreting estimates.",
            ("cohort flow", "attrition table", "eligibility diagram", "denominator panel"),
        ),
        _module(
            "baseline_table",
            "baseline_context",
            "core",
            "Baseline characteristics show who contributed to the estimate and whether adjustment is plausible.",
            ("Table 1", "baseline table", "descriptor table", "stratified baseline summary"),
        ),
        _module(
            "missingness_measurement_audit",
            "data_quality",
            "core",
            "ICU data availability is part of the scientific result and must be visible.",
            (
                "missingness heatmap",
                "measurement availability table",
                "data-quality panel",
                "source-coverage summary",
            ),
        ),
        _module(
            "primary_adjusted_estimate",
            "primary_estimand",
            "core",
            "The main association estimate should expose effect scale, uncertainty, and adjustment set.",
            (
                "adjusted effect table",
                "forest plot",
                "coefficient plot",
                "marginal-effect panel",
            ),
        ),
        _module(
            "absolute_risk_context",
            "descriptive_result",
            "core",
            "Readers need the exposure prevalence and absolute outcome risks before interpreting adjusted relative estimates.",
            (
                "prevalence estimate with CI",
                "absolute outcome risk by exposure",
                "risk-difference panel",
                "event-rate panel",
            ),
        ),
        _module(
            "robustness_grid",
            "robustness",
            "core",
            "Top-journal observational results usually need sensitivity over definitions and missing-data choices.",
            (
                "sensitivity grid",
                "robustness table",
                "specification curve",
                "alternative-definition panel",
            ),
        ),
        _module(
            "effect_modification",
            "heterogeneity",
            "recommended",
            "Clinically meaningful strata help reveal whether a pooled estimate hides important heterogeneity.",
            (
                "subgroup forest",
                "interaction table",
                "stratified effect panel",
                "database-specific estimate panel",
            ),
        ),
        _module(
            "full_model_appendix",
            "supplementary_provenance",
            "supplementary",
            "Full coefficients, definitions, and code provenance belong in the supplement or artifact manifest.",
            ("full coefficient table", "variable dictionary", "code manifest", "claim ledger"),
        ),
    ],
    "prediction": [
        _module(
            "modelling_cohort_flow",
            "cohort_accounting",
            "core",
            "Prediction studies need modelling denominators, exclusions, event counts, and validation splits.",
            ("cohort flow", "modelling denominator table", "train-test split diagram"),
        ),
        _module(
            "baseline_table",
            "baseline_context",
            "core",
            "Development and validation populations need baseline and outcome context.",
            ("Table 1", "baseline table", "development-validation characteristics"),
        ),
        _module(
            "validation_design",
            "validation",
            "core",
            "Internal, temporal, site, or external validation design must be explicit.",
            ("validation split table", "external validation table", "site validation panel"),
        ),
        _module(
            "discrimination",
            "model_performance",
            "core",
            "Discrimination quantifies ranking performance but is not sufficient by itself.",
            ("ROC curve", "AUROC table", "precision-recall curve", "AUPRC table"),
        ),
        _module(
            "calibration",
            "calibration",
            "core",
            "Clinical prediction reporting needs calibration to judge whether predicted risks are usable.",
            ("calibration plot", "calibration-in-the-large", "calibration slope", "Brier score"),
        ),
        _module(
            "missingness_leakage_audit",
            "data_quality",
            "core",
            "Feature availability, imputation, and leakage checks are part of model validity.",
            ("missingness table", "feature availability panel", "leakage audit", "preprocessing table"),
        ),
        _module(
            "clinical_utility",
            "clinical_utility",
            "conditional",
            "If action thresholds are clinically relevant, net benefit or threshold analyses are needed.",
            ("decision curve", "net-benefit plot", "threshold utility table", "risk-group table"),
            triggers=("action thresholds", "deployment", "triage", "clinical utility"),
        ),
        _module(
            "model_explanation",
            "explainability",
            "recommended",
            "Interpretability helps readers inspect model behaviour but does not replace validation.",
            ("feature importance", "coefficient plot", "SHAP summary", "partial-dependence panel"),
        ),
        _module(
            "model_card_appendix",
            "supplementary_provenance",
            "supplementary",
            "Model cards and preprocessing details are needed for reproducibility.",
            ("feature dictionary", "hyperparameter table", "model card", "code manifest"),
        ),
    ],
    "time_to_event": [
        _module(
            "risk_set_flow",
            "cohort_accounting",
            "core",
            "Survival studies need eligibility, time zero, follow-up, censoring, and risk-set accounting.",
            ("cohort flow", "risk-set table", "time-zero diagram", "follow-up summary"),
        ),
        _module(
            "baseline_table",
            "baseline_context",
            "core",
            "Baseline covariates define the starting risk set and adjustment context.",
            ("Table 1", "baseline table", "risk-set baseline summary"),
        ),
        _module(
            "survival_curve",
            "temporal_absolute_risk",
            "core",
            "Absolute-risk displays show time-patterns that a single hazard ratio cannot show.",
            (
                "Kaplan-Meier curve",
                "cumulative-incidence curve",
                "risk table",
                "absolute-risk curve",
            ),
        ),
        _module(
            "adjusted_survival_contrast",
            "survival_effect",
            "core",
            "Adjusted hazard or risk contrasts quantify the primary survival estimand.",
            ("Cox model table", "hazard-ratio forest", "risk-difference table"),
        ),
        _module(
            "survival_diagnostics",
            "diagnostics",
            "core",
            "Censoring and proportional-hazards assumptions need visible checks.",
            (
                "proportional-hazards diagnostic",
                "Schoenfeld residual plot",
                "censoring table",
                "follow-up distribution",
            ),
        ),
        _module(
            "survival_sensitivity",
            "robustness",
            "recommended",
            "Landmark, censoring, and competing-risk choices should be tested when plausible.",
            ("landmark analysis", "competing-risk sensitivity", "alternative censoring panel"),
        ),
    ],
    "phenotyping": [
        _module(
            "feature_availability_flow",
            "cohort_accounting",
            "core",
            "Phenotype discovery depends on which patients and features survive availability filters.",
            ("cohort flow", "feature availability table", "phenotyping denominator panel"),
        ),
        _module(
            "feature_quality_scaling",
            "data_quality",
            "core",
            "Scaling, missingness, and feature provenance influence cluster geometry.",
            ("feature missingness heatmap", "scaling summary", "feature provenance table"),
        ),
        _module(
            "phenotype_structure",
            "phenotype_structure",
            "core",
            "A structure display shows whether the discovered groups are separable or continuous.",
            ("embedding plot", "UMAP", "PCA", "cluster heatmap", "dendrogram"),
        ),
        _module(
            "phenotype_profiles",
            "phenotype_profile",
            "core",
            "Clinical profiles make unsupervised clusters interpretable.",
            ("cluster profile heatmap", "phenotype characteristics table", "radar plot"),
        ),
        _module(
            "cluster_stability",
            "stability",
            "core",
            "Stability evidence protects against overinterpreting arbitrary cluster cuts.",
            ("bootstrap stability grid", "consensus matrix", "alternative-k panel"),
        ),
        _module(
            "outcome_characterization",
            "downstream_characterization",
            "recommended",
            "Outcome associations characterize phenotypes but should be labelled exploratory.",
            ("outcome-by-cluster table", "survival by phenotype", "mortality by cluster panel"),
        ),
    ],
    "causal_emulation": [
        _module(
            "target_trial_protocol",
            "causal_protocol",
            "core",
            "The estimand is defined by eligibility, time zero, strategies, assignment, follow-up, and contrast.",
            ("target-trial table", "protocol schematic", "emulation specification table"),
        ),
        _module(
            "causal_cohort_flow",
            "cohort_accounting",
            "core",
            "Treatment-strategy groups and exclusions must be auditable from time zero.",
            ("cohort flow", "treatment assignment flow", "time-zero diagram"),
        ),
        _module(
            "baseline_balance",
            "balance_positivity",
            "core",
            "Balance and positivity diagnostics are required before interpreting a causal contrast.",
            ("standardized mean difference plot", "balance table", "positivity plot", "weight distribution"),
        ),
        _module(
            "primary_causal_contrast",
            "causal_contrast",
            "core",
            "The main display must state the causal contrast, estimator, effect scale, and uncertainty.",
            ("causal contrast table", "IPTW estimate", "g-computation estimate", "effect curve"),
        ),
        _module(
            "causal_sensitivity",
            "robustness",
            "core",
            "Unmeasured confounding, trimming, and estimator choices need sensitivity assessment.",
            ("unmeasured-confounding sensitivity", "trimming sensitivity", "alternative-estimator grid"),
        ),
        _module(
            "timing_dag_appendix",
            "supplementary_provenance",
            "supplementary",
            "Timing tables or DAGs clarify why variables are confounders, mediators, or excluded colliders.",
            ("variable timing table", "DAG", "confounder rationale table", "code manifest"),
        ),
    ],
    "descriptive": [
        _module(
            "cohort_denominator",
            "cohort_accounting",
            "core",
            "Descriptive work still needs explicit denominators and exclusions.",
            ("cohort flow", "denominator table", "eligibility summary"),
        ),
        _module(
            "descriptive_table",
            "baseline_context",
            "core",
            "A main descriptive table anchors the population and variables.",
            ("Table 1", "descriptive summary table", "stratified descriptor table"),
        ),
        _module(
            "distribution_prevalence",
            "distribution",
            "core",
            "Distributional or prevalence displays answer the primary descriptive question.",
            ("distribution plot", "prevalence bar chart", "density plot", "histogram", "ridge plot"),
        ),
        _module(
            "missingness_data_quality",
            "data_quality",
            "core",
            "Coverage and missingness determine how much of the descriptive result is interpretable.",
            ("missingness table", "coverage heatmap", "data-quality panel"),
        ),
        _module(
            "stratified_context",
            "heterogeneity",
            "recommended",
            "Clinically meaningful strata or databases prevent a pooled average from hiding structure.",
            ("stratified summary", "small multiples", "database comparison heatmap"),
        ),
    ],
}

_GENERIC_CONDITIONAL_MODULES = [
    _module(
        "cross_database_heterogeneity",
        "transportability",
        "conditional",
        "When multiple databases or sites are in scope, display source-level coverage and result heterogeneity.",
        (
            "database-specific panel",
            "transportability heatmap",
            "source-level estimate forest",
            "coverage-by-database table",
        ),
        triggers=("multiple databases", "external validation", "transportability", "cross database"),
    ),
    _module(
        "measurement_process_audit",
        "data_quality",
        "conditional",
        "When missingness or measurement frequency is central, show the measurement process explicitly.",
        (
            "measurement frequency panel",
            "missingness mechanism table",
            "availability-by-time heatmap",
        ),
        triggers=("missingness", "measurement process", "source absence"),
    ),
    _module(
        "exposure_outcome_distribution",
        "descriptive_result",
        "conditional",
        "When prevalence, incidence, or event rates are part of the question, show them before adjusted modelling.",
        (
            "exposure prevalence table",
            "outcome-by-exposure table",
            "prevalence and outcome figure",
            "event-rate by exposure panel",
        ),
        triggers=("prevalence", "incidence", "event rate", "mortality rate"),
    ),
]


_BRIEF_CHECK_TERMS: Dict[str, Sequence[str]] = {
    "cohort flow / attrition": ("cohort", "attrition", "flow", "eligibility", "denominator"),
    "cohort flow / modelling denominator": ("cohort", "denominator", "eligibility", "attrition"),
    "cohort flow / risk-set denominator": ("cohort", "risk-set", "risk set", "denominator"),
    "cohort flow / feature availability": ("cohort", "feature availability", "missingness"),
    "target-trial schematic or cohort flow": ("target trial", "cohort", "time zero", "eligibility"),
    "Table 1 baseline characteristics": ("table 1", "table_one", "baseline characteristic", "demographic"),
    "Table 1 or descriptive summary table": ("table 1", "table_one", "descriptive summary", "baseline"),
    "primary adjusted effect estimate": ("adjusted", "association", "effect", "odds ratio", "risk ratio"),
    "sensitivity / robustness summary": ("sensitivity", "robustness", "variant"),
    "missingness or data-quality summary": ("missingness", "data quality", "measurement"),
    "discrimination curve or metric panel": ("auroc", "auc", "discrimination", "roc"),
    "calibration plot": ("calibration",),
    "feature importance or coefficient summary": ("feature importance", "coefficient", "importance"),
    "survival or cumulative-incidence curve with risk table": ("survival", "cumulative incidence", "risk table"),
    "adjusted hazard/risk estimate": ("hazard", "cox", "survival model"),
    "assumption or censoring diagnostic": ("censoring", "proportional hazards", "assumption"),
    "feature missingness and scaling summary": ("missingness", "scaling", "feature"),
    "embedding or heatmap of phenotype structure": ("embedding", "heatmap", "umap", "pca", "phenotype"),
    "cluster characteristics table": ("cluster", "characteristic", "phenotype"),
    "stability or validation diagnostic": ("stability", "validation", "bootstrap"),
    "balance plot": ("balance", "standardized mean difference", "smd"),
    "primary causal contrast": ("causal contrast", "estimand", "iptw", "g-computation", "matching"),
    "distribution or prevalence figure": ("distribution", "prevalence", "histogram", "density"),
}

_ROLE_CHECK_TERMS: Dict[str, Sequence[str]] = {
    "cohort_accounting": (
        "cohort",
        "eligibility",
        "exclusion",
        "attrition",
        "denominator",
        "flow",
        "risk set",
        "risk-set",
    ),
    "baseline_context": (
        "table 1",
        "table_one",
        "baseline",
        "characteristic",
        "demographic",
        "descriptor",
    ),
    "data_quality": (
        "missing",
        "missingness",
        "measurement",
        "availability",
        "coverage",
        "data quality",
        "leakage",
        "preprocessing",
    ),
    "primary_estimand": (
        "adjusted",
        "association",
        "effect",
        "odds ratio",
        "risk ratio",
        "marginal effect",
        "coefficient",
    ),
    "robustness": (
        "sensitivity",
        "robustness",
        "variant",
        "alternative",
        "specification",
    ),
    "heterogeneity": (
        "subgroup",
        "interaction",
        "stratified",
        "heterogeneity",
        "database-specific",
    ),
    "validation": (
        "validation",
        "external",
        "temporal split",
        "train-test",
        "test set",
    ),
    "model_performance": (
        "auroc",
        "auc",
        "roc",
        "auprc",
        "precision-recall",
        "discrimination",
    ),
    "calibration": ("calibration", "brier", "calibration slope"),
    "clinical_utility": ("decision curve", "net benefit", "threshold", "utility"),
    "explainability": ("feature importance", "shap", "coefficient", "partial dependence"),
    "temporal_absolute_risk": (
        "survival curve",
        "kaplan",
        "cumulative incidence",
        "risk table",
        "absolute risk",
    ),
    "survival_effect": ("hazard", "cox", "survival model", "risk difference"),
    "diagnostics": (
        "diagnostic",
        "assumption",
        "censoring",
        "proportional hazards",
        "schoenfeld",
    ),
    "phenotype_structure": ("embedding", "umap", "pca", "heatmap", "dendrogram"),
    "phenotype_profile": ("cluster profile", "phenotype characteristic", "radar"),
    "stability": ("stability", "bootstrap", "consensus", "alternative k", "alternative-k"),
    "downstream_characterization": ("outcome by cluster", "mortality by cluster", "downstream"),
    "causal_protocol": ("target trial", "time zero", "strategy", "estimand", "protocol"),
    "balance_positivity": (
        "balance",
        "standardized mean difference",
        "smd",
        "positivity",
        "weight",
    ),
    "causal_contrast": (
        "causal contrast",
        "iptw",
        "g-computation",
        "matching",
        "doubly robust",
    ),
    "distribution": ("distribution", "prevalence", "histogram", "density", "ridge"),
    "descriptive_result": (
        "prevalence",
        "incidence",
        "event rate",
        "mortality rate",
        "outcome-by-exposure",
        "outcome by exposure",
    ),
    "transportability": (
        "cross database",
        "cross-database",
        "database-specific",
        "site-specific",
        "transportability",
        "generalization",
        "generalisation",
        "external validation",
    ),
    "supplementary_provenance": (
        "supplement",
        "appendix",
        "manifest",
        "claim ledger",
        "dictionary",
        "full coefficient",
    ),
}



def family_template(family: StudyDesignFamily) -> Dict[str, object]:
    return _FAMILY_TEMPLATES[family]


def design_principles_for_family(family: StudyDesignFamily) -> List[str]:
    return [*_BASE_DESIGN_PRINCIPLES, *_FAMILY_DESIGN_PRINCIPLES[family]]


def anti_patterns_for_brief() -> List[str]:
    return list(_BASE_ANTI_PATTERNS)


def display_modules_for_family(family: StudyDesignFamily) -> List[DisplayModuleSpec]:
    return [module.model_copy(deep=True) for module in _FAMILY_DISPLAY_MODULES[family]]


def triggered_generic_modules(triggers: Sequence[str]) -> List[DisplayModuleSpec]:
    trigger_blob = " ".join(triggers).lower()
    modules: List[DisplayModuleSpec] = []
    for module in _GENERIC_CONDITIONAL_MODULES:
        if module.module_id == "cross_database_heterogeneity":
            is_active = any(
                token in trigger_blob
                for token in ("cross_database", "cross database", "site", "transportability")
            )
        elif module.module_id == "measurement_process_audit":
            is_active = any(
                token in trigger_blob
                for token in ("missingness", "measurement", "source absence")
            )
        elif module.module_id == "exposure_outcome_distribution":
            is_active = any(
                token in trigger_blob
                for token in ("prevalence", "incidence", "event rate", "mortality rate")
            )
        else:
            is_active = any(trigger.lower() in trigger_blob for trigger in module.triggers)
        if is_active:
            modules.append(module.model_copy(deep=True))
    return modules


def brief_check_terms(item: str) -> Sequence[str] | None:
    return _BRIEF_CHECK_TERMS.get(item)


def role_check_terms(role: str) -> Sequence[str]:
    return _ROLE_CHECK_TERMS.get(role, ())


__all__ = [
    "DisplayModuleSpec",
    "DisplayTier",
    "StudyDesignFamily",
    "anti_patterns_for_brief",
    "brief_check_terms",
    "design_principles_for_family",
    "display_modules_for_family",
    "family_template",
    "role_check_terms",
    "triggered_generic_modules",
]
