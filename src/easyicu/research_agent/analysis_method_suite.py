"""Per-family analysis-method SUITE registry — the standard reviewer method set.

``capability_registry`` answers one question: *for a study-design family, what is
the ONE primary estimand, and is it computed deterministically?* This module
answers the very next question a reviewer (or a user who knows the field) asks:

    "For a prediction study you'd normally also see calibration, a
    decision-curve / net-benefit analysis, DeLong confidence intervals on the
    AUROC, and SHAP feature attribution. For survival you'd want a
    proportional-hazards check and Kaplan-Meier + log-rank. For causal work a
    covariate-balance love plot, a positivity check, and an E-value. Does
    EasyICU do these, and which are primary vs supporting vs still planned?"

Today that answer is scattered across three disconnected surfaces — the primary
estimand lives in ``capability_registry``, which *packages* import lives in
``method_capabilities``, which reviewer *items* a run satisfied lives in
``reporting_checklist`` — plus the figure renderers hard-code which panels exist.
There is no single place that declares, per family, the full method suite. This
module is that place.

Design contract (kept honest by
``tests/research_agent/test_analysis_method_suite.py``):

* Every suite ``family`` is a real :data:`StudyDesignFamily` and has a record in
  ``capability_registry``.
* Each suite has at least one ``primary`` method, and every deterministic primary
  method's ``runner`` is a runner ``capability_registry`` actually wires
  (``_PRIMARY_DETERMINISTIC_RUNNERS`` / ``AUXILIARY_DETERMINISTIC_RUNNERS``).
* ``tier`` and ``implementation`` use a **closed vocabulary**. A ``planned``
  method carries **no runner** — it is recognised but not implemented, so it must
  *fail closed* if requested as a primary estimand and is **never** silently
  approximated by a nearby method (competing-risks CIF stays a Cox-HR-is-not-a-CIF
  boundary, not a Cox HR wearing a CIF label).
* A ``deterministic`` method must name where it is produced (``runner`` non-None).
* ``docs/analysis_method_suite.md`` is regenerated from
  :func:`render_method_suite_markdown` and drift-tested for equality.

This module imports only the family type and the primary/auxiliary runner names
from ``capability_registry`` — no prompts, no LLM calls, no pipeline import — so
it stays a light, unit-testable declaration a reviewer can read top-to-bottom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .capability_registry import (
    AUXILIARY_DETERMINISTIC_RUNNERS,
    CAPABILITY_REGISTRY,
)
from .study_design_playbook import StudyDesignFamily

__all__ = [
    "METHOD_TIERS",
    "METHOD_IMPLEMENTATIONS",
    "AnalysisMethod",
    "MethodSuite",
    "METHOD_SUITE_REGISTRY",
    "get_suite",
    "methods_by_tier",
    "planned_methods",
    "deterministic_methods",
    "supporting_methods",
    "render_method_suite_markdown",
]


# Closed vocabularies. A method's place in the analysis is a *tier*; how it is
# produced is an *implementation*. Keeping the two orthogonal is deliberate: a
# standard-supporting method can be deterministic, llm-coded, or still planned.
METHOD_TIERS: Tuple[str, ...] = (
    "primary",  # the reported estimand — owns the headline
    "standard_supporting",  # diagnostics/robustness a competent reviewer expects
    "exploratory",  # optional deeper add-on, labelled exploratory (no overclaim)
    "planned",  # recognised, not implemented; fail-closed, never approximated
)

METHOD_IMPLEMENTATIONS: Tuple[str, ...] = (
    "deterministic",  # deterministic runner or figure panel; source-data-backed
    "llm_coded",  # produced by LLM analysis code when the plan asks; value-verified
    "planned",  # not implemented; must fail closed if required as a primary estimand
)


@dataclass(frozen=True)
class AnalysisMethod:
    """One method in a family's suite, with its tier and how it is produced."""

    key: str
    name: str
    purpose: str
    tier: str  # one of METHOD_TIERS
    implementation: str  # one of METHOD_IMPLEMENTATIONS
    produces: str  # source-data table / figure panel it emits (or would)
    runner: Optional[str] = None  # deterministic entrypoint / figure renderer / None
    reporting_items: Tuple[str, ...] = ()  # checklist items it helps satisfy
    notes: str = ""


@dataclass(frozen=True)
class MethodSuite:
    """The standard method suite for one study-design family."""

    family: StudyDesignFamily
    label: str
    methods: Tuple[AnalysisMethod, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# The registry. Ordered primary -> standard_supporting -> exploratory -> planned
# within each family so the markdown reads like a methods section.
# ---------------------------------------------------------------------------

_PREDICTION = MethodSuite(
    family="prediction",
    label="Prediction / risk modelling",
    methods=(
        AnalysisMethod(
            key="discrimination_calibration",
            name="Discrimination + calibration (AUROC + calibration curve)",
            purpose="Primary model performance: how well the model ranks and how usable its probabilities are.",
            tier="primary",
            implementation="llm_coded",
            produces="AUROC + calibration curve + ROC panels (figure deterministic)",
            runner="prediction",  # deterministic FIGURE renderer; the FIT is LLM-coded
            reporting_items=("TRIPOD+AI 15", "TRIPOD+AI 17"),
            notes="Model FIT is LLM-coded (value-provenance verified); the calibration+ROC FIGURE is deterministic.",
        ),
        AnalysisMethod(
            key="calibration_metrics",
            name="Calibration slope/intercept + Brier score",
            purpose="Quantify calibration beyond the visual curve; separate calibration-in-the-large from slope.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="calibration metrics row; Brier annotated on the calibration panel when present",
            runner=None,
            reporting_items=("TRIPOD+AI 10c", "TRIPOD+AI 15"),
            notes="Brier is displayed by the deterministic figure when the run emits it; a deterministic calibration-metric runner is PLANNED.",
        ),
        AnalysisMethod(
            key="delong_ci",
            name="DeLong CI / test on AUROC",
            purpose="Confidence interval on discrimination and formal comparison of two models' AUROCs.",
            tier="standard_supporting",
            implementation="planned",
            produces="auroc_delong.csv (auroc, ci_low, ci_high [, comparison p])",
            runner=None,
            reporting_items=("TRIPOD+AI 15",),
            notes="Deterministic supporting runner PLANNED (WS5). Until then discrimination CI is LLM-coded if the plan asks.",
        ),
        AnalysisMethod(
            key="decision_curve",
            name="Decision-curve analysis / net benefit (DCA)",
            purpose="Clinical utility across threshold probabilities vs treat-all / treat-none.",
            tier="standard_supporting",
            implementation="planned",
            produces="decision_curve.csv (threshold, net_benefit, net_benefit_all)",
            runner=None,
            reporting_items=("TRIPOD+AI 19",),
            notes="Already a TRIPOD+AI checklist item; deterministic DCA runner + panel PLANNED (WS5).",
        ),
        AnalysisMethod(
            key="threshold_metrics",
            name="Sensitivity/specificity/PPV/NPV at clinical thresholds",
            purpose="Operating-point performance and confusion matrix at decision-relevant cut-offs.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="threshold_metrics.csv; confusion matrix at chosen thresholds",
            runner=None,
            reporting_items=("TRIPOD+AI 15",),
        ),
        AnalysisMethod(
            key="feature_attribution",
            name="Feature attribution (SHAP / permutation importance)",
            purpose="Which predictors drive the model; per-feature contribution for interpretability.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="feature_importance.csv; beeswarm/bar summary",
            runner=None,
            reporting_items=("TRIPOD+AI 10a",),
            notes="`shap` is a curated importable package with a permutation-importance fallback (method_capabilities). A deterministic SHAP-summary panel is PLANNED (WS5).",
        ),
        AnalysisMethod(
            key="subgroup_fairness",
            name="Subgroup / fairness performance",
            purpose="Discrimination + calibration within demographic strata (age, sex).",
            tier="standard_supporting",
            implementation="deterministic",
            produces="per-subgroup performance table",
            runner="fairness",  # research_agent/methods/fairness.py
            reporting_items=("TRIPOD+AI 12", "TRIPOD+AI 18"),
        ),
        AnalysisMethod(
            key="internal_validation",
            name="Internal validation (bootstrap optimism / cross-validation)",
            purpose="Correct optimism from evaluating on the development data.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="internal-validation performance table",
            runner=None,
            reporting_items=("TRIPOD+AI 11",),
        ),
        AnalysisMethod(
            key="conformal_intervals",
            name="Conformal prediction intervals",
            purpose="Distribution-free per-patient predictive uncertainty with coverage guarantees.",
            tier="exploratory",
            implementation="deterministic",
            produces="conformal coverage table",
            runner="conformal",  # research_agent/methods/conformal.py
        ),
        AnalysisMethod(
            key="external_validation",
            name="External / cross-database validation",
            purpose="Transportability of discrimination + calibration to a second ICU database.",
            tier="planned",
            implementation="planned",
            produces="external-cohort performance + recalibration",
            runner=None,
            reporting_items=("TRIPOD+AI 16",),
        ),
        AnalysisMethod(
            key="reclassification",
            name="Net reclassification / IDI vs a baseline model",
            purpose="Incremental value of new predictors over an established score.",
            tier="planned",
            implementation="planned",
            produces="NRI / IDI table",
            runner=None,
        ),
        AnalysisMethod(
            key="dynamic_prediction",
            name="Dynamic / landmark prediction (time-updated risk)",
            purpose="Re-estimated risk from time-updated features (landmarking).",
            tier="planned",
            implementation="planned",
            produces="landmark performance over time",
            runner=None,
        ),
    ),
)

_SURVIVAL = MethodSuite(
    family="time_to_event",
    label="Survival / time-to-event",
    methods=(
        AnalysisMethod(
            key="cox_hr",
            name="Cox proportional-hazards hazard ratio",
            purpose="Primary adjusted effect of exposure on the event hazard.",
            tier="primary",
            implementation="llm_coded",
            produces="agent-declared Cox result (hazard_ratio, ci_low, ci_high) + deterministic forest panel",
            runner="time_to_event",
            reporting_items=("STROBE 16",),
            notes="The agent owns time zero, censoring, exposure, adjustment, and fit; the runner renders registered Cox/KM products only.",
        ),
        AnalysisMethod(
            key="km_logrank",
            name="Kaplan-Meier curves + log-rank by exposure",
            purpose="Unadjusted survival over time by exposure group with the log-rank test.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="kaplan_meier curve data + log-rank; KM panel",
            runner="time_to_event",  # deterministic figure renderer
            reporting_items=("STROBE 15",),
        ),
        AnalysisMethod(
            key="ph_check",
            name="Proportional-hazards check (Schoenfeld residuals / PH test)",
            purpose="Test the core Cox assumption; a violated PH makes a single HR misleading.",
            tier="standard_supporting",
            implementation="planned",
            produces="schoenfeld_test.csv (covariate, chi2, p) + schoenfeld_plot (diagnostics panel slot)",
            runner=None,
            reporting_items=("STROBE 12a",),
            notes="The survival figure already has a diagnostics panel that ACCEPTS a schoenfeld_plot; a deterministic PH-test runner to FILL it is PLANNED (WS5).",
        ),
        AnalysisMethod(
            key="subgroup_hr",
            name="Subgroup hazard ratios / interaction forest",
            purpose="Effect heterogeneity across prespecified subgroups.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="subgroup HR forest",
            runner=None,
            reporting_items=("STROBE 12b",),
        ),
        AnalysisMethod(
            key="rmst",
            name="Restricted mean survival time (RMST)",
            purpose="Difference in mean event-free time up to a horizon — interpretable when PH is dubious.",
            tier="standard_supporting",
            implementation="planned",
            produces="rmst.csv (group, rmst, ci) + difference",
            runner=None,
        ),
        AnalysisMethod(
            key="competing_risks_cif",
            name="Competing-risks cumulative incidence (Fine-Gray / CIF)",
            purpose="Cause-specific cumulative incidence when a competing event (e.g. death) precludes the outcome.",
            tier="planned",
            implementation="planned",
            produces="cause-specific CIF",
            runner=None,
            notes="A cause-naive Cox HR is NOT a CIF — this stays a KNOWN_UNSUPPORTED_ESTIMAND that fails closed, never approximated.",
        ),
        AnalysisMethod(
            key="time_varying_hr",
            name="Time-varying coefficients / landmark survival",
            purpose="Relax the constant-HR assumption over follow-up.",
            tier="planned",
            implementation="planned",
            produces="time-varying HR / landmark curves",
            runner=None,
        ),
    ),
)

_CAUSAL = MethodSuite(
    family="causal_emulation",
    label="Causal inference / target-trial emulation",
    methods=(
        AnalysisMethod(
            key="iptw_or",
            name="Stabilised-IPTW marginal odds ratio",
            purpose="Primary marginal causal contrast under a target-trial protocol.",
            tier="primary",
            implementation="llm_coded",
            produces="agent-declared causal effect table + target-trial/identification protocol",
            runner="causal_emulation",
            reporting_items=("STROBE 16",),
            notes="The agent owns the estimator, exposure/outcome, covariates, and identification assumptions; deterministic code renders registered products only.",
        ),
        AnalysisMethod(
            key="covariate_balance",
            name="Covariate balance (SMD love plot, before vs after weighting)",
            purpose="Show the weighting removed measured confounding imbalance.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="covariate_balance.csv (covariate, smd_before, smd_after) + love-plot panel",
            runner="causal_emulation",  # deterministic figure renderer + causal_audit
            reporting_items=("STROBE 9",),
        ),
        AnalysisMethod(
            key="positivity_overlap",
            name="Positivity / overlap (propensity distribution + trimming)",
            purpose="Confirm treated/untreated overlap so weights are not dominated by extreme units.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="propensity distribution + trimming report",
            runner=None,
            reporting_items=("STROBE 9",),
            notes="Computed by the agent under the declared causal method; the deterministic causal figure can display the registered overlap product.",
        ),
        AnalysisMethod(
            key="evalue",
            name="E-value (sensitivity to unmeasured confounding)",
            purpose="How strong an unmeasured confounder would have to be to explain away the effect.",
            tier="standard_supporting",
            implementation="planned",
            produces="evalue.csv (point_estimate, evalue, evalue_ci)",
            runner=None,
            reporting_items=("STROBE 12e",),
            notes="Deterministic E-value runner PLANNED (WS5); shared with the association family.",
        ),
        AnalysisMethod(
            key="negative_control",
            name="Negative-control outcome / exposure",
            purpose="Detect residual confounding via an association that should be null.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="negative-control contrast",
            runner=None,
            reporting_items=("STROBE 12e",),
        ),
        AnalysisMethod(
            key="doubly_robust",
            name="Doubly-robust / matching sensitivity (AIPW, TMLE, PS matching)",
            purpose="Re-estimate the effect under an alternative identification to test IPTW robustness.",
            tier="exploratory",
            implementation="planned",
            produces="alternative-estimator effect table",
            runner=None,
        ),
        AnalysisMethod(
            key="g_methods",
            name="g-formula / marginal structural models (time-varying treatment)",
            purpose="Handle time-varying treatment and confounding affected by prior treatment.",
            tier="planned",
            implementation="planned",
            produces="g-computation / MSM effect",
            runner=None,
        ),
    ),
)

_ASSOCIATION = MethodSuite(
    family="association",
    label="Association (general + graded ordinal / dose-response)",
    methods=(
        AnalysisMethod(
            key="ordinal_trend",
            name="Ordinal dose-response (OR per +1 stage, per-stage forest, monotonicity)",
            purpose="Primary graded-exposure trend when the exposure has >=3 ordered levels.",
            tier="primary",
            implementation="llm_coded",
            produces="agent-declared ordered trend + dose_response.csv; deterministic forest",
            runner=None,
            reporting_items=("STROBE 16",),
            notes="Validated ordered-trend primitives are available to agent code, but the framework never selects the exposure, scores, adjustment set, or model.",
        ),
        AnalysisMethod(
            key="adjusted_association",
            name="Adjusted association (logistic / linear)",
            purpose="Primary confounder-adjusted effect for a general (non-graded) exposure.",
            tier="primary",
            implementation="llm_coded",
            produces="adjusted effect estimate (bound via NumericClaim) + forest (deterministic figure)",
            runner=None,
            reporting_items=("STROBE 16",),
            notes="LLM-coded fit; the forest/strata/missingness FIGURE is deterministic (base_association_skill).",
        ),
        AnalysisMethod(
            key="multiple_adjustment",
            name="Multiple adjustment sets (crude / minimal / full)",
            purpose="Show the effect's sensitivity to the confounder set.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="nested-model effect table",
            runner=None,
            reporting_items=("STROBE 12a", "STROBE 16"),
        ),
        AnalysisMethod(
            key="effect_modification",
            name="Effect modification / interaction + subgroup forest",
            purpose="Whether the association differs across prespecified subgroups.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="interaction test + subgroup forest",
            runner=None,
            reporting_items=("STROBE 12b",),
        ),
        AnalysisMethod(
            key="missingness_audit",
            name="Missing-data audit + complete-case vs imputation sensitivity",
            purpose="Characterise missingness and show the effect is stable to the handling choice.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="missingness_summary.csv + sensitivity contrast",
            runner="missing_data",  # research_agent/methods/missing_data.py
            reporting_items=("STROBE 12c",),
        ),
        AnalysisMethod(
            key="multiple_testing",
            name="Multiple-testing correction (FDR / Bonferroni)",
            purpose="Control the family-wise / false-discovery rate across reported associations.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="adjusted p-value table",
            runner="multiple_testing",  # research_agent/methods/multiple_testing.py
            reporting_items=("STROBE 12e",),
        ),
        AnalysisMethod(
            key="rcs_spline",
            name="Restricted cubic spline dose-response (continuous exposure)",
            purpose="Non-linear exposure-response without imposing linearity or arbitrary cut-points.",
            tier="standard_supporting",
            implementation="planned",
            produces="spline_dose_response.csv (x, log_or, ci) + spline panel",
            runner=None,
        ),
        AnalysisMethod(
            key="evalue",
            name="E-value (sensitivity to unmeasured confounding)",
            purpose="Robustness of an observational association to an unmeasured confounder.",
            tier="standard_supporting",
            implementation="planned",
            produces="evalue.csv",
            runner=None,
            reporting_items=("STROBE 12e",),
        ),
        AnalysisMethod(
            key="robustness_panel",
            name="Robustness panel (alternative specifications)",
            purpose="Aggregate the effect across alternative cohort/model choices.",
            tier="exploratory",
            implementation="deterministic",
            produces="robustness_panel.csv",
            runner="robustness_sensitivity",  # deterministic_robustness + renderer
            reporting_items=("STROBE 12e",),
        ),
        AnalysisMethod(
            key="mediation",
            name="Mediation / quantitative bias analysis",
            purpose="Decompose direct/indirect effects or quantify plausible bias.",
            tier="planned",
            implementation="planned",
            produces="mediation / bias-analysis table",
            runner=None,
        ),
    ),
)

_PHENOTYPING = MethodSuite(
    family="phenotyping",
    label="Phenotyping / clustering (cross-sectional + longitudinal trajectory)",
    methods=(
        AnalysisMethod(
            key="cluster_solution",
            name="Cluster solution + stability (cross-sectional subphenotypes)",
            purpose="Primary unsupervised grouping of patients on baseline/early features.",
            tier="primary",
            implementation="llm_coded",
            produces="cluster assignments + stability; heatmap + stability + outcome-by-cluster figure (deterministic)",
            runner="phenotyping",  # deterministic figure renderer; the FIT is LLM-coded
            reporting_items=("internal_phenotype P4", "internal_phenotype P5"),
        ),
        AnalysisMethod(
            key="k_selection",
            name="Number-of-clusters selection (silhouette / gap / BIC)",
            purpose="Justify k rather than assume it.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="k-selection criterion curve",
            runner=None,
            reporting_items=("internal_phenotype P4",),
        ),
        AnalysisMethod(
            key="cluster_stability",
            name="Stability / reproducibility (bootstrap, consensus, adjusted Rand)",
            purpose="Show the clusters are not an artefact of the seed/sample.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="stability report",
            runner=None,
            reporting_items=("internal_phenotype P5",),
        ),
        AnalysisMethod(
            key="trajectory_cluster_stability",
            name="Typed trajectory-cluster stability refits",
            purpose=(
                "Replay an agent-specified subsampling/refit design for one already "
                "selected trajectory model and cluster count."
            ),
            tier="standard_supporting",
            implementation="deterministic",
            produces=(
                "digest-bound stability specification, refit ledger, adjusted-Rand "
                "table, aligned assignments, and freeze/report decision"
            ),
            runner="trajectory_cluster_stability",
            reporting_items=("internal_phenotype P5",),
            notes=(
                "The planner owns every scientific and randomization choice. The "
                "executor supports only the closed observed-data diagonal-GMM "
                "contract and fails closed outside it; general cluster stability "
                "remains agent-coded."
            ),
        ),
        AnalysisMethod(
            key="cluster_sizes",
            name="Cluster sizes + degenerate-cluster flag",
            purpose="Report each group's size and flag near-empty clusters.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="cluster_sizes.csv; figure panel",
            runner="phenotyping",
            reporting_items=("internal_phenotype P6",),
        ),
        AnalysisMethod(
            key="outcome_by_cluster",
            name="Outcome-by-cluster descriptive comparison",
            purpose="Compare outcomes across clusters — DESCRIPTIVE, explicitly not causal.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="outcome_by_cluster.csv; figure panel",
            runner="phenotyping",
            reporting_items=("internal_phenotype P8",),
            notes="figure_strategy blocks 'clusters are causal entities'; kept descriptive by contract.",
        ),
        # --- Longitudinal trajectory sub-suite ---
        AnalysisMethod(
            key="trajectory_feature_clustering",
            name="Trajectory-feature clustering",
            purpose="Longitudinal phenotyping using the agent-declared representation and unsupervised method appropriate to the planned question and data.",
            tier="primary",
            implementation="llm_coded",
            produces="agent-declared feature representation + cluster assignments + silhouette/stability/size QC + outcome-by-trajectory",
            runner="phenotyping",
            reporting_items=("internal_phenotype P4", "internal_phenotype P5", "internal_phenotype P9"),
            notes="The agent owns the feature representation, time horizon, clustering method, and k-selection. The `phenotyping` runner only renders standardized, source-backed products; it never chooses the science. Trajectory-feature clustering is deliberately not relabelled as LCGA.",
        ),
        AnalysisMethod(
            key="lcga_gbtm",
            name="LCGA / group-based trajectory modelling (GBTM)",
            purpose="Latent-class growth / group-based trajectory model — a model-based longitudinal method.",
            tier="planned",
            implementation="planned",
            produces="latent-class trajectory model",
            runner=None,
            notes="PLANNED. Do NOT relabel the deterministic feature-clustering path as LCGA; they are different methods.",
        ),
        AnalysisMethod(
            key="mixed_effects_growth",
            name="Mixed-effects / growth-mixture trajectory models",
            purpose="Random-effects longitudinal models with latent classes.",
            tier="planned",
            implementation="planned",
            produces="mixed-effects / GMM trajectory model",
            runner=None,
        ),
        AnalysisMethod(
            key="dtw_timeseries",
            name="DTW / time-series distance clustering",
            purpose="Cluster raw trajectories under a shape-aware (dynamic time warping) distance.",
            tier="planned",
            implementation="planned",
            produces="DTW distance clustering",
            runner=None,
        ),
        AnalysisMethod(
            key="landmark_trajectory_prediction",
            name="Landmark trajectory prediction",
            purpose="Predict outcome from trajectory shape up to a landmark time.",
            tier="planned",
            implementation="planned",
            produces="landmark trajectory-prediction model",
            runner=None,
        ),
    ),
)

_DESCRIPTIVE = MethodSuite(
    family="descriptive",
    label="Descriptive / measurement audit",
    methods=(
        AnalysisMethod(
            key="descriptive_summary",
            name="Descriptive summaries / Table 1 / measurement-process audit",
            purpose="Primary characterisation of the cohort and measurement process.",
            tier="primary",
            implementation="llm_coded",
            produces="descriptive summary table (bound); base figure deterministic",
            runner=None,
            reporting_items=("STROBE 14a",),
        ),
        AnalysisMethod(
            key="table_one",
            name="Baseline-characteristics table (Table 1)",
            purpose="Participant characteristics by group with appropriate summaries.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="table_one.csv",
            runner=None,
            reporting_items=("STROBE 14a",),
        ),
        AnalysisMethod(
            key="missingness_audit",
            name="Missing-data / completeness audit",
            purpose="Report per-variable missingness and measurement coverage.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="missingness_summary.csv",
            runner="missing_data",
            reporting_items=("STROBE 12c",),
        ),
    ),
)


METHOD_SUITE_REGISTRY: Tuple[MethodSuite, ...] = (
    _SURVIVAL,
    _CAUSAL,
    _ASSOCIATION,
    _PREDICTION,
    _PHENOTYPING,
    _DESCRIPTIVE,
)


# ---------------------------------------------------------------------------
# Names the registry may legitimately cite as a deterministic PRIMARY runner —
# exactly the runners capability_registry wires. Supporting/exploratory
# deterministic methods may cite a module or figure-renderer name (checked for
# shape, not membership, since those are not capability_registry primaries).
# ---------------------------------------------------------------------------

def _known_primary_runner_names() -> frozenset:
    names = {c.primary_runner for c in CAPABILITY_REGISTRY if c.primary_runner}
    names |= {a.name for a in AUXILIARY_DETERMINISTIC_RUNNERS}
    return frozenset(names)


KNOWN_PRIMARY_RUNNER_NAMES: frozenset = _known_primary_runner_names()


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def get_suite(family: StudyDesignFamily) -> Optional[MethodSuite]:
    """Return the method suite for a family, or ``None`` if unregistered."""
    for suite in METHOD_SUITE_REGISTRY:
        if suite.family == family:
            return suite
    return None


def methods_by_tier(family: StudyDesignFamily, tier: str) -> Tuple[AnalysisMethod, ...]:
    """Methods of a family at a given tier (empty tuple if none / unknown family)."""
    suite = get_suite(family)
    if suite is None:
        return ()
    return tuple(m for m in suite.methods if m.tier == tier)


def planned_methods() -> Tuple[Tuple[str, AnalysisMethod], ...]:
    """(family, method) pairs for every PLANNED method — the honest roadmap.

    A ``planned`` method is recognised but not implemented; it must fail closed
    if requested as a primary estimand and is never silently approximated.
    """
    out = []
    for suite in METHOD_SUITE_REGISTRY:
        for m in suite.methods:
            if m.tier == "planned" or m.implementation == "planned":
                out.append((suite.family, m))
    return tuple(out)


def deterministic_methods() -> Tuple[Tuple[str, AnalysisMethod], ...]:
    """(family, method) pairs for every method produced deterministically today."""
    out = []
    for suite in METHOD_SUITE_REGISTRY:
        for m in suite.methods:
            if m.implementation == "deterministic":
                out.append((suite.family, m))
    return tuple(out)


def supporting_methods(family: StudyDesignFamily) -> Tuple[AnalysisMethod, ...]:
    """Standard-supporting methods for a family (the reviewer-expected depth set)."""
    return methods_by_tier(family, "standard_supporting")


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _impl_badge(impl: str) -> str:
    if impl == "deterministic":
        return "deterministic ✅"
    if impl == "llm_coded":
        return "LLM-coded ⚠️"
    return "planned ⛔"


def render_method_suite_markdown() -> str:
    """Render the per-family method-suite matrix as Markdown.

    ``docs/analysis_method_suite.md`` is generated from this and drift-tested.
    """
    lines = [
        "# EasyICU research-agent analysis-method suite",
        "",
        "_Generated from `easyicu.research_agent.analysis_method_suite`. Do not edit "
        "by hand — edit the registry and regenerate._",
        "",
        "`capability_registry` declares the ONE primary estimand per family. This "
        "matrix declares the **full standard method suite** a reviewer expects for "
        "each family, and how each method is produced:",
        "",
        "- **Tier** — `primary` (the reported estimand) · `standard_supporting` "
        "(diagnostics/robustness a reviewer routinely expects) · `exploratory` "
        "(optional deeper add-on, labelled as such) · `planned` (recognised, not yet "
        "implemented — fails closed, never approximated).",
        "- **Implementation** — deterministic (source-data-backed runner/panel) · "
        "LLM-coded (value-provenance verified) · planned.",
        "",
        "A `planned` method carries no runner. It must fail closed if requested as a "
        "primary estimand — e.g. competing-risks CIF is never answered with a Cox HR.",
        "",
    ]
    for suite in METHOD_SUITE_REGISTRY:
        lines.append(f"## {suite.label}")
        lines.append("")
        lines.append("| Method | Tier | Implementation | Produces | Runner |")
        lines.append("| --- | --- | --- | --- | --- |")
        for m in suite.methods:
            runner = f"`{m.runner}`" if m.runner else "—"
            lines.append(
                f"| {m.name} | {m.tier} | {_impl_badge(m.implementation)} | "
                f"{m.produces} | {runner} |"
            )
        lines.append("")
    # A compact honest roadmap of what is planned.
    lines.append("## Planned methods (declared, not implemented — fail closed)")
    lines.append("")
    lines.append("| Family | Method | Why it matters |")
    lines.append("| --- | --- | --- |")
    for family, m in planned_methods():
        lines.append(f"| {family} | {m.name} | {m.purpose} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual regen
    print(render_method_suite_markdown())
