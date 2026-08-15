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
    "FIGURE_SOURCE_OBLIGATIONS",
    "AnalysisMethod",
    "MethodSuite",
    "METHOD_SUITE_REGISTRY",
    "get_suite",
    "methods_by_tier",
    "planned_methods",
    "deterministic_methods",
    "supporting_methods",
    "figure_product_source_obligations",
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

FIGURE_SOURCE_OBLIGATIONS: Tuple[str, ...] = (
    "effect:subgroup",
    "prediction:calibration",
    "prediction:decision",
    "prediction:performance",
    "prediction:roc",
    "prediction:time_varying_discrimination",
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
    figure_source_contracts: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    # Exact reviewed execution resources associated with this method. These are
    # coordinates only: the runtime still verifies that a package is installed,
    # and a Coder-reachable kernel does not become a deterministic host owner.
    # Keeping the binding beside the method prevents Planner and Coder catalogs
    # from maintaining two prose-derived maps of the same scientific capability.
    kernel_modules: Tuple[str, ...] = ()
    software_packages: Tuple[str, ...] = ()
    # Exact, reviewable expansion metadata.  It gives the action owner enough
    # information to explain a gap without guessing from method prose.  A
    # decomposition is scientifically equivalent only when declared here;
    # alternatives always require user confirmation and are never substituted.
    required_inputs: Tuple[str, ...] = ()
    composition_action_ids: Tuple[str, ...] = ()
    alternative_action_ids: Tuple[str, ...] = ()
    primary_for_analysis_types: Tuple[str, ...] = ()


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
            figure_source_contracts=(
                (
                    "figure:discrimination_calibration",
                    (
                        "prediction:performance",
                        "prediction:calibration",
                        "prediction:roc",
                    ),
                ),
                ("figure:model_performance", ("prediction:performance",)),
                ("figure:prediction_performance", ("prediction:performance",)),
                ("figure:discrimination", ("prediction:performance",)),
                ("figure:roc_curve", ("prediction:roc",)),
                (
                    "figure:receiver_operating_characteristic",
                    ("prediction:roc",),
                ),
                ("figure:calibration_curve", ("prediction:calibration",)),
                ("figure:calibration_plot", ("prediction:calibration",)),
            ),
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
            implementation="llm_coded",
            produces="auroc_delong.csv (auroc, ci_low, ci_high [, comparison p])",
            runner=None,
            reporting_items=("TRIPOD+AI 15",),
            notes=(
                "`methods.delong_auc` (delong_auc_ci / delong_test) is a "
                "reviewed, tested kernel offered to the Coder via "
                "CURATED_METHOD_KERNELS; the Coder calls it instead of "
                "re-deriving the DeLong variance. A deterministic host runner "
                "that OWNS the step is still planned."
            ),
            kernel_modules=("delong_auc",),
        ),
        AnalysisMethod(
            key="decision_curve",
            name="Decision-curve analysis / net benefit (DCA)",
            purpose="Clinical utility across threshold probabilities vs treat-all / treat-none.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="decision_curve.csv (threshold, net_benefit, net_benefit_all)",
            runner=None,
            reporting_items=("TRIPOD+AI 19",),
            notes=(
                "`methods.decision_curve` (net_benefit_curve / "
                "summarize_decision_curve) is a reviewed, tested kernel offered "
                "to the Coder via CURATED_METHOD_KERNELS. A deterministic host "
                "runner + panel that OWN the step are still planned."
            ),
            figure_source_contracts=(
                ("figure:decision_curve", ("prediction:decision",)),
            ),
            kernel_modules=("decision_curve",),
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
            notes="`shap` is a curated importable package with a permutation-importance fallback (contracts/method_packages.py). A deterministic SHAP-summary panel is PLANNED (WS5).",
            software_packages=("shap",),
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
            implementation="llm_coded",
            produces="conformal coverage table",
            runner=None,
            notes=(
                "Corrected 2026-07-30: this was declared deterministic, but no "
                "host code calls methods/conformal.py -- the old guard accepted "
                "'the module is importable' as proof of production. It is now a "
                "reviewed kernel offered to the Coder via CURATED_METHOD_KERNELS."
            ),
            kernel_modules=("conformal",),
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
            alternative_action_ids=("prediction.internal_validation",),
        ),
        AnalysisMethod(
            key="reclassification",
            name="Net reclassification / IDI vs a baseline model",
            purpose="Incremental value of new predictors over an established score.",
            tier="planned",
            implementation="planned",
            produces="NRI / IDI table",
            runner=None,
            alternative_action_ids=("prediction.decision_curve",),
        ),
        AnalysisMethod(
            key="dynamic_prediction",
            name="Dynamic / landmark prediction (time-updated risk)",
            purpose="Re-estimated risk from time-updated features (landmarking).",
            tier="primary",
            implementation="llm_coded",
            produces="landmark performance over time",
            runner=None,
            notes=(
                "The host compiles leakage-safe landmark rows and evaluates "
                "predictions with reviewed primitives; the Coder fits an exact "
                "Planner-declared sklearn Pipeline. Patient-level splitting, "
                "landmarks and horizons are mandatory. The capability remains "
                "analysis-only until a typed host model-fit/result validator exists."
            ),
            kernel_modules=("dynamic_prediction", "temporal_features"),
            software_packages=("sklearn",),
            required_inputs=(
                "longitudinal rows with patient/stay identity and measurement time",
                "prespecified prediction landmarks, lookback windows and target horizons",
                "event time plus follow-up/censoring time",
                "patient-level development/validation split",
            ),
            composition_action_ids=(
                "prediction.discrimination_calibration",
                "prediction.calibration_metrics",
                "prediction.internal_validation",
            ),
            alternative_action_ids=("prediction.discrimination_calibration",),
            primary_for_analysis_types=("dynamic_prediction",),
            figure_source_contracts=(
                (
                    "figure:time_varying_discrimination",
                    ("prediction:time_varying_discrimination",),
                ),
            ),
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
            implementation="deterministic",
            produces="host-bound Cox result + PH diagnostic + digest receipt",
            runner="survival_primary_cox",
            reporting_items=("STROBE 16",),
            notes="The Planner fixes time zero, censoring, exposure, adjustment and horizon; the sealed host owner fits and receipts that exact contract.",
            software_packages=("lifelines",),
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
            software_packages=("lifelines",),
        ),
        AnalysisMethod(
            key="ph_check",
            name="Proportional-hazards check (Schoenfeld residuals / PH test)",
            purpose="Test the core Cox assumption; a violated PH makes a single HR misleading.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="schoenfeld_test.csv (covariate, chi2, p) + schoenfeld_plot (diagnostics panel slot)",
            runner="survival_primary_cox",
            reporting_items=("STROBE 12a",),
            notes=(
                "`methods.ph_schoenfeld` (ph_test / run_ph_test, over "
                "lifelines.statistics.proportional_hazard_test) is a reviewed, "
                "tested kernel executed by the sealed primary Cox owner. The "
                "diagnostic table and its SHA are bound into the host receipt."
            ),
            kernel_modules=("ph_schoenfeld",),
            software_packages=("lifelines",),
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
            implementation="llm_coded",
            produces="rmst.csv (group, rmst, ci) + difference",
            runner=None,
            notes=(
                "`methods.rmst` (rmst / rmst_difference) is a reviewed, tested "
                "kernel offered to the Coder via CURATED_METHOD_KERNELS. It "
                "computes the integral-form sampling SE deliberately: "
                "lifelines' restricted_mean_survival_time(return_variance=True) "
                "returns the population variance, which inflates the CI by "
                "~sqrt(n). A deterministic host runner is still planned."
            ),
            kernel_modules=("rmst",),
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
            implementation="deterministic",
            produces="e_values.csv (term, odds_ratio, ci, e_value, e_value_lower_bound)",
            runner="sensitivity",  # methods/sensitivity.py::compute_e_value
            reporting_items=("STROBE 12e",),
            notes=(
                "Produced by the host in orchestration/finalize.py over every "
                "primary-effect row; shared with the association family. "
                "Converting an OR to an RR needs a baseline event rate, and it "
                "is ALWAYS this run's own observed rate, read from its "
                "outcome-rate product and named in e_values.md. When no "
                "unambiguous observed rate exists the E-value is not reported "
                "at all: compute_e_value raises rather than assume one. Until "
                "2026-07-30 it assumed 0.1, which for OR=2.0 gives E=3.04 "
                "against 2.68 at an observed 0.214 -- overstating robustness."
            ),
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
            implementation="deterministic",
            produces="host-bound adjusted effect + typed model/coefficient contract + deterministic forest",
            runner="adjusted_association_estimates",
            reporting_items=("STROBE 16",),
            notes="The sealed owner claims only one exact supported estimator with explicit coding for every term; interactions, splines and multi-model/free-form kernels remain agent-coded.",
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
            figure_source_contracts=(("figure:subgroup_forest", ("effect:subgroup",)),),
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
            implementation="deterministic",
            produces="e_values.csv",
            runner="sensitivity",  # methods/sensitivity.py::compute_e_value
            reporting_items=("STROBE 12e",),
            notes=(
                "Same host-produced artifact as the causal family's entry, "
                "including its observed-event-rate requirement."
            ),
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
            reporting_items=(
                "internal_phenotype P4",
                "internal_phenotype P5",
                "internal_phenotype P9",
            ),
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


def figure_product_source_obligations(product: object) -> Tuple[str, ...]:
    """Return reviewer-suite source obligations for one exact typed figure.

    This is the shared semantic registry consumed by figure lineage validation;
    validators must not maintain a second list of benchmark or suite product
    names. Multiple suites may register the same display, in which case every
    declared obligation is retained.
    """

    canonical = str(product or "").strip().lower()
    obligations = {
        obligation
        for suite in METHOD_SUITE_REGISTRY
        for method in suite.methods
        for figure_product, source_obligations in method.figure_source_contracts
        if figure_product == canonical
        for obligation in source_obligations
    }
    return tuple(sorted(obligations))


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
        "_Generated from `easyicu.research_agent.planning.analysis_method_suite`. Do not edit "
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
        "## Reviewed Planner → Coder resource bindings",
        "",
        "These exact coordinates are published to Planner through the scientific-action "
        "catalog and selected into Coder authority when that action is chosen. A reviewed "
        "kernel remains Coder-generated unless a deterministic runner is named above; "
        "the binding does not upgrade its claim ceiling.",
        "",
        "| Scientific action | Reviewed kernels | Runtime packages |",
        "| --- | --- | --- |",
    ]
    for suite in METHOD_SUITE_REGISTRY:
        for method in suite.methods:
            if not method.kernel_modules and not method.software_packages:
                continue
            kernels = ", ".join(f"`{value}`" for value in method.kernel_modules) or "—"
            packages = (
                ", ".join(f"`{value}`" for value in method.software_packages) or "—"
            )
            lines.append(
                f"| `{suite.family}.{method.key}` | {kernels} | {packages} |"
            )
    lines.append("")
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
