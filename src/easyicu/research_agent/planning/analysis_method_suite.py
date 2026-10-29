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
``tests/research_agent/execution/test_analysis_method_suite.py``):

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
from typing import Literal, Optional, Tuple

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

    family: StudyDesignFamily | Literal["data_preparation"]
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
            key="benchmark_comparison",
            name="Comparison with an existing score on the same stays",
            purpose=(
                "Compare the model with an existing score the question names on "
                "the same validation stays: discrimination always, calibration "
                "only when the score is a probability of the same outcome."
            ),
            tier="standard_supporting",
            implementation="deterministic",
            produces=(
                "benchmark_comparison.csv (per comparator: both AUROCs, their "
                "paired difference and interval; calibration side by side when "
                "compared; the comparator's information window)"
            ),
            runner="static_prediction_model",
            reporting_items=("TRIPOD+AI 15",),
            notes=(
                "Host-owned by the static prediction owner: it reads the "
                "primary's sealed scores, never refits or recalibrates the "
                "comparator, and reads what a comparator is from its one "
                "owner (planning.benchmark_comparator)."
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
            name="Subgroup prediction performance",
            purpose="Descriptive point estimates for frozen binary predictions in predeclared validation subgroups; retain sparse or undefined cells with reasons.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="per-subgroup performance table",
            runner="prediction_subgroup",
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
            tier="standard_supporting",
            implementation="deterministic",
            produces="per-database external performance, spline calibration figure, decision curve, case mix and TRIPOD map; separate recalibration",
            runner="prediction_external_validation",
            reporting_items=("TRIPOD 5–8", "TRIPOD 10", "TRIPOD 12–13", "TRIPOD 16–20"),
            kernel_modules=("prediction_validation",),
            required_inputs=("ExternalValidationSpec reviewed with development and validation databases",
                             "digest-bound fitted numeric logistic model and train-fitted preprocessing",
                             "independent-subject typed validation cohorts with verified concept metadata"),
            notes="Analysis only: transport to named cohorts; exact meaning changes require plan confirmation. No predictor deletion or model refit. Recalibration is apparent on its fitting cohort.",
        ),
        AnalysisMethod(
            key="reclassification",
            name="Net reclassification / IDI vs a baseline model",
            purpose="Incremental value of new predictors over an established score.",
            tier="exploratory",
            implementation="llm_coded",
            produces="bootstrap categorical/continuous NRI + IDI table",
            runner=None,
            notes=(
                "The reviewed kernel compares paired old/new probabilities on "
                "the same binary-outcome subjects. Results remain analysis-only: "
                "reclassification does not replace calibration, decision-curve "
                "or external-validation evidence."
            ),
            kernel_modules=("reclassification",),
            required_inputs=(
                "binary outcome with both event classes represented",
                "paired baseline-model and updated-model probabilities for identical subjects",
                "prespecified clinical cutoffs when categorical NRI is requested",
                "fixed bootstrap count, confidence level and random seed",
            ),
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
            notes="The survival primary owner emits log-rank for declared discrete groups on the Cox complete-case cohort and horizon; continuous exposure is not grouped implicitly.",
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
            implementation="deterministic",
            produces="rmst_summary.csv (group rmst/ci + difference) + receipt",
            runner="signed_rmst_contrast",
            notes=(
                "Host-owned executor `execution.runners.rmst_executor` claims "
                "only a step bound to one reviewed two-group RMST specification "
                "and the typed analysis cohort. `methods.rmst` "
                "(rmst / rmst_difference) owns the Kaplan-Meier estimate and its "
                "integral-form sampling SE. No Coder fallback; results remain "
                "analysis-only, and an external survRM2 oracle is still "
                "outstanding before any reportable promotion."
            ),
            kernel_modules=("rmst",),
        ),
        AnalysisMethod(
            key="competing_risks_cif",
            name="Competing risks: Aalen-Johansen CIF, Gray, cause-specific Cox and Fine-Gray",
            purpose="Separate event-free cause-specific hazards from subdistribution hazards and cumulative incidence.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="group CIF + Gray test + cause-specific Cox HR + Fine-Gray subdistribution HR + digest receipt",
            runner="survival_primary_cox",
            notes=(
                "Declare competing_risk_strategy=cause_specific_and_subdistribution, "
                "competing_event_value and competing_event_definition in the survival primary contract. "
                "The same complete-case cohort/horizon feeds CIF, Gray and local R survival::finegray+coxph. "
                "Gray retains its kernel's analysis-only ceiling and pooled censoring limitation. "
                "A cause-naive Cox HR must never substitute for cumulative incidence."
            ),
            kernel_modules=("competing_risks", "gray_test"),
            software_packages=("lifelines",),
            required_inputs=(
                "non-negative follow-up time from a declared time origin",
                "closed event coding with 0=censoring and positive competing-event codes",
                "distinct event of interest and competing event with an explicit definition",
                "declared discrete exposure levels and complete-case horizon cohort",
                "local R survival with finegray and coxph",
            ),
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
            name="Covariate balance before and after weighting",
            purpose="Describe measured imbalance with a fixed unweighted pooled sample-variance SMD denominator and effective sample sizes; balance does not establish absence of confounding.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="covariate_balance.csv and source-bound covariate_balance_receipt.json",
            runner="covariate_balance",
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
            key="propensity_adjustment",
            name="Propensity-score matching / stabilised weighting diagnostics",
            purpose="Construct prespecified matching or IPTW weights and verify overlap and covariate balance before an effect model.",
            tier="standard_supporting",
            implementation="llm_coded",
            produces="propensity scores + matched pairs or stabilised weights + SMD balance table",
            runner=None,
            reporting_items=("STROBE 9", "STROBE 12a"),
            notes=(
                "The reviewed kernel creates propensity scores, matches/weights "
                "and balance diagnostics only. It does not estimate an effect; "
                "the plan must bind its output to a separate outcome-model owner."
            ),
            kernel_modules=("propensity_weighting",),
            required_inputs=(
                "binary treatment and exact pretreatment covariate roster",
                "declared estimand and matching or weighting strategy",
                "caliper/trim/stabilisation policy fixed before outcome inspection",
                "separate downstream outcome estimator and contrast",
            ),
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
            name="Doubly-robust AIPTW sensitivity",
            purpose="Re-estimate a point-treatment binary-outcome ATE under a doubly robust identification strategy.",
            tier="exploratory",
            implementation="llm_coded",
            produces="AIPTW ATE + influence-function SE/CI + positivity diagnostics",
            runner=None,
            notes=(
                "The exact action reserves survival_inputs beside the AIPTW kernel "
                "so event-time reconciliation cannot be evicted by resource "
                "ranking. This is point-treatment analysis only, not TMLE, a "
                "longitudinal MSM or a publication-authorized causal finding."
            ),
            kernel_modules=("survival_inputs", "doubly_robust"),
            required_inputs=(
                "binary point treatment and binary outcome",
                "exact pretreatment adjustment matrix",
                "declared consistency, exchangeability and positivity assumptions",
                "fixed propensity trim bounds, outcome model and random seed",
            ),
        ),
        AnalysisMethod(
            key="mediation",
            name="Product-method mediation sensitivity",
            purpose="Decompose direct and indirect associations under explicit no-interaction and sequential-ignorability assumptions.",
            tier="exploratory",
            implementation="llm_coded",
            produces="natural direct/indirect/total effect table + bootstrap intervals",
            runner=None,
            notes=(
                "The reviewed kernel fails closed unless both identifying "
                "assumptions are explicitly declared and an empirical interaction "
                "screen passes. Binary-outcome effects are log-odds approximations."
            ),
            kernel_modules=("mediation",),
            required_inputs=(
                "temporally ordered exposure, mediator and outcome",
                "baseline covariates preceding exposure",
                "explicit no exposure-mediator interaction assumption",
                "explicit sequential-ignorability assumption and bootstrap policy",
            ),
        ),
        AnalysisMethod(
            key="g_methods",
            name="Longitudinal MSM / point-intervention g-formula",
            purpose=("Estimate a longitudinal MSM with sequential stabilized IPTW, optional IPCW and weighted pooled logistic outcomes, "
                     "or point-intervention standardized RD/RR with bootstrap CIs. Multi-interval g-formula is not implemented."),
            tier="exploratory",
            implementation="deterministic",
            produces="g_method_estimates.csv + g_method_weight_diagnostics.csv + g_method_receipt.json + g_method_checklist.md",
            runner="time_varying_g_methods",
            kernel_modules=("msm", "gformula"),
            reporting_items=("STROBE 9", "STROBE 12a", "STROBE 12e", "STROBE 16"),
            required_inputs=("one owner-verified person-period input from the selected database; no baseline-only substitution",
                             "g_methods(<GMethodSpec JSON>): treatment/confounder histories, exact lags, censoring policy, positivity bounds and truncation rule",
                             "explicit consistency, exchangeability and correct-specification assumptions"),
            notes="Only complete numeric histories and binary treatment/events. MSM reports interval overlap, weight distributions and ESS; pooled-logistic odds ratios are not risk ratios or Cox hazard ratios. Observational assumptions remain unverified; analysis only.",
        ),
    ),
)

_ASSOCIATION = MethodSuite(
    family="association",
    label="Association (general + graded ordinal / dose-response)",
    methods=(
        AnalysisMethod(
            key="mixed_effects", name="Linear mixed model for repeated outcomes",
            purpose="Estimate continuous outcome change with a random intercept and optional time slope; report CIs, variance components and ICC.",
            tier="standard_supporting", implementation="deterministic", runner="repeated_measures",
            produces="aggregate repeated_measure_effects.csv + repeated_measure_diagnostics.csv + repeated_measure_profile.csv",
            kernel_modules=("mixed_effects",), software_packages=("statsmodels",),
            required_inputs=("reviewed long-panel time axis, unit, window and aggregation", "explicit informative-death-dropout limitation and sensitivity"),
            primary_for_analysis_types=("repeated_measures",),
            notes="Linear or natural cubic spline fixed time; correlated random time slopes are optional. ICC with slopes is at time zero. Analysis only; no latent classes.",
        ),
        AnalysisMethod(
            key="gee", name="GEE for repeated binary or continuous outcomes",
            purpose="Estimate population-averaged change with robust SE and a planned exchangeable, AR1 or independent working correlation.",
            tier="standard_supporting", implementation="deterministic", runner="repeated_measures",
            produces="aggregate repeated_measure_effects.csv + repeated_measure_diagnostics.csv + repeated_measure_profile.csv",
            kernel_modules=("gee",), software_packages=("statsmodels",),
            required_inputs=("reviewed long-panel time axis, unit, window and aggregation", "working correlation", "explicit informative-death-dropout limitation and sensitivity"),
            primary_for_analysis_types=("repeated_measures",),
            notes="QIC is reported when available; compare on identical rows with a common scale. AR1 requires consecutive observed grid bins. Analysis only.",
        ),
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
            name="Adjusted association (binary / continuous / ordered / free-days / count)",
            purpose="Primary confounder-adjusted effect for a general (non-graded) exposure.",
            tier="primary",
            implementation="deterministic",
            produces="host-bound adjusted effect + typed model/coefficient contract + deterministic forest",
            runner="adjusted_association_estimates",
            reporting_items=("STROBE 16",),
            notes="Logistic remains the binary default. Typed free-days/ordinal outcomes use ordered logit with a PO score check; count outcomes declare Poisson or NB2 and exposure time/log offset with dispersion reporting. Report common OR or rate ratio with CIs, exact estimand and declared cluster covariance. Never change estimand, death score, horizon, levels or estimator without researcher confirmation.",
        ),
        AnalysisMethod(
            key="proportional_odds", name="Proportional odds for ordered outcomes",
            purpose="Estimate a common OR toward higher declared outcome levels and test proportional odds.",
            tier="standard_supporting", implementation="llm_coded",
            produces="common OR and CI + proportional-odds score test + ordered bar summary",
            reporting_items=("STROBE 12a", "STROBE 16"),
            software_packages=("statsmodels",),
            required_inputs=("ordered outcome levels", "explicit estimand", "declared model terms"),
            notes="The exact primary model is host-owned through association.adjusted_association; a violated PO assumption limits common-OR interpretation to analysis only.",
        ),
        AnalysisMethod(
            key="free_days", name="Support-free days with explicit death scoring",
            purpose="Derive complete free days from verified intervals and horizon survival; analyse as ordered outcomes.",
            tier="standard_supporting", implementation="llm_coded",
            produces="aggregate derivation audit + median/IQR + ordered distribution + common OR and CI",
            reporting_items=("STROBE 7", "STROBE 12a", "STROBE 16"),
            software_packages=("statsmodels",),
            required_inputs=("horizon", "support concept episodes", "death score -1 or 0", "survival follow-up"),
            notes="Unavailable support episodes or incomplete horizon ascertainment refuse derivation per source. No hurdle, zero-inflated or win-ratio sensitivity is implemented.",
        ),
        AnalysisMethod(
            key="count_outcome", name="Poisson or negative-binomial count association",
            purpose="Estimate a prespecified rate ratio using exposure time or log offset, with an NB2 dispersion score.",
            tier="standard_supporting", implementation="llm_coded",
            produces="rate ratio and CI + dispersion test + exposure/offset receipt",
            reporting_items=("STROBE 12a", "STROBE 16"),
            software_packages=("statsmodels",),
            required_inputs=("nonnegative integer counts", "explicit estimand", "exposure time or unit exposure", "prespecified family"),
            notes="The exact primary model is host-owned through association.adjusted_association. NB2 is never selected automatically after a Poisson diagnostic.",
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
            implementation="deterministic",
            produces="interaction test + subgroup forest",
            runner="baseline_association",
            reporting_items=("STROBE 12b",),
            figure_source_contracts=(("figure:subgroup_forest", ("effect:subgroup",)),),
            notes="The signed baseline family reports categorical stratum contrasts and an interaction test from one logistic model; unsigned or other model shapes are not claimed by this owner.",
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
            implementation="llm_coded",
            produces="spline_dose_response.csv (x, log_or, ci) + spline panel",
            runner=None,
            notes=(
                "The reviewed kernel uses declared Harrell-style knots, reports a "
                "Wald non-linearity test and emits delta-method prediction curves."
            ),
            kernel_modules=("rcs_dose_response",),
            required_inputs=(
                "continuous exposure and declared reference value",
                "binary or continuous outcome plus exact adjustment matrix",
                "prespecified knot count/locations and prediction grid",
                "declared linear/logistic model family and confidence level",
            ),
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
            purpose="Count frozen cluster assignments against the cohort roster, retain empty clusters and unassigned units, and flag near-empty clusters only using declared thresholds.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="cluster_sizes.csv and source-bound cluster_sizes_receipt.json",
            runner="cluster_sizes",
            reporting_items=("internal_phenotype P6",),
        ),
        AnalysisMethod(
            key="outcome_by_cluster",
            name="Outcome-by-cluster descriptive comparison",
            purpose="Compare outcomes across clusters — DESCRIPTIVE, explicitly not causal.",
            tier="standard_supporting",
            implementation="deterministic",
            produces="outcome_by_cluster.csv with clinical/outcome rows, observed denominators and missing counts",
            runner="phenotype_comparison",
            reporting_items=("internal_phenotype P8",),
            notes=("Source-bound descriptions on frozen cross-sectional assignments, or on the signed fixed-window suite's "
                   "frozen trajectory labels when its stability owner froze a class; no refit, inference or causal-entity claim. "
                   "Figures remain separately planned result consumers."),
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
            key="early_subtype_assignment",
            name="Early-feature assignment to frozen trajectory subtypes",
            purpose=(
                "Test whether clinical features observed before the phenotype-defining "
                "trajectory window can identify an already frozen subtype label."
            ),
            tier="exploratory",
            implementation="llm_coded",
            produces=(
                "held-out subtype probabilities + balanced accuracy/macro-F1/log-loss/"
                "multiclass-Brier + calibration/confusion/coefficient tables"
            ),
            runner=None,
            reporting_items=(
                "internal_phenotype P5",
                "TRIPOD+AI 10c",
                "TRIPOD+AI 15",
            ),
            notes=(
                "This is a supervised assignment model after phenotype discovery, not "
                "a second clustering fit and not outcome prediction from trajectory "
                "shape. Subtype labels must be frozen before modelling; early-feature "
                "construction must precede the trajectory window; development and "
                "validation patients must be disjoint. Internal performance remains "
                "analysis-only and does not establish a clinical diagnostic category."
            ),
            kernel_modules=("subtype_assignment",),
            software_packages=("sklearn",),
            required_inputs=(
                "frozen subtype labels from a completed trajectory solution",
                "early clinical feature roster excluding subtype labels and outcomes",
                "feature-window end strictly before the phenotype-defining trajectory window",
                "patient-disjoint development and validation cohorts",
                "prespecified missing-data, calibration and performance reporting rules",
            ),
        ),
        AnalysisMethod(
            key="lcga_gbtm",
            name="LCGA / group-based trajectory modelling (GBTM)",
            purpose="Latent-class growth / group-based trajectory model — a model-based longitudinal method.",
            tier="planned",
            implementation="planned",
            produces="latent-class trajectory model",
            runner=None,
            notes="PLANNED: the runner lock/image declares no validated lcmm, flexmix or Python LCGA package. Enabling needs BIC selection, minimum class size, APPA and entropy validation. Never relabel feature clustering or mixed growth curves as LCGA/GBTM.",
        ),
        AnalysisMethod(
            key="mixed_effects_growth",
            name="Mixed-effects growth curves",
            purpose="Describe continuous longitudinal growth with correlated random intercepts and time slopes.",
            tier="exploratory", implementation="deterministic", runner="repeated_measures",
            produces="aggregate growth fixed effects, random-effect covariance, ICC at time zero and observed time profiles",
            kernel_modules=("mixed_effects",), software_packages=("statsmodels",),
            required_inputs=("reviewed random-slope long-panel design", "explicit informative-death-dropout limitation and sensitivity"),
            notes="Analysis-only growth curves; no latent classes or growth mixture. LCGA/GBTM remains planned.",
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


_DATA_PREPARATION = MethodSuite(
    family="data_preparation", label="Data preparation", methods=(
        AnalysisMethod(
            key="outlier_rules", name="Plausibility ranges and winsorization",
            purpose="Apply reviewed per-concept range filtering or fixed-bound winsorization to verified raw values; report removed/capped counts separately per database.",
            tier="standard_supporting", implementation="deterministic",
            produces="data_preparation_provenance.json: outlier rule and removed/capped counts",
            runner="methods.missing", kernel_modules=("missing",),
            reporting_items=("STROBE 8", "STROBE 12c"),
            required_inputs=("verified raw numeric values in the selected database",),
            notes="Refuse without raw-value authority; no inferred or cross-database conversion. Claim ceiling: analysis_only.",
        ),
        AnalysisMethod(
            key="window_aggregation", name="Time-window aggregation",
            purpose="Summarize each variable in a reviewed [start,end) window using first/min/max/mean/last and chronological measurement order.",
            tier="standard_supporting", implementation="deterministic",
            produces="data_preparation_provenance.json: window, function and measurement counts",
            runner="methods.missing", kernel_modules=("missing",),
            reporting_items=("STROBE 8",), required_inputs=("raw measurement times and analysis-unit identity",),
            notes="Already aggregated exports cannot supply a different raw-data window. Claim ceiling: analysis_only.",
        ),
        AnalysisMethod(
            key="unit_harmonisation", name="Unit harmonisation status",
            purpose="Inspect the frozen concept-dictionary conversion rules applied by extraction, independently for all six databases.",
            tier="standard_supporting", implementation="deterministic",
            produces="data_preparation_provenance.json: captured per-database unit conversion status",
            runner="methods.missing", kernel_modules=("missing",),
            reporting_items=("STROBE 8",), required_inputs=("captured concept-dictionary source provenance",),
            notes="Read-only status; conversions are not reapplied to materialized values. Absent database snapshots remain not_recorded. Claim ceiling: analysis_only.",
        ),
        AnalysisMethod(
            key="missing_data", name="Complete case and multiple imputation",
            purpose="Choose the primary missing-data policy and retain its complementary sensitivity: stochastic FCS with PMM/logistic/polytomous models, outcome and auxiliaries, fixed seed and Rubin pooling.",
            tier="standard_supporting", implementation="deterministic",
            produces="pooled coefficients, within/between variance, df, FMI and aggregate diagnostics in data_preparation_provenance.json",
            runner="methods.missing_data", kernel_modules=("missing_data",),
            reporting_items=("STROBE 12c", "TRIPOD+AI 7"),
            required_inputs=("typed missing-data plan, observed outcome and supported conditional models",),
            notes="Association, ordinary Cox and static prediction; prediction fits imputation within each training split. Unsupported inputs refuse; no complete-case fallback. Claim ceiling: analysis_only.",
        ),
        AnalysisMethod(
            key="scaling_coding", name="Standardization, reference coding and spline knots",
            purpose="Execute reviewed standardization, model reference levels and explicit covariate spline knots; fit prediction scaling only on training partitions.",
            tier="standard_supporting", implementation="deterministic",
            produces="model design columns and data_preparation_provenance.json: reviewed scaling/coding choices",
            runner="methods.missing", kernel_modules=("missing", "rcs_dose_response"),
            reporting_items=("STROBE 11", "TRIPOD+AI 10"),
            required_inputs=("typed model terms and estimable training variance",),
            notes="Exposure splines require the existing curve contract. Prediction reference/spline overrides refuse until a matching owner exists. Claim ceiling: analysis_only.",
        ),
    ),
)


METHOD_SUITE_REGISTRY: Tuple[MethodSuite, ...] = (
    _DATA_PREPARATION,
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
            lines.append(f"| `{suite.family}.{method.key}` | {kernels} | {packages} |")
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
