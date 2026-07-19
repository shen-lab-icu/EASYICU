"""Exact paper-facing Canonical9 task suite.

This repository-local module contains manuscript evaluation material.  It is
excluded from the installed EasyICU wheel and must never be imported by the
Agent control plane or included in Planner/Coder prompts.
"""

from __future__ import annotations

from easyicu.research_agent.icu_agent_bench import (
    ICUAgentBenchSuite,
    ICUAgentBenchTask,
)


def easyicu_evaluation_protocol_suite() -> ICUAgentBenchSuite:
    """The v2 nine-question evaluation protocol (E1-H3).

    These nine clinically distinct research questions are the manuscript's
    cost-efficient-model reliability baseline (writing framework v2 §K3 / §M).
    Each carries a method dimension, an embedded *audit hazard* answer key in
    ``semantic_guardrails``, and the required display items in
    ``expected_outputs`` (Table 1 + >=1 result figure + applicable audit
    panel; hard tasks expect >=2 result figures).

    Gold answers are ``planned`` here on purpose: the result-validity
    dimension is scored against a **locked reference analysis** computed by a
    non-agent path (EasyICU API + statsmodels, ricu cross-check for the
    ordinal/survival/clustering tasks) that must be frozen before the
    cost-efficient-model batch is run (framework §M2 / A5). Until then the
    deterministic *audit-conclusion-safety* checks (hazard handling, forbidden
    conclusions) carried by the guardrails are the live Tier-1 signal.

    This is the EasyICU evaluation protocol, NOT a published benchmark (see
    module banner). Case-specific requirements live here, never in shared
    prompts (prompt hygiene).
    """
    return ICUAgentBenchSuite(
        name="EasyICU evaluation protocol (nine questions)",
        tasks=[
            ICUAgentBenchTask(
                task_id="e1_sepsis3_prevalence_mortality",
                kind="sepsis_onset",
                title="Sepsis-3 prevalence and in-hospital mortality",
                objective=(
                    "Estimate Sepsis-3 prevalence and its association with "
                    "in-hospital mortality, with a transparent, reproducible "
                    "cohort definition and visible denominator."
                ),
                expected_outputs=[
                    "cohort definition summary",
                    "table one",
                    "prevalence and mortality figure",
                ],
                semantic_guardrails=[
                    "Define Sepsis-3 with EasyICU derived concepts (suspected "
                    "infection timing + SOFA), never an ICD-code proxy.",
                    "State the cohort denominator and inclusion/exclusion "
                    "explicitly; do not silently pick one operationalisation.",
                    "Use diagnosis codes only for cohort membership, not as event "
                    "timing — they generally carry no reliable onset time "
                    "(database-dependent: e.g. eICU exposes a diagnosis offset, "
                    "MIMIC diagnoses_icd does not).",
                ],
                evaluation_notes=[
                    "Audit hazard: cohort-definition transparency; "
                    "ICD-as-membership not timing.",
                    "result-validity scored vs locked reference (planned, §M2).",
                ],
                target_databases=["miiv"],
                difficulty="basic",
                gold_answer_status="planned",
            ),
            ICUAgentBenchTask(
                task_id="e2_lactate_mortality",
                kind="descriptive_association",
                title="24h peak lactate vs in-hospital mortality",
                objective=(
                    "Quantify the descriptive association between first-24h "
                    "peak lactate and in-hospital mortality."
                ),
                expected_outputs=[
                    "table one",
                    "lactate-mortality association figure",
                    "missingness/within-window aggregation audit",
                ],
                semantic_guardrails=[
                    "Respect the within-window aggregation rule and units "
                    "(mmol/L vs mg/dL) when summarising peak lactate.",
                    "Report lactate distribution (skew) and use median where a "
                    "mean would mislead.",
                ],
                evaluation_notes=[
                    "Audit hazard: within-window aggregation and units.",
                    "result-validity scored vs locked reference (planned, §M2).",
                ],
                target_databases=["miiv", "mimic", "eicu", "aumc", "hirid", "sic"],
                difficulty="basic",
                gold_answer_status="planned",
            ),
            ICUAgentBenchTask(
                task_id="e3_kdigo_gradient",
                kind="ordinal_dose_response",
                title="KDIGO AKI stage gradient vs LOS and mortality",
                objective=(
                    "Characterise the dose-response gradient of first-24h KDIGO "
                    "AKI stage against ICU length of stay and mortality."
                ),
                expected_outputs=[
                    "table one",
                    "stage-stratified outcome figure",
                    "ordinal trend audit",
                ],
                semantic_guardrails=[
                    "Treat KDIGO stage as an ordered category; do not summarise "
                    "it with mean/SD as if it were continuous.",
                    "Report stage boundaries explicitly.",
                ],
                evaluation_notes=[
                    "Audit hazard: treating an ordinal stage as continuous.",
                    "result-validity scored vs locked reference (planned, §M2; "
                    "ricu cross-check).",
                ],
                target_databases=["miiv", "mimic", "eicu", "aumc", "hirid", "sic"],
                difficulty="basic",
                gold_answer_status="planned",
            ),
            ICUAgentBenchTask(
                task_id="m1_hepatobiliary_missingness",
                kind="missingness_robustness",
                title="Hepatobiliary missingness and the liver organ score",
                objective=(
                    "Assess the bilirubin / SOFA-2 liver component association "
                    "with mortality while making explicit how missingness "
                    "constrains interpretation."
                ),
                expected_outputs=[
                    "table one",
                    "missingness audit panel",
                    "adjusted association figure",
                ],
                semantic_guardrails=[
                    "Report the missing fraction and how complete-case analysis "
                    "shrinks the cohort before interpreting.",
                    "Do not present a complete-case estimate as if it were the "
                    "full-cohort estimate (complete-case bias).",
                ],
                evaluation_notes=[
                    "Audit hazard: complete-case bias under high missingness.",
                    "result-validity scored vs locked reference (planned, §M2).",
                ],
                target_databases=["miiv", "eicu"],
                difficulty="intermediate",
                gold_answer_status="planned",
            ),
            ICUAgentBenchTask(
                task_id="m2_mortality_prediction",
                kind="mortality_prediction",
                title="First-24h vitals+labs mortality prediction with calibration",
                objective=(
                    "Build an in-hospital mortality prediction model from "
                    "first-24h vitals and labs with an explicit train/test "
                    "split, discrimination, and calibration."
                ),
                expected_outputs=[
                    "table one",
                    "held-out discrimination figure (ROC)",
                    "calibration figure",
                    "split definition + leakage audit",
                ],
                semantic_guardrails=[
                    "Split by patient-level id (group by subject/patient), not by "
                    "row, so a patient cannot appear in both train and test; fit "
                    "any scaler/imputer on the training split only.",
                    "No post-outcome leakage features.",
                    "Report imbalance-aware metrics (recall/F1/AUROC/PR), not "
                    "accuracy alone; report calibration, not discrimination alone.",
                    "Bind every reported metric (AUROC, Brier) to a registered "
                    "value.",
                ],
                evaluation_notes=[
                    "Audit hazard: leakage, patient-level split, class imbalance, "
                    "numeric binding, calibration.",
                    "result-validity scored vs locked reference (planned, §M2).",
                ],
                target_databases=["miiv"],
                difficulty="intermediate",
                gold_answer_status="planned",
            ),
            ICUAgentBenchTask(
                task_id="m3_sepsis_subphenotype",
                kind="subphenotype_clustering",
                title="Sepsis subphenotype identification (unsupervised)",
                objective=(
                    "Identify candidate sepsis subphenotypes by unsupervised "
                    "clustering of first-24h labs and vitals (Seymour-style), "
                    "reporting cluster stability and profiles."
                ),
                expected_outputs=[
                    "cluster profile table",
                    "cluster visualisation figure",
                    "cluster-stability audit",
                ],
                semantic_guardrails=[
                    "State that clusters have no ground truth; report stability "
                    "(e.g. silhouette, bootstrap) before interpreting.",
                    "Do not over-claim clusters as established clinical " "subtypes.",
                ],
                evaluation_notes=[
                    "Audit hazard: clusters lack ground truth; over-interpretation.",
                    "result-validity scored vs locked reference (planned, §M2; "
                    "silhouette/stability band).",
                ],
                target_databases=["miiv"],
                difficulty="intermediate",
                gold_answer_status="planned",
            ),
            ICUAgentBenchTask(
                task_id="h1_ventilation_survival",
                kind="survival_analysis",
                title="Mechanical ventilation duration/status vs 28-day mortality",
                objective=(
                    "Estimate the association between mechanical ventilation "
                    "duration/status and 28-day mortality with time-to-event "
                    "methods that respect exposure timing."
                ),
                expected_outputs=[
                    "table one",
                    "Kaplan-Meier figure",
                    "Cox summary figure",
                    "immortal-time / PH diagnostics audit",
                ],
                semantic_guardrails=[
                    "Avoid immortal time bias: align exposure with follow-up "
                    "start; do not classify by a future event.",
                    "Include only incident cases: exclude patients whose "
                    "event/exposure already occurred before follow-up start "
                    "(prevalent cases) — a cohort-definition exclusion, distinct "
                    "from leakage.",
                    "Check proportional-hazards assumptions before reporting a "
                    "single hazard ratio.",
                ],
                evaluation_notes=[
                    "Audit hazard: immortal time bias; prevalent-vs-incident "
                    "exclusion.",
                    "result-validity scored vs locked reference (planned, §M2; "
                    "HR band, ricu cross-check).",
                ],
                target_databases=["miiv", "eicu", "aumc", "hirid"],
                difficulty="advanced",
                gold_answer_status="planned",
            ),
            ICUAgentBenchTask(
                task_id="h2_vasopressor_causal",
                kind="causal_inference",
                title="Early vasopressor exposure vs mortality (causal)",
                objective=(
                    "Estimate the effect of early vasopressor exposure on "
                    "mortality using PSM/IPTW, making confounding by indication "
                    "explicit."
                ),
                expected_outputs=[
                    "table one",
                    "covariate-balance figure",
                    "adjusted effect figure",
                    "confounding/positivity audit",
                ],
                semantic_guardrails=[
                    "Make confounding by indication explicit; report covariate "
                    "balance and positivity.",
                    "Do not state a causal conclusion that the design cannot "
                    "support; bound the claim to the assumptions.",
                ],
                evaluation_notes=[
                    "Audit hazard: confounding by indication; over-claimed "
                    "causality (forbidden: unqualified causal language).",
                    "result-validity scored vs locked reference (planned, §M2).",
                ],
                target_databases=["miiv", "eicu"],
                difficulty="advanced",
                gold_answer_status="planned",
            ),
            ICUAgentBenchTask(
                task_id="h3_trajectory_clustering",
                kind="longitudinal_trajectory_analysis",
                title="Organ-dysfunction trajectory clustering (longitudinal)",
                objective=(
                    "Cluster longitudinal organ-dysfunction trajectories "
                    "(SOFA components / lactate over time) into latent classes "
                    "and relate them to outcome."
                ),
                expected_outputs=[
                    "trajectory cluster profile table",
                    "trajectory figure",
                    "outcome-by-trajectory-class figure",
                    "cluster-count / alignment audit",
                ],
                semantic_guardrails=[
                    "Align trajectories on a fixed anchor; avoid immortal time "
                    "and length-biased sampling.",
                    "Justify the number of clusters; do not interpret a "
                    "trajectory class as a causal group.",
                ],
                evaluation_notes=[
                    "Audit hazard: trajectory alignment, immortal time, cluster-"
                    "count selection, trajectory-as-cause.",
                    "result-validity scored vs locked reference (planned, §M2).",
                ],
                target_databases=["miiv"],
                difficulty="advanced",
                gold_answer_status="planned",
            ),
        ],
    )
