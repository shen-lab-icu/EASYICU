"""Analysis Bench v1 — richer end-to-end ICU research tasks.

These items are intentionally harder than the rule-focused smoke bench.
Each task is framed as a mini-analysis package rather than a single
predictor/outcome check:

* the cohort contains multiple candidate predictors;
* missingness and ICU-specific artefacts are part of the task, not noise;
* the agent is expected to produce a workflow with descriptive summaries,
  missingness audit, outcome incidence, primary modelling and at least
  one robustness/sensitivity component;
* scoring can inspect whether the run surfaced the expected workflow
  steps and registered the expected artefact families.

The goal is not to perfectly simulate real ICU databases. The goal is
to create a paper-grade benchmark that is still deterministic and
small enough to run repeatedly across model variants.

Interpretation rule: benchmark hits should be read as alignment to
predefined synthetic-task rules unless separately supported by
literature, guidelines, or external validation. Do not rewrite
benchmark warnings as stand-alone clinical facts.
"""

from __future__ import annotations

from typing import Dict, List

from .items import BenchItem


def _demo(age, low, high):
    import numpy as np

    return np.clip(age, low, high)


def _rich_multisystem_cohort(seed: int, *, offset: int = 0, n: int = 1400, zero_artifact: bool = False,
                             vaso_missing: float = 0.72, liver_missing: float = 0.28) -> "object":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed + offset)
    age = _demo(rng.normal(66, 14, n), 18, 95)
    sex = rng.choice(["M", "F"], size=n, p=[0.56, 0.44])
    los_icu = rng.gamma(2.5, 2.2, size=n).clip(0.3, 60)

    # Physiologic axes
    resp = rng.normal(20, 5, size=n).clip(8, 45)
    spo2 = rng.normal(95, 4.5, size=n).clip(65, 100)
    map_v = rng.normal(76, 13, size=n).clip(40, 125)
    hr = rng.normal(92, 18, size=n).clip(40, 180)
    sbp = (map_v + rng.normal(22, 8, size=n)).clip(60, 220)
    dbp = (map_v - rng.normal(10, 5, size=n)).clip(25, 130)
    temp = rng.normal(37.0, 0.9, size=n).clip(33, 41.5)
    gcs = rng.choice(range(3, 16), size=n,
                     p=np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06,
                                 0.07, 0.08, 0.09, 0.12, 0.18, 0.21]) /
                       np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06,
                                 0.07, 0.08, 0.09, 0.12, 0.18, 0.21]).sum())

    # SOFA components
    sofa2_resp = np.clip(np.floor((97 - spo2) / 5).astype(int), 0, 4)
    sofa2_cardio = np.clip(np.floor((75 - map_v) / 8).astype(int), 0, 4)
    sofa2_cns = np.clip(np.floor((15 - gcs) / 3).astype(int), 0, 4)
    latent_renal = rng.normal(0, 1, n) + 0.35 * sofa2_cardio + 0.25 * (age - 65) / 10
    creat = np.exp(0.1 + 0.42 * latent_renal + rng.normal(0, 0.25, n)).clip(0.2, 12)
    kdigo_stage = np.clip(np.floor((creat - 0.8) / 0.9).astype(int), 0, 3)
    sofa2_renal = np.clip(np.floor((creat - 0.7) / 1.0).astype(int), 0, 4)

    latent_liver = rng.normal(0, 1, n) + 0.18 * sofa2_resp + 0.22 * (age - 65) / 10
    bili = np.exp(-0.25 + 0.5 * latent_liver + rng.normal(0, 0.45, n)).clip(0.2, 18)
    sofa2_liver = np.clip(np.floor((bili - 0.8) / 1.2).astype(int), 0, 4)

    plt_count = rng.normal(220, 70, size=n).clip(10, 500)
    sofa2_coag = np.clip(np.floor((180 - plt_count) / 45).astype(int), 0, 4)

    lact = np.exp(0.4 + 0.18 * sofa2_cardio + 0.18 * sofa2_resp + rng.normal(0, 0.45, n)).clip(0.4, 22)
    vaso_prob = 1.0 / (1.0 + np.exp(-(-1.8 + 0.65 * sofa2_cardio + 0.35 * np.log(lact + 0.2))))
    vaso = (rng.random(n) < vaso_prob).astype(int)

    # Missingness mechanisms
    liver_is_missing = rng.random(n) < liver_missing
    vaso_is_missing = rng.random(n) < vaso_missing
    bili_obs = bili.copy()
    sofa2_liver_obs = sofa2_liver.astype(float).copy()
    vaso_obs = vaso.astype(float).copy()
    bili_obs[liver_is_missing] = np.nan
    sofa2_liver_obs[liver_is_missing] = np.nan
    vaso_obs[vaso_is_missing] = np.nan

    sofa2 = sofa2_resp + sofa2_cardio + sofa2_cns + sofa2_renal + sofa2_coag + np.nan_to_num(sofa2_liver_obs, nan=0.0)

    if zero_artifact:
        artifact = rng.random(n) < 0.08
        sofa2[artifact] = 0
        sofa2_liver_obs[artifact] = np.nan
        bili_obs[artifact] = np.nan
        # Patients with artefactual zero are not truly low risk.
        sofa2_cardio[artifact] = np.maximum(sofa2_cardio[artifact], 1)
        sofa2_resp[artifact] = np.maximum(sofa2_resp[artifact], 1)

    logit = (
        -4.2
        + 0.11 * sofa2
        + 0.06 * (age - 65) / 5
        + 0.14 * np.log(lact + 0.2)
        + 0.18 * kdigo_stage
        - 0.035 * (map_v - 75)
        - 0.10 * (gcs - 11)
        + 0.22 * np.nan_to_num(vaso_obs, nan=0.0)
        + np.where(liver_is_missing, 0.35, 0.0)
        + np.where(vaso_is_missing, 0.25, 0.0)
    )
    death = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age,
        "sex": sex,
        "los_icu": los_icu,
        "death": death,
        "sofa2": sofa2.astype(int),
        "sofa2_resp": sofa2_resp.astype(int),
        "sofa2_coag": sofa2_coag.astype(int),
        "sofa2_liver": sofa2_liver_obs,
        "sofa2_cardio": sofa2_cardio.astype(int),
        "sofa2_cns": sofa2_cns.astype(int),
        "sofa2_renal": sofa2_renal.astype(int),
        "kdigo_stage": kdigo_stage.astype(int),
        "creat": creat,
        "lact": lact,
        "bili": bili_obs,
        "map": map_v,
        "hr": hr,
        "sbp": sbp,
        "dbp": dbp,
        "temp": temp,
        "spo2": spo2,
        "resp": resp,
        "vaso": vaso_obs,
        "gcs": gcs.astype(int),
    })


def _shock_discordance_cohort(seed: int) -> "object":
    df = _rich_multisystem_cohort(seed, offset=101, n=1500, zero_artifact=False, vaso_missing=0.64, liver_missing=0.24)
    import numpy as np

    rng = np.random.default_rng(seed + 101)
    discordant = rng.random(len(df)) < 0.14
    df.loc[discordant, "lact"] = np.clip(df.loc[discordant, "lact"] * 2.0, 0.5, 22)
    df.loc[discordant, "map"] = np.clip(df.loc[discordant, "map"] - 10, 40, 125)
    df.loc[discordant, "vaso"] = np.where(rng.random(discordant.sum()) < 0.7, 1.0, df.loc[discordant, "vaso"])
    return df


def _zero_artifact_cohort(seed: int) -> "object":
    return _rich_multisystem_cohort(seed, offset=211, n=1500, zero_artifact=True, vaso_missing=0.70, liver_missing=0.30)


def _hepatorenal_missingness_cohort(seed: int) -> "object":
    return _rich_multisystem_cohort(seed, offset=307, n=1300, zero_artifact=False, vaso_missing=0.78, liver_missing=0.38)


def _neuro_hemodynamic_cohort(seed: int) -> "object":
    df = _rich_multisystem_cohort(seed, offset=409, n=1400, zero_artifact=False, vaso_missing=0.55, liver_missing=0.22)
    return df


COMMON_ANALYSIS_STEPS: List[str] = [
    "table", "outcome", "missingness", "association",
]


ANALYSIS_BENCH_ITEMS: List[BenchItem] = [
    BenchItem(
        key="analysis_sofa_multisignal_mortality",
        name="SOFA-2 multisignal mortality analysis",
        research_question="In a first-ICU-stay adult cohort, quantify whether early SOFA-2 severity and selected first-24h physiology are associated with ICU mortality, explicitly auditing score==0 artefacts and missingness.",
        target_outcome="death",
        primary_predictor="sofa2",
        expected_or_direction=+1,
        cohort_factory=_zero_artifact_cohort,
        expected_finding_substrings=["non-monotonic", "missingness", "sofa2"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years", "ICU LoS ≥ 24 hours"],
        benchmark_family="analysis",
        difficulty="advanced",
        evidence_basis="consensus_inspired_synthetic",
        claim_scope="internal_benchmark_only",
        candidate_variables=["sofa2", "age", "sex", "map", "hr", "spo2", "resp", "vaso", "bili", "sofa2_liver"],
        expected_step_substrings=["table", "outcome", "missingness", "stratum", "association", "sensitivity"],
        expected_artifact_substrings=["table_one", "outcome_incidence", "missingness", "sofa2_stratum", "primary_association"],
        notes="Core EasyICU-style end-to-end analysis task; should behave like a mini paper pipeline rather than a one-regression toy item.",
        interpretation_note="The score==0 anomaly is benchmark-constructed; do not cite it as an external ICU epidemiology or pathophysiology claim.",
    ),
    BenchItem(
        key="analysis_shock_discordance",
        name="Shock physiology discordance audit",
        research_question="Characterise whether high lactate, low mean arterial pressure, and vasopressor exposure form discordant shock phenotypes with differential ICU mortality, and distinguish association from treatment-selection bias.",
        target_outcome="death",
        primary_predictor="lact",
        expected_or_direction=+1,
        cohort_factory=_shock_discordance_cohort,
        expected_finding_substrings=["vaso", "missingness", "lact"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years", "ICU LoS ≥ 12 hours"],
        benchmark_family="analysis",
        difficulty="advanced",
        evidence_basis="literature_inspired_synthetic",
        claim_scope="internal_benchmark_only",
        candidate_variables=["lact", "map", "vaso", "age", "hr", "sofa2_cardio"],
        expected_step_substrings=["table", "missingness", "association", "sensitivity"],
        expected_artifact_substrings=["table_one", "missingness", "primary_association"],
        notes="Requires the agent to avoid causal language around vasopressor use while still analysing a clinically coherent physiology pattern.",
        interpretation_note="Discordant shock patterns and vasopressor warnings here are synthetic benchmark conditions, not validated external phenotypes.",
    ),
    BenchItem(
        key="analysis_renal_stage_sensitivity",
        name="Renal injury staging and sensitivity analysis",
        research_question="Evaluate whether first-24h KDIGO stage is associated with ICU mortality, while reporting creatinine skewness and checking whether the conclusion is stable to complete-case versus reduced-variable models.",
        target_outcome="death",
        primary_predictor="kdigo_stage",
        expected_or_direction=+1,
        cohort_factory=_hepatorenal_missingness_cohort,
        expected_finding_substrings=["creat", "missingness"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years"],
        benchmark_family="analysis",
        difficulty="intermediate",
        evidence_basis="consensus_inspired_synthetic",
        claim_scope="internal_benchmark_only",
        candidate_variables=["kdigo_stage", "creat", "age", "sex", "sofa2_renal", "vaso"],
        expected_step_substrings=["table", "missingness", "association", "sensitivity"],
        expected_artifact_substrings=["table_one", "missingness", "primary_association"],
        notes="Tests whether the agent can pair an ordinal renal stage with a right-skewed lab and an explicit sensitivity statement.",
        interpretation_note="KDIGO and creatinine are clinically grounded, but the stability target in this task is a synthetic benchmark expectation.",
    ),
    BenchItem(
        key="analysis_neuro_hemodynamic",
        name="Neurologic-hemodynamic mortality model",
        research_question="Assess whether low Glasgow Coma Scale and low mean arterial pressure are jointly associated with ICU mortality, respecting the ordinal nature of GCS and the negative direction of hemodynamic stability.",
        target_outcome="death",
        primary_predictor="gcs",
        expected_or_direction=-1,
        cohort_factory=_neuro_hemodynamic_cohort,
        expected_finding_substrings=["gcs", "ordinal"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years", "ICU LoS ≥ 6 hours"],
        benchmark_family="analysis",
        difficulty="intermediate",
        evidence_basis="literature_inspired_synthetic",
        claim_scope="internal_benchmark_only",
        candidate_variables=["gcs", "map", "age", "sex", "hr", "sofa2_cns"],
        expected_step_substrings=["table", "missingness", "association"],
        expected_artifact_substrings=["table_one", "primary_association"],
        notes="Harder than the rule bench GCS item because it asks for a multivariable interpretation rather than a single-sign check.",
        interpretation_note="The target direction is benchmark-internal and must not be reported as an externally validated joint GCS-MAP effect.",
    ),
    BenchItem(
        key="analysis_respiratory_failure",
        name="Respiratory component severity analysis",
        research_question="Determine whether hypoxemia-related markers and the SOFA-2 respiratory component are associated with ICU mortality, and report whether the component behaves consistently with the total score severity gradient.",
        target_outcome="death",
        primary_predictor="sofa2_resp",
        expected_or_direction=+1,
        cohort_factory=_rich_multisystem_cohort,
        expected_finding_substrings=["sofa2", "resp"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years", "ICU LoS ≥ 24 hours"],
        benchmark_family="analysis",
        difficulty="advanced",
        evidence_basis="consensus_inspired_synthetic",
        claim_scope="internal_benchmark_only",
        candidate_variables=["sofa2_resp", "spo2", "resp", "age", "sofa2"],
        expected_step_substrings=["table", "outcome", "association", "stratum"],
        expected_artifact_substrings=["table_one", "outcome_incidence", "primary_association"],
        notes="Forces the agent to reason about a component score and surrounding continuous physiology together.",
        interpretation_note="Respiratory-severity expectations are literature-shaped, but the cohort and effect sizes remain synthetic benchmark constructs.",
    ),
    BenchItem(
        key="analysis_hepatobiliary_missingness",
        name="Hepatobiliary missingness and interpretability",
        research_question="Quantify whether bilirubin and the SOFA-2 liver component are associated with ICU mortality, while explicitly showing how missingness constrains interpretation.",
        target_outcome="death",
        primary_predictor="bili",
        expected_or_direction=+1,
        cohort_factory=_hepatorenal_missingness_cohort,
        expected_finding_substrings=["bili", "missingness"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years", "ICU LoS ≥ 24 hours"],
        benchmark_family="analysis",
        difficulty="advanced",
        evidence_basis="internal_stress_test_synthetic",
        claim_scope="internal_benchmark_only",
        candidate_variables=["bili", "sofa2_liver", "age", "sex", "sofa2"],
        expected_step_substrings=["table", "missingness", "association", "sensitivity"],
        expected_artifact_substrings=["table_one", "missingness", "primary_association"],
        notes="A good agent should avoid overclaiming from a partially observed laboratory component.",
        interpretation_note="The missingness structure is benchmark-designed and should not be cited as a real-world bilirubin availability estimate.",
    ),
    BenchItem(
        key="analysis_vasopressor_selection_bias",
        name="Vasopressor association without causal overreach",
        research_question="Estimate whether first-24h vasopressor exposure is associated with ICU mortality, but make treatment-selection bias and missingness explicit and avoid causal effect language.",
        target_outcome="death",
        primary_predictor="vaso",
        expected_or_direction=+1,
        cohort_factory=_shock_discordance_cohort,
        expected_finding_substrings=["vaso", "selection", "missingness"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years", "ICU LoS ≥ 12 hours"],
        benchmark_family="analysis",
        difficulty="advanced",
        evidence_basis="methods_consensus_synthetic",
        claim_scope="internal_benchmark_only",
        candidate_variables=["vaso", "lact", "map", "age", "sofa2_cardio"],
        expected_step_substrings=["table", "missingness", "association", "sensitivity"],
        expected_artifact_substrings=["table_one", "missingness", "primary_association"],
        notes="This is a language-discipline task as much as a modelling task.",
        interpretation_note="Selection-bias and causal-language checks reflect benchmark methodology rules, not external adjudication of vasopressor effects.",
    ),
    BenchItem(
        key="analysis_complete_case_stability",
        name="Complete-case stability under structured missingness",
        research_question="Build a mortality association model using early severity and physiology variables, then test whether the main conclusion survives reduced-variable and complete-case sensitivity analyses.",
        target_outcome="death",
        primary_predictor="sofa2",
        expected_or_direction=+1,
        cohort_factory=_hepatorenal_missingness_cohort,
        expected_finding_substrings=["missingness", "selection bias"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years", "ICU LoS ≥ 24 hours"],
        benchmark_family="analysis",
        difficulty="advanced",
        evidence_basis="methods_consensus_synthetic",
        claim_scope="internal_benchmark_only",
        candidate_variables=["sofa2", "age", "sex", "map", "hr", "bili", "vaso", "creat"],
        expected_step_substrings=["table", "missingness", "association", "sensitivity"],
        expected_artifact_substrings=["table_one", "missingness", "primary_association", "sensitivity"],
        notes="Directly targets the kind of robustness section a methods paper should benchmark.",
        interpretation_note="Complete-case and reduced-model expectations are internal methods-benchmark targets, not external clinical truth claims.",
    ),
    BenchItem(
        key="analysis_component_hierarchy",
        name="Organ-component hierarchy and risk ranking",
        research_question="Compare the relative mortality associations of SOFA-2 organ components and determine whether component-level risk ranking is coherent with the total severity score.",
        target_outcome="death",
        primary_predictor="sofa2_renal",
        expected_or_direction=+1,
        cohort_factory=_rich_multisystem_cohort,
        expected_finding_substrings=["ordinal", "sofa2"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years"],
        benchmark_family="analysis",
        difficulty="advanced",
        evidence_basis="literature_inspired_synthetic",
        claim_scope="internal_benchmark_only",
        candidate_variables=["sofa2_resp", "sofa2_cardio", "sofa2_cns", "sofa2_renal", "sofa2_liver", "sofa2_coag", "sofa2"],
        expected_step_substrings=["table", "association", "sensitivity"],
        expected_artifact_substrings=["primary_association", "component"],
        notes="This is close to the sort of figure-supporting task we actually care about for publication.",
        interpretation_note="Component-ranking coherence is evaluated only within this synthetic benchmark and should not be written up as an external clinical hierarchy.",
    ),
    BenchItem(
        key="analysis_data_quality_first",
        name="Data-quality-first mortality analysis",
        research_question="Before claiming any mortality association, determine whether data quality issues in liver- and vasopressor-related variables materially limit interpretation, then report the most defensible adjusted association model.",
        target_outcome="death",
        primary_predictor="sofa2",
        expected_or_direction=+1,
        cohort_factory=_zero_artifact_cohort,
        expected_finding_substrings=["missingness", "sofa2", "vaso", "bili"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years", "ICU LoS ≥ 24 hours"],
        benchmark_family="analysis",
        difficulty="advanced",
        evidence_basis="internal_stress_test_synthetic",
        claim_scope="internal_benchmark_only",
        candidate_variables=["sofa2", "bili", "sofa2_liver", "vaso", "age", "sex", "map"],
        expected_step_substrings=["missingness", "table", "association", "sensitivity", "stratum"],
        expected_artifact_substrings=["missingness", "table_one", "primary_association"],
        notes="This one asks the agent to prioritise auditability before elegance, which is exactly the paper-standard behavior we want.",
        interpretation_note="Warning hits in this task show alignment to benchmark audit rules, not independent validation of external ICU data-quality claims.",
    ),
]


__all__ = ["ANALYSIS_BENCH_ITEMS"]
