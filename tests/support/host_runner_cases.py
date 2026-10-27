"""Small synthetic inputs for desktop execution parity and ownership tests.

Factories are independent of test modules and contain no benchmark material.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.survival import (
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
    SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
)
from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_executor import (
    PHENOTYPE_ASSIGNMENTS_PRODUCT,
    PHENOTYPE_PROFILES_PRODUCT,
)
from easyicu.research_agent.execution.runners.descriptive_result_figure_executor import (
    DESCRIPTIVE_DISTRIBUTION_COLUMNS,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    CohortDescriptor,
    ConceptDescriptor,
    FamilyPrimaryResultRequirement,
    ResearchContext,
    VariableRole,
)


def cohort_summary_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="02_cohort_summary",
        planned_analysis_role="auxiliary",
        intent="Describe the exact closed cohort columns.",
        inputs=[
            "artifact:analysis_cohort",
            "age",
            "sex",
            "exposure",
            "outcome",
        ],
        expected_outputs=["table:cohort_summary"],
        method="descriptive_cohort_summary",
    )


def cohort_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [50.0, 60.0, 70.0, 80.0],
            "sex": ["Female", "Male", "Female", "Male"],
            "exposure": [0.0, 1.0, 0.0, 1.0],
            "outcome": [0, 0, 1, 1],
        }
    )


def cohort_summary_context() -> dict:
    return {
        "variables": [
            {
                "name": "age",
                "unit": "years",
                "observed_domain": {
                    "n_unique": 4,
                    "is_binary": False,
                    "min": 50.0,
                    "max": 80.0,
                },
            },
            {
                "name": "sex",
                "unit": None,
                "observed_domain": {
                    "n_unique": 2,
                    "is_binary": False,
                    "levels": ["Female", "Male"],
                },
            },
            {
                "name": "exposure",
                "unit": None,
                "observed_domain": {
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0.0, 1.0],
                },
            },
            {
                "name": "outcome",
                "unit": None,
                "observed_domain": {
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0, 1],
                },
            },
        ]
    }


cohort_summary = SimpleNamespace(
    _step=cohort_summary_step,
    _frame=cohort_summary_frame,
    _context=cohort_summary_context,
)


def descriptive_distribution_step(**updates) -> AnalysisStep:
    payload = {
        "step_id": "04_distribution",
        "planned_analysis_role": "auxiliary",
        "intent": "Summarize a continuous variable by a closed category.",
        "inputs": ["artifact:analysis_cohort", "sex", "los_icu"],
        "expected_outputs": ["table:distribution_prevalence"],
        "method": "descriptive_distribution_summary",
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


descriptive_distribution = SimpleNamespace(_step=descriptive_distribution_step)


def descriptive_association_step(**updates) -> AnalysisStep:
    payload = {
        "step_id": "05_age_los_association",
        "planned_analysis_role": "auxiliary",
        "intent": "Describe the monotonic association without a causal claim.",
        "inputs": ["artifact:analysis_cohort", "age", "los_icu"],
        "expected_outputs": ["statistic:age_los_spearman"],
        "method": "descriptive_association",
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


descriptive_association = SimpleNamespace(_step=descriptive_association_step)


def ordered_stratified_step() -> AnalysisStep:
    return AnalysisStep.model_validate(
        {
            "step_id": "ordered_trend",
            "planned_analysis_role": "secondary",
            "intent": "Compare an ordered exposure across two declared outcomes.",
            "inputs": [
                "artifact:analysis_cohort",
                "table:adjusted_association_estimates",
                "severity",
                "death",
                "duration",
            ],
            "expected_outputs": [
                "table:ordinal_trend_dose_response",
            ],
            "method": "ordinal_stratified_descriptive_analysis",
            "scientific_action_id": "association.ordinal_trend",
        }
    )


def ordered_stratified_plan(step: AnalysisStep | None = None) -> AnalysisPlan:
    child = step or ordered_stratified_step()
    primary = AnalysisStep.model_validate(
        {
            "step_id": "primary_model",
            "planned_analysis_role": "primary",
            "intent": "Fit the declared adjusted primary model.",
            "inputs": ["artifact:analysis_cohort", "severity", "death"],
            "expected_outputs": ["table:adjusted_association_estimates"],
            "method": "adjusted_association_models",
            "model_requirements": [
                {
                    "requirement_id": "primary",
                    "outcome": "death",
                    "outcome_type": "binary",
                    "method_family": "statsmodels_logit_mle",
                    "exposure_source": "severity",
                    "analysis_role": "primary",
                    "analysis_set": "source_aware",
                    "covariates": [],
                    "model_terms": [
                        {
                            "name": "severity",
                            "role": "exposure",
                            "coding": "ordinal_linear",
                            "levels": ["0", "1", "2"],
                            "reference_level": None,
                            "transform": "declared_level_index",
                        }
                    ],
                }
            ],
        }
    )
    return AnalysisPlan(research_question="Test", steps=[primary, child])


def ordered_stratified_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "severity": [0, 0, 1, 1, 2, 2, None],
            "death": [0, 1, 0, 1, 1, 1, 0],
            "duration": [1.0, 2.0, 2.0, 4.0, 5.0, 8.0, 9.0],
        }
    )


def ordered_stratified_primary_binding(
    plan: AnalysisPlan,
    frame: pd.DataFrame,
    run_dir: Path,
    *,
    requirement_update: dict | None = None,
) -> dict:
    """Fit the plan's primary model on ``frame`` and bind its estimates table.

    The ordered-trend owner reads the primary's
    ``table:adjusted_association_estimates`` to prove its refit used the same
    rows; the host binds it beside the cohort.  ``requirement_update`` fits a
    deliberately different primary, for the owner's mismatch stop.
    """

    from easyicu.research_agent.contracts.association_execution import (
        association_estimator_kind,
    )
    from easyicu.research_agent.execution.runners.adjusted_association_executor import (
        run_adjusted_association_from_env,
    )

    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    requirement = primary.model_requirements[0]
    if requirement_update:
        requirement = requirement.model_copy(update=requirement_update)
    out_dir = Path(run_dir) / "steps" / primary.step_id / "outputs"
    run_adjusted_association_from_env(
        requirement_id=requirement.requirement_id,
        exposure=requirement.exposure_source,
        outcome=requirement.outcome,
        covariates=list(requirement.covariates or []),
        model_terms=list(requirement.model_terms or []),
        estimator_kind=association_estimator_kind(requirement),
        analysis_set=requirement.analysis_set,
        analysis_role=requirement.analysis_role,
        method_family=requirement.method_family,
        primary_contrast_level=requirement.primary_contrast_level,
        dependence=requirement.dependence,
        missing_category_covariates=list(requirement.missing_category_covariates()),
        frame=frame,
        emit_step_summary=False,
        output_dir=out_dir,
    )
    path = out_dir / "adjusted_association_estimates.csv"
    table = pd.read_csv(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "relative_path": path.relative_to(run_dir).as_posix(),
        "sha256": digest,
        "declared_kind": "table",
        "product": "adjusted_association_estimates",
        "evidence_kind": "table",
        "evidence_id": "ev-primary-estimates",
        "produced_by_step": primary.step_id,
        "product_contract": {"columns": list(table.columns), "row_count": len(table)},
    }


ordered_stratified = SimpleNamespace(
    _step=ordered_stratified_step,
    _plan=ordered_stratified_plan,
    _frame=ordered_stratified_frame,
    _primary_binding=ordered_stratified_primary_binding,
)


def table_one_step(*, outputs: list[str] | None = None) -> AnalysisStep:
    return AnalysisStep(
        step_id="02_table_one",
        intent="Describe the locked analysis cohort by outcome.",
        inputs=[
            "artifact:analysis_cohort",
            "death",
            "age",
            "sex",
            "lact_max",
            "lact_measured",
            "lact_n",
        ],
        expected_outputs=outputs or ["table:table_one"],
        method="grouped baseline characteristics",
        table_one_spec={
            "group_by": "death",
            "group_levels": [0, 1],
            "variables": [
                {
                    "name": "age",
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "mann_whitney_or_kruskal",
                },
                {
                    "name": "sex",
                    "variable_kind": "categorical",
                    "summary": "count_percent",
                    "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                    "levels": ["Female", "Male"],
                },
                {
                    "name": "lact_max",
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "mann_whitney_or_kruskal",
                },
                {
                    "name": "lact_measured",
                    "variable_kind": "categorical",
                    "summary": "count_percent",
                    "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                    "levels": [0, 1],
                },
            ],
        },
    )


def table_one_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "death": [0, 0, 0, 1, 1, 1],
            "age": [50.0, 60.0, 70.0, 65.0, 75.0, 85.0],
            "sex": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "lact_max": [1.1, None, 2.0, 3.0, 4.0, None],
            "lact_measured": [1, 0, 1, 1, 1, 0],
            "lact_n": [1, 0, 2, 1, 3, 0],
        }
    )


table_one = SimpleNamespace(_step=table_one_step, _frame=table_one_frame)


def prediction_model_context(row_count: int) -> ResearchContext:
    return ResearchContext(
        research_question="Predict a binary outcome from prespecified variables.",
        cohort=CohortDescriptor(
            cohort_name="prediction_fixture",
            database="synthetic",
            n_stays=row_count,
            id_columns=["patient_stay_id"],
            outcome_columns=["death"],
            provenance={
                "replacement_row_identity": {
                    "output_identity_column": "patient_stay_id",
                    "mapping_file_sha256": "a" * 64,
                    "patient_group_derivation": {
                        "algorithm": "prefix_before_:s",
                        "delimiter": ":s",
                    },
                }
            },
        ),
        variables=[
            ConceptDescriptor(name="age", dtype="float64"),
            ConceptDescriptor(
                name="sex",
                dtype="object",
                observed_domain={"n_unique": 2, "levels": ["F", "M"]},
            ),
            ConceptDescriptor(name="marker", dtype="float64"),
            ConceptDescriptor(
                name="death",
                role=VariableRole.OUTCOME,
                dtype="int64",
                observed_domain={
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0, 1],
                },
            ),
        ],
        target_outcome="death",
    )


def prediction_model_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    subject_count = 90
    stays_per_subject = np.where(np.arange(subject_count) % 4 == 0, 2, 1)
    subjects = np.repeat(np.arange(subject_count), stays_per_subject)
    stay_number = np.concatenate([np.arange(count) + 1 for count in stays_per_subject])
    age = rng.normal(64, 12, len(subjects))
    marker = rng.normal(0, 1, len(subjects))
    sex = np.where(subjects % 2 == 0, "F", "M")
    logit = -2.0 + 0.025 * (age - 60) + 0.7 * marker + 0.25 * (sex == "M")
    probability = 1.0 / (1.0 + np.exp(-logit))
    death = rng.binomial(1, probability)
    # Keep both classes abundant so the one fixed group split is executable.
    death[:10] = np.arange(10) % 2
    frame = pd.DataFrame(
        {
            "patient_stay_id": [
                f"p{subject}:s{stay}"
                for subject, stay in zip(subjects, stay_number, strict=True)
            ],
            "age": age,
            "sex": sex,
            "marker": marker,
            "death": death,
        }
    )
    frame.loc[frame.index[::13], "marker"] = np.nan
    frame.loc[frame.index[::17], "sex"] = None
    return frame


def prediction_model_primary_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="primary_model",
        planned_analysis_role="primary",
        intent="Fit the prespecified static mortality model.",
        inputs=["age", "sex", "marker", "death", "artifact:analysis_cohort"],
        expected_outputs=["table:prediction_scores", "table:model_performance"],
        method="logistic prediction model",
        scientific_action_id="prediction.discrimination_calibration",
    )


prediction_model = SimpleNamespace(
    _primary_step=prediction_model_primary_step,
    _frame=prediction_model_frame,
    _context=prediction_model_context,
)


def cross_sectional_phenotyping_context(n: int) -> ResearchContext:
    return ResearchContext(
        research_question="Discover early cross-sectional phenotypes.",
        cohort=CohortDescriptor(
            cohort_name="phenotype_fixture",
            database="synthetic",
            n_stays=n,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="str"),
            ConceptDescriptor(name="marker_a", role=VariableRole.LAB, dtype="float64"),
            ConceptDescriptor(
                name="marker_b", role=VariableRole.VITAL, dtype="float64"
            ),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
        ],
        target_outcome="death",
    )


def cross_sectional_phenotyping_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 360
    group = np.repeat(np.arange(3), n // 3)
    return pd.DataFrame(
        {
            "stay_id": [f"stay_{index}" for index in range(n)],
            "marker_a": rng.normal(group * 2.5, 0.5),
            "marker_b": rng.normal((2 - group) * 2.0, 0.6),
            "death": rng.binomial(1, 0.2 + group * 0.1),
        }
    )


def cross_sectional_phenotyping_primary_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="primary_phenotypes",
        planned_analysis_role="primary",
        intent="Fit the Planner-selected early phenotype roster.",
        inputs=["stay_id", "marker_a", "marker_b", "death", "artifact:analysis_cohort"],
        expected_outputs=[PHENOTYPE_PROFILES_PRODUCT, PHENOTYPE_ASSIGNMENTS_PRODUCT],
        method="cross-sectional phenotyping",
        scientific_action_id="phenotyping.cluster_solution",
        phenotyping_feature_columns=["marker_a", "marker_b"],
    )


cross_sectional_phenotyping = SimpleNamespace(
    _primary_step=cross_sectional_phenotyping_primary_step,
    _frame=cross_sectional_phenotyping_frame,
    _context=cross_sectional_phenotyping_context,
)


def survival_primary_requirement() -> FamilyPrimaryResultRequirement:
    return FamilyPrimaryResultRequirement(
        analysis_family="survival",
        exposure_source="treatment",
        outcome="death",
        expected_result_product="table:survival_effect_estimates",
        input_product="table:analysis_cohort",
        estimator="cox_proportional_hazards",
        effect_scale="hazard_ratio",
        uncertainty_method="wald_95ci",
        population="eligible cohort",
        time_origin="ICU admission",
        time_column="follow_up_days",
        time_unit="days",
        event_column="death",
        event_value=1,
        event_definition="death",
        censoring_strategy="administrative at 28 days",
        competing_risk_strategy="none",
        time_horizon="28 days",
        time_horizon_value=28,
        effect_measure="hazard ratio",
        proportional_hazards_diagnostic="schoenfeld_global_test",
        proportional_hazards_alpha=0.05,
        proportional_hazards_policy="block_paper_authorization",
        covariates=["age"],
        model_terms=[
            {
                "name": "treatment",
                "role": "exposure",
                "coding": "binary",
                "levels": ["0", "1"],
                "reference_level": "0",
                "transform": "treatment_contrast",
            },
            {
                "name": "age",
                "role": "covariate",
                "coding": "continuous",
                "transform": "identity",
            },
        ],
        exposure_encoding="declared_model_terms",
        missing_data_policy="complete_case",
    )


def survival_primary_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="04_primary",
        intent="Estimate the declared survival contrast.",
        inputs=["table:analysis_cohort"],
        expected_outputs=[
            "table:survival_effect_estimates",
            SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
            SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
        ],
        method="cox_proportional_hazards",
        planned_analysis_role="primary",
        family_primary_result_requirement=survival_primary_requirement(),
    )


survival_primary = SimpleNamespace(
    _step=survival_primary_step, _requirement=survival_primary_requirement
)


def descriptive_result_figure_distribution_step(**updates: object) -> AnalysisStep:
    payload: dict[str, object] = {
        "step_id": "06_distribution_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Render the typed descriptive distribution.",
        "inputs": ["table:descriptive_distribution"],
        "expected_outputs": ["figure:descriptive_distribution"],
        "method": "visualization",
        "input_consumption_contracts": [
            ArtifactConsumptionContract(
                input_key="table:descriptive_distribution",
                mode="all_rows",
            )
        ],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def descriptive_result_figure_statistic_step(**updates: object) -> AnalysisStep:
    payload: dict[str, object] = {
        "step_id": "07_statistic_figure",
        "planned_analysis_role": "auxiliary",
        "intent": "Render one typed descriptive statistic.",
        "inputs": ["statistic:rank_correlation"],
        "expected_outputs": ["figure:rank_correlation_display"],
        "method": "visualization",
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def descriptive_result_figure_distribution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                "length_of_stay",
                "Overall",
                "overall",
                20,
                20,
                0,
                0.0,
                2.0,
                1.0,
                4.0,
                3.0,
                2.0,
            ],
            ["length_of_stay", "A", "group", 9, 9, 0, 0.0, 1.5, 0.8, 3.0, 2.4, 1.8],
            ["length_of_stay", "B", "group", 11, 11, 0, 0.0, 2.5, 1.2, 5.0, 3.5, 2.2],
        ],
        columns=DESCRIPTIVE_DISTRIBUTION_COLUMNS,
    )


def descriptive_result_figure_binding(
    tmp_path: Path,
    *,
    step: AnalysisStep,
    kind: str,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    run_dir = tmp_path / "run"
    parent_dir = run_dir / "steps" / "05_parent" / "outputs"
    parent_dir.mkdir(parents=True)
    input_key = step.inputs[0]
    product = input_key.partition(":")[2]
    if kind == "table":
        frame = descriptive_result_figure_distribution_frame()
        source_path = parent_dir / "typed_distribution.csv"
        frame.to_csv(source_path, index=False)
        product_contract: dict[str, object] = {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "columns": list(frame.columns),
            "row_count": len(frame),
        }
    else:
        payload = {
            "name": product,
            "value": -0.125,
            "effect_scale": "Spearman rank correlation",
            "unit": "unitless",
            "p_value": 0.04,
            "denominator": 20,
        }
        source_path = parent_dir / "typed_statistic.json"
        source_path.write_text(json.dumps(payload), encoding="utf-8")
        product_contract = {
            "schema_version": "easyicu.host_typed_product.v1",
            "json_structure": {
                "root_type": "object",
                "paths": {"": {"type": "object", "keys": list(payload)}},
            },
        }
    record = EvidenceStore(run_dir).register_file(
        kind=kind,
        description=f"Digest-bound {kind} parent.",
        source_path=source_path,
        evidence_id=f"{kind}_typed_parent",
        produced_by_step="05_parent",
        producer="deterministic_test",
        generation_mode="deterministic_standard",
    )
    path = run_dir / record.relative_path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {
        "declared_kind": kind,
        "evidence_id": record.evidence_id,
        "input_key": input_key,
        "produced_by_step": "05_parent",
        "product": product,
        "sha256": digest,
    }
    binding: dict[str, object] = {
        "relative_path": str(path.relative_to(run_dir)),
        "sha256": digest,
        "declared_kind": kind,
        "evidence_kind": kind,
        "evidence_id": record.evidence_id,
        "absolute_path": str(path),
        "produced_by_step": "05_parent",
        "product": product,
        "identity_row": identity,
        "product_contract": product_contract,
    }
    if kind == "table":
        binding["consumption_contract"] = {
            "schema_version": "easyicu.verified_artifact_consumption/1",
            "input_key": input_key,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(descriptive_result_figure_distribution_frame()),
        }
    manifest: dict[str, object] = {
        "schema_version": "2.1",
        "step_id": step.step_id,
        "inputs": {input_key: binding},
    }
    return run_dir, manifest, binding


descriptive_result_figure = SimpleNamespace(
    _distribution_step=descriptive_result_figure_distribution_step,
    _statistic_step=descriptive_result_figure_statistic_step,
    _distribution_frame=descriptive_result_figure_distribution_frame,
    _binding=descriptive_result_figure_binding,
)


def audit_panel_step(**updates: object) -> AnalysisStep:
    payload: dict[str, object] = {
        "step_id": "04_audit_panel",
        "planned_analysis_role": "auxiliary",
        "intent": "Render the framework audit panel.",
        "inputs": [],
        "expected_outputs": ["figure:audit_panel"],
        "method": "visualization",
        "icu_rule_refs": ["visualization_rule"],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


audit_panel = SimpleNamespace(_step=audit_panel_step)


def deterministic_missingness_cohort(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # lactate: measured for 60% of stays; crea: measured for 90%; rrt: measured
    # for NO stay (structural no-source in this cohort).
    lact_measured = (rng.random(n) < 0.60).astype(int)
    crea_measured = (rng.random(n) < 0.90).astype(int)
    rrt_measured = np.zeros(n, dtype=int)
    return pd.DataFrame(
        {
            "stay_id": np.arange(n),
            "age": rng.integers(40, 90, n),
            "sex": rng.choice(["Male", "Female"], n),
            "death": (rng.random(n) < 0.12).astype(int),
            "los_icu": rng.gamma(2.0, 2.0, n),
            "lactate": np.where(lact_measured == 1, rng.gamma(2.0, 1.5, n), np.nan),
            "lactate_measured": lact_measured,
            "lactate_n": lact_measured,
            "crea": np.where(crea_measured == 1, rng.gamma(2.0, 0.6, n), np.nan),
            "crea_measured": crea_measured,
            "crea_n": crea_measured,
            "rrt": np.full(n, np.nan),
            "rrt_measured": rrt_measured,
            "rrt_n": rrt_measured,
        }
    )


deterministic_missingness = SimpleNamespace(_cohort=deterministic_missingness_cohort)
