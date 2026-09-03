"""Advanced plan-contract owner tests extracted from pipeline integration."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

def test_advanced_plan_contract_preserves_prediction_evaluation_boundary(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import UserPreferences

    ctx = ra.ResearchContext(
        research_question="Build a mortality prediction model.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        user_preferences=UserPreferences(inferred_analysis_family="prediction_model"),
    )
    plan = ra.AnalysisPlan(
        research_question=ctx.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_train_model",
                intent="Train model.",
                inputs=["age", "sex"],
                expected_outputs=["model:trained_model"],
            ),
            ra.AnalysisStep(
                step_id="02_evaluate_auroc",
                intent="Evaluate AUROC from prior predictions.",
                inputs=["01_train_model"],
                expected_outputs=["statistic:auroc"],
            ),
        ],
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=ctx)

    assert [step.step_id for step in revised.steps] == [
        "01_train_model",
        "02_evaluate_auroc",
    ]
    assert revised == plan
    assert findings and findings[0].validator == "plan_contract"
    assert findings[0].detail.get("missing_structured_owner") is True


def test_advanced_plan_contract_normalizes_robustness_steps(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
        UserPreferences,
    )

    plan = AnalysisPlan(
        research_question="Compare lactate missing-data strategies.",
        steps=[
            AnalysisStep(
                step_id="01_missingness",
                intent="Summarize missingness.",
                expected_outputs=["table:missingness"],
            ),
            AnalysisStep(
                step_id="03_model_fitting_complete_case",
                intent="Fit complete-case logistic regression.",
                expected_outputs=["model:complete_case_model"],
            ),
            AnalysisStep(
                step_id="04_robustness_figure",
                intent="Generate robustness figure from model outputs.",
                expected_outputs=["figure:robustness_plot"],
            ),
        ],
    )
    context = ResearchContext(
        research_question="Compare lactate missing-data strategies.",
        cohort=CohortDescriptor(
            cohort_name="cohort",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
        user_preferences=UserPreferences(inferred_analysis_family="robustness"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == [
        "01_missingness",
        "03_model_fitting_complete_case",
        "04_robustness_figure",
    ]
    assert revised == plan
    assert findings[0].detail.get("missing_structured_owner") is True


def _survival_plan_with_only_association_steps(AnalysisPlan, AnalysisStep):
    """A locked-survival plan the LLM wrote as a pure association study."""
    return AnalysisPlan(
        research_question=(
            "Estimate the association between mechanical ventilation and 28-day "
            "mortality respecting exposure timing and censoring."
        ),
        analysis_type="survival",
        steps=[
            AnalysisStep(
                step_id="01_cohort_timezero_attrition",
                intent="Define time zero, follow-up, and cohort attrition.",
                method="descriptive",
                expected_outputs=["table:cohort_attrition"],
            ),
            AnalysisStep(
                step_id="03_table_one",
                intent="Baseline characteristics.",
                method="descriptive",
                expected_outputs=["table:table_one"],
            ),
            AnalysisStep(
                step_id="05_primary_landmark_association_model",
                intent="Fit adjusted association model for the landmark outcome.",
                method="association_study",
                expected_outputs=["table:adjusted_association_estimates"],
            ),
            AnalysisStep(
                step_id="05_primary_landmark_association_model_figure",
                intent="Forest plot of adjusted effects.",
                method="association_study",
                expected_outputs=["figure:effect_forest"],
            ),
            AnalysisStep(
                step_id="06_sensitivity_and_diagnostics",
                intent="Sensitivity and model diagnostics.",
                method="association_study",
                expected_outputs=["table:robustness_results"],
            ),
        ],
    )


def test_advanced_plan_contract_does_not_choose_survival_method_for_agent(ra):
    """A family mismatch is surfaced without rewriting the agent's science."""
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
    )

    plan = _survival_plan_with_only_association_steps(AnalysisPlan, AnalysisStep)
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="vent",
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert revised == plan
    assert findings and findings[0].validator == "plan_contract"
    assert findings[0].detail.get("missing_structured_owner") is True
    assert findings[0].detail.get("preserved_step_ids") == [
        step.step_id for step in plan.steps
    ]


def test_advanced_plan_contract_never_converts_primary_model_cohort(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract

    context = ra.ResearchContext(
        research_question="Estimate survival while preserving the planned owners.",
        cohort=ra.CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=50, n_stays=50
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="survival",
        steps=[
            ra.AnalysisStep(
                step_id="01_primary_model_cohort",
                intent=(
                    "Prepare the primary model cohort for survival analysis and "
                    "report attrition."
                ),
                method="cohort_definition",
                expected_outputs=["table:cohort_attrition"],
            ),
            ra.AnalysisStep(
                step_id="05_primary_association",
                intent="Estimate the prespecified adjusted association.",
                method="mixed_effects_regression",
                expected_outputs=["table:association_estimates"],
            ),
        ],
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert revised == plan
    assert [step.method for step in revised.steps] == [
        "cohort_definition",
        "mixed_effects_regression",
    ]
    assert findings[0].detail.get("missing_structured_owner") is True


def test_advanced_plan_contract_preserves_survival_cohort_owner_boundary(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract

    context = ra.ResearchContext(
        research_question="Estimate time-to-event survival with Cox regression.",
        cohort=ra.CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=20, n_stays=20
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="survival",
        steps=[
            ra.AnalysisStep(
                step_id="01_survival_cohort",
                intent="Define the survival cohort and report attrition.",
                method="cohort_definition",
                expected_outputs=["table:cohort_attrition"],
            ),
            ra.AnalysisStep(
                step_id="05_primary_cox",
                intent="Fit the prespecified Cox proportional-hazards model.",
                method="cox_proportional_hazards",
                expected_outputs=["table:hr"],
            ),
        ],
    )

    revised, _findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == [
        "01_survival_cohort",
        "05_primary_cox",
    ]
    assert revised.steps[0].method == "cohort_definition"
    assert revised.steps[0].expected_outputs == ["table:cohort_attrition"]
    assert revised.steps[1].method == "cox_proportional_hazards"
    assert "table:cox_summary" in revised.steps[1].expected_outputs


def test_advanced_plan_contract_preserves_explicit_kmeans_method(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract

    context = ra.ResearchContext(
        research_question="Discover longitudinal phenotypes with KMeans clustering.",
        cohort=ra.CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=20, n_stays=20
        ),
        variables=[],
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=[
            ra.AnalysisStep(
                step_id="05_kmeans_phenotyping",
                intent="Discover trajectory phenotypes with KMeans.",
                method="kmeans_clustering",
                expected_outputs=[
                    "table:cluster_assignments",
                    "table:cluster_characteristics",
                ],
            )
        ],
    )

    revised, _findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == ["05_kmeans_phenotyping"]
    assert revised.steps[0].method == "kmeans_clustering"
    assert "manifest:cluster_selection" in revised.steps[0].expected_outputs
    assert "statistic:silhouette_score" not in revised.steps[0].expected_outputs


def test_clustering_contract_does_not_invent_mortality_characterization(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract

    context = ra.ResearchContext(
        research_question="Discover longitudinal phenotypes without outcome analysis.",
        cohort=ra.CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=20, n_stays=20
        ),
        variables=[],
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=[
            ra.AnalysisStep(
                step_id="05_gmm_phenotyping",
                intent="Discover trajectory phenotypes with a Gaussian mixture model.",
                method="gaussian_mixture_model",
                expected_outputs=[
                    "table:cluster_assignments",
                    "statistic:cluster_count",
                ],
            )
        ],
    )

    revised, _findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    outputs = revised.steps[0].expected_outputs
    assert "table:cluster_mortality" not in outputs
    assert "table:outcome_by_cluster" not in outputs


def test_advanced_plan_contract_leaves_pure_association_family_alone(ra):
    """A genuine association study is NOT force-converted (upgrade-only guard)."""
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
        UserPreferences,
    )

    plan = AnalysisPlan(
        research_question="Is Sepsis-3 associated with mortality after adjustment?",
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Cohort attrition.",
                method="descriptive",
                expected_outputs=["table:cohort_attrition"],
            ),
            AnalysisStep(
                step_id="02_primary_association_model",
                intent="Adjusted logistic association.",
                method="association_study",
                expected_outputs=["table:adjusted_association_estimates"],
            ),
        ],
    )
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="sepsis3",
        user_preferences=UserPreferences(inferred_analysis_family="association_study"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    step_ids = [s.step_id for s in revised.steps]
    assert "01_survival_analysis" not in step_ids
    assert step_ids == ["01_cohort", "02_primary_association_model"]
    assert not any(
        f.detail and f.detail.get("converted_from_association") for f in findings
    )


def test_advanced_plan_contract_does_not_rewrite_cluster_robust_association(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract

    context = ra.ResearchContext(
        research_question=(
            "Estimate the mortality association using mixed effects with "
            "cluster-robust SE and hospital-level clustering."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=50, n_stays=50
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="association_study",
        steps=[
            ra.AnalysisStep(
                step_id="05_primary_association",
                intent=context.research_question,
                method="mixed_effects_regression",
                expected_outputs=["table:association_estimates"],
            )
        ],
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == ["05_primary_association"]
    assert revised.steps[0].method == "mixed_effects_regression"
    assert not any(item.detail.get("family") == "clustering" for item in findings)


def test_advanced_plan_contract_preserves_article_level_robustness_suite(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
        UserPreferences,
    )

    plan = AnalysisPlan(
        research_question=(
            "Estimate Sepsis-3 prevalence and adjusted mortality association "
            "with visible attrition, missingness, and robustness."
        ),
        steps=[
            AnalysisStep(
                step_id="01_primary_cohort_and_exposure_definition",
                intent="Define cohort eligibility, attrition, and Sepsis-3 exposure.",
                expected_outputs=["table:cohort_attrition", "derived_variable:sepsis3"],
            ),
            AnalysisStep(
                step_id="02_table_one_and_missingness",
                intent="Render Table 1 baseline characteristics and missingness audit.",
                expected_outputs=[
                    "table:table_one",
                    "table:missingness_measurement_audit",
                ],
            ),
            AnalysisStep(
                step_id="03_primary_adjusted_association",
                intent="Fit adjusted association model and report odds ratio.",
                expected_outputs=["table:adjusted_association_primary"],
            ),
            AnalysisStep(
                step_id="04_robustness_grid",
                intent="Run complete-case and alternative-definition sensitivity analyses.",
                expected_outputs=["figure:robustness_grid"],
            ),
        ],
    )
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="cohort",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="sepsis3",
        user_preferences=UserPreferences(inferred_analysis_family="robustness"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == [
        "01_primary_cohort_and_exposure_definition",
        "02_table_one_and_missingness",
        "03_primary_adjusted_association",
        "04_robustness_grid",
    ]
    robustness_step = revised.steps[-1]
    assert robustness_step.expected_outputs == ["figure:robustness_grid"]
    assert revised == plan
    assert findings[0].detail.get("missing_structured_owner") is True


def test_advanced_plan_contract_does_not_duplicate_dedicated_robustness_renderer(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
        UserPreferences,
    )

    plan = AnalysisPlan(
        research_question="Estimate an association with a complete-case sensitivity.",
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Define and account for the analysis cohort.",
                method="cohort_definition_and_attrition",
                expected_outputs=["artifact:analysis_cohort", "table:cohort_flow"],
            ),
            AnalysisStep(
                step_id="02_table_one",
                intent="Describe baseline characteristics.",
                method="descriptive",
                expected_outputs=["table:table_one"],
            ),
            AnalysisStep(
                step_id="03_measurement_audit",
                intent="Audit measurement availability.",
                method="missingness_measurement_audit",
                expected_outputs=["table:missingness_measurement_audit"],
            ),
            AnalysisStep(
                step_id="04_primary_model",
                planned_analysis_role="primary",
                intent="Estimate the prespecified association.",
                method="adjusted_association_models",
                expected_outputs=["table:adjusted_association_estimates"],
                model_requirements=[
                    {
                        "requirement_id": "primary_model",
                        "outcome": "death",
                        "outcome_type": "binary",
                        "method_family": "statsmodels_logit_mle",
                        "exposure_source": "exposure",
                        "analysis_role": "primary",
                        "analysis_set": "source_aware",
                        "required_for_step_success": True,
                        "covariates": [],
                        "model_terms": [
                            {
                                "name": "exposure",
                                "role": "exposure",
                                "coding": "binary",
                                "levels": ["0", "1"],
                                "reference_level": "0",
                                "transform": "treatment_contrast",
                            }
                        ],
                        "exposure_levels": ["0", "1"],
                        "exposure_reference_level": "0",
                        "primary_contrast_level": "1",
                    }
                ],
            ),
            AnalysisStep(
                step_id="05_robustness",
                planned_analysis_role="sensitivity",
                intent="Replay the primary model in the complete-case set.",
                method="robustness_sensitivity",
                expected_outputs=[
                    "statistic:primary_or",
                    "statistic:complete_case_n",
                    "table:robustness_summary",
                    "table:robustness_matrix",
                    "statistic:robustness_summary",
                    "log:missingness_strategy_notes",
                ],
            ),
            AnalysisStep(
                step_id="06_robustness_figure",
                planned_analysis_role="auxiliary",
                intent="Render the verified robustness matrix.",
                method="visualization",
                inputs=["table:robustness_matrix"],
                expected_outputs=["figure:robustness"],
                input_consumption_contracts=[
                    {
                        "input_key": "table:robustness_matrix",
                        "mode": "all_rows",
                    }
                ],
            ),
        ],
    )
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="cohort",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="exposure",
        user_preferences=UserPreferences(inferred_analysis_family="robustness"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert revised == plan
    assert findings == []
    figure_outputs = [
        output
        for step in revised.steps
        for output in step.expected_outputs
        if output.startswith("figure:")
    ]
    assert figure_outputs == ["figure:robustness"]


def test_advanced_plan_contract_normalizes_bias_audit_steps(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
        UserPreferences,
    )

    plan = AnalysisPlan(
        research_question="Estimate vasopressor association with mortality and audit selection bias.",
        steps=[
            AnalysisStep(
                step_id="01_cohort_summary",
                intent="Summarize cohort and vasopressor exposure.",
                expected_outputs=["table:cohort_summary"],
            ),
            AnalysisStep(
                step_id="02_outcome_incidence",
                intent="Report mortality incidence.",
                expected_outputs=["statistic:mortality_rate"],
            ),
            AnalysisStep(
                step_id="03_missingness_audit",
                intent="Audit norepinephrine-equivalent missingness.",
                expected_outputs=["table:missingness_profile"],
            ),
        ],
    )
    context = ResearchContext(
        research_question="Estimate vasopressor association with mortality and audit selection bias.",
        cohort=CohortDescriptor(
            cohort_name="cohort",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
        user_preferences=UserPreferences(inferred_analysis_family="bias_audit"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == [
        "01_cohort_summary",
        "02_outcome_incidence",
        "03_missingness_audit",
    ]
    assert revised == plan
    assert findings[0].detail.get("missing_structured_owner") is True


def test_advanced_plan_contract_does_not_rewrite_component_data_quality_audit(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
        UserPreferences,
    )

    plan = AnalysisPlan(
        research_question=(
            "Audit whether composite-score rows have enough measured components "
            "before any outcome model is fit."
        ),
        steps=[
            AnalysisStep(
                step_id="01_component_completeness_qc",
                intent="Check composite-score component completeness.",
                expected_outputs=[
                    "statistic:low_completeness_count",
                    "table:component_completeness",
                ],
            )
        ],
    )
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="cohort",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
        user_preferences=UserPreferences(inferred_analysis_family="data_quality_audit"),
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == ["01_component_completeness_qc"]
    assert findings == []


def test_advanced_plan_contract_infers_robustness_without_user_preferences(ra):
    from easyicu.research_agent.planning.advanced_plan_contract import _enforce_advanced_plan_contract
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
    )

    plan = AnalysisPlan(
        research_question="Compare complete-case, missing-indicator, and reduced-variable lactate models.",
        steps=[
            AnalysisStep(
                step_id="01_robustness_analysis",
                intent="Compare complete-case and missing-indicator robustness strategies.",
                expected_outputs=[
                    "table:robustness_summary",
                    "figure:robustness_figure",
                ],
            ),
        ],
    )
    context = ResearchContext(
        research_question="Compare complete-case, missing-indicator, and reduced-variable lactate models.",
        cohort=CohortDescriptor(
            cohort_name="cohort",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
        user_preferences=None,
    )

    revised, findings = _enforce_advanced_plan_contract(plan=plan, context=context)

    assert [step.step_id for step in revised.steps] == ["01_robustness_analysis"]
    assert revised.steps[0].expected_outputs == [
        "table:robustness_summary",
        "figure:robustness_figure",
    ]
    assert findings == []
