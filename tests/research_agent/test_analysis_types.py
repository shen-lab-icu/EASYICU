"""Tests for the EHR analysis-type registry used by the planner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.analysis_types import (
    is_concept_set_family,
    normalize_analysis_family,
)


def test_infer_analysis_type_quality_audit(ra):
    schema = ra.schema
    ctx = ra.ResearchContext(
        research_question="Audit bilirubin and vasopressor measurement completeness in this ICU cohort.",
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[
            ra.ConceptDescriptor(
                name="bili", role=schema.VariableRole.LAB, dtype="float64"
            ),
            ra.ConceptDescriptor(
                name="vaso", role=schema.VariableRole.INTERVENTION, dtype="float64"
            ),
            ra.ConceptDescriptor(
                name="death", role=schema.VariableRole.OUTCOME, dtype="int64"
            ),
        ],
        target_outcome="death",
    )
    spec = ra.infer_analysis_type(ctx, target_outcome="death")
    assert spec.key == "data_quality_audit"


def test_bare_word_model_does_not_force_prediction(ra):
    """The verb "model" must not stamp an association question as prediction.

    Regression for the E2 lactate item: "you may model lactate continuously" is a
    descriptive association, but a bare "model" strong-cue used to short-circuit
    infer_analysis_type to prediction_model (before the effect-size scoring),
    which then dragged the study-design family to prediction. Real prediction
    cues (predict/auroc/calibration/...) still route correctly (see M2).
    """
    from easyicu.research_agent.study_design import infer_study_design_family

    ctx = ra.ResearchContext(
        research_question=(
            "What is the descriptive association between first-24h peak lactate "
            "and in-hospital mortality? You may model lactate continuously and "
            "report an appropriate effect measure with uncertainty."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="lact",
    )
    assert ra.infer_analysis_type(ctx).key != "prediction_model"
    assert str(infer_study_design_family(ctx)) == "association"


def test_primary_cohort_workflow_boilerplate_does_not_override_scientific_family(ra):
    """One required cohort definition plus data QC is workflow, not sensitivity."""

    from easyicu.research_agent.study_design import infer_study_design_family

    ctx = ra.ResearchContext(
        research_question=(
            "Characterise the dose-response gradient of an ordered first-24h "
            "organ-dysfunction stage against mortality. Define one adult analysis "
            "cohort and state explicit inclusion criteria and exclusion criteria. "
            "Assess data quality and missingness before the outcome analysis."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[],
        primary_exposure="ordered_stage",
        target_outcome="death",
    )

    assert ra.infer_analysis_type(ctx).key == "association_study"
    assert str(infer_study_design_family(ctx)) == "association"


def test_alternative_eligibility_comparison_remains_cohort_sensitivity(ra):
    ctx = ra.ResearchContext(
        research_question=(
            "Compare the primary cohort definition with alternative eligibility "
            "criteria and report movement across definitions as a sensitivity analysis."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    assert ra.infer_analysis_type(ctx).key == "cohort_definition_sensitivity"


def test_disclaimed_causal_does_not_hijack_trajectory_clustering(ra):
    """A latent-class trajectory-clustering question that disclaims causality must
    route to trajectory_clustering, not causal_inference.

    Regression for H3: the RQ says "cluster LONGITUDINAL trajectories ... into
    latent classes" and, as a methods caveat, "do NOT interpret a trajectory
    class as a causal group". The bare "causal" strong-cue fired
    causal_inference before trajectory_clustering, imposing an unsatisfiable
    causal contract. Strong clustering framing now gates the causal family.
    """

    def _ctx(text):
        return ra.ResearchContext(
            research_question=text,
            cohort=ra.CohortDescriptor(
                cohort_name="c", database="synthetic", n_patients=10, n_stays=10
            ),
            variables=[],
            target_outcome="death",
        )

    h3 = (
        "Among adult ICU patients, cluster LONGITUDINAL organ-dysfunction "
        "trajectories (first-72h SOFA-2 binned into 6-hour windows) into latent "
        "classes, and relate the classes to in-hospital mortality DESCRIPTIVELY; "
        "do NOT interpret a trajectory class as a causal group."
    )
    assert ra.infer_analysis_type(_ctx(h3)).key == "trajectory_clustering"

    # H2 mirror: also disclaims causality ("do NOT state a causal conclusion")
    # but has NO clustering framing -> must STILL route to causal_inference.
    h2 = (
        "Estimate the association of early (first-24h) vasopressor exposure with "
        "in-hospital mortality, making confounding by indication explicit; report "
        "covariate balance and positivity for any weighted estimate, and do NOT "
        "state a causal conclusion the design cannot support."
    )
    assert ra.infer_analysis_type(_ctx(h2)).key == "causal_inference"

    # A genuine causal task that merely uses cluster-robust SEs must not be
    # dragged to clustering by the bare word "cluster".
    crobust = (
        "Estimate the causal effect of vasopressor exposure on mortality via "
        "IPTW with cluster-robust standard errors."
    )
    assert ra.infer_analysis_type(_ctx(crobust)).key == "causal_inference"


def test_existing_cluster_membership_remains_an_association_exposure(ra):
    from easyicu.research_agent.study_design import infer_study_design_family

    for membership in (
        "a previously assigned subphenotype",
        "existing latent class membership",
    ):
        ctx = ra.ResearchContext(
            research_question=(
                f"Estimate the association between {membership} and mortality "
                "using mixed-effects regression with cluster-robust standard errors."
            ),
            cohort=ra.CohortDescriptor(
                cohort_name="c", database="synthetic", n_patients=100, n_stays=100
            ),
            variables=[],
            primary_exposure="assigned_group",
            target_outcome="death",
        )
        assert ra.infer_analysis_type(ctx).key == "association_study"
        assert infer_study_design_family(ctx) == "association"


def test_clustering_variance_language_for_patients_is_not_phenotype_discovery(ra):
    from easyicu.research_agent.analysis_types import (
        strong_trajectory_clustering_framing,
    )
    from easyicu.research_agent.study_design import infer_study_design_family

    questions = (
        "Fit mixed effects with site-level clustering for patients and report "
        "the adjusted odds ratio.",
        "Use cluster-robust standard errors for longitudinal patient records "
        "when estimating the mortality association.",
        "Fit a mixed-effects model to account for clustering of patients "
        "within hospitals and report the adjusted odds ratio.",
        "Use GEE to account for clustering among patients within hospitals "
        "when estimating the mortality association.",
    )
    for question in questions:
        ctx = ra.ResearchContext(
            research_question=question,
            cohort=ra.CohortDescriptor(
                cohort_name="c", database="synthetic", n_patients=100, n_stays=100
            ),
            variables=[],
            primary_exposure="exposure",
            target_outcome="death",
        )
        assert not strong_trajectory_clustering_framing(question)
        assert ra.infer_analysis_type(ctx).key == "association_study"
        assert infer_study_design_family(ctx) == "association"


def test_gaussian_mixture_phenotype_discovery_is_clustering(ra):
    from easyicu.research_agent.analysis_types import (
        strong_trajectory_clustering_framing,
    )
    from easyicu.research_agent.study_design import infer_study_design_family

    question = (
        "Discover longitudinal patient phenotypes using a Gaussian mixture "
        "model, select the class count, and report cluster stability."
    )
    ctx = ra.ResearchContext(
        research_question=question,
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=100, n_stays=100
        ),
        variables=[],
        target_outcome="death",
    )

    assert strong_trajectory_clustering_framing(question)
    assert ra.infer_analysis_type(ctx).key == "trajectory_clustering"
    assert infer_study_design_family(ctx) == "phenotyping"


def test_causal_disclaimer_and_fixed_followup_endpoint_do_not_choose_science(ra):
    from easyicu.research_agent.study_design import infer_study_design_family

    questions = (
        "Estimate the adjusted association between exposure and mortality; do "
        "not make a causal claim.",
        "Assess binary 28-day mortality at follow-up using logistic regression.",
    )
    for question in questions:
        ctx = ra.ResearchContext(
            research_question=question,
            cohort=ra.CohortDescriptor(
                cohort_name="c", database="synthetic", n_patients=100, n_stays=100
            ),
            variables=[],
            primary_exposure="exposure",
            target_outcome="death",
        )
        assert ra.infer_analysis_type(ctx).key == "association_study"
        assert infer_study_design_family(ctx) == "association"


def test_chinese_negation_and_cluster_nuisance_do_not_hijack_family(ra):
    from easyicu.research_agent.study_design import infer_study_design_family

    questions = (
        "采用混合效应回归和医院层面聚类稳健标准误，评估既有轨迹分群与死亡的关联。",
        "使用混合效应回归识别医院聚类效应，不进行患者表型分群。",
        "识别医院聚类效应并使用混合效应回归，不做患者表型发现。",
        "评估暴露与死亡的调整关联，不作因果解释，使用逻辑回归。",
        "估计固定28天死亡结局的逻辑回归关联，不进行生存分析。",
    )
    for question in questions:
        ctx = ra.ResearchContext(
            research_question=question,
            cohort=ra.CohortDescriptor(
                cohort_name="c", database="synthetic", n_patients=100, n_stays=100
            ),
            variables=[],
            primary_exposure="exposure",
            target_outcome="death",
        )
        assert ra.infer_analysis_type(ctx).key == "association_study", question
        assert infer_study_design_family(ctx) == "association", question


def test_new_idea_mining_families_are_concept_set_shapes() -> None:
    assert normalize_analysis_family("measurement bias") == "measurement_bias_audit"
    assert (
        normalize_analysis_family("definition_sensitivity")
        == "cohort_definition_sensitivity"
    )
    assert normalize_analysis_family("imputation_policy") == "score_policy_sensitivity"
    assert is_concept_set_family("measurement_bias_audit")
    assert is_concept_set_family("cohort_definition_sensitivity")
    assert is_concept_set_family("score_policy_sensitivity")


def test_infer_analysis_type_measurement_bias_before_generic_quality_audit(ra):
    schema = ra.schema
    ctx = ra.ResearchContext(
        research_question=(
            "Audit measurement bias from selective laboratory testing frequency "
            "and missingness in this ICU cohort."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[
            ra.ConceptDescriptor(
                name="lactate", role=schema.VariableRole.LAB, dtype="float64"
            ),
            ra.ConceptDescriptor(
                name="death", role=schema.VariableRole.OUTCOME, dtype="int64"
            ),
        ],
        target_outcome="death",
    )

    spec = ra.infer_analysis_type(ctx, target_outcome="death")

    assert spec.key == "measurement_bias_audit"


def test_infer_analysis_type_respects_user_preference_hint(ra):
    schema = ra.schema
    ctx = ra.ResearchContext(
        research_question="Please compare ICU severity scores across cohorts.",
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[
            ra.ConceptDescriptor(
                name="sofa2", role=schema.VariableRole.COMPOSITE_SCORE, dtype="float64"
            ),
            ra.ConceptDescriptor(
                name="death", role=schema.VariableRole.OUTCOME, dtype="int64"
            ),
        ],
        target_outcome="death",
        user_preferences=schema.UserPreferences(
            inferred_analysis_family="validation",
            must_have_outputs="external validation table and calibration figure",
        ),
    )
    spec = ra.infer_analysis_type(ctx, target_outcome="death")
    assert spec.key == "validation"


def test_infer_analysis_type_prefers_validation_over_prediction_keywords(ra):
    schema = ra.schema
    ctx = ra.ResearchContext(
        research_question=(
            "Externally validate SOFA-2 and qSOFA for ICU mortality, "
            "compare discrimination, calibration, and transportability across cohorts."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=10, n_stays=10
        ),
        variables=[
            ra.ConceptDescriptor(
                name="sofa2", role=schema.VariableRole.COMPOSITE_SCORE, dtype="float64"
            ),
            ra.ConceptDescriptor(
                name="death", role=schema.VariableRole.OUTCOME, dtype="int64"
            ),
        ],
        target_outcome="death",
    )
    spec = ra.infer_analysis_type(ctx, target_outcome="death")
    assert spec.key == "validation"


def test_mock_planner_emits_prediction_analysis_and_publication_for_prediction_question(
    ra, tmp_path: Path
):
    cohort = pd.DataFrame(
        {
            "stay_id": range(1, 81),
            "age": [40 + (i % 30) for i in range(80)],
            "heart_rate": [70 + (i % 15) for i in range(80)],
            "death": [1 if i % 8 == 0 else 0 for i in range(80)],
        }
    )
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Build an ICU mortality prediction model and define evaluation metrics.",
        cohort=cohort,
        cohort_name="prediction_protocol_test",
        database="synthetic",
        target_outcome="death",
    )

    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    step_ids = [step["step_id"] for step in plan["steps"]]
    # The agent-selected prediction step identity is preserved; deterministic
    # plan handling may add product contracts but must not collapse the plan into
    # a benchmark-shaped canonical mega-step.
    assert "04_prediction_model_analysis" in step_ids, step_ids
    assert "04_prediction_model_analysis_figure" in step_ids, step_ids
    assert "04_primary_association" not in step_ids


def test_reused_mock_pipeline_refreshes_context_between_prediction_and_clustering_runs(
    ra, tmp_path: Path
):
    cohort = pd.DataFrame(
        {
            "stay_id": range(1, 61),
            "age": [45 + (i % 25) for i in range(60)],
            "lact_t0": [1.2 + (i % 5) * 0.3 for i in range(60)],
            "lact_t6": [1.1 + (i % 5) * 0.25 for i in range(60)],
            "map_t0": [75 + (i % 7) * 2 for i in range(60)],
            "map_t6": [78 + (i % 7) * 2 for i in range(60)],
            "death": [1 if i % 9 == 0 else 0 for i in range(60)],
        }
    )
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())

    first = pipeline.run(
        question="Build an ICU mortality prediction model and define evaluation metrics.",
        cohort=cohort,
        cohort_name="prediction_then_clustering",
        database="synthetic",
        target_outcome="death",
    )
    second = pipeline.run(
        question=(
            "Cluster ICU patients by first-24h lactate and MAP trajectories "
            "to identify hemodynamic subphenotypes and compare mortality."
        ),
        cohort=cohort,
        cohort_name="prediction_then_clustering",
        database="synthetic",
        target_outcome="death",
    )

    first_plan = json.loads(Path(first.plan_path).read_text(encoding="utf-8"))
    second_plan = json.loads(Path(second.plan_path).read_text(encoding="utf-8"))

    first_step_ids = [step["step_id"] for step in first_plan["steps"]]
    second_step_ids = [step["step_id"] for step in second_plan["steps"]]

    # Reusing one pipeline refreshes the context while preserving each agent's
    # method-specific step identity rather than forcing canonical mega-steps.
    assert "04_prediction_model_analysis" in first_step_ids, first_step_ids
    assert "04_trajectory_clustering_analysis" in second_step_ids, second_step_ids


def test_mock_planner_routes_survival_question_to_protocol_and_saves_user_preferences(
    ra, tmp_path: Path
):
    cohort = pd.DataFrame(
        {
            "stay_id": range(1, 81),
            "time_to_event_hours": [12 + (i % 30) for i in range(80)],
            "censor_time_hours": [36 + (i % 20) for i in range(80)],
            "death": [1 if i % 7 == 0 else 0 for i in range(80)],
            "age": [50 + (i % 20) for i in range(80)],
        }
    )
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question=(
            "Evaluate 28-day survival after ICU admission with explicit time zero, "
            "censoring rules, Kaplan-Meier curves, and a Cox-style model."
        ),
        cohort=cohort,
        cohort_name="survival_protocol_test",
        database="synthetic",
        target_outcome="death",
        user_preferences={
            "inferred_analysis_family": "survival",
            "timing_and_design": "time zero at ICU admission; 28-day follow-up",
            "must_have_outputs": "Kaplan-Meier plot and hazard ratio table",
        },
    )

    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    ctx = json.loads(
        (Path(result.workdir) / "research_context.json").read_text(encoding="utf-8")
    )
    step_ids = [step["step_id"] for step in plan["steps"]]

    # The mock agent chose a protocol-only survival plan. The framework may
    # review that choice, but it must not invent a Cox estimand or rewrite the
    # agent plan into a canonical method-shaped mega-step.
    assert plan["analysis_type"] == "survival"
    protocol = next(
        step for step in plan["steps"] if step["step_id"] == "04_survival_protocol"
    )
    assert protocol["method"] == "survival_protocol"
    assert protocol["expected_outputs"] == ["log:survival_protocol"]
    assert "01_survival_analysis" not in step_ids, step_ids
    assert "04_primary_association" not in step_ids
    assert ctx["user_preferences"]["inferred_analysis_family"] == "survival"
    assert "Kaplan-Meier" in (ctx["user_preferences"]["must_have_outputs"] or "")


def test_planner_prompt_suggests_inferred_family(ra):
    """The planner sees a focused suggestion plus the full catalog."""
    import importlib

    agents = importlib.import_module("easyicu.research_agent.agents")

    def _ctx(question: str):
        return ra.ResearchContext(
            research_question=question,
            cohort=ra.CohortDescriptor(
                cohort_name="c", database="miiv", n_patients=200, n_stays=200
            ),
            variables=[
                ra.ConceptDescriptor(
                    name="sofa",
                    role=ra.schema.VariableRole.COMPOSITE_SCORE,
                    dtype="float64",
                    is_ordinal=True,
                ),
                ra.ConceptDescriptor(
                    name="death", role=ra.schema.VariableRole.OUTCOME, dtype="int64"
                ),
            ],
            target_outcome="death",
        )

    cases = {
        "Cox proportional hazards time-to-event survival of 28-day mortality.": "survival",
        "Discover patient subphenotypes via trajectory clustering of vitals.": "trajectory_clustering",
        "Estimate admission SOFA association with ICU mortality.": "association_study",
    }
    for question, expected_family in cases.items():
        prompt = agents._build_planner_user_prompt(_ctx(question))
        assert "INFERRED ANALYSIS FAMILY SUGGESTION" in prompt
        suggested_line = next(
            line
            for line in prompt.splitlines()
            if "INFERRED ANALYSIS FAMILY SUGGESTION" in line
        )
        assert expected_family in suggested_line, (question, suggested_line)
        # The full catalog still follows as reference.
        assert "ANALYSIS-TYPE CATALOG" in prompt


def test_parse_fills_inferred_analysis_type_only_when_agent_omits_it(ra):
    import importlib
    import json

    agents = importlib.import_module("easyicu.research_agent.agents")

    ctx = ra.ResearchContext(
        research_question="Cox proportional hazards survival of 28-day mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="miiv", n_patients=200, n_stays=200
        ),
        variables=[
            ra.ConceptDescriptor(
                name="death", role=ra.schema.VariableRole.OUTCOME, dtype="int64"
            )
        ],
        target_outcome="death",
    )
    valid_plan = json.dumps(
        {
            "research_question": ctx.research_question,
            "steps": [
                {
                    "step_id": "01_fit",
                    "intent": "fit cox model",
                    "inputs": [],
                    "expected_outputs": ["table:hr"],
                    "method": "cox_ph",
                    "icu_rule_refs": [],
                }
            ],
            "rationale": "r",
        }
    )
    planner = agents.PlannerAgent.__new__(agents.PlannerAgent)
    planner.last_dropped_plan_keys = {"top_level": [], "steps": []}
    plan = agents.PlannerAgent._parse(planner, valid_plan, ctx)
    assert plan.analysis_type == "survival", plan.analysis_type


def test_parse_preserves_agent_selected_family_and_rationale(ra):
    import importlib

    agents = importlib.import_module("easyicu.research_agent.agents")
    ctx = ra.ResearchContext(
        research_question=(
            "Estimate a binary mortality association at 28 days after follow-up "
            "using logistic regression."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="miiv", n_patients=200, n_stays=200
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="exposure",
    )
    raw = json.dumps(
        {
            "research_question": ctx.research_question,
            "analysis_type": "association_study",
            "steps": [
                {
                    "step_id": "01_association",
                    "intent": "Fit the prespecified logistic association.",
                    "method": "logistic_regression",
                    "expected_outputs": ["table:association_estimates"],
                }
            ],
            "rationale": (
                "The outcome is a fixed binary endpoint; follow-up describes "
                "ascertainment and does not imply a time-to-event estimand."
            ),
        }
    )
    planner = agents.PlannerAgent.__new__(agents.PlannerAgent)
    planner.last_dropped_plan_keys = {"top_level": [], "steps": []}

    plan = agents.PlannerAgent._parse(planner, raw, ctx)

    assert plan.analysis_type == "association_study"
    assert "fixed binary endpoint" in plan.rationale


def test_parse_rejects_unknown_analysis_type_instead_of_bypassing_contract(ra):
    import importlib

    agents = importlib.import_module("easyicu.research_agent.agents")
    ctx = ra.ResearchContext(
        research_question="Estimate a time-to-event association.",
        cohort=ra.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=100, n_stays=100
        ),
        variables=[],
        target_outcome="death",
    )
    raw = json.dumps(
        {
            "research_question": ctx.research_question,
            "analysis_type": "survial",
            "steps": [
                {
                    "step_id": "01_model",
                    "intent": "Fit the declared model.",
                    "method": "cox_proportional_hazards",
                    "expected_outputs": ["table:hazard_ratio"],
                }
            ],
        }
    )
    planner = agents.PlannerAgent.__new__(agents.PlannerAgent)
    planner.last_dropped_plan_keys = {"top_level": [], "steps": []}

    with pytest.raises(ValueError, match="Unknown analysis_type"):
        agents.PlannerAgent._parse(planner, raw, ctx)


def test_infer_does_not_misclassify_lab_names_as_multimodal(ra):
    """Substring 'ct' (CT scan) inside lab names like 'lactate' must not score
    multimodal. Surfaced by a real gpt-5.4 run where an association cohort with a
    lactate covariate stamped analysis_type='multimodal'."""
    schema = ra.schema

    def _ctx(extra_var_names):
        variables = [
            schema.ConceptDescriptor(
                name="sofa",
                role=schema.VariableRole.COMPOSITE_SCORE,
                dtype="float64",
                is_ordinal=True,
            )
        ]
        for name in extra_var_names:
            variables.append(
                schema.ConceptDescriptor(
                    name=name, role=schema.VariableRole.LAB, dtype="float64"
                )
            )
        variables.append(
            schema.ConceptDescriptor(
                name="death", role=schema.VariableRole.OUTCOME, dtype="int64"
            )
        )
        return schema.ResearchContext(
            research_question=(
                "Is admission SOFA associated with ICU mortality after adjusting "
                "for age and lactate?"
            ),
            cohort=schema.CohortDescriptor(
                cohort_name="c", database="miiv", n_patients=500, n_stays=500
            ),
            variables=variables,
            target_outcome="death",
        )

    from easyicu.research_agent.analysis_types import infer_analysis_type

    # Lab names containing the substring 'ct'/'note' etc. must NOT score multimodal.
    assert infer_analysis_type(_ctx(["lactate"])).key == "association_study"
    assert infer_analysis_type(_ctx(["lactate", "extract_flag"])).key == "association_study"
    # Genuine modality variables must still be detected as multimodal.
    assert infer_analysis_type(_ctx(["ct_scan_present"])).key == "multimodal"
    assert infer_analysis_type(_ctx(["clinical_note"])).key == "multimodal"
