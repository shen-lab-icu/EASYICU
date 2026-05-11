"""Tests for the EHR analysis-type registry used by the planner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_infer_analysis_type_quality_audit(ra):
    schema = ra.schema
    ctx = ra.ResearchContext(
        research_question="Audit bilirubin and vasopressor measurement completeness in this ICU cohort.",
        cohort=ra.CohortDescriptor(cohort_name="c", database="synthetic", n_patients=10, n_stays=10),
        variables=[
            ra.ConceptDescriptor(name="bili", role=schema.VariableRole.LAB, dtype="float64"),
            ra.ConceptDescriptor(name="vaso", role=schema.VariableRole.INTERVENTION, dtype="float64"),
            ra.ConceptDescriptor(name="death", role=schema.VariableRole.OUTCOME, dtype="int64"),
        ],
        target_outcome="death",
    )
    spec = ra.infer_analysis_type(ctx, target_outcome="death")
    assert spec.key == "data_quality_audit"


def test_infer_analysis_type_respects_user_preference_hint(ra):
    schema = ra.schema
    ctx = ra.ResearchContext(
        research_question="Please compare ICU severity scores across cohorts.",
        cohort=ra.CohortDescriptor(cohort_name="c", database="synthetic", n_patients=10, n_stays=10),
        variables=[
            ra.ConceptDescriptor(name="sofa2", role=schema.VariableRole.COMPOSITE_SCORE, dtype="float64"),
            ra.ConceptDescriptor(name="death", role=schema.VariableRole.OUTCOME, dtype="int64"),
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
        cohort=ra.CohortDescriptor(cohort_name="c", database="synthetic", n_patients=10, n_stays=10),
        variables=[
            ra.ConceptDescriptor(name="sofa2", role=schema.VariableRole.COMPOSITE_SCORE, dtype="float64"),
            ra.ConceptDescriptor(name="death", role=schema.VariableRole.OUTCOME, dtype="int64"),
        ],
        target_outcome="death",
    )
    spec = ra.infer_analysis_type(ctx, target_outcome="death")
    assert spec.key == "validation"


def test_mock_planner_emits_prediction_analysis_and_publication_for_prediction_question(ra, tmp_path: Path):
    cohort = pd.DataFrame({
        "stay_id": range(1, 81),
        "age": [40 + (i % 30) for i in range(80)],
        "heart_rate": [70 + (i % 15) for i in range(80)],
        "death": [1 if i % 8 == 0 else 0 for i in range(80)],
    })
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
    # The planner emits ``04_prediction_model_analysis`` +
    # ``05_publication_figure_generation``, then the pipeline normalises
    # them to the canonical ``01_model_training`` step and splits the
    # figure outputs into ``01_model_training_figure`` (see
    # ``_normalise_plan_for_family`` and ``_split_table_and_figure_outputs_in_plan``).
    assert "01_model_training" in step_ids, step_ids
    assert "01_model_training_figure" in step_ids, step_ids
    assert "04_primary_association" not in step_ids


def test_reused_mock_pipeline_refreshes_context_between_prediction_and_clustering_runs(ra, tmp_path: Path):
    cohort = pd.DataFrame({
        "stay_id": range(1, 61),
        "age": [45 + (i % 25) for i in range(60)],
        "lact_t0": [1.2 + (i % 5) * 0.3 for i in range(60)],
        "lact_t6": [1.1 + (i % 5) * 0.25 for i in range(60)],
        "map_t0": [75 + (i % 7) * 2 for i in range(60)],
        "map_t6": [78 + (i % 7) * 2 for i in range(60)],
        "death": [1 if i % 9 == 0 else 0 for i in range(60)],
    })
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

    # Post-normalisation canonical step ids: prediction collapses to
    # ``01_model_training`` and clustering collapses to
    # ``01_trajectory_clustering`` (see ``_normalise_plan_for_family``).
    assert "01_model_training" in first_step_ids, first_step_ids
    assert "01_trajectory_clustering" in second_step_ids, second_step_ids


def test_mock_planner_routes_survival_question_to_protocol_and_saves_user_preferences(ra, tmp_path: Path):
    cohort = pd.DataFrame({
        "stay_id": range(1, 81),
        "time_to_event_hours": [12 + (i % 30) for i in range(80)],
        "censor_time_hours": [36 + (i % 20) for i in range(80)],
        "death": [1 if i % 7 == 0 else 0 for i in range(80)],
        "age": [50 + (i % 20) for i in range(80)],
    })
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
    ctx = json.loads((Path(result.workdir) / "research_context.json").read_text(encoding="utf-8"))
    step_ids = [step["step_id"] for step in plan["steps"]]

    assert "04_survival_protocol" in step_ids, step_ids
    assert "04_primary_association" not in step_ids
    assert ctx["user_preferences"]["inferred_analysis_family"] == "survival"
    assert "Kaplan-Meier" in (ctx["user_preferences"]["must_have_outputs"] or "")
