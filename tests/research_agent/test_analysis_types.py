"""Tests for the EHR analysis-type registry used by the planner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
    # The planner emits ``04_prediction_model_analysis`` +
    # ``05_publication_figure_generation``, then the pipeline normalises
    # them to the canonical ``01_model_training`` step and splits the
    # figure outputs into ``01_model_training_figure`` (see
    # ``_normalise_plan_for_family`` and ``_split_table_and_figure_outputs_in_plan``).
    assert "01_model_training" in step_ids, step_ids
    assert "01_model_training_figure" in step_ids, step_ids
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

    # Post-normalisation canonical step ids: prediction collapses to
    # ``01_model_training`` and clustering collapses to
    # ``01_phenotype_trajectory_clustering`` (see ``_normalise_plan_for_family``).
    assert "01_model_training" in first_step_ids, first_step_ids
    assert "01_phenotype_trajectory_clustering" in second_step_ids, second_step_ids


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

    # The advanced-plan contract now normalizes a survival question into a
    # self-contained canonical survival step carrying the KM/Cox figure
    # contract (replacing the mock planner's bare 04_survival_protocol). The
    # figure output is then split into a dedicated sibling figure step, so the
    # contract surfaces as 01_survival_analysis(+_figure) rather than a bare
    # association plan.
    assert "01_survival_analysis" in step_ids, step_ids
    assert "04_primary_association" not in step_ids
    all_outputs = [
        output for step in plan["steps"] for output in step["expected_outputs"]
    ]
    assert "figure:survival_curves" in all_outputs, step_ids
    assert "table:cox_summary" in all_outputs
    assert ctx["user_preferences"]["inferred_analysis_family"] == "survival"
    assert "Kaplan-Meier" in (ctx["user_preferences"]["must_have_outputs"] or "")


def test_planner_prompt_locks_inferred_family(ra):
    """The planner prompt must name the single inferred family for THIS study,
    not only present the generic catalog (regression: pilot plans came back
    with analysis_type=None and every question collapsed to logistic)."""
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
        assert "LOCKED ANALYSIS FAMILY FOR THIS STUDY" in prompt
        locked_line = next(
            line for line in prompt.splitlines() if "LOCKED ANALYSIS FAMILY" in line
        )
        assert expected_family in locked_line, (question, locked_line)
        # The full catalog still follows as reference.
        assert "ANALYSIS-TYPE CATALOG" in prompt


def test_parse_stamps_analysis_type_onto_plan(ra):
    """PlannerAgent._parse must stamp the deterministically inferred family
    onto the plan so downstream method/figure routing has a non-None signal."""
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
