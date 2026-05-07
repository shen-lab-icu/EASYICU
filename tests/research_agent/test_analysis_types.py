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


def test_mock_planner_emits_prediction_protocol_for_prediction_question(ra, tmp_path: Path):
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
    assert any("prediction_model_protocol" in sid for sid in step_ids), step_ids
    assert "04_primary_association" not in step_ids
