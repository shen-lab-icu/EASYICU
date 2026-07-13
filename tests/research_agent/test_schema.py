"""Schema round-trip and accessor tests."""

from __future__ import annotations

import json

import pytest


def test_research_context_round_trip(ra):
    ctx = ra.ResearchContext(
        research_question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic",
            n_patients=10, n_stays=10,
        ),
        variables=[
            ra.ConceptDescriptor(name="age", role="demographic", dtype="float64"),
            ra.ConceptDescriptor(name="sofa2", role="composite_score", dtype="int64",
                                 is_ordinal=True),
            ra.ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        ],
        target_outcome="death",
    )
    blob = ctx.model_dump_json()
    again = ra.ResearchContext.model_validate_json(blob)
    assert again.research_question == ctx.research_question
    assert again.cohort.n_stays == 10
    assert again.target_outcome == "death"
    # round-trip preserves the variables
    assert {v.name for v in again.variables} == {"age", "sofa2", "death"}


def test_variable_lookup(ra):
    ctx = ra.ResearchContext(
        research_question="x",
        cohort=ra.CohortDescriptor(cohort_name="c", database="s", n_patients=1, n_stays=1),
        variables=[
            ra.ConceptDescriptor(name="age", role="demographic", dtype="float64"),
            ra.ConceptDescriptor(name="sofa2", role="composite_score", dtype="int64"),
        ],
    )
    assert ctx.variable("age").role.value == "demographic"
    assert ctx.variable("sofa2").role.value == "composite_score"
    assert ctx.variable("absent_column") is None


def test_evidence_record_required_fields(ra):
    rec = ra.EvidenceRecord(
        evidence_id="x_1",
        kind="table",
        description="d",
        relative_path="evidence/x.csv",
        sha256="0" * 64,
    )
    blob = json.loads(rec.model_dump_json())
    assert blob["evidence_id"] == "x_1"
    assert blob["kind"] == "table"
    assert blob["sha256"] == "0" * 64


def test_pipeline_result_model(ra):
    res = ra.PipelineResult(
        run_id="run_x", workdir="/tmp/x",
        context_path="/tmp/x/context.json",
        plan_path="/tmp/x/plan.json",
        manifest_path="/tmp/x/manifest.json",
        report_path="/tmp/x/report.md",
        manuscript_path="/tmp/x/manuscript.md",
        evidence_count=3, findings_count=1,
    )
    paths = res.as_paths()
    assert "context" in paths and paths["context"].name == "context.json"


def test_analysis_plan_rejects_duplicate_step_ids(ra):
    with pytest.raises(ValueError, match="step_id values must be unique"):
        ra.AnalysisPlan(
            research_question="Test duplicate execution identity.",
            steps=[
                ra.AnalysisStep(step_id="01_model", intent="First owner."),
                ra.AnalysisStep(step_id="01_model", intent="Conflicting owner."),
            ],
        )
