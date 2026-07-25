"""Explicit execution/artifact/science/paper completion axes."""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.readiness import (
    execution_gate_status,
    write_readiness_artifacts,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Is a treatment exposure associated with mortality?",
        steps=[
            AnalysisStep(
                step_id="01_feasibility",
                intent="Check exposure feasibility before modelling.",
            )
        ],
    )


def _feasibility_failure_record() -> dict[str, object]:
    return {
        "step_id": "01_feasibility",
        "status": "ok",
        "step_summary": {
            "status": "completed_feasibility_failure",
            "reason": "no usable exposure contrast",
        },
    }


def test_outer_ok_does_not_hide_scientific_feasibility_failure():
    gate = execution_gate_status(
        plan=_plan(),
        per_step_records=[_feasibility_failure_record()],
    )

    assert gate["execution_complete"] is True
    assert gate["step_scientific_requirements_complete"] is False
    assert gate["scientific_incomplete_steps"] == [
        {
            "step_id": "01_feasibility",
            "summary_status": "completed_feasibility_failure",
        }
    ]
    assert gate["step_completion_states"][0] == {
        "schema_version": "easyicu.step_completion_state/1",
        "step_id": "01_feasibility",
        "execution_ok": True,
        "outer_status": "ok",
        "summary_status": "completed_feasibility_failure",
        "scientific_requirement_complete": False,
    }


def test_run_status_surfaces_four_distinct_completion_axes(tmp_path: Path):
    context = ResearchContext(
        research_question="Is a treatment exposure associated with mortality?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
    )
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text("Manuscript scaffold not generated.\n", encoding="utf-8")

    gates, _ = write_readiness_artifacts(
        context=context,
        plan=_plan(),
        findings=[],
        per_step_records=[_feasibility_failure_record()],
        evidence=EvidenceStore(tmp_path),
        run_dir=tmp_path,
        manuscript_path=manuscript,
        stop_after_analysis=True,
    )

    assert gates["completion_schema_version"] == "easyicu.run_completion_axes/1"
    assert gates["execution_ok"] is True
    assert gates["artifact_valid"] is False
    assert gates["scientific_requirement_complete"] is False
    assert gates["paper_authorized"] is False
    assert gates["analysis_validated"] is False
    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status["schema_version"] == "easyicu.run_status/2"
    assert status["status"] == "analysis_only"


def test_explicit_development_lane_forces_diagnostic_only(tmp_path: Path):
    context = ResearchContext(
        research_question="Is a treatment exposure associated with mortality?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
    )
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text("Development-only manuscript.\n", encoding="utf-8")

    gates, _ = write_readiness_artifacts(
        context=context,
        plan=_plan(),
        findings=[],
        per_step_records=[_feasibility_failure_record()],
        evidence=EvidenceStore(tmp_path),
        run_dir=tmp_path,
        manuscript_path=manuscript,
        stop_after_analysis=True,
        force_diagnostic_only=True,
    )

    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert gates["forced_diagnostic_only"] is True
    assert status["forced_diagnostic_only"] is True
    assert status["status"] == "diagnostic_only"
