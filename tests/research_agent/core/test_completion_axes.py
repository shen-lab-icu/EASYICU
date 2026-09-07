"""Explicit execution/artifact/science/paper completion axes."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest
from pathlib import Path

from easyicu.research_agent.reporting.completion import (
    RunCompletionDecision,
    RunCompletionFacts,
    readiness_status,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.readiness import (
    execution_gate_status,
    render_report,
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


def test_post_readiness_reviewer_inherits_final_paper_gate(tmp_path: Path):
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
    evidence = EvidenceStore(tmp_path)
    evidence.register_text(
        kind="log",
        description="Pre-readiness simulated reviewer report.",
        text=json.dumps(
            {
                "round": 0,
                "summary": {
                    "aggregated_recommendation": "accept",
                    "counts": {},
                },
                "critiques": [],
            }
        ),
        filename="reviewer_report.json",
        evidence_id="reviewer_report_json",
        aliases=["reviewer_report_json"],
        producer="pipeline",
        generation_mode="system",
    )

    gates, artifact_paths = write_readiness_artifacts(
        context=context,
        plan=_plan(),
        findings=[],
        per_step_records=[_feasibility_failure_record()],
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=manuscript,
        stop_after_analysis=True,
        force_diagnostic_only=True,
    )

    final_report = json.loads(
        (tmp_path / "reviewer_report_post_readiness.json").read_text(encoding="utf-8")
    )
    scientific_gate_comments = [
        comment
        for critique in final_report["critiques"]
        for comment in critique["comments"]
        if comment["topic"] == "scientific_gate"
    ]
    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))

    assert gates["post_readiness_reviewer_recommendation"] == "major_revision"
    assert scientific_gate_comments
    assert artifact_paths["reviewer_report_post_readiness_json"] == (
        "reviewer_report_post_readiness.json"
    )
    assert status["canonical_outputs"]["reviewer_report_post_readiness_json"] == (
        "reviewer_report_post_readiness.json"
    )
    assert evidence.get("run_status") is not None
    assert "run_status" in {
        record.evidence_id
        for record in evidence.current_verified_records([_feasibility_failure_record()])
    }


def test_report_keeps_analysis_only_distinct_from_diagnostic_only(tmp_path: Path):
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

    report = render_report(
        context=context,
        plan=None,
        findings=[],
        per_step_records=[],
        evidence=EvidenceStore(tmp_path),
        readiness={
            "execution_complete": True,
            "evidence_complete": False,
            "numeric_verified": False,
            "analysis_validated": False,
            "manuscript_ready": False,
            "display_suite_complete": False,
            "publication_ready": False,
        },
    )

    assert "## Status: ANALYSIS ONLY" in report


def _all_content_facts(**changes) -> RunCompletionFacts:
    return replace(
        RunCompletionFacts(
            execution_complete=True,
            evidence_complete=True,
            numeric_verified=True,
            analysis_validated=True,
            publication_figure_bundle_ready=True,
            publication_provenance_ready=True,
            display_suite_complete=True,
            article_contract_complete=True,
            article_figure_strategy_complete=True,
            scientific_maturity_article_grade=True,
            plan_truncated=False,
            replan_budget_exhausted=False,
            administrative_metadata_verified=True,
        ),
        **changes,
    )


def _authorized_decision(**fact_changes):
    return RunCompletionDecision(
        _all_content_facts(**fact_changes),
        execution_paper_eligible=True,
        plan_authority_verified=True,
        plan_authority_sha256="a" * 64,
    )


@pytest.mark.parametrize(
    "changes,expected_status",
    [
        ({"execution_complete": False}, "diagnostic_only"),
        ({"evidence_complete": False}, "analysis_only"),
        ({"numeric_verified": False}, "analysis_only"),
        ({"analysis_validated": False}, "analysis_only"),
        ({"publication_figure_bundle_ready": False}, "manuscript_ready"),
        ({"publication_provenance_ready": False}, "manuscript_ready"),
        ({"display_suite_complete": False}, "manuscript_ready"),
        ({"article_contract_complete": False}, "manuscript_ready"),
        ({"article_figure_strategy_complete": False}, "manuscript_ready"),
        ({"scientific_maturity_article_grade": False}, "manuscript_ready"),
        ({"plan_truncated": True}, "manuscript_ready"),
        ({"replan_budget_exhausted": True}, "diagnostic_only"),
    ],
)
def test_each_missing_scientific_requirement_blocks_paper_authority(
    changes, expected_status
):
    decision = _authorized_decision(**changes)
    assert decision.status == expected_status
    assert decision.paper_authorized is False
    assert readiness_status(decision.to_gates()) == expected_status


@pytest.mark.parametrize(
    "authority,expected_status,authorized",
    [
        ({}, "publication_ready", True),
        ({"execution_paper_eligible": False}, "publication_ready", False),
        ({"plan_authority_verified": False}, "publication_ready", False),
        ({"plan_authority_sha256": None}, "publication_ready", False),
        ({"plan_authority_sha256": "not-a-digest"}, "publication_ready", False),
        ({"plan_authority_sha256": "a" * 63}, "publication_ready", False),
        ({"forced_diagnostic_only": True}, "diagnostic_only", False),
    ],
)
def test_content_readiness_and_final_paper_authority_are_separate(
    authority, expected_status, authorized
):
    decision = replace(_authorized_decision(), **authority)
    assert decision.publication_ready is True
    assert decision.status == expected_status
    assert decision.paper_authorized is authorized
    gates = decision.to_gates()
    assert gates["publication_artifacts_ready"] is (
        expected_status == "publication_ready"
    )
    assert gates["paper_authorized"] is authorized
    assert readiness_status(gates) == expected_status


def test_no_identity_can_never_authorize_a_content_ready_run():
    decision = RunCompletionDecision(_all_content_facts())
    assert decision.status == "publication_ready"
    assert decision.publication_artifacts_ready is True
    assert decision.paper_authorized is False
    assert decision.plan_authority_sha256 is None


def test_administrative_metadata_does_not_change_scientific_authority():
    decision = _authorized_decision(administrative_metadata_verified=False)
    assert decision.paper_authorized is True
    assert decision.to_gates()["submission_ready"] is False


def test_completion_is_immutable_and_projections_are_independent():
    decision = _authorized_decision()
    with pytest.raises(FrozenInstanceError):
        decision.execution_paper_eligible = False
    with pytest.raises(FrozenInstanceError):
        decision.facts.evidence_complete = False
    projection = decision.to_gates()
    projection["paper_authorized"] = False
    assert decision.to_gates()["paper_authorized"] is True
    assert (
        replace(
            decision, plan_authority_sha256=" A" + "a" * 63 + " "
        ).plan_authority_sha256
        == "a" * 64
    )


@pytest.mark.parametrize("value", [None, 1, "false"])
def test_unknown_or_truthy_strings_are_not_boolean_completion_verdicts(value):
    with pytest.raises(TypeError, match="completion_fact_requires_boolean"):
        _all_content_facts(evidence_complete=value)
    with pytest.raises(TypeError, match="completion_authority_requires_boolean"):
        replace(_authorized_decision(), plan_authority_verified=value)


def test_a_report_rejects_a_contradictory_completion_status_projection():
    projection = _authorized_decision().to_gates()
    projection["forced_diagnostic_only"] = True
    with pytest.raises(ValueError, match="completion_status_projection_mismatch"):
        readiness_status(projection)
