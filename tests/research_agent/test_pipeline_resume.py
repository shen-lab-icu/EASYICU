from __future__ import annotations

import ast
from pathlib import Path

import pytest

from easyicu.research_agent.pipeline_resume import (
    ResumeController,
    upsert_step_record,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Resume policy test.",
        steps=[
            AnalysisStep(
                step_id="01_define",
                intent="Define the cohort.",
                expected_outputs=["table:cohort"],
            ),
            AnalysisStep(
                step_id="02_model",
                intent="Fit the model.",
                expected_outputs=["table:model"],
            ),
            AnalysisStep(
                step_id="03_figure",
                intent="Render the figure.",
                expected_outputs=["figure:model"],
            ),
        ],
    )


def test_resume_controller_drops_requested_step_and_stale_findings(tmp_path: Path):
    state = {
        "per_step_records": [
            {
                "step_id": "00_probe",
                "status": "ok",
                "step_summary": {"n_rows": 10},
            },
            {"step_id": "01_define", "status": "ok"},
            {"step_id": "02_model", "status": "ok"},
            {"step_id": "03_figure", "status": "ok"},
            {"step_id": "04_unplanned", "status": "ok"},
            {"step_id": "02_model", "status": "failed"},
        ],
        "findings": [
            {
                "validator": "step_contract",
                "severity": "error",
                "message": "stale issue for 02_model",
                "detail": {"step_id": "02_model"},
            },
            {
                "validator": "manuscript_gate",
                "severity": "error",
                "message": "stale downstream issue",
                "detail": {"failed_steps": [{"step_id": "03_figure"}]},
            },
            {
                "validator": "clinical",
                "severity": "warning",
                "message": "keep upstream warning",
                "detail": {"step_id": "01_define"},
            },
            {
                "validator": "cohort_auditor",
                "severity": "warning",
                "message": "legacy cohort warning is recomputed on resume",
            },
            {
                "validator": "runner",
                "severity": "error",
                "message": "prior runner failure for 01_define should clear",
            },
        ],
    }

    applied = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state=state,
        resume_from_step_id="02_model",
        stop_after_step_id="02_model",
    ).apply()

    assert applied.resumed_step_ids == {"00_probe", "01_define"}
    assert [record["step_id"] for record in applied.per_step_records] == [
        "00_probe",
        "01_define",
    ]
    assert applied.probe_summary == {"n_rows": 10}
    messages = [finding.message for finding in applied.findings]
    assert "keep upstream warning" in messages
    assert not any("stale" in message for message in messages)
    resume_findings = [
        finding for finding in applied.findings if finding.validator == "resume"
    ]
    assert resume_findings
    assert resume_findings[-1].detail["dropped_completed_step_ids"] == [
        "02_model",
        "03_figure",
        "04_unplanned",
    ]


def test_pipeline_resume_entrypoints_are_importable() -> None:
    from easyicu.research_agent.pipeline_resume import ResumeApplication

    assert ResumeController.__name__ == "ResumeController"
    assert ResumeApplication.__name__ == "ResumeApplication"
    assert callable(upsert_step_record)


def test_pipeline_resume_is_a_leaf_module() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "pipeline_resume.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {
        "pipeline",
        "pipeline_execute",
        "easyicu.research_agent.pipeline",
        "easyicu.research_agent.pipeline_execute",
    }
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module in forbidden
        for node in tree.body
    )


def test_pipeline_execute_delegates_resume_policy_to_resume_controller() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "pipeline_execute.py"
    )
    source = path.read_text(encoding="utf-8")

    assert "ResumeController(" in source
    assert "def _resume_code_for_step" not in source
    assert "def _remaining_steps" not in source
    assert "def _initial_step_index" not in source


def test_resume_controller_reuses_latest_valid_code_evidence(tmp_path: Path):
    bad_path = tmp_path / "evidence" / "bad.py"
    good_path = tmp_path / "evidence" / "good.py"
    bad_path.parent.mkdir()
    bad_path.write_text("{}", encoding="utf-8")
    good_path.write_text(
        "import os\nprint(os.environ['COHORT_PARQUET'])\n",
        encoding="utf-8",
    )
    state = {
        "evidence": [
            {
                "evidence_id": "code_good",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": "evidence/good.py",
            },
            {
                "evidence_id": "code_bad",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": "evidence/bad.py",
            },
        ]
    }

    code, record = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state=state,
        resume_from_step_id="02_model",
    ).prior_code_for_step("02_model")

    assert "COHORT_PARQUET" in code
    assert record["evidence_id"] == "code_good"


def test_resume_controller_rejects_code_evidence_outside_run_dir(tmp_path: Path):
    outside = tmp_path / "outside.py"
    outside.write_text("import os\nprint(os.environ['COHORT_PARQUET'])\n")
    state = {
        "evidence": [
            {
                "evidence_id": "code_escape",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": "../outside.py",
            },
            {
                "evidence_id": "code_absolute",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": str(outside),
            },
        ]
    }

    reused = ResumeController(
        plan=_plan(),
        run_dir=tmp_path / "run",
        resume_state=state,
        resume_from_step_id="02_model",
    ).prior_code_for_step("02_model")

    assert reused is None


def test_resume_controller_remaining_steps_respects_stop_point(tmp_path: Path):
    controller = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state=None,
        stop_after_step_id="02_model",
    )

    remaining = controller.remaining_steps(
        plan=_plan(),
        executed_step_ids={"01_define"},
    )

    assert [step.step_id for step in remaining] == ["02_model"]


def test_resume_controller_rejects_unknown_resume_or_stop_step(tmp_path: Path):
    with pytest.raises(ValueError, match="resume_from_step_id"):
        ResumeController(
            plan=_plan(),
            run_dir=tmp_path,
            resume_state=None,
            resume_from_step_id="missing",
        )

    with pytest.raises(ValueError, match="stop_after_step_id"):
        ResumeController(
            plan=_plan(),
            run_dir=tmp_path,
            resume_state=None,
            stop_after_step_id="missing",
        )


def test_upsert_step_record_replaces_pending_and_preserves_completed_records():
    records = [
        {"step_id": "02_model", "status": "executed_pending_review"},
        {"step_id": "03_figure", "status": "ok"},
    ]

    upsert_step_record(
        records,
        {"step_id": "02_model", "status": "ok"},
        replace_statuses={"executed_pending_review"},
    )
    upsert_step_record(
        records,
        {"step_id": "03_figure", "status": "contract_failed"},
        replace_statuses={"executed_pending_review"},
    )
    upsert_step_record(records, {"step_id": "04_extra", "status": "ok"})

    assert records == [
        {"step_id": "02_model", "status": "ok"},
        {"step_id": "03_figure", "status": "ok"},
        {"step_id": "03_figure", "status": "contract_failed"},
        {"step_id": "04_extra", "status": "ok"},
    ]
