"""A run launch logs how long each of its synchronous steps took.

The request that starts a run answers only once the host has read the source,
checked the setup and the data package, prepared the runner and recorded the
job; on a large package that took over a minute, and nothing said which step.
The launch now logs one line per started job: the job id and each step's
seconds, nothing about the study or its data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.webserver import research_run_submission as submission

_STEPS = ("describe_source", "readiness", "verify_package", "prepare_runner", "authorize", "start_job")


@pytest.fixture
def launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    study = {
        "id": "study-clock",
        "question": "How common is hypoglycaemia on the first ICU day?",
        "data_source": {"path": str(tmp_path / "private-export"), "database": "eicu"},
    }
    monkeypatch.delenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", raising=False)
    monkeypatch.setattr(submission.context_store, "get_context", lambda _: study)
    monkeypatch.setattr(submission.dataio, "describe_export_source", lambda _: {"ok": True})
    monkeypatch.setattr(submission.dataio, "prepared_export_manifest_path", lambda _: None)
    monkeypatch.setattr(
        submission, "build_research_workflow_snapshot",
        lambda **k: SimpleNamespace(planning_prerequisites_missing=[]),
    )
    monkeypatch.setattr(submission, "provider_environment_for_agent_run", lambda **k: {})
    monkeypatch.setattr(submission.settings_store, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(submission.capabilities, "validate_compute_target", lambda _: {"ok": True})
    monkeypatch.setattr(submission.agent_runs, "resolve_agent_provider_config", lambda **k: {})
    monkeypatch.setattr(submission.context_store, "build_agent_context_binding", lambda *a, **k: {})
    monkeypatch.setattr(
        submission, "research_pipeline_workspace",
        lambda: SimpleNamespace(project_root=lambda _: tmp_path),
    )
    monkeypatch.setattr(submission, "list_bound_run_history", lambda **k: [])
    monkeypatch.setattr(submission, "resumable_planner_checkpoint_job_id", lambda **k: "")
    monkeypatch.setattr(submission.context_store, "handoff_context", lambda *a, **k: {"revision": 1})
    monkeypatch.setattr(submission.capabilities, "record_tool_event", lambda kind, detail: None)
    monkeypatch.setattr(
        submission.agent_pipeline_runs, "make_research_pipeline_run_runner",
        lambda **kwargs: (lambda _job: {}),
    )
    monkeypatch.setattr(
        submission, "_submit_job",
        lambda *args: SimpleNamespace(id="job-clock", kind="agent-run", status="queued"),
    )

    def run(**changes: Any) -> Any:
        return submission.submit_research_run(
            submission.ResearchRunSubmissionRequest(
                study_context_id=study["id"],
                provider="mock",
                credential_source="pi_verified",
                external_llm_opt_in=True,
                intent="candidate_plan",
                planner_start_mode="fresh",
                **changes,
            ),
            authorize=lambda: None,
        )

    return SimpleNamespace(run=run, study=study)


def _launch_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.name == submission.__name__ and "research run launch" in record.getMessage()
    ]


def test_a_started_job_logs_each_steps_seconds(launch, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=submission.__name__):
        receipt = launch.run()

    assert receipt.job_id == "job-clock"
    (line,) = _launch_lines(caplog)
    pattern = "research run launch job-clock: " + " ".join(rf"{step}=\d+\.\ds" for step in _STEPS)
    assert re.fullmatch(pattern, line), line
    # Seconds and the job id only: nothing about the study or its data.
    assert launch.study["data_source"]["path"] not in line
    assert "private-export" not in line and launch.study["id"] not in line


def test_a_refused_launch_logs_no_step_line(launch, monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        submission.dataio, "describe_export_source", lambda _: {"ok": False, "error": "no_export_files"},
    )
    with caplog.at_level(logging.INFO, logger=submission.__name__):
        with pytest.raises(submission.ResearchRunSubmissionError) as refused:
            launch.run()

    assert refused.value.code == "no_export_files"
    assert _launch_lines(caplog) == []
