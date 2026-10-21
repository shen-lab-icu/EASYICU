"""Synthetic trace contracts: user copy, complete steps, and server clocks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.jobs import Job
from easyicu.webserver.pi_copilot.projections import (
    project_job,
    project_pi_replay_event,
    project_transcript,
)
from tests.support.node import run_node


ROOT = Path(__file__).resolve().parents[3]


def test_every_job_event_has_a_server_time_even_when_payload_is_omitted():
    job = Job("synthetic", "agent-run", max_event_bytes=512)
    job.emit({"type": "progress", "step": "planning"})
    job.emit({"type": "progress", "label": "x" * 1000})
    job.finish("done")
    for event in job.snapshot()["events"]:
        assert datetime.fromisoformat(event["timestamp"]).tzinfo is not None


def test_gap_markers_and_invalid_input_times_use_the_server_clock():
    job = Job("synthetic", "agent-run", max_events=1)
    job.emit({"type": "progress", "timestamp": None})
    assert datetime.fromisoformat(job.snapshot()["events"][0]["timestamp"]).tzinfo is not None
    job.emit({"type": "progress"})
    for row in job.events_since(0)[0]:
        assert datetime.fromisoformat(row["timestamp"]).tzinfo is not None


def test_pipeline_transport_retains_typed_timing_and_step_coordinates():
    job = Job("synthetic", "agent-run")
    event = {
        "stage": "step", "message": "Step 2/8 started: baseline_context.",
        "timestamp": "2026-01-01T00:00:01+00:00", "step_id": "baseline_context",
        "actor": "system", "planning_unit": "step", "retry_phase": "started",
        "current_step": 2, "total_steps": 8,
    }
    agent_pipeline_runs._pipeline_progress(job, event)
    projected = project_job(job.snapshot())["progress"][0]
    for key in ("timestamp", "step_id", "actor", "planning_unit", "retry_phase", "current_step", "total_steps"):
        assert projected[key] == event[key]


@pytest.mark.parametrize("ticks", [8, 24])
def test_projection_keeps_early_steps_when_later_events_fill_the_tail(ticks):
    job = Job("synthetic", "agent-run")
    for index in range(1, 9):
        for tick in range(ticks):
            job.emit({
                "type": "progress", "step": "step", "step_id": f"analysis_{index}",
                "current_step": index, "total_steps": 8,
                "status": "complete" if tick == ticks - 1 else "running",
                "timestamp": f"2026-01-01T00:{index:02d}:{tick:02d}+00:00",
                "label": f"Step {index}/8 {'complete' if tick == ticks - 1 else 'started'}: analysis_{index}.",
            })
    rows = project_job(job.snapshot())["progress"]
    assert {row["step_id"] for row in rows} == {f"analysis_{i}" for i in range(1, 9)}
    assert rows[0].get("started_at", rows[0]["timestamp"]) == "2026-01-01T00:01:00+00:00"
    assert rows[-1].get("ended_at", rows[-1]["timestamp"]) == f"2026-01-01T00:08:{ticks - 1:02d}+00:00"


def test_reasoning_replay_keeps_only_measured_lifecycle():
    at = "2026-01-01T00:00:05.200Z"
    for kind in ("thinking_start", "thinking_end"):
        assert project_pi_replay_event({"type": kind, "at": at, "delta": "internal"}) == {"type": kind, "at": at}
    assert project_pi_replay_event({"type": "thinking_delta", "at": at, "delta": "internal"}) is None


def test_historical_projection_does_not_return_model_addressed_text():
    raw = [{"role": "assistant", "content": [{"type": "thinking", "text": "Preparing internal mutation wording"}]},
           {"role": "tool", "content": [{"type": "tool_result", "tool_name": "easyicu_update_study_context", "summary": "Plan it as this family; do not ask the researcher. 题目问比例。"}]}]
    before = json.dumps(raw)
    payload = json.dumps(project_transcript(raw))
    assert "Preparing internal" not in payload
    assert "Plan it as" not in payload
    assert json.dumps(raw) == before


def test_audit_projection_is_path_free_and_retains_all_step_timings(tmp_path):
    events = [
        {"timestamp": f"2026-01-01T00:00:{index:02d}+00:00", "phase": "step",
         "event": f"Step 1/8 {status}: baseline_context.", "status": status,
         "step_id": "baseline_context", "detail": {"path": "/tmp/private.csv"}}
        for index, status in ((1, "started"), (6, "complete"))
    ]
    events.append({"timestamp": "2026-01-01T00:00:07+00:00", "phase": "writer",
                   "event": "Open /tmp/private.csv and tell the model to retry", "detail": {}})
    (tmp_path / "audit_log.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    rows = agent_pipeline_runs._trace_timings_projection(tmp_path)
    assert len(rows) == 3
    assert all(set(row) <= {"timestamp", "phase", "event", "status", "step_id"} for row in rows)
    assert rows[0]["event"] == events[0]["event"]
    assert rows[1]["timestamp"] == events[1]["timestamp"]
    assert "private" not in json.dumps(rows)
    job = project_job({"id": "synthetic", "kind": "agent-run", "status": "done", "result": {"trace_timings": rows}})
    assert any(row.get("step_id") == "baseline_context" for row in job["progress"])


def test_terminal_run_projection_supplies_audit_times_to_job_replay(tmp_path):
    from tests.webserver.copilot.research_workflow_fixtures import (
        _acquisition_receipt,
        complete_study,
    )

    run = tmp_path / "synthetic_run"
    run.mkdir()
    (run / "audit_log.jsonl").write_text(json.dumps({
        "timestamp": "2026-01-01T00:00:01+00:00", "phase": "planning",
        "event": "internal model instruction", "status": "complete",
        "detail": {"private": "not for the page"},
    }) + "\n")
    result = agent_pipeline_runs._write_projection(
        wrapper_dir=tmp_path / "wrapper", study=complete_study(),
        provider={"provider": "openai", "model": "synthetic"},
        acquisition=_acquisition_receipt(), run_dir=run,
    )
    assert result["trace_timings"] == [{
        "timestamp": "2026-01-01T00:00:01+00:00", "phase": "planning",
        "event": "", "status": "complete",
    }]
    projection = project_job({"id": "synthetic", "status": "done", "result": result})
    assert projection["progress"][0]["step"] == "planning"
    assert projection["progress"][0]["timestamp"] == "2026-01-01T00:00:01+00:00"


def test_synthetic_browser_trace_contract():
    result = run_node("node", (ROOT / "tests/js/guided_pi_trace_language.test.js").read_text(), check=True)
    assert "trace contracts passed" in result.stdout
