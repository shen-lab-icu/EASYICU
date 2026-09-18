"""Preparation replay must use host input state, not a finished job or filename."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from easyicu.webserver.pi_copilot.projections import project_job
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)


def _replay(job: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    transcript = _read("js/screens-guided-pi-transcript.js")
    resources = _read("js/screens-guided-pi-resources.js")
    script = f"""
      global.window = {{
        EU_LANG: 'zh',
        EU_GUIDED_PI_REPLAY: {{ lifecycleTurns: session => session.replayTurns || [] }},
      }};
      eval({transcript!r});
      eval({resources!r});
      const owner = window.EU_GUIDED_PI_TRANSCRIPT.create({{
        tr: (en, zh) => zh,
        activity: {{ focusLatest: rows => rows }},
        upsertActivityStep: (activity, step) => activity.steps.push(step),
        timeMs: value => Date.parse(value),
        resourceKey: resource => JSON.stringify(resource || null),
        modelErrorText: code => code,
        workflowActionCode: () => 'failed_pipeline_requires_fresh_plan',
      }});
      const session = {{
        transcript: [],
        replayTurns: [{{
          job_id: 'prepare', kind: 'host_action', action_code: 'prepare_analysis_data',
          child_job_id: 'child', status: 'done',
          started_at: '2026-09-07T10:00:00Z', ended_at: '2026-09-07T10:01:00Z',
        }}],
        archived_child_jobs: [{json.dumps(job)}],
      }};
      const before = JSON.stringify(session);
      const rows = owner.transcriptMessages(session);
      const assistant = rows.find(row => row.role === 'assistant');
      const renderer = window.EU_GUIDED_PI_RESOURCES.create({{ esc: String }});
      process.stdout.write(JSON.stringify({{
        rows, assistant,
        html: renderer.renderForMessage(assistant),
        sourceUnchanged: before === JSON.stringify(session),
      }}));
    """
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


def _job(**overrides) -> dict:
    return {
        "job_id": "child",
        "kind": "agent-run",
        "status": "done",
        "gate_status": "blocked",
        "gate_reason_code": "data_foundation_blocked",
        "human_review_pending": False,
        "artifact_refs": [
            {"kind": "research_artifact", "run_id": "run_blocked", "artifact": name,
             "sha256": "c" * 64}
            for name in ("source_run_manifest.json", "run_context.json")
        ],
        "progress": [{"step": "data_foundation", "status": "failed"}],
        **overrides,
    }


def test_blocked_legacy_preparation_is_not_replayed_as_ready() -> None:
    payload = _replay(_job())
    assert "未准备完成" in payload["assistant"]["text"]
    assert "已准备完成" not in payload["assistant"]["text"]
    assert "data-gpi-run-outcome-data" not in payload["html"]
    activity = next(row for row in payload["rows"] if row["role"] == "activity")
    assert activity["status"] == "error"
    assert not any(row["step"] == "inputs" for row in activity["steps"])
    assert next(row for row in activity["steps"] if row["step"] == "data_foundation")["status"] == "error"
    assert "打开数据来源清单" in payload["html"]  # Diagnostics remain readable.
    assert not any(row["role"] == "user" for row in payload["rows"])
    assert payload["sourceUnchanged"] is True


@pytest.mark.parametrize("state", [None, "unavailable", "metadata_only", "prepared"])
def test_preparation_claim_needs_both_input_state_and_current_plan(state) -> None:
    job = _job(human_review_pending=True, gate_reason_code="operator_plan_approval_required")
    if state is not None:
        job["research_input_state"] = state
    payload = _replay(job)
    assert "已准备完成" not in payload["assistant"]["text"]
    assert ("data-gpi-run-outcome-data" in payload["html"]) is (state == "prepared")
    assert payload["sourceUnchanged"] is True


def test_prepared_input_and_plan_replay_ready_without_granting_execution() -> None:
    job = _job(
        human_review_pending=True,
        gate_reason_code="operator_plan_approval_required",
        research_input_state="prepared",
        progress=[{"step": "data_foundation", "status": "complete"}],
    )
    job["artifact_refs"].append({
        "kind": "research_artifact", "run_id": "run_ready",
        "artifact": "agent_plan.json", "sha256": "d" * 64,
    })
    payload = _replay(job)
    assert "已准备完成" in payload["assistant"]["text"]
    assert "分析尚未开始" in payload["assistant"]["text"]
    assert "data-gpi-run-outcome-data" in payload["html"]
    assert not any(row["role"] == "user" for row in payload["rows"])
    assert payload["sourceUnchanged"] is True


@pytest.mark.parametrize("state", [None, "unavailable", "metadata_only"])
def test_plan_artifact_alone_does_not_prove_data_preparation(state) -> None:
    job = _job(human_review_pending=True, research_input_state=state)
    job["artifact_refs"].append({
        "kind": "research_artifact", "run_id": "run_catalog",
        "artifact": "agent_plan.json", "sha256": "d" * 64,
    })
    payload = _replay(job)
    assert "已准备完成" not in payload["assistant"]["text"]
    assert "现有回执不足以确认" in payload["assistant"]["text"]
    assert "data-gpi-run-outcome-data" not in payload["html"]


@pytest.mark.parametrize("status", ["failed", "cancelled", "interrupted", "done"])
def test_failed_preparation_does_not_offer_workbench_even_with_prepared_input(status) -> None:
    payload = _replay(_job(status=status, research_input_state="prepared"))
    assert "已准备完成" not in payload["assistant"]["text"]
    assert "data-gpi-run-outcome-data" not in payload["html"]


@pytest.mark.parametrize("state,expected", [
    ("prepared", "prepared"), ("metadata_only", "metadata_only"),
    ("unavailable", "unavailable"), (None, "unavailable"),
    ("/private/patient-input", "unavailable"), ({"path": "/private/input"}, "unavailable"),
])
def test_project_job_carries_only_closed_input_state(state, expected) -> None:
    projected = project_job({
        "id": "child", "status": "done",
        "result": {"run_id": "run_input", "research_input_state": state},
    })
    assert projected["research_input_state"] == expected
    assert "/private" not in json.dumps(projected)
