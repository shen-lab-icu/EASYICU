"""Timeout copy uses counted host actions, never provider messages."""

import json
from pathlib import Path

import pytest

from easyicu.webserver import agent_pipeline_runs, stop_next_actions
from easyicu.webserver.pi_copilot.projections import gate_detail_projection, project_run_outcome
from tests.support.node import run_node


def test_timeout_gate_and_conversation_preserve_the_actual_retry_count(tmp_path):
    error = TimeoutError("private synthetic provider text")
    error.easyicu_structured_attempt_metadata = [
        {"attempt": 1, "error_class": "timeout", "raw_chars": 0, "transport_attempts": 2},
    ]
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={}, exc=error,
        code="research_pipeline_provider_timeout", execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate["detail"]["transport_retries"] == 1
    assert gate_detail_projection(gate["detail"])["gate_transport_retries"] == 1
    outcome = project_run_outcome({"ok": True, "run_id": "synthetic", "gate": gate})
    assert outcome["gate_transport_retries"] == 1
    assert "private" not in json.dumps(gate)
    assert gate["next_actions"][0]["kind"] == "retry"


@pytest.mark.parametrize("count", [True, -1, 1000, "1", {}, None])
def test_invalid_retry_counts_are_never_public_coordinates(count):
    assert "transport_retries" not in stop_next_actions.safe_stop_coordinates({"transport_retries": count})


@pytest.mark.parametrize("count", [0, 1, 2])
def test_timeout_stop_names_provider_and_detail_keeps_the_retry_count(count):
    script = r"""
const modules = {};
global.window = { EU_HTML: {esc: x => x}, EasyICU: {guidedPi: {declare: (name, api) => {modules[name] = api;}}}};
require(process.argv[1]);
require(process.argv[3]);
const owner = modules.errorText.create({tr: (en, zh) => zh, staticPreview: false});
const stop = modules.runStop.create({tr: (en, zh) => zh, runFailureText: owner.runFailureText});
console.log(JSON.stringify([stop.render({savedRuns:[{
  run_id:'synthetic', run_status:'failed', error_code:'research_pipeline_provider_timeout',
  gate_transport_retries:Number(process.argv[2]),
}]}), owner.runFailureText('research_pipeline_provider_timeout', {transportRetries:Number(process.argv[2])})]));
"""
    path = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static/js/screens-guided-pi-error-text.js"
    result = run_node("node", script, str(path), str(count),
                      str(path.with_name("screens-guided-pi-run-stop.js")), check=True)
    top_line, detail = json.loads(result.stdout)
    assert "模型服务返回错误，本次运行已停止。网络或服务恢复后可直接重试。" in top_line
    assert "未及时返回" in detail
    assert f"重试 {count} 次" in detail
    assert "请稍后重试" not in detail


def test_recovery_text_names_the_actual_planning_restart():
    script = r"""
global.window = {EU_HTML: {esc: x => x}, EasyICU: {guidedPi: {declare: (n, a) => {global.api=a;}}}};
require(process.argv[1]);
const owner = api.create({tr: (en, zh) => zh});
console.log(JSON.stringify([
  owner.pipelineEventLabel({step:'provider_auto_resume',restart_stage:'metadata_planning'}),
  owner.pipelineEventLabel({step:'provider_recovery_wait'}),
]));
"""
    path = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static/js/screens-guided-pi-activity.js"
    restarted, waiting = json.loads(run_node("node", script, str(path), check=True).stdout)
    assert "重新生成候选计划" in restarted and "检查点" not in restarted
    assert "恢复后将自动继续" in waiting
