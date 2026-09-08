import json
import shutil
import subprocess

import pytest

from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)


@pytest.mark.parametrize("report_only", [True, False])
def test_explicit_click_sends_report_only_scope_without_plan_or_approval(report_only):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable")
    script = "global.window = {};\n" + _read("js/screens-guided-pi-replay.js") + "\n"
    script += """
      const calls = [];
      const replay = window.EasyICU.guidedPi.require('replay');
      (async () => {
        await replay.retryFailedExecution({
          reportOnly: REPORT_ONLY,
          session: {binding: {run_id: 'original', study_context_id: 'study'},
            research_provider: {provider: 'openai', credential_source: 'pi_verified'}},
          api: {
            loadStudyContext: async () => ({context: {question: 'Question', data_source: {path: '/prepared/miiv'}}}),
            startAgentRun: async request => {calls.push(request); return {job_id: 'repair'};},
          },
        });
        process.stdout.write(JSON.stringify(calls));
      })().catch(error => { console.error(error); process.exit(1); });
    """.replace("REPORT_ONLY", json.dumps(report_only))
    result = subprocess.run([node, "--eval", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = json.loads(result.stdout)
    assert len(calls) == 1
    assert calls[0].get("report_only", False) is report_only
    assert calls[0]["execution_resume_source_run_id"] == "original"
    assert "plan_revision_source_run_id" not in calls[0]
    assert "decision" not in calls[0] and "budget_mode" not in calls[0]


@pytest.mark.parametrize("reason,code,expected", [
    ("restore", "WRITER_ONLY_REGISTERED_INPUT_CHANGED", [True, False]),
    ("report_only", "WRITER_ONLY_REGISTERED_INPUT_CHANGED", [True]),
    ("restore", "WRITER_ONLY_STUDY_CHANGED", [True]),
    ("restore", "provider_auth_failed", [True]),
])
def test_general_restore_has_one_governed_fallback_but_explicit_report_scope_does_not(reason, code, expected):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = "global.window = {};\n" + _read("js/screens-guided-pi-plan-actions.js")
    script += """
      const calls = [], messages = [], errors = [];
      const host = {
        tr: (en,zh) => zh, errorText: e => e.code,
        busy: () => false, sessionIsStale: () => false, session: () => ({}),
        api: () => ({}), setBusy: () => {}, setError: e => errors.push(e),
        appendMessage: m => messages.push(m), render: () => {},
        recordHostAction: async () => {}, watchChildJob: () => {},
        replay: {retryFailedExecution: async request => {
          calls.push(request.reportOnly);
          // Fallback errors must surface, never start a third attempt.
          throw Object.assign(new Error('sealed input changed'), {code: ERROR_CODE});
        }},
      };
      (async () => {
        await window.EU_GUIDED_PI_PLAN_ACTIONS.create(host).retryFailedExecution(REASON);
        process.stdout.write(JSON.stringify({calls,messages,errors}));
      })().catch(e => {console.error(e);process.exit(1);});
    """.replace("REASON", json.dumps(reason)).replace("ERROR_CODE", json.dumps(code))
    result = subprocess.run([node, "--eval", script], capture_output=True, text=True, check=True)
    output = json.loads(result.stdout)
    assert output["calls"] == expected
    assert len([m for m in output["messages"] if m["role"] == "user"]) == 1
    assert output["errors"][-1] == code
    if len(expected) == 2:
        assert output["messages"][-1]["role"] == "assistant"
        assert "不重跑分析" not in output["messages"][0]["text"]


def test_duration_rounding_carries_into_minutes():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = "global.window = {};\n" + _read("js/screens-guided-pi-activity.js")
    script += """
      const owner = window.EU_GUIDED_PI_ACTIVITY.create({
        tr: (en,zh) => zh, esc: String, iconHtml: () => '',
        resourceName: () => '', resourceKey: () => '', resourceButton: () => '',
      });
      process.stdout.write(owner.render({role:'activity',status:'complete',
        startedAt:1000,endedAt:420900,steps:[]}));
    """
    result = subprocess.run([node, "--eval", script], capture_output=True, text=True, check=True)
    assert "7 分" in result.stdout and "60 秒" not in result.stdout


def test_real_outcome_click_preserves_restore_vs_explicit_report_only_scope():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = "global.window = {};\n" + _read("js/screens-guided-pi-events.js")
    script += """
      let click;
      const calls = [];
      const host = {querySelector:()=>null, addEventListener:(name,handler)=>{if(name==='click')click=handler;}};
      const owner = window.EasyICU.guidedPi.require('events').create({
        state:{host}, retryFailedExecution: reason => calls.push(reason),
      });
      owner.wire();
      for(const reason of ['restore','report_only','unknown']) {
        click({target:{closest:selector => selector==='[data-gpi-run-outcome-retry]'
          ? {dataset:{gpiRunOutcomeRetry:reason}} : null}});
      }
      process.stdout.write(JSON.stringify(calls));
    """
    result = subprocess.run([node, "--eval", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ["restore", "report_only", "validation_repair"]
