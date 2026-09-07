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
