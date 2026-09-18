"""An explicit retry preserves the stopped plan source, never its approval."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)


def exercise(case: str = "ready") -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    script = r"""
      global.window = {};
      eval(__CONFIRMATION_SOURCE__);
      eval(__ACTIONS_SOURCE__);
      const calls = [];
      let busy = false;
      const workflow = {
        next_action_code: 'agent_plan_revision_nonconvergent',
        plan_review_summary: {
          run_id: 'stopped-plan', authorization_questions: [],
          automatic_revision_blockers: [],
          remediation_buckets: {agent_plan_revision: ['ACCEPTED_BASELINE_CONTENT_MISSING']},
        },
      };
      const review = workflow.plan_review_summary;
      if (CASE === 'missing-run') delete review.run_id;
      if (CASE === 'no-planner-repair') review.remediation_buckets.agent_plan_revision = [];
      if (CASE === 'unknown-authority') delete review.authorization_questions;
      if (CASE === 'unknown-blockers') delete review.automatic_revision_blockers;
      if (CASE === 'authority-question') review.authorization_questions = [{code: 'source-review'}];
      if (CASE === 'external-blocker') review.automatic_revision_blockers = ['source-review'];
      if (CASE === 'stale-code') workflow.next_action_code = 'operator_plan_approval_required';
      const original = JSON.stringify(workflow);
      const host = {
        tr: (en, zh) => zh, esc: value => String(value), iconHtml: () => '',
        resourceButton: resource => `<button>${resource.label}</button>`,
        errorText: error => String(error), regeneration: {}, nextActions: {}, replay: {},
        session: () => ({
          session_id: 'session', archived_child_jobs: [],
          binding: {run_id: 'neighbour-job', study_context_id: 'study', study_revision: 7},
          research_provider: {provider: 'openai', credential_source: 'pi_verified'},
        }),
        workflow: () => workflow, busy: () => busy,
        sessionIsStale: () => CASE === 'stale-session',
        api: () => ({
          loadStudyContext: async id => ({context: {
            id, question: 'Unchanged question', data_source: {path: '/prepared/miiv'},
          }}),
          startAgentRun: async body => {calls.push(['plan', body]); return {job_id: 'new-job'};},
          submitAgentRunReview: async body => calls.push(['approval', body]),
        }),
        projectId: () => 'project', turnGrants: () => [],
        setBusy: value => {busy = value;}, setError: () => {}, render: () => {},
        recordHostAction: async (...args) => calls.push(['record', ...args]),
        watchChildJob: (...args) => calls.push(['watch', ...args]),
        appendMessage: message => calls.push(['message', message.text]),
        sendText: async (...args) => calls.push(['send', ...args]),
      };
      const confirmation = window.EU_GUIDED_PI_CONFIRMATION.create(host);
      const actions = window.EU_GUIDED_PI_PLAN_ACTIONS.create(host);
      (async () => {
        const spec = confirmation.workflowConfirmation();
        const html = confirmation.workflowConfirmationHtml();
        await actions.continueSystemOwnedPlanProgression();
        const bareContinuation = await actions.continueUserRequestedSystemProgression('请继续');
        await actions.startFormalPlanGeneration('agent_plan_revision_nonconvergent', {automatic: true});
        const automaticCalls = calls.slice();
        if (CASE === 'ready') {
          await actions.confirmWorkflow(spec);
          // The old card must not produce a second run before the new binding arrives.
          await actions.confirmWorkflow(spec);
          await actions.continueSystemOwnedPlanProgression();
        } else {
          await actions.startFormalPlanGeneration('agent_plan_revision_nonconvergent');
        }
        process.stdout.write(JSON.stringify({
          spec, html, calls, automaticCalls, bareContinuation,
          originalUnchanged: original === JSON.stringify(workflow),
        }));
      })().catch(error => {console.error(error); process.exit(1);});
    """
    script = script.replace("__CONFIRMATION_SOURCE__", json.dumps(_read("js/screens-guided-pi-confirmation.js")))
    script = script.replace("__ACTIONS_SOURCE__", json.dumps(_read("js/screens-guided-pi-plan-actions.js")))
    script = f"const CASE = {json.dumps(case)};\n" + script
    result = subprocess.run([node, "--eval", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def test_stopped_plan_has_explicit_replanning_but_no_analysis_approval() -> None:
    result = exercise()
    assert result["spec"]["nonApprovable"] is True
    assert result["spec"].get("retryPlanRevision") is True
    assert "修复后重新规划一次" in result["html"]
    assert "批准并开始分析" not in result["html"]
    assert "打开完整计划" in result["html"]
    assert "查看审阅详情" in result["html"]


def test_only_explicit_click_starts_one_source_bound_revision_not_a_fresh_catalogue() -> None:
    result = exercise()
    assert result["automaticCalls"] == []
    assert result["bareContinuation"] is False
    plans = [call[1] for call in result["calls"] if call[0] == "plan"]
    assert len(plans) == 1
    assert plans[0]["plan_revision_source_run_id"] == "stopped-plan"
    assert plans[0]["planner_start_mode"] == "auto"
    assert plans[0]["literature_search_authorized"] is True
    assert not any(call[0] in {"approval", "send"} for call in result["calls"])
    assert result["originalUnchanged"] is True


@pytest.mark.parametrize("case", [
    "missing-run", "no-planner-repair", "unknown-authority", "unknown-blockers", "authority-question",
    "external-blocker", "stale-code", "stale-session",
])
def test_manual_retry_cannot_bypass_an_unresolved_or_stale_source(case: str) -> None:
    result = exercise(case)
    assert result["calls"] == []
    assert "修复后重新规划一次" not in result["html"]
    assert result["originalUnchanged"] is True
