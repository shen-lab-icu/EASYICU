"""A bare 「继续」 claims the message only when the plan revision can start.

After a review asks for planner-owned changes, a bare continuation starts the
revision without a conversation turn.  When the revision cannot start (the
conversation's data source is not confirmed yet, the last run failed, or the
revision already started), the message must reach the conversation once as an
ordinary message: it used to be appended locally first and then sent again,
so the researcher saw it twice and the conversation answered a request the
host had silently declined.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)


def exercise(case: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    script = r"""
      global.window = {};
      eval(__ACTIONS_SOURCE__);
      const calls = [];
      let busy = false;
      const archived = CASE === 'last-run-failed'
        ? [{kind: 'agent-run', status: 'failed', created_at_epoch: 2}]
        : [{kind: 'agent-run', status: 'completed', created_at_epoch: 1}];
      const host = {
        tr: (en, zh) => zh, errorText: error => String(error && error.message || error),
        regeneration: {}, nextActions: {}, replay: {},
        projectId: () => 'p1', turnGrants: () => [],
        session: () => ({
          session_id: 's1', archived_child_jobs: archived,
          binding: {run_id: 'reviewed-run', study_context_id: 'study1'},
          research_provider: {provider: 'openai'},
        }),
        workflow: () => ({
          next_action_code: 'plan_scientific_changes_required',
          plan_review_summary: {
            authorization_questions: [], automatic_revision_blockers: [],
            remediation_buckets: {agent_plan_revision: ['PLANNER_OWNED_REPAIR']},
          },
        }),
        busy: () => busy, sessionIsStale: () => false,
        researchSourceReady: () => CASE !== 'source-unconfirmed',
        api: () => ({
          loadStudyContext: async () => ({context: {question: 'q', data_source: {path: '/prepared'}}}),
          startAgentRun: async body => {
            calls.push(['plan', body.plan_revision_source_run_id]);
            if (CASE === 'start-fails') throw new Error('provider_unavailable');
            return {job_id: 'repair-job'};
          },
        }),
        render: () => {}, recordHostAction: async () => {},
        watchChildJob: (...args) => calls.push(['child', ...args]),
        refreshSession: async () => {}, loadWorkflow: async () => {},
        setBusy: value => {busy = value;},
        setError: value => { if (value) calls.push(['error', value]); },
        setDraft: () => {},
        appendMessage: value => calls.push(['message', value.text]),
      };
      (async () => {
        const actions = window.EU_GUIDED_PI_PLAN_ACTIONS.create(host);
        const first = await actions.continueUserRequestedSystemProgression('继续');
        // A failed start is retried by the next message, so it is sent once.
        const second = CASE === 'start-fails'
          ? null
          : await actions.continueUserRequestedSystemProgression('继续');
        process.stdout.write(JSON.stringify({first, second, calls}));
      })().catch(error => { console.error(error); process.exit(1); });
    """
    script = script.replace(
        "__ACTIONS_SOURCE__", json.dumps(_read("js/screens-guided-pi-plan-actions.js"))
    )
    script = f"const CASE = {json.dumps(case)};\n" + script
    completed = run_node(node, script, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout)


def _messages(calls: list) -> list:
    return [call for call in calls if call[0] == "message"]


def _plans(calls: list) -> list:
    return [call for call in calls if call[0] == "plan"]


@pytest.mark.parametrize("case", ["source-unconfirmed", "last-run-failed"])
def test_a_revision_that_cannot_start_leaves_the_message_to_the_conversation(case):
    result = exercise(case)

    assert (result["first"], result["second"]) == (False, False)
    assert _messages(result["calls"]) == []
    assert _plans(result["calls"]) == []


def test_a_started_revision_claims_one_message_and_the_next_goes_to_the_conversation():
    result = exercise("ready")

    assert (result["first"], result["second"]) == (True, False)
    assert _messages(result["calls"]) == [["message", "继续"]]
    assert _plans(result["calls"]) == [["plan", "reviewed-run"]]
    assert ["child", "repair-job", "easyicu_full_run_submitted"] in result["calls"]


def test_a_claimed_message_whose_start_fails_reports_an_error_not_a_copy():
    result = exercise("start-fails")

    assert result["first"] is True
    assert _messages(result["calls"]) == [["message", "继续"]]
    assert ["error", "provider_unavailable"] in result["calls"]
