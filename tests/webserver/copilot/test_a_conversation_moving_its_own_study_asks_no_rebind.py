"""A conversation that moves its own study binding does not ask the reader to rebind it.

Starting a plan from the conversation changes the study's active job, so the
session's binding reads as stale until the turn or the background job ends,
when the page rebinds by itself.  On the hosted pilot the page showed
「权威状态已变化 … 重新绑定当前状态」 for the whole of plan generation, an
internal state a clinician cannot act on.  The banner now waits until nothing
of this conversation is running; a change found at rest still asks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.support.node import run_node

JS = Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "static" / "js"

_HARNESS = r"""
const path = require('path');
const [jsDir, input] = process.argv.slice(1, 3);
global.window = global;
window.EU_LANG = 'zh';
const registry = {};
window.EasyICU = { guidedPi: {
  declare: (name, api) => { registry[name] = api; },
  optional: name => registry[name] || null,
  require: name => { if (!registry[name]) throw new Error('missing owner ' + name); return registry[name]; },
} };
require(path.join(jsDir, 'html-escape.js'));
require(path.join(jsDir, 'screens-guided-pi-session-view.js'));
const tr = (en, zh) => zh || en;
const esc = window.EU_HTML.esc;
const INPUT = JSON.parse(input);
const stub = (overrides = {}) => new Proxy(overrides, { get: (target, key) => (key in target ? target[key] : () => '') });
const job = { id: 'job-1', role: 'activity', childJobId: 'job-1', status: 'running' };
const state = {
  session: { session_id: 'session-1', binding: {}, data_source_authorization: { status: 'confirmed' } },
  messages: [{ id: 'user-1', role: 'user', text: 'q', complete: true }],
  workflowReceipts: INPUT.job ? [job] : [], busy: INPUT.busy, projectLoading: false,
  draft: '', regenerating: false, planStarting: null,
};
const view = registry.sessionView.create({
  state, modules: stub({ optional: () => null }), dataConsent: stub({ requiresConfirmation: () => false }),
  starters: null, ideaSource: null, header: stub(), regeneration: null,
  studyWorkspace: stub({ messageView: () => ({}) }),
  activity: stub({ renderTimeline: () => '<article data-row></article>', durationText: () => '5 秒' }),
  runFiles: stub({ timeline: rows => rows }), resourceOwner: stub(), messageActions: stub(),
  transcript: stub({ latestTurnCompletedIdeaExploration: () => false }),
  runOutcome: stub({ collection: () => [] }), aside: stub({ layoutOptions: () => ({}) }),
  cohortEligibility: stub(), tr, esc, iconHtml: () => '', projectId: () => 'project-1',
  publicAssistantText: text => text, assistantTextHtml: text => text, sessionIsStale: () => true,
  agentMode: () => 'research', accessModeLabel: () => '', projectTitle: () => 'Project',
  navigationSessionTitle: () => 'Session', workflowConfirmationHtml: () => '',
  hostJobs: stub(), followUps: stub(), effortMenu: null,
});
const html = view.sessionPanel();
process.stdout.write(JSON.stringify({
  banner: html.includes('data-gpi-rebind'),
  // Nothing can be sent while the binding is stale, banner or not.
  sendEnabled: /<button[^>]*data-gpi-send(?![^>]*disabled)[^>]*>/.test(html),
}));
"""


def _shown(**payload: Any) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(node, _HARNESS, str(JS), json.dumps(payload), check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "running", [{"busy": True, "job": False}, {"busy": False, "job": True}], ids=["turn", "job"]
)
def test_while_its_own_turn_or_job_runs_the_page_asks_no_rebind(running: dict) -> None:
    shown = _shown(**running)

    assert shown == {"banner": False, "sendEnabled": False}


def test_a_change_found_at_rest_still_asks_for_a_rebind() -> None:
    shown = _shown(busy=False, job=False)

    assert shown == {"banner": True, "sendEnabled": False}
