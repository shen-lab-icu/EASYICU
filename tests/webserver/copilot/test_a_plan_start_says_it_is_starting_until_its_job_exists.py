"""A plan start says it is starting until the host answers with its job.

The request that starts a candidate plan returns only after the host has
checked the study's data package, which can take a minute or two on a large
one.  Until then the page has no job to follow or stop.  The composer shows
that the plan is starting, with its elapsed time and no Stop button (a Stop
there would reach the conversation turn that already ended), and hands over to
the job's own row once the host answers; a refused or abandoned start leaves
nothing behind.
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
const [jsDir, lang, files, input] = process.argv.slice(1, 5);
global.window = global;
window.EU_LANG = lang;
const registry = {};
window.EasyICU = { guidedPi: {
  declare: (name, api) => { registry[name] = api; },
  optional: name => registry[name] || null,
  require: name => { if (!registry[name]) throw new Error('missing owner ' + name); return registry[name]; },
} };
require(path.join(jsDir, 'html-escape.js'));
for (const file of JSON.parse(files)) require(path.join(jsDir, file));
const tr = (en, zh) => (lang === 'zh' ? zh : en) || en;
const esc = window.EU_HTML.esc;
const INPUT = JSON.parse(input);
const out = value => process.stdout.write(JSON.stringify(value === undefined ? null : value));
const settle = () => new Promise(resolve => setTimeout(resolve, 0));
(async () => {
__SCENARIO__
})().catch(error => { console.error(error && error.stack || String(error)); process.exit(1); });
"""


def _node(scenario: str, files: tuple[str, ...], payload: Any = None, *, lang: str = "zh") -> Any:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(
        node,
        _HARNESS.replace("__SCENARIO__", scenario),
        str(JS),
        lang,
        json.dumps(list(files)),
        json.dumps(payload or {}),
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


# --- the plan start (plan-actions) -------------------------------------------------


_START = """
const seen = [];
let busy = false;
let planStarting = null;
let session = {
  session_id: 'session-1', archived_child_jobs: [],
  binding: { study_context_id: 'study-1', study_revision: 4, run_id: '' },
  research_provider: { provider: 'openai', credential_source: 'pi_verified' },
};
const workflow = { next_action_code: 'provider_ready_to_generate_plan',
  host_decisions: { plan_transition: { family: 'plan_transition', next_action_code: 'provider_ready_to_generate_plan' } } };
let answer = null;
const host = {
  tr, esc, iconHtml: () => '', resourceButton: () => '', errorText: error => 'error:' + String(error && error.message),
  regeneration: {}, nextActions: {}, replay: {}, session: () => session, workflow: () => workflow,
  busy: () => busy, latestRun: () => null, sessionIsStale: () => false, projectId: () => 'project-1',
  turnGrants: () => [], setBusy: value => { busy = value; },
  setPlanStarting: value => { planStarting = value || null; }, planStarting: () => planStarting,
  setError: value => { if (value) seen.push(['error', value]); },
  render: () => seen.push(['render', planStarting ? planStarting.sessionId : null, busy]),
  refreshSession: async () => {}, loadWorkflow: async () => {},
  watchChildJob: (...args) => seen.push(['watch', args[0], planStarting]),
  appendMessage: () => {}, sendText: async () => {}, recordHostAction: async () => null,
  api: () => ({
    loadStudyContext: async id => ({ context: { id, question: 'q', data_source: { path: '/prepared/eicu' } } }),
    startAgentRun: () => new Promise((resolve, reject) => { answer = { resolve, reject }; }),
  }),
};
const actions = registry.planActions.create(host);
const started = actions.startFormalPlanGeneration('provider_ready_to_generate_plan', { automatic: true });
await settle(); await settle();
const during = { planStarting, busy, sessionId: planStarting && planStarting.sessionId,
  hasStartedAt: Boolean(planStarting && Number(planStarting.startedAt) > 0) };
if (INPUT.switchSession) session = { ...session, session_id: 'session-2' };
if (INPUT.refuse) answer.reject(Object.assign(new Error('package_invalid'), { code: 'research_pipeline_manifest_invalid' }));
else answer.resolve({ job_id: 'plan-job' });
const result = await started;
out({ during, after: planStarting, busy, result, seen });
"""
_START_FILES = ("screens-guided-pi-plan-actions.js",)


def test_the_start_is_shown_until_the_host_answers_with_its_job() -> None:
    shown = _node(_START, _START_FILES)

    assert shown["during"]["sessionId"] == "session-1"
    assert shown["during"]["hasStartedAt"] is True
    assert shown["during"]["busy"] is True
    # The first render already shows the start, not a bare busy composer.
    first_render = next(row for row in shown["seen"] if row[0] == "render")
    assert first_render == ["render", "session-1", True]
    assert shown["result"] is True
    assert shown["after"] is None
    # The job's own row takes over: the start was cleared before it.
    assert ["watch", "plan-job", None] in shown["seen"]


def test_a_refused_start_leaves_no_start_behind() -> None:
    shown = _node(_START, _START_FILES, {"refuse": True})

    assert shown["during"]["sessionId"] == "session-1"
    assert shown["result"] is False
    assert shown["after"] is None
    assert shown["busy"] is False
    assert ["error", "error:package_invalid"] in shown["seen"]
    # The banner is drawn with the start already gone.
    assert shown["seen"][-1] == ["render", None, False]


def test_a_start_whose_conversation_was_left_leaves_no_start_behind() -> None:
    shown = _node(_START, _START_FILES, {"switchSession": True})

    assert shown["result"] is False
    assert shown["after"] is None
    assert not any(row[0] == "watch" for row in shown["seen"])


# --- the composer (session-view) ---------------------------------------------------


_COMPOSER = """
const stub = (overrides = {}) => new Proxy(overrides, { get: (target, key) => (key in target ? target[key] : () => '') });
const state = {
  session: { session_id: 'session-1', binding: {}, data_source_authorization: { status: 'confirmed' } },
  messages: [{ id: 'user-1', role: 'user', text: 'q', complete: true }],
  workflowReceipts: [], busy: true, projectLoading: false, draft: '', regenerating: false,
  planStarting: INPUT.planStarting,
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
  publicAssistantText: text => text, assistantTextHtml: text => text, sessionIsStale: () => false,
  agentMode: () => 'research', accessModeLabel: () => '', projectTitle: () => 'Project',
  navigationSessionTitle: () => 'Session', workflowConfirmationHtml: () => '',
  hostJobs: stub(), followUps: stub(), effortMenu: null,
});
const html = view.sessionPanel();
const box = (html.match(/<div class="gpi-compose-running"[^]*?<\\/div>/) || [''])[0];
out({
  starting: html.includes('data-gpi-plan-starting'),
  stop: html.includes('data-gpi-stop'),
  composerRunning: html.includes('gpi-compose-card is-running'),
  text: box.replace(/<[^>]+>/g, ' ').replace(/\\s+/g, ' ').trim(),
  elapsed: (box.match(/data-gpi-live-elapsed="(\\d+)"/) || [])[1] || null,
});
"""
_COMPOSER_FILES = ("screens-guided-pi-session-view.js",)


def test_the_composer_says_the_plan_is_starting_and_offers_no_stop() -> None:
    shown = _node(
        _COMPOSER, _COMPOSER_FILES,
        {"planStarting": {"sessionId": "session-1", "startedAt": 1_700_000_000_000}},
    )

    assert shown["starting"] is True
    assert shown["stop"] is False
    assert shown["composerRunning"] is True
    assert shown["text"] == (
        "正在启动研究计划生成 EasyICU 先核对所选数据包，再开始生成；数据较大时需要一两分钟。 5 秒"
    )
    assert shown["elapsed"] == "1700000000000"


@pytest.mark.parametrize(
    "starting",
    [None, {"sessionId": "session-other", "startedAt": 1_700_000_000_000}],
)
def test_a_busy_turn_without_this_conversations_start_keeps_its_stop(starting: Any) -> None:
    shown = _node(_COMPOSER, _COMPOSER_FILES, {"planStarting": starting})

    assert shown["starting"] is False
    assert shown["stop"] is True


def test_the_start_reads_in_english() -> None:
    shown = _node(
        _COMPOSER, _COMPOSER_FILES,
        {"planStarting": {"sessionId": "session-1", "startedAt": 1}}, lang="en",
    )

    assert shown["text"].startswith(
        "Starting the research plan EasyICU checks the selected data package first; "
        "a large one takes a minute or two."
    )
