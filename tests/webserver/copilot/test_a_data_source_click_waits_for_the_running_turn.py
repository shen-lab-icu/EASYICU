"""A data-source click made while a turn runs waits for the turn, then runs.

The page used to drop the click silently while the assistant was still
thinking, so "选择 SICdb 数据" looked dead. The click is now remembered for
its conversation and sent once the turn ends; a conversation switch or a
source the turn already bound drops it.
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
const [jsDir, lang, input] = process.argv.slice(1, 4);
const listeners = {};
global.window = global;
window.EU_LANG = lang;
global.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init && init.detail; } };
global.document = {
  addEventListener: (type, handler) => (listeners[type] = listeners[type] || []).push(handler),
  dispatchEvent: event => (listeners[event.type] || []).forEach(handler => handler(event)),
};
const registry = {};
window.EasyICU = { guidedPi: {
  declare: (name, api) => { registry[name] = api; },
  optional: name => registry[name] || null,
  require: name => { if (!registry[name]) throw new Error('missing owner ' + name); return registry[name]; },
} };
require(path.join(jsDir, 'html-escape.js'));
for (const file of ['screens-guided-pi-data-consent.js', 'screens-guided-pi-error-text.js', 'screens-guided-pi-data-binding.js']) {
  require(path.join(jsDir, file));
}
const tr = (en, zh) => (lang === 'zh' ? zh : en) || en;
const INPUT = JSON.parse(input);
const out = value => process.stdout.write(JSON.stringify(value === undefined ? null : value));
const errorText = registry.errorText.create({ tr, staticPreview: false }).errorText;
const calls = [];
const errors = [];
let session = { session_id: 'session-1', binding: { study_context_id: 'study-1' },
  data_source_authorization: { status: INPUT.status || 'selection_in_progress' } };
let mounted = true;
let busy = Boolean(INPUT.busy);
let renders = 0;
const binding = registry.dataBinding.create({
  api: () => ({ authorizePiCopilotDataSource: async (sessionId, body) => {
    calls.push([sessionId, body]);
    if (INPUT.refusal) { const error = new Error('refused'); error.code = INPUT.refusal; throw error; }
    return { session: { ...session, data_source_authorization: { status: 'selection_in_progress' } }, resource: null };
  } }),
  render() { renders += 1; }, projectId: () => 'project-1', loadWorkflow: async () => {}, dataConsent: registry.dataConsent,
  errorText, rememberSession() {}, continueAfterDataSourceConfirmation: async () => false,
  session: () => session, busy: () => busy, root: () => (mounted ? {} : null),
  setError(value) { if (value) errors.push(value); }, setSession(value) { session = value; },
});
const settle = () => new Promise(resolve => setTimeout(resolve, 0));
(async () => {
__SCENARIO__
})().catch(error => { console.error(error && error.stack || String(error)); process.exit(1); });
"""

def _node(scenario: str, payload: Any = None, *, lang: str = "zh") -> Any:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(
        node,
        _HARNESS.replace("__SCENARIO__", scenario),
        str(JS),
        lang,
        json.dumps(payload or {}),
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


_CLICK_WHILE_BUSY = """
await binding.authorizeDataSource('begin_local_selection', { database: 'sicdb' });
const waitingWhileBusy = Boolean(binding.waitingAction());
const callsWhileBusy = calls.length;
binding.resumeWaiting();
await settle();
const callsStillBusy = calls.length;
if (INPUT.switchTo) session = { ...session, session_id: INPUT.switchTo };
if (INPUT.boundTo) session = { ...session, data_source_authorization: { status: INPUT.boundTo } };
busy = false;
binding.resumeWaiting();
await settle(); await settle();
binding.resumeWaiting();
await settle();
out({ waitingWhileBusy, callsWhileBusy, callsStillBusy, calls, renders, errors, waitingAfter: Boolean(binding.waitingAction()) });
"""


def test_a_click_while_the_turn_runs_is_sent_once_the_turn_ends() -> None:
    shown = _node(_CLICK_WHILE_BUSY, {"busy": True, "status": "pending"})

    assert shown["waitingWhileBusy"] is True
    assert shown["callsWhileBusy"] == 0
    assert shown["callsStillBusy"] == 0
    assert shown["renders"] >= 1
    assert shown["calls"] == [
        ["session-1", {"project_id": "project-1", "action": "begin_local_selection", "database": "sicdb"}]
    ]
    assert shown["waitingAfter"] is False
    assert shown["errors"] == []


def test_a_click_without_a_running_turn_is_sent_at_once() -> None:
    shown = _node(_CLICK_WHILE_BUSY, {"busy": False, "status": "pending"})

    assert shown["waitingWhileBusy"] is False
    assert shown["callsWhileBusy"] == 1
    assert len(shown["calls"]) == 1


def test_a_waiting_click_does_not_follow_the_researcher_to_another_conversation() -> None:
    shown = _node(_CLICK_WHILE_BUSY, {"busy": True, "status": "pending", "switchTo": "session-2"})

    assert shown["calls"] == []
    assert shown["waitingAfter"] is False


def test_a_waiting_click_is_dropped_when_the_turn_already_bound_a_source() -> None:
    shown = _node(_CLICK_WHILE_BUSY, {"busy": True, "status": "pending", "boundTo": "confirmed"})

    assert shown["calls"] == []
    assert shown["waitingAfter"] is False
