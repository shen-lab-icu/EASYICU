"""A researcher can explore EasyICU before connecting a model.

The connection page offers to explore first; the workspace then opens without
a model, and only a step that needs one (starting a research conversation,
sending a message) asks for the connection in a dialog and resumes after it.
Provider authority stays on the server: deferring the connection grants
nothing, it only moves where the connection is asked for.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    STATIC as STATIC,
    _ESCAPE_OWNER as _ESCAPE_OWNER,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read as _read,
)


_GATE_SCRIPT = r"""
const store = {};
const storage = process.argv[2] === 'blocked'
  ? { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } }
  : { getItem: key => (key in store ? store[key] : null), setItem: (key, value) => { store[key] = String(value); } };
global.window = { EU_LANG: 'zh', localStorage: storage };
global.document = { activeElement: null };
require(process.argv[1]);
const state = { showSetup: false, setupSaving: false, codexBusy: false, error: 'old' };
let ready = false, configured = false, renders = 0, finished = 0;
const resumed = [];
const gate = window.EasyICU.guidedPi.require('connect').create({
  state,
  tr: (en, zh) => zh,
  esc: value => String(value),
  render: () => { renders += 1; },
  connectionReady: () => ready,
  connectionConfigured: () => configured,
  finishProviderSetup: async () => { finished += 1; ready = configured; },
  body: () => '<form data-gpi-provider-form></form>',
});
const out = {};
out.firstVisitPage = gate.pageRequired();
gate.deferConnection();
out.afterDefer = { page: gate.pageRequired(), stored: store['easyicu.copilot.connectionDeferred'] || null, showSetup: state.showSetup };
state.showSetup = true;
out.settingsReopenPage = gate.pageRequired();
state.showSetup = false;

out.askWhileOffline = gate.ask('start_conversation', () => resumed.push('conversation'));
const offlineDialog = gate.dialogHtml();
out.offlineDialog = {
  modal: offlineDialog.includes('aria-modal="true"'),
  title: offlineDialog.includes('连接模型后继续'),
  reason: offlineDialog.includes('研究对话需要模型'),
  form: offlineDialog.includes('data-gpi-provider-form'),
  notNow: offlineDialog.includes('暂不连接'),
  continueOffered: offlineDialog.includes('data-gpi-connect-continue'),
};

state.setupSaving = true;
gate.dismiss();
out.dismissWhileSaving = Boolean(state.connectPrompt);
state.setupSaving = false;

(async () => {
  await gate.proceed();
  out.proceedBeforeConnected = { finished, resumed: resumed.length, open: Boolean(state.connectPrompt) };
  configured = true;
  out.connectedDialogContinues = gate.dialogHtml().includes('data-gpi-connect-continue');
  await gate.proceed();
  out.afterProceed = { finished, resumed: resumed.slice(), open: Boolean(state.connectPrompt) };

  out.askWhenReady = gate.ask('send_message', () => resumed.push('message'));
  out.readyLeavesNoDialog = gate.dialogHtml() === '';

  ready = false; configured = false;
  gate.ask('send_message');
  let prevented = false;
  out.escapeHandled = gate.handleKeydown({ key: 'Escape', preventDefault() { prevented = true; } });
  out.escape = { prevented, open: Boolean(state.connectPrompt) };
  out.otherKeyHandled = gate.handleKeydown({ key: 'Enter', preventDefault() {} });

  gate.ask('unknown_reason');
  out.unknownReasonIsManual = gate.dialogHtml().includes('浏览项目、数据和技能不需要模型');
  process.stdout.write(JSON.stringify(out));
})();
"""


def _run_gate(mode: str = "storage") -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    completed = run_node(
        node, _GATE_SCRIPT, str(STATIC / "js" / "screens-guided-pi-connect.js"), mode, check=True
    )
    return json.loads(completed.stdout)


@pytest.mark.requires_node
def test_exploring_first_defers_the_page_and_a_model_step_asks_then_resumes() -> None:
    out = _run_gate()

    assert out["firstVisitPage"] is True
    assert out["afterDefer"] == {"page": False, "stored": "1", "showSetup": False}
    # Opening connection settings still shows the full page.
    assert out["settingsReopenPage"] is True

    assert out["askWhileOffline"] is False
    assert out["offlineDialog"] == {
        "modal": True, "title": True, "reason": True, "form": True,
        "notNow": True, "continueOffered": False,
    }
    assert out["dismissWhileSaving"] is True
    # Continue does nothing until a connection is configured, then resumes once.
    assert out["proceedBeforeConnected"] == {"finished": 0, "resumed": 0, "open": True}
    assert out["connectedDialogContinues"] is True
    assert out["afterProceed"] == {"finished": 1, "resumed": ["conversation"], "open": False}
    # A ready connection never opens the dialog.
    assert out["askWhenReady"] is True
    assert out["readyLeavesNoDialog"] is True
    assert out["escapeHandled"] is True
    assert out["escape"] == {"prevented": True, "open": False}
    assert out["otherKeyHandled"] is False
    assert out["unknownReasonIsManual"] is True


@pytest.mark.requires_node
def test_blocked_browser_storage_falls_back_to_the_connection_page() -> None:
    out = _run_gate("blocked")

    assert out["firstVisitPage"] is True
    # Deferring still works for this visit; it just is not remembered.
    assert out["afterDefer"] == {"page": False, "stored": None, "showSetup": False}


def test_the_screen_routes_model_steps_through_the_connection_gate() -> None:
    owner = _read("js/screens-guided-pi.js")
    events = _read("js/screens-guided-pi-events.js")
    provider = _read("js/screens-guided-pi-provider.js")

    create_session = owner.split("async function createSession()", 1)[1].split("state.creating = true", 1)[0]
    assert "CONNECT.ask('start_conversation', createSession)" in create_session
    assert "state.showSetup = true" not in create_session
    send_text = owner.split("async function sendText(", 1)[1].split("text = String(text || '').trim();", 1)[0]
    assert "CONNECT.ask('send_message'" in send_text
    load_status = owner.split("async function loadStatus()", 1)[1].split("async function createSession()", 1)[0]
    assert "if (!connectionReady() && !state.connectionDeferred) state.showSetup = true;" in load_status
    # The dialog sits over the workspace, never over the connection page.
    assert "(setupFocused ? '' : CONNECT.dialogHtml())" in owner
    assert "CONNECT.focusDialog(state.host)" in owner

    click = events.split("state.host.addEventListener('click', event => {", 1)[1]
    assert click.lstrip().startswith("if (CONNECT && CONNECT.handleClick(event)) return;")
    assert "if (CONNECT && CONNECT.handleKeydown(event)) return;" in events

    setup = provider.split("function renderSetup(options)", 1)[1].split("function renderBindingSummary", 1)[0]
    assert "data-gpi-connect-later" in setup
    assert "先进去看看，用到模型时再连接" in setup
    assert "data-gpi-demo" in setup
    assert "renderConnectionBody(options)" in setup
    summary = provider.split("function renderBindingSummary(options)", 1)[1]
    assert "if (!options.connectionReady)" in summary
    assert "data-gpi-connect-open" in summary
    assert "未连接模型" in summary
    assert "guidedPi.declare('provider', { renderSetup, renderConnectionBody, renderBindingSummary })" in provider
    # The protocol note speaks about a saved connection, never before one exists.
    assert "config.connection_verified === true && transport !== 'openai-completions'" in provider
    assert "!options.apiResearchReady ? `<div class=\"gpi-config-note warn\">" not in provider


def test_the_connection_dialog_has_its_own_stylesheet_owner() -> None:
    sheet = _read("css/guided-pi-connect.css")
    base = _read("css/guided-pi.css")

    assert sheet.startswith("/* Owner: screens-guided-pi-connect.js")
    assert ".gpi-connect-backdrop{" in sheet and "position:fixed" in sheet
    assert ".gpi-provider-shell.gpi-connect-dialog{" in sheet
    assert "@media (max-width:560px)" in sheet
    # One set of provider tokens serves the page and the dialog.
    assert ".gpi-provider-setup,.gpi-connect-backdrop{" in base
    assert "--provider-ink:" not in sheet
