"""A folder panel the researcher closes without a source ends its selection.

The host keeps a conversation in ``selection_in_progress`` while its folder
panel is open (see test_an_abandoned_folder_selection_does_not_hold_the_source_gate).
The page ends that selection when the researcher closes the panel, so the
conversation shows the source gate again instead of a card whose only way out
is the panel; the card itself also offers to leave.  A selection the host keeps
(a data task still owns it) stays: closing the panel then says nothing, while
the card's own button says why.
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
const binding = registry.dataBinding.create({
  api: () => ({ authorizePiCopilotDataSource: async (sessionId, body) => {
    calls.push([sessionId, body]);
    if (INPUT.refusal) { const error = new Error('refused'); error.code = INPUT.refusal; throw error; }
    return { session: { ...session, data_source_authorization: { status: 'pending', reason: 'project_source_confirmation_required' } }, resource: null };
  } }),
  render() {}, projectId: () => 'project-1', loadWorkflow: async () => {}, dataConsent: registry.dataConsent,
  errorText, rememberSession() {}, continueAfterDataSourceConfirmation: async () => false,
  session: () => session, busy: () => false, root: () => (mounted ? {} : null),
  setError(value) { if (value) errors.push(value); }, setSession(value) { session = value; },
});
const settle = () => new Promise(resolve => setTimeout(resolve, 0));
(async () => {
__SCENARIO__
})().catch(error => { console.error(error && error.stack || String(error)); process.exit(1); });
"""

_PANEL = {
    "kind": "native_workspace", "route": "extraction", "state": "setup",
    "study_context_id": "study-1", "study_revision": 3, "entry_mode": "source_binding",
}


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


_CLOSE = """
document.dispatchEvent(new CustomEvent('easyicu:guided-preview-closed', { detail: INPUT.detail }));
await settle();
out({ calls, errors, status: session.data_source_authorization.status });
"""


def test_closing_the_folder_panel_ends_the_selection() -> None:
    shown = _node(_CLOSE, {"detail": {"resource": _PANEL, "projectId": "project-1"}})

    assert shown["calls"] == [
        ["session-1", {"project_id": "project-1", "action": "cancel_local_selection"}]
    ]
    assert shown["status"] == "pending"
    assert shown["errors"] == []


@pytest.mark.parametrize(
    "payload",
    [
        # Another panel or file, not the folder selection's own.
        {"detail": {"resource": {**_PANEL, "entry_mode": ""}, "projectId": "project-1"}},
        {"detail": {"resource": {"kind": "research_artifact", "file": "agent_plan.json"},
                    "projectId": "project-1"}},
        # The panel of another project's conversation.
        {"detail": {"resource": _PANEL, "projectId": "project-2"}},
        # No selection is open: the panel was opened from a confirmed source.
        {"detail": {"resource": _PANEL, "projectId": "project-1"}, "status": "confirmed"},
        {"detail": {"resource": _PANEL, "projectId": "project-1"}, "status": "pending"},
        {"detail": {}},
    ],
)
def test_closing_anything_else_leaves_the_session_as_it_is(payload: dict[str, Any]) -> None:
    shown = _node(_CLOSE, payload)

    assert shown["calls"] == []


def test_a_page_that_is_not_open_does_not_end_the_selection() -> None:
    shown = _node(
        "mounted = false;\n" + _CLOSE,
        {"detail": {"resource": _PANEL, "projectId": "project-1"}},
    )

    assert shown["calls"] == []


def test_a_selection_the_host_keeps_stays_without_a_banner() -> None:
    shown = _node(
        _CLOSE,
        {"detail": {"resource": _PANEL, "projectId": "project-1"},
         "refusal": "pi_session_local_selection_job_active"},
    )

    assert len(shown["calls"]) == 1
    assert shown["status"] == "selection_in_progress"
    assert shown["errors"] == []


def test_the_open_selection_card_offers_to_leave_it() -> None:
    shown = _node(
        """
        const html = registry.dataConsent.render(session, { tr, esc: window.EU_HTML.esc, icon: () => '' });
        out({ actions: [...html.matchAll(/data-gpi-data-source-action="([^"]+)"[^>]*>([^<]+)</g)].map(m => [m[1], m[2]]) });
        """
    )

    assert shown["actions"] == [
        ["begin_local_selection", "返回本地目录选择"],
        ["cancel_local_selection", "放弃本地选择"],
    ]


@pytest.mark.parametrize(
    ("refusal", "zh", "en"),
    [
        ("pi_session_local_selection_job_active",
         "本研究的数据任务还在运行，等它结束后才能放弃本地目录选择。",
         "A data task of this study is still running. The folder selection can be left once it finishes."),
        ("pi_session_local_selection_not_started",
         "本地目录选择已经结束，请刷新这段对话。",
         "The local folder selection has already ended. Refresh this conversation."),
    ],
)
def test_the_cards_leave_button_says_why_the_host_refused(refusal: str, zh: str, en: str) -> None:
    scenario = """
    await binding.authorizeDataSource('cancel_local_selection');
    out({ calls, errors, status: session.data_source_authorization.status });
    """
    for lang, text in (("zh", zh), ("en", en)):
        shown = _node(scenario, {"refusal": refusal}, lang=lang)

        assert shown["calls"] == [
            ["session-1", {"project_id": "project-1", "action": "cancel_local_selection"}]
        ]
        assert shown["errors"] == [text]
        assert shown["status"] == "selection_in_progress"


def test_the_preview_names_the_panel_the_researcher_closed() -> None:
    # The preview owner is exercised in the browser audit; here its close
    # handler is pinned to hand the closed resource to the event, read before
    # close() clears it.
    source = (JS / "screens-guided-pi-preview.js").read_text(encoding="utf-8")
    handler = source[source.index("if (event.target.closest('[data-gpi-preview-close]')) {"):]
    handler = handler[: handler.index("return;")]

    assert handler.index("const closed = state.resource;") < handler.index("close();")
    assert "new CustomEvent('easyicu:guided-preview-closed'" in handler
    assert "detail: { resource: closed, projectId: closedProjectId }" in handler
