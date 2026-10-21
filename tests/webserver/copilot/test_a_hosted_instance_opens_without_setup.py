"""A hosted clinician starts at the question, not at setup.

On a hosted instance (``/api/capabilities`` → ``deployment.hosted``) the
administrator connects the model and keeps every project in the clinician's
own study folder on the server. The page therefore never shows the model
connection choice or a server folder: a missing model is reported with the
administrator's contact, the clinician's project opens (or is created) by
itself, and a conversation starts once the configured model is usable. A
local build keeps today's connection page and folder dialog.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    STATIC as STATIC,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read as _read,
)


_SCRIPT = r"""
const store = {};
const storage = { getItem: key => (key in store ? store[key] : null), setItem: (key, value) => { store[key] = String(value); } };
const hosts = {};
global.window = {
  EU_LANG: 'zh', localStorage: storage,
  EU_CAPABILITIES: { deployment: { hosted: true, provider_provisioned: false } },
  EU_SETTINGS: { support_contact: '信息科 <王老师>\u0007 8123' },
};
global.document = { activeElement: null, getElementById: id => (hosts[id] || (hosts[id] = { id, innerHTML: '' })) };
require(process.argv[1]);
require(process.argv[2]);
require(process.argv[3]);
const tr = (en, zh) => zh;
const esc = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const hosted = window.EasyICU.guidedPi.require('hosted');
const out = {};
out.active = hosted.active();
out.provisioned = hosted.providerProvisioned();
out.defaultTitle = hosted.DEFAULT_PROJECT_TITLE;
out.landing = (hosted.landingProject([
  { id: 'older', updated_at: '2026-10-01T00:00:00Z' },
  { id: 'newest', updated_at: '2026-10-20T00:00:00Z' },
  { id: '' },
]) || {}).id;
out.landingNone = hosted.landingProject([]);
const base = { project: { id: 'p1' }, session: null, creating: false, projectLoading: false, demoMode: false, projectIssue: '' };
out.opens = {
  ready: hosted.opensConversation(base, true),
  modelMissing: hosted.opensConversation(base, false),
  hasSession: hosted.opensConversation({ ...base, session: { session_id: 's' } }, true),
  loading: hosted.opensConversation({ ...base, projectLoading: true }, true),
  demo: hosted.opensConversation({ ...base, demoMode: true }, true),
  noProject: hosted.opensConversation({ ...base, project: null }, true),
};
const missing = hosted.missingModelHtml({ tr, esc });
out.missing = { contactEscaped: missing.includes('信息科 &lt;王老师&gt;  8123'), ask: missing.includes('请联系管理员'), raw: missing.includes('<王老师>') };
const readySummary = hosted.summaryHtml({ tr, esc, connectionReady: true });
const missingSummary = hosted.summaryHtml({ tr, esc, connectionReady: false });
out.summary = {
  ready: readySummary.includes('模型已就绪'),
  readyControls: /data-gpi-(config|connect-open)/.test(readySummary),
  missing: missingSummary.includes('模型尚未配置') && missingSummary.includes('8123'),
  missingControls: /data-gpi-(config|connect-open)/.test(missingSummary),
};

const state = { showSetup: true, setupSaving: false, codexBusy: false, error: '' };
const gate = window.EasyICU.guidedPi.require('connect').create({
  state, tr, esc, render() {},
  connectionReady: () => false, connectionConfigured: () => false,
  finishProviderSetup: async () => {},
  body: () => '<form data-gpi-provider-form>ChatGPT / Codex</form>',
});
out.hostedPage = gate.pageRequired();
out.hostedAsk = gate.ask('start_conversation');
const dialog = gate.dialogHtml();
out.hostedDialog = {
  title: dialog.includes('模型尚未配置'), ok: dialog.includes('知道了'),
  form: dialog.includes('data-gpi-provider-form'), account: dialog.includes('ChatGPT'),
  continueOffered: dialog.includes('data-gpi-connect-continue'), contact: dialog.includes('8123'),
};
gate.dismiss();
out.dismissed = state.connectPrompt === null;

const projects = window.EU_GUIDED_PROJECTS;
const ctx = {
  t: tr, esc, attr: esc, icon: () => '', compactPath: value => String(value || ''),
  slugifyDraftFolder: value => String(value || ''),
  guidedDrafts: { data: { default_parent_dir: '/home/zhuhb/easyicu_dev9/users/a/home/projects' } },
  guidedFolderDialogMode: 'new', guidedFolderSeedTitle: '血糖与死亡', guidedFolderMenuOpen: true,
};
projects.renderFolderControls(ctx);
const controls = hosts.gdFolderControls.innerHTML;
projects.renderFolderDialog(ctx);
const hostedDialog = hosts.gdFolderDialogHost.innerHTML;
out.hostedProjects = {
  newOnly: controls.includes('data-folder-choice="new"') && !controls.includes('data-folder-choice="open"'),
  menu: controls.includes('gd-folder-menu'),
  title: hostedDialog.includes('data-draft-title') && hostedDialog.includes('血糖与死亡'),
  create: hostedDialog.includes('data-createdraft'),
  folderInputs: /data-draft-(slug|parent-dir)|data-browsedraftparent|data-folder-choice="open"/.test(hostedDialog),
  path: hostedDialog.includes('/home/zhuhb'),
};

window.EU_CAPABILITIES = { deployment: { hosted: false } };
state.showSetup = false;
out.localPage = gate.pageRequired();
gate.ask('start_conversation');
out.localDialogForm = gate.dialogHtml().includes('data-gpi-provider-form');
projects.renderFolderControls(ctx);
projects.renderFolderDialog(ctx);
out.localProjects = {
  openOffered: hosts.gdFolderControls.innerHTML.includes('data-folder-choice="open"'),
  parentField: hosts.gdFolderDialogHost.innerHTML.includes('data-draft-parent-dir'),
};
process.stdout.write(JSON.stringify(out));
"""


def _run() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    completed = run_node(
        node,
        _SCRIPT,
        str(STATIC / "js" / "screens-guided-pi-hosted.js"),
        str(STATIC / "js" / "screens-guided-pi-connect.js"),
        str(STATIC / "js" / "screens-guided-projects.js"),
        check=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.requires_node
def test_a_hosted_clinician_is_never_asked_to_connect_a_model_or_choose_a_folder() -> None:
    out = _run()

    assert out["active"] is True
    assert out["provisioned"] is False
    assert out["defaultTitle"] == "我的研究"
    assert out["landing"] == "newest"
    assert out["landingNone"] is None
    assert out["opens"] == {
        "ready": True, "modelMissing": False, "hasSession": False,
        "loading": False, "demo": False, "noProject": False,
    }
    # The administrator contact is shown escaped, control characters removed.
    assert out["missing"] == {"contactEscaped": True, "ask": True, "raw": False}
    assert out["summary"] == {
        "ready": True, "readyControls": False, "missing": True, "missingControls": False,
    }

    # Even with connection settings requested, no connection page is shown.
    assert out["hostedPage"] is False
    assert out["hostedAsk"] is False
    assert out["hostedDialog"] == {
        "title": True, "ok": True, "form": False, "account": False,
        "continueOffered": False, "contact": True,
    }
    assert out["dismissed"] is True

    assert out["hostedProjects"] == {
        "newOnly": True, "menu": False, "title": True, "create": True,
        "folderInputs": False, "path": False,
    }


@pytest.mark.requires_node
def test_a_local_build_keeps_the_connection_page_and_the_folder_dialog() -> None:
    out = _run()

    assert out["localPage"] is True
    assert out["localDialogForm"] is True
    assert out["localProjects"] == {"openOffered": True, "parentField": True}


def test_the_shell_opens_the_hosted_project_and_conversation_through_their_owners() -> None:
    index = _read("index.html")
    shell = _read("js/screens-guided.js")
    screen = _read("js/screens-guided-pi.js")
    provider = _read("js/screens-guided-pi-provider.js")

    hosted_tag = index.index("js/screens-guided-pi-hosted.js?v=20261021-hosted1")
    assert hosted_tag < index.index("js/screens-guided-pi-connect.js") < index.index("js/screens-guided-pi.js")

    landing = shell.split("async function openHostedLandingProject(piOwner)", 1)[1].split("function createLocalGuidedDraft", 1)[0]
    # Only when the address names no project; the server picks the folder.
    assert "piOwner.historyContext().projectId" in landing
    assert "hosted.landingProject(localDraftRows())" in landing
    assert "blankGuidedDraftPayload(hosted.DEFAULT_PROJECT_TITLE)" in landing
    assert "parent_dir" not in landing
    assert ".then(() => openHostedLandingProject(piOwner)).finally(() => {" in shell

    opener = screen.split("function openHostedConversation()", 1)[1].split("function unmount()", 1)[0]
    assert "hosted.opensConversation(state, connectionReady())" in opener
    assert "void createSession()" in opener
    assert screen.count("openHostedConversation();") == 2

    summary = provider.split("function renderBindingSummary(options)", 1)[1]
    assert summary.index("hosted.summaryHtml(") < summary.index("data-gpi-connect-open")
