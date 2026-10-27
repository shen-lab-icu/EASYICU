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
let started = 0;
hosted.openConversation(base, true, () => { started += 1; });
hosted.openConversation({ ...base, session: { session_id: 's' } }, true, () => { started += 1; });
hosted.openConversation(base, false, () => { started += 1; });
out.started = started;
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
  guidedDrafts: { data: { default_parent_dir: '/home/zhuhb/easyicu_dev9/users/a/home/projects' } },
  guidedFolderDialogMode: 'new', guidedFolderSeedTitle: '血糖与死亡',
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
// Local projects are created and opened from the project list's foot.
out.localProjects = {
  controls: hosts.gdFolderControls.innerHTML,
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
    assert out["started"] == 1
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
    assert out["localProjects"] == {"controls": "", "parentField": False}


def test_the_shell_opens_the_hosted_project_and_conversation_through_their_owners() -> None:
    index = _read("index.html")
    shell = _read("js/screens-guided.js")
    screen = _read("js/screens-guided-pi.js")
    provider = _read("js/screens-guided-pi-provider.js")

    hosted_tag = index.index("js/screens-guided-pi-hosted.js?v=20261021-help1")
    assert hosted_tag < index.index("js/screens-guided-pi-connect.js") < index.index("js/screens-guided-pi.js")

    landing = shell.split("async function openHostedLandingProject(piOwner)", 1)[1].split("function createLocalGuidedDraft", 1)[0]
    # Only when the address names no project; the server picks the folder.
    assert "piOwner.historyContext().projectId" in landing
    assert "hosted.landingProject(localDraftRows())" in landing
    assert "blankGuidedDraftPayload(hosted.DEFAULT_PROJECT_TITLE)" in landing
    assert "parent_dir" not in landing
    assert ".then(() => openHostedLandingProject(piOwner)).finally(() => {" in shell

    assert (
        "const openHostedConversation = () => { const hosted = MODULES.optional('hosted'); "
        "if (hosted) hosted.openConversation(state, connectionReady(), createSession); };"
    ) in screen
    assert screen.count("openHostedConversation();") == 2

    summary = provider.split("function renderBindingSummary(options)", 1)[1]
    assert summary.index("hosted.summaryHtml(") < summary.index("data-gpi-connect-open")


_SETTINGS_SCRIPT = r"""
global.window = {
  EU_LANG: 'zh', localStorage: { getItem: () => null, setItem() {} },
  sessionStorage: { getItem: () => null, removeItem() {} },
  EU_CAPABILITIES: { deployment: { hosted: process.argv[3] === 'hosted' } },
  EU_SETTINGS: {
    language: 'zh', ai_enabled: true, export_dir: '/home/zhuhb/exports',
    support_contact: '信息科 <王老师>', about: { version: '9.0', python: '3.11', config_path: '/home/zhuhb/easyicu/settings.json' },
  },
};
global.t = (en, zh) => zh;
global.icon = () => '';
global.document = { getElementById: () => null };
// index.html loads the database-name owner ahead of every screen.
require(process.argv[1].replace(/[^/]+$/, 'database-names.js'));
require(process.argv[1]);
require(process.argv[2]);
const page = window.SCREENS.settings;
const html = page.render();
const rail = page.rail();
process.stdout.write(JSON.stringify({
  hosted: html.includes('data-settings-hosted'),
  contact: html.includes('信息科 &lt;王老师&gt;'),
  language: html.includes('id="set-language"') && html.includes('data-setting="language"'),
  diagnostics: html.includes('data-settings-diagnostics'),
  adminControls: /data-setting="(ai_enabled|data_mode)"|id="set-(workspace|privacy|data-mode|capabilities)"|data-ext-manager/.test(html),
  paths: html.includes('/home/zhuhb'),
  reset: page.actionHtml.includes('data-settings-reset'),
  railTargets: (rail.match(/data-settings-jump="([^"]+)"/g) || []).length,
}));
"""


def _settings(mode: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    completed = run_node(
        node,
        _SETTINGS_SCRIPT,
        str(STATIC / "js" / "screens-guided-pi-hosted.js"),
        str(STATIC / "js" / "screens-settings.js"),
        mode,
        check=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.requires_node
def test_hosted_settings_keep_the_clinicians_display_and_the_administrator_contact() -> None:
    hosted = _settings("hosted")
    assert hosted == {
        "hosted": True, "contact": True, "language": True, "diagnostics": True,
        "adminControls": False, "paths": False, "reset": False, "railTargets": 3,
    }

    local = _settings("local")
    assert local["hosted"] is False
    assert local["adminControls"] is True
    assert local["reset"] is True
    # Personal instructions are per install, so only the local rail offers them.
    assert local["railTargets"] == 7


def test_a_hosted_skill_catalogue_offers_no_install() -> None:
    skills = _read("js/screens-skills.js")
    # The only upload entry sits in the menu a hosted instance leaves out.
    assert "const createMenu = hostedOwner && hostedOwner.active() ? '' : `" in skills
    assert "<h1>${tr('Skills', '技能')}</h1></div>${createMenu}</div>" in skills
    assert skills.count("data-sk-upload aria-label") == 1
    menu = skills.split("const createMenu = ", 1)[1].split("</details>`;", 1)[0]
    assert "data-sk-upload aria-label" in menu
    # Skill state is the administrator's: a hosted detail shows it, no toggle.
    assert "const stateOnly = method || Boolean(window.EasyICU?.guidedPi?.optional('hosted')?.active());" in skills
    assert '<div class="eusk-detail-actions">${stateOnly' in skills
    assert skills.count("data-sk-toggle=") == 1


def test_a_hosted_instance_uses_only_the_administrators_api_connection() -> None:
    shell = _read("js/screens-guided-pi.js")
    restore = shell.split("function restoreConfiguredResearchProvider() {", 1)[1].split("\n  }\n", 1)[0]
    # The ChatGPT/Codex account choice is never the hosted connection, and the
    # status read that decides it waits for the deployment facts.
    assert "if ((hosted && hosted.active()) || (" in restore
    assert "state.researchProvider = 'api';" in restore
    assert "state.startupPromise = capabilitiesReady.then(() => loadStatus())" in shell
