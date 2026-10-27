"""遇到问题 gives a clinician a reference the administrator can look up.

The conversation header's help menu copies the project, conversation, run and
job ids with the version and time — never a path, question text or patient
data — and names the administrator on a hosted instance. A hosted header
leaves out the expert menu and the connection settings the administrator
owns; a local header keeps them.
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
const copied = [];
global.window = {
  EU_LANG: 'zh',
  EU_CAPABILITIES: { deployment: { hosted: true } },
  EU_SETTINGS: { support_contact: '信息科 王老师 8123', about: { version: '9.0.1' } },
  navigator: { clipboard: { writeText: async text => { copied.push(text); } } },
};
global.document = {};
require(process.argv[1]);
require(process.argv[2]);
require(process.argv[3]);
const tr = (en, zh) => zh;
const esc = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const help = window.EasyICU.guidedPi.require('help').create({ tr, esc, context: () => ({
  projectId: 'draft_b2f90ee188ae', sessionId: 'pi_session_01', runId: 'run_20261021T090520_c4b7c9',
  jobId: '/home/zhuhb/easyicu_dev9/users/a 血糖与死亡',
}) });
const header = window.EasyICU.guidedPi.require('header');
const out = {};
const reference = help.referenceText();
out.reference = reference.split('\n');
const menu = help.render();
out.menu = {
  popover: menu.includes('data-popover-menu'), copy: menu.includes('复制问题编号'),
  contact: menu.includes('信息科 王老师 8123'),
};
const options = { tr, esc, icon: () => '', projectTitle: '我的研究', sessionTitle: '新对话', layout: {}, connectionLabel: 'deepseek · deepseek-chat' };
const hostedHead = header.render({ ...options, hosted: true, helpHtml: menu });
const localHead = header.render({ ...options, hosted: false, helpHtml: menu });
out.hostedHead = {
  help: hostedHead.includes('data-gpi-help'),
  expert: /gpi-head-overflow|data-gpi-mode-switch|data-gpi-demo|data-gpi-study-setup/.test(hostedHead),
  connection: header.renderModelControl({ ...options, hosted: true }).includes('data-gpi-config'),
};
out.localHead = {
  help: localHead.includes('data-gpi-help'),
  expert: /gpi-head-overflow/.test(localHead) && /data-gpi-mode-switch/.test(localHead),
  connection: header.renderModelControl({ ...options, hosted: false }).includes('data-gpi-config'),
};
const button = { textContent: '复制问题编号', closest: () => null };
out.handled = help.handleClick({ target: { closest: selector => (selector === '[data-gpi-help-copy]' ? button : null) } });
out.ignored = help.handleClick({ target: { closest: () => null } });
setImmediate(() => {
  out.copied = copied.length === 1 && copied[0].startsWith('EasyICU 9.0.1\n项目: draft_b2f90ee188ae');
  out.label = button.textContent;
  // A browser that refuses the clipboard: the reference is shown instead.
  window.navigator.clipboard.writeText = async () => { throw new Error('denied'); };
  const inserted = [];
  const holder = { querySelector: () => null, insertBefore: node => inserted.push(node) };
  document.createElement = tag => ({ tag, className: '', textContent: '' });
  window.document = document;
  const refused = { textContent: '复制问题编号', parentNode: holder, closest: () => null };
  help.handleClick({ target: { closest: selector => (selector === '[data-gpi-help-copy]' ? refused : null) } });
  setImmediate(() => {
    out.refused = {
      label: refused.textContent,
      shown: inserted.length === 1 && inserted[0].tag === 'pre' && inserted[0].className === 'gpi-help-reference'
        && inserted[0].textContent.startsWith('EasyICU 9.0.1\n项目: draft_b2f90ee188ae'),
    };
    process.stdout.write(JSON.stringify(out));
    process.exit(0);
  });
});
"""


def _run() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    completed = run_node(
        node,
        _SCRIPT,
        str(STATIC / "js" / "screens-guided-pi-hosted.js"),
        str(STATIC / "js" / "screens-guided-pi-help.js"),
        str(STATIC / "js" / "screens-guided-pi-header.js"),
        check=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.requires_node
def test_the_problem_reference_carries_ids_and_the_administrator_only() -> None:
    out = _run()

    reference = out["reference"]
    assert reference[:4] == [
        "EasyICU 9.0.1",
        "项目: draft_b2f90ee188ae",
        "对话: pi_session_01",
        "运行: run_20261021T090520_c4b7c9",
    ]
    # A value that is not an id (a path, question text) is left out.
    assert len(reference) == 5 and reference[4].startswith("时间: 20")
    assert "/home" not in "\n".join(reference) and "血糖" not in "\n".join(reference)
    assert out["menu"] == {"popover": True, "copy": True, "contact": True}
    assert out["handled"] is True and out["ignored"] is False
    assert out["copied"] is True
    assert out["label"] == "已复制"
    assert out["refused"] == {"label": "请选中下方编号复制，或截图", "shown": True}


@pytest.mark.requires_node
def test_a_hosted_header_drops_the_expert_menu_and_the_connection_settings() -> None:
    out = _run()

    assert out["hostedHead"] == {"help": True, "expert": False, "connection": False}
    assert out["localHead"] == {"help": True, "expert": True, "connection": True}


def test_the_help_menu_is_wired_through_its_owner() -> None:
    index = _read("index.html")
    screen = _read("js/screens-guided-pi.js")
    events = _read("js/screens-guided-pi-events.js")
    view = _read("js/screens-guided-pi-session-view.js")
    sheet = _read("css/guided-pi-help.css")

    assert index.index("js/screens-guided-pi-help.js?v=") < index.index("js/screens-guided-pi.js?v=")
    assert index.index("css/guided-pi-skin.css?") < index.index("css/guided-pi-help.css?v=")
    assert "MODULES.require('help').create({ tr, esc, context: () => ({" in screen
    assert "if (HELP && HELP.handleClick(event)) return;" in events
    assert "helpHtml: HELP ? HELP.render() : ''," in view
    assert sheet.startswith("/* Owner: screens-guided-pi-help.js")


def test_the_help_menu_opens_toward_the_page_at_every_width() -> None:
    # The control ends the conversation header at every width, so the menu
    # hangs left from its right edge. A narrow-width left anchor pushed it
    # off a 390 px screen.
    sheet = _read("css/guided-pi-help.css")
    menu = sheet.split(".gpi-help-menu{", 1)[1].split("}", 1)[0]

    assert "right:0" in menu
    assert ".gpi-help-menu{right:auto" not in sheet and "left:0" not in sheet
