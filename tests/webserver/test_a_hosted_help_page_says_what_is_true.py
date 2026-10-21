"""快速上手 on a hosted instance describes the hospital server, not a local install.

The local page promises that EasyICU runs on the reader's machine, that
nothing is uploaded and that a model token is the reader's to bring. None of
that holds for a clinician on the hospital server, so the hosted page names
where the data is, which databases the administrator prepared, that the
conversation uses the administrator's model, and whom to ask. A hosted
instance also drops the expert single-key shortcuts, the design-reference
page and the demo data switches; a local build keeps all of them.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node

STATIC = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "webserver" / "static"

_SCRIPT = r"""
const hosted = process.argv[3] === 'hosted';
const owners = {};
global.window = {
  EU_LANG: 'zh',
  EU_HTML: { esc: value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') },
  EU_CAPABILITIES: {
    deployment: { hosted },
    registered_sources: { by_database: { miiv: 1, eicu: 2, aumc: 0, hirid: 0, mimic: 0, sic: 0 } },
  },
  EU_SETTINGS: { support_contact: '信息科 <王老师> 8123' },
  EasyICU: { guidedPi: {
    declare: (name, owner) => { owners[name] = owner; },
    optional: name => owners[name] || null,
  } },
};
global.t = (en, zh) => zh;
global.icon = () => '';
require(process.argv[1]);
require(process.argv[2]);
const page = window.SCREENS.tutorial;
process.stdout.write(JSON.stringify({ action: page.actionHtml(), rail: page.rail(), body: page.render() }));
"""


def _page(deployment: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(
        node,
        _SCRIPT,
        str(STATIC / "js" / "screens-guided-pi-hosted.js"),
        str(STATIC / "js" / "screens-help.js"),
        deployment,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


@pytest.mark.requires_node
def test_the_hosted_page_names_the_server_the_prepared_data_and_the_administrator() -> None:
    page = _page("hosted")
    body, rail = page["body"], page["rail"]

    assert "运行在医院服务器上" in body
    assert "研究数据保存在医院服务器上" in body
    assert "这台服务器上已准备好：MIMIC-IV、eICU-CRD。" in body
    assert "由管理员配置的模型" in body
    assert "患者的逐行记录不会发给模型" in body
    assert "不能替代对个体患者的临床判断" in body
    assert "复制问题编号发给管理员（信息科 &lt;王老师&gt; 8123）" in body
    # The local install's promises are not made on the server.
    for local_claim in ("local-first", "你的机器", "本机", "本地导出", "开始演示"):
        assert local_claim not in body
    assert 'data-nav="guided"' in page["action"]
    assert "data-help-startdemo" not in page["action"]
    assert "快捷键" not in rail
    assert "信息科 &lt;王老师&gt; 8123" in rail


@pytest.mark.requires_node
def test_the_local_page_is_unchanged() -> None:
    page = _page("local")

    assert "local-first" in page["body"]
    assert "data-help-startdemo" in page["action"]
    assert "快捷键" in page["rail"]


def test_a_hosted_instance_drops_single_key_shortcuts_and_the_states_catalogue() -> None:
    app = (STATIC / "js" / "app.js").read_text(encoding="utf-8")

    assert "const HOSTED_HIDDEN_ROUTES = new Set(['states']);" in app
    assert "if (hostedInstance() && HOSTED_HIDDEN_ROUTES.has(route)) {" in app
    assert "if (typing || e.metaKey || e.ctrlKey || e.altKey || hostedInstance()) return;" in app
    assert "tgt.tagName === 'SELECT'" in app


def test_a_hosted_instance_offers_no_demo_data_switch() -> None:
    js = STATIC / "js"
    shell_css = (STATIC / "css" / "shell.css").read_text(encoding="utf-8")
    app = (js / "app.js").read_text(encoding="utf-8")
    i18n = (js / "i18n.js").read_text(encoding="utf-8")
    api = (js / "api.js").read_text(encoding="utf-8")

    assert ":root[data-eu-hosted] [data-datamode-switch]{ display:none; }" in shell_css
    assert "document.documentElement.toggleAttribute('data-eu-hosted', hostedInstance());" in app
    # Every page that renders a demo switch marks it for the shell rule.
    rendering = [
        path for path in sorted(js.glob("*.js"))
        if 'data-datamode="demo"' in path.read_text(encoding="utf-8")
    ]
    assert {path.name for path in rendering} == {
        "app.js", "screens-extraction.js", "screens-viz.js", "screens-viz-patient.js",
    }
    for path in rendering:
        text = path.read_text(encoding="utf-8")
        assert text.count("data-datamode-switch") >= text.count('data-datamode="demo"'), path.name
    assert "data-ex-sample data-datamode-switch" in (js / "screens-extraction.js").read_text(encoding="utf-8")
    # No code path switches a hosted instance to demo data, and a browser that
    # remembers demo mode opens it on the prepared data.
    assert (
        "if (m === 'demo' && ((window.EU_CAPABILITIES || {}).deployment || {}).hosted === true) return;"
        in i18n
    )
    assert "window.setDataMode('real', { force: true });" in api.split("await hydrateCapabilities();", 1)[1]


_PICKER = r"""
global.window = {
  t: (en, zh) => zh, icon: () => '', EU_HTML: { esc: value => String(value) },
  EU_CAPABILITIES: { deployment: { hosted: process.argv[2] === 'hosted' } },
};
require(process.argv[1]);
const picker = window.EU_EXTRACTION_FOLDER_PICKER;
const roots = [{ name: 'exports', path: '/srv/easyicu/exports' }];
process.stdout.write(JSON.stringify([
  picker.displayPath('/srv/easyicu/exports', roots),
  picker.displayPath('/srv/easyicu/exports/eicu_v2/exports', roots),
  picker.displayPath('/srv/easyicu/exports-other', roots),
]));
"""


@pytest.mark.requires_node
@pytest.mark.parametrize(
    ("deployment", "expected"),
    [
        ("hosted", ["数据目录 › exports", "数据目录 › exports › eicu_v2 › exports", ""]),
        ("local", ["", "", ""]),
    ],
)
def test_the_hosted_folder_picker_names_folders_from_the_data_root(deployment: str, expected: list) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(
        node, _PICKER, str(STATIC / "js" / "screens-extraction-folder-picker.js"), deployment, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    # The server path never shows in a hosted picker; a sibling folder that
    # merely shares the root's prefix is not inside it.
    assert json.loads(result.stdout) == expected


def test_hosted_review_names_the_prepared_data_without_paths_or_registration() -> None:
    viz = (STATIC / "js" / "screens-viz.js").read_text(encoding="utf-8")
    patient = (STATIC / "js" / "screens-viz-patient.js").read_text(encoding="utf-8")
    block = viz.split("function sourceRegistryBlock(mode) {", 1)[1].split("\n  function ", 1)[0]

    assert "const hosted = hostedServer();" in block
    assert "t('Prepared data', '已准备的数据')" in block
    # No server path, no rename/remove and no add-by-path on a hosted row.
    assert "${hosted ? '' : `<span class=\"src-path mono\">${esc(s.path)}</span>`}" in block
    assert "${hosted ? '' : `<span class=\"src-actions\">" in block
    assert "${hosted ? '' : `<div class=\"path-field editable src-add\">" in block
    assert "t('Load the data the administrator prepared on the server.', '加载管理员在服务器上准备好的数据。')" in patient
    assert "${hosted ? '' : `<div class=\"d mono\" style=\"margin-top:4px;\">path_hash=" in patient
    assert "hostedServer() ? t('Load this data', '加载这份数据')" in patient
    cohort = (STATIC / "js" / "screens-viz-cohort.js").read_text(encoding="utf-8")
    assert "hostedServer() ? t('Prepared data', '已准备的数据') : t('Local export', '本地导出')" in cohort
    assert "hostedServer() ? t('Load this data', '加载这份数据') : t('Load local export', '加载本地导出')" in cohort
    cohort_view = (STATIC / "js" / "screens-viz-cohort-view.js").read_text(encoding="utf-8")
    assert "if (hosted && Object.prototype.hasOwnProperty.call(HOSTED_TEXT, raw)) return t(...HOSTED_TEXT[raw]);" in cohort_view
    assert "'Local export': ['Prepared data', '已准备的数据']," in cohort_view
    crossdb = (STATIC / "js" / "screens-viz-crossdb-setup.js").read_text(encoding="utf-8")
    assert "hostedServer() ? h.t('Check a data folder on the server, then run bounded sampling.'" in crossdb


def test_the_hosted_settings_sidebar_lists_only_the_sections_the_page_has() -> None:
    shell = (STATIC / "js" / "desktop-module-shell.js").read_text(encoding="utf-8")

    assert (
        "].filter(([id]) => !hostedInstance() || id === 'set-language' || id === 'set-about').map("
        in shell
    )
