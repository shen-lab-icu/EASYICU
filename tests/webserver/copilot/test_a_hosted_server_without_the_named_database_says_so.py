"""Source cards distinguish missing hosted data from available named exports."""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _ESCAPE_OWNER as _ESCAPE_OWNER,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read as _read,
)


def _card(*, hosted: bool, reply: dict, contact: str = "") -> str:
    script = f"""
      global.window = global;
      window.EU_LANG = 'zh';
      window.EU_CAPABILITIES = {{ deployment: {{ hosted: {json.dumps(hosted)} }} }};
      window.EU_SETTINGS = {{ support_contact: {json.dumps(contact)} }};
      eval({_ESCAPE_OWNER!r});
      eval({_read("js/database-names.js")!r});
      eval({_read("js/screens-guided-pi-hosted.js")!r});
      eval({_read("js/screens-guided-pi-next-actions.js")!r});
      eval({_read("js/screens-guided-pi-host-jobs.js")!r});
      eval({_read("js/screens-guided-pi-data-consent.js")!r});
      const consent = window.EasyICU.guidedPi.require('dataConsent');
      const session = {{ session_id: 's1', data_source_authorization: {{ status: 'pending' }} }};
      const host = {{
        tr: (en, zh) => zh,
        api: () => ({{
          loadOfficialDemoSources: async () => ({{ sources: [] }}),
          loadPiCopilotRegisteredSourceOffer: async () => ({json.dumps(reply)}),
        }}),
        dataConsent: consent,
        session: () => session, projectId: () => 'proj-1',
        messages: () => [{{ role: 'user', text: '在 HiRID 数据库中描述体温记录的分布。' }}],
        render() {{}}, busy: () => false,
        workflowReceipts: () => [], setWorkflowReceipts() {{}}, setError() {{}}, errorText: e => String(e),
      }};
      const owner = window.EasyICU.guidedPi.require('hostJobs').create(host);
      const card = () => consent.render(session, {{
        tr: (en, zh) => zh, esc: window.EU_HTML.esc, icon: () => '', namedDemo: null,
        namedDatabase: owner.namedDatabase(),
      }});
      const tick = () => new Promise(resolve => setTimeout(resolve, 5));
      (async () => {{
        await tick(); card(); await tick();
        const html = card();
        owner.stopAll();
        process.stdout.write(JSON.stringify(html));
      }})().catch(err => {{ console.error(err); process.exitCode = 1; }});
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    return json.loads(run_node(node, script, check=True).stdout)


_EMPTY = {"database": "hirid", "recommended": None, "choices": []}


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_hosted_missing_database_names_the_administrator() -> None:
    html = _card(hosted=True, reply=_EMPTY, contact="信息科 <8888>")

    assert "尚未添加 HiRID 数据。" in html
    assert "信息科 &lt;8888&gt;" in html and "添加 HiRID 数据" in html
    assert "本版本暂不支持" not in html
    assert 'data-gpi-data-source-action="begin_local_selection">选择其他数据源</button>' in html
    assert "选择 HiRID 数据" not in html and "data-gpi-data-source-database" not in html


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_local_install_offers_the_named_database_folder() -> None:
    html = _card(hosted=False, reply=_EMPTY)

    assert "尚未添加 HiRID 数据。" in html
    assert "选择 HiRID 数据" in html
    assert 'data-gpi-data-source-database="hirid"' in html
    assert "本版本暂不支持" not in html
    assert "这台服务器上还没有" not in html


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_registered_export_is_offered_for_research() -> None:
    held = {"source_id": "src_hirid", "label": "HiRID v1.1.1", "stays": 12,
            "module_count": 2, "generated_date": "2026-10-19"}
    html = _card(hosted=True, reply={"database": "hirid", "recommended": held, "choices": []})

    assert "本版本暂不支持" not in html
    assert 'data-gpi-data-source-id="src_hirid"' in html
    assert 'data-gpi-data-source-database="hirid"' in html
    assert "12 例 ICU 住院" in html and "2 个模块" in html
    assert "这台服务器上还没有" not in html
