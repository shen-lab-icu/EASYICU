"""「在 eICU 数据里…」 is offered the eICU export EasyICU already holds.

When the opening question names a full database and the conversation still
needs a source, the card reads the host's path-free offer for that database
(GET .../registered-source-offer). A recommended export becomes the primary
action, 「使用这份数据」; exports EasyICU cannot choose between are listed
for the researcher; a failed read falls back to folder selection. The click
posts ``use_registered_source`` with that exact source id, which the host
binds and confirms in one step; the page's StudyContext copy follows the
host's new revision, and the conversation continues to the plan.
"""

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


_RECOMMENDED = {
    "source_id": "src_eicu_full", "label": "eICU-CRD v2.0", "stays": 200859,
    "module_count": 12, "generated_date": "2026-10-19",
}
_TIED = [
    {"choice": 1, "source_id": "src_a", "label": "eICU-CRD v2.0", "stays": 200859,
     "module_count": 12, "generated_date": "2026-10-19"},
    {"choice": 2, "source_id": 'src_"b"', "label": "eICU-CRD v2.0", "stays": 200859,
     "module_count": 12, "generated_date": "2026-10-18"},
]


def _offer_script(replies: list) -> str:
    return f"""
      global.window = global;
      window.EU_LANG = 'zh';
      eval({_ESCAPE_OWNER!r});
      eval({_read("js/screens-guided-pi-next-actions.js")!r});
      eval({_read("js/screens-guided-pi-host-jobs.js")!r});
      eval({_read("js/screens-guided-pi-data-consent.js")!r});
      const consent = window.EasyICU.guidedPi.require('dataConsent');
      const replies = {json.dumps(replies)};
      const calls = [];
      let renders = 0;
      let project = 'proj-1';
      const session = {{ session_id: 's1', data_source_authorization: {{ status: 'pending' }} }};
      const host = {{
        tr: (en, zh) => zh,
        api: () => ({{
          loadOfficialDemoSources: async () => ({{ sources: [] }}),
          loadPiCopilotRegisteredSourceOffer: async (...args) => {{
            calls.push(args);
            const reply = replies.shift();
            if (reply && reply.error) throw Object.assign(new Error('x'), {{ code: reply.error }});
            return reply;
          }},
        }}),
        dataConsent: consent,
        session: () => session, projectId: () => project,
        messages: () => [{{ role: 'user', text: '在 eICU 数据里评估乳酸和死亡率的关系' }}],
        render() {{ renders += 1; }}, busy: () => false,
        workflowReceipts: () => [], setWorkflowReceipts() {{}}, setError() {{}}, errorText: e => String(e),
      }};
      const owner = window.EasyICU.guidedPi.require('hostJobs').create(host);
      const esc = window.EU_HTML.esc;
      const card = () => consent.render(session, {{
        tr: (en, zh) => zh, esc, icon: () => '', namedDemo: null, namedDatabase: owner.namedDatabase(),
      }});
      const tick = () => new Promise(resolve => setTimeout(resolve, 5));
    """


def _run(script: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    return json.loads(run_node(node, script, check=True).stdout)


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_the_recommended_export_is_the_primary_action() -> None:
    out = _run(_offer_script([{"database": "eicu", "recommended": _RECOMMENDED, "choices": []}]) + """
      (async () => {
        await tick();   // the demo catalog answers first
        const loading = card();
        await tick();
        const ready = card();
        const rendersAfterAnswer = renders;
        card(); await tick();
        owner.stopAll();
        process.stdout.write(JSON.stringify({
          calls,
          loading: {
            copy: loading.includes('正在查找 EasyICU 中已有的 eICU 数据…'),
            folderOnly: loading.includes('选择本地文件夹') && !loading.includes('use_registered_source'),
            noPrimary: !loading.includes('btn primary'),
          },
          ready: {
            title: ready.includes('你的问题指定了 eICU'),
            facts: ready.includes('<p class="gpi-data-consent-facts">eICU-CRD v2.0 · 2026-10-19 生成 · 200,859 例 ICU 住院 · 12 个模块</p>'),
            use: ready.includes('<button class="btn primary" type="button" data-gpi-data-source-action="use_registered_source" data-gpi-data-source-database="eicu" data-gpi-data-source-id="src_eicu_full">使用这份数据</button>'),
            folder: ready.includes('<button class="btn" type="button" data-gpi-data-source-action="begin_local_selection" data-gpi-data-source-database="eicu">选择本地文件夹</button>'),
            notice: ready.includes('选择数据源不等于批准分析'),
            noPath: !/\\/|path/i.test(ready.replace(/<\\/[a-z]+>/g, '')),
          },
          rendered: rendersAfterAnswer >= 1,
          noRefetchWithinMinute: calls.length === 1,
        }));
      })().catch(err => { console.error(err); process.exitCode = 1; });
    """)

    assert out["calls"] == [["s1", "proj-1", "eicu"]]
    assert all(out["loading"].values()), out["loading"]
    assert all(out["ready"].values()), out["ready"]
    assert out["rendered"] is True
    assert out["noRefetchWithinMinute"] is True


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_tied_exports_are_listed_and_a_failed_read_falls_back_to_the_folder() -> None:
    out = _run(_offer_script([{"database": "eicu", "recommended": None, "choices": _TIED}]) + """
      (async () => {
        await tick(); card(); await tick();
        const tied = card();
        // Another project has its own offer; this one fails.
        project = 'proj-2';
        replies.push({ error: 'pi_session_data_source_authorization_not_pending' });
        card(); await tick();
        const failed = card();
        project = '';
        const noProject = card();
        owner.stopAll();
        process.stdout.write(JSON.stringify({
          tied: {
            count: tied.includes('EasyICU 中有 2 份验证过的 eICU 数据，无法替你判断这项研究该用哪一份。'),
            first: tied.includes('data-gpi-data-source-id="src_a">使用这份</button>'),
            escaped: tied.includes('data-gpi-data-source-id="src_&quot;b&quot;"') && !tied.includes('src_"b"'),
            dates: tied.includes('2026-10-19 生成') && tied.includes('2026-10-18 生成'),
            noPrimaryUse: !tied.includes('btn primary'),
          },
          failed: failed.includes('选择 eICU 数据') && !failed.includes('use_registered_source'),
          noProject: noProject.includes('选择 eICU 数据'),
          calls: calls.map(row => row[1]),
        }));
      })().catch(err => { console.error(err); process.exitCode = 1; });
    """)

    assert all(out["tied"].values()), out["tied"]
    assert out["failed"] is True
    assert out["noProject"] is True
    assert out["calls"] == ["proj-1", "proj-2"]


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_the_click_posts_the_exact_export_and_the_conversation_continues() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    script = f"""
      global.window = global;
      window.EU_LANG = 'zh';
      eval({_ESCAPE_OWNER!r});
      eval({_read("js/screens-guided-pi-data-consent.js")!r});
      eval({_read("js/screens-guided-pi-data-binding.js")!r});
      const consent = window.EasyICU.guidedPi.require('dataConsent');
      const confirmed = {{
        session_id: 's1', binding: {{ study_context_id: 'study-1' }},
        data_source_authorization: {{
          status: 'confirmed', confirmation_mode: 'select_registered_source', extraction_scope: 'study_required',
          source: {{ label: 'eICU-CRD', reference_release: '2.0' }}, confirmed_at: '2026-10-20T11:00:00Z',
        }},
      }};
      const posted = [];
      const steps = [];
      window.EU_STUDY_CONTEXT = {{
        hydrate: async () => {{ steps.push('hydrate'); }},
        active: () => ({{ id: 'study-1' }}),
        activate: async () => {{ steps.push('activate'); }},
        refreshActiveFromServer: async () => {{ steps.push('refresh'); }},
      }};
      let session = {{ session_id: 's1', binding: {{ study_context_id: 'study-1' }}, data_source_authorization: {{ status: 'pending' }} }};
      const binding = window.EasyICU.guidedPi.require('dataBinding').create({{
        api: () => ({{ authorizePiCopilotDataSource: async (id, body) => {{ posted.push([id, body]); return {{ session: confirmed, resource: null }}; }} }}),
        render() {{}}, projectId: () => 'proj-1',
        loadWorkflow: async () => {{ steps.push('workflow'); }},
        dataConsent: consent, errorText: e => String(e), rememberSession() {{}},
        continueAfterDataSourceConfirmation: async () => {{ steps.push('continue'); }},
        session: () => session, busy: () => false, setError() {{}}, setSession: value => {{ session = value; }},
      }});
      const target = {{ dataset: {{ gpiDataSourceAction: 'use_registered_source', gpiDataSourceDatabase: 'eicu', gpiDataSourceId: 'src_eicu_full' }} }};
      const event = {{ target: {{ closest: () => target }} }};
      const action = consent.actionFromEvent(event);
      (async () => {{
        await binding.authorizeDataSource(action, {{ database: consent.databaseFromEvent(event), sourceId: consent.sourceIdFromEvent(event) }});
        const ctx = {{ tr: (en, zh) => zh, esc: window.EU_HTML.esc, icon: () => '' }};
        const past = consent.renderPast(session, ctx);
        process.stdout.write(JSON.stringify({{
          posted, steps,
          past: past.includes('已确认数据源') && past.includes('使用 EasyICU 已有的数据 · 只准备研究所需数据'),
          matches: consent.matchesSourceSelection(session, 'eICU-CRD v2.0'),
          noCard: consent.render(session, {{ ...ctx, namedDemo: null, namedDatabase: null }}) === '',
        }}));
      }})().catch(err => {{ console.error(err); process.exitCode = 1; }});
    """
    out = json.loads(run_node(node, script, check=True).stdout)

    assert out["posted"] == [["s1", {
        "project_id": "proj-1", "action": "use_registered_source",
        "database": "eicu", "source_id": "src_eicu_full",
    }]]
    # The page's StudyContext copy follows the host-side binding before the
    # conversation continues to the plan, exactly once.
    assert out["steps"] == ["hydrate", "refresh", "workflow", "continue"]
    assert out["past"] is True
    assert out["matches"] is True
    assert out["noCard"] is True


def test_the_card_action_forwards_its_source_id() -> None:
    events = _read("js/screens-guided-pi-events.js")
    shell = _read("js/screens-guided-pi.js")
    assert "const sourceId = typeof DATA_CONSENT.sourceIdFromEvent === 'function' ? DATA_CONSENT.sourceIdFromEvent(event) : '';" in events
    assert "authorizeDataSource(dataSourceAction, database || sourceId ? { database, sourceId } : undefined);" in events
    # The host-jobs owner reads the offer for the shell's current project.
    assert "projectId: () => projectId()," in shell
    assert ".gpi-data-consent-choices{" in _read("css/guided-pi.css")
