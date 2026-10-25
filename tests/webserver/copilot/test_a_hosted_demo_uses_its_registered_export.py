"""Hosted demo cards use the registry; only a desktop can prepare a demo."""

from __future__ import annotations

import json

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _ESCAPE_OWNER as _ESCAPE_OWNER,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read as _read,
)


def _card(*, recommended=None, choices=(), export_ready=False, hosted=True, forbidden=False,
          database="eicu_demo", title="eICU Demo", reader_label=None, stays=12):
    script = f"""
      global.window = global;
      window.EU_CAPABILITIES = {{deployment: {{hosted: {json.dumps(hosted)}}}}};
      eval({_ESCAPE_OWNER!r});
      eval({_read('js/screens-guided-pi-hosted.js')!r});
      eval({_read('js/screens-guided-pi-error-text.js')!r});
      eval({_read('js/screens-guided-pi-host-jobs.js')!r});
      eval({_read('js/screens-guided-pi-data-consent.js')!r});
      const consent = window.EasyICU.guidedPi.require('dataConsent');
      const session = {{session_id: 's1', data_source_authorization: {{status: 'pending'}}}};
      const requested = [], prepared = [], errors = [];
      const host = {{tr: (en, zh) => zh, dataConsent: consent,
        api: () => ({{
          loadOfficialDemoSources: async () => ({{sources: [{{
            id: 'eicu_demo_v2_0_1', database: {json.dumps(database)}, title: {json.dumps(title)}, version: '2.0.1',
            scope: {{icu_stays: {json.dumps(stays)}}},
            reader_label: {json.dumps(reader_label)},
            status: {{export_ready: {json.dumps(export_ready)}}}
          }}]}}),
          loadPiCopilotRegisteredSourceOffer: async (...args) => {{
            requested.push(args);
            return {{database: 'eicu_demo', recommended: {json.dumps(recommended)}, choices: {json.dumps(choices)}}};
          }},
          startOfficialDemoSourcePrepare: async id => {{
            prepared.push(id);
            if ({json.dumps(forbidden)}) throw Object.assign(new Error('unavailable_when_hosted'), {{code: 'unavailable_when_hosted', status: 403}});
            return {{job_id: 'j1'}};
          }},
        }}),
        session: () => session, projectId: () => 'p1', busy: () => false,
        messages: () => [{{role: 'user', text: {json.dumps(f'Use {title} v2.0.1 to describe ICU stays')}}}],
        render() {{}}, workflowReceipts: () => [], setWorkflowReceipts() {{}},
        setError: text => errors.push(text), errorText: window.EasyICU.guidedPi.require('errorText')
          .create({{tr: (en, zh) => zh, staticPreview: () => false}}).errorText,
      }};
      const owner = window.EasyICU.guidedPi.require('hostJobs').create(host);
      const tick = () => new Promise(resolve => setTimeout(resolve, 5));
      (async () => {{
        owner.namedDemo(); await tick(); owner.namedDemo(); await tick();
        const named = owner.namedDemo();
        const html = consent.render(session, {{tr: host.tr, esc: EU_HTML.esc, icon: () => '', namedDemo: named}});
        const used = await owner.useNamedDemo('eicu_demo_v2_0_1');
        owner.stopAll();
        console.log(JSON.stringify({{named, html, requested, prepared, used, errors}}));
      }})().catch(error => {{console.error(error); process.exitCode = 1;}});
    """
    return json.loads(run_node("node", script, check=True).stdout)


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_hosted_demo_uses_the_exact_registered_export_even_if_the_catalog_is_not_ready():
    result = _card(recommended={
        "source_id": "src_demo", "database": "eicu_demo", "label": "eICU Demo v2.0.1", "stays": 12,
    })
    assert result["requested"] == [["s1", "p1", "eicu_demo"]]
    assert result["named"]["prepared"] is True
    assert "eICU Demo v2.0.1" in result["html"]
    assert "12 例 ICU 住院" in result["html"]
    assert 'data-gpi-data-source-action="use_registered_source"' in result["html"]
    assert 'data-gpi-data-source-database="eicu_demo"' in result["html"]
    assert 'data-gpi-data-source-id="src_demo"' in result["html"]
    assert "data-gpi-named-demo" not in result["html"]
    assert 'begin_local_selection" data-gpi-data-source-database="eicu_demo"' not in result["html"]
    assert result["prepared"] == []
    assert result["used"] is False


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_hosted_demo_without_a_registered_export_names_the_administrator_even_if_cached():
    result = _card(export_ready=True)
    assert result["named"]["prepared"] is False
    assert "这台服务器上的 Demo 数据由管理员准备。" in result["html"]
    assert "data-gpi-named-demo" not in result["html"]
    assert "use_registered_source" not in result["html"]
    assert result["prepared"] == []


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_tied_hosted_demo_exports_are_offered_without_preparation():
    choices = [
        {"source_id": f"src_{index}", "database": "eicu_demo", "label": "eICU Demo v2.0.1", "stays": 12}
        for index in (1, 2)
    ]
    result = _card(choices=choices)
    assert result["named"]["prepared"] is True
    assert all(f'data-gpi-data-source-id="{row["source_id"]}"' in result["html"] for row in choices)
    assert "data-gpi-named-demo" not in result["html"]
    assert result["prepared"] == []


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_desktop_demo_keeps_the_preparation_action():
    result = _card(hosted=False)
    assert 'data-gpi-named-demo="eicu_demo_v2_0_1"' in result["html"]
    assert result["prepared"] == ["eicu_demo_v2_0_1"]
    assert result["used"] is True
    assert "12 例 ICU 住院" in result["html"]


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_prepare_action_refused_with_403_reports_the_typed_visible_error():
    result = _card(hosted=False, forbidden=True)
    assert result["used"] is False
    assert result["errors"][-1] == "这项操作由管理员负责，请联系管理员。"
    # The session view renders the same error state as an alert outside the card.
    view = _read("js/screens-guided-pi-session-view.js")
    assert 'state.error ? `<div class="gpi-error" role="alert"><span>${esc(state.error)}</span>' in view


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_legacy_demo_sharing_a_full_database_key_never_offers_the_full_export():
    result = _card(database="miiv", title="MIMIC-IV Demo", recommended={
        "source_id": "src_full", "label": "MIMIC-IV v3.1", "stays": 40,
    })
    assert "src_full" not in result["html"]
    assert "use_registered_source" not in result["html"]
    assert result["prepared"] == []


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_named_demo_card_uses_its_profile_reader_label():
    result = _card(hosted=False, title="eICU Collaborative Research Database Demo",
                   reader_label="eICU Demo v2.0.1", stays=">2500")
    assert "eICU Demo v2.0.1" in result["html"]
    assert "Collaborative Research Database" not in result["html"]
    assert "&gt;2,500 例 ICU 住院" in result["html"]
