"""A hosted card naming the MIMIC-IV demo offers its registered export (pilot2 P1, 54420219b).

That demo shares the full database's key ``miiv``. The card ignored the
host's offer for that key and showed the administrator hint although the
demo was registered. It now reads the host's ``official_demo`` entry for this
exact release, and still never offers the full database's export.
"""

from __future__ import annotations

import json

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _ESCAPE_OWNER as _ESCAPE_OWNER,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read as _read,
)

DEMO_ID = "mimic_iv_demo_v2_2"
FULL = {"source_id": "src_full", "label": "MIMIC-IV v3.1", "stays": 90000}
DEMO_EXPORT = {"source_id": "src_demo", "label": "MIMIC-IV Clinical Database Demo v2.2", "stays": 140}


def _card(payload: dict) -> dict:
    script = f"""
      global.window = global;
      window.EU_CAPABILITIES = {{deployment: {{hosted: true}}}};
      eval({_ESCAPE_OWNER!r});
      eval({_read('js/screens-guided-pi-hosted.js')!r});
      eval({_read('js/screens-guided-pi-error-text.js')!r});
      eval({_read('js/screens-guided-pi-host-jobs.js')!r});
      eval({_read('js/screens-guided-pi-data-consent.js')!r});
      const consent = window.EasyICU.guidedPi.require('dataConsent');
      const session = {{session_id: 's1', data_source_authorization: {{status: 'pending'}}}};
      const requested = [], prepared = [];
      const host = {{tr: (en, zh) => zh, dataConsent: consent,
        api: () => ({{
          loadOfficialDemoSources: async () => ({{sources: [{{
            id: {json.dumps(DEMO_ID)}, database: 'miiv', title: 'MIMIC-IV Clinical Database Demo', version: '2.2',
            scope: {{icu_stays: 140}}, status: {{export_ready: false}}
          }}]}}),
          loadPiCopilotRegisteredSourceOffer: async (...args) => {{ requested.push(args); return {json.dumps(payload)}; }},
          startOfficialDemoSourcePrepare: async id => {{ prepared.push(id); return {{job_id: 'j1'}}; }},
        }}),
        session: () => session, projectId: () => 'p1', busy: () => false,
        messages: () => [{{role: 'user', text: 'Use MIMIC-IV Clinical Database Demo v2.2 to describe ICU stays'}}],
        render() {{}}, workflowReceipts: () => [], setWorkflowReceipts() {{}},
        setError() {{}}, errorText: window.EasyICU.guidedPi.require('errorText')
          .create({{tr: (en, zh) => zh, staticPreview: () => false}}).errorText,
      }};
      const owner = window.EasyICU.guidedPi.require('hostJobs').create(host);
      const tick = () => new Promise(resolve => setTimeout(resolve, 5));
      (async () => {{
        owner.namedDemo(); await tick(); owner.namedDemo(); await tick();
        const named = owner.namedDemo();
        const html = consent.render(session, {{tr: host.tr, esc: EU_HTML.esc, icon: () => '', namedDemo: named}});
        owner.stopAll();
        console.log(JSON.stringify({{named, html, requested, prepared}}));
      }})().catch(error => {{console.error(error); process.exitCode = 1;}});
    """
    return json.loads(run_node("node", script, check=True).stdout)


def _offer(*, demo_id=DEMO_ID, recommended=DEMO_EXPORT, choices=()):
    return {"database": "miiv", "recommended": FULL, "choices": [], "official_demo": {
        "id": demo_id, "database": "miiv", "label": "MIMIC-IV Clinical Database Demo v2.2",
        "registered_offer": {"state": "ready", "recommended": recommended, "choices": list(choices)},
    }}


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_the_hosted_mimic_iv_demo_card_offers_its_registered_export():
    result = _card(_offer())

    assert result["requested"] == [["s1", "p1", "miiv"]]
    assert result["named"]["prepared"] is True
    assert 'data-gpi-data-source-action="use_registered_source"' in result["html"]
    assert 'data-gpi-data-source-database="miiv"' in result["html"]
    assert 'data-gpi-data-source-id="src_demo"' in result["html"]
    assert "src_full" not in result["html"]
    assert "由管理员准备" not in result["html"]
    assert result["prepared"] == []


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_tied_demo_exports_are_offered_without_the_full_data():
    choices = [{**DEMO_EXPORT, "source_id": f"src_demo_{index}"} for index in (1, 2)]
    result = _card(_offer(recommended=None, choices=choices))

    assert all(f'data-gpi-data-source-id="{row["source_id"]}"' in result["html"] for row in choices)
    assert "src_full" not in result["html"]


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
@pytest.mark.parametrize("payload", (
    _offer(demo_id="another_release"),
    {"database": "miiv", "recommended": FULL, "choices": []},
))
def test_without_this_releases_offer_the_card_names_the_administrator(payload):
    result = _card(payload)

    assert result["named"]["prepared"] is False
    assert "这台服务器上的 Demo 数据由管理员准备。" in result["html"]
    assert "use_registered_source" not in result["html"]
    assert "src_full" not in result["html"]
