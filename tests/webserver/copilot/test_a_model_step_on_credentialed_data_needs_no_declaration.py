"""A model step on credentialed data needs no data-handling declaration.

A declaration (on-device, zero retention, the user's own acceptance) is
recorded with the connection, not required: an undeclared connection calls
the model for MIMIC-IV or eICU like any other source (owner decision
2026-10-27).  So the plan card offers its own approval, 生成候选计划, and no
card or error sends the user to declare data handling.  What content may
leave is unchanged, and an unknown source identity still fails closed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.providers.client_trust import (
    UNKNOWN_SOURCE_DATABASE,
    DataHandling,
    ProviderConfigurationError,
    require_credentialed_data_clearance,
    require_data_handling,
)
from tests.support.node import run_node

STATIC = Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "static"
JS = STATIC / "js"

_HARNESS = r"""
const path = require('path');
const [jsDir, lang, input] = process.argv.slice(1, 4);
global.window = { EU_LANG: lang };
const registry = {};
window.EasyICU = { guidedPi: {
  declare: (name, api) => { registry[name] = api; },
  optional: name => registry[name] || null,
  require: name => { if (!registry[name]) throw new Error('missing owner ' + name); return registry[name]; },
} };
require(path.join(jsDir, 'html-escape.js'));
require(path.join(jsDir, 'screens-guided-pi-confirmation.js'));
const tr = (en, zh) => (lang === 'zh' ? zh : en) || en;
const INPUT = JSON.parse(input);
const session = { session_id: 'session-1', binding: { study_context_id: 'study-1', study_revision: 2 } };
const owner = registry.confirmation.create({
  tr, esc: window.EU_HTML.esc, iconHtml: () => '', resourceButton: () => '',
  sessionIsStale: () => false, busy: () => false,
  session: () => session, workflow: () => ({ next_action_code: INPUT.code }),
});
const spec = owner.workflowConfirmation();
process.stdout.write(JSON.stringify({ grants: spec && spec.grants, html: owner.workflowConfirmationHtml() }));
"""


def _card(code: str, *, lang: str = "zh") -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(node, _HARNESS, str(JS), lang, json.dumps({"code": code}), check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


@pytest.mark.parametrize("database", ["miiv", "eicu", "mimic", "sic", "aumc", "hirid"])
def test_an_undeclared_connection_may_call_the_model_for_credentialed_data(database: str) -> None:
    require_data_handling(database, DataHandling.UNSPECIFIED)
    require_credentialed_data_clearance(object(), database)


def test_an_unknown_source_identity_still_fails_closed() -> None:
    with pytest.raises(ProviderConfigurationError) as raised:
        require_data_handling("no_such_database", DataHandling.UNSPECIFIED)
    assert raised.value.issue == UNKNOWN_SOURCE_DATABASE


def test_the_plan_card_generates_the_candidate_plan_directly() -> None:
    card = _card("provider_ready_to_generate_plan")
    assert "provider_run" in card["grants"]
    assert 'data-gpi-confirm-action>生成候选计划</button>' in card["html"]
    assert "声明数据处理方式" not in card["html"] and "data-gpi-config" not in card["html"]
    english = _card("provider_ready_to_generate_plan", lang="en")
    assert 'data-gpi-confirm-action>Generate candidate plan</button>' in english["html"]
    assert "Declare data handling" not in english["html"]


def test_no_page_sends_the_user_to_declare_data_handling() -> None:
    assert not (JS / "screens-guided-pi-data-clearance.js").exists()
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "screens-guided-pi-data-clearance.js" not in index
    for name in ("screens-guided-pi.js", "screens-guided-pi-confirmation.js",
                 "screens-guided-pi-error-text.js", "screens-guided-pi-model-menu.js"):
        source = (JS / name).read_text(encoding="utf-8")
        assert "provider_not_cleared_for_credentialed_data" not in source, name
        assert "dataClearance" not in source, name
