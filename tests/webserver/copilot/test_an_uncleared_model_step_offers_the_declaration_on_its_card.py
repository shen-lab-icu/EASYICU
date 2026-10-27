"""An uncleared model step offers the declaration on its card, not an error.

The host projects ``session.data_clearance``: whether the credentialed-data
gate would refuse the conversation's next model step.  A card whose approval
would call the model then names the missing declaration and opens the
connection settings instead of offering a button that can only fail.  Cards
that do not call the model, and cleared sessions, keep their own approval.
A refusal that still reaches a send opens the settings instead of an error
banner.  The gate itself is unchanged and stays at every model call.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.support.node import run_node

STATIC = Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "static"
JS = STATIC / "js"
_FILES = ("screens-guided-pi-data-clearance.js", "screens-guided-pi-confirmation.js")
_BLOCKED = {"status": "blocked", "reason_code": "provider_not_cleared_for_credentialed_data", "declarable": True}

_HARNESS = r"""
const path = require('path');
const [jsDir, lang, files, input] = process.argv.slice(1, 5);
global.window = { EU_LANG: lang };
const registry = {};
window.EasyICU = { guidedPi: {
  declare: (name, api) => { registry[name] = api; },
  optional: name => registry[name] || null,
  require: name => { if (!registry[name]) throw new Error('missing owner ' + name); return registry[name]; },
} };
require(path.join(jsDir, 'html-escape.js'));
for (const file of JSON.parse(files)) require(path.join(jsDir, file));
const tr = (en, zh) => (lang === 'zh' ? zh : en) || en;
const esc = window.EU_HTML.esc;
const INPUT = JSON.parse(input);
const session = { session_id: 'session-1', binding: { study_context_id: 'study-1', study_revision: 2 },
  data_clearance: INPUT.clearance };
const owner = registry.confirmation.create({
  tr, esc, iconHtml: () => '', resourceButton: () => '',
  sessionIsStale: () => false, busy: () => false,
  session: () => session, workflow: () => ({ next_action_code: INPUT.code }),
});
const spec = owner.workflowConfirmation();
process.stdout.write(JSON.stringify({ grants: spec && spec.grants, html: owner.workflowConfirmationHtml() }));
"""


def _card(code: str, clearance: dict[str, Any] | None, *, lang: str = "zh") -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(
        node, _HARNESS, str(JS), lang, json.dumps(list(_FILES)),
        json.dumps({"code": code, "clearance": clearance}), check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_a_blocked_plan_card_offers_the_declaration_instead_of_its_approval() -> None:
    card = _card("provider_ready_to_generate_plan", _BLOCKED)
    assert "provider_run" in card["grants"]
    assert "data-gpi-confirm-action" not in card["html"]
    assert "开始生成研究计划" not in card["html"]
    assert 'data-gpi-config>声明数据处理方式</button>' in card["html"]
    assert "当前模型还没有声明能否处理 MIMIC-IV、eICU 等需授权的数据。" in card["html"]
    english = _card("provider_ready_to_generate_plan", _BLOCKED, lang="en")
    assert "Declare data handling" in english["html"]


def test_an_account_connection_is_sent_to_change_the_connection() -> None:
    card = _card("provider_ready_to_generate_plan", {**_BLOCKED, "declarable": False})
    assert 'data-gpi-config>更换模型连接</button>' in card["html"]
    assert "声明数据处理方式" not in card["html"]


@pytest.mark.parametrize("clearance", [None, {"status": "cleared"}, {"status": "unknown"}])
def test_a_cleared_or_unread_session_keeps_the_approval(clearance) -> None:
    card = _card("provider_ready_to_generate_plan", clearance)
    assert "data-gpi-confirm-action" in card["html"]
    assert "开始生成研究计划" in card["html"]
    assert "data-gpi-clearance" not in card["html"]


def test_a_card_that_does_not_call_the_model_is_not_changed() -> None:
    card = _card("plan_ready", _BLOCKED)
    assert "provider_run" not in card["grants"]
    assert "data-gpi-confirm-action" in card["html"]
    assert "data-gpi-clearance" not in card["html"]


def test_a_refused_send_opens_the_settings_instead_of_an_error_banner() -> None:
    host = (JS / "screens-guided-pi.js").read_text(encoding="utf-8")
    assert (
        "if (clearance && clearance.isRefusal(error)) { state.error = ''; state.showSetup = true; }\n"
        "      else state.error = errorText(error);"
    ) in host
    owner = (JS / "screens-guided-pi-data-clearance.js").read_text(encoding="utf-8")
    assert "const REFUSAL = 'provider_not_cleared_for_credentialed_data';" in owner
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    assert index.index("js/screens-guided-pi-data-clearance.js?v=20261027-clr1") < index.index(
        "js/screens-guided-pi.js?v=20261027-clr1"
    )
