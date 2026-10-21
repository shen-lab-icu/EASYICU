"""An ended hosted session offers the way back to the login page.

On a hosted instance the login gateway answers an /api/ call whose session
ended with 401 and the code ``login_required``. The page does not leave at
once: one notice says the sign-in expired and links to /login, so a question
the clinician is still typing can be copied first. Any other refusal keeps
its own message and shows no notice, and the conversation's error owner
names the expiry in plain words.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node

STATIC = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "webserver" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


_SCRIPT = r"""
const appended = [];
function element(tag) {
  const node = {
    tag, className: '', textContent: '', href: '', attributes: {}, children: [], isConnected: false,
    setAttribute(name, value) { this.attributes[name] = String(value); },
    append(...items) { this.children.push(...items); },
  };
  return node;
}
global.window = { t: (en, zh) => zh };
global.document = {
  readyState: 'loading',
  addEventListener: () => {},
  querySelector: () => null,
  createElement: element,
  body: { appendChild(node) { node.isConnected = true; appended.push(node); } },
};
const [status, code] = [Number(process.argv[3]), process.argv[4]];
global.fetch = async () => ({
  ok: false, status, json: async () => ({ detail: { error: code } }),
});
require(process.argv[1]);
require(process.argv[2]);
(async () => {
  const errors = [];
  for (let i = 0; i < 2; i += 1) {
    try { await window.EU_API.getJSON('/api/capabilities'); } catch (err) { errors.push(err.code); }
  }
  process.stdout.write(JSON.stringify({
    errors,
    notices: appended.map(node => ({
      className: node.className,
      role: node.attributes.role,
      text: node.children.map(child => child.textContent),
      href: node.children.map(child => child.href).filter(Boolean),
    })),
  }));
})();
"""


def _run(status: int, code: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(
        node,
        _SCRIPT,
        str(STATIC / "js" / "hosted-session.js"),
        str(STATIC / "js" / "api.js"),
        str(status),
        code,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


@pytest.mark.requires_node
def test_a_login_required_reply_shows_one_notice_with_the_login_link() -> None:
    outcome = _run(401, "login_required")

    assert outcome["errors"] == ["login_required", "login_required"]
    assert outcome["notices"] == [{
        "className": "eu-session-expired",
        "role": "alert",
        "text": ["登录已过期。还没发送的内容请先复制。", "重新登录"],
        "href": ["/login"],
    }]


@pytest.mark.requires_node
@pytest.mark.parametrize(("status", "code"), [(401, "unauthorized"), (403, "login_required"), (403, "unavailable_when_hosted")])
def test_other_refusals_keep_their_message_and_show_no_notice(status: int, code: str) -> None:
    outcome = _run(status, code)

    assert outcome["errors"] == [code, code]
    assert outcome["notices"] == []


_ERROR_TEXT = r"""
let owner = null;
global.window = {
  EU_HTML: { esc: value => String(value) },
  EasyICU: { guidedPi: { declare: (name, api) => { if (name === 'errorText') owner = api; }, optional: () => null } },
};
require(process.argv[1]);
const text = owner.create({ tr: (en, zh) => zh, staticPreview: () => false }).errorText;
process.stdout.write(JSON.stringify(text(Object.assign(new Error('login_required'), { status: 401, code: 'login_required' }))));
"""


@pytest.mark.requires_node
def test_the_conversation_names_the_expiry_in_plain_words() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(node, _ERROR_TEXT, str(STATIC / "js" / "screens-guided-pi-error-text.js"), check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == "登录已过期，请重新登录后继续。"


def test_the_session_owner_loads_before_the_transport_and_has_its_style() -> None:
    index = _read("index.html")

    assert index.index("js/hosted-session.js?v=") < index.index("js/api.js?v=")
    assert ".eu-session-expired{" in _read("css/shell.css")
