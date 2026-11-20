"""Switching the model connection starts a conversation on the chosen one.

A conversation keeps the connection it was created with (its
``research_provider`` binding; the server routes every turn through it). The
setup panel let a researcher switch from a saved API connection to the
ChatGPT account and finish, while the open conversation stayed on the API
connection: the next turns went out on the model and account they had just
left (UX zh, 2026-11-20). Finishing setup on a different connection now starts
a conversation on it; the earlier one stays in the project. Fakes only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"

_SCRIPT = r"""
const path = require('node:path');
const dir = process.argv[1];
const scenario = JSON.parse(process.argv[2]);
global.window = { EU_LANG: 'zh' };
require(path.join(dir, 'screens-guided-pi-modules.js'));
require(path.join(dir, 'screens-guided-pi-events.js'));
let onClick;
const state = {
  host: { addEventListener: (type, fn) => { if (type === 'click') onClick = fn; } },
  session: scenario.session, researchProvider: scenario.from, researchModel: 'account-model',
  showSetup: true, codexAuth: { authentication_verified: true }, error: '',
};
let created = 0;
const owner = window.EasyICU.guidedPi.require('events').create({
  state, render() {}, projectId: () => 'draft_x', tr: (en, zh) => zh,
  createSession: async () => {
    created += 1;
    state.session = { session_id: 'pi_new', research_provider: {
      provider: state.researchProvider === 'codex' ? 'codex' : 'openai' } };
  },
  finishProviderSetup: async () => { state.showSetup = false; },
  apiResearchReady: () => true, loadCodexResearchStatus: () => {},
});
owner.wire();
const click = (selector, dataset = {}) => onClick({
  target: { closest: s => (s === selector ? { dataset, closest: () => null } : null) },
  preventDefault() {},
});
(async () => {
  click('[data-gpi-research-provider]', { gpiResearchProvider: scenario.to });
  click('[data-gpi-provider-done]');
  await new Promise(resolve => setTimeout(resolve, 20));
  process.stdout.write(JSON.stringify({
    created,
    session: state.session && state.session.session_id,
    provider: state.session && state.session.research_provider.provider,
  }));
})();
"""


def _finish_setup(session, *, from_provider: str, to_provider: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    scenario = {"session": session, "from": from_provider, "to": to_provider}
    result = run_node(
        node,
        _SCRIPT,
        str((STATIC / "js").resolve()),
        json.dumps(scenario),
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def _session(provider: str) -> dict:
    return {"session_id": "pi_old", "research_provider": {"provider": provider}}


@pytest.mark.parametrize(
    ("bound", "from_provider", "to_provider", "expected"),
    [
        ("openai", "api", "codex", "codex"),
        ("codex", "codex", "api", "openai"),
    ],
)
def test_finishing_setup_on_another_connection_starts_a_conversation_on_it(
    bound, from_provider, to_provider, expected
) -> None:
    after = _finish_setup(
        _session(bound), from_provider=from_provider, to_provider=to_provider
    )

    assert after == {"created": 1, "session": "pi_new", "provider": expected}


@pytest.mark.parametrize(("bound", "selected"), [("openai", "api"), ("codex", "codex")])
def test_finishing_setup_on_the_same_connection_keeps_the_conversation(
    bound, selected
) -> None:
    after = _finish_setup(_session(bound), from_provider=selected, to_provider=selected)

    assert after == {"created": 0, "session": "pi_old", "provider": bound}


def test_finishing_setup_without_a_conversation_still_starts_one() -> None:
    after = _finish_setup(None, from_provider="api", to_provider="codex")

    assert after == {"created": 1, "session": "pi_new", "provider": "codex"}


def test_a_conversation_sends_every_turn_on_its_own_connection(tmp_path) -> None:
    # The server side of the contract: the turn's gateway comes from the
    # conversation's own binding, never from the setup panel's selection.
    from types import SimpleNamespace

    from easyicu.webserver.pi_copilot.contracts import ResearchProviderBinding
    from easyicu.webserver.pi_copilot.service import PiCopilotService

    service = PiCopilotService.__new__(PiCopilotService)
    service.gateway = object()
    pooled = object()
    service.codex_gateway_pool = SimpleNamespace(
        gateway_for=lambda binding, **_: pooled
    )
    api = ResearchProviderBinding(
        provider="openai",
        credential_source="pi_verified",
        authentication_mode="api_key",
        model="api-model",
    )

    assert (
        service._conversation_gateway_for_binding(api, unified=True) is service.gateway
    )
