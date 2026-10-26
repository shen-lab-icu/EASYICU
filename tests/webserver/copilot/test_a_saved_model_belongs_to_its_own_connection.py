"""The connection form names the saved model only on the saved connection.

With a verified DeepSeek connection saved, the form on DeepSeek shows its
model and offers to reuse its credential. Switched to another service, it does
neither: that model and key belong to DeepSeek. Runtime readiness codes are
not listed on the connection page.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    STATIC as STATIC,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
)

_SCRIPT = r"""
global.window = { EU_LANG: 'zh' };
require(process.argv[1]);
const provider = window.EasyICU.guidedPi.require('provider');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const option = (value, selected, label) => `<option value="${esc(value)}"${value === selected ? ' selected' : ''}>${esc(label)}</option>`;
const config = {
  provider: 'deepseek', base_url: 'https://api.deepseek.com/v1', api_transport: 'openai-completions',
  model: 'synthetic-saved-model', credential_present: true, connection_verified: true,
};
const render = draft => provider.renderConnectionBody({
  state: { researchProvider: 'api', apiDraft: draft, error: '' },
  runtime: { configuration: config, blockers: ['runtime_integrity_verified'] }, config,
  tr: (en, zh) => zh, esc, option, providerPreset: () => 'deepseek', connectionConfigured: true, staticPreview: false,
});
const read = markup => ({
  model: markup.includes('synthetic-saved-model'),
  reuse: markup.includes('留空则沿用已保存的凭据'),
  blockers: markup.includes('连接前请处理') || markup.includes('runtime_integrity_verified'),
});
const other = provider.PRESETS.openrouter;
process.stdout.write(JSON.stringify({
  saved: read(render(null)),
  other: read(render({ preset: 'openrouter', provider: other.provider, base_url: other.base_url, api_transport: other.api_transport })),
}));
"""


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_the_saved_model_and_credential_stay_with_their_connection() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    out = json.loads(run_node(node, _SCRIPT, str(STATIC / "js" / "screens-guided-pi-provider.js"), check=True).stdout)

    assert out["saved"] == {"model": True, "reuse": True, "blockers": False}
    assert out["other"] == {"model": False, "reuse": False, "blockers": False}
