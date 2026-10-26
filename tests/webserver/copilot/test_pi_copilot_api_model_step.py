"""An API connection lists the service's models before the researcher picks one.

The form asks for the service, credential and (for a proxy or gateway) its
address; connecting reads the endpoint's model catalog through the opt-in
discovery route; only then is a model chosen and verified-and-saved. The
credential survives the form's re-renders in page memory, never in markup,
and a changed service or address invalidates the listed models.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    STATIC as STATIC,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read as _read,
)


_FLOW_SCRIPT = r"""
global.window = { EU_LANG: 'zh' };
require(process.argv[1]);
require(process.argv[2]);
const provider = window.EasyICU.guidedPi.require('provider');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const option = (value, selected, label) => `<option value="${esc(value)}"${value === selected ? ' selected' : ''}>${esc(label)}</option>`;
const tr = (en, zh) => zh;

// A tiny form: fields by name, parsed back from the rendered markup.
function parseForm(markup) {
  const fields = {};
  for (const tag of markup.matchAll(/<input name="([a-z_]+)"([^>]*)>/g)) {
    const value = tag[2].match(/value="([^"]*)"/);
    fields[tag[1]] = { value: value ? value[1] : '' };
  }
  const transport = markup.match(/<select name="api_transport">[\s\S]*?<option value="([^"]+)" selected/);
  if (transport) fields.api_transport = { value: transport[1] };
  const model = markup.match(/<select name="model" required>([\s\S]*?)<\/select>/);
  if (model) {
    const chosen = model[1].match(/<option value="([^"]*)" selected/);
    fields.model = { value: chosen ? chosen[1] : '' };
  }
  const preset = markup.match(/<select data-gpi-provider-preset>[\s\S]*?<option value="([^"]+)" selected/);
  const presetField = { value: preset ? preset[1] : '', matches: sel => sel === '[data-gpi-provider-preset]' };
  const form = {
    fields,
    elements: { namedItem: name => fields[name] || null },
    querySelector(sel) {
      if (sel === '[data-gpi-provider-preset]') return presetField;
      const named = sel.match(/^\[name="([a-z_]+)"\]$/);
      if (named) return fields[named[1]] || null;
      return null;
    },
  };
  return { form, presetField };
}
global.FormData = class {
  constructor(form) { this.form = form; }
  get(name) { return this.form.fields[name] ? this.form.fields[name].value : null; }
  has(name) { return Object.prototype.hasOwnProperty.call(this.form.fields, name); }
};

const state = { researchProvider: 'api', setupSaving: false, error: '', runtime: { configuration: {} } };
const calls = { discover: [], save: [] };
let discoverReply = { ok: true, models: ['m-large', 'm-small', '<b>x</b>'], recommended: 'm-small' };
const api = {
  discoverPiCopilotProviderModels: async body => { calls.discover.push(body); return discoverReply; },
  savePiCopilotProviderConfig: async body => { calls.save.push(body); return { runtime: { status: 'ready' } }; },
};
let current = null;   // the form the host shows
const host = { querySelector: sel => (sel === '[data-gpi-provider-form]' ? current && current.form : null) };
state.host = host;
let control;
let markup = '';
const options = () => ({
  state, runtime: state.runtime, config: state.runtime.configuration || {}, runtimeMissing: [], tr, esc, option,
  providerPreset: () => 'cliproxyapi', connectionConfigured: false, staticPreview: false,
});
function render() {
  control.captureForm(host);
  markup = provider.renderConnectionBody(options());
  current = parseForm(markup);
  control.restoreForm(host);
}
control = window.EasyICU.guidedPi.require('providerControl').create({
  state, api: () => api, tr, render, runtimeReady: () => true, shellReady: () => true,
  connectionConfigured: () => false, connectionReady: () => false, errorText: e => String((e && e.code) || e),
});
const out = {};
const pick = key => {
  current.presetField.value = key;
  control.handleChange({ target: { value: key, closest: () => current.form, matches: sel => sel === '[data-gpi-provider-preset]' } });
};
const beforeAdvanced = () => markup.split('<details class="gpi-api-advanced">')[0];
render();
// A first visit has no saved connection: the host's built-in proxy default
// is not the researcher's choice, so the form opens on DeepSeek.
out.firstVisit = {
  deepseekFirstAndSelected: /<select data-gpi-provider-preset><optgroup label="[^"]+"><option value="deepseek" selected>/.test(markup),
  address: current.form.fields.base_url.value,
  provider: current.form.fields.provider.value,
  addressUnderAdvanced: !beforeAdvanced().includes('name="base_url"'),
};
// Services on this computer and institutional clouds show their address.
pick('ollama');
out.local = { address: current.form.fields.base_url.value, visible: beforeAdvanced().includes('name="base_url"'), hint: markup.includes('它不校验凭据') };
pick('azure');
out.azure = { address: current.form.fields.base_url.value, visible: beforeAdvanced().includes('name="base_url"'), hint: markup.includes('模型请选择你的部署名') };
pick('cliproxyapi');
out.first = {
  connectButton: markup.includes('连接并读取模型'),
  noModelField: !markup.includes('name="model"'),
  noPresetModel: !/gpt-|claude-|gemini-|deepseek-chat/.test(markup),
  proxyAddressVisible: markup.split('<details class="gpi-api-advanced">')[0].includes('name="base_url"'),
  proxyHint: markup.includes('CLIProxyAPI 默认地址为 http://127.0.0.1:8317/v1'),
  notice: markup.includes('连接时会用这个凭据读取该服务的模型列表'),
};
(async () => {
  current.form.fields.api_key.value = 'sk-synthetic-secret';
  await control.configureProvider(current.form);
  out.discoverCall = calls.discover[0];
  out.savedBeforeChoice = calls.save.length;
  out.listed = {
    select: markup.includes('<select name="model" required>'),
    recommendedSelected: markup.includes('<option value="m-small" selected>m-small · 推荐</option>'),
    escaped: markup.includes('&lt;b&gt;x&lt;/b&gt;') && !markup.includes('<b>x</b>'),
    count: markup.includes('该服务返回了 3 个模型'),
    saveButton: markup.includes('验证并保存'),
    rediscover: markup.includes('data-gpi-provider-rediscover'),
    keyKept: current.form.fields.api_key.value === 'sk-synthetic-secret',
    keyNotInMarkup: !markup.includes('sk-synthetic-secret'),
  };
  // A changed address belongs to another endpoint: the list goes.
  current.form.fields.base_url.value = 'http://127.0.0.1:9000/v1';
  control.handleChange({ target: { closest: () => current.form, matches: sel => sel.includes('[name="base_url"]') } });
  out.addressChangeClearsList = !markup.includes('name="model"') && markup.includes('连接并读取模型');
  current.form.fields.base_url.value = 'http://127.0.0.1:8317/v1';
  await control.configureProvider(current.form);
  current.form.fields.model.value = 'm-large';
  await control.configureProvider(current.form);
  out.saveCall = calls.save[0];
  out.afterSave = { keyCleared: current.form.fields.api_key.value === '', draftHoldsNoKey: !JSON.stringify(state.apiDraft).includes('sk-'), catalog: state.modelCatalog };
  // Closing the form discards its draft, credential and list.
  const open = current; current = null; control.captureForm(host); current = open;
  out.closed = { draft: state.apiDraft, catalog: state.modelCatalog };
  control.restoreForm(host);
  out.closed.keyRestored = current.form.fields.api_key.value !== '';

  // A changed service clears the credential typed for the previous one.
  current.form.fields.api_key.value = 'sk-other-secret';
  current.presetField.value = 'deepseek';
  control.handleChange({ target: { value: 'deepseek', closest: () => current.form, matches: sel => sel === '[data-gpi-provider-preset]' } });
  out.serviceChange = {
    keyCleared: current.form.fields.api_key.value === '',
    addressUnderAdvanced: !markup.split('<details class="gpi-api-advanced">')[0].includes('name="base_url"')
      && markup.split('<details class="gpi-api-advanced">')[1].includes('value="https://api.deepseek.com/v1"'),
  };

  // The example address is never contacted.
  current.presetField.value = 'custom-openai';
  control.handleChange({ target: { value: 'custom-openai', closest: () => current.form, matches: sel => sel === '[data-gpi-provider-preset]' } });
  current.form.fields.base_url.value = 'https://llm-gateway.example/v1';
  current.form.fields.api_key.value = 'sk-synthetic-secret';
  const before = calls.discover.length;
  await control.configureProvider(current.form);
  out.exampleRefused = { calls: calls.discover.length - before, keyCleared: current.form.fields.api_key.value === '', error: Boolean(state.error) };

  // An empty catalog is said plainly and offers no model to save.
  discoverReply = { ok: true, models: [], recommended: null };
  current.form.fields.base_url.value = 'https://gateway.synthetic.test/v1';
  await control.configureProvider(current.form);
  out.emptyCatalog = { note: markup.includes('该服务没有返回任何模型'), noSelect: !markup.includes('name="model"'), saves: calls.save.length };
  process.stdout.write(JSON.stringify(out));
})().catch(err => { console.error(err); process.exitCode = 1; });
"""


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_connecting_lists_models_then_the_choice_is_verified_and_saved() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    completed = run_node(
        node, _FLOW_SCRIPT,
        str(STATIC / "js" / "screens-guided-pi-provider.js"),
        str(STATIC / "js" / "screens-guided-pi-provider-control.js"),
        check=True,
    )
    out = json.loads(completed.stdout)

    assert out["firstVisit"] == {
        "deepseekFirstAndSelected": True, "address": "https://api.deepseek.com/v1",
        "provider": "deepseek", "addressUnderAdvanced": True,
    }
    assert out["local"] == {"address": "http://127.0.0.1:11434/v1", "visible": True, "hint": True}
    assert out["azure"] == {"address": "", "visible": True, "hint": True}
    assert out["first"] == {
        "connectButton": True, "noModelField": True, "noPresetModel": True,
        "proxyAddressVisible": True, "proxyHint": True, "notice": True,
    }
    assert out["discoverCall"] == {
        "provider": "easyicu-local", "base_url": "http://127.0.0.1:8317/v1",
        "api_transport": "openai-completions", "api_key": "sk-synthetic-secret", "enable_ai": True,
    }
    assert out["savedBeforeChoice"] == 0
    assert all(out["listed"].values()), out["listed"]
    assert out["addressChangeClearsList"] is True
    assert out["saveCall"] == {
        "provider": "easyicu-local", "base_url": "http://127.0.0.1:8317/v1",
        "api_transport": "openai-completions", "api_key": "sk-synthetic-secret",
        "model": "m-large", "enable_ai": True,
    }
    assert out["afterSave"] == {"keyCleared": True, "draftHoldsNoKey": True, "catalog": None}
    assert out["closed"] == {"draft": None, "catalog": None, "keyRestored": False}
    assert out["serviceChange"] == {"keyCleared": True, "addressUnderAdvanced": True}
    assert out["exampleRefused"] == {"calls": 0, "keyCleared": True, "error": True}
    assert out["emptyCatalog"] == {"note": True, "noSelect": True, "saves": 1}


def test_presets_and_form_events_live_with_the_provider_owners() -> None:
    events = _read("js/screens-guided-pi-events.js")
    owner = _read("js/screens-guided-pi.js")
    provider = _read("js/screens-guided-pi-provider.js")
    control = _read("js/screens-guided-pi-provider-control.js")

    # The event wiring delegates; the preset table is the renderer's.
    assert "presets = {" not in events
    assert "if (PROVIDER_CONTROL && PROVIDER_CONTROL.handleChange(event)) return;" in events
    assert "if (PROVIDER_CONTROL && PROVIDER_CONTROL.handleClick(event)) return;" in events
    assert "const PRESETS = Object.freeze({" in provider
    # The credential is captured before and restored after every render.
    assert "PROVIDER_CONTROL.captureForm(state.host);" in owner
    assert "PROVIDER_CONTROL.restoreForm(state.host);" in owner
    assert "localStorage" not in control and "sessionStorage" not in control
    assert "state.apiKey" not in control


def test_every_connection_error_reads_in_the_interface_language() -> None:
    copy = _read("js/screens-guided-pi-error-text.js")
    for code in (
        "pi_provider_credential_required", "pi_provider_request_invalid",
        "pi_provider_verification_failed", "pi_provider_models_response_invalid",
        "pi_provider_base_url_rejected", "pi_provider_api_transport_unsupported",
        "external_llm_opt_in_required", "pi_provider_auth_failed",
        "pi_provider_connection_failed",
    ):
        branch = copy.split(f"error.code === '{code}'", 1)
        assert len(branch) == 2, code
        assert "tr(" in branch[1].split("}", 1)[0], code
