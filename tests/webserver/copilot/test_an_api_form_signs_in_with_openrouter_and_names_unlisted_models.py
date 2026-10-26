"""The API form signs in with OpenRouter and accepts a model a service does not list.

Back from OpenRouter's sign-in, the form opens on OpenRouter, asks for no
credential and lists models with the key the host holds. A service that
publishes no model list lets the researcher name the model; saving proves it.
The service list groups Chinese services first, after DeepSeek.
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
const landing = process.argv[3];
const replaced = [];
global.window = {
  EU_LANG: 'zh',
  location: { search: landing, pathname: '/', hash: '#guided', assign: url => { window.assigned = url; } },
  history: { replaceState: (_s, _t, url) => replaced.push(url) },
};
require(process.argv[1]);
require(process.argv[2]);
const provider = window.EasyICU.guidedPi.require('provider');
const esc = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const option = (value, selected, label) => `<option value="${esc(value)}"${value === selected ? ' selected' : ''}>${esc(label)}</option>`;
const tr = (en, zh) => zh;
function parseForm(markup) {
  const fields = {};
  for (const tag of markup.matchAll(/<input name="([a-z_]+)"([^>]*)>/g)) {
    const value = tag[2].match(/value="([^"]*)"/);
    fields[tag[1]] = { value: value ? value[1] : '' };
  }
  const transport = markup.match(/<select name="api_transport">[\s\S]*?<option value="([^"]+)" selected/);
  if (transport) fields.api_transport = { value: transport[1] };
  const preset = markup.match(/<select data-gpi-provider-preset>[\s\S]*?<option value="([^"]+)" selected/);
  const presetField = { value: preset ? preset[1] : '' };
  return { form: {
    fields,
    elements: { namedItem: name => fields[name] || null },
    querySelector(sel) {
      if (sel === '[data-gpi-provider-preset]') return presetField;
      const named = sel.match(/^\[name="([a-z_]+)"\]$/);
      return named ? fields[named[1]] || null : null;
    },
  }, presetField };
}
global.FormData = class {
  constructor(form) { this.form = form; }
  get(name) { return this.form.fields[name] ? this.form.fields[name].value : null; }
  has(name) { return Object.prototype.hasOwnProperty.call(this.form.fields, name); }
};
const state = { researchProvider: 'codex', setupSaving: false, error: '', runtime: { configuration: {} } };
const calls = { discover: [], save: [], login: [] };
let discover = async body => ({ ok: true, models: ['vendor/model-a'], recommended: '' });
const api = {
  discoverPiCopilotProviderModels: async body => { calls.discover.push(body); return discover(body); },
  savePiCopilotProviderConfig: async body => { calls.save.push(body); return { runtime: { status: 'ready' } }; },
  startPiCopilotOpenRouterLogin: async body => { calls.login.push(body); return loginReply; },
};
let loginReply = { auth_url: 'https://openrouter.ai/auth?callback_url=x&code_challenge=y&code_challenge_method=S256' };
let current = null;
const host = { querySelector: sel => (sel === '[data-gpi-provider-form]' ? current && current.form : null) };
state.host = host;
let markup = '';
let control;
const options = () => ({
  state, runtime: state.runtime, config: state.runtime.configuration || {}, runtimeMissing: [], tr, esc, option,
  providerPreset: () => 'deepseek', connectionConfigured: false, staticPreview: false,
});
function render() {
  control.captureForm(host);
  markup = state.showFormNow === false ? '' : provider.renderConnectionBody(options());
  current = markup ? parseForm(markup) : null;
  control.restoreForm(host);
}
const errorText = e => 'E:' + String((e && e.code) || e);
control = window.EasyICU.guidedPi.require('providerControl').create({
  state, api: () => api, tr, render, runtimeReady: () => true, shellReady: () => true,
  connectionConfigured: () => false, connectionReady: () => false, errorText,
});
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const out = { landing: { provider: state.researchProvider, setup: state.showSetup, replaced } };
(async () => {
  // The loading screen renders no form; the return waits for it.
  state.loading = true; state.showFormNow = false; render();
  state.loading = false; state.showFormNow = true; render();
  await tick(); await tick();
  out.afterLanding = {
    preset: current && current.presetField.value,
    noCredentialField: !markup.includes('name="api_key"'),
    signedInNote: markup.includes('已用 OpenRouter 账户登录'),
    discover: calls.discover[0] || null,
    error: state.error,
  };
  if (current && current.form.fields.model === undefined) current.form.fields.model = { value: 'vendor/model-a' };
  await control.configureProvider(current.form);
  out.save = calls.save[0] || null;
  out.afterSave = { login: state.openrouterLogin || null };

  // Signing in from the form: the page leaves for OpenRouter's /auth only.
  current.presetField.value = 'openrouter';
  control.handleClick({ target: { closest: sel => (sel === '[data-gpi-openrouter-login]' ? {} : null) } });
  await tick(); await tick();
  out.loginStart = { body: calls.login[0] || null, assigned: window.assigned || '' };
  window.assigned = '';
  loginReply = { auth_url: 'https://openrouter.ai.evil.test/auth?x=1' };
  control.handleClick({ target: { closest: sel => (sel === '[data-gpi-openrouter-login]' ? {} : null) } });
  await tick(); await tick();
  out.loginRefused = { assigned: window.assigned || '', error: state.error };

  // A service with no model list: the researcher names the model.
  state.error = '';
  current.presetField.value = 'zhipu';
  control.handleChange({ target: { value: 'zhipu', closest: () => current.form, matches: sel => sel === '[data-gpi-provider-preset]' } });
  discover = async () => { throw { code: 'pi_provider_model_list_unavailable' }; };
  current.form.fields.api_key = { value: 'sk-synthetic-zhipu' };
  await control.configureProvider(current.form);
  await tick(); await tick();
  out.manual = {
    input: /<input name="model" maxlength="256"/.test(markup),
    hint: markup.includes('该服务不提供模型列表'),
    saveLabel: markup.includes('验证并保存'),
    error: state.error,
  };
  current.form.fields.model.value = 'glm-synthetic';
  current.form.fields.api_key = { value: 'sk-synthetic-zhipu' };
  await control.configureProvider(current.form);
  out.manualSave = calls.save[1] || null;
  // Signed in, but the address was edited away from OpenRouter: the typed
  // credential goes, never the sign-in.
  state.openrouterLogin = 'signed_in';
  discover = async () => ({ ok: true, models: ['m'], recommended: '' });
  current.presetField.value = 'openrouter';
  control.handleChange({ target: { value: 'openrouter', closest: () => current.form, matches: sel => sel === '[data-gpi-provider-preset]' } });
  current.form.fields.base_url = { value: 'https://gateway.synthetic.test/v1' };
  current.form.fields.api_key = { value: 'sk-typed-for-gateway' };
  await control.configureProvider(current.form);
  await tick(); await tick();
  out.editedAddress = calls.discover[calls.discover.length - 1];
  out.groups = [...markup.matchAll(/<optgroup label="([^"]+)">/g)].map(m => m[1]);
  out.firstOptions = [...markup.matchAll(/<option value="([a-z-]+)"/g)].map(m => m[1]).slice(0, 7);
  process.stdout.write(JSON.stringify(out));
})().catch(err => { console.error(err); process.exitCode = 1; });
"""


def _run(landing: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    completed = run_node(
        node, _SCRIPT,
        str(STATIC / "js" / "screens-guided-pi-provider.js"),
        str(STATIC / "js" / "screens-guided-pi-provider-control.js"),
        landing,
        check=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_back_from_openrouter_the_form_lists_models_with_the_hosts_key() -> None:
    out = _run("?openrouter_login=signed_in")

    assert out["landing"] == {"provider": "api", "setup": True, "replaced": ["/#guided"]}
    assert out["afterLanding"] == {
        "preset": "openrouter", "noCredentialField": True, "signedInNote": True, "error": "",
        "discover": {
            "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
            "api_transport": "openai-completions", "api_key": "",
            "credential_source": "openrouter_login", "enable_ai": True,
        },
    }
    assert out["save"]["credential_source"] == "openrouter_login"
    assert out["save"]["api_key"] == "" and out["save"]["model"] == "vendor/model-a"
    assert out["afterSave"] == {"login": None}
    assert out["loginStart"]["body"] == {"enable_ai": True}
    assert out["loginStart"]["assigned"].startswith("https://openrouter.ai/auth?")
    assert out["loginRefused"] == {"assigned": "", "error": "E:openrouter_auth_url_invalid"}


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_failed_sign_in_reports_its_reason_and_asks_for_a_credential() -> None:
    out = _run("?openrouter_login=failed&reason=openrouter_login_exchange_failed")

    assert out["landing"]["provider"] == "api"
    assert out["afterLanding"]["preset"] == "openrouter"
    assert out["afterLanding"]["noCredentialField"] is False
    assert out["afterLanding"]["discover"] is None
    assert out["afterLanding"]["error"] == "E:openrouter_login_exchange_failed"


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_an_unlisted_model_is_named_by_the_researcher_and_saved_with_its_key() -> None:
    out = _run("?openrouter_login=signed_in")

    assert out["manual"] == {"input": True, "hint": True, "saveLabel": True, "error": ""}
    assert out["manualSave"] == {
        "provider": "zhipu", "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_transport": "openai-completions", "api_key": "sk-synthetic-zhipu",
        "model": "glm-synthetic", "enable_ai": True,
    }
    assert out["editedAddress"]["api_key"] == "sk-typed-for-gateway"
    assert "credential_source" not in out["editedAddress"]
    assert out["groups"] == ["国内服务", "国际服务", "机构云", "本机", "代理与网关"]
    assert out["firstOptions"] == ["deepseek", "qwen", "zhipu", "moonshot", "doubao", "siliconflow", "minimax"]
