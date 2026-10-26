'use strict';

// The credentialed-data declaration control: what each host state shows,
// what a save sends, and the copy for every refusal code.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const esc = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

function load(language) {
  const owners = {};
  const tr = (en, zh) => language === 'zh' ? zh : en;
  const context = vm.createContext({
    t: tr,
    icon: () => '',
    FormData: class { constructor(form) { this.form = form; } get(name) { return this.form.values[name]; } },
    window: {
      EU_LANG: language,
      EU_HTML: { esc },
      EasyICU: { guidedPi: {
        declare: (name, api) => { owners[name] = api; },
        optional: name => owners[name] || null,
      } },
    },
  });
  for (const path of process.argv.slice(2)) {
    vm.runInContext(fs.readFileSync(path, 'utf8'), context, { filename: path });
  }
  return { owners, tr };
}

const connection = { provider: 'easyicu-local', base_url: 'http://127.0.0.1:8317/v1', model: 'model-a' };
function record(overrides) {
  return {
    connection, value: 'unspecified', lapsed: false, may_change: true,
    change_authority: 'local_user', source: 'settings', status_code: 'pi_provider_declaration_unspecified',
    notice_key: 'physionet_llm_data_handling_declaration', notice_version: '2', ...overrides,
  };
}
const checked = html => (html.match(/value="([a-z_]+)" checked/) || [])[1] || '';

for (const language of ['en', 'zh']) {
  const { owners, tr } = load(language);
  const view = owners.providerDeclaration;
  const render = (state, config) => view.renderSection({ state, tr, esc, config: config || { connection_verified: true } });

  // Nothing to declare until a connection exists, and never in a static preview.
  assert.equal(render({}), '');
  assert.equal(render({ declaration: record({ status_code: 'pi_provider_declaration_connection_required' }) }), '');
  assert.equal(view.renderSection({ state: { declaration: record() }, tr, esc, staticPreview: true }), '');

  // An undeclared desktop connection: four choices, nothing to save yet.
  let html = render({ declaration: record() });
  assert.equal((html.match(/name="declaration"/g) || []).length, 4);
  assert.equal(checked(html), 'unspecified');
  assert.match(html, /gpi-declaration-save" type="submit" disabled/);
  assert.ok(html.includes(tr('Can this model handle MIMIC-IV or eICU data?', '这个模型能处理 MIMIC-IV、eICU 数据吗？')));

  // The user's own acceptance of an external service is its own choice, named as such.
  html = render({ declaration: record({ value: 'user_accepted_external', status_code: 'pi_provider_declaration_recorded' }) });
  assert.equal(checked(html), 'user_accepted_external');
  assert.ok(html.includes(tr('An external service I accept for these data', '外部服务（我确认可以使用）')));
  assert.ok(html.includes(tr('The third choice is your own decision.', '第三项由你自行决定。')));
  assert.match(html, /value="user_accepted_external"[^>]*><span>(An external service I accept for these data|外部服务（我确认可以使用）)<\/span>/);

  // A changed choice can be saved.
  html = render({ declaration: record(), declarationChoice: 'zero_retention_declared' });
  assert.equal(checked(html), 'zero_retention_declared');
  assert.doesNotMatch(html, /gpi-declaration-save" type="submit" disabled/);

  // A lapsed record belonged to another connection: nothing is chosen for this one.
  html = render({ declaration: record({ value: 'on_device', lapsed: true, status_code: 'pi_provider_declaration_lapsed' }) });
  assert.equal(checked(html), 'unspecified');
  assert.ok(html.includes(tr('The connection changed, so the earlier declaration no longer applies.', '连接已更改，之前的声明不再适用。')));
  assert.doesNotMatch(html, /gpi-declaration-save" type="submit" disabled/);

  // The administrator's configuration is shown, not editable.
  html = render({ declaration: record({ value: 'on_device', may_change: false, change_authority: 'administrator_configuration', status_code: 'pi_provider_declaration_administrator_configuration' }) });
  assert.equal(checked(html), 'on_device');
  assert.match(html, /<fieldset[^>]* disabled>/);
  assert.equal(html.includes('gpi-declaration-save'), false);

  // A recorded declaration applies only after the connection is verified again.
  const recorded = record({ value: 'on_device', status_code: 'pi_provider_declaration_recorded' });
  const verifyNote = tr('Verify the connection again to apply this declaration.', '请重新验证连接，这项声明才会生效。');
  assert.ok(render({ declaration: recorded }, { connection_verified: false }).includes(verifyNote));
  assert.equal(render({ declaration: recorded }, { connection_verified: true }).includes(verifyNote), false);

  // The connection page asks only when a stopped run sent the researcher
  // there, only for an API connection, and then the question leads the page.
  const pageOptions = (researchProvider, declarationAsked) => ({
    state: { researchProvider, declaration: record(), declarationAsked }, tr, esc, config: {}, runtime: {},
    option: (value, selected, label) => `<option value="${value}">${label}</option>`,
    providerPreset: () => 'cliproxyapi',
  });
  assert.equal(owners.provider.renderConnectionBody(pageOptions('api', false)).includes('data-gpi-declaration-form'), false);
  const asked = owners.provider.renderConnectionBody(pageOptions('api', true));
  assert.ok(asked.includes('data-gpi-declaration-form'));
  assert.ok(asked.indexOf('data-gpi-declaration-form') < asked.indexOf('data-gpi-provider-form'));
  assert.equal(owners.provider.renderConnectionBody(pageOptions('codex', true)).includes('data-gpi-declaration-form'), false);

  // Every refusal code has its own sentence in both languages.
  const errors = owners.errorText.create({ tr, staticPreview: () => false });
  const sentences = new Set();
  for (const suffix of ['settings_only', 'admin_required', 'notice_mismatch', 'value_invalid', 'administrator_configuration',
    'connection_required', 'connection_changed', 'record_unavailable', 'busy', 'gateway_unavailable']) {
    const code = `pi_provider_declaration_${suffix}`;
    const text = errors.errorText({ code, message: code });
    assert.ok(text && text !== code && !text.includes('pi_provider'), `${language}: ${code}`);
    sentences.add(text);
  }
  assert.equal(sentences.size, 10);
}

// A save sends the shown connection and notice binding, then refreshes the
// host status; a stale form reloads the record.
(async () => {
  const { owners, tr } = load('en');
  const calls = [];
  const state = { declaration: record() };
  let failWith = null;
  const control = owners.providerDeclaration.create({
    state, tr, render: () => {}, errorText: error => `text:${error.code}`,
    reloadStatus: async () => { calls.push(['status']); },
    api: () => ({
      loadPiCopilotProviderDeclaration: async () => { calls.push(['load']); return record({ status_code: 'pi_provider_declaration_unspecified' }); },
      savePiCopilotProviderDeclaration: async body => {
        calls.push(['save', body]);
        if (failWith) { const error = new Error('x'); error.code = failWith; throw error; }
        return record({ value: body.value, status_code: 'pi_provider_declaration_recorded' });
      },
    }),
  });
  const form = { values: { declaration: 'zero_retention_declared' } };
  const submit = { target: { closest: selector => (selector === '[data-gpi-declaration-form]' ? form : null) }, preventDefault() {} };
  assert.equal(control.handleSubmit(submit), true);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0])), ['save', {
    connection, value: 'zero_retention_declared', notice_key: 'physionet_llm_data_handling_declaration',
    notice_version: '2', channel: 'settings_ui',
  }]);
  assert.deepEqual(calls[1], ['status']);
  assert.equal(state.declaration.value, 'zero_retention_declared');
  assert.equal(state.declarationBusy, false);

  calls.length = 0; failWith = 'pi_provider_declaration_connection_changed';
  control.handleSubmit(submit);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(state.declarationError, 'text:pi_provider_declaration_connection_changed');
  assert.deepEqual(calls.map(call => call[0]), ['save', 'load']);

  // Another form's submit is not this owner's.
  assert.equal(control.handleSubmit({ target: { closest: () => null } }), false);
})().catch(error => { console.error(error); process.exit(1); });
