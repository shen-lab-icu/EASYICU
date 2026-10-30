'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname,
  '../../src/easyicu/webserver/static/js/screens-guided-pi-host-jobs.js'), 'utf8');
const title = 'eICU Collaborative Research Database Demo';
const row = (version = '2.0.1') => ({ id: `eicu-${version}`, title, version, database: 'eicu_demo' });
const other = { id: 'mimic', title: 'MIMIC-IV Clinical Database Demo', version: '2.2', database: 'miiv' };

async function reader(rows) {
  let owner;
  const window = { EU_HTML: { esc: String }, EasyICU: { guidedPi: {
    declare: (_name, value) => { owner = value; },
    optional: () => null,
  } } };
  vm.runInNewContext(source, { window });
  let text = '';
  const controller = owner.create({
    tr: en => en, session: () => ({ session_id: 'synthetic' }), busy: () => false,
    dataConsent: { requiresConfirmation: () => true },
    api: () => ({ loadOfficialDemoSources: async () => ({ sources: rows }) }),
    messages: () => [{ role: 'user', text }], render() {},
  });
  controller.namedDemo();
  await new Promise(resolve => setImmediate(resolve));
  return question => {
    text = question;
    return controller.namedDemo()?.id ?? null;
  };
}

test('a unique complete title and version wins over the shared product gap match', async () => {
  const ask = await reader([row(), row('2.0.2'), other]);
  assert.equal(ask(`Use ${title} v2.0.2 for synthetic counts.`), 'eicu-2.0.2');
  assert.equal(ask('Use eICU demo for synthetic counts.'), null);
});

for (const suffix of ['v2.0.0', 'v2.0.10', 'v2.0.1-beta', 'v2.0.1+build', 'v2.0.1_legacy', '']) {
  test(`a near-miss complete title is not an exact source choice: ${suffix || 'missing version'}`, async () => {
    const ask = await reader([row(), other]);
    assert.equal(ask(`Use ${title} ${suffix} for synthetic counts.`), null);
  });
}

test('complete titles and input use the same Unicode and case normalization', async () => {
  const ask = await reader([row(), row('2.0.2'), other]);
  const question = `Use ${title.toUpperCase()} v2.0.1 for synthetic counts.`;
  const fullWidth = question.replace(/[!-~]/g, char => String.fromCharCode(char.charCodeAt(0) + 0xfee0));
  assert.equal(ask(fullWidth), 'eicu-2.0.1');
});

test('naming a second database blocks even one unique complete demo title', async () => {
  const ask = await reader([row(), other]);
  assert.equal(ask(`Compare ${title} v2.0.1 with ＭＩＭＩＣ－ＩＶ full data.`), null);
  assert.equal(ask(`Compare ${title} v2.0.1 and ${other.title} v2.2.`), null);
});

test('duplicate exact catalog hits remain ambiguous', async () => {
  const ask = await reader([row(), { ...row(), id: 'duplicate' }, other]);
  assert.equal(ask(`Use ${title} v2.0.1.`), null);
});

test('short demo names remain supported without matching inside another word', async () => {
  const ask = await reader([row(), other]);
  assert.equal(ask('Use eICU demo for synthetic counts.'), 'eicu-2.0.1');
  assert.equal(ask('Use unrelatedeicu demo for synthetic counts.'), null);
  assert.equal(ask('Use eICU Collaborative Research Database. Another project is a demo.'), null);
});

test('a short name cannot discard an explicitly requested version', async () => {
  const ask = await reader([row(), other]);
  assert.equal(ask('Use eICU demo v2.0.1 for synthetic counts.'), 'eicu-2.0.1');
  assert.equal(ask('Use eICU demo v2.0.0 for synthetic counts.'), null);
  assert.equal(ask('Use eICU demo v2.0.1-beta for synthetic counts.'), null);
});
