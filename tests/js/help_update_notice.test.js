'use strict';

// Only an update_available answer fills the rails' update slots; a failed or
// quiet answer leaves them empty. The item is a popover menu, never a card
// over the page. Later snoozes the version and empties the slots; a browser
// that will not open leaves the address in the menu instead of a popup.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const UPDATE = { status: 'update_available', latest_version: '0.2.0', release_url: 'https://github.com/shen-lab-icu/EASYICU/releases/tag/v0.2.0' };

function element(html) {
  const listeners = {};
  const node = {
    innerHTML: html || '', open: false,
    addEventListener(name, handler) { listeners[name] = handler; },
    querySelector(selector) {
      if (selector === '.eu-update') return this.innerHTML.includes('class="eu-update"') ? (this.details ||= element()) : null;
      return null;
    },
    press(action) { return listeners.click({ target: { closest: () => ({ dataset: { updateAction: action } }) } }); },
  };
  return node;
}

function harness(load, opened = true) {
  const calls = [];
  const slots = [element(), element()];
  const window = {
    EU_LANG: 'zh',
    EU_API: {
      loadUpdateCheck: async () => { calls.push('load'); return load(); },
      snoozeUpdate: async version => { calls.push(`snooze:${version}`); return { status: 'snoozed' }; },
      openUpdatePage: async () => { calls.push('open'); return { opened }; },
    },
    addEventListener() {},
  };
  const panels = [];
  const document = {
    body: {},
    querySelectorAll(selector) {
      if (selector === '[data-update-slot]') return slots;
      if (selector === '.eu-update-panel') return panels;
      return [];
    },
  };
  const context = vm.createContext({ window, document, setTimeout });
  for (const file of ['html-escape.js', 'help-update-notice.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  return { notice: window.EU_UPDATE_NOTICE, calls, slots, panels };
}

const settle = () => new Promise(resolve => setTimeout(resolve, 0));

for (const [name, load] of [
  ['offline', async () => ({ status: 'unavailable', current_version: '0.1.0' })],
  ['current', async () => ({ status: 'current', latest_version: '0.1.0' })],
  ['snoozed', async () => ({ status: 'snoozed', latest_version: '0.2.0' })],
  ['not desktop', async () => ({ status: 'not_desktop' })],
  ['a failed request', async () => { throw new Error('HTTP 500'); }],
]) {
  test(`${name} leaves the rails' update slots empty`, async () => {
    const { notice, slots } = harness(load);
    await notice.check();
    assert.deepEqual(slots.map(slot => slot.innerHTML), ['', '']);
  });
}

test('an update fills every slot with a rail item whose popover menu names the version', async () => {
  const { notice, slots } = harness(async () => UPDATE);
  await notice.check();
  for (const slot of slots) {
    assert.match(slot.innerHTML, /^<details class="eu-update" data-popover-menu><summary title="EasyICU 0\.2\.0 可以更新了"/);
    assert.match(slot.innerHTML, /<span>更新<\/span>/);
    assert.match(slot.innerHTML, /<strong>EasyICU 0\.2\.0 可以更新了<\/strong>/);
    assert.match(slot.innerHTML, /data-update-action="later">稍后提醒/);
    assert.match(slot.innerHTML, /data-update-action="download">下载/);
    assert.doesNotMatch(slot.innerHTML, /github\.com/, 'the address appears only when the browser did not open');
  }
  const before = slots[0].innerHTML;
  notice.fill(document_of(slots));
  assert.equal(slots[0].innerHTML, before, 'a filled slot is not rebuilt');
});

function document_of(slots) { return { querySelectorAll: () => slots }; }

test('remind me later empties the slots and snoozes that version', async () => {
  const { notice, slots, calls } = harness(async () => UPDATE);
  await notice.check();
  slots[0].details.press('later');
  await settle();
  assert.deepEqual(slots.map(slot => slot.innerHTML), ['', '']);
  assert.deepEqual(calls, ['load', 'snooze:0.2.0']);
});

test('download opens the release page through the host and then empties the slots', async () => {
  const { notice, slots, calls } = harness(async () => UPDATE);
  await notice.check();
  slots[1].details.press('download');
  await settle();
  assert.deepEqual(calls, ['load', 'open']);
  assert.deepEqual(slots.map(slot => slot.innerHTML), ['', '']);
});

test('a browser that will not open leaves the address in the menu', async () => {
  const { notice, slots, panels } = harness(async () => UPDATE, false);
  await notice.check();
  panels.push({ innerHTML: '' });
  slots[0].details.press('download');
  await settle();
  assert.match(slots[0].innerHTML, /class="eu-update"/);
  assert.match(panels[0].innerHTML, /没能打开浏览器[^<]*<span>https:\/\/github\.com\/shen-lab-icu\/EASYICU\/releases\/tag\/v0\.2\.0<\/span>/);
});
