'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const STATIC = path.resolve(__dirname, '../../src/easyicu/webserver/static');
const plain = value => JSON.parse(JSON.stringify(value));
const settle = async () => { for (let i = 0; i < 8; i++) await new Promise(resolve => setImmediate(resolve)); };
const builtin = (id, layer, title, title_zh) => ({ source: 'builtin', id, layer, title, title_zh,
  enabled: true, version: 'synthetic-v1', digest: 'a'.repeat(64) });
const CATALOG = [
  builtin('survival-time', 'research_workflow', 'Survival outcome analysis', '生存分析'),
  builtin('synthetic-table', 'analysis_module', 'Synthetic table', '合成描述表'),
  { source: 'user', id: 'my-notes', title: 'My notes <b>', enabled: true, digest: 'b'.repeat(64) },
  builtin('synthetic-definition', 'data_definition', 'Synthetic definition', '合成数据定义'),
  { ...builtin('unavailable-module', 'analysis_module', 'Unavailable module', '未启用模块'), enabled: false },
];

// Only the DOM surface used by the composer owner. Events are dispatched
// through the same delegated handlers as a browser, with the real attachment owner.
function harness(handlers = {}) {
  const listeners = {};
  const document = { activeElement: null };
  let menu = null;
  let scrolled = 0;
  const attrs = {};
  const input = {
    value: '', selectionStart: 0, selectionEnd: 0, disabled: false,
    matches: selector => selector === '[data-gpi-input]',
    setAttribute: (key, value) => { attrs[key] = String(value); },
    removeAttribute: key => { delete attrs[key]; },
    focus() { document.activeElement = this; },
    setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; },
  };
  const card = { appendChild: node => { menu = node; }, contains: node => node === input || node === menu };
  const host = {
    addEventListener: (name, fn) => { (listeners[name] ||= []).push(fn); },
    querySelector: selector => selector === '[data-gpi-input]' ? input
      : selector === '.gpi-compose-card' ? card : selector === '[data-gpi-slash-menu]' ? menu : null,
  };
  document.createElement = () => ({
    dataset: {}, open: false, innerHTML: '', className: '',
    setAttribute(key, value) { this[key] = value; },
    remove() { if (menu === this) menu = null; },
    querySelector(selector) {
      const match = selector.match(/data-gpi-slash-row="(\d+)"/);
      if (match) return { scrollIntoView: () => { scrolled++; } };
      return null;
    },
  });
  const state = { host, session: { session_id: 's1', draft_attached_skills: [] }, draft: '', busy: false };
  const declared = {};
  const calls = [];
  const window = {
    EU_LANG: 'zh',
    EU_COMPOSER_KEYBOARD: { enterShouldSend: event => event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229 },
    EU_POPOVER_MENUS: { closeAll() { calls.push(['close-menus']); }, fit() {} },
    EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; }, optional: name => declared[name] } },
    EU_API: {
      loadPiSessionSkills: async () => handlers.catalog ? handlers.catalog() : { items: CATALOG, max_attached_skills: 2 },
      setPiDraftSkills: async (session, body) => {
        calls.push(['draft', session, plain(body)]);
        if (handlers.draft) return handlers.draft(body);
        return { attached_skills: body.attached_skills };
      },
    },
  };
  const context = vm.createContext({ window, document });
  for (const file of ['html-escape.js', 'screens-guided-pi-skill-attach.js', 'screens-guided-pi-slash-menu.js', 'screens-guided-pi-events.js']) {
    vm.runInContext(fs.readFileSync(path.join(STATIC, 'js', file), 'utf8'), context, { filename: file });
  }
  let renders = 0;
  const events = declared.events.create({ state, projectId: () => 'p1', render: () => { renders++; },
    COHORT_ELIGIBILITY: { actionFromEvent: () => null }, MESSAGE_ACTIONS: { handleClick: () => false },
    sendMessage: () => { calls.push(['send']); } });
  events.wire();
  const owner = { capture: events.captureComposer, restore: events.restoreComposer };
  input.focus();
  function emit(name, props = {}) {
    const event = { target: input, key: '', defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; }, ...props };
    for (const fn of listeners[name] || []) fn(event);
    return event;
  }
  async function type(value, caret = value.length, extra = {}) {
    input.value = value; input.setSelectionRange(caret, caret); state.draft = value;
    emit('input', extra); await settle();
  }
  const click = index => emit('click', { target: {
    closest: selector => selector === '[data-gpi-slash-row]' ? { dataset: { gpiSlashRow: String(index) } } : null,
  } });
  return { state, input, attrs, calls, window, owner, emit, type, click,
    menu: () => menu, html: () => menu ? menu.innerHTML : '', scrolled: () => scrolled, renders: () => renders };
}

test('slash opens a grouped popover and filters Chinese, English, id and group names', async () => {
  const h = harness();
  await h.type('/');
  assert.equal(h.menu().open, true);
  assert.equal(h.menu()['data-popover-menu'], '');
  assert.match(h.html(), /流程/); assert.match(h.html(), /模块/);
  assert.match(h.html(), /我的技能/); assert.match(h.html(), /其他/);
  assert.doesNotMatch(h.html(), /unavailable-module/);
  assert.match(h.html(), /My notes &lt;b&gt;/);
  assert.equal(h.input.value, '/');
  assert.equal(h.attrs['aria-expanded'], 'true');
  for (const query of ['生存', 'SURVIVAL', 'outcome', 'survival-time', 'workflow', '流程']) {
    await h.type('/' + query);
    assert.match(h.html(), /survival-time/);
    assert.doesNotMatch(h.html(), /synthetic-table/);
  }
  await h.type('/no-synthetic-match');
  assert.match(h.html(), /没有匹配/);
  assert.equal(h.attrs['aria-activedescendant'], undefined);
  h.window.EU_LANG = 'en';
  await h.type('/table');
  assert.match(h.html(), /Modules/);
  assert.match(h.html(), /Synthetic table/);
});

test('arrows navigate and scroll, Enter selects the exact identity without sending', async () => {
  const h = harness();
  await h.type('Explain /');
  assert.equal(h.emit('keydown', { key: 'ArrowDown' }).defaultPrevented, true);
  assert.equal(h.attrs['aria-activedescendant'], 'gpi-slash-row-1');
  h.emit('keydown', { key: 'ArrowUp' });
  assert.equal(h.attrs['aria-activedescendant'], 'gpi-slash-row-0');
  h.emit('keydown', { key: 'ArrowUp' });
  assert.equal(h.attrs['aria-activedescendant'], 'gpi-slash-row-3');
  h.emit('keydown', { key: 'ArrowDown' });
  assert.ok(h.scrolled() >= 4);
  assert.equal(h.emit('keydown', { key: 'Enter' }).defaultPrevented, true);
  await settle();
  assert.deepEqual(h.calls.find(row => row[0] === 'draft'), ['draft', 's1', {
    project_id: 'p1', attached_skills: [{ source: 'builtin', id: 'survival-time', version: 'synthetic-v1', digest: 'a'.repeat(64) }],
  }]);
  assert.equal(h.input.value, 'Explain '); assert.equal(h.state.draft, 'Explain ');
  assert.equal(h.menu(), null);
  assert.equal(h.attrs['aria-expanded'], undefined);
  assert.equal(h.renders(), 1);
  assert.equal(h.calls.some(row => row[0] === 'send'), false);
});

test('Tab and mouse selection remove the whole query at the caret and keep surrounding text', async () => {
  for (const method of ['tab', 'mouse']) {
    const h = harness();
    await h.type('Before /my-notes after', 11);
    if (method === 'tab') assert.equal(h.emit('keydown', { key: 'Tab' }).defaultPrevented, true);
    else h.click(0);
    await settle();
    assert.equal(h.state.draft, 'Before  after');
    assert.equal(h.input.selectionStart, 7);
    assert.deepEqual(plain(h.state.session.draft_attached_skills), [{ source: 'user', id: 'my-notes', digest: 'b'.repeat(64) }]);
  }
});

test('Escape closes, preserves the query and allows the next Enter to reach the composer', async () => {
  const h = harness(); await h.type('/sur');
  assert.equal(h.emit('keydown', { key: 'Escape' }).defaultPrevented, true);
  assert.equal(h.menu(), null); assert.equal(h.input.value, '/sur');
  h.emit('keydown', { key: 'Enter' });
  assert.equal(h.calls.filter(row => row[0] === 'send').length, 1);
  assert.equal(h.calls.some(row => row[0] === 'draft'), false);
});

test('Chinese composition suppresses input and navigation until compositionend', async () => {
  const h = harness(); await h.type('/sur');
  h.emit('keydown', { key: 'Enter', keyCode: 229 }); await settle();
  h.emit('keydown', { key: 'Enter', isComposing: true }); await settle();
  assert.equal(h.calls.some(row => row[0] === 'draft'), false);
  h.emit('compositionstart'); assert.equal(h.menu(), null);
  await h.type('/生'); assert.equal(h.menu(), null);
  assert.equal(h.emit('keydown', { key: 'Enter', isComposing: true }).defaultPrevented, false);
  h.emit('compositionend'); await settle(); assert.match(h.html(), /生存分析/);
  await h.type('/sur', 4, { isComposing: true }); assert.equal(h.menu(), null);
});

test('URLs, embedded slashes, selections and whitespace after the token close the menu', async () => {
  const h = harness();
  for (const text of ['http://synthetic.test/path', 'See https://synthetic.test/', 'word/sur', '//sur', '/sur/path', '/sur ']) {
    await h.type('/'); await h.type(text);
    assert.equal(h.menu(), null, text);
  }
  await h.type('/sur/path', 4); assert.equal(h.menu(), null);
  await h.type('text\n/sur'); assert.ok(h.menu());
  h.input.selectionEnd = h.input.selectionStart - 1; h.emit('input');
  assert.equal(h.menu(), null);
});

test('the shared attachment limit and already-attached state are stated on rows without writes', async () => {
  const h = harness();
  h.state.session.draft_attached_skills = CATALOG.slice(0, 2);
  await h.type('/');
  assert.match(h.html(), /已添加/); assert.match(h.html(), /已达 2 个技能上限/);
  assert.match(h.html(), /aria-disabled="true"/);
  h.click(0); h.click(2); h.emit('keydown', { key: 'Enter' }); await settle();
  assert.equal(h.calls.some(row => row[0] === 'draft'), false);
  assert.equal(h.calls.some(row => row[0] === 'send'), false);
  assert.equal(h.input.value, '/');
  h.state.session.draft_attached_skills.pop();
  await h.type('/my'); h.click(0); await settle();
  assert.equal(h.state.session.draft_attached_skills.length, 2);
});

test('host refusals keep the query and surface through the existing attachment chips', async () => {
  const h = harness({ draft: () => { throw { code: 'skill_attachment_family_conflict' }; } });
  await h.type('/sur'); h.click(0); await settle();
  assert.equal(h.input.value, '/sur'); assert.equal(h.state.session.draft_attached_skills.length, 0);
  assert.equal(h.renders(), 1);
});

test('a repaint keeps the open menu and the textarea selection', async () => {
  const h = harness(); await h.type('Before /sur after', 11);
  h.owner.capture(); h.menu().remove(); h.input.setSelectionRange(0, 0);
  h.owner.restore();
  assert.equal(h.menu().open, true); assert.equal(h.input.selectionStart, 11);
  assert.equal(h.attrs['aria-activedescendant'], 'gpi-slash-row-0');
});

test('late catalogue responses cannot reopen a dismissed menu or another session', async () => {
  for (const leave of ['escape', 'session']) {
    let resolve;
    const h = harness({ catalog: () => new Promise(done => { resolve = done; }) });
    await h.type('/sur');
    if (leave === 'escape') h.emit('keydown', { key: 'Escape' });
    else { h.state.session = { session_id: 's2', draft_attached_skills: [] }; h.owner.restore(); }
    resolve({ items: CATALOG, max_attached_skills: 2 }); await settle();
    assert.equal(h.menu(), null);
  }
});

test('a native popover close clears the slash state and asynchronous responses stay closed', async () => {
  const h = harness(); await h.type('/');
  const menu = h.menu(); menu.open = false; h.emit('toggle', { target: menu });
  assert.equal(h.menu(), null);
  h.emit('keydown', { key: 'Enter' });
  assert.equal(h.calls.filter(row => row[0] === 'send').length, 1);
});

test('an outside press followed by a synchronous repaint stays closed before the toggle event arrives', async () => {
  const h = harness(); await h.type('/');
  h.menu().open = false;
  h.owner.capture(); h.menu()?.remove(); h.owner.restore();
  assert.equal(h.menu(), null);
  assert.equal(h.attrs['aria-expanded'], undefined);
});

test('busy or stale conversations cannot open or select slash skills', async () => {
  for (const flag of ['busy', 'childJobId', 'projectLoading', 'stale']) {
    const h = harness();
    if (flag === 'stale') h.state.session.stale = { stale: true }; else h.state[flag] = true;
    await h.type('/'); assert.equal(h.menu(), null);
  }
  const h = harness(); await h.type('/'); h.state.busy = true; h.click(0); await settle();
  assert.equal(h.calls.some(row => row[0] === 'draft'), false);
});

test('a catalogue failure is readable and Enter never sends the unresolved slash token', async () => {
  const h = harness({ catalog: () => { throw new Error('synthetic failure'); } });
  await h.type('/'); assert.match(h.html(), /暂时无法加载/);
  assert.equal(h.emit('keydown', { key: 'Enter' }).defaultPrevented, true);
  assert.equal(h.calls.some(row => row[0] === 'draft'), false);
});

test('moving the caret outside a slash token closes the menu', async () => {
  const h = harness(); await h.type('Before /sur after', 11);
  h.input.setSelectionRange(1, 1); h.emit('keyup', { key: 'ArrowLeft' });
  assert.equal(h.menu(), null);
});

test('selecting after the caret moves cannot remove a stale token', async () => {
  const h = harness(); await h.type('Before /sur after', 11);
  h.input.setSelectionRange(1, 1); h.click(0); await settle();
  assert.equal(h.calls.some(row => row[0] === 'draft'), false);
  assert.equal(h.state.draft, 'Before /sur after');
});

test('a delayed attachment leaves newer text and another session draft intact', async () => {
  for (const change of ['typing', 'session']) {
    let resolve;
    const h = harness({ draft: () => new Promise(done => { resolve = done; }) });
    await h.type('/sur'); h.click(0); await settle();
    if (change === 'typing') await h.type('New synthetic question');
    else { h.state.session = { session_id: 's2', draft_attached_skills: [] }; h.input.value = 'New synthetic question'; h.state.draft = h.input.value; }
    resolve({ attached_skills: [CATALOG[0]] }); await settle();
    assert.equal(h.state.draft, 'New synthetic question');
  }
});

test('composer integration loads the owner, delegates before sending and keeps anchored scrollable targets', () => {
  const read = file => fs.readFileSync(path.join(STATIC, file), 'utf8');
  const index = read('index.html');
  assert.ok(index.indexOf('js/screens-guided-pi-skill-attach.js') < index.indexOf('js/screens-guided-pi-slash-menu.js'));
  assert.ok(index.indexOf('js/screens-guided-pi-slash-menu.js') < index.indexOf('js/screens-guided-pi-events.js'));
  const events = read('js/screens-guided-pi-events.js');
  assert.match(events, /slashMenu/); assert.match(events, /slash\.wire\(\)/);
  assert.match(events, /if \(event\.defaultPrevented\) return;/);
  assert.doesNotMatch(events, /\[@\/\]\$/);
  const css = read('css/guided-pi-composer-menus.css');
  assert.match(css, /\.gpi-compose-card \.gpi-slash-popover\{[^}]*bottom:calc\(100% \+ 8px\)/);
  assert.match(css, /\.gpi-entry-compose \.gpi-compose-card \.gpi-slash-popover\{[^}]*top:calc\(100% \+ 8px\)/);
  assert.match(css, /max-height:min\(320px,/); assert.match(css, /overflow-y:auto/);
  assert.match(css, /\.gpi-slash-row\{[^}]*min-height:36px/);
});
