/* A menu action closes its menu before the repaint it causes, and a menu the
   researcher is still using stays open through repaints. keepOpen reopens
   every menu that is open when the conversation repaints, so an owner whose
   action repaints closes its menu first: the composer's access level, the
   header's More menu and the composer's + menu (closed on the way down, in
   the capture phase), and a conversation's ••• rename and remove (closed
   before their prompt). A layout panel toggle and a group inside + end
   nothing, so their menu stays open through the next repaint. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const [menusFile, modulesFile, eventsFile, workspaceFile, shellFile] = process.argv.slice(2).map(file => path.resolve(file));

// A small element tree that understands the selectors these owners use: a
// tag, classes and attributes, compounds of them, descendants and lists.
function compoundMatches(node, text) {
  const tag = /^[a-z][\w-]*/i.exec(text);
  if (tag && node.tagName !== tag[0].toUpperCase()) return false;
  for (const [, name] of text.matchAll(/\.([\w-]+)/g)) if (!node.classes.has(name)) return false;
  for (const [, name, value] of text.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)) {
    if (!node.attrs.has(name) || (value !== undefined && node.attrs.get(name) !== value)) return false;
  }
  return true;
}
function selectorMatches(node, selector) {
  const parts = selector.trim().split(/\s+/);
  if (!compoundMatches(node, parts.pop())) return false;
  let at = node.parentNode;
  while (parts.length) {
    while (at && !compoundMatches(at, parts[parts.length - 1])) at = at.parentNode;
    if (!at) return false;
    parts.pop();
    at = at.parentNode;
  }
  return true;
}
function el(tag, attrs = {}, children = []) {
  const node = {
    tagName: tag.toUpperCase(), attrs: new Map(), classes: new Set(), parentNode: null, children: [],
    get className() { return [...this.classes].join(' '); },
    get dataset() {
      return Object.fromEntries([...this.attrs].filter(([name]) => name.startsWith('data-'))
        .map(([name, value]) => [name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase()), value]));
    },
    get open() { return this.attrs.has('open'); },
    set open(value) { if (value) this.attrs.set('open', ''); else this.attrs.delete('open'); },
    get lastElementChild() { return this.children[this.children.length - 1] || null; },
    getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; },
    setAttribute(name, value) { this.attrs.set(name, String(value)); },
    removeAttribute(name) { this.attrs.delete(name); },
    matches(selector) { return selector.split(',').some(one => selectorMatches(this, one)); },
    closest(selector) { for (let at = this; at; at = at.parentNode) if (at.matches(selector)) return at; return null; },
    contains(other) { for (let at = other; at; at = at.parentNode) if (at === this) return true; return false; },
    querySelectorAll(selector) {
      const found = [];
      const walk = parent => parent.children.forEach(child => { if (child.matches(selector)) found.push(child); walk(child); });
      walk(this);
      return found;
    },
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
  };
  for (const [name, value] of Object.entries(attrs)) {
    if (name === 'class') value.split(' ').forEach(one => node.classes.add(one));
    else node.attrs.set(name, String(value));
  }
  children.forEach(child => { child.parentNode = node; node.children.push(child); });
  return node;
}

global.window = global;
const documentListeners = [];
global.document = {
  addEventListener(type, handler, capture) { documentListeners.push({ type, handler, capture: capture === true }); },
  activeElement: null,
  getElementById: () => null,
};
global.EU_HTML = { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') };
require(menusFile);
require(modulesFile);
require(eventsFile);
require(workspaceFile);
const MENUS = global.EU_POPOVER_MENUS;
const modules = global.EasyICU.guidedPi;

// The conversation's floating menus as the shell paints them: a repaint
// builds new nodes with every menu closed, as innerHTML does, and the shell
// wraps it in keepOpen.
const handlers = {};
const host = el('div');
host.addEventListener = (name, handler) => { handlers[name] = handler; };
const live = {};
function paint() {
  const menu = (className, body) => el('details', { class: className, 'data-popover-menu': '' }, [el('summary'), body]);
  live.accessButton = el('button', { type: 'button', 'data-gpi-access-mode': 'full' });
  live.layoutToggle = el('button', { type: 'button', 'data-gpi-layout-toggle': 'context', 'aria-pressed': 'false' }, [el('span'), el('span')]);
  live.modeSwitch = el('button', { type: 'button', 'data-gpi-mode-switch': 'workspace' });
  live.picker = el('button', { type: 'button', role: 'menuitem', 'data-gpi-composer-picker': 'materials' });
  live.groupSummary = el('summary', { role: 'menuitem' });
  live.layout = menu('gpi-layout-control', el('div', {}, [live.layoutToggle]));
  live.more = menu('gpi-head-overflow', el('div', { class: 'gpi-head-overflow-menu', role: 'menu' }, [el('div', { class: 'gpi-mode-switch' }, [live.modeSwitch])]));
  live.plus = menu('gpi-idea-source-menu', el('div', { class: 'gpi-idea-source-popover', role: 'menu' }, [
    live.picker, el('details', { class: 'gpi-idea-source-group' }, [live.groupSummary, el('div')])]));
  live.access = menu('gpi-access-menu', el('div', { class: 'gpi-access-popover' }, [live.accessButton]));
  host.children = [];
  [live.layout, live.more, live.plus, live.access].forEach(node => { node.parentNode = host; host.children.push(node); });
}
const toggles = [];
const modes = [];
const opened = [];
const inert = { handleClick: () => false, actionFromEvent: () => null };
const state = { host, session: { session_id: 'session_a' }, busy: false, childJobId: '', accessMode: 'assist' };
const render = () => MENUS.keepOpen(host, paint);
const events = modules.require('events').create({
  state, render, tr: en => en, projectId: () => 'project_current',
  MESSAGE_ACTIONS: inert, STARTERS: inert, COHORT_ELIGIBILITY: inert, DATA_CONSENT: inert,
  ASIDE: { togglePanel: name => { toggles.push(name); return true; } },
  STUDY_WORKSPACE: { openMaterials: () => opened.push('materials'), openSkills: () => opened.push('skills') },
  // The mode switch may repaint before it returns.
  switchMode: mode => { modes.push(mode); render(); },
});
events.wire();
// A press as the browser dispatches it: pointerdown, then the click through
// the document's capture-phase listeners (popover-menus.js and the shell's
// dismissHeaderOverflow) to the conversation's own handler, then the
// document's bubbling listeners.
function press(target) {
  const event = { target, preventDefault() {}, stopPropagation() {} };
  const run = (type, capture) => documentListeners
    .filter(row => row.type === type && row.capture === capture).forEach(row => row.handler(event));
  run('pointerdown', true);
  run('pointerdown', false);
  run('click', true);
  events.dismissHeaderOverflow(event);
  handlers.click(event);
  run('click', false);
}

// Choosing an access level closes its menu through the repaint it causes.
paint();
live.access.open = true;
press(live.accessButton);
assert.equal(state.accessMode, 'full');
assert.equal(live.access.open, false, 'The access menu closes with the choice');

// A mode chosen in More, and a picker chosen in +, close their menu first.
paint();
live.more.open = true;
press(live.modeSwitch);
assert.deepEqual(modes, ['workspace']);
assert.equal(live.more.open, false, 'More closes before the mode switch repaints');
paint();
live.plus.open = true;
press(live.picker);
assert.deepEqual(opened, ['materials']);
assert.equal(live.plus.open, false, '+ closes before the picker it opens repaints');

// Closing after the action, as a bubbling listener did, finds only the old
// menu: the repainted one comes back open. Hence the capture phase.
paint();
live.more.open = true;
const stale = live.modeSwitch;
handlers.click({ target: stale, preventDefault() {} });
events.dismissHeaderOverflow({ target: stale });
assert.equal(live.more.open, true);
const shell = fs.readFileSync(shellFile, 'utf8');
assert.match(shell, /document\.addEventListener\('click', dismissHeaderOverflow, true\)/);
assert.match(shell, /document\.removeEventListener\('click', dismissHeaderOverflow, true\)/);

// A layout toggle switches its panel in place and leaves the menu open, so
// the next progress repaint keeps it open for the next panel.
paint();
live.layout.open = true;
press(live.layoutToggle);
assert.deepEqual(toggles, ['context']);
assert.equal(live.layoutToggle.getAttribute('aria-pressed'), 'true');
assert.equal(live.layout.open, true);
render();
assert.equal(live.layout.open, true, 'The layout menu survives the repaint after a toggle');

// Opening a group inside + chooses nothing, so + survives the repaint.
paint();
live.plus.open = true;
press(live.groupSummary);
render();
assert.equal(live.plus.open, true, '+ survives the repaint after a group opens');

// A conversation's ••• closes before rename and remove ask the researcher,
// so the repaint after their answer does not bring it back.
const workspace = modules.require('studyWorkspace').create({ tr: en => en, esc: EU_HTML.esc, iconHtml: () => '' });
const rail = { hidden: true, innerHTML: '', querySelectorAll: () => [], querySelector: () => null };
global.document.getElementById = id => (id === 'gdConversationRail' ? rail : null);
let rowMenu = null;
const seen = [];
function rowActions() {
  const rename = el('button', { type: 'button', 'data-gpi-rail-rename': 'pi_empty' });
  const remove = el('button', { type: 'button', 'data-gpi-rail-remove': 'pi_empty' });
  rowMenu = el('details', { class: 'gpi-conversation-menu', 'data-popover-menu': '', 'data-popover-key': 'pi_empty', open: '' }, [
    el('summary'), el('div', {}, [rename, remove])]);
  return { rename, remove };
}
workspace.syncNavigation({
  visible: true, projectId: 'project_current', loading: false, disabled: false,
  sessions: [{ session_id: 'pi_empty', title: 'New task', has_history: false, created_at: '2026-09-21T00:00:00Z' }],
  selectedId: 'pi_empty', title: row => row.title, status: () => 'Not started', time: () => '',
  open: () => {}, create: () => {}, resources: [],
  rename: row => seen.push(['rename', row.session_id, rowMenu.open]),
  remove: row => seen.push(['remove', row.session_id, rowMenu.open]),
});
rail.onclick({ target: rowActions().rename });
rail.onclick({ target: rowActions().remove });
assert.deepEqual(seen, [['rename', 'pi_empty', false], ['remove', 'pi_empty', false]]);
process.stdout.write('Menu actions close their menu before the repaint; menus still in use stay open.\n');
