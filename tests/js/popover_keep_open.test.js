/* An open floating menu survives its owner's repaint: a running job repaints
   the conversation on every progress event, which used to close the layout,
   more, composer and conversation menus under the researcher's pointer. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const [ownerFile, ...consumerFiles] = process.argv.slice(2).map(file => path.resolve(file));
let active = null;
global.window = global;
global.document = {
  addEventListener() {},
  querySelectorAll: () => [],
  get activeElement() { return active; },
};
require(ownerFile);
const MENUS = global.EU_POPOVER_MENUS;

function menu(className, popoverKey) {
  const summary = { focused: 0, focus() { this.focused += 1; active = summary; } };
  const item = { name: `${className}-item` };
  const m = {
    className, open: false, summary, item, popoverKey,
    contains: node => node === m || node === summary || node === item,
    querySelector: selector => (selector === ':scope > summary' ? summary : null),
    getAttribute: name => (name === 'data-popover-key' ? popoverKey || null : null),
  };
  return m;
}
// A root whose rebuild replaces every menu node, like an innerHTML repaint.
const root = {
  menus: [],
  querySelectorAll(selector) {
    if (selector === 'details[data-popover-menu][open]') return this.menus.filter(m => m.open);
    if (selector === 'details[data-popover-menu]') return this.menus.slice();
    return [];
  },
};
const paint = () => { root.menus = [menu('gpi-layout-control'), menu('gpi-conversation-menu'), menu('gpi-conversation-menu')]; };
paint();

// The second conversation's menu is open with focus inside it.
root.menus[2].open = true;
active = root.menus[2].item;
let rebuilt = 0;
assert.equal(MENUS.keepOpen(root, () => { rebuilt += 1; paint(); return 'painted'; }), 'painted');
assert.equal(rebuilt, 1);
assert.deepEqual(root.menus.map(m => m.open), [false, false, true], 'The same menu of its kind reopens');
assert.equal(root.menus[2].summary.focused, 1, 'Focus returns to its summary');

// Without focus inside, the menu reopens and focus stays where it was.
active = null;
MENUS.keepOpen(root, paint);
assert.deepEqual(root.menus.map(m => m.open), [false, false, true]);
assert.equal(root.menus[2].summary.focused, 0);

// A menu its owner closed before the rebuild stays closed.
root.menus[2].open = false;
MENUS.keepOpen(root, paint);
assert.deepEqual(root.menus.map(m => m.open), [false, false, false]);
assert.equal(MENUS.keepOpen(null, () => 'plain'), 'plain');

// A keyed menu follows its row when the rows reorder: the open ••• of
// conversation B reopens on B, never on whichever row took its place.
root.menus = [menu('gpi-conversation-menu', 'session-a'), menu('gpi-conversation-menu', 'session-b')];
root.menus[1].open = true;
MENUS.keepOpen(root, () => { root.menus = [menu('gpi-conversation-menu', 'session-c'), menu('gpi-conversation-menu', 'session-b'), menu('gpi-conversation-menu', 'session-a')]; });
assert.deepEqual(root.menus.filter(m => m.open).map(m => m.popoverKey), ['session-b']);
MENUS.keepOpen(root, () => { root.menus = [menu('gpi-conversation-menu', 'session-a')]; });
assert.deepEqual(root.menus.filter(m => m.open), [], 'A removed row reopens nothing');

// The conversation shell and its conversation list repaint through it.
for (const file of consumerFiles) {
  assert.match(fs.readFileSync(file, 'utf8'), /menus\.keepOpen\((state\.host|rail), \(\) => \{ (state\.host|rail)\.innerHTML = markup; \}\)/, path.basename(file));
}
process.stdout.write('Open menus survive a repaint.\n');
