/* An open floating menu keeps to the room it has. The model menu under the
   entry composer ran past a 680-pixel window and past the conversation's
   scrolling region; a panel now takes the room on the side it opens toward
   and scrolls inside it. */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');

const [ownerFile] = process.argv.slice(2).map(file => path.resolve(file));
const styles = new Map();
const body = { tagName: 'BODY' };
global.window = global;
global.innerHeight = 680;
global.getComputedStyle = node => styles.get(node) || { overflowY: 'visible', position: 'static' };
global.document = { addEventListener() {}, querySelectorAll: () => [], body, documentElement: { tagName: 'HTML' } };
require(ownerFile);
const MENUS = global.EU_POPOVER_MENUS;

const box = (top, height) => ({ top, bottom: top + height, height });
function element(tagName, parentElement, rect) {
  const style = {
    removeProperty(name) { if (name === 'max-height') delete this.maxHeight; },
  };
  return { tagName, parentElement, style, getBoundingClientRect: () => rect };
}
// A <details> menu: its summary at summaryTop, its panel at panelRect, inside
// a region (the conversation log) that clips at regionRect.
function menu({ summaryTop = 390, panelRect, regionRect = box(0, 680), regionOverflow = 'auto', panelPosition = 'absolute' }) {
  const region = element('DIV', body, regionRect);
  styles.set(region, { overflowY: regionOverflow, position: 'static' });
  const details = element('DETAILS', region, box(summaryTop, 28));
  const summary = element('SUMMARY', details, box(summaryTop, 28));
  const panel = element('DIV', details, panelRect);
  styles.set(panel, { overflowY: 'auto', position: panelPosition });
  details.children = [summary, panel];
  details.open = true;
  details.querySelector = selector => (selector === ':scope > summary' ? summary : null);
  details.matches = selector => selector === MENUS.SELECTOR;
  return { details, panel };
}

// Below the entry composer: 312 pixels from y=450 in a 680-pixel window.
let { details, panel } = menu({ panelRect: box(450, 312) });
MENUS.onToggle({ target: details });
assert.equal(panel.style.maxHeight, '222px', 'the panel stops 8 pixels above the window edge');
assert.equal(panel.style.overflowY, 'auto');

// The scrolling region that holds the composer ends before the window does.
({ details, panel } = menu({ panelRect: box(450, 312), regionRect: box(40, 560) }));
MENUS.fit(details);
assert.equal(panel.style.maxHeight, '142px', 'the region narrows the room');

// A region that neither scrolls nor clips does not narrow it.
({ details, panel } = menu({ panelRect: box(450, 312), regionRect: box(40, 560), regionOverflow: 'visible' }));
MENUS.fit(details);
assert.equal(panel.style.maxHeight, '222px');

// A fixed panel answers to the window alone.
({ details, panel } = menu({ panelRect: box(450, 312), regionRect: box(40, 560), panelPosition: 'fixed' }));
MENUS.fit(details);
assert.equal(panel.style.maxHeight, '222px');

// A panel that opens above its summary keeps to the top edge.
({ details, panel } = menu({ summaryTop: 300, panelRect: box(-40, 330) }));
MENUS.fit(details);
assert.equal(panel.style.maxHeight, '282px', 'from the top edge plus 8 pixels to the panel bottom');

// A panel that fits keeps its own height, and a bound from an earlier
// opening in a shorter window is dropped.
({ details, panel } = menu({ panelRect: box(450, 208) }));
panel.style.maxHeight = '100px';
MENUS.fit(details);
assert.equal(panel.style.maxHeight, undefined);

// A panel that grows after it opens, as the model list arriving from the
// host does, is fitted again; a closed one is no longer watched.
const watched = new Set();
let resized = null;
global.ResizeObserver = class {
  constructor(callback) { resized = callback; }
  observe(node) { watched.add(node); }
  unobserve(node) { watched.delete(node); }
};
let rect = box(450, 120);
({ details, panel } = menu({ panelRect: box(450, 120) }));
panel.getBoundingClientRect = () => rect;
MENUS.onToggle({ target: details });
assert.equal(panel.style.maxHeight, undefined, 'the loading panel fits');
assert.ok(watched.has(panel));
rect = box(450, 312);
resized([{ target: panel }]);
assert.equal(panel.style.maxHeight, '222px', 'the grown panel keeps to the room');
details.open = false;
resized([{ target: panel }]);
assert.ok(!watched.has(panel));

// Closing a menu fits nothing.
({ details, panel } = menu({ panelRect: box(450, 312) }));
details.open = false;
MENUS.onToggle({ target: details });
assert.equal(panel.style.maxHeight, undefined);
process.stdout.write('An open menu keeps to the room it has.\n');
