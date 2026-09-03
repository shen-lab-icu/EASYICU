'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const ownerPath = process.argv[2];
assert.ok(ownerPath, 'screens-extraction-embedded.js path is required');

let paintCount = 0;
let scroller = null;
const host = {
  isConnected: true,
  querySelector(selector) {
    return selector === '[data-gpi-extraction-embed]' ? scroller : null;
  },
  querySelectorAll() { return []; },
  set innerHTML(_value) {
    paintCount += 1;
    scroller = { scrollTop: 0 };
  },
};

global.icon = () => '';
global.t = en => en;
global.escHtml = value => String(value == null ? '' : value);
global.window = {
  EU_EXTRACTION_NATIVE_OWNER: {
    render: () => '<section></section>',
    bind: () => {},
    isReal: () => true,
    isPreparedExport: () => false,
  },
};

vm.runInThisContext(fs.readFileSync(ownerPath, 'utf8'), { filename: ownerPath });

window.EU_EXTRACTION_EMBEDDED_WORKSPACE.mount(host, {});
assert.equal(paintCount, 1);
scroller.scrollTop = 640;

window.EU_EXTRACTION_EMBEDDED_WORKSPACE.repaint();

assert.equal(paintCount, 2);
assert.equal(scroller.scrollTop, 640, 'repaint must preserve the right preview scroll position');
console.log('extraction embedded scroll contract passed');
