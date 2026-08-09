'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.resolve(process.argv[2]), 'utf8');
const sandbox = {
  window: {
    t: (english) => english,
    icon: () => '',
  },
};
vm.runInNewContext(source, sandbox);
const renderer = sandbox.window.AGENT_RENDER;

const hostileLabel = 'figure" onerror="globalThis.pwned=1';
const safePng = 'data:image/png;base64,iVBORw0KGgo=';
const escaped = renderer.figureGallery({
  figures: [{ label: hostileLabel, data_url: safePng }],
});
assert.ok(escaped.includes('&quot;'), 'attribute quotes must be entity escaped');
assert.ok(!escaped.includes('alt="figure" onerror='), 'label must not create a new attribute');

const hostileSource = renderer.figureGallery({
  figures: [{
    label: 'bad source',
    data_url: 'data:image/png;base64,AAAA" onerror="globalThis.pwned=1',
  }],
});
assert.equal(hostileSource, '', 'malformed image data URL must not create an image');

const activeData = renderer.figureGallery({
  figures: [{ label: 'html', data_url: 'data:text/html,<script>alert(1)</script>' }],
});
assert.equal(activeData, '', 'only bounded PNG data URLs may render');

process.stdout.write(JSON.stringify({ ok: true, cases: 3 }));
