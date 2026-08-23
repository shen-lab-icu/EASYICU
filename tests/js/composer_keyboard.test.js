/* Executable contract shared by both Guided Copilot composers. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
require(path.resolve(process.argv[2]));

const owner = global.EU_COMPOSER_KEYBOARD;
assert(owner, 'composer keyboard owner must be published');
assert.equal(owner.enterShouldSend({ key: 'Enter', shiftKey: false, isComposing: false, keyCode: 13 }), true);
assert.equal(owner.enterShouldSend({ key: 'Enter', shiftKey: true, isComposing: false, keyCode: 13 }), false);
assert.equal(owner.enterShouldSend({ key: 'Enter', shiftKey: false, isComposing: true, keyCode: 13 }), false);
assert.equal(owner.enterShouldSend({ key: 'Enter', shiftKey: false, isComposing: false, keyCode: 229 }), false);
assert.equal(owner.enterShouldSend({ key: 'a', shiftKey: false, isComposing: false, keyCode: 65 }), false);

process.stdout.write(JSON.stringify({ enter: true, shift_enter: true, ime_safe: true }));
