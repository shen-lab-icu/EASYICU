/* Executable contract: a guided startup never stays behind its cover silently.
   pilot1 11-17: the first load after a restart rendered the cover, sent no
   startup request and showed nothing for 600 s. A throw in the first render's
   synchronous startup, or a startup that never settles, is now a typed stop. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

function node(tag) {
  const element = {
    tag, attributes: {}, children: [], listeners: {}, textContent: '', className: '',
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return this.attributes[name]; },
    append(...items) { this.children.push(...items); },
    addEventListener(type, fn) { this.listeners[type] = fn; },
    querySelector(selector) { return selector === '.gd-startup-card' ? this.card : null; },
    remove() { this.removed = true; },
  };
  return element;
}
const shield = node('div');
shield.card = node('div');
const timers = [];
const errors = [];
global.window = global;
global.document = {
  documentElement: { lang: 'en' },
  createElement: node,
  querySelectorAll(selector) { return selector === '[data-guided-startup-shield]' ? [shield] : []; },
};
global.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length; };
global.clearTimeout = () => {};
global.console = { ...console, error: (...args) => errors.push(args) };
require(path.resolve(process.argv[2]));

const startup = global.EU_GUIDED_STARTUP;
assert.equal(typeof startup.guard, 'function');

// Outside the initial startup an error propagates unchanged.
assert.throws(() => startup.guard(() => { throw new TypeError('later'); }), TypeError);

// During startup a synchronous throw becomes a readable, typed stop.
startup.begin();
assert.equal(startup.guard(() => { throw new TypeError('x is undefined'); }), undefined);
assert.equal(startup.isActive(), false);
assert.equal(shield.getAttribute('data-guided-startup-failed'), 'guided_startup_render_failed');
assert.equal(shield.getAttribute('role'), 'alert');
assert.match(shield.card.children[0].textContent, /guided_startup_render_failed · TypeError/);
assert.equal(shield.card.children[1].textContent, 'Reload');
assert.equal(global.EU_GUIDED_STARTUP_FAILURE.code, 'guided_startup_render_failed');
assert.equal(errors.length, 1);

// A startup that never finishes is stopped by its bounded watchdog.
startup.begin();
const watchdog = timers[timers.length - 1];
assert.equal(watchdog.ms, 120000);
watchdog.fn();
assert.equal(shield.getAttribute('data-guided-startup-failed'), 'guided_startup_timeout');

// A startup that finished is never stopped afterwards.
startup.begin();
const settled = timers[timers.length - 1];
startup.finish(global.document);
shield.attributes['data-guided-startup-failed'] = 'unchanged';
settled.fn();
assert.equal(shield.getAttribute('data-guided-startup-failed'), 'unchanged');

process.stdout.write(JSON.stringify({ render_throw_typed: true, watchdog: true, settled_untouched: true }));
