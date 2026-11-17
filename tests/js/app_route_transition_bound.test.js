'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

function setup() {
  const marks = [];
  const rendered = [];
  const transitions = [];
  const timers = new Map();
  const htmlClasses = new Set();
  const listeners = {};
  const location = { hash: '#cohort' };
  let timerId = 0;
  let allowedMotion = true;
  const app = {
    addEventListener() {}, querySelector() { return null; },
    querySelectorAll() { return []; }, setAttribute() {},
    set innerHTML(value) {
      rendered.push(value);
      this.firstElementChild = { attributes: {}, setAttribute(name, val) { this.attributes[name] = val; } };
    },
  };
  const document = {
    getElementById: () => app,
    createElement: () => ({ setAttribute() {} }),
    body: { appendChild() {} },
    documentElement: {
      toggleAttribute() {},
      classList: { add: (name) => htmlClasses.add(name), remove: (name) => htmlClasses.delete(name) },
    },
    addEventListener() {},
    startViewTransition(update) {
      let resolve;
      let rejectReady;
      const finished = new Promise((done) => { resolve = done; });
      const ready = new Promise((_, fail) => { rejectReady = fail; });
      const transition = {
        finished, ready, updateCallbackDone: finished,
        skips: 0, skipTransition() { this.skips++; rejectReady(new Error('skipped')); },
      };
      transitions.push({ update, resolve, transition });
      return transition;
    },
  };
  const window = {
    SCREENS: Object.fromEntries(['cohort', 'patient', 'extraction'].map((route) => [
      route, { render: () => route },
    ])),
    scrollTo() {},
    addEventListener(type, callback) { listeners[type] = callback; },
    matchMedia(query) { return { matches: query.includes('min-width') && allowedMotion }; },
    EU_BOOT: { mark: (name) => marks.push(name) },
  };
  vm.runInNewContext(source, {
    document, window, location, requestAnimationFrame: (callback) => callback(),
    setTimeout(fn, ms) { timers.set(++timerId, { fn, ms }); return timerId; },
    clearTimeout(id) { timers.delete(id); },
  });
  return {
    app, marks, rendered, transitions, timers, htmlClasses, reduceMotion: () => { allowedMotion = false; },
    go(route) { location.hash = '#' + route; listeners.hashchange(); },
  };
}

async function main() {
  const stalled = setup();
  assert.deepEqual(stalled.rendered, ['cohort']);
  stalled.go('patient');
  assert.deepEqual(stalled.rendered, ['cohort']);
  assert.equal(stalled.htmlClasses.has('eu-route-transitioning'), true);
  assert.equal(stalled.timers.size, 1, 'a transition that never calls update needs a deadline');
  const [{ fn, ms }] = stalled.timers.values();
  assert.ok(ms > 0 && ms <= 1000);
  fn();
  assert.deepEqual(stalled.rendered, ['cohort', 'patient']);
  assert.ok('data-eu-app-shell' in stalled.app.firstElementChild.attributes);
  const delayed = stalled.transitions[0];
  assert.equal(delayed.transition.skips, 1);
  assert.equal(stalled.htmlClasses.has('eu-route-transitioning'), false);
  delayed.update(); // skipTransition still permits the browser to call update.
  delayed.resolve();
  await delayed.transition.finished;
  assert.deepEqual(stalled.rendered, ['cohort', 'patient']);
  assert.deepEqual(stalled.marks, [
    'app_script', 'first_render_start', 'first_render_done',
    'view_transition_start', 'view_transition_skipped', 'view_transition_finished',
  ]);

  const rapid = setup();
  rapid.go('patient');
  rapid.go('extraction');
  const [old, current] = rapid.transitions;
  old.update();
  old.resolve();
  await old.transition.finished;
  assert.deepEqual(rapid.rendered, ['cohort']);
  assert.equal(rapid.htmlClasses.has('eu-route-transitioning'), true);
  assert.equal(old.transition.skips, 1);
  current.update();
  for (const timer of rapid.timers.values()) timer.fn();
  current.resolve();
  await current.transition.finished;
  assert.deepEqual(rapid.rendered, ['cohort', 'extraction']);
  assert.equal(current.transition.skips, 0);
  assert.equal(rapid.htmlClasses.has('eu-route-transitioning'), false);
  assert.equal(rapid.timers.size, 0);

  const reduced = setup();
  reduced.go('patient');
  reduced.reduceMotion();
  reduced.go('extraction');
  reduced.transitions[0].update();
  assert.deepEqual(reduced.rendered, ['cohort', 'extraction']);
  assert.equal(reduced.timers.size, 0);
  process.stdout.write('Transition deadline, late callbacks, supersession, reduced motion\n');
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
