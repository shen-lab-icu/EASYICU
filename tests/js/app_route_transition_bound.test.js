'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const apiSource = fs.readFileSync(process.argv[3], 'utf8');

function setup() {
  const marks = [];
  const rendered = [];
  const focused = [];
  const scrolls = [];
  const transitions = [];
  const timers = new Map();
  const htmlClasses = new Set();
  const listeners = {};
  const documentListeners = {};
  const hydrationResponses = {
    '/api/catalog': { totalConcepts: 0 },
    '/api/settings': { language: 'en' },
    '/api/workspaces/registry': { sources: [] },
    '/api/capabilities': { deployment: { hosted: false } },
  };
  const location = { hash: '#cohort' };
  let timerId = 0;
  let allowedMotion = true;
  const app = {
    addEventListener() {}, querySelector() { return null; },
    querySelectorAll() { return []; }, setAttribute() {},
    dataset: {},
    focus() { focused.push(rendered.at(-1)); },
    set innerHTML(value) {
      rendered.push(value);
      this.firstElementChild = { attributes: {}, setAttribute(name, val) { this.attributes[name] = val; } };
    },
  };
  const document = {
    readyState: 'loading',
    querySelector() { return null; },
    getElementById(id) {
      if (id === 'app') return app;
      if (id === 'gdPiShell' && rendered.at(-1) === 'guided') return app.firstElementChild;
      return null;
    },
    createElement: () => ({ setAttribute() {} }),
    body: { appendChild() {} },
    documentElement: {
      toggleAttribute() {},
      classList: { add: (name) => htmlClasses.add(name), remove: (name) => htmlClasses.delete(name) },
    },
    addEventListener(type, callback) { documentListeners[type] = callback; },
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
    SCREENS: Object.fromEntries(['cohort', 'guided', 'patient', 'extraction'].map((route) => [
      route, { render: () => route },
    ])),
    EU_LANG: 'en',
    scrollX: 24,
    scrollY: 120,
    scrollTo(x, y) { scrolls.push([x, y]); },
    addEventListener(type, callback) { listeners[type] = callback; },
    matchMedia(query) { return { matches: query.includes('min-width') && allowedMotion }; },
    EU_BOOT: { mark: (name) => marks.push(name) },
  };
  const context = vm.createContext({
    document, window, location, requestAnimationFrame: (callback) => callback(),
    setTimeout(fn, ms) { timers.set(++timerId, { fn, ms }); return timerId; },
    clearTimeout(id) { timers.delete(id); },
    async fetch(path) {
      assert.ok(Object.hasOwn(hydrationResponses, path), path);
      return { ok: true, json: async () => hydrationResponses[path] };
    },
  });
  vm.runInContext(source, context);
  return {
    app, marks, rendered, focused, scrolls, transitions, timers, htmlClasses,
    reduceMotion: () => { allowedMotion = false; },
    go(route) { location.hash = '#' + route; listeners.hashchange(); },
    startHydration() {
      vm.runInContext(apiSource, context);
      return documentListeners.DOMContentLoaded();
    },
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

  for (const first of ['update', 'deadline']) {
    const hydration = setup();
    const boot = hydration.startHydration();
    hydration.go('guided');
    const [deferred] = hydration.transitions;
    const [timer] = hydration.timers.values();
    assert.deepEqual(hydration.rendered, ['cohort']);
    await boot;
    assert.ok(hydration.marks.includes('boot_rerender'));
    assert.deepEqual(hydration.rendered, ['cohort', 'guided']);
    const guidedShell = hydration.app.firstElementChild;
    const lateUpdates = { update: deferred.update, deadline: timer.fn };
    lateUpdates[first]();
    assert.equal(hydration.app.firstElementChild, guidedShell, `late ${first} replaced the Guided shell`);
    lateUpdates[first === 'update' ? 'deadline' : 'update']();
    deferred.resolve();
    await deferred.transition.finished;
    assert.deepEqual(hydration.rendered, ['cohort', 'guided']);
    assert.deepEqual(hydration.focused, ['cohort', 'guided']);
    assert.deepEqual(hydration.scrolls, [[0, 0], [0, 0]]);
    assert.equal(deferred.transition.skips, 1);
    assert.equal(hydration.timers.size, 0);
    assert.equal(hydration.htmlClasses.has('eu-route-transitioning'), false);
  }
  process.stdout.write('Transition deadline, late callbacks, supersession, reduced motion, hydration\n');
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
