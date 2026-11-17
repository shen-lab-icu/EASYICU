'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const appSource = fs.readFileSync(process.argv[3], 'utf8');

function element(tag) {
  return {
    tag, children: [], attributes: {}, listeners: {}, textContent: '',
    append(...children) {
      for (const child of children) child.parentNode = this;
      this.children.push(...children);
    },
    appendChild(child) { this.append(child); },
    remove() {
      const siblings = this.parentNode.children;
      siblings.splice(siblings.indexOf(this), 1);
      this.parentNode = null;
    },
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(type, callback) { this.listeners[type] = callback; },
  };
}

function setup(hash, headOnly = false) {
  const body = element('body');
  const app = element('div');
  body.append(app);
  app.querySelector = (selector) => {
    if (selector !== ':scope > [data-eu-app-shell]') return null;
    return app.children.find((child) => 'data-eu-app-shell' in child.attributes);
  };
  app.querySelectorAll = () => [];
  Object.defineProperties(app, {
    childElementCount: { get: () => app.children.length },
    firstElementChild: { get: () => app.children[0] },
    innerHTML: { set() { app.children.length = 0; app.append(element('main')); } },
  });
  const document = {
    body: headOnly ? null : body,
    createElement: element,
    readyState: headOnly ? 'loading' : 'complete',
    documentElement: { classList: { contains: () => true }, toggleAttribute() {} },
    getElementById: (id) => id === 'app' && document.body ? app : null,
    addEventListener() {},
    startViewTransition() {},
  };
  const timers = [];
  const listeners = {};
  const posts = [];
  let onBody;
  let now = 1500;
  let reloads = 0;
  const window = {
    innerWidth: 1440, SCREENS: {}, EU_API: {}, EU_GUIDED_STARTUP: {},
    location: { hash, reload() { reloads++; } },
    scrollTo() {},
    addEventListener(type, callback) { listeners[type] = callback; },
    fetch(url, opts) { posts.push({ url, opts }); return Promise.resolve(); },
  };
  const context = vm.createContext({
    document, window, performance: { now: () => now },
    location: window.location,
    requestAnimationFrame: (fn) => fn(),
    setTimeout: (fn, ms) => timers.push({ fn, ms }),
    MutationObserver: class {
      constructor(callback) { this.callback = callback; }
      observe() { onBody = this.callback; }
      disconnect() { onBody = null; }
    },
  });
  vm.runInContext(source, context);
  assert.equal(timers.length, 1);
  assert.equal(timers[0].ms, 43500);
  return {
    app, body, boot: window.EU_BOOT, listeners, posts,
    reloads: () => reloads,
    expire() { now = 45000; timers[0].fn(); },
    attachBody() { document.body = body; },
    notifyBody() { onBody?.(); },
    observingBody: () => !!onBody,
    mountApp() {
      window.SCREENS.guided = { render: () => '<main></main>' };
      vm.runInContext(appSource, context);
    },
  };
}

const healthy = setup('#guided');
const shell = element('main');
shell.setAttribute('data-eu-app-shell', '');
healthy.app.append(shell);
healthy.expire();
assert.equal(healthy.boot.failure, null);
assert.deepEqual(healthy.body.children, [healthy.app]);
assert.deepEqual(healthy.posts, []);

const stalled = setup('#patient?search=private%20question');
stalled.app.append(element('div'));
stalled.app.textContent = 'How does creatinine change after ICU admission?';
stalled.boot.mark('app_script');
stalled.boot.mark('first_render_start');
stalled.boot.mark('private mark');
stalled.listeners.error({ error: new TypeError('private message'), filename: '/asset?secret=1' });
stalled.listeners.unhandledrejection({ reason: { name: 'private error name' } });
for (let i = 0; i < 25; i++) stalled.listeners.error({ error: new RangeError() });
stalled.expire();
stalled.expire();

assert.equal(stalled.body.children.length, 2);
const stop = stalled.body.children[1];
assert.equal(stop.attributes.role, 'alert');
assert.equal(stop.attributes['data-app-boot-failed'], 'app_boot_stalled');
const [english, chinese, code, reload] = stop.children[0].children;
assert.match(english.textContent, /start/i);
assert.match(chinese.textContent, /启动/);
assert.equal(code.textContent, 'app_boot_stalled');
assert.equal(reload.tag, 'button');
assert.match(reload.textContent, /Reload/);
reload.listeners.click();
assert.equal(stalled.reloads(), 1);

assert.equal(stalled.posts.length, 1);
const [{ url, opts }] = stalled.posts;
assert.equal(url, '/api/client/boot-diagnostics');
assert.equal(opts.method, 'POST');
assert.equal(opts.credentials, 'same-origin');
assert.equal(opts.redirect, 'error');
test('the diagnostic POST requests an origin-only referrer policy', () => {
  assert.equal(opts.referrerPolicy, 'strict-origin');
});
assert.equal(opts.headers['Content-Type'], 'application/json');
assert.equal(stalled.boot.failure.code, 'app_boot_stalled');
assert.deepEqual(JSON.parse(opts.body), {
  code: 'app_boot_stalled',
  readyState: 'complete',
  marks: { app_script: 1500, first_render_start: 1500 },
  app_child_count: 1,
  app_text_length: stalled.app.textContent.length,
  presence: {
    SCREENS: true, __euRender: false, EU_API: true, EU_CATALOG: false,
    EU_CAPABILITIES: false, EU_GUIDED_STARTUP: true,
  },
  route_transitioning: true,
  view_transition_available: true,
  route_hash: '#patient',
  viewport_width_bucket: '1181_plus',
  errors: [
    { type: 'error', name: 'TypeError' },
    { type: 'unhandledrejection', name: 'UnknownError' },
    ...Array.from({ length: 6 }, () => ({ type: 'error', name: 'RangeError' })),
  ],
});

const unknownRoute = setup('#private-project-123/record');
unknownRoute.expire();
assert.equal(JSON.parse(unknownRoute.posts[0].opts.body).route_hash, '#other');

const headOnly = setup('#skills/private-record?token=private', true);
headOnly.expire();
assert.equal(headOnly.posts.length, 1);
assert.equal(headOnly.boot.failure.readyState, 'loading');
assert.equal(headOnly.boot.failure.app_child_count, 0);
assert.equal(headOnly.boot.failure.route_hash, '#skills');
headOnly.attachBody();
headOnly.notifyBody();
assert.equal(headOnly.body.children[1].attributes['data-app-boot-failed'], 'app_boot_stalled');
assert.equal(headOnly.posts.length, 1);

test('a late shell removes the watchdog overlay', () => {
  const late = setup('#guided');
  late.expire();
  assert.equal(late.body.children[1].attributes['data-app-boot-failed'], 'app_boot_stalled');
  late.mountApp();
  assert.ok('data-eu-app-shell' in late.app.firstElementChild.attributes);
  assert.equal(late.body.children.length, 2, 'only the app and route announcer remain');
  assert.equal(late.posts.length, 1);
  assert.equal(late.boot.failure.code, 'app_boot_stalled');
});

test('the body observer re-checks for a shell before showing the overlay', () => {
  const lateBody = setup('#guided', true);
  lateBody.expire();
  lateBody.attachBody();
  const mounted = element('main');
  mounted.setAttribute('data-eu-app-shell', '');
  lateBody.app.append(mounted);
  lateBody.notifyBody();
  assert.equal(lateBody.body.children.length, 1, 'the recovered app has no overlay');
  assert.equal(lateBody.observingBody(), false);
});

test('a shell mounted before observer delivery disconnects the watchdog observer', () => {
  const lateBody = setup('#guided', true);
  lateBody.expire();
  assert.equal(lateBody.observingBody(), true);
  lateBody.attachBody();
  lateBody.mountApp();
  assert.equal(lateBody.observingBody(), false);
  lateBody.notifyBody();
  assert.equal(lateBody.body.children.length, 2, 'only the app and route announcer remain');
});
process.stdout.write('Healthy shell, typed stop, bounded telemetry, safe route hashes, head-only timeout\n');
