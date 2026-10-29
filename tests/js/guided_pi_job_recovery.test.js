/* A vanished Copilot job releases the composer. A temporary disconnect and
   a late error from an earlier conversation keep the current turn intact. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const shell = fs.readFileSync(process.argv[2], 'utf8');
const start = shell.indexOf('  function watchJob(');
assert.ok(start >= 0);
const owner = shell.slice(start, shell.indexOf('  async function sendText(', start));

function harness(snapshot) {
  const state = { busy: true, messages: [], session: { session_id: 'session-A' }, sessionSelectionRevision: 1,
    regenerating: true, regeneration: { activity: { steps: [] }, message: {} }, source: null };
  const finished = [];
  let rendered = 0;
  const context = vm.createContext({
    state,
    EventSource: class {
      constructor(url) { this.url = url; this.closed = false; }
      close() { this.closed = true; }
    },
    closeSource() { if (state.source) state.source.close(); state.source = null; },
    api: () => ({ loadJobSnapshot: snapshot }),
    projectId: () => 'project-A',
    finishActivity: (...args) => finished.push(args),
    render: () => { rendered += 1; },
    tr: en => en,
  });
  vm.runInContext(owner, context);
  context.watchJob('job-A');
  return { context, state, finished, rendered: () => rendered };
}

(async () => {
  const gone = harness(async () => { throw Object.assign(new Error('unknown job'), { status: 404 }); });
  const source = gone.state.source;
  await source.onerror();
  assert.equal(gone.state.busy, false);
  assert.equal(gone.state.regenerating, false);
  assert.equal(gone.state.regeneration, null);
  assert.equal(source.closed, true);
  assert.equal(gone.state.source, null);
  assert.match(gone.state.error, /resend/);
  assert.equal(gone.finished[0][0], 'error');
  assert.equal(gone.rendered(), 1);

  for (const snapshot of [async () => ({ status: 'running' }), async () => { throw Object.assign(new Error('offline'), { status: 503 }); }]) {
    const temporary = harness(snapshot);
    const active = temporary.state.source;
    await active.onerror();
    assert.equal(temporary.state.busy, true);
    assert.equal(active.closed, false);
    assert.equal(temporary.finished.length, 0);
    assert.equal(temporary.rendered(), 0);
  }

  let release;
  let checks = 0;
  const switched = harness(() => {
    checks += 1;
    return new Promise((_resolve, reject) => { release = () => reject(Object.assign(new Error('unknown job'), { status: 404 })); });
  });
  const earlier = switched.state.source;
  const pending = earlier.onerror();
  await earlier.onerror();
  assert.equal(checks, 1, 'Concurrent disconnect events share one snapshot check');
  switched.context.watchJob('job-B');
  const current = switched.state.source;
  release();
  await pending;
  assert.equal(switched.state.busy, true);
  assert.equal(current.closed, false);
  assert.equal(switched.state.source, current);
  assert.equal(switched.finished.length, 0);
  switched.state.busy = false;
  await earlier.onerror();
  assert.equal(current.closed, false, 'An earlier handler cannot close the current stream');
  process.stdout.write('A missing Copilot job releases the composer without disturbing another turn.\n');
})().catch(error => { console.error(error); process.exitCode = 1; });
