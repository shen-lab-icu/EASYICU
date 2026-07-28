/* Executable lifecycle contract for Cross-DB raw-distribution reconnects. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(process.argv[2], 'utf8');
const KEY = 'easyicu_crossdb_raw_job_v1';
const META = {
  job_id: 'job_cross_01',
  kind: 'crossdb-raw-distribution',
  raw_root: '/raw/icu',
  source_identity: 'eicu,miiv',
  sample_mode: 'standard',
  feature_scope: 'all_catalog',
};

function storageWith(initial) {
  const values = new Map();
  if (initial !== undefined) values.set(KEY, initial);
  return {
    getItem(key) { return values.has(key) ? values.get(key) : null; },
    setItem(key, value) { values.set(key, String(value)); },
    removeItem(key) { values.delete(key); },
  };
}

function harness(options) {
  const opts = options || {};
  const calls = [];
  const timers = [];
  let timerId = 0;
  const state = { root: opts.currentRoot || '', identity: '' };
  const localStorage = storageWith(opts.stored);
  class FakeEventSource {
    static instances = [];
    constructor(url) {
      this.url = url;
      this.closed = false;
      FakeEventSource.instances.push(this);
    }
    close() { this.closed = true; }
    message(payload) { this.onmessage({ data: JSON.stringify(payload) }); }
    fail() { this.onerror(); }
  }
  const host = {
    canRestore() { return opts.canRestore !== false; },
    acceptResume(meta) {
      calls.push(['accept', meta.job_id]);
      if (state.root && state.root !== meta.raw_root) return false;
      state.root = meta.raw_root;
      state.identity = meta.source_identity;
      return true;
    },
    matchesSource(meta) { return state.root === meta.raw_root && state.identity === meta.source_identity; },
    onProbe(meta) { calls.push(['probe', meta.job_id]); },
    onRunning(meta, progress) { calls.push(['running', meta.job_id, progress && progress.phase]); },
    onProgress(meta, progress) { calls.push(['progress', meta.job_id, progress.current]); },
    onCancelRequested(meta) { calls.push(['cancel_requested', meta.job_id]); },
    onTerminal(meta, snapshot) {
      calls.push(['terminal', meta.job_id, snapshot.status, snapshot.result]);
      return opts.terminalAccept;
    },
    onUnavailable(meta) { calls.push(['unavailable', meta.job_id]); },
    onConnectionError(meta) { calls.push(['connection_error', meta.job_id]); },
  };
  let snapshotCalls = 0;
  const context = {
    console,
    encodeURIComponent,
    setTimeout(callback, delay) {
      const entry = { id: ++timerId, callback, delay, cancelled: false };
      timers.push(entry);
      return entry.id;
    },
    clearTimeout(id) {
      const entry = timers.find(item => item.id === id);
      if (entry) entry.cancelled = true;
    },
    EventSource: FakeEventSource,
    localStorage,
    EU_CROSSDB_JOB_HOST: host,
    EU_API: {
      async loadJobSnapshot() {
        snapshotCalls += 1;
        if (opts.snapshotError) throw opts.snapshotError;
        if (opts.snapshotFactory) return opts.snapshotFactory();
        return opts.snapshot;
      },
    },
  };
  context.window = context;
  vm.runInNewContext(source, vm.createContext(context), { filename: process.argv[2] });
  return {
    calls,
    context,
    EventSource: FakeEventSource,
    localStorage,
    snapshotCalls: () => snapshotCalls,
    nextTimerDelay() {
      const entry = timers.find(item => !item.cancelled);
      return entry ? entry.delay : null;
    },
    async runNextTimer() {
      const entry = timers.find(item => !item.cancelled);
      if (!entry) return false;
      entry.cancelled = true;
      await entry.callback();
      return true;
    },
  };
}

(async () => {
  const running = harness({
    stored: JSON.stringify(META),
    snapshot: {
      id: META.job_id,
      kind: META.kind,
      status: 'running',
      events: [{ type: 'progress', phase: 'database', current: 2, total: 6 }],
    },
  });
  assert.equal(await running.context.EU_CROSSDB_JOB_CONTINUITY.restoreIfNeeded(), true);
  assert.equal(running.snapshotCalls(), 1);
  assert.deepEqual(running.calls.slice(0, 3), [
    ['accept', META.job_id],
    ['probe', META.job_id],
    ['running', META.job_id, 'database'],
  ]);
  assert.equal(running.EventSource.instances.length, 1);
  assert.equal(running.EventSource.instances[0].url, '/api/jobs/job_cross_01/events');
  running.EventSource.instances[0].message({ type: 'progress', current: 3, total: 6 });
  assert.deepEqual(running.calls.at(-1), ['progress', META.job_id, 3]);
  running.EventSource.instances[0].message({ type: 'cancel_requested', reason: 'user_requested' });
  assert.deepEqual(running.calls.at(-1), ['cancel_requested', META.job_id]);
  const callsAfterCancel = running.calls.length;
  running.EventSource.instances[0].message({ type: 'progress', current: 4, total: 6 });
  assert.equal(running.calls.length, callsAfterCancel, 'late progress must not overwrite cancel state');
  const result = { source_type: 'raw_database_root', source_count: 2 };
  running.EventSource.instances[0].message({ type: 'end', status: 'cancelled', result: null });
  assert.deepEqual(running.calls.at(-1).slice(0, 3), ['terminal', META.job_id, 'cancelled']);
  assert.equal(running.calls.at(-1)[3], null);
  assert.equal(running.EventSource.instances[0].closed, true);
  assert.equal(running.localStorage.getItem(KEY), null, 'terminal metadata must not replay the same result or error after refresh');
  running.context.EU_CROSSDB_JOB_CONTINUITY.onSourceChanged('/raw/other', META.source_identity, META.sample_mode, META.feature_scope);
  assert.equal(running.localStorage.getItem(KEY), null, 'changing the raw root must forget the old job');

  for (const status of ['failed', 'cancelled']) {
    const terminal = harness({
      stored: JSON.stringify(META),
      snapshot: { id: META.job_id, kind: META.kind, status, events: [], result: null, error: status === 'failed' ? 'boom' : null },
    });
    assert.equal(await terminal.context.EU_CROSSDB_JOB_CONTINUITY.restoreIfNeeded(), true);
    assert.deepEqual(terminal.calls.at(-1).slice(0, 3), ['terminal', META.job_id, status]);
    assert.equal(terminal.localStorage.getItem(KEY), null);
  }

  const invalidResult = harness({
    stored: JSON.stringify(META),
    snapshot: { id: META.job_id, kind: META.kind, status: 'done', events: [], result: {} },
    terminalAccept: false,
  });
  assert.equal(await invalidResult.context.EU_CROSSDB_JOB_CONTINUITY.restoreIfNeeded(), false);
  assert.equal(invalidResult.localStorage.getItem(KEY), null, 'a rejected terminal result must not be replayed forever');

  const missing = harness({ stored: JSON.stringify(META), snapshotError: new Error('/api/jobs/job_cross_01 -> HTTP 404') });
  assert.equal(await missing.context.EU_CROSSDB_JOB_CONTINUITY.restoreIfNeeded(), false);
  assert.deepEqual(missing.calls.at(-1), ['unavailable', META.job_id]);
  assert.equal(missing.localStorage.getItem(KEY), null, 'server-restart/404 metadata must be cleared');

  const mismatchedRoot = harness({
    stored: JSON.stringify(META),
    snapshot: { id: META.job_id, kind: META.kind, status: 'running', events: [] },
    currentRoot: '/raw/other',
  });
  assert.equal(await mismatchedRoot.context.EU_CROSSDB_JOB_CONTINUITY.restoreIfNeeded(), false);
  assert.equal(mismatchedRoot.snapshotCalls(), 0);
  assert.equal(mismatchedRoot.localStorage.getItem(KEY), null);

  let resolveSnapshot;
  const pendingSnapshot = new Promise(resolve => { resolveSnapshot = resolve; });
  const changedDuringProbe = harness({
    stored: JSON.stringify(META),
    snapshotFactory: () => pendingSnapshot,
  });
  const pendingRestore = changedDuringProbe.context.EU_CROSSDB_JOB_CONTINUITY.restoreIfNeeded();
  changedDuringProbe.context.EU_CROSSDB_JOB_CONTINUITY.onSourceChanged('/raw/other', META.source_identity, META.sample_mode, META.feature_scope);
  resolveSnapshot({ id: META.job_id, kind: META.kind, status: 'running', events: [] });
  assert.equal(await pendingRestore, false);
  assert.equal(changedDuringProbe.EventSource.instances.length, 0, 'a stale snapshot must not attach after the root changes');

  const changedAfterConnect = harness({
    stored: JSON.stringify(META),
    snapshot: { id: META.job_id, kind: META.kind, status: 'running', events: [] },
  });
  assert.equal(await changedAfterConnect.context.EU_CROSSDB_JOB_CONTINUITY.restoreIfNeeded(), true);
  const oldStream = changedAfterConnect.EventSource.instances[0];
  changedAfterConnect.context.EU_CROSSDB_JOB_CONTINUITY.onSourceChanged('/raw/other', META.source_identity, META.sample_mode, META.feature_scope);
  assert.equal(oldStream.closed, true);
  const callsAfterSourceChange = changedAfterConnect.calls.length;
  oldStream.message({ type: 'progress', current: 5, total: 6 });
  oldStream.message({ type: 'end', status: 'done', result });
  assert.equal(changedAfterConnect.calls.length, callsAfterSourceChange, 'closed stale stream must not update the host');

  const replayWatermark = harness({
    stored: JSON.stringify(META),
    snapshot: {
      id: META.job_id,
      kind: META.kind,
      status: 'running',
      events: [
        { type: 'progress', phase: 'loading', current: 0, total: 6, seq: 0 },
        { type: 'progress', phase: 'database', current: 4, total: 6, seq: 4 },
      ],
    },
  });
  assert.equal(await replayWatermark.context.EU_CROSSDB_JOB_CONTINUITY.restoreIfNeeded(), true);
  const callsBeforeReplay = replayWatermark.calls.length;
  replayWatermark.EventSource.instances[0].message({ type: 'progress', current: 0, total: 6, seq: 0 });
  replayWatermark.EventSource.instances[0].message({ type: 'progress', current: 4, total: 6, seq: 4 });
  assert.equal(replayWatermark.calls.length, callsBeforeReplay, 'SSE replay must not regress snapshot progress');
  replayWatermark.EventSource.instances[0].message({ type: 'progress', current: 5, total: 6, seq: 5 });
  assert.deepEqual(replayWatermark.calls.at(-1), ['progress', META.job_id, 5]);

  const reconnectBackoff = harness({
    stored: JSON.stringify(META),
    snapshot: {
      id: META.job_id,
      kind: META.kind,
      status: 'running',
      events: [
        { type: 'progress', phase: 'loading', current: 0, total: 6, seq: 0 },
        { type: 'progress', phase: 'database', current: 4, total: 6, seq: 4 },
      ],
    },
  });
  assert.equal(await reconnectBackoff.context.EU_CROSSDB_JOB_CONTINUITY.restoreIfNeeded(), true);
  reconnectBackoff.EventSource.instances[0].fail();
  assert.equal(reconnectBackoff.snapshotCalls(), 1, 'stream failure must not probe synchronously');
  assert.equal(reconnectBackoff.nextTimerDelay(), 500);
  assert.equal(await reconnectBackoff.runNextTimer(), true);
  assert.equal(reconnectBackoff.snapshotCalls(), 2);
  assert.equal(reconnectBackoff.EventSource.instances.length, 2);
  reconnectBackoff.EventSource.instances[1].message({ type: 'progress', current: 0, total: 6, seq: 0 });
  reconnectBackoff.EventSource.instances[1].message({ type: 'progress', current: 4, total: 6, seq: 4 });
  const callsBeforeOldReconnectStream = reconnectBackoff.calls.length;
  reconnectBackoff.EventSource.instances[0].message({ type: 'progress', current: 5, total: 6, seq: 5 });
  reconnectBackoff.EventSource.instances[0].message({ type: 'end', status: 'done', result, seq: 6 });
  assert.equal(reconnectBackoff.calls.length, callsBeforeOldReconnectStream, 'replaced stream for the same job must stay fenced');
  reconnectBackoff.EventSource.instances[1].fail();
  assert.equal(reconnectBackoff.nextTimerDelay(), 1000);

  const invalid = harness({ stored: JSON.stringify({ ...META, raw_root: 'x'.repeat(4097) }), snapshot: null });
  assert.equal(await invalid.context.EU_CROSSDB_JOB_CONTINUITY.restoreIfNeeded(), false);
  assert.equal(invalid.localStorage.getItem(KEY), null);
  assert.equal(invalid.snapshotCalls(), 0);

  const started = harness({ snapshot: null });
  assert.equal(started.context.EU_CROSSDB_JOB_CONTINUITY.start(META, { phase: 'queued' }), true);
  assert.deepEqual(Object.keys(JSON.parse(started.localStorage.getItem(KEY))).sort(), [
    'feature_scope', 'job_id', 'kind', 'raw_root', 'sample_mode', 'source_identity',
  ]);
  started.context.EU_CROSSDB_JOB_CONTINUITY.onSourceChanged(META.raw_root, META.source_identity, 'deeper', META.feature_scope);
  assert.equal(started.localStorage.getItem(KEY), null, 'changing sample mode must not resume a differently scoped job');

  const scopeChanged = harness({ snapshot: null });
  assert.equal(scopeChanged.context.EU_CROSSDB_JOB_CONTINUITY.start(META, { phase: 'queued' }), true);
  scopeChanged.context.EU_CROSSDB_JOB_CONTINUITY.onSourceChanged(META.raw_root, META.source_identity, META.sample_mode, 'curated_core');
  assert.equal(scopeChanged.localStorage.getItem(KEY), null, 'changing feature scope must not resume a differently scoped job');

  process.stdout.write(JSON.stringify({
    restored: true,
    terminal_statuses: 3,
    missing_cleared: true,
    root_guard: true,
    late_progress_blocked: true,
    stale_stream_blocked: true,
    replay_watermark: true,
    reconnect_backoff: true,
    feature_scope_guard: true,
    same_job_stale_stream: true,
    terminal_pointer_cleared: true,
  }));
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
