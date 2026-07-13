/* Executable contract for extract/convert refresh continuity. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

const STORAGE_KEY = 'easyicu.extractionJob.v1';
const storage = new Map();
const snapshots = new Map();
const cancelled = [];
const began = [];
const events = [];
const missing = [];
const connectionLosses = [];
const streams = [];

global.window = global;
global.localStorage = {
  getItem: key => storage.has(key) ? storage.get(key) : null,
  setItem: (key, value) => storage.set(key, String(value)),
  removeItem: key => storage.delete(key),
};
global.EU_API = {
  loadJobSnapshot: async jobId => {
    const value = snapshots.get(jobId);
    if (value instanceof Error) throw value;
    return value;
  },
  cancelJob: async (jobId, reason) => { cancelled.push({ jobId, reason }); },
};
global.EU_EXTRACTION_JOB_HOST = {
  begin: record => began.push(record),
  applyEvent: (record, event) => events.push({ record, event }),
  missing: record => missing.push(record),
  connectionLost: (record, error) => connectionLosses.push({ record, error }),
};
global.EventSource = class FakeEventSource {
  constructor(url) {
    this.url = url;
    this.closed = false;
    streams.push(this);
  }
  close() { this.closed = true; }
  emit(message) { this.onmessage({ data: JSON.stringify(message) }); }
};

storage.set(STORAGE_KEY, JSON.stringify({
  job_id: 'convert_1',
  kind: 'convert',
  source: { path: '/data/raw', database: 'miiv', ignored: 'drop-me' },
  config: { ignored: 'drop-me' },
  result: { converted: 999 },
}));
snapshots.set('convert_1', {
  id: 'convert_1',
  kind: 'convert',
  status: 'running',
  events: [{ type: 'progress', current: 2, total: 8, file: 'chartevents.csv' }],
  result: null,
  error: null,
});

require(path.resolve(process.argv[2]));

(async () => {
  const owner = global.EU_EXTRACTION_JOB_CONTINUITY;
  await owner.ready;

  assert.equal(began.at(-1).job_id, 'convert_1');
  assert.equal(owner.isRunning(), true, 'a reconciled running snapshot is active');
  assert.equal(events.at(-1).event.current, 2);
  assert.equal(streams.at(-1).url, '/api/jobs/convert_1/events');
  assert.deepEqual(Object.keys(JSON.parse(storage.get(STORAGE_KEY))).sort(), ['config', 'job_id', 'kind', 'source']);
  assert.deepEqual(JSON.parse(storage.get(STORAGE_KEY)).config, {});

  streams.at(-1).emit({ type: 'end', status: 'cancelled', result: { converted: 2, failed: 0 } });
  assert.equal(events.at(-1).event.status, 'cancelled');
  assert.equal(owner.isRunning(), false, 'a terminal event is not still running');
  assert.equal(streams.at(-1).closed, true);
  assert.equal(storage.has(STORAGE_KEY), true, 'terminal metadata stays available for refresh restore');
  owner.abandon();
  assert.equal(storage.has(STORAGE_KEY), false, 'explicit continue/reset must remove terminal metadata');

  const staleTicket = owner.prepare({
    kind: 'extract',
    source: { path: '/exports/a', database: 'eicu' },
    config: { modules: ['vitals'] },
  });
  owner.abandon();
  assert.equal(owner.attach(staleTicket, 'stale_job'), null);
  await Promise.resolve();
  assert.deepEqual(cancelled.at(-1), { jobId: 'stale_job', reason: 'source_changed_before_tracking' });

  const oversizedModules = Array.from({ length: 100 }, (_, i) => `module-${i}-${'x'.repeat(200)}`);
  const extractTicket = owner.prepare({
    kind: 'extract',
    source: { path: '/exports/current', database: 'eicu' },
    config: {
      run_mode: 'recommended',
      modules: oversizedModules,
      format: 'parquet',
      merge: true,
      max_patients: 999999999,
      out_dir: '/exports/output',
      secret: 'must-not-persist',
    },
  });
  assert.ok(owner.attach(extractTicket, 'extract_2'));
  assert.equal(owner.isRunning(), true, 'a newly attached job is running');
  const persisted = JSON.parse(storage.get(STORAGE_KEY));
  assert.deepEqual(Object.keys(persisted).sort(), ['config', 'job_id', 'kind', 'source']);
  assert.equal(persisted.config.modules.length, 64);
  assert.ok(persisted.config.modules.every(module => module.length <= 128));
  assert.equal(persisted.config.max_patients, 10000000);
  assert.equal(Object.hasOwn(persisted.config, 'secret'), false);

  streams.at(-1).emit({ type: 'end', status: 'failed', error: 'disk full' });
  assert.equal(events.at(-1).event.status, 'failed');
  assert.equal(owner.isRunning(), false, 'a failed job is terminal');
  assert.equal(storage.has(STORAGE_KEY), true);

  snapshots.set('extract_2', {
    id: 'extract_2', kind: 'extract', status: 'done', events: [],
    result: { out_dir: '/exports/output/run-1', total_rows: 42 }, error: null,
  });
  await owner.restore();
  assert.equal(events.at(-1).event.status, 'done', 'terminal restore must use the server snapshot');
  assert.equal(events.at(-1).event.result.total_rows, 42);
  assert.equal(owner.isRunning(), false, 'a restored done snapshot is terminal');

  owner.abandon();
  storage.set(STORAGE_KEY, JSON.stringify({
    job_id: 'missing_3', kind: 'convert',
    source: { path: '/data/old', database: 'miiv' }, config: {},
  }));
  snapshots.set('missing_3', new Error('/api/jobs/missing_3 -> HTTP 404'));
  await owner.restore();
  assert.equal(missing.at(-1).job_id, 'missing_3');
  assert.equal(storage.has(STORAGE_KEY), false, 'server restart/expired history clears the pointer');
  assert.equal(connectionLosses.length, 0);

  process.stdout.write(JSON.stringify({ restored: true, bounded: true, missingCleared: true }));
})().catch(error => {
  process.stderr.write(String(error && error.stack || error));
  process.exitCode = 1;
});
