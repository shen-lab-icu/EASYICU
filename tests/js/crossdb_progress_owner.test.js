/* Executable contract for Cross-DB structured progress and deferred cancel. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(process.argv[2], 'utf8');
const cancelCalls = [];
const context = { console };
context.window = context;
vm.runInNewContext(source, vm.createContext(context), { filename: process.argv[2] });
const progress = context.EU_CROSSDB_PROGRESS;

progress.beginStart();
assert.equal(progress.snapshot().starting, true);
assert.equal(progress.requestCancel({
  api: { cancelJob(jobId, reason) { cancelCalls.push([jobId, reason]); return Promise.resolve({}); } },
}), true);
assert.equal(cancelCalls.length, 0, 'cancel before submission returns must stay deferred');

progress.attach('job-1', { phase: 'queued', databases: ['miiv', 'eicu'] });
assert.equal(progress.flushCancel({
  cancelJob(jobId, reason) { cancelCalls.push([jobId, reason]); return Promise.resolve({}); },
}), true);
assert.deepEqual(cancelCalls, [['job-1', 'user_requested']]);
assert.equal(progress.applyProgress({ type: 'progress', phase: 'database', database: 'miiv' }), false, 'late progress must not overwrite cancel state');
assert.equal(progress.snapshot().progress.phase, 'cancel');

let missingApiError = '';
progress.clear();
progress.beginStart();
progress.attach('job-missing-api', { phase: 'queued' });
progress.requestCancel({
  api: null,
  onError(error) { missingApiError = String(error && error.message || error); },
});
assert.match(missingApiError, /cancel API unavailable/);
assert.equal(progress.snapshot().cancelRequested, false, 'missing API must not leave cancel permanently disabled');
assert.equal(progress.snapshot().progress.phase, 'queued', 'failed cancel must restore the prior progress context');
const missingApiHtml = progress.render({
  errorMessage: missingApiError,
  esc(value) { return String(value); },
  icon() { return ''; },
  t(english) { return english; },
});
assert.match(missingApiHtml, /role="alert"/);
assert.match(missingApiHtml, /Cross-DB cancel API unavailable/);

let rejectedApiError = '';
progress.requestCancel({
  api: {
    cancelJob() {
      return { catch(handler) { handler(new Error('Cancel request rejected')); } };
    },
  },
  onError(error) { rejectedApiError = String(error && error.message || error); },
});
assert.match(rejectedApiError, /Cancel request rejected/);
assert.equal(progress.snapshot().cancelRequested, false, 'rejected cancel must restore the prior progress context');

progress.clear();
progress.resume('job-history', { type: 'cancel_requested' }, [
  { type: 'progress', phase: 'loading', databases: ['miiv', 'eicu'], current: 0, total: 2 },
  { type: 'progress', phase: 'database', database: 'miiv', database_label: 'MIMIC-IV', database_status: 'complete', current: 1, total: 2 },
  { type: 'cancel_requested', reason: 'user_requested' },
]);
assert.equal(progress.snapshot().cancelRequested, true);
assert.equal(progress.snapshot().databases[0].status, 'complete', 'resume must retain per-database history before cancel');

progress.clear();
progress.beginStart();
progress.attach('job-2', {
  type: 'progress', phase: 'loading', current: 0, total: 2,
  databases: ['miiv', 'eicu'], completed_chunks: 0, total_chunks: 4,
});
progress.applyProgress({
  type: 'progress', phase: 'database', database: 'miiv', database_label: 'MIMIC-IV',
  database_status: 'loading', current: 0, total: 2, completed_chunks: 0, total_chunks: 4,
  chunk_current: 0, chunk_total: 2,
});
progress.applyProgress({
  type: 'progress', phase: 'chunk', database: 'miiv', database_label: 'MIMIC-IV',
  database_status: 'loading', chunk_status: 'complete', current: 0, total: 2,
  completed_chunks: 1, total_chunks: 4, chunk_current: 1, chunk_total: 2,
});
progress.applyProgress({
  type: 'progress', phase: 'database', database: 'miiv', database_label: 'MIMIC-IV',
  database_status: 'complete', current: 1, total: 2, completed_chunks: 2, total_chunks: 4,
  chunk_current: 2, chunk_total: 2,
});

const html = progress.render({
  esc(value) { return String(value); },
  fmtInt(value) { return String(value); },
  icon() { return ''; },
  progressMessage(value) { return String(value || ''); },
  sampleProfile: { maxPatients: 200, sampleSize: 600 },
  statusLabel(value) { return String(value || ''); },
  t(english) { return english; },
});
assert.match(html, /role="status"/);
assert.match(html, /aria-live="polite"/);
assert.match(html, /<progress[^>]+value="50"/);
assert.match(html, /MIMIC-IV/);
assert.match(html, /eicu/);
assert.match(html, /is-complete/);
assert.match(html, /data-crossdb-cancel/);

process.stdout.write(JSON.stringify({
  deferred_cancel: true,
  cancel_api_guard: true,
  cancel_error_visible: true,
  cancel_history_restored: true,
  late_progress_blocked: true,
  structured_progress: true,
}));
