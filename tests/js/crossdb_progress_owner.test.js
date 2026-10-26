/* Executable contract for Cross-DB structured progress and deferred cancel. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(process.argv[2], 'utf8');
const cancelCalls = [];
const context = { console };
context.window = context;
const sandbox = vm.createContext(context);
// Load the database-name owner as index.html does, before this widget.
const namesPath = process.argv[2].replace(/[^/]+$/, 'database-names.js');
vm.runInContext(fs.readFileSync(namesPath, 'utf8'), sandbox, { filename: namesPath });
vm.runInContext(source, sandbox, { filename: process.argv[2] });
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
// A database still waiting is named by the owner, not shown as its key.
assert.match(html, /crossdb-progress-db-name">eICU-CRD</);
assert.doesNotMatch(html, />eicu</);
assert.match(html, /is-complete/);
assert.match(html, /data-crossdb-cancel/);

// A registered-export summary names where it runs, and that the first full
// comparison takes minutes because every row is summarized.
progress.clear();
progress.attach('summary-job', { phase: 'queued', job_kind: 'crossdb-summary' });
const summaryOptions = hosted => ({
  esc(value) { return String(value); },
  hosted,
  icon() { return ''; },
  t(english, chinese) { return chinese; },
});
const hostedSummary = progress.render(summaryOptions(true));
assert.match(hostedSummary, /正在汇总已准备的数据…/);
assert.match(hostedSummary, /在医院服务器上计算 · 只返回汇总结果/);
assert.match(hostedSummary, /data-crossdb-first-run>每一行数据都会参与汇总，所以第一次对比两份完整数据可能需要几分钟；之后再对比会很快。</);
assert.doesNotMatch(hostedSummary, /本机|注册导出/);
// While it runs, the card says which step it is on, in the page language,
// and shows no sampling limits: every row is summarized.
progress.applyProgress({ type: 'progress', phase: 'summarizing', current: 1, total: 2, job_kind: 'crossdb-summary',
  message: 'Computing bounded cohort aggregates for registered exports.' });
const summarizing = progress.render({ ...summaryOptions(true), sampleProfile: { maxPatients: 200, sampleSize: 600 },
  progressMessage(value) { return String(value || ''); }, statusLabel(value) { return String(value || ''); } });
assert.match(summarizing, /正在汇总第 2 份数据（共 2 份）…/);
assert.doesNotMatch(summarizing, /≤|summarizing|Computing bounded/);
progress.applyProgress({ type: 'progress', phase: 'finalizing', current: 2, total: 2, job_kind: 'crossdb-summary' });
assert.match(progress.render(summaryOptions(true)), /正在检查各份数据能否对比…/);
const localSummary = progress.render(summaryOptions(false));
assert.match(localSummary, /仅本机 · 不上传任何内容/);
assert.match(localSummary, /正在加载已注册导出的聚合摘要…/);
assert.match(localSummary, /data-crossdb-first-run/);

process.stdout.write(JSON.stringify({
  deferred_cancel: true,
  hosted_summary_copy: true,
  cancel_api_guard: true,
  cancel_error_visible: true,
  cancel_history_restored: true,
  late_progress_blocked: true,
  structured_progress: true,
}));
