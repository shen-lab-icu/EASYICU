'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
}

class FakeEventSource {
  constructor(jobId) {
    this.jobId = jobId;
    this.closed = false;
    this.onmessage = null;
  }
  emit(payload) {
    if (this.onmessage) this.onmessage({ data: JSON.stringify(payload) });
  }
  close() { this.closed = true; }
}

function memoryStorage() {
  const values = new Map();
  return {
    getItem(key) { return values.has(key) ? values.get(key) : null; },
    setItem(key, value) { values.set(key, String(value)); },
    removeItem(key) { values.delete(key); },
  };
}

const contexts = new Map([
  ['context-a', { id: 'context-a', current_stage: 'study_setup', active_job_id: null, revision: 1 }],
  ['context-b', { id: 'context-b', current_stage: 'study_setup', active_job_id: null, revision: 4 }],
]);
let activeId = 'context-b';
const patches = [];

global.window = global;
global.t = en => en;
global.icon = () => '';
global.addEventListener = () => {};
global.EU_STUDY_CONTEXT = {
  active: () => ({ ...contexts.get(activeId) }),
  patchContext(id, patch, options) {
    const current = contexts.get(id);
    if (!current) return null;
    if (Object.hasOwn(options || {}, 'expectedActiveJobId')
        && current.active_job_id !== options.expectedActiveJobId) return null;
    const next = { ...current, ...patch };
    contexts.set(id, next);
    patches.push({ id, patch: { ...patch }, options: { ...(options || {}) } });
    return { ...next };
  },
};

require(path.resolve(process.argv[2]));

const owner = global.EU_AGENT_STUDY_CONTEXT;
const channel = owner.createRunChannel();
const jobMemory = owner.createJobMemory(memoryStorage(), 'agent-jobs-test');
const uiEvents = [];

async function submitRun(label, contextPromise, submitPromise) {
  const capturedQuestion = `question-${label}`;
  let token = channel.start({
    surface: label,
    study_id: `study-${label}`,
    question: capturedQuestion,
    source_path: `/exports/${label}`,
  });
  assert.equal(Object.isFrozen(token), true);
  const context = await contextPromise;
  token = channel.bind(token, { context_id: context.id });
  const response = await submitPromise;
  token = channel.bind(token, {
    job_id: response.job_id,
    context_revision: response.study_context_revision,
  });
  owner.markContextRunning(token.context_id, token.job_id, token.context_revision);
  jobMemory.remember({ study_id: token.study_id, job_id: token.job_id, context_id: token.context_id });
  const stream = new FakeEventSource(token.job_id);
  stream.onmessage = event => {
    const payload = JSON.parse(event.data);
    if (payload.type === 'end') {
      owner.markContextFinished(
        token.context_id,
        payload.status,
        payload.result,
        token.job_id,
        payload.result && payload.result.study_context_revision,
      );
      jobMemory.clear(token.job_id, token.study_id);
    }
    if (channel.isCurrent(token)) uiEvents.push([label, payload.type, payload.label || payload.status]);
  };
  return { stream, token };
}

(async () => {
  const contextA = deferred();
  const submitA = deferred();
  const pendingA = submitRun('a', contextA.promise, submitA.promise);

  const contextB = deferred();
  const submitB = deferred();
  const pendingB = submitRun('b', contextB.promise, submitB.promise);

  contextA.resolve({ id: 'context-a' });
  submitA.resolve({ job_id: 'job-a', study_context_revision: 2 });
  const runA = await pendingA;
  assert.equal(channel.isCurrent(runA.token), false, 'late A submission must not retake the B UI channel');

  contextB.resolve({ id: 'context-b' });
  submitB.resolve({ job_id: 'job-b', study_context_revision: 5 });
  const runB = await pendingB;
  assert.equal(channel.isCurrent(runB.token), true);
  assert.equal(runA.token.question, 'question-a', 'the run keeps the question captured at start');

  runA.stream.emit({ type: 'progress', label: 'A progress' });
  runB.stream.emit({ type: 'progress', label: 'B progress' });
  assert.deepEqual(uiEvents, [['b', 'progress', 'B progress']], 'stale A progress must not paint over B');

  runA.stream.emit({
    type: 'end', status: 'done',
    result: { study_id: 'context-a', study_context_revision: 3, gate: { status: 'analysis_only' } },
  });
  assert.equal(contexts.get('context-a').current_stage, 'review');
  assert.equal(contexts.get('context-a').active_job_id, null);
  assert.equal(contexts.get('context-b').active_job_id, 'job-b', 'A terminal must not clear B');
  assert.equal(global.EU_AGENT_LAST_RUN, undefined, 'inactive A must not replace the global last run');
  assert.equal(jobMemory.get('study-a'), null);
  assert.equal(jobMemory.get('study-b').job_id, 'job-b', 'remembered jobs are scoped by study');

  runB.stream.emit({
    type: 'end', status: 'done',
    result: { study_id: 'context-b', study_context_revision: 6, gate: { status: 'blocked' } },
  });
  assert.equal(contexts.get('context-b').current_stage, 'review_blocked');
  assert.equal(contexts.get('context-b').revision, 6);
  assert.equal(global.EU_AGENT_LAST_RUN.study_context_id, 'context-b');
  assert.equal(jobMemory.get('study-b'), null);
  assert.ok(patches.some(row => row.id === 'context-a' && row.options.expectedActiveJobId === 'job-a'));
  assert.ok(patches.some(row => row.id === 'context-b' && row.options.expectedActiveJobId === 'job-b'));

  process.stdout.write(JSON.stringify({ ok: true, ui_events: uiEvents.length, patches: patches.length }));
})().catch(error => {
  process.stderr.write(String(error && error.stack || error));
  process.exitCode = 1;
});
