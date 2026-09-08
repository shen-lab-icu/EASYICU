'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
global.window = global;
global.document = { addEventListener() {} };
global.addEventListener = () => {};
let owner;
global.EasyICU = { guidedPi: { declare(_name, value) { owner = value; } } };
global.EU_HTML = { esc: String };
require(path.resolve(__dirname, '../../src/easyicu/webserver/static/js/screens-guided-pi-run-files.js'));

const run = (id, study = 'a') => ({ run_id: id, study_id: study, project_dir: `/runs/${study}/${id}` });
const review = row => ({ ok: true, ...row, readiness: { signable: true }, signed: false,
  artifacts: [{ name: 'result_tables.json' }, { name: 'evidence_ledger.json' }] });
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
const calls = [];
const api = {
  listStudyContexts: async () => ({ contexts: [{ id: 'a', title: 'A' }, { id: 'b', title: 'B' }] }),
  loadIdeaAgentProjects: async () => ({ projects: [{ study_id: 'a', project_dir: '/seed/a' }] }),
  loadGuidedDrafts: async () => ({ drafts: [] }),
  loadAgentRunHistory: async request => {
    calls.push(['history', request]);
    return { ok: true, runs: [run(request.project_seed_dir ? 'old' : 'new', request.study_id || 'a')], count: 1 };
  },
  loadAgentRunReview: async dir => review({ run_id: dir.split('/').pop(), study_id: dir.split('/')[2], project_dir: dir }),
  loadAgentRunArtifact: async (dir, name) => { calls.push(['artifact', dir, name]); return { ok: true, payload: { source: dir } }; },
  downloadAgentRunArtifact: async (dir, name) => { calls.push(['download', dir, name]); },
  downloadAgentRunBundle: async dir => { calls.push(['bundle', dir]); },
  signoffAgentRun: async (dir, body) => { calls.push(['sign', dir, body]); return { ...review(run('old')), signed: true, reportable: false, draft_unlocked: false }; },
};

(async () => {
  const messages = [
    { id: 'question', role: 'user', timelineAt: 1000, text: 'same words' },
    { id: 'receipt', role: 'activity', timelineAt: 2000, steps: [{ resources: [{ kind: 'research_artifact', run_id: 'old' }] }] },
    { id: 'answer', role: 'assistant', timelineAt: 3000, resources: [{ kind: 'research_artifact', run_id: 'old', artifact: 'result_tables.json' }] },
    { id: 'unrelated', role: 'assistant', timelineAt: 5000, resources: [{ kind: 'idea_plan', run_id: 'new' }] },
  ];
  const projected = owner.projectTimeline(messages, [run('old'), { ...run('new'), updated_at_epoch: 4 }]);
  assert.deepEqual(projected.find(row => row.id === 'answer').savedRuns.map(row => row.run_id), ['old']);
  assert.equal(projected.find(row => row.id === 'receipt').savedRuns.length, 0);
  assert.equal(projected.find(row => row.id === 'unrelated').savedRuns.length, 0);
  assert.deepEqual(projected.map(row => row.role), ['user', 'activity', 'assistant', 'saved_run', 'assistant']);
  assert.equal(projected[3].savedRuns[0].run_id, 'new');
  assert(!messages.some(row => row.savedRuns), 'projection must not mutate saved transcript');
  const conversationApi = { loadPiCopilotSessions: async (_limit, id) => ({ sessions: [
    { session_id: id + '-chat', title: 'Identical title', binding: { study_context_id: id === 'draft-old' ? 'old-study' : 'new-study' } },
  ] }) };
  const original = await owner.findConversation(conversationApi, [{ id: 'draft-new' }, { id: 'draft-old' }], 'old-study');
  assert.equal(original.draft.id, 'draft-old');
  assert.equal(original.session.session_id, 'draft-old-chat');
  assert.equal(await owner.findConversation(conversationApi, [{ id: 'draft-new' }], 'missing-study'), null);
  assert.equal(await owner.findConversation(conversationApi, [{ id: 'draft-old' }], 'old-study', () => false), null);
  const ctl = owner.createController(api);
  await ctl.catalog({ studyId: 'a', projectId: 'draft-a' });
  assert.equal(ctl.state.selected.id, 'a');
  assert.deepEqual(ctl.state.runs.map(row => row.run_id).sort(), ['new', 'old']);
  assert(calls.some(([, request]) => request && request.project_seed_dir === '/seed/a'));
  await ctl.openRun(run('old'));
  await ctl.artifact('result_tables.json');
  assert.equal(ctl.state.artifact.payload.source, '/runs/a/old');
  await ctl.download('result_tables.json'); await ctl.download();
  assert(calls.some(row => row[0] === 'download' && row[1] === '/runs/a/old'));
  assert(calls.some(row => row[0] === 'bundle' && row[1] === '/runs/a/old'));
  await ctl.artifact('../../secret'); await ctl.download('../../secret');
  assert(!calls.some(row => row.includes('../../secret')));
  await ctl.sign(['evidence_reviewed']); assert(!calls.some(row => row[0] === 'sign'));
  const checks = ['evidence_reviewed', 'claims_remain_locked', 'no_patient_rows_persisted'];
  await ctl.sign(checks);
  assert.equal(ctl.state.review.signed, true);
  assert.equal(ctl.state.review.reportable, false);
  assert.equal(ctl.state.review.draft_unlocked, false);
  await ctl.sign(checks); assert.equal(calls.filter(row => row[0] === 'sign').length, 1);

  // A slow request for A must never overwrite B's history or evidence.
  const pending = deferred();
  const race = owner.createController({ ...api, loadAgentRunReview: () => pending.promise });
  await race.select({ id: 'a' });
  const first = race.openRun(run('old'));
  await race.select({ id: 'b' });
  pending.resolve(review(run('old'))); await first;
  assert.equal(race.state.selected.id, 'b'); assert.equal(race.state.review, null);
  const artPending = deferred();
  const race2 = owner.createController({ ...api, loadAgentRunArtifact: () => artPending.promise });
  await race2.select({ id: 'a' }); await race2.openRun(run('old'));
  const loading = race2.artifact('result_tables.json'); await race2.select({ id: 'b' });
  artPending.resolve({ ok: true, payload: { wrong: true } }); await loading;
  assert.equal(race2.state.artifact, null);

  const failed = owner.createController({ ...api, loadAgentRunHistory: async () => { throw Error('offline'); } });
  await failed.select({ id: 'a' });
  assert.match(failed.state.error, /offline/); assert.equal(failed.state.loading, false);
  const bad = owner.createController({ ...api, loadAgentRunReview: async () => review(run('old', 'b')) });
  await bad.select({ id: 'a' }); await bad.openRun(run('old'));
  assert.match(bad.state.error, /identity/); assert.equal(bad.state.review, null);
  const imported = owner.createController(api);
  await imported.select({ id: 'a', readOnly: true }); await imported.openRun(run('old')); await imported.sign(checks);
  assert.equal(calls.filter(row => row[0] === 'sign').length, 1);
  const partial = owner.createController({ ...api, loadIdeaAgentProjects: async () => { throw Error('seed registry unavailable'); } });
  await partial.catalog({ studyId: 'b' });
  assert.match(partial.state.warning, /seed registry unavailable/); assert.equal(partial.state.selected.id, 'b');
  const signedFailure = owner.createController({ ...api, signoffAgentRun: async () => { throw Error('gate changed'); } });
  await signedFailure.select({ id: 'a' }); await signedFailure.openRun(run('old')); await signedFailure.sign(checks);
  assert.match(signedFailure.state.error, /gate changed/); assert.equal(signedFailure.state.review.signed, false);

  // Conversation changes invalidate run-file reads and late repaint callbacks.
  let context = { projectId: 'draft-a', studyId: 'a', sessionId: 'chat-a' };
  let repaints = 0;
  const threadPending = deferred();
  const thread = owner.create({ api: () => ({ ...api, loadAgentRunReview: () => threadPending.promise }),
    context: () => context, changed: () => { repaints++; }, resourceButton: () => '' });
  await thread.sync();
  const opening = thread.open('/runs/a/old');
  context = { projectId: 'draft-b', studyId: 'b', sessionId: 'chat-b' };
  await thread.sync();
  const settledPaints = repaints;
  threadPending.resolve(review(run('old'))); await opening;
  assert.equal(repaints, settledPaints);
  assert(thread.timeline([]).every(row => row.savedRuns.every(run => run.study_id === 'b')));
  thread.reset();
  console.log('Copilot history: exact-run reads/downloads/signoff, two-root recovery, races, partial failure, imported review and gates passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
