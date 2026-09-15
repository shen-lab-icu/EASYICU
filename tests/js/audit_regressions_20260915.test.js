'use strict';
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const read = name => fs.readFileSync(path.join(root, name), 'utf8');
function section(name, begin, end) {
  const source = read(name), a = source.indexOf(begin), b = source.indexOf(end, a + begin.length);
  assert(a >= 0 && b > a, `${name}: owner boundaries must exist`);
  return source.slice(a, b);
}
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
const flush = () => new Promise(resolve => setImmediate(resolve));
function context(values = {}) {
  const ctx = vm.createContext({ window: { t: en => en, icon: () => '' }, ...values });
  vm.runInContext(read('html-escape.js'), ctx);
  ctx.esc = ctx.window.EU_HTML.esc;
  return ctx;
}

test('P1-2 restored study text stays literal in the question, plan and JSON preview', () => {
  const hostile = '<img src=x onerror="globalThis.pwned=1"> & "value"';
  const c = context({ studyParams: {}, IDEA: { restoreSlot() {} }, REVIEW: { restoreSlot() {} },
    patientN: 10, mods: [], branch: 'predict', userQuestion: '', acceptedFrame: false, studyContract: null,
    unreadSlots: () => [], realMode: () => false, tg: en => en, icon: () => '',
    cardShell: (_a, _b, _c, _d, body, foot) => body + (foot || '') });
  vm.runInContext(section('screens-guided.js', '  function restoreGuidedSlotsFromSession(', '  function hasGuidedProjectMemory('), c);
  c.restoreGuidedSlotsFromSession({ slots: { study_params: { outcome: hostile, exposure: hostile, window: '24h' } } });
  vm.runInContext(section('screens-guided.js', '  function planFor(', '  /* Remember the user'), c);
  vm.runInContext(section('screens-guided.js', '  const CARD =', '  function summaryOf(') + '\nglobalThis.cards = CARD;', c);
  vm.runInContext(section('screens-guided.js', '  const ART =', '  function rocSvg('), c);
  for (const html of [c.cards.question(), c.artBody('analysis/cohort_summary.json')]) {
    assert(!html.includes('<img'), 'restored user text must not create an element');
    assert(html.includes('&lt;img'));
  }
  assert.equal(c.studyParams.outcome, hostile, 'escaping must not mutate scientific input');
  assert.match(c.artBody('analysis/cohort_summary.json'), /class="jk"/);
});

test('P2-5 runtime, unclassified and future routes all survive finding lanes', () => {
  const c = context(); vm.runInContext(read('screens-agent-render.js'), c);
  const findings = ['runtime_capability', 'unclassified', 'future_route'].map((route, i) => ({
    code: `UNKNOWN_${i}`, message: `finding-${i}`, remediation_route: route, requires_user_authorization: false,
  }));
  findings.push({ code: 'AUTHORIZED', message: 'one-decision', remediation_route: 'external_evidence', requires_user_authorization: true });
  const html = c.window.AGENT_RENDER.artifactStructuredView('scientific_plan_review.json', { approval_allowed: false, findings });
  for (let i = 0; i < 3; i++) assert(html.includes(`finding-${i}`));
  assert(html.includes('3 plan, evidence and runtime items'), 'authorized rows must not double count in evidence');
  assert(html.includes('Analysis paused'));
});

test('P2-1 cohort controls consume emitted dataset properties', () => {
  const source = 'screens-viz-cohort.js';
  const cases = [
    ['feature-scope', 'comp', 'cohortFeatureScope', 'featureScope', 'all'],
    ['sofa-matrix-mode', 'sofa-granularity', 'cohortSofaMatrixMode', 'sofaMatrixMode', 'count'],
    ['feature-module', 'feature-toggle', 'cohortFeatureModule', 'featureModule', 'renal'],
  ];
  for (const [control, next, attr, field, value] of cases) {
    let click, paints = 0;
    const c = context({ state: {}, root: { querySelectorAll: () => [{ dataset: { [attr]: value }, addEventListener: (_event, fn) => { click = fn; } }] }, repaintScreen: () => { paints++; } });
    vm.runInContext(section(source, `      root.querySelectorAll('[data-cohort-${control}]')`, `      root.querySelectorAll('[data-cohort-${next}]')`), c);
    click(); assert.equal(c.state[field], value); assert.equal(paints, 1);
    assert(read('screens-viz-cohort-view.js').includes(`data-cohort-${control}=`));
  }
});

test('P2-2 demo completion is invalid after reset, a new run or mode change', async () => {
  const c = context({ localStorage: { getItem: () => null, removeItem() {} }, location: { hash: '#guided' } });
  vm.runInContext(read('screens-extraction-job-continuity.js'), c);
  await c.window.EU_EXTRACTION_JOB_CONTINUITY.ready;
  c.continuity = c.window.EU_EXTRACTION_JOB_CONTINUITY;
  const code = section('screens-extraction.js', '      const ticket = continuity.capture();', '    } else {');
  for (const action of ['reset', 'new-run', 'real', 'normal']) {
    let callback; c.setTimeout = fn => { callback = fn; }; c.exView = 'running';
    c.dataMode = () => action === 'real' ? 'real' : 'demo'; c.backgroundRepaint = () => {};
    vm.runInContext(`(() => {${code}})()`, c);
    if (action === 'reset') c.continuity.abandon();
    if (action === 'new-run') c.continuity.prepare({ kind: 'extract', source: { path: '/synthetic' } });
    callback(); assert.equal(c.exView, action === 'normal' ? 'done' : 'running');
  }
});

test('P2-3/4 background progress paints embedded views and preserves other-route input', async () => {
  let embedded = 0, shell = 0;
  const c = context({ location: { hash: '#guided' }, localStorage: { setItem() {} } });
  c.window.__euRender = () => { shell++; };
  c.window.EU_EXTRACTION_EMBEDDED_WORKSPACE = { isMounted: () => true, repaint: () => { embedded++; } };
  vm.runInContext(section('screens-extraction.js', '  function repaint()', '  function exportDestinationLabel()'), c);
  c.backgroundRepaint(); assert.equal(embedded, 1); assert.equal(shell, 0);
  c.window.EU_EXTRACTION_EMBEDDED_WORKSPACE.isMounted = () => false;
  const pending = deferred(); c.window.EU_API = { registerWorkspaceSource: () => pending.promise };
  const result = c.rememberExportPath('/synthetic'); pending.resolve({ sources: [] }); await result;
  assert.equal(shell, 0); assert.equal(embedded, 1);
  c.location.hash = '#extraction'; c.backgroundRepaint(); assert.equal(shell, 1);
});

test('P2-8 real cohort never shows or consumes seeded strict/loosen controls', () => {
  let paints = 0;
  const c = context({ realMode: () => true, branch: 'predict', BRANCH: { predict: {} }, cohortPhase: 'empty',
    snapshotSummary: () => ({ stays: 57 }), fmtInt: n => n, fmtPct: () => '—', fmtNum: () => '—',
    activeExportLabel: () => 'local', cardShell: (...args) => args.join(''), icon: () => '' });
  vm.runInContext(section('screens-guided.js', '  const CARD =', '  function summaryOf(') + '\nglobalThis.cards = CARD;', c);
  const html = c.cards.cohort(); assert(html.includes('57')); assert(!html.includes('data-act="strict"')); assert(!html.includes('matched 0'));
  for (const action of ['strict', 'loosen']) {
    c.actEl = { dataset: { act: action } }; c.renderThread = () => { paints++; };
    vm.runInContext(`(() => {${section('screens-guided.js', '          const a = actEl.dataset.act;', "          if (a === 'open')")}})()`, c);
  }
  assert.equal(paints, 0);
});

test('P2-7 folder chip opens the existing folder dialog', () => {
  const calls = [], c = context({ tok: '@folderopen', label: 'Open folder', pushUser() {}, showGuidedDraftSetup: (...args) => calls.push(args) });
  vm.runInContext(`(() => {${section('screens-guided.js', "          if (tok === '@folderopen')", "          if (tok === '@hintN')")}})()`, c);
  assert.deepEqual(calls, [['Open folder', 'open']]);
});

test('P2-10 only the latest project click can restore memory or display its error', async () => {
  for (const staleError of [false, true]) {
    const requests = [deferred(), deferred()], bound = [];
    const c = context({ gen: 0, guidedCopilot: {}, document: { querySelectorAll: () => [] }, piProjectShellActive: () => true,
      projectTitle: x => x, t: en => en, guidedBackendContext: () => ({}), bindProjectToPi: (_result, row) => bound.push(row.id),
      pushBot: () => { throw Error('stale failure displayed'); }, renderThread() {} });
    c.window.EU_API = { openGuidedProject: () => requests.shift().promise };
    const [a, b] = requests.slice();
    vm.runInContext(section('screens-guided.js', '  function openGuidedProjectMemory(', '  function guidedGoalMeta('), c);
    const first = c.openGuidedProjectMemory({ id: 'A', project_dir: '/A' }, null, 'draft');
    const second = c.openGuidedProjectMemory({ id: 'B', project_dir: '/B' }, null, 'draft');
    b.resolve({ ok: true }); await second;
    if (staleError) a.reject(Error('old failure')); else a.resolve({ ok: true });
    await first; assert.deepEqual(bound, ['B']);
  }
});

test('P2-11 send and regenerate discard stale success and errors after session selection', async () => {
  for (const kind of ['send', 'regenerate']) for (const outcome of ['success', 'error']) {
    const pending = deferred(), watched = [];
    const c = context({ state: { session: { session_id: 'A' }, sessionSelectionRevision: 1, messages: [] }, projectId: () => 'project',
      sessionIsStale: () => false, sessionMatchesUiLanguage: () => true, IDEA_SOURCE: null, REGENERATION: null,
      PLAN_ACTIONS: { regenerationAuthority: () => ({ grants: [] }) }, turnGrants: () => [],
      ensureActivity: () => ({}), upsertActivityStep() {}, render() {}, finishActivity: () => { throw Error('old error changed activity'); },
      errorText: String, watchJob: id => watched.push(id),
      api: () => ({ sendPiCopilotMessage: () => pending.promise, regeneratePiCopilotMessage: () => pending.promise }) });
    vm.runInContext(section('screens-guided-pi.js', '  async function sendText(', '  async function continueAfterDataSourceConfirmation('), c);
    const result = kind === 'send' ? c.sendText('question') : c.regenerateMessage('entry', 'question');
    await flush(); c.state.session = { session_id: 'B' }; c.state.sessionSelectionRevision++; c.state.busy = false; c.state.jobId = 'B-job';
    if (outcome === 'success') pending.resolve({ job_id: 'A-job' }); else pending.reject(Error('A-error'));
    await result; assert.equal(c.state.jobId, 'B-job'); assert.deepEqual(watched, []);
  }
});

test('P2-11 formal plan checks identity at every awaited boundary', async () => {
  for (const boundary of ['load', 'start', 'record']) {
    let revision = 1; const pending = deferred(), calls = []; let current = { session_id: 'A', binding: { study_context_id: 'study' }, research_provider: {} };
    let owner; const c = context(); c.window.EasyICU = { guidedPi: { declare: (_name, value) => { owner = value; } } };
    vm.runInContext(read('screens-guided-pi-plan-actions.js'), c);
    const actions = owner.create({ tr: en => en, session: () => current, selectionRevision: () => revision, projectId: () => 'project', busy: () => false, sessionIsStale: () => false,
      workflow: () => ({ next_action_code: 'provider_ready_to_generate_plan' }), nextActions: {},
      appendMessage() {}, setBusy() {}, render() {}, setError: value => { if (value) throw Error(value); }, errorText: String,
      api: () => ({ loadStudyContext: () => boundary === 'load' ? pending.promise : Promise.resolve({ data_source: { path: '/synthetic' } }),
        startAgentRun: () => { calls.push('start'); return boundary === 'start' ? pending.promise : Promise.resolve({ job_id: 'job-A' }); } }),
      recordHostAction: () => { calls.push('record'); return boundary === 'record' ? pending.promise : Promise.resolve(); },
      watchChildJob: () => calls.push('watch') });
    const result = actions.startFormalPlanGeneration('provider_ready_to_generate_plan'); await flush();
    revision++; current = { session_id: 'A' }; // A -> B -> A must still invalidate the old action.
    pending.resolve(boundary === 'load' ? { data_source: { path: '/synthetic' } } : { job_id: 'job-A' });
    assert.equal(await result, false); assert(!calls.includes('watch'));
    if (boundary === 'load') assert(!calls.includes('start'));
    if (boundary === 'start') assert(!calls.includes('record'));
  }
});

test('P2-11 stale pre-start plan request does not poison the transition guard', async () => {
  const firstLoad = deferred(), calls = [];
  let current = { session_id: 'A', binding: { study_context_id: 'study', run_id: 'source-run' }, research_provider: {} };
  let owner;
  const c = context();
  c.window.EasyICU = { guidedPi: { declare: (_name, value) => { owner = value; } } };
  vm.runInContext(read('screens-guided-pi-plan-actions.js'), c);
  let loadCount = 0;
  const actions = owner.create({
    tr: en => en, session: () => current, projectId: () => 'project', busy: () => false,
    sessionIsStale: () => false,
    workflow: () => ({ next_action_code: 'agent_plan_revision_nonconvergent', plan_review_summary: {
      run_id: 'source-run', authorization_questions: [], automatic_revision_blockers: [],
      remediation_buckets: { agent_plan_revision: [{ code: 'repair' }] },
    } }), nextActions: {}, appendMessage() {}, setBusy() {}, render() {}, setError() {},
    errorText: String,
    api: () => ({
      loadStudyContext: () => (++loadCount === 1 ? firstLoad.promise : Promise.resolve({ data_source: { path: '/synthetic' } })),
      startAgentRun: () => { calls.push('start'); return Promise.resolve({ job_id: 'job-A' }); },
    }),
    recordHostAction: () => Promise.resolve(), watchChildJob: () => calls.push('watch'),
  });
  const first = actions.startFormalPlanGeneration('agent_plan_revision_nonconvergent');
  await flush();
  current = { session_id: 'B' };
  firstLoad.resolve({ data_source: { path: '/synthetic' } });
  assert.equal(await first, false);
  current = { session_id: 'A', binding: { study_context_id: 'study', run_id: 'source-run' }, research_provider: {} };
  assert.equal(await actions.startFormalPlanGeneration('agent_plan_revision_nonconvergent'), true);
  assert.deepEqual(calls, ['start', 'watch']);
});

test('P2-12 document and web opens invalidate older artifact responses', async () => {
  for (const kind of ['webpage', 'research_document']) {
    const pending = deferred(), c = context({ state: { request: 0, projectId: '', recentResources: [] },
      safeResource: value => value, safeWorkflowContext: value => value, rememberResource() {}, render() {} });
    c.loadResource = async () => { const ticket = ++c.state.request; const payload = await pending.promise; if (ticket === c.state.request) c.state.payload = payload; };
    vm.runInContext(section('screens-guided-pi-preview.js', '  function open(resource,', '  function openRunEvidence('), c);
    c.open({ kind: 'research_artifact' }, 'project');
    c.open({ kind }, 'project'); pending.resolve({ stale: true }); await flush();
    assert.equal(c.state.payload, null); assert.equal(c.state.resource.kind, kind); assert.equal(c.state.loading, false);
  }
});

test('P1-1 conversion filenames are escaped at the live progress renderer', () => {
  const c = context({ convResult: null, convErr: null, convProg: { current: 1, total: 2, file: '<img src=x onerror="bad()">.csv' },
    exPath: '/synthetic', icon: () => '', t: en => en, pathDisplay: x => x, preparedDestinationHint: () => 'prepared' });
  c.escHtml = c.esc;
  vm.runInContext(section('screens-extraction.js', '  function convertingState()', '  function startConvert()'), c);
  const html = c.convertingState();
  assert(html.includes('&lt;img')); assert(!html.includes('<img'));
});

test('P2-6 run-file evidence opens the registered run and digest; display links stay local', async () => {
  const calls = [], registered = {}, c = context({ document: { getElementById: () => null } });
  c.window.EasyICU = { guidedPi: { declare: (name, owner) => { registered[name] = owner; },
    require: name => registered[name], optional: name => registered[name] || null } };
  c.window.EU_PRODUCT_LABELS = { projectTitle: (value, fallback) => value || fallback };
  registered.evidencePreview = { kindLabel: () => 'Result' };
  const artifactSha = 'b'.repeat(64), evidenceSha = 'a'.repeat(64);
  const run = { run_id: 'original-run', study_id: 'study', project_dir: '/synthetic/original' };
  const api = { listStudyContexts: async () => ({ contexts: [{ id: 'study' }] }), loadIdeaAgentProjects: async () => ({ projects: [] }),
    loadGuidedDrafts: async () => ({ drafts: [] }), loadAgentRunHistory: async () => ({ runs: [run] }),
    loadAgentRunReview: async () => ({ ...run, artifacts: [{ name: 'manuscript_provenance.json', sha256: artifactSha }] }),
    loadAgentRunArtifact: async () => ({ payload: {} }),
    loadPiCopilotResearchArtifact: async (...args) => { calls.push(['artifact', ...args]); return { payload: {} }; },
    loadPiCopilotResearchEvidence: async (...args) => { calls.push(['evidence', ...args]); return { payload: {} }; } };
  c.window.EU_API = api;
  vm.runInContext(read('screens-guided-pi-preview.js'), c);
  vm.runInContext(read('screens-guided-pi-run-files.js'), c);
  const owner = registered.runFiles.create({ api: () => api, context: () => ({ projectId: 'project', studyId: 'study', sessionId: 'session' }), changed() {} });
  await owner.sync();
  let scrolled = 0;
  const sectionNode = { dataset: { runFiles: run.project_dir }, querySelectorAll: () => [{ dataset: { gpiDisplayAnchor: 'Table 1' }, scrollIntoView: () => { scrolled++; } }] };
  const click = (selector, element) => owner.handleClick({ preventDefault() {}, target: { closest: query => query === '[data-run-files]' ? sectionNode : query === selector ? element : null } });
  click('summary', { parentElement: sectionNode }); await flush();
  click('[data-run-files-artifact]', { dataset: { runFilesArtifact: '0' } }); await flush();
  const button = { dataset: { evidenceId: 'summary', evidenceSha256: evidenceSha }, closest: () => null };
  click('[data-gpi-evidence-open]', button); await flush();
  assert.deepEqual(calls.find(row => row[0] === 'artifact'), ['artifact', 'project', 'original-run', 'manuscript_provenance.json', artifactSha]);
  assert.deepEqual(calls.find(row => row[0] === 'evidence'), ['evidence', 'project', 'original-run', 'summary', evidenceSha]);
  button.dataset.evidenceSha256 = 'invalid'; click('[data-gpi-evidence-open]', button); await flush();
  assert.equal(calls.filter(row => row[0] === 'evidence').length, 1);
  click('[data-gpi-display]', { dataset: { gpiDisplay: 'Table 1' } }); assert.equal(scrolled, 1);
});

test('workspace snapshot request identity survives an A-B-A path cycle', async () => {
  const pending = [deferred(), deferred(), deferred()];
  const requested = [];
  const c = context({ workspaceSnapshot: null, workspaceSnapshotPath: '', renderThread() {}, renderAside() {} });
  c.window.EU_API = {
    loadWorkspaceSummary: path => {
      requested.push(path);
      return pending[requested.length - 1].promise;
    },
  };
  vm.runInContext(section('screens-guided.js', '  function loadWorkspaceSnapshot(', '  /* ---- dynamic plan/frame'), c);

  const firstA = c.loadWorkspaceSnapshot({ path: '/A' });
  const requestB = c.loadWorkspaceSnapshot({ path: '/B' });
  const secondA = c.loadWorkspaceSnapshot({ path: '/A' });
  assert.deepEqual(requested, ['/A', '/B', '/A']);

  pending[2].resolve({ marker: 'fresh-A2' });
  assert.deepEqual(await secondA, { marker: 'fresh-A2' });
  pending[0].resolve({ marker: 'stale-A1' });
  assert.equal(await firstA, null);
  pending[1].resolve({ marker: 'stale-B' });
  assert.equal(await requestB, null);
  assert.deepEqual(c.workspaceSnapshot, { marker: 'fresh-A2' });
});

test('a failed workspace switch cannot cache the previous path under the new path', async () => {
  const pending = deferred();
  const c = context({
    workspaceSnapshot: { marker: 'cached-A' },
    workspaceSnapshotPath: '/A',
    renderThread() {},
    renderAside() {},
  });
  let calls = 0;
  c.window.EU_API = { loadWorkspaceSummary: () => { calls += 1; return pending.promise; } };
  vm.runInContext(section('screens-guided.js', '  function loadWorkspaceSnapshot(', '  /* ---- dynamic plan/frame'), c);

  const firstB = c.loadWorkspaceSnapshot({ path: '/B' });
  pending.reject(Error('unavailable'));
  assert.equal(await firstB, null);
  assert.equal(c.workspaceSnapshot, null);
  c.window.EU_API.loadWorkspaceSummary = () => { calls += 1; return Promise.resolve({ marker: 'fresh-B' }); };
  assert.deepEqual(await c.loadWorkspaceSnapshot({ path: '/B' }), { marker: 'fresh-B' });
  assert.equal(calls, 2);
});

test('guided session requests are coalesced only for the same project identity', async () => {
  const pending = [deferred(), deferred()];
  const requested = [];
  const c = context({
    gen: 1,
    guidedSessionRequest: null,
    guidedCopilot: { loading: false, error: null, session: null, last: null },
    selectedGuidedDraft: { id: 'draft-A', project_dir: '/A', title: 'A' },
    guidedBackendContext: () => ({}),
    renderThread() {},
  });
  c.window.EU_API = {
    createGuidedSession: () => Promise.resolve({ session: null }),
    openGuidedProject: payload => {
      requested.push(payload.project_dir);
      return pending[requested.length - 1].promise;
    },
  };
  vm.runInContext(section('screens-guided.js', '  function trackGuidedSessionRequest(', '  function threadFromSessionMessage('), c);

  const first = c.ensureGuidedSession(false);
  c.selectedGuidedDraft = { id: 'draft-B', project_dir: '/B', title: 'B' };
  c.gen += 1;
  const second = c.ensureGuidedSession(false);
  assert.deepEqual(requested, ['/A', '/B']);

  pending[1].resolve({ session: { id: 'session-B', project_dir: '/B', memory_scope: 'project_folder' } });
  assert.equal((await second).id, 'session-B');
  pending[0].resolve({ session: { id: 'session-A', project_dir: '/A', memory_scope: 'project_folder' } });
  assert.equal(await first, null);
  assert.equal(c.guidedCopilot.session.id, 'session-B');
});
