/* Executable contract for source-boundary isolation and history activation. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

const listeners = {};
const storage = new Map();
global.window = global;
global.location = { hash: '#extraction' };
global.localStorage = {
  getItem: key => storage.has(key) ? storage.get(key) : null,
  setItem: (key, value) => storage.set(key, String(value)),
  removeItem: key => storage.delete(key),
};
global.CustomEvent = class CustomEvent {
  constructor(type, init) { this.type = type; this.detail = init && init.detail; }
};
global.addEventListener = (type, callback) => {
  listeners[type] = listeners[type] || [];
  listeners[type].push(callback);
};
global.dispatchEvent = event => {
  (listeners[event.type] || []).forEach(callback => callback(event));
};
global.document = {
  readyState: 'complete',
  addEventListener: global.addEventListener,
};

const persisted = new Map();
const saveBodies = [];
let serverContexts = [];
let serverActiveId = null;
function rememberServer(context) {
  persisted.set(context.id, { ...context });
  serverContexts = [{ ...context }].concat(serverContexts.filter(row => row.id !== context.id));
  serverActiveId = context.id;
}
global.EU_API = {
  loadActiveStudyContext: async () => ({
    ok: true,
    context: serverContexts.find(row => row.id === serverActiveId) || null,
  }),
  listStudyContexts: async () => ({ ok: true, contexts: serverContexts.map(row => ({ ...row })), active_id: serverActiveId }),
  saveStudyContext: async context => {
    saveBodies.push({ ...context });
    const current = persisted.get(context.id) || null;
    if (current && Object.keys(context).length === 1) {
      rememberServer(current);
      return { ok: true, context: { ...current } };
    }
    assert.equal(context.expected_revision, current ? current.revision : 0);
    const { expected_revision: _expectedRevision, ...metadata } = context;
    const saved = {
      ...(current || {}),
      ...metadata,
      revision: (current ? current.revision : 0) + 1,
    };
    if (current) {
      saved.current_stage = current.current_stage;
      saved.last_route = current.last_route;
      saved.active_job_id = current.active_job_id;
    }
    rememberServer(saved);
    return { ok: true, context: { ...saved } };
  },
  handoffStudyContext: async body => {
    const current = persisted.get(body.study_context_id);
    assert.equal(body.expected_revision, current.revision);
    const handed = {
      ...current,
      current_stage: body.current_stage,
      last_route: body.target_route,
      revision: current.revision + 1,
    };
    if (body.active_job_id !== undefined) handed.active_job_id = body.active_job_id;
    rememberServer(handed);
    return { ok: true, context: handed };
  },
};

require(path.resolve(process.argv[2]));

(async () => {
  const store = global.EU_STUDY_CONTEXT;
  const old = store.startNew({
    title: 'Old mortality study',
    question: 'Does old exposure predict mortality?',
    outcome: 'mortality',
    data_source: { path: '/exports/old', label: 'Old export', database: 'miiv' },
    cohort: { preset: 'adult_first' },
    modules: ['outcome'],
  }, { persist: false });

  let nextPath = '/exports/new';
  store.registerSource('extraction', current => ({
    title: current ? current.title : 'New extraction study',
    question: current ? current.question : 'Analyze the newly extracted cohort.',
    outcome: current ? current.outcome : '',
    data_source: { path: nextPath, label: 'New export', database: 'eicu' },
    cohort: { preset: 'sepsis3' },
    modules: ['vitals'],
    current_stage: 'data_prepared',
  }));

  const boundary = store.handoff({ sourceRoute: 'extraction', targetRoute: 'agent' });
  assert.notEqual(boundary.context.id, old.id, 'source change must create a new project id');
  assert.equal(boundary.context.question, 'Analyze the newly extracted cohort.');
  assert.equal(boundary.context.outcome, '', 'old outcome must not cross the source boundary');
  await boundary.persisted;

  const ids = store.all().map(context => context.id);
  assert(ids.includes(old.id));
  assert(ids.includes(boundary.context.id));
  await store.activate(old.id);
  assert.equal(store.active().id, old.id, 'history activation must switch the active project');
  assert.equal(store.active().question, 'Does old exposure predict mortality?');

  nextPath = '/exports/old';
  store.registerSource('patient', current => ({
    question: current ? current.question : 'Fresh patient review question',
    data_source: { path: nextPath, label: 'Old export', database: 'miiv' },
    cohort: { review: 'patient' },
    confirmations: { crossdb_plan_only: false },
    current_stage: 'patient_reviewed',
  }));
  const continued = store.handoff({ sourceRoute: 'patient', targetRoute: 'agent' });
  assert.equal(continued.context.id, old.id, 'same-source explicit handoff should continue the selected project');
  await continued.persisted;
  assert.equal(store.active().current_stage, 'patient_reviewed', 'handoff must send the prepared stage, not the metadata response stage');

  global.EU_VIZ_CONTEXT = {
    snapshot: route => ({
      data_source: { path: '/exports/shared', label: 'Shared export', database: 'miiv' },
      cohort: route === 'crossdb'
        ? { source_count: 2, source_type: 'prepared' }
        : route === 'patient'
          ? {
            entity_count: 94458,
            full_entity_count: 94458,
            review_entities: 500,
            review_entity_cap: 500,
            review_scope: 'browser_bounded_entity_sample',
            module_count: 19,
          }
          : { cohort_size: 10, comparison: 'outcome' },
      modules: ['vitals'],
      comparator: route === 'crossdb' ? 'cross_database_descriptive' : '',
      crossdb_selection: route === 'crossdb' ? {
        schema_version: 'crossdb-selection-v1',
        source_count: 2,
        sources: [
          { source_id: 'src_miiv', label: 'Primary MIIV', database: 'miiv', path_hash: 'aaaaaaaaaaaa' },
          { source_id: 'src_eicu', label: 'Comparator eICU', database: 'eicu', path_hash: 'bbbbbbbbbbbb' },
        ],
        selection_digest: 'c'.repeat(64),
      } : {},
    }),
  };
  require(path.resolve(process.argv[3]));
  const crossdb = store.handoff({ sourceRoute: 'crossdb', targetRoute: 'agent' });
  assert.equal(crossdb.context.current_stage, 'crossdb_plan_only');
  assert.equal(crossdb.context.confirmations.crossdb_plan_only, true);
  assert.equal(crossdb.context.data_source, null, 'Cross-DB must not collapse to one active export');
  assert.equal(crossdb.context.crossdb_selection.source_count, 2);
  assert.deepEqual(
    crossdb.context.crossdb_selection.sources.map(source => source.source_id),
    ['src_miiv', 'src_eicu'],
  );
  await crossdb.persisted;
  const patient = store.handoff({ sourceRoute: 'patient', targetRoute: 'agent' });
  assert.notEqual(patient.context.id, crossdb.context.id, 'leaving a Cross-DB plan must start a single-export project');
  assert.equal(patient.context.current_stage, 'patient_reviewed');
  assert.equal(patient.context.question, 'Analyze the reviewed ICU cohort using the active EasyICU export.');
  assert.notEqual(patient.context.analysis_goal, crossdb.context.analysis_goal);
  assert.notEqual(patient.context.comparator, 'cross_database_descriptive');
  assert.equal(patient.context.confirmations.crossdb_plan_only, false, 'Patient review must clear plan-only');
  assert.equal(patient.context.cohort.entity_count, 94458, 'full cohort denominator must remain explicit');
  assert.equal(patient.context.cohort.full_entity_count, 94458);
  assert.equal(patient.context.cohort.review_entities, 500, 'bounded browser review denominator must survive handoff');
  assert.equal(patient.context.cohort.review_entity_cap, 500);
  assert.equal(patient.context.cohort.review_scope, 'browser_bounded_entity_sample');
  assert.equal(patient.context.confirmations.patient_review_bounded_sample, true);
  assert.equal(patient.context.confirmations.patient_review_full_entity_set, false);
  assert.deepEqual(patient.context.crossdb_selection, {}, 'single-export review must clear Cross-DB selection');
  for (const key of ['source_count', 'source_type', 'comparison_mode']) {
    assert.equal(Object.hasOwn(patient.context.cohort, key), false, `Patient cohort must remove ${key}`);
  }
  await patient.persisted;

  const cohort = store.handoff({ sourceRoute: 'cohort', targetRoute: 'agent' });
  assert.equal(cohort.context.id, patient.context.id, 'same-export review routes should continue one project');
  assert.equal(cohort.context.current_stage, 'cohort_reviewed');
  assert.equal(cohort.context.question, patient.context.question, 'route review must preserve the project question');
  assert.equal(cohort.context.cohort.cohort_size, 10);
  assert.equal(cohort.context.cohort.comparison, 'outcome');
  assert.equal(cohort.context.confirmations.cohort_review_completed, true);
  for (const key of ['entity_count', 'full_entity_count', 'review_entities', 'review_entity_cap', 'review_scope', 'module_count']) {
    assert.equal(Object.hasOwn(cohort.context.cohort, key), false, `Cohort review must remove Patient field ${key}`);
  }
  for (const key of ['patient_review_completed', 'patient_review_bounded_sample', 'patient_review_full_entity_set']) {
    assert.equal(Object.hasOwn(cohort.context.confirmations, key), false, `Cohort review must remove Patient confirmation ${key}`);
  }
  await cohort.persisted;

  const patientAgain = store.handoff({ sourceRoute: 'patient', targetRoute: 'agent' });
  assert.equal(patientAgain.context.id, cohort.context.id, 'returning to Patient should keep the same export project');
  assert.equal(patientAgain.context.question, patient.context.question);
  assert.equal(patientAgain.context.cohort.review_entities, 500);
  assert.equal(patientAgain.context.confirmations.patient_review_bounded_sample, true);
  assert.equal(patientAgain.context.comparator, '', 'Patient review must not inherit Cohort UI grouping');
  for (const key of ['cohort_size', 'comparison']) {
    assert.equal(Object.hasOwn(patientAgain.context.cohort, key), false, `Patient review must remove Cohort field ${key}`);
  }
  assert.equal(Object.hasOwn(patientAgain.context.confirmations, 'cohort_review_completed'), false);
  await patientAgain.persisted;

  const crossdbAgain = store.handoff({ sourceRoute: 'crossdb', targetRoute: 'agent' });
  await crossdbAgain.persisted;
  global.EU_GUIDED_CONTEXT = {
    snapshot: () => ({
      question: 'Evaluate mortality in the active ICU cohort.',
      source: { path: '/exports/shared', label: 'Shared export', database: 'miiv' },
      cohort_preset: 'adult_first',
      max_patients: 500,
      modules: ['vitals'],
      outcome: 'mortality',
      window_preset: 'whole_stay',
      window_label: 'Whole stay',
      comparator: '',
      export_format: 'parquet',
      configured: true,
    }),
  };
  require(path.resolve(process.argv[4]));
  const guided = global.EU_GUIDED_STUDY_CONTEXT.handoff('agent');
  assert.notEqual(guided.context.id, crossdbAgain.context.id);
  assert.equal(guided.context.question, 'Evaluate mortality in the active ICU cohort.');
  assert.equal(guided.context.confirmations.crossdb_plan_only, false);
  for (const key of ['review', 'source_count', 'source_type', 'comparison_mode']) {
    assert.equal(Object.hasOwn(guided.context.cohort, key), false, `Guided cohort must remove ${key}`);
  }
  await guided.persisted;

  const home = store.startNew({
    question: 'Does the home question remain one project after Guided configuration?',
    data_source: { path: '/exports/shared', label: 'Shared export', database: 'miiv' },
    current_stage: 'study_setup',
    last_route: 'guided',
  }, { persist: false });
  await store.persist();
  global.EU_GUIDED_CONTEXT.snapshot = () => ({
    question: 'Evaluate mortality after Guided collected the study design.',
    source: { path: '/exports/shared', label: 'Shared export', database: 'miiv' },
    cohort_preset: 'adult_first', modules: ['vitals'], outcome: 'mortality',
    window_preset: 'whole_stay', window_label: 'Whole stay', comparator: '',
    export_format: 'parquet', configured: true,
  });
  const guidedContinuation = await global.EU_GUIDED_STUDY_CONTEXT.persistForRun('agent_preflight');
  assert.equal(guidedContinuation.id, home.id, 'Guided configuration must continue the home project');
  assert.equal(guidedContinuation.question, 'Evaluate mortality after Guided collected the study design.');

  store.registerSource('demo-check', current => ({
    title: current ? current.title : 'Demo study',
    question: current ? current.question : 'Fresh demo study question.',
    data_source: { path: '', label: 'Demo data', database: 'demo' },
    cohort: { preset: 'adult_first' }, modules: ['vitals'], outcome: current ? current.outcome : '',
    current_stage: 'data_prepared',
  }));
  const demo = store.handoff({ sourceRoute: 'demo-check', targetRoute: 'agent' });
  assert.notEqual(demo.context.id, home.id, 'real export to Demo identity must create a new project');
  assert.equal(demo.context.question, 'Fresh demo study question.');
  assert.equal(demo.context.data_source.database, 'demo');
  await demo.persisted;

  const longQuestion = 'Q'.repeat(300);
  const longContext = store.startNew({ question: longQuestion }, { persist: false });
  assert.equal(longContext.title.length, 160, 'derived title must respect the backend title limit');
  assert.equal(longContext.question, longQuestion, 'a valid >160 character question must remain intact');
  await store.persist();
  assert.equal(persisted.get(longContext.id).question, longQuestion);

  assert.throws(
    () => store.startNew({ question: 'unsafe', patient_rows: [{ patient_id: 'secret' }] }, { persist: false }),
    /Row-level StudyContext metadata is forbidden/,
  );
  assert.throws(
    () => store.startNew({ question: 'unsafe nested', cohort: { records: [{ stay_id: 9 }] } }, { persist: false }),
    /Row-level StudyContext metadata is forbidden/,
  );
  const sanitized = store.startNew({
    question: 'Canonical metadata only', unknown_payload: { secret: 'drop-me' },
    cohort: { preset: 'adult_first', unsupported: 'drop-me' }, modules: [{ bad: true }, 'vitals'],
  }, { persist: false });
  assert.equal(Object.hasOwn(sanitized, 'unknown_payload'), false);
  assert.equal(Object.hasOwn(sanitized.cohort, 'unsupported'), false);
  assert.deepEqual(sanitized.modules, ['vitals']);
  assert.equal(localStorage.getItem('easyicu.studyContext.active.v1').includes('drop-me'), false);

  const dirty = store.startNew({
    question: 'Unsynced browser question wins hydration.',
    data_source: { path: '/exports/dirty', label: 'Dirty export', database: 'miiv' },
  }, { persist: false });
  const staleServer = { ...dirty, question: 'Older server question.', current_stage: 'plan' };
  rememberServer(staleServer);
  await store.hydrate({ force: true });
  assert.equal(store.active().question, 'Unsynced browser question wins hydration.');
  assert.equal(store.all().find(row => row.id === dirty.id).question, 'Unsynced browser question wins hydration.');
  rememberServer({
    id: 'server-clean-b', title: 'Clean server B', question: 'Server-selected clean project.',
    data_source: { path: '/exports/server-b', label: 'Server B', database: 'eicu' },
    current_stage: 'study_setup', last_route: 'entry',
  });
  await store.hydrate({ force: true });
  assert.equal(store.active().id, dirty.id, 'hydration must not replace dirty active A with clean server-active B');
  assert.equal(store.active().question, 'Unsynced browser question wins hydration.');
  await store.persist();

  const other = store.startNew({ question: 'Other clean project.' }, { persist: false });
  await store.persist();
  const advanced = { ...persisted.get(dirty.id), current_stage: 'review', active_job_id: null };
  rememberServer(advanced);
  serverActiveId = other.id;
  const saveCount = saveBodies.length;
  const activated = await store.activate(dirty.id);
  assert.deepEqual(saveBodies[saveCount], { id: dirty.id }, 'clean activation must send an id-only merge');
  assert.equal(activated.current_stage, 'review', 'activation must not roll back a server-advanced stage');

  require(path.resolve(process.argv[6]));
  require(path.resolve(process.argv[5]));
  global.EU_WORKSPACE_REGISTRY = {
    active_path: '/exports/active-single',
    sources: [{ path: '/exports/active-single', label: 'Unrelated active export', summary: { stays: 140 } }],
  };
  const crossdbProject = {
    planOnly: true,
    studyContext: crossdb.context,
  };
  assert.equal(
    global.EU_AGENT_STUDY_CONTEXT.sourceFor(crossdbProject, global.EU_WORKSPACE_REGISTRY.sources[0]),
    null,
    'Cross-DB selection receipt must not fall back to one active registry export',
  );
  assert.equal(
    global.EU_AGENT_STUDY_CONTEXT.sourceFor({
      planOnly: true,
      studyContext: {
        data_source: global.EU_WORKSPACE_REGISTRY.sources[0],
        crossdb_selection: {},
      },
    }, global.EU_WORKSPACE_REGISTRY.sources[0]),
    null,
    'A legacy or damaged plan-only receipt must fail closed instead of using the active registry export',
  );
  assert.equal(
    global.EU_AGENT_STUDY_CONTEXT.sourceFor({
      planOnly: false,
      studyContext: {
        data_source: global.EU_WORKSPACE_REGISTRY.sources[0],
        crossdb_selection: crossdb.context.crossdb_selection,
      },
    }, global.EU_WORKSPACE_REGISTRY.sources[0]),
    null,
    'A multi-source receipt must stay authoritative even if the lifecycle stage drifts',
  );
  const guidedBinding = await global.EU_AGENT_STUDY_CONTEXT.prepareGuidedHandoff(crossdbProject);
  assert.equal(guidedBinding.schema_version, 'easyicu.guided-project-handoff/1');
  assert.equal(guidedBinding.project_id, crossdb.context.id);
  assert.equal(guidedBinding.binding_receipt.schema_version, 'easyicu.pi-project-binding-handoff/1');
  assert.equal(guidedBinding.binding_receipt.study_context_id, crossdb.context.id);
  assert.equal(guidedBinding.binding_receipt.study_context_revision, persisted.get(crossdb.context.id).revision);
  assert.equal(JSON.stringify(guidedBinding).includes('/exports/'), false, 'Pi handoff must remain path-free');
  assert.deepEqual(global.EU_AGENT_STUDY_CONTEXT.takeGuidedHandoff(), guidedBinding);
  assert.equal(global.EU_AGENT_STUDY_CONTEXT.takeGuidedHandoff(), null, 'handoff is consumed once');
  const unboundProject = {
    id: 'idea-project-unbound',
    name: ['Idea project', 'Idea project'],
  };
  const unboundGuided = await global.EU_AGENT_STUDY_CONTEXT.prepareGuidedHandoff(unboundProject);
  assert.equal(unboundGuided.project_id, unboundProject.id);
  assert.equal(unboundGuided.binding_receipt, null, 'unbound projects initialize inside Copilot');
  assert.deepEqual(global.EU_AGENT_STUDY_CONTEXT.takeGuidedHandoff(), unboundGuided);
  const jobContext = store.startNew({
    question: 'Job lifecycle context',
    data_source: { path: '/exports/jobs', label: 'Jobs export', database: 'miiv' },
  }, { persist: false });
  await store.persist();
  global.EU_AGENT_STUDY_CONTEXT.markContextRunning(jobContext.id, 'job-old');
  global.EU_AGENT_STUDY_CONTEXT.markContextRunning(jobContext.id, 'job-new');
  global.EU_AGENT_STUDY_CONTEXT.markContextFinished(jobContext.id, 'done', { study_id: jobContext.id, gate: { status: 'analysis_only' } }, 'job-old');
  let jobRow = store.all().find(row => row.id === jobContext.id);
  assert.equal(jobRow.active_job_id, 'job-new', 'stale terminal must not clear the newer job');
  assert.equal(jobRow.current_stage, 'analyze');
  assert.equal(global.EU_AGENT_LAST_RUN, undefined, 'stale terminal must not replace the global last run');
  const blockedResult = { study_id: jobContext.id, gate: { status: 'blocked', reason: 'source_invalid', checks: [{ id: 'source_valid', passed: false }] } };
  global.EU_AGENT_STUDY_CONTEXT.markContextFinished(jobContext.id, 'done', blockedResult, 'job-new');
  jobRow = store.all().find(row => row.id === jobContext.id);
  assert.equal(jobRow.current_stage, 'review_blocked');
  assert.equal(jobRow.active_job_id, null);
  assert.equal(global.EU_AGENT_LAST_RUN.study_context_id, jobContext.id);

  const guidedJob = store.startNew({ question: 'Guided running context' }, { persist: false });
  await store.persist();
  global.EU_AGENT_STUDY_CONTEXT.markContextRunning(guidedJob.id, 'guided-job');
  const switched = store.startNew({ question: 'Context selected while Guided runs' }, { persist: false });
  await store.persist();
  const lastRunBeforeInactiveTerminal = global.EU_AGENT_LAST_RUN;
  global.EU_AGENT_STUDY_CONTEXT.markContextFinished(guidedJob.id, 'done', { study_id: guidedJob.id, gate: { status: 'analysis_only' } }, 'guided-job');
  assert.equal(store.active().id, switched.id, 'Guided terminal must not switch the active context');
  assert.equal(store.all().find(row => row.id === guidedJob.id).current_stage, 'review');
  assert.equal(store.all().find(row => row.id === switched.id).current_stage, 'study_setup');
  assert.equal(global.EU_AGENT_LAST_RUN, lastRunBeforeInactiveTerminal, 'inactive Guided A must not replace the global B-facing last run');

  const listKey = 'easyicu.studyContext.list.v1';
  const activeKey = 'easyicu.studyContext.active.v1';
  localStorage.setItem(listKey, JSON.stringify([
    switched,
    { id: 'unsafe-list-row', patient_rows: [{ patient_id: 'cache-secret' }] },
  ]));
  dispatchEvent({ type: 'storage', key: listKey, newValue: localStorage.getItem(listKey) });
  const scrubbedList = JSON.parse(localStorage.getItem(listKey));
  assert.equal(scrubbedList.length, 1, 'unsafe cached list rows must be physically removed');
  assert.equal(JSON.stringify(scrubbedList).includes('cache-secret'), false);
  localStorage.setItem(activeKey, JSON.stringify({ id: 'unsafe-active', cohort: { records: [{ stay_id: 'active-secret' }] } }));
  dispatchEvent({ type: 'storage', key: activeKey, newValue: localStorage.getItem(activeKey) });
  assert.equal(localStorage.getItem(activeKey), null, 'unsafe active cache must be physically removed');
  process.stdout.write(JSON.stringify({ patient: patient.context, guided: guided.context }));
})();
