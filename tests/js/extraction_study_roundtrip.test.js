'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const ownerPath = process.argv[2];
assert.ok(ownerPath, 'screens-extraction-study-context.js path is required');

let supplier = null;
let applied = null;
global.window = {
  EU_STUDY_CONTEXT: {
    registerSource(route, callback) {
      assert.equal(route, 'extraction');
      supplier = callback;
    },
  },
  EU_EXTRACTION_CONTEXT: {
    applyStudySetup(setup) { applied = setup; return setup; },
    snapshot() {
      return {
        data_source: { path: '/private/export/run-1', label: 'MIMIC-IV', database: 'miiv' },
        cohort: { preset: 'icd', age_min: 18, include_diagnoses: ['A41'] },
        modules: ['blood_gas'],
        preset_label: 'Diagnosis / ICD cohort',
        export_format: 'parquet',
        observation_hours: 24,
      };
    },
  },
};

const namesPath = require('node:path').join(require('node:path').dirname(ownerPath), 'database-names.js');
vm.runInThisContext(fs.readFileSync(namesPath, 'utf8'), { filename: namesPath });
vm.runInThisContext(fs.readFileSync(ownerPath, 'utf8'), { filename: ownerPath });

const context = {
  id: 'study-a41',
  revision: 7,
  question: 'How many adult MIMIC-IV ICU stays have ICD A41, and what is lactate coverage?',
  data_source: { path: '/private/raw/mimiciv', label: 'MIMIC-IV 3.1', database: 'miiv' },
  cohort: {
    preset: 'icd', age_min: 18, age_max: 100, min_icu_los_hours: 0,
    observation_window_hours: 24, exclude_readmissions: false,
    icd_enabled: true, include_diagnoses: ['A41'], exclude_diagnoses: [],
  },
  modules: ['blood_gas'],
  execution_concepts: { primary_exposure: 'lact', covariates: [] },
  time_window: { observation_hours: 24, anchor: 'ICU admission' },
  export_format: 'parquet',
};

const projected = window.EU_EXTRACTION_STUDY_CONTEXT.project(context, 'miiv');
assert.equal(projected.study_context_id, 'study-a41');
assert.equal(projected.revision, 7);
assert.equal(projected.expected_database, 'miiv');
assert.deepEqual(projected.cohort.include_diagnoses, ['A41']);
assert.deepEqual(projected.modules, ['blood_gas']);
assert.equal(projected.execution_concepts.primary_exposure, 'lact');
assert.equal(JSON.stringify(projected).includes('/private/'), false, 'inbound setup projection must be path-free');

window.EU_EXTRACTION_STUDY_CONTEXT.hydrate(context, 'miiv');
assert.deepEqual(applied, projected);
assert.equal(window.EU_EXTRACTION_STUDY_CONTEXT.matchesDatabase('miiv', 'mimiciv'), true);
assert.equal(window.EU_EXTRACTION_STUDY_CONTEXT.matchesDatabase('miiv', 'sicdb'), false);

assert.ok(supplier, 'outbound extraction supplier must stay registered');
const outgoing = supplier(context);
assert.equal(outgoing.question, context.question, 'round-trip must preserve the original research question');
assert.equal(outgoing.data_source.database, 'miiv');
assert.deepEqual(outgoing.cohort.include_diagnoses, ['A41']);
assert.deepEqual(outgoing.modules, ['blood_gas']);
assert.equal(outgoing.export_format, 'parquet');

// Execute the actual form owner: a study identity alone must not promote the
// form's defaults to research choices, or retain the previous study's choices.
const path = require('node:path');
const root = path.dirname(ownerPath);
const page = vm.createContext({
  window: {}, t: (_en, zh) => zh, icon: () => '',
  setTimeout: () => 1, clearTimeout() {},
  document: { querySelector: () => null },
  location: { hash: '#extraction' },
});
page.window.EU_STUDY_CONTEXT = { registerSource(_route, callback) { page.supplier = callback; } };
for (const file of ['database-names.js', 'html-escape.js', 'screens-extraction.js', 'screens-extraction-study-context.js']) {
  vm.runInContext(fs.readFileSync(path.join(root, file), 'utf8'), page, { filename: file });
}
const copy = value => JSON.parse(JSON.stringify(value));
const bridge = page.window.EU_EXTRACTION_STUDY_CONTEXT;
const form = page.window.EU_EXTRACTION_CONTEXT;
const render = () => page.window.SCREENS.extraction.render();

const complete = {
  id: 'study-complete', revision: 1, data_source: { database: 'miiv' },
  cohort: { preset: 'adult_first', age_min: 18, age_max: 80, observation_window_hours: 24 },
  modules: ['blood_gas'], export_format: 'csv',
};
bridge.hydrate(complete);
const completeHtml = render();
assert.equal(form.snapshot().cohort.age_max, 80);
assert.equal(form.snapshot().observation_hours, 24);

const sparse = { id: 'study-new', revision: 0, question: 'Discuss a research idea before choosing data.' };
bridge.hydrate(sparse);
const fresh = copy(form.snapshot());
assert.equal(fresh.cohort.preset, 'all_icu', 'form defaults reset across studies');
assert.equal(fresh.cohort.age_min, 0);
assert.equal(fresh.cohort.age_max, 100);
assert.equal(fresh.observation_hours, 720);
assert.equal(fresh.export_format, 'parquet');
assert.deepEqual(fresh.modules, []);
assert.match(completeHtml, /data-ex-prefill-state="ready"/);
assert.deepEqual(fresh.configuration_fields.cohort, [], 'unset criteria stay unspecified');
assert.match(render(), /data-ex-prefill-state="pending"/);
assert.doesNotMatch(render(), /全部 ICU 住院|全部年龄|30天上限|已从 Copilot 带入/);
const sparseReturn = copy(page.supplier(sparse));
assert.deepEqual(sparseReturn.cohort, {}, 'a visit must not write all-ICU defaults into the study');
assert.deepEqual(sparseReturn.time_window, {});
assert.equal(sparseReturn.export_format, '');
assert.deepEqual(sparseReturn.confirmations, { extraction_completed: false, crossdb_plan_only: false });

const partial = {
  id: 'study-partial', revision: 0, cohort: { age_min: 18 },
  time_window: { observation_hours: 24 }, modules: [],
};
bridge.hydrate(partial);
assert.match(render(), /data-ex-prefill-state="partial"/);
assert.match(render(), /年龄 ≥ 18/);
assert.match(render(), /前 24h/);
assert.match(render(), /尚未确定/);
assert.doesNotMatch(render(), /全部 ICU 住院|30天上限/);
const partialReturn = copy(page.supplier(partial));
assert.deepEqual(partialReturn.cohort, { age_min: 18, observation_window_hours: 24 });
assert.deepEqual(partialReturn.time_window, { observation_hours: 24 });

// Local pending revisions can change before the server allocates a new revision.
bridge.hydrate({ ...partial, cohort: { age_min: 21 }, time_window: { observation_hours: 12 } });
assert.equal(form.snapshot().cohort.age_min, 21);
assert.equal(form.snapshot().observation_hours, 12);

const handlers = {};
const ageControl = { dataset: { exRange: 'age_max' }, value: '75', type: 'number', addEventListener(name, callback) { handlers[name] = callback; } };
page.window.SCREENS.extraction.afterRender({
  querySelector: () => null,
  querySelectorAll: selector => selector === '[data-ex-range]' ? [ageControl] : [],
});
handlers.change();
const edited = copy(page.supplier({ ...partial, cohort: { age_min: 21 } }));
assert.equal(edited.cohort.age_max, 75, 'actual user edits do round-trip');
assert.equal(Object.hasOwn(edited.cohort, 'preset'), false, 'an age edit does not specify a starting population');
assert.match(render(), /当前提取设置 · 待核对/);

// Selecting the six core feature modules must not silently switch the final
// button to a shortcut that replaces the user's cohort and observation window.
const posted = [];
page.window.EU_SETTINGS = { export_dir: '/test/no-files-are-written' };
page.window.EventSource = function EventSource() {};
page.window.EU_API = { postJSON: (url, body) => { posted.push({ url, body: copy(body) }); return Promise.resolve({ job_id: 'test-only' }); } };
page.window.EU_EXTRACTION_JOB_CONTINUITY = { abandon() {}, prepare: () => 'ticket', attach: () => true, isPending: () => false };
bridge.hydrate({
  ...complete, id: 'study-core-modules',
  modules: ['demographics', 'vitals', 'chemistry', 'sofa2_score', 'sepsis3_sofa2', 'outcome'],
});
const conversion = { kind: 'convert', job_id: 'test-conversion', source: { path: '/test/source', database: 'miiv' } };
page.window.EU_EXTRACTION_JOB_HOST.begin(conversion);
page.window.EU_EXTRACTION_JOB_HOST.applyEvent(conversion, { type: 'end', status: 'done', result: {} });
function clickControl(selector, dataset, many) {
  let callback;
  const control = { dataset, addEventListener(_name, handler) { callback = handler; } };
  page.window.SCREENS.extraction.afterRender({
    querySelector: candidate => !many && candidate === selector ? control : null,
    querySelectorAll: candidate => many && candidate === selector ? [control] : [],
  });
  assert.ok(callback, `control is wired: ${selector}`);
  callback();
}
clickControl('[data-ex-convdone]', {});
const stepSelector = '[data-ex-step-next], [data-ex-step-back], [data-ex-step-target]';
clickControl(stepSelector, { exStepNext: 'cohort' }, true);
clickControl(stepSelector, { exStepNext: 'export' }, true);
const finalButton = render().match(/class="btn primary ex-run-primary" data-ex-run="([^"]+)"/);
assert.ok(finalButton, 'the final extraction action is present');
clickControl('[data-ex-run]', { exRun: finalButton[1] }, true);
assert.equal(posted.length, 1);
assert.equal(posted[0].url, '/api/jobs/extract');
assert.deepEqual(posted[0].body.modules, ['demographics', 'vitals', 'chemistry', 'sofa2_score', 'sepsis3_sofa2', 'outcome']);
assert.equal(posted[0].body.cohort.preset, 'adult_first');
assert.equal(posted[0].body.cohort.age_min, 18);
assert.equal(posted[0].body.cohort.age_max, 80);
assert.equal(posted[0].body.cohort.observation_window_hours, 24);
assert.equal(posted[0].body.format, 'csv');

const oldRun = {
  kind: 'extract', job_id: 'old-study-run', source: { path: '/test/source', database: 'miiv' },
  config: { run_mode: 'recommended', modules: ['vitals'] },
};
page.window.EU_EXTRACTION_JOB_HOST.begin(oldRun);
page.window.EU_EXTRACTION_JOB_HOST.applyEvent(oldRun, {
  type: 'end', status: 'done', result: { out_dir: '/test/old-export', registered_source: { ok: true } },
});
assert.equal(form.snapshot().completed, true);
const afterRun = { id: 'study-after-old-run', revision: 0 };
page.window.EU_SOURCES = { activeSource: () => ({ id: 'old-registry-source', path: '/test/old-export', label: 'Old study export' }) };
bridge.hydrate(afterRun);
assert.equal(form.snapshot().completed, false, 'a finished run is not a result for a new study');
assert.deepEqual(copy(form.snapshot().modules), [], 'a prior run cannot supply the new study\'s module selection');
assert.equal(form.snapshot().data_source, null, 'the globally active registry export is not the new study\'s selected folder');
assert.deepEqual(copy(page.supplier(afterRun).cohort), {}, 'a prior recommended run cannot turn unknown criteria into defaults');

page.window.EU_EXTRACTION_JOB_CONTINUITY.isRunning = () => true;
assert.equal(bridge.hydrate({ id: 'cannot-steal-running-form' }), null);
assert.equal(form.snapshot().study_context_id, afterRun.id);
assert.throws(() => page.supplier({ id: 'cannot-steal-running-form' }), /still attached to another study/);

// The running job owns the configuration that will label its terminal result.
// Exercise the actual native/embedded owners without issuing backend requests.
function runningOwnerPage(study) {
  const current = vm.createContext({
    window: {}, t: (_en, zh) => zh, icon: () => '',
    setTimeout: () => 1, clearTimeout() {},
    document: { querySelector: () => null }, location: { hash: '#extraction' },
  });
  current.window.EU_STUDY_CONTEXT = { registerSource(_route, callback) { current.supplier = callback; } };
  for (const file of ['database-names.js', 'html-escape.js', 'screens-extraction.js', 'screens-extraction-study-context.js', 'screens-extraction-embedded.js']) {
    vm.runInContext(fs.readFileSync(path.join(root, file), 'utf8'), current, { filename: file });
  }
  current.running = false;
  current.abandoned = 0;
  current.window.EU_EXTRACTION_JOB_CONTINUITY = {
    isRunning: () => current.running,
    abandon() { current.abandoned += 1; current.running = false; },
  };
  if (study) current.window.EU_EXTRACTION_STUDY_CONTEXT.hydrate(study);
  current.run = {
    kind: 'extract', job_id: 'original-running-job', source: { path: '/test/source', database: 'miiv' },
    config: { run_mode: 'recommended', modules: ['vitals'], format: 'csv', max_patients: 250 },
  };
  current.window.EU_EXTRACTION_JOB_HOST.begin(current.run);
  current.running = true;
  return current;
}
function finishOriginalRun(current, study) {
  current.window.EU_EXTRACTION_JOB_HOST.applyEvent(current.run, {
    type: 'end', status: 'done', result: { out_dir: '/test/original-finished-export', registered_source: { ok: true } },
  });
  current.running = false;
  const outgoing = copy(current.supplier(study));
  assert.equal(outgoing.confirmations.extraction_completed, true, 'original tracked job still supplies its terminal result');
  assert.equal(outgoing.data_source.path, '/test/original-finished-export');
  assert.deepEqual(outgoing.modules, ['vitals']);
  assert.equal(outgoing.cohort.preset, 'all_icu');
  assert.equal(outgoing.cohort.age_min, 0);
  assert.equal(outgoing.cohort.observation_window_hours, 720);
  assert.equal(outgoing.cohort.max_patients, 250);
  assert.equal(outgoing.export_format, 'csv');
  return outgoing;
}
const revised = { ...complete, revision: 2, cohort: { age_min: 21 }, modules: ['chemistry'], export_format: 'parquet' };
const sameStudyRun = runningOwnerPage(complete);
const sameStudyBefore = copy(sameStudyRun.window.EU_EXTRACTION_CONTEXT.snapshot());
assert.ok(sameStudyRun.window.EU_EXTRACTION_STUDY_CONTEXT.hydrate(complete), 'identical hydration is a harmless no-op');
assert.deepEqual(copy(sameStudyRun.window.EU_EXTRACTION_CONTEXT.snapshot()), sameStudyBefore);
assert.ok(sameStudyRun.window.EU_EXTRACTION_STUDY_CONTEXT.hydrate({ ...complete, revision: 2 }), 'revision-only refresh is also a no-op');
assert.deepEqual(copy(sameStudyRun.window.EU_EXTRACTION_CONTEXT.snapshot()), sameStudyBefore);
assert.equal(sameStudyRun.window.EU_EXTRACTION_STUDY_CONTEXT.hydrate(revised), null, 'same-ID revision cannot replace running job configuration');
assert.deepEqual(copy(sameStudyRun.window.EU_EXTRACTION_CONTEXT.snapshot()), sameStudyBefore);
assert.equal(sameStudyRun.window.EU_EXTRACTION_NATIVE_OWNER.applyExtractionScope('all_supported'), false);
assert.equal(sameStudyRun.window.EU_EXTRACTION_NATIVE_OWNER.beginSourceBinding(), false);
assert.equal(sameStudyRun.abandoned, 0);
assert.equal(sameStudyRun.running, true);
const sameStudyHost = { isConnected: true, querySelector: () => null, querySelectorAll: () => [] };
assert.equal(sameStudyRun.window.EU_EXTRACTION_EMBEDDED_WORKSPACE.mount(sameStudyHost, {
  studyContext: complete,
  resource: { study_context_id: complete.id, study_revision: complete.revision, extraction_scope: 'all_supported' },
}), false, 'an identical study does not permit changing the running job module scope');
assert.deepEqual(copy(sameStudyRun.window.EU_EXTRACTION_CONTEXT.snapshot()), sameStudyBefore);
finishOriginalRun(sameStudyRun, complete);
assert.ok(sameStudyRun.window.EU_EXTRACTION_STUDY_CONTEXT.hydrate(revised));
assert.equal(sameStudyRun.window.EU_EXTRACTION_CONTEXT.snapshot().completed, false, 'a settled result cannot be relabelled with changed study settings');
assert.deepEqual(copy(sameStudyRun.window.EU_EXTRACTION_CONTEXT.snapshot().modules), ['chemistry']);
assert.equal(sameStudyRun.window.EU_EXTRACTION_CONTEXT.snapshot().cohort.age_min, 21);

const unboundRun = runningOwnerPage();
const unboundBefore = copy(unboundRun.window.EU_EXTRACTION_CONTEXT.snapshot());
assert.equal(unboundBefore.study_context_id, '');
assert.equal(unboundRun.window.EU_EXTRACTION_STUDY_CONTEXT.hydrate(complete), null, 'first study hydration cannot relabel a native running job');
assert.deepEqual(copy(unboundRun.window.EU_EXTRACTION_CONTEXT.snapshot()), unboundBefore);
assert.equal(unboundRun.abandoned, 0);
finishOriginalRun(unboundRun, {});
assert.equal(unboundRun.window.EU_EXTRACTION_CONTEXT.snapshot().study_context_id, '');

const embeddedRun = runningOwnerPage(complete);
const embeddedBefore = copy(embeddedRun.window.EU_EXTRACTION_CONTEXT.snapshot());
const embeddedHost = { isConnected: true, querySelector: () => null, querySelectorAll: () => [] };
assert.equal(embeddedRun.window.EU_EXTRACTION_EMBEDDED_WORKSPACE.mount(embeddedHost, {
  studyContext: { id: 'other-study', revision: 1 },
  resource: { entry_mode: 'source_binding', study_context_id: 'other-study', expected_database: 'miiv', extraction_scope: 'all_supported' },
}), false);
assert.deepEqual(copy(embeddedRun.window.EU_EXTRACTION_CONTEXT.snapshot()), embeddedBefore);
assert.equal(embeddedRun.abandoned, 0, 'rejected embedded hydration cannot abandon tracking');
assert.equal(embeddedRun.running, true);
assert.match(embeddedHost.innerHTML, /当前数据任务仍在跟踪/);
assert.match(embeddedHost.innerHTML, /新研究配置尚未应用/);
assert.doesNotMatch(embeddedHost.innerHTML, /exPathInput|data-gpi-source-binding-confirm|data-gpi-extraction-sync/);
finishOriginalRun(embeddedRun, complete);
assert.ok(embeddedRun.window.EU_EXTRACTION_STUDY_CONTEXT.hydrate({ id: 'other-study', revision: 1 }), 'new setup can be opened once the old task has settled');

console.log('extraction StudyContext round-trip and sparse-config contract passed');
