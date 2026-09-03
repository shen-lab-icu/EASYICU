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

console.log('extraction StudyContext round-trip contract passed');
