/* Data Extraction -> Agent StudyContext mapping.
   Owner: extraction handoff metadata. Closure state is supplied only through
   window.EU_EXTRACTION_CONTEXT by the extraction screen owner. */
(function () {
  'use strict';

  const DATABASE_ALIASES = {
    miiv: 'miiv', mimiciv: 'miiv', mimic_iv: 'miiv',
    mimic: 'mimic', miii: 'mimic', mimiciii: 'mimic', mimic_iii: 'mimic',
    eicu: 'eicu', eicu_crd: 'eicu',
    aumc: 'aumc', amsterdamumcdb: 'aumc',
    hirid: 'hirid', sic: 'sic', sicdb: 'sic',
  };
  const COHORT_FIELDS = [
    'preset', 'age_min', 'age_max', 'min_icu_los_hours',
    'observation_window_hours', 'max_patients', 'exclude_readmissions',
    'icd_enabled', 'icd_include', 'icd_exclude',
  ];

  function cleanDatabase(value) {
    const key = String(value || '').trim().toLowerCase().replace(/[-\s]+/g, '_');
    return DATABASE_ALIASES[key] || '';
  }

  function cleanTextList(value, limit) {
    return Array.from(new Set((Array.isArray(value) ? value : [])
      .slice(0, limit || 64)
      .map(item => String(item || '').trim())
      .filter(Boolean)));
  }

  function project(context, expectedDatabase) {
    const current = context && typeof context === 'object' ? context : {};
    const source = current.data_source && typeof current.data_source === 'object'
      ? current.data_source : {};
    const rawCohort = current.cohort && typeof current.cohort === 'object'
      ? current.cohort : {};
    const cohort = {};
    COHORT_FIELDS.forEach(key => {
      if (Object.prototype.hasOwnProperty.call(rawCohort, key)) cohort[key] = rawCohort[key];
    });
    cohort.include_diagnoses = cleanTextList(rawCohort.include_diagnoses, 16);
    cohort.exclude_diagnoses = cleanTextList(rawCohort.exclude_diagnoses, 16);
    const rawExecution = current.execution_concepts && typeof current.execution_concepts === 'object'
      ? current.execution_concepts : {};
    const executionConcepts = {
      outcome: String(rawExecution.outcome || '').trim(),
      primary_exposure: String(rawExecution.primary_exposure || '').trim(),
      covariates: cleanTextList(rawExecution.covariates, 64),
    };
    const rawWindow = current.time_window && typeof current.time_window === 'object'
      ? current.time_window : {};
    return {
      study_context_id: String(current.id || current.study_context_id || '').trim(),
      revision: Number.isInteger(current.revision) && current.revision >= 0 ? current.revision : 0,
      expected_database: cleanDatabase(expectedDatabase || source.database),
      cohort,
      modules: cleanTextList(current.modules, 64),
      execution_concepts: executionConcepts,
      time_window: {
        ...(Number.isFinite(rawWindow.hours) ? { hours: rawWindow.hours } : {}),
        ...(Number.isFinite(rawWindow.observation_hours) ? { observation_hours: rawWindow.observation_hours } : {}),
        ...(rawWindow.anchor ? { anchor: String(rawWindow.anchor).slice(0, 160) } : {}),
      },
      export_format: ['parquet', 'csv', 'excel'].includes(String(current.export_format || '').toLowerCase())
        ? String(current.export_format).toLowerCase() : '',
    };
  }

  function hydrate(context, expectedDatabase) {
    const setup = project(context, expectedDatabase);
    const adapter = window.EU_EXTRACTION_CONTEXT;
    if (!adapter || typeof adapter.applyStudySetup !== 'function') return setup;
    adapter.applyStudySetup(setup);
    return setup;
  }

  function matchesDatabase(expected, actual) {
    const wanted = cleanDatabase(expected);
    if (!wanted) return true;
    return !!cleanDatabase(actual) && wanted === cleanDatabase(actual);
  }

  function build(current) {
    const existing = current || {};
    const adapter = window.EU_EXTRACTION_CONTEXT;
    const snapshot = adapter && adapter.snapshot ? adapter.snapshot() : {};
    const label = snapshot.preset_label || 'ICU cohort';
    return {
      title: existing.title || `${label} study`,
      // Selecting data authorizes the source and extraction setup only. A new
      // Copilot project must still collect the research question and goal from
      // the user instead of inheriting generic analysis prose as if confirmed.
      question: existing.question || '',
      purpose: existing.purpose || '',
      data_source: snapshot.data_source || existing.data_source || null,
      cohort: snapshot.cohort || existing.cohort || {},
      modules: snapshot.modules && snapshot.modules.length ? snapshot.modules : (existing.modules || []),
      outcome: existing.outcome || '',
      time_window: Object.assign({}, existing.time_window || {}, { observation_hours: snapshot.observation_hours }),
      comparator: existing.comparator || '',
      export_format: snapshot.export_format || existing.export_format || '',
      analysis_goal: existing.analysis_goal || '',
      confirmations: Object.assign({}, existing.confirmations || {}, { extraction_completed: !!snapshot.completed, crossdb_plan_only: false }),
      current_stage: snapshot.completed ? 'data_prepared' : 'study_setup',
    };
  }

  window.EU_EXTRACTION_STUDY_CONTEXT = { project, hydrate, matchesDatabase };

  if (window.EU_STUDY_CONTEXT && window.EU_STUDY_CONTEXT.registerSource) {
    window.EU_STUDY_CONTEXT.registerSource('extraction', build);
  }
})();
