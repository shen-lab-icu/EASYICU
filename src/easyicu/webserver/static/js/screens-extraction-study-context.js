/* Data Extraction -> Agent StudyContext mapping.
   Owner: extraction handoff metadata. Closure state is supplied only through
   window.EU_EXTRACTION_CONTEXT by the extraction screen owner. */
(function () {
  'use strict';

  const COHORT_FIELDS = [
    'preset', 'age_min', 'age_max', 'min_icu_los_hours',
    'observation_window_hours', 'max_patients', 'exclude_readmissions',
    'icd_enabled', 'icd_include', 'icd_exclude',
  ];

  function cleanDatabase(value) {
    // A study coordinate names a full database; a demo key does not clean to one.
    const key = String(value || '').trim().toLowerCase().replace(/[-\s]+/g, '_');
    const names = window.EU_DATABASE_NAMES;
    return names.isDemo(key) ? '' : names.key(key);
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
    ['include_diagnoses', 'exclude_diagnoses'].forEach(key => {
      if (Object.prototype.hasOwnProperty.call(rawCohort, key)) {
        cohort[key] = cleanTextList(rawCohort[key], 16);
      }
    });
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
    return adapter.applyStudySetup(setup);
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
    if (snapshot.study_context_id && snapshot.study_context_id !== String(existing.id || existing.study_context_id || '')) {
      throw new Error('The extraction workspace is still attached to another study. Finish its task before switching.');
    }
    const label = snapshot.preset_label || 'ICU cohort';
    // A folder-only visit may contain form defaults, not research decisions.
    // Only return fields the study supplied or the user actually configured.
    const fields = snapshot.configuration_fields;
    const cohort = fields && Array.isArray(fields.cohort) && !snapshot.completed
      ? Object.fromEntries(fields.cohort.filter(key => Object.prototype.hasOwnProperty.call(snapshot.cohort || {}, key))
        .map(key => [key, snapshot.cohort[key]]))
      : snapshot.cohort;
    const hasWindow = !fields || snapshot.completed || fields.cohort.includes('observation_window_hours');
    const hasFormat = !fields || snapshot.completed || fields.export_format;
    return {
      title: existing.title || `${label} study`,
      // Selecting data authorizes the source and extraction setup only. A new
      // Copilot project must still collect the research question and goal from
      // the user instead of inheriting generic analysis prose as if confirmed.
      question: existing.question || '',
      purpose: existing.purpose || '',
      data_source: snapshot.data_source || existing.data_source || null,
      cohort: fields && !snapshot.completed
        ? Object.assign({}, existing.cohort || {}, cohort || {})
        : (snapshot.cohort || existing.cohort || {}),
      modules: Array.isArray(snapshot.modules) ? snapshot.modules : (existing.modules || []),
      outcome: existing.outcome || '',
      time_window: Object.assign({}, existing.time_window || {}, hasWindow ? { observation_hours: snapshot.observation_hours } : {}),
      comparator: existing.comparator || '',
      export_format: (hasFormat && snapshot.export_format) || existing.export_format || '',
      analysis_goal: existing.analysis_goal || '',
      confirmations: Object.assign({}, existing.confirmations || {}, { extraction_completed: !!snapshot.completed, crossdb_plan_only: false }),
      current_stage: snapshot.completed ? 'data_prepared' : 'study_setup',
    };
  }

  // Read-only view helpers use the extraction owner's explicit snapshot and
  // display dependencies; they never write form state or scientific criteria.
  function formatHours(value, t) {
    const n = Math.max(0, Number(value || 0));
    if (!n) return t('Any', '不限');
    if (n % 24 === 0) return n + 'h · ' + (n / 24) + 'd';
    return n + 'h';
  }

  function formatObservationWindow(value, { t, defaultHours, maxHours }) {
    const n = Math.max(1, Number(value || defaultHours));
    if (n >= maxHours) return t('full available · 30d cap', '全可用 · 30天上限');
    return t('first ', '前 ') + formatHours(n, t);
  }

  function formatAgeRange(ageMin, ageMax, t) {
    if (ageMin <= 0 && ageMax >= 100) return t('all ages', '全部年龄');
    if (ageMin <= 0) return '≤ ' + ageMax;
    if (ageMax >= 100) return '≥ ' + ageMin;
    return ageMin + '–' + ageMax;
  }

  function folderHint(expectedDatabase, { t, hostedCopy }) {
    const name = window.EU_DATABASE_NAMES.release(expectedDatabase);
    return name
      ? hostedCopy('folderHintNamed', t(`Choose the ${name} data folder; EasyICU then identifies its layout automatically. No patient rows are uploaded or returned.`, `选择 ${name} 数据文件夹后，EasyICU 会自动判断目录结构。不上传，也不返回患者行。`), name)
      : hostedCopy('folderHint', t('Choose the ICU data folder; EasyICU then identifies its database and layout automatically. No patient rows are uploaded or returned.', '选择 ICU 数据文件夹后，EasyICU 会自动判断数据库和目录结构。不上传，也不返回患者行。'));
  }

  function renderPrefill(snapshot, { t, icon, esc, edited, cohortLabel, adult, ageLabel, firstOnly, windowLabel }) {
    if (!snapshot.study_context_id) return '';
    const cohort = snapshot.cohort;
    const database = window.EU_DATABASE_NAMES.release(snapshot.expected_database) || snapshot.expected_database.toUpperCase();
    const diagnoses = cohort.preset === 'icd' && Array.isArray(cohort.include_diagnoses) ? cohort.include_diagnoses.join(', ') : '';
    const fields = new Set(snapshot.configuration_fields && snapshot.configuration_fields.cohort || []);
    const has = key => fields.has(key);
    const population = has('preset') || diagnoses || (has('icd_enabled') && cohort.icd_enabled === true);
    const age = has('age_min') || has('age_max') || adult;
    const first = has('exclude_readmissions') || cohort.preset === 'adult_first';
    const observation = has('observation_window_hours');
    const parts = [
      database,
      population ? (diagnoses ? `ICD ${diagnoses}` : cohortLabel) : '',
      age ? t('age ', '年龄 ') + ageLabel : '',
      first ? (firstOnly ? t('first ICU stay', '首次 ICU 住院') : t('repeat stays allowed', '不排除重复入住')) : '',
      observation ? windowLabel : '',
      snapshot.modules.join(', '),
      snapshot.configuration_fields && snapshot.configuration_fields.export_format ? String(snapshot.export_format || '').toUpperCase() : '',
    ].filter(Boolean);
    const pending = [
      !database && t('database', '数据库'), !population && t('population', '起点人群'),
      !age && t('age range', '年龄范围'), !first && t('first-stay rule', '首次入住规则'),
      !observation && t('measurement window', '测量窗口'), !snapshot.modules.length && t('features', '特征'),
    ].filter(Boolean);
    const state = !parts.length ? 'pending' : (pending.length ? 'partial' : 'ready');
    const title = edited
      ? t('Current extraction choices · review required', '当前提取设置 · 待核对')
      : state === 'pending' ? t('Extraction setup not yet synced', '尚未同步提取配置')
        : state === 'partial' ? t('Some extraction choices loaded', '已带入部分提取条件')
          : t('Loaded from Copilot · review required', '已从 Copilot 带入 · 待核对');
    const description = state === 'pending'
      ? t('Research decisions are still in the conversation. Form defaults are not synced research criteria.', '研究条件仍在对话中讨论；表单默认值尚未成为本研究的提取配置。')
      : parts.join(' · ') + (pending.length ? t(' · Not yet specified: ', ' · 尚未确定：') + pending.join('、') : '');
    return `
      <div class="ex-connect-primary ex-copilot-prefill" role="note" data-ex-prefill-state="${state}">
        <div class="ex-connect-copy">
          <div class="ex-connect-copy-ico">${icon('spark', 15)}</div>
          <div>
            <div class="ex-connect-copy-title">${title}</div>
            <div class="ex-connect-copy-desc">${esc(description)}</div>
          </div>
        </div>
      </div>`;
  }

  window.EU_EXTRACTION_STUDY_CONTEXT = { project, hydrate, matchesDatabase, renderPrefill, folderHint, formatHours, formatObservationWindow, formatAgeRange };

  if (window.EU_STUDY_CONTEXT && window.EU_STUDY_CONTEXT.registerSource) {
    window.EU_STUDY_CONTEXT.registerSource('extraction', build);
  }
})();
