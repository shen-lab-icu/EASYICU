/* Cross-module StudyContext owner.
   Keeps an immediate browser cache for navigation continuity, then mirrors the
   same context to the local FastAPI store. Route modules only register a source
   supplier; API transport remains in api.js and shell navigation remains in
   app.js. */
(function () {
  'use strict';

  const STORAGE_KEY = 'easyicu.studyContext.active.v1';
  const LIST_STORAGE_KEY = 'easyicu.studyContext.list.v1';
  const DIRTY_STORAGE_KEY = 'easyicu.studyContext.dirty.v1';
  const EVENT_NAME = 'easyicu:study-context';
  const PERSISTED_FIELDS = [
    'id', 'title', 'question', 'purpose', 'data_source', 'crossdb_selection', 'cohort', 'modules',
    'outcome', 'primary_exposure', 'covariates', 'covariate_selection',
    'covariate_rationales', 'covariate_temporal_roles', 'execution_concepts',
    'analysis_design', 'sensitivity_specs', 'time_window', 'comparator',
    'export_format', 'analysis_goal', 'current_stage', 'last_route',
    'active_job_id', 'confirmations', 'idea_handoff',
  ];
  const ROW_LEVEL_KEYS = new Set([
    'tablerows', 'rowdata', 'rows', 'records', 'values', 'observations', 'series',
    'patient', 'patients', 'patientid', 'patientids', 'stayid', 'stayids',
    'subjectid', 'subjectids', 'hadmid', 'hadmids', 'entityid', 'entityids',
  ]);
  const COHORT_SCHEMA = {
    preset: 'text', label: 'text', review: 'text', review_scope: 'text', comparison: 'text',
    source_type: 'text', comparison_mode: 'text', icd_include: 'text', icd_exclude: 'text',
    age_min: 'number', age_max: 'number', min_icu_los_hours: 'number',
    observation_window_hours: 'number', max_patients: 'number', entity_count: 'number',
    full_entity_count: 'number', review_entities: 'number', review_entity_cap: 'number',
    module_count: 'number', cohort_size: 'number', source_count: 'number',
    exclude_readmissions: 'bool', icd_enabled: 'bool',
    include_diagnoses: 'text_list', exclude_diagnoses: 'text_list',
    sepsis_definition: {
      record_scope: 'text', runtime_profile: 'text', implementation_profile: 'text',
      score_family: 'text', definition_locked: 'bool',
      suspected_infection: {
        mode: 'text', abx_win_hours: 'number', samp_win_hours: 'number',
        abx_count_win_hours: 'number', abx_min_count: 'number',
        positive_cultures_required: 'bool',
      },
      sofa_increase: {
        si_window: 'text', window_before_si_hours: 'number', window_after_si_hours: 'number',
        delta_function: 'text', threshold: 'number', keep_components: 'bool',
      },
      review_options: { si_window: 'text_list' },
      locked_core: {
        suspected_infection_windows: 'text', sofa_window: 'text', delta_rule: 'text',
        sofa_threshold: 'text',
      },
    },
  };
  const TIME_WINDOW_SCHEMA = {
    hours: 'number', observation_hours: 'number', anchor: 'text', preset: 'text', label: 'text',
  };
  const EXECUTION_CONCEPTS_SCHEMA = {
    outcome: 'text', primary_exposure: 'text', covariates: 'text_list',
  };
  const ANALYSIS_DESIGN_SCHEMA = {
    analysis_unit: 'text', variance_estimator: 'text', cluster_unit: 'text',
  };
  const IDEA_HANDOFF_SCHEMA = {
    schema_version: 'text', run_id: 'text', idea_id: 'text',
    canonical_handoff_sha256: 'text', status: 'text', accepted_at: 'text',
    go_no_go: 'text', go_no_go_reason: 'text',
  };
  const suppliers = {};
  let activeContext = readCache();
  let contexts = readContextList();
  if (activeContext) rememberContext(activeContext);
  let revision = 0;
  const contextRevisions = new Map();
  let persistQueue = Promise.resolve();
  let dirtyIds = new Set();
  try {
    const storedDirty = JSON.parse(localStorage.getItem(DIRTY_STORAGE_KEY) || '[]');
    if (Array.isArray(storedDirty)) dirtyIds = new Set(storedDirty.map(text).filter(Boolean));
  } catch (_) {}
  let syncState = {
    state: activeContext ? 'cached' : 'empty',
    error: null,
    updated_at: null,
  };
  let hydratePromise = null;

  function text(value) {
    return value == null ? '' : String(value).trim();
  }

  function metadataText(value, maxLength) {
    return typeof value === 'string' ? value.trim().slice(0, maxLength) : '';
  }

  function contextRevision(id) {
    return contextRevisions.get(id) || 0;
  }

  function bumpContextRevision(id) {
    contextRevisions.set(id, contextRevision(id) + 1);
  }

  function localId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
      return 'study-' + window.crypto.randomUUID();
    }
    return 'study-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 9);
  }

  function cleanObject(value) {
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  }

  function cleanList(value) {
    return Array.isArray(value) ? value.filter(item => item != null) : [];
  }

  function assertMetadataOnly(value, path) {
    if (Array.isArray(value)) {
      value.forEach((item, index) => assertMetadataOnly(item, `${path || 'context'}[${index}]`));
      return;
    }
    if (!value || typeof value !== 'object') return;
    Object.keys(value).forEach(key => {
      const normalizedKey = String(key).toLowerCase().replace(/[^a-z0-9]+/g, '');
      if (ROW_LEVEL_KEYS.has(normalizedKey)) {
        throw new Error(`Row-level StudyContext metadata is forbidden at ${(path || 'context')}.${key}`);
      }
      assertMetadataOnly(value[key], `${path || 'context'}.${key}`);
    });
  }

  function cleanSchemaObject(value, schema) {
    const raw = cleanObject(value);
    const result = {};
    Object.keys(schema).forEach(key => {
      if (!Object.prototype.hasOwnProperty.call(raw, key)) return;
      const kind = schema[key];
      const candidate = raw[key];
      if (kind && typeof kind === 'object') {
        if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
          result[key] = cleanSchemaObject(candidate, kind);
        }
      } else if (kind === 'text') {
        if (typeof candidate === 'string') result[key] = metadataText(candidate, 500);
      } else if (kind === 'number') {
        if (typeof candidate === 'number' && Number.isFinite(candidate)) result[key] = candidate;
      } else if (kind === 'bool') {
        if (typeof candidate === 'boolean') result[key] = candidate;
      } else if (kind === 'text_list' && Array.isArray(candidate)) {
        result[key] = candidate.slice(0, 64).filter(item => typeof item === 'string').map(item => metadataText(item, 160)).filter(Boolean);
      }
    });
    return result;
  }

  function cleanConfirmations(value) {
    const raw = cleanObject(value);
    const result = {};
    Object.keys(raw).slice(0, 64).forEach(key => {
      if (/^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/.test(key) && typeof raw[key] === 'boolean') {
        result[key] = raw[key];
      }
    });
    return result;
  }

  function cleanSensitivitySpecs(value) {
    if (!Array.isArray(value)) return [];
    return value.slice(0, 16).map(candidate => {
      const raw = cleanObject(candidate);
      const row = {
        spec_id: metadataText(raw.spec_id, 80),
        axis: metadataText(raw.axis, 40),
        strategy: metadataText(raw.strategy, 80),
        execution_variables: Array.from(new Set(cleanList(raw.execution_variables)
          .slice(0, 16)
          .filter(item => typeof item === 'string')
          .map(item => metadataText(item, 80))
          .filter(Boolean))),
        require_alive_at_landmark: raw.require_alive_at_landmark === true,
        exclude_negative_event_times: raw.exclude_negative_event_times === true,
      };
      if (typeof raw.landmark_hours === 'number' && Number.isFinite(raw.landmark_hours)) {
        row.landmark_hours = raw.landmark_hours;
      }
      return row;
    }).filter(row => row.spec_id && row.axis && row.strategy);
  }

  function cleanDataSource(value) {
    const source = cleanObject(value);
    const clean = {
      path: metadataText(source.path, 4096),
      label: metadataText(source.label, 160),
      database: metadataText(source.database, 64),
    };
    return clean.path || clean.label || clean.database ? clean : null;
  }

  function cleanCrossdbSelection(value) {
    const raw = cleanObject(value);
    if (!Object.keys(raw).length) return {};
    const sources = cleanList(raw.sources).slice(0, 64).map(source => {
      const row = cleanObject(source);
      return {
        source_id: metadataText(row.source_id, 80),
        label: metadataText(row.label, 160),
        database: metadataText(row.database, 64),
        path_hash: metadataText(row.path_hash, 64).toLowerCase(),
      };
    }).filter(source => source.source_id && /^[0-9a-f]{12,64}$/.test(source.path_hash));
    const sourceCount = Number(raw.source_count);
    const digest = metadataText(raw.selection_digest, 64).toLowerCase();
    if (raw.schema_version !== 'crossdb-selection-v1'
        || !Number.isInteger(sourceCount) || sourceCount < 2
        || sourceCount !== sources.length || !/^[0-9a-f]{64}$/.test(digest)) return {};
    return {
      schema_version: 'crossdb-selection-v1',
      source_count: sourceCount,
      sources,
      selection_digest: digest,
    };
  }

  function normalize(value) {
    const raw = cleanObject(value);
    assertMetadataOnly(raw, 'context');
    const id = text(raw.id || raw.study_context_id) || localId();
    const question = metadataText(raw.question, 1200);
    return {
      id,
      revision: (typeof raw.revision === 'number' && Number.isInteger(raw.revision) && raw.revision >= 0) ? raw.revision : 0,
      title: metadataText(raw.title, 160) || question.slice(0, 160) || 'Untitled ICU study',
      question,
      purpose: metadataText(raw.purpose, 800),
      data_source: cleanDataSource(raw.data_source),
      crossdb_selection: cleanCrossdbSelection(raw.crossdb_selection),
      cohort: cleanSchemaObject(raw.cohort, COHORT_SCHEMA),
      modules: Array.from(new Set(cleanList(raw.modules).slice(0, 64).filter(item => typeof item === 'string').map(item => metadataText(item, 80)).filter(Boolean))),
      outcome: metadataText(raw.outcome, 500),
      primary_exposure: metadataText(raw.primary_exposure, 160),
      covariates: Array.from(new Set(cleanList(raw.covariates).slice(0, 64).filter(item => typeof item === 'string').map(item => metadataText(item, 160)).filter(Boolean))),
      covariate_selection: ['planner_selectable', 'exact'].includes(text(raw.covariate_selection)) ? text(raw.covariate_selection) : 'planner_selectable',
      covariate_rationales: Object.fromEntries(Object.entries(cleanObject(raw.covariate_rationales)).slice(0, 64).filter(([key, value]) => /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/.test(key) && typeof value === 'string').map(([key, value]) => [key, metadataText(value, 500)]).filter(([, value]) => value)),
      covariate_temporal_roles: Object.fromEntries(Object.entries(cleanObject(raw.covariate_temporal_roles)).slice(0, 64).filter(([key, value]) => /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/.test(key) && ['baseline_static', 'at_or_before_time_zero'].includes(value))),
      execution_concepts: cleanSchemaObject(raw.execution_concepts, EXECUTION_CONCEPTS_SCHEMA),
      analysis_design: cleanSchemaObject(raw.analysis_design, ANALYSIS_DESIGN_SCHEMA),
      sensitivity_specs: cleanSensitivitySpecs(raw.sensitivity_specs),
      time_window: cleanSchemaObject(raw.time_window, TIME_WINDOW_SCHEMA),
      comparator: metadataText(raw.comparator, 500),
      export_format: metadataText(raw.export_format, 40),
      analysis_goal: metadataText(raw.analysis_goal, 1200),
      confirmations: cleanConfirmations(raw.confirmations),
      idea_handoff: cleanSchemaObject(raw.idea_handoff, IDEA_HANDOFF_SCHEMA),
      current_stage: text(raw.current_stage) || 'study_setup',
      last_route: text(raw.last_route) || 'entry',
      active_job_id: text(raw.active_job_id) || null,
      created_at: text(raw.created_at),
      updated_at: text(raw.updated_at) || new Date().toISOString(),
    };
  }

  function persistPayload(context) {
    const payload = {};
    PERSISTED_FIELDS.forEach(field => { payload[field] = context[field]; });
    return payload;
  }

  function readCache() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? normalize(JSON.parse(raw)) : null;
    } catch (_) {
      try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
      return null;
    }
  }

  function readContextList() {
    try {
      const raw = localStorage.getItem(LIST_STORAGE_KEY);
      const rows = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(rows)) throw new Error('StudyContext cache list must be an array');
      const sanitized = [];
      rows.forEach(row => {
        if (!row || typeof row !== 'object' || !(row.id || row.study_context_id)) return;
        try { sanitized.push(normalize(row)); } catch (_) {}
      });
      try { localStorage.setItem(LIST_STORAGE_KEY, JSON.stringify(sanitized)); } catch (_) {}
      return sanitized;
    } catch (_) {
      try { localStorage.removeItem(LIST_STORAGE_KEY); } catch (_) {}
      return [];
    }
  }

  function writeContextList() {
    try { localStorage.setItem(LIST_STORAGE_KEY, JSON.stringify(contexts)); } catch (_) {}
  }

  function isDirty(id) {
    return !!(id && dirtyIds.has(id));
  }

  function setDirty(id, value) {
    if (!id) return;
    if (value) dirtyIds.add(id);
    else dirtyIds.delete(id);
    try {
      if (dirtyIds.size) localStorage.setItem(DIRTY_STORAGE_KEY, JSON.stringify(Array.from(dirtyIds)));
      else localStorage.removeItem(DIRTY_STORAGE_KEY);
    } catch (_) {}
  }

  function rememberContext(context) {
    if (!context || !context.id) return;
    const normalized = normalize(context);
    contexts = [normalized].concat(contexts.filter(row => row.id !== normalized.id)).slice(0, 80);
    writeContextList();
  }

  function writeCache(context) {
    try {
      if (context) localStorage.setItem(STORAGE_KEY, JSON.stringify(context));
      else localStorage.removeItem(STORAGE_KEY);
    } catch (_) {}
  }

  function clone(value) {
    if (!value) return null;
    try { return JSON.parse(JSON.stringify(value)); } catch (_) { return value; }
  }

  function status() {
    return Object.assign({}, syncState);
  }

  function emit(reason) {
    window.dispatchEvent(new CustomEvent(EVENT_NAME, {
      detail: { context: clone(activeContext), sync: status(), reason: reason || 'update' },
    }));
  }

  function setSync(state, error) {
    syncState = {
      state,
      error: error ? String(error.message || error) : null,
      updated_at: new Date().toISOString(),
    };
    emit('sync');
  }

  function setLocal(value, reason) {
    revision += 1;
    activeContext = normalize(value);
    bumpContextRevision(activeContext.id);
    setDirty(activeContext.id, true);
    rememberContext(activeContext);
    writeCache(activeContext);
    syncState = {
      state: 'pending',
      error: null,
      updated_at: new Date().toISOString(),
    };
    emit(reason || 'local');
    return clone(activeContext);
  }

  function responseContext(response) {
    if (!response || typeof response !== 'object') return null;
    return response.context || (response.result && response.result.context) || null;
  }

  function persist(context) {
    const api = window.EU_API;
    if (!api || typeof api.saveStudyContext !== 'function') {
      setSync('local-only', 'StudyContext API is unavailable');
      return Promise.resolve(clone(context));
    }
    const requestRevision = revision;
    const requestContextRevision = contextRevision(context.id);
    setSync('syncing');
    const payload = persistPayload(context);
    payload.expected_revision = context.revision;
    const request = () => api.saveStudyContext(payload).then(response => {
      const saved = responseContext(response) || context;
      if (requestContextRevision === contextRevision(context.id)) {
        const normalizedSaved = normalize(saved);
        rememberContext(normalizedSaved);
        setDirty(context.id, false);
      }
      if (requestRevision === revision && activeContext && activeContext.id === context.id) {
        activeContext = normalize(saved);
        rememberContext(activeContext);
        writeCache(activeContext);
        setSync('synced');
        emit('persisted');
      }
      return clone(requestRevision === revision ? activeContext : saved);
    }).catch(error => {
      if (requestRevision === revision) setSync('error', error);
      throw error;
    });
    persistQueue = persistQueue.catch(() => null).then(request);
    return persistQueue;
  }

  function update(patch, options) {
    const opts = options || {};
    const next = Object.assign({}, activeContext || {}, cleanObject(patch), {
      updated_at: new Date().toISOString(),
    });
    const context = setLocal(next, opts.reason || 'update');
    if (opts.persist !== false) persist(context).catch(error => {
      console.warn('[EasyICU] StudyContext sync failed:', error);
    });
    return context;
  }

  function startNew(patch, options) {
    const opts = options || {};
    const context = setLocal(Object.assign({}, cleanObject(patch), {
      id: localId(),
      updated_at: new Date().toISOString(),
    }), opts.reason || 'start-new');
    if (opts.persist !== false) persist(context).catch(error => {
      console.warn('[EasyICU] new StudyContext sync failed:', error);
    });
    return context;
  }

  function activate(id) {
    const context = contexts.find(row => row.id === text(id));
    if (!context) return Promise.reject(new Error('StudyContext not found: ' + text(id)));
    if (isDirty(context.id)) {
      const selected = setLocal(context, 'activate-dirty');
      return persist(selected);
    }

    revision += 1;
    const activateRevision = revision;
    activeContext = normalize(context);
    rememberContext(activeContext);
    writeCache(activeContext);
    syncState = { state: 'syncing', error: null, updated_at: new Date().toISOString() };
    emit('activate');
    const api = window.EU_API;
    if (!api || typeof api.saveStudyContext !== 'function') {
      setSync('local-only', 'StudyContext API is unavailable');
      return Promise.resolve(clone(activeContext));
    }
    const request = () => api.saveStudyContext({ id: context.id }).then(response => {
      const saved = responseContext(response) || context;
      if (activateRevision === revision && activeContext && activeContext.id === context.id) {
        activeContext = normalize(saved);
        rememberContext(activeContext);
        writeCache(activeContext);
        setSync('synced');
        emit('activated');
      }
      return clone(activateRevision === revision ? activeContext : saved);
    }).catch(error => {
      if (activateRevision === revision) setSync('error', error);
      throw error;
    });
    persistQueue = persistQueue.catch(() => null).then(request);
    return persistQueue;
  }

  function patchContext(id, patch, options) {
    const contextId = text(id);
    const opts = options || {};
    const current = contexts.find(row => row.id === contextId);
    if (!current) return null;
    if (Object.prototype.hasOwnProperty.call(opts, 'expectedActiveJobId')
        && current.active_job_id !== opts.expectedActiveJobId) return null;
    const next = normalize(Object.assign({}, current, cleanObject(patch), { id: contextId }));
    revision += 1;
    bumpContextRevision(contextId);
    rememberContext(next);
    if (activeContext && activeContext.id === contextId) {
      activeContext = next;
      writeCache(activeContext);
    }
    if (opts.dirty === true) setDirty(contextId, true);
    emit(opts.reason || 'context-history');
    return clone(next);
  }

  function registerSource(route, supplier) {
    const id = text(route);
    if (!id || typeof supplier !== 'function') return;
    suppliers[id] = supplier;
  }

  function sourceIdentity(value) {
    const source = cleanDataSource(value);
    if (!source) return '';
    if (source.path) return 'path:' + source.path;
    if (source.database) return 'database:' + source.database.toLowerCase();
    return source.label ? 'label:' + source.label.toLowerCase() : '';
  }

  function prepare(options) {
    const opts = options || {};
    const sourceRoute = text(opts.sourceRoute || opts.source_route || location.hash.slice(1)) || 'entry';
    const current = clone(activeContext);
    let supplied = suppliers[sourceRoute] ? suppliers[sourceRoute](current) : {};
    const explicitPatch = cleanObject(opts.patch);
    let patch = Object.assign({}, cleanObject(supplied), explicitPatch);
    const currentSource = sourceIdentity(current && current.data_source);
    const nextSource = sourceIdentity(patch.data_source);
    const sourceBoundary = !!(currentSource && nextSource && currentSource !== nextSource);
    const currentQuestion = text(current && current.question);
    const nextQuestion = text(patch.question);
    const questionBoundary = !opts.continueExisting && !!(currentQuestion && nextQuestion && currentQuestion !== nextQuestion);
    const nextStage = text(opts.currentStage || opts.current_stage || patch.current_stage);
    const leavingCrossdbPlan = !!(current && current.current_stage === 'crossdb_plan_only' && nextStage && nextStage !== 'crossdb_plan_only');
    const scopeBoundary = leavingCrossdbPlan;
    if (sourceBoundary || questionBoundary || scopeBoundary) {
      supplied = suppliers[sourceRoute] ? suppliers[sourceRoute](null) : {};
      patch = Object.assign({}, cleanObject(supplied), explicitPatch);
    }
    patch = Object.assign({}, patch, {
      current_stage: text(opts.currentStage || opts.current_stage || (supplied && supplied.current_stage)) || 'agent_handoff',
      last_route: sourceRoute,
    });
    return sourceBoundary || questionBoundary || scopeBoundary
      ? startNew(patch, { persist: false, reason: sourceBoundary ? 'source-boundary' : (scopeBoundary ? 'scope-boundary' : 'question-boundary') })
      : update(patch, { persist: false, reason: 'handoff-local' });
  }

  function handoff(options) {
    const opts = options || {};
    const sourceRoute = text(opts.sourceRoute || opts.source_route || location.hash.slice(1)) || 'entry';
    const targetRoute = text(opts.targetRoute || opts.target_route) || 'agent';
    const context = prepare(opts);
    const handoffRevision = revision;
    const saved = persist(context);
    const handoffRequest = () => saved.then(savedContext => {
      if (handoffRevision !== revision) return clone(activeContext);
      const api = window.EU_API;
      if (!api || typeof api.handoffStudyContext !== 'function') return savedContext;
      return api.handoffStudyContext({
        study_context_id: savedContext.id,
        expected_revision: savedContext.revision,
        // The metadata endpoint intentionally preserves server-owned lifecycle
        // fields on existing contexts. Carry the locally prepared transition
        // into the dedicated CAS-protected handoff endpoint instead of echoing
        // the metadata response's prior stage.
        current_stage: context.current_stage,
        last_route: sourceRoute,
        target_route: targetRoute,
        active_job_id: context.active_job_id || undefined,
      }).then(response => {
        const handed = responseContext(response);
        if (handed && handoffRevision === revision) {
          activeContext = normalize(handed);
          rememberContext(activeContext);
          writeCache(activeContext);
          setDirty(activeContext.id, false);
        }
        if (handoffRevision === revision) {
          setSync('synced');
          emit('handoff-persisted');
        }
        return clone(activeContext || savedContext);
      });
    });
    persistQueue = persistQueue.catch(() => null).then(handoffRequest);
    const persisted = persistQueue.catch(error => {
      if (handoffRevision === revision) setSync('error', error);
      throw error;
    });
    return { context, persisted };
  }

  function hydrate(options) {
    const force = options === true || !!(options && options.force);
    if (hydratePromise && !force) return hydratePromise;
    if (force) hydratePromise = null;
    const api = window.EU_API;
    if (!api || typeof api.loadActiveStudyContext !== 'function') {
      setSync(activeContext ? 'cached' : 'empty');
      hydratePromise = Promise.resolve(clone(activeContext));
      return hydratePromise;
    }
    const hydrateRevision = revision;
    setSync('loading');
    hydratePromise = Promise.allSettled([
      api.loadActiveStudyContext(),
      typeof api.listStudyContexts === 'function' ? api.listStudyContexts() : Promise.resolve({ contexts: [] }),
    ]).then(responses => {
      const activeResult = responses[0];
      const listResult = responses[1];
      if (activeResult.status === 'rejected' && listResult.status === 'rejected') {
        throw activeResult.reason || listResult.reason || new Error('StudyContext hydration failed');
      }
      const response = activeResult.status === 'fulfilled' ? activeResult.value : null;
      const listResponse = listResult.status === 'fulfilled' ? (listResult.value || {}) : {};
      const serverContexts = Array.isArray(listResponse.contexts) ? listResponse.contexts : [];
      const serverIds = new Set(serverContexts.map(context => text(context && context.id)).filter(Boolean));
      // The server owns product-list membership.  Remove clean cached rows
      // that are no longer returned (for example internal evaluation records),
      // while preserving unsynced local edits and the currently active record.
      contexts = contexts.filter(context =>
        serverIds.has(context.id)
        || isDirty(context.id)
        || (activeContext && activeContext.id === context.id),
      );
      serverContexts.forEach(context => {
        if (!isDirty(context && context.id)) rememberContext(context);
      });
      writeContextList();
      if (activeContext && isDirty(activeContext.id)) rememberContext(activeContext);
      const serverContext = responseContext(response)
        || contexts.find(row => row.id === listResponse.active_id)
        || null;
      const browserActiveIsDirty = !!(activeContext && isDirty(activeContext.id));
      if (serverContext && hydrateRevision === revision && !browserActiveIsDirty && !isDirty(serverContext.id)) {
        activeContext = normalize(serverContext);
        rememberContext(activeContext);
        writeCache(activeContext);
      }
      if (hydrateRevision === revision) {
        setSync(activeContext ? (isDirty(activeContext.id) ? 'cached-dirty' : 'synced') : 'empty');
        emit('hydrated');
      }
      return clone(activeContext);
    }).catch(error => {
      if (hydrateRevision === revision) setSync(activeContext ? 'cached' : 'error', error);
      hydratePromise = null;
      return clone(activeContext);
    });
    return hydratePromise;
  }

  window.EU_STUDY_CONTEXT = {
    STORAGE_KEY,
    EVENT_NAME,
    active: () => clone(activeContext),
    all: () => clone(contexts) || [],
    activate,
    status,
    hydrate,
    patchContext,
    prepare,
    update,
    startNew,
    persist: () => activeContext ? persist(clone(activeContext)) : Promise.resolve(null),
    handoff,
    registerSource,
  };

  document.addEventListener('click', event => {
    const trigger = event.target.closest('[data-study-handoff]');
    if (!trigger) return;
    event.preventDefault();
    const sourceRoute = trigger.dataset.studySource || location.hash.slice(1) || 'entry';
    const targetRoute = trigger.dataset.studyTarget || 'agent';
    const result = handoff({ sourceRoute, targetRoute });
    location.hash = '#' + targetRoute;
    result.persisted.catch(error => {
      console.warn('[EasyICU] StudyContext handoff stayed local:', error);
    });
  }, true);

  window.addEventListener('storage', event => {
    if (event.key === LIST_STORAGE_KEY) {
      contexts = readContextList();
      emit('storage-list');
      return;
    }
    if (event.key === DIRTY_STORAGE_KEY) {
      try {
        const rows = JSON.parse(event.newValue || '[]');
        dirtyIds = new Set(Array.isArray(rows) ? rows.map(text).filter(Boolean) : []);
      } catch (_) { dirtyIds = new Set(); }
      return;
    }
    if (event.key !== STORAGE_KEY) return;
    revision += 1;
    activeContext = readCache();
    if (activeContext) {
      bumpContextRevision(activeContext.id);
      rememberContext(activeContext);
    }
    syncState = { state: activeContext ? 'cached' : 'empty', error: null, updated_at: new Date().toISOString() };
    emit('storage');
  });

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', hydrate);
  else hydrate();
})();
