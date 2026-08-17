/* Guided Copilot StudyContext mapping.
   Owner: Guided -> Agent metadata handoff. The Guided IIFE exposes only a
   sanitized snapshot through window.EU_GUIDED_CONTEXT. */
(function () {
  'use strict';

  function build(stage, base) {
    const store = window.EU_STUDY_CONTEXT;
    const existing = arguments.length > 1
      ? (base || {})
      : (store && store.active ? (store.active() || {}) : {});
    const adapter = window.EU_GUIDED_CONTEXT;
    const snapshot = adapter && adapter.snapshot ? adapter.snapshot() : {};
    const question = snapshot.question || existing.question || '';
    const cohort = Object.assign({}, existing.cohort || {});
    ['review', 'source_count', 'source_type', 'comparison_mode'].forEach(key => delete cohort[key]);
    return {
      title: existing.title || question.slice(0, 160) || 'Guided ICU study',
      question,
      purpose: existing.purpose || 'Continue the study configured in Guided Copilot.',
      data_source: snapshot.source || existing.data_source || null,
      cohort: Object.assign(cohort, {
        preset: snapshot.cohort_preset || 'adult_first',
        max_patients: snapshot.max_patients,
      }),
      modules: snapshot.modules && snapshot.modules.length ? snapshot.modules : (existing.modules || []),
      outcome: snapshot.outcome || existing.outcome || '',
      time_window: Object.assign({}, existing.time_window || {}, {
        preset: snapshot.window_preset || 'whole_stay',
        label: snapshot.window_label || '',
      }),
      comparator: snapshot.comparator || existing.comparator || '',
      export_format: snapshot.export_format || existing.export_format || '',
      analysis_goal: existing.analysis_goal || question,
      confirmations: Object.assign({}, existing.confirmations || {}, { guided_configuration_collected: !!snapshot.configured, crossdb_plan_only: false }),
      current_stage: stage || 'agent_handoff',
    };
  }

  function persistForRun(stage) {
    const store = window.EU_STUDY_CONTEXT;
    if (!store || !store.prepare || !store.persist) return Promise.resolve(null);
    const current = store.active ? store.active() : null;
    const fresh = build(stage || 'agent_preflight', null);
    if (current && current.current_stage === 'crossdb_plan_only' && fresh.question === current.question) {
      return Promise.reject(new Error('Reframe the Cross-DB plan as a single-export question before running Agent preflight.'));
    }
    store.prepare({
      sourceRoute: 'guided',
      currentStage: stage || 'agent_preflight',
      continueExisting: true,
    });
    return store.persist().then(context => {
      if (context && context.current_stage === 'crossdb_plan_only') {
        throw new Error('Reframe the Cross-DB plan as a single-export question before running Agent preflight.');
      }
      return context;
    });
  }

  function activeId() {
    const store = window.EU_STUDY_CONTEXT;
    const current = store && store.active ? store.active() : null;
    return current && current.id ? current.id : '';
  }

  function handoff(targetRoute) {
    const store = window.EU_STUDY_CONTEXT;
    if (!store || !store.handoff) return { context: null, persisted: Promise.resolve(null) };
    const current = store.active ? store.active() : null;
    const fresh = build('agent_handoff', null);
    if (current && current.current_stage === 'crossdb_plan_only' && fresh.question === current.question) {
      return {
        context: current,
        persisted: Promise.reject(new Error('Reframe the Cross-DB plan as a single-export question before starting a governed analysis.')),
      };
    }
    return store.handoff({
      sourceRoute: 'guided',
      targetRoute: targetRoute || 'agent',
      currentStage: 'agent_handoff',
      continueExisting: true,
    });
  }

  if (window.EU_STUDY_CONTEXT && window.EU_STUDY_CONTEXT.registerSource) {
    window.EU_STUDY_CONTEXT.registerSource('guided', current => build('agent_handoff', current));
  }
  window.EU_GUIDED_STUDY_CONTEXT = { activeId, build, handoff, persistForRun };
})();
