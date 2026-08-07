/* Owner: study-stage progress derived from the active StudyContext.

   The sidebar's three section labels (Discovery & Plan / Data & Review /
   Analysis & Evidence) already ARE the product's main line, but they were
   rendered as static category headings, so a user could not tell which stage
   they had reached or what was still missing before a draft is possible.

   The shell is not allowed to read window.EU_STUDY_CONTEXT directly — that
   boundary is locked by test_route_handoffs_have_sources_and_viz_mapping_has_
   its_own_owner. So this module owns the derivation and hands the shell a
   small typed snapshot to render, and nothing else.

   States are evidence-based, never optimistic: a stage is 'done' only when the
   context carries the artifact that stage produces. */
(function () {
  'use strict';

  // Stages the agent surface writes once a run has left the plan.
  const AGENT_TERMINAL = ['review', 'review_blocked', 'agent_failed', 'agent_cancelled'];
  const REVIEW_CONFIRMATIONS = [
    'patient_review_completed',
    'cohort_review_completed',
    'crossdb_review_completed',
  ];

  function activeContext() {
    const store = window.EU_STUDY_CONTEXT;
    if (!store || typeof store.active !== 'function') return null;
    try { return store.active(); } catch (e) { return null; }
  }

  function has(value) {
    return typeof value === 'string' ? value.trim().length > 0 : !!value;
  }

  function discoveryDone(context) {
    if (!context) return false;
    const confirmations = context.confirmations || {};
    return has(context.question) || !!confirmations.guided_configuration_collected;
  }

  function dataDone(context) {
    if (!context) return false;
    const confirmations = context.confirmations || {};
    if (confirmations.extraction_completed) return true;
    if (REVIEW_CONFIRMATIONS.some(key => confirmations[key])) return true;
    return context.current_stage === 'data_prepared';
  }

  function analysisDone(context) {
    if (!context) return false;
    return AGENT_TERMINAL.indexOf(String(context.current_stage || '')) !== -1;
  }

  // Cross-DB comparison is deliberately a plan-only scope: it can shape an
  // analysis plan but is not itself a reviewed cohort, so it must not light
  // the data stage as complete.
  function planOnly(context) {
    return !!(context && context.current_stage === 'crossdb_plan_only');
  }

  function snapshot() {
    const context = activeContext();
    const done = {
      discovery: discoveryDone(context),
      data: dataDone(context),
      analysis: analysisDone(context),
    };
    const order = ['discovery', 'data', 'analysis'];
    // The active stage is the first unfinished one — but only once the study
    // has actually started. With no context at all every stage is 'todo', so
    // a first-time user is not told they are mid-way through something.
    const started = !!context && (done.discovery || done.data || done.analysis || planOnly(context));
    const firstOpen = order.filter(id => !done[id])[0] || null;
    const stages = order.map(id => ({
      id,
      state: done[id] ? 'done' : (started && id === firstOpen ? 'active' : 'todo'),
    }));
    return {
      started,
      stale: !!window.EU_STALE,
      planOnly: planOnly(context),
      question: (context && context.question) || '',
      stages,
      byId: stages.reduce((acc, stage) => { acc[stage.id] = stage.state; return acc; }, {}),
    };
  }

  window.EU_STUDY_PROGRESS = { snapshot };
})();
