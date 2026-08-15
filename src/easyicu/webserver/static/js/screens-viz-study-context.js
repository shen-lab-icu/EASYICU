/* Patient/Cohort/Cross-DB -> Agent StudyContext mapping.
   Owner: visualization handoff metadata. Runtime screen state is supplied only
   through window.EU_VIZ_CONTEXT; this sibling never reaches into its IIFE. */
(function () {
  'use strict';

  const CONFIG = {
    patient: {
      title: 'Patient-reviewed ICU cohort',
      question: 'Analyze the reviewed ICU cohort using the active EasyICU export.',
      stage: 'patient_reviewed',
    },
    cohort: {
      title: 'Cohort statistics study',
      question: 'Which cohort-level patterns warrant evidence-bound analysis in the active EasyICU export?',
      stage: 'cohort_reviewed',
    },
    crossdb: {
      title: 'Cross-database ICU study',
      question: 'Assess whether the selected cohort and feature definitions are portable across the reviewed ICU databases.',
      stage: 'crossdb_plan_only',
    },
  };

  const ROUTE_COHORT_FIELDS = {
    patient: ['entity_count', 'full_entity_count', 'review_entities', 'review_entity_cap', 'review_scope', 'module_count'],
    cohort: ['cohort_size', 'comparison'],
    crossdb: ['source_count', 'source_type', 'comparison_mode'],
  };
  const ROUTE_CONFIRMATION_FIELDS = [
    'patient_review_completed',
    'cohort_review_completed',
    'crossdb_review_completed',
    'patient_review_bounded_sample',
    'patient_review_full_entity_set',
  ];

  function nonCrossdbCohort(value, route) {
    const cohort = Object.assign({}, value || {});
    delete cohort.review;
    Object.entries(ROUTE_COHORT_FIELDS).forEach(([owner, fields]) => {
      if (owner !== route) fields.forEach(key => delete cohort[key]);
    });
    return cohort;
  }

  function routeComparator(existing, snapshot, route) {
    if (route === 'crossdb') return snapshot.comparator || 'cross_database_descriptive';
    if (route === 'cohort') return snapshot.comparator || existing.comparator || '';
    return existing.current_stage === 'cohort_reviewed' ? '' : (existing.comparator || snapshot.comparator || '');
  }

  function build(route, current) {
    const existing = current || {};
    const adapter = window.EU_VIZ_CONTEXT;
    const snapshot = adapter && adapter.snapshot ? adapter.snapshot(route) : {};
    const config = CONFIG[route];
    const cohort = Object.assign({}, route === 'crossdb' ? {} : nonCrossdbCohort(existing.cohort, route), snapshot.cohort || {}, { review: route });
    const confirmations = Object.assign({}, existing.confirmations || {});
    ROUTE_CONFIRMATION_FIELDS.forEach(key => delete confirmations[key]);
    confirmations[`${route}_review_completed`] = true;
    confirmations.crossdb_plan_only = route === 'crossdb';
    if (route === 'patient') {
      confirmations.patient_review_bounded_sample = cohort.review_scope === 'browser_bounded_entity_sample';
      confirmations.patient_review_full_entity_set = cohort.review_scope === 'full_entity_set';
    }
    return {
      title: existing.title || config.title,
      question: existing.question || config.question,
      purpose: existing.purpose || 'Continue from data review to an auditable Agent analysis.',
      data_source: route === 'crossdb' ? null : (snapshot.data_source || existing.data_source || null),
      crossdb_selection: route === 'crossdb' ? (snapshot.crossdb_selection || {}) : {},
      cohort,
      modules: snapshot.modules && snapshot.modules.length ? snapshot.modules : (existing.modules || []),
      outcome: existing.outcome || snapshot.outcome || '',
      time_window: existing.time_window || {},
      comparator: routeComparator(existing, snapshot, route),
      export_format: existing.export_format || '',
      analysis_goal: route === 'crossdb'
        ? 'Create a reviewable cross-database analysis plan without executing a single-export Agent run.'
        : (existing.analysis_goal || 'Run an evidence-bound analysis of the reviewed cohort.'),
      confirmations,
      current_stage: config.stage,
    };
  }

  if (window.EU_STUDY_CONTEXT && window.EU_STUDY_CONTEXT.registerSource) {
    Object.keys(CONFIG).forEach(route => {
      window.EU_STUDY_CONTEXT.registerSource(route, current => build(route, current));
    });
  }
})();
