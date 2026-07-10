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

  function nonCrossdbCohort(value) {
    const cohort = Object.assign({}, value || {});
    ['review', 'source_count', 'source_type', 'comparison_mode'].forEach(key => delete cohort[key]);
    return cohort;
  }

  function build(route, current) {
    const existing = current || {};
    const adapter = window.EU_VIZ_CONTEXT;
    const snapshot = adapter && adapter.snapshot ? adapter.snapshot(route) : {};
    const config = CONFIG[route];
    return {
      title: existing.title || config.title,
      question: existing.question || config.question,
      purpose: existing.purpose || 'Continue from data review to an auditable Agent analysis.',
      data_source: snapshot.data_source || existing.data_source || null,
      cohort: Object.assign({}, route === 'crossdb' ? {} : nonCrossdbCohort(existing.cohort), snapshot.cohort || {}, { review: route }),
      modules: snapshot.modules && snapshot.modules.length ? snapshot.modules : (existing.modules || []),
      outcome: existing.outcome || snapshot.outcome || '',
      time_window: existing.time_window || {},
      comparator: route === 'crossdb' ? (snapshot.comparator || 'cross_database_descriptive') : (existing.comparator || snapshot.comparator || ''),
      export_format: existing.export_format || '',
      analysis_goal: route === 'crossdb'
        ? 'Create a reviewable cross-database analysis plan without executing a single-export Agent run.'
        : (existing.analysis_goal || 'Run an evidence-bound analysis of the reviewed cohort.'),
      confirmations: Object.assign({}, existing.confirmations || {}, {
        [`${route}_review_completed`]: true,
        crossdb_plan_only: route === 'crossdb',
      }),
      current_stage: config.stage,
    };
  }

  if (window.EU_STUDY_CONTEXT && window.EU_STUDY_CONTEXT.registerSource) {
    Object.keys(CONFIG).forEach(route => {
      window.EU_STUDY_CONTEXT.registerSource(route, current => build(route, current));
    });
  }
})();
