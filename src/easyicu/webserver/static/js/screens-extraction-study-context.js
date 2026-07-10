/* Data Extraction -> Agent StudyContext mapping.
   Owner: extraction handoff metadata. Closure state is supplied only through
   window.EU_EXTRACTION_CONTEXT by the extraction screen owner. */
(function () {
  'use strict';

  function build(current) {
    const existing = current || {};
    const adapter = window.EU_EXTRACTION_CONTEXT;
    const snapshot = adapter && adapter.snapshot ? adapter.snapshot() : {};
    const label = snapshot.preset_label || 'ICU cohort';
    return {
      title: existing.title || `${label} study`,
      question: existing.question || `Analyze the selected ${label} using the prepared feature modules.`,
      purpose: existing.purpose || 'Continue from Data Extraction to an auditable Agent analysis.',
      data_source: snapshot.data_source || existing.data_source || null,
      cohort: snapshot.cohort || existing.cohort || {},
      modules: snapshot.modules && snapshot.modules.length ? snapshot.modules : (existing.modules || []),
      outcome: existing.outcome || '',
      time_window: Object.assign({}, existing.time_window || {}, { observation_hours: snapshot.observation_hours }),
      comparator: existing.comparator || '',
      export_format: snapshot.export_format || existing.export_format || '',
      analysis_goal: existing.analysis_goal || 'Run an evidence-bound analysis of the prepared cohort.',
      confirmations: Object.assign({}, existing.confirmations || {}, { extraction_completed: true, crossdb_plan_only: false }),
      current_stage: 'data_prepared',
    };
  }

  if (window.EU_STUDY_CONTEXT && window.EU_STUDY_CONTEXT.registerSource) {
    window.EU_STUDY_CONTEXT.registerSource('extraction', build);
  }
})();
