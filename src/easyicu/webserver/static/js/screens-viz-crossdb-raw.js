/* Cross-DB raw-run owner: curated first-run concepts and hard-bounded request. */
(function () {
  'use strict';

  const CORE_FEATURES = Object.freeze([
    'hr', 'map', 'sbp', 'dbp', 'resp', 'temp',
    'spo2', 'crea', 'lact', 'wbc', 'plt', 'glu',
  ]);

  function coreFeatures() {
    return CORE_FEATURES.slice();
  }

  function buildRequest({ dataRoot, databases, maxPatients, sampleSize }) {
    return {
      data_root: String(dataRoot || '').trim(),
      databases: Array.from(new Set((databases || []).map(String).filter(Boolean))),
      features: coreFeatures(),
      feature_scope: 'curated_core',
      coverage_min: 2,
      max_patients: maxPatients,
      sample_size: sampleSize,
    };
  }

  window.EU_CROSSDB_RAW = { coreFeatures, buildRequest };
})();
