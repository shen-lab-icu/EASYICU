/* Cross-DB raw-run owner: explicit quick/full feature scope and bounded request. */
(function () {
  'use strict';

  const FEATURE_SCOPES = Object.freeze({
    core: 'curated_core',
    all: 'all_catalog',
  });
  const CORE_FEATURES = Object.freeze([
    'hr', 'map', 'sbp', 'dbp', 'resp', 'temp',
    'spo2', 'crea', 'lact', 'wbc', 'plt', 'glu',
  ]);

  function coreFeatures() {
    return CORE_FEATURES.slice();
  }

  function normalizeFeatureScope(value) {
    return value === 'all' || value === FEATURE_SCOPES.all ? 'all' : 'core';
  }

  function apiFeatureScope(value) {
    return FEATURE_SCOPES[normalizeFeatureScope(value)];
  }

  function buildRequest({ dataRoot, databases, maxPatients, sampleSize, featureScope }) {
    const scope = normalizeFeatureScope(featureScope);
    const request = {
      data_root: String(dataRoot || '').trim(),
      databases: Array.from(new Set((databases || []).map(String).filter(Boolean))),
      feature_scope: FEATURE_SCOPES[scope],
      coverage_min: 2,
      max_patients: maxPatients,
      sample_size: sampleSize,
    };
    if (scope === 'core') request.features = coreFeatures();
    return request;
  }

  window.EU_CROSSDB_RAW = {
    apiFeatureScope,
    buildRequest,
    coreFeatures,
    normalizeFeatureScope,
  };
})();
